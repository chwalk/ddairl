import time
import random


class TabularQBrain:


    def __init__(self, safe_reward, crash_reward, advances_learning_interval, base_discount, \
        num_actions, step_size, random_move_probability, num_road_sections_in_q_values):
        self.MOVE_LEFT_ACTION = -1
        self.STAY_STILL_ACTION = 0
        self.MOVE_RIGHT_ACTION = 1

        self.safe_reward = safe_reward
        self.crash_reward = crash_reward
        self.advances_learning_interval = advances_learning_interval
        self.base_discount = base_discount
        self.num_actions = num_actions
        self.step_size = step_size
        self.random_move_probability = random_move_probability
        self.num_road_sections_in_q_values = num_road_sections_in_q_values

        self.DEBUG_MESSAGES = False


    def on_series(self, num_lanes):
        self.num_lanes = num_lanes
        self.latest_qvalues = {}
        self.max_qvalues = {}


    def on_before_move(self, car_position, current_road_section, road):
        action = 0 # The default action is to stay still.
        left = self.__state_action_to_qvalue(self.MOVE_LEFT_ACTION, car_position, road)
        stay = self.__state_action_to_qvalue(self.STAY_STILL_ACTION, car_position, road)
        right = self.__state_action_to_qvalue(self.MOVE_RIGHT_ACTION, car_position, road)

        maximum_value = max(left, stay, right)
        if (left == maximum_value):
            action = self.MOVE_LEFT_ACTION
        elif (right == maximum_value):
            action = self.MOVE_RIGHT_ACTION
        # else: stay still

        return action


    def on_after_move(self, action, crashed, num_advances, recent_road_states):
        if (crashed or (num_advances % self.advances_learning_interval == self.advances_learning_interval-1)):
            reward = self.safe_reward
            if (crashed):
                reward = self.crash_reward

            self.__update_qvalues(action, reward, recent_road_states)


    def on_crashed(self, fast_mode, game_number, display_frequency, road_width, num_advances, max_advances):
        if ((not fast_mode) or (game_number % display_frequency == 0)):
            print("Crashed! Road width: {0}, game num: {1}, num advances: {2}, max advances: {3}.".format(road_width, game_number, num_advances, max_advances))
            time.sleep(1)
    

    def __update_qvalues(self, action, reward, recent_road_states):
        learning_states = recent_road_states[-self.advances_learning_interval:]

        discount_power = len(learning_states)
        for current_state in learning_states:
            discount = self.base_discount ** discount_power

            road_sections = current_state[0]
            car_position = current_state[1]
            action = current_state[2]

            qvalues_tuple = self.__state_action_to_qvalues_tuple(action, car_position, road_sections)
            if (qvalues_tuple in self.latest_qvalues):
                qvalue = self.__bellmans_equation(self.latest_qvalues[qvalues_tuple], reward
                    , discount, self.max_qvalues[qvalues_tuple])
            
                self.latest_qvalues[qvalues_tuple] = qvalue
                if (qvalue > self.max_qvalues[qvalues_tuple]):
                    self.max_qvalues[qvalues_tuple] = qvalue
            else:
                qvalue = self.__bellmans_equation(0, reward, discount, 0)
                self.latest_qvalues[qvalues_tuple] = qvalue
                self.max_qvalues[qvalues_tuple] = qvalue

            discount_power -= 1


    def __bellmans_equation(self, last_q_value, reward, discount, max_q_value):
        result = (((1 - self.step_size)*last_q_value) + (self.step_size*(reward + (discount*max_q_value))))
        return result


    def __state_action_to_qvalue(self, action, car_position, road_sections):
        qvalue = 0
        qvalues_tuple = self.__state_action_to_qvalues_tuple(action, car_position, road_sections)
        if (qvalues_tuple in self.latest_qvalues):
            qvalue = self.__bellmans_equation(self.latest_qvalues[qvalues_tuple],
                0, self.base_discount, self.max_qvalues[qvalues_tuple])
        return qvalue


    def __state_action_to_qvalues_tuple(self, action, car_position, road_sections):
        qvalues_list = [0] * (self.num_lanes + (self.num_lanes
            * self.num_road_sections_in_q_values) + self.num_actions)

        qvalues_list[car_position-1] = 1

        for road_section_index, road_section in enumerate(road_sections):
            for road_obstacle_it in range(1,len(road_section)-1):
                if (road_section[road_obstacle_it] != ' '):
                    qvalues_list[self.num_lanes+(self.num_lanes*road_section_index)+road_obstacle_it-1] = 1
                else:
                    qvalues_list[self.num_lanes+(self.num_lanes*road_section_index)+road_obstacle_it-1] = 0

        qvalues_list[action - 2] = 1

        qvalues_tuple = tuple(qvalues_list)

        return qvalues_tuple
