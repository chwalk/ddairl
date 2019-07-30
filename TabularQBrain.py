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


    def on_before_move(self, car_position, current_road_section, road):
        action = 0 # The default action is to stay still.
        return action


    def on_after_move(self, action, crashed, num_advances, recent_road_states):
        print('In after move.')


    def on_crashed(self, fast_mode, game_number, display_frequency, road_width, num_advances, max_advances):
        if ((not fast_mode) or (game_number % display_frequency == 0)):
            print("Crashed! Road width: {0}, game num: {1}, num advances: {2}, max advances: {3}.".format(road_width, game_number, num_advances, max_advances))
            time.sleep(1)
