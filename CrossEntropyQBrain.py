import time
import random
import tensorflow


class CrossEntropyQBrain:


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
        self.tensorflow_session = None

        self.DEBUG_MESSAGES = False


    def on_series(self, num_lanes):
        self.num_lanes = num_lanes
        self.car_road_state = None
        if (self.tensorflow_session != None):
            self.tensorflow_session.close()
        self.__initialize_tensorflow()
        

    def on_before_move(self, car_position, current_road_section, road):
        self.car_road_state = []
        self.car_road_state.append(self.__state_to_qvalues_list(car_position, road))
        action = 0 # The default action is to stay still.
        predicted_action = self.tensorflow_session.run(self.sample_actions_tensor
            , feed_dict={self.car_road_tensor: self.car_road_state})
        action = predicted_action[0][0]-1
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
    

    def __initialize_tensorflow(self):
        number_states = self.num_lanes + (self.num_lanes * self.num_road_sections_in_q_values)
        number_action = self.num_actions

        # Ask for prediction.
        tensorflow.reset_default_graph()
        self.car_road_tensor = tensorflow.placeholder(shape=[None, number_states]
            , dtype=tensorflow.float32)
        hidden_layer_tensor = tensorflow.layers.dense(self.car_road_tensor, 128
            , activation=tensorflow.nn.relu)
        prediction_tensor = tensorflow.layers.dense(hidden_layer_tensor, number_action)
        self.sample_actions_tensor = tensorflow.multinomial(logits = prediction_tensor
            , num_samples = 1)

        # Train.
        self.chosen_action_tensor = tensorflow.placeholder(shape=[None], dtype=tensorflow.uint8)
        self.rewards_tensor = tensorflow.placeholder(shape=[None], dtype=tensorflow.float32)
        cross_entropies_tensor = tensorflow.losses.softmax_cross_entropy(
            onehot_labels=tensorflow.one_hot(self.chosen_action_tensor, number_action)
            , logits=prediction_tensor)
        loss_tensor = tensorflow.reduce_sum(self.rewards_tensor * cross_entropies_tensor)

        optimizer_tensor = tensorflow.train.RMSPropOptimizer(learning_rate=0.001, decay=0.99)
        self.train_tensor = optimizer_tensor.minimize(loss_tensor)

        initializer = tensorflow.global_variables_initializer()
        self.tensorflow_session = tensorflow.Session()
        self.tensorflow_session.run(initializer)
    

    def __update_qvalues(self, action, reward, recent_road_states):
        learning_states = recent_road_states[-self.advances_learning_interval:]

        for current_state in learning_states:
            road_sections = current_state[0]
            car_position = current_state[1]
            action = current_state[2]+1

            road_sections_and_car_position = [self.__state_to_qvalues_list(car_position, road_sections)]
            self.tensorflow_session.run(self.train_tensor, feed_dict
                                                            ={self.car_road_tensor: road_sections_and_car_position,
                                                            self.chosen_action_tensor: [action],
                                                            self.rewards_tensor: [reward]})


    # def __bellmans_equation(self, last_q_value, reward, discount, max_q_value):
    #     result = (((1 - self.step_size)*last_q_value) + (self.step_size*(reward + (discount*max_q_value))))
    #     return result


    # def __state_action_to_qvalue(self, action, car_position, road_sections):
    #     qvalue = 0
    #     qvalues_tuple = self.__state_action_to_qvalues_tuple(action, car_position, road_sections)
    #     if (qvalues_tuple in self.latest_qvalues):
    #         qvalue = self.__bellmans_equation(self.latest_qvalues[qvalues_tuple],
    #             0, self.base_discount, self.max_qvalues[qvalues_tuple])
    #     return qvalue


    def __state_to_qvalues_list(self, car_position, road_sections):
        qvalues_list = [0] * (self.num_lanes + (self.num_lanes
            * self.num_road_sections_in_q_values))

        qvalues_list[car_position-1] = 1

        for road_section_index, road_section in enumerate(road_sections):
            for road_obstacle_it in range(1,len(road_section)-1):
                if (road_section[road_obstacle_it] != ' '):
                    qvalues_list[self.num_lanes+(self.num_lanes
                        *road_section_index)+road_obstacle_it-1] = 1
                else:
                    qvalues_list[self.num_lanes+(self.num_lanes
                        *road_section_index)+road_obstacle_it-1] = 0

        return qvalues_list
