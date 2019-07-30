import random
import time
import tensorflow


class DeepQNeuralBrain:


    def __init__(self, safe_reward, crash_reward, advances_learning_interval, base_discount, \
        gamma, num_actions, step_size, deep_q_learning_interval, random_move_probability,
        num_road_sections_in_q_values):
        self.MOVE_LEFT_ACTION = -1
        self.STAY_STILL_ACTION = 0
        self.MOVE_RIGHT_ACTION = 1

        self.safe_reward = safe_reward
        self.crash_reward = crash_reward
        self.advances_learning_interval = advances_learning_interval
        self.base_discount = base_discount
        self.gamma = gamma
        self.num_actions = num_actions
        self.step_size = step_size
        self.deep_q_learning_interval = deep_q_learning_interval
        self.random_move_probability = random_move_probability
        self.num_road_sections_in_q_values = num_road_sections_in_q_values
        self.tensorflow_session = None

        self.DEBUG_MESSAGES = False


    def on_series(self, num_lanes):
        self.num_lanes = num_lanes
        self.car_road_state = None
        self.training_inputs = []
        # We completely retrain the neural network every time the size of the road changes. The
        # tensors making up the neural network rely on the size of the road.
        if (self.tensorflow_session != None):
            self.tensorflow_session.close()
        self.__initialize_tensorflow(1)


    def on_before_move(self, car_position, current_road_section, road):
        self.car_road_state = []
        self.car_road_state.append(self.__state_to_qvalues_list(car_position, road))

        action = 0 # The default action is to stay still.
        # With some presumably small chance, move randomly. This is likely not necessary with this
        # application, but is a good idea with many. The idea is that the game may not try some
        # avenues with an improbable, but highly valuable reward. If there's some randomness baked
        # in, then the game will try it eventually.
        #
        # The example I recall is that there is a door that 9 times out of 10 gives no reward, but
        # 1 out of 10 gives a reward of 100. There is another door that always gives a reward of 1.
        # The agent will quickly learn to always open the door of reward 1. However, it would be
        # better off to play the odds and get a reward of 100 10% of the time. I think I saw this
        # in Serena Yeung's excellent Stanford video called Reinforcement Learning (Lecture 14)?
        if (random.random() < self.random_move_probability):
            # Move left a third of the time, move right a third of the time and stay still a third
            # of the time.
            move_probability = random.random()
            if (move_probability < 1/3):
                action = self.MOVE_LEFT_ACTION
            elif (move_probability > 2/3):
                action = self.MOVE_RIGHT_ACTION
            #else don't move.
        else:
            # In the much more likely case that the agent is using its past learning to determine
            # the next move, determine the q-value to decide how to move. Essentially, take a
            # snapshot of the state -- where the car is and where the boulders are, and retrieve
            # the preferred action.
            predicted_action = self.tensorflow_session.run(self.sample_actions_tensor, feed_dict={self.car_road_tensor: self.car_road_state})
            # We subtract 1 because the action is stored as an unsigned int in tensorflow (0-2),
            # however we prefer to work in terms of -1, 0 and 1.
            action = predicted_action[0][0] - 1

        return action


    def on_after_move(self, action, crashed, num_advances, recent_road_states):
        # The game only learns if the car crashes (learns from its mistakes) or after a number of
        # successful runs (learns from its successes).
        if (crashed or (num_advances % self.advances_learning_interval == self.advances_learning_interval-1)):
            reward = self.safe_reward
            if (crashed):
                reward = self.crash_reward

            if (self.DEBUG_MESSAGES):
                print('action: {0}, crashed: {1}, na: {2}, reward: {3}, recent_road_states: {4}'.format(action, crashed, num_advances, reward, recent_road_states))

            # The main purpose of this method: Learn.
            self.__update_qvalues(reward, recent_road_states)


    def on_crashed(self, fast_mode, game_number, display_frequency, road_width, num_advances, max_advances):
        if ((not fast_mode) or (game_number % display_frequency == 0)):
            print("Crashed! Road width: {0}, game num: {1}, num advances: {2}, max advances: {3}.".format(road_width, game_number, num_advances, max_advances))
            time.sleep(1)


    def __initialize_tensorflow(self, hidden_layers):
        # What information needs to be stored for the state? The car position, the road and any
        # obstacles.
        #
        # This information is stored in a binary manner, for reasons that will become clear later.
        # Here's how the structure looks:
        # |n|n|n  |n|n|n |n|n|n |n|n|n
        #  car          roadway
        #
        # If the car is on the left side of the road, the left-most number will be 1. In the
        # middle, the second number will be 1 and on the right side of the road, the third will be
        # set. The numbers for the roadway will be all 0s for a roadway with no obstacles. If there
        # is 1 obstacle, then a 1 will be set describing its position.
        #
        # The action is simply three values: moving left, moving right or staying still):
        # |n|n|n
        # action
        # A move to the left will set the number third from the right, staying still will set the
        # number second from the right and a move to the right will set the last number.
        #
        # For example, a car on the right of the road that comes across a boulder similarly on the
        # road but two spots away, and consequently moving left will have a state and action like
        # this:
        #
        # state:  |0|0|1  |0|0|0 |0|0|0 |0|0|1
        # action: |1|0|0

        number_states = self.num_lanes + (self.num_lanes * self.num_road_sections_in_q_values)
        number_actions = self.num_actions

        # Automatically reset tensorflow variables. Needed since we are resetting the tensorflow session for every road width.
        tensorflow.reset_default_graph()

        # The car position and the road ahead, including obstacles.
        self.car_road_tensor = tensorflow.placeholder(shape=[None, number_states],
            dtype=tensorflow.float32, name="car_road_tensor")

        # The car can make three moves: right, left and stay still. 0 is left, 1 is stay still, 2 is right.
        self.chosen_action_tensor = tensorflow.placeholder(shape=[None],
            dtype=tensorflow.uint8, name="chosen_action_tensor")

        # Takes on two values: 1 if the car doesn't crash, or -1 if it does.
        self.rewards_tensor = tensorflow.placeholder(shape=[None],
            dtype=tensorflow.float32, name="rewards_tensor")

        # The Q value associated with the previous three tensors.
        self.q_prime_tensor = tensorflow.placeholder(shape=[None, number_actions],
            dtype=tensorflow.float32, name="q_prime_tensor")

        # Relatively quick learning with relu. The relu function is just y=x, x>=0 and y=0, x<0
        hidden_layer_tensor = tensorflow.layers.dense(self.car_road_tensor, 128, activation=tensorflow.nn.relu)
        # The action logits tensor consists of a whopping three nodes.
        self.car_road_logits_tensor = tensorflow.layers.dense(hidden_layer_tensor,
            number_actions, name="car_road_logits_tensor")

        # When called, grabs a single, preferred action. The call to multinomial returns a
        # probability distribution, a multinomial probability distribution defined by the
        # training of the neural network.
        self.sample_actions_tensor = tensorflow.multinomial(logits = self.car_road_logits_tensor,
            num_samples = 1, name="sample_actions_tensor")

        # We define the quality of the prediction as the tensor determining the action. (Yes,
        # the following is the tensor determining the action.)
        self.q_tensor = self.car_road_logits_tensor

        # The following block is all just to change one value. It seems a little messy so I tried
        # to find other, simpler ways but it became a rabbit hole. Revisit on a rainy day.
        self.one_hot_tensor = tensorflow.one_hot(self.chosen_action_tensor, number_actions)
        self.one_hot_complement_tensor = tensorflow.one_hot(self.chosen_action_tensor, number_actions, 0.0, 1.0)
        # The equation for the q value is the reward + gamma * previous q value.
        self.target_q_value = self.one_hot_tensor * (self.rewards_tensor + (self.gamma * self.q_prime_tensor))
        # Complete the target q tensor.
        self.target_q_tensor = (self.one_hot_complement_tensor * self.q_prime_tensor) + self.target_q_value

        # Define the loss using least squares regression.
        self.sq_diff_tensor = tensorflow.squared_difference(self.target_q_tensor, self.q_tensor)
        self.loss_tensor = tensorflow.reduce_mean(self.sq_diff_tensor, name="loss_tensor")

        # Taking a walk downhill.
        optimizer_tensor = tensorflow.train.RMSPropOptimizer(learning_rate=0.001, decay=0.99, name="optimizer_tensor")
        self.train_tensor = optimizer_tensor.minimize(self.loss_tensor, name="train_tensor")

        initializer = tensorflow.global_variables_initializer()
        self.tensorflow_session = tensorflow.Session()
        self.tensorflow_session.run(initializer)


    def __update_qvalues(self, reward, recent_road_states):
        # We don't necessarily learn from all the states. Grab the latest x states.
        learning_states = recent_road_states[-self.advances_learning_interval:]

        for current_state in learning_states:
            # Grab the relevant pieces from the state.
            road_sections = current_state[0]
            car_position = current_state[1]
            # We add 1 because we want to store the action as an unsigned int in tensorflow (0-2)
            # in the following step, however the code up to this point worked in terms of -1, 0 and
            # 1.
            action = current_state[2] + 1

            road_sections_and_car_position = [self.__state_to_qvalues_list(car_position, road_sections)]

            # Ask the neural network for the q value.
            q_value = self.tensorflow_session.run(self.q_tensor, feed_dict={self.car_road_tensor: road_sections_and_car_position})

            # With deep Q learning, we don't immediately update the neural network. Push the values
            # on a list that we will add to the neural network later.
            self.training_inputs.append((road_sections_and_car_position, [action], [reward], q_value))

            if (len(self.training_inputs) >= self.deep_q_learning_interval):
                self.__push_values_into_neural_net()


    def __state_to_qvalues_list(self, car_position, road_sections):
        # What information needs to be stored? The car position, the road and any obstacles.
        #
        # This information is stored in a binary manner, for reasons that will become clear later.
        # Here's how the structure looks:
        # |n|n|n  |n|n|n |n|n|n |n|n|n
        #  car          roadway
        #
        # If the car is on the left side of the road, the left-most number will be 1. In the
        # middle, the second number will be 1 and on the right side of the road, the third will be
        # set. The numbers for the roadway will be all 0s for a roadway with no obstacles. If there
        # is 1 obstacle, then a 1 will be set describing its position.
        #
        # For example, a car on the right of the road that comes across a boulder similarly on the
        # road but two spots away will have a state like this:
        #
        # |0|0|1  |0|0|0 |0|0|0 |0|0|1
        #
        # Create the list with the appropriate number of values and fill it will all 0s.
        qvalues_list = [0] * (self.num_lanes + (self.num_lanes * self.num_road_sections_in_q_values))

        # Put a 1 in the place representing the car's position.
        qvalues_list[car_position-1] = 1

        # Look for obstacles and put a 1 in the appropriate spots if we find them.
        for road_section_index, road_section in enumerate(road_sections):
            for road_obstacle_it in range(1,len(road_section)-1):
                if (road_section[road_obstacle_it] != ' '):
                    qvalues_list[self.num_lanes+(self.num_lanes*road_section_index)+road_obstacle_it-1] = 1
                else:
                    qvalues_list[self.num_lanes+(self.num_lanes*road_section_index)+road_obstacle_it-1] = 0

        return qvalues_list


    def __push_values_into_neural_net(self):
        while (len(self.training_inputs) > 0):
            training_input = self.training_inputs.pop(0)

            car_road = training_input[0]
            action = training_input[1]
            reward = training_input[2]
            q_value = training_input[3]

            self.tensorflow_session.run(self.train_tensor, feed_dict={self.car_road_tensor: car_road,
                                                                self.chosen_action_tensor: action,
                                                                self.rewards_tensor: reward,
                                                                self.q_prime_tensor: q_value})
