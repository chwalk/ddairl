import random


class QValueBrain:


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
            # the q-values for moving left, staying still or moving right.
            left = self.__state_action_to_qvalue(self.MOVE_LEFT_ACTION, car_position, road)
            stay = self.__state_action_to_qvalue(self.STAY_STILL_ACTION, car_position, road)
            right = self.__state_action_to_qvalue(self.MOVE_RIGHT_ACTION, car_position, road)

            if (self.DEBUG_MESSAGES):
                print('l: {0}, s: {1}, r: {2}, car_position: {3}, road: {4}'.format(left, stay, right, car_position, road))
            # Figure out whether moving left, staying still or moving right has the highest
            # q-value.
            maximum_value = max(left, stay, right)

            if (left == maximum_value):
                action = self.MOVE_LEFT_ACTION
            elif (right == maximum_value):
                action = self.MOVE_RIGHT_ACTION
            # else don't move.

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
            self.__update_qvalues(action, reward, recent_road_states)
    

    def on_crashed(self, fast_mode, game_number, display_frequency, road_width, num_advances, max_advances):
        if ((not fast_mode) or (game_number % display_frequency == 0)):
            print("Crashed! Road width: {0}, game num: {1}, num advances: {2}, max advances: {3}.".format(road_width, game_number, num_advances, max_advances))
            time.sleep(1)


    def __update_qvalues(self, action, reward, recent_road_states):
        # We don't necessarily learn from all the states. Grab the latest x states.
        learning_states = recent_road_states[-self.advances_learning_interval:]

        # Discount the earlier frames less than the more recent ones.
        discount_power = len(learning_states)
        for current_state in learning_states:
            # Calculates the discount for this frame. You'll note the earlier frames are discounted
            # less than the more recent ones.
            discount = self.base_discount ** discount_power

            # Grab the relevant pieces from the state.
            road_sections = current_state[0]
            car_position = current_state[1]
            action = current_state[2]

            # Create our "key".
            qvalues_tuple = self.__state_action_to_qvalues_tuple(action, car_position, road_sections)
            if (qvalues_tuple in self.latest_qvalues):
                if (qvalues_tuple not in self.max_qvalues):
                    print('Sanity error.')

                # Bellman's equation is at the heart of reinforcement learning. It's nice to
                # understand Bellman's equation to some extent, but frankly we can just look at it
                # as magic. Magic that works.
                qvalue = self.__bellmans_equation(self.latest_qvalues[qvalues_tuple], reward, \
                    discount, self.max_qvalues[qvalues_tuple])
                
                # To learn, we must keep track of the latest q-value and the max q-value. Overwrite
                # the latest, since this is the new latest.
                self.latest_qvalues[qvalues_tuple] = qvalue

                # And reset the max q-value if indeed the new value is larger.
                if (qvalue > self.max_qvalues[qvalues_tuple]):
                    self.max_qvalues[qvalues_tuple] = qvalue
            else:
                # We fall into this else clause if we hit some new road condition not encountered
                # before.

                # Calculate the initial qvalue assuming the latest q-value and the max q-value are
                # 0.
                qvalue = self.__bellmans_equation(0, reward, discount, 0)

                # Set the latest and max.
                self.latest_qvalues[qvalues_tuple] = qvalue
                self.max_qvalues[qvalues_tuple] = qvalue

            if (self.DEBUG_MESSAGES):
                print('current_state: {0}, qvalue: {1}'.format(current_state, qvalue))
            # Decreasing the power will actually increase the discount in the next iteration. For
            # example, .9 squared is less than .9 because .81 is less than .9.
            discount_power -= 1


    def __bellmans_equation(self, last_q_value, reward, discount, max_q_value):
        # A lot can be said about Bellman's equation and it's worth a good google or book. Some
        # important things to note:
        # -It is calculating a q-value.
        # -Initially, the result is mostly impacted by the last q-value. If we crashed the last
        #   time we tried "this", we'll likely try something else next time.
        # -It also takes into account "ancient history". Perhaps we did crash the last time we
        #   tried this, but prior to that the action worked a hundred times in a row.
        # -Over time, the right-hand side of the equation has more impact.
        # TODO: Instrument and make sure these statements are true.
        result = (((1 - self.step_size)*last_q_value) + (self.step_size*(reward + (discount*max_q_value))))
        return result


    # Given what the agent has learned, this returns the ranking of a given state and action.
    def __state_action_to_qvalue(self, action, car_position, road_sections):
        qvalue = 0
        qvalues_tuple = self.__state_action_to_qvalues_tuple(action, car_position, road_sections)
        if (qvalues_tuple in self.latest_qvalues):
            if (qvalues_tuple not in self.max_qvalues):
                print('Sanity error.')
            qvalue = self.__bellmans_equation(self.latest_qvalues[qvalues_tuple], 0, self.base_discount, self.max_qvalues[qvalues_tuple])
        return qvalue


    # The states are stored in a map, or otherwise referenced by a key. This returns that key for
    # any given state and action.
    def __state_action_to_qvalues_tuple(self, action, car_position, road_sections):
        # We start with a list and convert it to a tuple later. What information needs to be
        # stored? The car position, the road and any obstacles, as well as the action (moving left,
        # moving right or staying still).
        #
        # This information is stored in a binary manner, for reasons that will become clear later.
        # Here's how the structure looks:
        # |n|n|n  |n|n|n |n|n|n |n|n|n  |n|n|n
        #  car          roadway         action
        #
        # If the car is on the left side of the road, the left-most number will be 1. In the
        # middle, the second number will be 1 and on the right side of the road, the third will be
        # set. The numbers for the roadway will be all 0s for a roadway with no obstacles. If there
        # is 1 obstacle, then a 1 will be set describing its position. Finally, a move to the left
        # will set the number third from the right, staying still will set the number second from
        # the right and a move to the right will set the last number.
        #
        # For example, a car on the right of the road that comes across a boulder similarly on the
        # road but two spots away, and consequently moving left will have a state like this:
        #
        # |0|0|1  |0|0|0 |0|0|0 |0|0|1  |1|0|0
        #
        # Create the list with the appropriate number of values and fill it will all 0s.
        qvalues_list = [0] * (self.num_lanes + (self.num_lanes * self.num_road_sections_in_q_values) + self.num_actions)

        # Put a 1 in the place representing the car's position.
        qvalues_list[car_position-1] = 1

        # Look for obstacles and put a 1 in the appropriate spots if we find them.
        for road_section_index, road_section in enumerate(road_sections):
            for road_obstacle_it in range(1,len(road_section)-1):
                if (road_section[road_obstacle_it] != ' '):
                    qvalues_list[self.num_lanes+(self.num_lanes*road_section_index)+road_obstacle_it-1] = 1
                else:
                    qvalues_list[self.num_lanes+(self.num_lanes*road_section_index)+road_obstacle_it-1] = 0

        # Put a 1 in the last three numbers of the list representing the direction the car is
        # moving.
        qvalues_list[action - 2] = 1

        # Convert the list to a tuple so that Python will generate a hash for us.
        qvalues_tuple = tuple(qvalues_list)

        return qvalues_tuple
