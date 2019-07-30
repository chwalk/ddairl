import random
import time
from ExperienceReplay import ExperienceReplay


"""This class handles all the details of the game; drawing the screen, maintaining data
structures, etc. There is no learning logic in here."""
class GameStructure:


    def __init__(self, starting_road_width, ending_road_width, num_advances_level_complete, \
            display_rate, random_obstacle_probability, max_number_display_road_states, \
            max_number_road_states, advances_learning_interval, max_history, fast_mode):
        self.starting_road_width = starting_road_width
        self.ending_road_width = ending_road_width
        self.num_advances_level_complete = num_advances_level_complete
        self.display_rate = display_rate
        self.random_obstacle_probability = random_obstacle_probability
        self.num_blank_lines_between_screens = 20
        self.max_number_display_road_states = max_number_display_road_states
        self.max_number_road_states = max_number_road_states
        self.advances_learning_interval = advances_learning_interval
        self.max_history = max_history
        self.fast_mode = fast_mode

        self.DEBUG_FIXED_OBSTACLES = False
        self.DISPLAY_EVERY_XTH_GAME = 500
        self.FAST_DISPLAY_RATE = 0.1


    def start(self, brain):
        self.brain = brain
        # Learn how to drive the three lane road. Then add a lane, and then another.
        for road_width in range(self.starting_road_width, self.ending_road_width):
            self.road_width = road_width
            self.empty_road_section = [' '] * self.road_width
            self.empty_road_section[0] = '|'
            self.empty_road_section[-1] = '|'
            self.game_number = -1
            # Keep track of each advance, so that we know how well we are learning.
            self.num_advances = 0
            self.num_advances_for_road_width = 0
            self.__play_series()


    def __play_series(self):
        self.brain.on_series(self.road_width - 2)
        self.experience_replay = ExperienceReplay(self.max_history, self.advances_learning_interval)

        # Run many games, learning to drive with each game. Once the car advances 2000 sections
        # (or whatever num_advances_level_complete is set to), consider the level completed.
        while (self.num_advances < self.num_advances_level_complete):
            self.game_number += 1
            self.__play_game()


    def __play_game(self):
        # Keep track of each advance, so that we know how well we are learning.
        self.num_advances = 0

        # This keeps the last several states -- that is, the way the road looked, the car
        # position and the action taken. In the event of a crash, we go back and learn from
        # them. This is known as reinforcement learning.
        self.recent_road_states = []

        # Start with the car in the middle of the road (or close to it).
        self.car_position = ((self.road_width - 2) // 2) + 1

        # Start with the car driving straight down the road, not to the left or right.
        self.action = 0
        self.previous_road_section_num_obstacles = 0

        # Draw the road.
        self.__draw_entrance()

        # Add a section of road, navigate, add another section and so on until we crash.
        crashed = False
        while (not crashed):
            # This creates the row at the bottom of the screen, which may or may not have obstacles
            # in it.
            self.__create_next_road_section()
            crashed = self.__move()


    NUMBER_SECTIONS_IN_ENTRANCE = 2


    def __draw_entrance(self):
        # The road ahead.
        self.road = []
        if (self.experience_replay.is_empty() == False):
            (car_position, road, future_states) = self.experience_replay.pop()
            self.car_position = car_position
            self.road = road
            if (future_states is not None):
                for future_state in future_states:
                    self.future_road.append(future_state[0][-1].copy())
        else:
            self.future_road = []
            for entrance_num in range(self.NUMBER_SECTIONS_IN_ENTRANCE):
                self.road.append(self.empty_road_section.copy())
        self.__scroll()


    def __move(self):
        # Keep track of each advance, so that we know how well we are learning.
        self.num_advances += 1

        if (self.num_advances > self.num_advances_for_road_width):
            self.num_advances_for_road_width = self.num_advances

        current_road_section = self.road[0]

        # Call out to our artificial intelligence "brain" and see whether we want to move left,
        # right or continue straight.
        self.action = self.brain.on_before_move(self.car_position, current_road_section, self.road)

        # Keep track of the game states.
        self.__update_recent_road_states()

        # Move the car.
        self.car_position += self.action

        crashed = False
        # Crashing involves hitting either the curb or a boulder.
        if ((current_road_section[self.car_position] == '|')
            or (current_road_section[self.car_position] == 'O')):
                crashed = True # Crash!
                self.experience_replay.push(self.recent_road_states)

        # Call out to our brain and let it know whether we crashed.
        self.brain.on_after_move(self.action, crashed, self.num_advances, self.recent_road_states)

        # Actually draw the road.
        self.__scroll(crashed)

        if (crashed):
            self.brain.on_crashed(self.fast_mode, self.game_number, self.DISPLAY_EVERY_XTH_GAME, self.road_width, self.num_advances, self.num_advances_for_road_width)

        return crashed


    # Keep track of the game states.
    def __update_recent_road_states(self):
        road_copy = []
        for road_section in self.road:
            road_copy.append(road_section.copy())
        self.recent_road_states.append([road_copy, self.car_position, self.action])
        if (len(self.recent_road_states) > self.max_number_road_states):
            self.recent_road_states.pop(0)


    def __create_next_road_section(self):
        curb = 1
        road_width_without_curbs = self.road_width - 2
        if (len(self.future_road) > 0):
            next_road_section = self.future_road.pop(0)
        else:
            next_road_section = self.empty_road_section.copy()
            if (self.previous_road_section_num_obstacles > 1):
                num_obstacles_allowed_in_land_row = road_width_without_curbs - 2
            else:
                num_obstacles_allowed_in_land_row = road_width_without_curbs - 1
            num_obstacles_in_road_section = 0
            if (self.DEBUG_FIXED_OBSTACLES):
                spots = []
                if (self.num_advances % 4 == 0):
                    spots = [1]
                elif (self.num_advances % 2 == 0):
                    spots = [0, 2]
                for spot in spots:
                    next_road_section[curb + spot] = 'O'
                    num_obstacles_in_road_section += 1
            else:
                spots = random.sample(list(range(road_width_without_curbs)), num_obstacles_allowed_in_land_row)
                for spot in spots:
                    if (random.random() < self.random_obstacle_probability):
                        next_road_section[curb + spot] = 'O'
                        num_obstacles_in_road_section += 1
            self.previous_road_section_num_obstacles = num_obstacles_in_road_section
        self.road.append(next_road_section)
        if (len(self.road) > self.max_number_display_road_states):
            self.road.pop(0)


    def __scroll(self, crashed = False):
        if (self.fast_mode):
            if (self.game_number % self.DISPLAY_EVERY_XTH_GAME != 0):
                return
        self.__scroll_screen()
        self.__draw_previous_road_sections()
        current_road_section = self.road[0].copy()
        if (crashed):
            # Draw the burning embers of the crashed car, engulfed in roiling clouds of
            # burning gasoline. Or an X. Same thing.
            current_road_section[self.car_position] = 'X'
        else:
            current_road_section[self.car_position] = 'H'
        self.__draw_road_section(current_road_section)
        for road_section in self.road[1:]:
            self.__draw_road_section(road_section)
        if (self.fast_mode):
            time.sleep(self.FAST_DISPLAY_RATE)
        else:
            time.sleep(self.display_rate)


    def __draw_road_section(self, road_section):
        left_margin_size = (80 - len(road_section))//2 # Integer arithmetic.
        left_margin_list = [' '] * left_margin_size
        left_margin = ''.join(left_margin_list)
        print(left_margin + ''.join(road_section))


    def __scroll_screen(self):
        for blank_line_num in range(self.num_blank_lines_between_screens):
            print()


    def __draw_previous_road_sections(self):
        if (len(self.recent_road_states) > self.max_number_display_road_states):
            # Why the strange offset of -(self.max_number_display_road_states+1)? The number of
            # road states can grow larger than the number we wish to display. Frankly, I don't
            # recall why. If it turns out the extra states aren't needed, get rid of them and clean
            # up this code.
            for prev_road_section in self.recent_road_states[-(self.max_number_display_road_states+1)][0]:
                self.__draw_road_section(prev_road_section)

