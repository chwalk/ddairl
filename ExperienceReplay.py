class ExperienceReplay:


    def __init__(self, max_history, max_snapshot):
        self.experience_replay_history = []
        self.future_road = []
        self.max_history = max_history
        self.max_snapshot = max_snapshot


    def is_empty(self):
        return (len(self.experience_replay_history) == 0)


    def pop(self):
        snapshot = self.experience_replay_history[0]
        current_state = snapshot[0]
        car_position = current_state[1]
        road = current_state[0].copy()
        if (len(snapshot) > 0):
            return (car_position, road, snapshot[1:])
        else:
            return (car_position, road, None)


    def push(self, recent_road_states):
        snapshot = []
        for recent_state in recent_road_states[-self.max_snapshot:]:
            snapshot.append(recent_state)
        self.experience_replay_history.insert(0, snapshot)
        if (len(self.experience_replay_history) > self.max_history):
            self.experience_replay_history.pop()
