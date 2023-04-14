class CF:

    def __init__(self, cf_state, terminal, actions, cumulative_reward, value, searched_nodes, time, path=[], num_steps=0, action_path=[]):
        self.cf_state = cf_state
        self.terminal = terminal
        self.actions = actions
        self.cumulative_reward = cumulative_reward
        self.value = value
        self.searched_nodes = searched_nodes
        self.time = time
        self.path = path
        self.num_steps = num_steps
        self.action_path = action_path
