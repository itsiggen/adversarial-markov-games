class agent_selector():
    '''
        Outputs an agent in the given order whenever agent_select is called. Can reinitialize to a new order
    '''
    def __init__(self, agents):
        self.reinit(agents)

    def reinit(self, agents):
        self.agents = agents
        self._current_agent = 0
        self.selected_agent = 0

    def reset(self):
        self.reinit(self.agents)
        return self.next()

    def next(self):
        self._current_agent = (self._current_agent + 1) % len(self.agents)
        self.selected_agent = self.agents[self._current_agent - 1]
        return self.selected_agent