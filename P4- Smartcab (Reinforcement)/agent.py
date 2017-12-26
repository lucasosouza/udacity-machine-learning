import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import random
from collections import defaultdict
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.qvalues = defaultdict(float)
        self.possible_actions = [None, 'forward', 'left', 'right']
        self.completion_times = []
        self.failures_count = 0
        self.trial_number = 0 

        # added metrics
        self.total_reward = 0
        self.penalties_count = 0
        self.penalties_per_trial = []

        # customizable parameters. also defined in run method
        self.discount_factor = .5
        self.learning_rate = .5
        self.exploration_rate = .5


    def reset(self, destination=None):
        """ Resets some agent parameters at each trial """

        self.planner.route_to(destination)

        # TODO: Prepare for a new trip; reset any variables here, if required

        self.trial_number += 1
        self.total_reward = 0
        self.penalties_count = 0

        # Decaying exploration_rate: each round I divide epsilon by number of trials. 
        self.exploration_rate = self.exploration_rate / self.trial_number

    def update(self, t):
        """
            Arguments:
                t - time

            Relevant Info:
                inputs is the response from the car sensors
                it is represented as a dictionary with 4 keys:
                    - whether the light is green or red
                    - for each car approaching, shows the directions it is heading. the car may be approching:
                        - oncoming (straight)
                        - from the right
                        - from the left

                deadline is an number, showing how many actions you have left.

                next waypoint is likely the output of the gps, it shows in which direction is the waypoint.
        """

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state 
        self.state = self.state_transform(self.next_waypoint, inputs)
        
        # TODO: Select action according to your policy
        # account for exploration_rate

        if np.random.random > self.exploration_rate:
            action = self.best_qvalue_action(self.state)
        else:
            action = np.random.choice(self.possible_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.total_reward += reward

        # store completion time, to evaluate performance
        # I will only store for the last 20 trials, after the algorithm learned
        if self.trial_number >= 90:
            if reward == -1:
                self.penalties_count += 1
            # finish state 1
            if reward == 12.0:
                time_to_complete = (t*1.0)/(t+deadline)
                self.completion_times.append(time_to_complete)
                self.penalties_per_trial.append(self.penalties_count)
            # finish state 2    
            if deadline == 0: # 
                self.failures_count +=1
                self.penalties_per_trial.append(self.penalties_count)

        # Compute the state s', the new state after taking the action
        new_inputs = self.env.sense(self)
        new_next_waypoint = self.planner.next_waypoint()
        new_state = tuple([new_next_waypoint] + new_inputs.values())

        # TODO: Learn policy based on state, action, reward
        self.qvalues[(self.state, action)] = self.qvalues[(self.state, action)] + self.learning_rate *\
         (reward + self.discount_factor * max(self.calc_qvalues(new_state)) - self.qvalues[(self.state, action)])

        # print "LearningAgent.update(): qvalue = {}, state = {}, deadline = {}, inputs = {}, action = {}, reward = {}".format(self.qvalues[(self.state,action)], self.state, deadline, inputs, action, reward)  # [debug]

    def calc_qvalues(self, state):
        """ Recover list of qvalues for each combination of state and action """
        
        return map(lambda action: self.qvalues[(state, action)], self.possible_actions)


    def best_qvalue_action(self, state):
        """ Chooses action based on best q-value """

        # get qvalues
        qvalues = self.calc_qvalues(state)

        # select maximum values
        argmaxs = np.argwhere(qvalues == np.amax(qvalues)).flatten()

        # randomly choose an item of the most q values. if it is just one, will select that one
        selected_action = self.possible_actions[np.random.choice(argmaxs)]

        return selected_action

    def state_transform(self, next_waypoint, inputs):
        """ Maps sensors inputs to state """

        return tuple([next_waypoint] + inputs.values())


    def random_action(self):
        """ Chooses a random action from the set of possible actions at each intersection """

        return np.random.choice(self.possible_actions)

    def simple_reactive_action(self, next_waypoint):
        """ Chooses action according to next waypoint, with disregard to other sensors """

        return next_waypoint

    def define_parameters(self, alpha, gamma, epsilon):
        """ Define parameters to be optimized through grid search """

        self.learning_rate = alpha
        self.discount_factor = gamma
        self.exploration_rate = epsilon

    def report_metrics(self):
        """ Report relevant metrics:
                - Average time to complete (in percentage of total time available)
                - Number of failures
        """

        return np.mean(self.completion_times)*100, self.failures_count, np.mean(self.penalties_per_trial)


def run(alpha=.5, gamma=.15, epsilon= .1):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    ### Define alpha and gamma for agent
    a.define_parameters(alpha, gamma, epsilon)

    # Now simulate it
    #sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    sim = Simulator(e, update_delay=0, display=False)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "Average required time to complete: {:.2f}% ; Number of failures: {} ; Average number of penalties: {:.2f} ".format(*a.report_metrics())

    ### return metrics
    return a.report_metrics()

def grid_search():
    """ Search over different parameters of discount factor gamma, learning rate alpha, and exploration rate epsilon
        
        Warning: adjust the interval rate for alpha, gamma and epsilon intervals before running. 
        Current settings will run for approximately 20 minutes
    """

    # run the game and get metrics
    metrics = {}
    for alpha in np.arange(0,.9,.05):
        for gamma in np.arange(0,.9,.05):
            for epsilon in np.arange(0,1.1,.1):
                metrics[alpha, gamma, epsilon] = run(alpha, gamma, epsilon)

    # sort ascending by 1. number of failures first, 2. average number of penalties, 3. average time to complete
    sorted_metrics = sorted(metrics.items(), key=lambda x:(x[1][1],x[1][2],x[1][0]))[:20]

    # pretty print
    print " Alpha | Gamma | Epsilon | Avg time to complete | Number of failures | Avg number of penalties "
    for sm in sorted_metrics:
        print "{:.2f} | {:.2f}  | {:.2f} | {:.2f}% | {} | {:.2f}".format(
            sm[0][0], sm[0][1], sm[0][2], sm[1][0], sm[1][1],sm[1][2])
    
if __name__ == '__main__':
    # run() 
    grid_search() # disable grid search if you only need to run one simulation
