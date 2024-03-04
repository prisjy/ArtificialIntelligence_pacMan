# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** BEGIN YOUR CODE HERE ***"
        # util.raiseNotDefined()
        # STEP1. iterate and update the value
        for _ in range(0, self.iterations):
            curVal = util.Counter()

            # STEP2. get the state
            for curState in self.mdp.getStates():
                if self.mdp.isTerminal(curState):
                    curVal[curState] = self.values[curState]
                    continue
                
                # STEP3. get the possible actions and update the value
                curVal[curState] = -1000000000000000000000000000000000000
                
                for curAction in self.mdp.getPossibleActions(curState):
                    qValue = 0
                    for sNext in self.mdp.getTransitionStatesAndProbs(curState, curAction):
                        nextVal = sNext[0]
                        nextActionProb = sNext[1]
                        qValue = qValue + nextActionProb * (self.mdp.getReward(curState, curAction, nextVal) + self.discount * self.values[nextVal])
                    curVal[curState] = max(curVal[curState], qValue)
                    
            self.values = curVal
        "*** END YOUR CODE HERE ***"

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** BEGIN YOUR CODE HERE ***"
        # util.raiseNotDefined()
        qValue = 0
        for sNext in self.mdp.getTransitionStatesAndProbs(state, action):
            nextVal = sNext[0]
            nextActionProb = sNext[1]
            qValue = qValue + nextActionProb * (self.mdp.getReward(state, action, nextVal) + self.discount * self.values[nextVal])
        return qValue
        "*** END YOUR CODE HERE ***"
        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** BEGIN YOUR CODE HERE ***"
        # util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return None
        
        qValueAction = util.Counter()

        for action in self.mdp.getPossibleActions(state):
            qValueAction[action] = self.computeQValueFromValues(state, action)

        return qValueAction.argMax()
        "*** END YOUR CODE HERE ***"

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Write value iteration code here
        "*** BEGIN YOUR CODE HERE ***"
        # util.raiseNotDefined()
        allStates = self.mdp.getStates()
        numOfStates = len(allStates)

        # STEP1. iterate and update the value
        for i in range(0, self.iterations):
            curVal = util.Counter()
            curState = allStates[i % numOfStates]

            if self.mdp.isTerminal(curState):
                continue
            
            # STEP3. get the possible actions and update the value
            maxQ = -1000000000000000000000000000000000000
            
            for curAction in self.mdp.getPossibleActions(curState):
                qValue = self.computeQValueFromValues(curState, curAction)
                maxQ = max(maxQ, qValue)
                    
            self.values[curState] = maxQ
        "*** END YOUR CODE HERE ***"

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** BEGIN YOUR CODE HERE ***"
        # util.raiseNotDefined()
        pQueue = util.PriorityQueue()

        predecessors = {}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                        if prob > 0:
                            if nextState not in predecessors:
                                predecessors[nextState] = set()
                            predecessors[nextState].add(state)

        for curState in self.mdp.getStates():
            if not self.mdp.isTerminal(curState):
                qValues = []
                for curAction in self.mdp.getPossibleActions(curState):
                    qValue = self.computeQValueFromValues(curState, curAction)
                    qValues.append(qValue)
                maxQ = max(qValues) if qValues else None
                if maxQ is not None:
                    diff = abs(self.values[curState] - maxQ)
                    pQueue.update(curState, -diff)

        for _ in range(self.iterations):
            if pQueue.isEmpty():
                break

            curState = pQueue.pop()
            if not self.mdp.isTerminal(curState):
                qValues = []
                for curAction in self.mdp.getPossibleActions(curState):
                    qValue = self.computeQValueFromValues(curState, curAction)
                    qValues.append(qValue)
                maxQ = max(qValues)
                self.values[curState] = maxQ

            for p in predecessors.get(curState, []):
                qValues = []
                for curAction in self.mdp.getPossibleActions(p):
                    qValue = self.computeQValueFromValues(p, curAction)
                    qValues.append(qValue)
                maxQ = max(qValues)
                diff = abs(self.values[p] - maxQ)
                if diff > self.theta:
                    pQueue.update(p, -diff)
        "*** END YOUR CODE HERE ***"
        

