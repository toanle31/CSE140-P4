from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import counter
from pacai.util import probability
from pacai.util import util
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.defense.DefensiveReflexAgent',
        second = 'pacai.agents.capture.offense.OffensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class QLAgent(CaptureAgent):
    """
    Base Q learning agent that will be used for
    both of the agents.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index)
        self.values = counter.Counter()
        self.features = None
        self.weights = None
        self.alpha = None
        self.epsilon = None
        self.discount = None

    def registerInitialState(self, state):
        """
        This function is called to set up our Agents
        So make sure our two agents override this and call
        super().regiserInitialState(state)
        """
        super().registerInitialState(state)
    
    def getValue(self, state):
        """
        Return the value of the highest valued action
        "action" should be received through getPolicy
        """
        action = self.getPolicy(state)
        value = 0.0
        if action is not None:
            value = self.getQValue(state, action)
        return value

    def getQValue(self, state, action):
        """
        Q value is the dot product of weights and features 
        """
        weights = self.weights
        features = self.getFeatures(state, action)
        return weights * features

    def getFeatures(self, state, action):
        """
        Each of the agent class should override this function
        and define their own features. This will be used in
        getQValue and update function. Returns a "Counter" dictionary
        """

        """
        note: maybe look at this:
        pacai.core.featureExtractors.IdentityExtractor
        To model our getFeatures function
        Though we will have to come up with features that would work
        with our attack/defense agent separately since they're going to
        behave differently
        """
        return self.features

    def getPolicy(self, state):
        """
        This function should return the best action
        available to our agent, this shoud be evaluated
        using the self.getQValue function
        """
        action = None
        legalActions = state.getLegalActions(state)[self.index]
        if legalActions:
            maxValue = float("-inf")
            for legalAction in legalActions:
                #  note: might wanna skip Directions.STOP here
                #  though for now just leaving it in
                qValue = self.getQValue(state, legalAction)
                nextState = self.getSuccessor(state, legalAction)
                #  reward should be calculated here based on the score
                #  of current state and next state potentially right now just 0
                reward = 0
                #  calling update here:
                #  this should probably be commented out for when
                #  we submit our agent
                self.update(state, action, nextState, reward)
                if (maxValue < qValue):
                    maxValue = qValue
                    action = legalAction
        return action

    def chooseAction(self, state):
        """
        This is the function we're supposed to override to
        return a legal action.
        """
        legalActions = state.getLegalActions(self.index)
        if not legalActions:
            return None
        else:
            if probability.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        """
        This function updates all the weights for our features
        It should be called in getPolicy.
        """

        """
        Formula for the correction value:
        w <- w + α[correction]f(s, a)
        correction = (R(s,a) + 𝛾V(s')) - Q(s, a)
        """
        features = self.getFeatures(state, action)
        for feature in features:
            discount = self.discount
            qPrime = self.getValue(nextState)
            qValue = self.getQValue(state, action)
            alpha = self.alpha
            correction = reward + discount  * qPrime - qValue
            self.weights[feature] += alpha * correction * features[feature]

    def getSuccessor(self, state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        ** Coppied from the Reflex Agent class **
        """
        successor = state.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    """
    Not sure what to do with this function yet
    Though it is called at the end of the game
    So our agents should override this
    and probably print it to a file
    def final(self, state):
    """

class AttackAgent(QLAgent):
    """
    Extends QLAgent to implement Attack Agent
    # TODO: override registerInitialState(self, state):
            override getFeatures(self, state, action)
    """

class DefenseAgent(QLAgent):
    """
    Extends QLAgent to implement Defense Agent
    # TODO: override registerInitialState(self, state)
            override getFeatures(self, state, action)
    """