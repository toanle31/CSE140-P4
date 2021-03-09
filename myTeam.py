from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import counter
from pacai.util import probability
from pacai.util import util
from pacai.core.actions import Actions
from pacai.core.search import search
from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.student.qlearningAgents import QLearningAgent
from pacai.student.qlearningAgents import ApproximateQAgent
from pacai.core import distance
from pacai.core.directions import Directions
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.offense.OffensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    # firstAgent = AttackAgent
    secondAgent = AttackAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class QLAgent(ReinforcementAgent):
    """
    Base Q learning agent that will be used for
    both of the agents.
    """
    # def __init__(self, index, **kwargs):
    #     super().__init__(index)
    #     self.values = counter.Counter()
    #     self.features = None
    #     self.weights = counter.Counter()
    #     self.alpha = 0.2
    #     self.epsilon = 0.05
    #     self.discount = 0.8
    #     self.actionList = []
    #     self.reward = 0
    #     self.q_vals = counter.Counter()

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = counter.Counter()
        self.attack = self.AttackAgent(self.index)

    # def registerInitialState(self, state):
    #     """
    #     This function is called to set up our Agents
    #     So make sure our two agents override this and call
    #     super().regiserInitialState(state)
    #     """
    #     super().registerInitialState(state)
    #     pass
    
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

    # def getQValue(self, state, action):
    #     """
    #     Q value is the dot product of weights and features 
    #     """
    #     # weights = self.weights
    #     # features = self.getFeatures(state, action)
    #     # return weights * features
    #     return self.q_vals[(state, action)]

    # def getFeatures(self, state, action):
    #     """
    #     Each of the agent class should override this function
    #     and define their own features. This will be used in
    #     getQValue and update function. Returns a "Counter" dictionary
    #     """

    #     """
    #     note: maybe look at this:
    #     pacai.core.featureExtractors.IdentityExtractor
    #     To model our getFeatures function
    #     Though we will have to come up with features that would work
    #     with our attack/defense agent separately since they're going to
    #     behave differently
    #     """
    #     pass

    def getPolicy(self, state):
        """
        This function should return the best action
        available to our agent, this shoud be evaluated
        using the self.getQValue function
        """
        action = None
        legalActions = state.getLegalActions(self.index)
        # print(legalActions)
        if legalActions:
            maxValue = float("-inf")
            for legalAction in legalActions:
                #  note: might wanna skip Directions.STOP here
                #  though for now just leaving it in
                qValue = self.getQValue(state, legalAction)
                # print(maxValue, qValue)
                # nextState = self.getSuccessor(state, legalAction)
                #  reward should be calculated here based on the score
                #  of current state and next state potentially right now just 0
                # reward = self.getScore(state)
                #  calling update here:
                #  this should probably be commented out for when
                #  we submit our agent
                if (maxValue < qValue):
                    maxValue = qValue
                    action = legalAction
        # nextState = self.getSuccessor(state, action)
        # reward = self.getScore(state)
        # self.update(state, action, nextState, reward)
        return action

    def getAction(self, state):
        """
        This is the function we're supposed to override to
        return a legal action.
        """

        # if self.actionList:
        #     self.update(state, self.actionList[-1], self.getScore(state))
        legalActions = state.getLegalActions(self.index)
        if not legalActions:
            return None
        else:
            if probability.flipCoin(self.getEpsilon()):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)
        # self.actionList.append(action)
        # self.update(state, action, self.getScore(state))
        return action

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            pass

    def getQValue(self, state, action):
        # get all features for the state action pair
        features = self.attack.getFeatures(state, action)
        print(features)
        # dot product of features and their assigned weights
        return self.weights * features

    # this function updates all weights for features
    def update(self, state, action, newState, reward):
        # get all features for the state action pair
        features = self.attack.getFeatures(state, action)
        alpha = self.getAlpha()  # learning rate
        discountRate = self.getDiscountRate()  # discount rate
        new_val = self.getValue(newState)  # value of next state
        q_val = self.getQValue(state, action)  # Q-value
        correction = (reward + (discountRate * new_val)) - q_val

        # update weights for all features
        for feature in features:
            self.weights[feature] += alpha * correction * features[feature]

    # def update(self, state, action, newState, reward):
    #     alpha = self.getAlpha()  # learning rate
    #     discountRate = self.getDiscountRate()  # discount rate
    #     val = self.getValue(state)  # value of current state
    #     new_val = self.getValue(newState)  # value of next state
    #     self.q_vals[(state, action)] = (1 - alpha) * val + alpha * (reward + discountRate * new_val)

    # def update(self, state, action, reward):
    #     """
    #     This function updates all the weights for our features
    #     It should be called in getPolicy.
    #     """

    #     """
    #     Formula for the correction value:
    #     w <- w + α[correction]f(s, a)
    #     correction = (R(s,a) + 𝛾V(s')) - Q(s, a)
    #     """
    #     nextState = self.getSuccessor(state, action)
    #     features = self.getFeatures(state, action)
    #     for feature in features:
    #         discount = self.discount
    #         qPrime = self.getValue(nextState)
    #         qValue = self.getQValue(state, action)
    #         alpha = self.alpha
    #         correction = reward + discount  * qPrime - qValue
    #         self.weights[feature] += alpha * correction * features[feature]

    # def getSuccessor(self, state, action):
    #     """
    #     Finds the next successor which is a grid position (location tuple).
    #     ** Coppied from the Reflex Agent class **
    #     """
    #     successor = state.generateSuccessor(self.index, action)
    #     pos = successor.getAgentState(self.index).getPosition()

    #     if (pos != util.nearestPoint(pos)):
    #         # Only half a grid position was covered.
    #         return successor.generateSuccessor(self.index, action)
    #     else:
    #         return successor
    """
    Not sure what to do with this function yet
    Though it is called at the end of the game
    So our agents should override this
    and probably print it to a file
    def final(self, state):
    """

    class AttackAgent(CaptureAgent):
        """
        Extends QLAgent to implement Attack Agent
        # TODO: override registerInitialState(self, state):
                override getFeatures(self, state, action)
        """

        def __init__(self, index, **kwargs):
            super().__init__(index)

        def registerInitialState(self, state):
            """
            This function is called to set up our Agents
            So make sure our two agents override this and call
            super().regiserInitialState(state)
            """
            super().registerInitialState(state)

        def getSuccessor(self, gameState, action):
            """
            Finds the next successor which is a grid position (location tuple).
            """

            if gameState is not None:
                successor = gameState.generateSuccessor(self.index, action)
                pos = successor.getAgentState(self.index).getPosition()

                if (pos != util.nearestPoint(pos)):
                    # Only half a grid position was covered.
                    if successor is not None:
                        return successor.generateSuccessor(self.index, action)
                    else:
                        return None
                else:
                    return successor
            else:
                return None

        def getFeatures(self, state, action):
            # food = self.getFood(state)
            # walls = state.getWalls()
            # ghosts = self.getOpponents(state)
            # ghost_pos = [state.getAgentState(g).getPosition() for g in ghosts]

            # features = counter.Counter()

            # features["bias"] = 1.0

            # # Compute the location of pacman after he takes the action.
            # x, y = state.getAgentState(self.index).getPosition()
            # dx, dy = Actions.directionToVector(action)
            # next_x, next_y = int(x + dx), int(y + dy)

            # # Count the number of ghosts 1-step away.
            # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in
            #         Actions.getLegalNeighbors(g, walls) for g in ghost_pos)

            # # If there is no danger of ghosts then add the food feature.
            # if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            #     features["eats-food"] = 1.0


            # foodList = self.getFood(state).asList()

            # # This should always be True, but better safe than sorry.
            # if (len(foodList) > 0):
            #     myPos = (x, y)
            #     minDistance = min([distance.manhattan(myPos, food_pos) for food_pos in foodList])
            #     features['distanceToFood'] = minDistance / (walls.getWidth() * walls.getHeight())
            # # prob = AnyFoodSearchProblem(state, start = (next_x, next_y))
            # # dist = len(search.bfs(prob))
            # # if dist is not None:
            # #     # Make the distance a number less than one otherwise the update will diverge wildly.
            # #     features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())

            # features.divideAll(10.0)
            # return features

            features = counter.Counter()
            successor = self.getSuccessor(state, action)
            if not successor:
                return features
            agentState = successor.getAgentState(self.index)
            agentPos = agentState.getPosition()
            # gameScore feature
            features["score"] = self.getScore(successor)

            # FEATURE: distance to nearestfood
            foodList = self.getFood(successor).asList()
            features['num_opponents_food_left'] = len(foodList)
            features['num_my_food_left'] = len(self.getFoodYouAreDefending(successor).asList())
            distanceToFood = min([distance.manhattan(agentPos, food) for food in foodList])
            features["distanceToFood"] = 1/(distanceToFood+1)

            # FEATURE: distance to defenders
            # We want to run away from Braveghost if they're too close
            # We're only scared braveghost so we check enemy.isBraveGhost()
            # amount of enemy defenders
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            defenders = [enemy for enemy in enemies if enemy.isBraveGhost() \
                        and enemy.getPosition() is not None]
            if (len(defenders) > 0):  # only take this in account if there are defenders
                dists = [distance.manhattan(agentPos, defender.getPosition()) for defender in defenders]
                distanceToDefender = min(dists)
                # Only need to be very scared and run away if it's enemy's defender is too close
                features["defenderDistance"] = 1 / (2 ** (2 - distanceToDefender))

            # FEATURE: Power capsules
            # We prioritize getting power capsules if there are more than 0 defenders
            # note: there aren't too many capsules on the map, so most of the time it's 0
            capsuleList = self.getCapsules(successor)
            if (len(capsuleList) > 0): 
                if (len(defenders) > 0):
                    dists = [distance.manhattan(agentPos, capsule) for capsule in capsuleList]
                    distanceToCapsule = min(dists)
                    features["capsules"] = 1 / (distanceToCapsule+1)
                else:
                    features["capsules"] = -15

            # FEATURE: stop
            # We want to avoid stopping if possible
            if (action == Directions.STOP):
                features["stop"] = 1

            # FEATURE: isPacman
            # We want our attack agent to go straight to the enemy base
            # So we do negative weights when we're in our own base
            if not agentState.isPacman():
                features["isPacman"] = 1
            """
            # FEATURE: defense assist
            # We want to assist our defense agent when we get sent back to our base
            # note: weight shouldn't be too large, but once our base is cleared 
            # these weights will be 0, so it wouldn't be too much of a problem
            attackers = [enemy for enemy in enemies if enemy.isPacman() and enemy.getPosition() is not None]
            if (len(attackers) > 0 and agentState.isBraveGhost()):
                dists = [self.getMazeDistance(agentPos, attacker.getPosition()) for attacker in attackers]
                distanceToAttacker = min(dists)
                features["defenseAssist"] = distanceToAttacker
            """
            return features

        def chooseAction(self, state):
            """
            This is the function we're supposed to override to
            return a legal action.
            """
            obj = QLAgent(self.index)
            return obj.getAction(state)

class AttackAgent(CaptureAgent):
    """
    Extends QLAgent to implement Attack Agent
    # TODO: override registerInitialState(self, state):
            override getFeatures(self, state, action)
    """

    # def __init__(self, index, **kwargs):
    #     super().__init__(index)

    def getFeatures(self, state, action):
        print('here')
        food = self.getFood(state)
        walls = state.getWalls()
        ghosts = self.getOpponents(state)
        ghost_pos = [state.getAgentState(g).getPosition() for g in ghosts]

        features = counter.Counter()

        features["bias"] = 1.0

        # Compute the location of pacman after he takes the action.
        x, y = state.getAgentState(self.index).getPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # Count the number of ghosts 1-step away.
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in
                Actions.getLegalNeighbors(g, walls) for g in ghost_pos)

        # If there is no danger of ghosts then add the food feature.
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0


        foodList = self.getFood(state).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = (x, y)
            minDistance = min([self.getMazeDistance(myPos, food_pos) for food_pos in foodList])
            features['distanceToFood'] = minDistance / (walls.getWidth() * walls.getHeight())
        # prob = AnyFoodSearchProblem(state, start = (next_x, next_y))
        # dist = len(search.bfs(prob))
        # if dist is not None:
        #     # Make the distance a number less than one otherwise the update will diverge wildly.
        #     features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())

        features.divideAll(10.0)
        return features

    def chooseAction(self, state):
        """
        This is the function we're supposed to override to
        return a legal action.
        """
        obj = QLAgent(self.index)
        return obj.getAction(state)
        


class DefenseAgent(QLAgent):
    """
    Extends QLAgent to implement Defense Agent
    # TODO: override registerInitialState(self, state)
            override getFeatures(self, state, action)
    """