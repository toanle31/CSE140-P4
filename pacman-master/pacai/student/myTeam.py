from pacai.util import reflection
from pacai.agents.capture.capture import CaptureAgent
from pacai.util import counter
from pacai.util import probability
from pacai.util import util
from pacai.util import stack, queue, priorityQueue
# from pacai.student.qlearningAgents import ApproximateQAgent
from pacai.core.actions import Actions
import random

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    # firstAgent = reflection.qualifiedImport(first)
    # secondAgent = reflection.qualifiedImport(second)
    firstAgent = MyAgent
    secondAgent = MyAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class MyAgent(CaptureAgent):

    # def aStarSearch(problem, heuristic):
    #     """
    #     Search the node that has the lowest combined cost and heuristic first.
    #     """

    #     # *** Your Code Here ***
    #     state_pq = priorityQueue.PriorityQueue()
    #     visited = []
    #     start_state = problem.startingState()
    #     # initial state pushed to priority queue
    #     state_pq.push((start_state, [], 0), heuristic(start_state, problem))

    #     while not state_pq.isEmpty():
    #         curr_state, action_list, total_cost = state_pq.pop()
    #         if problem.isGoal(curr_state):  # if goal is reached return list of actions to take
    #             return action_list

    #         if curr_state not in visited:
    #             for succ in problem.successorStates(curr_state):
    #                 state, action, cost = succ

    # # push all non-visited successor states to queue and update action list,
    # # action cost and get estimated action cost using heuristic function
    #                 if state not in visited:
    #                     state_pq.push((state, action_list + [action], total_cost + cost),
    #                     problem.actionsCost(action_list + [action]) + heuristic(state, problem))
    #         visited.append(curr_state)

    #     return action_list

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """
        food = self.getFood(gameState)
        walls = gameState.getWalls()
        ghosts = self.getOpponents(gameState)
        ghost_pos = [gameState.getAgentState(g).getPosition() for g in ghosts]

        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        next_x, next_y = successor.getAgentState(self.index).getPosition()
        next_x, next_y = int(next_x), int(next_y)

        # Count the number of ghosts 1-step away.
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in
                Actions.getLegalNeighbors(g, walls) for g in ghost_pos)

        # If there is no danger of ghosts then add the food feature.
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = 1 / minDistance

        return features

    def getWeights(self, gameState, action):
        """
        Returns a dict of weights for the state.
        The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
        """

        return {
            'successorScore': 10.0,
            'distanceToFood': 5.0,
            '#-of-ghosts-1-step-away': -100,
            'eats-food': 0.0
        }