from pacai.util import reflection
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.core.directions import Directions
from pacai.util import counter

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
class AttackAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        agentState = successor.getAgentState(self.index)
        agentPos = agentState.getPosition()
        # gameScore feature
        features["score"] = self.getScore(successor)
        # distance to nearestfood feature ##EAT
        foodList = self.getFood(successor).asList()
        distanceToFood = min([self.getMazeDistance(agentPos, food) for food in foodList])
        features["distanceToFood"] = distanceToFood
        # amount of enemy defenders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # distance to defenders feature ##RUNAWAY
        # We're only scared braveghost so we check enemy.isBraveGhost()
        defenders = [enemy for enemy in enemies if enemy.isBraveGhost() and enemy.getPosition() is not None]
        if (len(defenders) > 0):  # only take this in account if there are defenders
            dists = [self.getMazeDistance(agentPos, defender.getPosition()) for defender in defenders]
            features["defenderDistance"] = min(dists)
        # stop feature, we don't want to stop if we can help it
        if (action == Directions.STOP):
            features["stop"] = 1

        # Power capsules feature, we want to prioritize capsules if
        #   there are more than one BraveGhost
        #   length of defenders should be 0 if there are no BraveGhost
        #   meaning we don't care if defenders are scared
        capsuleList = self.getCapsules(successor).asList()
        if (len(capsuleList) > 0 and len(defenders) > 0): 
            distanceToCapsule = min([self.getMazeDistance(agentPos, capsule) for capsule in capsuleList])
            features["capsules"] = distanceToCapsule
        return features
    # these weights can use improvements
    def getWeights(self, gameState, action):
        return {
            "score": 100,
            "distanceToFood": -1.5,
            "stop": -65,
            "capsules": -2.5,
            "defenderDistance": 8
        }
"""
class DefenseAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)
    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        agentState = successor.getAgentState(self.index)
        agentPos = agentState.getPosition()

        return features
    def getWeights(self, gameState, action):
        return {

        }
"""