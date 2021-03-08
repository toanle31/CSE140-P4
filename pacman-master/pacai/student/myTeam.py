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
    defenseAgent = DefenseAgent(firstIndex) 
    attackAgent = AttackAgent(secondIndex)
    return [
        defenseAgent,
        attackAgent
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

        # FEATURE: distance to nearestfood
        foodList = self.getFood(successor).asList()
        distanceToFood = min([self.getMazeDistance(agentPos, food) for food in foodList])
        features["distanceToFood"] = distanceToFood

        # FEATURE: distance to defenders
        # We want to run away from Braveghost if they're too close
        # We're only scared braveghost so we check enemy.isBraveGhost()
        # amount of enemy defenders
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        defenders = [enemy for enemy in enemies if enemy.isBraveGhost() \
                    and enemy.getPosition() is not None]
        if (len(defenders) > 0):  # only take this in account if there are defenders
            dists = [self.getMazeDistance(agentPos, defender.getPosition()) for defender in defenders]
            distanceToDefender = min(dists)
            # Only need to be very scared and run away if it's enemy's defender is too close
            features["defenderDistance"] = distanceToDefender if distanceToDefender < 3 else 0

        # FEATURE: Power capsules
        # We prioritize getting power capsules if there are more than 1 defenders
        # note: there aren't too many capsules on the map, so most of the time it's 0
        capsuleList = self.getCapsules(successor)
        if (len(capsuleList) > 0 and len(defenders) > 0): 
            distanceToCapsule = min([self.getMazeDistance(agentPos, capsule) for capsule in capsuleList])
            features["capsules"] = distanceToCapsule

        # FEATURE: stop
        # We want to avoid stopping if possible
        if (action == Directions.STOP):
            features["stop"] = 1

        # FEATURE: reverse
        # We want to avoid reversing if possible
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == reverse):
            features["reverse"] = 1

        # FEATURE: isPacman
        # We want our attack agent to go straight to the enemy base
        # So we do negative weights when we're in our own base
        # note: this weight shouldn't be too large because we want to
        # also help our defender sometimes.
        if not agentState.isPacman():
            features["isPacman"] = 1

        # FEATURE: defense assist
        # We want to assist our defense agent when we get sent back to our base
        # note: weight shouldn't be too large, but once our base is cleared 
        # these weights will be 0, so it wouldn't be too much of a problem
        attackers = [enemy for enemy in enemies if enemy.isPacman() and enemy.getPosition() is not None]
        if (len(attackers) > 0 and agentState.isBraveGhost()):
            dists = [self.getMazeDistance(agentPos, attacker.getPosition()) for attacker in attackers]
            distanceToAttacker = min(dists)
            features["defenseAssist"] = distanceToAttacker

        return features

    # these weights can use improvements
    def getWeights(self, gameState, action):
        return {
            "score": 100,
            "distanceToFood": -2,
            "defenderDistance": 8.5,
            "capsules": -2.5,
            "stop": -50,
            "reverse": -2,
            "isPacman": -45,
            "defenseAssist": -65
        }

class DefenseAgent(ReflexCaptureAgent):
    # Right now this is just the provided ReflexDefenseAgent
    # remove these comments when done implementing the agent
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        agentState = successor.getAgentState(self.index)
        agentPos = agentState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (agentState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(agentPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1
        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2
        }