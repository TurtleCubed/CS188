# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        # score = 0

        # make Pacman avoid ghosts
        for pos in successorGameState.getGhostPositions():
            if util.manhattanDistance(pos, newPos) <= 1:
                score -= 100

        # make Pacman move generally towards food and grab adjacent food
        nearestDist = 1e9
        for foodPos in newFood.asList():
            dist = util.manhattanDistance(foodPos, newPos)
            if dist < nearestDist:
                nearestDist = dist
            if foodPos == newPos:
                score += 50
        score += 1/nearestDist

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        value, move = self.pacminimax(0, 0, gameState)
        return move

    def pacminimax(self, agentNum, treeDepth, gameState: GameState):
        agentNum = agentNum % gameState.getNumAgents()

        # Check if the node is terminal -> win/lose or max depth
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None
        if treeDepth == self.depth:
            return self.evaluationFunction(gameState), None

        # Maximize for Pacman and minimize for the ghosts
        newTreeDepth = treeDepth
        if agentNum == gameState.getNumAgents() - 1:
            newTreeDepth = treeDepth + 1
        if agentNum == 0:
            v = float('-inf')
            move = None
            for la in gameState.getLegalActions(agentNum):
                v2, a2 = self.pacminimax(agentNum + 1, newTreeDepth, gameState.generateSuccessor(agentNum, la))
                if v2 > v:
                    v = v2
                    move = la
        else:
            v = float('inf')
            move = None
            for la in gameState.getLegalActions(agentNum):
                v2, a2 = self.pacminimax(agentNum + 1, newTreeDepth, gameState.generateSuccessor(agentNum, la))
                if v2 < v:
                    v = v2
                    move = la
        return v, move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value, move = self.pacAB(0, 0, gameState, float('-inf'), float('inf'))
        return move

    def pacAB(self, agentNum, treeDepth, gameState: GameState, alpha, beta):
        agentNum = agentNum % gameState.getNumAgents()

        # Check if the node is terminal -> win/lose or max depth
        if gameState.isLose() or gameState.isWin() or treeDepth == self.depth:
            return self.evaluationFunction(gameState), None

        newTreeDepth = treeDepth
        if agentNum == gameState.getNumAgents() - 1:
            newTreeDepth = treeDepth + 1

        # Maximize for Pacman and minimize for the ghosts
        if agentNum == 0:
            v, move = float('-inf'), None
            for la in gameState.getLegalActions(agentNum):
                v2, a2 = self.pacAB(agentNum + 1, newTreeDepth, gameState.generateSuccessor(agentNum, la), alpha, beta)
                if v2 > v:
                    v, move = v2, la
                    alpha = max(alpha, v)
                if v > beta:
                    return v, move
            return v, move
        else:
            v, move = float('inf'), None
            for la in gameState.getLegalActions(agentNum):
                v2, a2 = self.pacAB(agentNum + 1, newTreeDepth, gameState.generateSuccessor(agentNum, la), alpha, beta)
                if v2 < v:
                    v, move = v2, la
                    beta = min(beta, v)
                if v < alpha:
                    return v, move
            return v, move


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        value, move = self.expectimax(0, 0, gameState)
        return move

    def expectimax(self, agentNum, treeDepth, gameState: GameState):
        agentNum = agentNum % gameState.getNumAgents()

        # Check if the node is terminal -> win/lose or max depth
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None
        if treeDepth == self.depth:
            return self.evaluationFunction(gameState), None

        # Maximize for Pacman and minimize for the ghosts
        newTreeDepth = treeDepth
        if agentNum == gameState.getNumAgents() - 1:
            newTreeDepth = treeDepth + 1

        if agentNum == 0:
            v = -1e9
            move = None
            for la in gameState.getLegalActions(agentNum):
                v2, a2 = self.expectimax(agentNum + 1, newTreeDepth, gameState.generateSuccessor(agentNum, la))
                if v2 > v:
                    v = v2
                    move = la
        else:
            v = 0
            value_list = []
            move = None
            for la in gameState.getLegalActions(agentNum):
                v2, a2 = self.expectimax(agentNum + 1, newTreeDepth, gameState.generateSuccessor(agentNum, la))
                value_list.append(v2)
            v = sum(value_list)/len(value_list)
                    
        return v, move

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    score = currentGameState.getScore()
    # score = 0

    # Make Pacman avoid ghosts
    for i in range(1, currentGameState.getNumAgents()):
        if util.manhattanDistance(pos, currentGameState.getGhostPosition(i)) < 1:
            score -= 100
    
    # make Pacman move generally towards food and grab adjacent food
    nearestDist = float('inf')
    for foodPos in food.asList():
        # dist = (foodPos[0]-pos[0])**2 + (foodPos[1] - pos[1])**2
        dist = util.manhattanDistance(foodPos, pos)
        if dist < nearestDist:
            nearestDist = dist
    score += 1 / nearestDist
    # print(currentGameState.getPacmanState().getDirection())

    return score

# Abbreviation
better = betterEvaluationFunction
