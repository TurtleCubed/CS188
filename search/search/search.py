# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from pydoc import doc
from sre_constants import FAILURE
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

# state = coords (state) , action, path-cost
class Node: 
    def __init__(self, state=[], parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __str__(self):
        return f'Node({self.state}, {self.parent}, {self.action}, {self.path_cost})'

    def __repr__(self):
        return f'Node({self.state}, {self.parent}, {self.action}, {self.path_cost})'


def bestFirstSearch(problem, f):
    node = Node(state=problem.getStartState())
    frontier = util.PriorityQueue()
    frontier.update(node, f(node))
    # frontier = util.Stack()
    # frontier.push(node)
    reached = {node.state: node}
    while not frontier.isEmpty():
        node = frontier.pop()
        # print(node.state)
        if problem.isGoalState(node.state):
            return node
        # reached[node.state] = node
        # if not is_cycle(node):
        for child in expand(problem, node):
            s = child.state
            # print(s, child.path_cost)
            if (s not in reached) or (child.path_cost < reached[s].path_cost):
                reached[s] = child
                # frontier.push(child)
                frontier.update(child, f(child))

    return FAILURE


def expand(problem, node):
    s = node.state
    for successor in problem.getSuccessors(s):
        state = successor[0]
        action = successor[1]
        action_cost = successor[2]
        s_prime = state
        cost = node.path_cost + action_cost 
        yield Node(state=s_prime, parent=node, action=action, path_cost=cost)


def getPath(node): 
    path = []
    while node.action:
        path.insert(0, node.action)
        node = node.parent
    return path


def is_cycle(node): 
    seen = set()
    while node.action:
        if node.state in seen: return True
        seen.add(node.state)
        node = node.parent
    return False


def getDepth(node): 
    depth = 0
    while node.action:
        depth += 1 
        node = node.parent
    return depth

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # node = problem.getStartState()
    # frontier = util.PriorityQueue()
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(node))
    def f(node):
        return -getDepth(node)

    node = bestFirstSearch(problem, f)
    path = getPath(node) if node else [] 
    return path

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    def f(node):
        return getDepth(node)

    node = bestFirstSearch(problem, f)
    path = getPath(node) if node else [] 
    return path

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def f(node):
        return node.path_cost

    node = bestFirstSearch(problem, f)
    path = getPath(node) if node else [] 
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def f(node):
        return node.path_cost + heuristic(node.state, problem)

    node = bestFirstSearch(problem, f)
    path = getPath(node) if node else [] 
    return path

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
