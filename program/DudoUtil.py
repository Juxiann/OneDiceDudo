import pickle
from typing import List

def createEmptyTree() -> dict:
    '''
    Creates an empty tree for 1DD game.
    '''
    tree = dict()
    rolled = ['1', '2', '3', '4', '5', '6']
    def createEmptyTreeRecursive(tree, infoSet):
        from DudoNode import DudoNode
        node = DudoNode()
        node.infoSet = infoSet
        available = node.availableChoices()
        if len(infoSet) == 1:
            available.remove('d')
        NUM_ACTIONS = len(available)
        node.regretSum, node.strategySum, node.strategy = [0] * NUM_ACTIONS, [0] * NUM_ACTIONS, [0] * NUM_ACTIONS
        # Careful not to use the following code as it maps the three items to the same list.
        # Changing the list will change all three!
        # node.regretSum, node.strategySum, node.strategy = [[0] * NUM_ACTIONS] * 3
        node.children = available
        tree[str(infoSet)] = node

        for nextAction in available:
            newIS = infoSet + [nextAction]
            createEmptyTreeRecursive(tree, newIS)
        return tree

    for number in rolled:
        createEmptyTreeRecursive(tree, [number])
    return tree


def rankCount(rolled: List[int]) -> List[int]:
    '''
    According to the rolled dice, returns the number of each roll (from 1 to 6)
    A roll of 1 is considered 'wild' and counts as any number.
    rolled: List of rolled dice.
    >>> rankCount([1, 4, 4, 2])
    [1, 2, 1, 3, 1, 1]
    >>> rankCount([1, 4])
    [1, 1, 1, 2, 1, 1]
    '''
    output = [0] * 6
    for roll in rolled:
        if roll == 1:
            output = [x + 1 for x in output]
        else:
            output[roll - 1] += 1
    return output

def gameValue(nodeMap):
    '''
    Each terminal node profit multiplied by its probability.
    :return:

    '''
    value = 0
    diceList = []
    for i in range(1, 7):
        for j in range(1, 7):
            diceList.append([i, j])
    for dice in diceList:
        def valueRecursive(infoSet: List[str]) -> float:
            curr_node = nodeMap[str(infoSet)]
            if curr_node.isTerminal():
                return curr_node.returnPayoff(dice)
            # Not a terminal node
            else:
                curr_player = (len(infoSet) - 1) % 2
                other = 1 - curr_player
                otherInfo = [str(dice[other])] + infoSet[1:]
                strategy = curr_node.getAverageStrategy()
                value = 0
                for i in range(len(curr_node.children)):
                    value += -valueRecursive(otherInfo + [curr_node.children[i]]) * strategy[i]
                return value
        value += valueRecursive([str(dice[0])])
    value = value / 36
    return value

def resetSS(nodeMap: dict):
    '''
    Resets the strategySum of all nodes in nodeMap to zero
    :return:
    '''
    for item in nodeMap:
        nodeMap[item].strategySum = [0] * len(nodeMap[item].strategySum)


def prune(nodeMap: dict, threshold: int):
    for item in nodeMap:
        NUM_ACTIONS = len(nodeMap[item].regretSum)
        nodeMap[item].promising_branches = list(range(NUM_ACTIONS))
        for i in range(NUM_ACTIONS):
            if nodeMap[item].regretSum[i] < threshold:
                nodeMap[item].promising_branches.remove(i)

def readNodeMap(filepath: str):
    with open(filepath, 'rb') as f:
        nodeMap = pickle.load(f)
    return nodeMap

