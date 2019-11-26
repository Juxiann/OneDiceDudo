import random
from typing import *
import pickle
from os import getcwd
from DudoUtil import createEmptyTree, gameValue, resetSS, readNodeMap

nodeMap = createEmptyTree()

def continueTrain(file, iterations: int, savePath):
    global nodeMap, log
    log = []
    nodeMap = readNodeMap(file)
    train(iterations, savePath)


def train(iterations: int, savePath):
    t1 = time.time()
    util = 0
    for i in range(1, iterations):
        # Sample an outcome of roll. First one is self rolled, second is opponent.
        rolledDice = [random.randint(1, 6), random.randint(1, 6)]
        util += cfr(rolledDice, [str(rolledDice[0])], 1, 1)
        # Reset strategy sum
        # if iterations == 0:
        #     resetSS(nodeMap)

        # Progress
        if i % (10000) == 0:
            print(f"Dudo trained {i} iterations. {str(10000 / (time.time() - t1))} iterations per second.")
            print("Theoretical game value: " + str(gameValue(nodeMap)))
            log.append(f"Dudo trained {i} iterations. {str(10000 / (time.time() - t1))} iterations per second.")
            log.append("Theoretical game value: " + str(gameValue(nodeMap)))
            t1 = time.time()
    # print("Theoretical game value: " + str(gameValue(nodeMap)))
    #     if i % (10 ** 6) == 0:
    #         name_log = f"log-dt500kDc{i}"
    #         with open(savePath, 'wb') as f:
    #             pickle.dump(nodeMap, f)
    #         with open(name_log, 'wb') as f:
    #             pickle.dump(log, f)

    # Save the trained algorithm
    with open(savePath, 'wb') as f:
        pickle.dump(nodeMap, f)
    name_log = f"log-dt500kDc3.5M2"
    with open(name_log, 'wb') as f:
        pickle.dump(log, f)

def cfr(rolledDice: List[float], infoSet: List[str], p0: float, p1: float) -> float:
    '''
    Returns the counterfactual regret of the information set.
    p0 is the probability of reaching the state assuming player 1 plays to reach the current state
    p1 is the probability of reaching the state assuming player 0 plays to reach the current state
    '''
    plays = len(infoSet) - 1
    curr_player = plays % 2
    other_player = 1 - curr_player

    curr_node = nodeMap[str(infoSet)]
    curr_node.times_visited += 1
    # Return Payoff for terminal nodes.
    if curr_node.isTerminal():
        return curr_node.returnPayoff(rolledDice)

    realization_weight = p1 if curr_player == 0 else p0

    # if realization_weight == 0:
    #     return 0
    curr_node.count_realization += 1
    curr_node.realization_sum += realization_weight
    # curr_node.realization_sum += realization_weight * (iteration) / (iteration + 1)
    # This gets the current strategy based on regretSum,
    # also adds the regret sum to the cumulative regretSum

    strategy = curr_node.getStrategy(realization_weight)

    # nodeUtil is the weighted average of the cfr of each branch,
    # weighted by the probability of traversing down a branch
    nodeUtil = 0

    NUM_ACTIONS = len(curr_node.children)
    # This is the cf-utility of each subsequent choice.
    util = [0] * NUM_ACTIONS

    # For each action, recursively call cfr with additional history and probability
    for a in range(NUM_ACTIONS):
        nextIS = [str(rolledDice[other_player])] + infoSet[1:] + [curr_node.children[a]]
        # The first probability is player 1's counterfactual probability
        if curr_player == 0:
            util[a] = -cfr(rolledDice, nextIS, p0 * strategy[a], p1)
        # Current player is 1
        else:
            util[a] = -cfr(rolledDice, nextIS, p0, p1 * strategy[a])
        nodeUtil += strategy[a] * util[a]

    # For each action, compute and accumulate counterfactual regret
    # curr_node.accumulateAvgRegret(util, nodeUtil, p1 if curr_player == 0 else p0)
    # curr_node.accumulateAvgRegretDc(util, nodeUtil, p1 if curr_player == 0 else p0, iteration)
    for a in range(NUM_ACTIONS):
        regret = util[a] - nodeUtil
        curr_node.regretSum[a] += realization_weight * regret

    return nodeUtil

if __name__ == '__main__':
    # train(10000)
    # print('long')
    import time
    start_time = time.time()
    cwd = getcwd()
    continueTrain(cwd + '/trainedTrees/Discounted/dudoTrained-500kDc', 3 * 10 **6, cwd + '/trainedTrees/Discounted/dt-3.5MReg2')
    print("--- %s seconds ---" % (time.time() - start_time))
