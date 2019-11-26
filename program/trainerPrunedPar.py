from typing import List
import random, pickle
from os import getcwd
import time
from DudoTrainer import cfr
from DudoUtil import readNodeMap, gameValue
import multiprocessing

def continueTrain(file, iterations: int, savePath):
    global nodeMap
    nodeMap = readNodeMap(file)
    train(iterations)
    # Save the trained algorithm
    with open(savePath, 'wb') as f:
        pickle.dump(nodeMap, f)

def train(iterations: int):
    t1 = time.time()
    util = 0
    with multiprocessing.Pool() as pool:
        m = multiprocessing.Manager()
        d = m.dict(nodeMap)
        for i in range(iterations):
            rr = random.random()
            # Sample an outcome of roll. First one is self rolled, second is opponent.
            rolledDice = [random.randint(1, 6), random.randint(1, 6)]
            if rr < .95:
                util += cfrPrunedPar(rolledDice, [str(rolledDice[0])], 1, 1, pool, d, m)
            else:
                util += cfr(rolledDice, [str(rolledDice[0])], 1, 1)
            # Reset strategy sum
            # if iterations == 0:
            #     resetSS(nodeMap)

            # Progress
            print(i)
            if i % (10000) == 0:
                print(f"Dudo trained {i} iterations. {str(10000 / (time.time() - t1))} iterations per second.")
                print("Theoretical game value: " + str(gameValue(nodeMap)))
                t1 = time.time()
        for key in d:
            nodeMap[key] = d[key]

def cfrPruned(rolledDice: List[float], infoSet: List[str], p0: float, p1: float) -> float:
    '''
    Cfr with pruning.

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
    # This gets the current strategy based on regretSum,
    # also adds the regret sum to the cumulative regretSum

    strategy = curr_node.getStrategy(realization_weight)

    # nodeUtil is the weighted average of the cfr of each branch,
    # weighted by the probability of traversing down a branch
    nodeUtil = 0

    NUM_ACTIONS = len(curr_node.regretSum)
    # This is the cf-utility of each subsequent choice.
    util = [0] * NUM_ACTIONS

    # For each action, recursively call cfr with additional history and probability
    for a in curr_node.promising_branches:
        nextIS = [str(rolledDice[other_player])] + infoSet[1:] + [curr_node.children[a]]
        # The first probability is player 1's counterfactual probability
        if curr_player == 0:
            util[a] = -cfrPruned(rolledDice, nextIS, p0 * strategy[a], p1)
        # Current player is 1
        else:
            util[a] = -cfrPruned(rolledDice, nextIS, p0, p1 * strategy[a])
        nodeUtil += strategy[a] * util[a]

    # For each action, compute and accumulate counterfactual regret
    for a in curr_node.promising_branches:
        regret = util[a] - nodeUtil
        curr_node.regretSum[a] += (p1 if curr_player == 0 else p0) * regret

    return nodeUtil

def cfrPrunedPar(rolledDice: List[float], infoSet: List[str], p0: float, p1: float, pool, d, m) -> float:
    '''
    Cfr with pruning paralleled in all iterations.

    '''
    plays = len(infoSet) - 1
    curr_player = plays % 2
    other_player = 1 - curr_player

    curr_node = d[str(infoSet)]
    curr_node.times_visited += 1
    # Return Payoff for terminal nodes.
    if curr_node.isTerminal():
        return curr_node.returnPayoff(rolledDice)

    realization_weight = p1 if curr_player == 0 else p0
    # This gets the current strategy based on regretSum,
    # also adds the regret sum to the cumulative regretSum

    strategy = curr_node.getStrategy(realization_weight)

    nodeUtil = 0
    NUM_ACTIONS = len(curr_node.regretSum)
    # This is the cf-utility of each subsequent choice.
    util = [0] * NUM_ACTIONS

    # For each action, recursively call cfr with additional history and probability
    # for a in curr_node.promising_branches:
    #     nextIS = [str(rolledDice[other_player])] + infoSet[1:] + [curr_node.children[a]]
    #     nodeUtil_a, util_a = cfrRecursiveWrapper(curr_node, rolledDice, nextIS, p0, p1, strategy[a], nodeUtil)
    #     nodeUtil += nodeUtil_a
    #     util[a] = util_a

    args = []
    d2 = m.dict()
    for a in curr_node.promising_branches:
        nextIS = [str(rolledDice[other_player])] + infoSet[1:] + [curr_node.children[a]]
        args.append([curr_player, rolledDice, nextIS, p0, p1, strategy[a], d, d2, a])

    pool.starmap(cfrRecursiveWrapper, args)

    nodeUtil = sum(d2[a][0] for a in d2)
    # For each action, compute and accumulate counterfactual regret
    for a in curr_node.promising_branches:
        regret = d2[a][1] - nodeUtil
        curr_node.regretSum[a] += (p1 if curr_player == 0 else p0) * regret

    return nodeUtil

def cfrRecursiveWrapper(curr_player, rolledDice: List[float], nextIS: List[str],
                        p0: float, p1: float, strategy_a, dict, d2, a):
    # The first probability is player 1's counterfactual probability
    if curr_player == 0:
        util_a = -cfrPrunedrec(rolledDice, nextIS, p0 * strategy_a, p1, dict)
    # Current player is 1
    else:
        util_a = -cfrPrunedrec(rolledDice, nextIS, p0, p1 * strategy_a, dict)
    nodeUtil_a = strategy_a * util_a
    d2[a] = [nodeUtil_a, util_a]


def cfrPrunedrec(rolledDice: List[float], infoSet: List[str], p0: float, p1: float, d) -> float:
    '''
    Cfr with pruning.

    '''
    plays = len(infoSet) - 1
    curr_player = plays % 2
    other_player = 1 - curr_player

    curr_node = d[str(infoSet)]
    curr_node.times_visited += 1
    # Return Payoff for terminal nodes.
    if curr_node.isTerminal():
        return curr_node.returnPayoff(rolledDice)

    realization_weight = p1 if curr_player == 0 else p0

    # This gets the current strategy based on regretSum,
    # also adds the regret sum to the cumulative regretSum

    strategy = curr_node.getStrategy(realization_weight)

    # nodeUtil is the weighted average of the cfr of each branch,
    # weighted by the probability of traversing down a branch
    nodeUtil = 0

    NUM_ACTIONS = len(curr_node.regretSum)
    # This is the cf-utility of each subsequent choice.
    util = [0] * NUM_ACTIONS

    # For each action, recursively call cfr with additional history and probability
    for a in curr_node.promising_branches:
        nextIS = [str(rolledDice[other_player])] + infoSet[1:] + [curr_node.children[a]]
        # The first probability is player 1's counterfactual probability
        if curr_player == 0:
            util[a] = -cfrPrunedrec(rolledDice, nextIS, p0 * strategy[a], p1, d)
        # Current player is 1
        else:
            util[a] = -cfrPrunedrec(rolledDice, nextIS, p0, p1 * strategy[a], d)
        nodeUtil += strategy[a] * util[a]

    # For each action, compute and accumulate counterfactual regret
    for a in curr_node.promising_branches:
        regret = util[a] - nodeUtil
        curr_node.regretSum[a] += (p1 if curr_player == 0 else p0) * regret

    return nodeUtil

if __name__ == '__main__':
    start_time = time.time()
    cwd = getcwd()
    continueTrain(cwd + '/trainedTrees/Discounted/dt-500kDcPruned', 100, cwd + '/trainedTrees/Discounted/dt-1MCons1')
    print("--- %s seconds ---" % (time.time() - start_time))