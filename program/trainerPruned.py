from typing import List
import random, pickle
from os import getcwd
import time
from DudoTrainer import cfr
from DudoUtil import readNodeMap, gameValue
import multiprocessing

def continueTrain(file, iterations: int, savePath, log_path):
    global nodeMap
    nodeMap = readNodeMap(file)
    train(iterations, savePath, log_path)
    # Save the trained algorithm


def train(iterations: int, savePath, log_path):
    log = ""
    t1 = time.time()
    util = 0
    for i in range(1, iterations):
        rr = random.random()
        # Sample an outcome of roll. First one is self rolled, second is opponent.
        rolledDice = [random.randint(1, 6), random.randint(1, 6)]
        if rr < .95:
            util += cfrPruned(rolledDice, [str(rolledDice[0])], 1, 1)
        else:
            util += cfr(rolledDice, [str(rolledDice[0])], 1, 1)
        # Reset strategy sum
        # if iterations == 0:
        #     resetSS(nodeMap)

        # Progress
        print_freq = 10000
        if i % (print_freq) == 0:
            print(f"Dudo trained {i} iterations. {str(print_freq / (time.time() - t1))} iterations per second.")
            print("Theoretical game value: " + str(gameValue(nodeMap)))
            log += f"Dudo trained {i} iterations. {str(print_freq / (time.time() - t1))} iterations per second. \n"
            log += "Theoretical game value: " + str(gameValue(nodeMap)) + "\n"
            t1 = time.time()
            # print("Theoretical game value: " + str(gameValue(nodeMap)))
        # if i % (10 ** 6) == 0:
        #     name_log = f"log-dt500kDcPruned{i}its"
        #     with open(savePath, 'wb') as f:
        #         pickle.dump(nodeMap, f)
        #     with open(name_log, 'wb') as f:
        #         pickle.dump(log, f)

    with open(savePath, 'wb') as f:
        pickle.dump(nodeMap, f)
    with open(log_path, 'w') as f:
        f.write(log)

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

if __name__ == '__main__':
    start_time = time.time()
    cwd = getcwd()
    continueTrain(cwd + '/trainedTrees/Discounted/dt-5.5MPr', 2.5 * 10 ** 6, cwd + '/trainedTrees/Discounted/dt-8MPr', cwd + '/trainedTrees/logs/logdt-8MPr.txt')
    print("--- %s seconds ---" % (time.time() - start_time))