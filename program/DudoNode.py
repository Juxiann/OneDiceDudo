from typing import List
from DudoUtil import rankCount


class DudoNode():
    # First item in list is the dice rolled, then the claimed number * claimed rank, if 'd' then dudo.
    infoSet: List[str]
    regretSum: List[float]
    strategy: List[float]
    strategySum: List[float]
    children: List
    times_visited: int

    def __init__(self):
        self.children = []
        self.times_visited = 0
        self.count_realization = 0
        self.realization_sum = 0

    def __str__(self):
        return str(self.infoSet) + ' ' + ', '.join(str(e) for e in self.getAverageStrategy())

    def strength(self) -> int:
        '''
        Returns strength of previous play, return -1 if no plays precede, and 12 if 'd' dudo.
        Define '1*2' as strength 0, '1*3' as strength 1, ..., '2*1' as strength 11,
        formula (6) of page 19 of 'cfr' has error
        >>> dn = DudoNode()
        >>> dn.infoSet = ['2', '2*1']
        >>> dn.strength()
        11
        '''
        if len(self.infoSet) <= 1:
            return -1
        else:
            if self.infoSet[-1] == 'd':
                return 12
            number = int(self.infoSet[-1][-3])
            rank = int(self.infoSet[-1][-1])
            if rank != 1:
                return 6 * number + rank - 8
            else:
                return 6 * number - 1

    def availableChoices(self) -> List[str]:
        '''
        Returns a list of all possible actions, depending on the previous player's action.
        >>> dn = DudoNode()
        >>> dn.infoSet = ['2', '1*6']
        >>> dn.availableChoices()
        ['1*1', '2*2', '2*3', '2*4', '2*5', '2*6', '2*1', 'd']
        >>> dn.infoSet = ['2', '1*6', 'd']
        >>> dn.availableChoices()
        []
        '''
        # All options (except dudo), corresponding to strength 0 to 11
        allActions = ['1*2', '1*3', '1*4', '1*5', '1*6', '1*1', '2*2', '2*3', '2*4', '2*5', '2*6', '2*1']
        prevStrength = self.strength()
        if prevStrength == 12:
            return []
        return allActions[prevStrength + 1: ] + ['d']

    def returnPayoff(self, rolledDice: List) -> int:
        '''
        Returns the payoff for terminal nodes, raise error if not a terminal node.
        >>> dn = DudoNode()
        >>> dn.infoSet = ['2', '1*2', '2*3', 'd']
        >>> dn.returnPayoff([3, 1])
        1
        '''
        if not self.isTerminal():
            raise Exception('Not a terminal node.')
        else:
            claim = self.infoSet[-2]
            claimNumber = int(claim[0])
            claimRank = int(claim[2])
            actualCount = rankCount(rolledDice)
            if actualCount[claimRank - 1] >= claimNumber:
                return 1
            else:
                return -1


    def getStrategy(self, realizationWeight: float) -> List[float]:
        normalizingSum = 0
        NUM_ACTIONS = len(self.regretSum)
        for i in range(NUM_ACTIONS):
            if self.regretSum[i] > 0:
                self.strategy[i] = self.regretSum[i]
            else:
                self.strategy[i] = 0
            normalizingSum += self.strategy[i]

        for i in range(NUM_ACTIONS):
            if normalizingSum > 0:
                self.strategy[i] /= normalizingSum
            else:
                self.strategy[i] = 1 / NUM_ACTIONS
            self.strategySum[i] += realizationWeight * self.strategy[i]

        return self.strategy

    def getStrategyDc(self, realizationWeight: float, iterations:int) -> List[float]:
        '''
        Get strategy but with linear discount. See Note02.
        '''
        normalizingSum = 0
        NUM_ACTIONS = len(self.regretSum)
        for i in range(NUM_ACTIONS):
            if self.regretSum[i] > 0:
                self.strategy[i] = self.regretSum[i]
            else:
                self.strategy[i] = 0
            normalizingSum += self.strategy[i]

        for i in range(NUM_ACTIONS):
            if normalizingSum > 0:
                self.strategy[i] /= normalizingSum
            else:
                self.strategy[i] = 1 / NUM_ACTIONS
            self.strategySum[i] += realizationWeight * self.strategy[i]

        return self.strategy

    def getAverageStrategy(self) -> List[float]:
        NUM_ACTIONS = len(self.regretSum)
        avgStrategy = [0] * NUM_ACTIONS
        normalizingSum = sum(self.strategySum)
        for i in range(NUM_ACTIONS):
            if normalizingSum > 0:
                avgStrategy[i] = self.strategySum[i] / normalizingSum
            else:
                avgStrategy[i] = 1 / NUM_ACTIONS
        for a in range(NUM_ACTIONS):
            if avgStrategy[a] < 0.01:
                avgStrategy[a] = 0
        normalizingSum = sum(avgStrategy)
        for a in range(NUM_ACTIONS):
            avgStrategy[a] /= normalizingSum
        return avgStrategy

    def getAverageStrategyDc(self) -> List[float]:
        '''
        getAverageStrategy but with linear discount. See Note02.
        '''
        NUM_ACTIONS = len(self.regretSum)
        avgStrategy = [0] * NUM_ACTIONS
        normalizingSum = sum(self.strategySum)
        for i in range(NUM_ACTIONS):
            if normalizingSum > 0:
                avgStrategy[i] = self.strategySum[i] / normalizingSum
            else:
                avgStrategy[i] = 1 / NUM_ACTIONS
        return avgStrategy

    def accumulateAvgRegret(self, util: List[float], avgNodeUtil: float, realizationWeight: float):
        for a in range(len(self.regretSum)):
            regret = util[a] - avgNodeUtil
            self.regretSum[a] += realizationWeight * regret

    def accumulateAvgRegretDc(self, util: List[float], avgNodeUtil: float, realizationWeight: float, iterations: int):
        for a in range(len(self.regretSum)):
            regret = util[a] - avgNodeUtil
            self.regretSum[a] += realizationWeight * regret
            self.regretSum[a] *= iterations / (iterations + 1)

    def isTerminal(self):
        return self.infoSet[-1] == 'd'
