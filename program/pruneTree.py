from typing import List


def prune(self, threshold: str):
    for item in self.nodeMap:
        self.nodeMap[item].promising_branches = list(range(2))
        for i in range(2):
            if self.nodeMap[item].regretSum[i] < threshold:
                self.nodeMap[item].promising_branches.remove(i)