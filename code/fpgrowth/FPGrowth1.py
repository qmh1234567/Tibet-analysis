import os
import sys
import FPTree

class FPGrowth1:
    def __init__(self, minsup=2):
        self.fp = []
        self.minsup = minsup
    
    def growth(self, tree, postNodes):
        if tree.isUniquePath():
            nodeCombinations = []
            tree.getCombinationFromPath(nodeCombinations)
            for combination in nodeCombinations:
                support = self._getMinSupport(combination)
                if support is None or support < self.minsup:
                    continue
                #gen pattern
                pattern = ([],support)
                for node in combination:
                    pattern[0].append(node["name"])
                for node in postNodes:
                    pattern[0].append(node)
                if len(pattern[0]) > 1:
                    self.fp.append(pattern)
                    #self._printPattern(pattern)
        else:
            for item in tree.itemTable:
                #gen pattern
                pattern = ([],tree.itemTable[item][0])
                pattern[0].append(item)
                for node in postNodes:
                    pattern[0].append(node)
                if len(pattern[0]) > 1 and pattern[1] > self.minsup: 
                    self.fp.append(pattern)  
                    #self._printPattern(pattern)
                #construct conditional pattern base
                baseSet = []
                tree.getConditionalPatternBase(item,baseSet)  
                tmpTree = FPTree.FPTree(baseSet, minsup=self.minsup) 
                tmpTree.build()
                if not tmpTree.isEmpty():
                    self.growth(tmpTree, pattern[0])       
            
    def _getMinSupport(self, nodes):
        if len(nodes) == 0:
            return None
        support = nodes[0]["support"]
        for node in nodes:
            if node["support"] < support:
                support = node["support"]
        return support
    
    def _printPattern(self, pattern):
        if len(pattern[0]) < 2:
            return
        print("*******************")
        print(pattern[0])
        print(pattern[1])
        print("*******************")   



            
        