import os
import sys

#Params:
#1. transSet=[[[itemx,...,itemy],support],...]
#Interfaces:
#1. build()
#2. isUniquePath()
#3. printTree()
#4. getCombinationFromPath(container) #container=[[node,...],...]
#5. itemTable
#6. getConditionalPatternBase(itemName,baseSet)
#7. isEmpty()
class FPTree:
    def __init__(self, transSet, minsup = 50):
        self.transSet = transSet
        self.minsup = minsup
        self._preProcess()
        self.sortedTransSet = self.transSet
        self.itemTable = {} #{itemName:[support, nextNode],...}
        #Node = {"name":,"support":,"children":{name:node},"next":,"parent":node}
        self.tree = {"name":None,"support":0,"children":{},"next":None,"parent":None} 
        self.currNode = self.tree
        self._initItemTable()
    
    def _preProcess(self):
        #compute trans length & statistic frequent items
        typecounter = {}#{typeid:count}
        transLen = []
        for trans in self.transSet:
            transLen.append(len(trans[0]))
            for type in trans[0]:
                if type not in typecounter:
                    typecounter[type] = 0
                typecounter[type] += 1            
        #sort
        sortedtypes = sorted(typecounter.items(), key=lambda d:d[1], reverse = False)
        for typeitem in sortedtypes:
            for i in range(len(self.transSet)):
                if typeitem[0] in self.transSet[i][0]:
                    #swap 
                    pos = self.transSet[i][0].index(typeitem[0])                  
                    tmp = self.transSet[i][0][pos]
                    self.transSet[i][0][pos] = self.transSet[i][0][transLen[i]-1]
                    self.transSet[i][0][transLen[i]-1] = tmp
                    transLen[i] -= 1 
        typecounter.clear()
    
    def isEmpty(self):
        if len(self.tree["children"]) == 0:
            return True
        else:
            return False
    
    def getConditionalPatternBase(self,itemName,baseSet):
        tableItem = self.itemTable[itemName]
        node = tableItem[1]
        while node is not None:
            support = node["support"]
            tmp = [[],support]
            pnode = node["parent"]
            while pnode is not None:
                if pnode["name"] is None:
                    break
                tmp[0].append(pnode["name"])
                pnode = pnode["parent"]
            tmp[0].reverse()
            baseSet.append(tmp)
            node = node["next"]
    
    def getCombinationFromPath(self, container):
        nodes = []
        for itemName in self.itemTable:
            node = self.itemTable[itemName][1]
            nodes.append(node)
        size = len(nodes)
        for i in range(1,2**size):
            tmp = []
            for j in range(size):
                if i & 2**j != 0:
                    tmp.append(nodes[j])
            container.append(tmp)
            
    def isUniquePath(self):
        flag = True
        children = self.tree["children"]
        while len(children) > 0:
            if len(children) != 1:
                flag = False
                break
            for name in children:
                children = children[name]["children"]
        return flag
        
    def _initItemTable(self):
        for trans in self.sortedTransSet:
            for item in trans[0]:
                if item not in self.itemTable:
                    self.itemTable[item] = [0,None]
        
    def build(self):
        for trans in self.sortedTransSet:
            self.currNode = self.tree
            support = trans[1]
            items = trans[0]
            for i in range(len(items)):
                self._insertNode(items, i, support)
        self._minsupCheck()
        #self.printTree()
    
    def _minsupCheck(self):
        #{itemName:[support, nextNode],...}
        names = []
        for key in self.itemTable.keys():
            names.append(key)
        for itemName in names:
            if self.itemTable[itemName][0] < self.minsup:
                item = self.itemTable[itemName][1]
                #del parent's children attribute
                if len(item["parent"]) != 0:
                    item["parent"]["children"].pop(itemName)
                #del node self
                item.clear()
                #del itemTabel
                self.itemTable.pop(itemName)
       
    def _insertNode(self, items, index, support):
        item = items[index]
        if item not in self.currNode["children"]:
            self.currNode["children"][item] = {"name":item,"support":support,"children":{},"next":None,"parent":self.currNode}
            tableItem = self.itemTable[item]
            tableItem[0] += support
            nextNode = tableItem[1]
            if nextNode is None:
                tableItem[1] = self.currNode["children"][item]
            else:
                while nextNode["next"] is not None:
                    nextNode = nextNode["next"]
                nextNode["next"] = self.currNode["children"][item]
        else:
            self.currNode["children"][item]["support"] += support
            self.itemTable[item][0] += support  
        self.currNode = self.currNode["children"][item]  
    
    def printPath(self, itemName):
        if itemName not in self.itemTable:
            return
        tableItem = self.itemTable[itemName]
        node = tableItem[1]
        while node is not None:
            print("=========path==========")
            print(node["name"]+":"+str(node["support"]))
            pnode = node["parent"]
            while pnode is not None:
                if pnode["name"] is None:
                    break
                print(pnode["name"]+":"+str(pnode["support"]))
                pnode = pnode["parent"]
            node = node["next"]
        

    def printTree(self):
        print("******************************")
        print("Item table:") 
        for item in self.itemTable:
            print("name:"+item+"|support:"+str(self.itemTable[item][0])+"|next:"+str(self.itemTable[item][1]["name"])+"\n")    
        print("******************************")
        print("FP-Tree:")
        no = 0
        nodes = {no:self.tree}
        while len(nodes) != 0:
            names = []
            for name in nodes.keys():
                names.append(name)
            for name in names:
                node = nodes.pop(name)
                parent = node["parent"]
                if parent is None:
                    parentName = None
                else:
                    parentName = parent["name"]
                next = node["next"]
                if next is None:
                    nextName = None
                else:
                    nextName = next["name"]
                print("name:"+str(node["name"])+"|support:"+str(node["support"])+"|children:"+str(node["children"].keys())+"|parent:"+str(parentName)+"|next:"+str(nextName)+"\n")
                for childName in node["children"]:
                    no += 1
                    nodes[no] = node["children"][childName] 
        print("******************************")
       
def test():
    testcase = [[["i2","i1","i5"],1],[["i2","i4"],1],[["i2","i3"],1],[["i2","i1","i4"],1],[["i1","i3"],1],[["i2","i3"],1],[["i1","i3"],1],[["i2","i1","i3","i5"],1],[["i2","i1","i3"],1]]
    #testcase = []
    #testcase = [(["i1"],1),(["i1","i2","i3"],1)]
    #testcase = [(['i1'], 1)]
    tree = FPTree(testcase)      
    tree.build()
    tree.printTree()
    #print(tree.isUniquePath())c
    #c=[]
    #tree.getCombinationFromPath(c)
    #for s in c:
        #print("****************")
        #for n in s:
            #print(n["name"])
    #baseSet = []
    #tree.getConditionalPatternBase("i3",baseSet)
    #for base in baseSet:
        #print("*************")
        #print(base[0])
        #print(base[1])
    
# if __name__ == "__main__":
#     test()    
            
        