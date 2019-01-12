from fp_data import TF_IDF_keyword,write_lists_to_file,Hanlp_keyword
import FPTree
from FPGrowth1 import FPGrowth1
# import FPGrowth1
import sys
sys.path.append(r'../common/')
from txt_Word2Vec import Read_file, LoadWordList

rule_types = ['society', 'culture', 'politics']
# rule_type = rule_types[0]
rule_type = rule_types[1]
# rule_type = rule_types[2]

# 文件的路径
jsonfile='./../../Resources/jsonfiles/' + rule_type +'.json'
CutWordtxt='./../../Resources/CutWordPath/' + rule_type + '.txt'
keywordtxt='./../../Resources/Keywordfiles/' + rule_type + '.txt'
# keywordfile='./../../Resources/Keywordfiles/keyword_hanlp.txt'

# 加载数据
def load_data(keywordfile):
    keywordlists=[]
    with open(keywordfile,'r',encoding='utf-8') as f:
        keyword_list=f.read().splitlines()
        for item in keyword_list:
            item=item.strip()#去除空字符
            item_list=item.split(' ')
            keywordlists.append([item_list,1])
    return keywordlists

# 生成关联规则
def get_rules(keywordfile):
    #testcase = [[["i2","i1","i5"],1],[["i2","i4"],1],[["i2","i3"],1],[["i2","i1","i4"],1],[["i1","i3"],1],[["i2","i3"],1],[["i1","i3"],1],[["i2","i1","i3","i5"],1],[["i2","i1","i3"],1]]
    print("getting rules...")
    testcase=load_data(keywordfile)
    tree = FPTree.FPTree(testcase, minsup=2)
    tree.build()
    algorithm = FPGrowth1(minsup=2)
    algorithm.growth(tree, [])
    res = sorted(algorithm.fp, key=lambda d: d[1], reverse=True)
    with open("rule_" + rule_type + "_TF_IDF.txt", 'w', encoding='utf-8') as f:
        for rule in res:
            f.write(str(rule)+"\n")
    print("关联规则写入文件成功")


if __name__ == '__main__':
    # 读取json文件，不去停用词，保留。 
    # 修改文件后必须调用一次

    # 分词
    contents=Read_file(jsonfile,CutWordtxt,flag_stop=True)

    # TF_IDF 提取关键字
    keywordlists=TF_IDF_keyword(CutWordtxt,keywordCount=5)
    write_lists_to_file(keywordlists,keywordtxt)

    # Hanlp 提取关键字
    # content_List=LoadWordList(CutWordtxt)
    # Hanlp_keyword(content_List,CutWordtxt,keywordtxt)

    # 挖掘关联规则
    get_rules(keywordtxt)

    # testcase = []
    # data = ["A B C E F O", "A C G", "E I", "A C D E G", "A C E G L", "E J", "A B C E F P", "A C D", "A C E G M", "A C E G N"]
    # for item in data:
    #     testcase.append([item.split(" "), 1])
    #
    # tree = FPTree.FPTree(testcase, minsup=2)
    # # 这里的minsup是对数据集中的支持度进行筛选，即出现频率小于minsup的数据都会被过滤
    # tree.build()
    # algorithm = FPGrowth1(minsup=2)
    # # 这里的minsuo是对频繁项集的支持度进行筛选，即出现次数小于minsup的频繁项集都会被过滤
    # # 总的来说 第二个minsup起决定性作用
    # algorithm.growth(tree, [])
    # result = sorted(algorithm.fp, key=lambda d: d[1], reverse=True)
    # print("\nThis is the rusult: ")
    # print(result)


