from fp_data import TF_IDF_keyword,write_lists_to_file,Hanlp_keyword
import FPTree
from FPGrowth1 import FPGrowth1
# import FPGrowth1
import sys
sys.path.append(r'../common/')
from txt_Word2Vec import Read_file,LoadWordList


# 文件的路径
jsonfile='./../../Resources/jsonfiles/culture.json'   
CutWordtxt='./../../Resources/CutWordPath/culture.txt'
# keywordtxt='./../../Resources/Keywordfiles/data_train_new.txt'
keywordfile='./../../Resources/Keywordfiles/culture_keyword.txt'


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
    print("正在生成关联规则..")
    #testcase = [[["i2","i1","i5"],1],[["i2","i4"],1],[["i2","i3"],1],[["i2","i1","i4"],1],[["i1","i3"],1],[["i2","i3"],1],[["i1","i3"],1],[["i2","i1","i3","i5"],1],[["i2","i1","i3"],1]]
    testcase=load_data(keywordfile)
    tree = FPTree.FPTree(testcase,minsup=2)      
    tree.build()
    algorithm = FPGrowth1(minsup=2)
    algorithm.growth(tree,[])
    res = sorted(algorithm.fp, key=lambda d:d[1], reverse = True )
    with open("./rule.txt",'w',encoding='utf-8') as f:
        for rule in res:
            f.write(str(rule)+"\n")
    print("关联规则写入文件成功")

if __name__ == '__main__':
    # 读取json文件，不去停用词，保留。 
    # 修改文件后必须调用一次
    contents=Read_file(jsonfile,CutWordtxt,flag_stop=False)
    keywordlists=TF_IDF_keyword(CutWordtxt,keywordCount=5)
    write_lists_to_file(keywordlists,keywordfile)
    get_rules(keywordfile)# 生成关联规则


    # # 读取分词后的文件
    # content_List=LoadWordList(CutWordtxt)
    # Hanlp_keyword(content_List,CutWordtxt,keywordtxt)
    # get_rules(keywordtxt)



    