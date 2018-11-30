import json
from txt_preprocess import HanLp_Segment
# 提取json文件的content标签的内容，返回列表
def Read_jsonfile():
    jsonfile='../../Resources/jsonfiles/data_train.json'
    with open(jsonfile,"r",encoding='utf-8') as f:
        count=0
        dicts=json.load(f)
        content_list=[]
        for dict_item in dicts:
            content_list.append(dict_item["content"])
    print("文章长度为",len(content_list))
    return content_list

# 
def Searach_word(word,content_list):
    wordlist=["党委","唐卡","罗布林卡","青年林卡","林卡","墨竹工卡","鲁朗林海","澎波河卡孜河段","卡若拉冰川","卡加拉","茶卡","卡洛镇","玛卡","卡垫","卡布","卡赛","达因卡","卡斯木村","卡定沟","宗通卡"]
    count=0
    
    for single_new in content_list:
            single_new_list=single_new.split('，')
            for sentence in single_new_list: # 查找每篇新闻的每句话
                flag=False;
                for oldword in wordlist:
                    if oldword in sentence:
                        flag=True
                        break
                if(flag==False):
                    if word in sentence:
                    # for oldword in wordlist:  #不在已有的单词表
                        # if oldword not in sentence:
                        count=count+1
                        print("*"*100)
                        print(sentence)
                        s1=HanLp_Segment(sentence,flag_stop=True)    
                        print(s1)  
                        break
    return count




if __name__ == '__main__':
    content_list=Read_jsonfile()
    word='党'
    count=Searach_word(word,content_list)
    print("该单词一共出现了%d次" % count)
    