# 对json文件或txt文件进行处理
import json
import re
import os
import txt_Word2Vec
# 读取json文件，只读type和content的内容，返回列表
def readfile(filename):
    with open(filename,'r',encoding='utf-8') as f:
        dict_sentiment=json.load(f)
        content_list=[]
        type_list=[]
        for dict1 in dict_sentiment:     
            # dict1['content'] =txt_Word2Vec.Process_News(dict1['content'],flag_stop=True)
            content_list.append(dict1['content'])
            type_list.append(dict1['type'])
        print('读入文件成功')
    return type_list,content_list


# 根据json文件生成txt训练集,结果为neg.txt或positive.txt
def generate_trainset(jsonfile):
    with open(jsonfile,'r',encoding='utf-8') as f:
        lisdicts=json.load(f)
        neglist=[]
        poslist=[]
        for d in lisdicts:
            if d['type']=='0':
                neglist.append(d['content'])
            else:
                poslist.append(d['content'])
        # 将列表写入文件
        with open('./hotel/neg.txt','w',encoding='utf-8') as f:
            for word in neglist:
                f.write(word+'\n')
        with open('./hotel/positive.txt','w',encoding='utf-8') as f:
            for word in poslist:
                f.write(word+'\n')



# 将一份txt文件的每一行都写入一个txt文件
def Write_to_many_txt(Cutwordtxt,filepath):
    with open(Cutwordtxt,'r',encoding='utf-8') as f:
        news_list=f.read().splitlines()
    # 遍历所有新闻
    for i in range(len(news_list)):
        # 使用正则表达式去除标题的特殊符号和空格
        # titles[i]=re.sub(r'[\s | / \d ＂＂ " ? : “ ” \. \- +]','',titles[i])
        filename=str(i)+'.txt'
        file=os.path.join(filepath,filename)  # 创建路径
        # os.mknod(file)  # 生成文件
        with open(file,'w',encoding='utf-8') as f:
            f.write(news_list[i])
    print("写入多个txt文件成功")

# 将正负txt文件转成json文件
def load_ChnSentiCorp(neg_filepath,pos_filepath,filename):
    # filename='./../../Resources/jsonfiles/ChnSentiCrop.json'
    neg_flist=os.listdir(neg_filepath)
    pos_flist=os.listdir(pos_filepath)
    list1=read_from_filelist(neg_filepath,neg_flist,'0')
    list2=read_from_filelist(pos_filepath,pos_flist,'1')
    list1.extend(list2)
    # 将列表写入json文件
    with open(filename,'w',encoding='utf-8') as f:
        json.dump(list1,f)
        print('生成json文件成功')


# 将一个目录下的多个txt读成一个列表
def covert_txts_to_list(filepath):
    # 该路径下的所有文件名字
    fileList=os.listdir(filepath)
    contentlist=[]
    for file in fileList:
        with open(file,'r',encoding='utf-8') as f:
            contentlist.append(f.read())
    return contentlist

# 将分类的多个文件目录做成json文件
def convert_folders_to_json(folder_path,jsonfile):
    Fudan_list=[]
    category_list=os.listdir(folder_path)
    for dr in category_list:
        category_path=os.path.join(folder_path,dr)  # 子文件夹路径
        fileList=os.listdir(category_path) # 该子文件夹下的所有文件名字
        for f in fileList:
            Fudan_dict={
            "type":'',
            'content':[]
            }
            file=os.path.join(category_path,f) # 文件路径
            Fudan_dict['type']=dr # 类别
            # 获取文件大小
            if os.path.getsize(file)>10913:  # 如果文件大小超过10kb，则丢弃
                continue
            with open(file,'rb') as f:
                strwords=f.read().decode('gb2312',errors='ignore')
                strwords=re.sub(r'[\s 《 》\r \n / ＂＂ " ? : “ ” \. \- + % （ ） ]','',strwords)
                strwords=re.sub(r'【责任编辑】|【参考文献】|【校对者】', '', strwords)
                strwords=re.sub(r'【.*】', '', strwords)
                Fudan_dict['content']=strwords
            # 格式不对的文本需要过滤掉
            if len(Fudan_dict['content'])<10:
                print(file)
                print(Fudan_dict)
            else:
                # 对每条新闻进行分词
                # Fudan_dict['content'] =txt_Word2Vec.Process_News(Fudan_dict['content'],flag_stop=True)
                Fudan_list.append(Fudan_dict)
    # 写入json文件
    with open(jsonfile,'w',encoding='utf-8') as f:
        json.dump(Fudan_list,f)
    print("写入json文件成功")
    print("新闻数为",len(Fudan_list))

if __name__ == '__main__':
    # # 存放路径
    # filepath='./../../Resources/sentiment_folders/weiboCorp/negtive'
    # # 待写入的txt文件
    # Cutwordtxt='./../../Resources/sentiment_folders/weibo/neg1.txt'
    # Write_to_many_txt(Cutwordtxt,filepath)

    folder_path='./../../Resources/classfication_folders/test_corpus/'
    jsonfile='./../../Resources/jsonfiles/fudan_test.json'
    convert_folders_to_json(folder_path,jsonfile)