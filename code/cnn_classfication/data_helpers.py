import json
from collections import Counter
import numpy as np
import itertools
import sys
from gensim.models import word2vec,KeyedVectors
sys.path.append(r'../common/')
from txt_Word2Vec import Process_News
from gensim.models import word2vec,KeyedVectors

class w2v_wrapper:
     def __init__(self,file_path):
        # w2v_file = os.path.join(base_path, "vectors_poem.bin")
        self.model =word2vec.Word2Vec.load(file_path)  # 加载词向量文件
        if 'unknown' not  in self.model.wv.vocab:
            unknown_vec = np.random.uniform(-0.1,0.1,size=128)  # 产生300个[-0.1,0.1)的数 从均匀分布中随机采样
            self.model.wv.vocab['unknown'] = len(self.model.wv.vocab)
            self.model.wv.vectors = np.row_stack((self.model.wv.vectors,unknown_vec))
            

'''标签说明 政治 2  文化1  社会0 '''
# 统计新闻
def Statistic_News():
    jsonfile='../../Resources/jsonfiles/data_test.json'
    with open(jsonfile,"r",encoding='utf-8') as f:
        count=0
        dicts=json.load(f)
        contents={
        "政治":0,
        "文化":0,
        "社会":0,
        }
        for dict_item in dicts:
            if dict_item['type'] in contents.keys():
                contents[dict_item['type']]+=1
                count+=1
            else:
                continue
    print(contents)


#　统计一个文档的新闻字数分布
def Draw_new_len(file_txt):
    with open(file_txt,'r',encoding='utf-8') as f:
        politic_list=f.readlines()
        news_static={
            "1-300":0,
            "300-600":0,
            "600-1000":0,
            ">1000":0,
        }
        count=len(politic_list)
        for item in politic_list:
            if len(item)>=1 and len(item)<=300:
                news_static["1-300"]+=1
            elif len(item)>300 and len(item)<=600:
                news_static["300-600"]+=1
            elif len(item)>600 and len(item)<1000:
                news_static["600-1000"]+=1
            else:
                news_static[">1000"]+=1
        print(news_static)
        name_list =list(news_static.keys())
        num_list = list(news_static.values())
        rects=plt.bar(range(len(num_list)), num_list, color='rgby')
        # X轴标题
        index=np.arange(len(name_list))
        print(index)
        index=[float(c)+0.4 for c in index]
        plt.ylim(ymax=6000, ymin=0)
        plt.xticks(index, name_list)
        plt.ylabel("长度") #X轴标签
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
        plt.show()


# 填充句子  squence_length=最大长度  sentences=文档所有新闻列表  
def pad_sentences(squence_length,sentences,padding_word="<PAD/>"):
    # 遍历每条新闻 找到最大长度的新闻
    # squence_length=max(len(x) for x in sentences)
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i] # 取出一条新闻
        num_padding=squence_length-len(sentence) # 计算需要填充的长度
        new_sentence=sentence+ padding_word*num_padding # 填充该条新闻
        padded_sentences.append(new_sentence)
    return padded_sentences  # 返回填充后的所有新闻列表



'''加载数据集 
输入：cutwordfile 分词后的文件路径  jsonfile：待处理的json文件 labelfile: 处理后的标签文件路径
输出：x_text：三种类型的新闻列表 y：每条新闻的标签
备注： 第一次调用或输入改变时需要用到返回值，第二次调用则直接读取文件
'''
def load_data_and_labels(cutwordfile=None,jsonfile=None,labelfile=None):
    x_text=[]
    with open(jsonfile,"r",encoding='utf-8') as f:
        dicts=json.load(f)
        # 新闻类别
        y=[]
        # 新闻内容
        x_text=[]
        # 标签说明
        label={
            # 1所在的索引位置
            "政治":[0,0,1],
            # "时政要闻":[0,0,1],
            '文化':[0,1,0],
            '社会':[1,0,0],
        }
        print('正在对文章进行分词...')
        newcontent=[]
        for dict_item in dicts:
            if dict_item['type'] in label.keys() :
                key=label[dict_item['type']]
                y.append(key)
                # 调用分词函数处理每篇文章
                content = Process_News(dict_item['content'],flag_stop=True)
                # 每篇文章提取400个词语
                contentlist=content.split(" ")[0:500]
                # 添加处理后的新闻
                content=" ".join(contentlist)
                # content=content[0:599]  #截断每篇文章
                x_text.append(content)
            else:
                continue            
    with open(cutwordfile,'w',encoding='utf-8') as f:
        # 截断新闻之后再写入，否则会增加写入的时间
        # padded_sentences=pad_sentences(600,x_text,padding_word="<PAD/>")
        for sentence in x_text:
            f.write(sentence+'\n')
    # 转化为np.array类型
    y=np.array(y)
    # 将y写入txt文件
    np.savetxt(labelfile,y)
    return [x_text,y]


# 根据word2vec建立词典
def get_text_idx(text,vocab,max_document_length):
    text_array = np.zeros([len(text), max_document_length],dtype=np.int32)
    for i,x in  enumerate(text):
        words = x.split(" ")
        for j, w in enumerate(words):  # j代表词出现的位置
            if w in vocab:
                text_array[i, j] = vocab[w].index  # 将index添加到词典中
            else :
                text_array[i, j] = vocab['unknown']
    return text_array


# 针对一个数据集生成一个批迭代器  生成批次数据
# 使用生成器生成 num_epochsx num_batches_per_epoch 个批次的数据用于训练
def batch_iter(data,batch_size,num_epochs,shuffle=True):
    data=np.array(data)
    data_size=len(data)
    # 每个epoch 的num_batch
    num_batches_per_epoch=int(len(data)/batch_size)+1
    for epoch in range(num_epochs):
        # shuffle_indices 的len为 data_size 随机元素组成的列表
        # 在每次迭代中随机打乱数据
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data        
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,data_size)
            yield shuffled_data[start_index:end_index]


