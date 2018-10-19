import json
from collections import Counter
from txt_Word2Vec import Process_News
import numpy as np
import itertools

# 标签说明 ：宗教 0  文化1  生态2 
# 统计新闻
def Read_file():
    cutwordfile='./Resources/CutWordPath/'
    jsonfile='./Resources/jsonfiles/xzw_total_train.json'
    x_text=[]
    with open(jsonfile,"r",encoding='utf-8') as f:
        dicts=json.load(f)
        # 新闻类别
        all_labels=[]
        # 新闻内容
        x_text=[]
        # 标签说明
        label={
            "宗教":'0',
            '文化':'1',
            '生态':'2'
        }
        print('正在对文章进行分词...')
        for dict_item in dicts:
            if dict_item['type'] in label.keys() :
                key=label[dict_item['type']]
                all_labels.append(key)
                # 调用分词函数处理每篇文章
                content = Process_News(dict_item['content'])
                x_text.append(content)
            else:
                continue            
    with open(cutwordfile+'xzw.txt','w',encoding='utf-8') as f:
        for line in x_text:
            f.write(line+'\n')
    return [x_text,all_labels]

# 填充句子  
def pad_sentences(sentences,padding_word="<PAD/>"):
    # 遍历每条新闻 找到最大长度的新闻
    squence_length=max(len(x) for x in sentences)
    print(squence_length)
    padded_sentences=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        num_padding=squence_length-len(sentence)
        new_sentence=sentence+[padding_word]*num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

# 建立词典
def build_vocab(sentences):
    # 统计每个词的词频
    word_counts=Counter(itertools.chain(*sentences))
    # 索引到单词的映射 most_common返回一个TopN列表
    vocabulary_inv=[x[0] for x in word_counts.most_common()]
    # 单词到索引的映射
    vocabulary={x:i for i,x in enumerate(vocabulary_inv)}
    return [vocabulary,vocabulary_inv]

def bulid_input_data(sentences,labels,vocabulary):
    # 根据词典将标签和句子映射成向量
    x=np.array([vocabulary[word] for word in sentence] for sentence in sentences)
    y=np.array(labels)
    return [x,y]

# 整理数据
def load_data():
    # 返回输入向量、标签，字典,词
    [x_text,all_labels]=Read_file()
    # x_text=list(open('t1.txt',"r",encoding='utf-8').read().splitlines())
    # 将新闻做成二级列表
    sentences=[]
    for x in x_text:
        x=x.split(' ')
        sentences.append(x)
    # 填充句子
    padded_sentences=pad_sentences(sentences)
    # 创建词典
    [vocabulary,vocabulary_inv]=build_vocab(sentences)
    # 建立输入
    x,y=bulid_input_data(padded_sentences,all_labels,vocabulary)
    return [x,y,vocabulary,vocabulary_inv]

# 针对一个数据集生成一个批迭代器
def batch_iter(data,batch_size,num_epochs):
    data=np.array(data)
    data_size=len(data)
    num_batches_per_epoch=int(len(data)/batch_size)+1
    for epoch in range(num_epochs):
        # shuffle_indices 的len为 data_size 随机元素组成的列表
        shuffle_indices=np.random.permutation(np.arange(data_size))
        shuffled_data=data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num*batch_size
            end_index=min((batch_num+1)*batch_size,data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
#    [x,y,vocabulary,vocabulary_inv]=load_data()
   print(x)
   print("*"*70)
   print(y)
   
    