import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
corpus_raw='He is the king . The king is royal . She is the royal  queen'

# 转化成小写字母
corpus_raw=corpus_raw.lower()
''''
创建词典，确定每个单词的索引
'''
words=[]
for word in corpus_raw.split():
    if word!='.':
        words.append(word)

words=set(words)  # 去除重复单词
print(words)

word2int={}
int2word={}
# 词典的总长度
vocab_size=len(words)

for i,word in enumerate(words):
    word2int[word]=i 
    int2word[i]=word  

print(word2int)
print(int2word)


'''
将句子向量转化为单词列表
'''
# 将句子向量转化成单词列表
raw_sentences=corpus_raw.split('.')
print(raw_sentences)
sentences=[]
for sentence in raw_sentences:
    sentences.append(sentence.split())# 以空格分割
print(sentences)

# 产生训练数据
data=[]
Window_size=2  # 窗口大小
for sentence in sentences:
    for word_index,word in enumerate(sentence):
        # 寻找中心词附近的词
        # print("sentence=",sentence[max(word_index-Window_size,0):min(word_index+Window_size,len(sentence))+1])
        for nb_word in sentence[max(word_index-Window_size,0):min(word_index+Window_size,len(sentence))+1]:
            if nb_word != word:
                data.append([word,nb_word])
print("data=",data)

# 将训练数据转化为one-shoot向量表示
def to_one_shoot(data_point_index,vocab_size):
    temp=np.zeros(vocab_size) # onshot 向量的长度=字典长度
    temp[data_point_index]=1
    return temp

x_train=[]  
y_train=[]
for data_word in data:
    x_train.append(to_one_shoot(word2int[ data_word[0] ],vocab_size))
    y_train.append(to_one_shoot(word2int[ data_word[1] ] ,vocab_size))
# 将列表转化成numpy arrays
x_train=np.asarray(x_train)
y_train=np.asarray(y_train)
# print(x_train,y_train)

print(x_train.shape,y_train.shape)

# 构建tensorflow模型  placeholder 占位符  (None,vocab_size) 行不定，列为vocab_size
x=tf.placeholder(tf.float32,shape=(None,vocab_size))
y_label=tf.placeholder(tf.float32,shape=(None,vocab_size))

Embedding_dim=5
# tf.random_normal 从正态分布中去除随机值  tf.Variable 构造张量
W1=tf.Variable(tf.random_normal([vocab_size,Embedding_dim]))  # 权重
b1=tf.Variable(tf.random_normal([Embedding_dim]))    # 偏置
# tf.matmul 矩阵的乘法， 输入矩阵*权重矩阵
hidden_representation=tf.add(tf.matmul(x,W1),b1)

W2=tf.Variable(tf.random_normal([Embedding_dim,vocab_size]))
b2=tf.Variable(tf.random_normal([vocab_size]))
prediction=tf.nn.softmax(tf.add(tf.matmul(hidden_representation,W2),b2))

# print(hidden_representation)
## 训练tensorflow模型 tf.Session 运行tf操作的类
sess=tf.Session()
# 初始化模型的参数
init=tf.global_variables_initializer()
# 训练
sess.run(init)
# 定义损失函数 tf.reduce_mean 计算张量在各个维度上的平均值 tf.reduce_sum计算一个张量在各个维度的和  tf.log计算对数 
corss_entropy_loss=tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(prediction),reduction_indices=[1]))
# 定义训练步骤
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(corss_entropy_loss)
n_iters=1000
for _ in range(n_iters):
    # 第一个参数表示执行内容，feed_dict：每次迭代中更新x和y的值
    sess.run(train_step,feed_dict={x:x_train,y_label:y_train})
    print('loss is:',sess.run(corss_entropy_loss,feed_dict={x:x_train,y_label:y_train}))


# 一个 0 1向量与W1相乘时的结果就是对应的词向量，还需要加上偏置
vectors=sess.run(W1+b1)
print(vectors)
print("-"*70)
# 查看queen的词向量
print(vectors[word2int['queen']],word2int['queen'])

# 计算两个向量的欧氏距离
def enuclidean_dist(vec1,vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

# 找到与某个单词最接近的词向量
def find_closet(word_index,vectors):
    min_dist=1000
    min_index=-1
    query_vector=vectors[word_index]
    for index,vector in enumerate(vectors):
        eud_dist=enuclidean_dist(vector,query_vector)
        if eud_dist <min_dist and not np.array_equal(vector,query_vector):
            min_dist=eud_dist
            min_index=index
    return min_index

print(int2word[find_closet(word2int['king'],vectors)])
print(int2word[find_closet(word2int['queen'],vectors)])
print(int2word[find_closet(word2int['royal'],vectors)])

# 画出向量相关图 将向量从5维压缩到2维显示
# n_components 嵌入式空间的维度   
model=TSNE(n_components=2,random_state=0)
# 设置精度，过小的结果会被压缩
np.set_printoptions(suppress=True)
# 归一化
vectors=model.fit_transform(vectors)
# 创建一个正则器
normalizer=preprocessing.Normalizer()
# l2表示样本各个特征值除以各个特征值平方之和
vectors=normalizer.fit_transform(vectors,'l2')


fig,ax=plt.subplots()
# 约束坐标轴范围
plt.axis([-1,1,-1,1])

print("vectors=",vectors)

for word in words:
    print(word,vectors[word2int[word]][1])
    ax.annotate(word,(vectors[word2int[word]][0],vectors[word2int[word]][1]))
plt.show()
