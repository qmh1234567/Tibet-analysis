# 隐含狄利克雷分布（LDA) 主题词抽取
import pandas as pd
import jieba
import pyLDAvis
import pyLDAvis.sklearn
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import time

# 分词
def CutWordList(CutWordtxt):
    # CutWordtxt='Resources/CutWordPath/tibet.txt'
    contents=open(CutWordtxt,'r',encoding='utf-8').read()
    content_List=contents.splitlines()  # 分词之后的新闻内容列表
    return content_List

# 特征提取，
def TF_IDF(n_features,content_List):
    t1=time.time()
    # n_features = 1000  # 1000个最重要的特征关键词
    # 因为 CountVectorizer 只是单纯的统计词频，改用TfidfVectorizer
    tf_vectorizer=TfidfVectorizer(strip_accents='unicode',
                                  max_features=n_features,
                                  stop_words='english',
                                  max_df=0.5,
                                  min_df=10)
    tf = tf_vectorizer.fit_transform(content_List)
    print("\词向量表示为",tf.toarray())
    t2=time.time()
    print("生成词向量所需时间为",(t2-t1))
    return tf,tf_vectorizer

# # 构建训练语料
# def CreateTrainset(content_List):
#     # 生成词典
#     cont_list = []
#     for item in content_List:
#         cont_list.append(item.split(' ')[:-2])
#     # 生成字典
#     dictionary=Dictionary(cont_list)
#     # 使用词带模型生成向量语料
#     corpus=[ dictionary.doc2bow(text) for text in cont_list]
#     return dictionary,corpus

# # LDA模型训练
# def TrainLDA(dictionary,corpus,n_topics):
    
#     lda=LdaModel(corpus=corpus,id2word=dictionary,num_topics=n_topics)
#     print("打印主题\n")
#     print(lda.print_topics(n_topics,20))
#     return lda


def LDA_featureExtract(n_topics,tf):
    print("正在查找主题...")
    t1=time.time()
    # n_topics = 10
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    t2=time.time()
    print("查找主题所花时间为",(t2-t1))
    return lda

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


if __name__ == '__main__': 
    n_features=1000
    n_topics=10  # 主题数目
    CutWordtxt='Resources/CutWordPath/tibet.txt'
    contlist=CutWordList(CutWordtxt)

    # dictionary,corpus=CreateTrainset(contlist)

    # TrainLDA(dictionary,corpus,n_topics)

    tf,tf_vectorizer=TF_IDF(n_features,contlist)
    # 输出前20个关键词
    n_top_words = 20
    tf_feature_names = tf_vectorizer.get_feature_names()

   
    lda=LDA_featureExtract(n_topics,tf)
    print_top_words(lda, tf_feature_names, n_top_words)

   
    data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.show(data)

    # 显示动态图
    pyLDAvis.enable_notebook()
    # pyLDAvis.enable_notebook()
    pyLDAvis.sklearn.prepare(lda,tf, tf_vectorizer)





