# 隐含狄利克雷分布（LDA) 主题词抽取
import pandas as pd
import jieba
import pyLDAvis
import pyLDAvis.sklearn
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.decomposition import LatentDirichletAllocation
import time




'''
查找主题
'''
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

'''打印结果'''
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
            for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

'''构建训练语料'''
def CreateTrainset(content_List):
    # 生成词典
    cont_list = []
    for item in content_List:
        cont_list.append(item.split(' ')[:-2])
    # 生成字典
    dictionary=Dictionary(cont_list)
    # 使用词带模型生成向量语料
    corpus=[ dictionary.doc2bow(text) for text in cont_list]
    return dictionary,corpus

    

'''LDA模型训练 '''
def TrainLDA(dictionary,corpus,n_topics):
    lda=LdaModel(corpus=corpus,id2word=dictionary,num_topics=n_topics)
    print("打印主题\n")
    print(lda.print_topics(n_topics))
    return lda






