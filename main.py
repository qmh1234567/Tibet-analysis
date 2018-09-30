from text_LDA import *
from txt_Word2Vec import *

# 文件的路径
jsonfile='Resources/tibet.json'  # json文件路径
CutWordtxt='Resources/CutWordPath/tibet.txt' 
Binarypath='Resources/TibetWord2vec'
CutWordPath='Resources/CutWordPath/'
KeyWordPath='Resources/keyword.txt'

def Do_txt_Word2Vec():
     # 判断文件是否更新有待完成
    # if not os.path.exists(Binarypath):
    # if not os.path.exists(CutWordtxt):
    # 读取json文件
    Contlist=ConjsontoList(jsonfile)
    print("正在分词中..")
    # 分词，并将结果写入文件
    SegmentContList(Contlist,CutWordtxt)
    # 训练词向量
    Tibet_Word2Vec(CutWordPath,Binarypath)
    # 加载词向量的二进制文件
    model=word2vec.Word2Vec.load(Binarypath)
    # # 词向量可视化
    # KeyWordView(KeyWordPath,CutWordtxt,model,100)
    # Test_Word2vec(model)


def Do_txt_LDA(CutWordtxt):
    n_features=1000   # 特征数目
    n_topics=10  # 主题数目
    n_top_words = 20 # 前20个关键词
    contlist=CutWordList(CutWordtxt)
    # dictionary,corpus=CreateTrainset(contlist)
    # TrainLDA(dictionary,corpus,n_topics)
    # TF-IDF 训练词向量
    tf,tf_vectorizer=TF_IDF(n_features,contlist)
    tf_feature_names = tf_vectorizer.get_feature_names()
    # 训练lda模型
    lda=LDA_featureExtract(n_topics,tf)
    # 输出lda训练结果
    print_top_words(lda, tf_feature_names, n_top_words)
    data = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    pyLDAvis.show(data)
    # 显示动态图
    pyLDAvis.enable_notebook()
    # pyLDAvis.enable_notebook()
    pyLDAvis.sklearn.prepare(lda,tf, tf_vectorizer)


if __name__ == '__main__': 
    Do_txt_Word2Vec()
    # Do_txt_LDA(CutWordtxt)