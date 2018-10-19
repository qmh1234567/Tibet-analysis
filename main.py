from text_LDA import CutWordList,LDA_featureExtract,print_top_words,TF_IDF
from txt_Word2Vec import Process_News,Read_file,Tibet_Word2Vec
from gensim.models import word2vec
import pyLDAvis
# 文件的路径
jsonfile='./Resources/jsonfiles/xzw_total_train.json'  # json文件路径
CutWordtxt='Resources/CutWordPath/xzw_total.txt' 
Binarypath='Resources/Binaryfiles/xzw_total_WC'

TibetUnTxt = 'Resources/CutWordPath/tibetUn.txt'
CorrectTxt = 'Resources/CutWordPath/tibetUn_1.txt'
Nametxt='Resources/CutWordPath/人名.txt'
Placetxt='Resources/CutWordPath/地名.txt'

# KeyWordPath='Resources/keyword.txt'

def Do_txt_Word2Vec():
    # 读取分词后的文件
    contents=Read_file(jsonfile,CutWordtxt)
    # 训练词向量
    Tibet_Word2Vec(CutWordtxt,Binarypath)
    # 加载词向量的二进制文件
    model=word2vec.Word2Vec.load(Binarypath)
    # 去除tibetUn.txt里的编辑人、新闻来源、&gt、英文人名、A115、时间日期等
    # DataClean(TibetUnTxt, CorrectTxt)
    print(model.most_similar('中国'))
    


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
    print(data)
    pyLDAvis.show(data)
    # 显示动态图
    pyLDAvis.enable_notebook()
    # pyLDAvis.enable_notebook()
    pyLDAvis.sklearn.prepare(lda,tf, tf_vectorizer)


if __name__ == '__main__': 
    Do_txt_Word2Vec()
    # Do_txt_LDA(CutWordtxt)