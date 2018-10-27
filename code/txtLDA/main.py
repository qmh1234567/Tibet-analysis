# coding=utf-8
from text_LDA import LDA_featureExtract,print_top_words
from gensim.models import word2vec
import pyLDAvis
import pandas as pd
import sys

sys.path.append(r'../common/')
from txt_preprocess import LoadWordList,TF_IDF

# 文件的路径
CutWordtxt='./../../Resources/CutWordPath/xzw_total.txt' 
Binaryfile='./../../Resources/Binaryfiles/xzw_total_WC'

TibetUnTxt = './../../Resources/CutWordPath/tibetUn.txt'
CorrectTxt = './../../Resources/CutWordPath/tibetUn_1.txt'



def Do_txt_LDA(CutWordtxt):
    n_features=1000   # 特征数目
    n_topics=10  # 主题数目
    n_top_words = 20 # 前20个关键词
    contlist=LoadWordList(CutWordtxt)
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
    # Do_txt_Word2Vec()
    Do_txt_LDA(CutWordtxt)


# KeyWordPath='Resources/keyword.txt'

# def Do_txt_Word2Vec():
#     # 读取分词后的文件
#     # contents=Read_file(jsonfile,CutWordtxt)
#     # 训练词向量
#     # Tibet_Word2Vec(CutWordtxt,Binaryfile)
#     # 加载词向量的二进制文件
#     model=word2vec.Word2Vec.load(Binaryfile)
#     # 去除tibetUn.txt里的编辑人、新闻来源、&gt、英文人名、A115、时间日期等
#     # DataClean(TibetUnTxt, CorrectTxt)
#     # print(model.most_similar('国务院新闻办公室'))
#     s="国务院新闻办公室 北京 发表 中国 保障 宗教 信仰 自由 政策 实践 中国 学者 受 访 时 宗教 信仰 自由 中国 传统 更是 中国 现代化 进程 缺少 政策 中国社科院世界宗教研究所 副所长 郑筱筠 中新社 记者 中国 宗教 信仰 植根于 中国 传统 文化 厚重 文化 土壤 优秀 中国 传统 文化 民族 宗教 文化 交流 融合 产物 个体 信仰 宗教 自由 尊重 自由 公民 信仰 宗教 宗教 并存 国家 保障 宗教 信仰 自由 宗教 领域 长期 和谐 稳定 中国 创举 成就 郑筱筠 提到 宗教 处 社会 相 宗教 发展 规律 中国 宗教 社会 相 公民 放弃 宗教信仰 宗教 教义 宗教 活动 法律 社会 发展 文明 相 符合 信教 群众 宗教 利益 公民 享有 宗教 信仰 自由 权利 承担 法律 义务 原 国家宗教事务局宗教研究中心 研究部 副主任 张弩说 受 中国 传统 优秀 文化 多元 通和 合共生 观念 中国 宗教 主流 宗教 信仰 自由 基础上 中国 宗教 尊重 和睦相处 倡导 自由 责任 权利 义务 相 统一 求 差异 求 宽容 交流 求 共识 构建 健康 宗教 和谐 复旦大学 国际 政治系 主任 徐以骅 指出 宗教界 人士 改革开放 中国 宗教 迎来 历史 时期 宗教 信仰 自由 政策 落实 中国 宗教 走上 健康 发展 轨道 宗教 法治化 建设 宗教 学术 交流 长足 发展 中国 宗教 宗教 团体 走出 国门 走向 世界 中国 对外 民间 交流 一道 亮丽 风景线"
#     # f=open(CutWordtxt,'r')
#     # print(f.readline())
#     s=s.split(' ')
#     pd.Series(Extract_keywords(s,model))