from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import sys
sys.path.append(r'../common/')
from txt_Word2Vec import Read_file,Tibet_Word2Vec,LoadWordList
from txt_preprocess import TF_IDF
import matplotlib.pyplot as plt
import numpy as np

# 文件的路径
jsonfile='./../../Resources/jsonfiles/politics.json'  
# CutWordtxt='./../../Resources/CutWordPath/xzxw_st.txt'
CutWordtxt='./../../Resources/CutWordPath/politics.txt'
ImagePath='./../../Resources/1.jpg'
font_path='./../../Resources/fangsong_GB2312.ttf'
word_dict_file='wordcloud.txt'

'''生成词云: str_words为字符串 生成词云图片'''
def GenWordCloud(str_words):
    image=plt.imread(ImagePath)
    wc=WordCloud(background_color='white',
                font_path=font_path, # 设置字体格式，不设置显示不了中文
                max_words=2000,
                mask=image, # 词云形状
                )
    wc.generate(str_words)
    image_colors=ImageColorGenerator(image)
    wc.recolor(color_func=image_colors)
    # plt.imshow(wc)
    # plt.axis("off")
    # plt.show()
    wc.to_file("politic.jpg")

def TF_IDF_NEWS(word_dict_file,CutWordtxt,n_features=1000):
    content_List=LoadWordList(CutWordtxt)
    tf,tf_vectorizer=TF_IDF(n_features,content_List)
    # 获取词袋模型的所有词语
    word_dict=tf_vectorizer.get_feature_names()
    word=np.array(word_dict)
    # 得到tfidf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
    weight=tf.toarray()
    # keyword=word[word_index]
    dict_word={}
    for i in range(len(weight)):
        for j in range(len(word)):
            dict_word[word[j]]=int(weight[i][j]*1000)
    word_list=sorted(dict_word.items(),key = lambda x:x[1],reverse = True)

    with open(word_dict_file,'w',encoding='utf-8') as f:
        for i in range(30):
            f.write("{\"name\": \""+str(word_list[i][0])+"\",\"value\":"+str(word_list[i][1])+"},\n")
    print("写入文件成功")


if __name__ == '__main__':
    # # 读取文件 第一次调用
    content_list=Read_file(jsonfile,CutWordtxt,flag_stop=True)
    str_content="".join(content_list)
    # TF_IDF_NEWS(word_dict_file,CutWordtxt,n_features=1000)

    '''词云图片'''
    # # 第二次调用
    # with open(CutWordtxt,'r',encoding='utf-8') as f:
    #     str_content=f.read()
    GenWordCloud(str_content)
     
      
    