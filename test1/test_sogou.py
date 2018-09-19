from jpype import *
from pyhanlp import *
import sys
from txt_preprocess import  *
import re
import logging
import os.path
import multiprocessing
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence

def ReadFile(filepath,filesavepath):
    k=1
    Wordlist=[]
    print(os.getcwd())
    # 打开语料库
    with open(filepath,encoding='utf-8') as f:
        print(os.getcwd())
        for line in f:
            if k>=1000:
                break
            else:
                # 使用正则表达式去掉<content></content>
                raw_content=re.sub(r'<content>|</content>','',line)
                # 使用hanlp对每行进行分词
                Line_seg=HanLp_Segment(raw_content)
                # 将分词之后的行添加到列表中
                Wordlist.append(Line_seg)
                # 将分词后的结果写入文件
                with open(filesavepath,'a',encoding='utf-8') as fw:
                    fw.write(Line_seg)
                k+=1

   
    
# 查看并显示词向量
def Use_Word2vec(Binarypath):
    model=word2vec.Word2Vec.load(Binarypath)
   
    print("与财富最相近的词语是\n")
    result=model.most_similar('财富')

    for word in result:
        print(word[0],word[1])

   
    print("民众和记者的相似度为:\n")
    print(model.similarity(u'民众',u'记者'))
    
    
    print(model['民众'])

   
   

if __name__ == '__main__':
    filepath='./test1/corpus.txt'
    filesavepath='./test1/corpusDone.txt'
    ReadFile(filepath,filesavepath) 

    # logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",level=logging.INFO)
    # 导入训练集
    sentences=word2vec.Text8Corpus(filesavepath)
    # 构建词向量
    model=word2vec.Word2Vec(sentences,size=300,workers=2,min_count=5,window=5)
    # 保存至二进制文件
    Binarypath='./test1/corpusWord2vec'
    model.save(Binarypath)
    Use_Word2vec(Binarypath)


    # program=os.path.basename(sys.argv[0])
    # logger=logging.getLogger(program)

    # logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
    # logging.root.setLevel(level=logging.INFO)
    # logging.info("running %s" % ' '.join(sys.argv))

    # if(sys.argv)<4
    #  print(global()['__doc__'] % locals)
    #  sys.exit(1)
    
    # inp,outp,outp2=sys.argv[1:4]
    

    