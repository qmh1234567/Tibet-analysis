# Tibet-analysis
## 项目目录说明
- Resources       =>存放项目的资源文件

    Binaryfiles     =>存放word2vec处理后的二进制文件
    
    CutWordPath    =>存放分词后的txt文件
    
    jsonfiles      =>存放爬虫爬取后的json文件
    
    stopwords.txt     =>停用词文件
    
- test1 此为测试文件夹，不用理会
- Classification.py  => 文本分类脚本，目前进行到预处理阶段
- main.py  => 主文件
- text-cnn => 定义TextCNN类的脚本
- text_LDA => 主题词提取+显示脚本
- txt_preprocess => 数据预处理脚本，主要使用里面的分词函数
- txt_Word2Vec  => 语料清洗、生成词向量、词向量可视化
