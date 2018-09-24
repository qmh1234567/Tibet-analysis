from pyhanlp import *
import re
def Define_segment():
    # 自定义分词
    print("=" * 30 + " 自定义分词" + "=" * 30)
    CustomDictionary = JClass('com.hankcs.hanlp.dictionary.CustomDictionary')
    CustomDictionary.add('攻城狮')
    CustomDictionary.add('单身狗')
    HanLP = JClass('com.hankcs.hanlp.HanLP')
    print(HanLP.segment('攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰'))
    print("-" * 70)

def Segment():
    # 中文分词
    print(HanLP.segment('国家主席习近平2日在人民大会堂同南非总统拉马福萨举行会谈。'))
    print("-" * 70)       
    # 标准分词
    print("=" * 30 + "标准分词" + "=" * 30)
    StandardTokenizer = JClass('com.hankcs.hanlp.tokenizer.StandardTokenizer')
    print(StandardTokenizer.segment('国家主席习近平2日在人民大会堂同南非总统拉马福萨举行会谈。'))
    print("-" * 70)
    print("="*30+"NLP分词"+"="*30)
    NLPTokenizer=JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
    print(NLPTokenizer.segment('国家主席习近平2日在人民大会堂同南非总统拉马福萨举行会谈'))

def ExtractKeyword(document):
    ## 关键词提取和自动摘要
    # document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
    #         "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
    #         "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
    #         "严格地进行水资源论证和取水许可的批准。"
    print("=" * 30 + "关键词提取" + "=" * 30)
    print(HanLP.extractKeyword(document, 8))
    print("-" * 70)

    print("=" * 30 + "自动摘要" + "=" * 30)
    print(HanLP.extractSummary(document, 3))
    print("-" * 70)



# document1=  ["签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。" ,
#     "王国强、高峰、汪洋、张朝阳光着头、韩寒、小四," ,
#     "张浩和胡健康复员回家了," ,
#     "王总和小丽结婚了," ,
#     "编剧邵钧林和稽道青说:" ,
#     "这里有关天培的有关事迹," ,
#     "龚学平等领导,邓颖超生前." ]

## 中国人名识别
def ChineseNameRecognize(document1):
    segment = HanLP.newSegment().enableTranslatedNameRecognize(True);
    # segment = HanLP.newSegment().enableNameRecognize(True);
    for sentence in document1:
        term_list = segment.seg(sentence)
        print(term_list)

if __name__ == '__main__':
    document = "水利部水资源司司长陈明忠9月29日在国务院新闻办举行的新闻发布会上透露，" \
            "根据刚刚完成了水资源管理制度的考核，有部分省接近了红线的指标，" \
            "有部分省超过红线的指标。对一些超过红线的地方，陈明忠表示，对一些取用水项目进行区域的限批，" \
            "严格地进行水资源论证和取水许可的批准。"
    words=HanLP.segment(document)    
    str1 = ""
    for word in words:
        str1 = str1 + (str(word).split('/')[0]) + ' '
    print(str1)

    ExtractKeyword(document)