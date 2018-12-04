import json
from matplotlib import pyplot as plt
import numpy as np
from pyecharts import Pie,Bar,Map
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from PIL import Image
from gensim.models import word2vec
import sys
import zhihu
import re
import pyhanlp

def Read_file(jsonfile):
    with open(jsonfile,'r',encoding='utf-8') as f:
        dict_content=json.load(f)
    return dict_content



#  词云
def plt_wordcloud(new_words):
    coloring = np.array(Image.open("./../../Resources/wordcloudpics/img.jpg"))
    # simkai.ttf 必填项 识别中文的字体，例：simkai.ttf，
    my_wordcloud = WordCloud(background_color="white", max_words=800,
                    mask=coloring, max_font_size=120, random_state=30, scale=2,font_path="./../../Resources/fangsong_GB2312.ttf").generate(new_words)
    image_colors = ImageColorGenerator(coloring)
    plt.imshow(my_wordcloud.recolor(color_func=image_colors))
    plt.imshow(my_wordcloud)
    plt.axis("off")
    plt.show()
    # 保存图片
    my_wordcloud.to_file('./../../Resources/wordcloudpics/1.jpg')




# 文章以及作者信息
class ZhihuData(object):
    def __init__(self,dict_content):
        self.gender=dict_content['gender']
        self.comment_number=dict_content['comment_number']
        self.yes_number=dict_content['yes_number']
        self.topic_tag=dict_content['topic_tag']
        self.title=dict_content['title']
    # 性别分布
    def plt_Pie(self,name,htmlfile):
        # 1 男  0 女  -1 未知
        value=[self.gender.count(1),self.gender.count(0),self.gender.count(-1)]
        attr=['男','女','未知']
        pie=Pie(name,width=500,height=300)
        pie.add("性别",attr,value,is_label_show=True)
        pie.render(htmlfile)
   
    # 点赞数统计
    def plt_Yes(self,htmlfile):
        dict_yes_number={
            "<1000":0,
            "1000-5000":0,
            "5000-10000":0,
            ">10000":0,
        }
        dict_comment_number={
            "<100":0,
            "100-500":0,
            "500-1000":0,
            ">1000":0,
        }
        for item in self.yes_number:
            if item<1000:
                dict_yes_number["<1000"]+=1
            elif item>=1000 and item < 5000:
                dict_yes_number["1000-5000"]+=1
            elif item >=5000 and item <10000:
                dict_yes_number["5000-10000"]+=1
            else:
                dict_yes_number[">10000"]+=1

        for item in self.comment_number:
            if item<100:
                dict_comment_number["<100"]+=1
            elif item>=100 and item < 500:
                dict_comment_number["100-500"]+=1
            elif item >=500 and item <1000:
                dict_comment_number["500-1000"]+=1
            else:
                dict_comment_number[">1000"]+=1
        bar=Bar("话题点赞数统计")
        bar.add("点赞数",list(dict_yes_number.keys()),list(dict_yes_number.values()))
        bar.add("评论数",list(dict_comment_number.keys()),list(dict_comment_number.values()),is_convert=True)
        bar.show_config()
        bar.render(htmlfile) 


    def static_tag(self,htmlfile,tag_json):
        ditc_tag={
            "政治":[],
            "政治标题": [],
            "文化":[],
            "文化标题":[],
            "社会":[],
            "社会标题":[],
            "其他":[],
            "其他标题":[]
        }
        politic=["印度军事","地缘政治","援藏","基础设施建设","制度","国家统一","政治","外交","领土","军事","独立运动","国际关系","国际政治","台湾","美国政治"]
        society=["地质灾害","打工换宿","川藏线","社会现状","爱情","中国经济","登山","青年旅舍","背包客","户外","骑行","户外运动","经济","农业","旅行","社会","西藏旅游","西藏自助游","穷游","旅游","自驾游","医疗","医生","女文青","电视选秀节目","心灵鸡汤","汽车","支教"]
        education=["辞职","当兵","教育","职业发展","工作","西藏大学","大学生活","教师","学生"]
        society.extend(education) # 将教育归入社会，不需要返回值
        culture=["花卉","寺庙","建筑","突厥","藏文","音乐","唐卡","藏语","医学","古人类学","工程学","文化习俗","地质学","中国历史","中国地理","历史地理学","习俗","文化","电影","历史","佛教","宗教","西藏历史","人文地理","美食","饮食","阅读","民俗","藏传佛教","民族","地理","西域","文艺青年","传统文化","纪录片"]
        for li in zip(self.title, self.topic_tag):
            if len(list(set(li[1]).intersection(set(politic)))):
                ditc_tag["政治"].append(li[1])
                ditc_tag["政治标题"].append(li[0])
            elif len(list(set(li[1]).intersection(set(society)))):
                ditc_tag["社会"].append(li[1])
                ditc_tag["社会标题"].append(li[0])
            elif len(list(set(li[1]).intersection(set(culture)))):
                ditc_tag["文化"].append(li[1])
                ditc_tag["文化标题"].append(li[0])
            else:
                ditc_tag["其他"].append(li[1])
                ditc_tag["其他标题"].append(li[0])
        # 写入json文件
        zhihu.write_to_jsonfile(ditc_tag,tag_json)
        list1=['政治标题','文化标题','社会标题','其他标题']
        for i in list1:
            ditc_tag.pop(i)
        bar=Bar('文章类别统计')
        attr=list(ditc_tag.keys())
        value=[]
        for i in ditc_tag.values():
            value.append(len(i))
        bar.add("类别",attr,value,is_label_show=True,make_point=['average'])
        bar.show_config()
        bar.render(htmlfile)

    def plt_Tibet_map(self,htmlfile):
        value = [478.3, 218.00, 180.00, 133.31, 148.50,
            121, 46.5]
        attr = [u'拉萨市', u'日喀则市',  u'昌都市', u'林芝市', u'山南市', 
                u'那曲地区', u"阿里地区"]
        map = Map(u"17年西藏各城市GDP分布", width=1200, height=600)
        map.add("", attr, value, maptype=u'西藏',visual_range=[0,500],
                is_visualmap=True, visual_text_color='#000')
        map.show_config()
        map.render(htmlfile)


if __name__ == '__main__':
    jsonfile="./../../Resources/jsonfiles/zhihu.json"
    gender_htmlfile='./../../Resources/htmlfiles/gender.html'
    yesnumber_htmlfile='./../../Resources/htmlfiles/Number.html'
    catagory_htmlfile='./../../Resources/htmlfiles/category.html'
    tag_jsonfile="../../Resources/jsonfiles/tag.json"
    map_jsonfile='./../../Resources/htmlfiles/Tibetmap.html'

    dict_content=Read_file(jsonfile)
    zh=ZhihuData(dict_content)
    zh.plt_Pie("话题作者性别分布",gender_htmlfile)
    print("性别统计执行结束,%s文件生成"% gender_htmlfile)

    zh.plt_Yes(yesnumber_htmlfile)
    print("点赞数统计执行结束,%s文件生成"% yesnumber_htmlfile)

    zh.static_tag(catagory_htmlfile,tag_jsonfile)
    print("统计标签执行结束,%s文件生成"% yesnumber_htmlfile)

    zh.plt_Tibet_map(map_jsonfile)
    print("统计标签执行结束,%s文件生成"% map_jsonfile)






