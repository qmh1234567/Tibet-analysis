import json
from matplotlib import pyplot as plt
from pyecharts import Map,Bar,Pie
import numpy as np
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
from PIL import Image

def Read_file(jsonfile):
    with open(jsonfile,'r',encoding='utf-8') as f:
        dict_content=json.load(f)
    return dict_content

class ZhihuData(object):
    def __init__(self,dict_content):
        self.title=dict_content['title']
        self.gender=dict_content['gender']
        self.comment_number=dict_content['comment_number']
        self.yes_number=dict_content['yes_number']
        self.topic_tag=dict_content['topic_tag']
    
    # 性别分布
    def plt_Pie(self):
        # 1 男  0 女  -1 未知
        value=[self.gender.count(1),self.gender.count(0),self.gender.count(-1)]
        attr=['男','女','未知']
        pie=Pie("性别分析",width=500,height=300)
        pie.add("z",attr,value,is_label_show=True)
        pie.render('./../../../Resources/htmlfiles/gender.html')

    # 点赞数统计
    def plt_Yes(self):
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
        # bar.show_config()
        bar.render('./../../../Resources/htmlfiles/Number.html') 
    
    def plt_tag(self):
        # 关系图
        # print(self.topic_tag)
        tag_list=[]
        # 变成一级列表
        for li in self.topic_tag:
            for item in li:
                tag_list.append(item)
        new_words=" ".join(tag_list)
        # 图片
        coloring = np.array(Image.open("../../../Resources/wordcloudpics/img.jpg"))
        # simkai.ttf 必填项 识别中文的字体，例：simkai.ttf，
        my_wordcloud = WordCloud(background_color="white", max_words=800,
                     mask=coloring, max_font_size=120, random_state=30, scale=2,font_path="./../../../Resources/fangsong_GB2312.ttf").generate(new_words)
        image_colors = ImageColorGenerator(coloring)
        plt.imshow(my_wordcloud.recolor(color_func=image_colors))
        plt.imshow(my_wordcloud)
        plt.axis("off")
        plt.show()
        # 保存图片
        my_wordcloud.to_file('./../../../Resources/wordcloudpics/1.jpg')





    def plt_Tibet_map(self):
        value = [180, 70, 30, 45, 80,
            90, 5]
        attr = [u'拉萨市', u'日喀则市',  u'昌都市', u'林芝市', u'山南市', 
                u'那曲市', u"阿里地区"]
        map = Map(u"西藏地图示例", width=1200, height=600)
        map.add("", attr, value, maptype=u'西藏',
                is_visualmap=True, visual_text_color='#000')
        map.show_config()
        map.render('../../../Resources/htmlfiles/Tibetmap.html')


if __name__ == '__main__':
    jsonfile="./../../../Resources/jsonfiles/zhihu.json"
    dict_content=Read_file(jsonfile)
    zh=ZhihuData(dict_content)
    zh.plt_Yes()
    zh.plt_Pie()
    zh.plt_tag()




