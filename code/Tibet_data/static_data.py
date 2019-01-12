from pyecharts import Pie,Bar,Map

# age distribution
def plt_Age_distribution():
    pass


# 人口分布
def plt_Tibet_population(htmlfile):
    value = [55, 70, 65, 19, 32,
        19, 9]
    attr = [u'拉萨市', u'日喀则市',  u'昌都市', u'林芝市', u'山南市', 
            u'那曲地区', u"阿里地区"]
    map = Map(u"17年西藏各城市人口分布", width=1200, height=600)
    map.add("万人", attr, value, maptype=u'西藏',visual_range=[0,80],
            is_visualmap=True, visual_text_color='#000')
    map.show_config()
    map.render(htmlfile)


# 西藏人口受教育程度  
def ple_edcation(htmlfile):
    value=[165332,131024,385788,1098474]
    attr=['大学(大专以上)','高中(含中专)','初中','小学']
    pie=Pie("西藏人口受教育程度",width=800,height=300)
    pie.add("学历",attr,value,is_label_show=True)
    pie.render(htmlfile)

# 西藏各城市人口受教育程度  
def ple_edcation_of_eachcity(htmlfile):
    lasa=[0.46,5.56,7.20,9.47,21.10,36.62,19.59] # 拉萨市
    cangdu=[0.04,0.94,1.75,2.36,10.28,39.92,44.7] #昌都市
    sanlan=[0.05,2.41,3.39,5.54,16.08,50.99,21.55] #山南市
    rkz=[0.03,1.68,2.41,3.79,14.87,43.88,33.35] # 日喀则市
    naqu=[0.03,1.16,2.29,2.78,8.38,35.25,50.11] #那曲地区
    ali=[0.05,2.73,4.64,5.22,10.89,32.54,43.93] #阿里地区
    linz=[0.14,2.52,3.46,4.84,14.26,40.60,34.19] #林芝地区
    attr=['研究生','本科','专科','高中','初中','小学','未上过学']
    bar=Bar("2010年西藏各城市学历",width=1000,height=400)
    bar.add("拉萨市",attr,lasa)
    bar.add("昌都市",attr,cangdu)
    bar.add("山南市",attr,sanlan)
    bar.add("日喀则市",attr,rkz)
    bar.add("那曲地区",attr,naqu)
    bar.add("阿里地区",attr,ali)
    bar.add("林芝地区",attr,linz)
    bar.show_config()
    bar.render(htmlfile) 



# 西藏各城市土地面积分布


### 民族构成
def ple_ethnic(htmlfile):
    value=[2716389,245263,40514]
    attr=['藏族','汉族','其他']
    pie=Pie("西藏民族构成",width=800,height=400)
    pie.add("民族",attr,value,is_label_show=True)
    pie.render(htmlfile)


if __name__ == '__main__':
    htmlfile='./population.html'
    edu_html='./education.html'
    ethnic_html='./ethnic.html'
    edu_city_html='./education_of_eachcity.html'
    plt_Tibet_population(htmlfile)
    ple_edcation(edu_html)
    ple_ethnic(ethnic_html)
    ple_edcation_of_eachcity(edu_city_html)