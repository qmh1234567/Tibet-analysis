from pyecharts import Map,Bar,Pie,Graph,Geo,Funnel
import json
import re
import pyhanlp

def Read_file(jsonfile):
    with open(jsonfile,'r',encoding='utf-8') as f:
        dict_content=json.load(f)
    return dict_content

# 提取每种类别的文章的评论者性别、地点等信息
def extract_article_list(dict_comment):
    article_gender_list=[] # 性别
    article_place_list=[] # 位置
    article_business_list=[] # 行业
    article_major_list=[]  # 专业

    # 政治话题的评论者性别分布
    for article_info in dict_comment["社会"]:
        # 性别
        article_gender=article_info["dict_every_article_info"]["gender"]
        article_gender_list.extend(article_gender)
        # 位置
        article_place=article_info["dict_every_article_info"]["locations"]
        article_place_l=[]
        for itemlist in article_place:
            article_place_l.extend(itemlist)
        article_place_list.extend(article_place_l)
        # 行业
        article_business=article_info["dict_every_article_info"]["business"]
        article_business_list.extend(article_business)
        # 专业
        article_major=article_info["dict_every_article_info"]["majors"]
        article_mj=[]
        for itemlist in article_major:
            article_mj.extend(itemlist)
        article_major_list.extend(article_mj)

    return article_gender_list,article_place_list,article_business_list,article_major_list

# 性别分布
def plt_Pie(name,gender,htmlfile):
    # 1 男  0 女  -1 未知
    value=[gender.count(1),gender.count(0),gender.count(-1)]
    attr=['男','女','未知']
    pie=Pie(name,width=500,height=300)
    pie.add("性别",attr,value,is_label_show=True)
    pie.render(htmlfile)

# 对城市名进行清理
def clean_cities_name(place):
    # 过滤掉错误的城市名字
    str_place=" ".join(place)
    place_list=pyhanlp.HanLP.segment(str_place)
    correct_place=re.findall(r'(\w+)/ns',str(place_list))
    cleaned_place=[]
    # 删掉市
    for item in correct_place:
        item=item.replace(u'伊犁','伊犁哈萨克自治州')
        except_list=['市','广安','黔东南','丽江','大兴安岭']
        for i in except_list:
            item=item.replace(i,'')
        cleaned_place.append(item)
    # 读取城市名单,过滤掉世界其他地方的地名
    with open("./cities.txt",'r',encoding='utf-8') as f:
        cities_name=f.read().split("、")
    wrong_cities=set(cleaned_place).difference(set(cities_name))
    city_list=list(set(cleaned_place).difference(set(wrong_cities))) # 去重后的正确城市列表
    city_name=[]  # 没去重的正确城市列表
    for item in cleaned_place:
        if item not in wrong_cities:
            city_name.append(item)
    return city_name,city_list

# 地点分布
def plt_place(place,htmlfile):
    city_name,city_list=clean_cities_name(place) # 对城市名进行清理
    city_dict = {city_list[i]:0 for i in range(len(city_list))}
    for i in range(len(city_list)):
        city_dict[city_list[i]]=city_name.count(city_list[i])
    print(city_dict)
    # 根据数量排序
    sort_dict=sorted(city_dict.items(),key=lambda d:d[1],reverse=True)
    city_name=[]
    city_num=[]
    for i in range(len(sort_dict)):
        city_name.append(sort_dict[i][0])
        city_num.append(sort_dict[i][1])
    data=list(zip(city_name,city_num))
    geo=Geo("评论者位置分布","数据来源知乎",title_color="#fff", title_pos="center",
width=1200, height=600, background_color='#404a59')
    attr,value=geo.cast(data)
    # 注意修改显示范围
    geo.add('城市',attr,value,visual_range=[0,20],
    visual_text_color='#fff',symbol_size=10,
    is_visualmap=True,is_picewise=False,visual_split_number=10)
    geo.render(htmlfile)

# 清理专业
def clear_business(majorlist):
    # 根据专业名单过滤掉一些不重要的专业
    with open("./major.txt",'r',encoding='utf-8') as f:
        majors=f.read().split(",")
    new_major_list=[]
    for item in majorlist:
        if item in majors:
            new_major_list.append(item)
        else:
            new_major_list.append('其他')
    return new_major_list


# 行业分布
def plt_business(name,business,htmlfile):
    business_list=list(set(business))
    business_dict={ business_list[i]: 0 for i in range(len(business_list)) }
    for i in range(len(business_list)):
        business_dict[business_list[i]]=business.count(business_list[i])
    # 检查键值是否为空
    if business_dict.__contains__(''):
        business_dict.update({'未知':business_dict.pop('')})
    # 按照value降序排列
    business_dict=sorted(business_dict.items(),key=lambda x:x[1],reverse=True)
    print(business_dict)
    attr=[]
    value=[]
    for i in range(len(business_dict)):
        attr.append(business_dict[i][0])
        value.append(business_dict[i][1])
    print("长度为",len(attr))
    bar=Bar("评论用户行业分布")
    bar.use_theme("roma")
    bar.width=1000
    bar.height=340
    bar.add(
        "行业",
        attr,
        value,
        xaxis_interval=0,
        xaxis_rotate=45,
        is_datazoom_show=True,
    )
    bar.render(htmlfile)

if __name__ == '__main__':
    gender_htmlfile='./../../Resources/htmlfiles/article_comment_gender.html'
    location_htmlfile='./../../Resources/htmlfiles/loaction.html'
    business_htmlfile='./../../Resources/htmlfiles/business.html'
    major_htmlfile='./../../Resources/htmlfiles/major.html'
    comment_class_file='./../../Resources/htmlfiles/comment_class.html'
    jsonfile1='./../../Resources/jsonfiles/processed_zh_comment.json'

    dict_comment=Read_file(jsonfile1)
    attr=list(dict_comment.keys())
    value=[]
    for key in attr:
        value.append(len(dict_comment[key]))
    bar=Bar('带评论的文章的类别分布')
    bar.add('类别',attr,value,is_label_show=True,make_point=['average'])
    bar.show_config()
    bar.render(comment_class_file)

    # 提取需要的信息
    article_gender_list,article_place_list,article_business_list,article_major_list=extract_article_list(dict_comment)

    plt_Pie("社会话题的性别分布",article_gender_list,gender_htmlfile)
    print("评论用户性别分布执行结束,%s文件生成"% gender_htmlfile)

    plt_place(article_place_list,location_htmlfile)
    print("评论用户位置分布执行结束,%s文件生成"% location_htmlfile)

    plt_business("评论用户行业分布",article_business_list,business_htmlfile)
    print("评论用户行业分布执行结束,%s文件生成"% business_htmlfile)

    major_list=clear_business(article_major_list)
    plt_business("评论用户专业分布",major_list,major_htmlfile)
    print("评论用户专业分布执行结束,%s文件生成"% major_htmlfile)

    