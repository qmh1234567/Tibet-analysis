import json

def read_json_file(filename):
    with open(filename,'r',encoding='utf-8') as f:
        dict1=json.load(f)
    return dict1

# 提取每篇文章的信息
def extract_artcile(dict_comment):
    dict_total={
        "title":[],
        "dict_every_article_info":[]
    }
    for itemdict in dict_comment:
        if itemdict["info"]["title"] in dict_total["title"]:
            dict_every_article_info["gender"].append(itemdict["info"]["gender"])
            dict_every_article_info["content"].append(itemdict["info"]["content"])
            dict_every_article_info["business"].append(itemdict["user_info"]["business"])
            dict_every_article_info["locations"].append(itemdict["user_info"]["locations"])
            dict_every_article_info["majors"].append(itemdict["user_info"]["majors"])
        else:
            dict_total["title"].append(itemdict["info"]["title"])
            dict_every_article_info={
                "gender":[],
                "content":[],
                "business":[],
                "locations":[],
                "majors":[],
            }
            if dict_every_article_info:
                dict_total["dict_every_article_info"].append(dict_every_article_info)       
    return dict_total
    
#  将文章分类
def select_catalogue(tag_file,dict_total,jsonfile):
    all_dict={
        "政治":[],
        "文化":[],
        "社会":[],
        "其他":[],
    }
    tag_dict=read_json_file(tag_file)
    for item in range(len(dict_total["title"])):
        dict_signal={
            "title":dict_total["title"][item],
            "dict_every_article_info":dict_total["dict_every_article_info"][item]
        }
        title=dict_total["title"][item]
        if title in tag_dict["政治标题"]:
            all_dict["政治"].append(dict_signal)
        elif title in tag_dict["文化标题"]:
            all_dict["文化"].append(dict_signal)
        elif title in tag_dict["社会标题"]:
            all_dict["社会"].append(dict_signal)
        else:
            all_dict["其他"].append(dict_signal)
    with open(jsonfile,"w",encoding='utf-8') as f:
        jsobj=json.dumps(all_dict)
        f.write(jsobj)
        print("写入%s文件成功"% jsonfile)


if __name__ == '__main__':
    filename='./../../Resources/jsonfiles/zh_comment.json'
    tag_file='./../../Resources/jsonfiles/tag.json'
    jsonfile='./../../Resources/jsonfiles/processed_zh_comment.json'
    dict_zhihu_comment=read_json_file(filename)
    dict_total=extract_artcile(dict_zhihu_comment)
    select_catalogue(tag_file,dict_total,jsonfile)
    