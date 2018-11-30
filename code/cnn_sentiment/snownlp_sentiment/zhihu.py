import requests
from pyquery import *
import re
import time
import json



dict_zh={
    'title':[],
    'author':[],
    'gender':[],
    'member_tag':[],
    'yes_number':[],
    'comment_number':[],
    'topic_tag':[],
    'essay_url':[],
}

def get_topic():
    url='https://www.zhihu.com/topic/19559654/top-answers'
    my_cookies='_zap=4fbaed0d-c8a2-4246-98be-6d19751c7072; d_c0="AMBlcBIVNg6PTobW-UWbvub3jUn76wmcvUE=|1536927571"; _xsrf=iijilMzpQRIDVkJIOPQ38BpyOsbOBQCN; __gads=ID=d97e8645acf20e12:T=1542008229:S=ALNI_MZyItIPyJhsDjtGmvO7mjB2fwYkug; q_c1=fac64d02b67c45619eb4ade1b10690b9|1543396310000|1537260806000; __utmz=51854390.1543396313.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmv=51854390.000--|3=entry_date=20180918=1; tst=r; __utma=51854390.1131470212.1543396313.1543396313.1543405569.2; l_cap_id="ZDMzODc4Mzk5M2YwNDA2M2JiZTlhZDJhZWIyMzZlOTc=|1543461996|96cf47131b653c0db6a2aef40664f5b961be7d02"; r_cap_id="YjY1MmU1MGVlNmZkNDExZmE0YWFjOWZlYWEyZjgyN2M=|1543461996|adcf88a319890b95ea95970c2823732863d72235"; cap_id="MzRmMjIzZjViYTM2NDA0ODlhZDdhODJhYmM0ZjVjZGI=|1543461996|30f02ed565088b669d5c11ebe8b877a10693e1d7"; tgw_l7_route=4902c7c12bebebe28366186aba4ffcde; capsion_ticket="2|1:0|10:1543539590|14:capsion_ticket|44:MDVkNzAwZjlmZGFhNDI5OGFjZDE2ODE0NzU4MzFkNTQ=|2438d17deada282bd01dcf54f9e15f56725f14926a245ecbf90bc84f4ee4e36f"'
    headers={
        'content-type':'application/json',
        'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'cookie':my_cookies,
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36'
    }


    nexturl="http://www.zhihu.com/api/v4/topics/19559654/feeds/essence?include=data%5B%3F%28target.type%3Dtopic_sticky_module%29%5D.target.data%5B%3F%28target.type%3Danswer%29%5D.target.content%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%3Bdata%5B%3F%28target.type%3Dtopic_sticky_module%29%5D.target.data%5B%3F%28target.type%3Danswer%29%5D.target.is_normal%2Ccomment_count%2Cvoteup_count%2Ccontent%2Crelevant_info%2Cexcerpt.author.badge%5B%3F%28type%3Dbest_answerer%29%5D.topics%3Bdata%5B%3F%28target.type%3Dtopic_sticky_module%29%5D.target.data%5B%3F%28target.type%3Darticle%29%5D.target.content%2Cvoteup_count%2Ccomment_count%2Cvoting%2Cauthor.badge%5B%3F%28type%3Dbest_answerer%29%5D.topics%3Bdata%5B%3F%28target.type%3Dtopic_sticky_module%29%5D.target.data%5B%3F%28target.type%3Dpeople%29%5D.target.answer_count%2Carticles_count%2Cgender%2Cfollower_count%2Cis_followed%2Cis_following%2Cbadge%5B%3F%28type%3Dbest_answerer%29%5D.topics%3Bdata%5B%3F%28target.type%3Danswer%29%5D.target.annotation_detail%2Ccontent%2Chermes_label%2Cis_labeled%2Crelationship.is_authorized%2Cis_author%2Cvoting%2Cis_thanked%2Cis_nothelp%3Bdata%5B%3F%28target.type%3Danswer%29%5D.target.author.badge%5B%3F%28type%3Dbest_answerer%29%5D.topics%3Bdata%5B%3F%28target.type%3Darticle%29%5D.target.annotation_detail%2Ccontent%2Chermes_label%2Cis_labeled%2Cauthor.badge%5B%3F%28type%3Dbest_answerer%29%5D.topics%3Bdata%5B%3F%28target.type%3Dquestion%29%5D.target.annotation_detail%2Ccomment_count%3B&limit="+"10"+"&offset=0"
    for i in range(20):    
        res=requests.session().get(url=nexturl,headers=headers)
        if res.status_code== 200:
            print("正在爬取...请等待")
            dict_content=res.json()
            nexturl = dict_content['paging']['next']
            dict_zh=query_info(dict_content)
        else:
            print("获取失败")
    return dict_zh


'''获取话题的标题、作者、点赞数、以及评论数'''
def query_info(dict_content):
    for dict1 in dict_content["data"]:
        dict_zh["author"].append(dict1['target']['author']['name']) 
        dict_zh["member_tag"].append(dict1['target']['author']['edu_member_tag']['member_tag'])
        dict_zh["gender"].append(dict1['target']['author']['gender'])
        try:
            dict_zh["title"].append(dict1['target']['question']['title'])
            id1=dict1['target']['question']['id']
            id2=dict1['target']['id']
            childurl="https://www.zhihu.com/question/"+str(id1)+"/answer/"+str(id2)   #需要注意问题的网址和文章的网址不一样
            dict_zh["essay_url"].append(childurl)
            get_childtopic(childurl)
        except:
            dict_zh["title"].append(dict1['target']['title'])
            childurl=dict1['target']['url']
            dict_zh["essay_url"].append(childurl)
            get_childtopic(childurl)
        dict_zh["yes_number"].append(dict1['target']['voteup_count'])
        dict_zh["comment_number"].append(dict1['target']['comment_count'])
    return dict_zh

    
# 获取话题的标签
def query_tag(res):
    doc_obj=PyQuery(res.text)
    # 获取两种类型的文章标签
    tag=doc_obj('.QuestionHeader-topics').find('.Popover').text()
    if not tag.strip():
        tag=doc_obj('.Post-topicsAndReviewer').find('.Popover').text()
    taglist=tag.split(' ')
    dict_zh['topic_tag'].append(taglist)

# 获取子话题
# childurl="http://www.zhihu.com/api/v4/answers/185556632"
def get_childtopic(childurl):
    cookies="_zap=4fbaed0d-c8a2-4246-98be-6d19751c7072; d_c0=\"AMBlcBIVNg6PTobW-UWbvub3jUn76wmcvUE=|1536927571\"; _xsrf=iijilMzpQRIDVkJIOPQ38BpyOsbOBQCN; __gads=ID=d97e8645acf20e12:T=1542008229:S=ALNI_MZyItIPyJhsDjtGmvO7mjB2fwYkug; q_c1=fac64d02b67c45619eb4ade1b10690b9|1543396310000|1537260806000; __utmz=51854390.1543396313.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmv=51854390.000--|3=entry_date=20180918=1; tst=r; __utma=51854390.1131470212.1543396313.1543396313.1543405569.2; capsion_ticket=\"2|1:0|10:1543539590|14:capsion_ticket|44:MDVkNzAwZjlmZGFhNDI5OGFjZDE2ODE0NzU4MzFkNTQ=|2438d17deada282bd01dcf54f9e15f56725f14926a245ecbf90bc84f4ee4e36f\"; l_n_c=1; n_c=1; tgw_l7_route=931b604f0432b1e60014973b6cd4c7bc; l_cap_id=\"OWIyMWM1NDVhNzU2NGUzNmFkZDFlZGRjMzJiNGVkNDA=|1543546012|ef05c17fb9a4da8e82b6dfb84cfef976fca79ae0\"; r_cap_id=\"NmMwZDY1ZDg1NTNiNDZjZGE1NGNkYzgwY2ZhYWNmZDM=|1543546012|dfa3d463208ca05fe79f6b9cf08c6a81b9d3c414\"; cap_id=\"MmQ0MWJhMDJmODFmNGMxNmEwNDljZDVjNzMxMmJjMGI=|1543546012|8896a3ae1f58adab3587a990991c6bdec9fcf729\""
    headers={
        'content-type':'application/json',
        "accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "cookie":cookies,
        "user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36",
    }
    try:
        res=requests.session().get(url=childurl,headers=headers)
        if res.status_code== 200:
            query_tag(res)
        else:
            print("获取失败")
        return res
    except Exception as e:
        print('error:',e)


def write_to_jsonfile(dict_obj):
    jsobj=json.dumps(dict_obj)
    with open("zhihu.json",'w') as f:
        f.write(jsobj)
    print("写入文件成功")

if __name__ == '__main__':
    dict_zh=get_topic()
    print(dict_zh)
    write_to_jsonfile(dict_zh)
