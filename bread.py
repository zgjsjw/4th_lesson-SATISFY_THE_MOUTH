# -*- coding:utf-8 -*-
import os
import pandas as pd
from pyecharts import Bar
from pyecharts import Pie
import matplotlib.pyplot as plt

data_in = './data/City_bread_Data.xlsx'
data_out = './result'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


import json
from urllib.request import urlopen, quote
def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = 'd8FVvCWpwUdIt69uAVWD8yNnc7V49v3h' # 浏览器端密钥
    address = quote(address) # 由于本文地址变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + address  + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode()
    temp = json.loads(res)
    lat = temp['result']['location']['lat']
    lng = temp['result']['location']['lng']
    return lat, lng


def clean_data(data_path):
    raw_data = pd.read_excel(data_path)
    data_after_drop = raw_data[raw_data['City'] == '上海'].drop(['City'], axis=1)  # 丢弃上海，默认均为上海地域，保留店铺URL，留着查看玩玩
    print('初始数据总共有{}条;\n'.format(len(data_after_drop)))
    data_na = data_after_drop.apply(lambda t: t.isnull().sum())  # 查看数据缺失情况
    print('数据缺失情况为：\n', data_na)
    data_after_dropna = data_after_drop.dropna()  # 由于没有那种关键性必须保留数据，所以对缺失数据做删除操作
    data_after_dropdup = data_after_dropna.drop_duplicates()  # 对重复数据进行删除
    data_na_after_clean = data_after_dropdup.apply(lambda t: t.isnull().sum())
    print('\n清洗后数据还有{}条;\n'.format(len(data_after_dropdup)))
    print('数据缺失情况为：\n', data_na_after_clean)
    #data_after_dropdup['lat_lng'] = data_after_dropdup['Location'].map(lambda t: getlnglat("上海" + t))
    data_after_dropdup.to_csv(os.path.join(data_out, 'new_bread.csv'), encoding='utf-8-sig', index=False)
    return data_after_dropdup


# 首先来看下菜系品种
def kinds_of_bread(data):
    kinds = data['Cuisine'].value_counts()
    kinds_fig = Bar("Cuisine")
    kinds_fig.add('', kinds.index, kinds.values, xaxis_rotate=60, bar_category_gap='50%', is_label_show=True)
    kinds_fig.show_config()
    kinds_fig.render(os.path.join(data_out, 'kind_of_bread.html'))


# 再来看看餐厅星级分布情况
def stars_of_bread(data):
    stars = data['Star'].value_counts().sort_index()
    print(stars.index)
    starts_fig = Pie("Stars of Restaurant", title_pos='center')
    starts_fig.add('', ['0星','2星','3星','3.5星','4星','4.5星','5星'], stars.values, legend_pos='left',
                   legend_orient='vertical', is_label_show=True, rosetype='radius')
    starts_fig.show_config()
    starts_fig.render(os.path.join(data_out, 'stars_of_bread.html'))

#尝试看下能不能分析地域图（用echarts或者Tableau）
#需等待百度地图的配额提升才能进行地址和经纬度之间的转换

#人均消费分析
def pcc_of_bread(data):
    pcc = data['Per_Consumption']
    print("人均最高{}元".format(pcc.max()))
    print("人均最低{}元".format(pcc.min()))
    print("平均消费{0:.2f}元".format(pcc.mean()))
    section = ['50以下', '51~100', '101~150', '151~200', '201~300', '301~500', '501~1000', '1000以上']
    number_result = pd.Series([0, 0, 0, 0, 0, 0, 0, 0], index=section)
    for t in pcc:
        if t <= 50:
            number_result['50以下'] = number_result['50以下'] + 1
        elif 50<t<=100:
            number_result['51~100'] = number_result['51~100'] + 1
        elif 100<t<=150:
            number_result['101~150'] = number_result['101~150'] + 1
        elif 150<t<=200:
            number_result['151~200'] = number_result['151~200'] + 1
        elif 200<t<=300:
            number_result['201~300'] = number_result['201~300'] + 1
        elif 300<t<=500:
            number_result['301~500'] = number_result['301~500'] + 1
        elif 500<t<=1000:
            number_result['501~1000'] = number_result['501~1000'] + 1
        else:
            number_result['1000以上'] = number_result['1000以上'] + 1

    pcc_fig = Bar('人均消费分布', title_pos='center')
    pcc_fig.add('',number_result.index, number_result.values, is_label_show=True, bar_category_gap='50%',
                label_color= ['#123456'], is_convert=True, label_pos='right')
    pcc_fig.show_config()
    pcc_fig.render(os.path.join(data_out, 'pcc_of_bread.html'))
    print('我（价）最（格）想（最）吃（贵）：\n', data.loc[data['Per_Consumption'].idxmax()])


#评论数分析
def comments_of_bread(data):
    comments = data['Comments']
    print("评论最多{}".format(comments.max()))
    print('人气最旺：\n', data.loc[data['Comments'].idxmax()])
    print("评论最少{}".format(comments.min()))
    comments_fig = Bar('评论数分布情况')
    comments_fig.add('', comments.index, comments.values, mark_point =['max'], is_xaxis_show=False, label_color=['#224466'])
    comments_fig.show_config()
    comments_fig.render(os.path.join(data_out, 'comments_of_bread.html'))



#评分指标分析，根据5分以下，5~6分，。。。，9分以上来查看下

#星级和评分的关系，看下雷达图的情况，看下评分最高的几个餐厅，看下最贵的菜系和最受欢迎的菜系，查看下星级--评分情况（散点图）

#机器学习部分


def main():
    if not os.path.exists(os.path.join(data_out, 'new_bread.csv')):
        bread_data = clean_data(data_in)
    else:
        bread_data = pd.read_csv(os.path.join(data_out, 'new_bread.csv'))

    #kinds_of_bread(bread_data)
    #stars_of_bread(bread_data)
    pcc_of_bread(bread_data)
    comments_of_bread(bread_data)



if __name__ == '__main__':
    main()
