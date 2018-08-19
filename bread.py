# -*- coding:utf-8 -*-
import os
import pandas as pd
from pyecharts import Bar
from pyecharts import Pie

data_in = './data/City_Food_Data.xlsx'
data_out = './result'


def clean_data(data_path):
    raw_data = pd.read_excel(data_path)
    data_after_drop = raw_data[raw_data['City'] == '上海'].drop(['City'], axis=1)  # 丢弃上海，默认均为上海地域，保留店铺URL，留着查看玩玩
    print('初始数据总共有{}条;\n'.format(len(data_after_drop)))
    data_na = data_after_drop.apply(lambda t: t.isnull().sum())  # 查看数据缺失情况
    print('数据缺失情况为：\n', data_na)
    data_after_dropna = data_after_drop.dropna()  # 由于没有那种关键性必须保留数据，所以对缺失数据做删除操作
    data_after_dropdup = data_after_dropna.drop_duplicates()  # 对重复数据进行删除
    data_after_dropdup.to_csv(os.path.join(data_out, 'new_bread.csv'), encoding='utf-8-sig', index=False)
    data_na_after_clean = data_after_dropdup.apply(lambda t: t.isnull().sum())
    print('\n清洗后数据还有{}条;\n'.format(len(data_after_dropdup)))
    print('数据缺失情况为：\n', data_na_after_clean)
    return data_after_dropdup


# 首先来看下菜系品种
def kinds_of_food(data):
    kinds = data['Cuisine'].value_counts()
    kinds_fig = Bar("Cuisine")
    kinds_fig.add('', kinds.index, kinds.values, xaxis_rotate=60, bar_category_gap='50%', is_label_show=True)
    kinds_fig.show_config()
    kinds_fig.render(os.path.join(data_out, 'kind_of_food.html'))


# 再来看看餐厅星级分布情况
def stars_of_food(data):
    stars = data['Star'].value_counts().sort_index()
    print(stars.index)
    starts_fig = Pie("Stars of Restaurant", title_pos='center')
    starts_fig.add('', ['0星','2星','3星','3.5星','4星','4.5星','5星'], stars.values, legend_pos='left',
                   legend_orient='vertical', is_label_show=True, rosetype='radius')
    starts_fig.show_config()
    starts_fig.render(os.path.join(data_out, 'stars_of_food.html'))

#尝试看下能不能分析地域图（用echarts或者Tableau）
#人均消费分析
#评论数分析
#评分指标分析
#星级和评分的关系
#机器学习部分


def main():
    if not os.path.exists(os.path.join(data_out, 'new_bread.csv')):
        food_data = clean_data(data_in)
    else:
        food_data = pd.read_csv(os.path.join(data_out, 'new_bread.csv'))

    # print(food_data.info())

    #kinds_of_food(food_data)
    stars_of_food(food_data)


if __name__ == '__main__':
    main()
