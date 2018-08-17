# -*- coding:utf-8 -*-
import os
import pandas as pd

data_in = './data/FOOD_DATA.csv'
data_out = './result'

def clean_data(data_path):
    raw_data = pd.read_csv(data_path)
    #print(raw_data.head())
    data_after_drop = raw_data.drop(['店铺所在城市'], axis=1)#丢弃上海，默认均为上海地域，保留店铺URL，留着查看玩玩
    #print(len(data_after_drop))
    data_after_dropna = data_after_drop.dropna()
    #print(len(data_after_dropna))
    data_after_dropdup = data_after_dropna.drop_duplicates()
    #print(len(data_after_dropdup))
    return data_after_dropdup

def main():
    if not os.path.exists(data_out):
        os.mkdir(data_out)

    clean_data(data_in)


if __name__ == '__main__':
    main()
