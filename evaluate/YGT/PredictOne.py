import os
import lightgbm as lgb
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
from sys import path
path.append(os.path.dirname(os.path.realpath(__file__)))
import kernel as k
# from . import kernel as k

import re

beginDate = k.beginDate
endDate = k.endDate
newdisk_path = k.newdisk_path
property_path = k.property_path
address_path = k.address_path
medprice_path = k.medprice_path
arealabel_path = k.arealabel_path
platelabel_path = k.platelabel_path
modulelable_path = k.modulelabel_path
cache_path_model = os.path.dirname(os.path.realpath(__file__))+'/cache/model_%s-%s.txt' % (beginDate, endDate)
cache_path_guapai = os.path.dirname(os.path.realpath(__file__))+'/cache/guapai_%s-%s.hdf' % (beginDate, endDate)

# 加载基础数据 & 训练数据
print('加载基础 & 训练数据...')
if not (os.path.exists(cache_path_model) and os.path.exists(cache_path_guapai)):
    print("请先训练模型！")    # 报错信息记得返回到前端
else:
    meta_df = pd.read_hdf(cache_path_guapai, 'meta')
    gbm = lgb.Booster(model_file=cache_path_model)

# 加载其它训练信息
print('加载模型参数...')
if not (os.path.exists(medprice_path) and os.path.exists(arealabel_path) and os.path.exists(platelabel_path)\
        and os.path.exists(modulelable_path)):
    print("模型参数不全！")    # 报错信息记得返回到前端
else:
    med_price = pd.read_csv(medprice_path)
    arealabel = pd.read_csv(arealabel_path,usecols=["label","area"])
    arealabel.set_index(["area"],inplace=True)
    arealabel = arealabel.to_dict()["label"]
    platelabel = pd.read_csv(platelabel_path,usecols=["label","plate"])
    platelabel.set_index(["plate"],inplace=True)
    platelabel = platelabel.to_dict()["label"]
    modulelabel = pd.read_csv(modulelable_path,usecols=["Module","unit_price"])
    modulelabel.set_index(["Module"],inplace=True)

# 使用训练过的数据进行预测

def predict(houses):

    # cols = ["项目名称","住宅","竣工年份","总楼层数","所属层","建筑面积(m2)","楼盘编号"]
    cols = ["address","house_type","time","all_floor","floor","acreage","NewDiskID"]
    data = DataFrame(data=houses,columns=cols)
    data.reset_index(inplace=True)
    x_data_raw = data
    x_data = data[['index','time','all_floor','floor', 'acreage', 'NewDiskID']]    # r1
    x_data = pd.merge(x_data, meta_df.drop(['PropertyID'], axis=1), on='NewDiskID', how='inner')    # r2
    x_data = pd.merge(x_data, med_price, on='NewDiskID', how='left')    # r3

    floor_map = lambda x: list(pd.cut(x, [0, 3, 6, 9, np.inf], labels=['低层', '多层', '小高层', '高层']))
    x_data['floor_section'] = floor_map(x_data.floor)
    x_data['time'] = x_data.time.apply(lambda x: min(2018 - x, 100) if 0 < x <= 2018 else None)
    x_data['area'] = x_data.area.apply(lambda x: arealabel[x])
    x_data['Plate'] = x_data.Plate.apply(lambda x: platelabel[x])
    i = pd.Series(range(0, modulelabel.shape[0]), index=modulelabel.index).to_dict()
    x_data.Module = x_data.Module.map(i)
    coors = k.make_coordinates(x_data.Coordinates.values)
    coors.index = x_data.index
    x_data = pd.concat((x_data, coors), axis=1).drop('Coordinates', axis=1)
    x_data.floor_section = x_data.floor_section.map({'低层': 0, '小高层': 1, '多层': 2, '高层': 3})  # r4
    # x_data.drop('NewDiskID', inplace=True, axis=1)
    # x_data.drop('unit_price', axis=1, inplace=True)

    # 顺序要和训练时的x_train和x_test一致
    x_data = x_data[["acreage","all_floor","floor","time","area","Plate","Module","floor_section","loc_x","loc_y","median"]]
    y_pred = gbm.predict(x_data)
    x_data_raw["predict_price"] = y_pred
    return x_data_raw

def address_filter(address):
    re.compile('^*[路弄街]]').split()[0]

def find_DiskID(address):

    # 为了改进体验，希望以后可以在网页中输入地址时就判断出是否存在某个小区，而不是在预测前再返回错误信息！
    # address = address_filter(address)
    address_df = pd.read_csv(address_path, usecols=['RoadLaneNo', 'NewDiskID'])
    address_df.rename(columns={'RoadLaneNo': 'address'}, inplace=True)
    address_all = pd.merge(meta_df[["NewDiskID","name"]],address_df, how='left', on='NewDiskID').dropna(axis=0, how='any')
    address_fit = address_all[address_all.address.str.contains(address)]
    address_fit = address_fit.head(1)   # 取第一个匹配的
    if address_fit.empty:
        print("找不到对应的小区！")      # 报错信息记得返回到前端
        return (None,None,None)
    else:
        print(address_fit)
        NewDiskID = address_fit.iat[0, 0]
        return (NewDiskID, address_fit.iat[0,1],address_fit.iat[0,2])

def find_DiskID_ByName(diskname_input):
    address_df = pd.read_csv(address_path, usecols=['RoadLaneNo', 'NewDiskID'])
    address_df.rename(columns={'RoadLaneNo': 'address'},inplace=True)
    name_all = pd.merge(meta_df[['NewDiskID','name']],address_df, how='left', on='NewDiskID').dropna(axis=0, how='any')
    name_fit = name_all[name_all.name.str.contains(diskname_input)]
    name_fit = name_fit.head(1)
    if name_fit.empty:
        print("找不到对应的小区！")
        return (None,None,None)
    else:
        print(name_fit)
        NewDiskID = name_fit.iat[0, 0]
        return (NewDiskID, name_fit.iat[0,1], name_fit.iat[0,2])


def init(district,address,house_type,time,all_floor,floor,acreage):
    NewDiskID = find_DiskID(address)[0]
    return[address, house_type, time, all_floor, floor, acreage, NewDiskID]

if __name__ == '__main__':  # 前端传入如下数据（可以是多套房子），组装数据后进行预测，返回给前端预测值。
    district = "黄浦区"
    address = "西藏南路1739弄"
    house_type = "住宅"
    time = 2004
    all_floor = 20
    floor = 10
    acreage = 103
    houses = []
    houses.append(init(district,address,house_type,time,all_floor,floor,acreage))
    prediction = predict(houses)
    print(prediction["predict_price"][0])

