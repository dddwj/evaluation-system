import pandas as pd
import os
import numpy as np

from pandas.core.frame import DataFrame
import lightgbm as lgb

class predict:
    def __init__(self):
        self.setConstants()

    def setConstants(self):
        self.houses = []
        from ..models import models_logs
        model = models_logs.objects.get(inUseFlag=1, trainSuccess=1)
        print("将使用%s号模型" % model.id)
        model_id = model.id
        self.beginDate = model.startMonth.strftime('%Y-%m')
        self.endDate = model.endMonth.strftime('%Y-%m')
        '''
            测试用
            print("将使用%s号模型" % 57)
            model_id = 57
            self.beginDate = '2017-02'
            self.endDate = '2017-03'
        '''
        # 新房数据表路径
        self.newdisk_path = os.path.dirname(os.path.realpath(__file__)) + '/data/AD_NewDisk.csv'
        # 房源属性数据表路径
        self.property_path = os.path.dirname(os.path.realpath(__file__)) + '/data/AD_Property.csv'
        # 地址数据表路径
        self.address_path = os.path.dirname(os.path.realpath(__file__)) + '/data/AD_NewDiskAddress.csv'
        # 挂牌数据路径
        self.data_path = os.path.dirname(os.path.realpath(__file__)) + '/data/'
        self.model_dir = os.path.dirname(os.path.realpath(__file__)) + '/cache/model_%s/' % (model_id)
        if not os.path.exists(self.model_dir):
            print("模型训练有问题")
            return "模型训练有问题"
        # 房源中位数价格路径
        self.medprice_path = self.model_dir + '/medprice.csv'
        # 区名特征化路径
        self.arealabel_path = self.model_dir + '/arealabel.csv'
        # 板块名特征化路径
        self.platelabel_path = self.model_dir + '/platelabel.csv'
        # 内中外环特征化路径
        self.modulelabel_path = self.model_dir + 'modulelabel.csv'
        # 模型缓存路径
        self.cache_path_model = self.model_dir + '/model.txt'
        # 挂牌缓存路径
        self.cache_path_guapai = os.path.dirname(os.path.realpath(__file__)) + '/cache/guapai_%s-%s.hdf' % (self.beginDate, self.endDate)
        # 预处理缓存路径
        self.cache_path_feats = os.path.dirname(os.path.realpath(__file__)) + '/cache/feats_%s-%s.hdf' % (self.beginDate, self.endDate)

        self.meta_df = pd.read_hdf(self.cache_path_guapai, 'meta')
        self.gbm = lgb.Booster(model_file=self.cache_path_model)
        self.med_price = pd.read_csv(self.medprice_path)
        self.arealabel = pd.read_csv(self.arealabel_path, usecols=["label", "area"])
        self.arealabel.set_index(["area"], inplace=True)
        self.arealabel = self.arealabel.to_dict()["label"]
        self.platelabel = pd.read_csv(self.platelabel_path, usecols=["label", "plate"])
        self.platelabel.set_index(["plate"], inplace=True)
        self.platelabel = self.platelabel.to_dict()["label"]
        self.modulelabel = pd.read_csv(self.modulelabel_path, usecols=["Module", "unit_price"])
        self.modulelabel.set_index(["Module"], inplace=True)

    def make_coordinates(self, data):
        coors = []
        # for i in tqdm(data):
        for i in data:
            if type(i) == str and i != '公寓' and i != '商业' and i != '其它':
                coors.append(i.split(','))
            else:
                coors.append([None, None])
        coors = pd.DataFrame(coors, columns=['loc_x', 'loc_y'])
        # coors=pd.DataFrame([coor.split(',') for coor in all_df.Coordinates],columns=['loc_x','loc_y'],index=all_df.index)
        coors = coors.astype(float)
        return coors

    def find_DiskID(self, address):
        # 为了改进体验，希望以后可以在网页中输入地址时就判断出是否存在某个小区，而不是在预测前再返回错误信息！
        # address = address_filter(address)
        address_df = pd.read_csv(self.address_path, usecols=['RoadLaneNo', 'NewDiskID'])
        # address_df = tools.read_basic_table("AD_NewDiskAddress")
        address_df.rename(columns={'RoadLaneNo': 'address'}, inplace=True)
        address_all = pd.merge(self.meta_df[["NewDiskID", "name"]], address_df, how='left', on='NewDiskID').dropna(axis=0,
                                                                                                              how='any')
        address_fit = address_all[address_all.address.str.contains(address)]
        address_fit = address_fit.head(1)  # 取第一个匹配的
        if address_fit.empty:
            print("找不到对应的小区！")  # 报错信息记得返回到前端
            return (None, None, None)
        else:
            print(address_fit)
            NewDiskID = address_fit.iat[0, 0]
            return (NewDiskID, address_fit.iat[0, 1], address_fit.iat[0, 2])

    ###############################################
    ## 需要改写，将查找功能放到数据库，不要在本地查找。##
    def find_DiskID_ByName(self, diskname_input):
        address_df = pd.read_csv(self.address_path, usecols=['RoadLaneNo', 'NewDiskID'])
        address_df.rename(columns={'RoadLaneNo': 'address'}, inplace=True)
        name_all = pd.merge(self.meta_df[['NewDiskID', 'name']], address_df, how='left', on='NewDiskID').dropna(axis=0,
                                                                                                           how='any')
        name_fit = name_all[name_all.name.str.contains(diskname_input)]
        name_fit = name_fit.head(1)
        if name_fit.empty:
            print("找不到对应的小区！")
            return (None, None, None)
        else:
            print(name_fit)
            NewDiskID = name_fit.iat[0, 0]
            return (NewDiskID, name_fit.iat[0, 1], name_fit.iat[0, 2])

    def predict(self):
        # cols = ["项目名称","住宅","竣工年份","总楼层数","所属层","建筑面积(m2)","楼盘编号"]
        cols = ["address", "house_type", "time", "all_floor", "floor", "acreage", "NewDiskID"]
        data = DataFrame(data=self.houses, columns=cols)
        data.reset_index(inplace=True)
        x_data_raw = data
        x_data = data[['index', 'time', 'all_floor', 'floor', 'acreage', 'NewDiskID']]  # r1
        x_data = pd.merge(x_data, self.meta_df.drop(['PropertyID'], axis=1), on='NewDiskID', how='inner')  # r2
        x_data = pd.merge(x_data, self.med_price, on='NewDiskID', how='left')  # r3

        floor_map = lambda x: list(pd.cut(x, [0, 3, 6, 9, np.inf], labels=['低层', '多层', '小高层', '高层']))
        x_data['floor_section'] = floor_map(x_data.floor)
        x_data['time'] = x_data.time.apply(lambda x: min(2018 - x, 100) if 0 < x <= 2018 else None)
        x_data['area'] = x_data.area.apply(lambda x: self.arealabel[x])
        x_data['Plate'] = x_data.Plate.apply(lambda x: self.platelabel[x])
        i = pd.Series(range(0, self.modulelabel.shape[0]), index=self.modulelabel.index).to_dict()
        x_data.Module = x_data.Module.map(i)
        coors = self.make_coordinates(x_data.Coordinates.values)
        coors.index = x_data.index
        x_data = pd.concat((x_data, coors), axis=1).drop('Coordinates', axis=1)
        x_data.floor_section = x_data.floor_section.map({'低层': 0, '小高层': 1, '多层': 2, '高层': 3})  # r4

        # 顺序要和训练时的x_train和x_test一致
        x_data = x_data[
            ["acreage", "all_floor", "floor", "time", "area", "Plate", "Module", "floor_section", "loc_x", "loc_y",
             "median"]]
        y_pred = self.gbm.predict(x_data)
        x_data_raw["predict_price"] = y_pred
        return x_data_raw["predict_price"]


    def parseTools(self, district, address, house_type, time, all_floor, floor, acreage):
        NewDiskID = self.find_DiskID(address)[0]
        return [address, house_type, time, all_floor, floor, acreage, NewDiskID]

    def addCase(self, district,address,house_type,time,all_floor,floor,acreage):
        self.houses.append(self.parseTools(district,address,house_type,time,all_floor,floor,acreage))



if __name__ == '__main__':  # 前端传入如下数据（可以是多套房子），组装数据后进行预测，返回给前端预测值。
    p = predict()

    district = "黄浦区"
    address = "西藏南路1739弄"
    house_type = "住宅"
    time = 2004
    all_floor = 20
    floor = 10
    acreage = 103
    p.addCase(district,address,house_type,time,all_floor,floor,acreage)
    p.addCase(district,address,house_type,2005,all_floor,floor,acreage)
    res = p.predict()
    print(res)
