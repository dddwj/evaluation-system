import os
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import re
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

class trainModel:
    def setConstants(self, model_id):
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
            os.mkdir(self.model_dir)
        # 房源中位数价格路径
        self.medprice_path = self.model_dir + '/medprice.csv'
        # 区名特征化路径
        self.arealabel_path = self.model_dir + '/arealabel.csv'
        # 板块名特征化路径
        self.platelabel_path = self.model_dir + '/platelabel.csv'
        # 内中外环特征化路径
        self.modulelabel_path = self.model_dir + 'modulelabel.csv'
        # 模型缓存路径
        self.cache_path = self.model_dir + '/model.txt'

    def setParams(self, model_id):
        from ..models import models_logs
        model = models_logs.objects.get(id=model_id)
        self.beginDate = model.startMonth.strftime('%Y-%m')
        self.endDate = model.endMonth.strftime('%Y-%m')
        self.objective = model.objective
        self.metric = model.metric
        self.learning_rate = model.learning_rate
        self.feature_fraction = model.feature_fraction
        self.bagging_fraction = model.bagging_fraction
        self.max_depth = model.max_depth
        self.num_leaves = model.num_leaves
        self.bagging_freq = model.bagging_freq
        self.min_data_in_leaf = model.min_data_in_leaf
        self.min_gain_to_spilt = model.min_gain_to_split
        self.lambda_l1 = model.lambda_l1
        self.lambda_l2 = model.lambda_l2
        self.verbose = model.verbose

    def name_filter(self, name):
        """小区名正则过滤"""
        n = re.compile('\(|\（|一期|二期').split(name)[0]
        n = re.sub(r'\(.*?\)', '', re.sub(r'\（.*?\）', '', n))
        n = n.strip('*0123456789(（）)')
        n = n.split('第')[0]
        return n

    def address_filter(self, address):
        """小区地址清洗"""
        n = re.compile(',|，|、').split(address)[0]
        n = re.sub(r'\(.*?\)', '', re.sub(r'\（.*?\）', '', n))
        n = n.strip('*0123456789')
        return n

    def time_map(self, time):
        if type(time) == str:
            split_char = '/' if '/' in time else '-'
            return int(time.split(split_char)[0])
        return None

    def floor_map(self, floor):
        # 楼层映射
        return list(pd.cut(floor, [0, 3, 6, 9, np.inf], labels=['低层', '多层', '小高层', '高层']))

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

    def load_guapai(self, name, month):
        """读取挂牌数据"""
        # 训练模型，使用本地数据，提高效率。
        with open(os.path.join(self.data_path, name, '挂牌.txt'), encoding='utf-8') as f:
            l = []
            for i in f.readlines():
                l.append(i.split('\t'))
            df = pd.DataFrame(l)
            drop_col = [0, 15, 16, 18, 19, 20, 21]
            if len(df.columns) == 23:
                drop_col.append(22)
            df.drop(drop_col, axis=1, inplace=True)  # 去除无用列
            df.columns = ['area', 'address', 'name', 'price', 'unit_price', 'acreage', 'room_type', 'all_floor',
                          'floor',
                          'shore', 'house_type', 'fitment', 'time', 'Source', 'house_trait']
            df['month'] = month
            print('load %s' % name)
            return df

    def load_data(self):
        """加载训练数据"""
        print('加载挂牌训练数据...')
        cache_path = os.path.dirname(os.path.realpath(__file__)) + '/cache/guapai_%s-%s.hdf' % (self.beginDate, self.endDate)
        if os.path.exists(cache_path):
            # 加载缓存
            meta_df = pd.read_hdf(cache_path, 'meta')
            all_df = pd.read_hdf(cache_path, 'data')
        else:
            # pool = Pool()
            # files=[i for i in os.listdir(data_path) if os.path.dirname(os.path.realpath(__file__))+'' not in i]
            files = np.unique(
                [datetime.strftime(x, '%Y-%m') for x in list(pd.date_range(start=self.beginDate, end=self.endDate))])
            # files = sorted(files)
            # dfs = [pool.apply_async(load_guapai, (name, month)) for month, name in enumerate(files)]
            # pool.close()
            # pool.join()
            # dfs = [i.get() for i in dfs]
            dfs = []
            for month, name in enumerate(files):
                dfs.append(self.load_guapai(name, str(month)))
            print('共加载%s个月份的挂牌数据...' % len(dfs))
            all_df = pd.concat(dfs, ignore_index=True)

            # 获取经纬度信息
            newdisk_df = pd.read_csv(self.newdisk_path, usecols=['NewDiskID', 'PropertyID', 'NewDiskName', 'Coordinates'])
            # newdisk_df = tools.read_basic_table('AD_NewDisk') 训练模型，使用本地数据，不再读取数据库。
            newdisk_df.rename(columns={'NewDiskName': 'name'}, inplace=True)

            # 获取板块、环线信息
            property_df = pd.read_csv(self.property_path, usecols=['PropertyID', 'Area', 'Plate', 'Module', 'HousingName'])
            property_df.rename(columns={'Area': 'area', 'HousingName': 'name'}, inplace=True)

            # 获取楼盘地址信息
            address_df = pd.read_csv(self.address_path, usecols=['RoadLaneNo', 'NewDiskID'])
            address_df.rename(columns={'RoadLaneNo': 'address'}, inplace=True)

            # merge them
            meta_df = pd.merge(newdisk_df, property_df.drop('name', axis=1), on='PropertyID', how='left')
            # meta_df=pd.merge(meta_df,address_df,on='NewDiskID',how='left')

            # 小区名称清洗
            index = meta_df.name.notnull()
            meta_df.loc[index, 'name'] = meta_df.loc[index, 'name'].apply(self.name_filter)
            all_df.name = all_df.name.apply(self.name_filter)

            address_df.address = address_df.address.apply(self.address_filter)
            all_df.address = all_df.address.apply(self.address_filter)

            # 转换数值类型 str->float
            numerical_columns = ['price', 'unit_price', 'acreage', 'all_floor', 'floor']
            all_df[numerical_columns] = all_df[numerical_columns].astype(float)

            all_df['No'] = range(all_df.shape[0])
            address_match = pd.merge(all_df[['No', 'address']], address_df, on='address', how='inner')
            name_match = pd.merge(all_df[['No', 'name', 'area']], meta_df[['name', 'area', 'NewDiskID']],
                                  on=['name', 'area'],
                                  how='inner')

            match = pd.concat((address_match[['No', 'NewDiskID']], name_match[['No', 'NewDiskID']]), ignore_index=True)
            match.drop_duplicates(keep='first', inplace=True)
            match = match.sort_values('No')

            all_df = all_df.loc[match.No]
            all_df['NewDiskID'] = match.NewDiskID.values
            all_df.drop('No', axis=1, inplace=True)

            all_df = pd.merge(all_df, meta_df[['NewDiskID', 'Coordinates', 'Plate', 'Module']], on='NewDiskID',
                              how='left')
            meta_df.to_hdf(cache_path, 'meta')
            all_df.to_hdf(cache_path, 'data')
        return meta_df, all_df


    def preprocess(self, all_df):
        """特征预处理"""
        print('清洗挂牌数据...')
        cache_path = os.path.dirname(os.path.realpath(__file__)) + '/cache/feats_%s-%s.hdf' % (self.beginDate, self.endDate)
        if os.path.exists(cache_path):
            all_df = pd.read_hdf(cache_path, 'data')
        else:
            # 修正面积
            acreage_log = np.log(all_df.acreage)
            mean = acreage_log.mean()
            std = acreage_log.std()
            i = acreage_log[(acreage_log <= mean + 2 * std) & (acreage_log >= mean - 1 * std)].index
            sns.set({'figure.figsize': (8, 4)})
            sns.boxplot(all_df.loc[i].acreage)
            all_df.loc[i].acreage.describe()
            all_df = all_df.loc[i]

            # 修正单价
            unit_price_log = np.log1p(all_df.unit_price)
            mean = unit_price_log.mean()
            std = unit_price_log.std()
            i = unit_price_log[(unit_price_log <= mean + 3 * std) & (unit_price_log >= mean - 3.2 * std)].index
            sns.set({'figure.figsize': (8, 4)})
            sns.boxplot(all_df.loc[i].unit_price)
            all_df.loc[i].unit_price.describe()
            all_df = all_df.loc[i]

            # 修复总价
            # 修复总价单位误差
            all_df.loc[all_df.price <= 10000, 'price'] *= 10000
            # 差价分布
            anomaly_price = np.abs(all_df.unit_price * all_df.acreage - all_df.price)
            anomaly_price_index = anomaly_price[anomaly_price > 100000].index  # 差价太多为异常点
            # 直接删除异常样本
            all_df.drop(anomaly_price_index, axis=0, inplace=True)

            # 环线
            all_df.loc[all_df[all_df.Module == '所有'].index, 'Module'] = '内环内'
            # sorted_module = all_df[['unit_price', 'Module']].groupby('Module').median().sort_values('unit_price')
            # i = pd.Series(range(0, sorted_module.shape[0]), index=sorted_module.index)
            # all_df.Module = all_df.Module.map(i.to_dict())

            # 楼层映射
            all_df.loc[all_df.floor < 0, 'floor'] = np.nan
            # 分段映射
            all_df['floor_section'] = self.floor_map(all_df.floor)

            # 朝向因素
            # 暂无为缺省字段
            all_df.shore.replace({'暂无数据': '暂无', '      ': '暂无', '': '暂无'}, inplace=True)
            sorted_shore = all_df[['unit_price', 'shore']].groupby('shore').mean().sort_values('unit_price')
            i = pd.Series(range(0, sorted_shore.shape[0]), index=sorted_shore.index)
            all_df.shore = all_df.shore.map(i.to_dict())

            # 房屋类型
            all_df.loc[all_df[(all_df.house_type == '其它') | (all_df.house_type == '工厂')].index, 'house_type'] = '公寓'
            sorted_house_type = all_df[['house_type', 'unit_price']].groupby('house_type').median().sort_values(
                'unit_price')
            i = pd.Series(range(0, sorted_house_type.shape[0]), index=sorted_house_type.index)
            i.to_dict()
            all_df.house_type = all_df.house_type.map(i.to_dict())

            # 装修情况
            default_fit = '暂无'  # 缺省字段填充
            all_df.fitment.replace({'': default_fit, '暂无数据': default_fit, '豪华装': '豪装', '其他': default_fit}, inplace=True)
            all_df.fitment = all_df.fitment.apply(lambda x: x.strip('修'))
            sorted_fitment = all_df[['fitment', 'unit_price']].groupby('fitment').median().sort_values('unit_price')
            i = pd.Series(range(0, sorted_fitment.shape[0]), index=sorted_fitment.index)
            all_df.fitment = all_df.fitment.map(i.to_dict())

            # 房型
            r = re.compile('室|厅|厨|卫')  # 正则提取房型数据
            l = [map(int, r.split(i)[:-1]) for i in all_df.room_type]
            room_type_df = pd.DataFrame(l, index=all_df.index, columns=['室', '厅', '厨', '卫'])
            all_df = pd.concat((all_df, room_type_df), axis=1)

            # 时间
            all_df.time = all_df.time.apply(lambda x: self.time_map(x)).astype(int)
            all_df.time = all_df.time.apply(lambda x: min(2018 - x, 100) if 0 < x <= 2018 else None)

            # 经纬度
            coors = self.make_coordinates(all_df.Coordinates.values)
            all_df.index = coors.index
            all_df = pd.concat((all_df, coors), axis=1).drop('Coordinates', axis=1)

            # 缓存特征矩阵
            all_df = all_df[all_df.unit_price.notnull()]
            all_df.to_hdf(cache_path, 'data')
        print('共有%d条训练数据' % all_df.shape[0])
        return all_df

    def train_model(self, x_train, y_train):
        # * LightGBM
        # cache_path = os.path.dirname(os.path.realpath(__file__))+'/cache/model_%s-%s_%s.txt' % (beginDate, endDate, x_train.shape[1])
        if os.path.exists(self.cache_path):
            print('使用缓存中的模型，不再训练...')
            gbm = lgb.Booster(model_file=self.cache_path)
        else:
            print('开始模型训练...')
            # 设置模型参数
            # params = {
            #     'objective': 'regression',
            #     'metric': 'mse',
            #     'learning_rate': 0.2,
            #     'feature_fraction': 0.6,
            #     'bagging_fraction': 0.6,
            #     'max_depth': 14,
            #     'num_leaves': 220,
            #     'bagging_freq': 5,
            #     'min_data_in_leaf': 10,
            #     'min_gain_to_split': 0,
            #     'lambda_l1': 1,
            #     'lambda_l2': 1,
            #     'verbose': 0,
            # }
            params = {
                'objective': self.objective,
                'metric': self.metric,
                'learning_rate': self.learning_rate,
                'feature_fraction': self.feature_fraction,
                'bagging_fraction': self.bagging_fraction,
                'max_depth': self.max_depth,
                'num_leaves': self.num_leaves,
                'bagging_freq': self.bagging_freq,
                'min_data_in_leaf': self.min_data_in_leaf,
                'min_gain_to_split': self.min_gain_to_spilt,
                'lambda_l1': self.lambda_l1,
                'lambda_l2': self.lambda_l2,
                'verbose': self.verbose,
            }

            lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=['area', 'Plate', 'Module', 'floor_section'])
            gbm = lgb.train(params, lgb_train, num_boost_round=750)
            gbm.save_model(self.cache_path)
        return gbm

    def make_train_set(self, all_df):
        '''计算单套价格'''
        # 训练集
        x_train = all_df[
            ['acreage', 'all_floor', 'floor', 'time', 'NewDiskID',
             'area', 'Plate', 'Module', 'floor_section', 'loc_x', 'loc_y']]
        # 计算小区房价中位数
        med_price = pd.concat((x_train.NewDiskID, all_df.unit_price), axis=1)
        med_price = med_price.groupby('NewDiskID', as_index=False)['unit_price'].agg({'median': 'mean'})
        med_price.to_csv(self.medprice_path, index=False)
        x_train = pd.merge(x_train, med_price, on='NewDiskID', how='left')


        # 将离散型变量转换成整型，用于lgb训练
        area_le = LabelEncoder()  # 大版块
        arealabel_name = pd.unique(x_train.area)
        x_train.area = area_le.fit_transform(x_train.area)
        arealabel = pd.DataFrame({"area": arealabel_name, "label": area_le.transform(arealabel_name)})
        arealabel.to_csv(self.arealabel_path)  # 把标签对应量记录下来，方便日后用模型预测时使用同一套标签。

        plate_le = LabelEncoder()  # 小板块
        platelabel_name = pd.unique(x_train.Plate)
        x_train.Plate = plate_le.fit_transform(x_train.Plate)
        platelabel = pd.DataFrame({"plate": platelabel_name, "label": plate_le.transform(platelabel_name)})
        platelabel.to_csv(self.platelabel_path)

        # 环线
        sorted_module = all_df[['unit_price', 'Module']].groupby('Module').median().sort_values('unit_price')
        sorted_module.to_csv(self.modulelabel_path)
        i = pd.Series(range(0, sorted_module.shape[0]), index=sorted_module.index).to_dict()
        x_train.Module = x_train.Module.map(i)

        section_map = {'低层': 0, '小高层': 1, '多层': 2, '高层': 3}
        x_train.floor_section = x_train.floor_section.replace(section_map)  # -> r4

        x_train.drop('NewDiskID', inplace=True, axis=1)  # -> r5
        return x_train

    def train(self, model_id):
        self.setConstants(model_id)
        self.setParams(model_id)
        '''加载数据'''
        meta_df, all_df = self.load_data()
        '''处理训练集'''
        all_df = self.preprocess(all_df)
        '''分割'''
        y_train = all_df.unit_price
        x_train = self.make_train_set(all_df)
        '''模型训练'''
        gbm = self.train_model(x_train, y_train)
