# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns
import time
import os
import gc
#import re
import lightgbm as lgb
from datetime import datetime
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
#from geohash import encode
from scipy.stats import skew
from sklearn.model_selection import train_test_split
# from multiprocessing import Pool


import wx


class Example(wx.Frame):
    def __init__(self, parent, title, size):
        super(Example, self).__init__(parent, title=title, size=size)
        self.panel = wx.Panel(self)
        self.tranningData_1 = wx.TextCtrl(self.panel)  #训练数据
        self.tranningData_2 = wx.TextCtrl(self.panel)
        self.path = wx.TextCtrl(self.panel)      #文件保存路径
        #self.InitUI()
        self.sizer = wx.GridBagSizer(0, 0)
        self.sizer.Add(wx.StaticText(self.panel, label="训练数据范围"), pos=(1, 0), flag=wx.ALL, border=5)
        self.sizer.Add(self.tranningData_1, pos=(2, 0), flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer.Add(self.tranningData_2, pos=(2, 1), flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer.Add(wx.StaticText(self.panel, label="评估结果保存路径"), pos=(3, 0), flag=wx.ALL, border=5)
        self.sizer.Add(self.path, pos=(4, 0), flag=wx.ALL, border=5)
        self.button = wx.Button(self.panel, label="开始评估")
        self.sizer.Add(self.button, pos=(4, 1), flag=wx.ALL, border=5)
        self.panel.SetSizerAndFit(self.sizer)
        self.Bind(wx.EVT_BUTTON,self.onClick,self.button)
        self.button.SetDefault()
        self.Centre()
        self.Show()
        
           

    #def InitUI(self):
        

    def onClick(self,event):
        a=str(self.tranningData_1.GetValue())
        b=str(self.tranningData_2.GetValue())
        c=str(self.path.GetValue())        
        self.model(a,b,c)
        
       
    @staticmethod
    def model(a,b,c):
        import re

        # 小区名正则过滤
        name_filter=lambda x:re.sub(r'\(.*?\)', '', re.sub(r'\（.*?\）', '', x))
        # 楼层映射
        floor_map=lambda x:list(pd.cut(x,[0,3,6,9,np.inf],labels=['低层','多层','小高层','高层']))

        # 挂牌数据路径
        data_path='./data/'
        # Ad_property表路径
        property_path='./data/GC_dbo_AD_Property.csv'
        # AD_NewDisk表路径
        newdisk_path='./data/GC_dbo_AD_NewDisk.csv'
        # 结果存放路径
        result_path=c
        # 挂牌数据范围
        beginDate=a
        endDate=b
        files=np.unique([datetime.strftime(x,'%Y-%m') for x in list(pd.date_range(start=beginDate, end=endDate))])
        encoding='utf-8'

        # ## 加载数据

        # In[12]:

        # 读取挂牌数据
        def load_data(name,month):
            with open(os.path.join(data_path,name,'挂牌.txt'),encoding=encoding) as f:
                l=[]
                for i in f.readlines():
                    l.append(i.split('\t'))
                df=pd.DataFrame(l)
                drop_col=[0,15,16,18,19,20,21]
                if len(df.columns)==23:
                    drop_col.append(22)
                df.drop(drop_col,axis=1,inplace=True) # 去除无用列
                df.columns=['area','address','name','price','unit_price','acreage','room_type','all_floor','floor','shore','house_type','fitment','time','Source','house_trait']
                df['month']=month
                return df

        cache_name='%s-%s.hdf'%(beginDate,endDate)
        if os.path.exists(cache_name):
            all_df=pd.read_hdf(cache_name,'data')
        else:
            # 加载挂牌数据
            # pool=Pool()
            # files=[i for i in os.listdir(data_path) if '.' not in i]
            # files=sorted(files)[-10:]
            dfs=[]
            for month,name in enumerate(files):
                print(month,name)
                dfs.append(load_data(name,str(month)))
            
            # dfs=[pool.apply_async(load_data,(name,month)) for month,name in enumerate(files)]
            # pool.close()
            # pool.join()
            # dfs=[i.get() for i in dfs]
            all_df=pd.concat(dfs,ignore_index=True)

            # 获取经纬度信息
            newdisk_df=pd.read_csv(newdisk_path,usecols=['NewDiskID','PropertyID','NewDiskName','Coordinates'],encoding='gbk')
            newdisk_df.rename(columns={'NewDiskName':'name'},inplace=True)
            newdisk_df.dropna(inplace=True)
            newdisk_df.drop_duplicates('PropertyID',inplace=True)

            # 获取板块、环线信息
            property_df=pd.read_csv(property_path,usecols=['PropertyID','Area','Plate','Module','HousingName'],encoding='gbk')
            property_df.rename(columns={'Area':'area','HousingName':'name'},inplace=True)

            # merge them
            property_df=pd.merge(property_df,newdisk_df[['PropertyID','Coordinates']],on='PropertyID',how='inner')
            property_df.head()

            # 小区名称清洗
            property_df.drop_duplicates(['area','name'],inplace=True)
            property_df.name=property_df.name.apply(name_filter)
            all_df.name=all_df.name.apply(name_filter)

            # 合并信息
            all_df=pd.merge(all_df,property_df,on=['name','area'],how='inner')

            # 转换数值类型
            numerical_columns=['price','unit_price','acreage','all_floor','floor'] # str->float
            all_df[numerical_columns]=all_df[numerical_columns].astype(float)

            # ## 特征预处理

            # * 修正面积
            acreage_log=np.log(all_df.acreage)
            mean=acreage_log.mean()
            std=acreage_log.std()
            i=acreage_log[(acreage_log<=mean+2*std)&(acreage_log>=mean-1*std)].index
            # sns.set({'figure.figsize':(8,4)})
            # sns.boxplot(all_df.loc[i].acreage)
            all_df=all_df.loc[i]

            # * 修正单价
            unit_price_log=np.log(all_df.unit_price)
            mean=unit_price_log.mean()
            std=unit_price_log.std()
            i=unit_price_log[(unit_price_log<=mean+3*std)&(unit_price_log>=mean-3.2*std)].index
            #sns.set({'figure.figsize':(8,4)})
            #sns.boxplot(all_df.loc[i].unit_price)
            all_df=all_df.loc[i]

            # * 修复总价

            # 修复总价单位误差
            all_df.loc[all_df.price<=10000,'price']*=10000
            # 差价分布
            anomaly_price=np.abs(all_df.unit_price*all_df.acreage-all_df.price)
            # sns.set({'figure.figsize':(8,4)})
            # sns.boxplot(anomaly_price)
            anomaly_price_index=anomaly_price[anomaly_price>100000].index # 差价太多为异常点
            anomaly_price[anomaly_price_index].sort_values()
            # 修正异常的总价
            # train.loc[anomaly_price_index,'price']=train.loc[anomaly_price_index,'unit_price']*train.loc[anomaly_price_index,'acreage']

            # 直接删除异常样本
            all_df.drop(anomaly_price_index,axis=0,inplace=True)

            # * 地区因素

            # 环线
            all_df.loc[all_df[all_df.Module=='所有'].index,'Module']='内环以内'
            sorted_module=all_df[['unit_price','Module']].groupby('Module').mean().sort_values('unit_price')
            print(sorted_module)
            #sns.barplot(x=sorted_module.index,y='unit_price',data=sorted_module)

            # id映射
            i=pd.Series(range(0,sorted_module.shape[0]),index=sorted_module.index)
            i.to_dict()

            # all_df.Module=all_df.Module.map(i.to_dict())

            # * 楼层因素
            all_df.loc[all_df.floor<0,'floor']=np.nan

            # 分段映射
            all_df['floor_section']=floor_map(all_df.floor)

            # 统计
            all_df.groupby('floor_section',as_index=False)['floor_section'].agg({'count':'count'})

            # * 朝向因素
            # 暂无为缺省字段
            all_df.shore.replace({'暂无数据':'暂无','      ':'暂无','':'暂无'},inplace=True)
            sorted_shore=all_df[['unit_price','shore']].groupby('shore').mean().sort_values('unit_price')
            # print(sorted_shore)
            # sns.barplot(x=sorted_shore.index,y='unit_price',data=sorted_shore)
            i=pd.Series(range(0,sorted_shore.shape[0]),index=sorted_shore.index)
            i.to_dict()
            # all_df.shore=all_df.shore.map(i.to_dict())

            # * 房屋类型
            all_df.loc[all_df[(all_df.house_type=='其它') | (all_df.house_type=='工厂')].index,'house_type']='公寓'
            all_df.house_type.unique()
            sorted_house_type=all_df[['house_type','unit_price']].groupby('house_type').mean().sort_values('unit_price')
            # sorted_house_type
            i=pd.Series(range(0,sorted_house_type.shape[0]),index=sorted_house_type.index)
            # i.to_dict()
            # all_df.house_type=all_df.house_type.map(i.to_dict())

            # * 装修情况
            # 缺省字段填充
            default_fit='暂无'
            all_df.fitment.replace({'':default_fit,'暂无数据':default_fit,'豪华装':'豪装'},inplace=True)
            all_df.fitment=all_df.fitment.apply(lambda x:x.strip('修'))
            # all_df.fitment.unique()
            sorted_fitment=all_df[['fitment','unit_price']].groupby('fitment').median().sort_values('unit_price')
            # print(sorted_fitment)
            # sns.barplot(x=sorted_fitment.index,y='unit_price',data=sorted_fitment)
            i=pd.Series(range(0,sorted_fitment.shape[0]),index=sorted_fitment.index)
            # i.to_dict()
            # all_df.fitment=all_df.fitment.map(i.to_dict())

            # * 房型
            import re

            # 正则提取房型数据
        #    r=re.compile('室|厅|厨|卫')
        #    l=[map(int,r.split(i)[:-1]) for i in all_df.room_type]
        #    room_type_df=pd.DataFrame(l,index=all_df.index,columns=['室','厅','厨','卫'])
        #    all_df=pd.concat((all_df,room_type_df),axis=1)

            # * 时间
            def convert_time(x):
                try:
                    if '/' in x:
                        return int(x.split('/')[0])
                    else:
                        return int(x.split('-')[0])
                except:
                    return 2013

            all_df.time=all_df.time.apply(convert_time).astype(int)
            all_df.time=all_df.time.apply(lambda x:2018-x if x<=2018 else 0)

            # * 经纬度
            coors=pd.DataFrame([coor.split(',') for coor in all_df.Coordinates],columns=['loc_x','loc_y'],index=all_df.index)
            coors=coors.astype(float)
            all_df=pd.concat((all_df,coors),axis=1).drop('Coordinates',axis=1)
            # 缓存特征矩阵
            all_df.to_hdf(cache_name,'data')

        print('共有%d条训练数据'%all_df.shape[0])


        # * 处理测试集

        # 读取楼盘信息数据表
        property_df=pd.read_csv(property_path,encoding='gbk',usecols=['PropertyID','Area','Plate','Module'])
        newdisk_df=pd.read_csv(newdisk_path,encoding='gbk',usecols=['NewDiskID','PropertyID','Coordinates','NewDiskName','NewDiskType'])
        x_test=pd.merge(newdisk_df,property_df,on=['PropertyID'],how='left')
        coors=pd.DataFrame([coor.split(',') if type(coor)==str else [None]*2 for coor in x_test.Coordinates],
                           columns=['loc_x','loc_y'],
                           index=x_test.index)
        coors=coors.astype(float)
        x_test.drop(['PropertyID','Coordinates','NewDiskType'],axis=1,inplace=True)
        x_test=pd.concat((x_test,coors),axis=1)
        x_test.rename(columns={'Area':'area','NewDiskName':'name'},inplace=True)
        x_train=all_df[['area','Plate','Module','loc_x','loc_y','unit_price','name','time']]
        times=all_df[['name','time']].groupby(['name'],as_index=False)['time'].agg({'time':'mean'})
        x_test=pd.merge(x_test,times,on='name',how='left')
        print(x_test.shape)

        # In[56]:
        x_train.loc[x_train[x_train.Module=='所有'].index,'Module']='内环以内'
        x_test.loc[x_test[x_test.Module=='所有'].index,'Module']='内环以内'
        sorted_module=x_train[['unit_price','Module']].groupby('Module').mean().sort_values('unit_price')
        y_train=x_train.unit_price
        mid_price=x_train[['name','unit_price']].groupby('name',as_index=False)['unit_price'].agg({'mid':'median'})
        x_train=pd.merge(x_train,mid_price,on='name',how='left')
        x_test=pd.merge(x_test,mid_price,on='name',how='left')
        x_train.drop(['name','unit_price'],axis=1,inplace=True)
        i=pd.Series(range(0,sorted_module.shape[0]),index=sorted_module.index)
        x_train.Module=x_train.Module.map(i.to_dict())
        x_test.Module=x_test.Module.map(i.to_dict())

        # 将离散型变量转换成整型，用于lgb训练
        from sklearn.preprocessing import LabelEncoder
        encoder=LabelEncoder() # 小板块
        x_train.area=encoder.fit_transform(x_train.area)
        x_test.area=encoder.transform(x_test.area)
        x_train.Plate=encoder.fit_transform(x_train.Plate)
        x_test.Plate=encoder.fit_transform(x_test.Plate)

        print(x_train.columns)
        print(x_test.columns)
        # ## 模型训练

        # * LightGBM

        # In[57]:

        params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate':0.2,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
            'max_depth': 14,
            'num_leaves':180,
            'bagging_freq':5,
            'min_data_in_leaf':10,
            'min_gain_to_split':0,
            'lambda_l1':1,
            'lambda_l2':1,
            'verbose':0,
        }

        val_loss=[]

        def eval_error(pred, df):
            """自定义评价函数"""
            score = mean_squared_error(df.get_label(),pred)
            val_loss.append(score)
            return ('mse',score,False)

        # lgb_train = lgb.Dataset(x1,y1,categorical_feature=['area','Plate','Module'])
        # lgb_val=lgb.Dataset(x2,y2,reference=lgb_train)
        # # gbm = lgb.cv(params,lgb_train,num_boost_round=3000,early_stopping_rounds=20,feval=eval_error,nfold=3)
        # gbm = lgb.train(params,lgb_train,num_boost_round=500,valid_sets=lgb_val,early_stopping_rounds=15,feval=eval_error)

        cache_name='model_%s-%s.txt'%(beginDate,endDate)
        if os.path.exists(cache_name):
            gbm=lgb.Booster(model_file=cache_name)
        else:
            lgb_train = lgb.Dataset(x_train,y_train,categorical_feature=['area','Plate','Module'])
            gbm = lgb.train(params,lgb_train,num_boost_round=500)
            gbm.save_model(cache_name)

        # In[84]:
        mid_price=all_df[['name','unit_price']].groupby('name',as_index=False)['unit_price'].median()
        df=pd.merge(x_test[['name']],mid_price,on='name',how='left')
        df['mark']=np.isnan(df['unit_price'])*1

        # 输出结果
        ans=pd.DataFrame({'NewDiskID':x_test.NewDiskID,'楼盘':x_test.name,'单价':gbm.predict(x_test[x_train.columns]),'mark':df.mark})
        ans=ans[['NewDiskID','楼盘','单价','mark']]
        ans.to_csv(result_path,index=False,encoding='gbk')




app = wx.App()
Example(None, title='GridBag Demo - www.yiibai.com', size=(500, 500))
app.MainLoop()


    
