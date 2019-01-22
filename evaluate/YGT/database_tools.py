import pymysql
import pandas as pd

def read_basic_table(tableName):
    if(tableName == 'AD_NewDisk'):
        conn = pymysql.connect(host='101.132.154.2', user='housing', passwd='housing', db='House_Basic', port=3306, charset='utf8')
        cur = conn.cursor()
        sql = "select NewDiskID, PropertyID, NewDiskName, Coordinates from AD_NewDisk"
        cur.execute(sql)
        data = list(cur.fetchall())
        df = pd.DataFrame(data, columns=['NewDiskID', 'PropertyID', 'NewDiskName', 'Coordinates'])
        cur.close()
        conn.close()
        print("从数据库获取AD_NewDisk表...完成！")
        return df

    if(tableName == 'AD_NewDiskAddress'):
        conn = pymysql.connect(host='101.132.154.2', user='housing', passwd='housing', db='House_Basic', port=3306,
                               charset='utf8')
        cur = conn.cursor()
        sql = "select RoadLaneNo, NewDiskID from AD_NewDiskAddress"
        cur.execute(sql)
        data = list(cur.fetchall())
        df = pd.DataFrame(data, columns=['RoadLaneNo', 'NewDiskID'])
        cur.close()
        conn.close()
        print("从数据库获取AD_NewDiskAddress表...完成！")
        print(df)
        return df


