from django.shortcuts import render
from django.http import *
from .YGT import PredictOne
# Create your views here.

def index(request):
    return render(request, "evaluate/index.html")
    pass
def evaluate(request):
    return render(request, "evaluate/searchinput.html")

def searchlandmark(request):
    return render(request, "evaluate/searchlandmark.html")

def getDisk(request):
    area = request.GET['area']
    diskname_input = request.GET['diskNameInput']
    (diskid, name, address) = PredictOne.find_DiskID_ByName(diskname_input)
    if diskid != None and name != None and address != None:
        return render(request, "evaluate/searchinput.html", {'diskname': name ,'address': address})
    else:
        (diskid, name, address) = PredictOne.find_DiskID(diskname_input)
        if diskid != None and name != None and address != None:
            return render(request, "evaluate/searchinput.html", {'diskname': name ,'address': address})
        else:
            return render(request, "evaluate/searchlandmark.html", {'error': "找不到对应的小区！"})

def result(request):
    # 与网页中的name属性匹配
    district=''
    address=request.GET['address']
    house_type='住宅'
    time=int(request.GET['time'])
    all_floor=int(request.GET['groFloor'])
    floor=int(request.GET['curFloor'])
    acreage=float(request.GET['square'])

    houses=[]
    houses.append(PredictOne.init(district, address, house_type, time, all_floor, floor, acreage))
    result = str(PredictOne.predict(houses)["predict_price"][0])

    return render(request, "evaluate/result.html",{'result': result[0:8]})
    pass

def admin(request):
    return render(request, "evaluate/administrator.html")

def trend(request):
    return render(request, "evaluate/trend.html")

def average(request):
    return render(request, "evaluate/average.html")

def averageQuery(request):
    queryAttribute = request.path.split('/')[3]
    if queryAttribute == 'diskName':
        params = request.GET.keys()
        if params.__contains__('diskName') and params.__contains__('month'):
            diskName = request.GET['diskName']
            (year, m) = request.GET['month'].split('-')
            import pymysql
            conn = pymysql.connect(host='101.132.154.2', user='housing', passwd='housing', db='House_OnSale', charset='utf8')
            cursor = conn.cursor()
            res = {}
            for mons in range(1,4):  # 暂时先查询01-03月的数据，等数据库全部数据导入完成再改为range(1,13)
                if mons < 10:
                    month = "" + year + "0" + str(mons)
                else:
                    month = "" + year + str(mons)
                print("Querying:" + month + "," + diskName)
                try:
                    cursor.callproc('avg_diskName_month',(diskName, month))
                except pymysql.err.ProgrammingError:
                    return JsonResponse({'status': 'ok', 'month': m, 'diskName': diskName, 'data': 'month out of range'})
                if cursor.rowcount == 0:
                    return JsonResponse({'status': "ok", 'month': m, 'diskName': diskName, 'data': 'no match'})
                price = cursor.fetchone()[1]
                res[month] = price
            return JsonResponse({'status': "ok", 'month': m, 'diskName': diskName, 'data': res})
        else:
            return JsonResponse({'status': "params not fully specified"})

    elif queryAttribute == 'plate':     # 只能查看201901月后的板块/小区均价。之前的数据库中plate内容有丢失。
        params = request.GET.keys()
        if params.__contains__('plate'):
            plate = request.GET['plate']
            month = request.GET['month']
            res = {}
            import pymysql
            conn = pymysql.connect(host='101.132.154.2', user='housing', passwd='housing', db='House_OnSale',
                                   charset='utf8')
            cursor = conn.cursor()
            try:
                cursor.callproc("avg_plate_month", (plate, month) )
            except pymysql.err.ProgrammingError:
                return JsonResponse({'status': 'ok', 'data': 'sql error'})
            avg_disks = cursor.fetchall()
            for avg_disk in avg_disks:
                res[avg_disk[0]] = avg_disk[1]
            print(res)
            return JsonResponse({'status': 'ok', 'data': res})
        else:
            return JsonResponse({'status': "params not fully specified"})
    else:
        return JsonResponse({'status': "invalid api access"})

def baseQuery(request):
    queryAttribute = request.path.split('/')[3]
    if queryAttribute == 'plate':   # 本来应该查询House_Basic数据库里的基础数据的，但是基础数据和我们爬下来的链家数据不一致。比较之后认为，基础数据不够全面，因此暂时使用201901月中链家的板块划分。
        params = request.GET.keys()
        if params.__contains__('area'):
            area = request.GET['area']
            import pymysql
            conn = pymysql.connect(host='101.132.154.2', user='housing', passwd='housing', db='House_OnSale',
                                   charset='utf8')
            cursor = conn.cursor()
            sql = "select distinct plate from `201901` where area = '%s' " % area
            cursor.execute(sql)
            responses = cursor.fetchall()
            plates = [list(response)[0] for response in responses]
            return JsonResponse({'status': 'ok', 'data': plates})
        else:
            return JsonResponse({'status': "params not fully specified"})
    else:
        return JsonResponse({'status': "invalid api access"})

def chooseDisk(request):
    # lineNumber = request.GET['line']
    searchInput = request.GET['diskNameInput']
    return render(request, "evaluate/chooseDisk.html", {'searchInput': searchInput })


def diskDetail(request):
    return render(request, "evaluate/diskDetail.html")