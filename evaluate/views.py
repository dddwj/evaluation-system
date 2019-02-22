from django.shortcuts import render
from django.http import *
from django.core import serializers
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

def adminPage(request):
    return render(request, "evaluate/admin_login.html")

def modelPage(request):
    return render(request, "evaluate/admin_model.html")

def spiderPage(request):
    return render(request, "evaluate/admin_spider.html")

def admin(request):
    from django.utils.datastructures import MultiValueDictKeyError
    try:
        adminName = request.GET['adminName']
        adminPwd = request.GET['password']
        if adminName == 'root' and adminPwd == 'root':
            return render(request, "evaluate/admin.html",  locals())
        else:
            return render(request, "evaluate/admin_login.html", {'error_msg': '用户名密码错误'})
    except MultiValueDictKeyError:
        return render(request, "evaluate/admin_login.html")



def trend(request):
    return render(request, "evaluate/trend.html")

def average(request):
    return render(request, "evaluate/average.html")

def averageQuery(request):
    queryAttribute = request.path.split('/')[3]
    import pymysql
    conn = pymysql.connect(host='101.132.154.2', user='housing', passwd='housing', db='House_OnSale',
                           charset='utf8')
    if queryAttribute == 'diskName':
        params = request.GET.keys()
        if params.__contains__('diskName') and params.__contains__('month'):
            diskName = request.GET['diskName']
            (year, m) = request.GET['month'].split('-')
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
            cursor.close()
            return JsonResponse({'status': "ok", 'month': m, 'diskName': diskName, 'data': res})
        else:
            return JsonResponse({'status': "params not fully specified"})

    elif queryAttribute == 'plate':     # 只能查看201901月后的板块/小区均价。之前的数据库中plate内容有丢失。
        params = request.GET.keys()
        if params.__contains__('plate') and params.__contains__('month'):
            plate = request.GET['plate']
            month = request.GET['month']
            res = {}
            cursor = conn.cursor()
            try:
                cursor.callproc("avg_plate_month", (plate, month) )
            except pymysql.err.ProgrammingError:
                return JsonResponse({'status': 'ok', 'data': 'sql error'})
            avg_disks = cursor.fetchall()
            for avg_disk in avg_disks:
                res[avg_disk[0]] = (avg_disk[1],avg_disk[2])
            print(res)
            cursor.close()
            return JsonResponse({'status': 'ok', 'data': res})
        else:
            return JsonResponse({'status': "params not fully specified"})

    elif queryAttribute == 'allDisks':
        params = request.GET.keys()
        if params.__contains__('month'):
            month = str(request.GET['month'])
            if not tableExists(month):
                return JsonResponse({'status': 'ok', 'month': month, 'data': 'month out of range'})
            cursor = conn.cursor()
            try:    # 存储过程'avg_allDisk_month'作用：查询指定月份内全市所有小区的平均每平米价格。若上一次统计为五天前，那么重新统计。
                cursor.callproc('avg_allDisk_month', args=(month,))        # 存储过程运行完成，数据没有commit，当前conn会话仍持有锁，因此insert没有在数据库中永久生效，所以在navicat中不可见新的数据。
                data = cursor.fetchall()
                conn.commit()   # 必须写commit，用来释放锁，否则insert不会在数据库中最终生效。
            except pymysql.err.ProgrammingError:
                return JsonResponse({'status': 'ok', 'data': 'sql error'})
            return JsonResponse({'status': 'ok', 'month': month, 'data': data})
        else:
            return JsonResponse({'status': "params not fully specified"})
    elif queryAttribute == 'allDistricts':
        params = request.GET.keys()
        if params.__contains__('month'):
            month = str(request.GET['month'])
            if not tableExists(month):
                return JsonResponse({'status': 'ok', 'month': month, 'data': 'month out of range'})
            cursor = conn.cursor()
            try:
                sql = "select avg(monthAveragePrice), district, jingwei from `" + month + "_average` group by district;"
                cursor.execute(sql)
                data = cursor.fetchall()
            except pymysql.err.ProgrammingError:
                return JsonResponse({'status': 'ok', 'data': 'sql error'})
            return JsonResponse({'status': 'ok', 'month': month, 'data': data})
        else:
            return JsonResponse({'status': "params not fully specified"})
    elif queryAttribute == 'allPlates':
        params = request.GET.keys()
        if params.__contains__('month'):
            month = str(request.GET['month'])
            if not tableExists(month):
                return JsonResponse({'status': 'ok', 'month': month, 'data': 'month out of range'})
            cursor = conn.cursor()
            try:
                sql = "select avg(monthAveragePrice), plate, jingwei from `" + month + "_average` group by plate;"
                cursor.execute(sql)
                data = cursor.fetchall()
            except pymysql.err.ProgrammingError:
                return JsonResponse({'status': 'ok', 'data': 'sql error'})
            return JsonResponse({'status': 'ok', 'month': month, 'data': data})
        else:
            return JsonResponse({'status': "params not fully specified"})
    else:
        return JsonResponse({'status': "invalid api access"})

def tableExists(tableName):
    import pymysql
    conn_table = pymysql.connect(host='101.132.154.2', user='housing', passwd='housing', db='House_OnSale',
                           charset='utf8')
    cursor_table = conn_table.cursor()
    sql = "show tables like '" + tableName + "';"
    cursor_table.execute(sql)
    match = cursor_table.rowcount
    cursor_table.close()
    conn_table.close()
    if match != 0:
        return True
    return False

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

def modelControl(request, controlAttribute):
    from .models import models_logs
    print(controlAttribute)

    if controlAttribute == 'startTraining':
        # request.GET.get('test', 'test1')
        params = request.GET.dict()
        from datetime import datetime
        model = models_logs()
        model.trainer = 'root1'
        model.startMonth = datetime.strptime(params['startMonth'], '%Y-%m')
        model.endMonth = datetime.strptime(params['endMonth'], '%Y-%m')
        model.trainDate = datetime.strftime(datetime.now(),'%Y-%m-%d')
        model.comment = params['comment']
        model.objective = params['objective']
        model.metric = params['metric']
        model.learning_rate = params['learning_rate']
        model.feature_fraction = params['feature_fraction']
        model.bagging_fraction = params['bagging_fraction']
        model.num_leaves = params['num_leaves']
        model.bagging_freq = params['bagging_freq']
        model.min_data_in_leaf = params['min_data_in_leaf']
        model.min_gain_to_split = params['min_gain_to_split']
        model.lambda_l1 = params['lambda_l1']
        model.lambda_l2 = params['lambda_l2']
        model.verbose = params['verbose']
        model.save()
        return JsonResponse({'status': 'ok', 'model_id': model.id})

    if controlAttribute == 'modelList':
        from .models import models_logs
        modelList = list(models_logs.objects.all().values())
        return JsonResponse({'status': 'ok', 'modelList': modelList})

    if controlAttribute == 'getCurrentModel':
        from django.forms.models import model_to_dict
        model = model_to_dict(models_logs.objects.get(inUseFlag=1))    # get方法仅返回一条记录，且返回的是models_logs对象，而不是QuerySet
        return JsonResponse({'status': 'ok', 'model': model})

    if controlAttribute == 'chooseModel':
        model_id = request.GET["model_id"]
        models_logs.objects.filter(inUseFlag=1).update(inUseFlag=0)
        models_logs.objects.filter(id=model_id).update(inUseFlag=1)
        return JsonResponse({'status': 'ok'})



