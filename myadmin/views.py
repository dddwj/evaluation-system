from django.contrib import auth
from django.contrib.auth.decorators import user_passes_test
from django.http import *
from django.shortcuts import render
from scrapyd_api import ScrapydAPI
import requests
from evaluate.YGT import train, predict

SCRAPYD_URL='http://localhost:6800'
PROJECT_NAME='lianjia'
#PROJECT_NAME='default'
SPIDER_NAME='lianjia_spider'

scrapyd = ScrapydAPI(SCRAPYD_URL)

# Create your views here.

def loginCheck(func):
    def wrapper(request, *args, **kwargs):
        is_login = request.session.get('IS_LOGIN', False)
        if True:
            ret = func(request, *args, **kwargs)
            return ret
        else:
            return JsonResponse({"status": "not login"})

    return wrapper


def login(request):
    account = request.GET.get('account')
    password = request.GET.get('password')
    print(account)
    userAuthent = auth.authenticate(username=account, password=password)
    if account == 'root' and password == '123123':
        request.session['IS_LOGIN'] = True
        request.session['account'] = account
        return JsonResponse({"status": "ok", "data": "success"})
    return JsonResponse({"status": "ok", "data": "error"})


@loginCheck
def scrapy_job_schedule(request):
    res = scrapyd.schedule(PROJECT_NAME, SPIDER_NAME)
    return JsonResponse({"status": "ok", "data": res})


@loginCheck
def scrapy_job_list(request):
    res = scrapyd.list_jobs(PROJECT_NAME)
    return JsonResponse({"status": "ok", "data": res})


@loginCheck
def scrapy_job_cancel(request):
    job_id = request.GET.get('id')
    res = scrapyd.cancel(PROJECT_NAME, job_id)
    return JsonResponse({"status": "ok", "data": res})


@loginCheck
def scrapy_job_status(request):
    job_id = request.GET.get('id')
    # http://127.0.0.1:6800/logs/lianjia/lianjia_spider/2a6ea51c3cd011e991f89061aedfe3a9.json
    url = "%s/logs/%s/%s/%s.json" % (SCRAPYD_URL,PROJECT_NAME,SPIDER_NAME,job_id)
    scrapy_log = eval(requests.get(url).text)
    res = {
        "datas": scrapy_log.get('datas'),#可用于echart
        "pages": scrapy_log.get('pages'),
        "items": scrapy_log.get('items')
    }
    return JsonResponse({"status": "ok", "data": res})

@loginCheck
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
        t = train.trainModel()
        t.train(model.id)
        model.trainSuccess = 1
        model.save()
        print(model.id)
        if not model.id:
            return JsonResponse({'status': 'failed'})
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

    if controlAttribute == 'predict':   # 预测一套房子
        print(request.get_raw_uri())
        district = request.GET.get('district')
        address = request.GET.get('address')
        house_type = '住宅'
        time = int(request.GET.get('time'))
        all_floor = int(request.GET.get('all_floor'))
        floor = int(request.GET.get('floor'))
        acreage = int(request.GET.get('acreage'))
        from evaluate.YGT import predict
        p = predict.predict()
        p.addCase(district, address, house_type, time, all_floor, floor, acreage)
        res = p.predict()[0]
        print(res)
        return JsonResponse({'status': 'ok', 'res': res})
