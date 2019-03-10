from django.contrib import auth
from django.contrib.auth.decorators import user_passes_test
from django.http import *
from django.shortcuts import render
from scrapyd_api import ScrapydAPI
import requests

SCRAPYD_URL='http://localhost:6800'
# PROJECT_NAME='lianjia'
PROJECT_NAME='default'
SPIDER_NAME='lianjia_spider'

scrapyd = ScrapydAPI(SCRAPYD_URL)

# Create your views here.

def loginCheck(func):
    def wrapper(request, *args, **kwargs):
        is_login = request.session.get('IS_LOGIN', False)
        if is_login:
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
