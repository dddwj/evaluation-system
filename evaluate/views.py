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
    diskname_input = request.GET['diskname_input']
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

    return render(request, "evaluate/result.html",{'result':result})
    pass