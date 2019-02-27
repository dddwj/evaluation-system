from django.shortcuts import render
from django.http import *
# from .YGT import PredictOne
# Create your views here.

def index(request):
    return render(request, "evaluate/index.html")
    pass

def evaluate(request):
    return render(request, "evaluate/searchinput.html")

def searchlandmark(request):
    return render(request, "evaluate/searchlandmark.html")

def getDisk(request):

    selected_disk = request.GET['diskname']

    import sqlalchemy
    from sqlalchemy.ext.automap import automap_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine

    engine = create_engine('mysql+pymysql://housing:housing@101.132.154.2/House_Basic', encoding='utf-8', echo=True)
    Base = automap_base()

    Base.prepare(engine, reflect=True)
    db = sessionmaker(bind=engine)()
    ADNewDisk = Base.classes.AD_NewDisk
    ADNewDiskAddress = Base.classes.AD_NewDiskAddress

    disk = db.query(ADNewDisk).filter(ADNewDisk.NewDiskName == selected_disk).first()

    address = db.query(ADNewDiskAddress).filter(ADNewDiskAddress.NewDiskID == disk.NewDiskID).first()

    return render(request, "evaluate/searchinput.html", {'selected_disk': selected_disk, 'address': address})

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

def chooseDisk(request):
    # lineNumber = request.GET['line']
    searchInput = request.GET['diskNameInput']

    import sqlalchemy
    from sqlalchemy.ext.automap import  automap_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import create_engine

    engine = create_engine('mysql+pymysql://housing:housing@101.132.154.2/House_Basic', encoding='utf-8', echo=True)
    Base = automap_base()

    Base.prepare(engine, reflect=True)
    db = sessionmaker(bind=engine)()
    ADNewDisk = Base.classes.AD_NewDisk

    disk = db.query(ADNewDisk).filter(ADNewDisk.NewDiskName == searchInput).all()

    if len(disk) == 1:
        # estate1 = disk[0].NewDiskName

        return render(request, "evaluate/chooseDisk.html",
                      {'searchInput': searchInput, 'disks': disk})
    else:
        disk2 = db.query(ADNewDisk).filter(ADNewDisk.NewDiskName.like("%" + searchInput + "%")).all()

        return render(request, "evaluate/chooseDisk.html",
                      {'searchInput': searchInput, 'disks': disk2})


def diskDetail(request):
    return render(request, "evaluate/diskDetail.html")