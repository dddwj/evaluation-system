from django.db import models

# Create your models here.

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://housing:housing@101.132.154.2/House_Basic',encoding='utf-8',echo=True)
Base = automap_base()

Base.prepare(engine, reflect=True)
ADArea = Base.classes.AD_Area

db = sessionmaker(bind=engine)()

ret = db.query(ADArea).first()
print(ret.areaName)

