"""house URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from django.shortcuts import render
from evaluate import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    url('^$',views.index),
    url('^index',views.index),
    url('^searchlandmark', views.searchlandmark),
    url('^searchinput', views.getDisk),
    url('^evaluate',views.evaluate),
    url('^result',views.result),
    url('^administrator',views.admin),
    url('^trend.html',views.trend),
    url('^average.html',views.average),
    url('^api/average/', views.averageQuery),
    url('^api/base/', views.baseQuery),
    url('^chooseDisk.html',views.chooseDisk),
    url('^diskDetail.html', views.diskDetail)



]
