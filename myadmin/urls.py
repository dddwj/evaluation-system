from django.conf.urls import url

from myadmin import views


urlpatterns = [
    url(r'^$', views.index),
    url(r'^api/login', views.login),
    url(r'^api/scrapy/job/list', views.scrapy_job_list),
    url(r'^api/scrapy/job/schedule', views.scrapy_job_schedule),
    url(r'^api/scrapy/job/cancel', views.scrapy_job_cancel),
    url(r'^api/scrapy/job/status', views.scrapy_job_status),
    url(r'^api/model/(.+)/$', views.modelControl),  # eg: /myadmin/api/model/modelList

    # url...
]