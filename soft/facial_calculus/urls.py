from django.conf.urls import url

from . import views

from django.conf.urls import patterns, url

#urlpatterns = patterns('myapp.views',
  #  url(r'^list/$', 'list', name='list'),
#)

urlpatterns = [
    url(r'^$', views.list, name='list'),
	url(r'^redirect_to_analyze/(?P<pic_id>[0-9]+)$', views.redirect_to_analyze, name='redirect_to_analyze'),
	url(r'^delete/(?P<pic_id>[0-9]+)$', views.delete, name='delete'),
]