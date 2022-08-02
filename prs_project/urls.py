"""prs_project URL Configuration"""
# from django.conf.urls import url, include
from django.urls import path,include
from django.contrib import admin

from moviegeeks import views

urlpatterns = [
    path('', views.index, name='index'),
    path('movies/', include('moviegeeks.urls')),
    path('collect/', include('collector.urls')),
    path('analytics/', include('analytics.urls')),
    path('admin/', admin.site.urls),
    path('rec/', include('recommender.urls'))
]
