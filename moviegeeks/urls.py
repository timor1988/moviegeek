from django.conf.urls import url
from django.urls import path

from moviegeeks import views

urlpatterns = [
    path('', views.index, name='index'),
    url(r'^movie/(?P<movie_id>\d+)/$', views.detail, name='detail'),
    url(r'^genre/(?P<genre_id>[\w-]+)/$', views.genre, name='genre'),
    path('search/', views.search_for_movie, name='search_for_movie'),
]
