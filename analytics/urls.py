from django.urls import path
from django.conf.urls import url
from analytics import views

urlpatterns = [
    path('', views.index, name='index'),
    url(r'^user/(?P<user_id>\d+)/$', views.user, name='user'),
    url(r'^content/(?P<content_id>\d+)/$', views.content, name='content'),
    url(r'^cluster/(?P<cluster_id>\d+)/$', views.cluster, name='cluster'),
    path('api/get_statistics', views.get_statistics, name='get statistics'),
	path('api/events_on_conversions', views.events_on_conversions, name='events_on_conversions'),
    path('api/ratings_distribution', views.ratings_distribution, name='ratings_distribution'),
	path('api/top_content', views.top_content, name='top_content'),
    path('api/clusters', views.clusters, name='clusters'),
    path('lda', views.lda, name='lda'),
    path('similarity', views.similarity_graph, name='similarity_graph'),
]