from django.urls import path

from collector import views

urlpatterns = [
    path('log/', views.log, name='log'),
]


