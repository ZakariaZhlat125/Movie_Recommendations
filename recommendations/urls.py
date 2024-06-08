# recommendations/urls.py

from django.urls import path
from . import views
# from .views import recommend_movie

urlpatterns = [
    path('', views.recommend_movie, name='recommend_movie'),
    path('apriori/', views.apriori_analysis, name='apriori_analysis'),
    path('knn/', views.knn_recommend_view, name='knn'),
    path('contentBased/', views.cb_recommend_view, name='content-based'),
    path('about/', views.about, name='about'),
]
