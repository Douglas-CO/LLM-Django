from django.contrib import admin
from django.urls import path
from api import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/verificar_frase/", views.verificar_frase_view, name="verificar_frase"),
]
