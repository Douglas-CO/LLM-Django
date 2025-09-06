from django.urls import path
from . import views

urlpatterns = [
    path("", views.verificar_frase, name="verificar_frase"),
]
