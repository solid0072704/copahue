from django.urls import path

from . import views

urlpatterns = [
    path("comprar/<slug:slug>/", views.comprar, name="comprar"),
    path("orden/<str:codigo>/", views.orden, name="orden"),
    path("prueba/resolver/", views.resolver_prueba, name="resolver_prueba"),
    path("prueba/<str:codigo>/", views.pago_de_prueba, name="pago_de_prueba"),
    path("<str:clave>/confirmacion/", views.confirmacion, name="pago_confirmacion"),
    path("<str:clave>/retorno/", views.retorno, name="pago_retorno"),
]
