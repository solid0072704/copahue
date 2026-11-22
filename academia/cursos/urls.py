from django.urls import path

from . import views

urlpatterns = [
    path("", views.catalogo, name="catalogo"),
    path("cursos/<slug:slug>/", views.curso, name="curso"),
    path("mi-aprendizaje/", views.mi_aprendizaje, name="mi_aprendizaje"),
    path("aprender/<slug:slug>/", views.aprender, name="aprender"),
    path("aprender/<slug:slug>/leccion/<int:pk>/", views.leccion, name="leccion"),
    path("aprender/<slug:slug>/prueba/<int:pk>/", views.prueba, name="prueba"),
    path("aprender/<slug:slug>/certificado/", views.certificado, name="certificado"),
    path("intentos/<int:pk>/", views.intento, name="intento"),
    path("lecciones/<int:pk>/video/", views.video, name="video"),
    path("lecciones/<int:pk>/subtitulos/", views.subtitulos, name="subtitulos"),
    path("lecciones/<int:pk>/avance/", views.guardar_avance, name="guardar_avance"),
    path("lecciones/<int:pk>/marcar/", views.marcar, name="marcar"),
    path("recursos/<int:pk>/", views.recurso, name="recurso"),
    path("certificados/<str:codigo>/", views.verificar, name="verificar"),
    path("medios/publico/<path:ruta>", views.medio_publico, name="medio_publico"),
]
