from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("cuentas/", include("allauth.urls")),
    path("pagos/", include("academia.pagos.urls")),
    path("", include("academia.cursos.urls")),
]
admin.site.site_header = admin.site.site_title = "Academia · administración"
admin.site.index_title = "Cursos, alumnos y ventas"
