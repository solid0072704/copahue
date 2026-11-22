"""El panel donde se suben y se administran los cursos: un curso con sus módulos, cada módulo con sus lecciones
(aquí se sube el video), y cada evaluación con sus preguntas. También los alumnos inscritos, su país y sus certificados."""

from django import forms
from django.contrib import admin
from django.db import models

from . import paises
from .formularios import CursoForm, ModuloForm, PreguntaForm
from .models import Certificado, Curso, Evaluacion, Inscripcion, Instructor, Intento, Leccion, Modulo, Perfil, Pregunta, Recurso


class ConComaDecimal:
    """Los números con decimales se escriben como en español («49,90»), en la ficha y en la lista editable."""

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if isinstance(db_field, models.DecimalField):
            kwargs.setdefault("localize", True)
        return super().formfield_for_dbfield(db_field, request, **kwargs)


class ModulosDelCurso(admin.TabularInline):
    model = Modulo
    fields = ("numero", "titulo", "grupo", "etapa")
    extra = 0
    show_change_link = True


class RecursosDelCurso(admin.TabularInline):
    model = Recurso
    fields = ("titulo", "archivo", "leccion")
    extra = 0


@admin.register(Curso)
class CursoAdmin(ConComaDecimal, admin.ModelAdmin):
    form = CursoForm
    list_display = ("titulo", "categoria", "precio_usd", "publicado", "destacado", "inscritos")
    list_editable = ("precio_usd", "publicado", "destacado")
    prepopulated_fields = {"slug": ("titulo",)}
    inlines = [ModulosDelCurso, RecursosDelCurso]
    fieldsets = [
        (None, {"fields": ("titulo", "slug", "bajada", "descripcion", "categoria", "instructor", "imagen")}),
        ("Venta", {"fields": ("precio_usd", "publicado", "destacado", "orden", "certifica")}),
        ("Página del curso", {"fields": ("logros", "preguntas")}),
        ("Lección de muestra", {"fields": ("muestra_titulo", "muestra_video", "muestra_poster", "muestra_subtitulos")}),
    ]

    @admin.display(description="inscritos")
    def inscritos(self, curso) -> int:
        return curso.inscripciones.count()


class LeccionesDelModulo(admin.StackedInline):
    model = Leccion
    fields = (("numero", "titulo", "tipo", "minutos"), "descripcion", ("video", "video_url"), ("poster", "subtitulos", "muestra"))
    extra = 0


@admin.register(Modulo)
class ModuloAdmin(admin.ModelAdmin):
    form = ModuloForm
    list_display = ("__str__", "curso", "grupo", "etapa", "lecciones_publicadas")
    list_filter = ("curso",)
    fields = ("curso", ("numero", "titulo"), "descripcion", "logros", "practica", ("grupo", "etapa"))
    inlines = [LeccionesDelModulo]

    @admin.display(description="lecciones con video")
    def lecciones_publicadas(self, modulo) -> str:
        todas = list(modulo.lecciones.all())
        return f"{sum(x.publicada for x in todas)} de {len(todas)}"


class PreguntasDeLaEvaluacion(admin.StackedInline):
    model = Pregunta
    form = PreguntaForm
    fields = (("tipo", "activa"), "enunciado", "opciones", ("correcta", "tolerancia", "unidad"), "explicacion", ("leccion", "tema"))
    extra = 0
    autocomplete_fields = ("leccion",)


@admin.register(Evaluacion)
class EvaluacionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "curso", "tipo", "preguntas_por_intento", "aprobacion", "en_el_banco")
    list_filter = ("curso", "tipo")
    inlines = [PreguntasDeLaEvaluacion]

    @admin.display(description="preguntas en el banco")
    def en_el_banco(self, evaluacion) -> int:
        return evaluacion.preguntas.filter(activa=True).count()


@admin.register(Leccion)
class LeccionAdmin(admin.ModelAdmin):
    list_display = ("__str__", "modulo", "tipo", "minutos", "publicada", "muestra")
    list_filter = ("modulo__curso", "muestra")
    search_fields = ("titulo", "clave")

    @admin.display(boolean=True, description="con video")
    def publicada(self, leccion) -> bool:
        return leccion.publicada


@admin.register(Inscripcion)
class InscripcionAdmin(admin.ModelAdmin):
    list_display = ("usuario", "curso", "origen", "creada")
    list_filter = ("curso", "origen")
    search_fields = ("usuario__email", "usuario__first_name")
    autocomplete_fields = ("usuario",)


@admin.register(Intento)
class IntentoAdmin(admin.ModelAdmin):
    list_display = ("usuario", "evaluacion", "nota", "aprobado", "entregado")
    list_filter = ("evaluacion__curso", "aprobado")
    readonly_fields = ("usuario", "evaluacion", "preguntas", "respuestas", "buenas", "total", "nota", "aprobado", "entregado")


@admin.register(Certificado)
class CertificadoAdmin(admin.ModelAdmin):
    list_display = ("nombre", "curso", "codigo", "emitido")
    search_fields = ("nombre", "codigo", "usuario__email")


@admin.register(Perfil)
class PerfilAdmin(admin.ModelAdmin):
    """El país de cada alumno: el que indicó al registrarse y decide en qué moneda ve los precios."""
    list_display = ("usuario", "nombre_del_pais")
    list_filter = ("pais",)
    search_fields = ("usuario__email", "usuario__first_name")
    autocomplete_fields = ("usuario",)

    def formfield_for_dbfield(self, db_field, request, **kwargs):
        if db_field.name == "pais":
            return forms.ChoiceField(label="País", choices=[("", "Sin indicar")] + paises.OPCIONES, required=False)
        return super().formfield_for_dbfield(db_field, request, **kwargs)


admin.site.register(Instructor)
