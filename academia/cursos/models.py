"""Los datos de la academia: cursos con sus módulos y lecciones, lo que hace cada alumno y las evaluaciones.

Un curso se puede crear a mano en el panel de administración o importar de un paquete de `academia/contenido`
(`manage.py importar_curso`). Las `clave` son los identificadores estables de ese paquete: permiten volver a
importar sin duplicar ni perder el avance de los alumnos.
"""

from __future__ import annotations

import secrets

from django.conf import settings
from django.db import models
from django.urls import reverse
from django.utils import timezone


class Instructor(models.Model):
    nombre = models.CharField(max_length=120)
    titulo = models.CharField("título profesional", max_length=200, blank=True)
    bio = models.TextField("presentación", blank=True)
    foto = models.ImageField(upload_to="publico/instructores/", blank=True)

    class Meta:
        verbose_name_plural = "instructores"

    def __str__(self) -> str:
        return self.nombre


class Curso(models.Model):
    CERTIFICA = [("evaluaciones", "Al aprobar las pruebas de los módulos y el examen final"), ("lecciones", "Al completar todas las lecciones")]
    slug = models.SlugField("dirección", unique=True, help_text="Lo que va en la dirección web: /cursos/<esto>/")
    titulo = models.CharField("título", max_length=160)
    bajada = models.CharField(max_length=300, help_text="Una línea bajo el título")
    descripcion = models.TextField("descripción", blank=True)
    categoria = models.CharField("categoría", max_length=60, blank=True)
    logros = models.JSONField("lo que vas a poder hacer", default=list, blank=True, help_text="Lista de frases")
    preguntas = models.JSONField("preguntas frecuentes", default=list, blank=True, help_text="Lista de pares [pregunta, respuesta]")
    instructor = models.ForeignKey(Instructor, on_delete=models.PROTECT, related_name="cursos")
    imagen = models.ImageField(upload_to="publico/cursos/", blank=True, help_text="Portada del curso en el catálogo (16:9)")
    precio = models.PositiveIntegerField(default=0, help_text="En la moneda del sitio, sin decimales. Con 0 el curso es gratuito.")
    publicado = models.BooleanField(default=False, help_text="Se ve en el catálogo y se puede comprar")
    destacado = models.BooleanField(default=False)
    orden = models.PositiveIntegerField(default=0)
    certifica = models.CharField("el certificado se obtiene", max_length=20, choices=CERTIFICA, default="evaluaciones")
    # la lección de muestra: se ve en la página del curso sin comprar
    muestra_titulo = models.CharField("título de la lección de muestra", max_length=160, blank=True)
    muestra_video = models.FileField("video de muestra", upload_to="publico/muestras/", blank=True)
    muestra_poster = models.ImageField("imagen del video de muestra", upload_to="publico/muestras/", blank=True)
    muestra_subtitulos = models.FileField("subtítulos de la muestra (.vtt)", upload_to="publico/muestras/", blank=True)
    creado = models.DateTimeField(auto_now_add=True)
    actualizado = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["orden", "titulo"]

    def __str__(self) -> str:
        return self.titulo

    def get_absolute_url(self) -> str:
        return reverse("curso", args=[self.slug])

    @property
    def gratuito(self) -> bool:
        return self.precio == 0

    def lecciones(self):
        return Leccion.objects.filter(modulo__curso=self).select_related("modulo")

    def resumen(self) -> dict:
        """Módulos, lecciones y minutos, del curso completo y de lo ya publicado."""
        lecciones = list(self.lecciones())
        publicadas = [x for x in lecciones if x.publicada]
        return dict(modulos=self.modulos.count(), lecciones=len(lecciones), minutos=sum(x.minutos for x in lecciones),
                    publicadas=len(publicadas), minutos_publicados=sum(x.minutos for x in publicadas))


class Modulo(models.Model):
    curso = models.ForeignKey(Curso, on_delete=models.CASCADE, related_name="modulos")
    clave = models.CharField(max_length=20, blank=True)
    numero = models.PositiveIntegerField("número")
    titulo = models.CharField("título", max_length=200)
    descripcion = models.TextField("descripción", blank=True)
    logros = models.JSONField("al terminar vas a poder", default=list, blank=True)
    practica = models.TextField("práctica", blank=True)
    grupo = models.CharField(max_length=120, blank=True, help_text="Bloque del programa al que pertenece, si el curso los tiene")
    etapa = models.PositiveSmallIntegerField(default=1)

    class Meta:
        ordering = ["numero"]
        constraints = [models.UniqueConstraint(fields=["curso", "numero"], name="un_numero_por_curso")]
        verbose_name = "módulo"

    def __str__(self) -> str:
        return f"{self.numero}. {self.titulo}"

    @property
    def minutos(self) -> int:
        return sum(x.minutos for x in self.lecciones.all())


class Leccion(models.Model):
    TIPOS = [("clase", "Clase"), ("demo", "Demostración"), ("laboratorio", "Laboratorio")]
    modulo = models.ForeignKey(Modulo, on_delete=models.CASCADE, related_name="lecciones")
    clave = models.CharField(max_length=20, blank=True)
    numero = models.PositiveIntegerField("número")
    titulo = models.CharField("título", max_length=200)
    descripcion = models.TextField("descripción", blank=True)
    tipo = models.CharField(max_length=20, choices=TIPOS, default="clase")
    minutos = models.PositiveSmallIntegerField(default=0)
    video = models.FileField(upload_to="lecciones/", blank=True, help_text="El video de la lección. Solo lo ven los inscritos.")
    video_url = models.URLField("video externo", blank=True, help_text="Si el video está en un servicio de video, su dirección")
    poster = models.ImageField("imagen del video", upload_to="publico/lecciones/", blank=True)
    subtitulos = models.FileField("subtítulos (.vtt)", upload_to="subtitulos/", blank=True)
    muestra = models.BooleanField("vista previa gratuita", default=False, help_text="Se puede ver sin comprar el curso")

    class Meta:
        ordering = ["modulo__numero", "numero"]
        constraints = [models.UniqueConstraint(fields=["modulo", "numero"], name="un_numero_por_modulo")]
        verbose_name = "lección"
        verbose_name_plural = "lecciones"

    def __str__(self) -> str:
        return f"{self.modulo.numero}.{self.numero} {self.titulo}"

    @property
    def publicada(self) -> bool:
        return bool(self.video or self.video_url)

    @property
    def curso(self) -> Curso:
        return self.modulo.curso


class Recurso(models.Model):
    """Un archivo para descargar: del curso completo (el caso, las plantillas) o de una lección."""
    curso = models.ForeignKey(Curso, on_delete=models.CASCADE, related_name="recursos")
    leccion = models.ForeignKey(Leccion, on_delete=models.CASCADE, related_name="recursos", null=True, blank=True)
    titulo = models.CharField("título", max_length=160)
    archivo = models.FileField(upload_to="recursos/")

    def __str__(self) -> str:
        return self.titulo


class Inscripcion(models.Model):
    ORIGENES = [("compra", "Compra"), ("gratis", "Curso gratuito"), ("cortesia", "Cortesía")]
    usuario = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="inscripciones")
    curso = models.ForeignKey(Curso, on_delete=models.PROTECT, related_name="inscripciones")
    origen = models.CharField(max_length=20, choices=ORIGENES, default="compra")
    creada = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["usuario", "curso"], name="una_inscripcion_por_curso")]
        verbose_name = "inscripción"
        verbose_name_plural = "inscripciones"

    def __str__(self) -> str:
        return f"{self.usuario} → {self.curso}"


class Avance(models.Model):
    usuario = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="avances")
    leccion = models.ForeignKey(Leccion, on_delete=models.CASCADE, related_name="avances")
    segundos = models.PositiveIntegerField(default=0)
    completada = models.BooleanField(default=False)
    actualizado = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["usuario", "leccion"], name="un_avance_por_leccion")]


class Evaluacion(models.Model):
    TIPOS = [("diagnostico", "Diagnóstico inicial"), ("modulo", "Prueba de módulo"), ("final", "Examen final")]
    curso = models.ForeignKey(Curso, on_delete=models.CASCADE, related_name="evaluaciones")
    modulo = models.OneToOneField(Modulo, on_delete=models.CASCADE, related_name="evaluacion", null=True, blank=True)
    tipo = models.CharField(max_length=20, choices=TIPOS, default="modulo")
    preguntas_por_intento = models.PositiveSmallIntegerField(default=10)
    aprobacion = models.PositiveSmallIntegerField("nota de aprobación (%)", default=70, help_text="El diagnóstico no se aprueba ni se reprueba: usa 0")
    recomendaciones = models.JSONField(default=list, blank=True, help_text="Solo el diagnóstico: pares [respuestas correctas desde, recomendación]")

    class Meta:
        verbose_name = "evaluación"
        verbose_name_plural = "evaluaciones"

    def __str__(self) -> str:
        return self.nombre

    @property
    def nombre(self) -> str:
        return f"Prueba del módulo {self.modulo.numero}" if self.modulo_id else self.get_tipo_display()

    @property
    def lleva_nota(self) -> bool:
        return self.tipo != "diagnostico"


class Pregunta(models.Model):
    TIPOS = [("unica", "Una correcta"), ("multiple", "Varias correctas"), ("numero", "Respuesta numérica"), ("orden", "Ordenar")]
    evaluacion = models.ForeignKey(Evaluacion, on_delete=models.CASCADE, related_name="preguntas")
    tipo = models.CharField(max_length=20, choices=TIPOS, default="unica")
    enunciado = models.TextField()
    opciones = models.JSONField(default=list, blank=True, help_text="Lista de alternativas. En «ordenar», en el orden correcto.")
    respuesta = models.JSONField(null=True, blank=True, help_text="Una correcta: su posición (desde 0). Varias: lista de posiciones. Numérica: el número.")
    tolerancia = models.FloatField(default=0)
    unidad = models.CharField(max_length=20, blank=True)
    explicacion = models.TextField("explicación", help_text="Se muestra al corregir")
    leccion = models.ForeignKey(Leccion, on_delete=models.SET_NULL, null=True, blank=True, related_name="+", help_text="Dónde se enseña")
    tema = models.ForeignKey(Modulo, on_delete=models.SET_NULL, null=True, blank=True, related_name="+", help_text="Solo el diagnóstico: el módulo al que apunta")
    activa = models.BooleanField(default=True)

    def __str__(self) -> str:
        return self.enunciado[:80]


class Intento(models.Model):
    usuario = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="intentos")
    evaluacion = models.ForeignKey(Evaluacion, on_delete=models.CASCADE, related_name="intentos")
    creado = models.DateTimeField(auto_now_add=True)
    entregado = models.DateTimeField(null=True, blank=True)
    preguntas = models.JSONField(default=list)       # [{"id": pregunta, "orden": [posiciones de las alternativas como se mostraron]}]
    respuestas = models.JSONField(default=dict)      # {"<id>": lo que marcó}
    buenas = models.PositiveSmallIntegerField(default=0)
    total = models.PositiveSmallIntegerField(default=0)
    nota = models.PositiveSmallIntegerField(default=0)
    aprobado = models.BooleanField(default=False)

    class Meta:
        ordering = ["-creado"]

    def __str__(self) -> str:
        return f"{self.usuario} · {self.evaluacion} · {self.nota} %"


def _codigo() -> str:
    return secrets.token_hex(5).upper()


class Certificado(models.Model):
    usuario = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="certificados")
    curso = models.ForeignKey(Curso, on_delete=models.PROTECT, related_name="certificados")
    codigo = models.CharField("código de verificación", max_length=12, unique=True, default=_codigo)
    nombre = models.CharField(max_length=160)
    horas = models.DecimalField(max_digits=5, decimal_places=1, default=0)
    emitido = models.DateTimeField(default=timezone.now)

    class Meta:
        constraints = [models.UniqueConstraint(fields=["usuario", "curso"], name="un_certificado_por_curso")]

    def __str__(self) -> str:
        return f"{self.nombre} · {self.curso} · {self.codigo}"
