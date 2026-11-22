"""Formularios: el registro de alumnos y los campos que hacen cómodo cargar un curso a mano en el panel."""

from __future__ import annotations

from django import forms

from .models import Curso, Modulo, Pregunta


class Registro(forms.Form):
    """Lo que se pide al crear la cuenta, además del correo y la contraseña: el nombre que irá en los certificados."""
    nombre = forms.CharField(label="Nombre y apellido", max_length=150, widget=forms.TextInput(attrs={"autocomplete": "name"}))
    field_order = ["nombre", "email", "password1", "password2"]

    def signup(self, request, user) -> None:
        user.first_name = self.cleaned_data["nombre"].strip()
        user.save(update_fields=["first_name"])


class Lineas(forms.CharField):
    """Una lista que se escribe a razón de un elemento por línea."""
    widget = forms.Textarea(attrs={"rows": 6, "cols": 90})

    def prepare_value(self, valor):
        return "\n".join(str(x) for x in valor) if isinstance(valor, (list, tuple)) else valor

    def to_python(self, valor) -> list[str]:
        return [x.strip() for x in (valor or "").splitlines() if x.strip()]


class Pares(Lineas):
    """Una lista de pares (pregunta y respuesta) que se escribe como «pregunta | respuesta», uno por línea."""

    def prepare_value(self, valor):
        return "\n".join(" | ".join(str(y) for y in x) for x in valor) if isinstance(valor, (list, tuple)) else valor

    def to_python(self, valor) -> list[list[str]]:
        pares = [[y.strip() for y in x.split("|", 1)] for x in super().to_python(valor)]
        if any(len(x) != 2 or not all(x) for x in pares):
            raise forms.ValidationError("Cada línea lleva la pregunta, una barra vertical y la respuesta: «¿Pregunta? | Respuesta».")
        return pares


class CursoForm(forms.ModelForm):
    logros = Lineas(label="Lo que vas a poder hacer", required=False, help_text="Una frase por línea")
    preguntas = Pares(label="Preguntas frecuentes", required=False, help_text="Una por línea: «¿Pregunta? | Respuesta»")

    class Meta:
        model = Curso
        fields = "__all__"


class ModuloForm(forms.ModelForm):
    logros = Lineas(label="Al terminar vas a poder", required=False, help_text="Una frase por línea")

    class Meta:
        model = Modulo
        fields = "__all__"


class PreguntaForm(forms.ModelForm):
    """Las alternativas van una por línea y la respuesta correcta se indica contando desde 1, como las ve una persona."""
    opciones = Lineas(label="Alternativas", required=False, widget=forms.Textarea(attrs={"rows": 4, "cols": 90}),
                      help_text="Una por línea. En «ordenar», escríbelas en el orden correcto.")
    correcta = forms.CharField(label="Respuesta correcta", required=False, help_text=(
        "Una correcta: el número de la alternativa (1, 2, 3…). Varias correctas: los números separados por comas. "
        "Numérica: el número. Ordenar: se deja en blanco."))

    class Meta:
        model = Pregunta
        exclude = ["respuesta"]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        r = self.instance.respuesta if self.instance and self.instance.pk else None
        if isinstance(r, list):
            self.fields["correcta"].initial = ", ".join(str(k + 1) for k in r)
        elif r is not None:
            self.fields["correcta"].initial = str(r + 1) if self.instance.tipo == "unica" else str(r).replace(".", ",")

    def clean(self) -> dict:
        datos = super().clean()
        tipo, opciones, texto = datos.get("tipo"), datos.get("opciones") or [], (datos.get("correcta") or "").strip()
        try:
            if tipo == "numero":
                datos["respuesta"] = float(texto.replace(".", "").replace(",", ".")) if "," in texto else float(texto)
            elif tipo == "orden":
                datos["respuesta"] = None
            else:
                posiciones = sorted({int(x) - 1 for x in texto.replace(";", ",").split(",") if x.strip()})
                if not posiciones or not all(0 <= k < len(opciones) for k in posiciones) or (tipo == "unica" and len(posiciones) != 1):
                    raise ValueError
                datos["respuesta"] = posiciones[0] if tipo == "unica" else posiciones
        except ValueError:
            raise forms.ValidationError("La respuesta correcta no calza con el tipo de pregunta ni con sus alternativas.") from None
        if tipo != "numero" and len(opciones) < 2:
            raise forms.ValidationError("Escribe al menos dos alternativas, una por línea.")
        return datos

    def save(self, commit: bool = True):
        self.instance.respuesta = self.cleaned_data["respuesta"]
        return super().save(commit)
