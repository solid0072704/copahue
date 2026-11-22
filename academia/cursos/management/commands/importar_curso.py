from django.core.management.base import BaseCommand

from academia.cursos.importar import importar


class Command(BaseCommand):
    help = "Carga o actualiza un curso desde un paquete de academia/contenido (por ejemplo, academia.contenido.claude_finanzas)."

    def add_arguments(self, parser) -> None:
        parser.add_argument("paquete")

    def handle(self, *args, paquete: str, **opciones) -> None:
        curso = importar(paquete)
        r = curso.resumen()
        self.stdout.write(self.style.SUCCESS(
            f'«{curso.titulo}»: {r["modulos"]} módulos, {r["lecciones"]} lecciones, {curso.evaluaciones.count()} evaluaciones. '
            + ("Publicado." if curso.publicado else "Sin publicar: define el precio y publícalo en el panel de administración.")))
