from django.core.management.base import BaseCommand, CommandError

from academia.pagos import cambio
from academia.pagos.models import Moneda


class Command(BaseCommand):
    help = "Trae los tipos de cambio del día para las monedas que no están fijas. Sirve como tarea programada diaria."

    def handle(self, *args, **opciones) -> None:
        try:
            r = cambio.actualizar()
        except Exception as e:          # noqa: BLE001 — la fuente no respondió o respondió otra cosa
            raise CommandError(f"No se pudieron traer los tipos de cambio: {e}") from e
        for m in Moneda.objects.all():
            estado = ("fijo" if m.fija else "sin aplicar: revisar" if m.codigo in r["rechazadas"] else
                      "la fuente no lo trae" if m.codigo in r["sin_dato"] else f"{m.actualizada:%d-%m-%Y}")
            self.stdout.write(f"{m.codigo}  {m.por_dolar if m.por_dolar is not None else '—':>16}  {estado}")
        self.stdout.write(self.style.SUCCESS(f"{len(r['actualizadas'])} monedas actualizadas."))
        if r["rechazadas"]:
            self.stdout.write(self.style.WARNING("Sin aplicar por estar muy lejos del valor guardado: " + ", ".join(r["rechazadas"])))
