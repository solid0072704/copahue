from django.db import migrations, models

# Las monedas con que parte la academia: (código, nombre, símbolo, decimales con que se cobra). El tipo de cambio de
# cada una llega con la primera actualización (`manage.py actualizar_monedas`, o sola al mostrar precios).
MONEDAS = [
    ("USD", "dólares", "US$", 2), ("CLP", "pesos chilenos", "$", 0), ("ARS", "pesos argentinos", "$", 2), ("BOB", "bolivianos", "Bs", 2),
    ("BRL", "reales", "R$", 2), ("CAD", "dólares canadienses", "CA$", 2), ("COP", "pesos colombianos", "$", 0), ("CRC", "colones", "₡", 2),
    ("DOP", "pesos dominicanos", "RD$", 2), ("EUR", "euros", "€", 2), ("GTQ", "quetzales", "Q", 2), ("HNL", "lempiras", "L", 2),
    ("MXN", "pesos mexicanos", "$", 2), ("NIO", "córdobas", "C$", 2), ("PEN", "soles", "S/", 2), ("PYG", "guaraníes", "₲", 0),
    ("UYU", "pesos uruguayos", "$", 2),
]


def sembrar(apps, schema_editor) -> None:
    Moneda = apps.get_model("pagos", "Moneda")
    for codigo, nombre, simbolo, decimales in MONEDAS:
        dolar = codigo == "USD"
        Moneda.objects.get_or_create(codigo=codigo, defaults=dict(nombre=nombre, simbolo=simbolo, decimales=decimales, por_dolar=1 if dolar else None, fija=dolar))


class Migration(migrations.Migration):
    """Las monedas en que se muestran los precios, y en cada orden el precio en dólares y el tipo de cambio usados."""

    dependencies = [
        ("pagos", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Moneda",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("codigo", models.CharField(help_text="El código ISO de la moneda: CLP, MXN, EUR…", max_length=3, unique=True, verbose_name="código")),
                ("nombre", models.CharField(help_text="En plural y con minúscula: «pesos chilenos»", max_length=60)),
                ("simbolo", models.CharField(default="$", help_text="Si es solo «$», el precio lleva además el código: $100.000 CLP", max_length=6, verbose_name="símbolo")),
                ("decimales", models.PositiveSmallIntegerField(default=2, help_text="Decimales con que se cobra en esta moneda. El peso chileno, 0.")),
                ("por_dolar", models.DecimalField(blank=True, decimal_places=6, help_text="Cuántas unidades de esta moneda vale un dólar. En blanco, los precios no se muestran en ella.", max_digits=18, null=True, verbose_name="unidades por dólar")),
                ("fija", models.BooleanField(default=False, help_text="Marcado, la actualización automática de cada día no cambia este valor", verbose_name="tipo de cambio fijo")),
                ("actualizada", models.DateTimeField(blank=True, editable=False, help_text="La fecha que informó la fuente. Vacío si el valor se escribió a mano.", null=True, verbose_name="valor de la fuente del")),
                ("observacion", models.CharField(blank=True, editable=False, max_length=200, verbose_name="observación")),
            ],
            options={"ordering": ["codigo"]},
        ),
        migrations.AddField(
            model_name="orden",
            name="cambio",
            field=models.DecimalField(decimal_places=6, default=0, help_text="Unidades de la moneda de la orden por dólar con que se calculó el monto", max_digits=18, verbose_name="tipo de cambio"),
        ),
        migrations.AddField(
            model_name="orden",
            name="precio_usd",
            field=models.DecimalField(decimal_places=2, default=0, help_text="El precio del curso al comprar", max_digits=8, verbose_name="precio en dólares"),
        ),
        migrations.AlterField(
            model_name="orden",
            name="monto",
            field=models.DecimalField(decimal_places=2, help_text="Lo que se cobra, en la moneda de la orden", max_digits=12),
        ),
        migrations.RunPython(sembrar, migrations.RunPython.noop),
    ]
