import django.core.validators
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    """Los precios pasan a dólares y cada alumno tiene un perfil con su país.

    El campo del precio cambia de nombre y de tipo, y conserva su número: un curso que tenía precio en la moneda del
    sitio queda con ese mismo número en dólares, a la vista en el panel para corregirlo, y nunca gratuito por error."""

    dependencies = [
        ("cursos", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.RenameField(model_name="curso", old_name="precio", new_name="precio_usd"),
        migrations.AlterField(
            model_name="curso",
            name="precio_usd",
            field=models.DecimalField(
                decimal_places=2,
                default=0,
                help_text="En dólares (USD). Cada alumno lo ve además en la moneda de su país. Con 0 el curso es gratuito.",
                max_digits=8,
                validators=[django.core.validators.MinValueValidator(0)],
                verbose_name="precio en dólares",
            ),
        ),
        migrations.CreateModel(
            name="Perfil",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("pais", models.CharField(blank=True, help_text="El código del país, de la lista de `paises.py`", max_length=2, verbose_name="país")),
                ("usuario", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="perfil", to=settings.AUTH_USER_MODEL)),
            ],
            options={"verbose_name_plural": "perfiles"},
        ),
    ]
