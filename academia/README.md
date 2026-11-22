# Academia

Una academia en línea propia, al estilo de las plataformas de cursos: catálogo con varios cursos, cuentas de alumnos
(con Google o con correo y contraseña), compra con pasarela de pago, aula con avance guardado, evaluaciones
corregidas en el servidor, certificado verificable y un panel para subir cursos. Es una aplicación Django.

| Qué | Dónde |
|---|---|
| Catálogo, página del curso, aula, pruebas, certificado, panel | `cursos/` |
| Órdenes de compra y pasarelas (Flow, Mercado Pago y una de prueba) | `pagos/` |
| Configuración y direcciones | `plataforma/` |
| Los cursos escritos como paquetes (el primero: «Claude para Finanzas y Contabilidad») | `contenido/` |

## Probarla en el computador

```bash
pip install -r academia/requirements.txt
python academia/manage.py migrate
python academia/manage.py importar_curso academia.contenido.claude_finanzas
python academia/manage.py createsuperuser        # tu cuenta de administración
python academia/manage.py runserver
```

Queda en http://localhost:8000 y el panel en `/admin/`. Sin configurar nada usa una base de datos local
(`academia/datos/`), muestra los correos en la consola y cobra con la **pasarela de prueba**, que simula el pago.
El curso importado llega sin publicar: en el panel se le pone precio y se marca «Publicado».

## Cómo se sube un curso

- **En el panel** (`/admin/`): se crea el curso, sus módulos y, dentro de cada módulo, sus lecciones; ahí mismo se
  sube el video de cada lección, sus subtítulos y los archivos para descargar. Las listas se escriben una por
  línea, y en las preguntas la respuesta correcta se indica contando desde 1.
- **Como paquete** en `contenido/` (así está hecho el primero): la ficha en `__init__.py`, el temario en
  `programa.py` y las preguntas en `evaluaciones.py`. Se carga con `manage.py importar_curso <paquete>`. Importar de
  nuevo actualiza textos y preguntas sin tocar el precio, los videos ni lo que hicieron los alumnos.

Una lección sin video aparece como «En producción». Una lección marcada «vista previa gratuita» se puede ver sin
comprar, y cada curso puede tener además un video de muestra en su página.

## Cuentas

Correo y contraseña, o Google. El botón de Google aparece cuando están `GOOGLE_CLIENT_ID` y `GOOGLE_CLIENT_SECRET`
(un cliente OAuth «aplicación web» de Google Cloud, con la dirección de retorno
`https://<tu dominio>/cuentas/google/login/callback/`). Quien entra con Google y ya tenía cuenta con ese correo
queda en la misma cuenta. Recuperar la contraseña y confirmar el correo necesitan correo saliente (`ACADEMIA_SMTP_*`).

## Pagos

Comprar crea una orden pendiente y lleva al alumno a la pasarela. La orden se da por pagada solo cuando la pasarela
lo avisa al servidor y el servidor lo comprueba con ella, por el monto exacto; recién entonces se crea la
inscripción y se envía el correo de confirmación. Un aviso repetido no hace nada.

| Pasarela | Se activa con | Estado |
|---|---|---|
| Flow | `FLOW_API_KEY`, `FLOW_SECRET_KEY`, `FLOW_URL` | Escrita según su documentación y probada con respuestas simuladas. Falta probarla en su ambiente sandbox. |
| Mercado Pago | `MERCADOPAGO_ACCESS_TOKEN` (y `MERCADOPAGO_WEBHOOK_SECRET` para validar la firma de los avisos) | Igual: falta probarla con credenciales de prueba. |
| De prueba | solo en desarrollo, o con `ACADEMIA_PASARELAS=prueba` | Funciona: es la que usan las pruebas automáticas. |

Un pago recibido por otra vía (una transferencia, por ejemplo) se registra en el panel: en Órdenes, «Dar por
pagadas». Una cortesía se da creando la inscripción. La academia no emite boletas ni facturas.

## Evaluaciones y certificado

Cada evaluación tiene un banco de preguntas (una correcta, varias correctas, numérica u ordenar). Un intento toma
preguntas al azar y baraja las alternativas; la página no recibe las respuestas, y las explicaciones llegan al
entregar. El diagnóstico no lleva nota y recomienda por dónde partir. El certificado se emite al aprobar las
pruebas y el examen final (o al completar las lecciones, según el curso) y cualquiera puede comprobarlo en
`/certificados/<código>/`.

## Configuración (variables de entorno)

| Variable | Para qué |
|---|---|
| `ACADEMIA_DEBUG` | `0` en el servidor |
| `ACADEMIA_SECRET_KEY` | Clave secreta de la aplicación (obligatoria en el servidor) |
| `ACADEMIA_HOSTS`, `ACADEMIA_URL` | El dominio (`academia.ejemplo.cl`) y la dirección pública (`https://academia.ejemplo.cl`) |
| `DATABASE_URL` | La base de datos PostgreSQL |
| `ACADEMIA_NOMBRE`, `ACADEMIA_MONEDA` | Nombre del sitio y moneda de los precios (por defecto CLP) |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Entrar con Google |
| `ACADEMIA_SMTP_HOST`, `_PORT`, `_USUARIO`, `_CLAVE`, `ACADEMIA_REMITENTE` | Correo saliente |
| `ACADEMIA_VERIFICAR_CORREO` | `mandatory` para exigir confirmar el correo al registrarse |
| `FLOW_*`, `MERCADOPAGO_*`, `ACADEMIA_PASARELAS` | Pagos |
| `ACADEMIA_S3_BUCKET`, `_ENDPOINT`, `_CLAVE`, `_SECRETO`, `_REGION` | Videos e imágenes en un almacenamiento compatible con S3, con enlaces firmados que caducan |

En el servidor: `pip install -r academia/requirements.txt`, `python academia/manage.py collectstatic --noinput`,
`python academia/manage.py migrate` y, para atender, `gunicorn academia.plataforma.wsgi`.

## Pruebas

```bash
pip install pytest pytest-django
pytest tests/test_academia.py tests/test_academia_web.py -q
```

Cubren el contenido del primer curso y la aplicación: registro e ingreso, acceso a las lecciones, avance, compra
con cada pasarela (las externas, con respuestas simuladas), corrección de pruebas, certificado y panel.

## Lo que falta

- **Publicarla**: un servidor con base de datos, el dominio, y las credenciales de Google, de la pasarela, del
  correo y del almacenamiento de videos. Nada de eso está contratado.
- **Probar Flow y Mercado Pago contra sus ambientes de prueba** antes de cobrar.
- **Las lecciones del primer curso**: se producen con el motor del curso gratuito (`curso_youtube/`), una por video.
- Términos y condiciones, política de privacidad y la emisión de boletas.
- Cupones de descuento, reseñas y buscador: no están en esta versión.
