# Academia

Una academia en línea propia, al estilo de las plataformas de cursos: catálogo con varios cursos, cuentas de alumnos
(con Google o con correo y contraseña), precios en dólares y en la moneda del país de cada alumno, compra con
pasarela de pago, aula con avance guardado, evaluaciones corregidas en el servidor, certificado verificable y un
panel para subir cursos. Es una aplicación Django.

| Qué | Dónde |
|---|---|
| Catálogo, página del curso, aula, pruebas, certificado, panel | `cursos/` |
| Monedas y tipos de cambio, precios, órdenes de compra y pasarelas (Flow, Mercado Pago y una de prueba) | `pagos/` |
| Configuración y direcciones | `plataforma/` |
| Los cursos escritos como paquetes (el primero: «Claude para Finanzas y Contabilidad») | `contenido/` |

## Probarla en el computador

```bash
pip install -r academia/requirements.txt
python academia/manage.py migrate
python academia/manage.py importar_curso academia.contenido.claude_finanzas
python academia/manage.py actualizar_monedas     # los tipos de cambio de hoy
python academia/manage.py createsuperuser        # tu cuenta de administración
python academia/manage.py runserver
```

Queda en http://localhost:8000 y el panel en `/admin/`. Sin configurar nada usa una base de datos local
(`academia/datos/`), muestra los correos en la consola y cobra con la **pasarela de prueba**, que simula el pago.
El curso importado llega sin publicar: en el panel se le pone el precio en dólares y se marca «Publicado».

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

Correo y contraseña, o Google. Al registrarse, cada persona escribe su nombre (el que irá en sus certificados) y
elige su **país**, que decide en qué moneda ve los precios; quien llega por primera vez con Google pasa por el mismo
paso, con su nombre ya escrito. Después se cambian en «Mi aprendizaje → Mi nombre y mi país».

El botón de Google aparece cuando están `GOOGLE_CLIENT_ID` y `GOOGLE_CLIENT_SECRET` (un cliente OAuth «aplicación
web» de Google Cloud, con la dirección de retorno `https://<tu dominio>/cuentas/google/login/callback/`). Quien entra
con Google y ya tenía cuenta con ese correo queda en la misma cuenta. Recuperar la contraseña y confirmar el correo
necesitan correo saliente (`ACADEMIA_SMTP_*`).

## Precios y monedas

El precio de cada curso se escribe **en dólares** (en el panel; admite centavos, «49,90»). De ahí sale todo:

- **Lo que se ve.** Todos ven el precio en dólares. Quien tiene cuenta lo ve además en la moneda de su país: exacto
  si es la moneda en que se cobra («US$ 99 · $100.394 CLP») y como referencia si es otra («≈ $1.893 MXN»). Donde se
  usa el dólar (Ecuador, El Salvador, Panamá, Puerto Rico, Estados Unidos; también Cuba, Venezuela y «otro país») se
  ve solo en dólares. A quien no tiene cuenta, o no ha dicho su país, la página le dice cómo verlo en su moneda.
- **Lo que se cobra.** Las pasarelas cobran en una sola moneda (`ACADEMIA_MONEDA`; por defecto pesos chilenos): el
  precio en dólares por el tipo de cambio de ese día, redondeado a pesos. La página de pago muestra ese monto, el
  tipo de cambio usado y, a quien vive en otro país, que la conversión final la hace su banco. La orden guarda el
  precio en dólares y el tipo de cambio con que se calculó.
- **Los tipos de cambio** están en el panel, en Ventas → Monedas. Llegan una vez al día de la dirección abierta de
  [ExchangeRate-API](https://www.exchangerate-api.com) (no pide clave; exige mostrar «Rates By Exchange Rate API»
  junto a los precios convertidos, y las páginas lo hacen). La academia los va a buscar sola cuando los guardados
  ya tienen más de un día; también se piden con `manage.py actualizar_monedas` (sirve como tarea programada) o con
  la acción «Traer ahora» del panel. Si la fuente falla se sigue con los valores guardados; sin tipo de cambio de la
  moneda de cobro no se puede pagar, y la página lo dice.
- **Un precio que no se mueva.** En Monedas se puede escribir el tipo de cambio a mano y marcarlo «fijo»: la
  actualización diaria no lo toca. Con el dólar fijo en 1.010, un curso de 99 dólares cuesta siempre $99.990.
- **Resguardo.** Un valor de la fuente que difiere más de 30 % del guardado no se aplica solo: queda anotado en la
  moneda para revisarlo, porque con ese número se calcula lo que se cobra.

Los países y la moneda de cada uno están en `cursos/paises.py`; las monedas, en el panel. Para agregar un país basta
una línea en esa lista (y crear su moneda en el panel si es nueva).

## Pagos

Comprar crea una orden pendiente, por el monto en la moneda de las pasarelas (ver «Precios y monedas»), y lleva al
alumno a la pasarela. La orden se da por pagada solo cuando la pasarela lo avisa al servidor y el servidor lo
comprueba con ella, por el monto exacto; recién entonces se crea la inscripción y se envía el correo de
confirmación. Un aviso repetido no hace nada. Si el tipo de cambio cambia entre que el alumno ve el monto y aprieta
«Pagar», no se cobra: se le muestra el monto nuevo.

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
| `ACADEMIA_NOMBRE` | Nombre del sitio |
| `ACADEMIA_MONEDA` | La moneda en que cobran las pasarelas (por defecto CLP). Los precios de los cursos van en dólares |
| `ACADEMIA_CAMBIO_AUTOMATICO`, `ACADEMIA_CAMBIO_URL` | Con `0`, los tipos de cambio solo se actualizan con `actualizar_monedas` o desde el panel. La segunda cambia la fuente |
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

Cubren el contenido del primer curso y la aplicación: registro e ingreso (con correo y, simulado, con Google), los
precios que ve cada país, los tipos de cambio (con respuestas simuladas de la fuente), acceso a las lecciones,
avance, compra con cada pasarela (las externas, con respuestas simuladas), corrección de pruebas, certificado y panel.

## Lo que falta

- **Publicarla**: un servidor con base de datos, el dominio, y las credenciales de Google, de la pasarela, del
  correo y del almacenamiento de videos. Nada de eso está contratado.
- **Probar Flow y Mercado Pago contra sus ambientes de prueba** antes de cobrar.
- **Cobrar fuera de Chile en la moneda de cada país**: hoy todos pagan en la moneda de las pasarelas chilenas y el
  precio en otra moneda es una referencia. Cobrar en pesos mexicanos o soles pide una pasarela internacional.
- **Las lecciones del primer curso**: se producen con el motor del curso gratuito (`curso_youtube/`), una por video.
- Términos y condiciones, política de privacidad y la emisión de boletas.
- Cupones de descuento, reseñas y buscador: no están en esta versión.
