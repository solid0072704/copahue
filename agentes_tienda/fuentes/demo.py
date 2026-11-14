"""Catálogo ficticio para ver la tienda funcionando sin claves de API.

Los productos de demostración se marcan con demo=True: mientras existan, el sitio muestra
un aviso y le pide a los buscadores que no lo indexen (noindex).
"""

# (nicho, título "estilo proveedor", costo USD, pedidos, valoración %, días de envío, comisión %)
CATALOGO = [
    ("Hogar y cocina", "Organizador giratorio 360 para especias, cocina y baño, estante de almacenamiento", 6.8, 8400, 97.1, 12, 8.0),
    ("Hogar y cocina", "Picador de verduras manual multifunción, cortador de cebolla y ajo, utensilios de cocina", 9.5, 15200, 96.4, 10, 7.0),
    ("Hogar y cocina", "Dispensador automático de jabón espumoso, sensor infrarrojo, recargable USB", 11.2, 6300, 95.8, 14, 9.0),
    ("Hogar y cocina", "Selladora de bolsas mini portátil al calor, conserva alimentos frescos", 4.1, 22400, 93.2, 11, 6.0),
    ("Hogar y cocina", "Tira de luz LED con sensor de movimiento para clóset y cocina, recargable", 7.4, 9100, 96.9, 13, 8.0),
    ("Hogar y cocina", "Báscula de cocina digital de precisión, pantalla LCD, acero inoxidable", 8.9, 4700, 97.5, 12, 7.0),
    ("Mascotas", "Cepillo autolimpiante para perros y gatos, quita pelo muerto con un botón", 5.6, 31000, 97.8, 10, 9.0),
    ("Mascotas", "Fuente de agua para gatos 2L, filtro triple, bomba silenciosa", 14.9, 7800, 96.2, 15, 10.0),
    ("Mascotas", "Comedero lento antiahogo para perros, plato laberinto, digestión sana", 4.9, 12800, 97.0, 12, 8.0),
    ("Mascotas", "Pelota interactiva inteligente para gatos, movimiento automático, recargable", 9.8, 5400, 94.1, 14, 9.0),
    ("Mascotas", "Botella de agua portátil para perros de paseo, con dispensador de comida", 7.2, 16300, 96.6, 11, 8.0),
    ("Bienestar", "Masajeador cervical eléctrico para cuello y hombros, calor, pulso de baja frecuencia", 18.5, 11900, 95.3, 13, 9.0),
    ("Bienestar", "Pistola de masaje muscular profunda, 6 cabezales, 30 velocidades, silenciosa", 32.0, 9800, 96.0, 12, 8.0),
    ("Bienestar", "Difusor de aromas ultrasónico, humidificador con luz LED de 7 colores, 300 ml", 12.4, 14300, 96.7, 14, 9.0),
    ("Bienestar", "Rodillo de jade y gua sha facial, set de masaje para el rostro", 5.2, 26700, 95.9, 12, 7.0),
    ("Bienestar", "Antifaz de dormir 3D con audífonos bluetooth, música para descansar", 9.9, 8800, 94.8, 13, 8.0),
    ("Accesorios para auto", "Soporte magnético de celular para auto, rejilla de ventilación, giratorio 360", 4.3, 41200, 96.8, 10, 7.0),
    ("Accesorios para auto", "Aspiradora portátil para auto inalámbrica, alta succión, 120W", 19.9, 6900, 94.6, 14, 9.0),
    ("Accesorios para auto", "Organizador para maletero plegable, gran capacidad, impermeable", 13.5, 5200, 96.3, 15, 8.0),
    ("Accesorios para auto", "Cámara de retroceso con visión nocturna, impermeable, gran angular", 21.0, 3100, 92.4, 16, 9.0),
    ("Tecnología útil", "Mini proyector portátil 4K wifi bluetooth, cine en casa, compatible con celular", 45.0, 7300, 93.9, 15, 9.0),
    ("Tecnología útil", "Cargador inalámbrico 3 en 1 plegable para celular, reloj y audífonos", 16.8, 9600, 95.7, 12, 9.0),
    ("Tecnología útil", "Lámpara de escritorio LED con carga inalámbrica, regulable, protección ocular", 15.2, 4100, 96.4, 13, 8.0),
    ("Tecnología útil", "Audífonos inalámbricos estilo AirPods Pro réplica 1:1", 7.9, 52000, 89.0, 10, 6.0),
    ("Tecnología útil", "Localizador inteligente bluetooth para llaves y billetera, alarma antipérdida", 6.4, 18700, 94.2, 11, 8.0),
    ("Fitness en casa", "Set de bandas elásticas de resistencia 11 piezas, entrenamiento en casa", 9.1, 19800, 96.9, 11, 8.0),
    ("Fitness en casa", "Rueda abdominal con rebote automático y rodillera, ejercicio de core", 11.6, 8700, 96.1, 13, 8.0),
    ("Fitness en casa", "Cuerda para saltar inteligente con contador digital de calorías", 6.9, 10400, 95.4, 12, 8.0),
    ("Fitness en casa", "Mat de yoga antideslizante TPE 6 mm, con líneas de alineación", 14.0, 6200, 97.2, 14, 7.0),
    ("Fitness en casa", "Barra de dominadas para puerta sin tornillos, ajustable, carga 150 kg", 18.9, 2900, 91.5, 18, 8.0),
]


def productos():
    salida = []
    for i, (nicho, titulo, costo, pedidos, valoracion, dias, comision) in enumerate(CATALOGO, start=1):
        salida.append(
            {
                "fuente": "demo",
                "id_fuente": f"{i:03d}",
                "titulo_original": titulo,
                "categoria_fuente": nicho,
                "imagenes": [],
                "video": None,
                "costo_usd": costo,
                "precio_lista_usd": None,
                "envio_usd": None,
                "pedidos": pedidos,
                "valoracion": valoracion,
                "comision": comision,
                "dias_envio": dias,
                "url_proveedor": "",
                "url_afiliado": None,
                "nicho": nicho,
                "busqueda": "",
                "demo": True,
            }
        )
    return salida
