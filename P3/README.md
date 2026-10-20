# Práctica 3

En esta tarea tratamos de distinguir diferentes objetos en la imagen utilizando parámetros como: el contorno, area, perimetro, diametro...

Hecho por:
  - [Eduardo Arbelo Rua-Figueroa](https://github.com/edulanarf)
  - [Pablo Campos Rico](https://github.com/pabr0)

## Índice

- [Monedas](#monedas)
- [Partículas](#partículas)

## Monedas

## Partículas

En este trata de separar los diferentes elementos de la imagen en 3 grupos: FRA (fragments), PEL (pellets) y TAR. Para esto se han utilizado imágenes que contienen los 3 grupos por separado para obtener las características de estos y utilizar estas para diferenciar en una imagen con los 3 grupos conjuntos los elementos.

En primer lugar, se establece un area mínima para evitar utilizar contornos muy pequeños (desperfectos de la imagen):

```
MIN_AREA = 10
```

### **Función loadImage()**  
Mediante esta función, cargamos una imagen en concreto y se lanza una excepción en caso de no encontrar una imagen:

```
def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"No se encuentra la imagen: {path}")

    return image
```

### **Función extract_particles()**  

Esta función se encarga de recortar las diferentes partículas de la imagen.

En primer lugar se aplica el umbralizado (OTSU):

```
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
```

Luego obtenemos el color predominante del fondo: De esta manera podemos saber si el borde de la partícula es blanco y el fondo negro o al contrario, borde negro y fondo blanco:

```
    border = np.concatenate((
        mask[0, :],
        mask[-1, :],
        mask[:, 0],
        mask[:, -1],
    ))

    # Se determina el color del fondo:
    if np.mean(border) > 127:
        mask = cv2.bitwise_not(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
```

Por último se devuelven todos los contornos que tengan un area mayor a la propuesta (MIN_AREA=10):

```
    return [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_AREA
    ]
```

### **Función extract_features()**

Haciendo uso de esta función, extraemos las diferentes características de cada una de las partículas y las metemos en un array de Numpy:

- Área
- Circularidad
- Ratio de aspecto
- Ratio de elipse
- Solidez
- Colores (HSV)

### **Función build_reference_features()**

Mediante esta funcion se calculan las características reales de cada imagen de referencia FRA, PEL y TAR y las guardamos para poder compararlas mas adelante:

```
    for label, path in train_images.items():
        image = load_image(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        contours = extract_particles(image, gray)
        features_list = [
            features
            for contour in contours
            if (
                features := extract_features(
                    image,
                    contour,
                    gray,
                    hsv,
                )
            ) is not None ]
```

