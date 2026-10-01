# Práctica 2
## Índice

- [Canny](#canny)
- [Sobel](#sobel)
- [](#)

## Canny

Para esta tarea, se realiza la cuenta de píxeles blancos por filas y se determina el valor máximo de píxeles blancos para filas, maxfil, mostrando el número de filas y sus respectivas posiciones, con un número de píxeles blancos mayor o igual que 0.90 * maxfil.  

Mediante el uso de la función de Canny, se detectan los bordes de la imagen asignando un umbral mínimo descartando el borde si la intensidad del gradiente es menor y asignando también un umbral máximo descartando el borde si la intensidad del gradiente es mayor. En este caso el umbral mínimo elegido sería 100 y el máximo 200:  
  
```  
canny = cv2.Canny(gris, 100, 200)
``` 
  
A continuación se suman los valores de los píxeles de las filas de la imagen:  
  
```
row_counts = cv2.reduce(canny, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32SC1)
```

Se normaliza en base al número de columnas (segundo valor del shape) y al valor máximo del píxel. Esto devuelve la cantidad de pixeles blancos por filas:

```
rows = row_counts / (255 * canny.shape[1])
```

Para finalizar, se obtiene el valor máximo de estos píxeles blancos multiplicándolo por 0.9 obteniendo así el umbral:


```
rows = row_counts / (255 * canny.shape[1])
umbral = max(rows) * 0.9
filas_sel = np.where(rows >= umbral)[0]
```

# Sobel
