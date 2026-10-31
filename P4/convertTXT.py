
import os
import json
import random
import shutil
from pathlib import Path

#Mediante este fichero, pasamos a txt todos los json y dividimos las imagenes entre train, val y test
# Configuración
IMAGES_DIR = Path("fotos")
ANNOTATIONS_DIR = Path("labels")
DATASET_DIR = Path("dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

random.seed(42)

# Buscar todas las imágenes
image_extensions = [".jpg", ".jpeg", ".png"]

images = [
    p for p in IMAGES_DIR.iterdir()
    if p.suffix.lower() in image_extensions
]

random.shuffle(images)

# Dividir dataset
total = len(images)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": images[:train_end],
    "val": images[train_end:val_end],
    "test": images[val_end:]
}

# Procesar cada partición
for split, split_images in splits.items():

    for image_path in split_images:

        json_path = ANNOTATIONS_DIR / (image_path.stem + ".json")

        if not json_path.exists():
            print(f"Falta anotación: {json_path}")
            continue

        # Leer JSON
        with open(json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # Copiar imagen
        destination_image = (
            DATASET_DIR / split / "images" / image_path.name
        )

        shutil.copy2(image_path, destination_image)

        # Crear archivo TXT YOLO
        destination_label = (
            DATASET_DIR / split / "labels" / (image_path.stem + ".txt")
        )

        with open(destination_label, "w", encoding="utf-8") as f:

            for annotation in annotations:

                x_center, y_center, width, height = annotation["rectangle"]

                # Una única clase: matrícula (0)
                f.write(
                    f"0 {x_center} {y_center} {width} {height}\n"
                )

print("Dataset preparado correctamente.")

for split, split_images in splits.items():
    print(f"{split}: {len(split_images)} imágenes")
