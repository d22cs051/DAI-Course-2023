import albumentations as A
import cv2
from matplotlib import pyplot as plt

# Declare an augmentation pipeline
transform = A.Compose([
    A.Resize(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CLAHE(),
    A.RandomRotate90(),
    A.Transpose(),
    A.Blur(blur_limit=3),
    A.OpticalDistortion(),
    A.GridDistortion(),
    A.HueSaturationValue(),
])

k = 1
for i in range(1, 10):
    # Read an image with OpenCV and convert it to the RGB colorspace
    image = cv2.imread(f"ORIGINAL_PICS/{i}.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for j in range(12):
        # Augment an image
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        cv2.imwrite(f"data/original_pic_augmented/img{k}.jpg",transformed_image)
        print(f"SAVED: 'data/original_pic_augmented/{k}.jpg'")
        k+=1