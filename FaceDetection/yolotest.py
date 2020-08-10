import tensorflow as tf
import yolo
import os
import numpy as np
dataset = yolo.Dataset("data/dataset.yml", "train", image_size=416, augment=True)
val_dataset = yolo.Dataset("data/dataset.yml", "test", image_size=416, augment=False)

model = yolo.models.YOLOv4(num_classes=1)

model.compile(
    optimzer=tf.optimizers.Adam()
)

model.fit(dataset, val_dataset=val_dataset,  epochs = 100, start_lr = 5e-6, end_lr = 1e-8)