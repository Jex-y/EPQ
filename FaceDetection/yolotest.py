import tensorflow as tf
import yolo
import os

dataset = yolo.Dataset("data/dataset.yml", "train")


physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = yolo.models.YOLOv4(num_classes=1)

next(dataset)
