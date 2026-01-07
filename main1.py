import numpy as np
import cv2, os
from PIL import Image
import tensorflow as tf
from keras.preprocessing.image import img_to_array, save_img, array_to_img
from tensorflow.keras.models import load_model

model_path = r"C:\Users\subha\Downloads\generator.h5"
image_path = r"D:\dataLOL\eval15\low\1007.png"

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
original_shape = img.shape[:2]

img_arr = (img_to_array(img) - 127.5) / 127.5
resized = cv2.resize(img_arr, (256, 256), interpolation=cv2.INTER_AREA)
ready_img = np.expand_dims(resized, axis=0)


model = load_model(model_path)


pred = model.predict(ready_img)
pred = (pred[0] + 1) / 2 
pred = (pred * 255).astype(np.uint8) 


pred_resized = cv2.resize(pred, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
pred_resized = array_to_img(pred_resized)


save_img("./output.png", pred_resized)
import matplotlib.pyplot as plt


original = cv2.imread(image_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)  

enhanced = cv2.imread("./output.png")
enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)  


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title("Low-Light Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced)
plt.title("Enhanced Image")
plt.axis("off")

plt.show()
