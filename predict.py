import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import os

# ✅ Load the model
model_path = "best_model.h5"  # Adjust if your model is in another folder
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")
model = load_model(model_path)

# ✅ Correct class names (based on your folders)
class_names = ['Bacterial_dermatosis', 'flea_allergy', 'healthy', 'hotspot', 'mange', 'ringworm']

# ✅ Get user input image
img_path = input("👉 Enter full path to the image: ")
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found at {img_path}")

# ✅ Preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# ✅ Predict
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
confidence = float(np.max(predictions)) * 100

# ✅ Get predicted class
if predicted_index < len(class_names):
    predicted_class = class_names[predicted_index]
else:
    predicted_class = "Unknown"

# ✅ Output result
print(f"\n✅ Predicted Class: {predicted_class}")
print(f"🔍 Confidence: {confidence:.2f}%")

# ✅ Display image with prediction
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f}%)", fontsize=14)
plt.axis('off')
plt.show()
