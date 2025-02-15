model.save("alzheimer_cnn_model.h5")

import matplotlib.pyplot as plt

plt.subplot(1, 2, 1)
plt.subplot(1, 2, 2)
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# model = load_model("alzheimer_cnn_model.h5")
# Sinon, si le modèle est déjà dans la variable `model`, continuez.

# Définition des paramètres (doivent correspondre à ceux utilisés pour l'entraînement)
IMG_SIZE = (150, 150)  # Taille d'entrée du modèle



# Chemin vers l'image à tester (remplacez par le chemin de votre image)
image_path = "/home/yuba/Bureau/Python/CNN/Alzheimer/Alzheimer_s Dataset/test/VeryMildDemented/26 (68).jpg"

# Chargement et redimensionnement de l'image
img = load_img(image_path, target_size=IMG_SIZE)
img_array = img_to_array(img)


# Afficher l'image originale
