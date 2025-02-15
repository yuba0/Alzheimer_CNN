import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.data import AUTOTUNE

# Définition des paramètres
IMG_SIZE = (150, 150)     # Taille des images
BATCH_SIZE = 32
EPOCHS = 50               # Nombre d'époques élevé pour tenter d'atteindre 80 % de précision

# Chemins vers vos dossiers de données
TRAIN_DIR = '/home/yuba/Bureau/Python/CNN/Alzheimer/Alzheimer_s Dataset/train'
VALIDATION_DIR = '/home/yuba/Bureau/Python/CNN/Alzheimer/Alzheimer_s Dataset/test'

# Chargement des datasets 
# Cette fonction lit les images depuis les répertoires et génère automatiquement les labels basés sur le nom des dossiers.
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',         # Les labels sont inférés depuis le nom des sous-dossiers
    label_mode='categorical',  # Pour la classification multi-classes (4 classes)
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123
)

# Optimisation des performances : mise en cache et préchargement
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



# Définition du modèle CNN
model = models.Sequential([
    # La couche de rescaling normalise les pixels entre 0 et 1
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

    # Bloc de convolution 1
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Bloc de convolution 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Bloc de convolution 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Bloc de convolution 4
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Passage en couche entièrement connectée
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Pour réduire le surapprentissage

    # Couche de sortie à 4 neurones (une par classe) avec activation softmax
    layers.Dense(4, activation='softmax')
])



# Compilation du modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Affichage du résumé du modèle
model.summary()



# Entraînement du modèle
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)



# Sauvegarde du modèle entraîné
model.save("alzheimer_cnn_model.h5")

import matplotlib.pyplot as plt

# Récupération des métriques dans l'objet 'history'
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)



# Affichage des courbes de précision
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Précision entraînement')
plt.plot(epochs_range, val_acc, label='Précision validation')
plt.legend(loc='lower right')
plt.title('Précision entraînement et validation')



# Affichage des courbes de perte
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Perte entraînement')
plt.plot(epochs_range, val_loss, label='Perte validation')
plt.legend(loc='upper right')
plt.title('Perte entraînement et validation')

plt.show()



# Affichage des précisions finales
print(f"Précision finale sur l'entraînement : {acc[-1]*100:.2f}%")
print(f"Précision finale sur la validation   : {val_acc[-1]*100:.2f}%")


import numpy as np
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
plt.imshow(img_array.astype("uint8"))
plt.axis("off")
plt.title("Image à tester")
plt.show()



# Préparation de l'image pour le modèle :
# Ajouter une dimension pour simuler un batch de taille 1.
img_batch = np.expand_dims(img_array, axis=0)
# Note : Comme le modèle commence par une couche Rescaling(1./255), il n'est pas nécessaire de normaliser ici.



# Prédiction de la classe
predictions = model.predict(img_batch)
# Récupérer l'indice de la classe prédite et la confiance associée
predicted_index = np.argmax(predictions[0])
confidence = predictions[0][predicted_index]

# Définir les noms de classes (doivent être dans le même ordre que lors de la création du dataset)
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
predicted_class = class_names[predicted_index]



# Affichage du résultat de la prédiction
print(f"Prédiction : {predicted_class} avec une confiance de {confidence * 100:.2f}%")


