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
