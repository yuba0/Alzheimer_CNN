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


