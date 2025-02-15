    epochs=EPOCHS
)



# Sauvegarde du modèle entraîné
# Récupération des métriques dans l'objet 'history'
acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)



# Affichage des courbes de précision
plt.figure(figsize=(12, 6))

plt.plot(epochs_range, acc, label='Précision entraînement')
plt.plot(epochs_range, val_acc, label='Précision validation')
plt.legend(loc='lower right')
plt.title('Précision entraînement et validation')



# Affichage des courbes de perte
plt.plot(epochs_range, loss, label='Perte entraînement')
plt.plot(epochs_range, val_loss, label='Perte validation')
plt.legend(loc='upper right')
plt.title('Perte entraînement et validation')

plt.show()



# Affichage des précisions finales
print(f"Précision finale sur l'entraînement : {acc[-1]*100:.2f}%")
print(f"Précision finale sur la validation   : {val_acc[-1]*100:.2f}%")


import numpy as np
