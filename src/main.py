from src.preprocessing import load_data
from src.model import build_model
from src.train import train_model
from src.inference import predict_image
import argparse


def main():
    print("\n🔹 Chargement et prétraitement des données...")
    X_train, X_test, y_train, y_test = load_data()
    
    print("\n🔹 Initialisation du modèle CNN...")
    model = build_model()
    
    print("\n🔹 Entraînement du modèle...")
    train_model(model, X_train, y_train, X_test, y_test)
    
    print("\n✅ Modèle entraîné avec succès !")
    
    # Optionnel : Prédiction sur une image test
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Chemin vers une image à prédire')
    args = parser.parse_args()
    
    if args.image:
        print("\n🔹 Prédiction sur l'image :", args.image)
        predict_image(model, args.image)


if __name__ == "__main__":
    main()

