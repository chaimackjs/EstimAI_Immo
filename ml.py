import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Chemin vers les données prétraitées
DATA_PATH = os.path.join("data", "clean", "dfv_dpe.csv")

def preparer_donnees(df_model, target="prix_m2"):

    # Sélectionner les colonnes numériques pertinentes pour les features
    colonnes_features = [
        "surface_reelle_bati",
        "nombre_pieces_principales",
        "surface_terrain",
        "nombre_de_lots",
        "annee_mutation",
        "mois_mutation",
        "annee_dpe",
        "etiquette_dpe",
        "etiquette_ges",
        "qualite_isolation_enveloppe",
        "qualite_isolation_murs",
        "qualite_isolation_menuiseries",
        "surface_habitable_logement",
        "annee_construction",
    ]
    
    # Filtrer les colonnes qui existent dans le DataFrame
    colonnes_existantes = [col for col in colonnes_features if col in df_model.columns]
    print(f"Colonnes utilisées: {colonnes_existantes}")
    
    # Supprimer les lignes avec des valeurs manquantes
    df_clean = df_model[colonnes_existantes + [target]].dropna()
    
    print(f"Nombre d'échantillons après suppression des valeurs manquantes: {len(df_clean)}")
    
    # Séparer les features et la cible
    X = df_clean[colonnes_existantes]
    y = df_clean[target]
    
    # Supprimer les valeurs aberrantes (prix_m2 > 0 et < 50000)
    if target == "prix_m2":
        mask = (y > 0) & (y < 50000)
        X = X[mask]
        y = y[mask]
    
    print(f"Nombre d'échantillons après suppression des valeurs aberrantes: {len(X)}")
    
    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normaliser les features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, colonnes_existantes

def entrainer_modele(X_train, X_test, y_train, y_test, model):
    
    print(f"\nEntraînement du modèle ...")
    
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculer les métriques
    metriques = {
        "rmse_train": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
    }
    
    return model, metriques

def afficher_metriques(metriques, model_nom):
    """Affiche les métriques d'évaluation du modèle"""
    print(f"\n{'='*50}")
    print(model_nom)
    print(f"{'='*50}")
    print(f"RMSE Train: {metriques['rmse_train']:.2f}")
    print(f"RMSE Test:  {metriques['rmse_test']:.2f}")
    print(f"MAE Train:  {metriques['mae_train']:.2f}")
    print(f"MAE Test:   {metriques['mae_test']:.2f}")
    print(f"R² Train:   {metriques['r2_train']:.4f}")
    print(f"R² Test:    {metriques['r2_test']:.4f}")
    print(f"{'='*50}\n")


def enregistrer_model(model,nom_model):
    os.makedirs("models",exist_ok=True)

    chemin=os.path.join("models",f"{nom_model}.joblib")

    joblib.dump(model,chemin)

    print(f"Modèle enregistré : {chemin}")


def main():
    
    # Charger les données
    print("Chargement des données...")
    df_model = pd.read_csv(DATA_PATH)

    df_model=df_model.sample(n=300000)


    # Préparer les données
    print("\nPréparation des données...")
    X_train, X_test, y_train, y_test, scaler, colonnes_features = preparer_donnees(df_model, target="prix_m2")


    model_list = [
        {"model_nom":"LinearRegression","model":LinearRegression()},
        {"model_nom":"RandomForestRegressor","model":RandomForestRegressor()},
#        {"model_nom":"LinearRegression","model":LinearRegression()},
#        {"model_nom":"LinearRegression","model":LinearRegression()},
    ]

    for model_dic in model_list:

        # Entraîner le modèle
        model, metriques = entrainer_modele(X_train, X_test, y_train, y_test,model_dic["model"])
        
        # Afficher les métriques
        afficher_metriques(metriques,model_dic["model_nom"])

        enregistrer_model(model,model_dic["model_nom"])
    
if __name__ == "__main__":
    main()
