import pandas as pd
import os

def lire_fichier(chemin): 
    return pd.read_csv(chemin,sep="|")

def afficher_infos_generales(df):

    print("Nombre de lignes : ",df.shape[0])
    print("Nombre de colonnes : ",df.shape[1])

    print("Valeur nulles : \n", df.isnull().sum())

    print("\nInformations générales : ")
    df.info();

    print("Les 5 premières lignes : ")
    print(df.head())


def nettoyage(df):
    # Standariser les noms de colonnes (maj->min) et remplacer les espaces par des "_"
    df.columns = (df.columns.str.lower().str.replace(" ","_"))
    # Transformer les data de la colonne "date_mutation" de type"str" en vrai dates qu'on pourra utiliser
    df["date_mutation"] = pd.to_datetime(df["date_mutation"],dayfirst=True)
    # Les valeurs de la colonne contiennent des virgules ce qui empêche l'utilisation numérique des valeurs
    # on transforme donc les "," en "." puis on transfomr en valeur numérique avec "pd.to_numeric"
    df["valeur_fonciere"] = pd.to_numeric(df["valeur_fonciere"].str.replace(",","."))
    # On supprime dans la colonne "nature_mutation' les lignes qui ne correspondent pas à des mutations de nature vente
    df = df[ df["nature_mutation"]=="Vente"  ]
    # On supprime dans la colonne "type_local' les lignes qui ne correspondent pas à des locaux de type maison ou appartement
    df = df[ df["type_local"].isin(["Maison","Appartement"])  ]
    # On supprime les colonnes complétement vides
    df = df.dropna(axis=1, how="all")
    # On supprime les colonnes non pertinante après première analyse
    df = df.drop(columns=["no_disposition","b/t/q","no_plan","type_local","nature_culture","nature_culture_speciale"])

    return df
    


if __name__== '__main__':

    df2021= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2021.txt"))
    df2022= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2022.txt"))
    df2023= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2023.txt"))
    df2024= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2024.txt"))
    df2025= lire_fichier(os.path.join("data","dvf","ValeursFoncieres-2025.txt"))

    print("\nDonnées 2021 : \n")
    afficher_infos_generales(df2021)


    print("Valeurs nature_mutation :", df2021["Nature mutation"].unique())
    print("Valeurs type_local :", df2021["Type local"].unique())


    print("\nDonnées 2022 : \n")
    afficher_infos_generales(df2022)

    print("\nDonnées 2023 : \n")
    afficher_infos_generales(df2023)

    print("\nDonnées 2024 : \n")
    afficher_infos_generales(df2024)

    print("\nDonnées 2025 : \n")
    afficher_infos_generales(df2025)

    # print(df2021.columns[df2021.isnull().all()])

    df2021 = nettoyage(df2021)
    df2022 = nettoyage(df2022)
    df2023 = nettoyage(df2023)
    df2024 = nettoyage(df2024)
    df2025 = nettoyage(df2025)
            

    print("\nDonnées 2021 : \n")
    afficher_infos_generales(df2021)

    print()
