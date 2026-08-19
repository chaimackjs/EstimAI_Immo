import pandas as pd 

def lire_fichier(chemin): 
    return pd.read_csv(chemin,sep="|")



if __name__== '__main__':

    df2021= lire_fichier("data\\dvf\\ValeursFoncieres-2021.txt")
    df2022= lire_fichier("data\\dvf\\ValeursFoncieres-2022.txt")
    df2023= lire_fichier("data\\dvf\\ValeursFoncieres-2023.txt")
    df2024= lire_fichier("data\\dvf\\ValeursFoncieres-2024.txt")
    df2025= lire_fichier("data\\dvf\\ValeursFoncieres-2025.txt")

    print("Données 2021 : \n")

    print("Nombre de lignes : ",df2021.shape[0])
    print("Nombre de colonnes : ",df2021.shape[1])

    print("Colonnes : ", df2021.columns)
    print("Types des colonnes : ", df2021.dtypes)

    print("Valeur nulles : \n", df2021.isnull().sum())

    print("\nInformations générales : ")
    df2021.info();

    print("Les 5 premières lignes : ")
    print(df2021.head())