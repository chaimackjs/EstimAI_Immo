# EstimAI_Immo

## Récupération des données dvf:

Le lancement manuel du programme nécessite de récupérer manuellement les données disponibles sur le site « https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres ».

Il faut ensuite télécharger les fichiers correspondant aux cinq dernières années disponibles. Une fois les fichiers téléchargés, il faut les décompresser, récupérer les fichiers texte qu’ils contiennent et les placer dans un dossier nommé data/dvf, qui devra être créé au préalable.

## Récupération des données dpe: 

Il est possible de récupérer les données dpe sous forme de fichier : "https://www.data.gouv.fr/dataservices/dpe-logements-existants-depuis-juillet-2021" dans la partie : "Les outils à votre disposition" puis selectionner "Un accès en téléchargement aux données sous forme d’une base de données".
Cependant, la taille du fichier à récupérer est énorme, nous devons opter pour la deuxième option qui est celle de récupérer lesd onnées directement à part de l'api. Le programme récupère un certain nombre de lignes fixes qu'il est possible de changer depuis "main.py", dans la fonction" def recuperer_dpe(nb_ligne=10000)
    