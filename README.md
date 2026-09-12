# EstimAI_Immo

## Interface d'estimation

L'interface de production est une petite application Streamlit. Installez les dépendances puis préparez les données comme indiqué ci-dessous :

```bash
pip install -r requirements.txt
python preparer_donnees.py
python ml.py
streamlit run app.py
```

L'application est ensuite accessible à l'adresse affichée par Streamlit, généralement `http://localhost:8501`. Elle utilise le modèle `models/HistGradientBoostingRegressor.joblib` et estime le prix au m² à partir des caractéristiques saisies.

## Déploiement gratuit depuis GitHub

GitHub Pages héberge uniquement des fichiers statiques et ne peut pas exécuter cette application Python. Pour conserver le code sur GitHub et obtenir une URL publique gratuitement, utilisez **Streamlit Community Cloud**.

1. Générez le modèle sur votre ordinateur :

	```bash
	python preparer_donnees.py
	python ml.py
	```

2. Vérifiez que `models/HistGradientBoostingRegressor.joblib` existe, puis envoyez les fichiers sur GitHub :

	```bash
	git add app.py requirements.txt README.md models/HistGradientBoostingRegressor.joblib
	git commit -m "Ajouter l interface de production"
	git push origin main
	```

3. Ouvrez [share.streamlit.io](https://share.streamlit.io), connectez votre compte GitHub et cliquez sur **Create app**.
4. Sélectionnez le dépôt `chaimackjs/EstimAI_Immo`, la branche `main` et le fichier principal `app.py`.
5. Cliquez sur **Deploy**. Streamlit fournira une URL publique gratuite pour l'interface.

Le dépôt doit contenir le fichier du modèle : Streamlit Cloud exécute l'application, mais ne lance pas automatiquement l'entraînement des données.

## Récupération des données dvf:

Le lancement manuel du programme nécessite de récupérer manuellement les données disponibles sur le site « https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres ».

Il faut ensuite télécharger les fichiers correspondant aux cinq dernières années disponibles. Une fois les fichiers téléchargés, il faut les décompresser, récupérer les fichiers texte qu’ils contiennent et les placer dans un dossier nommé data/dvf, qui devra être créé au préalable.

## Récupération des données dpe: 

Il est possible de récupérer les données dpe sous forme de fichier : "https://www.data.gouv.fr/dataservices/dpe-logements-existants-depuis-juillet-2021" dans la partie : "Les outils à votre disposition" puis selectionner "Un accès en téléchargement aux données sous forme d’une base de données".
Cependant, la taille du fichier à récupérer est énorme, nous devons opter pour la deuxième option qui est celle de récupérer lesd onnées directement à part de l'api. Le programme récupère un certain nombre de lignes fixes qu'il est possible de changer depuis "main.py", dans la fonction" def recuperer_dpe(nb_ligne=10000)


## Utilisation des données prétraitées:

Le programme principal récupère les données depuis le dossier cache s'il existe, cela permet d'exécuter le programme plus rapidement sans passer le programme de nettoyage. 
Pour régénérer le programme il suffit de supprimer et de réexécuter le dossier cache