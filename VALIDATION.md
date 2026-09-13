# Validation technique

Date : 12 septembre 2026.

Commande executee :

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m pytest -q
```

Resultat : **40 passed**.

Les donnees des tests sont artificielles. Aucun score d'evaluation immobiliere
reel n'a ete calcule. Les appels HTTP sont simules dans les tests ; la collecte
complete doit etre executee dans l'environnement de l'utilisateur.

Environnement :
- numpy : 2.3.5
- pandas : 2.2.3
- scikit-learn : 1.8.0
- catboost : 1.2.8
- requests : 2.32.5
- joblib : 1.5.3
- pytest : 9.0.2
- Python : 3.13.5
