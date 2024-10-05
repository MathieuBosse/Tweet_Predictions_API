# Modèles de Prédiction de Sentiment et Pipeline MLOps

Ce projet consiste à développer des modèles de prédiction de sentiment à partir de données textuelles et à mettre en place un pipeline **MLOps** pour leur déploiement et gestion en production. 

## Objectifs du projet

- **Développement de Modèles IA** : Création de modèles de sentiment utilisant des méthodes comme la régression logistique, les réseaux de neurones, et des modèles avancés comme BERT.
- **Pipeline MLOps** : Mise en place d'un pipeline complet de déploiement continu avec tests unitaires, suivi des modèles, et gestion en production avec **MLFlow**.
- **Présentation des résultats** : Préparation de supports pour une audience non technique et rédaction d'un article de blog expliquant les résultats et la démarche.

## Technologies Utilisées

- **Langage** : Python
- **Bibliothèques IA** : scikit-learn, TensorFlow, Hugging Face
- **MLOps** : MLFlow, Docker, GitHub Actions, Prometheus & Grafana
- **API** : FastAPI pour exposer les modèles via une API REST

## Installation et Utilisation

Pour installer et utiliser ce projet, commencez par cloner le dépôt GitHub avec `git clone https://github.com/votre-utilisateur/sentiment-analysis-mlops.git`, puis accédez au répertoire du projet avec `cd sentiment-analysis-mlops`. Ensuite, créez un environnement virtuel avec `python3 -m venv venv` et activez-le. Sur Linux ou macOS, utilisez `source venv/bin/activate`, et sur Windows, exécutez `venv\Scripts\activate`. 

Une fois l'environnement activé, installez les dépendances nécessaires avec la commande `pip install -r requirements.txt`.

Pour le suivi des modèles avec MLFlow, lancez-le en exécutant `mlflow ui`. Si vous souhaitez entraîner un modèle, exécutez par exemple `python src/train_model.py --model logistic_regression --data_path ./data/dataset.csv` pour un modèle de régression logistique.

Après l'entraînement, vous pouvez déployer le modèle en production via une API avec FastAPI en exécutant `uvicorn src.app:app --reload`. Enfin, pour vérifier que tout fonctionne correctement, lancez les tests unitaires avec `pytest tests/`.

Pour surveiller les performances du modèle en production, veuillez suivre les instructions détaillées dans le fichier `monitoring/README.md`.
