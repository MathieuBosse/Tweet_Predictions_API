# Étape 1 : Utiliser une image de base Python
FROM python:3.12

# Étape 2 : Définir le répertoire de travail à l'intérieur du conteneur
WORKDIR /app

# Étape 3 : Copier le fichier requirements.txt pour installer les dépendances
COPY requirements.txt ./

# Étape 4 : Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 5 : Copier tout le code source dans le conteneur
COPY src/ /app/src/
COPY models/ /app/models/
COPY tests/ /app/tests/


# Étape 6 : Exposer le port sur lequel Flask écoute (8080)
EXPOSE 8080

# Étape 7 : Définir la commande pour exécuter l'API Flask
CMD ["python", "src/main.py"]

