"""Entraînement des modèles de prévision de consommation.

`dataset` construit le jeu d'apprentissage depuis la base, `train` entraîne un
modèle et enregistre le run dans MLflow. Le service d'inférence ne rejoue
jamais ce code : il charge un modèle déjà enregistré, désigné par son tag.
"""
