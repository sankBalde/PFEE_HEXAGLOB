# PFEE_HEXAGLOB

### Prérequis
- `Python 3.7+`
- `spaCy 3.x`
- Modèles pré-entraînés de `spaCy` (par exemple, `fr_core_news_lg` pour le français)
- `pose-format` pour la manipulation de Pose

## 1. Transformateur Texte-vers-Glossaire

### Fonctionnalités
- Réorganisation des clauses et des triplets en fonction de la structure des phrases.
- Règles personnalisées de glossification pour les verbes, pronoms, négations et mots de localisation.


## 2. Transformateur de Videos .webm en .pose
- Les videos sont prises du dataset : `https://github.com/parlr/lsf-data/tree/master`
- Transformation des videos .webm en .mp4
- Lancer la commande `videos_to_poses --format mediapipe --directory /path/to/videos` pour convertir du .mp4 en .pose
- Ecriture des mots avec leur path pose et gloss respectifs dans un csv


