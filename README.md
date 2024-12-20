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
- Lancer la commande `python filter_data_AND_convert_in_pose/video_pose.py --format mediapipe -i converted_videos_mp4/elix/achat.mp4 -o achat.pose` pour convertir du .mp4 en .pose
- Ecriture des mots avec leur path pose et gloss respectifs dans un csv

## 3. Concatenation de Pose
- Voir le dossier `gloss_to_pose`
  - Un dossier test de poses existe: `pose_test`
  - Pour lancer le test: `python main.py`
  - Pour visualiser le test: `python ../filter_data_AND_convert_in_pose/pose_visualisation.py -i ecole_aller.pose -o ecole_aller.mp4`


