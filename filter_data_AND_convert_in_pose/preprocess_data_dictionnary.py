import json
import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from text_to_gloss_dir.rules import *

def convert_json_to_csv(input_json, output_csv):
    # Lire le fichier JSON
    with open(input_json, 'r') as file:
        data = json.load(file)

    # Ouvrir ou créer un fichier CSV pour l'écriture
    with open(output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Écrire l'en-tête du fichier CSV
        writer.writerow(['words', 'path', 'glosses'])

        # Parcourir chaque élément dans le JSON
        for item in data:
            words = item['key']
            video_path = item['video']
            glosses = text_to_gloss(words, "fr")

            # Changer l'extension de .webm à .pose
            new_video_path = os.path.splitext(video_path)[0] + '.pose'


            # Écrire la clé et le nouveau chemin de la vidéo dans le CSV
            if len(glosses) == 0:
                writer.writerow([words, new_video_path, words])
            else:
                writer.writerow([words, new_video_path, glosses[0][1]])


# Exemple d'utilisation :
input_json = 'converted_videos_mp4/vocabulaire.json'  # Le chemin vers ton fichier JSON
output_csv = 'pose_videos/vocabulary.csv'  # Le nom du fichier CSV de sortie
convert_json_to_csv(input_json, output_csv)
