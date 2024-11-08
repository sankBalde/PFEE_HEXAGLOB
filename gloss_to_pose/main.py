from lookup import PoseLookup
from concatenate import concatenate_poses

# Supposons que vous ayez une liste de données pour initialiser `rows`
rows = [
    {
        "words": "bonjour",
        "glosses": "bonjour",
        "spoken_language": "fr",
        "signed_language": "fr",
        "path": "bonjour.pose",
        "start": 0,
        "end": 10
    },
    {
        "words": "aller",
        "glosses": "aller",
        "spoken_language": "fr",
        "signed_language": "fr",
        "path": "aller.pose",
        "start": 0,
        "end": 10
    },
    {
        "words": "école",
        "glosses": "école",
        "spoken_language": "fr",
        "signed_language": "fr",
        "path": "ecole.pose",
        "start": 0,
        "end": 10
    },
    {
        "words": "je",
        "glosses": "je",
        "spoken_language": "fr",
        "signed_language": "fr",
        "path": "aller.pose",
        "start": 0,
        "end": 10
    },
    # Ajoutez d'autres éléments si nécessaire
]

# Spécifiez le répertoire où les fichiers de pose sont stockés
directory = "pose_test"

# Instancier la classe PoseLookup
pose_lookup = PoseLookup(rows=rows, directory=directory)
output_path = "ecole_aller.pose"
try:
    pose_ecole = pose_lookup.read_pose("ecole.pose")
    pose_aller = pose_lookup.read_pose("aller.pose")

    # Concaténer les poses
    concatenated_pose = concatenate_poses([pose_ecole, pose_aller])
    print("Pose concaténée avec succès :", concatenated_pose)
    print('Saving to disk ...')
    with open(output_path, "wb") as f:
        concatenated_pose.write(f)

except Exception as e:
    print("Erreur lors du chargement de la pose :", e)

