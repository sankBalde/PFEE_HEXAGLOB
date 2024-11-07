import cv2
from ultralytics import YOLO

# Charger le modèle YOLOv8 (assurez-vous d'avoir le modèle YOLOv8 pré-entrainé, sinon téléchargez-le)
model = YOLO('yolov8n.pt')  # ou yolov8s.pt pour une version plus grande

# Charger la vidéo d'entrée
input_video_path = 'tmp/bonjour-3.mp4'
output_video_path = 'output_video.mp4'

cap = cv2.VideoCapture(input_video_path)

# Obtenir les propriétés de la vidéo d'entrée
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Définir le codec et le writer pour la vidéo de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer la détection avec YOLOv8
    results = model(frame)

    # Annoter le frame avec les détections
    annotated_frame = results[0].plot()

    # Écrire le frame annoté dans la vidéo de sortie
    out.write(annotated_frame)

    # Afficher les résultats en temps réel (optionnel)
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
out.release()
cv2.destroyAllWindows()
