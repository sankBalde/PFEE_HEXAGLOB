import cv2
import mediapipe as mp

# Initialiser MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Indices des landmarks à extraire
FACE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155,  # Oeil gauche
                  362, 263, 387, 386, 385, 384, 373, 380, 381, 382, 383, 362,  # Oeil droit
                  70, 63, 105, 66, 107, 55, 46, 53, 52,  # Sourcil gauche
                  336, 296, 334, 293, 300, 285, 276, 283, 282,  # Sourcil droit
                  61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 375, 321, 405, 314, 17, 84, 181, 91, 146,  # Bouche
                  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]  # Contour du visage


# Fonction pour extraire les landmarks
def extract_landmarks(results):
    landmarks = {
        "face": [],
        "left_hand": [],
        "right_hand": [],
        "body": []
    }

    # Extraire les landmarks du visage
    if results.face_landmarks:
        landmarks["face"] = [(results.face_landmarks.landmark[i].x,
                              results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z) for i in FACE_LANDMARKS]

    # Extraire les landmarks des mains
    if results.left_hand_landmarks:
        landmarks["left_hand"] = [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark]

    if results.right_hand_landmarks:
        landmarks["right_hand"] = [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark]

    ignore_list = [8, 6, 5, 4, 1, 2, 3, 7, 0, 10, 9]
    # Extraire les landmarks du corps (pose) en ignorant les indices dans ignore_list
    if results.pose_landmarks:
        landmarks["body"] = [
            (lm.x, lm.y, lm.z) for idx, lm in enumerate(results.pose_landmarks.landmark) if idx not in ignore_list
        ]

    return landmarks


# Fonction principale pour capturer les landmarks à partir d'un fichier vidéo
def get_landmarks_from_video(video_path):
    # Ouvrir le fichier vidéo
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []

    # Initialiser l'objet Holistic de MediaPipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Convertir l'image en RGB car MediaPipe utilise des images RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Traiter l'image pour obtenir les landmarks
            results = holistic.process(image_rgb)

            # Revenir à l'image en BGR pour l'affichage avec OpenCV
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Extraire les landmarks
            landmarks = extract_landmarks(results)
            landmarks_list.append(landmarks)

            # Afficher les landmarks du visage simplifiés
            for point in landmarks["face"]:
                x = int(point[0] * frame.shape[1])
                y = int(point[1] * frame.shape[0])
                cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)

            # Afficher les landmarks du corps simplifiés
            for point in landmarks["body"]:
                x = int(point[0] * frame.shape[1])
                y = int(point[1] * frame.shape[0])
                cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)

            # Afficher les landmarks des mains simplifiés
            for point in landmarks["left_hand"]:
                x = int(point[0] * frame.shape[1])
                y = int(point[1] * frame.shape[0])
                cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)

            for point in landmarks["right_hand"]:
                x = int(point[0] * frame.shape[1])
                y = int(point[1] * frame.shape[0])
                cv2.circle(image_bgr, (x, y), 3, (0, 255, 0), -1)

            # Afficher l'image avec les landmarks
            cv2.imshow('Holistic Landmarks', image_bgr)

            # Quitter si la touche 'q' est pressée
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Libérer la capture et fermer les fenêtres
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "tmp/bonjour-3.mp4"
    get_landmarks_from_video(video_path)
