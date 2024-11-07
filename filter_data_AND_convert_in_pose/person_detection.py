import cv2
def detection_person_frame(model, frame):
    # Charger l'image
    try:
        H, W, _ = frame.shape
        print (frame.shape)
    except AttributeError:
        # Fermer toutes les fenêtres OpenCV
        cv2.destroyAllWindows()
        return frame

    # Effectuer la détection sur l'image
    results = model(frame)[0]

    # Seuil de confiance pour les détections
    threshold = 0.7
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        name_class = results.names[int(class_id)].upper()
        if (score > threshold) and (name_class == 'PERSON'):

            extract_object_frame = frame[int(y1):int(y2), int(x1):int(x2)]
            print("return the exactframe", x1, y1, x2, y2)

            return extract_object_frame

    # Fermer toutes les fenêtres OpenCV
    cv2.destroyAllWindows()
    return frame

