import cv2
import mediapipe as mp
import os
import json

def save_mediapipe_pose_from_video(video_path: str, output_directory: str):
    """
    Process a video file and save the face, pose, and hand landmarks as JSON files for each frame.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_directory : str
        Directory where the landmark data will be saved.
    """
    # Initialize Mediapipe Holistic model
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    FACE_LANDMARKS = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155, # Oeil gauche
                        362, 263, 387, 386, 385, 384, 373, 380, 381, 382, 383, 362,  # Oeil droit
                            70, 63, 105, 66, 107, 55, 46, 53, 52,  # Sourcil gauche
                            336, 296, 334, 293, 300, 285, 276, 283, 282,  # Sourcil droit
                            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78, 95, 88, 178,
                  87, 14, 317, 402, 318, 324, 308, 415, 375, 321, 405, 314, 17, 84, 181, 91, 146] # Bouche
    FACEMESH_CONTOURS_POINTS = [
        p for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))
    ]
    for i in FACEMESH_CONTOURS_POINTS:
        FACE_LANDMARKS.append(i)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Finished processing video.")
            break

        # Convert the image from BGR to RGB for Mediapipe processing
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # Improves performance

        # Process the frame with Mediapipe Holistic
        results = holistic.process(image_rgb)

        # Only save if landmarks for face, pose, and hands are detected
        if results.face_landmarks and results.pose_landmarks and results.left_hand_landmarks and results.right_hand_landmarks:
            # Extract landmarks for face, pose, and hands

            face_landmarks = [(results.face_landmarks.landmark[i].x,
                                           results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z)
                                          for i in FACE_LANDMARKS]
            #face_landmarks = [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
            pose_landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            left_hand_landmarks = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            right_hand_landmarks = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]

            # Save the landmarks for the current frame
            frame_data = {
                "face_landmarks": {
                    "num_landmarks": len(face_landmarks),
                    "landmarks": [",".join(map(str, lm)) for lm in face_landmarks]
                },
                "pose_landmarks": {
                    "num_landmarks": len(pose_landmarks),
                    "landmarks": [",".join(map(str, lm)) for lm in pose_landmarks]
                },
                "left_hand_landmarks": {
                    "num_landmarks": len(left_hand_landmarks),
                    "landmarks": [",".join(map(str, lm)) for lm in left_hand_landmarks]
                },
                "right_hand_landmarks": {
                    "num_landmarks": len(right_hand_landmarks),
                    "landmarks": [",".join(map(str, lm)) for lm in right_hand_landmarks]
                }
            }

            # Save the frame data as a JSON file
            output_path = os.path.join(output_directory, f"{frame_id:04d}.json")
            with open(output_path, 'w') as f:
                json.dump(frame_data, f, indent=4)

            frame_id += 1
            print(f"Processed frame {frame_id}")

    # Release the video capture and Mediapipe resources
    cap.release()
    holistic.close()

    print("All frames processed and landmarks saved.")

# Example usage:
save_mediapipe_pose_from_video("tmp/bonjour-3.mp4", "bonjour_output_landmarks")