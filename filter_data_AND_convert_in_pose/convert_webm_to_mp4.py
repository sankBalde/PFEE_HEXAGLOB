import cv2
import os


def convert_webm_to_mp4(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.webm'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.mp4'
            output_path = os.path.join(output_folder, output_filename)

            cap = cv2.VideoCapture(input_path)


            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break


                out.write(frame)

            cap.release()
            out.release()

    cv2.destroyAllWindows()


# Exemple d'utilisation :
input_folder = 'videos_webm'
output_folder = 'converted_videos_mp4'
for doc in os.listdir(input_folder):
    output_folder_mp4 = output_folder + '/' + doc
    input_folder_mp4 = input_folder + '/' + doc
    convert_webm_to_mp4(input_folder_mp4, output_folder_mp4)
