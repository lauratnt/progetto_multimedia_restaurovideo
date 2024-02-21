import cv2
import os
from PIL import Image

import os

#pulisce la cartella prima dell'esecuzione del codice
def clear_directory(directory):
    if not os.path.exists(directory):
        print(f"La directory {directory} non esiste.")
        return
    
    files = os.listdir(directory)
    for file in files:
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    print(f"La directory {directory} è stata svuotata.")

clear_directory("provaim")

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore nell'apertura del video.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_path = os.path.join(output_folder, f"frame_{frame_count}.png") #salva il frame in formato .png
        cv2.imwrite(output_path, frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

def crop_images_in_directory(input_dir, output_dir, crop_amount):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            width, height = img.size

            cropped_img = img.crop((crop_amount, crop_amount, width - crop_amount, height - crop_amount)) #croppa l'immagine

            output_path = os.path.join(output_dir, filename)
            cropped_img.save(output_path)

            print(f"{filename} cropped and saved.")


#video da cui estrarre i frame
video_path = "nonproc.mp4"

#cartella dei frame
output_folder = "provaim"

extract_frames(video_path, output_folder)

output_directory = "cropped_images"

#pixel da croppare
crop_amount = 40

#salvataggio immagini croppate
crop_images_in_directory(output_folder, output_directory, crop_amount)
