#post processing delle immagini e componimento del video
import cv2
import os

#processa tutte le immagini nella cartella
def restore_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        restored = restore_image(image)

        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, restored)

        print(f"Immagine ripristinata salvata: {output_path}")

#applica i filtri per restaurare l'immagine
def restore_image(image):
    # Applico il filtro bilaterale per ridurre il rumore mantenendo i dettagli
    denoised = cv2.bilateralFilter(image, 9, 75, 75)
    # Applico la correzione del contrasto per migliorare la nitidezza
    sharpened = cv2.addWeighted(denoised, 1.5, cv2.GaussianBlur(denoised, (0, 0), 10), -0.5, 0)
    # Applico il filtro di sharpening per accentuare i contorni
    restored = cv2.addWeighted(image, 1.5, sharpened, -0.5, 0)
    return restored

def create_video_from_restored_images(input_folder, output_video_path, fps=24):
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("Nessuna immagine trovata nella cartella di input.")
        return

    image_files.sort() #non si sa mai anche se dovrebbero essere ordinate

    sample_image = cv2.imread(os.path.join(input_folder, image_files[0]))
    height, width, _ = sample_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    #creazione video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    out.release()

    print(f"Video creato con successo: {output_video_path}")


input_folder = 'img_finali/'
output_folder_restored = 'output_vidfin_restored/'
output_video_path = 'output_video_restored.mp4'


restore_images(input_folder, output_folder_restored)

create_video_from_restored_images(output_folder_restored, output_video_path)
