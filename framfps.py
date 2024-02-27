import cv2

def compose_video(input_video_path, output_video_path, fps=25):
    # Apre il video sorgente
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Errore nell'apertura del video sorgente.")
        return

    # Ottiene le informazioni sul video sorgente
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definisce il codec e crea l'oggetto VideoWriter per il video di output
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Scrive il frame nel video di output
        out.write(frame)
        frame_number += 1

        # Stampa il progresso
        print(f"Frame {frame_number}/{total_frames} elaborati.")

    # Rilascia le risorse
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Composizione del video completata.")

# Utilizzo:
input_video_path = 'video1.mp4'
output_video_path = 'video11.mp4'
compose_video(input_video_path, output_video_path)
