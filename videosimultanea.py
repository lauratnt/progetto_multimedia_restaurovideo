import cv2

# Fa la resize dei video se non fossero della stessa dimensione
def resize_video(video, target_width, target_height):
    ret, frame = video.read()
    if not ret:
        return None
    frame_resized = cv2.resize(frame, (target_width, target_height))

    return frame_resized

def play_side_by_side(video1_path, video2_path, target_width, target_height):
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    if not video1.isOpened() or not video2.isOpened():
        print("Impossibile aprire i video")
        return

    video1_resized = cv2.VideoCapture(video1_path)
    video2_resized = cv2.VideoCapture(video2_path)

    paused = False  # per mettere in pausa

    while True:
        if paused:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)

        # Premo 'p' per cambiare stato di riproduzione del video
        if key & 0xFF == ord('p'):
            paused = not paused

        # Chiudo file
        if key & 0xFF == ord('q'):
            break

        if not paused:
            ret1, frame1_resized = video1_resized.read()
            ret2, frame2_resized = video2_resized.read()
            if not ret1 or not ret2:
                break

            frame1_resized = resize_video(video1, target_width, target_height)
            frame2_resized = resize_video(video2, target_width, target_height)

            # Unisce frame orizzontalmente
            side_by_side = cv2.hconcat([frame1_resized, frame2_resized])

            cv2.imshow('Video Originale | Video Processato', side_by_side)

    video1.release()
    video2.release()
    cv2.destroyAllWindows()


video1_path = 'nonproc.mp4'
video2_path = 'rest.mp4'
target_width = 430  # Cambio parametri di resize
target_height = 280  
play_side_by_side(video1_path, video2_path, target_width, target_height)
