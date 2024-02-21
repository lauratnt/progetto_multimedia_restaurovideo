#codice definitivo
#premo 'r' = reset
#premo 'i' = interpolazione manuale
#premo 'f' = interpolazione automatica con la maschera nel dominio delle frequenze (non funziona sempre)
#premo 'e' = esegue estrazione dei contorni e inspessimento (se serve)
#premo 'p' = processo l'immagine in corso
#tutte le immagini vengono inserite nella cartella di output 'img_finali'

import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# Fa il resize dell'immagine per migliorare la visibilità
def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 50)
    height = int(image.shape[0] * scale_percent / 50)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA) #interpola se non trova pixel
    return resized_image

selected_regions = []

#Seleziona regione con area del mouse per il riquadro
def select_region(event, x, y, flags, param):
    global selected_regions

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_regions.append([(x, y)])

    elif event == cv2.EVENT_LBUTTONUP:
        if len(selected_regions[-1]) == 1:
            selected_regions[-1].append((x, y))


# Funzione per elaborare un'immagine nel dominio delle frequenze con la maschera
def freq_image(image):
    # Converte in scala di grigi
    gray_image = np.mean(image, axis=2)

    #Calcola la trasformata di Fourier dell'immagine
    gray_image_freq = np.fft.fft2(gray_image)
    gray_image_freq_shifted = np.fft.fftshift(gray_image_freq) #sposta basse frequenze al centro dell'immagine

    # Applica un filtro notch per rimuovere il rumore
    notch_filter = np.ones_like(gray_image_freq_shifted)
    center_x, center_y = gray_image_freq_shifted.shape[0] // 2, gray_image_freq_shifted.shape[1] // 2
    notch_width = 11  # Larghezza del filtro
    notch_filter[center_x - notch_width:center_x + notch_width, center_y - notch_width:center_y + notch_width] = 0

    filtered_freq = gray_image_freq_shifted * notch_filter #immagine filtrata con il notch

    # Calcola l'inversa
    filtered_image_freq = np.fft.ifftshift(filtered_freq)
    filtered_image = np.fft.ifft2(filtered_image_freq)
    filtered_image = np.abs(filtered_image)

    # Normalizza tra 0 e 255
    normalized_filtered_image = (filtered_image - np.min(filtered_image)) / (np.max(filtered_image) - np.min(filtered_image)) * 255

    # Applica la sogliatura all'immagine filtrata- se cambio i parametri ho soglia più stretta o più alta
    thresholded_image = np.copy(normalized_filtered_image)
    thresholded_image[(thresholded_image < 50) | (thresholded_image > 190)] = 0
    thresholded_image[(thresholded_image >= 50) & (thresholded_image <= 190)] = 255

    # Crea una maschera delle aree bianche dalla soglia fatta
    mask = (thresholded_image == 255)

    # Interpolazione con Gaussiana per fondere le aree bianche con lo sfondo
    merged_image = image.copy()
    mask_smoothed = gaussian_filter(mask.astype(float), sigma=3)  # Regola sigma per il livello di smoothing

    # Trova gli indici dei pixel vuoti
    blank_indices = np.where(mask_smoothed > 0.1)

    # Riempie le aree vuote nell'immagine colorata usando l'interpolazione
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    x_blank, y_blank = x[blank_indices], y[blank_indices]
    x_valid, y_valid = x[~mask_smoothed.astype(bool)], y[~mask_smoothed.astype(bool)]
    for channel in range(3):
        filled_values = griddata((x_valid.ravel(), y_valid.ravel()), image[:,:,channel][~mask_smoothed.astype(bool)].ravel(), (x_blank.ravel(), y_blank.ravel()), method='linear')
        merged_image[:,:,channel][blank_indices] = filled_values

    return merged_image


def process_images(input_folder, output_folder):
    global selected_regions 

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            image = cv2.imread(filepath)
            clone = image.copy()

            scale_percent = 50
            image = resize_image(image, scale_percent)

            cv2.namedWindow('Select Region')
            cv2.setMouseCallback('Select Region', select_region)

            while True:
                image_show = image.copy()

                for region in selected_regions:
                    if len(region) == 2:
                        cv2.rectangle(image_show, region[0], region[1], (0, 255, 0), 2)

                cv2.imshow('Select Region', image_show)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('i'): # Esegui interpolazione se premo "i"
                    for region in selected_regions:
                        if len(region) == 2:
                            x1, y1 = region[0]
                            x2, y2 = region[1]

                            # Crea la maschera sull'area selezionata
                            mask = np.zeros_like(image, dtype=np.uint8)
                            cv2.rectangle(mask, region[0], region[1], (255, 255, 255), -1)

                            # Fai l'inpaint x interpolare con il metodo di navier stokes (c'è anche il telea ma non inpainta bene come questo)
                            interpolated = cv2.inpaint(image, mask[:, :, 0], inpaintRadius=3, flags=cv2.INPAINT_NS)

                            # Sostituisce l'area con quella interpolata
                            image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = interpolated[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]

                    selected_regions = []
                    cv2.imshow('Select Region', image)

                # Se premo r resetta le modifiche
                elif key == ord('r'):
                    image = clone.copy()
                    image = resize_image(image, scale_percent)
                    selected_regions = []
                    cv2.imshow('Select Region', image)

                # Applico filtro mediano con kernel 5x5 //posso cambiare dimensione del kernel senza problemi
                elif key == ord('m'):
                    image = cv2.medianBlur(image, 5)
                    cv2.imshow('Select Region', image)

                # Processa immagine sul dominio delle frequenze - non funziona bene su tutte le immagini
                elif key == ord('f'):
                    processed_image = freq_image(image)
                    cv2.imshow('Processed Image', processed_image)

                elif key == ord('e'): # Estrae i contorni e li inspessisce
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    thickened_contours = np.zeros_like(image)
                    for contour in contours:
                        cv2.drawContours(thickened_contours, [cv2.approxPolyDP(contour, 3, True)], -1, (255, 255, 255), 3)

                    image = cv2.addWeighted(image, 0.7, thickened_contours, 0.3, 0)

                    cv2.imshow('Select Region', image)

                elif key == ord('p'):
                    break
                #processo l'immagine attuale

            cv2.destroyAllWindows()

            output_filepath = os.path.join(output_folder, f"{filename}")
            cv2.imwrite(output_filepath, image)

input_folder = 'cropped_images/' #prende le immagini preprocessate
output_folder = 'img_finali/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

process_images(input_folder, output_folder)
