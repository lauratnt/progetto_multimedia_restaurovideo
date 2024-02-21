# progetto_multimedia_restaurovideo
spiegazione in breve del codice:
    Estrazione dei Frame da un Video e Cropping:
        In questa sezione, vengono estratti i frame da un video e salvati come immagini PNG. Successivamente, queste immagini vengono sottoposte a un processo di ritaglio per rimuovere parti indesiderate.

    Elaborazione delle Immagini e Interpolazione:
        Qui vengono implementate diverse funzionalità per l'elaborazione delle immagini, tra cui l'interpolazione, l'applicazione di filtri e l'elaborazione nel dominio delle frequenze. Viene data all'utente la possibilità di selezionare manualmente delle regioni specifiche sull'immagine per l'interpolazione. Le altre opzioni consentono di ripristinare l'immagine allo stato originale ("r"), applicare un filtro mediano ("m"), eseguire l'elaborazione delle immagini nel dominio delle frequenze ("f"), estrarre i contorni e ispessirli ("e"), e infine, processare l'immagine attuale ("p").

    Post-processing e Creazione di un Video:
        Infine, le immagini elaborate vengono sottoposte a un processo di restauro tramite filtri bilaterali, correzione del contrasto e sharpening. Le immagini restaurate vengono utilizzate per creare un nuovo video.

In sintesi, il codice prende un video in input, ne estrae i frame, esegue diverse operazioni di elaborazione e manipolazione delle immagini su questi frame, e infine crea un nuovo video a partire dalle immagini elaborate. Durante l'elaborazione delle immagini, viene data all'utente la possibilità di interagire e controllare il processo attraverso l'interfaccia grafica, consentendo di selezionare manualmente le regioni di interesse sull'immagine e di eseguire diverse operazioni di elaborazione in base alle proprie esigenze.
