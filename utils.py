import os
import zipfile

def unzip_and_delete(source_folder, output_folder):
    """
    Estrae tutti i file .zip dalla source_folder verso la output_folder
    e poi elimina i file .zip originali.
    """

    # 1. Controlla se la cartella sorgente esiste
    if not os.path.exists(source_folder):
        print(f"Errore: La cartella sorgente '{source_folder}' non esiste.")
        return

    # 2. Crea la cartella di destinazione se non esiste
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Creata cartella di destinazione: {output_folder}")
        except OSError as e:
            print(f"Errore nella creazione della cartella {output_folder}: {e}")
            return

    # 3. Itera sui file
    files_found = False
    for item in os.listdir(source_folder):
        if item.endswith(".zip"):
            files_found = True
            zip_path = os.path.join(source_folder, item)

            print(f"Sto estraendo: {item} -> in {output_folder}")

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Qui specifichiamo la cartella di output
                    zip_ref.extractall(output_folder)

                # Se l'estrazione va a buon fine, elimina lo zip originale
                print(f"Estrazione completata. Elimino {item}")
                os.remove(zip_path)

            except zipfile.BadZipFile:
                print(f"ERRORE: Il file {item} è corrotto.")
            except Exception as e:
                print(f"Errore generico con {item}: {e}")

    if not files_found:
        print("Nessun file .zip trovato nella cartella sorgente.")
    else:
        print("\nOperazione completata!")