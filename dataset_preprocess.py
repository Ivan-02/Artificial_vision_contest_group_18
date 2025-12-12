from utils import unzip_and_delete

if __name__ == '__main__':
    # Cartella DOVE SI TROVANO ora i file zip scaricati
    cartella_con_zip = "./path/to/SoccerNet/tracking-2023"

    # Cartella DOVE VUOI METTERE i file estratti (Dataset pronto)
    cartella_destinazione = "./dataset/"

    # Avvia lo script
    unzip_and_delete(cartella_con_zip, cartella_destinazione)