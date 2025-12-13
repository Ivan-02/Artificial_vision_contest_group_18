from utils import *
from ultralytics import YOLO
import cv2 as cv

if __name__ == '__main__':
    # Cartella DOVE SI TROVANO ora i file zip scaricati
    cartella_con_zip = "./path/to/SoccerNet/tracking-2023"

    # Cartella DOVE VUOI METTERE i file estratti (Dataset pronto)
    cartella_destinazione = "./dataset/"

    # Avvia lo script
    unzip_and_delete(cartella_con_zip, cartella_destinazione)

    train_input_root = "./dataset/train"
    test_input_root = "./dataset/test"

    train_output_root = "./dataset/yolo/"
    test_output_root = "./dataset/yolo/"

    convert_mot_to_yolo(train_input_root, train_output_root, target_class_id=0, sub_folder='train')
    convert_mot_to_yolo(test_input_root, test_output_root, target_class_id=0, sub_folder='test')

    model = YOLO("yolo11m.pt")

    for image in os.listdir('./dataset/yolo/test/images'):

        if image.endswith('.jpg'):

            image_path = os.path.join('./dataset/yolo/test/images', image)

            results = model.predict(image_path)

            cv.imshow('ciao', results[0].plot())
            if cv.waitKey(0) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()

    """yaml_file = 'data.yaml'
    data_per_yaml = {
        'train': output_train_images_path,
        'val': output_validation_images_path,
        'test': output_test_images_path,
        'nc': num_classes,
        'names': class_list
    }


    try:
        with open(yaml_file, 'w') as f:
            yaml.dump(data_per_yaml, f, sort_keys= False, default_flow_style=False)

    except Exception as e:
        print(e)"""
