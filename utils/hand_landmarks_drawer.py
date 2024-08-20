import os
import cv2
import mediapipe as mp
from mediapipe.python.solutions.hands import Hands
from tqdm import tqdm
from PIL import Image

mp_drawing = mp.solutions.drawing_utils

hands = Hands(static_image_mode=True, )

base_dir = './GestureDataset/ego_gesture/images'
dest_dir = './images_Drawed'

def draw_landmarks(base_dir, dest_dir):
    print(f"Elaborazione cartella: {base_dir}")
    print(f"Cartella di destinazione: {dest_dir}")

    for root, dirs, files in os.walk(base_dir):
        if(len(files)!=0):
            splitted_path = str(root).split("/")

            relative_path = os.path.relpath(root, base_dir)

            output_dir = os.path.join(dest_dir, relative_path)
            output_dir += '_Drawed'
            os.makedirs(output_dir, exist_ok=True)

            print(f"Current: {splitted_path[-4]}")

            
            if(splitted_path[-2] != "Depth"):
                print("Elaborating: "+root)
                for file in tqdm(files):
                    if file.endswith(".jpg"):
                        input_file = os.path.join(root, file)
                        output_file = os.path.join(output_dir, file)                    

                        image = cv2.imread(input_file)
                        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        
                        cv2.imwrite(output_file, image)
            else:
                print("Elaborating: "+root)
                for file in tqdm(files):
                    if file.endswith(".jpg"):
                        input_file = os.path.join(root, file)
                        output_file = os.path.join(output_dir, file)

                        im = Image.open(input_file)
                        im.save(output_file)


                



    hands.close()
    cv2.destroyAllWindows()

draw_landmarks(base_dir, dest_dir)


