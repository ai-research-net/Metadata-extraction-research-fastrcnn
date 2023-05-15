import json
import numpy as np
import cv2
import os

counter = 0
for file in os.scandir("../data/pdfs"):
    if file.name.endswith(".jpg"):
        #json_file = file.name[:-4] + ".json"
        #with open("../data/pdfs/" +json_file) as f:
            #sections = json.load(f)
            #for section in sections:
        
                #if not isinstance(section[-1], str):
                        print(file.name)
                    #try:
                        #cor = section[1]
                        img = cv2.imread(file.path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        #crop_img = img[int(cor[1]):int(cor[3]), int(cor[0]):int(cor[2])]
                        crop_img = img
                        image_name = f"../data/images/{file.name}.jpg"
                        cv2.imwrite(image_name, crop_img)
                        counter +=1
                    #except:
                    #    print("Error")
                