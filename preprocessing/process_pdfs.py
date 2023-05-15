from convertPdfToXml import convertPdfToXml
from parseDocument import parseDocument
from similarity import similarity
import pandas as pd
import gc
import time

metadata = pd.read_csv('../data/metadata/metadata.csv')
metadata = metadata.fillna('')

i = 0
for index, row in metadata.iterrows():
    if index < 0:
        continue
    else:
        print("File {}".format(i))
        i+=1
        file_name = row['file_name']
        try:
            print(file_name)
            convertPdfToXml(file_name)
            parseDocument()
            similarity(row, file_name[:-4]+'.json')
            gc.collect()
            time.sleep(3)
        except:
            print("Error with the file {}".format(file_name))
