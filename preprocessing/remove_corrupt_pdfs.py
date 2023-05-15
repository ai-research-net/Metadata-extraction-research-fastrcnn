import os 
import PyPDF2

i = 1
for file in os.scandir("../data/pdfs"):
    if file.name.endswith(".pdf"):
        try:
            with open(file.path, 'rb') as f:
                fileReader = PyPDF2.PdfFileReader(f)
                
        except:
            print("File removed",i)
            os.remove(file.path)
            i+=1

print(i-1)

