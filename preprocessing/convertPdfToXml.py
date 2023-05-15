import cv2
from tesseract2dict import TessToDict
from pdf2image import convert_from_path


def convertPdfToXml(fileName):
    pages = convert_from_path(fileName)
    newFileName = fileName[:-4] + '.jpg'
    for page in pages:
        page.save(newFileName, "JPEG")
        break


    td=TessToDict()

    inputImage=cv2.imread(newFileName)
    # Convert PDF to XML and output the file under the folder "outfolder" => ./outfolder/out.hocr
    word_dict=td.tess2dict(inputImage,'out','outfolder')
    del td
    del inputImage
    del word_dict
