import xml.etree.ElementTree as ET

def parseDocument():
    mytree = ET.parse("./outfolder/out.hocr")
    root = mytree.getroot()
    import pickle5 as pickle

    content = []

    for child in root:
        if child.tag == "{http://www.w3.org/1999/xhtml}body":
            
            headers = child.findall(".//{http://www.w3.org/1999/xhtml}span[@class='ocr_header']")
            for header in headers:
                header_feature = []

                header_text = ''
                header_position = header.attrib['title'].split()[1:5]
                header_baselines = header.attrib['title'].split()[6:8]
                header_size = header.attrib['title'].split()[9:10]
                
                header_size = [float(x.replace(';', '')) for x in header_size]
                header_baselines = [float(x.replace(";", '')) for x in header_baselines]
                header_position = [float(x.replace(";", '')) for x in header_position]
                
                for word in header:
                    header_text += word.text + ' '
                header_feature.append(header_text.strip())
                header_feature.append(header_position)
                header_feature.append(header_baselines)
                header_feature.append(header_size)

                content.append(header_feature)

            pars = child.findall('.//{http://www.w3.org/1999/xhtml}p[@class="ocr_par"]')
            for par in pars:
                paragraph_feature = []

                paragraph_text = ''
                paragraph_position = par.attrib['title'].split()[1:5]
                paragraph_position = [float(x.replace(";", "")) for x in paragraph_position]

                words = par.findall(".//{http://www.w3.org/1999/xhtml}span[@class='ocrx_word']")
                for word in words:
                    paragraph_text += word.text + ' '

                paragraph_feature.append(paragraph_text.strip())
                paragraph_feature.append(paragraph_position)

                content.append(paragraph_feature)
            
        
    with open("./doc_content.pkl", "wb") as f:
        pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)