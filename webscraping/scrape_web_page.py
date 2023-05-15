import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrapWebpage(url):
    # Get the webpage HTML response
    response = requests.get(url)
    
    # Parse the webpage into a BeautifulSoup object
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the list of documents
    documents = soup.find_all('div',class_='artifact-description')

    # Create dataframe columns
    df = pd.read_csv('../data/metadata/metadata.csv')
    i = 0
    for document in documents:
        i +=1
        print("Document {}".format(i))
        link = document.find('a')
        link = "https://www.ssoar.info" + link.get('href')

        newDocumentResponse = requests.get(link)

        documentPage = BeautifulSoup(newDocumentResponse.text, 'html.parser')
        try: 
            title = documentPage.find('h2', class_='page-header').text
        except:
            title = ''
        
        
        try:
            authors = documentPage.find("div", class_="simple-item-view-authors").text.strip()
        except:
            authors= ''

        try:
            affiliation = documentPage.find("span", class_="resourceDetailTableCellValue").text.strip()
        except:
            affiliation = ''
        try:
            abstract = documentPage.find("p", class_="abstract_long").text.strip().replace("... view less", '')
        except:
            abstract = ''
        
        language = ''
        classification = ''
        year = ''
        address = ''
        keywords = ''
        series = ''
        issn = ''
        urn = ''

        ps = documentPage.find_all("span", class_="resourceDetailTableCellLabel")
        for p in ps:
            if "language" in p.text:
                language = p.find_next_sibling("span").text.strip()
            if "Classification" in p.text:
                classification = p.find_next_sibling("span").text.strip().split("\n")
                classification = [c for c in classification if c != ""]
            if "Year" in p.text:
                year = p.find_next_sibling("span").text.strip()
            if "City" in p.text:
                address = p.find_next_sibling("span").text.strip()
            if "Keywords" in p.text:
                keywords = p.find_next_sibling("span").text.strip().split("\n")
                keywords = [k for k in keywords if k != ""]
            if "Series" in p.text:
                series = p.find_next_sibling("span").text.strip()
            if "ISSN" in p.text:
                issn = p.find_next_sibling("span").text.strip()

        for h in documentPage.find_all("h5", string="Citation Suggestion"):
            urn = h.find_next_sibling("p").findChildren("a")[0].text
        
        fileName = title + ".pdf"
        fileName = fileName.replace("/", "_")
        fileName = fileName.replace(":", "_")
        fileName = "../data/pdfs/" + fileName[0:250]
        # Download the file
        try:
            documentPdfLink = documentPage.find("div", id="file-section-entry").findChildren("a")[0].get('href')
            downloadFile(documentPdfLink, fileName)
        except:
            print("No pdf file found")
            
        try:
            # Remove the first page
            removeFirstPagePDF(fileName)
        except:
            print("Could not remove first page")

        # Write the data to a csv file
        row = [fileName, authors, title, urn, address, affiliation, "", series, abstract, year, issn, language, classification, keywords]
        df.loc[len(df)] = row
        df.to_csv("../data/metadata/metadata.csv", index=False)
    return True

def downloadFile(url, filename):
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as out_file:
        out_file.write(response.content)
    return True

def removeFirstPagePDF(filename):
    from PyPDF2 import PdfFileReader, PdfFileWriter
    pdf = PdfFileReader(filename, 'rb')
    output = PdfFileWriter()
    for i in range(1, pdf.getNumPages()):
        output.addPage(pdf.getPage(i))
    with open("../data/pdfs/" + filename.split("/")[-1], "wb") as outputStream:
        output.write(outputStream)
    return True

page = 1
for i in range(1):
    link = "https://www.ssoar.info/ssoar/handle/community/10000/discover?rpp=10&etal=0&group_by=none&page="+str(page)
    scrapWebpage(link)
    page += 1

