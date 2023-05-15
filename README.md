# Metadata-extraction-research
This main project contains the research pipeline, it contains: the data scrapping, preprocessing and a Fast R-CNN model.


## Folder structure

```
Metadata-extraction-research
│   README.md      
└───webscraping 
│   │   scrape_web_page.py
│   
└───preprocessing
│   │   convertPdfToXml.py
│   │   convert_to_gray.py
│   │   create_text_map.py
│   │   parseDocument.py
│   │   process_pdfs.py
│   │   remove_corrupt_pdfs.py
│   │   similarity.py
│
└───model
│   │   fast_rcnn_model.py
└───data
    └───images
    └───metadata
    │   metadata.csv
    └───pdfs
```



## Execution steps
Before the execution, the data folder and subfolders need to be created accorting to the folder structure in the previous section.

* First step is to execute the **scrape_web_page.py** file to get the data from the website [website](https://www.ssoar.info/ssoar/handle/community/10000/discover?rpp=10&etal=0&group_by=none&page=), the script works on the current html structure of the website, therefore, if the website design and structure changes, a newer data scrapping script need to be implimented.
* The file **metadata.csv** need to be created empty with the headers, an example is shown in the file.
* The file **remove_corrupt_pdfs.py** needs to be run, to remove the corrupted pdfs from the dataset.
* The file **process_pdfs.py** needs to be run, it will automatically call other scripts in the preprocessing folder.
* The file **create_text_map.py** needs to be run to create the text maps.
* The file **convert_to_gray.py** needs to be run.
* The file **fast_rcnn_model.py** needs to be run.


**create_text_map.py** and **convert_to_gray.py** could be run at the same time as they don't overlap.

