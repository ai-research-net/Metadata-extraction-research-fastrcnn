import math
import re
from collections import Counter
import pickle5 as pickle
import pandas as pd
import numpy as np
import json

def similarity(labels, fileName):
    WORD = re.compile(r"\w+")

    def get_cosine(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator


    def text_to_vector(text):
        words = WORD.findall(text)
        return Counter(words)


    with open("./doc_content.pkl", "rb") as f:
        content = pickle.load(f)


    author = labels[1]
    title = labels[2]
    doi =  labels[3]
    address = labels[4]
    affiliation = labels[5]
    email = labels[6]
    journal = labels[7]
    abstract = labels[8]
    date = labels[9]

    for i in range(len(content)):
        part = content[i]
        # Similarity between part and author
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(author).lower())
        cosine = get_cosine(vector1, vector2)
     
        if (cosine > 0.60):
            part_label = "author"
            content[i].append("author")
            continue

        # Similarity between part and title
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(title).lower())
        cosine = get_cosine(vector1, vector2)


        if (cosine > 0.60):
            part_label = "title"
            content[i].append("title")

            continue
        
        # Similarity between part and DOI
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(doi).lower())
        cosine = get_cosine(vector1, vector2)

        if (cosine > 0.60):
            part_label = "doi"
            content[i].append("doi")

            continue

        # Similarity between part and address
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(address).lower())
        cosine = get_cosine(vector1, vector2)

        if (cosine > 0.60):
            part_label = "address"
            content[i].append("address")
            continue

        # Similarity between part and affiliation
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(affiliation).lower())
        cosine = get_cosine(vector1, vector2)

        if (cosine > 0.60):
            part_label = "affiliation"
            content[i].append("affiliation")
            continue

        # Similarity between part and email
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(email).lower())
        cosine = get_cosine(vector1, vector2)

        if (cosine > 0.60):
            part_label = "email"
            content[i].append("email")
            continue
        
        # Similarity between part and journal
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(journal).lower())
        cosine = get_cosine(vector1, vector2)

        if (cosine > 0.60):
            part_label = "journal"
            content[i].append("journal")
            continue

        # Similarity between part and abstract
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(abstract).lower())
        cosine = get_cosine(vector1, vector2)

        if (cosine > 0.60):
            part_label = "abstract"
            content[i].append("abstract")
            continue
        
        # Similarity between part and date
        vector1 = text_to_vector(str(part[0]).lower())
        vector2 = text_to_vector(str(date).lower())
        cosine = get_cosine(vector1, vector2)

        if (cosine > 0.50):
            part_label = "date"
            content[i].append("date")
            continue
            

    # Create JSON file for the document
    json_str = json.dumps(content, ensure_ascii=False)

    with open(fileName, "w") as f:
        f.write(json_str)
