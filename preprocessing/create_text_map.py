import torch
import os
import pickle5 as pickle
import json
from PIL import Image
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from transformers import AutoModel, AutoTokenizer
import scipy.interpolate as interp
import numpy as np
from random import randrange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

for file in os.scandir("../data/pdfs/"):
    if (file.name.endswith(".jpg")):
        image = Image.open(file.path)
        w, h = image.size
        image = image.resize((451, 679))
        x_resize = w / 451
        y_resize = h / 679
        
        with open(file.path.replace('.jpg', '.json'), 'r') as f:
            data = json.load(f)
        
        final_metadata = []
        for metadata in data:
            if isinstance(metadata[-1], str):
                #print("OLD Y1", metadata[1][1])
                metadata[1][0] = int(metadata[1][0] / x_resize)
                metadata[1][1] = int(metadata[1][1] / y_resize)
                metadata[1][2] = int(metadata[1][2] / x_resize)
                metadata[1][3] = int(metadata[1][3] / y_resize)
                #print("New Y1", metadata[1][1])
                final_metadata.append(metadata)
        if len(final_metadata) == 0:
            continue
        #print("Final metadata:", final_metadata)
        tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        model = AutoModel.from_pretrained('bert-base-german-cased', output_hidden_states=True)
        max_len = 1080
        text_map = np.zeros((451, 679))

        for metadata in final_metadata:
            sentence = "[CLS] " + " ".join([txt + ". [SEP]" for txt in metadata[0].split(".")]) + " [SEP]"
            #sentence = metadata[0]
            try:
                tokenized_text = tokenizer.tokenize(sentence)
                masked_index = randrange(0, len(tokenized_text) - 1)
                tokenized_text[masked_index] = '[MASK]'
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                #print(tokenized_text)
                current_seg = 1
                segment_ids = []
                for token in tokenized_text:
                    segment_ids.append(current_seg)
                    if (token == '[SEP]'):
                        current_seg+=1

                tokens_tensor = torch.tensor([indexed_tokens])
                segments_tensors = torch.tensor([segment_ids])

                with torch.no_grad():
                    embedding = model(tokens_tensor, segments_tensors)
            except:
                print("Error with token size")
                break
            hidden_states = embedding[2]

            #token_embeddings = torch.stack(hidden_states, dim=0)
            #token_embeddings = torch.squeeze(token_embeddings, dim=1)
            #token_embeddings = token_embeddings.permute(1, 0, 2)

            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)

            x1 = int(metadata[1][0])
            y1 = metadata[1][1]
            x2 = int(metadata[1][2])
            y2 = int(metadata[1][3])
            
            if (y1 > 679) or (y2>679) or (x1>451) or (x2>451):
                continue
            #print(metadata[1])
            #try:
            #print(sentence_embedding)
            #try:
            sentence_embg_i = 0
            for i in range(x1, x2):
                for j in range(y1, y2):
                    if sentence_embg_i >= sentence_embedding.size()[0]:
                        sentence_embg_i = 0
                    text_map[i, j] = sentence_embedding[sentence_embg_i]
                    sentence_embg_i += 1   
            #except:
                #print(x1, x2)
                #print(y1, y2)
        with open("../data/text_maps/"+file.name.replace(".jpg", ".pkl"), "wb") as f:
            pickle.dump(text_map, f)
        
        pd.DataFrame(text_map).to_csv("../data/text_maps/"+file.name.replace(".jpg", ".csv"))
        #break
        print("File: {} -- Done".format(file.name))