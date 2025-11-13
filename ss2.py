#sentense simility2(ss2.py)
"""
Bert Model from transformers
"""
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

'''
uses torch
'''
def  get_embedding(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors = 'pt',
        truncation = True,
        padding = True
        )
    with torch.no_grad():
        outputs = model(**inputs)
    #returns MEAN Pooling
    return outputs.last_hidden_state.mean(dim = 1)

'''
functional from nn from torch
'''
from torch.nn import functional as F

def sentence_similarity(sent1, sent2):
    emb1, emb2 = get_embedding(sent1), get_embedding(sent2)
    return F.cosine_similarity(emb1, emb2).item()

#sentence1 and sentense2
#0.90257174 +1same
s1, s2 = "hii my name is R.LaxmanSayaCharan" , 'hii R.LaxmanSatyaCharan is my name'

similarity_score =sentence_similarity(s1, s2)
print(f'Similarity Score: {similarity_score : 0.8f}')