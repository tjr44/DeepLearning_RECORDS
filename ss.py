#sentense simility(ss.py)
'''
BERT model from transformers
'''
from transformers import BertTokenizer, BertModel
import torch

#record date and time
from rdt_engine import run_m
run_m.ntime_code_name(__file__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

'''
uses torch 
'''
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling

'''
functional from nn from torch
'''
import torch.nn.functional as F

def sentence_similarity(sent1, sent2):
    emb1 = get_embedding(sent1)
    emb2 = get_embedding(sent2)
    return F.cosine_similarity(emb1, emb2).item()

"""
given
"""
'''
similarity most of all same if same  `voco` file uses in this `Google Vocolabary` pretarined
'''

#0.7429#0.7429
s1 = "How are you?"
s2 = "What is your mood today?"

#0.7260 #0.7260#0.7260
s1 = "what is your name?"
s2 = 'can you tell your name' 

#if same sentennse gaves returns 1.0000
s1 = s2 = 'hello'

#0.8451
s1 = 'hello'
s2 = 'hey'

similarity_score = sentence_similarity(s1, s2)
print(f"Similarity Score: {similarity_score:.4f}")