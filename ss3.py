#record date and time
from rdt_engine import ntime_code_name
ntime_code_name(__file__)

#Sentense Simility3(ss3.py)
#Bert Model from tranformers
from transformers import BertTokenizer, BertModel
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

'''
uses torch
'''
def get_embedding(sentence):
    inputs = tokenizer(
           sentence,
           return_tensors = 'pt',
           truncation = True,
           padding = True,           
    )
    with torch.no_grad():
        outputs = model(**inputs)
    #returns MEAN Pooling
    return outputs.last_hidden_state.mean(dim = 1)

'''
functional from nn from torch
'''
import torch.nn.functional as F

def sentence_similarity(sent1,sent2):
    #if tuple unpacking is NoneType.__format__
    emb1, emb2 = get_embedding(sent1),get_embedding(sent2),
    return F.cosine_similarity(emb1,emb2).item()

#sentence1 and sentence2
#None if tuple unpacking `deeplearning`
#`1` if not float point
s1, s2 = (
          "what is deeplearning",
          'explain deeplearning'
)
#not, if tuple unpacking returns import ss3None(no issue in tuple unpacking)
#mostly returns `1`
#s1, s2 = 'what is your name?', 'your name is?'
#1 if 0.0f
#s1, s2 = 'what is your name?', 'who are you'
#0.9026   
#s1, s2 = "hii my name is R.LaxmanSayaCharan" , 'hii R.LaxmanSatyaCharan is my name' #not None

#function varible(object)
similarity_score = sentence_similarity(s1, s2)

print(f'{similarity_score : 0.0f}')#floating (float | tensor) varible : 0.nf n as number