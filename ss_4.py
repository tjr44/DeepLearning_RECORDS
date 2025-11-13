#sentence similarity file4
"""
Bert model of sentence similarity from transformers
"""
from transformers import BertTokenizer, BertModel
import torch
from torch import Tensor
#functional from nn from torch dynamic deeplearning frame work
import torch.nn.functional as F
#type hint
from typing import Any,Tuple,Union

'''record date and time'''
from rdt_engine import run_m
run_m.ntime_code_name(__file__)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#uses torch embedding sentence
def get_embedding(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors = 'pt',
        truncation = True,
        padding = True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim = 1) # mean pooling

#function for cosine_similarity returns
'''
`Any` return types
[or]
Tuple[str,...] more tuple of strings by (ellipsis)[...]
'''
def sentence_similarity(sent1 : str = '', sent2 : str = "") -> Any | Tuple[str,...]:
    embedding1 , embedding2 = get_embedding(sent1) , get_embedding(sent2)
    return F.cosine_similarity(embedding1, embedding2).item()


#starting point of entere codebase
'''
type hint | type annoation Union of string numbers
'''
def ___main___()->None | Union[str,Tensor]:
    global s1, s2
     
    '''current module and datetime was recored:) ---> `ss_4.py` , `13-Nov-2025 >>> 09:24:04 PM`  at package :  >>> `models_by_pytorch`

       Similarity Scoring is :  0.6343863010406494140625000000000000000000000000000000000000000000000000000000000000000000000000000000
    '''
    
    s1 , s2 = 'what is deep learning', "explain deeplearning"
    return print(f'Similarity Scoring is : {sentence_similarity(s1, s2) : .100f}')


#`True` returns
if __name__ == f'__main__':
    ___main___()