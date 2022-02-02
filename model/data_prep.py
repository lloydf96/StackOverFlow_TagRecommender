import spacy
import re
import pandas as pd
import numpy as np
import random
import sys
sys.path.append('model')
import torch
import torch.nn as nn
import torch.nn.functional as F

class processText():
    def __init__(self):
        
        nlp = spacy.load("en_core_web_sm")

        special_cases = {
                        "c#" : [{"ORTH": "csharp"}],
                        "c++" : [{"ORTH": "c++"}],
                         ".net" : [{"ORTH": "asp.net"}],
                         "asp.net": [{"ORTH": "asp.net"}],
                        }

        nlp.tokenizer.rules = nlp.tokenizer.rules.update(special_cases)
        suffixes = list(nlp.Defaults.suffixes)
        suffixes.remove(">")
        suffixes.remove(":")

        suffix_regex = spacy.util.compile_suffix_regex(suffixes)
        nlp.tokenizer.suffix_search = suffix_regex.search

        prefixes = list(nlp.Defaults.prefixes)
        prefixes.remove("<")
        prefixes.remove(":")

        prefix_regex = spacy.util.compile_prefix_regex(prefixes)
        nlp.tokenizer.prefix_search = prefix_regex.search
        self.all_stopwords = nlp.Defaults.stop_words
        self.nlp = nlp

    def process_sentence(self,sentence):
        doc = self.nlp(sentence.lower())
        return [token.text for token in doc if not self.regex_prep(token.text)]

    def regex_prep(self,word):
        #if any of them are true then do not continue
        is_true = bool(re.search(r"^\W+$",word))
        is_true = (True if word in self.all_stopwords else False) or is_true
        return is_true

    def process(self,sentence):
        return self.process_sentence(sentence)

class process_text_for_model():
    def __init__(self,model_qs):
        self.model_qs = model_qs
        self.qs_vocab = model_qs.wv.index_to_key
        self.process_text = processText()
        self.qs_vocab_length = len(self.qs_vocab)
        self.qs_vocab_dict = dict(zip(self.qs_vocab,list(range(self.qs_vocab_length))))

    def process(self,text):
        processed_text = self.process_text.process(text)
        processed_text_vector = [self.model_qs.wv[word] for word in processed_text if word in self.qs_vocab]

        if len(processed_text_vector) == 0:
            processed_text_vector = [self.model_qs.wv['<unknown_word_SOF>']]

        text_vector = self.one_hot_qs(processed_text_vector)
        return text_vector[None,:]
     
    def embedding_qs(self,text):
        text_vector = torch.from_numpy(np.sum(np.array(text),axis = 0)).type(torch.float)
        return text_vector

    def one_hot_qs(self,qs_val):
        qs_val = torch.IntTensor(np.array([self.qs_vocab_dict[word] for word in qs_val])).to(torch.int64)
        qs_val = torch.sum(F.one_hot(qs_val,num_classes = self.qs_vocab_length),axis = 0)
        return qs_val.to(torch.float)

