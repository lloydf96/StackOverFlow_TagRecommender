from model_class import *
from data_prep import *
from gensim.models import Word2Vec
import os
import sys
sys.path.append('model')
model_tags = Word2Vec.load('../model/model_tags.model')
model_qs = Word2Vec.load('../model/model_qs.model')

class recommend_tags():
    def __init__(self,model_path):
        self.process_text = processText()
        self.process_qn = process_text_for_model(model_qs)
        self.model = DNN(4081,10,model_tags)
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()
        
    def get_tag(self,question):
        question_vector = self.process_qn.process(question)
        return self.model.infer(question_vector)[0]
