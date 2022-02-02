import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('model')
class DNN(nn.Module):
    def __init__(self,input_dim,output_dim,model_tags,metric = 'cosine',k = 5):
        super(DNN, self).__init__()
        self.L1 = nn.Sequential(nn.Linear(input_dim,1000),
                                nn.BatchNorm1d(1000),
                                nn.ReLU(),
                                nn.Linear(1000,500),
                                nn.BatchNorm1d(500),
                                nn.ReLU(),
                                nn.Linear(500,100),
                                nn.BatchNorm1d(100),
                                nn.ReLU(),
                                nn.Linear(100,100),
                                nn.BatchNorm1d(100),
                                nn.ReLU(),
                                nn.Linear(100,100),
                                nn.BatchNorm1d(100),
                                nn.ReLU(),
                                nn.Linear(100,75),
                                nn.BatchNorm1d(75),
                                nn.ReLU(),
                                nn.Linear(75,50),
                                nn.BatchNorm1d(50),
                                nn.ReLU()
                                )

        self.fc = nn.Linear(50,output_dim)
        self.model_tags = model_tags
        self.k = k
        self.model_tags_vector = torch.FloatTensor(self.model_tags.wv.vectors)
        self.metric = metric
        
    def forward(self,qs,tags):
        tag_model_op = self.predict(qs)
        tag_op = torch.FloatTensor([self.model_tags.wv[tag] for tag in tags])
        return tag_model_op,tag_op
    
    def infer(self,qs):
        tag_model = self.predict(qs)
        tag_distance = [self.distance_metric(tag) for tag in tag_model]
        nearest_k_tags = [torch.topk(i_distance,self.k).indices for i_distance in tag_distance]
        nearest_k_tags = [ [self.model_tags.wv.index_to_key[tag_index] for tag_index in tag_list] for tag_list in nearest_k_tags]
        return nearest_k_tags
        
    def predict(self,qs):
        L1_op = self.L1(qs)
        op = self.fc(L1_op)
        return op
    
    def distance_metric(self,tag):
        
        if self.metric == 'cosine':
#             print((tag @ self.model_tags_vector.T).shape)
#             print(((tag.data.norm(2))* self.model_tags_vector.T.norm(2,dim = 0)).shape)
            return (tag @ self.model_tags_vector.T) / ((tag.data.norm(2))* self.model_tags_vector.T.norm(2,dim = 0))
        else :
#             print(torch.sum((tag - self.model_tags_vector)**2,dim = 1).shape)
            return -1*torch.sum((tag - self.model_tags_vector)**2,dim = 1)