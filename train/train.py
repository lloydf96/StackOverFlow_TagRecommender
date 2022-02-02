from gensim.models import Word2Vec
import os
import gensim
import random

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import re
import spacy

from torch.utils.data import DataLoader

class SOFDataset(Dataset):
    def __init__(self,qs,tags,model_qs,model_tags,process_text,use_embedding = False,get_all_tags = False,test_on_known_tags = True,add_negative_sample = False):
        self.qs = qs
        self.tags = tags
        self.model_qs = model_qs
        self.tags.reset_index(inplace = True, drop = True)
        self.tags['Tag'] = self.tags.Tag.apply(lambda x : self.process_tags(x))
        self.qs_vocab = model_qs.wv.index_to_key
        self.process_text = process_text
        self.use_embedding = use_embedding
        self.get_all_tags = get_all_tags
        self.model_tags = model_tags
        self.add_negative_sample = add_negative_sample
        self.qs['processed_text'] = self.qs.Title.apply(lambda x:[word if word in self.qs_vocab else '<unknown_word_SOF>' for word in self.process_text.process(x)])
        if test_on_known_tags:
            self.tags = self.tags[self.tags.Tag.isin(model_tags.wv.index_to_key)].reset_index(drop = True)
            self.qs = self.qs[self.qs.Id.isin(self.tags.Id)].reset_index(drop = True)
        if not self.use_embedding :
            self.qs_vocab_length = len(self.qs_vocab)
            self.qs_vocab_dict = dict(zip(self.qs_vocab,list(range(self.qs_vocab_length))))
            
    def __len__(self):
        return self.tags.shape[0]
    
    def __getitem__(self,idx):
        '''Return qs vector and tag word (for training word would be converted to vector using model_tags)'''
        
        tags_val = self.tags.Tag[idx]
        tags_idx = self.tags.Id[idx]
        tags_val = self.process_tags(tags_val)
        qs_val = self.qs[self.qs.Id == tags_idx].processed_text.item()
        
        if self.use_embedding:
            qs_val = [self.model_qs.wv[word] for word in qs_val if word in self.qs_vocab]
            if len(qs_val) == 0:
                qs_val = [self.model_qs.wv['<unknown_word_SOF>']]
            qs_val = self.embedding_qs(qs_val)
        else:
            qs_val = self.one_hot_qs(qs_val)
            
        if self.get_all_tags:
            tags_all = list(self.tags.Tag[self.tags.Id == tags_idx].values)
            if len(tags_all)<7:
                tags_all+=['None']*(7-len(tags_all))
            return qs_val,tags_val,tags_all
        elif self.add_negative_sample:
            tags_all = list(self.tags.Tag[self.tags.Id == tags_idx].values)
            return qs_val,tags_val, random.choice(list(set(self.model_tags.wv.index_to_key) - set(tags_all)))
        else:
            return qs_val,tags_val
        
    def one_hot_qs(self,qs_val):
        qs_val = torch.IntTensor(np.array([self.qs_vocab_dict[word] for word in qs_val])).to(torch.int64)
        qs_val = torch.sum(F.one_hot(qs_val,num_classes = self.qs_vocab_length),axis = 0)
        return qs_val.to(torch.float)
        
    def process_tags(self,tag):
        return str(tag).lower().strip()
           
    def embedding_qs(self,qs_val):
        
        qs_val = torch.from_numpy(np.sum(np.array(qs_val),axis = 0)).type(torch.float)
        
        return qs_val

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

def log_statement(statement = None):
    if statement is None:
        with open('logs.txt','w') as f:
            f.write('Logging started... ')
            
    else:
        with open('logs.txt','a') as f:
            f.write('\n'+statement)
            
def model_eval(model,dl,max_steps = 50,training = True):
    dl.dataset.get_all_tags = True
    tag_exists = 0
    samples = 0
    loss = 0
    for step,(qs,tag,all_tags) in enumerate(dl):
        qs,tag,all_tags = qs,tag,all_tags
        if training:
            model_op = model(qs,tag)
            loss+= loss_function(model_op[0],model_op[1],target = torch.tensor([1]))
        recommended_tags = model.infer(qs)
        all_tags = np.array(all_tags).T
        tag_exists += sum([1 for i,tags in enumerate(recommended_tags) if len(set(all_tags[i]).intersection(set(tags))) > 0])
        samples += qs.shape[0]
        if step > max_steps:
            break
        
    dl.dataset.get_all_tags = False
    
    return loss/samples,tag_exists/samples

def train(model,optimizer,scheduler,loss_function,train_dl,test_dl,epochs,max_samples_per_epoch,device):
    model.to(device)
    loss_function.to(device)
    log_statement()
    for i in range(epochs):
        print("Epoch No : ",i)
        model.train()
        for step,(qs,tag,neg_tag) in enumerate(train_dl):
            qs,tag = qs.to(device),tag
            model.zero_grad()
            model_op = model(qs,tag)
            neg_tag = torch.FloatTensor(model.model_tags.wv[neg_tag]).to(device)
            predicted = model_op[0].to(device)
            target = model_op[1].to(device)
            positive_case = torch.tensor([1]).to(device)
            negative_case = torch.tensor([-1]).to(device)
            loss = loss_function(predicted,target,target = positive_case) + loss_function(predicted,neg_tag,target = negative_case)
             
            #model(qs) will give both ip and op torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = 2,norm_type = 2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = 1,norm_type = 'inf')
            optimizer.step()
            
#             print("\nmodel weights")
#             print(model.L1[0].weight)
#             print("\nmodel grad")
#             print(model.L1[0].weight.grad)
            
            with torch.no_grad():
                print('\nloss')
                print(loss.detach())
                print('model_grad')
                print(model.L1[0].weight.grad.data.norm(2))
                if step>max_samples_per_epoch//batch_size:
                    break
        
        scheduler.step()
        model.eval()
        
        with torch.no_grad():
            print("\nmodel weights")
            print(model.L1[0].weight)
            print("\nmodel grad")
            print(model.L1[0].weight.grad)
            model.to('cpu')
            loss_function.to('cpu')
            train_loss,train_model_metric = model_eval(model,train_dl,max_samples_per_epoch//batch_size)
            test_loss,test_model_metric = model_eval(model,test_dl,training = False)
            
            statement = "Train loss is %f and Train model metric is %f"%(train_loss,train_model_metric)
            print(statement)
            log_statement(statement)
            
            statement = "Test loss is %f and Test model metric is %f"%(test_loss,test_model_metric)
            print(statement)
            log_statement(statement)
            model.to(device)
            loss_function.to(device)
            
    return model

def eval_baseline(recommended_tags,dl,max_steps = 50):
    samples = 0
    tag_exists = 0
    no_of_tags = 0
    dl.dataset.get_all_tags = True
    for step,(qs,tag,all_tags) in enumerate(dl):
        qs,tag,all_tags = qs,tag,all_tags

        all_tags = np.array(all_tags).T
        tag_exists +=  sum([1 for tags in all_tags if len(set(tags).intersection(set(recommended_tags))) > 0])
        no_of_tags += sum([ len(set(tags).intersection(set(recommended_tags))) for tags in all_tags])
        samples += qs.shape[0]
        if step > max_steps:
            break
    
    dl.dataset.get_all_tags = True
        
    return tag_exists/samples,no_of_tags/samples       

if __name__ = "__main__":
    qs_train = pd.read_csv('/input/qs_train.csv')
    qs_test = pd.read_csv('/input/qs_test.csv')
    tags_train = pd.read_csv('/input/tags_train.csv')
    tags_test = pd.read_csv('/input/tags_test.csv')

    model_tags = Word2Vec.load('/input/model_tags.model')
    model_qs = Word2Vec.load('/input/model_qs.model')

    qs_train = qs_train[qs_train.word_len >= 1]
    tags_train = tags_train[tags_train.Id.isin(qs_train.Id)]
    process_text = processText()

    batch_size = 2048

    train_ds = SOFDataset(qs_train,tags_train,model_qs,model_tags,process_text,use_embedding = False,get_all_tags = False,add_negative_sample = True)
    test_ds = SOFDataset(qs_test,tags_test,model_qs,model_tags,process_text,use_embedding = False,get_all_tags = False,test_on_known_tags = True)
    train_dl = DataLoader(train_ds, batch_size= batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=True)

    loss_function = nn.CosineEmbeddingLoss()
    epochs = 30
    max_samples_per_epoch = 20000
    input_dim = test_ds[0][0].shape[0]
    model = DNN(input_dim,10,model_tags)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl.dataset.get_all_tags = False
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs,eta_min = 5e-7)
    model = train(model,optimizer,scheduler,loss_function,train_dl,test_dl,epochs,max_samples_per_epoch*4,device)

    k= 5
    model.eval()
    torch.save(model.state_dict(), '/model/model.pth')
    counter=collections.Counter(tags_train.prepped_tags)
    total_words = np.max(np.array(list(counter.values())))
    recommended_tags = [word for word,val in sorted(counter.items(),key=lambda x : -1*x[1])[:k]]

    model.to('cpu')
    for k in [5,7,10,15,20,30]:
        model.k = k
        recommended_tags = [word for word,val in sorted(counter.items(),key=lambda x : -1*x[1])[:k]]
        questions_predicted,tags_predicted = model_eval(model,test_dl,training = False)
        baseline_q_predicted,baseline_tags_predicted = eval_baseline(recommended_tags,test_dl,1000)
        print("\n\nTop %d accuracy are : "%(k))
        print("Model percent questions with one correct tag: %f \nnumber of relevant predictions on average: %f"%(questions_predicted,tags_predicted))
        print("\nBaseline percent questions with one correct tag: %f \nnumber of relevant predictions on average: %f"%(baseline_q_predicted,baseline_tags_predicted))


