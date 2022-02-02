import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
import collections

import gensim
import spacy
import re

def distance(a,b):
    return np.sum((a - b)**2,axis = 1)

def replace_rare_words(word,words_unknown):
    if word in words_unknown:
        return '<unknown_word_SOF>'
    else:
        return word

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

if __name__ == '__main__':
    qs = pd.read_csv('/input/Questions.csv',encoding = "ISO-8859-1")
    ans = pd.read_csv('/input/Answers.csv',encoding = "ISO-8859-1")
    tags = pd.read_csv('/input/Tags.csv',encoding = "ISO-8859-1")
    qs = qs[qs.Score>=5]
    qs_train,qs_test = train_test_split(qs, test_size=0.2)

    #select tags_sample add frequency and keep only those tags more than 10
    tags = tags[tags.Id.isin(qs.Id)]
    tags_train = tags[tags.Id.isin(qs_train.Id)]
    tags_test = tags[tags.Id.isin(qs_test.Id)]

    tags_train['prepped_tags'] = tags_train.Tag.apply(lambda x : str(x).lower().strip())
    #check distribution of word and frequency drop lower 1percentile
    tag_frequency = tags_train.prepped_tags.value_counts()
    #tag_frequency.plot(kind = 'bar')
    total_samples = sum(tag_frequency.values)
    tags_to_keep = tag_frequency[tag_frequency >= 10]
    tags_train = tags_train[tags_train.prepped_tags.isin(tags_to_keep.index)]
    tags_sentences = tags_train.groupby('Id').apply(lambda x : list(x.prepped_tags.values))

    model_tags = gensim.models.Word2Vec(vector_size = 10,window = 5,sg = 1,epochs = 10,compute_loss = True)
    model_tags.build_vocab(tags_sentences.values)
    model_tags.train(tags_sentences.values, total_examples=model_tags.corpus_count, epochs=10,compute_loss = True)   

    nearest_label = {}

    vectors = np.asarray(model_tags.wv.vectors)
    labels = np.asarray(model_tags.wv.index_to_key) 

    k = 10
    for i,label in enumerate(labels):
        label_vec = vectors[i,:]
        indices = np.argpartition(distance(label_vec,vectors), k+1)[0:k+1]
        nearest_label[label] = list(set(list(labels[indices])) - set([label]))

    neighbour_data = pd.DataFrame(data = {"label" : list(nearest_label.keys()),"neighbours" : nearest_label.values()})

    neighbour_data['len'] = neighbour_data.neighbours.apply(lambda x : len(x))

    tag_list = labels
    tag_dict = {}

    for label in labels:
        tags_label = tags_train[tags_train.Id.isin(tags_train[tags_train.prepped_tags == label].Id)]
        tags_label = tags_label[tags_label.prepped_tags != label].prepped_tags.value_counts().reset_index()[:k]
        tag_dict.update({label: [row[1]['index'] for row in tags_label.iterrows()]})
        
    neighbours_from_graph = pd.DataFrame(data = {'label':list(tag_dict.keys()),'neighbours_from_graph':tag_dict.values()}) 
    neighbours_from_graph= pd.DataFrame(data = {'label':list(tag_dict.keys()),'neighbours_from_graph':tag_dict.values()})   
    neighbours_from_graph.set_index('label',inplace = True)
    neighbour_data.set_index('label',inplace = True)
    neighbours = pd.concat([neighbours_from_graph,neighbour_data],axis = 1)
    neighbours['similar_tags'] = neighbours.apply(lambda x : len(set(x[0]).intersection(set(x[1]))),axis = 1)
    
    print("Tag similarity is :",neighbours.similar_tags.mean()/k)
    qs_train = qs_train[qs_train.Id.isin(tags_train.Id)]
    process_text = processText()
    qs_train['processed_sample'] = qs_train.Title.apply(lambda x : process_text.process(x))
    qs_train['word_len'] = qs_train.processed_sample.apply(lambda x : len(x))

    qs_vocab = [word for sublist in list(qs_train['processed_sample'].values) for word in sublist]

    counter=collections.Counter(qs_vocab)
    total_words = np.max(np.array(list(counter.values())))
    counter = {key:val for key, val in counter.items() if val<=10}
    words_unknown = [key for key,_ in list(counter.items())]

    qs_train['prepped_sentences'] = qs_train.processed_sample.apply(lambda x : [  replace_rare_words(word,words_unknown) for word in x])

    qs_train.to_csv('/input/qs_train.csv')
    qs_test.to_csv('/input/qs_test.csv')
    tags_train.to_csv('/input/tags_train.csv')
    tags_test.to_csv('/input/tags_test.csv')

    model_tags.save('/model/model_tags.model')
    model_qs.save('/model/model_qs.model')
        # Input data files are available in the read-only "../input/" directory
        # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
