import pandas as pd
import os
import sys
sys.path.append('..\model')
import torch
from recommend_tags import *

if __name__=='__main__':

    question = "Hey how does java work?"
    infer = recommend_tags('../model/model.pth')
    tags = infer.get_tag(question)
    print(tags)

