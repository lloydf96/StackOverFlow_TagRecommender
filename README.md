# StackOverFlow Tag Recommender app
The goal of this project was to recommend 5 tags which best represents a StackOverFlow question.

## Model description:
1. The model was trained on the [StackSample (10% of StackOverFlow Q&A) data](https://www.kaggle.com/stackoverflow/stacksample). Only those questions with a score of over 5 were considered, giving a total of 93,153 questions. 
2. The Word2Vec embedding model was used on tags to generate a lower dimensional (10D) representation of over 3000 tags, with the assumption that similar tags would cluster together. The embeddings ability to represent tags using a lower dimensional representation was verified by comparing  10 nearest neighbours for each tag in the embedding with a tag graph generated using tag co-occurance frequency and by visual verification using t-SNE plots on embedding.
3. A simple feed forward neural network was used to fit the question (represented as Bag of Words) to each of the corresponding tag vector.
4. Used a cosine triplet loss function to compare a positive and a negative tag vector with the model output.
5. The model achieved a Top 5 accuracy of 0.58 over the baseline of 0.42. The baseline model consisted of 5 most frequently used tags in the training data.

The model is at the heart of the application, which was developed using HTML, CSS and Flask and dockerised. 


https://user-images.githubusercontent.com/77821166/151679179-968b2849-1eaf-42ed-959b-5c93fb5044a6.mp4

