# ------ imports ------
import spacy
import gensim
from gensim.models import KeyedVectors
import os
from collections import Counter
# ---------------------

'''
spacy - removing recurring, redundant words (e.g., is, a, so, I, etc.)
-> Installing:
    pip install spacy
-> Downloading English language model:
    python -m spacy download en_core_web_sm

word2vec - pip install gensim, downloading Google's pre-trained model
-> Make sure to download the Google News pre-trained model from the Word2Vec repository.
'''

# Loading the pre-trained Word2Vec model
path = './GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(path, binary=True)
nlp = spacy.load('en_core_web_sm')

class PersonClassifier:
    def __init__(self,path):
        self.people_data = {}
        self.path = path
    

    #---------- Method to prepare and clean diary data ----------
    def prepare_data(self, data_route):
        diary_text = ''
        if os.path.isfile(data_route):
            with open(data_route, 'r', encoding='utf-8') as file:
                diary_text = file.read()
        else:
            for filename in os.listdir(data_route):
                if filename.endswith(".txt"):
                    with open(os.path.join(data_route, filename), 'r', encoding='utf-8') as file:
                        diary_text += file.read() + ' '
        
        doc = nlp(diary_text)
        cleaned_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
        
        return cleaned_words
    #------------------------------------------------------------
        