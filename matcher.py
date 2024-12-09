#------------------------- importok -------------------------
import spacy
import gensim
from gensim.models import KeyedVectors
import os
from collections import Counter
from textblob import TextBlob
from gensim import corpora
# -----------------------------------------------------------

#----------- pre-trained Word2Vec model behuzasa ------------
path = './GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(path, binary=True)
nlp = spacy.load('en_core_web_sm')
#------------------------------------------------------------


class PersonClassifier:
    def __init__(self, path):
        self.people_data = {}
        self.path = path
    
    #------------------ Adat letisztítása ismétlődő szavaktól spacy lib. segitsegevel ------------------
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
        
        return diary_text, cleaned_words
    #---------------------------------------------------------------------------------------------------

    #-------------- 1. szempont: Hangulatelemzés - (Sentiment analysis) ----------------
    '''
        - Indoklás: A szovegben felfedezhetunk erzelmeket kifejezo szavakat ez alapjan pedig meg tudjuk allapitanni,
          hogy az adott delikvens optimistabb/pesszimistabb latasmodu. Ez mindenfelekeppen segithet a parvalasztasnal.

        - Megkozelites: polarity arra szolgal, hogy visszaadja a szoveg pozitivitásának erteket, amely -1 es 1 kozotti erteket vesz fel: negativ esetben pesszimista, pozitiv esetben optimista hangulatot tukroz.
    '''
    def sentiment_analysis(self, diary_text):
        analysis = TextBlob(diary_text).sentiment
        return analysis.polarity
    #--------------------------------------------------
    
    #-------------- 2. szempont: erzelmi polaritas ----------------------
    def word2vec_similarity(self, cleaned_words):
        # List of keywords related to lifestyle, values, or other preferences
        keywords = ['love', 'adventure', 'family', 'career', 'fun']
        
        similarities = {}
        # For each keyword, calculate the average similarity of all cleaned words to this keyword
        for keyword in keywords:
            if keyword in model:
                # Compute the similarity of the keyword to each word in the cleaned_words list
                keyword_vector = model[keyword]
                word_similarities = []
                for word in cleaned_words:
                    if word in model:
                        word_vector = model[word]
                        # Compute similarity between the keyword and the current word
                        similarity = model.similarity(word, keyword)
                        word_similarities.append(similarity)
                
                # If similarities were found, calculate the average similarity
                if word_similarities:
                    avg_similarity = sum(word_similarities) / len(word_similarities)
                    similarities[keyword] = avg_similarity
        
        return similarities


    #------------------------------------------------------------------------------

    #------------------------ 3. szempont: Pronounok megszamlalasa -----------------------------
    '''
        - Indoklas: Ha a vizsgalt szemely tobbszor hasznal 'we', 'us' vagy mas tobb szemelyre utalo pronount
          akkor valoszinu, hogy csoport orientalt inkabb tarsasagi szemely.
        - Megkozelites:
    '''

    def personal_pronouns(self, doc):
        pronouns = [token.text for token in doc if token.pos_ == "PRON"]
        return Counter(pronouns)

    #------------------------------------------------------------------------------

    #--------------- 4. szempont: Fotema 1 ---------------

    #----------------------------------------

    #--------------- 5.szempont Fotema 2 ---------------

    #----------------------------------------

#------- Program running ---------
for i in range(20):
    people = PersonClassifier(path=f'./male/male{i+1}.txt')
    diary_text, cleaned_words = people.prepare_data(people.path)
    sentiment = people.sentiment_analysis(diary_text)
    word_similarities = people.word2vec_similarity(cleaned_words)

    print(f"Word2Vec Similarities: {word_similarities}")
    print(f"Sentiment Polarity: {sentiment:.5f}")
    
