#---------------- importok ------------------

import csv
import spacy
import gensim
from gensim.models import KeyedVectors
import os
from textblob import TextBlob
from gensim import corpora
from difflib import SequenceMatcher
import math
#--------------------------------------------

#----------- pre-trained Word2Vec model behuzasa ------------
path = './GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(path, binary=True)
nlp = spacy.load('en_core_web_sm')
#------------------------------------------------------------

'''
    - Ebben a classban a diary feldolgozása, valamint mind a tíz paraméter,ezek pedig_
        - Hangulatelemzés - pozitív vagy negatív (1 és -1 közötti érték)
        - Érzelmi polaritás - 'love', 'adventure', 'family', 'career', 'fun' elemek közül a szöveg szavai melyik témához állnal legközelebb - (végeredmény az egyik téma)
        - Leggyakoribb szó
        - Írásjelek száma
        - Paragrafusok száma
        - Szóhasználat minősége - (egyedi szavak / összes szó)
        - Átlagos mondathossz
        - Kérdések száma
        - Legjellemzőbb érzelem - (happines, sadness, anger ,fear)
        -Szövegsűrűség.
'''
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
        - Indoklás: A szovegben felfedezhetunk erzelmeket kifejezo szavakat, ez alapjan pedig meg tudjuk
          állapítani, hogy az adott delikvens optimistabb/pesszimistabb látásmódú. Ez segíthet a párválasztásnál.
        - Megközelítés: A polarity arra szolgál, hogy visszaadja a szöveg pozitivitásának értékét, amely -1 és 1 közötti
          értéket vesz fel: negatív esetben pesszimista, pozitív esetben optimista hangulatot tükröz.
    '''
    def sentiment_analysis(self, diary_text):
        analysis = TextBlob(diary_text).sentiment
        return analysis.polarity
    #--------------------------------------------------

    #-------------- 2. szempont: Érzelmi polaritás - Word2Vec hasonlóság ----------------
    '''
        - Indoklás: Az emberek különféle értékeket képviselnek, például szeretetet, kalandot, családot stb.
          A szavak közötti hasonlóság segíthet megérteni, hogy az egyén milyen értékeket részesít előnyben.
        - Megközelítés: Word2Vec model segítségével mérhetjük a tisztított szavak és bizonyos kulcsszavak közötti hasonlóságot.
    '''
    def word2vec_similarity(self, cleaned_words):
        keywords = ['love', 'adventure', 'family', 'career', 'fun']
        
        similarities = {}
        for keyword in keywords:
            if keyword in model:
                keyword_vector = model[keyword]
                word_similarities = []
                for word in cleaned_words:
                    if word in model:
                        word_vector = model[word]
                        similarity = model.similarity(word, keyword)
                        word_similarities.append(similarity)
                
                if word_similarities:
                    avg_similarity = sum(word_similarities) / len(word_similarities)
                    similarities[keyword] = avg_similarity
        
        sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
        
        if len(sorted_similarities) >= 3:
            return sorted_similarities[2][0]
        else:
            return None
    #--------------------------------------------------

    #-------------- 3. szempont: Leggyakoribb szó ----------------
    '''
        - Indoklás: Az elsődleges téma meghatározása segít megérteni, mi foglalkoztatja leginkább a vizsgált személyt.
        - Megközelítés: A tisztított szavakból szótári elemeket használunk fel a téma azonosításához.
    '''
    def get_first_main_theme(self, cleaned_words):
        dictionary = corpora.Dictionary([cleaned_words])
        most_common_words = dictionary.token2id
        last_word = list(dictionary.token2id.keys())[-1]  # Az utolsó kulcs
        return last_word
    #--------------------------------------------------

    #-------------- 4. szempont: írásjelek száma ----------------
    '''
        - Indoklás: Az írásjelek száma a szöveg stilisztikai jellemzőire utalhat.
          Például a sok felkiáltójel vagy kérdőjel érzelmes vagy kérdező szövegre utalhat.
        - Megközelítés: A szöveg tokenizálása után megszámoljuk az írásjeleket.
    '''
    def punctuation_count(self, diary_text):
        doc = nlp(diary_text)
        punctuation_tokens = [token for token in doc if token.is_punct]
        return len(punctuation_tokens)
    #--------------------------------------------------


    #-------------- 5. szempont: Paragrafusok száma ----------------
    '''
        - Indoklás: A paragrafusok száma segíthet megérteni a szöveg struktúráját és a gondolatok elrendezését.
        - Megközelítés: Az új sorok (`\n`) száma alapján meghatározzuk a paragrafusok számát.
    '''
    def get_paragraph_count(self, diary_text):
        paragraphs = diary_text.split('\n')
        return len([p for p in paragraphs if p.strip()])  # Counts non-empty paragraphs
    #--------------------------------------------------


    #-------------- 6. szempont: szóhasználat ----------------
    '''
        - Indoklás: A szókincs gazdagsága megmutatja, hogy mennyire változatos a szöveg szókincse.
        - Megközelítés: Az egyedi szavak számát osztjuk el az összes szavak számával a tisztított szavak között.
    '''
    def vocabulary_richness(self, cleaned_words):
        total_words = len(cleaned_words)
        unique_words = len(set(cleaned_words))
        if total_words == 0:
            return 0
        return unique_words / total_words
    #--------------------------------------------------


    #-------------- 7. szempont: Átlagos mondathossz ----------------
    '''
        - Indoklás: Az átlagos mondathossz segíthet a szöveg stílusának és komplexitásának értékelésében.
          A rövidebb mondatok egyszerűbb kifejezésmódot tükrözhetnek, míg a hosszabbak mélyebb gondolatmenetre utalhatnak.
        - Megközelítés: Felosztjuk a szöveget mondatokra, majd kiszámítjuk az egy mondatra eső szavak átlagos számát.
    '''
    def average_sentence_length(self, diary_text):
        doc = nlp(diary_text)
        sentences = list(doc.sents)
        if not sentences:
            return 0
        total_words = sum(len([token for token in sentence if not token.is_punct]) for sentence in sentences)
        avg_length = total_words / len(sentences)
        return avg_length
    #--------------------------------------------------

    #-------------- 8. szempont: Kérdések száma a szövegben ----------------
    '''
        - Indoklás: A szövegben lévő kérdések száma utalhat az egyén kíváncsiságára vagy érdeklődési szintjére.
        - Megközelítés: Megszámoljuk a kérdőjellel ("?") végződő mondatokat a szövegben.
    '''
    def count_questions(self, diary_text):
        doc = nlp(diary_text)
        questions = [sentence for sentence in doc.sents if sentence.text.strip().endswith('?')]
        return len(questions)
    #--------------------------------------------------

    #-------------- 9. szempont: Legjellemzőbb érzelem Word2Vec segítségével ----------------
    '''
        - Indoklás: Az emberek érzelmeik alapján jellemezhetők, például boldogság, szomorúság, düh, félelem stb.
          A szövegben található szavak Word2Vec alapú hasonlósága segíthet meghatározni a domináns érzelmi tónust.
        - Megközelítés: Kulcsérzelmekhez tartozó szavak listáját használjuk, és kiszámítjuk ezek hasonlóságát a tisztított szöveg szavaival.
    '''
    def dominant_emotion(self, cleaned_words):
        emotion_keywords = {
            'happiness': ['happy', 'joy', 'pleasure', 'delight'],
            'sadness': ['sad', 'sorrow', 'grief', 'melancholy'],
            'anger': ['angry', 'fury', 'outrage', 'resentment'],
            'fear': ['fear', 'anxiety', 'worry', 'terror']
        }

        emotion_scores = {emotion: 0 for emotion in emotion_keywords}

        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in model:
                    for word in cleaned_words:
                        if word in model:
                            emotion_scores[emotion] += model.similarity(word, keyword)

        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        return dominant_emotion, emotion_scores[dominant_emotion]
    #--------------------------------------------------

    #-------------- 10. szempont: Szövegsűrűség ----------------
    '''
        - Indoklás: A szöveg sűrűsége (word density) arra utal, hogy a szöveg mennyire tömör, hány szó esik egy egységnyi karakterre.
        - Megközelítés: Az összes szó számát elosztjuk a szöveg karaktereinek számával.
    '''
    def word_density(self, diary_text):
        total_characters = len(diary_text.replace(" ", ""))  # Szóközök nélkül számított karakterek
        total_words = len([token for token in nlp(diary_text) if not token.is_punct])
        if total_characters == 0:
            return 0
        return total_words / total_characters
    #--------------------------------------------------

os.makedirs('./male_tables', exist_ok=True)
os.makedirs('./female_tables', exist_ok=True)

def create_table(folder, gender, i, classifier, diary_text, cleaned_words):
    sentiment = classifier.sentiment_analysis(diary_text)
    word2vec_similarities = classifier.word2vec_similarity(cleaned_words)
    common_words = classifier.get_first_main_theme(cleaned_words)
    punctuation_count = classifier.punctuation_count(diary_text)
    paragraph_count = classifier.get_paragraph_count(diary_text)
    vocabulary_richness = classifier.vocabulary_richness(cleaned_words)
    avg_sentence_length = classifier.average_sentence_length(diary_text)
    question_count = classifier.count_questions(diary_text)
    dominant_emotion, emotion_score = classifier.dominant_emotion(cleaned_words)
    word_density = classifier.word_density(diary_text)
    
    row = {
        "Sentiment Polarity": sentiment,
        "Word2Vec Similarities": word2vec_similarities,
        "Most Common Theme": common_words,
        "Punctuation Count": punctuation_count,
        "Paragraph Count": paragraph_count,
        "Vocabulary Richness": vocabulary_richness,
        "Average Sentence Length": avg_sentence_length,
        "Question Count": question_count,
        "Dominant Emotion": dominant_emotion,
        "Dominant Emotion Score": emotion_score,
        "Word Density": word_density,
    }

    with open(f'{folder}/{gender}{i+1}_table.txt', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

for i in range(20):
    classifier = PersonClassifier(f'./male/male{i+1}.txt')
    diary_text, cleaned_words = classifier.prepare_data(f'./male/male{i+1}.txt')
    create_table('./male_tables', 'male', i, classifier, diary_text, cleaned_words)

for i in range(20):
    classifier = PersonClassifier(f'./female/female{i+1}.txt')
    diary_text, cleaned_words = classifier.prepare_data(f'./female/female{i+1}.txt')
    create_table('./female_tables', 'female', i, classifier, diary_text, cleaned_words)


male_data = []
female_data = []

for i in range(20):
    male_txt = f'./male_tables/male{i+1}_table.txt'
    female_txt = f'./female_tables/female{i+1}_table.txt'

    with open(male_txt, 'r') as mf:
        processed_data = []
        lines = mf.readlines()
        title = lines[0].strip().split(',')
        value = lines[1].strip().split(',')
        
        for j in range(len(title)):
            processed_data.append({title[j]: value[j]})
        male_data.append(processed_data)

    with open(female_txt, 'r') as mf:
        processed_data = []
        lines = mf.readlines()
        title = lines[0].strip().split(',')
        value = lines[1].strip().split(',')
        for j in range(len(title)):
            processed_data.append({title[j]: value[j]})
        female_data.append(processed_data)


from difflib import SequenceMatcher
import math

from difflib import SequenceMatcher
import math

def string_similarity(str1, str2):
    """Visszaadja a hasonlóság arányát SequenceMatcher segítségével."""
    return SequenceMatcher(None, str1, str2).ratio()

def compute_match_score(male, female):
    """A férfi és a nő közötti match pontszám kiszámítása."""
    score = 0
    total_fields = len(male)

    for i in range(total_fields):
        male_key = list(male[i].keys())[0]  # A férfi adat kulcsának kinyerése
        male_value = male[i][male_key]      # A férfi adat értékének kinyerése
        female_value = female[i][male_key]  # A nő adat értékének kinyerése

        try:
            # Ha numerikus értékekről van szó, akkor a különbséget normalizáljuk
            male_value = float(male_value)
            female_value = float(female_value)
            score += 1 / (1 + abs(male_value - female_value))  # A különbség inverzének hozzáadása
        except ValueError:
            # Ha nem numerikus értékekről van szó, akkor a szöveges hasonlóságot számoljuk
            score += string_similarity(male_value, female_value)

    # A teljes match pontszámot normalizáljuk a mezők számával
    return score / total_fields

matches = []
matched_females = set()  # Halmaz, amelyben tároljuk azokat a nőket, akik már párosítva lettek

for male_index, male in enumerate(male_data):
    best_match = None
    best_female_index = None
    best_score = -math.inf  # Kezdetben nagyon alacsony pontszámot adunk meg

    for female_index, female in enumerate(female_data):
        if female_index in matched_females:
            continue  # Ha a nő már párosítva lett, akkor kihagyjuk

        match_score = compute_match_score(male, female)  # A match pontszám kiszámítása
        if match_score > best_score:  # Ha jobb pontszámot találtunk, frissítjük a legjobb párost
            best_score = match_score
            best_match = female
            best_female_index = female_index

    # Ha találtunk legjobb párost, hozzáadjuk a találatok listájához
    if best_female_index is not None:
        matches.append((male_index + 1, best_female_index + 1, best_score))  # Párosítás tárolása
        matched_females.add(best_female_index)  # A nő párosítva lett, hozzáadjuk a halmazhoz

# Az eredmények kiírása százalékos formátumban
for male_idx, female_idx, score in matches:
    match_percentage = score * 100  # A pontszámot százalékra konvertáljuk
    print(f"Male {male_idx} - Female {female_idx}: Match Score = {match_percentage:.2f}%")