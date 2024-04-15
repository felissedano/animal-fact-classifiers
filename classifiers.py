import nltk as nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer      
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
import numpy as np


# The function for stemming, used inside tfidfvectorizer
def tokenize_stem_removepunc(text: str) -> list[str]:
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    tokens_no_punc = [word for word in tokens if word not in string.punctuation]
    stems = [stemmer.stem(token) for token in tokens_no_punc]
    return stems

# The function for lemmatizing, used inside tfidfvectorizer
def tokenize_lemma_removepunc(text:str) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens_no_punc = [word for word in tokens if word not in string.punctuation]
    lemmas = [lemmatizer.lemmatize(token) for token in tokens_no_punc]
    return lemmas



# initialize training and testing data
label = ["fake","fact"]
train_corpus = []
train_corpus_labels = []

test_corpus = []
test_corpus_labels = []

# Load the corpus and lavels for both training and test data
fakes_file = open("./fakes.txt")
facts_file = open("./facts.txt")
for i in range(4):
    for j in range(100):
        if(j < 70):
            train_corpus.append(facts_file.readline())
            train_corpus_labels.append("fact")
            train_corpus.append(fakes_file.readline())
            train_corpus_labels.append("fake")
        else:
            test_corpus.append(facts_file.readline())
            test_corpus_labels.append("fact")
            test_corpus.append(fakes_file.readline())
            test_corpus_labels.append("fake")
facts_file.close()
fakes_file.close()

# convert the data from normal python list to numpy array for better performance
train_corpus = np.array(train_corpus)
train_corpus_labels = np.array(train_corpus_labels)

test_corpus = np.array(test_corpus)
test_corpus_labels = np.array(test_corpus_labels)

# Initialize the lists with parameters as the elements
tokenizers  = [tokenize_stem_removepunc, tokenize_lemma_removepunc, tokenize_stem_removepunc, tokenize_lemma_removepunc]
stopwords   = ['english', 'english','english', 'english']
ngram = [(1,1),(1,1),(1,2),(1,2)]
use_idf_options = [True,True,True,True]
# classifiers = [MultinomialNB, LogisticRegression, SVC]
best_model = None
best_model_description = ""
best_score = 0
 
file = open("felis-a1-result.txt","w")

# run a for loop and do 4-fold CV on all three models with all four preprocessing setups
# Record the best model and then run the final test on it
for i in range(4):

    tf_vectorizer = TfidfVectorizer(tokenizer=tokenizers[i],stop_words=stopwords[i])
    x_train_tf = tf_vectorizer.fit_transform(train_corpus)

    x_test_tf = tf_vectorizer.transform(test_corpus)

    nb_pipeline = make_pipeline(TfidfVectorizer(tokenizer=tokenizers[i],stop_words=stopwords[i],ngram_range=ngram[i]),MultinomialNB())
    nb_score = cross_val_score(nb_pipeline,train_corpus,train_corpus_labels,cv=4)
    mean = nb_score.mean()
    description = " mean of 4-fold cv with {}, {}, {}".format(str(tokenizers[i].__name__),"Naive-Bayes",str(ngram[i])) + "\n"
    file.write(str(mean) + description)
    if mean > best_score:
        best_score = mean
        best_model = nb_pipeline
        best_model_description = description
    

    svc_pipeline =  make_pipeline(TfidfVectorizer(tokenizer=tokenizers[i],stop_words=stopwords[i],ngram_range=ngram[i]),SVC())
    svc_score = cross_val_score(svc_pipeline,train_corpus,train_corpus_labels,cv=4)
    mean = svc_score.mean()
    description = " mean of 4-fold cv with {}, {}, {}".format(str(tokenizers[i].__name__),"SVC",str(ngram[i])) + "\n"
    file.write(str(mean) + description)

    if mean > best_score:
        best_score = mean
        best_model = svc_pipeline
        best_model_description = description

    
    lr_pipeline =  make_pipeline(TfidfVectorizer(tokenizer=tokenizers[i],stop_words=stopwords[i],ngram_range=ngram[i]),LogisticRegression(C=1))
    lr_score = cross_val_score(lr_pipeline,train_corpus,train_corpus_labels,cv=4)
    mean = lr_score.mean()
    description = " mean of 4-fold cv with {}, {}, {}".format(str(tokenizers[i].__name__),"LogReg",str(ngram[i])) + "\n"
    file.write(str(mean) + description)
    if mean > best_score:
        best_score = mean
        best_model = lr_pipeline
        best_model_description = description

# Use the best model found in for loop to train on the final test data and do the prediction to get the final score.
best_model.fit(train_corpus,train_corpus_labels)
predicted = best_model.predict(test_corpus)

file.write("Score of best model on the final test data is " + str(np.mean(predicted == test_corpus_labels)) + best_model_description)


file.close()

while True:
    inp = input("\nPlease input an animal fact\n")
    print(best_model.predict([inp]))


# Data
# Yujun Zhong for cat
# Anshita Saxena for duck
