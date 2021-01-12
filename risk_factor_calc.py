# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 15:31:32 2020
@author: DELL
"""

import csv
import string
import numpy as numpy
import pandas as pd
import nltk ,re, string, collections
from nltk.util import ngrams
from nltk import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

Filename= "C:\\Users\\DELL\\Desktop\\SIH\\chats\\Fin.csv"

def tokenization(text):
    tokens=text.split()
    table=str.maketrans('','',string.punctuation)
    tokens=[w.translate(table) for w in tokens]
    tokens=[word.lower() for word in tokens]
    #sentences = sent_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]
    return tokens

def to_ngrams(tokens,n):
    #list2=[]
    to_ngram=ngrams(tokens,n)
    list_ngram=list(to_ngram)
    ngram_freq=collections.Counter(to_ngram) 
    freq_dict=dict(ngram_freq)
    return list_ngram

Data= pd.read_csv(Filename)
Data_new= Data.iloc[:, 1]
Data_new= pd.DataFrame(Data_new)
Chat= Data.iloc[:, 2]
text_lemma= Data.iloc[:, 3]
text_lemma.fillna("Not Applicable", inplace = True) 
Data['Stage']= [0 if x==4 else x for x in Data['Stage']]
stage= Data.iloc[:, 4]
stage.fillna(4, inplace = True) 
column_values = pd.Series(Chat)
c2= pd.Series(text_lemma)
c3= pd.Series(stage)
Data_new.insert(loc=1, column='Chats', value=column_values)
Data_new.insert(loc=2, column='Text_Lemma', value=c2)
Data_new.insert(loc=3, column='Stage', value=c3)
Main0= Data_new[Data_new['Stage']==0]
Main1= Data_new[Data_new['Stage']==1]
Main2= Data_new[Data_new['Stage']==2]
Main3= Data_new[Data_new['Stage']==3]
Data_new= Main0.head(300)
Data_new= Data_new.append(Main1.head(600))
Data_new= Data_new.append(Main2.head(600))
Data_new= Data_new.append(Main3.head(600))
#Temp= Data_new.loc[Data_new['0'] =='hum_366 ']
#Temp= Temp.loc[Temp['Chats'] !=' ']
#Data_new['Stage']= [0 if x==4 else x for x in Data_new['Stage']]
texts= list(Data_new.iloc[:, 2])
labels= Data_new.iloc[:, 3]
lemma=Data_new.iloc[:,2]
size=len(labels)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(
            ngram_range=(1, 3),
            use_idf=True,
            smooth_idf=False,
            norm=None,
            decode_error='replace',
            max_features=10000,
            min_df=5,
            max_df=0.5)

vect_file = open('vect_file', 'ab') 
pickle.dump(vectorizer,vect_file)

#for each in lemma: 
lemma_file=open('lemma_file', 'ab')
pickle.dump(lemma,lemma_file)
ngramlist=vectorizer.fit_transform(lemma)
ngram2_file=open('ngram2_file', 'ab')
pickle.dump(ngramlist,ngram2_file)


dense_mat=ngramlist.todense()
#feature_names = vectorizer.get_feature_names()
#df = pd.DataFrame(dense_mat, columns=feature_names)
X=dense_mat
Y= labels.astype("int16")
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Model 1
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)
y_pred1 = classifier1.predict(X_test)
Accuracy1= accuracy_score(y_test,y_pred1)
from sklearn.metrics import f1_score
F11_mac=f1_score(y_test, y_pred1, average='macro')
F11_mic=f1_score(y_test, y_pred1, average='micro')
from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test,y_pred1)
from sklearn.metrics import classification_report
target_names=['neutral','trust development','grooming','approach']
print(classification_report(y_test, y_pred1, target_names=target_names))



#Model 2
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'linear', random_state = 0)
classifier2.fit(X_train, y_train)

y_pred2 = classifier2.predict(X_test)
Accuracy2= accuracy_score(y_test,y_pred2)
F12_mac=f1_score(y_test, y_pred2, average='macro')
F12_mic=f1_score(y_test, y_pred2, average='micro')

#Model 3
from sklearn.naive_bayes import GaussianNB
classifier3= GaussianNB()
classifier3.fit(X_train, y_train)
y_pred3 = classifier3.predict(X_test)
Accuracy3= accuracy_score(y_test,y_pred3)
F13_mac=f1_score(y_test, y_pred3, average='macro')
F13_mic=f1_score(y_test, y_pred3, average='micro')
cm3=confusion_matrix(y_test,y_pred3)



from sklearn import tree
cfier=tree.DecisionTreeClassifier(max_leaf_nodes=12, max_depth=6)
cfier.fit(X_train, y_train)
#getting the details of the fitted model, tree_ attribute gives the tree structure
n_nodes = cfier.tree_.node_count
children_left = cfier.tree_.children_left
children_right = cfier.tree_.children_right
feature = cfier.tree_.feature
threshold = cfier.tree_.threshold

import pickle
LR_file2 = open('LR_file2', 'ab') 
pickle.dump(classifier1,LR_file2)



from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

#from textblob import TextBlob

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


#nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV} # Pos tag, used Noun, Verb, Adjective and Adverb



def stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


#Creating function for tokenization
def tokenization(text):
    text = re.split('\W+', text)
    return text
def text_clean(text):

    #text  = text.lower()
    #text = text.replace('[^\w\s]','')
    text = stopwords(text)
    text = lemmatize_words(text)

    return text
"""
lemma2="when do you want to meet?"
#lemma3=[]
lemma_fin=text_clean(lemma2)
"""

test_frame = pd.read_csv (r'C:\\Users\\DELL\\Desktop\\SIH\\chats\\predator2.csv')
test_frame.dropna(subset=['text_lemma'], inplace=True)
#test_frame['training_label']=0
score=0
#lemma2=['hi beautiful']
lemma2=test_frame.iloc[:,8]
ngramlist2=vectorizer.transform(lemma2)

dense_mat2=ngramlist2.todense()
y_pred_test= classifier1.predict(dense_mat2)
counter=len(y_pred_test)
weight1=0.224
weight2=0.274
weight3= 0.5
count1=0
count2=0
count3=0
threshold=0.15
score1=0
score2=0
score3=0

for val in y_pred_test:
    if(val==1):
        score+=weight1
        count1+=1
        score1+=weight1
    elif(val==2):
        score+=weight2
        count2+=1
        score2+=weight2
    elif(val==3):
        score+=weight3
        count3+=1
        score3+=weight3
#av1=score1/score
av2=score2/score
av3=score3/score
risk_list=[]
list_av=[]
#list_av.append(av1)
list_av.append(av2)
list_av.append(av3)
list_av2=[]
for i in list_av:
    list_av2.append(i-threshold)
for i in range(0,len(list_av2)):
    if(list_av2[i]>0):
        risk_list.append(list_av2[i]/list_av[i])

if(len(risk_list)>1):
    max_risk=risk_list[0]
    for i in range(0,len(risk_list)):
       if(risk_list[i]>max_risk):
           max_risk=risk_list[i]
           index=i+2
else:
    max_risk=risk_list[0]
    index=3
print("The risk factor is %s for the level %s"%((max_risk*100),index))
"""
risk=((av3-threshold)/av3)*100
print(score)
print(score/counter)
for val in y_pred_test:
    print(val)
"""
