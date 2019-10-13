#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:42:49 2019

@author: eddie
"""
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten
from keras.optimizers import Adam, SGD
import pandas as pd
import json
import random
import h5py
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')

#intents and patterns 
with open('pattern.json') as json_data:
    intents= json.load(json_data)
    
#Pre_processing 1: tokenizing 
classes= []
words= []
documents= []
ignore_words= ['?','.',',','!']
stemmer= LancasterStemmer()
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w= nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words= [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words= sorted(list(set(words)))
classes= sorted(list(set(classes)))

#pre_processing 2 preparing training data and label
training= []
output_empty= np.zeros(len(classes))

for doc in documents:
    bag = []
    pattern_words= doc[0]
    pattern_words= [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        
    output_row= list(output_empty)
    output_row[classes.index(doc[1])]= 1
    
    training.append([bag, output_row])

random.shuffle(training)
training= np.array(training)

x_train= list(training[:,0])
y_train= list(training[:,1])

#deap learning model with optimized h_parameters 
try:
    model= load_model('cbot.h5')
except:
    model = Sequential()
    model.add(Dense(128, input_shape= (len(x_train[0]),), activation= 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation= 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(y_train[0]), activation= 'softmax'))
    sgd= SGD(lr= 0.01, decay= 1e-6, momentum= 0.9, nesterov= True)
    model.compile(loss='categorical_crossentropy', optimizer= sgd, metrics= ['accuracy'])
    model.fit(np.array(x_train), np.array(y_train), epochs= 200, batch_size= 5, verbose= 0)
    model.save('cbot.h5')

#creating bag of words
def BoW(sents,words):
    Bag= np.zeros(len(words))
    SW= nltk.word_tokenize(sents)
    SW= [stemmer.stem(w.lower()) for w in SW if w not in ignore_words]
    for _ in SW:
         for i,j in enumerate(words):
                if _ == j:
                    Bag[i]= 1
    Bag=np.reshape(Bag,(1,69))
    return np.array(Bag)
#a function to find a string sector within a data frame 
def sec_finder(df,clss,sector):
    a=[]
    for i in range(1,len(df.columns)):
        try:
            b=nltk.word_tokenize(df.iloc[clss,i])
            if sector in b:
                a.append([df.iloc[clss,i], df.columns[i]])
        except:
            continue
    return a

#chatbot body
def chat():
    print('My name is Robo and I am a chatbot! You can leave the chat by typing: quit')
    name= input('What is your name?  ',)
    print('@[*-*]@ : Hi '+name, ' How are you? ')
    threshold= 0.5
    schl= pd.read_csv('School.csv')
    school= None 
    
    while True:
        inpt= input('\n'+name+ ':  ')
        if inpt== 'quit':
            break
        result= model.predict([BoW(inpt,words)])
        indx= np.argmax(result)
        tag= classes[indx]
        for t in intents['intents']:
            if tag== t['tag'] and (np.max(result)- np.mean(result)> threshold):
                print('@[*-*]@ : ', random.choice(t['responses']))
                if tag in ['help','Next_Event', 'deadline', 'Exam', 'Lecture'] and school ==None:
                    #while school is None:
                        try:
                            school= int(input('@[*-*]@: please provide your school number from this list: \n1.IT \n2.Business \n3.Art \n4.Engineering \n5.Law\n'))-1
                        except:
                            print('@[*-*]@: sorry, it is not a valid number. I may ask you later to put it again')
                if tag== 'Next_Event':
                    fst_ev= next(i for i, j in enumerate([schl.iloc[school,b] for b in range(1,13)]) if type(j)==str)+1
                    print('@[*-*]@ : The closest one is: ', schl.iloc[(school),fst_ev], ' on ',schl.columns[fst_ev])
                elif tag== 'deadline':
                    print('@[*-*]@ : The deadline for assignment submission is ', ('11/2/13'if school !=3 else '12/2/13'))
                elif tag== 'Exam':
                    print('@[*-*]@ : your school exams scheduled for this semester is/are: ', sec_finder(schl,school,"Exam")  )
                elif tag== 'Lecture':
                    print('@[*-*]@ : your school Lecture scheduled for this semester is/are: ', sec_finder(schl,school,"Lecture")  )
            elif tag== t['tag'] and (np.max(result)- np.mean(result)< threshold):
                print('@[*-*]@ :I do not understand, would you please make it more clear')
    return        


chat()    
