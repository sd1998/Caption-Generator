import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import listdir
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

def extract_features(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(intput = model.inputs,output=model.layers[-1].output)
    features = dict()
    for filename in listdir(directory):
        path = directory + '/' + filename
        image = load_img(path,target_size=(224,224))
        image = img_to_array(image)
        image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
        image = preprocess_input(image)
        feature = model.predict(image,verbose=0)
        image_id = filename.split('.')[0]
        features[image_id] = feature
    return features

directory = 'Flicker8k_Dataset'
features = extract_features(directory)
dump(features,open('features.pkl','wb'))

file = open('Flickr8k_text/Flickr8k.token.txt','r')
descr = file.read()
file.close()

description = dict()
for line in descr.split('\n'):
    tokens = line.split()
    if len(line) < 2:
        continue
    image_id = tokens[0]
    image_desc = tokens[1:]
    image_id = image_id.split('.')[0]
    image_desc = ' '.join(image_desc)
    if image_id not in description:
        description[image_id] = list()
    description[image_id].append(image_desc)    
    
import string

def preprocess_descriptions(description):
    table = str.maketrans('','',string.punctuation)
    for key,desc_list in description.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            desc = desc.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            desc_list[i] = ' '.join(desc)
            
preprocess_descriptions(description)

desc = set()
for key in description.keys():
    [desc.update(word.split()) for word in description[key]]
vocabulary = desc

lines = list()
for key,desc_list in description.items():
    for desc in desc_list:
        lines.append(key + ' ' + desc)
data = '\n'.join(lines)
file = open('description.txt','w')
file.write(data)
file.close()

from keras.layers import Dense,LSTM,Embedding
from keras.models import Model

file = open('Flickr8k_text/Flickr_8k.trainImages.txt','r')
dataset = file.read()
file.close()

train_dataset = list()
for doc in dataset.split('\n'):
    if len(doc) < 1:
        continue
    image_id = doc.split('.')[0]
    train_dataset.append(image_id)
    
file = open('description.txt','r')
description = file.read()
file.close()

descriptions = dict()
for line in description.split('\n'):
    tokens = line.split()
    image_id = tokens[0]
    image_desc = tokens[1:]
    if image_id in train_dataset:
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descript = 'startseq ' + ' '.join(image_desc) + ' endseq'
        descriptions[image_id].append(descript)
        
image_features = load(open('features.pkl','rb'))
features = {k:image_features[i] for i in train_dataset}

lines = list()
for key in descriptions.keys():
    [lines.append(val) for val in  descriptions[key]]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
vocab_size = len(tokenizer.wod_index) + 1

def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

max_length = max_length(descriptions)

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

X1_train,X2_train,y_train = list(),list(),list()
for key,desc_list in descriptions.items():
    for desc in desc_list:
        seq = tokenizer.texts_to_sequence([desc])[0]
        for i in range(1,len(seq)):
            in_seq,out_seq = seq[:1],seq[i]
            in_seq = pad_sequences([in_seq],max_length=max_length)[0]
            out_seq = to_catgorical([out_seq],num_classes=vocab_size)[0]
            X1_train.append(image_features[key][0])
            X2_train.append(in_seq)
            y_train.append(out_seq)

input1 = Input(shape=(4096,))
drop1 = Dropout(0.3)(input1)
dense1 = Dense(256,activation='relu')(drop1)
input2 = Input(shape=(max_length,))
embedding = Embedding(vocab_size,256,mask_zero=True)(input2)
drop2 = Dropout(0.3)(embedding)
lstm = LSTM(256)(drop2)
decoder1 = add([desne1,lstm])
dense2 = Dense(256,activation='relu')(decoder1)
output = Dense(vocab_size,activation='softmax')(dense2)
model = Model(inputs=[input1,input2],outputs=output)

model.compile(loss='categorical_crossentropy',optimizer='adam')

model.fit([X1_train,X2_train],y_train,epochs=10,validation_data=([X1_test,X2_test],y_test))
