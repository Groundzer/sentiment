import pandas as pd
df = pd.read_csv('data_sentiment_IPD.csv')
#print(df.head())
import numpy as np
import pythainlp
from deepcut import tokenize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import re
import itertools
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
from nltk import NaiveBayesClassifier as nbc
import codecs
from itertools import chain

accuracy = 0.0
while(accuracy <= 0.9):
      df = df[df['Summary'] != 0]
      df['Summary'] = df['Summary'].apply(lambda rating : 1 if rating == 1 else float(-1))
      index = df.index
      df['random_number'] = np.random.randn(len(index))
      train = df[df['random_number'] <= 0.80]
      test = df[df['random_number'] > 0.80]
      #print(test.head())

      pattern = r'[0-9]'
      patternN = '\n'

      positive = train[train['Summary'] == 1]
      negative = train[train['Summary'] == -1]
      #print(positive.count(1!=1))
      #print(negative.count(1!=1))

      ###List Positive
      li_pos = []
      for i in positive.sentiment:
            i = re.sub(pattern, '', i)
            i = re.sub(patternN, '', i)
            #i = re.sub(pattern_punctuation, '', i)
            words = pythainlp.tokenize.word_tokenize(i, engine='deepcut')
            removetable = str.maketrans('', '', '@#%`~!$%^&*()_-+={[}}|\:;"<>.?/')
            words = [s.translate(removetable) for s in words]
            li_pos.append(words)
            #print(words)


      li_pos = list(itertools.chain(*li_pos))

      #print(li)

      stopwords = set(thai_stopwords())
      stopwords.update([" ", "\u200b", '', '❤', '️️', '️', '😆'])

      list_pos = [i for i in li_pos if i not in stopwords]
      list_pos = list(dict.fromkeys(list_pos))
      #print(list_pos)

      #print("-------------------------------------------------------------")

      ###List Negative

      li_neg = []
      for i in negative.sentiment:
            i = re.sub(pattern, '', i)
            i = re.sub(patternN, '', i)
            words = pythainlp.tokenize.word_tokenize(i, engine='deepcut')
            removetable = str.maketrans('', '', '@#%`~!$%^&*()_-+={[}}|\:;"<>.?/')
            words = [s.translate(removetable) for s in words]
            #print("words : ", words)
            li_neg.append(words)
            #print(words)

      li_neg = list(itertools.chain(*li_neg))

      list_neg = [i for i in li_neg if i not in stopwords]
      list_neg = list(dict.fromkeys(list_neg))

      #print(list_neg)

      ### Model

      pos1=['pos']*len(list_pos)
      neg1=['neg']*len(list_neg)
      training_data = list(zip(list_pos,pos1)) + list(zip(list_neg,neg1))
      vocabulary = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))
      feature_set = [({i:(i in word_tokenize(sentence.lower())) for i in vocabulary},tag) for sentence, tag in training_data]
      #print(feature_set)
      classifier = nbc.train(feature_set)


      result = []
      for summary in test.Summary:
            if summary == 1:
                  result.append("pos")
            elif summary == -1:
                  result.append("neg")
            else:
                  result.append("0")
      #print(result)
      tp = tn = fn = fp = i = 0
      for test_sentence in test.sentiment:
            #test_sentence = input('\nข้อความ : ')
            featurized_test_sentence =  {i:(i in word_tokenize(test_sentence.lower())) for i in vocabulary}
            #print("test_sent:",test_sentence)
            tag_test = classifier.classify(featurized_test_sentence)
            #print(tag_test, type(tag_test))
            #print("tag:", tag_test) # ใช้โมเดลที่ train ประมวลผล
            #print(' | ', tag_test)
            if result[i] == 'pos' and tag_test == 'pos':
                  tp+=1
            elif result[i] == "neg" and tag_test == 'neg':
                  tn+=1
            elif result[i] == "pos" and tag_test == 'neg':
                  fn+=1
            elif result[i] == "neg" and tag_test == 'pos':
                  fp+=1
            else:
                  print("skip")     

            i+=1

      print('           A Positive  | A Negative')
      print('P Positive  ', tp, '        | ', fp)
      print('P Negative  ', fn, '        | ', tn)

      accuracy = (tp + tn) / (tp + tn + fp + fn)
      print('Accuracy = ', accuracy)

import joblib
filename = 'finalized_model.sav'
joblib.dump(classifier, filename)
