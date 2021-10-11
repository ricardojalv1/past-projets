# -*- coding: utf-8 -*-
"""
@author: Ricardo Alvarez
"""

import pandas as pd, numpy as np, sklearn, nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import *
nltk.download('wordnet')

# Open the positive reviews file as a list of strings (reviews separated by a new line)
pos_list = open("rt-polaritydata/rt-polarity.pos", "r").readlines()

# Open the negative reviews file as a list of strings (reviews separated by a new line)
neg_list = open("rt-polaritydata/rt-polarity.neg", "r").readlines()


# Convert pos_list to dataframe
pos_df = pd.DataFrame (pos_list, columns = ['Review'])
# Convert neg_list to dataframe
neg_df = pd.DataFrame (neg_list, columns = ['Review'])

# Add an indication for positive review (1)
pos_df['Labels'] = 1
# Add an indication for negative review (0)
neg_df['Labels'] = 0

#Combine the positive and negative dataframes into one and shuffle rows
combined_df = pd.concat([pos_df, neg_df]).sample(frac=1)
#Reset index of rows
combined_df = combined_df.reset_index(drop=True)

test_data = combined_df[int(len(combined_df)- (0.1*(len(combined_df)))):]

df_train_val = combined_df[:int(0.9*(len(combined_df)))]


def porter(text):
    
    return [nltk.stem.porter.PorterStemmer().stem(word) for word in nltk.word_tokenize(text)]

def snowball(text):
    
    return [nltk.stem.snowball.SnowballStemmer("english").stem(word) for word in nltk.word_tokenize(text)]

def lemma(text):
    
    return [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in nltk.word_tokenize(text)]
       
    

cv_options = [
    CountVectorizer(), \
    CountVectorizer(binary=True), \
    CountVectorizer(binary=True, stop_words='english', analyzer = 'word'), \
    CountVectorizer(binary=True, stop_words='english', analyzer = 'word', min_df = 10, max_df = 0.90), \
    CountVectorizer(binary=True, stop_words='english', analyzer = porter, min_df = 10, max_df = 0.90), \
    CountVectorizer(binary=True, stop_words='english', analyzer = snowball, min_df = 10, max_df = 0.90), \
    CountVectorizer(binary=True, stop_words='english', analyzer = lemma, min_df = 10, max_df = 0.90), \
    CountVectorizer(binary=True, stop_words='english', analyzer = 'word', min_df = 10, max_df = 0.90, ngram_range=(1,2)), \
    CountVectorizer(binary=True, analyzer = 'word', ngram_range=(1,2)), \
    CountVectorizer(binary=True, analyzer = 'word', ngram_range=(2,3)), \
    CountVectorizer(binary=True, tokenizer = nltk.word_tokenize, analyzer = 'word', ngram_range=(1,2)), 
                ]


#Uncommented and indent rest of code when doing validation for model parameters selection 
#for cv_option in cv_options:

    #print(str(cv_option)) 
    
    
#uncomment when validating. 90% * 11% is roughly 10% of total data (used as validation set)
#train, test = sklearn.model_selection.train_test_split(df_train_val, test_size = 0.11, shuffle = True, stratify = df_train_val['Labels'], random_state=123)
train, test = sklearn.model_selection.train_test_split(combined_df, test_size = 0.10, stratify = combined_df['Labels'], random_state=123)

#Uncomment when validating 
#cv = cv_option
cv = CountVectorizer(binary=True, stop_words='english', analyzer = snowball, min_df = 10, max_df = 0.90)
cv.fit_transform(train['Review'].values)
train_feature_set = cv.transform(train['Review'].values)
test_feature_set = cv.transform(test['Review'].values)
y_train = train['Labels'].values
y_test = test['Labels'].values

# tried penalty = 'l1', default = 'l2'
lr = sklearn.linear_model.LogisticRegression(solver = 'liblinear', random_state=123, max_iter=1000)

lr.fit(train_feature_set,y_train)

y_predicted = lr.predict(test_feature_set)

feature_importance = lr.coef_[0]


sorted_feature = np.argsort(feature_importance)
top_10_pos = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(w)] for w in sorted_feature[range(-1,-11, -1)]]
print("Top 10 positive words: ")
print(top_10_pos)

top_10_neg = [list(cv.vocabulary_.keys())[list(cv.vocabulary_.values()).index(w)] for w in sorted_feature[:10]]
print("Top 10 negative words: ")
print(top_10_neg)

print("Accuracy: ",round(sklearn.metrics.accuracy_score(y_test,y_predicted),3))
print("F1: ",round(sklearn.metrics.f1_score(y_test, y_predicted),3))
print("\n")

cmatrix = sklearn.metrics.plot_confusion_matrix(lr, test_feature_set, y_test,
                                 display_labels=['0', '1'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')

cmatrix.ax_.set_title('Confusion Matrix')