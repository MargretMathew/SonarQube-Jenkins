#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import *
from sklearn import tree
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('train.csv')
df


# In[3]:


df['class'].unique()


# > in first 20000 rows there is only one class, and loading more is causing memory issues in later steps, so using small sections of multiple classes

# In[4]:


figurative = df[df['class'] == 'figurative']
figurative = figurative[:250]


# In[5]:


irony = df[df['class'] == 'irony']
irony = irony[:250]


# In[6]:


regular = df[df['class'] == 'regular']
regular = regular[:250]


# In[7]:


sarcasm = df[df['class'] == 'sarcasm']
sarcasm = sarcasm[:250]


# In[8]:


df = pd.concat([figurative, irony, regular, sarcasm])


# In[9]:


df = df.reset_index()
df


# In[10]:


df.drop('index', axis=1, inplace=True)


# In[11]:


df


# ### Pre-processing the tweets into bag of words also removal of stop words

# In[12]:


vectorizer = CountVectorizer(binary=True, stop_words="english")
X = vectorizer.fit_transform(df['tweets'])

df_tf_bag = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
df_tf_bag


# ### Examine mapping of words to feature indexes

# In[13]:


vectorizer.vocabulary_


# ### Forming matrix with bag of words

# In[14]:


X.todense()


# In[15]:


vectorizer


# ### TF-IDF

# In[16]:


vectorizer = TfidfVectorizer(norm="l1")
X = vectorizer.fit_transform(df['tweets'])
df_tf_idf = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
df_tf_idf


# In[17]:


df_tf_idf.apply(np.sum, axis=1)


# In[18]:


vectorizer = TfidfVectorizer(norm=None)
X = vectorizer.fit_transform(df['tweets'])
df_tf_none = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
df_tf_none


# ### Ngrams and also stop words removal

# In[19]:


vectorizer = CountVectorizer(ngram_range=(1,3), stop_words="english")
X = vectorizer.fit_transform(df['tweets'])
df_tf_ngram = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
df_tf_ngram


# In[20]:


nltk.download('stopwords')
nltk.download('all')


# In[21]:


stopwords = set(nltk.corpus.stopwords.words("english"))
stopwords


# ### Converting to lowercase, removing punctuation, tokenizing each words, POS tagging and removing stopwords from the 'tweets'

# In[22]:


lower = map(str.lower, df['tweets'])

no_punc = map(lambda x: re.sub("[^a-z]", " ", x), lower)

tokenized = map(nltk.word_tokenize, no_punc)

tagged = map(nltk.pos_tag, tokenized)

stopwords = nltk.corpus.stopwords.words("english")
def remove_stopwords(doc):
    out = []
    for word in doc:
        if word[0] not in stopwords: out.append(word)
    return out

no_stopwords = list(map(remove_stopwords, tagged))

no_stopwords


# ### Converting list of POS tagged words to string

# In[23]:


tagged_docs = list(map(str, no_stopwords))
tagged_docs


# In[24]:


vectorizer = CountVectorizer(token_pattern=r"\('[^ ]+', '[^ ]+'\)", lowercase=False)
X = vectorizer.fit_transform(tagged_docs)
df_tf_POS = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
df_tf_POS


# ### Applying machine learning models using bag of words: df_tf_bag

# ### Partition the dataset to 70% training data and 30% testing data

# In[25]:


train_x, test_x, train_y, test_y = train_test_split(df_tf_bag, df["class"], test_size=0.3, random_state=0)

df_train_x = pd.DataFrame(train_x, columns=df_tf_bag.columns)
df_test_x = pd.DataFrame(test_x, columns=df_tf_bag.columns)
df_train_y = pd.DataFrame(train_y, columns=["class"])
df_test_y = pd.DataFrame(test_y, columns=["class"])


# In[26]:


print (df_train_x.shape)
print (df_test_x.shape)
print (df_train_y.shape)
print (df_test_y.shape)


# In[27]:


print (df["class"].value_counts())
print (df_train_y["class"].value_counts())
print (df_test_y["class"].value_counts())


# ### Decision Tree

# In[28]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[29]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### LinearSVC

# In[30]:


clf = LinearSVC()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[31]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Multinomial Naive Bayes

# In[32]:


clf = MultinomialNB()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[33]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Logistic Regression

# In[34]:


clf = LogisticRegression()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[35]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Creating model with feature selection

# In[36]:


f_val, p_val = chi2(df_train_x, df_train_y["class"]) 

df_scores = pd.DataFrame(zip(df_tf_bag, f_val, p_val), columns=["feature", "chi2", "p"])
df_scores["chi2"] = df_scores["chi2"].round(2)
df_scores["p"] = df_scores["p"].round(3)

sel_cols = df_scores[df_scores["p"]<0.05]["feature"].values
print ("\nSelected features: %d" % len(sel_cols))
print (sel_cols)


# ### LinearSVC

# In[37]:


clf = LinearSVC()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[38]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Decision Tree

# In[39]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[40]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Logistic Regression

# In[41]:


clf = LogisticRegression()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[42]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Multinomial Naive Bayes

# In[43]:


clf = MultinomialNB()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[44]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Cross Validation and Fine Tuning the model with feature selection

# In[45]:



skf = StratifiedKFold(n_splits=5, random_state=None)

fold = 0
f1 = []
precision =[]
recall=[]
accuracy=[]
features = []
for train_index, test_index in skf.split(df_tf_bag,df["class"]):
    fold += 1
    print ("Fold %d" % fold)
    # partition
    train_x, test_x = df_tf_bag.iloc[train_index], df_tf_bag.iloc[test_index]
    train_y, test_y = df["class"].iloc[train_index], df["class"].iloc[test_index]
    
    # vectorize
    #vectorizer = CountVectorizer(tokenizer=tokenize, binary=True, stop_words='english')
    #X = vectorizer.fit_transform(train_x)
    X = train_x
    #X_test = vectorizer.transform(test_x)
    X_test = test_x
    
    # convert numpy arrays to data frames
    df_train_x = pd.DataFrame(train_x, columns=df_tf_bag.columns)
    df_test_x = pd.DataFrame(test_x, columns=df_tf_bag.columns)
    df_train_y = pd.DataFrame(train_y, columns=["class"])
    df_test_y = pd.DataFrame(test_y, columns=["class"])
    
    #feature selection
    f_val, p_val = chi2(df_train_x, df_train_y["class"]) 
    #f_val, p_val = chi2(train_x, train_y["ReviewRate "]) 

    # print the Chi-squared valus and p values
    df_scores = pd.DataFrame(zip(df_tf_bag.columns, f_val, p_val), columns=["feature", "chi2", "p"])
    df_scores["chi2"] = df_scores["chi2"].round(2)
    df_scores["p"] = df_scores["p"].round(3)
    #print df_scores.sort_values("chi2", ascending=False)

    # use features with p < 0.05
    sel_ohe_cols = df_scores[df_scores["p"]<0.05]["feature"].values
    
    # train model
    
    clf = LogisticRegression(random_state=fold)
    grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'accuracy')
    grid_clf_acc.fit(X[sel_ohe_cols], train_y)
    #grid_clf_acc.fit(X, train_y)
    
    
    
    #clf.fit(X[sel_ohe_cols], train_y)
    # predict
    pred=grid_clf_acc.predict(X_test[sel_ohe_cols])
    #pred = clf.predict(X_test[sel_ohe_cols])
    # classification results
    for line in metrics.classification_report(test_y, pred).split("\n"):
        print (line)
    f1.append(metrics.f1_score(test_y, pred, average="micro"))
    precision.append(metrics.precision_score(test_y, pred, average="micro"))
    recall.append(metrics.recall_score(test_y, pred, average="micro"))
    accuracy.append(metrics.accuracy_score(test_y, pred))
    #features.append(len(vectorizer.vocabulary_))
    
print ("Average F1: %.2f" % np.mean(f1))
print ("Average prcesion: %.2f" % np.mean(precision))
print ("Average recall: %.2f" % np.mean(recall))
print ("Average accuracy: %.2f" % np.mean(accuracy))
#print ("Average F1: %.2f" % np.mean(features))


# ### Artificial Neural Networks

# In[46]:


input_dim = train_x.shape[1]
input_dim


# In[47]:


model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[48]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[49]:


model.summary()


# In[50]:


mapper = {item: i for i, item in enumerate(train_y.unique())}
mapper


# In[51]:


train_y = train_y.map(mapper)
train_y


# In[52]:


test_y = test_y.map(mapper)
test_y


# In[53]:


train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
test_y = np.asarray(test_y)


# In[54]:


history = model.fit(train_x, train_y, epochs=100, verbose=False, validation_data=(test_x, test_y), batch_size=10)
history


# In[55]:


history.params


# In[56]:


pd.DataFrame(history.history)


# In[57]:


history.history['accuracy']


# In[ ]:





# In[58]:


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# In[59]:


plot_history(history)


# In[60]:


loss, accuracy = model.evaluate(train_x, train_y, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_x, test_y, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[ ]:





# ### Applying machine learning models using Ngram: df_tf_ngram

# In[ ]:





# ### Partition the dataset to 70% training data and 30% testing data

# In[61]:


train_x, test_x, train_y, test_y = train_test_split(df_tf_ngram, df["class"], test_size=0.3, random_state=0)

df_train_x = pd.DataFrame(train_x, columns=df_tf_ngram.columns)
df_test_x = pd.DataFrame(test_x, columns=df_tf_ngram.columns)
df_train_y = pd.DataFrame(train_y, columns=["class"])
df_test_y = pd.DataFrame(test_y, columns=["class"])


# In[62]:


print (df_train_x.shape)
print (df_test_x.shape)
print (df_train_y.shape)
print (df_test_y.shape)


# In[63]:


print (df["class"].value_counts())
print (df_train_y["class"].value_counts())
print (df_test_y["class"].value_counts())


# ### Decision Tree

# In[64]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[65]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### LinearSVC

# In[66]:


clf = LinearSVC()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[67]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Multinomial Naive Bayes

# In[68]:


clf = MultinomialNB()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[69]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Logistic Regression

# In[70]:


clf = LogisticRegression()
clf = clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)


# In[71]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Creating model with feature selection

# In[72]:


f_val, p_val = chi2(df_train_x, df_train_y["class"]) 

df_scores = pd.DataFrame(zip(df_tf_ngram, f_val, p_val), columns=["feature", "chi2", "p"])
df_scores["chi2"] = df_scores["chi2"].round(2)
df_scores["p"] = df_scores["p"].round(3)

sel_cols = df_scores[df_scores["p"]<0.05]["feature"].values
print ("\nSelected features: %d" % len(sel_cols))
print (sel_cols)


# ### LinearSVC

# In[73]:


clf = LinearSVC()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[74]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Decision Tree

# In[75]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[76]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Logistic Regression

# In[77]:


clf = LogisticRegression()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[78]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Multinomial Naive Bayes

# In[79]:


clf = MultinomialNB()
clf = clf.fit(df_train_x[sel_cols], train_y)
pred_y = clf.predict(df_test_x[sel_cols])


# In[80]:


print ("f1:" + str(f1_score(pred_y, test_y, average="micro")))
print ("accuracy:" + str(accuracy_score(pred_y, test_y)))
print ("precision:" + str(precision_score(pred_y, test_y, average="micro")))
print ("recall:" + str(recall_score(pred_y, test_y, average="micro")))


# ### Cross Validation and Fine Tuning the model with feature selection

# In[81]:



skf = StratifiedKFold(n_splits=5, random_state=None)

fold = 0
f1 = []
precision =[]
recall=[]
accuracy=[]
features = []
for train_index, test_index in skf.split(df_tf_ngram,df["class"]):
    fold += 1
    print ("Fold %d" % fold)
    # partition
    train_x, test_x = df_tf_ngram.iloc[train_index], df_tf_ngram.iloc[test_index]
    train_y, test_y = df["class"].iloc[train_index], df["class"].iloc[test_index]
    
    # vectorize
    #vectorizer = CountVectorizer(tokenizer=tokenize, binary=True, stop_words='english')
    #X = vectorizer.fit_transform(train_x)
    X = train_x
    #X_test = vectorizer.transform(test_x)
    X_test = test_x
    
    # convert numpy arrays to data frames
    df_train_x = pd.DataFrame(train_x, columns=df_tf_ngram.columns)
    df_test_x = pd.DataFrame(test_x, columns=df_tf_ngram.columns)
    df_train_y = pd.DataFrame(train_y, columns=["class"])
    df_test_y = pd.DataFrame(test_y, columns=["class"])
    
    #feature selection
    f_val, p_val = chi2(df_train_x, df_train_y["class"]) 
    #f_val, p_val = chi2(train_x, train_y["ReviewRate "]) 

    # print the Chi-squared valus and p values
    df_scores = pd.DataFrame(zip(df_tf_ngram.columns, f_val, p_val), columns=["feature", "chi2", "p"])
    df_scores["chi2"] = df_scores["chi2"].round(2)
    df_scores["p"] = df_scores["p"].round(3)
    #print df_scores.sort_values("chi2", ascending=False)

    # use features with p < 0.05
    sel_ohe_cols = df_scores[df_scores["p"]<0.05]["feature"].values
    
    # train model
    
    clf = LogisticRegression(random_state=fold)
    grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
    grid_clf_acc = GridSearchCV(clf, param_grid = grid_values,scoring = 'accuracy')
    grid_clf_acc.fit(X[sel_ohe_cols], train_y)
    #grid_clf_acc.fit(X, train_y)
    
    
    
    #clf.fit(X[sel_ohe_cols], train_y)
    # predict
    pred=grid_clf_acc.predict(X_test[sel_ohe_cols])
    #pred = clf.predict(X_test[sel_ohe_cols])
    # classification results
    for line in metrics.classification_report(test_y, pred).split("\n"):
        print (line)
    f1.append(metrics.f1_score(test_y, pred, average="micro"))
    precision.append(metrics.precision_score(test_y, pred, average="micro"))
    recall.append(metrics.recall_score(test_y, pred, average="micro"))
    accuracy.append(metrics.accuracy_score(test_y, pred))
    #features.append(len(vectorizer.vocabulary_))
    
print ("Average F1: %.2f" % np.mean(f1))
print ("Average prcesion: %.2f" % np.mean(precision))
print ("Average recall: %.2f" % np.mean(recall))
print ("Average accuracy: %.2f" % np.mean(accuracy))
#print ("Average F1: %.2f" % np.mean(features))


# ### Artificial Neural Networks

# In[83]:


input_dim = train_x.shape[1]
input_dim


# In[84]:


model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# In[85]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[86]:


model.summary()


# In[87]:


mapper = {item: i for i, item in enumerate(train_y.unique())}
mapper


# In[88]:


train_y = train_y.map(mapper)
train_y


# In[89]:


test_y = test_y.map(mapper)
test_y


# In[90]:


history = model.fit(train_x, train_y, epochs=100, verbose=False, validation_data=(test_x, test_y), batch_size=10)
history


# In[91]:


history.params


# In[92]:


pd.DataFrame(history.history)


# In[93]:


history.history['accuracy']


# In[ ]:





# In[94]:


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# In[95]:


plot_history(history)


# In[96]:


loss, accuracy = model.evaluate(train_x, train_y, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_x, test_y, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

