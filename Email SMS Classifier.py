#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#df = pd.read_csv("C:\\Users\\welcome\\Downloads\\DataFrame\\spam.csv")
#Try reading the CSV file with a different encoding
df = pd.read_csv('C:\\Users\\welcome\\Downloads\\DataFrame\\spam.csv', encoding='ISO-8859-1')
# or
#df = pd.read_csv('spam.csv', encoding='cp1252')


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[5]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# # 1.Data Cleaning

# In[6]:


df.info()


# In[7]:


#drop last 3 cols
df.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)


# In[8]:


df.head()


# In[9]:


#renaming the cols

df.rename(columns ={'v1':'target','v2':'text'},inplace = True)


# In[10]:


df.head()


# In[11]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[12]:


df['target']  = encoder.fit_transform(df['target'])


# In[13]:


df.head()


# In[14]:


#missing value
df.isnull().sum()


# In[15]:


#check for duplicate values
df.duplicated().sum()


# In[16]:


# remove duplicates
df = df.drop_duplicates(keep = 'first')


# In[17]:


df.duplicated().sum()


# In[18]:


df.shape


# # 2.EDA

# In[19]:


df.head()


# In[20]:


df['target'].value_counts()


# In[21]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels = ['ham','spam'],autopct="%0.2f")
plt.show()


# In[22]:


#data is imbalanced


# In[49]:


import nltk
import re
nltk.download('stopwords')


# In[50]:


pip install nltk


# In[51]:


nltk.download('punkt')


# In[52]:


df['num_characters'] = df['text'].apply(len)
df.head()


# In[53]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[54]:


df.head()


# In[55]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))
df.head()


# In[56]:


df[['num_characters','num_words','num_sentences']].describe()


# In[57]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[58]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[59]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[60]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[61]:


sns.pairplot(df,hue='target')


# In[62]:


sns.heatmap(df.corr(),annot=True)


# In[69]:


from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
ps = PorterStemmer()


# # 3. Data Preprocessing
# 1 Lower case
# 2 Tokenization
# 3 Removing special characters
# 4 Removing stop words and punctuation
# 5 Stemming

# In[70]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[ ]:





# In[71]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[72]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[73]:


df['transformed_text'] = df['text'].apply(transform_text)
df.head()


# In[74]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[75]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[76]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[77]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[78]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[79]:


df.head()


# In[80]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[81]:


len(spam_corpus)


# In[83]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[84]:


len(ham_corpus)


# In[86]:


# Text Vectorization
# using Bag of Words
df.head()


# # 4. Model Building

# In[87]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[88]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[90]:


X.shape


# In[91]:


y = df['target'].values


# In[92]:


from sklearn.model_selection import train_test_split


# In[93]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[94]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[95]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[96]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[97]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[98]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[99]:


# tfidf --> MNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[100]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[101]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[102]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[103]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[104]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[105]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df


# In[106]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1


# In[107]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[108]:


# model improve
# 1. Change the max_features parameter of TfIdf
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)
new_df = performance_df.merge(temp_df,on='Algorithm')
new_df_scaled = new_df.merge(temp_df,on='Algorithm')
temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[109]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[110]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[111]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[ ]:


voting.fit(X_train,y_train)


# In[ ]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[ ]:


from sklearn.ensemble import StackingClassifier


# In[ ]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[ ]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[ ]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




