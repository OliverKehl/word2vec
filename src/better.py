import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

stop = set(stopwords.words('english'))
vectorizer = None

def filter(review):
    res = BeautifulSoup(review)
    res = re.sub('[^a-zA-Z]',' ',res.get_text())
    words = res.lower().split()
    words =[w for w in words if w not in stop]
    return ' '.join(words)
    

def readFile(filename):
    
    global vectorizer
    
    train_data = pd.read_csv(filename, header=0, delimiter='\t', quoting=3)
    train_size = train_data.shape[0]
    
    
    
    clean_train = []
    for i in xrange(0,train_size):
        clean_train.append(filter(train_data['review'][i]))
        #if i%1000 ==0:
        #    print '%d reviews processed...' %i
   
    
    #vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    if vectorizer==None:
        vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5, max_features = 50000)
        train_data_feature = vectorizer.fit_transform(clean_train)
    else:
        vec = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
        train_data_feature = vec.fit_transform(clean_train)
        

    print train_data_feature.shape
    if 'test' in filename:
        return train_data['id'], train_data_feature
    else:
        return train_data['id'], train_data_feature, train_data['sentiment']
    #test = pd.read_csv('',header=0,delimiter='\t', quoting=3)

def train(train_file):
    _,x,y = readFile(train_file)
    tmp_array = np.arange(x.shape[0])
    res_forest = None
    res_score = 0
    for i in range(10):
        print 'loop %d ...' %(i+1)
        train_i, test_i = train_test_split(tmp_array, train_size = 0.8, random_state = i)
        train_x = x[train_i]
        train_y = y[train_i]
        test_x = x[test_i]
        test_y = y[test_i]
    
        forest = RandomForestClassifier(n_estimators = 200, max_features = 0.2, random_state = 50)
        forest = forest.fit(train_x,train_y)
        res = forest.predict(test_x)
        score = roc_auc_score(test_y, res)
        if(score>res_score):
            res_forest = forest
            res_score = score
            print 'Num %d forest used...'%(i+1)
        del forest,train_i,test_i,train_x,train_y,test_x,test_y
    return res_forest

def predict(test_file,forest):
    mid,x = readFile(test_file)
    y = forest.predict(x)
    
    output = pd.DataFrame( data={"id":mid, "sentiment":y} )
    output.to_csv( "/home/kehl/Desktop/cross.csv", index=False, quoting=3 )
    
if __name__=='__main__':
    model = train('/home/kehl/Desktop/labeledTrainData.tsv')
    predict('/home/kehl/Desktop/testData.tsv',model)
    print 'done...'
    