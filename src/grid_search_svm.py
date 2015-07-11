import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

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
        if i%1000 ==0:
            print '%d reviews processed...' %i
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    if vectorizer==None:
        vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.9,ngram_range=(1,3),max_features=100000)
        train_data_feature = vectorizer.fit_transform(clean_train)
    else:
        vec = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
        train_data_feature = vec.fit_transform(clean_train)
        

    print train_data_feature.shape
    if 'test' in filename:
        return train_data['id'], train_data_feature
    else:
        return train_data['id'], train_data_feature, train_data['sentiment']

def forest_train(train_file):
    _,x,y = readFile(train_file)
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(100)
    forest = forest.fit(x,y)
    return forest

def forest_predict(test_file,forest):
    id,x = readFile(test_file)
    y = forest.predict(x)
    
    output = pd.DataFrame( data={"id":id, "sentiment":y} )
    output.to_csv( "/Users/oliverkehl/Desktop/Bag_of_Words_model_tfidf.csv", index=False, quoting=3 )

def svm_train(train_file,test_file):
    _,x,y = readFile(train_file)
    id, tx = readFile(test_file)
    #feature selection
    from sklearn.feature_selection import SelectKBest,chi2
    fselect = SelectKBest(chi2, k =5000)
    x = fselect.fit_transform(x,y)
    tx = fselect.transform(tx)

    print x.shape
    print tx.shape    

    hehe = np.concatenate((x,tx))
    from sklearn.preprocessing import scale
    hehe = scale(hehe,with_mean=False)
    x = hehe[0:x.shape[0]]
    tx = hehe[x.shape[0]:]
    
    from sklearn.cross_validation import train_test_split
    tmp_array = np.arange(x.shape[0])
    train_i,test_i = train_test_split(tmp_array, train_size = 0.8, random_state = 1024)
    train_x = x[train_i]
    train_y = y[train_i]
    test_x = x[test_i]
    test_y = y[test_i]

    from sklearn.svm import SVC
    model = SVC(probability=True)
    model.fit(x,y)
    
    res1 = model.predict_proba(train_x)
    res2 = model.predict_proba(test_x)
    from sklearn.metrics import roc_auc_score
    score1 = roc_auc_score(train_y, res1[:,1])
    score2 = roc_auc_score(test_y, res2[:,1])
    print score1
    print score2

    res = model.predict_proba(tx)
    output = pd.DataFrame( data={"id":id, "sentiment":res[:,1]} )
    output.to_csv( "/home/chuangxin/SVM_result.csv", index=False, quoting=3 )
    
    return model
       
if __name__=='__main__':
    model = svm_train('/home/chuangxin/labeledTrainData.tsv','/home/chuangxin/testData.tsv')
