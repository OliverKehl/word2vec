import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re


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
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    
    #vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    if vectorizer==None:
        vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5, max_features=50000)
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

def forest_train(train_file):
    _,x,y = readFile(train_file)
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 100, max_features = 'auto', random_state = 50)
    forest = forest.fit(x,y)
    return forest

def forest_predict(test_file,forest):
    id,x = readFile(test_file)
    y = forest.predict(x)
    
    output = pd.DataFrame( data={"id":id, "sentiment":y} )
    output.to_csv( "/home/kehl/Desktop/random_forest.csv", index=False, quoting=3 )
    
def svm_train(train_file):
    _,x,y = readFile(train_file)
    
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    from sklearn.cross_validation import StratifiedShuffleSplit
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv,n_jobs = 3)
    grid.fit(x, y)
    print grid.best_params_
    print grid.best_score_
    return grid.best_estimator_

def svm_predict(test_file,forest):
    id,x = readFile(test_file)
    y = forest.predict(x)
    output = pd.DataFrame( data={"id":id, "sentiment":y} )
    output.to_csv( "/Users/oliverkehl/Desktop/svm_gridsearch_result.csv", index=False, quoting=3 )
    
if __name__=='__main__':
    model = svm_train('/Users/oliverkehl/Downloads/labeledTrainData.tsv')
    svm_predict('/Users/oliverkehl/Downloads/testData.tsv',model)
