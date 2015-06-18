import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
from sklearn.linear_model import LogisticRegression as LR


stop = set(stopwords.words('english'))
vectorizer = None

def filter(review):
    res = BeautifulSoup(review)
    res = re.sub('[^a-zA-Z]',' ',res.get_text())
    words = res.lower().split()
    #words =[w for w in words if w not in stop]
    return ' '.join(words)
    

def readFile(filename):
    
    global vectorizer
    
    train_data = pd.read_csv(filename, header=0, delimiter='\t', quoting=3)
    train_size = train_data.shape[0]
    
    
    
    clean_train = []
    for i in xrange(0,train_size):
        clean_train.append(filter(train_data['review'][i]))
        #if i%1000 ==0:
            #print '%d reviews processed...' %i
    from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
    
    #vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)
    if vectorizer==None:
        vectorizer = TfidfVectorizer(sublinear_tf=True,ngram_range = ( 1, 3 ), max_features=40000)
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
    forest = RandomForestClassifier(n_estimators = 100, max_features = 'auto',  random_state = 50)
    forest = forest.fit(x,y)
    return forest

def forest_predict(test_file,forest):
    id,x = readFile(test_file)
    y = forest.predict(x)
    
    output = pd.DataFrame( data={"id":id, "sentiment":y} )
    output.to_csv( "/home/kehl/Desktop/random_forest.csv", index=False, quoting=3 )
    
def svm_train(train_file):
    _,x,y = readFile(train_file)
    
    from sklearn.cross_validation import train_test_split
    tmp_array = np.arange(x.shape[0])
    train_i, test_i = train_test_split(tmp_array, train_size = 0.8, random_state = 44)
    
    train_x = x[train_i]
    test_x = x[test_i]
    train_y = y[train_i]
    test_y = y[test_i]
    #from sklearn.svm import SVC
    #classifier = SVC(C=1.0,gamma = 0.0)
    #classifier.fit(train_x,train_y)
    
    lr = LR()
    lr.fit(train_x,train_y)
    #res = lr.predict_proba(test_x)
    res = lr.predict_proba(test_x)
    print res.shape
    from sklearn.metrics import roc_auc_score
    score = roc_auc_score(test_y,res[:,1])
    print score
    return lr
    
    '''
    grid search
    
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
    '''

def svm_predict(test_file,model):
    id,x = readFile(test_file)
    y = model.predict_proba(x)
    output = pd.DataFrame( data={"id":id, "sentiment":y[:,1]} )
    output.to_csv( "/home/chuangxin/LR_result.csv", index=False, quoting=3 )
    
if __name__=='__main__':
    model = svm_train('/home/chuangxin/labeledTrainData.tsv')
    svm_predict('/home/chuangxin/testData.tsv',model)
