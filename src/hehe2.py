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
        vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5)
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
    forest = RandomForestClassifier(100)
    forest = forest.fit(x,y)
    return forest

def forest_predict(test_file,forest):
    id,x = readFile(test_file)
    y = forest.predict(x)
    
    output = pd.DataFrame( data={"id":id, "sentiment":y} )
    output.to_csv( "/Users/oliverkehl/Desktop/Bag_of_Words_model_tfidf.csv", index=False, quoting=3 )
    
def nn_train(train_file):
    _,x,y = readFile(train_file)
    import sklearn.neural_network as nn
    model = nn.BernoulliRBM(n_components = 200, learning_rate = 0.1,batch_size = 2000, n_iter = 50, random_state = 1 , verbose=1)
    model = model.fit(x,y)
    return model

def nn_predict(test_file,forest):
    id,x = readFile(test_file)
    y = forest.predict(x)
    output = pd.DataFrame( data={"id":id, "sentiment":y} )
    output.to_csv( "/Users/oliverkehl/Desktop/svm_result.csv", index=False, quoting=3 )
    
if __name__=='__main__':
    model = nn_train('/Users/oliverkehl/Downloads/labeledTrainData.tsv')
    nn_predict('/Users/oliverkehl/Downloads/testData.tsv',model)
    
    
    
    
    
    
    
    