import tensorflow_hub as hub
import os
import tensorflow as tf
import time
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def vectorize(df):    
    """
    :param df: Dataframe; Processed training DataFrame 
    :return: target_exp: list[str]; List of target expansions
    :return: regressors: list[list[int]]: List of vectors from tokens
    Returns target expansions and vectors of corresponsing articles
    """
    
    tagged = df.apply(lambda x: TaggedDocument(words = x['TOKEN'], tags = [x['LABEL']]), axis=1)
    
    vect = Doc2Vec(dm=0, vector_size=100, min_count=2, window = 2)
    vect.build_vocab(tagged.values)
    vect.train(tagged.values, total_examples=len(tagged.values), epochs=30)    
    
    target_exp = [doc.tags[0] for doc in tagged.values]    
    
    regressors = [vect.infer_vector(doc.words) for doc in tagged.values]  
    
    return target_exp, regressors, vect 

def train(x, y, exp2id):
    """
    :param x: np.array[str]
    :param y: np.array[int]
    :param exp2id: dict; used to determine the output layer structure
    :return: tf.keras.Sequential
    Train and return the model.
    """
    start_train = time.time()
    hub_layer = hub.KerasLayer("https://tfhub.dev/google/nnlm-en-dim50/2", input_shape=[], dtype=tf.string,
                               trainable=True)
    model = tf.keras.Sequential([
        hub_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(exp2id), activation='softmax')
    ])
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_val = x[:x.shape[0] // 10]
    partial_x_train = x[x.shape[0] // 10:]

    y_val = y[:y.shape[0] // 10]
    partial_y_train = y[y.shape[0] // 10:]

    model.fit(partial_x_train,
              partial_y_train,
              epochs=30,
              batch_size=2048,
              validation_data=(x_val, y_val),
              verbose=1)

    end_train = time.time()
    print(f'Time elapsed for training the data: {end_train - start_train}')
    # model.save(os.path.join('..', 'Models', 'my_model'))
    return model




def train_lr(train):
    """
    :param train: Dataframe; Processed training DataFrame
    :return: sklearn.linear_model.LogisticRegression
    :return abv_dict: dict{str, str}; Dictionnary of expansions (keys) to abbreviations (values)
    :return: vect: Doc2Vec model trained and used to predict abbreviation expansions
    Trains and returns the logictic regression model and a trained Doc2Vec model.
    """

    y_train, X_train, vect = vectorize(train)
    
    # For cross-validation purposes
    """
    #0.001, 0.01, 0.1, 1, 10, 100
    #['sag', 'saga','lbfgs','newton-cg']
    param_grid = {'C':[0.1, 1, 10], \
                  'solver':['sag', 'lbfgs']}
        
    grid_model = GridSearchCV(LogisticRegression(n_jobs=-1, random_state=42), param_grid, verbose=10, cv=3, scoring='accuracy', n_jobs = -1)   
    grid_model.fit(X_train, y_train)

    
    print(grid_model.best_params_)
    print(grid_model.best_score_)
    print(grid_model.cv_results_)
    
    """
    lr = LogisticRegression(C=10, solver='saga', n_jobs=-1, random_state=42, verbose=10)
    #lr = LogisticRegression(n_jobs=-1, C=1)
    lr.fit(X_train, y_train)
    
    # Create abbreviation dictionary 
    abvs = train[["LABEL", "ABV"]].drop_duplicates()
    abvs_dict = dict(zip(abvs.LABEL, abvs.ABV))

    return lr, abvs_dict, vect
    
    
def train_rf(train):
    """
    :param train: Dataframe; Processed training DataFrame
    :return: sklearn.ensemble.RandomForestClassifier
    :return: vect: Doc2Vec model trained and used to predict abbreviation expansions
    Trains and returns the Random Forest Classifier model and a trained Doc2Vec model.
    """
    y_train, X_train, vect = vectorize(train)
    
    # For cross-validation purposes 
    """
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = { 
    'n_estimators': [100,1000, 10000],
    'max_features': ['auto', 'log2'],
    'max_depth' : [7,8], 
    'criterion': ['entropy', 'gini']
    }
    
    grid_model = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3, verbose=10, n_jobs=-1, scoring='accuracy')
    grid_model.fit(X_train, y_train)
    
    print(grid_model.best_params_)
    print(grid_model.best_score_)
    print(grid_model.cv_results_)
    """
    
    
    rfc = RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 1000, max_depth=8, criterion='gini', verbose=10, n_jobs=-1)
    rfc.fit(X_train, y_train)
    
    return rfc, vect