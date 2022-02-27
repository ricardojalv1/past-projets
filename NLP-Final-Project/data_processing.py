import pandas as pd
import os
import numpy as np
from collections import defaultdict, Counter
import warnings
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

import pickle

ADAM_PATH = os.path.join('..', 'Data', 'valid_adam.txt')
DATA_PATH = os.path.join('..', 'Data', 'train.csv')
DATA_PATH_SAMPLE = os.path.join('..', 'Data', 'sample.csv')
DATA_PATH_VALID = os.path.join('..', 'Data', 'valid.csv')
DATA_PATH_TEST = os.path.join('..', 'Data', 'test.csv')

warnings.filterwarnings("ignore")
TEST_SIZE = 0.3

def encode_expansions(acro2exp):
    """
    :param acro2exp: dict[str, list[str]]; a dictionary mapping acronyms to their possible expansions.
    :return: dict[str, int]
    Encode all expansions into int form. Return the mapping of expansion to its id.
    """
    cur_id = 0
    exp2id = {}
    for acronym, expansions in acro2exp.items():
        for exp in expansions:
            if exp not in exp2id:
                exp2id[exp] = cur_id
                cur_id += 1
    return exp2id


def get_windowed_x_and_acronyms(articles, locations, radius=15):
    """
    :param articles: list[str]
    :param locations: list[int]
    :param radius: int
    :return: np.array[str], list[str]
    Take the articles and the locations of the acronyms in these articles, modify the articles to the form of
    concatenation of {radius} words before the acronym and {radius} words after the acronym.
    Return the modified articles and the acronyms
    """
    acronyms = []
    for i, (article, location) in enumerate(zip(articles, locations)):
        lst = article.split(' ')
        start = max(0, location - radius)
        end = min(location + radius + 1, len(lst))
        window = ' '.join(lst[start:location] + lst[location + 1:end])
        articles[i] = window
        acronyms.append(lst[location])
    return np.array(articles), acronyms


def get_acronym_to_expansion_dict(path):
    """
    :param path: ADAM file path
    :return: dict[str, list[str]]
    Read the dictionary that maps the acronyms their possible expansions.
    Return the mapping.
    """
    df = pd.read_csv(path, sep='\t')
    acro2exp = defaultdict(list)
    for acro, exp in zip(df.PREFERRED_AB, df.EXPANSION):
        acro2exp[acro].append(exp)
    return acro2exp


def get_expansion_range(acronyms, exp2id, acro2exp):
    """
    :param acronyms: list[str]; Acronyms to be disambiguated
    :param exp2id: dict[str, int]; The mapping of expansion to its id.
    :param acro2exp: dict[str, list[str]]; The mapping of acronyms to their possible expansions.
    :return: list[list[int]]
    Return each acronym's possible expansion ids.
    """
    expansions_range = []
    for acro in acronyms:
        expansions = acro2exp[acro]
        expansions_range.append([exp2id[exp] for exp in expansions])
    return expansions_range

    
def create_abv_col(df):
    """
    :param df: DataFrame: dataframe from which to create abbreviation column
    :return: list[str] 
    Returns a list of abbreviations.
    """    
    return [text.split(' ')[loc] for text,loc in zip(df['TEXT'], df['LOCATION'])]

def create_tokens(df):
    """
    :param df: DataFrame: dataframe from which to tokenize text
    :return: list[list[str]]
    Returns a list of lists of tokens.
    """   
    return df['TEXT'].apply(lambda x: x.split(' '))

def drop_cols(df):
    """
    :param df: DataFrame: dataframe from which to drop useless columns
    :return: df: DataFrame
    Returns a DataFrame without unnecessary columns.
    """  
    return df.drop(columns=['ABSTRACT_ID', 'LOCATION', 'TEXT'])


def remove_stopwords(df):
    """
    :param df: DataFrame: dataframe from which to remove stopwords
    :return: df: DataFrame
    Returns a DataFrame without unnecessary columns.
    """  
    stopWords = stopwords.words('english')
    # Ignore stopwords that overlap with abbreviations 
    [stopWords.remove(w) for w in df['ABV'].str.lower() if w in stopWords]
    
    return df['TOKEN'].apply(lambda i: [item for item in i if not item in stopWords])


def process_data(path, load=False, save=False):
    """
    :param path: Data file path
    :param load: True/False; if True, save and overwrite the returned variables using pickle module under ../Data/
                             so they can be read without processing next time. Please refer the code for file names
    :param save: True/False; if True, read the files from ../Data/ without actually processing the files
    :return: x_train: np.array[str]
    :return: x_train: np.array[int]
    :return: exp_range_train: list[list[int]]; The possible expansion ids for each instance in training set
    :return: x_train: np.array[str]
    :return: x_train: np.array[int]
    :return: exp_range_test list[list[int]]; The possible expansion ids for each instance in testing set
    :return: exp2id: dict[str, int]; The mapping of expansion to its id.
    """
    if not load:
        df = pd.read_csv(path)
        print('Raw data loaded')
        articles_locations = []
        for article, location in zip(df.TEXT, df.LOCATION):
            articles_locations.append([article, location])

        acro2exp = get_acronym_to_expansion_dict(ADAM_PATH)
        exp2id = encode_expansions(acro2exp)
        y_all = np.array([exp2id[label] for label in df.LABEL])

        articles_locations_train, articles_locations_test, y_train, y_test = \
            train_test_split(articles_locations, y_all, test_size=TEST_SIZE)

        articles_train = []
        locations_train = []
        for article, location in articles_locations_train:
            articles_train.append(article)
            locations_train.append(location)

        articles_test = []
        locations_test = []
        for article, location in articles_locations_test:
            articles_test.append(article)
            locations_test.append(location)

        x_train, acro_train = get_windowed_x_and_acronyms(articles_train, locations_train)
        x_test, acro_test = get_windowed_x_and_acronyms(articles_test, locations_test)
        exp_range_train = get_expansion_range(acro_train, exp2id, acro2exp)
        exp_range_test = get_expansion_range(acro_test, exp2id, acro2exp)

        if save:
            pickle.dump(x_train, open(os.path.join('..', 'Data', 'x_train.pkl'), "wb"))
            pickle.dump(x_test, open(os.path.join('..', 'Data', 'x_test.pkl'), "wb"))
            pickle.dump(y_train, open(os.path.join('..', 'Data', 'y_train.pkl'), "wb"))
            pickle.dump(y_test, open(os.path.join('..', 'Data', 'y_test.pkl'), "wb"))
            pickle.dump(exp_range_train, open(os.path.join('..', 'Data', 'exp_range_train.pkl'), "wb"))
            pickle.dump(exp_range_test, open(os.path.join('..', 'Data', 'exp_range_test.pkl'), "wb"))
            pickle.dump(exp2id, open(os.path.join('..', 'Data', 'exp2id.pkl'), "wb"))
            pickle.dump(acro_train, open(os.path.join('..', 'Data', 'acro_train.pkl'), "wb"))
            pickle.dump(acro_test, open(os.path.join('..', 'Data', 'acro_test.pkl'), "wb"))

    else:
        x_train = pickle.load(open(os.path.join('..', 'Data', 'x_train.pkl'), "rb"))
        x_test = pickle.load(open(os.path.join('..', 'Data', 'x_test.pkl'), "rb"))
        y_train = pickle.load(open(os.path.join('..', 'Data', 'y_train.pkl'), "rb"))
        y_test = pickle.load(open(os.path.join('..', 'Data', 'y_test.pkl'), "rb"))
        exp_range_train = pickle.load(open(os.path.join('..', 'Data', 'exp_range_train.pkl'), "rb"))
        exp_range_test = pickle.load(open(os.path.join('..', 'Data', 'exp_range_test.pkl'), "rb"))
        exp2id = pickle.load(open(os.path.join('..', 'Data', 'exp2id.pkl'), "rb"))

    return x_train, x_test, exp_range_train, y_train, y_test, exp_range_test, exp2id


def process_data2(train_path, valid_test_path):

    """
    :param train_path: Training data file path
    :param valid_test_path: Validation or testing data file path
    :return: train: Dataframe; Processed training DataFrame 
    :return: valid_test: Dataframe; Processed validation or testing DataFrame 
    Returns processed training, validation or testing DataFrames.
    """
    df_train = pd.read_csv(train_path)
    df_valid_test = pd.read_csv(valid_test_path)
    
    print('Raw data loaded')
    train = df_train
    valid_test = df_valid_test
    
    train['ABV'] = create_abv_col(df_train)
    valid_test['ABV'] = create_abv_col(df_valid_test)
            
    # Only keep training and validation data about the 20 most common abbreviation expansions
    # Higher occurences will make it easier to gather context data and disambiguate 
    grouped = train.groupby(by=['ABV', 'LABEL'], as_index = False, sort = False).count()
    grouped = grouped.sort_values(by='TEXT', ascending = False)
    topLabels = grouped['ABV'][:20]
    train = train[train['ABV'].isin(topLabels)]
    valid_test = valid_test[valid_test['ABV'].isin(topLabels)]
    
    train['TOKEN'] = create_tokens(train)
    train = drop_cols(train)
    train['TOKEN'] = remove_stopwords(train)
    
    valid_test['TOKEN'] = create_tokens(valid_test)
    valid_test = drop_cols(valid_test)
    valid_test['TOKEN'] = remove_stopwords(valid_test)
    
    
    return train, valid_test