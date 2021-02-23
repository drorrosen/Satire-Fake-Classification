

import os
import re
import pandas as pd
import numpy as np


#preprocessing
from sentence_transformers import SentenceTransformer
from pattern.en import sentiment
from pattern.en import parse, Sentence
from pattern.en import modality, mood
import spacy
#spacy.cli.download("en_core_web_sm")


nlp = spacy.load("en_core_web_sm")

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


#modeling
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


#evaluation
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix

#saving model 
import joblib


#paths
FAKE_SOURCE_DIR = 'StoryText 2/Fake/finalFake'
SATIRE_SOURCE_DIR = 'StoryText 2/Satire/finalSatire'


class CleaningDf:
    """
    Cleaning the dataset and merging the text documents to the dataset.
    Returns a clean version of the data
    """
    def __init__(self,path = 'Fake News Stories.xlsx', target_col='Fake or Satire?', text_col='Text'):
        """
        :param df: The dataset with the Text and target columns
        :param target_col: the target column
        :param text_col: the text column
        """
        self.df = pd.read_excel(path)
        self.target_col = target_col
        self.text_col = text_col
        
    def extract_texts(self):
        """
        Read the text articles and create a temporary dataframe.
        returns a dataframe with two columns: Article Number and Text.
        """
        text_dict = {'Article Number': [] , 'Text': []}
        
        for folder in [FAKE_SOURCE_DIR, SATIRE_SOURCE_DIR]:
            for text in os.listdir(folder):
                if text.endswith('.txt'):
                    article_num = int(text.split('.txt')[0])
                    full_path = os.path.join(folder,text)
                    with open(full_path, encoding='Latin-1') as txt:
                        text_dict['Text'].append(txt.read())
                        text_dict['Article Number'].append(article_num)
                        
                        
        return pd.DataFrame(text_dict)

    def merge_dfs(self):
        """
        Merging the original dataframe  with the dataframe with the text, on Article Number.
        """
        self.df = pd.merge(left=self.df, right=self.extract_texts(), on='Article Number')
        self.df.loc[self.df['Fake or Satire?'] == 'Satire ', 'Fake or Satire?'] = 'Satire'
        self.df = self.df[['Text','Fake or Satire?', 'URL of rebutting article']]
        
        return self.df[[self.text_col, self.target_col]]
    





class Preprocessing:
    """
    Doing data Engineering and split to train and test.
    """
    def __init__(self,df, test_size=0.25, dim_reduction = True, n_components=100):
        """
        :param df: The cleaned dataframe
        :param test_size: The size of the test dataset
        :param dim_reduction: use PCA or not.
        :param n_components: number of components to choose for PCA
        """
        self.df = df
        self.test_size= 0.25
        self.dim_reduction = dim_reduction
        self.n_components = 100
    
    
                
        
    def adding_bert_features(self):
        """
        BERT Embeddings on the text column.
        """
        model = SentenceTransformer('stsb-bert-base')
        embbed_text = self.df['Text'].apply(lambda x: model.encode(x))
        embbed_text = np.array(embbed_text.tolist())
        embbed_df = pd.DataFrame(embbed_text)
        self.df = pd.concat([self.df, embbed_df],axis=1)
        
                
    
    def _create_datasets(self):
        """
        creates train and test datasets.
        :return: train and test datasets
        
        """
        X = self.df.drop(columns='Fake or Satire?')
        y = self.df['Fake or Satire?']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        
        self.X_train = X_train.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

        
    
    
    def feature_engineering(self):
        """
        Adding new features to the datasets.
        Returns training and test datasets after feature engineering.
        """
            

        #clean text
        self.X_train['Text'] = self.X_train['Text'].apply(lambda x: self.get_clean_text(x))
        self.X_test['Text'] = self.X_test['Text'].apply(lambda x: self.get_clean_text(x))

        #add number of words 
        self.X_train['word_num'] = self.X_train['Text'].apply(lambda s: len(s.split()))
        self.X_test['word_num'] = self.X_test['Text'].apply(lambda s: len(s.split()))
        
        
        # #adding modality and sentiment
        self.X_train = self.adding_sentiment(self.X_train)
        self.X_test = self.adding_sentiment(self.X_test)

        #drop Text column
        self.X_train.drop(columns=['Text'], inplace=True)
        self.X_test.drop(columns=['Text'], inplace=True)
        
        if self.dim_reduction:
            pca = PCA(n_components=self.n_components)
            self.X_train = pca.fit_transform(self.X_train)
            self.X_test = pca.transform(self.X_test)
    
            
        return self.X_train, self.X_test, self.y_train, self.y_test
            
                        

    def get_clean_text(self, row):
        """
        Some basic NLP preprocessing - tokenization and removing stopwords.
        """
        #lower letters
        row = row.lower()
        #removing stopwords
        row = ' '.join([word for word in row.split() if word not in nlp.Defaults.stop_words])
        #tokenization
        doc = nlp(row)
        row = ' '.join([token.text for token in doc])
        
        return row
    

        
        
    def adding_sentiment(self, data):
        """
        Adding sentiment, mood and modality as new features.
        """

        data['polarity'], data['subjectivity'] = data['Text'].apply(lambda x: sentiment(x)).str
        data['mood'], data['modality'] = data['Text'].apply(lambda x: self.adding_mood_modality(x)).str


        hot_encoded = pd.get_dummies(data['modality'])
        data = pd.concat([data.drop(columns='modality'), hot_encoded], sort=False, axis=1)


        return data



    def adding_mood_modality(self, row):
        """
        Calculating mood and modality from a text/row
        """
        row = parse(row, lemmata=True)
        row = Sentence(row)

        return modality(row), mood(row)
    


    
    
class Model:
    """
    Training the model and saving it as .pkl.
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        :param X_train: X_train dataset
        :param X_test: X_test dataset
        :param y_train: target train column
        :param y_test: target test column
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = LogisticRegression()
        
    
    def fit(self):
        """
        Doing cross validation to find the best hyperparameters 
        for the model and then fitting the model on the train dataset.
        """
        
        # Create first pipeline for base without reducing features.

        pipe = Pipeline([('classifier' , self.model)])

        # Create param grid.

        param_grid = [
            {'classifier' : [self.model],
             'classifier__penalty' : ['l1', 'l2'],
            'classifier__solver' : ['liblinear']}]

        # Create grid search object

        clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5)
        
        self.model = clf.fit(self.X_train, self.y_train)
        
        
    def predict(self):
        """
        Predicting the test dataset based on the fitted model.
        """
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred
        
    
    def evaluate(self):
        """
        model evaluation.
        """
        print(f'accuracy_score is {accuracy_score(self.y_test, self.y_pred)}]\n')
        print(f'confusion_matrix:\n {confusion_matrix(self.y_test, self.y_pred)}')
        print(f'classification report:\n {classification_report(self.y_test, self.y_pred)}')
    
    def save_model(self):
        """
        Saving model
        """
        joblib.dump(self.model, 'model_Fake_Satire.pkl')




if __name__ == '__main__':
    
    #Cleaning data
    df = CleaningDf()
    df = df.merge_dfs()
    
    #initiating the class preprocessing
    pre = Preprocessing(df)
    #adding embeddings
    pre.adding_bert_features()
    #splitting data
    pre._create_datasets()
    #feature engineering 
    X_train, X_test, y_train, y_test = pre.feature_engineering()
    #Training model
    
    model = Model(X_train, X_test, y_train, y_test)
    model.fit()
    model.predict()
    #evauluation
    model.evaluate()
    
    #save the model to the local machine
    model.save_model()

