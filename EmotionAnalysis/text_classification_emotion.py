# -*- coding: utf-8 -*-
"""emotion-classification-2.ipynb

Converted into class form

Pre requisite:
pip install -U sentence-transformers -q
"""
import pandas as pd
import numpy as np
import re
import argparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sentence_transformers import SentenceTransformer, util
from sklearn.datasets import make_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


class TrainEmotionModel():
    def __init__(self, input_path, output_path, test_df_path=None):
        """
        Initialize the dataframes and static column names
        @param input_path: Path to the file named "dataSetForPred.csv
        @param output_path: Path to save the test dataset
        @param [OPTIONAL] test_df_path: Path for the file to be tested for inference. If not entered will
        use a predefined dataset

        """
        self.input_path = input_path
        self.emotionNames = ['satisfaction', 'anger', 'relief', 'disappointment', 'disgust',
                             'boredom', 'reminiscence', 'guilt', 'worry', 'fear', 'sluggish',
                             'sickly', 'confusion', 'stimulation', 'excitement/adventurous',
                             'comfort', 'curiosity', 'delight', 'amazed', 'enjoyment/amusement',
                             'healthy/energetic', 'intense', 'sophisticated']
        self.traincols = ['[REDACTED]']
        self.test_df_path = test_df_path
        self.output_path = output_path
        self.model = SentenceTransformer('microsoft/mpnet-base')
        self.embedder = FunctionTransformer(
            lambda item: self.model.encode(item, convert_to_tensor=True,
                                           show_progress_bar=False).detach().cpu().numpy())
        self.preprocessor = ColumnTransformer(
            transformers=[('embedder', self.embedder, 'context')],
            remainder='passthrough'
        )

    def proc_df(self):
        """
        Initial df processing
        @return: processed dataframe
        """

        df = pd.read_csv(self.input_path, index_col=0)
        df = df.loc[:10000]
        df.drop_duplicates(inplace=False)
        df.drop(df[df['rating'] == 3].index, inplace=True)

        return df

    def divide_train(self):
        '''
        Divide dataset into train and test sets
        @return:train and test sets
        '''
        df = self.proc_df()
        train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

        train = train.reset_index(drop=True)
        # train2 = train2.reset_index(drop=True)
        validate = validate.reset_index(drop=True)
        test = test.reset_index(drop=True)

        X_train = train[self.traincols]
        Y_train = train[self.emotionNames]
        X_test = test[self.traincols]
        Y_test = test[self.emotionNames]
        X_validate = validate[self.traincols]
        Y_validate = validate[self.emotionNames]

        return [X_train, Y_train, X_test, Y_test, X_validate, Y_validate]

    def create_model(self):
        '''
        Createe the model with the data
        @return: dataframe with the predictions of the model.
        '''

        X_train, Y_train, X_test, Y_test, X_validate, Y_validate = self.divide_train()

        X_train_prep = self.preprocessor.fit_transform(X_train)
        forest = RandomForestClassifier(random_state=1)
        multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
        self.pred = multi_target_forest.fit(X_train_prep, Y_train)

        X_test_prep = self.preprocessor.fit_transform(X_test)
        Y_pred = self.pred.predict(X_test_prep)

        X_validate_prep = self.preprocessor.fit_transform(X_validate)
        Y_validate_pred = self.pred.predict(X_validate_prep)

        dfYPred = pd.DataFrame(list(Y_pred), columns=self.emotionNames)
        dftoCheck = pd.concat([X_test, dfYPred], axis=1)

        return dftoCheck

    def test_model_man(self):
        '''
        Manual testing df
        @return: manual testing results df
        '''
        if self.test_df_path is None:
            dfManTest = pd.DataFrame({'context': ['the chips are rotten', 'the taste reminds me of childhood',
                                                  'the chips are too salty', 'the taste is fresh',
                                                  'this was a popular snack at the party'],
                                      'sentiment': [-0.8, 0.4, -0.5, 0.6, 0.8], 'rating': [1, 4, 2, 5, 5]})
        else:
            dfManTest = pd.DataFrame(self.test_df_path, index=None)
        X_man_test = self.preprocessor.fit_transform(dfManTest[self.traincols])
        Y_man_test = self.pred.predict(X_man_test)

        dfYManPred = pd.DataFrame(list(Y_man_test), columns=self.emotionNames)

        dfManCheck = pd.concat([dfManTest, dfYManPred], axis=1)
        dfManCheck.to_csv(self.output_path)
        return dfManCheck


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_path",
                        metavar="input_path", help="input file name")
    parser.add_argument("-test", dest="test_file_path",
                        metavar="test-csv-file-path", help="test csv file path")
    parser.add_argument("-o", dest="output_file_path",
                        metavar="out-file-path", help="output csv file path")

    options = parser.parse_args()
    print(options)

    marking = TrainEmotionModel(
        input_path=options.input_path, output_path=options.output_path,
        test_df_path=options.test_file_path
    )

    df_Test = marking.create_model()
    df_Test.head()

    marking.test_model_man()
