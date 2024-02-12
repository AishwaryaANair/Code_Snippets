import math
import os
import argparse
import numpy as np
import pandas as pd
import sklearn.model_selection as ms
import sklearn.preprocessing as p
import tensorflow as tf
import transformers as trfs
from topic_modelling.new_modelling_code.ModelTrainingAPI.base_model import BaseModel

def load_files(PATH):
    # load all files in given path
    print(PATH)
    brand_str = dict()
    path = []
    for root, dirs, files in os.walk(os.path.abspath(PATH)):
        for file in files:
            path.append(os.path.join(root, file))

    # print('Path', path)

    for i in path:
        brand_str[i.split('/')[-1]] = pd.read_csv(i, index_col=0)
        # print(len(brand_str[i.split('/')[-1]]))
    return brand_str


def return_keys_for_training(input_path, theme_name, sentiment):
    # return keys from training files

    training_set = load_files(input_path + '/' + theme_name + sentiment + '/Balancedtrain')

    train_dict = dict()

    for i in range(0, len(training_set.keys())):
        train_dict[i] = list(training_set.keys())[i]

    return train_dict


class MakeModels:
    """
    initialize training loop and base models and start training

    """

    def __init__(self, theme, sentiment, test_num, model_folder_path, input_path, training_set_num=None, epochs=1,
                 return_keys=None):
        '''
        load config variables

        theme = Theme in title case (for example: Design Or Style)
        sentiment =  sentiment in lower case (positive/negative)
        test_num = add test number to model name
        model_folder_path = FOLDER where the model will be stored after generation
        input_path = FOLDER where the train and validate balanced files are located
        training_set_num = train model with particular sets
        epochs = number of epocs (default is 2)
        return_keys = Use return keys to only return the keys that will be used for training,
        when set yes - does not train the model
`
        '''
        self.theme_name = theme
        self.sentiment = sentiment
        self.test_num = test_num
        self.MAX_SEQUENCE_LENGTH = 64
        self.PRETRAINED_MODEL_NAME = 'microsoft/mpnet-base'
        self.BATCH_SIZE = 16
        self.EPOCHS = epochs
        self.MODELFOLDER = model_folder_path
        self.INPUT_PATH = input_path
        self.total_sets = training_set_num
        self.return_keys = return_keys

    def create(self, train, validate, epochs, fit_model):
        # trainer function that tokenizes and feeds data into model

        tokenizer = trfs.MPNetTokenizer.from_pretrained(self.PRETRAINED_MODEL_NAME)
        X_train, y_train, = train.sentence.values, train.label.values
        X_val, y_val = validate.sentence.values, validate.label.values

        X_train = self.batch_encode(X_train, tokenizer, self.MAX_SEQUENCE_LENGTH)
        X_val = self.batch_encode(X_val, tokenizer, self.MAX_SEQUENCE_LENGTH)

        if self.db is None:
            fit_model.fit(
                x=X_train.data,
                y=y_train,
                validation_data=(X_val.data, y_val),
                epochs=epochs,
                batch_size=self.BATCH_SIZE,
            )
        else:
            fit_model.fit(
                x=X_train.data,
                y=y_train,
                validation_data=(X_val.data, y_val),
                epochs=epochs,
                batch_size=self.BATCH_SIZE,
                callbacks=[self.custom_callback]
            )

        return fit_model

    def batch_encode(self, X, tokenizer, MAX_SEQUENCE_LENGTH):
        # encode dataset for training
        return tokenizer.batch_encode_plus(
            X,
            max_length=MAX_SEQUENCE_LENGTH,  # set the length of the sequences
            add_special_tokens=True,  # add [CLS] and [SEP] tokens
            return_attention_mask=True,
            return_token_type_ids=False,
            pad_to_max_length=True,  # add 0 pad tokens to the sequences less than max_length
            return_tensors='tf'
        )

    def create_path(self, PATH):
        # create folder if not exists
        if not os.path.exists(PATH):
            # Create a new directory because it does not exist
            print('Path created ', PATH)
            os.makedirs(PATH)
            # print(PATH)

    def trainer_function(self, training_set, validate_set):
        # train models in a loop
        if self.EPOCHS is None:
            self.EPOCHS = 2

        if self.total_sets is None:
            self.total_sets = [0]
        else:
            self.total_sets = [int(i) for i in self.total_sets.split(',')]

        # print('training_set', training_set.keys())
        self.create_path(self.MODELFOLDER + '/' + self.theme_name + self.sentiment + 'TrainedModel')

        MODELPATH = self.MODELFOLDER + '/' + self.theme_name + self.sentiment + 'TrainedModel/' + self.test_num + '.h5'

        model_obj = BaseModel()
        model = model_obj.base_model_init()
        model_list = list()


        for num in self.total_sets:
            if self.total_sets.index(num) in [0]:
                print(str(list(training_set.keys())[num]), 'Zero')
                model_function = self.create(training_set[str(list(training_set.keys())[num])],
                                             validate_set[str(list(validate_set.keys())[num])], 1, model)
                model_list.append(model_function)

                model_list[-1].save(MODELPATH)
            else:
                print(str(list(training_set.keys())[num]), num)
                model_function = self.create(training_set[str(list(training_set.keys())[num])],
                                             validate_set[str(list(validate_set.keys())[num])], 1, model_list[-1])
                model_list.append(model_function)

                model_list[-1].save(MODELPATH)

        return MODELPATH

    def return_keys_for_training(self):

        # return keys from training files

        training_set = load_files(self.INPUT_PATH + '/' + self.theme_name + self.sentiment + '/Balancedtrain')

        train_dict = dict()

        for i in range(0, len(training_set.keys())):
            train_dict[i] = list(training_set.keys())[i]

        return train_dict

    def initiator_fx(self):
        # aggregate the modules
        if self.return_keys is not None:
            return {'keys': list(self.return_keys_for_training())}

        else:
            training_set = load_files(self.INPUT_PATH + '/' + self.theme_name + self.sentiment + '/Balancedtrain')
            print(training_set.keys())
            print(self.INPUT_PATH + '/' + self.theme_name + self.sentiment + '/Balancedtrain')
            validate_set = load_files(self.INPUT_PATH + '/' + self.theme_name + self.sentiment + '/Balancedvalidate')
            # print(validate_set.keys())

            if self.total_sets is None:
                if len(training_set.keys()) > len(validate_set.keys()):
                    self.total_sets = range(0, len(validate_set.keys()))

                else:
                    self.total_sets = range(0, len(training_set.keys()))

            SAVEDPATH = self.trainer_function(training_set, validate_set)
            return {'saved_path': SAVEDPATH}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="input_file_path", metavar="input-csv-file-path", help="input csv file path")
    parser.add_argument("-o", dest="model_output_folder_path", metavar="out-model-file-path",
                        help="model output FOLDER path")

    parser.add_argument("-sen", dest="sentiment", metavar="sentiment", help="Sentiment type [positive or negative]")
    parser.add_argument("-th", dest="theme", metavar="theme", help="theme name")

    parser.add_argument("-test", dest="test_number", metavar="test_num", help="Test Number")
    parser.add_argument("-num", dest="training_set_number", metavar="training-set-number", help="training set numbers")
    parser.add_argument("-ep", dest="epochs", metavar="epochs", help="epochs")

    parser.add_argument("-key", dest="return_keys", metavar="return-training-keys-number", help="return training keys")

    # sentiment, topic_fil, test_num, model_folder_path, input_path, num = None
    options = parser.parse_args()
    print(options)

    if not (options.sentiment and options.theme and options.input_file_path and options.model_output_folder_path):
        print('Theme, sentiment, input path, output path are required.')
        exit(0)

    marking = MakeModels(options.theme,
                         options.sentiment, options.test_number, options.model_output_folder_path,
                         options.input_file_path, options.training_set_number, options.epochs, options.return_keys)

    marking.initiator_fx()
