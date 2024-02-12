import numpy as np
import pandas as pd
import tensorflow as tf
import transformers as trfs
import math

class BaseModel:
    """
    initialize base model from the hub and add layers to the pretrained model
    """
    def __init__(self):
        # load config files
        self.PRETRAINED_MODEL_NAME = '[REDACTED]'

    def create_model(self, max_sequence, model_name):
        # load pretrained model and add layers to the model

        model = trfs.TFMPNetForSequenceClassification.from_pretrained(model_name, num_labels=5)
        # print(model.config)
        # This is the input for the tokens themselves(words from the dataset after encoding):
        input_ids = tf.keras.layers.Input(shape=(max_sequence,), dtype=tf.int32, name='input_ids')

        attention_mask = tf.keras.layers.Input(shape=(max_sequence,), dtype=tf.int32, name='attention_mask')

        # Use previous inputs as BERT inputs:
        output = model([input_ids, attention_mask])

        output = tf.keras.layers.Dense(1, activation='sigmoid')(output[0])
        # print(output.shape)
        # Final model:
        model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output)
        return model

    def base_model_init(self):
        # create model and compile

        model = self.create_model(64, self.PRETRAINED_MODEL_NAME)
        opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model
