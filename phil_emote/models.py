import os
import numpy as np
import pandas as pd
from keras.models import model_from_json
from .tweetokenize.tokenizer import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class PhilEmoteModel():

    file_dir = os.path.dirname(__file__)
    model_dir = os.path.join(file_dir, 'model')
    lex_dir = os.path.join(file_dir, 'lexicon')

    def __init__(self, 
        json_file = os.path.join(model_dir, 'phil_emote.json'),
        weight_file = os.path.join(model_dir, 'phil_emote.h5'),
        words_file = os.path.join(lex_dir, 'words.tsv'),
        classes_file = os.path.join(lex_dir, 'classes.tsv'),
        maxlen = 50):
    
        self.classes_file = classes_file
        self.maxlen = maxlen
    
        self.tokenizer = Tokenizer()
        self.word_index = self.load_words(words_file=words_file)
        self.model = self.load_model(
            json_file=json_file, weight_file=weight_file)

    def load_model(self, json_file, weight_file):
        # load json and create model
        with open(json_file, 'r') as json_data:
            loaded_model_json = json_data.read()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(weight_file)
        return loaded_model

    def load_words(self, words_file):
        word_dataset = pd.read_csv(
            words_file, sep='\t', quoting=3, names=['index', 'word'])
        word_index = {word_dataset['word'][idx]: word_dataset['index'][idx]
                      for idx, i in enumerate(word_dataset['index'])}
        return word_index

    def load_classes(self, classes_file, output_type='emotion'):
        class_dataset = pd.read_csv(classes_file, sep='\t', names=[
                                    'index', 'emoji', 'emotion', 'sentiment'])
        class_dict = {class_dataset['index'][idx]: class_dataset[output_type][idx]
                      for idx, i in enumerate(class_dataset['emoji'])}

        return class_dict

    def tokenize_dataset(self, dataset, tokenizer):
        if type(dataset) == str:
            return [tokenizer.tokenize(dataset)]
        else:
            return tokenizer.tokenize_set(dataset)

    def vectorize_dataset(self, dataset, word_index, maxlen):
        input_vectors = []
        for word_tokens in dataset:
            index_tokens = []
            for token in word_tokens:
                try:
                    index_tokens.append(word_index[token])
                except:
                    index_tokens.append(1)
            input_vectors.append(index_tokens)
        return pad_sequences(input_vectors, maxlen=maxlen)

    def predict_dataset(self, dataset, output_type='emotion'):
        dataset = self.tokenize_dataset(dataset, tokenizer=self.tokenizer)
        dataset = self.vectorize_dataset(
            dataset, word_index=self.word_index, maxlen=self.maxlen)
        prediction = self.model.predict(dataset)

        class_dict = self.load_classes(
            classes_file=self.classes_file, output_type=output_type)

        return [class_dict[np.argmax(i)] for i in prediction]
