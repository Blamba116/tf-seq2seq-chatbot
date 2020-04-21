import numpy as np
import datasets
import yaml
import pickle

import tensorflow as tf
from sklearn.model_selection import train_test_split

import models


class Chatbot():
    
    def __init__(self, model_name):
        self.model_name = model_name
        self._read_params()
        
    def _read_params(self, path='input_params/input_params.yaml'):
        # read parameters
        with open(path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            params = params[self.model_name]

            # read data params
            self.sent_len = params['READ_DATA_PARAMS']['SENT_LEN']
            self.kind = params['READ_DATA_PARAMS']['KIND']
            # vocab params
            self.max_len = params['VOCAB_PARAMS']['MAX_LEN']
            self.min_len = params['VOCAB_PARAMS']['MIN_LEN']
            self.num_words = params['VOCAB_PARAMS']['NUM_WORDS']
            # model params
            self.batch_size = params['MODEL_PARAMS']['BATCH_SIZE']
            self.drop_prob = params['MODEL_PARAMS']['DROP_PROB']
            self.embedding_dims = params['MODEL_PARAMS']['EMBEDDING_DIMS']
            self.rnn_units = params['MODEL_PARAMS']['RNN_UNITS']
            self.attention_units = params['MODEL_PARAMS']['ATTENTION_UNITS']
            self.if_use_attention = params['MODEL_PARAMS']['IF_USE_ATTENTION']
            self.trainable_embedding = params['MODEL_PARAMS']['TRAINABLE_EMBEDDING']
            self.encoder_config = params['MODEL_PARAMS']['ENCODER_CONFIG']
            self.decoder_config = params['MODEL_PARAMS']['DECODER_CONFIG']
            # train params
            self.epochs = params['TRAIN_PARAMS']['EPOCHS']
            self.eval_freq = params['TRAIN_PARAMS']['EVAL_FREQ']
            self.save_path = params['TRAIN_PARAMS']['SAVE_PATH']

    def make_dataset(self, test_size=0.05, seed=12345):
        # read data
        cornell = datasets.readCornellData('cornell/', max_len=self.sent_len, kind=self.kind)

        sentences = ['<start>' + ' ' + i[0] + ' ' + '<eos>' for i in cornell]
        replies = ['<start>' + ' ' + i[1] + ' ' + '<eos>' for i in cornell]

        # filter sentences by length
        sent_mask = [self.min_len <= len(i.split(' ')) <= self.max_len for i in sentences]
        replies_mask = [self.min_len <= len(i.split(' ')) <= self.max_len for i in replies]
        full_mask = [i and j for (i, j) in zip(sent_mask, replies_mask)]

        sentences = np.array(sentences)[full_mask].tolist()
        replies = np.array(replies)[full_mask].tolist()

        # tokenize
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.num_words, filters='', oov_token='<unk>')
        tokenizer.fit_on_texts(sentences + replies)

        sentences_en = tokenizer.texts_to_sequences(sentences)
        replies_en = tokenizer.texts_to_sequences(replies)

        sentences_en = tf.keras.preprocessing.sequence.pad_sequences(sentences_en, maxlen=None, padding='post', value=0)
        replies_en = tf.keras.preprocessing.sequence.pad_sequences(replies_en, maxlen=None, padding='post', value=0)

        # train-test split
        X_train, X_test, y_train, y_test = train_test_split(sentences_en, replies_en, test_size=test_size, random_state=seed)

        buffer_size = len(X_train)
        steps_per_epoch = buffer_size // self.batch_size

        # dataset preparation
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size, seed=seed).batch(self.batch_size, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_dataset = val_dataset.shuffle(buffer_size, seed=seed).batch(self.batch_size, drop_remainder=True)
        
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.steps_per_epoch = steps_per_epoch

    def _build(self):
        self.vocab_size = min(self.num_words, len(self.tokenizer.word_index)) + 2
        
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dims)
        self.encoder = models.BidirectionalMultiLayersEncoder(self.encoder_config, self.rnn_units, self.drop_prob, return_sequences=True)
        self.decoder = models.MultiLayersDecoder(self.decoder_config, self.rnn_units*2, self.vocab_size, self.drop_prob)
        self.seq2seq = models.Seq2Seq(self.embedding, self.encoder, self.decoder, if_use_attention=self.if_use_attention,
                                      trainable_embedding=self.trainable_embedding)        
        
    def train(self, delete_after_training=True):
        self._build()
        self.loss_hist = self.seq2seq.fit(self.dataset, self.epochs, self.steps_per_epoch, self.val_dataset, self.eval_freq)
        if delete_after_training:
            # delete datasets
            self.dataset = None
            self.val_dataset = None
        
    def save_model(self):
        self.seq2seq.save_weights(self.save_path)
        pickle.dump(self.tokenizer, open(self.save_path + self.model_name + '_tokenizer.pickle', 'wb'))
        pickle.dump(self.loss_hist, open(self.save_path + self.model_name + '_loss_hist.pickle', 'wb'))
        
    def load_model(self):
        self.tokenizer = pickle.load(open(self.save_path + self.model_name + '_tokenizer.pickle', 'rb'))
        self._build()
        self.seq2seq.load_weights(self.save_path)
        
    def get_answer(self, input_raw):
        input_lines = []
        for i in input_raw:
            tmp_line = '<start>' + ' ' + datasets.extractText(i, kind=self.kind) + ' ' + '<eos>'
            input_lines.append(tmp_line)

        input_lines = self.tokenizer.texts_to_sequences(input_lines)
        input_lines = tf.keras.preprocessing.sequence.pad_sequences(input_lines, maxlen=self.max_len, padding='post')
        input_lines = tf.convert_to_tensor(input_lines)

        predictions = self.seq2seq.predict(input_lines, self.max_len, self.tokenizer.word_index['<start>'])
        predictions = self.tokenizer.sequences_to_texts(predictions)
        
        for i, j in enumerate(predictions):
            predictions[i] = j.split('<eos>')[0][len('<start>')+1:-1]

        return predictions