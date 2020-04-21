import numpy as np
import datasets
import yaml
import pickle
import argparse

import tensorflow as tf
from sklearn.model_selection import train_test_split

import models



def make_dataset(sent_len, kind, min_len, max_len, num_vords, batch_size, test_size=0.05, seed=12345):
    # read data
    cornell = datasets.readCornellData('cornell/', max_len=sent_len, kind=kind)

    sentences = ['<start>' + ' ' + i[0] + ' ' + '<eos>' for i in cornell]
    replies = ['<start>' + ' ' + i[1] + ' ' + '<eos>' for i in cornell]

    # filter sentences by length
    sent_mask = [min_len <= len(i.split(' ')) <= max_len for i in sentences]
    replies_mask = [min_len <= len(i.split(' ')) <= max_len for i in replies]
    full_mask = [i and j for (i, j) in zip(sent_mask, replies_mask)]

    sentences = np.array(sentences)[full_mask].tolist()
    replies = np.array(replies)[full_mask].tolist()

    # tokenize
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, filters='', oov_token='<unk>')
    tokenizer.fit_on_texts(sentences + replies)

    sentences_en = tokenizer.texts_to_sequences(sentences)
    replies_en = tokenizer.texts_to_sequences(replies)

    sentences_en = tf.keras.preprocessing.sequence.pad_sequences(sentences_en, maxlen=None, padding='post', value=0)
    replies_en = tf.keras.preprocessing.sequence.pad_sequences(replies_en, maxlen=None, padding='post', value=0)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(sentences_en, replies_en, test_size=test_size, random_state=seed)

    buffer_size = len(X_train)
    steps_per_epoch = buffer_size // batch_size
    vocab_size = min(num_vords, len(tokenizer.word_index)) + 2 # + 1 for padding symbol + 1 for <unk>

    # dataset preparation
    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size, seed=seed).batch(batch_size, drop_remainder=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.shuffle(buffer_size, seed=seed).batch(batch_size, drop_remainder=True)
    
    return tokenizer, dataset, val_dataset, vocab_size, steps_per_epoch


parser = argparse.ArgumentParser()
parser.add_argument('model_name')
args = parser.parse_args()
MODEL_NAME = args.model_name

# read parameters
with open('input_params/input_params.yaml') as f:
    
    PARAMS = yaml.load(f, Loader=yaml.FullLoader)
    PARAMS = PARAMS[MODEL_NAME]
    
    # read data params
    SENT_LEN = PARAMS['READ_DATA_PARAMS']['SENT_LEN']
    KIND = PARAMS['READ_DATA_PARAMS']['KIND']
    # vocab params
    MAX_LEN = PARAMS['VOCAB_PARAMS']['MAX_LEN']
    MIN_LEN = PARAMS['VOCAB_PARAMS']['MIN_LEN']
    NUM_WORDS = PARAMS['VOCAB_PARAMS']['NUM_WORDS']
    # model params
    BATCH_SIZE = PARAMS['MODEL_PARAMS']['BATCH_SIZE']
    DROP_PROB = PARAMS['MODEL_PARAMS']['DROP_PROB']
    EMBEDDING_DIMS = PARAMS['MODEL_PARAMS']['EMBEDDING_DIMS']
    RNN_UNITS = PARAMS['MODEL_PARAMS']['RNN_UNITS']
    ATTENTION_UNITS = PARAMS['MODEL_PARAMS']['ATTENTION_UNITS']
    IF_USE_ATTENTION = PARAMS['MODEL_PARAMS']['IF_USE_ATTENTION']
    TRAINABLE_EMBEDDING = PARAMS['MODEL_PARAMS']['TRAINABLE_EMBEDDING']
    ENCODER_CONFIG = PARAMS['MODEL_PARAMS']['ENCODER_CONFIG']
    DECODER_CONFIG = PARAMS['MODEL_PARAMS']['DECODER_CONFIG']
    # train params
    EPOCHS = PARAMS['TRAIN_PARAMS']['EPOCHS']
    EVAL_FREQ = PARAMS['TRAIN_PARAMS']['EVAL_FREQ']
    SAVE_PATH = PARAMS['TRAIN_PARAMS']['SAVE_PATH']

# prepare dataset
tokenizer, dataset, val_dataset, vocab_size, steps_per_epoch = make_dataset(sent_len=SENT_LEN,
                                                                            kind=KIND,
                                                                            min_len=MIN_LEN,
                                                                            max_len=MAX_LEN,
                                                                            num_vords=NUM_WORDS,
                                                                            batch_size=BATCH_SIZE)

# Seq2Seq model
embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIMS)
encoder = models.BidirectionalMultiLayersEncoder(ENCODER_CONFIG, RNN_UNITS, DROP_PROB, return_sequences=True)
decoder = models.MultiLayersDecoder(DECODER_CONFIG, RNN_UNITS*2, vocab_size, DROP_PROB)
seq2seq = models.Seq2Seq(embedding, encoder, decoder, if_use_attention=IF_USE_ATTENTION, trainable_embedding=TRAINABLE_EMBEDDING)

# train
loss_hist = seq2seq.fit(dataset, EPOCHS, steps_per_epoch, val_dataset, EVAL_FREQ)

# save model and tokenizer
seq2seq.save_weights(SAVE_PATH)
pickle.dump(tokenizer, open(SAVE_PATH + MODEL_NAME + '_tokenizer.pickle', 'wb'))
pickle.dump(loss_hist, open(SAVE_PATH + MODEL_NAME + '_loss_hist.pickle', 'wb'))