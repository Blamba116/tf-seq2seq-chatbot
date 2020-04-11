import tensorflow as tf
from layers import *

# Basic Encoder
class Encoder(tf.keras.Model):
    def __init__(self, rnn_units, drop_prob, return_sequences=False):
        super().__init__()
        self.rnn_units = rnn_units
        self.drop_prob = drop_prob
        self.return_sequences = return_sequences
        self.encoder_rnnlayer = tf.keras.layers.LSTM(rnn_units, return_sequences=self.return_sequences, return_state=True
                                                     , recurrent_dropout=drop_prob)

    def call(self, initial_input):
        output, hidden_state, cell_state = self.encoder_rnnlayer(initial_input)

        return output, hidden_state, cell_state
    
# Bidirectional Encoder
class BidirectionalEncoder(tf.keras.Model):
    def __init__(self, rnn_units, drop_prob, return_sequences=False):
        super().__init__()
        self.rnn_units = rnn_units
        self.drop_prob = drop_prob
        self.return_sequences = return_sequences
        self.layer = tf.keras.layers.LSTM(rnn_units, return_sequences=self.return_sequences, return_state=True
                                          , recurrent_dropout=drop_prob)
        self.encoder_rnnlayer = tf.keras.layers.Bidirectional(self.layer)

    def call(self, initial_input):
        output, forward_h, forward_c, backward_h, backward_c = self.encoder_rnnlayer(initial_input)
        hidden_state = tf.concat([forward_h, backward_h], -1)
        cell_state = tf.concat([forward_c, backward_c], -1)

        return output, hidden_state, cell_state

# Bidirectional multi layers Encoder
class BidirectionalMultiLayersEncoder(tf.keras.Model):
    def __init__(self, rnn_config, rnn_units, drop_prob, return_sequences=False):
        super().__init__()
        self.rnn_units = rnn_units
        self.drop_prob = drop_prob
        self.return_sequences = return_sequences
        for i in rnn_config:
            layer = tf.keras.layers.LSTM(rnn_units, return_sequences=self.return_sequences, return_state=True
                                         , recurrent_dropout=drop_prob)
            layer = tf.keras.layers.Bidirectional(layer)
            setattr(self, i, layer)
            
    def call(self, initial_input):
        hidden_states = []
        cell_states = []
        for i, rnn_layer in enumerate(self.layers):
            if i == 0:
                output, forward_h, forward_c, backward_h, backward_c = rnn_layer(initial_input)
            else:
                output, forward_h, forward_c, backward_h, backward_c = rnn_layer(output)
            hidden_states.append(tf.concat([forward_h, backward_h], -1))
            cell_states.append(tf.concat([forward_c, backward_c], -1))

        return output, hidden_states, cell_states

# Basic Decoder
class Decoder(tf.keras.Model):
    def __init__(self, rnn_units, dense_units, drop_prob):
        super().__init__()
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.drop_prob = drop_prob
        self.decoder_rnnlayer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True
                                                     , recurrent_dropout=drop_prob)

        self.dense_layer = tf.keras.layers.Dense(dense_units)

    def call(self, initial_input, initial_state):
        output, hidden_state, cell_state = self.decoder_rnnlayer(initial_input, initial_state)
        logits = self.dense_layer(output)

        return logits, hidden_state, cell_state
    
# Decoder from tfa
class TFADecoder(tf.keras.Model):
    def __init__(self, rnn_units, vocab_size, drop_prob):
        super().__init__()
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.drop_prob = drop_prob
        self.rnn_cell = tf.keras.layers.LSTMCell(rnn_units, recurrent_dropout=drop_prob)
        self.dense_layer = tf.keras.layers.Dense(vocab_size)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler=self.sampler, output_layer=self.dense_layer)
        
    def call(self, initial_input, initial_state):        
        outputs, hidden_state, cell_state = self.decoder(initial_input, initial_state=initial_state
                                                         , sequence_length=BATCH_SIZE*[Ty-1])

        logits = outputs.rnn_output
        
        return logits, hidden_state, cell_state
    
# Attention Decoder
class AttentionDecoder(tf.keras.Model):
    def __init__(self, rnn_units, dense_units, attention_units, drop_prob):
        super().__init__()
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.attention_units = attention_units
        self.drop_prob = drop_prob
        self.decoder_rnnlayer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True
                                                     , recurrent_dropout=drop_prob)
        self.attention = BahdanauAttention(self.attention_units)

        self.dense_layer = tf.keras.layers.Dense(dense_units)
        
    def call(self, initial_input, initial_state, encoder_output, seq_len):
        hidden, cell = initial_state * 1 # copy tensor
        outputs = tf.zeros([initial_input.shape[0], 1, self.dense_units])
        for i in range(seq_len):
            # slice inputs
            inputs = initial_input[:, i, :]
            
            #build context vector using attention mechanism
            context_vector, attention_weights = self.attention(hidden, encoder_output)
            
            #build decoder input via concating context vector + sliced input
            decoder_input = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(inputs, 1)], axis=-1)

            # run rnn cell
            output, hidden, cell = self.decoder_rnnlayer(decoder_input, [hidden, cell])
            
            # get logits
            output = tf.reshape(output, (-1, output.shape[2]))
            logits = self.dense_layer(output)
            
            outputs = tf.concat([outputs, tf.expand_dims(logits, 1)], 1)
        outputs = outputs[:, 1:, :]

        return outputs, hidden, cell
    
# Attention multi layers Decoder
class AttentionMultiLayersDecoder(tf.keras.Model):
    def __init__(self, rnn_config, rnn_units, dense_units, attention_units, drop_prob):
        super().__init__()
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.attention_units = attention_units
        self.drop_prob = drop_prob
        for i in rnn_config:
            layer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, recurrent_dropout=drop_prob)
            setattr(self, i, layer)
            
        self.attention = BahdanauAttention(self.attention_units)
        self.dense_layer = tf.keras.layers.Dense(dense_units)
        
    def call(self, initial_input, initial_state, encoder_output, seq_len):
        hidden, cell = initial_state
        outputs = tf.zeros([initial_input.shape[0], 1, self.dense_units])
        for i in range(seq_len):
            # slice inputs
            inputs = initial_input[:, i, :]
            
            #build context vector using attention mechanism
            context_vector, attention_weights = self.attention(hidden[-1], encoder_output)
            
            #build decoder input via concating context vector + sliced input
            decoder_input = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(inputs, 1)], axis=-1)

            # run rnn cell
            for j, rnn_layer in enumerate(self.layers):
                if isinstance(rnn_layer, tf.keras.layers.LSTM):
                    if j == 0:
                        output, hidden[j], cell[j] = rnn_layer(decoder_input, [hidden[j], cell[j]])
                    else:
                        output, hidden[j], cell[j] = rnn_layer(output, [hidden[j], cell[j]])
            
            # get logits
            output = tf.reshape(output, (-1, output.shape[2]))
            logits = self.dense_layer(output)
            
            outputs = tf.concat([outputs, tf.expand_dims(logits, 1)], 1)
        outputs = outputs[:, 1:, :]

        return outputs, hidden, cell