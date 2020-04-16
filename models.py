import tensorflow as tf
from layers import *

# Bidirectional multi layers Encoder
class BidirectionalMultiLayersEncoder(tf.keras.Model):
    """Implementation of bidirectional multi layer LSTM"""
    
    def __init__(self, rnn_config, rnn_units, drop_prob, return_sequences=False):
        """rnn_config - list(str) - list with LSTM layers' names
           rnn_units - int - number of hidden units in rnn
           drop_prob - float - recurrent dropout probability (1 - keep_prob)
           return_sequences- bool - return output as a sequence
        """
        
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
        """inputs:         
               initial_input - tensor (batch_size, embedding_dim, sequence_length)
            
           outputs:
               outputs - tensor (-1, rnn_units * 2) - sequence of logits
               hidden - list of tensors (-1, rnn_units * 2) - last hidden state
               cell - list of tensors (-1, rnn_units * 2) - last cell state
        """
        
        hidden_states = []
        cell_states = []
        # forward pass through all layers
        for i, rnn_layer in enumerate(self.layers):
            if i == 0:
                output, forward_h, forward_c, backward_h, backward_c = rnn_layer(initial_input)
            else:
                output, forward_h, forward_c, backward_h, backward_c = rnn_layer(output)
            hidden_states.append(tf.concat([forward_h, backward_h], -1))
            cell_states.append(tf.concat([forward_c, backward_c], -1))

        return output, hidden_states, cell_states
    
# Multi layers Decoder
class MultiLayersDecoder(tf.keras.Model):
    """Implementation of multi layer LSTM"""    
    
    def __init__(self, rnn_config, rnn_units, dense_units, drop_prob):
        """rnn_config - list(str) - list with LSTM layers' names
           rnn_units - int - number of hidden units in rnn
           dense_units - int - number of hidden units in output layer
           drop_prob - float - recurrent dropout probability (1 - keep_prob)
        """
        
        super().__init__()
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.drop_prob = drop_prob
        for i in rnn_config:
            layer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, recurrent_dropout=drop_prob)
            setattr(self, i, layer)
        # output layer
        self.dense_layer = tf.keras.layers.Dense(dense_units)
        
    def call(self, initial_input, initial_state):
        """inputs:         
               initial_input - tensor (batch_size, embedding_dim, sequence_length)
               initial_state - list of tensors (-1, rnn_units) - initial hidden state (usually, encoder last hidden state)
            
           outputs:
               outputs - tensor (-1, rnn_units * 2) - sequence of logits
               hidden - list of tensors (-1, rnn_units * 2) - last hidden state
               cell - list of tensors (-1, rnn_units * 2) - last cell state
        """
        
        hidden, cell = initial_state
        # forward pass through all layers
        for j, rnn_layer in enumerate(self.layers):
            if isinstance(rnn_layer, tf.keras.layers.LSTM):
                if j == 0:
                    output, hidden[j], cell[j] = rnn_layer(initial_input, [hidden[j], cell[j]])
                else:
                    output, hidden[j], cell[j] = rnn_layer(output, [hidden[j], cell[j]])
                    
        outputs = self.dense_layer(output)

        return outputs, hidden, cell
    
# Attention multi layers Decoder
class AttentionMultiLayersDecoder(tf.keras.Model):
    """implementation of bidirectional multi layer LSTM with Bahdanau attention"""    
    
    def __init__(self, rnn_config, rnn_units, dense_units, attention_units, drop_prob):
        """rnn_config - list(str) - list with LSTM layers' names
           rnn_units - int - number of hidden units in rnn
           dense_units - int - number of hidden units in output layer
           attention_units - int - number of hidden units in attention layer
           drop_prob - float - recurrent dropout probability (1 - keep_prob)
        """
        
        super().__init__()
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.attention_units = attention_units
        self.drop_prob = drop_prob
        for i in rnn_config:
            layer = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, recurrent_dropout=drop_prob)
            setattr(self, i, layer)        
        # attention layer
        self.attention = BahdanauAttention(self.attention_units)
        # output layer
        self.dense_layer = tf.keras.layers.Dense(dense_units)
        
    def call(self, initial_input, initial_state, encoder_output, seq_len):
        """inputs:         
               initial_input - tensor (-1, embedding_dim, sequence_length)
               initial_state - list of tensors (-1, rnn_units) - initial hidden state (usually, encoder last hidden state)
               encoder_output - tensor (-1, encoder rnn_units) - sequence of encoder outputs
               seq_len - int - length of input sequence
            
           outputs:
               outputs - tensor (-1, dense_units) - sequence of logits
               hidden - list of tensors (-1, rnn_units) - last hidden state
               cell - list of tensors (-1, rnn_units) - last cell state
        """
        
        hidden, cell = initial_state
        # prepare tensor to store output
        outputs = tf.zeros([initial_input.shape[0], 1, self.dense_units])     
        
        # forward pass through words in sequence
        for i in range(seq_len):
            # slice inputs
            inputs = initial_input[:, i, :]
            
            # build context vector using attention mechanism
            context_vector, attention_weights = self.attention(hidden[-1], encoder_output)
            
            # build decoder input via concating context vector + sliced input
            decoder_input = tf.concat([tf.expand_dims(context_vector, 1), tf.expand_dims(inputs, 1)], axis=-1)

            # forward pass through all layers
            for j, rnn_layer in enumerate(self.layers):
                if isinstance(rnn_layer, tf.keras.layers.LSTM):
                    if j == 0:
                        output, hidden[j], cell[j] = rnn_layer(decoder_input, [hidden[j], cell[j]])
                    else:
                        output, hidden[j], cell[j] = rnn_layer(output, [hidden[j], cell[j]])
            
            # get logits
            output = tf.reshape(output, (-1, output.shape[2]))
            logits = self.dense_layer(output)
            
            # store current outputs
            outputs = tf.concat([outputs, tf.expand_dims(logits, 1)], 1)
        outputs = outputs[:, 1:, :]

        return outputs, hidden, cell