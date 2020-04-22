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
    
# Seq2Seq class
class Seq2Seq(tf.keras.Model):
    def __init__(self, embedding, encoder, decoder, if_use_attention=False
                 , trainable_embedding=True):
        super().__init__()
           
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = tf.keras.optimizers.Adam()
        self.if_use_attention = if_use_attention
        self.trainable_embedding = trainable_embedding
        
    def loss_function(self, y_pred, y):
        entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = entropy(y_true=y, y_pred=y_pred)
        mask = tf.logical_not(tf.math.equal(y, 0))
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss
        loss = tf.reduce_mean(loss)

        return loss
    
    def forward_pass(self, input_batch, output_batch):
        encoder_emb_inp = self.embedding(input_batch)
        decoder_input = output_batch[:, :-1] # ignore <end>
        decoder_output = output_batch[:, 1:] #ignore <start>
        decoder_emb_inp = self.embedding(decoder_input)

        a, a_tx, c_tx = self.encoder(encoder_emb_inp)

        if self.if_use_attention:
            logits, _, _ = self.decoder(decoder_emb_inp, [a_tx, c_tx], a, decoder_emb_inp.shape[1])
        else:
            logits, _, _ = self.decoder(decoder_emb_inp, [a_tx, c_tx])
            
        return logits, decoder_output
        
    def train_on_batch(self, input_batch, output_batch):
        with tf.GradientTape() as tape:
            logits, decoder_output = self.forward_pass(input_batch, output_batch)
            #Calculate loss
            batch_loss = self.loss_function(logits, decoder_output)

        if self.trainable_embedding:
            variables = self.trainable_variables
        else:
            variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(batch_loss, variables)
        grads_and_vars = zip(gradients, variables)
        self.optimizer.apply_gradients(grads_and_vars)
        
        return batch_loss
    
    def evaluate(self, val_dataset):
        val_loss = 0.0
        for iters, (input_batch, output_batch) in enumerate(val_dataset):
            logits, decoder_output = self.forward_pass(input_batch, output_batch)
            val_loss += self.loss_function(logits, decoder_output)
            
        return val_loss, val_loss / (iters + 1)
    
    def fit(self, dataset, epochs, steps_per_epoch, val_dataset=None, eval_freq=None):
        history = {'train':[], 'avg_train':[], 'val':[], 'avg_val':[]}
        
        for i in range(1, epochs + 1):
            total_loss = 0.0
            progbar = tf.keras.utils.Progbar(steps_per_epoch)
            for (batch , (input_batch, output_batch)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_on_batch(input_batch, output_batch)
                total_loss += batch_loss
                progbar.add(1, values=[("train loss", batch_loss)])
                
            history['train'].append(total_loss.numpy())
            history['avg_train'].append(total_loss.numpy() / (batch + 1))            
            if i % eval_freq == 0 and val_dataset is not None:
                val_loss, val_avg_loss = self.evaluate(val_dataset)
                history['val'].append(val_loss.numpy())
                history['avg_val'].append(val_avg_loss.numpy())
                print ('\n val loss = %f, val avg loss = %f \n'%(history['val'][-1], history['avg_val'][-1]))
            
        return history
    
    def predict(self, input_batch, max_len, start_token):
        encoder_emb_inp = self.embedding(input_batch)
        a, a_tx, c_tx = self.encoder(encoder_emb_inp)

        decoder_input = [[start_token]] * input_batch.shape[0]
        decoder_input = tf.convert_to_tensor(decoder_input, dtype=tf.int64)

        predictions = decoder_input * 1 # copy tensor
        for i in range(max_len):
            decoder_emb_inp = self.embedding(decoder_input)

            if self.if_use_attention:
                logits, a_tx, c_tx = self.decoder(decoder_emb_inp, [a_tx, c_tx], a, decoder_emb_inp.shape[1])
            else:
                logits, a_tx, c_tx = self.decoder(decoder_emb_inp, [a_tx, c_tx])

            decoder_input = tf.argmax(logits, -1)
            predictions = tf.concat([predictions, decoder_input], -1)

        return predictions.numpy()