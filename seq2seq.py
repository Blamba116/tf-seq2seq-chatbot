import tensorflow as tf

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