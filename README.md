# Seq2Seq based chatbot

### Descriprion

This is a end-to-end implementation of chatbot based on deep RNN with LSTM cells.

Model parameters are set in input_params.yaml:

    READ_DATA_PARAMS:
        SENT_LEN: maximum sentence lenths
        KIND: preprocessing type
    VOCAB_PARAMS:
        MAX_LEN: maximum sequence lenths
        MIN_LEN: maximum sequence lenths
        NUM_WORDS: number of words in vocabulary
    MODEL_PARAMS:
        BATCH_SIZE: batch size
        DROP_PROB: dropout probability (1 - keep_prob)
        EMBEDDING_DIMS: embedding dimension
        RNN_UNITS: number of rnn hidden units
        ATTENTION_UNITS: number of attention hidden units
        IF_USE_ATTENTION: enables attention mechanism
        TRAINABLE_EMBEDDING: allows model to train embedding layer
        ENCODER_CONFIG: list of encoder layers
        DECODER_CONFIG: list of decoder layers
    TRAIN_PARAMS:
        EPOCHS: number of epochs
        EVAL_FREQ: validation frequency
        SAVE_PATH: directory for storing weights
		
### Chatbot training

Set all parameters in input_params.yaml and use train.py to train model:
		
```bash
python train.py <MODEL_NAME>
```

Weights, tokenizer and metrics are stored in SAVE_PATH directory.

### Catbot usage

Example of chatbot answering questions can be found in chat_example.py. This script requires <MODEL_NAME> of a trainded model.

```bash
python chat_example.py <MODEL_NAME>
```

### Telegram chatbot

TODO