# Seq2Seq based chatbot

### Descriprion

This is a straightforward end-to-end implementation of chatbot based on deep RNN with LSTM cells. Chatbot is trained on Cornell Movie-Dialogs Corpus. The idea of this project (and some scripts) is base on [HSE NLP Coursera course](https://www.coursera.org/learn/language-processing) honor assignment.

### Model parameters

Config file input_params.yaml has the following structure:
- <MODEL_NAME>:
	- READ_DATA_PARAMS:
		- SENT_LEN: maximum sentence lenths
		- KIND: preprocessing type
	- VOCAB_PARAMS:
		- MAX_LEN: maximum sequence lenths
		- MIN_LEN: maximum sequence lenths
		- NUM_WORDS: number of words in vocabulary
	- MODEL_PARAMS:
		- BATCH_SIZE: batch size
		- DROP_PROB: dropout probability (1 - keep_prob)
		- EMBEDDING_DIMS: embedding dimension
		- RNN_UNITS: number of rnn hidden units
		- ATTENTION_UNITS: number of attention hidden units
		- IF_USE_ATTENTION: enables attention mechanism
		- TRAINABLE_EMBEDDING: allows model to train embedding layer
		- ENCODER_CONFIG: list of encoder layers
		- DECODER_CONFIG: list of decoder layers
	- TRAIN_PARAMS:
		- EPOCHS: number of epochs
		- EVAL_FREQ: validation frequency
		- SAVE_PATH: directory for storing weights
	
### Installation

To install all packages simply run

```bash
pip install -U -r requirements.txt
```

Requirements are
* tensorflow==2.1.0
* numpy==1.18.2
* PyYAML==5.3.1
* scikit-learn==0.22.2.post1
* nltk==3.4.5
* tqdm==4.44.0
	
Run build_dependencies.py to download training corpus.

### Chatbot training

To train chatbot set all parameters in input_params.yaml and run train.py:
		
```bash
python train.py <MODEL_NAME>
```

Model weights, tokenizer and metrics are stored in SAVE_PATH directory.

### Chatbot test example

To test chatbot run chat_example.py:

```bash
python chat_example.py <MODEL_NAME>
```

### Telegram chatbot

It is also possible to start interactive telegram bot by running

```bash
python telegram_bot.py <MODEL_NAME> --token <telegram token>
```

To create telegram bot:

1. start chat with @BotFather
2. type /newbot
3. set chatbot name and username
4. use <telegram token> (HTTP API token) to access chatbot

