import argparse
import chatbot

parser = argparse.ArgumentParser()
parser.add_argument('model_name')
args = parser.parse_args()
model_name = args.model_name

bot = chatbot.Chatbot(model_name)
bot.load_model()

input_raw = ['how are you', 'do you like me', 'what is your hobby', 'do you like football',
             , 'Hi!', "whats your name?", 'Tell me about yourself'
             , 'Do you love me?', "What's the meaning of life?", 'How is the weather today?', "Let's have a dinner!"
             , 'Are you a bot?', 'Why not?']

predictions = bot.get_answer(input_raw)

for k, (i, j) in enumerate(zip(input_raw, predictions)):
    print('%i\n O: %s\n B: %s'%(k, i, j))