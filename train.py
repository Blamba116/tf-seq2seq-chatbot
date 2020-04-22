import argparse
import chatbot

parser = argparse.ArgumentParser()
parser.add_argument('model_name')
args = parser.parse_args()
model_name = args.model_name

bot = chatbot.Chatbot(model_name)
bot.make_dataset()
bot.train()
bot.save_model()