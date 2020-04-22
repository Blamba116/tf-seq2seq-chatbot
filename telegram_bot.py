# code is taken from examples in https://www.coursera.org/learn/language-processing

import requests
import time
import argparse
import os
import json

from requests.compat import urljoin
import argparse
import chatbot


class BotHandler(object):
    """
        BotHandler is a class which implements all back-end of the bot.
        It has tree main functions:
            'get_updates' — checks for new messages
            'send_message' – posts new message to user
            'get_answer' — computes the most relevant on a user's question
    """

    def __init__(self, token, dialogue_manager):
        self.token = token
        self.api_url = "https://api.telegram.org/bot{}/".format(token)
        self.dialogue_manager = dialogue_manager

    def get_updates(self, offset=None, timeout=30):
        params = {"timeout": timeout, "offset": offset}
        raw_resp = requests.get(urljoin(self.api_url, "getUpdates"), params)
        try:
            resp = raw_resp.json()
        except json.decoder.JSONDecodeError as e:
            print("Failed to parse response {}: {}.".format(raw_resp.content, e))
            return []

        if "result" not in resp:
            return []
        return resp["result"]

    def send_message(self, chat_id, text):
        params = {"chat_id": chat_id, "text": text}
        return requests.post(urljoin(self.api_url, "sendMessage"), params)

    def get_answer(self, question):
        if question == '/start':
            return "Hi, I am your project bot. How can I help you today?"
        return self.dialogue_manager.get_answer([question])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--token', type=str, default='')
    return parser.parse_args()


def is_unicode(text):
    return len(text) == len(text.encode())
        

def main():
    args = parse_args()
    model_name = args.model_name
    token = args.token

    if not token:
        if not "TELEGRAM_TOKEN" in os.environ:
            print("Please, set bot token through --token or TELEGRAM_TOKEN env variable")
            return
        token = os.environ["TELEGRAM_TOKEN"]
    
    simple_manager = chatbot.Chatbot(model_name)
    simple_manager.load_model()
    bot = BotHandler(token, simple_manager)

    print("Ready to talk!")
    offset = 0
    while True:
        updates = bot.get_updates(offset=offset)
        for update in updates:
            print("An update received.")
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                if "text" in update["message"]:
                    text = update["message"]["text"]
                    if is_unicode(text):
                        print("Update content: {}".format(update))
                        bot.send_message(chat_id, bot.get_answer(update["message"]["text"]))
                    else:
                        bot.send_message(chat_id, "Hmm, you are sending some weird characters to me...")
            offset = max(offset, update['update_id'] + 1)
        time.sleep(1)

if __name__ == "__main__":
    main()
