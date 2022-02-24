import json
import os
import requests
import telebot

import config
from lib.utils import web


with open(config.DEMO_BOT_TOKEN_PATH, 'r') as f:
    TOKEN = f.read()


bot = telebot.TeleBot(TOKEN)


def get_file_path(file_id):
    url_template = 'https://api.telegram.org/bot{}/getFile?file_id={}'
    resp = requests.get(
        url_template.format(
            TOKEN,
            file_id,
        ),
    )
    return json.loads(resp.content)['result']['file_path']


def save_file(file_id):
    if not os.path.exists(config.BOT_DATA_DIR):
        os.mkdir(config.BOT_DATA_DIR)
    url_template = 'https://api.telegram.org/file/bot{}/{}'
    file_path = get_file_path(file_id)
    local_path = os.path.join(config.BOT_DATA_DIR, file_path.split('/')[-1])
    web.download_file(
        url_template.format(TOKEN, file_path),
        local_path,
    )
    return local_path



@bot.message_handler(commands=['start', 'help'], content_types=['text'])
def send_welcome(message):
    bot.send_message(
        message.from_user.id,
        'Hello. Send me your photo!'
    )

@bot.message_handler(content_types=['photo'])
def show_demo(message):
    bot.send_message(
        message.from_user.id,
        'Processing...',
    )
    local_path = save_file(message.photo[-1].file_id)


if __name__ == '__main__':
    bot.remove_webhook()
    bot.infinity_polling()
