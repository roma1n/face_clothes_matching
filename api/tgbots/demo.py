import json
import os
import requests
import telebot

import config
from lib.inference import fashion_item_recomendator
from lib.utils import web

suggester = fashion_item_recomendator.FashionItemRecomedator()

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
        'Processing. It takes up about 20 seconds.',
    )
    local_path = save_file(message.photo[-1].file_id)

    suggestions = suggester.process_face_img(local_path)

    for fashion_item_img_path, fashion_item_url in zip(
        suggestions['img_paths'],
        suggestions['urls'],
    ):
        with open(fashion_item_img_path, 'rb') as photo:
            bot.send_photo(
                message.from_user.id,
                photo,
                caption=fashion_item_url,
            )



if __name__ == '__main__':
    bot.remove_webhook()
    bot.infinity_polling()
