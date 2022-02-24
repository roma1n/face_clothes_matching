import flask
import json
import pandas as pd
import requests
import telebot

from api.tgbots import demo
import config


def run_with_webhook(bot_module):
    bot_module.bot.remove_webhook()
    with open(config.SSL_CERT_PATH, 'r') as f:
        bot_module.bot.set_webhook(
            url='https://{}:{}/tgbots/{}'.format(config.API_HOST, config.API_PORT, bot_module.TOKEN),
            certificate=f,
        )


def get_bot_updates_processor(bot_module):
    def func():
        try:
            bot_module.bot.process_new_updates(
                [telebot.types.Update.de_json(flask.request.stream.read().decode('utf-8'))]
            )
        except Exception as e:
            return e, 200
        return 'Ok', 200
    return func


def get_bot_uri(bot_module):
    return '/tgbots/{}'.format(bot_module.TOKEN)


def stats():
    try:
        url_template = 'https://api.telegram.org/bot{}/getWebhookInfo'
        result = []
        for bot_module in bot_modules:
            result.append(
                json.loads(requests.get(url_template.format(bot_module.TOKEN)).content)['result']
            )
        df = pd.DataFrame(result)
        return df.to_html(), 200
    except Exception as e:
        return e, 200


def urls_to_view_funcs():
    d = {
        '/tgbots': stats,
    }
    d.update({
        get_bot_uri(bot_module): get_bot_updates_processor(bot_module) for bot_module in bot_modules
    })
    return d


bot_modules = [
    demo,
]

for bot_module in bot_modules:
    run_with_webhook(bot_module)
