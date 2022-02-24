import logging
import os
import ssl
import sys

import flask

from api import (
    ping,
    tgbots,
)
import config

app = flask.Flask('imagin')
app.logger.setLevel(logging.INFO)

for api_module in [
    ping,
    tgbots,
]:
    for url, view_func in api_module.urls_to_view_funcs().items():
        app.add_url_rule(url, view_func=view_func, methods=['POST', 'GET'])


context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain(config.SSL_CERT_PATH, config.SSL_PKEY_PATH)


if __name__ == '__main__':
    print(tgbots.stats())
    app.run(
        host='0.0.0.0',
        port=config.API_PORT,
        ssl_context=context,
    )
