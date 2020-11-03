import datetime
import logging
import yaml
from logging import Formatter, Handler

import requests
from dateutil import tz
from pathlib import Path


file_dir = Path(__file__).parent.resolve()
file_name = file_dir/"values.yaml"
if file_name.exists():
    stream = open(file_name, 'r')
    values = yaml.load(stream, Loader=yaml.FullLoader)

    TELEGRAM_TOKEN = values['TELEGRAM_TOKEN']
    TELEGRAM_CHAT_ID = values['CHAT_ID']
else:
    print("No configuration file for telegram logger!")


class RequestsHandler(Handler):
    def emit(self, record):
        log_entry = self.format(record)
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": log_entry, "parse_mode": "HTML"}
        return requests.post(
            "https://api.telegram.org/bot{token}/sendMessage".format(
                token=TELEGRAM_TOKEN
            ),
            data=payload,
        ).content


class LogstashFormatter(Formatter):
    def __init__(self):
        super(LogstashFormatter, self).__init__()

    def format(self, record):
        """
        Formatting the output in a nice, convenient way.
        """
        # get the current utc time
        t = datetime.datetime.utcnow()

        # specifiy the timezones
        to_zone = tz.gettz("Europe/Berlin")
        from_zone = tz.gettz("UTC")

        # transform to whished timezone
        # taken from
        # https://stackoverflow.com/questions/4770297/convert-utc-datetime-string-to-local-datetime
        t = t.replace(tzinfo=from_zone)
        t = t.astimezone(to_zone).strftime("%d.%m.%Y %H:%M:%S")

        return "<i>{datetime}</i><pre>\n{message}</pre>".format(
            message=record.msg, datetime=t
        )


def make_notifier():
    logger = logging.getLogger("Notifier")

    if not len(logger.handlers):
        handler = RequestsHandler()
        formatter = LogstashFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger
