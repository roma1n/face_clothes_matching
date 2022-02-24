import logging
import os
import pandas as pd
import sys
import tqdm

from lib.parsers import lamoda_parser
from lib.utils import web


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_man_id(man_ids, raw_man_id):
    man_id = man_ids[raw_man_id] if raw_man_id in man_ids.keys() else len(man_ids)
    man_ids[raw_man_id] = man_id
    return man_id


def parse_raw_csv(raw_csv, out_csv, img_dir):
    logging.info('Parsing raw csv file')

    df = pd.read_csv(raw_csv)

    gender_ru_to_en = {
        'ж': 'f',
        'м': 'm',
    }
    man_ids = {}

    result = [
        {
            'man_id': get_man_id(man_ids, row['id_man']),
            'id': lamoda_parser.parse_lamoda_id_from_item_url(item_url_or_id) \
                if item_url_or_id.startswith('https') else lamoda_parser.normalize_lamoda_id(item_url_or_id),
            'id_type': 'lamoda_id',
            'face_img': '{}.png'.format(row['face_img']),
            'gender': gender_ru_to_en[row['id_man'][0]],
        } for row in df.to_dict('records') for item_url_or_id in row['id_lamoda'].split('\n')
    ]

    logging.info('Parsing img urls from lamoda')

    for row in tqdm.tqdm(result):
        row.update({'img_url': lamoda_parser.parse_item_by_id(row['id'])['img_url']})

    logging.info('Validation sample len: {}'.format(len(result)))

    result_df = pd.DataFrame(result)
    result_df.to_csv(
        out_csv,
        index=False,
    )

    logging.info('Validation meta saved to {}'.format(out_csv))

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    logging.info('Downloading pictures from lamoda')

    for row in tqdm.tqdm(result):
        web.download_file(
            url=row['img_url'],
            local_filename=os.path.join(img_dir, '{}.jpg'.format(row['id'])),
        )

    logging.info('Done!')


def main():
    parse_raw_csv(
        raw_csv=os.path.join(
            os.environ['PROJECT_DIR'],
            'data',
            'lamoda',
            'hornest_val_raw.csv',
        ),
        out_csv=os.path.join(
            os.environ['PROJECT_DIR'],
            'data',
            'lamoda',
            'validation',
            'meta.csv',
        ),
        img_dir=os.path.join(
            os.environ['PROJECT_DIR'],
            'data',
            'lamoda',
            'validation',
            'fashion_items',
        ),
    )


if __name__ == '__main__':
    main()
