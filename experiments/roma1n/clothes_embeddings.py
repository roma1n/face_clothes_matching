import json

import config
from lib.models import fashion_item_embedder
from lib.utils import pipeline


def main():
    result = pipeline.process_dir_with_model(
        model=fashion_item_embedder.FashionItemEmbedder(),
        batch_size=100,
        input_path=config.LAMODA_IMG_WITH_SEGMENTATION_DIR,
        preprocessed=True,
        postprocess_result=lambda x: x.tolist(),
    )

    print('Saving to file...')

    with open(config.LAMODA_FASHION_ITEM_EMBDDEINGS_PATH, 'w') as f:
        f.write(json.dumps(result, indent=4))

    print('Done!')


if __name__ == '__main__':
    main()
