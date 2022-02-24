import bs4
import json
import requests


def parse_item_by_url(url, **kwargs):
    html_doc = requests.get(url, allow_redirects=True).content

    soup = bs4.BeautifulSoup(html_doc, 'html.parser')

    # TODO: parse other needed properties. Example:
    # 'link': 'https://www.lamoda.ru/p/mp002xm1gzy3/clothes-northland-bryuki-gornolyzhnye/',
    # 'img': 'MP002XM1GZY3_13289924_1_v1_2x.jpg',
    # 'category': 'Брюки',
    # 'season': 'демисезон',
    # 'color': 'черный',
    # 'print': 'однотонный',
    # 'country': 'Вьетнам',
    # 'brand': 'Northland'
    result = {
        'img_url': 'https:'+ soup.find_all('img')[0]['src'],
    }

    result.update(kwargs)

    return result


def parse_idea_by_url(url, **kwargs):
    # TODO: implement (not working now)

    html_doc = requests.get(url, allow_redirects=True).content

    soup = bs4.BeautifulSoup(html_doc, 'html.parser')

    imgs = []
    img_ids = []

    result = {
        'idea_url': url,
        'items': img_ids,
    }

    result.update(kwargs)

    return result


def normalize_lamoda_id(lamoda_id):
    assert isinstance(lamoda_id, str)
    return lamoda_id.split(' ')[-1].split('#')[-1].split('/')[-1].upper()


def suggest_url(normalized_lamoda_id):
    return 'https://www.lamoda.ru/p/{}'.format(normalized_lamoda_id)


def parse_lamoda_id_from_img_url(url):
    return url.split('/')[-1].split('_')[0]


def parse_lamoda_id_from_item_url(url):
    return url.split('/')[-3].upper()


def parse_item_by_id(lamoda_id):
    normalized_lamoda_id = normalize_lamoda_id(lamoda_id)
    return parse_item_by_url(
        suggest_url(normalized_lamoda_id.lower()),
        id=normalized_lamoda_id,
        id_type='lamoda_id',
    )


def parse_lamoda_items_to_json(item_ids):
    result = []
    for lamoda_id in item_ids:
        result.append(parse_item_by_id(lamoda_id))

    return json.dumps(result, indent=4)

def main():
    item_ids = [
        'Брюки Befree: #MP002XM09L02',
        'Худи Mango Man: #RTLAAY403301',
    ]
    print(parse_lamoda_items_to_json(item_ids[:3]))

def idea():
    url = 'https://www.lamoda.ru/discovery-women/?sitelink=topmenuM&l=1&id=1208'
    print(parse_idea_by_url(url))


if __name__ == '__main__':
    main()
    # idea()
