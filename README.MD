# Face clothes matching

## Кодстайл

1. pep8
2. Импортировать модули, а не библиотеки. Пример: вместо
```python
from lib.unet import UNET
model = UNET()
```

Можно сделать так:

```python
from lib import unet
model = unet.UNET()
```

3. Сортировка импортов по алфавиту, если из одной директории импортируется несколько модулей, то импортировать все одним импортом, каждый с новой строчки, после последнего модуля тоже запятая. Привер: вместо

```python
from lib.torch_models import unet
from lib.torch_models import dssm
```

Или

```python
from lib.torch_models import dssm, unet
```

Можно сделать так:

```python
from lib.torch_models import (
    dssm,
    unet,
)
```

В репо уже могут быть ошибки, их стоит исправлять :)

## lib

### torch_models

Необходимые стандартные архитектуры, реализованные на torch

### ml_tasks

ML задачи, реализованные в общем виде (например, классификация, сегментация и тп) с помощью pytorch lightning

### models

ML модели, которые используются в пайплайне. Если используются готовые модели, то загрузка и применение моделей. Если модели обучаются через ml_tasks, то запуск обучения (загрузка правильных датасетов, установка параметров)

### raw_datasets

Чистка / приведение к удобному виду датасетов

### inference

Задачи для инференса (например, обучить модели, посчитать эмбеддинги одежды)

## Настройка окружения

### Установка необходимых пакетов (нужно для удобства / корректной работы некоторых пакетов в окружении)

```bash
sudo apt install tmux htop python-is-python3 pipenv zip -y
sudo apt install ffmpeg libsm6 libxext6  -y
```

### Утсановка python3.7 (нужно для запуска окружения)

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
```

### Яндекс.Диск (для загрузки датасетов с/на диск; НЕ нужно чтобы скачивать датасеты по паблик ссылкам)

Установка:

```bash
echo "deb http://repo.yandex.ru/yandex-disk/deb/ stable main" | sudo tee -a /etc/apt/sources.list.d/yandex-disk.list > /dev/null && wget http://repo.yandex.ru/yandex-disk/YANDEX-DISK-KEY.GPG -O- | sudo apt-key add - && sudo apt-get update && sudo apt-get install -y yandex-disk

yandex-disk setup
```

[Дока Яндекс.Диска](https://yandex.ru/support/disk-desktop-linux/cli-commands.html)

### Запуск среды (создает pipenv окружение и запускает его)

```bash
./run_env.sh
```

### Tensorboard (чтобы смотреть графики при обучении через pytorch lightning)

Установка
```
pip install tensorflow tensorboard
```

Запуск

```bash
python /home/roma1n/.local/lib/python3.8/site-packages/tensorboard/main.py --logdir=lightning_logs/version_5/ --port 10200 --bind_all
```