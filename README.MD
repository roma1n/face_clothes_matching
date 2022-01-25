# Face clothes matching

## Настройка окружения

### Установка необходимых пакетов

```bash
sudo apt install tmux htop python-is-python3 pipenv zip
```

### Утсановка python3.7

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
```

### Яндекс.Диск

Установка:

```bash
echo "deb http://repo.yandex.ru/yandex-disk/deb/ stable main" | sudo tee -a /etc/apt/sources.list.d/yandex-disk.list > /dev/null && wget http://repo.yandex.ru/yandex-disk/YANDEX-DISK-KEY.GPG -O- | sudo apt-key add - && sudo apt-get update && sudo apt-get install -y yandex-disk

yandex-disk setup
```

[Дока Яндекс.Диска](https://yandex.ru/support/disk-desktop-linux/cli-commands.html)

### Запуск среды

```bash
./run_env.sh
```

### Tensorboard

Установка
```
pip install tensorflow tensorboard
```

Запуск

```bash
python /home/roma1n/.local/lib/python3.8/site-packages/tensorboard/main.py --logdir=lightning_logs/version_5/ --port 10200 --bind_all
```