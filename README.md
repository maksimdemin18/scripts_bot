Локально:

cd TelegramBot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python script_runner_bot_v8.py


Через Docker:

cd TelegramBot
docker compose build
docker compose up -d
docker compose logs -f tg-scripts-bot


## Добавление скриптов

* Классический способ — описать их в `bot.yaml` (пример уже в репозитории).
* Теперь можно класть автономные конфиги в `scripts/<name>/script.yaml` — бот найдёт их автоматически при старте или команде `/cfg reload`.
* Корневая папка с автопоиском задаётся через переменную окружения `SCRIPTS_DIR` (по умолчанию `scripts`).
