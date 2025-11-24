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
