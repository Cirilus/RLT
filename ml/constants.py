MODEL_EMB_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda"
MODEL_CHAT_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SYSTEM_PROMPT = "Ты — Ассистент для Мегафон, ты отвечаешь на их вопросы и помогаешь им."


SYSTEM_TOKEN = 1587
USER_TOKEN = 2188
BOT_TOKEN = 12435
LINEBREAK_TOKEN = 13

ROLE_TOKENS = {
    "user": USER_TOKEN,
    "bot": BOT_TOKEN,
    "system": SYSTEM_TOKEN
}

