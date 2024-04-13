MODEL_EMB_NAME = "intfloat/multilingual-e5-large-instruct"
DEVICE = "cuda"
MODEL_CHAT_NAME = "ml/model-q8_0.gguf"
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

