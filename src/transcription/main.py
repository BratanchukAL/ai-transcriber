import argparse
import json
import os
import time

import torch
import whisper

# Requirement PATH : to ffmpeg

# Аргументы командной строки
p = argparse.ArgumentParser()
# Обязательный аргумент -- путь к файлу.
p.add_argument("path", help="Аудио‑ или видеофайл")

# Необязательный аргумент --lang, по умолчанию -- русский.
p.add_argument("--lang", default="ru")

# Параметр случайности (температура) для модели, влияет на вариативность результатов.'''
p.add_argument("--temperature", type=float, default=0)

# Размер beam search (если используется), влияет на точность/варианты распознавания.
p.add_argument("--beam_size", type=int)

# Булев флаг: использовать ли предыдущий текст как контекст при распознавании.
p.add_argument("--condition", action="store_true",
               help="condition_on_previous_text")

# Необязательный текстовый ввод -- вводная для модели (можно подсказать контекст/темы).
p.add_argument("--prompt", default="",
               help="Вводная: тема, участники, термины...")

# Парсит все аргументы из командной строки в объект args.
args = p.parse_args()

if not os.path.isfile(args.path):
    raise SystemExit(f"Файл не найден: {args.path}")

# Проверяет, доступна ли видеокарта CUDA (GPU). Если нет -- используется CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Выводит на каком устройстве будет происходить транскрипция.
print(f"Устройство: {device.upper()} --",
      torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

# Загружает модель large-v3 на выбранное устройство (GPU или CPU).
model = whisper.load_model("large-v3", device=device)

# Сообщение о старте и начало отсчёта времени.
print("Транскрибирование...")
t0 = time.time()
result = model.transcribe(
    args.path,
    language=args.lang,
    temperature=args.temperature,
    beam_size=args.beam_size,
    condition_on_previous_text=args.condition,
    initial_prompt=args.prompt or None
)

# Показывает, сколько времени заняла транскрипция.
print(f"{round(time.time() - t0, 2)} с")

# Сохранение результата в JSON и TXT
base = os.path.splitext(args.path)[0]
with open(base + ".json", "w", encoding="utf-8") as f:
    json.dump(result["segments"], f, ensure_ascii=False, indent=4)
with open(base + ".txt", "w", encoding="utf-8") as f:
    f.write(result["text"])
print("Сохранено:", base + ".json / .txt")
