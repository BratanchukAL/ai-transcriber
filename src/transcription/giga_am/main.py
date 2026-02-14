import argparse
import json
import os
import time

import gigaam


#
def run_giga_am(audio_path):
    # Audio embeddings
    model_name = "v3_ssl"  # Options: `v1_ssl`, `v2_ssl`, `v3_ssl`
    model = gigaam.load_model(model_name)
    embedding, _ = model.embed_audio(audio_path)
    print(embedding)

    # ASR
    model_name = "v3_e2e_rnnt"  # Options: any model version with suffix `_ctc` or `_rnnt`
    model = gigaam.load_model(model_name)
    transcription = model.transcribe(audio_path)
    print(transcription)

    # Emotion recognition
    model = gigaam.load_model("emo")
    emotion2prob = model.get_probs(audio_path)
    print(", ".join([f"{emotion}: {prob:.3f}" for emotion, prob in emotion2prob.items()]))


if __name__ == "__main__":
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

    run_giga_am(args.path)
