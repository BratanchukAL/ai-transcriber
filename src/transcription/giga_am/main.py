import argparse
import json
import os
import time

import gigaam


#
import hydra
import torch
from gigaam import GigaAMASR
from gigaam.model import LONGFORM_THRESHOLD

from segmentation import transcribe


class ExGigaAMASR(GigaAMASR):
    def __init__(self, model: GigaAMASR):
        super().__init__(model.cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    @torch.inference_mode()
    def transcribe(self, wav_file: str) -> str:
        """
        Transcribes a short audio file into text.
        """
        wav, length = self.prepare_wav(wav_file)
        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        return self.decoding.decode(self.head, encoded, encoded_len)[0]

    @property
    def device(self) -> torch.device:
        return super()._device

def run_giga_am(audio_path: str):
    # Audio embeddings
    # model_name = "v3_ssl"  # Options: `v1_ssl`, `v2_ssl`, `v3_ssl`
    # model = gigaam.load_model(model_name)
    # embedding, _ = model.embed_audio(audio_path)
    # print(embedding)

    # ASR
    model_name = "v3_e2e_rnnt"  # Options: any model version with suffix `_ctc` or `_rnnt`
    model = gigaam.load_model(model_name)
    wav, length = model.prepare_wav(audio_path)
    encoded, encoded_len = model.forward(wav, length)

    transcribe(ExGigaAMASR(model),
               audio_path,
               )

    # wav, length = model.prepare_wav(audio_path)
    # if length.item() > LONGFORM_THRESHOLD:
    #     pass
    # encoded, encoded_len = model.forward(wav, length)
    # a  = model.decoding.decode(model.head, encoded, encoded_len)[0]

    # utterances = model.transcribe_longform(audio_path)
    # for utt in utterances:
    #     transcription, (start, end) = utt["transcription"], utt["boundaries"]
    #     print(f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}")

    # model_name = "v3_e2e_rnnt"  # Options: any model version with suffix `_ctc` or `_rnnt`
    # model = gigaam.load_model(model_name)
    transcription = model.transcribe(audio_path)
    # print(transcription)
    #
    # model_name = "v3_e2e_ctc"  # Options: any model version with suffix `_ctc` or `_rnnt`
    # model = gigaam.load_model(model_name)
    # transcription = model.transcribe(audio_path)
    # print(transcription)

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
