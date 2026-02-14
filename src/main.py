import glob
import logging
import os
import subprocess
import sys
from functools import reduce
from pathlib import Path
from typing import Union, Iterable, Tuple, List

logger = logging.getLogger(__name__)


def load_files(file_or_dir: Union[str, Path]) -> Iterable[Path]:
    """ Загружает пути к файлам в указанном каталоге и его поддиректориях """
    file_or_dir = Path(file_or_dir)
    files: Union[List[Union[str, Path]]] = []
    strict_file_formats = ['wav', 'mp3', 'm4a']

    if file_or_dir.is_file():
        files = [file_or_dir]
    else:
        files = reduce(lambda accum, cur: accum + glob.glob(os.path.join(file_or_dir, "*." + cur), recursive=False)
        , strict_file_formats, [])

        if not files:
            files = [Path(file_or_dir.__str__() + '.(wav|mp3)')]

    if files and not Path(files[0]).exists():
        files = []

    if not files:
        if file_or_dir.is_dir():
            logger.error(f"В каталоге {file_or_dir} и его поддиректориях не найдено ни одного файла")
        else:
            logger.error(f" {file_or_dir} не найден файл или директория")

    #
    for filename in files:
        try:
            logger.info(f"В обработке файл: {filename} ")
            yield Path(filename)
            logger.info(f"Закончена обработка файла: {filename} ")
        except Exception as e:
            logger.error(f"Ошибка обработки файла {filename}: {str(e)}")


if __name__ == "__main__":

    FFMPEG_PATH_TO_FOLDER = os.environ['FFMPEG_PATH_TO_FOLDER']

    os.environ["PATH"] += os.pathsep + Path(FFMPEG_PATH_TO_FOLDER).absolute().__str__()

    temperature = 0
    beam_size = None
    condition = False
    prompt = ""

    for audio_file in load_files('./../user_data'):
        logger.info(f"ШАГ 1: Транскрипция Whisper")
        # Шаг 1: Транскрипция Whisper
        cmd = [
            sys.executable, "./transcription/whisper/main.py",
            audio_file,
            "--lang", "ru",
            "--temperature", str(temperature),
        ]
        if beam_size:
            cmd += ["--beam_size", str(beam_size)]
        if condition:
            cmd.append("--condition")
        if prompt:
            cmd += ["--prompt", prompt]
        subprocess.run(cmd)

        # # Шаг 2: Диаризация NeMo
        # logger.info(f"Шаг 2: Диаризация NeMo")
        # json_file = os.path.splitext(audio_file)[0] + ".json"
        # subprocess.run([
        #     sys.executable,
        #     "./diarization/main.py",
        #     audio_file,  # <audio.wav>
        #     json_file,  # <whisper.json>
        #     "12"  # <max_speakers>
        # ])

    # def run_pipeline(filepath, use_clean=True, do_summary=True, temperature="0", beam_size=None, condition=False,
    #                  prompt=""):
    #     if not filepath or not os.path.exists(filepath):
    #         raise FileNotFoundError(f"Файл не найден: {filepath}")
    #
    #     base_name = os.path.splitext(filepath)[0]
    #     cleaned_file = base_name + "_cleaned.wav"
    #     audio_file = cleaned_file if use_clean else filepath
    #
    #     # Шаг 1: Очистка аудио (опционально)
    #     if use_clean:
    #         subprocess.run(["python", "clean_audio.py", filepath])
    #
    #     # Шаг 2: Транскрипция Whisper
    #     cmd = [
    #         "python", "transcribe.py",
    #         audio_file,
    #         "--lang", "ru",
    #         "--temperature", temperature,
    #     ]
    #     if beam_size:
    #         cmd += ["--beam_size", str(beam_size)]
    #     if condition:
    #         cmd.append("--condition")
    #     if prompt:
    #         cmd += ["--prompt", prompt]
    #     subprocess.run(cmd)
    #
    #     # Шаг 3: Диаризация NeMo
    #     json_file = os.path.splitext(audio_file)[0] + ".json"
    #     subprocess.run(["python", "diarize_nemo_auto.py", audio_file, json_file, "12"])
    #
    #     # Шаг 4: Конвертация в TXT / MD
    #     tagged_json = os.path.splitext(audio_file)[0] + "_tagged.json"
    #     # if os.path.exists(tagged_json):
    #     #     from convert_tagged_json_to_txt_md import convert_tagged_json_to_txt_md
    #     #     convert_tagged_json_to_txt_md(tagged_json)
    #
    #     # Шаг 5: Генерация саммари (опционально)
    #     if do_summary:
    #         subprocess.run(["python", "summarize_json.py", tagged_json])