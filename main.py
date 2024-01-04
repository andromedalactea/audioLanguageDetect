from scripts.detect_language import detect_language
from scripts.pre_procesing_audio import convert_audio_format
import timeit

def main(audio_path, model_name='tiny'):

    # Detect language
    code_language = detect_language(audio_path, model_name)

    # Return the code
    return code_language




# print(timeit.timeit(lambda: main("test_audios/Chapter 9 - Dialogue.mp3"), number=10) * 10)
