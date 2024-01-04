from detect_language import detect_language

def detect_language(audio_path, model_name="tiny"):
    """
    Detect the spoken language in an audio file.

    Args:
    audio_path (str): Path to the audio file.
    model_name (str, optional): Name of the Whisper model to use. Defaults to "base".

    Returns:
    int or str: Code representing the detected language (0 for Japanese, 1 for English,
                2 for Chinese), or a message if the language is not Japanese, Chinese, or English.
    """
    # Load the Whisper model
    model = whisper.load_model(model_name)

    # Load and prepare the audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Create a Mel spectrogram of the audio
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)

    # Return a code based on the detected language
    if detected_language == 'ja':
        code_language = 0
    elif detected_language == 'en':
        code_language = 1
    elif detected_language == 'zh':
        code_language = 2
    else:
        code_language = "The language of the audio isn't Japanese, Chinese, or English."

    return code_language


# Lista de archivos de audio
audio_files = ["test_audios/Chapter 9 - Dialogue.mp3", "test_audios/Chapter 17 - Dialogue.mp3", 'test_audios/fishing.mp3', 'test_audios/japanes_1.mp3',  'test_audios/video1548721825-[AudioTrimmer.com].wav', 'test_audios/video1548721825-[AudioTrimmerf32.com]f32.wav']

# Modelos de Whisper a probar
models = ["tiny", 'base', 'small']

# Lista para almacenar los resultados
results = []

# Testear cada archivo con cada modelo
for file in audio_files:
    # Obtener la duración del audio
    duration = librosa.get_duration(path=file)

    for model in models:
        # Medir el tiempo de ejecución
        execution_time = timeit.timeit(lambda: detect_language(file, model_name=model), number=1) 

        # Agregar los resultados al array
        results.append({"File": file, "Model": model, "Audio Duration (s)": duration , "Execution Time (s)": execution_time})

# Crear un DataFrame con los resultados
df = pd.DataFrame(results)


print(df)
# Exportar a CSV
df.to_csv("test_audios/time_execution_audioLanguageDetect.csv", index=False)

# Exportar a Excel
df.to_excel("test_audios/time_execution_audioLanguageDetect.xlsx", index=False)
