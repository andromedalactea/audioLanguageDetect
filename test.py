import whisper

def detect_language(audio_path, model_name="base"):
    # Cargar el modelo de Whisper
    model = whisper.load_model(model_name)

    # Cargar y preparar el audio
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # Crear un espectrograma Mel del audio
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detectar el idioma hablado
    _, probs = model.detect_language(mel)    
    detected_language = max(probs, key=probs.get)
    
    # Imprimir el texto reconocido
    return detected_language

# Uso de la función con un archivo de audio
print(detect_language("/home/andromeda/freelancer/audioLanguageDetect/test_audios/fishing.mp3"))
