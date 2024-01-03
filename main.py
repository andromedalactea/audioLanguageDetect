import whisper

def detect_language(audio_path, model_name="base"):
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

# Test the function with an audio file
print(detect_language("/home/andromeda/freelancer/audioLanguageDetect/test_audios/fishing.mp3"))
