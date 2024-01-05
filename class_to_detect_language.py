import whisper

class LanguageDetector:
    def __init__(self, model_name="tiny"):
        """
        Initialize the language detector with a specific Whisper model.

        Args:
        model_name (str, optional): Name of the Whisper model to use. Defaults to "tiny".
        """
        self.model = whisper.load_model(model_name, device='cpu')

    def detect_language(self, audio_path):
        """
        Detect the spoken language in an audio file.

        Args:
        audio_path (str): Path to the audio file.

        Returns:
        int or str: Code representing the detected language (0 for Japanese, 1 for English,
                    2 for Chinese), or a message if the language is not Japanese, Chinese, or English.
        """
        # Load and prepare the audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        # Create a Mel spectrogram of the audio
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # Detect the spoken language
        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)

        # Return a code based on the detected language
        if detected_language == 'ja':
            return 0
        elif detected_language == 'en':
            return 1
        elif detected_language == 'zh':
            return 2
        else:
            return "The language of the audio isn't Japanese, Chinese, or English."

# Example usage:
detector = LanguageDetector()
print(detector.detect_language("test_audios/video1548721825-[AudioTrimmerf32f32.com]f32f32.wav"))
