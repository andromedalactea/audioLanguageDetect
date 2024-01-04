import librosa
import soundfile as sf

def convert_audio_format(audio_path):
    # Cargar el audio con librosa para conversión automática a float32
    audio, sample_rate = librosa.load(audio_path, sr=None)

    # Changing the ouput path
    output_path = audio_path.replace('.', 'f32.')
    # Guardar el audio convertido en float32
    sf.write(output_path, audio, sample_rate)

    return output_path