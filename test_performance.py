from main import main
import librosa
import timeit
import pandas as pd

# List of audio files
audio_files = ['test_audios/english.wav', 'test_audios/japanese.wav', 'test_audios/chinese.wav']

# Whisper models to test
models = ["tiny", 'base', 'small']

# List to store the results
results = []

# Test each file with each model
for file in audio_files:
    # Get the duration of the audio
    duration = librosa.get_duration(path=file)

    for model in models:
        # Measure execution time and get the result from main
        start_time = timeit.default_timer()
        language_result = main(file, model_name=model)
        execution_time = timeit.default_timer() - start_time

        # Add the results to the array
        results.append({
            "File": file, 
            "Model": model, 
            "Audio Duration (s)": duration, 
            "Execution Time (s)": execution_time,
            "Language Detected": language_result  
        })

# Create a DataFrame with the results
df = pd.DataFrame(results)

print(df)
# Export to CSV
df.to_csv("test_audios/time_execution_audioLanguageDetect.csv", index=False)

# Export to Excel
df.to_excel("test_audios/time_execution_audioLanguageDetect.xlsx", index=False)
