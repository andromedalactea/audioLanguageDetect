# Audio Language Detection with Whisper

## Overview
This repository contains a Python script that utilizes OpenAI's Whisper model to detect the spoken language in an audio file. It's designed to be a simple and effective tool for identifying whether the language in an audio file is Japanese, English, or Chinese.

## Features
- Detects if the spoken language in an audio file is Japanese, English, or Chinese.
- Returns a numerical code representing the detected language (0 for Japanese, 1 for English, 2 for Chinese).
- Informs if the detected language is not among the three mentioned.

## Requirements
- Python 3.8 or higher
- Whisper package from OpenAI

## Installation
Install the Whisper package using pip:
```
pip install openai-whisper
```

## Usage
To use the script, run the `detect_language` function with the path to your audio file:
```python
from scripts.detect_language import detect_language

language_code = detect_language("path_to_your_audio_file")
print(language_code)
```

## Contributing
Contributions to this project are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.
