
# Getting Started:

Whisper: https://huggingface.co/openai/whisper-large-v2/ <br>
Dataset from: https://commonvoice.mozilla.org/en/dataset (Common Voice Delta Segment 13.0)

## Installation Requirements: (Windows 11, Python 3.10.11) <br>
1. Pytorch: https://pytorch.org/get-started/locally/
     <br> ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117```
2. Chocolatey: https://chocolatey.org/install
     <br> Open Powershell as Administrator and run: ```Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))```
3. FFMPEG: Run in command line:
     <br> ```choco install ffmpeg```
4. Hugging Face Transformers: (To use for Whisper)
     <br> ```pip install transformers``` and ```pip install transformers[torch]```
5. Hugging Face Evaluate: https://huggingface.co/evaluate-metric
     <br> ```pip install jiwer``` and then ```pip install evaluate```
6. Hugging Face Sentence Transformers: https://huggingface.co/tasks/sentence-similarity   
     <br> ```pip install -U sentence-transformers```
7. Pandas:
     <br> ```pip install pandas```
8. soundfile:
     <br> ```pip install soundfile``` and ```pip install librosa```

Performance Report: https://docs.google.com/document/d/1Kc378I8IxEUm957ndcmMiJkQyHnHVgJjNDjVGVXzHfk/edit?usp=sharing

## Summary:
Used Hugging Face and its transformers library to use the Whisper model on the Common Voice Delta Segment 13.0 dataset and analyzed its transcribed text results
Attempted to create my own model for binary gender classification using facebook's wav2vec2 model for audio classification



Hugging Face Models used:
- cardiffnlp/twitter-roberta-base-sentiment-latest (for sentiment analysis)
- openai/whisper-large-v2/ (for audio transcription)
- facebook/wav2vec2-base (for audio classification)


## Resources:
- https://huggingface.co/openai/whisper-large-v2/
- https://huggingface.co/docs/transformers/
- https://huggingface.co/evaluate-metric
- https://huggingface.co/tasks/sentence-similarity
- https://chocolatey.org/install
- https://pytorch.org/
- https://huggingface.co/docs/datasets/main/en/
- https://www.sltinfo.com/wp-content/uploads/2014/01/type-token-ratio.pdf
