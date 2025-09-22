## Pre-trained Models

Due to GitHub’s file size restrictions, large model files are **not included** in this repository.  
Please download them separately and place them in the appropriate project directories.

- **Flan-T5 Large** (used for fallback text generation)  
  [Hugging Face Model Card](https://huggingface.co/google/flan-t5-large)

- **Emotion Classification Model Files**  
  [Hugging Face Emotion Dataset / Models](https://huggingface.co/datasets/dair-ai/emotion)  
  (You may use a fine-tuned DistilBERT/Emotion classifier; see report for details.)

- **Whisper Tiny (English)** (used for speech-to-text)  
  [OpenAI Whisper Model – Tiny EN](https://huggingface.co/openai/whisper-tiny.en)

👉 After downloading, place them in your project directory as follows:
MindmateClean/
├── main_gui.py
├── logger.py
├── cortana_widget.py
├── requirements.txt
├── models/
│ ├── flan-t5-large/
│ ├── emotion/
│ └── tiny_en/
