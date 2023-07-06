from transformers import pipelines
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperConfig
from transformers import WhisperForConditionalGeneration


model_types = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2"
}


def setup_whisper(whisper_model_type: str) -> pipelines.Pipeline:

    """
    Sets up whisper, based on the whisper model type
    """

    if not whisper_model_type in model_types:
        raise Exception("Model Type does not exist", "valid types are -> ", list(model_types.keys()))
        return None
    


    whisper_model_name = model_types[whisper_model_type]

    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

    whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model_name)

    whisper_ft_extractor = WhisperFeatureExtractor()

    whisper_pipe = pipelines.pipeline("automatic-speech-recognition", model=whisper_model,tokenizer=whisper_tokenizer,feature_extractor=whisper_ft_extractor)
    

    # sample_test_data = "dataset\clips\common_voice_en_37027966.mp3"

    return whisper_pipe