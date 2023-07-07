import evaluate
import csv
from sentence_transformers import SentenceTransformer, util
import torchaudio
import numpy

def get_validated_data() -> dict:
    """
    Creates a dictionary that maps the validated file name to the data \n
    {[audio_file_name] -> {"client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents", "variant", "locale", "segment"}}
    """

    validated_dict = {}

    with open("./dataset/validated.tsv", 'r', encoding='utf8') as validated_file:
        
        validated_data = csv.reader(validated_file, delimiter='\t')

        header = next(validated_data)
        
        for data in validated_data:

            data_item = {}

            for i in range(len(header)):
                header_name = header[i]
                data_item[header_name] = str(data[i])
            
            file_name = data_item['path']

            validated_dict[file_name] = data_item

    return validated_dict

def wer_calculator():
    """
    returns calculator to calculate word error rate (from evaluate module). call `.compute`
    """
    calc = evaluate.load("wer")
    return calc

def cos_sim_calc(sentence_model: SentenceTransformer, prediction_sentence: str, reference_sentence: str) -> float:
    # taken from https://huggingface.co/tasks/sentence-similarity; calculates sentence semantic
    '''
    calculates cos_sim of two sentences
    '''
    prediction_embedding = sentence_model.encode(prediction_sentence)
    reference_embedding = sentence_model.encode(reference_sentence)

    cosine_similarity =  util.pytorch_cos_sim(prediction_embedding,reference_embedding)

    return cosine_similarity

def get_audio(audio_path: str):
    '''
    gets waveform (in numpy form) and sampling rate of the audio with the path, do not include directory
    '''
    waveform, sampling_rate = torchaudio.load(audio_path)
    # print(waveform,sample_rate)
    waveform = waveform.numpy()[0].astype('float32')
    return waveform, sampling_rate