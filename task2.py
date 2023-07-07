from transformers import AutoFeatureExtractor
import utils
from datasets import Dataset, Audio
# import datasets
import pandas as pd
import csv
import evaluate
import os


# Preprocessing/ filtering data
audio_dir = "./dataset/clips/"
available_clips = set()
with os.scandir(audio_dir) as audio_files:
    for mp3file in audio_files:
        available_clips.add(mp3file.name)


binary_genders = ['male','female']

# 'other' gender is excluded

validated_data = utils.get_validated_data()
validated_with_gender = {data for data in validated_data if validated_data[data]['gender'] in binary_genders}

dataset_path = "./dataset/validated.tsv"
gender_validated_path = "./dataset/gender_validated.tsv"


def create_gender_validated_data(input_file: str, output_file: str):
    '''
    creates data in output_file with only gender_validated files from input_file
    '''
    with open(input_file,'r',encoding='utf-8') as file:

        output_arr = []
        headers = next(file)
        header_info = headers.split('\t')
        path = header_info.index('path')

        output_arr.append(headers)

        for line in file:

            split_data = line.split('\t')
            path_name = split_data[path]

            if path_name in validated_with_gender and path_name in available_clips:
                output_arr.append(line)

        # print(output_arr[:10])
        with open(output_file, 'w', encoding='utf-8') as out:
            out.writelines(output_arr)

create_gender_validated_data(dataset_path,gender_validated_path)


# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# includes 'other' gender, excluded for training
testing_data_path = "./dataset/testing_data.tsv"
data_files = {'train': gender_validated_path, 'test': testing_data_path}


# test_dataframe = pd.read_csv(testing_data_path,delimiter="\t")
# test_dataframe = pd.DataFrame(test_dataframe)
# test_ds = Dataset.from_pandas(test_dataframe, split="test")

train_dataframe = pd.read_csv(gender_validated_path, delimiter='\t')
train_dataframe = pd.DataFrame(train_dataframe)

def audio_data(audio_path) -> dict:
    data = {}
    audio, audio_sample_rate = utils.get_audio(audio_path) 

    data['audio'] = audio
    data['sample_rate'] = audio_sample_rate

    return data

train_dataframe['audio'] = train_dataframe.apply(lambda df: audio_data(audio_dir+df['path']
                                                                       ),axis=1)
# print(train_dataframe)


train_ds = Dataset.from_pandas(train_dataframe, split="train")

genders_data = train_ds
genders_data = genders_data.train_test_split(test_size = 0.2)




print(genders_data)