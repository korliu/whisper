from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer
from transformers import Wav2Vec2Processor
import utils
from datasets import Dataset, Audio, ClassLabel
# import datasets
import pandas as pd
import numpy as np
import csv
import evaluate
import os
import torch

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

    data['audio_wav'] = np.asarray(audio)
    data['path'] = audio_path
    data['sampling_rate'] = audio_sample_rate

    return data


train_dataframe['audio'] = train_dataframe.apply(lambda df: audio_data(audio_dir+df['path']),axis=1)
train_dataframe['gender'] = train_dataframe['gender'].replace('male',0).replace('female',1)

# print(train_dataframe)


train_ds = Dataset.from_pandas(train_dataframe, split="train")

genders_ds = train_ds
genders_ds = genders_ds.train_test_split(test_size = 0.2)
cols_to_remove = ['client_id','sentence','up_votes','down_votes','age','accents','variant','locale','segment','path']
genders_ds = genders_ds.remove_columns(cols_to_remove)


label2id, id2label = dict(), dict()
for i, label in enumerate(binary_genders):
    label2id[label] = str(i)
    id2label[str(i)] = label

# print(id2label[str(1)])
# print(genders_ds, genders_ds['train'][0]['gender'])


# preprocess to use for Wav2Vec2
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def convert_data(dataset):
    audio_waves = [np.asarray(data["audio_wav"]) for data in dataset["audio"]]
    
    converted = feature_extractor(audio_waves, sampling_rate=feature_extractor.sampling_rate,max_length=16000, truncation=True)
    converted

    return converted
# print(genders_ds['train'])

# print(convert_data(genders_ds['train']))

encoded_gender_ds = genders_ds.map(convert_data, remove_columns="audio", batched=True)
encoded_gender_ds = encoded_gender_ds.rename_column("gender", "labels")


# print(encoded_gender_ds['train']['label'])


accuracy = evaluate.load("accuracy")
def compute_acc(evaluations):
    predictions = np.argmax(evaluations.predictions, axis=1)
    return accuracy.compute(predictions=predictions,references=evaluations.eval_pred.label_ids)



num_labels = len(id2label)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label,
)


training_args = TrainingArguments(
    output_dir="gender_classification_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_gender_ds["train"],
    eval_dataset=encoded_gender_ds["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_acc,
)

trainer.train()

trainer.push_to_hub()