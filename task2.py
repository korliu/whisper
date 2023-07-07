from transformers import AutoFeatureExtractor
import utils
import datasets
import csv
import evaluate

# Preprocessing/ filtering data

binary_genders = {'male','female'}
# 'other' gender is excluded

validated_data = utils.get_validated_data()
validated_with_gender = {data for data in validated_data if validated_data[data]['gender'] in binary_genders}

dataset_path = "./dataset/validated.tsv" ## train
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

            if split_data[path] in validated_with_gender:
                output_arr.append(line)

        # print(output_arr[:10])
        with open(output_file, 'w', encoding='utf-8') as out:
            out.writelines(output_arr)

create_gender_validated_data(dataset_path,gender_validated_path)


# feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# includes 'other' gender, excluded for training
testing_data_path = "./dataset/"
data_files = {'train': gender_validated_path, 'test': testing_data_path}
dataset = datasets.load_dataset(dataset_path, data_files=data_files)
