from api import whisper_query
import csv
import os


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


def init_output_arr():

    output_headings = ["Whisper Transcription", "Expected Transcription", "Word Error Rate"]

    heading = "\t".join(output_headings)
    heading += "\n"

    return [heading]


## task1
N_CLIPS = 10

output_arr = init_output_arr()

whisper = whisper_query.setup_whisper("tiny")

validated_files = get_validated_data()

data_path = "./dataset/clips/"
with os.scandir(data_path) as audio_files:
    total_clips = 0
    for mp3file in audio_files:

        if total_clips > N_CLIPS:
            break
        
        audio_file = mp3file.name
        file_name = data_path + audio_file

        # print(audio_file)
        if audio_file in validated_files:
            total_clips += 1

            whisper_transcription = whisper(file_name)

            transcription_text = whisper_transcription['text'].strip()
            expected_text = validated_files[audio_file]['sentence'].strip()


            output_data = "\t".join([transcription_text,expected_text]) + "\n"
            output_arr.append(output_data)
            

        else:
            continue

with open("outputs/task1_output.txt", 'w', encoding='utf-8') as output:
    output.writelines(output_arr)