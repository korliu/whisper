from api import whisper_query, sentence_transformer
import utils
import csv
import os



def init_output_arr(headers: list):

    output_headings = headers

    heading = "\t".join(output_headings)
    heading += "\n"

    return [heading]



## task1
N_CLIPS = 100
wh_model_type = "medium"

output_arr = init_output_arr(["whisper_transcription", "expected_transcription", "word_error_rate", "cosine_similarity"])
whisper = whisper_query.setup_whisper(wh_model_type)
validated_files = utils.get_validated_data()
wer = utils.wer_calculator()
sentence_model = sentence_transformer.get_sentence_model()

data_path = "./dataset/clips/"
with os.scandir(data_path) as audio_files:
    total_clips = 0
    for mp3file in audio_files:

        if total_clips >= N_CLIPS:
            break
        
        audio_file = mp3file.name
        file_name = data_path + audio_file

        # print(audio_file)
        if audio_file in validated_files:
            total_clips += 1

            whisper_transcription = whisper(file_name)

            transcription_text = str(whisper_transcription['text'].strip())
            expected_text = str(validated_files[audio_file]['sentence'].strip())

            word_error_rate = wer.compute(predictions=[transcription_text], references=[expected_text])
            cos_sim = utils.cos_sim_calc(sentence_model, transcription_text, expected_text).item()


            output_data = "\t".join([transcription_text,expected_text, str(word_error_rate), str(cos_sim)]) + "\n"
            output_arr.append(output_data)
            

        else:
            continue

with open("outputs/task1_output.txt", 'w', encoding='utf-8') as output:
    output.writelines(output_arr)


# Find Avg WER and Cosine Similarity
with open("outputs/task1_output.txt", 'r', encoding='utf-8') as input_file:

    data = csv.reader(input_file, delimiter="\t")

    headers = next(data)
    wer = headers.index("word_error_rate")
    cos_sim = headers.index("cosine_similarity")

    total_wer = 0
    total_cos_sim = 0

    count = 0
    for line in data:
        count += 1
        total_wer += float(line[wer])
        total_cos_sim += float(line[cos_sim])
    
    print(f"DATA FROM {N_CLIPS} CLIPS using whisper-{wh_model_type} \n\tAVG WORD ERROR RATE: {total_wer/count} \n\tAVG COSINE SIMILARITY: {total_cos_sim/count}")

