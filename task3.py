import csv
from collections import defaultdict
from transformers import pipeline
import re

# TASK 3

output_dir = "./outputs/"
word_freq_file = output_dir + "task3-word-freq.csv"
sentiment_score_file = output_dir + "task3-sentiment.csv"
type_token_ratio_file = output_dir + "task3-type-token-ratio.csv"

file_to_analyze = "./outputs/task1-medium_output.txt"

analyze_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

word_freq_histogram = defaultdict(int)
sentiment_scores = defaultdict(int)

#calculate word_frequency and sentiment_scores
with open(file_to_analyze,'r',encoding='utf-8') as file:

    data = csv.reader(file,delimiter='\t')
    headers = next(data)

    info = {header: i for i, header in enumerate(headers)}

    for line in data:
        if not line:
            continue

        clip = line[info["file_name"]]
        text = line[info["whisper_transcription"]]

        #sentiment
        get_sentiment_score = analyze_sentiment(text)
        sentiment_scores[clip] = (get_sentiment_score)

        text = re.sub('[,.]','',text).split()
        for word in text:
            word = word.lower()
            word_freq_histogram[word] = word_freq_histogram[word] + 1

# print(sentiment_scores, word_freq_histogram)
# output word frequency
with open(word_freq_file,'w',encoding='utf-8') as out:
    headers = ["WORD","FREQUENCY"]

    output_arr = []
    output_arr.append(",".join(headers)+"\n")

    sorted_by_freq = sorted(list(word_freq_histogram.items()),key=lambda x: x[1], reverse=True)[:100]
    
    for word_info in sorted_by_freq:
        word, freq = word_info
        output_arr.append(",".join([word,str(freq)])+"\n")

    out.writelines(output_arr)


type_token = {}
# calculate type-token-ratio
with open(file_to_analyze,'r',encoding='utf-8') as file:

    data = csv.reader(file,delimiter='\t')
    headers = next(data)

    transcription = headers.index("whisper_transcription")

    curr_group = 1
    words_encountered = set()
    words_in_group = 0
    clip_num = 0
    for line in data:
        if not line:
            continue
        
        text = re.sub('[,.]','',line[transcription]).split()
        if clip_num % 10 == 0 and clip_num > 0:
            type_token_val = len(words_encountered)/words_in_group

            type_token[curr_group] = type_token_val

            words_encountered = set()
            words_in_group = 0
            curr_group += 1

        for word in text:

            words_in_group += 1
            words_encountered.add(word)

        clip_num += 1

        if words_in_group != 0:
            type_token_val = len(words_encountered)/words_in_group

            type_token[curr_group] = type_token_val


# print(type_token)