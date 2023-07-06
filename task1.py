from api import whisper_query
import csv


def get_validated_data() -> dict:
    """
    Creates a dictionary that maps the validated file to the data
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



