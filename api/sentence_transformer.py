## taken from https://huggingface.co/tasks/sentence-similarity

from sentence_transformers import SentenceTransformer


def get_sentence_model():
    '''
    Creates a SentenceTransformerModel to use
    '''
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return sentence_model