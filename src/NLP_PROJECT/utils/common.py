from NLP_PROJECT import logger
import pickle


def load_object1(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise e