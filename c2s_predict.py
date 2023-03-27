from common import Common
from extractor import Extractor
from typing import Iterator, Iterable

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
EXTRACTION_API = 'https://po3g2dx2qa.execute-api.us-east-1.amazonaws.com/production/extractmethods'


class C2SPredictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config, EXTRACTION_API, self.config.MAX_PATH_LENGTH, max_path_width=2)

    @staticmethod
    def read_file(input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def predict(self, predict_lines: Iterable[str]):
        # align the number of columns to the model's expected input
        for i, line in enumerate(predict_lines):
            cols = line.split(' ')
            if len(cols) < self.config.DATA_NUM_CONTEXTS + 1:
                cols += [''] * (self.config.DATA_NUM_CONTEXTS + 1 - len(cols))
            else:
                del cols[self.config.DATA_NUM_CONTEXTS + 1:]
            predict_lines[i] = ' '.join(cols)

        model_results = self.model.predict(predict_lines)

        prediction_results = Common.parse_results(model_results, None, topk=SHOW_TOP_CONTEXTS)
        return prediction_results
