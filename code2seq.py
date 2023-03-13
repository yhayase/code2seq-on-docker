from argparse import ArgumentParser
import numpy as np
from c2s_predict import C2SPredictor
import tensorflow as tf

from config import Config
from interactive_predict import InteractivePredictor
from file_predict import FilePredictor
from model import Model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument("--batch_java_src", dest="batch_java_src_path_file",
                        help="path to java source paths",  metavar="FILE", 
                        required=False)
    parser.add_argument("--predict_c2s", dest="predict_c2s_file",
                        help="path to c2s file for prediction",  metavar="FILE", 
                        required=False)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    model = Model(config)
    print('Created model')
    if config.TRAIN_PATH:
        model.train()
    elif config.TEST_PATH and not args.data_path:
        results, precision, recall, f1, rouge = model.evaluate()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Rouge: ', rouge)
    elif args.batch_java_src_path_file:
        predictor = FilePredictor(config, model)
        for source_path in open(args.batch_java_src_path_file):
            source_path = source_path.rstrip('\n')
            prediction_results = predictor.predict(source_path)
            for index, method_prediction in prediction_results.items():
                print('  %s:' % method_prediction.original_name)
                for predicted_seq in method_prediction.predictions:
                    print('    %s with probability %f' % (predicted_seq.prediction, predicted_seq.score))
    elif args.predict_c2s_file:
        c2s_predictor = C2SPredictor(config, model)
        with open(args.predict_c2s_file) as f:
            c2s_lines = f.read().splitlines()
            for line in c2s_lines:
                prediction_results = c2s_predictor.predict([line])
                for index, method_prediction in prediction_results.items():
                    print('  %s:' % method_prediction.original_name)
                    for predicted_seq in method_prediction.predictions:
                        print('    %s with probability %f' % (predicted_seq.prediction, predicted_seq.score))
    elif args.predict:
        predictor = InteractivePredictor(config, model)
        predictor.predict()
    elif args.release and args.load_path:
        model.evaluate(release=True)
    model.close_session()
