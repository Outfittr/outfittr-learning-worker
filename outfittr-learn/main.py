import argparse
import json
import os
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

from learn import train, test, FeatureExtractor, create_multilayer_perceptron
from data_operations import construct_dataset, train_test_split, get_images, extract_all_features

feature_extractors = {
                      "ResNet50": lambda: ResNet50(weights='imagenet', include_top=False),
                      "VGG19": lambda: VGG19(weights='imagenet', include_top=False),
                      "VGG16": lambda: VGG16(weights='imagenet', include_top=False)
                     }


parser = argparse.ArgumentParser(description='Outfittr Learning Module')
subparsers = parser.add_subparsers(help='Different uses of the Learning Module')


""" Train Parser """
parser_train = subparsers.add_parser('train', help='Train the learning module on a dataset')
parser_train.add_argument('-d',
                          '--data',
                          help='Dataset path',
                          type=str,
                          required=True)
# parser_train.add_argument('-a',
#                           '--arch',
#                           help='The path to the JSON architecture of the model')
parser_train.add_argument('-o',
                          '--output',
                          help='The output file model (.h5) path',
                          type=str,
                          required=True)
# parser_train.add_argument('-m',
#                           '--model',
#                           help='The model file (.h5) if training is continuing',
#                           type=str)
parser_train.set_defaults(which='train')

""" Test Parser """
parser_test = subparsers.add_parser('test', help='Test the learning module on data')
parser_test.add_argument('path',
                         help='Path of model file to test from',
                         type=str)
parser_test.add_argument('-d',
                         '--data',
                         help='Path to testing data',
                         type=str,
                         required=True)
parser_test.set_defaults(which='test')

""" Extract Parser """
parser_extract = subparsers.add_parser('extract', help='Extract features from a clothing image set')
parser_extract.add_argument('-e',
                            '--extractor',
                            help='The feature extractor to use',
                            choices=list(feature_extractors.keys()),
                            required=True)
parser_extract.add_argument('-o',
                            '--output',
                            help='The output path of the feature vector files',
                            type=str,
                            required=True)
parser_extract.add_argument('-d',
                            '--data',
                            help='Path to images that will be feature extracted',
                            type=str,
                            required=True)
parser_extract.set_defaults(which='extract')


if __name__ == "__main__":
    args = parser.parse_args()

    if args.which is 'train':
        try:
            # Check if data files exist
            if not os.path.exists(args.data):
                raise FileNotFoundError("Data path does not exist")
            if not os.path.exists(os.path.normpath(os.path.join(args.data, './survey.json'))):
                raise FileNotFoundError("Data-point file does not exist")

            _surveys = json.load(open(os.path.join(args.data, './survey.json')))
            dataset = construct_dataset(_surveys, args.data)
            train_in, test_in, train_out, test_out = train_test_split(dataset[0],
                                                                      dataset[1],
                                                                      test_size=0.2,
                                                                      shuffle=True)

            history = train((train_in, train_out),
                            load_path=args.output,
                            architecture=create_multilayer_perceptron,
                            device='/device:CPU:0')

        except FileNotFoundError as e:
            print(e)
            raise

    elif args.which is 'test':
        _surveys = json.load(open(args.data+'/survey.json'))
        dataset = construct_dataset(_surveys, args.data)
        train_in, test_in, train_out, test_out = train_test_split(dataset[0],
                                                                  dataset[1],
                                                                  test_size=0.2,
                                                                  shuffle=True)

        test((test_in, test_out), args.path)
    elif args.which is 'extract':
        try:
            if args.data is None or args.extractor is None:
                raise Exception("Data or Extractor is invalid")

            items = get_images(args.data)
            extract_all_features(clothing_items=items, feature_extractor=feature_extractors[args.extractor])
        except Exception as e:
            print(e)
            raise
