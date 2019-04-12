import argparse
import data_operations
import learn
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16


feature_extractors = {
                      "ResNet50": lambda: ResNet50(weights='imagenet', include_top=False),
                      "VGG19": lambda: VGG19(weights='imagenet', include_top=False),
                      "VGG16": lambda: VGG16(weights='imagenet', include_top=False)
                     }


if __name__ == "__main__":
    args = parser.parse_args()

    if args.which is 'train':
        if args.model is not None:
            _surveys = json.load(open(args.data+'/survey.json'))
            dataset = data_operations.construct_dataset(_surveys, args.data)
            train_in, test_in, train_out, test_out = train_test_split(dataset[0], dataset[1], test_size=0.2, shuffle=True)

            learn_model = learn.OutfitterModel(args.model, [1, 2, 3, 4, 5])
            # Build model architecture
            history = learn_model.train((train_in, train_out))
        elif args.arch is not None and args.output is not None:
            pass
    elif args.which is 'test':
         _surveys = json.load(open(args.data+'/survey.json'))
        dataset = data_operations.construct_dataset(_surveys, args.data)
        train_in, test_in, train_out, test_out = train_test_split(dataset[0], dataset[1], test_size=0.2, shuffle=True)

        learn.test((test_in, test_out), args.path)
        pass
    elif args.which is 'extract':
        pass

parser = argparse.ArgumentParser(description='Outfittr Learning Module')
subparsers = parser.add_subparsers(help='Different uses of the Learning Module')

parser_train = subparsers.add_parser('train', help='Train the learning module on a dataset')
parser_train.add_argument('-d','--data', help='Dataset path', type=str, required=True)
parser_train.add_argument('-a','--arch', help='The path to the JSON architecture of the model')
parser_train.add_argument('-o', '--output', help='The output file model (.h5) path', type=str)
parser_train.add_argument('-m', '--model', help='The model file (.h5) if training is continuing', type=argparse.FileType('rw'))
parser_train.set_defaults(which='train')

parser_test = subparsers.add_parser('test', help='Test the learning module on data')
parser_test.add_argument('path', help='Path of model file to test from', type=str, required=True)argparse.FileType('rw')
parser_test.add_argument('-d', '--data', help='Path to testing data', type=str, required=True)
parser_test.set_defaults(which='test')

parser_extract = subparsers.add_parser('extract', help='Extract features from a clothing imag set')
parser_extract.add_argument('-e', '--extractor', help='The feature extractor to use', choices=list(feature_extractors.keys()),required=True)
parser_extract.add_argument('-o', '--output', help='The output path of the feature vector files', type=str, required=True)
parser_extract.add_argument('-d', '--data', help='Path to images that will be feature extracted', type=str, required=True)
parser_extract.set_defaults(which='extract')