import argparse

parser = argparse.ArgumentParser(description='Outfittr Learning Module')
subparsers = parser.add_subparsers(help='Different uses of the Learning Module')

parser_train = subparsers.add_parser('train', help='Train the learning module on a dataset')
parser_train.add_argument()

parser_test = subparsers.add_parser('test', help='Test the learning module on data')

parser_extract = subparsers.add_parser('extract', help='Extract features from a clothing imag set')

if __name__ == "__main__":
    args = parser.parse_args()

    # Do stuff...

    pass