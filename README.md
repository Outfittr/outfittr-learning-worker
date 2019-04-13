# Outfittr Learn Project
The machine learning module for the Outfittr application

## How to Use Main.py
usage: main.py [-h] {train,test,extract} ...

Outfittr Learning Module

### positional arguments:
  
  | Command              | Description                           |
  |----------------------|:----------------------------------------:|
  | {train,test,extract} | Different uses of the Learning Module |
  | train                | Train the learning module on a dataset |
  | test                | Test the learning module on data|
  | extract             | Extract features from a clothing image set|

### optional arguments:

  | Command              | Description                           |
  |----------------------|:----------------------------------------:|
  |-h, --help            | show this help message and exit| 

## How to use Train Command
usage: main.py train [-h] -d DATA -o OUTPUT

### optional arguments:
  | Command              | Description                           |
  |----------------------|:----------------------------------------:|
  |-h, --help            |show this help message and exit|
  |-d DATA, --data DATA  |Dataset path|
  |-o OUTPUT, --output OUTPUT| The output file model (.h5) path|

## How to use Test Command
usage: main.py test [-h] -d DATA path

### positional arguments:
| Command              | Description                           |
  |----------------------|:----------------------------------------:|
  |path|                  Path of model file to test from|

### optional arguments:
| Command              | Description                           |
  |----------------------|:----------------------------------------:|
  |-h, --help|           |show this help message and exit|
  |-d DATA, --data DATA  |Path to testing data|
  
## How to use Extract Command
usage: main.py extract [-h] -e {VGG19,ResNet50,VGG16} -o OUTPUT -d DATA

### optional arguments:
| Command              | Description                           |
  |----------------------|:----------------------------------------:|
  |-h, --help|            show this help message and exit|
  |-e {VGG19,ResNet50,VGG16}, --extractor {VGG19,ResNet50,VGG16} | The feature extractor to use|
  |-o OUTPUT, --output OUTPUT|         The output path of the feature vector files|
  |-d DATA, --data DATA  |Path to images that will be feature extracted|


