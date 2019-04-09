'''
    main.py

    @author Daquaris Chadwick
    
    The main entry point
'''
from .outfitter_model import OutfitterModel, FeatureExtractor, create_multilayer_perceptron, process_outfit_features
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
import numpy as np
import os
import re
import json


def get_images(root_dir):
    result = {'tops': [], 'bottoms': []}
    for parent, _, files in os.walk(root_dir):
        for filename in files:
            if not filename.endswith(('.jpg', '.png')):
                continue
            path = os.path.join(parent, filename)
            match = re.match(r'.*/(tops|bottoms).*', path)
            if not match:
                continue
            clothing = match.groups()[0]
            result[clothing].append((path, filename))
    return result


def extract_all_features(clothing_items):
    model = FeatureExtractor(ResNet50(weight='imagenet', include_top=False))
    output = {'data': {}}

    for i in clothing_items:
        output['data'].update({i: []})
        length = len(clothing_items[i])
        curr = 0

        if not os.path.exists('output/' + i):
            os.makedirs('output/' + i)

        print("\n" + i)
        print("Progress: 0/" + str(length))
        for j in clothing_items[i]:
            url_item = j[0]
            item = image.load_img(url_item, target_size=(224, 224))
            item = image.img_to_array(item)
            item = model.get_features(item).flatten().tolist()

            out = open("output/" + i + "/" + str(j[1]) + ".json", "w")
            out.write(json.dumps(item))
            out.close()
            
            curr += 1
            print("Progress: " + str(curr) + "/" + str(length))


def get_features(survey_data_item):
    ret_val = []

    for i in survey_data_item["createdOutfit"]:
        mapping = json.load(open("output/" + i.lower() + "/mappings.json"))

        ret_val.append(json.load(open("output/" + i.lower() + "/" + mapping[survey_data_item["createdOutfit"][i]])))

    return ret_val


def construct_dataset(surveys, feature_path):
    dataset_input = []
    dataset_output = []

    for survey in surveys:
        rating = np.zeros(5)
        rating[int(survey['createRating']) - 1] = 1
        context_vector = [ 
            survey['state'],
            survey['sex'],
            survey['factors']['temperature'],
            survey['factors']['weather'],
            survey['factors']['temperature'],
            survey['factors']['formality'],
            survey['factors']['season']
        ]

        for outfit_type in ['createdOutfit', 'randOutfit']:
            # Ensure the outfit type key exists; randOutfit may not
            if outfit_type not in survey:
                continue
            
            outfit_json = survey[outfit_type]

            if not outfit_json:
                continue

            try:
                clothing_vectors = []
                clothing_vectors.append(json.load(open('output/tops/' + outfit_json['Tops'] + '.json')))
                clothing_vectors.append(json.load(open('output/bottoms/' + outfit_json['Bottoms'] + '.json')))

                feature_vector = process_outfit_features(clothing_vectors)
                input_vector = np.concatenate((feature_vector, [np.asarray(context_vector)]), axis=None)

                dataset_input.append(input_vector)
                dataset_output.append(rating)
            except IOError as e:
                print('Failure. Skipping survey...', outfit_json)

    return dataset_input, dataset_output


items = get_images('data/')

# extract_all_features(items)
_surveys = json.load(open('data/surveys.json'))
dataset = construct_dataset(_surveys, 'data/')

# Construct dataset using OutfitterModel.create_model_input_vector()
train_in, test_in, train_out, test_out = train_test_split(dataset[0], dataset[1], test_size=0.2, shuffle=True)

learn_model = OutfitterModel('my_model.h5', [1, 2, 3, 4, 5], create_multilayer_perceptron)
history = learn_model.train((train_in, train_out))
print(history)
test_info = learn_model.test((test_in, test_out))
print(test_info)
