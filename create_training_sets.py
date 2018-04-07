import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from noise_maker import noise_maker

# Contants
DATA_LOCATION = "./data"

def create_trainingsets(location = DATA_LOCATION):
    int_to_vocab = pickle.load(open( "./data/int_to_vocab.pkl", "rb"))
    max_length = 92
    min_length = 50
    good_sentences = pickle.load(open(location + "/good_sentences.pkl", "rb"))
    training, testing = train_test_split(good_sentences, test_size = 0.15, random_state = 2)

    threshold = 0.9
    for sentence in training[:5]:
        print("Sentence: ")
        print("".join([int_to_vocab[i] for i in sentence]))
        print("With Noise: ")
        print("".join([int_to_vocab[i] for i in noise_maker(sentence, threshold)]))
        print()
    with open(location + '/training.pkl', 'wb') as pkl:
        print('Writing trainingset to pickle...')
        pickle.dump(training, pkl)
    with open(location + '/testing.pkl', 'wb') as pkl:
        print('Writing testingset to pickle...')
        pickle.dump(testing, pkl)

    training_mini = training[:50000]
    testing_mini = testing[:9000]


    with open(location + '/training_mini.pkl', 'wb') as pkl:
        print('Writing training_mini to pickle...')
        pickle.dump(training_mini, pkl)
    with open(location + '/testing_mini.pkl', 'wb') as pkl:
        print('Writing testing_mini to pickle...')
        pickle.dump(testing_mini, pkl)

    print("Trainingset: {0} sentences".format(len(training)))
    print("Testingset: {0} sentences".format(len(testing)))
    print("Trainingset Mini: {0} sentences".format(len(training_mini)))
    print("Testingset Mini: {0} sentences".format(len(testing_mini)))

if __name__ == "__main__":
    create_trainingsets(DATA_LOCATION)
