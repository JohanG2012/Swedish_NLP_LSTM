import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from noise_maker import noise_maker, write_apart, switch_gender

# Contants
DATA_LOCATION = "./data"

def create_debug_set(location = DATA_LOCATION):
    int_to_vocab = pickle.load(open(location + "/int_to_vocab.pkl", "rb"))
    good_sentences = pickle.load(open(location + "/good_sentences.pkl", "rb"))
    print("📢 Injecting noise into training set...")
    training = good_sentences[:15]
    noisy_training_sentences = []
    threshold = 0.9

    print("📢 Injecting noise into training set...")
    for sentence in training:
        noisy_training_sentences.append(noise_maker(sentence, threshold))

    for sentence in training[:5]:
        print("Sentence: ")
        print("".join([int_to_vocab[i] for i in sentence]))
        print("With Noise: ")
        print("".join([int_to_vocab[i] for i in noise_maker(sentence, threshold)]))
        print()

    with open(location + '/training_debug.pkl', 'wb') as pkl:
        print('Writing trainingset to pickle...')
        pickle.dump(training, pkl)

    with open(location + '/noisy_training_debug.pkl', 'wb') as pkl:
        print('Writing noisy training set to pickle...')
        pickle.dump(noisy_training_sentences, pkl)

    print("Trainingset: {0} sentences".format(len(training)))
    print("Noisy training set: {0} sentences".format(len(noisy_training_sentences)))

def triple_list(l):
    return [x for x in l for y in range(3)]

def create_trainingsets(location = DATA_LOCATION):
    int_to_vocab = pickle.load(open(location + "/int_to_vocab.pkl", "rb"))
    good_sentences = pickle.load(open(location + "/good_sentences.pkl", "rb"))
    noisy_training_sentences = []
    noisy_testing_sentences = []
    noisy_validation_sentences = []
    training, testing = train_test_split(good_sentences, test_size = 0.4, random_state = 2)
    testing, validation = train_test_split(testing, test_size = 0.5, random_state = 2)

    threshold = 0.9

    print("📢 Injecting noise into training set...")
    for sentence in training:
        noisy_training_sentences.append(noise_maker(sentence, threshold))
        noisy_training_sentences.append(write_apart(sentence))
        noisy_training_sentences.append(switch_gender(sentence))

    print("📢 Injecting noise into testing set...")
    for sentence in testing:
        noisy_testing_sentences.append(noise_maker(sentence, threshold))
        noisy_testing_sentences.append(write_apart(sentence))
        noisy_testing_sentences.append(switch_gender(sentence))

    print("📢 Injecting noise into validation set...")
    for sentence in validation:
        noisy_validation_sentences.append(noise_maker(sentence, threshold))
        noisy_validation_sentences.append(write_apart(sentence))
        noisy_validation_sentences.append(switch_gender(sentence))

    print("Expanding sets to match noisy sets...")
    training = triple_list(training)
    testing = triple_list(testing)
    validation = triple_list(validation)

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
    with open(location + '/validation.pkl', 'wb') as pkl:
        print("Writing validation set to pickle...")
        pickle.dump(validation, pkl)
    with open(location + '/noisy_training.pkl', 'wb') as pkl:
        print('Writing noisy training set to pickle...')
        pickle.dump(noisy_training_sentences, pkl)
    with open(location + '/noisy_testing.pkl', 'wb') as pkl:
        print('Writing noisy testingset to pickle...')
        pickle.dump(noisy_testing_sentences, pkl)
    with open(location + '/noisy_validation.pkl', 'wb') as pkl:
        print('Writing noisy validation set to pickle...')
        pickle.dump(noisy_validation_sentences, pkl)

    training_mini = training[:50000]
    noisy_training_mini = noisy_training_sentences[:50000]
    testing_mini = testing[:9000]
    noisy_testing_mini = noisy_testing_sentences[:9000]
    validation_mini = validation[:9000]
    noisy_validation_mini = noisy_validation_sentences[:9000]

    with open(location + '/training_mini.pkl', 'wb') as pkl:
        print('Writing training_mini to pickle...')
        pickle.dump(training_mini, pkl)
    with open(location + '/noisy_training_mini.pkl', 'wb') as pkl:
        print('Writing noisy_training_mini to pickle...')
        pickle.dump(noisy_training_mini, pkl)
    with open(location + '/testing_mini.pkl', 'wb') as pkl:
        print('Writing testing_mini to pickle...')
        pickle.dump(testing_mini, pkl)
    with open(location + '/noisy_testing_mini.pkl', 'wb') as pkl:
        print('Writing noisy testing mini to pickle...')
        pickle.dump(noisy_testing_mini, pkl)
    with open(location + '/validation_mini.pkl', 'wb') as pkl:
        print('Writing validation_mini to pickle...')
        pickle.dump(validation_mini, pkl)
    with open(location + '/noisy_validation_mini.pkl', 'wb') as pkl:
        print('Writing noisy validation mini to pickle...')
        pickle.dump(noisy_validation_mini, pkl)

    print("Trainingset: {0} sentences".format(len(training)))
    print("Noisy training set: {0} sentences".format(len(noisy_training_sentences)))
    print("Testingset: {0} sentences".format(len(testing)))
    print("Noisy testing set: {0} sentences".format(len(noisy_testing_sentences)))
    print("Trainingset Mini: {0} sentences".format(len(training_mini)))
    print("Noisy training set Mini: {0} sentences".format(len(noisy_training_mini)))
    print("Testingset Mini: {0} sentences".format(len(testing_mini)))
    print("Noisy testingset Mini: {0} sentences".format(len(noisy_testing_mini)))
    print("Validation set Mini: {0} sentences".format(len(validation_mini)))
    print("Noisy validation set Mini: {0} sentences".format(len(noisy_validation_mini)))

if __name__ == "__main__":
    create_trainingsets(DATA_LOCATION)
    #create_debug_set()
