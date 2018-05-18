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
    typos_testing_sentences = []
    gender_testing_sentences = []
    two_pass_testing_sentences = []
    noisy_validation_sentences = []
    training, testing = train_test_split(good_sentences, test_size = 0.4, random_state = 2)
    testing, validation = train_test_split(testing, test_size = 0.5, random_state = 2)

    threshold = 0.9
    training_loop = training[:]
    testing_loop = testing[:]
    validation_loop = validation[:]

    training.clear()
    testing.clear()
    validation.clear()
    gender_testing = []
    two_pass_testing = []
    typos_testing = []

    print("📢 Injecting noise into training set...")
    for sentence in training_loop:
        new_sentence = noise_maker(sentence, threshold)
        noisy_training_sentences.append(new_sentence)
        training.append(sentence)
        new_sentence = write_apart(sentence)
        if (new_sentence != sentence):
            noisy_training_sentences.append(new_sentence)
            training.append(sentence)
        new_sentence = switch_gender(sentence)
        if (new_sentence != sentence):
            noisy_training_sentences.append(new_sentence)
            training.append(sentence)

    print("📢 Injecting noise into testing set...")
    for sentence in testing_loop:
        new_sentence = noise_maker(sentence, threshold)
        if (new_sentence != sentence):
            typos_testing_sentences.append(new_sentence)
            typos_testing.append(sentence)
        new_sentence = write_apart(sentence)
        if (new_sentence != sentence):
            two_pass_testing_sentences.append(new_sentence)
            two_pass_testing.append(sentence)
        new_sentence = switch_gender(sentence)
        if (new_sentence != sentence):
            gender_testing_sentences.append(new_sentence)
            gender_testing.append(sentence)

    print("📢 Injecting noise into validation set...")
    for sentence in validation_loop:
        new_sentence = noise_maker(sentence, threshold)
        noisy_validation_sentences.append(new_sentence)
        validation.append(sentence)
        new_sentence = write_apart(sentence)
        if (new_sentence != sentence):
            noisy_validation_sentences.append(new_sentence)
            validation.append(sentence)
        new_sentence = switch_gender(sentence)
        if (new_sentence != sentence):
            noisy_validation_sentences.append(new_sentence)
            validation.append(sentence)

    for sentence in training[:5]:
        print("Sentence: ")
        print("".join([int_to_vocab[i] for i in sentence]))
        print("With Noise: ")
        print("".join([int_to_vocab[i] for i in noise_maker(sentence, threshold)]))
        print()
    with open(location + '/training.pkl', 'wb') as pkl:
        print('Writing trainingset to pickle...')
        pickle.dump(training, pkl)
    with open(location + '/gender_testing.pkl', 'wb') as pkl:
        print('Writing gender testingset to pickle...')
        pickle.dump(gender_testing, pkl)
    with open(location + '/two_pass_testing.pkl', 'wb') as pkl:
        print('Writing two-pass testingset to pickle...')
        pickle.dump(two_pass_testing, pkl)
    with open(location + '/typos_testing.pkl', 'wb') as pkl:
        print('Writing typos testingset to pickle...')
        pickle.dump(typos_testing, pkl)
    with open(location + '/validation.pkl', 'wb') as pkl:
        print("Writing validation set to pickle...")
        pickle.dump(validation, pkl)
    with open(location + '/noisy_training.pkl', 'wb') as pkl:
        print('Writing noisy training set to pickle...')
        pickle.dump(noisy_training_sentences, pkl)
    with open(location + '/noisy_typos_testing.pkl', 'wb') as pkl:
        print('Writing typos testingset to pickle...')
        pickle.dump(typos_testing_sentences, pkl)
    with open(location + '/noisy_gender_testing.pkl', 'wb') as pkl:
        print('Writing gender testingset to pickle...')
        pickle.dump(gender_testing_sentences, pkl)
    with open(location + '/noisy_two_pass_testing.pkl', 'wb') as pkl:
        print('Writing two-pass testingset to pickle...')
        pickle.dump(two_pass_testing_sentences, pkl)
    with open(location + '/noisy_validation.pkl', 'wb') as pkl:
        print('Writing noisy validation set to pickle...')
        pickle.dump(noisy_validation_sentences, pkl)

    training_mini = training[:50000]
    noisy_training_mini = noisy_training_sentences[:50000]
    gender_testing_mini = gender_testing[:9000]
    two_pass_testing_mini = two_pass_testing[:9000]
    typos_testing_mini = typos_testing[:9000]
    noisy_typos_testing_mini = typos_testing_sentences[:9000]
    noisy_gender_testing_mini = gender_testing_sentences[:9000]
    noisy_two_pass_testing_mini = two_pass_testing_sentences[:9000]
    validation_mini = validation[:9000]
    noisy_validation_mini = noisy_validation_sentences[:9000]

    with open(location + '/training_mini.pkl', 'wb') as pkl:
        print('Writing training_mini to pickle...')
        pickle.dump(training_mini, pkl)
    with open(location + '/noisy_training_mini.pkl', 'wb') as pkl:
        print('Writing noisy_training_mini to pickle...')
        pickle.dump(noisy_training_mini, pkl)
    with open(location + '/typos_testing_mini.pkl', 'wb') as pkl:
        print('Writing typos testing mini to pickle...')
        pickle.dump(typos_testing_mini, pkl)
    with open(location + '/gender_testing_mini.pkl', 'wb') as pkl:
        print('Writing gender testing mini to pickle...')
        pickle.dump(gender_testing_mini, pkl)
    with open(location + '/two_pass_testing_mini.pkl', 'wb') as pkl:
        print('Writing two-pass testing mini to pickle...')
        pickle.dump(two_pass_testing_mini, pkl)

    with open(location + '/noisy_typos_testing_mini.pkl', 'wb') as pkl:
        print('Writing typos testing mini to pickle...')
        pickle.dump(noisy_typos_testing_mini, pkl)
    with open(location + '/noisy_gender_testing_mini.pkl', 'wb') as pkl:
        print('Writing gender testing mini to pickle...')
        pickle.dump(noisy_gender_testing_mini, pkl)
    with open(location + '/noisy_two_pass_testing_mini.pkl', 'wb') as pkl:
        print('Writing two-pass testing mini to pickle...')
        pickle.dump(noisy_two_pass_testing_mini, pkl)

    with open(location + '/validation_mini.pkl', 'wb') as pkl:
        print('Writing validation_mini to pickle...')
        pickle.dump(validation_mini, pkl)
    with open(location + '/noisy_validation_mini.pkl', 'wb') as pkl:
        print('Writing noisy validation mini to pickle...')
        pickle.dump(noisy_validation_mini, pkl)

    print("Trainingset: {0} sentences".format(len(training)))
    print("Noisy training set: {0} sentences".format(len(noisy_training_sentences)))
    print("Testingset: {0} sentences".format(len(testing)))
    print("Typos testing set: {0} sentences".format(len(typos_testing_sentences)))
    print("Gender testing set: {0} sentences".format(len(gender_testing_sentences)))
    print("Two-pass testing set: {0} sentences".format(len(two_pass_testing_sentences)))
    print("Trainingset Mini: {0} sentences".format(len(training_mini)))
    print("Noisy training set Mini: {0} sentences".format(len(noisy_training_mini)))
    print("Typos testingset Mini: {0} sentences".format(len(typos_testing_mini)))
    print("Gender testingset Mini: {0} sentences".format(len(gender_testing_mini)))
    print("Two-pass testingset Mini: {0} sentences".format(len(two_pass_testing_mini)))
    print("Validation set Mini: {0} sentences".format(len(validation_mini)))
    print("Noisy validation set Mini: {0} sentences".format(len(noisy_validation_mini)))

if __name__ == "__main__":
    create_trainingsets(DATA_LOCATION)
    #create_debug_set()
