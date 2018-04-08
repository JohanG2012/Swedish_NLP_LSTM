import pickle
import numpy as np

DATA_FOLDER = './data'

def noise_maker(sentence, threshold):
    vocab_to_int = pickle.load(open("{}/vocab_to_int.pkl".format(DATA_FOLDER), "rb"))
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z','å', 'ä', 'ö']
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)

        # threshold high = Most characters correct
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance - swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance - Add extra letter from letters[]
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% chance - Do not type a character
            else:
                pass
        i += 1
    return noisy_sentence
