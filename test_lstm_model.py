from lstm_model import build_graph
import tensorflow as tf
import pickle
import numpy as np
from math import ceil
import time
from argparse import ArgumentParser
import os

arguments = ArgumentParser()
arguments.add_argument("name")
args = arguments.parse_args()

DATA_FOLDER = './data'
CHECKPOINT_FOLDER = './checkpoints/{}'.format(args.name)
USE_CHECKPOINT_FOLDER = './use_checkpoints/{}'.format(args.name)
tensorflow_log = 'tensorflow.log'
validation_restore = 'validation_restore.log'
loss_log = 'loss.log'
accuracy_log = 'accuracy.log'
success_log = 'success.log'
failed_log = 'failed.log'

training_sorted = pickle.load( open( "{}/training_mini.pkl".format(DATA_FOLDER), "rb" ) )
noisy_training_sorted = pickle.load(open("{}/noisy_training_mini.pkl".format(DATA_FOLDER), "rb"))
#training_sorted = pickle.load( open( "./data/training_sorted.pkl", "rb" ) )
testing_sorted = pickle.load( open( "{}/testing_mini.pkl".format(DATA_FOLDER), "rb" ) )
noisy_testing_sorted = pickle.load(open("{}/noisy_testing_mini.pkl".format(DATA_FOLDER), "rb"))
#testing_sorted = pickle.load( open( "./data/testing_sorted.pkl", "rb" ) )
vocab_to_int = pickle.load( open( "{}/vocab_to_int.pkl".format(DATA_FOLDER), "rb" ) )
int_to_vocab = pickle.load( open( "{}/int_to_vocab.pkl".format(DATA_FOLDER), "rb" ) )

# Training parameters
epochs = 10000
batch_size = 32
num_layers = 4
rnn_size = 512
embedding_size = 128
learning_rate = 0.00001
direction = 2
threshold = 0.95
keep_probability = 0.3
display_step = 1 # How often (batch) progress should be printed
stop = 3000 # After how many testing/validation the training should stop, if the batch loss have'nt decreased
per_epoch = 1 # How many times per epoch the training should be tested/validated

# Pad sentences to the same length
def pad_sentence_batch(sentence_batch):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(sentences, noisy_sentences, batch_size):

    # For each batch
    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]

        # Create noisy batch
        sentences_batch_noisy = noisy_sentences[start_i:start_i + batch_size]

        # Add EOS tokens
        sentences_batch_eos = []
        for sentence in sentences_batch:
            sentence.append(vocab_to_int['<EOS>'])
            sentences_batch_eos.append(sentence)

        # Add PAD tokens
        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))

        # Get lengths
        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))
        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))

        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths


def test(model):
    epochs = 1;

    # Start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        # Summary of testing loss
        testing_loss_summary = []

        print()
        print("Testing LSTM Model...")

        # Keep track of which batch iteration is being trained
        iteration = 0
        stop_early = 0
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        saver.restore(sess,"{}/lstm.ckpt".format(USE_CHECKPOINT_FOLDER))
        epoch_loss = 1
        batch_loss = 0
        batch_time = 0
        train_acc = []
        tested = 0
        is_correct = 0

        # Per batch
        for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(get_batches(training_sorted, noisy_training_sorted, batch_size)):

            # Run validation testing
            if batch_i % testing_check == 0 and batch_i > 0:
                val_acc = []
                batch_loss_testing = 0
                batch_time_testing = 0

                n_batches_testing = batch_i + 1

                validloss = batch_loss_testing / n_batches_testing

                print_tested_each = 100

                for i in range(0, len(noisy_testing_sorted)):

                    if (tested > print_tested_each):
                        print_tested_each  += 100
                        print("Tested {}% of test set".format((ceil(i / len(noisy_testing_sorted) * 100) * 100) / 100.0))

                    text = noisy_testing_sorted[i]
                    correct = testing_sorted[i]
                    answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size,
                                                             model.inputs_length: [len(text)]*batch_size,
                                                             model.targets_length: [len(text)+1],
                                                             model.keep_prob: [1.0]})[0]

                    # Remove <PAD> from output
                    pad = vocab_to_int["<PAD>"]
                    eos = vocab_to_int["<EOS>"]

                    answer_logits = "".join([int_to_vocab[i] for i in answer_logits if i != eos])
                    correct = "".join([int_to_vocab[i] for i in correct if i != eos])

                    tested += 1
                    if (answer_logits == correct):
                        is_correct += 1
                        with open("{0}/{1}".format(USE_CHECKPOINT_FOLDER, success_log),'a') as f:
                            f.write('  Validation Input: {}\n'.format("".join([int_to_vocab[i] for i in text])))
                        with open("{0}/{1}".format(USE_CHECKPOINT_FOLDER, success_log),'a') as f:
                            f.write('  Validation Output: {}\n\n'.format(answer_logits))
                    else:
                        with open("{0}/{1}".format(USE_CHECKPOINT_FOLDER, failed_log),'a') as f:
                            f.write('  Validation Input: {}\n'.format("".join([int_to_vocab[i] for i in text])))
                        with open("{0}/{1}".format(USE_CHECKPOINT_FOLDER, failed_log),'a') as f:
                            f.write('  Validation Output: {}\n\n'.format(answer_logits))

                # Reset
                batch_time_testing = 0
                print("Accuracy %: {}%".format((ceil((is_correct / tested) * 100) * 100) / 100.0))
                print("Exact Accuracy: {}".format(is_correct / tested))
                with open("{0}/{1}".format(USE_CHECKPOINT_FOLDER, accuracy_log),'a') as f:
                    f.write("Accuracy %: {}%\n".format((ceil((is_correct / tested) * 100) * 100) / 100.0))
                with open("{0}/{1}".format(USE_CHECKPOINT_FOLDER, accuracy_log),'a') as f:
                    f.write("Exact Accuracy: {}\n\n".format(is_correct / tested))

def test_lstm_model():
    for keep_probability in [0.3]:
        for num_layers in [4]:
            for threshold in [0.95]:
                model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)
                test(model)

if __name__ == "__main__":
    test_lstm_model()
