from lstm_model import build_graph
import tensorflow as tf
import pickle
import numpy as np
import time
from argparse import ArgumentParser
import os
import yaml
config = yaml.safe_load(open("./hyperparameters.yaml"))

arguments = ArgumentParser()
arguments.add_argument("name")
args = arguments.parse_args()

DATA_FOLDER = './data'
CHECKPOINT_FOLDER = './checkpoints/{}'.format(args.name)
USE_CHECKPOINT_FOLDER = './use_checkpoints/{}'.format(args.name)
tensorflow_log = 'tensorflow.log'
validation_restore = 'validation_restore.log'
loss_log = 'loss.log'

training_sorted = pickle.load( open( "{}/training_mini.pkl".format(DATA_FOLDER), "rb" ) )
noisy_training_sorted = pickle.load(open("{}/noisy_training_mini.pkl".format(DATA_FOLDER), "rb"))
testing_sorted = pickle.load( open( "{}/validation_mini.pkl".format(DATA_FOLDER), "rb" ) )
noisy_testing_sorted = pickle.load(open("{}/noisy_validation_mini.pkl".format(DATA_FOLDER), "rb"))
vocab_to_int = pickle.load( open( "{}/vocab_to_int.pkl".format(DATA_FOLDER), "rb" ) )
int_to_vocab = pickle.load( open( "{}/int_to_vocab.pkl".format(DATA_FOLDER), "rb" ) )

# Training parameters
display_step = 1 # How often (batch) progress should be printed
stop = 10 # After how many testing/validation the training should stop, if the batch loss have'nt decreased
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


def train(model, epochs):

    # Start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()

        # Summary of testing loss
        testing_loss_summary = []

        print()
        print("Training LSTM Model...")

        # Keep track of which batch iteration is being trained
        iteration = 0
        stop_early = 0
        testing_check = (len(training_sorted)//config['batch_size']//per_epoch)-1
        try:
            epoch_i = int(open(tensorflow_log,'r').read().split('\n')[-2])+1
            validrestore = float(open("{0}/{1}".format(CHECKPOINT_FOLDER, validation_restore),'r').read().split('\n')[-2])
            print("Loading validation loss from log checkpoint. Current min. validation loss: {}".format(validrestore))
            print('Loading Checkpoint. Starting at epoch nr: {}'.format(epoch_i))
        except:
            epoch_i = 1
            validrestore = 10000

        # Per epoch
        while epoch_i <= epochs:
            if epoch_i != 1:
                saver.restore(sess,"{}/lstm.ckpt".format(CHECKPOINT_FOLDER))
            epoch_loss = 1
            batch_loss = 0
            batch_time = 0
            train_acc = []

            # Per batch
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(get_batches(training_sorted, noisy_training_sorted, config['batch_size'])):
                start_time = time.time()
                summary, loss, _ = sess.run([model.merged, model.cost, model.train_op],
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: config['keep_probability']})

                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time
                iteration += 1

                # Print info
                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(training_sorted) // config['batch_size'],
                                  batch_loss / display_step,
                                  batch_time))
                    # Reset
                    batch_loss = 0
                    batch_time = 0

                # Run validation testing
                if batch_i % testing_check == 0 and batch_i > 0:
                    val_acc = []
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(get_batches(testing_sorted, noisy_testing_sorted, config['batch_size'])):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged, model.cost],
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})
                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                    n_batches_testing = batch_i + 1

                    # Print result
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing, batch_time_testing))

                    validloss = batch_loss_testing / n_batches_testing

                    if not os.path.exists(CHECKPOINT_FOLDER):
                        os.makedirs(CHECKPOINT_FOLDER)

                    with open("{0}/{1}".format(CHECKPOINT_FOLDER, loss_log),'a') as f:
                        f.write("Epoch: {0} Validation loss: {1}\n".format(epoch_i, validloss))

                    for i in range(10, 20):
                        text = noisy_testing_sorted[i]
                        correct = testing_sorted[i]
                        answer_logits = sess.run(model.predictions, {model.inputs: [text]*config['batch_size'],
                                                                 model.inputs_length: [len(text)]*config['batch_size'],
                                                                 model.targets_length: [len(text)+1],
                                                                 model.keep_prob: [1.0]})[0]

                        # Remove <PAD> from output
                        pad = vocab_to_int["<PAD>"]

                        print('  Validation Input: {}'.format("".join([int_to_vocab[i] for i in text])))
                        print('  Validation Output: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))
                        print('  Correct: {}'.format("".join([int_to_vocab[i] for i in correct if i != pad])))
                        print('  Is Correct: {}'.format(answer_logits == correct))
                        print()

                    # Reset
                    batch_time_testing = 0

                    # Save new model if new minimum
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary) and validloss <= validrestore:
                        print('Model is improved, Saving!')
                        stop_early = 0
                        checkpoint = "{}/lstm.ckpt".format(USE_CHECKPOINT_FOLDER)
                        saver.save(sess, checkpoint)
                        with open("{0}/{1}".format(CHECKPOINT_FOLDER, validation_restore),'a') as f:
                            f.write(str(validloss)+'\n')

                    else:
                        print("Model has not improved, will not save.")
                        stop_early += 1
                        if stop_early == stop:
                            break

                    checkpoint = "{}/lstm.ckpt".format(CHECKPOINT_FOLDER)
                    saver.save(sess, checkpoint)
                    with open("{0}/{1}".format(CHECKPOINT_FOLDER, tensorflow_log),'a') as f:
                        f.write(str(epoch_i)+'\n')
                    epoch_i += 1

            if stop_early == stop:
                print("Model has not improved for a while, stopping to avoid overfitting")
                break

def train_lstm_model():
    for config['keep_probability'] in [0.3]:
        for config['num_layers'] in [4]:
            for config['threshold'] in [0.95]:
                model = build_graph(config['keep_probability'], config['rnn_size'], config['num_layers'], config['batch_size'], config['learning_rate'], config['embedding_size'], config['direction'])
                train(model, config['epochs'])

if __name__ == "__main__":
    train_lstm_model()
