from lstm_model import build_graph
import tensorflow as tf
import pickle
from noise_maker import noise_maker
import numpy as np
import time

training_sorted = pickle.load( open( "./data/training_mini.pkl", "rb" ) )
#training_sorted = pickle.load( open( "./data/training_sorted.pkl", "rb" ) )
testing_sorted = pickle.load( open( "./data/testing_mini.pkl", "rb" ) )
#testing_sorted = pickle.load( open( "./data/testing_sorted.pkl", "rb" ) )
vocab_to_int = pickle.load( open( "./data/vocab_to_int.pkl", "rb" ) )

# Training parameters
epochs = 1
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75
display_step = 1 # How often (batch) progress should be printed
stop = 3 # After how many testing/validation the training should stop, if the batch loss have'nt decreased
per_epoch = 5 # How many times per epoch the training should be tested/validated


# Pad sentences to the same length
def pad_sentence_batch(sentence_batch):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(sentences, batch_size, threshold):

    # For each batch
    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]

        # Create noisy batch
        sentences_batch_noisy = []
        for sentence in sentences_batch:
            sentences_batch_noisy.append(noise_maker(sentence, threshold))

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

        # Summary of testing loss
        testing_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0
        stop_early = 0
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training LSTM Model...")

        # Per epoch
        for epoch_i in range(1, epochs+1):
            batch_loss = 0
            batch_time = 0

            # Per batch
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()
                summary, loss, _ = sess.run([model.merged, model.cost, model.train_op],
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: keep_probability})


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
                                  len(training_sorted) // batch_size,
                                  batch_loss / display_step,
                                  batch_time))
                    # Reset
                    batch_loss = 0
                    batch_time = 0

                # Run validation testing
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(get_batches(testing_sorted, batch_size, threshold)):
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

                    # Reset
                    batch_time_testing = 0

                    # Save new model if new minimum
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('Model is improved, Saving!')
                        stop_early = 0
                        checkpoint = "./checkpoints/lstm.ckpt"
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("Model has not improved, will not save.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Model has not improved for a while, stopping to avoid overfitting")
                break

def train_lstm_model():
    for keep_probability in [0.75]:
        for num_layers in [2]:
            for threshold in [0.95]:
                model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)
                train(model, epochs)

if __name__ == "__main__":
    train_lstm_model()
