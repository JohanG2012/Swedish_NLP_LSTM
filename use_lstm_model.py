import numpy as np
import pickle
from noise_maker import noise_maker
from train_lstm_model import get_batches
from lstm_model import build_graph
import tensorflow as tf

int_to_vocab = pickle.load( open( "./data/int_to_vocab.pkl", "rb" ) )
vocab_to_int = pickle.load( open( "./data/vocab_to_int.pkl", "rb" ) )
testing_sorted = pickle.load( open( "./data/testing_mini.pkl", "rb" ) )
noisy_testing_sorted = pickle.load( open("./data/noisy_testing_mini.pkl", "rb"))
#testing_sorted = pickle.load( open( "./data/testing_sorted.pkl", "rb" ) )

batch_size = 16
num_layers = 4
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
keep_probability = 0.3

def use_lstm_model():

    # Get random testing sentence, add noise
    random = np.random.randint(0,len(testing_sorted))
    text = testing_sorted[30:60]
    #text = noise_maker(text, 0.95)

    checkpoint = './checkpoints/lstm.ckpt'

    # Build LSTM model.
    model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        val_acc = []
        batch_loss_testing = 0
        batch_time_testing = 0
        for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(get_batches(testing_sorted, noisy_testing_sorted, batch_size)):
            #start_time_testing = time.time()
            summary, loss, accuracy = sess.run([model.merged, model.cost, model.accuracy],
                                            {model.inputs: input_batch,
                                            model.targets: target_batch,
                                            model.inputs_length: input_length,
                                            model.targets_length: target_length,
                                            model.keep_prob: 1})
            batch_loss_testing += loss
            val_acc.append(accuracy)
            #end_time_testing = time.time()
            #batch_time_testing += end_time_testing - start_time_testing

        n_batches_testing = batch_i + 1

        # Print result
        print('Testing Loss: {:>6.3f}, Accuracy: {:>6.3f}'
                .format(batch_loss_testing / n_batches_testing, sum(val_acc)/len(val_acc)))

        for i in range(10, 20):
            text = noisy_testing_sorted[i]
            answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size,
                                                        model.inputs_length: [len(text)]*batch_size,
                                                        model.targets_length: [len(text)+1],
                                                        model.keep_prob: [1.0]})[0]

            # Remove <PAD> from output
            pad = vocab_to_int["<PAD>"]

            print('  Validation Input: {}'.format("".join([int_to_vocab[i] for i in text])))
            print('  Validation Output: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))

    # Restore checkpoint
    # for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(get_batches(testing_sorted, noisy_testing_sorted, batch_size)):
    #     #text = noisy_testing_sorted[i]
    #     with tf.Session() as sess:
    #         saver = tf.train.Saver()
    #         saver.restore(sess, checkpoint)

    #         #Multiply by batch_size to match the model's input parameters
    #         answer_logits, loss, accuracy = sess.run([model.predictions, model.cost, model.accuracy], {model.inputs: input_batch, model.targets: target_batch,
    #                                                     model.inputs_length: input_length,
    #                                                     model.targets_length: target_length,
    #                                                     model.keep_prob: [1.0]})

    #     # Remove <PAD> from output
    #     pad = vocab_to_int["<PAD>"]

    #     print('accuracy: ' + str(accuracy) + ' loss: ' + str(loss))
    #     #print(input_batch)
    #     print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in input_batch[1]])))
    #     print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits[1] if i != pad])))


if __name__ == "__main__":
    use_lstm_model()
