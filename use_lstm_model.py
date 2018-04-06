import numpy as np
import pickle
from noise_maker import noise_maker
from lstm_model import build_graph
import tensorflow as tf

int_to_vocab = pickle.load( open( "./data/int_to_vocab.pkl", "rb" ) )
vocab_to_int = pickle.load( open( "./data/vocab_to_int.pkl", "rb" ) )
testing_sorted = pickle.load( open( "./data/testing_mini.pkl", "rb" ) )
#testing_sorted = pickle.load( open( "./data/testing_sorted.pkl", "rb" ) )

batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
keep_probability = 0.75

def use_lstm_model():

    # Get random testing sentence, add noise
    random = np.random.randint(0,len(testing_sorted))
    text = testing_sorted[random]
    text = noise_maker(text, 0.95)

    checkpoint = "./checkpoints/lstm.ckpt"

    # Build LSTM model.
    model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)

    # Restore checkpoint
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size,
                                                     model.inputs_length: [len(text)]*batch_size,
                                                     model.targets_length: [len(text)+1],
                                                     model.keep_prob: [1.0]})[0]

    # Remove <PAD> from output
    pad = vocab_to_int["<PAD>"]

    print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))
    print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


if __name__ == "__main__":
    use_lstm_model()
