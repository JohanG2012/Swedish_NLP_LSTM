import numpy as np
import pickle
from noise_maker import noise_maker
from noise_maker import encode
from lstm_model import build_graph
import tensorflow as tf
from argparse import ArgumentParser
import yaml
config = yaml.safe_load(open("./hyperparameters.yaml"))

arguments = ArgumentParser()
arguments.add_argument("name")
arguments.add_argument("text")
args = arguments.parse_args()

int_to_vocab = pickle.load( open( "./data/int_to_vocab.pkl", "rb" ) )
vocab_to_int = pickle.load( open( "./data/vocab_to_int.pkl", "rb" ) )
testing_sorted = pickle.load( open( "./data/noisy_validation_mini.pkl", "rb" ) )

def use_lstm_model():

    # Get random testing sentence, add noise
    #random = np.random.randint(0,len(testing_sorted))
    #text = testing_sorted[random]
    text = encode(args.text.lower())
    #text = noise_maker(text, 0.95)

    checkpoint = "./use_checkpoints/{}/lstm.ckpt".format(args.name)

    # Build LSTM model.
    model = build_graph(config['keep_probability'], config['rnn_size'], config['num_layers'], config['batch_size'], config['learning_rate'], config['embedding_size'], config['direction'])

    # Restore checkpoint
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)

        #Multiply by config['batch_size'] to match the model's input parameters
        answer_logits = sess.run(model.predictions, {model.inputs: [text]*config['batch_size'],
                                                     model.inputs_length: [len(text)]*config['batch_size'],
                                                     model.targets_length: [len(text)+1],
                                                     model.keep_prob: [1.0]})[0]

    # Remove <PAD> from output
    pad = vocab_to_int["<PAD>"]

    print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))
    print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))


if __name__ == "__main__":
    use_lstm_model()
