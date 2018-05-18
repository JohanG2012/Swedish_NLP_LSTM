import tensorflow as tf
import pickle
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from collections import namedtuple
from preprocess_data import noise_maker
import numpy as np
import time
from preprocess_data import clean_text

DATA_FOLDER = './data'

vocab_to_int = pickle.load( open( "{}/vocab_to_int.pkl".format(DATA_FOLDER), "rb" ) )
int_to_vocab = pickle.load( open( "{}/int_to_vocab.pkl".format(DATA_FOLDER), "rb" ) )
training_sorted = pickle.load( open( "{}/training_mini.pkl".format(DATA_FOLDER), "rb" ) )

# Model input pipes for data feed to the model
'''
Notes:

Tensorflow placeholders acts as pipes into the model.
name_scope makes sure so given values are from the same graph.

'''
def model_inputs():

    with tf.name_scope('inputs'):
        # ARGS: dtype, shape of the tensor to be fed, name for operation
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')

    with tf.name_scope('targets'):
        # ARGS: dtype, shape of the tensor to be fed, name for operation
        targets = tf.placeholder(tf.int32, [None, None], name='targets')

    # ARGS: dtype, name for operation
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # ARGS: dtype, shape of the tensor to be fed, name for operation
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    # ARGS: dtype, shape of the tensor to be fed, name for operation
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    # ARGS: input tensor, name for operation
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length


# Remove last word from each batch, add <GO> token to the start of each batch
def process_encoding_input(targets, vocab_to_int, batch_size):

    with tf.name_scope("processing_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input

'''
Note:

Encoding layer takes the input sequence and creates a encoded representation
'''
def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):

    if direction == 1:

        with tf.name_scope("RNN_Encoder_Cell_1D"):

            for layer in range(num_layers):

                with tf.variable_scope('encoder_{0}'.format(layer)):

                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
                    enc_output, enc_state = tf.nn.dynamic_rnn(drop, rnn_inputs, sequence_length, dtype=tf.float32)

            return enc_output, enc_state

    if direction == 2:

        with tf.name_scope("RNN_Encoder_Cell_2D"):

            for layer in range(num_layers):

                with tf.variable_scope('encoder_{0}'.format(layer)):

                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_inputs, sequence_length, dtype=tf.float32)

            # Concat outputs
            enc_output = tf.concat(enc_output, 2)

            # Use only forwarded state
            return enc_output, enc_state[0]

# Create training logits
def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, vocab_size, max_target_length):

    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input, sequence_length=targets_length, time_major=False)
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, initial_state, output_layer)
        training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_target_length)

        return training_logits

# Create inference logits
def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer, max_target_length, batch_size):
     print(max_target_length)

     with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings, start_tokens, end_token)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, initial_state, output_layer)
        inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder, output_time_major=False, impute_finished=True, maximum_iterations=max_target_length)

        return inference_logits

#Create the decoding cell and attention.
def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):

    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)

    output_layer = Dense(vocab_size, kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))

    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size, enc_output, inputs_length, normalize=False, name='BahdanauAttention')

    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attn_mech, rnn_size)
    initial_state = dec_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
    initial_state = initial_state.clone(cell_state=enc_state)

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, vocab_size, max_target_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings, vocab_to_int['<GO>'], vocab_to_int['<EOS>'], dec_cell, initial_state, output_layer, max_target_length, batch_size)

    return training_logits, inference_logits

def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length, vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):

    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers, enc_embed_input, keep_prob, direction)

    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    training_logits, inference_logits  = decoding_layer(dec_embed_input, dec_embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction)

    return training_logits, inference_logits

def build_accuracy(predictions, targets):
    correct_prediction = tf.equal(tf.cast(tf.round(predictions), tf.int32), targets)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

# Build graph
def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):

    tf.reset_default_graph()

    # Load model inputs
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # Create the training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]), targets, keep_prob, inputs_length, targets_length, max_target_length, len(vocab_to_int)+1, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction)

    # Create tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')

    with tf.name_scope("cost"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    accuracy = build_accuracy(predictions, targets)

    # Merge summaries
    merged = tf.summary.merge_all()

    # Export the nodes
    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length', 'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph
