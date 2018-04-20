import time
import pickle
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

# modeling
batch_size = 128
doc_batch = None
number_of_word = None
word_dimension = 10
output_dimension = 2
hidden_dimension = 20

encoder_x = tf.placeholder(dtype=tf.float32, shape=[doc_batch, number_of_word, word_dimension], name='encoder_inputs')
decoder_x = tf.placeholder(dtype=tf.float32, shape=[doc_batch, number_of_word, word_dimension], name='decoder_inputs')
decoder_y = tf.placeholder(dtype=tf.float32, shape=[doc_batch, output_dimension], name='decoder_targets')

encoder_seq_len = tf.placeholder(dtype=tf.int32, shape=[None])
decoder_seq_len = tf.placeholder(dtype=tf.int32, shape=[None])

encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_dimension)
decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_dimension)

_, encoder_output = tf.nn.dynamic_rnn(cell=encoder_cell, inputs=encoder_x, sequence_length=encoder_seq_len, dtype=tf.float32, scope="encoder")
_, decoder_output = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_x, sequence_length=decoder_seq_len, initial_state=encoder_output, dtype=tf.float32, scope="decoder")

decoder_logits = tf.contrib.layers.fully_connected(decoder_output.h, 2, weights_initializer=tf.contrib.layers.variance_scaling_initializer())
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=decoder_y,logits=decoder_logits))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

## load data
load_path = "S:/github/workspace/Text_Summarization/preprocessed_data/"
gensim_model = Word2Vec.load(load_path + "word2vec_model")
with open(load_path + "input_x", 'rb') as file:
    input_x = np.array(pickle.load(file))
with open(load_path + "input_y", 'rb') as file:
    input_y = np.array(pickle.load(file))

# x=input_x; y=input_y; word2vec_model = gensim_model; batch=200
# batch function
def make_batch(x, y, batch, word_dimension, word2vec_model):
    batch_idx = np.random.choice(x.shape[0], batch, False)
    xs = x[batch_idx]
    ys = y[batch_idx]

    total_input = {"encoder_input" : [],
                   "decoder_input" : [],
                   "decoder_output" : [],
                   "encoder_sequence_length" : [],
                   "decoder_sequence_length" : []}

    max_encoder_length = 0
    max_decoder_length = 0
    for sentences in xs:
        tmp_encoder_length = 0
        for words in sentences:
            tmp_decoder_length = len(words)
            tmp_encoder_length += len(words)
            if tmp_decoder_length > max_decoder_length:
                max_decoder_length = tmp_decoder_length
        if tmp_encoder_length > max_encoder_length:
            max_encoder_length = tmp_encoder_length

    tmp_encoder_input = []
    tmp_decoder_input = []
    for sentences in xs:
        count = 0
        for words in sentences:
            count += 1
            total_input["decoder_sequence_length"].append(len(words))
            for word in words:
                tmp_decoder_input.append(word2vec_model[word].tolist())
                tmp_encoder_input.append(word2vec_model[word].tolist())
            total_input["decoder_input"].append(tmp_decoder_input + (max_decoder_length - len(tmp_decoder_input)) * [[0.0] * word_dimension])
            # print("dn : ", len(tmp_decoder_input))
            # print("en : ", len(tmp_encoder_input))
            tmp_decoder_input = []
        for iter in range(count):
            total_input["encoder_input"].append(tmp_encoder_input + (max_encoder_length - len(tmp_encoder_input)) * [[0.0] * word_dimension])
            total_input["encoder_sequence_length"].append(len(tmp_encoder_input))
        tmp_encoder_input = []

    for labels in ys:
        for label in labels:
            total_input["decoder_output"].append(label)

    return {encoder_x: total_input["encoder_input"],
            decoder_x: total_input["decoder_input"],
            decoder_y: total_input["decoder_output"],
            encoder_seq_len: total_input["encoder_sequence_length"],
            decoder_seq_len: total_input["decoder_sequence_length"]}

for i in range(50000):
    start = time.time()
    batch = make_batch(input_x, input_y, batch_size, word_dimension, gensim_model)
    pred, ls, tr = sess.run([decoder_logits, loss, train], feed_dict=batch)
    if i % 50 == 0:
        print("iteration :", i, "loss : ", ls)
    if i % 500 == 0:
        print(pred)
    if i == 49999:
        print("inference time : ", time.time() - start)