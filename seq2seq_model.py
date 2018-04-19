import pickle
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

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

x=input_x
y=input_y
word2vec_model = gensim_model
batch = 1
gensim_model['했']
gensim_model['<EOS>']
def make_batch(x,y,batch,word2vec_model):
    batch_idx = np.random.choice(x.shape[0], batch, False)
    xs = x[batch_idx][:]
    ys = y[batch_idx][:]

    encoder_input = []
    dynamic_sequence_length = []
    for sentences in xs:
        dynamic_sequence_length.append(len(sentences))
        for words in sentences:
            for word in words:
                encoder_input.append(word2vec_model[word].tolist())

    return {encoder_x:encoder_input}
batch = make_batch(input_x, input_y, doc_batch, gensim_model)
pred, ls, tr = sess.run([decoder_logits, loss, train], feed_dict=batch)

    return {encoder_x:encoder_input,
            decoder_x:b,
            decoder_y:c,
            encoder_seq_len:dynamic_sequence_length,
            decoder_seq_len:e}

    # 문서 하나에 단어를 쭉 나열해서 입력으로 사용
    # 한 문서 내 최대 단어 수 : max_word_len, 한 문서 내 최대 문장 수 : max_seq_len,
    max_number_of_word = 0
    max_number_of_sequence = 0
    dyseq_length = []
    for sentences in xs:
        word_length_sum = 0
        tmp_seq_len = len(sentences)
        dyseq_length.append(tmp_seq_len)
        for words in sentences:
            word_length_sum += len(words)
        if max_number_of_word < word_length_sum:
            max_number_of_word = word_length_sum
        if max_number_of_sequence < tmp_seq_len:
            max_number_of_sequence = tmp_seq_len

    # 단어 길이가 max보다 적은 문서는 0 패딩을 붙인다
    seq_eles = []
    seq_lens = []
    doc_idx = 0
    for content in xs:
        tmp_eles = []
        tmp_lens = []
        word_length_sum = 0
        for element in content:
            tmp_element = []
            for word in element:
                tmp_element.append(gensim_model[word].tolist())
            tmp_eles += tmp_element
            word_length_sum += len(element)
            tmp_lens += [[doc_idx, word_length_sum - 1]]
        tmp_eles += ([[0.0] * word_dimension] * (max_word_len - len(tmp_eles)))
        for i in range(-(max_seq_len - len(tmp_lens)), 0):
            tmp_lens += ([[doc_idx, max_word_len + i]])
        seq_eles.append(tmp_eles)
        seq_lens.append(tmp_lens)
        doc_idx += 1

    y_label = []
    for idx, label in enumerate(ys):
        tmp_ys = [[0, 0]] * max_seq_len
        tmp_ys[:len(ys[idx])] = ys[idx]
        y_label.append(tmp_ys)

    return {encoder_x:, decoder_x:, decoder_y:, encoder_seq_len:max_seq_len, decoder_seq_len:}
    return {model_x: seq_eles, dyseq_len: dyseq_length, model_y: y_label, sequence_length: seq_lens}


for i in range(50000):
    batch = make_batch(input_x, input_y, doc_batch, word_dimension, gensim_model)
    pred, ls, tr = sess.run([decoder_logits, loss, train], feed_dict=batch)
    if i % 50 == 0:
        print("iteration :", i, "loss : ", ls)
    if i % 500 == 0:
        print(pred)

###########################




x = [[5, 7, 8], [6, 3], [3], [1]]
import helpers
xt, xlen = helpers.batch(x)
x
xt
xlen


import numpy as np
import tensorflow as tf
import helpers

tf.reset_default_graph()
sess = tf.InteractiveSession()

tf.__version__

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)

del encoder_outputs


encoder_final_state

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,

    initial_state=encoder_final_state,

    dtype=tf.float32, time_major=True, scope="plain_decoder",
)


decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

decoder_prediction = tf.argmax(decoder_logits, 2)

decoder_logits

stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

batch_ = [[6], [3, 4], [9, 8, 7]]

batch_, batch_length_ = helpers.batch(batch_)
print('batch_encoded:\n' + str(batch_))

din_, dlen_ = helpers.batch(np.ones(shape=(3, 1), dtype=np.int32),
                            max_sequence_length=4)
print('decoder inputs:\n' + str(din_))

pred_ = sess.run(decoder_prediction,
    feed_dict={
        encoder_inputs: batch_,
        decoder_inputs: din_,
    })
print('decoder predictions:\n' + str(pred_))

batch_size = 100

batches = helpers.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

print('head of the batch:')
for seq in next(batches)[:10]:
    print(seq)


def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

loss_track = []

max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')

%matplotlib inline
import matplotlib.pyplot as plt
plt.plot(loss_track)
print('loss {:.4f} after {} examples (batch_size={})'.format(loss_track[-1], len(loss_track)*batch_size, batch_size))