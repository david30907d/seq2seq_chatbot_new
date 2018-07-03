import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np


tf.app.flags.DEFINE_integer('rnn_size', 50, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 200, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_string('data_path', 'data/gensim2file.pkl', 'Filepath of intput data')

FLAGS = tf.app.flags.FLAGS

word2id, id2word, trainingSamples = loadDataset(FLAGS.data_path)

with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='decode', use_attention=True, beam_search=True, beam_size=5, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        print(model.infer(sess, sentence, isBatch=False))
        print("> ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()

