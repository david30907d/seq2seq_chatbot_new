import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import os, logging, random
import numpy as np
from tqdm import tqdm

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
tf.app.flags.DEFINE_string('CUDA', '0', 'GPU device')
logging.basicConfig(filename='validation.log',level=logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

word2id, id2word, trainingSamples = loadDataset('data/gensim2file.pkl')
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.CUDA

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.batch_size, word2id, mode='decode', use_attention=True, beam_search=True, beam_size=5, max_gradient_norm=5.0)
with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state('model/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logging.info('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise VadlueError('No such file:[{}]'.format('model/'))
    random.shuffle(trainingSamples)
    for sentence, _ in trainingSamples:
        logging.info(model.infer(sess, sentence, id2word, isBatch=True))
