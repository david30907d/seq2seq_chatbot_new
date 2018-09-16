import tensorflow as tf
from data_helpers import loadDataset, getBatches, createBatch
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os
from nmt.utils.evaluation_utils import evaluate
import numpy as np

tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 30, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_string('data_path', 'data/testing.pkl', 'filepath of dataset')
FLAGS = tf.app.flags.FLAGS

word2id, id2word, trainingSamples = loadDataset(FLAGS.data_path)

def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            return " ".join(predict_seq)

def gen_testing_data():
    with open('data/testing.output', 'w', encoding='utf-8') as f:
        for input, label in trainingSamples:
            f.write(' '.join([id2word[i] for i in label]) + '\n')

gen_testing_data()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True

with tf.Session(config=CONFIG) as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id,
                         mode='decode', use_attention=True, beam_search=True, beam_size=5, max_gradient_norm=5.0)
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))

    with open('data/testing.predict', 'w', encoding='utf-8') as target:
        for input_wordIds, label_wordIds in trainingSamples:
            batch = createBatch([[input_wordIds, []]])
            predicted_ids = model.infer(sess, batch)
            print(predict_ids_to_seq(predicted_ids, id2word, 1))
            target.write(predict_ids_to_seq(predicted_ids, id2word, 1) + '\n')

    for metric in ['bleu', 'rouge']:
        score = evaluate('data/testing.output', 'data/testing.predict', metric)
        print(metric, score / 100)
