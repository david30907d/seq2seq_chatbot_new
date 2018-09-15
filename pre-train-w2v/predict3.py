import tensorflow as tf
from dataHelper3 import sentence2enco
from model3 import Seq2SeqModel
from tqdm import tqdm
import math
import os
import sys
import numpy as np
import json
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.app.flags.DEFINE_integer('rnn_size', 512, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 4, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 512, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 10, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 1, 'Maximum epochs of training')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 5, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
tf.app.flags.DEFINE_string('model_name', 'chatbot.ckpt', 'File name used for model checkpoints')
tf.app.flags.DEFINE_integer('train_size', 100, 'Training data size')
tf.app.flags.DEFINE_integer('validation_size', 10, 'Validation data')
tf.app.flags.DEFINE_boolean('add_noise', True, 'Add noise to sentences')
tf.app.flags.DEFINE_float('noise_prob', 0.1, 'denoise autoencoder omission and swap probability')
tf.app.flags.DEFINE_boolean('use_attention', False, 'use attention or not')
tf.app.flags.DEFINE_boolean('use_pre_train', True, 'use gensim pre-train word embedding or not')
# tf.app.flags.DEFINE_boolean('decoder_init', False, 'decoder weight initializer')
tf.app.flags.DEFINE_boolean('lock', True, 'Training mode lock encoder or not')
tf.app.flags.DEFINE_boolean('only_wuxia', True, 'denoise autoencoder omission and swap probability')
tf.app.flags.DEFINE_boolean('second_step', True, 'styleTransfer second training (use wuxia novel to train decoder)')
tf.app.flags.DEFINE_integer('beam_size', 30, 'beam search size')
FLAGS = tf.app.flags.FLAGS


def output_words_probability(softmaxResult, word_ids, id2word, beam_szie):
    for i in range(len(softmaxResult[0])):
        count = 0
        for j in range(len(softmaxResult[0][i])):
            # print(j)
            count += softmaxResult[0][i][j]
            print(id2word[str(word_ids[0][i][j])], '%.10f' % softmaxResult[0][i][j])
            # print("------")
        # print(i)
        print(count)
        print("*****")
        break


def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    # print("predict_ids: ",predict_ids)
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            # print("predict_list: ",predict_list)
            predict_seq = [id2word[str(idx)] for idx in predict_list[0]]
            print(" ".join(predict_seq))


word2idFile = '../novel_sentence4/word2id'
id2wordFile = '../novel_sentence4/id2word'
if FLAGS.only_wuxia:
    target_word2idFile = '../novel_sentence4/decoder_word2id_compress'
    target_id2wordFile = '../novel_sentence4/decoder_id2word_compress'
else:
    target_word2idFile = word2idFile
    target_id2wordFile = id2wordFile   
pre_trainFile = '../novel_sentence4/gensim_5W.txt'

word2id = json.load(open(word2idFile,'r'))
id2word = json.load(open(id2wordFile,'r'))
target_word2id = json.load(open(target_word2idFile,'r'))
target_id2word = json.load(open(target_id2wordFile,'r'))
pre_train_embedding = pickle.load(open(pre_trainFile,'rb'))


input_vocab_size = len(word2id)
if FLAGS.only_wuxia:
    output_vocab_size = len(target_word2id)
else:
    output_vocab_size = len(word2id)

with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id, pre_train_embedding, input_vocab_size,
                         output_vocab_size, use_pre_train=FLAGS.use_pre_train, mode='decode', use_attention=FLAGS.use_attention, beam_search=True,
                         beam_size=FLAGS.beam_size, lock=FLAGS.lock, max_gradient_norm=5.0)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(FLAGS.model_dir))

    # sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        # sentence = sentence.encode('UTF-8')
        batch = sentence2enco(sentence, word2id, FLAGS.noise_prob, id2word, target_word2id, FLAGS.only_wuxia, noise=False)
        # print("batch: ",batch)
        predicted_ids, softmaxResult, word_ids= model.infer(sess, batch)
        # print("softmaxResult", softmaxResult)
        # print("softmaxResult shape", np.shape(softmaxResult))
        # print("word_ids", word_ids)
        # print("word_ids", np.shape(word_ids))
        if FLAGS.only_wuxia:
            output_words_probability(softmaxResult, word_ids, target_id2word, FLAGS.beam_size)
            predict_ids_to_seq([predicted_ids], target_id2word, FLAGS.beam_size)
        else:
            output_words_probability(softmaxResult, word_ids, id2word, FLAGS.beam_size)
            predict_ids_to_seq([predicted_ids], id2word, FLAGS.beam_size)

        print("> ", "")
        sys.stdout.flush()
        try:
            sentence = sys.stdin.readline()
        except Exception as e:
            print(e)
            continue