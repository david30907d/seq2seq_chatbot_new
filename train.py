import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
from tqdm import tqdm
import math, os, subprocess
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
tf.app.flags.DEFINE_string('validation', 'validation.out', 'log of validation')
FLAGS = tf.app.flags.FLAGS
checkpoint_path = os.path.join(FLAGS.model_dir, FLAGS.model_name)
word2id, id2word, trainingSamples = loadDataset(FLAGS.data_path)

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.allow_growth = True
model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate, word2id, mode='train', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0, gpu_num='0')
with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    current_step = 0
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir, graph=sess.graph)
    for e in range(FLAGS.numEpochs):
        print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
        batches = getBatches(trainingSamples, FLAGS.batch_size)
        for nextBatch in tqdm(batches, desc="Training"):
            loss, summary = model.train(sess, nextBatch)
            current_step += 1
            if current_step % FLAGS.steps_per_checkpoint == 0:
                model.saver.save(sess, checkpoint_path, global_step=current_step)
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, loss, perplexity))
                summary_writer.add_summary(summary, current_step)
                subprocess.call(['python3', 'validation.py' '--batch_size', str(FLAGS.batch_size), '--data_path', str(FLAGS.data_path), '--rnn_size', str(FLAGS.rnn_size), '--embedding_size', str(FLAGS.embedding_size), '--numEpochs', str(FLAGS.numEpochs), '--CUDA', '1'])
