import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
# from tensorflow.contrib.tensorboard.plugins import projector

class Seq2SeqModel():

    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, pre_train_embedding, input_vocab_size, output_vocab_size,
                 use_pre_train, mode, use_attention, beam_search, beam_size, lock, max_gradient_norm=5.0):
        
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learing_rate = learning_rate
        self.word_to_idx = word_to_idx
        self.pre_train_embedding = pre_train_embedding
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.use_pre_train = use_pre_train
        self.mode = mode
        self.use_attention = use_attention
        self.beam_search = beam_search
        self.beam_size = beam_size
        self.lock = lock
        self.max_gradient_norm = max_gradient_norm
        if self.use_pre_train:
            (_, self.embedding_size) = np.shape(self.pre_train_embedding)
        else:
            self.embedding_size = embedding_size
        self.build_model()



    def _create_rnn_cell(self):
        def single_rnn_cell():
            # 创建单个cell，这里需要注意的是一定要使用一个single_rnn_cell的函数，不然直接把cell放在MultiRNNCell
            # 的列表中最终模型会发生错误
            single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size, initializer=tf.orthogonal_initializer)
            #添加dropout,output_keep_prob參數代表多少比例不drop，若為1則代表不進行dropout
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        #列表中每个元素都是调用single_rnn_cell函数
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_model(self):
        print('building model......')

        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')

        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope('encoder'):
            encoder_cell = self._create_rnn_cell()



            if self.use_pre_train == True:
                print("use pre_train embedding")
                self.embedding = tf.Variable(tf.constant(0.0, shape=[self.input_vocab_size, self.embedding_size]), trainable=False, name="W")
                self.embedding_placeholder = tf.placeholder(tf.float32, [self.input_vocab_size, self.embedding_size])
                self.embedding_init = self.embedding.assign(self.embedding_placeholder)
            else:
                print("use random embedding")
                self.embedding = tf.get_variable('embedding', [self.input_vocab_size, self.embedding_size], dtype=tf.float32,
                                                 initializer=tf.truncated_normal_initializer(stddev=1e-4))


            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length,
                                                               dtype=tf.float32)

        with tf.variable_scope('decoder'):
            encoder_inputs_length = self.encoder_inputs_length
            if self.beam_search:
                print("use beamsearch decoding..")
                # 如果使用beam_search，则需要将encoder的输出进行tile_batch，其实就是复制beam_size份。
                # tile_batch()只能接受tensor，encoder_state非tensor而是structure所以要用nest.map_structure
                encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
                encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

                
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, memory=encoder_outputs,
                                                                       memory_sequence_length=encoder_inputs_length)
            decoder_cell = self._create_rnn_cell()
            if self.use_attention:
                print('use attention')
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                                   attention_layer_size=self.rnn_size, name='Attention_Wrapper')


            # tf.Print(decoder_outputs, [decoder_outputs], message="This is a: ") 
            # self.a = decoder_cell
            # self.a = tf.shape(decoder_cell)

            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            #定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
            if self.use_attention == True:
                decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            else:
                decoder_initial_state = encoder_state

            
            output_layer = tf.layers.Dense(self.output_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        if self.mode == 'train':
            # 定义decoder阶段的输入，其实就是在decoder的target开始处添加一个<go>,并删除结尾处的<end>,并进行embedding。
            # tf.strided_slice(intput, begin, end, strides步長) batch中每一筆sequence都取開頭到-1，意即最後的<end>不取
            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['__GO__']), ending], 1)
            decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                sequence_length=self.decoder_targets_length,
                                                                time_major=False, name='training_helper')
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                               initial_state=decoder_initial_state, output_layer=output_layer)

            # tf.contrib.seq2seq.dynamic_decode()    Return: (final_outputs, final_state, final_sequence_lengths)
            # decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
            # sample_id: [batch_size, decoder_targets_length], tf.int32，保存最终的编码结果。可以表示最后的答案
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=self.max_target_sequence_length)

            self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output) #不知道為啥要用tf.identity
            self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train') #返回最大值的索引值

            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                         targets=self.decoder_targets, weights=self.mask)

            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()

            optimizer = tf.train.AdamOptimizer(self.learing_rate)
            # t_vars = tf.trainable_variables()
            # train_vars = [var for var in t_vars if var.name.startswith('train')]
            trainable_params = tf.trainable_variables()
            if self.lock == True:
                trainable_params = [var for var in trainable_params if not var.name.startswith('encoder')]
            gradients = tf.gradients(self.loss, trainable_params)
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

        elif self.mode == 'decode':
            start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['__GO__'] #可改
            end_token = self.word_to_idx['__EOS__']

            if self.beam_search:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=self.embedding,
                                                                         start_tokens=start_tokens, end_token=end_token,
                                                                         initial_state=decoder_initial_state,
                                                                         beam_width=self.beam_size,
                                                                         output_layer=output_layer,
                                                                         length_penalty_weight=0.0)
                # decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding,
                #                                                            start_tokens=start_tokens, end_token=end_token)
                # inference_decoder2 = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                #                                                     initial_state=decoder_initial_state,
                #                                                     output_layer=output_layer)
            else:
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding,
                                                                           start_tokens=start_tokens, end_token=end_token)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=decoding_helper,
                                                                    initial_state=decoder_initial_state,
                                                                    output_layer=output_layer)


            self.decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                      maximum_iterations=35)   

            # decoder_outputs2, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder2,
            #                                                           maximum_iterations=35) 
            self.softmax = tf.nn.softmax(self.decoder_outputs.beam_search_decoder_output.scores)
            # self.aaa = tf.identity(self.decoder_outputs.predicted_ids)
            # self.a = tf.Print(decoder_outputs, [decoder_outputs], message="This is a: ") 


            if self.beam_search:
                self.decoder_predict_decode = self.decoder_outputs.predicted_ids
            else:
                self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)      

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    def init_embedding(self, sess):
        sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: self.pre_train_embedding})

    def train(self, sess, batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 0.8,
                      self.batch_size: len(batch.encoder_inputs)}

        _, loss, summary, global_step = sess.run([self.train_op, self.loss, self.summary_op, self.global_step], feed_dict=feed_dict)
        return loss, summary, global_step

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 0.8,
                      self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        #infer阶段只需要运行最后的结果，不需要计算loss，所以feed_dict只需要传入encoder_input相应的数据即可
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        # predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        predict, softmaxResult, word_ids = sess.run([self.decoder_predict_decode, self.softmax, self.decoder_outputs.beam_search_decoder_output.predicted_ids], feed_dict=feed_dict)

        return predict, softmaxResult, word_ids