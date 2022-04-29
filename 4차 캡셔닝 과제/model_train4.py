# -*- coding: utf-8 -*-
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import sys
import json
import pandas as pd
import random
from datasetProcessing import DataTrain
from datasetProcessing import DataVal
from Attention import attention

'''
주석 상, [1], [2]는 각 과제를 수행할 때 필수적으로 수정해야 하는 부분을 나타냈습니다.
환경 설치 상 문의 사항: yellowjs0304@gmail.com
out of memory 오류 시, 메모리 부족에 의한 오류임. 
batch_size를 줄이거나, 모델 구조의 파라미터(ex. 유닛수)를 줄여야함
#주의사항#
[1] 과제는 Bidirectional()가 아닌, tf.nn.rnn_cell.BasicLSTMCell(https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/BasicLSTMCell?hl=ko)을 사용하여 biLSTM을 정의하시면 됩니다.
[2] 과제는 다른 오픈소스 사용을 방지하기 위해 제공해드린 Tensorflow자료(https://www.tensorflow.org/tutorials/text/image_captioning) 내 class BahdanauAttention과 주석 라인 등을 참고하셔서 attention을 수행하시면 되겠습니다.
'''

'''
[2] Attention 클래스 다른 파일로 정의 후, Import 하여 사용.
'''

# :: 본인 컴퓨터 성능에 따라서 자유롭게 조절.
# 가장 최고의 성능을 낼 수 있도록 설정 후 실험 진행.
# num_saver_epoches를 조절하여 몇 Epoch마다 모델을 저장할지 조절 가능.
# 모델 크기가 크기 때문에 본인 메모리에 따라서 자유롭게 조절

load_saver = False
test_mode = False
learning_rate = 0.001
num_epoches = 20  # 전체 Epoch 횟수
batch_size = 128  # Batch Size 조절
num_display_steps = 15  # 몇 Epoch마다 결과 창을 보여줄지
num_saver_epoches = 5  # 모델을 몇 Epoch마다 저장할지
'''
- save_dir : 학습 모델이 저장될 경로
- log_dir : log(Tensorboard) 파일이 저장될 경로
'''
save_dir = 'save_model/'
log_dir = 'logs/'
output_filename = 'final_output.txt'

'''
본인 경로로 변경
'''
data_dir = 'C:\\Users\\Kim DongBeen\\Desktop\\homework_4-master\\homework_4-master\\data'
test_dir = 'C:\\Users\\Kim DongBeen\\Desktop\\homework_4-master\\homework_4-master\\data\\testing_data'

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
phases = {'train': 0, 'val': 1, 'test': 2}
random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

n_inputs = 4096  # 이미지 Feature Input Size
n_hidden = 256  # LSTM Hidden State Size
val_batch_size = 100  # Validation Batch Size
n_frames = 80  # 연속으로 처리하는 프레임 개수
max_caption_len = 50  # 문장 최대 길이
forget_bias_red = 1.0
forget_bias_gre = 1.0
dropout_prob = 0.5

'''
[2] n_attention 정의 
: hidden size와 동일하게 정의..
'''
n_attention = 256


# Video Captioning 모델 정의
class vedio2text:
    def __init__(self, vocab_num=0, lr=learning_rate):

        self.vocab_num = vocab_num
        self.learning_rate = lr
        self.saver = None

    def set_saver(self, saver):
        self.saver = saver

    def build_model(self, feat, captions=None, cap_len=None, sampling=None, phase=0):

        weights = {
            'W_feat': tf.Variable(tf.random_uniform([n_inputs, n_hidden], -0.1, 0.1), name='W_feat'),
            'W_dec': tf.Variable(tf.random_uniform([n_hidden, self.vocab_num], -0.1, 0.1), name='W_dec')
        }
        biases = {
            'b_feat': tf.Variable(tf.zeros([n_hidden]), name='b_feat'),
            'b_dec': tf.Variable(tf.zeros([self.vocab_num]), name='b_dec')
        }
        embeddings = {
            'emb': tf.Variable(tf.random_uniform([self.vocab_num, n_hidden], -0.1, 0.1), name='emb')
        }
        '''
        [2] Attention Layer 정의.
        unit 개수 == n_hidden 
        '''
        x = attention(n_attention)

        batch_size = tf.shape(feat)[0]

        if phase != phases['test']:
            cap_mask = tf.sequence_mask(cap_len, max_caption_len, dtype=tf.float32)

        if phase == phases['train']:  # add noise
            noise = tf.random_uniform(tf.shape(feat), -0.1, 0.1, dtype=tf.float32)
            feat = feat + noise

        if phase == phases['train']:
            feat = tf.nn.dropout(feat, dropout_prob)

        '''
        Image Embedding 처리
        '''

        feat = tf.reshape(feat, [-1, n_inputs])
        image_emb = tf.matmul(feat, weights['W_feat']) + biases['b_feat']
        image_emb = tf.reshape(image_emb, [-1, n_frames, n_hidden])
        image_emb = tf.transpose(image_emb, perm=[1, 0, 2])

        '''
        [1] biLSTM 정의
        lstm_red, lstm_gre 마다 Backward용 LSTMCell, Wrapper를 따로 정의
        tensorflow.keras.layer의 Bidirectional 사용 금지.

        '''
        # LSTM 1 : 프레임의 시간적 흐름 모델링 / CNN을 거친 이미지 프레임(.npy값)을 받아 처리
        with tf.variable_scope('LSTM1'):
            lstm_red = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=forget_bias_red, state_is_tuple=True)
            if phase == phases['train']:
                lstm_red = tf.contrib.rnn.DropoutWrapper(lstm_red, output_keep_prob=dropout_prob)

        # LSTM 2 : 단어 흐름 모델링 / 1번 LSTM의 hidden embedding을 받아 처리
        with tf.variable_scope('LSTM2'):
            lstm_gre = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=forget_bias_gre, state_is_tuple=True)
            if phase == phases['train']:
                lstm_gre = tf.contrib.rnn.DropoutWrapper(lstm_gre, output_keep_prob=dropout_prob)

                # 초기 LSTM zero state로 설정
        state_red = lstm_red.zero_state(batch_size, dtype=tf.float32)
        state_gre = lstm_gre.zero_state(batch_size, dtype=tf.float32)
        '''
        [1] Backward용 LSTMCell에 맞춰 zero State 설정
        '''

        padding = tf.zeros([batch_size, n_hidden + n_attention])
        '''
        2. attention 일 경우, padding의 사이즈를 늘려줘야 함..
        기존 : [batch size, n_hidden]
        개선 : [batch_size, n_hidden + n_attention]
        '''

        h_src = []
        '''

        이미지 프레임 (Encoding)처리 단
        [1] backward 용 output, state 정의, 
        forward, backward에 나온 결과 값을 Concat하여 기존 output을 대체하면 됨..
        마지막 h_src에도 
        '''

        ## 이미지 프레임(Visual 정보 처리 단)##
        # LSTM_RED : image / LSTM_GREEN : [Pad]
        for i in range(0, n_frames):
            with tf.variable_scope("LSTM1"):
                output_red, state_red = lstm_red(image_emb[i, :, :], state_red)

            with tf.variable_scope("LSTM2"):
                output_gre, state_gre = lstm_gre(tf.concat([padding, output_red], axis=1), state_gre)
                h_src.append(output_gre)  # even though padding is augmented, output_gre/state_gre's shape not change
        ######################################

        h_src = tf.stack(h_src, axis=0)

        bos = tf.ones([batch_size, n_hidden])
        padding_in = tf.zeros([batch_size, n_hidden])

        logits = []
        max_prob_index = None

        cross_ent_list = []

        '''
        문장 단어 (Decoding) 처리 단
        [1] backward 용 output, state 정의, 
        forward, backward에 나온 결과 값을 Concat하여 기존 output을 대체하면 됨..
        '''
        ## 단어 시퀀스(Textual 정보 처리 단) ##
        # LSTM_RED : [PAD] / LSTM_GREEN : Word Embedding
        for i in range(0, max_caption_len):
            with tf.variable_scope("LSTM1"):
                output_red, state_red = lstm_red(padding_in, state_red)

            if i == 0:
                with tf.variable_scope("LSTM2"):
                    con = tf.concat([bos, output_red], axis=1)
                    '''
                    [2]
                    h_src를 2차원으로 변경 후,
                    ex) (80, none, 256) --> (none,256)

                    Attention Layer 추가.
                    h_src, state_gre[1]을 Input으로 사용.
                    Decoder(lstm_gre) 모델에 적용

                    lstm_gre의 Input에 기존 con과 context_vector를 연결 후, 기존 con을 대체

                    '''
                    h_src = tf.squeeze(h_src)

                    context_vector = x.call(h_src, state_gre[1])
                    con = tf.concat([con, context_vector], axis=1)
                    output_gre, state_gre = lstm_gre(con, state_gre)
            else:
                if phase == phases['train']:
                    if sampling[i] == True:
                        feed_in = captions[:, i - 1]
                    else:
                        feed_in = tf.argmax(logit_words, 1)
                else:
                    feed_in = tf.argmax(logit_words, 1)
                with tf.device("/cpu:0"):
                    embed_result = tf.nn.embedding_lookup(embeddings['emb'], feed_in)
                with tf.variable_scope("LSTM2"):
                    con = tf.concat([embed_result, output_red], axis=1)
                    '''
                    [2] Attention Layer 추가.
                    h_src, state_gre[1]을 Input으로 사용.
                    Decoder(lstm_gre) 모델에 적용

                    lstm_gre의 Input에 기존 con과 context_vector를 연결 후, 기존 con을 대체

                    '''
                    context_vector = x.call(h_src, state_gre[1])
                    con = tf.concat([con, context_vector], axis=1)
                    output_gre, state_gre = lstm_gre(con, state_gre)

            '''
            [1] Backward, Forward한 값(Concat)을 tf.matmul의 입력으로 제공.
            '''

            logit_words = tf.matmul(output_gre, weights['W_dec']) + biases['b_dec']
            logits.append(logit_words)

            if phase != phases['test']:
                labels = captions[:, i]
                one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value=1, off_value=None, axis=1)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=one_hot_labels)
                cross_entropy = cross_entropy * cap_mask[:, i]
                cross_ent_list.append(cross_entropy)
        #########################################

        loss = 0.0
        if phase != phases['test']:
            cross_entropy_tensor = tf.stack(cross_ent_list, 1)
            loss = tf.reduce_sum(cross_entropy_tensor, axis=1)
            loss = tf.divide(loss, tf.cast(cap_len, tf.float32))
            loss = tf.reduce_mean(loss, axis=0)

        logits = tf.stack(logits, axis=0)
        logits = tf.reshape(logits, (max_caption_len, batch_size, self.vocab_num))
        logits = tf.transpose(logits, [1, 0, 2])

        summary = None
        if phase == phases['train']:
            summary = tf.summary.scalar('training loss', loss)
        elif phase == phases['val']:
            summary = tf.summary.scalar('validation loss', loss)

        return logits, loss, summary

    def inference(self, logits):

        dec_pred = tf.argmax(logits, 2)
        return dec_pred

    def optimize(self, loss_op):

        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)  # .minimize(loss_op)
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, params))
        return train_op


def train():
    datasetTrain = DataTrain(data_dir, batch_size)
    datasetTrain.build_train_data_obj_list()
    vocab_num = datasetTrain.dump_tokenizer()
    datasetVal = DataVal(data_dir, val_batch_size)
    datasetVal.build_val_data_obj_list()
    _ = datasetVal.load_tokenizer()  # vocab_num are the same

    train_graph = tf.Graph()
    val_graph = tf.Graph()

    with train_graph.as_default():
        feat = tf.placeholder(tf.float32, [None, n_frames, n_inputs])
        captions = tf.placeholder(tf.int32, [None, max_caption_len])
        sampling = tf.placeholder(tf.bool, [max_caption_len])
        cap_len = tf.placeholder(tf.int32, [None])

        # 모델 정의
        model = vedio2text(vocab_num=vocab_num,
                           lr=learning_rate)
        # logit 계산.
        logits, loss_op, summary = model.build_model(feat, captions, cap_len, sampling, phases['train'])
        dec_pred = model.inference(logits)
        train_op = model.optimize(loss_op)

        model.set_saver(tf.train.Saver(max_to_keep=3))
        init = tf.global_variables_initializer()
    train_sess = tf.Session(graph=train_graph)

    with val_graph.as_default():
        feat_val = tf.placeholder(tf.float32, [None, n_frames, n_inputs])
        captions_val = tf.placeholder(tf.int32, [None, max_caption_len])
        cap_len_val = tf.placeholder(tf.int32, [None])

        model_val = vedio2text(vocab_num=vocab_num, lr=learning_rate)
        logits_val, loss_op_val, summary_val = model_val.build_model(feat_val,
                                                                     captions_val, cap_len_val, phase=phases['val'])
        dec_pred_val = model_val.inference(logits_val)

        model_val.set_saver(tf.train.Saver(max_to_keep=3))
    val_sess = tf.Session(graph=val_graph)

    load = load_saver
    if not load:
        train_sess.run(init)
    else:
        saver_path = save_dir
        latest_checkpoint = tf.train.latest_checkpoint(saver_path)
        model.saver.restore(train_sess, latest_checkpoint)

    ckpts_path = save_dir + "save_net.ckpt"
    summary_writer = tf.summary.FileWriter(log_dir + '/train')
    summary_writer.add_graph(train_graph)
    summary_writer.add_graph(val_graph)

    samp_prob = 0.6
    for epo in range(num_epoches):
        datasetTrain.shuffle_perm()
        num_steps = int(datasetTrain.batch_max_size / batch_size)
        epo_loss = 0
        for i in range(0, num_steps):
            data_batch, label_batch, caption_lens_batch, id_batch = datasetTrain.next_batch()
            samp = datasetTrain.schedule_sampling(samp_prob, caption_lens_batch)
            if i % num_display_steps == 1:
                # training
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                _, loss, p, summ = train_sess.run([train_op, loss_op, dec_pred, summary],
                                                  feed_dict={feat: data_batch,
                                                             captions: label_batch,
                                                             cap_len: caption_lens_batch,
                                                             sampling: samp},
                                                  options=run_options)
                summary_writer.add_summary(summ, global_step=(epo * num_steps) + i)
                print("\n Training Epoch " + str(epo) + ",......")

            else:
                _, loss, p = train_sess.run([train_op, loss_op, dec_pred],
                                            feed_dict={feat: data_batch,
                                                       captions: label_batch,
                                                       cap_len: caption_lens_batch,
                                                       sampling: samp})
            epo_loss += loss
        av_loss = epo_loss / num_steps
        print("\n Finished Epoch " + str(epo) + \
              ", (Training Loss (avarage loss in this epoch): " + "{:.4f}".format(av_loss) + ")")

        if epo % num_saver_epoches == 0:
            ckpt_path = model.saver.save(train_sess, ckpts_path,
                                         global_step=(epo * num_steps) + num_steps - 1)
            # validation
            model_val.saver.restore(val_sess, ckpt_path)
            print("\n Validating Epoch " + str(epo) + ",......")

            num_steps_val = int(datasetVal.batch_max_size / val_batch_size)
            total_loss_val = 0
            for j in range(0, num_steps_val):
                data_batch, label_batch, caption_lens_batch, id_batch = datasetVal.next_batch()
                loss_val, p_val, summ = val_sess.run([loss_op_val, dec_pred_val, summary_val],
                                                     feed_dict={feat_val: data_batch,
                                                                captions_val: label_batch,
                                                                cap_len_val: caption_lens_batch})

                total_loss_val += loss_val
                summary_writer.add_summary(summ, global_step=(epo * num_steps_val) + j)

            print("Validation: " + str((j + 1) * val_batch_size) + "/" + \
                  str(datasetVal.batch_max_size) + ", done..." \
                  + "Total Loss: " + "{:.4f}".format(total_loss_val))
    print('\n\nTraining finished!')


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run(main=main)
