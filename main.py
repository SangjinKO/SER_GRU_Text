import tensorflow as tf
import pickle
import random

from tensorflow.core.framework import summary_pb2
import numpy as np

import os
import time
import datetime


from modelRNN import *
from configRNN import *

### Functions  for data processing
class ProcessDataText:
    # store data
    train_set = []
    dev_set = []
    test_set = []

    def __init__(self, data_path):
        self.data_path = data_path

        # load data
        self.train_set = self.load_data(DATA_TRAIN_TRANS, DATA_TRAIN_LABEL)
        self.dev_set = self.load_data(DATA_DEV_TRANS, DATA_DEV_LABEL)
        self.test_set = self.load_data(DATA_TEST_TRANS, DATA_TEST_LABEL)

        self.dic_size = 0
        # with open( data_path + DIC ) as f:
        with open(data_path + DIC, "rb") as f:
            self.dic_size = len(pickle.load(f))

    def load_data(self, text_trans, label):

        print('load data : ' + text_trans + ' ' + label)
        output_set = []
        tmp_text_trans = np.load(self.data_path + text_trans)
        tmp_label = np.load(self.data_path + label)

        for i in range(len(tmp_label)):
            output_set.append([tmp_text_trans[i], tmp_label[i]])
        print('[completed] load data')

        return output_set

    def get_glove(self):
        return np.load(self.data_path + GLOVE)

    def get_batch(self, data, batch_size, encoder_size, is_test=False, start_index=0):

        encoder_inputs, encoder_seq, labels = [], [], []
        index = start_index

        # Get a random batch of encoder and encoderR inputs from data,
        # pad them if needed
        for _ in range(batch_size):

            if is_test is False:
                # train case -  random sampling
                trans, label = random.choice(data)

            else:
                # dev, test case = ordered data
                if index >= len(data):
                    trans, label = data[0]  # won't be evaluated
                    index += 1
                else:
                    trans, label = data[index]
                    index += 1

            tmp_index = np.where(trans == 0)[0]  # find the pad index
            if (len(tmp_index) > 0):  # pad exists
                seqN = np.min((tmp_index[0], encoder_size))
            else:  # no-pad
                seqN = encoder_size

            encoder_inputs.append(trans[:encoder_size])
            encoder_seq.append(seqN)

            tmp_label = np.zeros(N_CATEGORY, dtype=np.float)
            tmp_label[label] = 1
            labels.append(tmp_label)

        return encoder_inputs, encoder_seq, np.reshape(labels, (batch_size, N_CATEGORY))

### Functions for evaluation (used in validation and testing)
def run_test(sess, model, batch_gen, data):
    list_batch_ce = []
    list_batch_correct = []

    list_pred = []
    list_label = []

    max_loop = len(data) / model.batch_size
    remaining = len(data) % model.batch_size

    max_loop = int(round(max_loop + 0.5))
    print("TESTTEST: ", max_loop)

    # evaluate data ( N of chunk (batch_size) + remaining( +1) )
    for test_itr in range(max_loop + 1):

        raw_encoder_inputs, raw_encoder_seq, raw_label = batch_gen.get_batch(
            data=data,
            batch_size=model.batch_size,
            encoder_size=model.encoder_size,
            is_test=True,
            start_index=(test_itr * model.batch_size)
        )

        # prepare data which will be push from pc to placeholder
        input_feed = {}

        input_feed[model.encoder_inputs] = raw_encoder_inputs
        input_feed[model.encoder_seq] = raw_encoder_seq
        input_feed[model.y_labels] = raw_label
        input_feed[model.dr_prob] = 1.0  # no drop out while evaluating

        try:
            bpred, bloss = sess.run([model.batch_pred, model.batch_loss], input_feed)
        except:
            print("excepetion occurs in valid step : " + str(test_itr))
            pass

        # remaining data case (last iteration)
        if test_itr == (max_loop):
            bpred = bpred[:remaining]
            bloss = bloss[:remaining]
            raw_label = raw_label[:remaining]

        # batch loss
        list_batch_ce.extend(bloss)

        # batch accuracy
        list_pred.extend(np.argmax(bpred, axis=1))
        list_label.extend(np.argmax(raw_label, axis=1))

    if IS_LOGGING:
        with open('../analysis/inference_log/text.txt', 'w') as f:
            f.write(' '.join([str(x) for x in list_pred]))

        with open('../analysis/inference_log/text_label.txt', 'w') as f:
            f.write(' '.join([str(x) for x in list_label]))

    list_batch_correct = [1 for x, y in zip(list_pred, list_label) if x == y]

    sum_batch_ce = np.sum(list_batch_ce)
    accr = np.sum(list_batch_correct) / float(len(data))

    value1 = summary_pb2.Summary.Value(tag="valid_loss", simple_value=sum_batch_ce)
    value2 = summary_pb2.Summary.Value(tag="valid_accuracy", simple_value=accr)
    summary = summary_pb2.Summary(value=[value1, value2])

    return sum_batch_ce, accr, summary, list_pred

### Functions  for training
def train_step(sess, model, batch_gen):
    raw_encoder_inputs, raw_encoder_seq, raw_label = batch_gen.get_batch(
        data=batch_gen.train_set,
        batch_size=model.batch_size,
        encoder_size=model.encoder_size,
        is_test=False
    )

    # prepare data which will be push from pc to placeholder
    input_feed = {}

    input_feed[model.encoder_inputs] = raw_encoder_inputs
    input_feed[model.encoder_seq] = raw_encoder_seq
    input_feed[model.y_labels] = raw_label
    input_feed[model.dr_prob] = model.dr

    _, summary = sess.run([model.optimizer, model.summary_op], input_feed)

    return summary

def train_model(model, batch_gen, num_train_steps, valid_freq, is_save=0, graph_dir_name='default'):
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    summary = None
    val_summary = None

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())
        early_stop_count = MAX_EARLY_STOP_COUNT

        if model.use_glove == 1:
            sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: batch_gen.get_glove()})
            print('[completed] loading pre-trained embedding vector to placeholder')

        # if exists check point, starts from the check point
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('save/' + graph_dir_name + '/'))
        if ckpt and ckpt.model_checkpoint_path:
            print('from check point!!!')
            saver.restore(sess, ckpt.model_checkpoint_path)

        writer = tf.summary.FileWriter('./graph/' + graph_dir_name, sess.graph)

        initial_time = time.time()

        min_ce = 1000000
        best_dev_accr = 0
        test_accr_at_best_dev = 0

        for index in range(num_train_steps):

            try:
                # run train
                summary = train_step(sess, model, batch_gen)
                writer.add_summary(summary, global_step=model.global_step.eval())

            except:
                print("excepetion occurs in train step")
                pass

            # run validation
            if (index + 1) % valid_freq == 0:

                dev_ce, dev_accr, dev_summary, _ = run_test(sess=sess,
                                                            model=model,
                                                            batch_gen=batch_gen,
                                                            data=batch_gen.dev_set)

                writer.add_summary(dev_summary, global_step=model.global_step.eval())

                end_time = time.time()

                if index > CAL_ACCURACY_FROM:

                    if (dev_ce < min_ce):
                        min_ce = dev_ce

                        # save best result
                        if is_save is 1:
                            saver.save(sess, 'save/' + graph_dir_name + '/', model.global_step.eval())

                        early_stop_count = MAX_EARLY_STOP_COUNT

                        test_ce, test_accr, _, _ = run_test(sess=sess,
                                                            model=model,
                                                            batch_gen=batch_gen,
                                                            data=batch_gen.test_set)

                        best_dev_accr = dev_accr
                        test_accr_at_best_dev = test_accr

                    else:
                        # early stopping
                        if early_stop_count == 0:
                            print("early stopped")
                            break

                        test_accr = 0
                        early_stop_count = early_stop_count - 1

                    print(str(int(end_time - initial_time) / 60) + " mins" + \
                          " step/seen/itr: " + str(model.global_step.eval()) + "/ " + \
                          str(model.global_step.eval() * model.batch_size) + "/" + \
                          str(round(model.global_step.eval() * model.batch_size / float(len(batch_gen.train_set)), 2)) + \
                          "\tdev: " + '{:.3f}'.format(dev_accr) + "  test: " + '{:.3f}'.format(
                        test_accr) + "  loss: " + '{:.2f}'.format(dev_ce))

        writer.close()

        print('Total steps : {}'.format(model.global_step.eval()))

        # result logging to file
        with open('./TEST_run_result.txt', 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + '\t' + \
                    batch_gen.data_path.split('/')[-2] + '\t' + \
                    graph_dir_name + '\t' + str(best_dev_accr) + '\t' + str(test_accr_at_best_dev) + '\n')


def create_dir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


#### MAIN ####
data_path = ''
batch_size = 128
encoder_size = 128
num_layer = 1
hidden_dim = 200
lr = 0.001
num_train_steps = 10000
is_save = 0
graph_prefix = 'SER_GRU_TEXT'
use_glove = 1
dr = 0.3

embed_train = ''
if EMBEDDING_TRAIN == False:
    embed_train = 'F'

graph_name = graph_prefix + \
             '_D' + (data_path).split('/')[-2] + \
             '_b' + str(batch_size) + \
             '_es' + str(encoder_size) + \
             '_L' + str(num_layer) + \
             '_H' + str(hidden_dim) + \
             '_G' + str(use_glove) + embed_train + \
             '_dr' + str(dr)

graph_name = graph_name + '_' + datetime.datetime.now().strftime("%m-%d-%H-%M")

if is_save is 1:
    create_dir('save/')
    create_dir('save/' + graph_name)

create_dir('graph/')
create_dir('graph/' + graph_name)

batch_gen = ProcessDataText(data_path)

model = SingleEncoderModelText(
    dic_size=batch_gen.dic_size,
    use_glove=use_glove,
    batch_size=batch_size,
    encoder_size=encoder_size,
    num_layer=num_layer,
    lr=lr,
    hidden_dim=hidden_dim,
    dr=dr
)

model.build_graph()

valid_freq = int(len(batch_gen.train_set) * EPOCH_PER_VALID_FREQ / float(batch_size)) + 1
print("[Info] Valid Freq = " + str(valid_freq))

train_model(model, batch_gen, num_train_steps, valid_freq, is_save, graph_name)