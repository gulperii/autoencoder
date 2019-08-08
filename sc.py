import json
import random
import re
from string import punctuation
import tensorflow as tf
import numpy as np
# from scipy import stats
import hyperparameter as hp
from tensorflow.contrib.rnn import LSTMCell

# TODO: DATA NASIL TUTULACAK

# all words with frequencies in vocab data
dictWithFreq = {}
vocabToIndex = {}
indexToVocab = {}
# only words with freq > 50
vocab = []
sentenceLengths = [2, 3, 4, 1000, 5]


# # before batch
# #TODO: Stop token ekle

# Batch size kadar data okuyor



# Input must be a matrix indexed



# ??
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     average_loss = 0.0
#     for index in range(NUM_TRAIN_STEPS):
#         batch = batch_gen.next()
#         loss_batch, _ = sess.run([loss, optimizer],
#                                  feed_dict={input: batch[0], output: batch[1]})
#         average_loss += loss_batch
#         if (index + 1) % 2000 == 0:
#             print('Average loss at step {}: {:5.1f}'.format(index + 1, average_loss / (index + 1)))

