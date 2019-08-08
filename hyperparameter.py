BATCH_SIZE = 32
EMBEDDING_SIZE = 128
VOCAB_PATH = "data/all news.txt"
SEQUENCE_LENGTH = 16
HIDDEN_SIZE = 256
DATA_SIZE = 20000
DATA_PATH = "data/all news.txt"
import numpy as np
#
#
# temp = [1,2,3,4]
# temp = temp + [BATCH_SIZE] * 5
#
# print(temp)
# for i in range(0, DATA_SIZE, hp.BATCH_SIZE):
#     yield data[i:i + hp.BATCH_SIZE]
#
#
#     def splitAndProcess(DATA_PATH):
#         gen = readLargeFile(DATA_PATH)
#         for i in range(0, DATA_SIZE, hp.BATCH_SIZE):
#             data = preprocess(next(gen), SEQ_LEN)
#             # TODO : ?
#             random.shuffle(data)
#             yield data


# list = [[1,2,3],[2,3,4],[5,6,7]]
# print(list)
# list = np.array(list)
# print (list)
#
#
# from batcher import Batcher
# from preprocess import Preprocesser
# import tensorflow as tf
# import json
# import numpy as np
#
# #batch = Batcher().get_batch()
#
# with open('hyperparams.json') as my_json:
#     parsed = json.load(my_json)
#
# SENT_LEN  = parsed["preprocesser"]["nn_input_size"]
# BATCH_SIZE= parsed["batcher"]["batch_size"]
# EYE_SIZE = parsed["preprocesser"]["vocab_size"]
# EMBEDDING_SIZE = 128
# HID_SIZE = 256
# ATTENTION = True
# BEAMER_ACTIVE = True
# BEAM_SIZE = 3
#
# CONCAT_SIZE =  EMBEDDING_SIZE+HID_SIZE
# DECODER_CONCAT_SIZE =  EMBEDDING_SIZE+HID_SIZE if ATTENTION is False else EMBEDDING_SIZE+HID_SIZE+SENT_LEN
#
# class Autoencoder():
#
#     def __init__(self):
#         self.batch = Batcher().get_batch()
#         self.preper= Preprocesser()
#         self.sess  = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1))
#         self.input_placeholder, self.loss, self.summary_merge, self.generateds = self.create_graph()
#         self.train_step = 0
#         self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)
#         self.optimizer = tf.train.AdamOptimizer()
#         self.gradient_limit = tf.constant(6.0, dtype=tf.float32, name='grad_limit')
#         grads_and_vars = self.optimizer.compute_gradients(self.loss)
#
#         clipped_grads_and_vars = []
#         for grad, var in grads_and_vars:
#             clipped_grad = tf.clip_by_value(grad, -self.gradient_limit, self.gradient_limit)
#             clipped_grads_and_vars.append((clipped_grad, var))
#         self.apply_grads = self.optimizer.apply_gradients(clipped_grads_and_vars)
#         self.sess.run(tf.global_variables_initializer())
#     @staticmethod
#     def create_placeholders():
#
#         input  = tf.placeholder(shape=[BATCH_SIZE,SENT_LEN],dtype=tf.int64)
#         return input
#
#     def create_graph(self):
#
#         input = self.create_placeholders()
#
#         with tf.name_scope("ENCODER"):
#             W_embedd = tf.Variable(tf.random_normal([EMBEDDING_SIZE,EYE_SIZE]),name="W_embedd")
#             b_embedd = tf.Variable(tf.random_normal([EMBEDDING_SIZE, 1]), name="b_embedd")
#
#             W_forget = tf.Variable(tf.random_normal([HID_SIZE,CONCAT_SIZE]),name="W_forget")
#             b_forget = tf.Variable(tf.zeros([HID_SIZE,1]),name="b_forget")
#
#             W_input = tf.Variable(tf.random_normal([HID_SIZE,CONCAT_SIZE]),name="W_input")
#             b_input = tf.Variable(tf.zeros([HID_SIZE,1]),name="b_input")
#
#             W_Cell = tf.Variable(tf.random_normal([HID_SIZE, CONCAT_SIZE]),name="W_cell")
#             b_Cell = tf.Variable(tf.zeros([HID_SIZE,1]),name="b_cell")
#
#             W_output = tf.Variable(tf.random_normal([HID_SIZE, CONCAT_SIZE]),name="W_output")
#             b_output = tf.Variable(tf.zeros([HID_SIZE,1]),name="b_output")
#
#             prev_hidden_st = tf.Variable(tf.zeros([HID_SIZE, BATCH_SIZE]),dtype=tf.float32,name="prev_hidden", trainable=False)
#             prev_cell_st   = tf.Variable(tf.zeros([HID_SIZE, BATCH_SIZE]),dtype=tf.float32,name="prev_cell_stat", trainable=False)
#
#             eh_list = []
#             for t,token in enumerate(tf.split(input,SENT_LEN,axis=1)):  # loop on words in sentences
#
#                 embedded_word = tf.sigmoid(tf.matmul(W_embedd,tf.reshape(tf.one_hot(token,EYE_SIZE),[EYE_SIZE,BATCH_SIZE]))+b_embedd)
#                 Z_conc = tf.concat([embedded_word,prev_hidden_st],0)   # Z in diagram
#
#                 forget_t = tf.sigmoid(tf.matmul(W_forget,Z_conc)+b_forget)  # (256,1)
#                 input_t  = tf.sigmoid(tf.matmul(W_input,Z_conc)+b_input)    # (256,1)
#                 cell_t   = tf.tanh(tf.matmul(W_Cell,Z_conc)+b_Cell)         # (256,1)
#                 output_t = tf.sigmoid(tf.matmul(W_output,Z_conc)+b_output)  # (256,1)
#
#                 cell_state  = tf.add(tf.multiply(prev_cell_st,forget_t),tf.multiply(input_t,cell_t))
#                 hidden_state= tf.multiply(output_t,tf.tanh(cell_state))
#                 prev_cell_st = cell_state
#                 prev_hidden_st = hidden_state
#                 eh_list.append(hidden_state)
#
#             final_encoded_state = prev_hidden_st
#
#         ### DECODER
#         with tf.name_scope("DECODER"):
#             start_seq_embedd = tf.constant(0.0,shape=[EMBEDDING_SIZE,BATCH_SIZE],dtype=tf.float32,name="Start_Sequence")
#             prev_hidden_st = final_encoded_state #final encoded go there;
#             prev_cell_st = tf.Variable(tf.zeros([HID_SIZE, BATCH_SIZE]), dtype=tf.float32, name="prev_cell_stat",trainable=False)
#
#             W_embedd_dec = tf.Variable(tf.random_normal([EMBEDDING_SIZE, EYE_SIZE]), name="W_embedd_decoder")
#             b_embedd_dec = tf.Variable(tf.random_normal([EMBEDDING_SIZE, 1]), name="b_embedd_decoder")
#
#             W_forget_dec = tf.Variable(tf.random_normal([HID_SIZE, DECODER_CONCAT_SIZE]), name="W_forget_decoder")
#             b_forget_dec = tf.Variable(tf.zeros([HID_SIZE, 1]), name="b_forget_decoder")
#
#             W_input_dec = tf.Variable(tf.random_normal([HID_SIZE, DECODER_CONCAT_SIZE]), name="W_input_decoder")
#             b_input_dec = tf.Variable(tf.zeros([HID_SIZE, 1]), name="b_input_decoder")
#
#             W_Cell_dec = tf.Variable(tf.random_normal([HID_SIZE, DECODER_CONCAT_SIZE]), name="W_cell_decoder")
#             b_Cell_dec = tf.Variable(tf.zeros([HID_SIZE, 1]), name="b_cell_decoder")
#
#             W_output_dec = tf.Variable(tf.random_normal([HID_SIZE, DECODER_CONCAT_SIZE]), name="W_output_decoder")
#             b_output_dec = tf.Variable(tf.zeros([HID_SIZE, 1]), name="b_output_decoder")
#
#             W_prediction = tf.Variable(tf.random_normal([EYE_SIZE,HID_SIZE]),name="W_prediction")
#             b_prediction = tf.Variable(tf.zeros([EYE_SIZE,1]),name="b_prediction")
#
#             Generated_sentences = []
#             #Start with propagating the start word
#             if ATTENTION is False:
#                 Z_conc = tf.concat([start_seq_embedd, prev_hidden_st], 0)  # Z in diagram
#             else:
#                 context_vector = tf.stack([tf.matmul(tf.transpose(tf.Variable(tf.zeros([HID_SIZE, 1]),dtype=tf.float32)),eh_x) for eh_x in eh_list])
#                 Z_conc = tf.concat([start_seq_embedd,tf.reshape(context_vector,[SENT_LEN,BATCH_SIZE]),prev_hidden_st],0)
#                 print("sing itt!")
#
#             forget_t = tf.sigmoid(tf.matmul(W_forget_dec, Z_conc) + b_forget_dec)  # (256,1)
#             input_t = tf.sigmoid(tf.matmul(W_input_dec, Z_conc) + b_input_dec)  # (256,1)
#             cell_t = tf.tanh(tf.matmul(W_Cell_dec, Z_conc) + b_Cell_dec)  # (256,1)
#             output_t = tf.sigmoid(tf.matmul(W_output_dec, Z_conc) + b_output_dec)  # (256,1)
#
#             cell_state = tf.add(tf.multiply(prev_cell_st, forget_t), tf.multiply(input_t, cell_t))
#             hidden_state = tf.multiply(output_t, tf.tanh(cell_state))
#
#             logits = tf.add(tf.matmul(W_prediction, hidden_state),b_prediction)
#             normalized_probs = tf.nn.softmax(logits,axis=0) #apply softmax to axis with 250.000 elems
#
#             prev_cell_st = cell_state
#             prev_hidden_st = hidden_state
#             #here it comes a loop that generates tokens
#             for t in range(SENT_LEN):              # loop words times in sentences
#                 embedded_word = tf.sigmoid(tf.matmul(W_embedd_dec, tf.reshape(tf.one_hot(tf.argmax(normalized_probs), EYE_SIZE), [EYE_SIZE, BATCH_SIZE])) + b_embedd_dec)
#
#                 if ATTENTION is False:
#                     Z_conc = tf.concat([embedded_word, prev_hidden_st], 0)  # Z in diagram
#                 else:
#                     context_vector = tf.stack([tf.matmul(tf.transpose(tf.Variable(tf.zeros([HID_SIZE, 1]), dtype=tf.float32)), eh_x) for eh_x in eh_list])
#                     Z_conc = tf.concat([embedded_word, tf.reshape(context_vector, [SENT_LEN, BATCH_SIZE]), prev_hidden_st], 0)
#
#                 forget_t = tf.sigmoid(tf.matmul(W_forget_dec, Z_conc) + b_forget_dec)  # (256,1)
#                 input_t = tf.sigmoid(tf.matmul(W_input_dec, Z_conc) + b_input_dec)  # (256,1)
#                 cell_t = tf.tanh(tf.matmul(W_Cell_dec, Z_conc) + b_Cell_dec)  # (256,1)
#                 output_t = tf.sigmoid(tf.matmul(W_output_dec, Z_conc) + b_output_dec)  # (256,1)
#
#                 cell_state = tf.add(tf.multiply(prev_cell_st, forget_t), tf.multiply(input_t, cell_t))
#                 hidden_state = tf.multiply(output_t, tf.tanh(cell_state))
#
#                 logits = tf.add(tf.matmul(W_prediction, hidden_state), b_prediction)
#                 normalized_probs = tf.nn.softmax(logits, axis=0)  # apply softmax to axis with vocab_size elems
#
#                 Generated_sentences.append(normalized_probs)
#                 prev_cell_st = cell_state
#                 prev_hidden_st = hidden_state
#             Generated_sentences = tf.stack(Generated_sentences)
#
#         with tf.name_scope("BEAMER"):
#             prev_hidden_st = final_encoded_state  # final encoded go there;
#             prev_cell_st = tf.Variable(tf.zeros([HID_SIZE, BATCH_SIZE]), dtype=tf.float32, name="prev_cell_stat",trainable=False)
#             if ATTENTION is False:  #update z_conc with start sequence
#                 Z_conc = tf.concat([start_seq_embedd, prev_hidden_st], 0)  # Z in diagram
#             else:
#                 context_vector = tf.stack([tf.matmul(tf.transpose(tf.Variable(tf.zeros([HID_SIZE, 1]),dtype=tf.float32)),eh_x) for eh_x in eh_list])
#                 Z_conc = tf.concat([start_seq_embedd,tf.reshape(context_vector,[SENT_LEN,BATCH_SIZE]),prev_hidden_st],0)
#                 print("sing itt!")
#             forget_t = tf.sigmoid(tf.matmul(W_forget_dec, Z_conc) + b_forget_dec)  # (256,1)
#             input_t = tf.sigmoid(tf.matmul(W_input_dec, Z_conc) + b_input_dec)  # (256,1)
#             cell_t = tf.tanh(tf.matmul(W_Cell_dec, Z_conc) + b_Cell_dec)  # (256,1)
#             output_t = tf.sigmoid(tf.matmul(W_output_dec, Z_conc) + b_output_dec)  # (256,1)
#
#             cell_state = tf.add(tf.multiply(prev_cell_st, forget_t), tf.multiply(input_t, cell_t))
#             hidden_state = tf.multiply(output_t, tf.tanh(cell_state))
#
#             logits = tf.add(tf.matmul(W_prediction, hidden_state), b_prediction)
#             normalized_probs = tf.nn.softmax(logits, axis=0)  # apply softmax to axis with 250.000 elems
#             prev_cell_st_list = [cell_state for i in range(BEAM_SIZE)] # every prevs should be kept in list since we'll use them in next step
#             prev_hidden_st_list = [hidden_state for i in range(BEAM_SIZE)]
#             top_vals_list = []
#             top_indices_list = []
#
#             top_indices = [0 for i in range(BATCH_SIZE)]
#             top_vals    = [0 for i in range(BATCH_SIZE)]
#             for cnt, sent_prob_vector in enumerate(tf.split(normalized_probs,BATCH_SIZE,axis=1)): #find top 3 probs for every sentence
#                 top_vals[cnt], top_indices[cnt] = tf.nn.top_k(tf.reshape(sent_prob_vector, [EYE_SIZE]), k=3)
#
#             top_vals_list.append(top_vals)
#             top_indices_list.append(top_indices)
#
#             top_stacked = tf.stack(top_indices) # 32x3 tensor includes top 3 index of every sentence
#             for t in range(SENT_LEN):              # loop words times in sentences
#                 normalized_probs = []
#                 for cnt, top_elem in enumerate(tf.split(top_stacked,BEAM_SIZE,1)):
#                     embedded_word = tf.sigmoid(tf.matmul(W_embedd_dec, tf.reshape(tf.one_hot(top_elem, EYE_SIZE), [EYE_SIZE, BATCH_SIZE])) + b_embedd_dec)
#
#                     if ATTENTION is False:
#                         Z_conc = tf.concat([embedded_word, prev_hidden_st_list[cnt]], 0)  # Z in diagram
#                     else:
#                         context_vector = tf.stack([tf.matmul(tf.transpose(tf.Variable(tf.zeros([HID_SIZE, 1]), dtype=tf.float32)), eh_x) for eh_x in eh_list])
#                         Z_conc = tf.concat([embedded_word, tf.reshape(context_vector, [SENT_LEN, BATCH_SIZE]), prev_hidden_st_list[cnt]], 0)
#
#                     forget_t = tf.sigmoid(tf.matmul(W_forget_dec, Z_conc) + b_forget_dec)  # (256,1)
#                     input_t = tf.sigmoid(tf.matmul(W_input_dec, Z_conc) + b_input_dec)  # (256,1)
#                     cell_t = tf.tanh(tf.matmul(W_Cell_dec, Z_conc) + b_Cell_dec)  # (256,1)
#                     output_t = tf.sigmoid(tf.matmul(W_output_dec, Z_conc) + b_output_dec)  # (256,1)
#
#                     cell_state = tf.add(tf.multiply(prev_cell_st_list[cnt], forget_t), tf.multiply(input_t, cell_t))
#                     hidden_state = tf.multiply(output_t, tf.tanh(cell_state))
#
#                     logits = tf.add(tf.matmul(W_prediction, hidden_state), b_prediction)
#                     normalized_probs.append(tf.nn.softmax(logits, axis=0))  #append prob vector of all three candidates
#                     prev_cell_st_list[cnt]   = cell_state
#                     prev_hidden_st_list[cnt] = hidden_state
#
#                 normalized_probs = tf.concat(normalized_probs,axis=0) # stack them and get a shape(3*V,batch_size) tensor
#
#                 for cnt, sent_prob_vector in enumerate(tf.split(normalized_probs, BATCH_SIZE, axis=1)):  # find top 3 probs for every sentence
#                     top_vals[cnt], top_indices[cnt] = tf.nn.top_k(tf.reshape(sent_prob_vector, [EYE_SIZE*BEAM_SIZE]), k=3)
#                 top_stacked = tf.stack(top_indices)  # 32x3 tensor includes top 3 index of every sentence
#                 top_vals_list.append(tf.stack(top_vals))
#                 top_indices_list.append(tf.stack(top_indices))
#
#
#             print("la")
#
#         Target_sentences = []
#         for t,token in enumerate(tf.split(input,SENT_LEN,axis=1)):  # loop on words in sentences
#             Target_sentences.append(tf.reshape(tf.one_hot(token,EYE_SIZE),[EYE_SIZE,BATCH_SIZE]))
#
#         Target_sentences = tf.stack(Target_sentences)
#
#         Target_sentences = tf.reshape(Target_sentences,[BATCH_SIZE,SENT_LEN,EYE_SIZE])
#         Generated_sentences = tf.reshape(Generated_sentences, [BATCH_SIZE, SENT_LEN, EYE_SIZE])
#
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Target_sentences, logits=Generated_sentences),axis=1)
#         tf.summary.scalar('lol_loss',tf.reduce_mean(loss))
#         merged = tf.summary.merge_all()
#         return [input,loss,merged,Generated_sentences]
#
#     def run_graph(self,batch):
#         input,loss_loses,mergy,_ = self.create_graph()
#
#         with tf.Session() as sess:
#             writer = tf.summary.FileWriter('./graphs', sess.graph)
#             init = tf.global_variables_initializer()
#             sess.run(init)
#             dicty = {input: batch.reshape(32,30)}
#
#             calced_loss = sess.run(loss_loses,feed_dict=dicty)
#             print("typeof: ",type(calced_loss))
#             print("ov my loss shape: ", calced_loss.shape)
#             print(calced_loss)
#
#     def train_on_batch(self,batch):
#
#         dicty = {self.input_placeholder: batch.reshape(32,30)}
#
#         my_summary, calced_loss, _, gen = self.sess.run([self.summary_merge, self.loss, self.apply_grads,self.generateds], feed_dict=dicty)
#         self.train_writer.add_summary(my_summary,self.train_step)
#         print(np.average(np.array(calced_loss)))
#         print("Step: ",self.train_step)
#         deced = np.argmax(gen,axis=2).reshape(BATCH_SIZE,SENT_LEN)
#         humanic_sent = [self.preper.index2word.get(i) for i in deced[0]]
#         input_sent = [self.preper.index2word.get(i) for i in batch[0]]
#         print(humanic_sent)
#         print(input_sent)
#         self.train_step += 1
#
#     def continuous_train(self):
#
#         while True:
#             self.train_on_batch(next(self.batch))
#             #print(next(self.batch)[0])
# if __name__ == '__main__':
#
#     my_neuro = Autoencoder()
#     my_neuro.continuous_train()