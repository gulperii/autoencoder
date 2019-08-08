def run_RNN(self):
    hidden_state = self.create_RNN()
    previous_state = np.zeros((self.hidden_size, 1))
    cell_state = np.zeros((self.hidden_size, 1))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    has_more_batch = True
    while has_more_batch:
        try:
            current_batch = next(self.form_batch)
            for row_number, row in enumerate(current_batch):
                embedded_word_vectors = np.zeros((self.max_seq_length, self.embedding_size), dtype=np.float32)
                for i, index in enumerate(row):
                    embedded_word_vectors[i] = self.embedding_matrix[index]
                feed_dict = {self.inputs: np.transpose(embedded_word_vectors),
                             self.prev_hidden_state_as_input: previous_state, self.cell_prev_state: cell_state}
                previous_hidden_state_for_sample = sess.run([hidden_state], feed_dict=feed_dict)

                print("Row number {}".format(row_number + 1))
                print(previous_hidden_state_for_sample)
                print("==============================\n")

                has_more_batch = False
        except:
            has_more_batch = False
            logging.info("No more batches")