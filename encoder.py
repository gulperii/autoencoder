import tensorflow as tf
import hyperparameter as hp
import LSTM
import batch
import preprocess

class Encoder(object):

    def __init__(self):
        self.preprocess = preprocess.Preprocess()
        self.batcher = batch.Batcher(self.preprocess)
        self.batchGen = self.batcher.batchGen
        self.embeddingMatrix = self.preprocess.embeddingMatrix


    def initializeStates(self):
        prevHiddenState = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
        prevCellState = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
        return prevHiddenState, prevCellState

    def runGraph(self):
        encoderCell = LSTM.Cell()

        hiddenState = encoderCell.calculateStates()
        prevHiddenState, prevCellState = self.initializeStates()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        init = tf.
        sess.run(init)

        hasMoreBatch = True
        while hasMoreBatch:
            try:
                currentBatch = next(self.batchGen)
                embeddedInput = tf.nn.embedding_lookup(self.embeddingMatrix, currentBatch)
                for sentence in embeddedInput:
                    feedDict = {encoderCell.input: sentence, encoderCell.prevCellState: prevCellState,
                                encoderCell.prevHiddenState: prevHiddenState}
                    prevHiddenStateForSample = sess.run([hiddenState], feed_dict=feedDict)
                    print(prevHiddenStateForSample)
                    hasMoreBatch = False

            except:
                hasMoreBatch = False
                print("Run out of batches !")


encoder = Encoder()
encoder.runGraph()