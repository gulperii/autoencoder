import tensorflow as tf
import hyperparameter as hp


class Cell(object):
    def __init__(self):
        self.input = tf.placeholder(shape=[hp.EMBEDDING_SIZE, hp.SEQUENCE_LENGTH], dtype=tf.int32, name='input')
        self.prevCellState = tf.placeholder(shape=(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE + hp.EMBEDDING_SIZE), dtype=tf.int32)
        self.prevHiddenState = tf.placeholder(shape=(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE + hp.EMBEDDING_SIZE),
                                              dtype=tf.int32)
        self.bInput, self.bOutput, self.bCell, self.bForget = self.initializeBias()
        self.wInput, self.wOutput, self.wCell, self.wForget = self.initializeWeight()
        # self.prevCellState, self.prevHiddenState = self.initializeStates()

    def initializeBias(self):
        bInput = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
        bOutput = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
        bCell = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
        bForget = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
        return bInput, bOutput, bCell, bForget

    # def initializeStates(self):
    #     prevHiddenState = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
    #     prevCellState = tf.random_normal(shape=(hp.HIDDEN_SIZE, 1))
    #     return prevCellState, prevHiddenState

    def initializeWeight(self):
        wInput = tf.random_normal(shape=(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE + hp.EMBEDDING_SIZE))
        wOutput = tf.random_normal(shape=(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE + hp.EMBEDDING_SIZE))
        wCell = tf.random_normal(shape=(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE + hp.EMBEDDING_SIZE))
        wForget = tf.random_normal(shape=(hp.HIDDEN_SIZE, hp.HIDDEN_SIZE + hp.EMBEDDING_SIZE))
        return wInput, wOutput, wCell, wForget

    def calculateStates(self):
        for vector in self.input:
            Z = tf.concat([vector, self.prevHiddenState], axis=0)
            sForget = tf.sigmoid(tf.math.add(tf.matmul(self.wForget, Z), self.wForget))
            sInput = tf.sigmoid(tf.math.add(tf.matmul(self.wInput, Z), self.bForget))
            sDCell = tf.tanh(tf.math.add(tf.matmul(self.wCell, Z), self.bCell))
            sCell = tf.math.add(tf.tensordot(sForget, self.prevCellState), tf.tensordot(sInput, sDCell))
            sOutput = tf.sigmoid(tf.math.add(tf.matmul(self.wOutput, Z), self.bOutput))
            sHidden = tf.tensordot(sOutput, tf.tanh(sCell))
            self.prevCellState = sCell
            self.prevHiddenState = sHidden
            print("calculated states")
        return sHidden
