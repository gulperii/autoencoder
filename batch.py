import numpy as np
import hyperparameter as hp
import preprocess

class Batch(object):
    def __init__(self, preprocess):
        self.preprocess = preprocess
        self.batchMatrix = np.zeros((hp.BATCH_SIZE, hp.SEQUENCE_LENGTH), dtype=np.int32)

    def createBatchMatrix(self, data):
        indexedData = self.preprocess.indexData(data)
        self.batchMatrix = np.array(indexedData)
        print("crated batch matrix")
        return self.batchMatrix


class Batcher(object):
    def __init__(self,preprocess):
        self.preprocess = preprocess
        self.batch = Batch(self.preprocess)
        self.dataGen = self.preprocess.dataGen
        self.batchGen = self.batchGenerator()

    def batchGenerator(self):
        for i in range(0, hp.DATA_SIZE, hp.BATCH_SIZE):
            data = next(self.dataGen)
            indexedBatch = self.batch.createBatchMatrix(data)
            yield indexedBatch

