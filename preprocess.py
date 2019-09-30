import re

# import scipy
import tensorflow as tf

import hyperparameter as hp

#TODO: blank

class Preprocess(object):
    alphabets = "([A-Za-z])"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    PAD_TOKEN = '[PAD]'
    UNKNOWN_TOKEN = '[UNK]'
    STOP_TOKEN = '[EOS]'

    def __init__(self):

        self.sentenceLengths = []
        self.sequenceLength = 0
        self.frequencyThreshold = 50
        self.dataSize = 0
        self.vocab = self.createVocab(hp.VOCAB_PATH)
        self.vocabToIndex = {word: index for index, word in enumerate(self.vocab)}
        self.indexToVocab = {index: word for index, word in enumerate(self.vocab)}
        self.embeddingMatrix = self.createEmbeddingMatrix()
        self.dataGen = self.readLargeFile(hp.DATA_PATH)

    def preprocess(self, sentences):

        #sentences = self.splitSentences(text)
        # words = re.findall(r"[\w']+", text)
        processedSentences = []
        self.dataSize = len(sentences)
        # snippet numpy
        for sentence in sentences:
            sentence = self.removePunct(sentence)
            sentence = sentence.split()
            self.sentenceLengths.append(len(sentence))
            temp = [word if word in self.vocabToIndex else self.UNKNOWN_TOKEN for word in sentence]
            temp.append(self.STOP_TOKEN)
            padCount = max(0, hp.SEQUENCE_LENGTH - len(temp))
            temp = temp + [self.PAD_TOKEN] * padCount
            temp = temp[:hp.SEQUENCE_LENGTH]
            temp = " ".join(temp)
            processedSentences.append(temp)
        return processedSentences

    def splitSentences(self, text):
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub("\s" + self.alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]" + self.alphabets + "[.]",
                      "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(self.alphabets + "[.]" + self.alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + self.alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace("...", ".")
        text = text.replace(":", ".")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def createVocab(self, path):
        dictWithFreq = {}
        vocab = []
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        text = self.removePunct(text)
        text = text.lower().split()

        for i in range(len(text)):
            dictWithFreq[text[i]] = dictWithFreq.get(text[i], 0) + 1

        # with open(VOCAB_PATH, "w", encoding="utf-8") as file:
        #     json.dump(dictWithFreq, file)

        vocab.append(self.PAD_TOKEN)
        vocab.append(self.UNKNOWN_TOKEN)
        vocab.append(self.STOP_TOKEN)
        for key in dictWithFreq:
            if dictWithFreq[key] > self.frequencyThreshold:
                vocab.append(key)
        print("created vocab")
        return vocab

    def removePunct(self, text):
        text=text.strip("\'")
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'[,{}@_*>()\\#%+=\[\]]', '', text)
        text = re.sub('a0', '', text)
        text = re.sub('\'91', ' ', text)
        text = re.sub('\'92', ' ', text)
        text = re.sub('\'93', ' ', text)
        text = re.sub('\'94', ' ', text)
        text = re.sub('\.', ' ', text)
        text = re.sub('\!', ' ', text)
        text = re.sub('-', ' ', text)
        text = re.sub('\?', ' ', text)
        text = re.sub(' +', ' ', text)
        return text

    def indexData(self, data):
        data = self.preprocess(data)
        indexedData = [[self.vocabToIndex[word] for word in sentence.split()] for sentence in data]
        return indexedData

    # def getSeqLen(self):
    #     z = np.abs(scipy.stats.zscore(self.sentenceLengths))
    #     sentenceLengthsO = self.sentenceLengths[(z < 3).all(axis=1)]
    #     self.sequenceLength = sentenceLengthsO.mean()

    def readLargeFile(self, DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            while True:
                data = []
                for i in range(hp.BATCH_SIZE):
                    data.append(f.readline())
                if not data:
                    break
                yield data

    def createEmbeddingMatrix(self):
        embeddingMatrix = tf.Variable(tf.random_uniform([len(self.vocab), hp.EMBEDDING_SIZE], -1.0, 1.0))
        return embeddingMatrix

