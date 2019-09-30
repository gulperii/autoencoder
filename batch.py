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



def cevap(matrix):
    row_length = len(matrix[0])
    col_length = len(matrix)
    coor_to_index = lambda i, j: i * row_length + j
    index_to_coor = lambda node: (int(node / row_length), node % row_length)
    adj_list = {}

    wall_list = []
    for i in range(col_length):
        for j in range(row_length):
            index = coor_to_index(i, j)
            if matrix[i][j] == 1: wall_list.append(index)
            adj_list[index] = neighbours_and_weights(index, matrix)

    initial_distances = shortest_path(matrix, 1)
    initial_shortest_path = initial_distances[-1]
    shortest = initial_shortest_path
    for wall in wall_list:
        broken_wall = initial_distances[wall] - 99
        if broken_wall > initial_shortest_path:
            continue
        wall_i, wall_j = index_to_coor(wall)
        sliced_matrix = [matrix[i][wall_j:] for i in range(wall_i,col_length)]
        new_attempt = shortest_path(sliced_matrix, broken_wall)[-1]
        if new_attempt < shortest:
            shortest = new_attempt

    return shortest


def shortest_path(matrix, initial_weight):
    row_length = len(matrix[0])
    col_length = len(matrix)
    size = row_length * col_length
    # index = lambda i, j: i * row_length + j

    visited = size * [False]
    distances = size * [MAX_INT]

    # min heap (weight,index)
    pq = [(initial_weight, 0)]
    distances[0] = initial_weight
    heapq.heapify(pq)

    while pq:
        cur_dist, cur_node = heapq.heappop(pq)
        if not visited[cur_node]:
            visited[cur_node] = True
            adj_list = neighbours_and_weights(cur_node, matrix)
            for node, weight in adj_list.items():
                if visited[node]: continue
                old_weight = distances[node]
                new_weight = cur_dist + weight
                if new_weight < old_weight:
                    distances[node] = new_weight
                heapq.heappush(pq, (distances[node], node))

    return distances


def neighbours_and_weights(index, matrix):
    index_to_coor = lambda node: (int(node / row_length), node % row_length)
    coor_to_index = lambda i, j: i * row_length + j
    weight = lambda x: 100 if x == 1 else 1

    row_length = len(matrix[0])
    col_length = len(matrix)

    i, j = index_to_coor(index)
    is_one = matrix[i][j]
    adj = {}
    if i > 0: adj[coor_to_index(i - 1, j)] = 100 if is_one else weight(matrix[i - 1][j])
    if j > 0: adj[coor_to_index(i, j - 1)] = 100 if is_one else weight(matrix[i][j - 1])
    if i + 1 < col_length: adj[coor_to_index(i + 1, j)] = 100 if is_one else weight(matrix[i + 1][j])
    if j + 1 < row_length: adj[coor_to_index(i, j + 1)] = 100 if is_one else weight(matrix[i][j + 1])

    return adj
matrix = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]
matrx = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
print(cevap(matrix))

# print(shortest_path([[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 0]]))
