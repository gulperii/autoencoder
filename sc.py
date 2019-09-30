import json
import random
import re
from string import punctuation
import tensorflow as tf
import numpy as np
# from scipy import stats
import hyperparameter as hp
from tensorflow.contrib.rnn import LSTMCell
# all words with frequencies in vocab data

# # before batch
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
import math


def answer(height, nodes):
    root_node = math.pow(2, height) - 1
    parent = -1
    return [int(solution_helper(height, node, root_node, parent)) for node in nodes]


def solution_helper(height, target_node, current_root, parent):
    left_most_node = math.pow(2, height) - 1
    left_child = current_root - ((left_most_node + 1) / 2)
    right_child = current_root - 1

    if target_node == current_root:
        return parent

    elif target_node <= left_child:
        return solution_helper(height - 1, target_node, left_child, current_root)

    else:
        return solution_helper(height - 1, target_node, right_child, current_root)


from fractions import Fraction


def solution(pegs):
    is_even = True if len(pegs) % 2 == 0 else False
    distances = [(-1) * (pegs[i] - pegs[i + 1]) for i in range(len(pegs) - 1)]
    flip = 1
    sum = 0
    for dist in distances:
        sum += (flip * dist)
        flip *= -1

    possible_r = Fraction(2 * (float(sum) / 3 if is_even else sum)).limit_denominator()

    if is_valid_r(possible_r, distances):
        return [possible_r.numerator, possible_r.denominator]
    return [-1, -1]


def is_valid_r(cand_r, distances):
    current_r = cand_r
    for dist in distances:
        next_r = dist - current_r
        if next_r < 1 or current_r < 1:
            return False
        current_r = next_r
    return True


import heapq

MAX_INT = 1000


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
    if shortest == row_length + col_length:
        return shortest
    for wall in wall_list:
        broken_wall = initial_distances[wall] - 99
        if broken_wall > initial_shortest_path:
            continue
        wall_i, wall_j = index_to_coor(wall)
        sliced_matrix = [matrix[i][wall_j:] for i in range(wall_i, col_length)]
        sliced_matrix[0][0] = 0
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


matrix = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1],
          [0, 0, 0, 0, 0, 0]]
matrx = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
print(cevap(matrx))

# print(shortest_path([[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 0]]))
