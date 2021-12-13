import random
import copy
from random import shuffle
import pickle
import os
import numpy as np


def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n + 1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n + 1: 2 * n + 1]]

        return piece_type, previous_board, board


def readOutput(path="output.txt"):
    with open(path, 'r') as f:
        position = f.readline().strip().split(',')

        if position[0] == "PASS":
            return "PASS", -1, -1

        x = int(position[0])
        y = int(position[1])

    return "MOVE", x, y


# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ProjectName: HW2
# FileName: write
# Description:
# TodoList:

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)


def writePass(path="output.txt"):
    with open(path, 'w') as f:
        f.write("PASS")


def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    res = ""
    res += str(piece_type) + "\n"
    for item in previous_board:
        res += "".join([str(x) for x in item])
        res += "\n"

    for item in board:
        res += "".join([str(x) for x in item])
        res += "\n"

    with open(path, 'w') as f:
        f.write(res[:-1])


# def hash(board):
#     # zoborist hash
#     h = 0
#     for i in board:
#         for j in i:
#             h = h ^ j
#     return h

def rolling_window(a, shape):
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


def euler(board, piece_type):
    b = np.array(board)
    q1b = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    q1w = np.array([[[2, 0], [0, 0]], [[0, 2], [0, 0]], [[0, 0], [2, 0]], [[0, 0], [0, 2]]])

    q2b = np.array([[[0, 1], [1, 1]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[1, 1], [1, 0]]])
    q2w = np.array([[[0, 2], [2, 2]], [[2, 0], [2, 2]], [[2, 2], [0, 2]], [[2, 2], [2, 0]]])

    qdb = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]])
    qdw = np.array([[[[2, 0], [2, 0]], [[0, 2], [0, 2]]]])

    if piece_type == 1:
        q1 = q2 = qd = 0
        for i in rolling_window(b, (2, 2)):
            for j in i:
                if j in q1w:
                    q1 += 1
                elif j in q2w:
                    q2 += 1
                elif j in qdw:
                    qd += 1
    else:
        q1 = q2 = qd = 0
        for i in rolling_window(b, (2, 2)):
            for j in i:
                if j in q1b:
                    q1 += 1
                elif j in q2b:
                    q2 += 1
                elif j in qdb:
                    qd += 1
    return (q1 - q2 + 2 * qd) / 4


def eval2(state, piece_type):
    b = be = w = we = 0
    for row in state:
        b += row.count(1)
        w += row.count(2)

    for i in range(1, len(state) - 1):
        if state[i][0] == 1:
            be += 1
        if state[i][-1] == 1:
            be += 1
        if state[i][0] == 2:
            we += 1
        if state[i][-1] == 2:
            we += 1
    be += state[0].count(1) + state[-1].count(1)
    we += state[0].count(2) + state[-1].count(2)

    l = len(all_liberties(state, piece_type)) - len(all_liberties(state, 3 - piece_type))
    e = euler(state, piece_type)

    if piece_type == 1:
        return l + b + be - 2*(w - we)
        # return l + (-4 * e) + 5 * b + be
    elif piece_type == 2:
        # return l + (-4 * e) + 5 * w + we
        return l + 5 * w + we - 2*(b - be)


def eval(board, piece_type):
    player_stones, adversary_stones, player_liberties, adversary_liberties = 0, 0, 0, 0
    player, pb, b = readInput(5)
    for x in range(5):
        for y in range(5):
            if board[x][y] != 0:
                if board[x][y] == player:
                    player_stones = player_stones + 1
                else:
                    adversary_stones = adversary_stones + 1
    player_liberties = len(all_liberties(board, player))
    adversary_liberties = len(all_liberties(board, 3 - player))
    if player == piece_type:
        return (player_stones + 1 * player_liberties) - (adversary_stones + 2 * adversary_liberties)
    else:
        return - (player_stones + 1 * player_liberties) + (adversary_stones + 2 * adversary_liberties)


def all_moves(board):
    moves = []
    for x in range(5):
        for y in range(5):
            if board[x][y] == 0:
                moves.append((x, y))
    return moves


def all_connected_friendly_neighbors(x, y, piece_type, board):
    not_visited = [(x, y)]
    connected_neighbors = []
    while not_visited:
        x, y = not_visited.pop()
        connected_neighbors.append((x, y))
        neighbors_friendly = friendly_neighbors(board, x, y, piece_type)
        for friend in neighbors_friendly:
            if friend not in not_visited and friend not in connected_neighbors:
                not_visited.append(friend)
    return connected_neighbors


def liberty(board, x, y, piece_type):
    connected_neighbors = all_connected_friendly_neighbors(x, y, piece_type, board)
    liberties = []
    for x, y in connected_neighbors:
        neighbors = all_neighbors(x, y)
        for x, y in neighbors:
            if board[x][y] == 0 and (x, y) not in liberties:
                liberties.append((x, y))
    return liberties


def all_neighbors(x, y):
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < 4:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < 4:
        neighbors.append((x, y + 1))
    return neighbors


def friendly_neighbors(board, x, y, piece_type):
    current_friendly_neighbors = []
    neighbors = all_neighbors(x, y)
    for x, y in neighbors:
        if board[x][y] == piece_type and (x, y) not in current_friendly_neighbors:
            current_friendly_neighbors.append((x, y))
    return current_friendly_neighbors


def find_dead_stones(piece_type, board):
    dead_stones = []
    for x in range(5):
        for y in range(5):
            if board[x][y] == piece_type:
                liberties = liberty(board, x, y, piece_type)
                if not liberties and (x, y) not in dead_stones:
                    dead_stones.append((x, y))
    return dead_stones


def remove_dead_stones(board, dead_stones):
    for x, y in dead_stones:
        board[x][y] = 0
    return board


def same_board(board, prev_board):
    for x in range(5):
        for y in range(5):
            if board[x][y] != prev_board[x][y]:
                return False
    return True


def all_liberties(board, piece_type):
    moves = all_moves(board)
    l = []
    for x, y in moves:
        curr = copy.deepcopy(board)
        curr[x][y] = piece_type
        l.append(liberty(curr, x, y, piece_type))
    return [lib for sublist in l for lib in sublist]


def legal_moves(board, prev_board, piece_type):
    moves = all_moves(board)
    legal = []
    for x, y in moves:
        curr = copy.deepcopy(board)
        curr[x][y] = piece_type
        curr_copy = copy.deepcopy(curr)
        l = liberty(curr, x, y, piece_type)
        if not l:
            dead = find_dead_stones(3 - piece_type, curr)
            if dead:
                curr = remove_dead_stones(curr, dead)
            l = liberty(curr, x, y, piece_type)
        if l:
            dead = find_dead_stones(3 - piece_type, curr_copy)
            if dead:
                curr_copy = remove_dead_stones(curr_copy, dead)
            if not (dead and same_board(curr_copy, prev_board)):
                # KO
                legal.append((x, y))
    return legal


def apply(board, piece_type, x, y):
    curr = copy.deepcopy(board)
    curr[x][y] = piece_type
    return curr


def score(board):
    b = w = 0
    for row in board:
        b += row.count(1)
        w += row.count(2)

    if b > w + 2.5:
        return 1
    else:
        return 2


def alpha_beta_search(board, prev_board, depth, alpha, beta, piece_type,
                      max_player):
    # global table
    # global table_moves
    moves = legal_moves(board, prev_board, piece_type)
    # shuffle(moves)

    # if table['moves'] >= 25:
    #     if score(board) == piece_type:
    #         return 1000, None
    #     else:
    #         return -1000, None

    # h = hash(board)
    # if h in table:
    #     return table[h], table_moves[h]

    if depth == 0:
        # table[h], table_moves[h] = eval(board, piece_type), None
        # return table[h], table_moves[h]
        # return eval(board, piece_type), None
        return eval2(board, piece_type), None

    if not moves:
        curr_best = ['PASS']
        return 0, curr_best

    curr_best = None

    if max_player:
        value = -1000
        for x, y in moves:
            curr = apply(board, piece_type, x, y)
            dead = find_dead_stones(3 - piece_type, curr)
            curr = remove_dead_stones(curr, dead)
            a = alpha_beta_search(curr, board, depth - 1, alpha,
                                  beta, 3 - piece_type, False)
            if value < a[0]:
                value = a[0]
                alpha = max(alpha, value)
                curr_best = (x, y)
            if alpha >= beta:
                break
            # table[h], table_moves[h] = value, curr_best
        if curr_best is None:
            return value, None
        return value, curr_best
    else:
        value = 1000
        for x, y in moves:
            curr = apply(board, piece_type, x, y)
            dead = find_dead_stones(3 - piece_type, curr)
            curr = remove_dead_stones(curr, dead)
            a = alpha_beta_search(curr, board, depth - 1, alpha,
                                  beta, 3 - piece_type, True)
            if value > a[0]:
                value = a[0]
                beta = min(beta, value)
                curr_best = (x, y)
            if alpha >= beta:
                break
            # table[h], table_moves[h] = value, curr_best
        if curr_best is None:
            return value, None
        return value, curr_best


def play(board, prev_board, piece_type):
    global table
    moves = legal_moves(board, prev_board, piece_type)
    if len(moves) == 25 and piece_type == 1:
        next_move = (2, 2)
    elif len(moves) == 24 and piece_type == 1:
        if board[2][2] == 0:
            next_move = (2, 2)
        else:
            next_move = [random.choice([(2, 3), (2, 1), (1, 2), (3, 2)])]
    else:
        value, next_move = alpha_beta_search(board, prev_board, 2, -1000, 1000, piece_type,
                                             True)
    # table['moves'] += 1
    return next_move


# def save_table():
#     global table
#     # global table_moves
#     with open('myfile.txt', 'wb') as pkl_file:
#         pickle.dump(table, pkl_file)

# with open('myfile1.txt', 'wb') as pkl_file:
#     pickle.dump(table_moves, pkl_file)


if __name__ == "__main__":
    # if os.path.exists('myfile.txt'):
    #     with open('myfile.txt', 'rb') as pkl_file:
    #         table = pickle.load(pkl_file)
    # else:
    #     table = {}
    #
    # if os.path.exists('myfile1.txt'):
    #     with open('myfile1.txt', 'rb') as pkl_file:
    #         table_moves = pickle.load(pkl_file)
    # else:
    #     table_moves = {}

    piece_type, prev_board, board = readInput(5)

    # if 'moves' not in table:
    #     table['moves'] = 0
    # if piece_type == 2:
    #     table['moves'] += 1
    writeOutput(play(board, prev_board, piece_type))
    # save_table()
