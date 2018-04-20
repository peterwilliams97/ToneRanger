import numpy as np


edge = '+--'
body = '|  '
full = '|xx'
origin = '|++'

N = 8


def make_board(filled, start):
    assert start not in filled, (start, filled)
    edge_line = list(edge * 8 + '+')
    # body_line = body * 8 + '|'
    board = []
    for y in range(N):
        board.append(edge_line)
        row = []
        for x in range(N):
            cell = full if (x + 1, y + 1) in filled else body
            if x + 1 == start[0] and y + 1 == start[1]:
                cell = origin
            row.append(cell)
        row.append('|')
        board.append(row)
    board.append(edge_line)
    # print(len(board))
    # print(len(board[0]))
    return board


def print_board(board):
    # print(len(board))
    # print(len(board[0]))
    for row in board:
        print(''.join(row))


board = np.ones((N, N), dtype=int) * N * N
x0, y0 = start


def distance(x, y):
    if x == x0 and y == y0:
        return
    xl, yl = max(0, x - 1), max(0, y - 1)
    xh, yh = min(N - 1, x + 1), min(N - 1, y + 1)
    neighbors = [(x_, y_) for x_ in (xl, xh + 1) for y_ in (yl, yh + 1)]
    distance = min(distance(x_, y_) for x_, y_ in neighbors)
    return distance

    r = N * N - len(filled)
    x, y = start
    board[x, y] = 0
    n = 1
    _fill(board, filled, x, y)


def _fill(board, filled, x0, y0, n):
    n = board[x0, y0]
    if n >= N * N - len(filled):
        return
    x, y = x0, y0
    n += 1
    while True:
        x -= 1
        if x == 0 or (x, y) in filled:
            break
        board[x, y] = min(board[x, y], n)
    x, y = x0, y0
    while True:
        x += 1
        if x == N - 1 or (x, y) in filled:
            break
        board[x, y] = min(board[x, y], n)
    x, y = x0, y0
    while True:
        y -= 1
        if y == 0 or (x, y) in filled:
            break
        board[x, y] = min(board[x, y], n)
    x, y = x0, y0
    while True:
        y += 1
        if y == N - 1 or (x, y) in filled:
            break
            board[x, y] = min(board[x, y], n)






filled = {(3, 2), (4, 3), (5, 4)}
start = (6, 2)
board = make_board(filled, start)
print_board(board)

# print(edge_line)
# print(body_line)
# print(edge_line)
