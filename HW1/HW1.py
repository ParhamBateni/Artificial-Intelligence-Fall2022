"""
We have a n*n square area in which numbers [0,n^2-1] are placed. We can shift rows and columns in every direction
and given n, the initial state of the numbers in square and the goal state of the square we would like to know how we
can approach the goal state by shifting the rows and columns. For example:
n=2

initial state=[[0,2],[1,3]]
goal state=[[2,1],[3,0]]
the answer to this example would be:
up 1
right 1
right 2

0 2                          1 2                         2 1                         2 1
1 3  -> shift column 1 up -> 0 3 -> shift row 1 right -> 0 3 -> shift row 2 right -> 3 0 -> we have reached the goal!
"""


from time import time

import numpy as np


class Node:
    def __init__(self, state, parent, parent_operation):
        self.state = state
        self.parent = parent
        self.parent_operation = parent_operation
        self.h = 0
        self.g = 0
        self.f = 0

    def __eq__(self, other):
        return np.equal(self.state, other.state).all()

    def __lt__(self, other):
        return self.f < other.f

    def __repr__(self):
        representation = ''
        box_size = len(str(np.max(self.state)))
        box_n = self.state.shape[0]

        for i in range(2 * box_n + 1):
            if i % 2 == 0:
                representation += '-' * (box_size * box_n + box_n) + '\n'
                continue
            for j in range(2 * box_n + 1):
                if j % 2 == 0:
                    representation += '|'
                    continue
                number = str(self.state[(i - 1) // 2][(j - 1) // 2])
                representation += number + ' ' * (box_size - len(number))
            representation += '\n'
        return representation


def shift_row(state: np.array, row, right=True, amount=1):
    if right:
        state[row] = np.array(np.append(state[row][-amount:], state[row][:-amount]))
    else:
        state[row] = np.array(np.append(state[row][amount:], state[row][:amount]))
    return state


def shift_column(state: np.array, column, down=True, amount=1):
    state = state.transpose()
    if down:
        state[column] = np.array(np.append(state[column][-amount:], state[column][:-amount]))
    else:
        state[column] = np.array(np.append(state[column][amount:], state[column][:amount]))
    return state.transpose()


def heuristic_function(current_state: np.array, goal_state: np.array) -> int:
    h = 0
    n = current_state.shape[0]
    for i in range(n):
        for j in range(n):
            num = goal_state[i, j]
            num_position_in_current_state = np.argwhere(current_state == num)[0]
            x_diff, y_diff = np.abs(num_position_in_current_state - [i, j])
            h += min(x_diff, n - x_diff) + min(y_diff, n - y_diff)
    return 10 * h


def custom_hash(node: Node):
    n = node.state.shape[0]
    state_str = ''
    for i in range(n):
        for j in range(n):
            state_str += str(node.state[i, j])
    return hash(state_str)


def main():
    n = int(input())
    t0 = time()
    X_goal = []
    for _ in range(n):
        X_goal.append(list(map(int, input().split())))
    X_goal = Node(np.array(X_goal), None, None)

    X0 = []
    for _ in range(n):
        X0.append(list(map(int, input().split())))
    X0 = Node(np.array(X0), None, None)

    """
    A* algorithm:
        At each step it picks the node according to a value f which is a parameter equal to the sum of two other
        parameters g and h.
        g= the movement cost to move from the starting state up to the given state following the path generated to
            get there.
        h= the estimated movement cost to move from that given state to the final goal state
    """
    frontier = {}
    closed = {}
    X0.h = -1
    frontier[custom_hash(X0)] = X0
    path = []

    count_nodes_expanded = 0
    while frontier:
        key, current_node = min(frontier.items(), key=lambda x: x[1])
        frontier.pop(key)
        count_nodes_expanded += 1
        closed[custom_hash(current_node)] = 1
        if current_node.h == 0:
            while current_node != X0:
                path.append(current_node)
                current_node = current_node.parent
            path.append(X0)
            path = path[::-1]
            break
        neighbors = []
        for i in range(n):
            for right in [True, False]:
                if right:
                    direction = 'right'
                else:
                    direction = 'left'
                neighbors.append(
                    Node(shift_row(current_node.state.copy(), i, right), current_node, f'{direction} {i + 1}'))
            for down in [True, False]:
                if down:
                    direction = 'down'
                else:
                    direction = 'up'
                neighbors.append(
                    Node(shift_column(current_node.state.copy(), i, down), current_node, f'{direction} {i + 1}'))
        for neighbor in neighbors:
            neighbor_key = custom_hash(neighbor)
            if closed.get(neighbor_key) is not None: continue
            neighbor.g = neighbor.parent.g + 1
            neighbor.h = heuristic_function(neighbor.state, X_goal.state)
            neighbor.f = neighbor.g + neighbor.h
            frontier[neighbor_key] = neighbor
            old_node = frontier.get(neighbor_key)
            if old_node is None:
                frontier[neighbor_key] = neighbor
            elif old_node.f > neighbor.f:
                frontier[neighbor_key] = neighbor

    print(len(path) - 1)
    for node in path[1:]:
        print(node.parent_operation)
    # print(time()-t0)


main()
