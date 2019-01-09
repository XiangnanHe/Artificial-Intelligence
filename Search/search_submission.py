# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

from __future__ import division

import copy
import heapq
import os
# import pickle
import cPickle as pickle

from scipy.spatial import distance


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
        current (int): The index of the current node in the queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        return heapq.heappop(self.queue)

        # raise NotImplementedError

    def remove(self, node_id):
        """
        Remove a node from the queue.

        This is a hint, you might require this in ucs,
        however, if you choose not to use it, you are free to
        define your own method and not use it.

        Args:
            node_id (int): Index of node in queue.
        """

        raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        # if not self.queue:
        heapq.heappush(self.queue, node)

        # raise NotImplementedError

    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n for _, n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self == other

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    exploredd_set = set()
    # graph.explored_nodes.add(start)
    while frontier:
        if frontier.size() == 0:
            return []
        curr = frontier.pop()
        # Check if curr path last node is goal
        if curr[1][-1] == goal:
            return curr[1]

        if curr[1][-1] not in exploredd_set:
            exploredd_set.add(curr[1][-1])

            for new_node in graph[curr[1][-1]]:
                new_path = copy.deepcopy(curr[1])
                if new_node not in exploredd_set:
                    if new_node == goal:
                        new_path.append(new_node)
                        return new_path
                    else:
                        # graph.explored_nodes.add(new_node)
                        new_path.append(new_node)
                        frontier.append((len(new_path), new_path))

    # raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    exploredd_set = set()
    while frontier:
        if frontier.size() == 0:
            return []
        curr = frontier.pop()
        # Check if curr path last node is goal
        if curr[1][-1] == goal:
            return curr[1]
        if curr[1][-1] not in exploredd_set:
            exploredd_set.add(curr[1][-1])
            weight = curr[0]
            for new_node in graph.neighbors(curr[1][-1]):
                new_path = copy.deepcopy(curr[1])

                if new_node not in exploredd_set:
                    # if new_node == goal:
                    new_path.append(new_node)
                    frontier.append((graph[curr[1][-1]][new_node]['weight'] + weight, new_path))

    # raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    if v == goal:
        return 0
    distance_v = graph.node[v]['pos']
    distance_goal = graph.node[goal]['pos']

    return distance.euclidean(distance_v, distance_goal)

    # raise NotImplementedError


def get_path_weight(graph, new_path):
    path_weight_sum = 0
    for i in range(1, len(new_path)):
        path_weight_sum += graph[new_path[i - 1]][new_path[i]]['weight']
    return path_weight_sum


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier = PriorityQueue()
    frontier.append((0, [start]))
    exploredd_set = set()
    while frontier:
        if frontier.size() == 0:
            return []
        curr = frontier.pop()
        # Check if curr path last node is goal
        if curr[1][-1] == goal:
            return curr[1]
        if curr[1][-1] not in exploredd_set:
            exploredd_set.add(curr[1][-1])

            for new_node in graph.neighbors(curr[1][-1]):
                new_path = copy.deepcopy(curr[1])
                if new_node not in exploredd_set:
                    # if new_node == goal:
                    new_path.append(new_node)
                    weight_g = get_path_weight(graph, new_path)
                    weight_h = heuristic(graph, new_node, goal)
                    frontier.append((weight_g + weight_h, new_path))

    # raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier_start = PriorityQueue()
    frontier_start.append((0, [start]))
    frontier_end = PriorityQueue()
    frontier_end.append((0, [goal]))
    explored_set_start = set()
    explored_set_end = set()
    best_cost = float('inf')
    best_path = []
    node_cost_start, node_cost_end = {}, {}
    node_cost_start[start] = (0, [start])
    node_cost_end[goal] = (0, [goal])

    while frontier_start and frontier_end:
        if frontier_start.size() == 0 or frontier_end.size() == 0:
            return []
        if frontier_start.top()[0] + frontier_end.top()[0] >= best_cost:
            return best_path
        elif frontier_start.top()[0] >= frontier_end.top()[0]:
            curr_end = frontier_end.pop()
            if curr_end[1][-1] == start:
                if curr_end[0] < best_cost:
                    best_cost = curr_end[0]
                    best_path = copy.deepcopy(curr_end[1][::-1])
            elif curr_end[1][-1] in node_cost_start:
                if curr_end[0] + node_cost_start[curr_end[1][-1]][0] < best_cost:
                    best_cost = curr_end[0] + node_cost_start[curr_end[1][-1]][0]
                    best_path = copy.deepcopy(node_cost_start[curr_end[1][-1]][1][0:-1] + curr_end[1][::-1])
            if curr_end[1][-1] not in explored_set_end:
                explored_set_end.add(curr_end[1][-1])
                for new_node in graph[curr_end[1][-1]]:
                    if new_node not in explored_set_end:
                        new_path = copy.deepcopy(curr_end[1])
                        new_path.append(new_node)
                        frontier_end.append((graph[curr_end[1][-1]][new_node]['weight'] + curr_end[0], new_path))
                        if new_node not in node_cost_end:
                            node_cost_end[new_node] = (
                            graph[curr_end[1][-1]][new_node]['weight'] + curr_end[0], new_path)
                        else:
                            if node_cost_end[new_node][0] > graph[curr_end[1][-1]][new_node]['weight'] + curr_end[0]:
                                node_cost_end[new_node] = (
                                graph[curr_end[1][-1]][new_node]['weight'] + curr_end[0], new_path)
        else:
            curr_start = frontier_start.pop()
            if curr_start[1][-1] == goal:
                if curr_start[0] < best_cost:
                    best_cost = curr_start[0]
                    best_path = copy.deepcopy(curr_start[1])
            elif curr_start[1][-1] in node_cost_end:
                if curr_start[0] + node_cost_end[curr_start[1][-1]][0] < best_cost:
                    best_cost = curr_start[0] + node_cost_end[curr_start[1][-1]][0]
                    best_path = copy.deepcopy(curr_start[1][0:-1] + node_cost_end[curr_start[1][-1]][1][::-1])
            if curr_start[1][-1] not in explored_set_start:
                explored_set_start.add(curr_start[1][-1])
                for new_node in graph[curr_start[1][-1]]:
                    if new_node not in explored_set_start:
                        new_path = copy.deepcopy(curr_start[1])
                        new_path.append(new_node)
                        frontier_start.append((graph[curr_start[1][-1]][new_node]['weight'] + curr_start[0], new_path))
                        if new_node not in node_cost_start:
                            node_cost_start[new_node] = (
                            graph[curr_start[1][-1]][new_node]['weight'] + curr_start[0], new_path)
                        elif node_cost_start[new_node][0] > graph[curr_start[1][-1]][new_node]['weight'] + curr_start[
                            0]:
                            node_cost_start[new_node] = (
                            graph[curr_start[1][-1]][new_node]['weight'] + curr_start[0], new_path)
    return best_path

    # raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier_start = PriorityQueue()
    frontier_start.append((0, [start]))
    frontier_end = PriorityQueue()
    frontier_end.append((0, [goal]))
    explored_set_start = set()
    explored_set_end = set()
    best_cost = float('inf')
    best_path = []
    node_cost_start, node_cost_end = {}, {}
    node_cost_start[start] = (0, [start])
    node_cost_end[goal] = (0, [goal])

    while frontier_start and frontier_end:
        if frontier_start.size() == 0 or frontier_end.size() == 0:
            return []
        #if frontier_start.top()[0] >= best_cost and frontier_end.top()[0] >= best_cost:
        if get_path_weight(graph, frontier_start.top()[1]) + get_path_weight(graph, frontier_end.top()[1]) >= best_cost:
            return best_path

        #elif frontier_start.top()[0] >= frontier_end.top()[0] and frontier_end.top()[0] < best_cost:
        if frontier_end.top()[0] < best_cost:
            curr_end = frontier_end.pop()
            if curr_end[1][-1] == start:
                cost = get_path_weight(graph, curr_end[1])
                if cost < best_cost:
                    best_cost = cost
                    best_path = copy.deepcopy(curr_end[1][::-1])
            elif curr_end[1][-1] in node_cost_start:
                cost_curr = get_path_weight(graph, curr_end[1])
                cost_node_start = get_path_weight(graph, node_cost_start[curr_end[1][-1]][1])
                if cost_curr + cost_node_start < best_cost:
                    best_cost = cost_curr + cost_node_start
                    best_path = copy.deepcopy(node_cost_start[curr_end[1][-1]][1][0:-1] + curr_end[1][::-1])
            if curr_end[1][-1] not in explored_set_end:
                explored_set_end.add(curr_end[1][-1])
                for new_node in graph[curr_end[1][-1]]:
                    if new_node not in explored_set_end:
                        new_path = copy.deepcopy(curr_end[1])
                        new_path.append(new_node)
                        weight_g = get_path_weight(graph, new_path)
                        weight_h = heuristic(graph, start, new_node)
                        frontier_end.append((weight_g + weight_h, new_path))
                        if new_node not in node_cost_end:
                            node_cost_end[new_node] = (weight_g + weight_h, new_path)
                        else:
                            if node_cost_end[new_node][0] > weight_g + weight_h:
                                node_cost_end[new_node] = (weight_g + weight_h, new_path)

        #elif frontier_start.top()[0] <= frontier_end.top()[0] and frontier_start.top()[0] < best_cost:
        if frontier_start.top()[0] < best_cost:
            curr_start = frontier_start.pop()
            if curr_start[1][-1] == goal:
                cost = get_path_weight(graph, curr_start[1])
                if cost < best_cost:
                    best_cost = cost
                    best_path = copy.deepcopy(curr_start[1])
            elif curr_start[1][-1] in node_cost_end:
                cost_curr = get_path_weight(graph, curr_start[1])
                cost_node_end = get_path_weight(graph, node_cost_end[curr_start[1][-1]][1])
                if cost_curr + cost_node_end < best_cost:
                    best_cost = cost_curr + cost_node_end
                    best_path = copy.deepcopy(curr_start[1][0:-1] + node_cost_end[curr_start[1][-1]][1][::-1])
            if curr_start[1][-1] not in explored_set_start:
                explored_set_start.add(curr_start[1][-1])
                for new_node in graph[curr_start[1][-1]]:
                    if new_node not in explored_set_start:
                        new_path = copy.deepcopy(curr_start[1])
                        new_path.append(new_node)
                        weight_g = get_path_weight(graph, new_path)
                        weight_h = heuristic(graph, new_node, goal)
                        frontier_start.append((weight_g + weight_h, new_path))
                        if new_node not in node_cost_start:
                            node_cost_start[new_node] = (weight_g + weight_h, new_path)
                        elif node_cost_start[new_node][0] > weight_g + weight_h:
                            node_cost_start[new_node] = (weight_g + weight_h, new_path)
    return best_path


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    goal_1, goal_2, goal_3 = goals
    if goal_1 == goal_2 or goal_1 == goal_3 or goal_2 == goal_3:
        return []
    frontier_goal_1 = PriorityQueue()
    frontier_goal_1.append((0, [goal_1]))
    frontier_goal_2 = PriorityQueue()
    frontier_goal_2.append((0, [goal_2]))
    frontier_goal_3 = PriorityQueue()
    frontier_goal_3.append((0, [goal_3]))
    explored_set_goal_1 = set()
    explored_set_goal_2 = set()
    explored_set_goal_3 = set()
    best_cost_12 = float('inf')
    best_path_12 = []
    best_cost_23 = float('inf')
    best_path_23 = []
    best_cost_31 = float('inf')
    best_path_31 = []
    node_cost_goal_1, node_cost_goal_2, node_cost_goal_3 = {}, {}, {}
    node_cost_goal_1[goal_1] = (0, [goal_1])
    node_cost_goal_2[goal_2] = (0, [goal_2])
    node_cost_goal_3[goal_3] = (0, [goal_3])
    best_path = []

    while True:
        if frontier_goal_1.size == 0 or frontier_goal_2.size() == 0 or frontier_goal_3.size() == 0:
            return []
        goal_1_min_cost = frontier_goal_1.top()[0]
        goal_2_min_cost = frontier_goal_2.top()[0]
        goal_3_min_cost = frontier_goal_3.top()[0]

        #if goal_1_min_cost >= best_cost_12 and goal_2_min_cost >= best_cost_23 and goal_3_min_cost >= best_cost_31:
        #if goal_1_min_cost + goal_2_min_cost >= best_cost_12 and \
        #        goal_2_min_cost + goal_3_min_cost >= best_cost_23 and \
        #        goal_3_min_cost + goal_1_min_cost >= best_cost_31:

        if (goal_1_min_cost + goal_2_min_cost >= best_cost_12 and goal_2_min_cost + goal_3_min_cost >= best_path_23) or \
            (goal_2_min_cost + goal_3_min_cost >= best_cost_23 and goal_3_min_cost + goal_1_min_cost >= best_cost_31) or \
            (goal_3_min_cost + goal_1_min_cost >= best_cost_31 and goal_1_min_cost + goal_2_min_cost >= best_cost_12):

            """
            if (goal_1_min_cost >= best_cost_12 and goal_2_min_cost >= best_cost_23) or \
                (goal_2_min_cost >= best_cost_23 and goal_3_min_cost >= best_cost_31) or \
                (goal_3_min_cost >= best_cost_31 and goal_1_min_cost >= best_cost_12):
            """
            if best_cost_12 + best_cost_23 < best_cost_23 + best_cost_31 and \
                    best_cost_12 + best_cost_23 < best_cost_31 + best_cost_12:
                if set(best_path_12).issubset(set(best_path_23)):
                    best_path = best_path_23
                elif set(best_path_23).issubset(set(best_path_12)):
                    best_path = best_path_12
                else:
                    best_path = best_path_12[:-1] + best_path_23
            elif best_cost_23 + best_cost_31 < best_cost_12 + best_cost_23 and \
                    best_cost_23 + best_cost_31 < best_cost_31 + best_cost_12:
                if set(best_path_23).issubset(set(best_path_31)):
                    best_path = best_path_31
                elif set(best_path_31).issubset(set(best_path_23)):
                    best_path = best_path_23
                else:
                    best_path = best_path_23[:-1] + best_path_31
            else:
                if set(best_path_31).issubset(set(best_path_12)):
                    best_path = best_path_12
                elif set(best_path_12).issubset(set(best_path_31)):
                    best_path = best_path_31
                else:
                    best_path = best_path_31[:-1] + best_path_12

            return best_path

        # Expand goal 1
        if goal_1_min_cost <= best_cost_12 or goal_1_min_cost <= best_cost_31:
            curr_goal_1 = frontier_goal_1.pop()
            if curr_goal_1[1][-1] == goal_2:
                if curr_goal_1[0] < best_cost_12:
                    best_cost_12 = curr_goal_1[0]
                    best_path_12 = copy.deepcopy(curr_goal_1[1])
            if curr_goal_1[1][-1] in node_cost_goal_2:
                if curr_goal_1[0] + node_cost_goal_2[curr_goal_1[1][-1]][0] < best_cost_12:
                    best_cost_12 = curr_goal_1[0] + node_cost_goal_2[curr_goal_1[1][-1]][0]
                    best_path_12 = copy.deepcopy(curr_goal_1[1][0:-1] + node_cost_goal_2[curr_goal_1[1][-1]][1][::-1])
            if curr_goal_1[1][-1] in node_cost_goal_3:
                if curr_goal_1[0] + node_cost_goal_3[curr_goal_1[1][-1]][0] < best_cost_31:
                    best_cost_31 = curr_goal_1[0] + node_cost_goal_3[curr_goal_1[1][-1]][0]
                    best_path_31 = copy.deepcopy(node_cost_goal_3[curr_goal_1[1][-1]][1][:-1] + curr_goal_1[1][::-1])
            if curr_goal_1[1][-1] not in explored_set_goal_1:
                explored_set_goal_1.add(curr_goal_1[1][-1])
                for new_node in graph[curr_goal_1[1][-1]]:
                    if new_node not in explored_set_goal_1:
                        new_path = copy.deepcopy(curr_goal_1[1])
                        new_path.append(new_node)
                        frontier_goal_1.append(
                            (graph[curr_goal_1[1][-1]][new_node]['weight'] + curr_goal_1[0], new_path))
                        if new_node not in node_cost_goal_1:
                            node_cost_goal_1[new_node] = \
                                (graph[curr_goal_1[1][-1]][new_node]['weight'] + curr_goal_1[0], new_path)
                        elif node_cost_goal_1[new_node][0] > graph[curr_goal_1[1][-1]][new_node]['weight'] + \
                                    curr_goal_1[0]:
                            node_cost_goal_1[new_node] = (
                            graph[curr_goal_1[1][-1]][new_node]['weight'] + curr_goal_1[0], new_path)

        # Expand goal 2
        if goal_2_min_cost <= best_cost_23 or goal_2_min_cost <= best_cost_12:
            curr_goal_2 = frontier_goal_2.pop()
            if curr_goal_2[1][-1] == goal_3:
                if curr_goal_2[0] < best_cost_23:
                    best_cost_23 = curr_goal_2[0]
                    best_path_23 = copy.deepcopy(curr_goal_2[1])
            if curr_goal_2[1][-1] in node_cost_goal_3:
                if curr_goal_2[0] + node_cost_goal_3[curr_goal_2[1][-1]][0] < best_cost_23:
                    best_cost_23 = curr_goal_2[0] + node_cost_goal_3[curr_goal_2[1][-1]][0]
                    best_path_23 = copy.deepcopy(curr_goal_2[1][:-1] + node_cost_goal_3[curr_goal_2[1][-1]][1][::-1])
            if curr_goal_2[1][-1] in node_cost_goal_1:
                if curr_goal_2[0] + node_cost_goal_1[curr_goal_2[1][-1]][0] < best_cost_12:
                    best_cost_12 = curr_goal_2[0] + node_cost_goal_1[curr_goal_2[1][-1]][0]
                    best_path_12 = copy.deepcopy(node_cost_goal_1[curr_goal_2[1][-1]][1][:-1] + curr_goal_2[1][::-1])
            if curr_goal_2[1][-1] not in explored_set_goal_2:
                explored_set_goal_2.add(curr_goal_2[1][-1])
                for new_node in graph[curr_goal_2[1][-1]]:
                    if new_node not in explored_set_goal_2:
                        new_path = copy.deepcopy(curr_goal_2[1])
                        new_path.append(new_node)
                        frontier_goal_2.append(
                            (graph[curr_goal_2[1][-1]][new_node]['weight'] + curr_goal_2[0], new_path))
                        if new_node not in node_cost_goal_2:
                            node_cost_goal_2[new_node] = (
                                graph[curr_goal_2[1][-1]][new_node]['weight'] + curr_goal_2[0], new_path)
                        elif node_cost_goal_2[new_node][0] > graph[curr_goal_2[1][-1]][new_node]['weight'] + \
                                curr_goal_2[0]:
                            node_cost_goal_2[new_node] = (
                                graph[curr_goal_2[1][-1]][new_node]['weight'] + curr_goal_2[0], new_path)

        # Expand goal 3
        if goal_3_min_cost <= best_cost_31 or goal_3_min_cost < best_cost_23:
            curr_goal_3 = frontier_goal_3.pop()
            if curr_goal_3[1][-1] == goal_1:
                if curr_goal_3[0] < best_cost_31:
                    best_cost_31 = curr_goal_3[0]
                    best_path_31 = copy.deepcopy(curr_goal_3[1])
            if curr_goal_3[1][-1] in node_cost_goal_1:
                if curr_goal_3[0] + node_cost_goal_1[curr_goal_3[1][-1]][0] < best_cost_31:
                    best_cost_31 = curr_goal_3[0] + node_cost_goal_1[curr_goal_3[1][-1]][0]
                    best_path_31 = copy.deepcopy(curr_goal_3[1][0:-1] + node_cost_goal_1[curr_goal_3[1][-1]][1][::-1])
            if curr_goal_3[1][-1] in node_cost_goal_2:
                if curr_goal_3[0] + node_cost_goal_2[curr_goal_3[1][-1]][0] < best_cost_23:
                    best_cost_23 = curr_goal_3[0] + node_cost_goal_2[curr_goal_3[1][-1]][0]
                    best_path_23 = copy.deepcopy(node_cost_goal_2[curr_goal_3[1][-1]][1][:-1] + curr_goal_3[1][::-1])
            if curr_goal_3[1][-1] not in explored_set_goal_3:
                explored_set_goal_3.add(curr_goal_3[1][-1])
                for new_node in graph[curr_goal_3[1][-1]]:
                    if new_node not in explored_set_goal_3:
                        new_path = copy.deepcopy(curr_goal_3[1])
                        new_path.append(new_node)
                        frontier_goal_3.append(
                            (graph[curr_goal_3[1][-1]][new_node]['weight'] + curr_goal_3[0], new_path))
                        if new_node not in node_cost_goal_3:
                            node_cost_goal_3[new_node] = (
                            graph[curr_goal_3[1][-1]][new_node]['weight'] + curr_goal_3[0], new_path)
                        elif node_cost_goal_3[new_node][0] > graph[curr_goal_3[1][-1]][new_node]['weight'] + \
                                curr_goal_3[0]:
                            node_cost_goal_3[new_node] = (
                            graph[curr_goal_3[1][-1]][new_node]['weight'] + curr_goal_3[0], new_path)
    return best_path


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 3: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:f
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    goal_1, goal_2, goal_3 = goals
    if goal_1 == goal_2 or goal_1 == goal_3 or goal_2 == goal_3:
        return []
    frontier_goal_1 = PriorityQueue()
    frontier_goal_1.append((0, [goal_1]))
    frontier_goal_2 = PriorityQueue()
    frontier_goal_2.append((0, [goal_2]))
    frontier_goal_3 = PriorityQueue()
    frontier_goal_3.append((0, [goal_3]))
    explored_set_goal_1 = set()
    explored_set_goal_2 = set()
    explored_set_goal_3 = set()
    best_cost_12 = float('inf')
    best_path_12 = []
    best_cost_23 = float('inf')
    best_path_23 = []
    best_cost_31 = float('inf')
    best_path_31 = []
    node_cost_goal_1, node_cost_goal_2, node_cost_goal_3 = {}, {}, {}
    node_cost_goal_1[goal_1] = (0, [goal_1])
    node_cost_goal_2[goal_2] = (0, [goal_2])
    node_cost_goal_3[goal_3] = (0, [goal_3])
    best_path = []

    while True:
        if frontier_goal_1.size == 0 or frontier_goal_2.size() == 0 or frontier_goal_3.size() == 0:
            return []
        goal_1_min_cost = frontier_goal_1.top()[0]
        goal_2_min_cost = frontier_goal_2.top()[0]
        goal_3_min_cost = frontier_goal_3.top()[0]

        #if goal_1_min_cost >= best_cost_12 and goal_2_min_cost >= best_cost_23 and goal_3_min_cost >= best_cost_31:
        #if goal_1_min_cost + goal_2_min_cost >= best_cost_12 and \
        #        goal_2_min_cost + goal_3_min_cost >= best_cost_23 and \
        #        goal_3_min_cost + goal_1_min_cost >= best_cost_31:

        if (goal_1_min_cost >= best_cost_12 and goal_2_min_cost >= best_path_23) or \
            (goal_2_min_cost >= best_cost_23 and goal_3_min_cost >= best_cost_31) or \
            (goal_3_min_cost >= best_cost_31 and goal_1_min_cost >= best_cost_12):

            """
            if (goal_1_min_cost >= best_cost_12 and goal_2_min_cost >= best_cost_23) or \
                (goal_2_min_cost >= best_cost_23 and goal_3_min_cost >= best_cost_31) or \
                (goal_3_min_cost >= best_cost_31 and goal_1_min_cost >= best_cost_12):
            """
            if best_cost_12 + best_cost_23 < best_cost_23 + best_cost_31 and \
                    best_cost_12 + best_cost_23 < best_cost_31 + best_cost_12:
                if set(best_path_12).issubset(set(best_path_23)):
                    best_path = best_path_23
                elif set(best_path_23).issubset(set(best_path_12)):
                    best_path = best_path_12
                else:
                    best_path = best_path_12[:-1] + best_path_23
            elif best_cost_23 + best_cost_31 < best_cost_12 + best_cost_23 and \
                    best_cost_23 + best_cost_31 < best_cost_31 + best_cost_12:
                if set(best_path_23).issubset(set(best_path_31)):
                    best_path = best_path_31
                elif set(best_path_31).issubset(set(best_path_23)):
                    best_path = best_path_23
                else:
                    best_path = best_path_23[:-1] + best_path_31
            else:
                if set(best_path_31).issubset(set(best_path_12)):
                    best_path = best_path_12
                elif set(best_path_12).issubset(set(best_path_31)):
                    best_path = best_path_31
                else:
                    best_path = best_path_31[:-1] + best_path_12

            return best_path

        # Expand goal 1
        if goal_1_min_cost <= best_cost_12 or goal_1_min_cost <= best_cost_31:
            curr_goal_1 = frontier_goal_1.pop()
            if curr_goal_1[1][-1] == goal_2:
                cost = get_path_weight(graph, curr_goal_1[1])
                if cost < best_cost_12:
                    best_cost_12 = cost
                    best_path_12 = copy.deepcopy(curr_goal_1[1])
            if curr_goal_1[1][-1] in node_cost_goal_2:
                cost_goal_1 = get_path_weight(graph, curr_goal_1[1])
                cost_node_goal_2 = get_path_weight(graph, node_cost_goal_2[curr_goal_1[1][-1]][1])
                if cost_goal_1 + cost_node_goal_2 < best_cost_12:
                    best_cost_12 = cost_goal_1 + cost_node_goal_2
                    best_path_12 = copy.deepcopy(curr_goal_1[1][0:-1] + node_cost_goal_2[curr_goal_1[1][-1]][1][::-1])
            if curr_goal_1[1][-1] in node_cost_goal_3:
                cost_goal_1 = get_path_weight(graph, curr_goal_1[1])
                cost_node_goal_3 = get_path_weight(graph, node_cost_goal_3[curr_goal_1[1][-1]][1])
                if cost_goal_1 + cost_node_goal_3 < best_cost_31:
                    best_cost_31 = cost_goal_1 + cost_node_goal_3
                    best_path_31 = copy.deepcopy(node_cost_goal_3[curr_goal_1[1][-1]][1][:-1] + curr_goal_1[1][::-1])
            if curr_goal_1[1][-1] not in explored_set_goal_1:
                explored_set_goal_1.add(curr_goal_1[1][-1])
                for new_node in graph[curr_goal_1[1][-1]]:
                    if new_node not in explored_set_goal_1:
                        new_path = copy.deepcopy(curr_goal_1[1])
                        new_path.append(new_node)
                        weight_g = get_path_weight(graph, new_path)
                        weight_h = heuristic(graph, new_node, goal_2)
                        frontier_goal_1.append((weight_h + weight_g, new_path))
                        if new_node not in node_cost_goal_1:
                            node_cost_goal_1[new_node] = (weight_g + weight_h, new_path)
                        elif node_cost_goal_1[new_node][0] > weight_h + weight_g:
                            node_cost_goal_1[new_node] = (weight_g + weight_h, new_path)

        # Expand goal 2
        if goal_2_min_cost <= best_cost_23 or goal_2_min_cost <= best_cost_12:
            curr_goal_2 = frontier_goal_2.pop()
            if curr_goal_2[1][-1] == goal_3:
                cost = get_path_weight(graph, curr_goal_2[1])
                if cost < best_cost_23:
                    best_cost_23 = cost
                    best_path_23 = copy.deepcopy(curr_goal_2[1])
            if curr_goal_2[1][-1] in node_cost_goal_3:
                cost_goal_2 = get_path_weight(graph, curr_goal_2[1])
                cost_node_goal_3 = get_path_weight(graph, node_cost_goal_3[curr_goal_2[1][-1]][1])
                if cost_goal_2 + cost_node_goal_3 < best_cost_23:
                    best_cost_23 = cost_goal_2 + cost_node_goal_3
                    best_path_23 = copy.deepcopy(curr_goal_2[1][:-1] + node_cost_goal_3[curr_goal_2[1][-1]][1][::-1])
            if curr_goal_2[1][-1] in node_cost_goal_1:
                cost_goal_2 = get_path_weight(graph, curr_goal_2[1])
                cost_node_goal_1 = get_path_weight(graph, node_cost_goal_1[curr_goal_2[1][-1]][1])
                if cost_goal_2 + cost_node_goal_1 < best_cost_12:
                    best_cost_12 = cost_goal_2 + cost_node_goal_1
                    best_path_12 = copy.deepcopy(node_cost_goal_1[curr_goal_2[1][-1]][1][:-1] + curr_goal_2[1][::-1])
            if curr_goal_2[1][-1] not in explored_set_goal_2:
                explored_set_goal_2.add(curr_goal_2[1][-1])
                for new_node in graph[curr_goal_2[1][-1]]:
                    if new_node not in explored_set_goal_2:
                        new_path = copy.deepcopy(curr_goal_2[1])
                        new_path.append(new_node)
                        weight_g = get_path_weight(graph, new_path)
                        weight_h = heuristic(graph, new_node, goal_3)
                        frontier_goal_2.append((weight_g + weight_h, new_path))
                        if new_node not in node_cost_goal_2:
                            node_cost_goal_2[new_node] = (weight_h + weight_g, new_path)
                        elif node_cost_goal_2[new_node][0] > weight_g + weight_h:
                            node_cost_goal_2[new_node] = (weight_g + weight_h, new_path)

        # Expand goal 3
        if goal_3_min_cost <= best_cost_31 or goal_3_min_cost < best_cost_23:
            curr_goal_3 = frontier_goal_3.pop()
            if curr_goal_3[1][-1] == goal_1:
                cost = get_path_weight(graph, curr_goal_3[1])
                if cost < best_cost_31:
                    best_cost_31 = cost
                    best_path_31 = copy.deepcopy(curr_goal_3[1])
            if curr_goal_3[1][-1] in node_cost_goal_1:
                cost_goal_3 = get_path_weight(graph, curr_goal_3[1])
                cost_node_goal_1 = get_path_weight(graph, node_cost_goal_1[curr_goal_3[1][-1]][1])
                if cost_goal_3 + cost_node_goal_1 < best_cost_31:
                    best_cost_31 = cost_goal_3 + cost_node_goal_1
                    best_path_31 = copy.deepcopy(curr_goal_3[1][0:-1] + node_cost_goal_1[curr_goal_3[1][-1]][1][::-1])
            if curr_goal_3[1][-1] in node_cost_goal_2:
                cost_goal_3 = get_path_weight(graph, curr_goal_3[1])
                cost_node_goal_2 = get_path_weight(graph, node_cost_goal_2[curr_goal_3[1][-1]][1])
                if cost_goal_3 + cost_node_goal_2 < best_cost_23:
                    best_cost_23 = cost_goal_3 + cost_node_goal_2
                    best_path_23 = copy.deepcopy(node_cost_goal_2[curr_goal_3[1][-1]][1][:-1] + curr_goal_3[1][::-1])
            if curr_goal_3[1][-1] not in explored_set_goal_3:
                explored_set_goal_3.add(curr_goal_3[1][-1])
                for new_node in graph[curr_goal_3[1][-1]]:
                    if new_node not in explored_set_goal_3:
                        new_path = copy.deepcopy(curr_goal_3[1])
                        new_path.append(new_node)
                        weight_g = get_path_weight(graph, new_path)
                        weight_h = heuristic(graph, new_node, goal_1)
                        frontier_goal_3.append((weight_g + weight_h, new_path))
                        if new_node not in node_cost_goal_3:
                            node_cost_goal_3[new_node] = (weight_g + weight_h, new_path)
                        elif node_cost_goal_3[new_node][0] > weight_g + weight_h:
                            node_cost_goal_3[new_node] = (weight_g + weight_h, new_path)


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Xiangnan He"


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to bonnie, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start == goal:
        return []
    frontier_start = PriorityQueue()
    frontier_start.append((0, [start]))
    frontier_end = PriorityQueue()
    frontier_end.append((0, [goal]))
    explored_set_start = set()
    explored_set_end = set()
    best_cost = float('inf')
    best_path = []
    node_cost_start, node_cost_end = {}, {}
    node_cost_start[start] = (0, [start])
    node_cost_end[goal] = (0, [goal])

    while frontier_start and frontier_end:
        if frontier_start.size() == 0 or frontier_end.size() == 0:
            return []
        #if frontier_start.top()[0] >= best_cost and frontier_end.top()[0] >= best_cost:
        if get_path_weight(graph, frontier_start.top()[1]) + get_path_weight(graph, frontier_end.top()[1]) >= best_cost:
            return best_path

        #elif frontier_start.top()[0] >= frontier_end.top()[0] and frontier_end.top()[0] < best_cost:
        if frontier_end.top()[0] < best_cost:
            curr_end = frontier_end.pop()
            if curr_end[1][-1] == start:
                cost = get_path_weight(graph, curr_end[1])
                if cost < best_cost:
                    best_cost = cost
                    best_path = copy.deepcopy(curr_end[1][::-1])
            elif curr_end[1][-1] in node_cost_start:
                cost_curr = get_path_weight(graph, curr_end[1])
                cost_node_start = get_path_weight(graph, node_cost_start[curr_end[1][-1]][1])
                if cost_curr + cost_node_start < best_cost:
                    best_cost = cost_curr + cost_node_start
                    best_path = copy.deepcopy(node_cost_start[curr_end[1][-1]][1][0:-1] + curr_end[1][::-1])
            if curr_end[1][-1] not in explored_set_end:
                explored_set_end.add(curr_end[1][-1])
                for new_node in graph[curr_end[1][-1]]:
                    if new_node not in explored_set_end:
                        new_path = copy.deepcopy(curr_end[1])
                        new_path.append(new_node)
                        weight_g = get_path_weight(graph, new_path)
                        weight_h = euclidean_dist_heuristic(graph, start, new_node)
                        frontier_end.append((weight_g + weight_h, new_path))
                        if new_node not in node_cost_end:
                            node_cost_end[new_node] = (weight_g + weight_h, new_path)
                        else:
                            if node_cost_end[new_node][0] > weight_g + weight_h:
                                node_cost_end[new_node] = (weight_g + weight_h, new_path)

        #elif frontier_start.top()[0] <= frontier_end.top()[0] and frontier_start.top()[0] < best_cost:
        if frontier_start.top()[0] < best_cost:
            curr_start = frontier_start.pop()
            if curr_start[1][-1] == goal:
                cost = get_path_weight(graph, curr_start[1])
                if cost < best_cost:
                    best_cost = cost
                    best_path = copy.deepcopy(curr_start[1])
            elif curr_start[1][-1] in node_cost_end:
                cost_curr = get_path_weight(graph, curr_start[1])
                cost_node_end = get_path_weight(graph, node_cost_end[curr_start[1][-1]][1])
                if cost_curr + cost_node_end < best_cost:
                    best_cost = cost_curr + cost_node_end
                    best_path = copy.deepcopy(curr_start[1][0:-1] + node_cost_end[curr_start[1][-1]][1][::-1])
            if curr_start[1][-1] not in explored_set_start:
                explored_set_start.add(curr_start[1][-1])
                for new_node in graph[curr_start[1][-1]]:
                    if new_node not in explored_set_start:
                        new_path = copy.deepcopy(curr_start[1])
                        new_path.append(new_node)
                        weight_g = get_path_weight(graph, new_path)
                        weight_h = euclidean_dist_heuristic(graph, new_node, goal)
                        frontier_start.append((weight_g + weight_h, new_path))
                        if new_node not in node_cost_start:
                            node_cost_start[new_node] = (weight_g + weight_h, new_path)
                        elif node_cost_start[new_node][0] > weight_g + weight_h:
                            node_cost_start[new_node] = (weight_g + weight_h, new_path)
    return best_path
    #raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph[node1][node2]['weight']
    """

    # nodes = graph.nodes()
    return None
