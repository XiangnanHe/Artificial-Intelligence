"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
#from exampleinference import inferenceExample

#inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)

"""
WRITE YOUR CODE BELOW.
"""


from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
import Distribution
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine, EnumerationEngine

from random import randint, random
from numpy.random import choice
import numpy as np


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    # TODO: finish this function    
    A_node = BayesNode(0, 2, name = 'alarm')
    FA_node = BayesNode(1, 2, name = 'faulty alarm')
    G_node = BayesNode(2, 2, name = 'gauge')
    FG_node = BayesNode(3, 2, name = 'faulty gauge')
    T_node = BayesNode(4, 2, name = 'temperature')

    nodes.extend([A_node, FA_node, G_node, FG_node, T_node])

    G_node.add_parent(T_node)
    T_node.add_child(G_node)

    FG_node.add_parent(T_node)
    T_node.add_child(FG_node)

    G_node.add_parent(FG_node)
    FG_node.add_child(G_node)

    A_node.add_parent(G_node)
    G_node.add_child(A_node)

    A_node.add_parent(FA_node)
    FA_node.add_child(A_node)

    return BayesNet(nodes)


def make_cargo_flight_net():
    """Create a Bayes Net representation of the above power plant problem.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    # TODO: finish this function
    paint_node = BayesNode(0, 2, name = 'paint')
    weather_node = BayesNode(1, 2, name = 'weather')
    oring_node = BayesNode(2, 2, name = 'oring')
    shuttle_node = BayesNode(3, 2, name = 'shuttle')
    booster_node = BayesNode(4, 2, name = 'booster')
    mission_node = BayesNode(5, 2, name='mission')

    nodes.extend([paint_node, weather_node, oring_node, shuttle_node, booster_node, mission_node])

    shuttle_node.add_parent(paint_node)
    paint_node.add_child(shuttle_node)

    shuttle_node.add_parent(weather_node)
    weather_node.add_child(shuttle_node)

    booster_node.add_parent(weather_node)
    weather_node.add_child(booster_node)

    booster_node.add_parent(oring_node)
    oring_node.add_child(booster_node)

    mission_node.add_parent(shuttle_node)
    shuttle_node.add_child(mission_node)

    mission_node.add_parent(booster_node)
    booster_node.add_child(mission_node)

    return BayesNet(nodes)


def make_final_exam_net():
    """Create a Bayes Net representation of the above power plant problem.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    # TODO: finish this function
    N_node = BayesNode(0, 2, name = 'netflix')
    E_node = BayesNode(1, 2, name = 'exercise')
    A_node = BayesNode(2, 2, name = 'assignment')
    C_node = BayesNode(3, 2, name = 'club')
    S_node = BayesNode(4, 2, name = 'social')
    G_node = BayesNode(5, 2, name='graduate')


    nodes.extend([N_node, E_node, A_node, C_node, S_node, G_node])

    A_node.add_parent(N_node)
    N_node.add_child(A_node)

    A_node.add_parent(E_node)
    E_node.add_child(A_node)

    S_node.add_parent(C_node)
    C_node.add_child(S_node)

    G_node.add_parent(A_node)
    A_node.add_child(G_node)

    G_node.add_parent(S_node)
    S_node.add_child(G_node)

    return BayesNet(nodes)

def set_probability_final_exam(bayes_net):
    """Set probability distribution for each node in the power plant system."""
    netflix_node = bayes_net.get_node_by_name("netflix")
    exercise_node = bayes_net.get_node_by_name("exercise")
    assignment_node = bayes_net.get_node_by_name("assignment")
    club_node = bayes_net.get_node_by_name("club")
    social_node = bayes_net.get_node_by_name("social")
    graduate_node = bayes_net.get_node_by_name("graduate")
    nodes = [netflix_node, exercise_node, assignment_node, club_node, social_node, graduate_node]
    # TODO: set the probability distribution for each node
    #raise NotImplementedError

    netflix_distribution = DiscreteDistribution(netflix_node)
    index = netflix_distribution.generate_index([], [])
    netflix_distribution[index] = [0.27, 0.73]
    netflix_node.set_dist(netflix_distribution)

    exercise_distribution = DiscreteDistribution(exercise_node)
    index = exercise_distribution.generate_index([], [])
    exercise_distribution[index] = [0.8, 0.2]
    exercise_node.set_dist(exercise_distribution)

    # Before new arrival
    club_distribution = DiscreteDistribution(club_node)
    index = club_distribution.generate_index([], [])
    club_distribution[index] = [0.64, 0.36]
    club_node.set_dist(club_distribution)

    dist = zeros([club_node.size(), social_node.size()], dtype=float32)  # Note the order of G_node, A_node
    dist[0, :] = [0.31, 0.69]  # probabilities for A when G is FALSE
    dist[1, :] = [0.22, 0.78]  # probabilities for A given G is TRUE
    socal_distribution = ConditionalDiscreteDistribution(nodes=[club_node, social_node], table=dist)
    social_node.set_dist(socal_distribution)

    dist = zeros([netflix_node.size(), exercise_node.size(), assignment_node.size()], dtype=float32)
    dist[0, 0, :] = [0.07, 0.93]
    dist[0, 1, :] = [0.11, 0.89]
    dist[1, 0, :] = [0.82, 0.18]
    dist[1, 1, :] = [0.56, 0.44]
    assignment_distribution = ConditionalDiscreteDistribution(nodes = [netflix_node, exercise_node, assignment_node], table=dist)
    assignment_node.set_dist(assignment_distribution)


    dist = zeros([assignment_node.size(), social_node.size(), graduate_node.size()], dtype=float32)
    dist[0, 0, :] = [0.7, 0.3]
    dist[0, 1, :] = [0.91, 0.09]
    dist[1, 0, :] = [0.23, 0.77]
    dist[1, 1, :] = [0.48, 0.52]
    graduate_distribution = ConditionalDiscreteDistribution(nodes = [assignment_node, social_node, graduate_node], table=dist)
    graduate_node.set_dist(graduate_distribution)

    return bayes_net

def set_probability_cargo_flight_badbooster(bayes_net):
    """Set probability distribution for each node in the power plant system."""
    paint_node = bayes_net.get_node_by_name("paint")
    weather_node = bayes_net.get_node_by_name("weather")
    oring_node = bayes_net.get_node_by_name("oring")
    shuttle_node = bayes_net.get_node_by_name("shuttle")
    booster_node = bayes_net.get_node_by_name("booster")
    mission_node = bayes_net.get_node_by_name("mission")
    nodes = [paint_node, weather_node, oring_node, shuttle_node, booster_node, mission_node]
    # TODO: set the probability distribution for each node
    #raise NotImplementedError

    weather_distribution = DiscreteDistribution(weather_node)
    index = weather_distribution.generate_index([], [])
    weather_distribution[index] = [0.1, 0.9]
    weather_node.set_dist(weather_distribution)

    paint_distribution = DiscreteDistribution(paint_node)
    index = paint_distribution.generate_index([], [])
    paint_distribution[index] = [0.01, 0.99]
    paint_node.set_dist(paint_distribution)

    # Before new arrival
    oring_distribution = DiscreteDistribution(oring_node)
    index = oring_distribution.generate_index([], [])
    oring_distribution[index] = [0.75, 0.25]
    oring_node.set_dist(oring_distribution)

    # After new arrival
    #oring_distribution = DiscreteDistribution(oring_node)
    #index = oring_distribution.generate_index([], [])
    #oring_distribution[index] = [0.15, 0.85]
    #oring_node.set_dist(oring_distribution)

    dist = zeros([paint_node.size(), weather_node.size(), shuttle_node.size()], dtype=float32)
    dist[0, 0, :] = [0.4, 0.6]
    dist[0, 1, :] = [0.4, 0.6]
    dist[1, 0, :] = [0.85, 0.15]
    dist[1, 1, :] = [0.02, 0.98]
    shuttle_distribution = ConditionalDiscreteDistribution(nodes = [paint_node, weather_node, shuttle_node], table=dist)
    shuttle_node.set_dist(shuttle_distribution)


    dist = zeros([weather_node.size(), oring_node.size(), booster_node.size()], dtype=float32)
    dist[0, 0, :] = [0.95, 0.05]
    dist[0, 1, :] = [0.15, 0.85]
    dist[1, 0, :] = [0.95, 0.05]
    dist[1, 1, :] = [0.05, 0.95]
    booster_distribution = ConditionalDiscreteDistribution(nodes = [weather_node, oring_node, booster_node], table=dist)
    booster_node.set_dist(booster_distribution)

    dist = zeros([shuttle_node.size(), booster_node.size(), mission_node.size()], dtype=float32)
    dist[0, 0, :] = [0.55, 0.45]
    dist[0, 1, :] = [0.2, 0.8]
    dist[1, 0, :] = [0.2, 0.8]
    dist[1, 1, :] = [0.05, 0.95]
    mission_distribution = ConditionalDiscreteDistribution(nodes = [shuttle_node, booster_node, mission_node], table=dist)
    mission_node.set_dist(mission_distribution)

    return bayes_net

def set_probability_cargo_flight_goodbooster(bayes_net):
    """Set probability distribution for each node in the power plant system."""
    paint_node = bayes_net.get_node_by_name("paint")
    weather_node = bayes_net.get_node_by_name("weather")
    oring_node = bayes_net.get_node_by_name("oring")
    shuttle_node = bayes_net.get_node_by_name("shuttle")
    booster_node = bayes_net.get_node_by_name("booster")
    mission_node = bayes_net.get_node_by_name("mission")
    nodes = [paint_node, weather_node, oring_node, shuttle_node, booster_node, mission_node]
    # TODO: set the probability distribution for each node
    #raise NotImplementedError

    weather_distribution = DiscreteDistribution(weather_node)
    index = weather_distribution.generate_index([], [])
    weather_distribution[index] = [0.1, 0.9]
    weather_node.set_dist(weather_distribution)

    paint_distribution = DiscreteDistribution(paint_node)
    index = paint_distribution.generate_index([], [])
    paint_distribution[index] = [0.01, 0.99]
    paint_node.set_dist(paint_distribution)

    # Before new arrival
    #oring_distribution = DiscreteDistribution(oring_node)
    #index = oring_distribution.generate_index([], [])
    #oring_distribution[index] = [0.75, 0.25]
    #oring_node.set_dist(oring_distribution)

    # After new arrival
    oring_distribution = DiscreteDistribution(oring_node)
    index = oring_distribution.generate_index([], [])
    oring_distribution[index] = [0.15, 0.85]
    oring_node.set_dist(oring_distribution)

    dist = zeros([paint_node.size(), weather_node.size(), shuttle_node.size()], dtype=float32)
    dist[0, 0, :] = [0.4, 0.6]
    dist[0, 1, :] = [0.4, 0.6]
    dist[1, 0, :] = [0.85, 0.15]
    dist[1, 1, :] = [0.02, 0.98]
    shuttle_distribution = ConditionalDiscreteDistribution(nodes = [paint_node, weather_node, shuttle_node], table=dist)
    shuttle_node.set_dist(shuttle_distribution)


    dist = zeros([weather_node.size(), oring_node.size(), booster_node.size()], dtype=float32)
    dist[0, 0, :] = [0.95, 0.05]
    dist[0, 1, :] = [0.15, 0.85]
    dist[1, 0, :] = [0.95, 0.05]
    dist[1, 1, :] = [0.05, 0.95]
    booster_distribution = ConditionalDiscreteDistribution(nodes = [weather_node, oring_node, booster_node], table=dist)
    booster_node.set_dist(booster_distribution)

    dist = zeros([shuttle_node.size(), booster_node.size(), mission_node.size()], dtype=float32)
    dist[0, 0, :] = [0.55, 0.45]
    dist[0, 1, :] = [0.2, 0.8]
    dist[1, 0, :] = [0.2, 0.8]
    dist[1, 1, :] = [0.05, 0.95]
    mission_distribution = ConditionalDiscreteDistribution(nodes = [shuttle_node, booster_node, mission_node], table=dist)
    mission_node.set_dist(mission_distribution)

    return bayes_net

def get_A_given_E(bayes_net):
    E_node = bayes_net.get_node_by_name('exercise')
    A_node = bayes_net.get_node_by_name('assignment')

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[E_node] = True
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    assignment_prob = Q[index]
    return (assignment_prob)

def get_not_G_given_not_N(bayes_net):
    N_node = bayes_net.get_node_by_name('netflix')
    G_node = bayes_net.get_node_by_name('graduate')

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[N_node] = False
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([False], range(Q.nDims))
    graduate_prob = Q[index]
    return (graduate_prob)

def get_C_given_G_E(bayes_net):
    C_node = bayes_net.get_node_by_name('club')
    G_node = bayes_net.get_node_by_name('graduate')
    E_node = bayes_net.get_node_by_name('exercise')

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[G_node] = True
    engine.evidence[E_node] = True
    #engine.evidence[N_node] = False
    Q = engine.marginal(C_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    club_prob = Q[index]
    return (club_prob)

def get_E_given_G_N(bayes_net):
    E_node = bayes_net.get_node_by_name('exercise')
    G_node = bayes_net.get_node_by_name('graduate')
    N_node = bayes_net.get_node_by_name('netflix')

    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[G_node] = True
    engine.evidence[N_node] = True
    #engine.evidence[N_node] = False
    Q = engine.marginal(E_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    exercise_prob = Q[index]
    return (exercise_prob)

def get_mission_prob_badbooster(bayes_net):
    """Calculate the conditional probability
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    #raise NotImplementedError
    weather_node = bayes_net.get_node_by_name('weather')
    mission_node = bayes_net.get_node_by_name('mission')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[weather_node] = False
    Q = engine.marginal(mission_node)[0]
    index = Q.generate_index([False], range(Q.nDims))
    mission_prob = Q[index]
    print(mission_prob)


def get_mission_prob_goodbooster(bayes_net):
    """Calculate the marginal
    probability of the alarm
    ringing in the
    power plant system."""
    # TODO: finish this function
    #raise NotImplementedError
    mission_node = bayes_net.get_node_by_name('mission')
    #T_node = bayes_net.get_node_by_name('temperature')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(mission_node)[0]
    index = Q.generate_index([False], range(Q.nDims))
    mission_prob = Q[index]
    print(mission_prob)


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""    
    A_node = bayes_net.get_node_by_name("alarm")
    F_A_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    F_G_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, F_A_node, G_node, F_G_node, T_node]
    # TODO: set the probability distribution for each node
    #raise NotImplementedError
    dist = zeros([G_node.size(), F_A_node.size(), A_node.size()], dtype=float32)
    dist[0, 0, :] = [0.9, 0.1]
    dist[0, 1, :] = [0.55, 0.45]
    dist[1, 0, :] = [0.1, 0.9]
    dist[1, 1, :] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes = [G_node, F_A_node, A_node], table=dist)
    A_node.set_dist(A_distribution)

    F_A_distribution = DiscreteDistribution(F_A_node)
    index = F_A_distribution.generate_index([], [])
    F_A_distribution[index] = [0.85, 0.15]
    F_A_node.set_dist(F_A_distribution)

    dist = zeros([T_node.size(), F_G_node.size(), G_node.size()], dtype=float32)
    dist[0, 0, :] = [0.95, 0.05]
    dist[0, 1, :] = [0.2, 0.8]
    dist[1, 0, :] = [0.05, 0.95]
    dist[1, 1, :] = [0.8, 0.2]
    G_distribution = ConditionalDiscreteDistribution(nodes = [T_node, F_G_node, G_node], table=dist)
    G_node.set_dist(G_distribution)

    dist = zeros([T_node.size(), F_G_node.size()], dtype=float32)  # P(FG|T)
    dist[0, :] = [0.95, 0.05] # Probability of FG when T is FALSE
    dist[1, :] = [0.2, 0.8] # Probability of FG when T is TRUE
    F_G_distribution = ConditionalDiscreteDistribution(nodes = [T_node, F_G_node], table = dist)
    F_G_node.set_dist(F_G_distribution)

    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([], [])
    T_distribution[index] = [0.8, 0.2]
    T_node.set_dist(T_distribution)

    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    #raise NotImplementedError
    A_node = bayes_net.get_node_by_name('alarm')
    #T_node = bayes_net.get_node_by_name('temperature')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    alarm_prob = Q[index]
    return alarm_prob


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    #raise NotImplementedError
    G_node = bayes_net.get_node_by_name('gauge')
    #T_node = bayes_net.get_node_by_name('temperature')
    #F_G_node = bayes_net.get_node_by_name('faulty gauge')
    engine = JunctionTreeEngine(bayes_net)
    #engine.evidence[T_node] = True
    #engine.evidence[F_G_node] = True
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    gauge_prob = Q[index]
    return gauge_prob

def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    #raise NotImplementedError
    A_node = bayes_net.get_node_by_name('alarm')
    T_node = bayes_net.get_node_by_name('temperature')
    F_G_node = bayes_net.get_node_by_name('faulty gauge')
    F_A_node = bayes_net.get_node_by_name('faulty alarm')
    engine = JunctionTreeEngine(bayes_net)
    engine.evidence[A_node] = True
    engine.evidence[F_G_node] = False
    engine.evidence[F_A_node] = False
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    temp_prob = Q[index]
    return temp_prob

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    # TODO: fill this out
    #raise NotImplementedError
    A_node = BayesNode(0, 4, name = 'A')
    B_node = BayesNode(1, 4, name = 'B')
    C_node = BayesNode(2, 4, name = 'C')
    AvB_node = BayesNode(3, 3, name = 'AvB')
    BvC_node = BayesNode(4, 3, name = 'BvC')
    CvA_node = BayesNode(5, 3, name = 'CvA')
    nodes.extend([A_node, B_node, C_node, AvB_node, BvC_node, CvA_node])

    AvB_node.add_parent(A_node)
    AvB_node.add_parent(B_node)
    A_node.add_child(AvB_node)
    B_node.add_child(AvB_node)

    BvC_node.add_parent(B_node)
    BvC_node.add_parent(C_node)
    B_node.add_child(BvC_node)
    C_node.add_child(BvC_node)

    CvA_node.add_parent(A_node)
    CvA_node.add_parent(C_node)
    A_node.add_child(CvA_node)
    C_node.add_child(CvA_node)

    A_distribution = DiscreteDistribution(A_node)
    index = A_distribution.generate_index([], [])
    A_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    A_node.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B_node)
    index = B_distribution.generate_index([], [])
    B_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    B_node.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C_node)
    index = C_distribution.generate_index([], [])
    C_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    C_node.set_dist(C_distribution)


    dist = zeros([A_node.size(), B_node.size(), AvB_node.size()], dtype=float32)
    dist[0, 0, :] = [0.1, 0.1, 0.8]
    dist[0, 1, :] = [0.2, 0.6, 0.2]
    dist[0, 2, :] = [0.15, 0.75, 0.1]
    dist[0, 3, :] = [0.05, 0.9, 0.05]
    dist[1, 0, :] = [0.6, 0.2, 0.2]
    dist[1, 1, :] = [0.1, 0.1, 0.8]
    dist[1, 2, :] = [0.2, 0.6, 0.2]
    dist[1, 3, :] = [0.15, 0.75, 0.1]
    dist[2, 0, :] = [0.75, 0.15, 0.1]
    dist[2, 1, :] = [0.6, 0.2, 0.2]
    dist[2, 2, :] = [0.1, 0.1, 0.8]
    dist[2, 3, :] = [0.2, 0.6, 0.2]
    dist[3, 0, :] = [0.9, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.1]
    dist[3, 2, :] = [0.6, 0.2, 0.2]
    dist[3, 3, :] = [0.1, 0.1, 0.8]
    AvB_distribution = ConditionalDiscreteDistribution(nodes = [A_node, B_node, AvB_node], table=dist)
    AvB_node.set_dist(AvB_distribution)

    dist = zeros([B_node.size(), C_node.size(), BvC_node.size()], dtype=float32)
    dist[0, 0, :] = [0.1, 0.1, 0.8]
    dist[0, 1, :] = [0.2, 0.6, 0.2]
    dist[0, 2, :] = [0.15, 0.75, 0.1]
    dist[0, 3, :] = [0.05, 0.9, 0.05]
    dist[1, 0, :] = [0.6, 0.2, 0.2]
    dist[1, 1, :] = [0.1, 0.1, 0.8]
    dist[1, 2, :] = [0.2, 0.6, 0.2]
    dist[1, 3, :] = [0.15, 0.75, 0.1]
    dist[2, 0, :] = [0.75, 0.15, 0.1]
    dist[2, 1, :] = [0.6, 0.2, 0.2]
    dist[2, 2, :] = [0.1, 0.1, 0.8]
    dist[2, 3, :] = [0.2, 0.6, 0.2]
    dist[3, 0, :] = [0.9, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.1]
    dist[3, 2, :] = [0.6, 0.2, 0.2]
    dist[3, 3, :] = [0.1, 0.1, 0.8]
    BvC_distribution = ConditionalDiscreteDistribution(nodes = [B_node, C_node, BvC_node], table=dist)
    BvC_node.set_dist(BvC_distribution)

    dist = zeros([C_node.size(), A_node.size(), CvA_node.size()], dtype=float32)
    dist[0, 0, :] = [0.1, 0.1, 0.8]
    dist[0, 1, :] = [0.2, 0.6, 0.2]
    dist[0, 2, :] = [0.15, 0.75, 0.1]
    dist[0, 3, :] = [0.05, 0.9, 0.05]
    dist[1, 0, :] = [0.6, 0.2, 0.2]
    dist[1, 1, :] = [0.1, 0.1, 0.8]
    dist[1, 2, :] = [0.2, 0.6, 0.2]
    dist[1, 3, :] = [0.15, 0.75, 0.1]
    dist[2, 0, :] = [0.75, 0.15, 0.1]
    dist[2, 1, :] = [0.6, 0.2, 0.2]
    dist[2, 2, :] = [0.1, 0.1, 0.8]
    dist[2, 3, :] = [0.2, 0.6, 0.2]
    dist[3, 0, :] = [0.9, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.1]
    dist[3, 2, :] = [0.6, 0.2, 0.2]
    dist[3, 3, :] = [0.1, 0.1, 0.8]
    CvA_distribution = ConditionalDiscreteDistribution(nodes = [C_node, A_node, CvA_node], table=dist)
    CvA_node.set_dist(CvA_distribution)
    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0, 0, 0]
    # TODO: finish this function
    AvB_node = bayes_net.get_node_by_name('AvB')
    CvA_node = bayes_net.get_node_by_name('CvA')
    BvC_node = bayes_net.get_node_by_name('BvC')
    engine = EnumerationEngine(bayes_net)

    engine.evidence[AvB_node] = 0
    engine.evidence[CvA_node] = 2

    Q = engine.marginal(BvC_node)[0]
    index = Q.generate_index([], range(Q.nDims))

    posterior = Q[index]
    #print(posterior)

    return tuple(posterior) # list


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)    
    #print(sample)
    # TODO: finish this function
    #raise NotImplementedError
    if not initial_state or initial_state == []:
        sample = tuple([randint(0, 3), randint(0, 3), randint(0, 3), 0, randint(0, 2), 2])

    sample = list(sample)
    sample[3] = 0
    sample[5] = 2
    A = bayes_net.get_node_by_name("A")
    team_table_A = A.dist.table
    B = bayes_net.get_node_by_name("B")
    team_table_B = B.dist.table
    C = bayes_net.get_node_by_name("C")
    team_table_C = C.dist.table
    AvB = bayes_net.get_node_by_name("AvB")
    match_table_AvB = AvB.dist.table
    BvC = bayes_net.get_node_by_name("BvC")
    match_table_BvC = BvC.dist.table
    CvA = bayes_net.get_node_by_name("CvA")
    match_table_CvA = CvA.dist.table

    rand_idx = choice([0, 1, 2, 4])
    #print(rand_idx)
    sample = list(sample)

    num_team_value = len(team_table_A)
    num_match_value = len(match_table_AvB[0, 0, :])

    # When random index = 0 or team A
    if rand_idx == 0:
        post_prob = zeros(num_team_value, dtype=float32)
        for i in range(num_team_value):
            # Only consider relevant team and matches and skip other teams or matches
            # given that they doesn't matter after normalization
            post_prob[i] = team_table_A[i]*match_table_AvB[i, sample[1], sample[3]]*\
                match_table_CvA[sample[2], i, sample[5]]
        post_prob = post_prob/sum(post_prob)
        sample[rand_idx] = choice(num_team_value, p = post_prob)
    # When random index = 1 or team B
    elif rand_idx == 1:
        post_prob = zeros(num_team_value, dtype=float32)
        for i in range(num_team_value):
            # Only consider relevant team and matches and skip other teams or matches
            # given that they doesn't matter after normalization
            post_prob[i] = team_table_B[i]*match_table_BvC[i, sample[2], sample[4]]*\
                match_table_AvB[sample[0], i, sample[3]]
        post_prob = post_prob / sum(post_prob)
        sample[rand_idx] = choice(num_team_value, p = post_prob)
    # When random index = 2 or team C
    elif rand_idx == 2:
        post_prob = zeros(num_team_value, dtype=float32)
        for i in range(num_team_value):
            # Only consider relevant team and matches and skip other teams or matches
            # given that they doesn't matter after normalization
            post_prob[i] = team_table_C[i]*match_table_CvA[i, sample[0], sample[5]]*\
                match_table_BvC[sample[1], i, sample[4]]
        post_prob = post_prob / sum(post_prob)
        sample[rand_idx] = choice(num_team_value, p = post_prob)
    # When random index = 4 or match BvC
    else:
        # Only consider relevant team and matches and skip other teams or matches
        # given that they doesn't matter after normalization
        # post prob of BvC will be the match_table given sample[1] (team B) and sample[2](team C)
        post_prob = match_table_BvC[sample[1], sample[2], :]
        sample[rand_idx] = choice(num_match_value, p = post_prob)

    #print(sample)
    return tuple(sample)

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A= bayes_net.get_node_by_name("A")      
    AvB= bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_state)    
    # TODO: finish this function
    if not initial_state or initial_state == []:
        sample = tuple([randint(0, 3), randint(0, 3), randint(0, 3), 0, randint(0, 2), 2])

    cand_state = [randint(0, 3), randint(0, 3), randint(0, 3), 0, randint(0, 2), 2]

    prob_initial = team_table[initial_state[0]]*team_table[initial_state[1]]*team_table[initial_state[2]]* \
        match_table[initial_state[0], initial_state[1], initial_state[3]]* \
        match_table[initial_state[1], initial_state[2], initial_state[4]]* \
        match_table[initial_state[2], initial_state[0], initial_state[5]]
    prob_cand = team_table[cand_state[0]] * team_table[cand_state[1]] * team_table[cand_state[2]] * \
        match_table[cand_state[0], cand_state[1], cand_state[3]] * \
        match_table[cand_state[1], cand_state[2], cand_state[4]] * \
        match_table[cand_state[2], cand_state[0], cand_state[5]]

    ratio = 1 if prob_cand/prob_initial > 1 else prob_cand/prob_initial

    prob_rand = random()

    if ratio < prob_rand:
        return tuple(initial_state)
    else:
        return tuple(cand_state)

def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    if not initial_state or initial_state == []:
        initial_state =[randint(0, 3), randint(0, 3), randint(0, 3), 0, randint(0, 2), 2]

    burn_in_count = 200000
    delta = 0.0001
    N = 100
    #print("step1")
    # Gibbs sampling
    sample_Gibbs = initial_state
    for i in range(burn_in_count):
        new_sample_Gibbs = Gibbs_sampler(bayes_net, sample_Gibbs)
        Gibbs_convergence[new_sample_Gibbs[4]] += 1
        Gibbs_count += 1
        sample_Gibbs = new_sample_Gibbs
    #print("step2")
    count = 0
    Gibbs_convergence_prob = np.asarray(Gibbs_convergence, dtype=float32) / sum(Gibbs_convergence)

    while count < N:
        max_diff = 0
        print(Gibbs_convergence_prob)
        new_sample_Gibbs = Gibbs_sampler(bayes_net, sample_Gibbs)
        Gibbs_convergence[new_sample_Gibbs[4]] += 1
        Gibbs_count += 1
        Gibbs_convergence_prob_new = np.asarray(Gibbs_convergence, dtype=float32) / sum(Gibbs_convergence)
        sample_Gibbs = new_sample_Gibbs
        for i in range(3):
            max_diff = max(max_diff, abs(Gibbs_convergence_prob_new[i] - Gibbs_convergence_prob[i]))
        if max_diff > delta:
            count = 0
        elif max_diff < delta:
            count += 1
        Gibbs_convergence_prob = Gibbs_convergence_prob_new


    #print("step3")
    # MH sampling
    sample_MH = initial_state
    for i in range(burn_in_count):
        new_sample_MH = MH_sampler(bayes_net, sample_MH)
        MH_convergence[new_sample_MH[4]] += 1
        MH_count += 1
        if new_sample_MH == sample_MH:
            MH_rejection_count += 1
        sample_MH = new_sample_MH
    #print("step4")
    count = 0
    MH_convergence_prob = np.asarray(MH_convergence, dtype=float32) / sum(MH_convergence)

    while count < N:
        max_diff = 0
        print(MH_convergence_prob)
        new_sample_MH = MH_sampler(bayes_net, sample_MH)
        MH_convergence[new_sample_MH[4]] += 1
        MH_count += 1
        if new_sample_MH == sample_MH:
            MH_rejection_count += 1
        MH_convergence_prob_new = np.asarray(MH_convergence, dtype=float32) / sum(MH_convergence)
        sample_MH = new_sample_MH
        for i in range(3):
            max_diff = max(max_diff, abs(MH_convergence_prob_new[i] - MH_convergence_prob[i]))
        if max_diff > delta:
            count = 0
        elif max_diff < delta:
            count += 1
        MH_convergence_prob = MH_convergence_prob_new

    return tuple([np.asarray(Gibbs_convergence, dtype=float32)/sum(Gibbs_convergence),
                 np.asarray(MH_convergence, dtype=float32)/sum(MH_convergence), Gibbs_count, MH_count,
                 MH_rejection_count])


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Xiangnan He"
    #raise NotImplementedError

if __name__ ==  "__main__":
    net1 = make_cargo_flight_net()
    net1 = set_probability_cargo_flight_badbooster(net1)
    get_mission_prob_badbooster(net1)

    net2 = make_cargo_flight_net()
    net2 = set_probability_cargo_flight_goodbooster(net2)
    get_mission_prob_goodbooster(net2)