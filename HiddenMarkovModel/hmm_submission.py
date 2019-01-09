import numpy as np


def part_1_a():
    """Provide probabilities for the letter HMMs outlined below.

    Letters Y and Z.

    See README.md for example probabilities for the letter A.
    See README.md for expected HMMs probabilities.
    See README.md for tuple of states.

    Returns:
        ( prior probabilities for all states for letter Y,
          transition probabilities between states for letter Y,
          emission probabilities for all states for letter Y,
          prior probabilities for all states for letter Z,
          transition probabilities between states for letter Z,
          emission probabilities for all states for letter Z )

        Sample Format (not complete):

        ( {'Y1': prob_of_starting_in_Y1, ...},
          {'Y1': {'Y1': prob_of_transition_from_Y1_to_Y1,
                  'Y2': prob_of_transition_from_Y1_to_Y2}, ...},
          {'Y1': [prob_of_observing_0, prob_of_observing_1], ...},
          {'Z1': prob_of_starting_in_Z1, ...},
          {'Z1': {'Z1': prob_of_transition_from_Z1_to_Z1,
                  'Z2': prob_of_transition_from_Z1_to_Z2}, ...},
          {'Z1': [prob_of_observing_0, prob_of_observing_1], ...} )
    """

    # TODO: complete this function.
    # raise NotImplemented()

    """Letter Y"""
    # prior probabilities for all states for letter Y
    y_prior_probs = {'Y1': 1.0,
                     'Y2': 0.0,
                     'Y3': 0.0,
                     'Y4': 0.0,
                     'Y5': 0.0,
                     'Y6': 0.0,
                     'Y7': 0.0,
                     'Yend': 0.0

                     }

    # transition probabilities between states for letter Y
    y_transition_probs = {'Y1': {'Y1': 0.667,
                                 'Y2': 0.333,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y2': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 1.0,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y3': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 1.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y4': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 1.0,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y5': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.667,
                                 'Y6': 0.333,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y6': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 1.0,
                                 'Yend': 0.0},
                          'Y7': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.667,
                                 'Yend': 0.333},
                          'Yend': {'Y1': 0.0,
                                   'Y2': 0.0,
                                   'Y3': 0.0,
                                   'Y4': 0.0,
                                   'Y5': 0.0,
                                   'Y6': 0.0,
                                   'Y7': 0.0,
                                   'Yend': 1.0},

                          }

    # emission probabilities for all states for letter Y
    y_emission_probs = {'Y1': [0.0, 1.0],
                        'Y2': [1.0, 0.0],
                        'Y3': [0.0, 1.0],
                        'Y4': [1.0, 0.0],
                        'Y5': [0.0, 1.0],
                        'Y6': [1.0, 0.0],
                        'Y7': [0.0, 1.0],
                        'Yend': [0.0, 0.0],

                        }

    """Letter Z"""
    # prior probabilities for all states for letter Z
    z_prior_probs = {'Z1': 1.0,
                     'Z2': 0.0,
                     'Z3': 0.0,
                     'Z4': 0.0,
                     'Z5': 0.0,
                     'Z6': 0.0,
                     'Z7': 0.0,
                     'Zend': 0.0
                     }

    # transition probabilities between states for letter Z
    z_transition_probs = {'Z1': {'Z1': 0.667,
                                 'Z2': 0.333,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z2': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 1.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z3': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.667,
                                 'Z4': 0.333,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z4': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 1.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z5': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 1.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z6': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 1.0,
                                 'Zend': 0.0},
                          'Z7': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 1.0},
                          'Zend': {'Z1': 0.0,
                                   'Z2': 0.0,
                                   'Z3': 0.0,
                                   'Z4': 0.0,
                                   'Z5': 0.0,
                                   'Z6': 0.0,
                                   'Z7': 0.0,
                                   'Zend': 1.0},

                          }

    # emission probabilities for all states for letter Z
    z_emission_probs = {'Z1': [0.0, 1.0],
                        'Z2': [1.0, 0.0],
                        'Z3': [0.0, 1.0],
                        'Z4': [1.0, 0.0],
                        'Z5': [0.0, 1.0],
                        'Z6': [1.0, 0.0],
                        'Z7': [0.0, 1.0],
                        'Zend': [0.0, 0.0],
                        }

    return (y_prior_probs, y_transition_probs, y_emission_probs,
            z_prior_probs, z_transition_probs, z_emission_probs)


def viterbi(evidence_vector, states, prior_probs, transition_probs,
            emission_probs):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list(int)): List of 0s (Silence) or 1s (Dot/Dash).
            example: [1, 0, 1, 1, 1]
        states (list(string)): List of all states.
            example: ['A1', 'A2', 'A3', 'Aend']
        prior_probs (dict): prior distribution for each state.
            example: {'A1'  : 1.0,
                      'A2'  : 0.0,
                      'A3'  : 0.0,
                      'Aend': 0.0}
        transition_probs (dict): dictionary representing transitions from
            each state to every other state, including self.
            example: {'A1'  : {'A1'  : 0.0,
                               'A2'  : 1.0,
                               'A3'  : 0.0,
                               'Aend': 0.0},
                      'A2'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 1.0,
                               'Aend': 0.0},
                      'A3'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.667,
                               'Aend': 0.333},
                      'Aend': {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.0,
                               'Aend': 1.0}}
        emission_probs (dict): dictionary of probabilities of outputs from
            each state.
            example: {'A1'  : [0.0, 1.0],
                      'A2'  : [1.0, 0.0],
                      'A3'  : [0.0, 1.0],
                      'Aend': [0.0, 0.0]}

    Returns:
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

        Example:
            ( ['A1', 'A2', 'A3', 'A3', 'A3'],
              0.445 )
    """
    # TODO: complete this function.
    # raise NotImplemented()

    num_states = len(states)
    num_observations = len(evidence_vector)

    sequence = list([None] * num_observations)
    probability = 0.0

    # T1 stores the max probability
    T1 = np.empty((num_states, num_observations))

    # T2 stores the max probable step taken in previous step
    T2 = np.empty((num_states, num_observations))

    # Initialize T1 and T2 matrices
    for i in range(num_states):
        T1[i, 0] = prior_probs[states[i]] * emission_probs[states[i]][evidence_vector[0]]
        T2[i, 0] = 0.0

    # Update T1 and T2 by looping over the evidence vector
    for i in range(1, num_observations):
        for j, state in enumerate(states):
            max_prob = -1
            max_prob_idx = -1

            # Loop over all the previous states to current state transition probability to find the max prob
            for k in range(num_states):
                # check if transition probability contains states
                if states[k] in transition_probs and states[j] in transition_probs[states[k]]:
                    curr_prob = T1[k, i - 1] * transition_probs[states[k]][states[j]] * \
                                emission_probs[states[j]][evidence_vector[i]]
                else:
                    curr_prob = 0.0
                if max_prob < curr_prob:
                    max_prob = curr_prob
                    max_prob_idx = k



            # Update T1 and T2 with the max prob
            T1[j, i] = max_prob
            T2[j, i] = max_prob_idx

    # Find state index for the last max probable evidence
    Z = np.empty(num_observations, dtype=int)
    Z[num_observations - 1] = np.argmax(T1[:, num_observations - 1])

    # Fill in the last max probably state in sequence
    sequence[num_observations - 1] = states[Z[num_observations - 1]]

    # Filling in the sequence path from the end
    for i in range(num_observations - 1, 0, -1):
        Z[i - 1] = T2[Z[i], i]
        sequence[i - 1] = states[Z[i - 1]]

    probability = np.max(T1[:, num_observations - 1])
    # Return if max probability is very close to 0
    #if abs(probability - 0.0) <= 1.0e-15:
    if probability == 0.0:
        return [], 0.0

    return sequence, probability


def part_2_a():
    """Provide probabilities for the NOISY letter HMMs outlined below.

    Letters A, Y, Z, letter pause, word space

    See README.md for example probabilities for the letter A.
    See README.md for expected HMMs probabilities.

    Returns:
        ( list of all states for letter A,
          prior probabilities for all states for letter A,
          transition probabilities between states for letter A,
          emission probabilities for all states for letter A,
          list of all states for letter Y,
          prior probabilities for all states for letter Y,
          transition probabilities between states for letter Y,
          emission probabilities for all states for letter Y,
          list of all states for letter Z,
          prior probabilities for all states for letter Z,
          transition probabilities between states for letter Z,
          emission probabilities for all states for letter Z,
          list of all states for letter pause,
          prior probabilities for all states for letter pause,
          transition probabilities between states for letter pause,
          emission probabilities for all states for letter pause,
          list of all states for word space,
          prior probabilities for all states for word space,
          transition probabilities between states for word space,
          emission probabilities for all states for word space )

        Sample Format (not complete):

        ( ['A1', ...],
          ['A1': prob_of_starting_in_A1, ...],
          {'A1': {'A1': prob_of_transition_from_A1_to_A1,
                  'A2': prob_of_transition_from_A1_to_A2}, ...},
          {'A1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['Y1', ...],
          ['Y1': prob_of_starting_in_Y1, ...],
          {'Y1': {'Y1': prob_of_transition_from_Y1_to_Y1,
                  'Y2': prob_of_transition_from_Y1_to_Y2}, ...},
          {'Y1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['Z1', ...],
          ['Z1': prob_of_starting_in_Z1, ...],
          {'Z1': {'Z1': prob_of_transition_from_Z1_to_Z1,
                  'Z2': prob_of_transition_from_Z1_to_Z2}, ...},
          {'Z1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['L1', ...]
          ['L1': prob_of_starting_in_L1, ...],
          {'L1': {'L1': prob_of_transition_from_L1_to_L1,
                  'L2': prob_of_transition_from_L1_to_L2}, ...},
          {'L1': [prob_of_observing_0, prob_of_observing_1], ...},
          ['W1', ...]
          ['W1': prob_of_starting_in_W1, ...],
          {'W1': {'W1': prob_of_transition_from_W1_to_W1,
                  'W2': prob_of_transition_from_W1_to_W2}, ...},
          {'W1': [prob_of_observing_0, prob_of_observing_1], ...} )
        """

    # TODO: complete this function.
    # raise NotImplemented()

    """Letter A"""
    # expected states names for letter A
    a_states = ['A1',
                'A2',
                'A3',
                'Aend'

                ]

    # prior probabilities for all states for letter A
    a_prior_probs = {'A1': 0.333,
                     'A2': 0.0,
                     'A3': 0.0,
                     'Aend': 0.0

                     }

    # transition probabilities between states for letter A
    a_transition_probs = {'A1': {'A1': 0.2,
                                 'A2': 0.8,
                                 'A3': 0.0,
                                 'Aend': 0.0},
                          'A2': {'A1': 0.0,
                                 'A2': 0.2,
                                 'A3': 0.8,
                                 'Aend': 0.0},
                          'A3': {'A1': 0.0,
                                 'A2': 0.0,
                                 'A3': 0.667,
                                 'Aend': 0.111,
                                 'L1': 0.111,
                                 'W1': 0.111},
                          'Aend': {'A1': 0.0,
                                   'A2': 0.0,
                                   'A3': 0.0,
                                   'Aend': 1.0}

                          }

    # emission probabilities for all states for letter A
    a_emission_probs = {'A1': [0.2, 0.8],
                        'A2': [0.8, 0.2],
                        'A3': [0.2, 0.8],
                        'Aend': [0.0, 0.0]
                        }

    """Letter Y"""
    # expected states names for letter Y
    y_states = ['Y1',
                'Y2',
                'Y3',
                'Y4',
                'Y5',
                'Y6',
                'Y7',
                'Yend'

                ]

    # prior probabilities for all states for letter Y
    y_prior_probs = {'Y1': 0.333,
                     'Y2': 0.0,
                     'Y3': 0.0,
                     'Y4': 0.0,
                     'Y5': 0.0,
                     'Y6': 0.0,
                     'Y7': 0.0,
                     'Yend': 0.0

                     }

    # transition probabilities between states for letter Y
    y_transition_probs = {'Y1': {'Y1': 0.667,
                                 'Y2': 0.333,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y2': {'Y1': 0.0,
                                 'Y2': 0.2,
                                 'Y3': 0.8,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y3': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.2,
                                 'Y4': 0.8,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y4': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.2,
                                 'Y5': 0.8,
                                 'Y6': 0.0,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y5': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.667,
                                 'Y6': 0.333,
                                 'Y7': 0.0,
                                 'Yend': 0.0},
                          'Y6': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.2,
                                 'Y7': 0.8,
                                 'Yend': 0.0},
                          'Y7': {'Y1': 0.0,
                                 'Y2': 0.0,
                                 'Y3': 0.0,
                                 'Y4': 0.0,
                                 'Y5': 0.0,
                                 'Y6': 0.0,
                                 'Y7': 0.667,
                                 'Yend': 0.111,
                                 'L1': 0.111,
                                 'W1': 0.111},
                          'Yend': {'Y1': 0.0,
                                   'Y2': 0.0,
                                   'Y3': 0.0,
                                   'Y4': 0.0,
                                   'Y5': 0.0,
                                   'Y6': 0.0,
                                   'Y7': 0.0,
                                   'Yend': 1.0},

                          }

    # emission probabilities for all states for letter Y
    y_emission_probs = {'Y1': [0.2, 0.8],
                        'Y2': [0.8, 0.2],
                        'Y3': [0.2, 0.8],
                        'Y4': [0.8, 0.2],
                        'Y5': [0.2, 0.8],
                        'Y6': [0.8, 0.2],
                        'Y7': [0.2, 0.8],
                        'Yend': [0.0, 0.0],

                        }

    """Letter Z"""
    # expected states names for letter Z
    z_states = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Zend'

    ]

    # prior probabilities for all states for letter Z
    z_prior_probs = {'Z1': 0.333,
                     'Z2': 0.0,
                     'Z3': 0.0,
                     'Z4': 0.0,
                     'Z5': 0.0,
                     'Z6': 0.0,
                     'Z7': 0.0,
                     'Zend': 0.0

                     }

    # transition probabilities between states for letter Z
    z_transition_probs = {'Z1': {'Z1': 0.667,
                                 'Z2': 0.333,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z2': {'Z1': 0.0,
                                 'Z2': 0.2,
                                 'Z3': 0.8,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z3': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.667,
                                 'Z4': 0.333,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z4': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.2,
                                 'Z5': 0.8,
                                 'Z6': 0.0,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z5': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.2,
                                 'Z6': 0.8,
                                 'Z7': 0.0,
                                 'Zend': 0.0},
                          'Z6': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.2,
                                 'Z7': 0.8,
                                 'Zend': 0.0},
                          'Z7': {'Z1': 0.0,
                                 'Z2': 0.0,
                                 'Z3': 0.0,
                                 'Z4': 0.0,
                                 'Z5': 0.0,
                                 'Z6': 0.0,
                                 'Z7': 0.2,
                                 'Zend': 0.267,
                                 'L1': 0.267,
                                 'W1': 0.267},
                          'Zend': {'Z1': 0.0,
                                   'Z2': 0.0,
                                   'Z3': 0.0,
                                   'Z4': 0.0,
                                   'Z5': 0.0,
                                   'Z6': 0.0,
                                   'Z7': 0.0,
                                   'Zend': 1.0},

                          }

    # emission probabilities for all states for letter Z
    z_emission_probs = {'Z1': [0.2, 0.8],
                        'Z2': [0.8, 0.2],
                        'Z3': [0.2, 0.8],
                        'Z4': [0.8, 0.2],
                        'Z5': [0.2, 0.8],
                        'Z6': [0.8, 0.2],
                        'Z7': [0.2, 0.8],
                        'Zend': [0.0, 0.0],

                        }

    """Pause between letters"""
    # expected states names for letter pause
    letter_pause_states = ['L1' ]

    # prior probabilities for all states for letter pause
    letter_pause_prior_probs = {'L1': 0.0

                                }

    # transition probabilities between states for letter pause
    letter_pause_transition_probs = {'L1': {'L1': 0.667,
                                            'A1': 0.111,
                                            'Y1': 0.111,
                                            'Z1': 0.111
                                            }

                                     }

    # emission probabilities for all states for letter pause
    letter_pause_emission_probs = {'L1': [0.8, 0.2]

    }

    """Space between words"""
    # expected states names for word space
    word_space_states = ['W1'

                         ]

    # prior probabilities for all states for word space
    word_space_prior_probs = {'W1': 0.0

                              }

    # transition probabilities between states for word space
    word_space_transition_probs = {'W1': {'W1': 0.857,
                                          'A1': 0.048,
                                          'Y1': 0.048,
                                          'Z1': 0.048}

                                   }

    # emission probabilities for all states for word space
    word_space_emission_probs = {'W1' : [0.8, 0.2]

    }

    return (a_states,
            a_prior_probs,
            a_transition_probs,
            a_emission_probs,
            y_states,
            y_prior_probs,
            y_transition_probs,
            y_emission_probs,
            z_states,
            z_prior_probs,
            z_transition_probs,
            z_emission_probs,
            letter_pause_states,
            letter_pause_prior_probs,
            letter_pause_transition_probs,
            letter_pause_emission_probs,
            word_space_states,
            word_space_prior_probs,
            word_space_transition_probs,
            word_space_emission_probs)


def quick_check():
    """Returns a few select values to check for accuracy.

    Returns:
        The following probabilities:
            ( prior probability of Z1,
              transition probability from Y7 to Y7,
              transition probability from Z3 to Z4,
              transition probability from W1 to W1,
              transition probability from L1 to Y1 )
    """

    # TODO: fill in the probabilities below where each shows None.
    # raise NotImplemented()

    part2a = part_2_a()

    # prior probability for Z1
    prior_prob_Z1 = part2a[9]['Z1']  # TODO

    # transition probability from Y7 to Y7
    transition_prob_Y7_Y7 = part2a[6]['Y7']['Y7']  # TODO

    # transition probability from Z3 to Z4
    transition_prob_Z3_Z4 = part2a[10]['Z3']['Z4']  # TODO

    # transition probability from W1 to W1
    transition_prob_W1_W1 = part2a[18]['W1']['W1']  # TODO

    # transition probability from L1 to Y1
    transition_prob_L1_Y1 = part2a[14]['L1']['Y1']  # TODO

    return (prior_prob_Z1,
            transition_prob_Y7_Y7,
            transition_prob_Z3_Z4,
            transition_prob_W1_W1,
            transition_prob_L1_Y1)


def part_2_b(evidence_vector, states, prior_probs, transition_probs,
             emission_probs):
    """Decode the most likely string generated by the evidence vector.

    Note: prior, states, transition_probs, and emission_probs will now contain
    all the letters, pauses, and spaces from part_2_a.

    For example, prior is now:

    prior_probs = {'A1'   : 0.333,
                   'A2'   : 0.0,
                   'A3'   : 0.0,
                   'Aend' : 0.0,
                   'Y1'   : 0.333,
                   'Y2'   : 0.0,
                   .
                   .
                   .
                   'Z1'   : 0.333,
                   .
                   .
                   .
                   'L1'  : 0.0,
                   'W1   : 0.0}

    Expect the same type of combinations for all probability and state input
    arguments.
l test case 1\npart_2_b runs successfully\nfail test case 2\npart_2_b runs successfully\nfail test case 3\npart_2_b runs successfully\nfail test case 4\n",
                "points_awarded": 20
    Essentially, the built Viterbi Trellis will contain all states for A, Y, Z,
    letter pause, and word space.

    Args:
        evidence_vector (list(int)): List of 0s (Silence) or 1s (Dot/Dash).
            example: [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
        states (list(string)): List of all states.
            example: ['A1', 'A2', 'A3', 'Aend']
        prior_probs (dict): prior distribution for each state.
            example: {'A1'  : 1.0,
                      'A2'  : 0.0,
                      'A3'  : 0.0,
                      'Aend': 0.0}
        transition_probs (dict): dictionary representing transitions from
            each state to every other state, including self.
            example: {'A1'  : {'A1'  : 0.0,
                               'A2'  : 1.0,
                               'A3'  : 0.0,
                               'Aend': 0.0},
                      'A2'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 1.0,
                               'Aend': 0.0},
                      'A3'  : {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.667,
                               'Aend': 0.333},
                      'Aend': {'A1'  : 0.0,
                               'A2'  : 0.0,
                               'A3'  : 0.0,
                               'Aend': 1.0}}
        emission_probs (dict): dictionary of probabilities of outputs from
            each state.
            example: {'A1'  : [0.0, 1.0],
                      'A2'  : [1.0, 0.0],
                      'A3'  : [0.0, 1.0],
                      'Aend': [0.0, 0.0]}

    Returns:
        ( A string that best fits the evidence,
          probability of that string being correct as a float. )

        For example:
            an evidence vector of [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]
            would return the String 'AA' with it's probability
    """
    sequence = ''
    probability = 0.0

    # TODO: complete this function.
    #raise NotImplemented()
    #print(evidence_vector, states, prior_probs, transition_probs, emission_probs)
    state_sequence, probability = viterbi(evidence_vector, states, prior_probs, transition_probs, emission_probs)
    #print(state_sequence, probability)
    if state_sequence is None or len(state_sequence) == 0:
        return [], 0.0

    def check_state(x):
        return{
            'A1': 'A',
            'A2': 'A',
            'A3': 'A',
            'Aend': 'A',
            'Y1': 'Y',
            'Y2': 'Y',
            'Y3': 'Y',
            'Y4': 'Y',
            'Y5': 'Y',
            'Y6': 'Y',
            'Y7': 'Y',
            'Yend': 'Y',
            'Z1': 'Z',
            'Z2': 'Z',
            'Z3': 'Z',
            'Z4': 'Z',
            'Z5': 'Z',
            'Z6': 'Z',
            'Z7': 'Z',
            'Zend': 'Z',
            'L1': '',
            'W1': ' '
        }[x]

    idx = 0
    '''
    while idx <= len(state_sequence)-1 and check_state(state_sequence[idx]) == ' ':
        idx += 1
    if idx >= len(state_sequence):
        return [], 0.0
    '''
    sequence += check_state(state_sequence[idx])

    for i in range(idx+1, len(state_sequence)):
        curr_state = check_state(state_sequence[i])
        if curr_state != check_state(state_sequence[i - 1]):
            sequence += curr_state

    sequence = sequence.strip()


    return sequence, probability
