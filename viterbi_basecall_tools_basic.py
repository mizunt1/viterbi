"""
Viterbi algorithm for basecalling compatible with different kmers
The algorythm does not account for stays. i.e. will classify stays in the
same way as any other state.
"""


def viterbi(posterior, transition_func, k=5, norm_interval=4):
    """
    input 2d posterior probabilities i.e. (1025, 1600) and the transition
    function which determines the transition probability between two kmers.
    Forms a dynamic programming table and a back trace table from the
    probabilities.
    :param posterior: posterior probabilities
    :param transition_func: function which determines transition probabilities
    :param k: k for kmer
    :param norm_interval: after a certain number of events, the dynamic

    between kmers
    :type posterior: numpy array / list
    :type transition_func: python function
    :type k: int
    :type norm_interval: int
   returns dynamic programming table, backtrace table
    """
    states = len(posterior)
    events = len(posterior[0])
    V = [[0 for i in range(events)] for j in range(states)]
    B = [[0 for i in range(events)] for j in range(states)]
    for st in range(states):
        # initialise first "day" of dp table
        V[st][0] = posterior[st][0]
        B[st][0] = st
        # posterior[0] : all the probabilities for A for three days
        # posterior[0][0]: probability for A for day 1.
        # V[0]: results for day 1
        # V[0][0] results for day 1 base 1
    for t in range(1, events):
        # for each "day"
        for st in range(states):
            # look at a single branch of probabilities out of states.
            # (i.e. A C G T )
            # see for this "day" which path has maximum probability
            # for that state
            max_prob = 0
            prev_st_save = 0
            for prev_st in range(states):
                # find the correct st and prob to save in to the dpgraph
                trans = transition_func(prev_st, st, k=k)
                if trans != 0:
                    prob = V[prev_st][t-1]*trans
                    if prob > max_prob:
                        max_prob = prob
                        prev_st_save = prev_st
                else:
                    pass

            max_prob_until = max_prob * posterior[st][t]
            V[st][t] = max_prob_until
            B[st][t] = prev_st_save

        if t % norm_interval == 0:
            sum = 0
            for i in range(states-1):

                sum += V[i][t]
            for i in range(states-1):
                V[i][t] = V[i][t]/sum

    return V, B


def determine_path(V, B):
    """
    given a dynamic programming table and a backtrace table, returns
    the most likely path using the viterbi algoithm.
    :param V: dynamic programming table.
    :param B: backtrace table
    :type V: 2d list
    :type B: 2d list
    returns most likely path of kmers
    """
    events = len(B[0])
    last_prob = [V[i][-1] for i in range(len(V))]
    # list of final values of probabilities
    kmer_max_prob = last_prob.index(max(last_prob))
    likely_path = [kmer_max_prob]
    # index of the maximum probability i.e. if
    # kmer_max_prob = 1 > AAA was last base in path
    # this is G. ie. 2.
    for t in range(1, events):
        kmer_of_interest = B[kmer_max_prob]
        previous_kmer = kmer_of_interest[-1*t]
        likely_path.insert(0, previous_kmer)
        kmer_max_prob = previous_kmer
    return likely_path


def transition(first, second, k=5):
    """
    calculates probability of transition from first to second
    step has probability of 1 and the rest is zero.
    N.B. This probability does not have to be normalised because
    we are looking at a relative probability, as the maximum will
    be taken from the probabilities.
    rough estimate of transition probabilities. Seems to work
    relatively well.
    """
    first_num = first - 1
    second_num = second - 1
    if second_num == -1:
        return 1
    # If first num is a stay, next number can be any
    elif first_num == -1:
        return 1
    elif (first_num & (2**((k*2)-2) - 1)) == (second_num >> 2):
        return 1
    else:
        return 0
