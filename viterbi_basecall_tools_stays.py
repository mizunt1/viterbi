"""
viterbi algorithm for basecalling, compatible with different kmers
which incoporates the existence of stays but does not classify
stays as a state, but rather replaces stays with the kmer which came before.
The viterbi function produces a dynamic programming table and a backtrace
table. The first row of the backtrace table indicates the branch which
started with AAAAA. i.e. there are no stay states in the backtrace table.
"""
import sys
from utils import number_to_kmer_nostay

# the transition function outputs integers depending on whether the output
# is not allowed, stay, skip or step. These are translated in to probabilities
# using a dictionary

transition_dict = {-1: 0, 0: 0.01, 1: 0.01, 2: 0.1}


def viterbi(posterior, transition_func, transition_dict=transition_dict,
            k=5, norm_interval=4):
    """
    input 2d posterior probabilities i.e. (1025, 1600) and the transition
    function which determines the transition probability between two kmers.
    Forms a dynamic programming table and a back trace table from the
    probabilities.
    :param posterior: posterior probabilities
    :param transition_func: function which determines transition probabilities
    between kmers
    :param transition_dict: dictionary matching output of transition function
    to probabilities
    :param k: k for kmer
    :param norm_interval: after a certain number of events, the dynamic
    programming table will be normalised.
    :type posterior: numpy array / list
    :type transition_func: python function
    :type transition_dict: dictionary
    :type k: int
    :type norm_interval: int
    returns dynamic programming table, backtrace table and transitions table.
    N.B. the first row of the backtrace table will represent the branch which
    started with the AAAAA kmer, and the value of 1 in the backtrace table will
    represent the first kmer, i.e. AAAAA. Stays are not considered a state,
    and therefore there will be no zeros in the table. Empty slots on the table
    will be indicated by None to minimise confusion with stays.
    Each element of the transition table shows the transition needed to
    transition in to that state. Neccessary for stringing together kmers. i.e.
    T[1][2] will indicate the transition needed to get to the state b[1][2].
    """
    states = len(posterior)
    events = len(posterior[0])
    V = [[0 for i in range(events)] for j in range(states - 1)]
    B = [[None for i in range(events-1)] for j in range(states - 1)]
    T = [[None for i in range(events-1)] for j in range(states - 1)]
    for st in range(1, states):
        # initialise first "day" of dp table
        V[st-1][0] = posterior[st][0]
        # posterior[0] : all the probabilities for A for three days
        # posterior[0][0]: probability for A for day 1.
        # V[0]: results for day 1
        # V[0][0] results for day 1 base 1
    for t in range(1, events):
        # for each "day"
        # st is the state we are transitioning in to
        for st in range(states):
            # look at a single branch of probabilities out of states.
            # (i.e. A C G T )
            # see for this "day" which path has maximum probability
            # for that state
            max_prob = 0
            prev_st_save = 0
            trans_save = None
            # prev_st is the transition we are transitioning from
            # cannot transition from a stay state therefore start at 1
            for prev_st in range(1, states):
                # find the correct st and prob to save in to the dpgraph
                trans_int = transition_func(prev_st, st, k=k)
                trans = transition_dict[trans_int]
                if trans != 0:
                    # note the first row of posterior represents stays
                    # the first row of B and V represents AA.
                    # hence [prev_st -1] when indexing V.
                    prob = V[prev_st-1][t-1] * trans
                    if prob > max_prob:
                        max_prob = prob
                        prev_st_save = prev_st
                        trans_save = trans_int
                    # initialy fill each possible stay state in column st.
                    # These will be overwritten if non stay states
                    # are more likely
                    # i.e. V[prev_st=AA][current t] =
                    #         prob * posterior[stay state][current t]
                    # the V[prev_st] is going to be the column where
                    # V[st] is, as this is the nature of a stay.
                    # prev_st = st. Hence for non stay state, we update
                    # V[st] but for stay we can update V[prev_st]
                    # repeat for all prev_st
                    if st == 0:
                        prob = prob * posterior[st][t]
                        V[prev_st-1][t] = prob
                        if prob != 0 and t < (events):
                            B[prev_st-1][t-1] = prev_st
                            T[prev_st-1][t-1] = trans_int
                else:
                    pass
            if st != 0:
                max_prob_until = max_prob * posterior[st][t]
                # if the non stay state transition probability is
                # larger than the stay state i.e.
                # GA --> AA is greater than
                # GA --> AA (stay)
                # then replace V[st -1][t] with the non stay probability
                if max_prob_until > V[st - 1][t]:
                    V[st - 1][t] = max_prob_until
                    if t < (events):
                        B[st - 1][t-1] = prev_st_save
                        T[st-1][t-1] = trans_save

        if t % norm_interval == 0:
            sum_is = 0.0
            for i in range(states-1):
                sum_is += V[i][t]
            try:
                for i in range(states-1):
                    V[i][t] = (V[i][t])/sum_is
            except ZeroDivisionError:
                print(
                    "impossible transition in posteriors or underflow, around event", t)
                sys.exit(1)
    return V, B, T


def determine_path(V, B, T):
    """
    given a dynamic programming table, a backtrace table and a
    transition table and returns
    the most likely path using the viterbi algoithm.
    The backtrace table must be formatted so that the first row of the
    table is not stays but AAA etc. A value of 1 in the backtrace
    table is AAA.
    :param V: dynamic programming table.
    :param B: backtrace table
    :param T: transition table
    :type V: 2d list
    :type B: 2d list
    :type T: 2d list
    returns most likely path of kmers
    """
    events = len(B[0])
    last_prob = [V[i][-1] for i in range(len(V))]
    # list of final values of probabilities
    kmer_max_prob = last_prob.index(max(last_prob))
    likely_path = [kmer_max_prob + 1]
    # index of the maximum probability i.e. if
    # kmer_max_prob = 1 > AAA was last base in path
    # this is G. ie. 2.
    transition = [T[kmer_max_prob][events-1]]
    row_of_interest = B[kmer_max_prob]
    for t in range(1, events + 1):
        previous_kmer = row_of_interest[-1*t]
        likely_path.insert(0, previous_kmer)
        if t < events:
            transition.insert(0, T[previous_kmer-1][(-1*t) - 1])
        try:
            row_of_interest = B[previous_kmer-1]
        except TypeError:
            print(
                "None values in backtrace table due to impossible\
                transitions in posteriors")
            sys.exit(1)
    return likely_path, transition


def transition(first, second, k=5):
    """
    outputs an integer depending on whether the transition is a stay (0),
    skip (1), step (2) or the transition is forbidden (-1).
    These transitions must then be changed in to probabilities by a
    transition dictionary.
    N.B. This probability does not have to be normalised because
    we are looking at a relative probability, as the maximum will
    be taken from the probabilities.
    rough estimate of transition probabilities. Seems to work
    relatively well.
    """
    first_num = first - 1
    second_num = second - 1
    # If first num is a stay, transition is 0.1 likely
    # NB cannot transition from a stay

    # steps
    if (first_num & (2**((k*2)-2) - 1)) == (second_num >> 2):
        return 2
    # skips
    elif (first_num & (2**((k*2)-4) - 1)) == (second_num >> 4):
        return 1
    # stays
    elif second_num == -1:
        return 0

    else:
        return -1


def stitch_kmers(kmers, transitions, k=5):
    """
    This function returns a single string of a dna sequence from a series
    of kmers.
    """
    sequence = kmers[0]-1
    # seq_len in bases
    seq_len = k
    for i in range(len(kmers)-1):
        # step
        if transitions[i] == 2:
            # mask the last letter of kmer
            # pretend that kmer does not account for stays
            last_added = (kmers[i+1] - 1) & 3
            # move the sequence up two bits
            sequence = (sequence << 2) + last_added
            seq_len += 1
        # skip
        if transitions[i] == 1:
            # mask the last two letters of kmer
            last_added = (kmers[i+1] - 1) & 15
            sequence = (sequence << 4) + last_added
            seq_len += 2
        # stay
        if transitions[i] == 0:
            pass
        string_seq = number_to_kmer_nostay(sequence, k=seq_len)
    return string_seq
