"""
using the example of the code on wikipedia to classify healthy and fever,
applied to bases. dp table and backprop table are saved in one dictionary
"""

def transition(first_kmer, second_kmer):
    """
    input kmers undergoing transition from first_kmer to second_kmer
    in numerical form, then output the probability of that
    transition occuring.
    """
    if -4*(first_kmer//3) + first_kmer + 1 == second_kmer:
        return 1
    else:
        return 0


def viterbi(posterior):
    """
    input 2d posterior probabilities
    """
    states = len(posterior)
    kmers = len(posterior[0])
    # states = posterior.shape[0]
    # i.e. for 5mkers, 2**5 states
    # kmers = posterior.shape[1]
    # number of "days"
    V = [{}]
    for st in range(states):
        # initialise first "day" of dp table
        V[0][st] = {"prob": posterior[st][0], "prev": None}
        # posterior[0] : all the probabilities for A for three days
        # posterior[0][0]: probability for A for day 1.
        # V[0]: results for day 1
        # V[0][0] results for day 1 base 1
    # transition probabilities now count for something
    for t in range(1, kmers):
        V.append({})
        for st in range(states):
            # look at a single branch of probabilities out of states.
            # (i.e. A G T C)
            # see for this "day" which path has maximum probability
            # for that state
            max_prob = max(
                V[t-1][prev_st]["prob"]*transition(prev_st, st) for prev_st in range(states)
            )
            # print("max chosen out of")
            # print(
            #   [V[t-1][prev_st]["prob"]*transition(prev_st, st) for prev_st in range(states)])
            # print([transition(prev_st, st) for prev_st in range(states)])
            for prev_st in range(states):
                # find the correct st and prob to save in to the dpgraph
                if V[t-1][prev_st]["prob"]*transition(prev_st, st) == max_prob:
                    max_prob_until = max_prob * posterior[st][t]
                    V[t][st] = {"prob":max_prob_until, "prev":prev_st}
                    break
    for line in dptable(V):
        print(line)
    for i in V:
        print(i)
    opt = []
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break

    for t in range(len(V) - 2, - 1, - 1):
        opt.insert(0, V[t+1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print(
        'The steps of states are ' + ' '.join(str(opt)) + ' with highest probability of %s' % max_prob)


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

posteriors = [
    [0.5, 0.1, 0.3], [0.3, 0.4, 0.2], [0.2, 0.2, 0.1], [0.1, 0.3, 0.4]]

viterbi(posteriors)
