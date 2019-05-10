obs = ('normal', 'cold', 'dizzy', 'dizzy', 'dizzy')
states = ('Healthy', 'Fever', 'ok')
start_p = {'Healthy': 0.6, 'Fever': 0.3, 'ok':0.1 }
trans_p = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.2, 'ok': 0.1},
   'Fever' : {'Healthy': 0.3, 'Fever': 0.6, 'ok': 0.1},
    'ok' :  {'Healthy': 0.4, 'Fever': 0.1, 'ok': 0.5}
}
emit_p = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    'ok' : {'normal': 0.3, 'cold': 0.2, 'dizzy': 0.5}
   }


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    # v will be a list containing nested dictionaries.
    # pairs of healthy and fever on each day, and the previous state and the probability
    # of it having taken that path.
    # every time a pair of healthy and fever dictionary element is added, it is another day.
    print("start loop 1 \n")
    for st in states:
        # st is healthy or fever
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
        print("print emit_p[st][obs[0]]", emit_p[st][obs[0]], "st", st, "obs[0]", obs[0])
        print("v[0][st]")
        print(V[0][st])
        print("V")
        print(V)
    # Run Viterbi when t > 0
    print("for loop 2 \n")
    for t in range(1, len(obs)):
        V.append({})
        print("for loop 2.1 t:", t)
        print("\n")
        for st in states:
            print("prev_st")
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states)
            print("max_tr_prob for st:", st)
            print(max_tr_prob)
            print("values from which max was chosen")
            print([V[t-1][prev_st]["prob"]*trans_p[prev_st][st] for prev_st in states])
            print("for loop 2.2 \n")
            print("\n")
            for prev_st in states:
                print("if loop 2.3 prev_st:", prev_st)
                print("\n")
                if V[t-1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    print("V")
                    for i in V:
                        print(i)

                    break

    for line in dptable(V):
        print(line)
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print(
        'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

viterbi(obs,
        states,
        start_p,
        trans_p,
        emit_p)
