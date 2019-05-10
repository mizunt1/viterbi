import time
import argparse
import numpy as np
from utils import load_training_data, number_to_kmer, get_posteriors, number_to_kmer_nostay
import viterbi_basecall_tools_basic as basic
import viterbi_basecall_tools_stays as stays
# A G T C posterior probabilities for three reads
posteriors = [
    [0.3, 0.2, 0.3], [0.1, 0.4, 0.2], [0.2, 0.2, 0.1], [0.4, 0.2, 0.4]]
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", type=str,
                    choices=["r", "e1", "e2", "e3", "e4"], default="e1",
                    dest="mode", help='list of options on examples',
                    required=True)

args = parser.parse_args()
mode = args.mode

posteriors_stays = np.zeros((17, 4))
posteriors_stays[1][0] = 0.9
posteriors_stays[0][1] = 0.9
posteriors_stays[1][1] = 0.05
posteriors_stays[2][1] = 0.05
posteriors_stays[2][3] = 0.1
posteriors_stays[0][2] = 0.9
posteriors_stays[0][3] = 0.9
posteriors_stays[5][2] = 0.1

number_of_posteriors = 100


def find_path(posteriors, labels):
    print("posteriors shape:", len(posteriors), ",", len(posteriors[0]))
    print("Determining viterbi path ...")
    start_time = time.time()
    number_of_posteriors = len(posteriors[0])
    V, B, T = stays.viterbi(posteriors, stays.transition, k=5, norm_interval=4)
    viterbi_path, transitions = stays.determine_path(V, B, T)
    end_time = time.time()
    print(
        "time taken to determine path: ", (end_time - start_time),
        "num posteriors: ", number_of_posteriors)
    return viterbi_path, transitions

if mode == "r":
    """
    Loads data from a hdf5 file, produces posteriors by running the data
    through a neural net, then puts the posteriors through a viterbi
    algorithm, then strings the bases together in to a basecall.
    """
    # load some data from a random part of the data set
    data, labels = load_training_data("../../r941_ch8000_5mer_stride5.h5", 10)

    print("labels shape", labels.shape)
    # put data through model and normalise
    # labels = labels[0][0:number_of_posteriors]
    labels = labels[0]

    print("data input shape and type", data.shape, type(data))

    model_path = "../catfish/trained_models/rgrgr_e_40_60000.pt"
    model_class_path = "../catfish/catfish/models/raw_rgrgr_mod_torch.py"
    posteriors = get_posteriors(data, model_class_path, model_path)
    # posteriors = posteriors[0][:, 0:number_of_posteriors]
    posteriors = posteriors[0]
    path, transitions = find_path(posteriors, labels)
    sequence = stays.stitch_kmers(path, transitions)

    print("final sequence:", sequence)

if mode == "e1":
    """
    this example does not incoporate stays, and uses artificial arbitrary
    posteriors defined at the top of this script along with arbitrary
    transition probabilities to produce a sequence of 1mers.
    """
    def transition_fake(first, second, k=1):
        if -4*(first // 3) + first + 1 == second:
            return 1
        else:
            return 0

    print("Determining viterbi path...")
    start_time = time.time()
    number_of_posteriors = len(posteriors[0])
    V, B = basic.viterbi(posteriors, transition_fake, k=2, norm_interval=1)
    path = basic.determine_path(V, B)
    end_time = time.time()
    print("DP table")
    for i in range(len(V)):
        print(V[i])
    print("backtrace table")
    for i in range(len(B)):
        print(B[i])
    print("time taken per posterior:",
          (end_time - start_time) / number_of_posteriors)
    print("number of posteriors:", number_of_posteriors)
    print(path)
    print("most likely path is:")
    print([number_to_kmer_nostay(i, k=1) for i in path])
    print("should be ['C', 'A', 'G']")


if mode == "e2":
    """
    This example incoporates stays and uses the arbitrary posterior
    probabilities and transition function to calculates the most likely
    path of 2mers.
    """
    print("example using stays")
    print(posteriors_stays)
    posteriors = posteriors_stays
    print("posteriors shape")
    print(posteriors.shape)
    number_of_posteriors = len(posteriors[0])
    print("Determining viterbi path ...")
    start_time = time.time()
    V, B, T = stays.viterbi(posteriors, stays.transition, k=5, norm_interval=1)
    print("DP table")
    for i in range(len(V)):
        print(V[i])
    print("backtrace table")
    for i in range(len(B)):
        print(B[i])
    print("transition table")
    for i in range(len(T)):
        print(T[i])

    viterbi_path, transition = stays.determine_path(V, B, T)
    end_time = time.time()
    print("calculated path: ", [number_to_kmer(i, k=2) for i in viterbi_path])
    print("should be ['AA', 'AA', 'AA', 'AC']")
    print(
        "time taken per posterior:",
        (end_time - start_time) / number_of_posteriors)
    print("number of posteriors", number_of_posteriors)

if mode == "e3":
    """
    A 5mer example using artificial posteriors, stitches together 5mers to
    create a basecall.
    """
    print("example using stays and artificial posteriors")
    posteriors = [[0 for i in range(8)] for i in range(1025)]
    # posteriors[states][time]
    posteriors[1][0] = 1.0 # AAAAA
    posteriors[0][1] = 1.0 # stay
    posteriors[3][2] = 1.0 # AAAAG
    posteriors[11][3] = 1.0 # AAAGG
    posteriors[42][4] = 1.0 # AAGGC
    posteriors[0][5] = 1.0 # stay
    posteriors[165][6] = 1.0 # AGGCA
    posteriors[582][7] = 1.0 # GCACC

    print("example using stays")
    print("posteriors used:")

    labels = ["AAAAA", "AAAAG", "AAAGG", "AAGGC", "AAGGC", "AGGCA", "GCACC"]
    labels = [1, 0, 3, 11, 42, 0, 165, 582]
    path, transitions = find_path(posteriors, labels)
    sequence = stays.stitch_kmers(path, transitions)

    print("final sequence:", sequence)

if mode == "e4":
    """
    similar to example 3 but with illegal transitions in posteriors.
    """
    print("example with illegal transitions in posteriors")
    posteriors = [[0 for i in range(7)] for i in range(1025)]
    # posteriors[states][time]
    posteriors[1][0] = 1.0 # AAAAA
    posteriors[0][1] = 1.0 # stay
    posteriors[3][2] = 1.0 # AAAAG
    posteriors[11][3] = 1.0 # AAAGG
    posteriors[683][4] = 1.0 # GGGGG
    posteriors[0][5] = 1.0 # stay
    posteriors[165][6] = 1.0 # AGGCA
    labels = [1, 0, 3, 11, 165, 0, 165]
    find_path(posteriors, labels)
