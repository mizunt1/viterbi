import h5py
import numpy as np
import types
import importlib.machinery
import torch
import torch.nn as nn


def number_to_kmer(number, k=5):
    """
    input number of kmer and returns the string representation
    stays are represented as 0's or "stay"
    :params number: integer representation of kmer
    :param k: k for kmer
    :type number: int
    :type k: int
    returns string representation of kmer
    """
    if number == 0:
        return "stay"
    else:
        number = number - 1
        base_dict = {0: "A", 1: "C", 2: "G", 3: "T"}
        string = ""
        for i in range(k):
            bit = base_dict[(number >> 2*i) & 3]
            string = bit + string
        return string


def kmer_to_number(kmer, k=5):
    """
    input string representation of kmer and output number representation
    stays are represented as 0's or "stay"
    :params kmer: string representation of kmer
    :param k: k for kmer
    :type kmer: string
    :type k: int
    returns integer representation of kmer
    """
    if kmer == "stay":
        number_to_return = 0
    else:
        base_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
        kmer_list = list(kmer)
        number_to_return = 1
        for i in range(k):
            base = kmer_list[k-i-1]
            number = base_dict[base]
            shifted = number << 2 * i
            number_to_return += shifted
    return number_to_return


def number_to_kmer_nostay(number, k=5):
    base_dict = {0: "A", 1: "C", 2: "G", 3: "T"}
    string = ""
    for i in range(k):
        bit = base_dict[(number >> 2*i) & 3]
        string = bit + string
    return string


def kmer_to_number_nostay(kmer, k=5):
    base_dict = {"A": 0, "C": 1, "G": 2, "T": 3}
    kmer_list = list(kmer)
    number_to_return = 0
    for i in range(k):
        base = kmer_list[k-i-1]
        number = base_dict[base]
        shifted = number << 2 * i
        number_to_return += shifted
    return number_to_return


def load_training_data(training_data_path, training_samples):
    """
    loads training data from a hdf5 file and returns
    data and labels as two different arrays
    :param training_data_path: path to data
    :param training_samples: number of samples to load. i.e.
    how many reads of length 8000 to load
    :type training_data_path: string
    :type training_samples: int
    :returns array of training data, array of labels
    """
    with h5py.File(training_data_path, 'r') as h5:
        num_chunks = len(h5['chunks'][:])
        print(
            "Selecting {} out of {} chunks".format(
                training_samples, num_chunks))
        random_selection = np.random.choice(
            num_chunks, training_samples, replace=False)
        training_chunks = h5['chunks'][:][random_selection]
        training_labels = h5['labels'][:][random_selection]
    return training_chunks, training_labels


def get_posteriors(data, model_class_path, model_path, sample_grouping=5):
    """
    gets the posterior probabilities from a trained neural network model
    :param data: data to be inputted in to model in form NHC
    :param model_class_path: path to model class
    :param model_path: path pickle of train model
    :param sample_grouping: the number of samples which will get
    classifed in to one kmer
    :type data: numpy array
    :type model_class_path: string
    :type model_path: string
    :type sample_grouping:int
    :returns: posterior probabilities for each event for each kmer
    """
    loader = importlib.machinery.SourceFileLoader('Model', model_class_path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    model = module.Model(sample_grouping)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    data = torch.from_numpy(data).to(device)
    data = data.permute(0, 2, 1)
    output = model(data)
    m = nn.Softmax(dim=1)
    normalised_output = m(output)
    return normalised_output
