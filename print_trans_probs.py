"""
script to print and visualise transition probabilities
"""

from utils import transition
kmer = [(i+1) for i in range(64)]
probs = [[transition(j, i) for i in kmer] for j in kmer]

for i in range(len(probs)):
    print(sum(probs[i]))
for i in range(len(probs)):
    print(probs[i])
