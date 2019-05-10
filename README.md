# Viterbi

Python code implementing the Viterbi algorithm to classify sequence of 5mers from a hdf5 file.

## Note on Hidden Markov Models

Lets imagine a robotic dog must determine whether the owner is happy or sad depending on a set of observations of the owner. These observations fall in to one of four categories, sleeping, crying, on facebook or watching made in chelsea.
let Y denote one of these four observations which are simplified to the four letters s, c, f, and w respectively.
Let X denote the unknown state, which can be either happy, h or sad s.

We want to answer such questions as what is the porbability of you being happy given that you are facebooking.

P(X=h | Y=f)

This is refered to as the emission probability, which maps each observation to the state.

Given:

P(Y|X) =


|   |w  |s  |c  |f  |
| --- | --- | --- | --- | --- | 
|s  |0.1|0.3|0.5|0.1|
|h  |0.4|0.4|0.2|0  |




We can say that the probability of the owner being facebooking given that they are happy can be read off the table

P(f | h) = 0

We want to invert this for the robot. 

What is the probability that the owner is happy, given they are facebooking. We need to use Bayes rule inorder to calculate this.
https://en.wikipedia.org/wiki/Bayes%27_theorem

Now let us introduce a temporal element. eg. The likelihood that if the owner is happy, they will remain happy.

This is refered to as a transition probability

|    |   |xt=2  |   |
|----|---|---|--- |
|    |   |s  |h   |
|xt=1|s  |0.99|0.01|
|    |h  |0.1|0.9|




By using the emission probabilities and the transition probabilities, it is possible to calculate the probability of the owner being happy or sad at the current time, given a series of observations.


## Notes on the Viterbi algorithm

The viterbi algorithm is used to decide the most likely sequence of states (i.e. happy or sad) given the emission and transition probabilities. 

## Example with 1mers

To illustrate the Viterbi algorithm with a simplified example, the scripts can be run using a sequence of DNA consisting of a series of three 1mers with arbitrary transition probabilities.
N.B in this example, the number_to_kmer function used is unique to this example as it does not account for stays. Also the ordering AGTC is used, but for the rest of the examples, the
alphabetical ACGT convention will be followed. This was just a mistake on my part.

The "emission probabilities" in this case are referred to as posterior probabilities. These are the probabilities of a window of data points being attributed to a specific kmer.
For example a series of raw signal data [0.1233, 0.7890, 0.7897, 0.8989] will be converted in to a set of probabilities which corresponds to the probability of that part of the signal 
being one of the four 1mers A, G, T or C. These probabilities are only dependent on the window on to the raw signal data and is not dependent on any other
information outside of that set of data points. These numbers are outputted from a machine learning algorithm.

Using posterior probabilities and transition probabilities, the viterbi algorithm will determine the most likely sequence of kmers. 

Let us outline the arbitraty posterior probabilities and transition probabilities used in this example.

sequence in time >

|   |t=1|t=2|t=3|
|---|---|---|---|
| A |0.3|0.2|0.3|
| G |0.1|0.4|0.2|
| T |0.2|0.2|0.1|
| C |0.4|0.2|0.4|

transition probabilities


|    |xt=2|   |   |   |   |
|----|---|---|---|---|---|
|    |   |A  |G  |T  |C  |
|xt=1|A  |  0|  1|  0|  0|
|    |G  |  0|  0|  1|  0|
|    |T  |  0|  0|  0|  1|
|    |C  |  1|  0|  0|  0|


A > G = 1

G > T = 1

T > C = 1

C > A = 1


All else is zero.

A dynamic programming table, and a back trace table must be created in the process of determining the most likely path.

![HMM](hmm.png?raw=true)


Figure 1. Diagram showing how a dynamic programming table is constructed

The transition probabilities are in red, the posterior probabilities are in blue, the numbers which are inserted in to the DP table are in green.

Dynamic programming table

|Events  |   |   |   |
|---|---|---|---|
|   |  1|     2|  3|
|A  |0.3|  0.08|  0.012|
|G  |0.1|  0.12| 0.016|
|T  |0.2|  0.02|  0.012|
|C  |0.4|  0.04|  0.008|



Backtrace table

|Branch start state|Events |  |   |
|----|---|---|---|
|    |  0|1  |2  |
|A    |A  |  C|  C|
|G    |G  |  A|  A|
|T    |T  |  G|  G|
|C    |C  |  T|  T|

### filling out the backtrace table


The number of hidden states in this 1mer example is 4. So along the diagram, there will always be 4 branches each characterised by which state they start on.
Each element in the backtrace table represents the previous state at that event for that branch.
In order to find out which branch this state belongs to, we must go through the backtrace table and through all the previous states until we are at the beginning, where we can determine what was
the starting state for that branch. 
i.e. if we want to find out which branch the final T state came from, we look at row T, event 2 which is G. So now we look at row G event 1 which is A. Then we look as row A event 0. 
So the branch that ended in a T started on A and followed the sequence A, G, T


### filling out the dynamic programming table


At each time step, a comparison is made between each states. i.e. at time step 1, what is the most likely path through the states thats leads to the current state being G.

P(A > G) = 0.3

P(G > G) = 0

P(T > G) = 0

P(C > G) = 0


So the most likely path taken for time step 1 to be state G is P(A > G). So this is the only path which is taken forward and the DP is updated with 0.12 for timestep 1, event G.
The backtrace table is updated with A for row G event 1, as G came from A. 


This is done for time step 1 with all the other states A, C and T, until the first column of the DP table is full, and the first column of the backtrace table is full.


This will then continue on to further events, until we arrive at the end of the events, and the value of the DP table which is the largest is the most likely state for the final state (i.e. G).
The path taken to this final state is the most likely path taken to arrive at this state therefore, the path taken is the most likely path taken through the states.
So we look at row G on the back trace table. We have state A, so the state which followed G must be state A.
We go to column 1 state A, and we see that the state which followed A is state C. 
So the most likely path is C A G.

## Run the 1mer example script

This example does not incoporate stays, and uses artificial arbitrary posteriors along with arbitrary transition probabilities to produce most likely path of 1mers.

    python run_viterbi.py -m e1
    
## 2mer example which incoporates stays, records transitions and using arbitrary posteriors

The posteriors used will be printed, aswell as the dynamic programming table the backtrace table and the transition table.
Each element in the transition table shows what transition was used to get to that state. 0 for stay, 1 for skip, 2 for step. This will become useful when stringing the kmers together.
The transition function is used, so transition probabilities are not arbitrary.

    python run_viterby -m e2


    
## 5mer example which incoporates stays and uses fake posteriors

Takes arbitrary posteriors and runs it through a viterbi algorythm and strings together the kmers to create a basecall using the transition table.

    python run_viterbi.py -m e3

## 5mer example with impossible transitions in posteriors which raises an error

Similar to example 3 pub with illegal transitions in posteriors

    python run_viterbi.py -m e4
    
    
## Running script with real data from a hdf5 file and accounting for stays

Takes one read from a hdf5 file, puts it through a trained neural net to obtain posterior probabilities, then runs these posteriors
through a viterbi algorythm and strings together the kmers into a basecall. The only example in this script which uses a neural net to produce posteriors. 

    python run_viterbi.py -m r


