# rich-noisy-python
A Python implementation of the EM algorithm in Newman, M.E.J. 2018. “Network structure from rich but noisy data.” Nature Physics 14 6 (June 1): 542–545.

rich-noisy-python takes a matrix of observed ties between nodes in a network and either a total number of trials where ties could be observed or a matrix describing how often each edge could have been observed. By iterating an expectation and a maximization step between estimates of generative parameters and probabilities of underlying similarity, it converges on a matrix of probabilities describing the likelihood of a "true" edge connecting every set of nodes, as well as a set of parameters defining the overall probability of similarity (rho), the true positive rate (alpha), and the false positive rate (beta).

This is a work in progress. Two important notes:

At present, rich-noisy-python implements two variants of the algorithm, one in which all edges have the same total number of trials to be observed, and one in which every edge has the option of a different number of trials. I will continue to implement the other variants of the algorithm as published by Newman.

The implementation at present needs continued tweaking. If you are interested in this package for immediate use, please email me at ahfc (at) umich (dot) edu.
