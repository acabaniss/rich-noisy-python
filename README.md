# rich-noisy-python
A Python implementation of the EM algorithm in Newman, M.E.J. 2018. “Network structure from rich but noisy data.” Nature Physics 14 6 (June 1): 542–545.

## Description ##

rich_noisy takes a matrix of observed ties between nodes in a network and either a total number of trials where ties could be observed or a matrix describing how often each edge could have been observed. By iterating an expectation and a maximization step between estimates of generative parameters and probabilities of underlying similarity, it converges on a matrix of probabilities describing the likelihood of a "true" edge connecting every set of nodes, as well as a set of parameters defining the overall probability of similarity (rho), the true positive rate (alpha), and the false positive rate (beta).


## Notes ##

**This is a work in progress.** Some important notes:

- At present, rich-noisy-python implements two variants of the algorithm, one in which all edges have the same total number of trials to be observed, and one in which every edge has the option of a different number of trials. They are both included in the main functions; see the documentation. I will continue to implement the other variants of the algorithm as published by Newman.

- At present, only symmetric matrices/undirected networks can be taken as inputs. This will be changed in the future.

The implementation at present needs continued tweaking. Docstrings do exist, and should provide some help. If you are interested in this package and are having trouble, please email me at ahfc (at) umich (dot) edu and I'm happy to help out.
