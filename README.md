# NSGA-II
The implementation of NSGA-II [1] with Python

1. nd_sort.py is the non-dominated sorting method using the efficient non-dominated sorting method in [2].

2. variation.py is the genetic operations in NSGA-II, where the simulated binary crossover (SBX) and polunomial muation (PM) in [3] are used.

3. crowding_distance.py is the density estimation method in NSGA-II, where the extreme solutions in each Pareto front are set to inf.

4. GLOBAL.py involves the problem and parameters.





[1]. Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on Evolutionary Computation, 2002, 6(2): 182-197.

[2]. Zhang X, Tian Y, Cheng R, et al. An efficient approach to nondominated sorting for evolutionary multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(2): 201-213.

[3]. Deb K, Beyer H G. Self-adaptive genetic algorithms with simulated binary crossover. Secretary of the SFB 531, 1999.
