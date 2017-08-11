# NSGA-II
The implementation of NSGA-II [1] with Python:

1. nd_sort.py is the non-dominated sorting method using the efficient non-dominated sorting method in [2].

2. crowding_distance.py is the density estimation method in NSGA-II, where the extreme solutions in each Pareto front are set to inf.

3. environment_selection.py is the environmental selection procedure in NSGA-II.

4. nsgaii.py is the main file.

5. GLOBAL.py involves the problem and parameters settings, meanwhile, the genetic operations (simulated binary crossover and polynomial mutation [3]) are presented.

6. wof_nsgaii.py is the future file for a vatiation of nsga-ii on large-scale multi-objective optimization from [4].





[1]. Deb K, Pratap A, Agarwal S, et al. A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on Evolutionary Computation, 2002, 6(2): 182-197.

[2]. Zhang X, Tian Y, Cheng R, et al. An efficient approach to nondominated sorting for evolutionary multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2015, 19(2): 201-213.

[3]. Deb K, Beyer H G. Self-adaptive genetic algorithms with simulated binary crossover. Secretary of the SFB 531, 1999.

[4]. Zille H, Ishibuchi H, Mostaghim S, et al. A Framework for Large-scale Multi-objective Optimization based on Problem Transformation. IEEE Transactions on Evolutionary Computation, 2017.
