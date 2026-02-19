**Section 1: The BBO capstone project and its purpose**

In the Black-Box optimization project, we are given eight experiments in which we are trying to find the optimum of a given function.  In each of the cases, the target function is unknown, and presumed to be difficult to evaluate.  We are given a number of initial data points with function inputs and outputs, and during the project we will be allowed to submit one dozen additional points for evaluation.  The target is to ultimately find the value of the inputs that optimize the function.
The functions range in dimension of their domain from two to eight, and are real-valued.  Each represents a real-world case scenario, including radiation detection, chemical processes, and Machine-Learning parameter optimization, among others.  These are realistic cases and assumptions, including the fact that function evaluation is often expensive, and therefore the skills developed during the project are useful and applicable.
The key issue in decision-making when choosing a new point every week is the trade-off between exploration and exploitation.  The question is: given the information we have so far, is it better to exploit the region where the current highest values have been found (in the hopes of finding even higher values), or to explore new regions of the domain (in the hopes of finding out that the function is even higher there)?  This trade-off manifests itself differently according to the characteristics of the problem.

**Section 2: Experiment Inputs and Outputs:**

As mentioned above, we are given eight functions, all of which are real-valued and have inputs in space of dimensions between two and eight.  In each case, we are given a number of initial data points for which the function have already been evaluated.  We are also given a description of the real-world scenario that the data points correspond to, each experiment having been designed around a specific concrete context.
 Our task is to choose each week a new data point in the function domain, and submit it for evaluation.  We then receive the value of the function at that point, and repeat the exercise the following week with an additional data point to consider.  The project lasts a fixed number of weeks, and the target is to find the global optimum for each function.

**Section 3: What are you trying to achieve within the BBO capstone project?**

The objective is to find the optimum for all eight functions using the process described above.  The key limitations are:
> We can only query one data point per week during the project;
> We do not have specific knowledge of functional forms of the objective function;
> In some cases, the function evaluation is stochastic, so it contains random noise.

**Section 4: Technical Approach**

I am approaching this problem using a Bayesian Optimization approach, described earlier in the course.  In a nutshell, this approach takes the information we have so far (existing data points) to come up with estimates for best guesses of the value of the objective function throughout the domain, as well as confidence bounds.  These estimates are based on Gaussian Processes.
The choice of the next data point to evaluate uses an Upper Confidence Bound (UCB) objective function, which trades off the mean and standard deviation of the model estimates at each point.  Higher emphasis on standard deviation imply that the possible range of values of the function widen, especially in points that are distant from any point that we have already evaluated, and make it more likely that the model will want to explore such a region.  Conversely, lower emphasis on standard deviation will emphasize exploitation of known points where the function value is high.
In order to determine the optimal trade-off between exploration and exploitation, I have carried out experiments that simulate the project with known objective functions.  The code for these simulations is available in this repository.  In each case, the simulations indicate that default parameters for this trade-off work well in the early stages, and switching to greater emphasis on exploitation tends to occur earlier in lower-dimensional experiments, due to a phenomenon called the Curse of Dimensionality.
There have been a couple of approaches mentioned in the course that can be used to potentially weed out regions of the domain and make the search process more efficient.  Among these, I am particularly interested in Support-Vector Machines, which can potentially give me additional useful information regarding which parts of the domain to explore and which ones to ignore.  I hope to be able to implement these improvements soon.

Update (19 Feb 2026): From Version 02, I implement two technical changes.  The first involves preceding the Bayesian Optimization by an SVM estimation that tries to determine if certain regions of the domain seem more promising for exploration.  The second is a dynamic implementation of the parameter kappa, to emphasize greater exploitation as the experiment comes closer to its conclusion.
