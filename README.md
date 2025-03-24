<h1 align="center"> The "Black-Box" Optimization Problem: Zero-Order Accelerated Stochastic Method via Kernel Approximation </h1>

<h3 align="center"> Abstract </h3>

<p align="justify">This paper addresses an optimization problem where computing gradients is impractical, placing it within the realm of black-box optimization. In this context, oracles furnish only the objective function's value at a given point, potentially with bounded stochastic noise. Assuming convexity and increased smoothness, our contribution introduces a novel zero-order accelerated stochastic gradient descent (ZO-AccSGD) method, leveraging kernel approximation to exploit increased smoothness. Theoretical analysis reveals the superiority convergence results of our ZO-AccSGD algorithm over state-of-the-art (SOTA) methods, particularly in improved iteration complexity estimation. Moreover, our analysis demonstrates <i>a surprising way to improve convergence to the error floor by utilizing the overbatching effect</i>. Validation of our theoretical results is demonstrated both on the model function and on functions of interest in the field of machine learning. A comprehensive discussion underscores the superiority of our proposed algorithm, offering advancements in solving the original problem compared to existing state-of-the-art methods.

<h2 align="left"> Intro </h2>
This repository contains code for experiments, provided in the article. 

<h2 align="left"> Files </h2>
<ul>
  <li> algorithms_comp.py contains the implementations of the methods considered in the experimental part of the paper </li>
  <li> functions.py contains implementation of logistic regression and negative log-likelihood functions </li>
  <li> utils.py contains functions for preparing data and plotting the results </li>
</ul>

<h2 align="left"> Folders </h2>
<ul>
  <li> <b>dump</b> is a folder for experiments results </li>
  <li> <b>datasets</b> contains used in the experiments datasets from <a href="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html">LIBSVM library</a> </li>
  <li> <b>figures</b> contains graphs presented in the paper </li>
</ul>

<h2 align="left"> Notebooks </h2>
There are three Jupyter Notebooks for logistic regression optimization. Datasets name as <i>phishing, diabetes</i> or <i>hearts</i> can be seen in the title. Notebook toy_example.ipynb represents the experiments on the minimization problem of solving p linear equations.
