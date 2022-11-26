# Automatic-Selection-of-SVM-Kernels-Using-Meta-Learning

The never-ending dilemma of which kernel to choose for a kernel-based learning method like support vector machines (SVM) is what we are trying to address in this paper. The current Trial 
and error method for selecting kernels is very time consuming and computationally ineffective.This paper presents a new Meta-learning based approach for selecting the kernels automatically
for classification problems. The Research is done on 7 different Classification problems which belong to Medical, Financial, Chemical and Phytology domains, using the Support Vector Machine
Algorithm (SVM). To enable in choosing the right kernel automatically, we make use of spatial statistical characteristics of the dataset by defining a Variogram model for every dataset. A 
variogram is derived by using the base statistical parameters calculated for every attribute of the dataset. We have evaluated the performance of three different SVM kernels and picked the best 
performing kernel based on the accuracy. Considering the Variogram parameters and the results of the best kernel chosen we use a Decision tree algorithm to generate a rule that’s effective in 
automatic SVM kernel selection.
