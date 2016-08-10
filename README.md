# Working_memory
This repository contains a model of a working memory task detailed here: jn.physiology.org/content/91/3/1424

One of the neurobiological properties that the reaserchers who worked on that paper measured is the activity of so-called “persistent memory pro/antisaccades”. So, essetially, after putting something into working memory and then clearing it, they were able to measure the following effect:  

![alt tag](https://github.com/agaikova/Working_memory/blob/master/what_are_axises.gif)

Since the axies are not very clear, I used graphing softawre to estimate the values of the spiking pattern at various points in the task. At 0.5 s, the spike value is ~ 28hz. At 1.75s, the spiking value is ~34 hz. The peak value, which is reached ~0.8s into the simulation, is around 54 hz. This data is used in the objective function of every hyperopt program written for this repository. 

Terry (@tcstewar) initially wrote the model (https://github.com/tcstewar/testing_notebooks/blob/master/Working%20memory%20overshoot.ipynb). He had some doubt as to the filter threshold used in the origional paper, so I ran hyperopt with multiple parameters for the filter threshold values of 0.0, 0.2, and 0.4. When testing the optimized values for 0.2 and 0.4 (see wm_benchmark_02 and wm_benchmaek_04), I found that they weren't very close to the neurobiological data so we decided to proceed by keeping filter_threshold == 0.0. Then I ran individual hyperopt tests for all the parameters considered in this simulation. Those individual hyperopt tests didn't result in much change in optimal values.

In the testing notebook, you can find the values from the 3 different filter thresholds and the final individually optimized values for filter_threshold == 0.0. 
