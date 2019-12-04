# Implement NSGA2
Based on <https://www.iitk.ac.in/kangal/Deb_NSGA-II.pdf>

Using <https://github.com/google-research/nasbench>

## Setup
1. download cuda 10.0, tensorflow 1.15, python 3.6
2. execute setup.py

## How to use

## Data Structure
* population is list of elems.
  
  elem = {'acc': data['validation_accuracy'], 'time': data['training_time'], 'spec': spec}

* 

## Flow
1. Initialize population randomly.
2. Generate offspring.
    1. Choose two parents by tournament selection.
    2. 
    
