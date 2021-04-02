# Stochastic COVID-19 Simulation
Stochastic simulation of a simple COVID-19 model on Networks with individual infectiousness variations.

Copyright: 2021, [Gerrit Großmann](https://mosi.uni-saarland.de/people/gerrit/), [Group of Modeling and Simulation](https://mosi.uni-saarland.de/) at [Saarland University](http://www.cs.uni-saarland.de/)

Version: 0.1 (Please note that this is proof-of-concept code in a very early development stage.)

**Caveat lector**: This is an academic model, do not use academic models as a basis for political decision-making.

## Overview
------------------
![Animation](https://github.com/gerritgr/StochasticNetworkedCovid19/raw/master/anim-opt.gif)


## Installation
------------------
The tool is based on Python3. Install the required dependencies using:
With:
```console
pip install -r requirements.txt
```

## Example Usage
-----------------
With
```console
python simulation.py
```


## Output
-----------------
Two output folders are created: `output_graphs/` and  `output_dynamics/`.

### output_graph
-----------------
`output_graphs/` contains example contact networks which are generated. Note that for each simulation run a new contact network is generated using a random graph model.
Moreover, the folder contains summary statistics (degree distribution) and network visalizations (only for networks < 200 nodes).

### output_dynamics
-----------------
`output_dynamics/` contains different files describing the stochastic dynamics of the system. Files with `evolution` in their name report the fraction of nodes in each compartment over time.
The `rvalues` file reports for each infected node in each simulatino run: (1) when the node became infected, (2) number of secondary infections of that node. 
Visualization code is not provided. 

