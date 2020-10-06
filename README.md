# Citation Network Analysis using HITS Algorithm

This repository contains an implementation for HITS algorithm. 
Given a citation network in `graph.txt` (zipped as `graph.tar.gz`), this implementation uses the HITS algorithm to analyze the hub and authority scores of each web page.

## Prerequisite (Dependencies)

I used Scipy and Numpy for implementation.

```bash
pip install scipy==1.5.0
pip install numpy==1.18.1
```

## Note
`graph_obj.py` contains Citation Network class that stores every attribute of a web page as a single namedtuple.
`graph.py` contains Citation Network class that only uses the index and reference index to build an adjacency matrix for a citation network.

## How to use
After cloning the code to your repository, generate Kronecker graph with the following command:
```bash
main.py [-h] [-f FILENAME]
```