"""[Fall 2020] AI607 Graph Mining and Social Network Analysis.
Homework #2 : Citation Network Analysis using HITS

This program prints the top 10 Hubs and Authority nodes for the given graph

Usage:
    python main.py -f [file]
"""

from __future__ import absolute_import
import argparse
from graph import CitationNetwork, hits, print_top_k


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(
        description="Get the top 10 Hubs and Authority nodes for the given graph"
    )
    parser.add_argument(
        "-f",
        "--file",
        action="store",
        default="graph.txt",
        type=str,
        help="A file path for an initial matrix",
    )
    args = parser.parse_args()

    # Hyperparameter settings
    MAX_ITER = 100  # The number of maximum iteration for HITS algorithm
    TOL = 1e-5  # Tolerance for HITS algorithm
    K = 10  # To find top-k scores

    # Generate the graph
    graph = CitationNetwork(args.file)

    # Run HITS algorithm
    hubs, authorities = hits(graph, MAX_ITER, TOL)

    # Print top-k hub and authority scores
    print(f"--- Top-{K} hubs ---")
    print_top_k(hubs, K)

    print(f"--- Top-{K} authorities ---")
    print_top_k(authorities, K)