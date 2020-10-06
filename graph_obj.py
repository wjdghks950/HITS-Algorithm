"""This file contains an incomplete implementation of the CitationNetwork class and HITS algorithm.
Your tasks are as follows:
    1. Complete the CitationNetwork class
    2. Complete the hits method
    3. Complete the print_top_k method
"""

from __future__ import absolute_import
from typing import Dict, Tuple

############################################################################
# You may import additional python standard libraries, numpy and scipy.
# Other libraries are not allowed.
############################################################################
from scipy.sparse import lil_matrix
import numpy as np
import os
from collections import namedtuple
from tqdm import tqdm
import math

class CitationNetwork:
    """Graph structure for the analysis of the citation network
    """

    def __init__(self, file_path: str) -> None:
        """The constructor of the CitationNetwork class.
        It parses the input file and generates a graph.

        Args:
            file_path (str): The path of the input file which contains papers and citations
        """

        ######### Task 1. Complete the constructor of CitationNetwork ##########
        # Load the input file and process it to a graph
        # You may declare any class variable or method if needed
        ########################################################################

        # raise NotImplementedError("CitationNetwork class is not implemented")
        fields = ("title", "author", "year", "venue", "idx", "ref", "abst")
        Document = namedtuple("Document", fields, defaults=(None,) * len(fields))
        if os.path.isfile(file_path):
            self.doc_list = []
            with open(file_path) as f_in:
                # Construct (N x N) sparse matrix (e.g., scipy.sparse.lil_matrix)
                self.N = int(f_in.readline().strip())
                self.citation_adj = lil_matrix((self.N, self.N), dtype="int32")
                doc = Document()
                ref_list = []
                for line in tqdm(f_in):
                    if line == "\n":
                        self.doc_list.append(doc)
                        if doc.ref is not None:
                            if isinstance(doc.ref, list):
                                for ref_idx in doc.ref:
                                    self.citation_adj[doc.idx, ref_idx] = 1
                            # if self.citation_adj[doc.idx, doc.ref] >= 1:
                            #     self.citation_adj[doc.idx, doc.ref] += 1
                            #     print(self.citation_adj[doc.idx, doc.ref])
                            # else:
                            #     self.citation_adj[doc.idx, doc.ref] = 1
                    else:
                        if line[:2] == "#*":
                            doc = doc._replace(title=line[2:].strip())
                        elif line[:2] == "#@":
                            doc = doc._replace(author=line[2:].strip())
                        elif line[:2] == "#t":
                            doc = doc._replace(year=line[2:].strip())
                        elif line[:2] == "#c":
                            doc = doc._replace(venue=line[2:].strip())
                        elif line[:6] == "#index":
                            doc = doc._replace(idx=int(line[6:].strip()))
                        elif line[:2] == "#%":
                            ref_list.append(int(line[2:].strip()))
                            doc = doc._replace(ref=ref_list)
                        elif line[:2] == "#!":
                            doc = doc._replace(abst=line[2:].strip())
                        else:  # Exception case: Incorrect Prefix
                            raise ValueError("Incorrect Prefix: Must be one of the following: (#*, #@, #t, #c, #index, #%, #!)")

                assert self.N == len(self.doc_list)
                assert self.num_ref == self.citation_adj.count_nonzero()
                print("Document Instance (sample): ", self.doc_list[:2])
                print("74th index should be = 9: >> Answer: ", self.citation_adj[74].count_nonzero())
                print("Citation network (Adj.): {} / (Non-zero counts: {})".format(self.citation_adj.get_shape(), self.citation_adj.count_nonzero()))
                print("Number of references: {}".format(self.num_ref))
    ############################################################################
    # You may add additional functions for convenience                         #
    ############################################################################

    # def adjacency_matrix(self):
    #     return cls.citation_adj
    
    def num_nodes(self):
        return self.N


def hits(
    graph: CitationNetwork, max_iter: int, tol: float
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """An implementation of HITS algorithm.
    It uses the power iteration method to compute hub and authority scores.
    It returns the hub and authority scores of each node.

    Args:
        graph (CitationNetwork): A CitationNetwork
        max_iter (int): Maximum number of iterations in the power iteration method
        tol (float): Error tolerance to check convergence in the power iteration method

    Returns:
        (hubs, authorities) (Tuple[Dict[int, float], Dict[int, float]]): Two-tuple of dictionaries.
            For each dictionary, the key is the paper index (int) and the value is its score (float)
    """

    ################# Task2. Complete the hits function ########################
    # Compute hub and authority scores of each node using the power iteration method
    ############################################################################

    # raise NotImplementedError("hits method is not implemented")
    authority_score = np.array([1 / math.sqrt(graph.num_nodes())] * graph.num_nodes())
    hub_score = np.array([1 / math.sqrt(graph.num_nodes())] * graph.num_nodes())
    print("Authority & Hub score (shape): ", authority_score.shape)

    for i in tqdm(range(max_iter)):
        next_hub_score = graph.citation_adj.dot(authority_score)
        next_hub_score = next_hub_score / np.linalg.norm(next_hub_score)
        next_authority_score = graph.citation_adj.T.dot(next_hub_score)
        next_authority_score = next_authority_score / np.linalg.norm(next_authority_score)
        # Normalize scores
        if i > 0:
            diff_norm = np.linalg.norm(next_hub_score - hub_score)
            print("Diff: {} vs. Tol.: {}".format(diff_norm, tol))
            if diff_norm < tol:
                break
        hub_score = next_hub_score.copy()
        authority_score = next_authority_score.copy()
        print(next_hub_score.shape)
        print(next_authority_score.shape)
        break

    hub_dict = dict(enumerate(next_hub_score, start=0))
    authority_dict = dict(enumerate(next_authority_score, start=0))
    return (hub_dict, authority_dict)



def print_top_k(scores: Dict[int, float], k: int) -> None:
    """Print top-k scores in the decreasing order and the corresponding indices.
    The printing format should be as follows:
        <Index 1>\t<score>
        <Index 2>\t<score>
        ...
        <Index k>\t<score>

    Args:
        scores (Dict[int, float]): Hub or Authority scores.
            For each dictionary, the key is the paper index (int) and the value is its score (float)
        k (int): The number of top scores to print.
    """

    ############## Task3. Complete the print_top_k function ####################
    # Print top-k scores in the decreasing order
    ############################################################################

    # raise NotImplementedError("print_top_k method is not implemented")
    sorted_scores = {idx: score for idx, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    doc_idx = list(sorted_scores.keys())
    doc_score = list(sorted_scores.values())
    del sorted_scores
    for i in range(k):
        print(doc_idx[i], "\t", doc_score[i])
