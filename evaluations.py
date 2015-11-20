"""This module contains the code to perform the analysis
on the evaluations about the coherence of methods
made by involved annotators.

In more details, the module provides:
    `Judge`: class that embeds annotations of a single user/evaluator

    `calculate_agreement_scores`: function to compute the *Agreement Score Matrix*
            according to the evaluations of two input Judges.

    `cohense_kappa`: function to compute the (weighted) Cohen's Kappa Index according
            to input agreement rates.

    `mean_precision`: function to calculate the *Precision Index* among judges
            on a given set of evaluations.
"""
# Author:  Valerio Maggio <valeriomaggio@gmail.com>
# License: BSD 3 clause

from django.contrib.auth.models import User
from source_code_analysis.models import SoftwareProject
import numpy as np


class Judge:
    """
    This class embeds information about a single Judge.
    """

    CODES = 5

    def __init__(self, username, project_name, project_version=None):
        """
        Parameters:
        ----------
        username: str
            Username of the Judge used to retrieve it from the Database

        project_name: str
            The name of the project to get the data for

        project_version: str, optional (default: None)
            The specific version of the project to fetch from the database.
        """
        try:
            self.user_model = User.objects.get(username__iexact=username)
        except User.DoesNotExist:
            print('Warning: No User Found with the input username: %s' % username)
            self.user_model = None

        try:
            if project_version:
                project = SoftwareProject.objects.get(name__iexact=project_name, version__iexact=project_version)
            else:
                project = SoftwareProject.objects.get(name__iexact=project_name)
        except SoftwareProject.DoesNotExist:
            print('Warning: No SoftwareProject Found with the input name: %s' % project_name)
        except SoftwareProject.MultipleObjectsReturned:
            print('Warning: Multiple SoftwareProject Found with the given name: %s' % project_name)
        else:
            self.project = project


        if self.user_model and self.project:
            # This will be a list of sets containing the ids of reference methods, for every evaluation Code.
            self._evaluations = list()
            self.fetch_agreement_evaluations()


    def fetch_agreement_evaluations(self):
        """
        Fetches and stores the agreement evaluations of the associated
        user on the selected Project
        """
        evaluations = self.user_model.evaluations.filter(reference_method__project__id=
                                                         self.project.id).exclude(evaluated=False).exclude(wrong_association=True)

        # Judgements have been mapped on a Nominal Scale of values ranging from 0 to 4 (see CODES)
        # namely:"Strong Disagreement, Disagreement, Don't Know, Agreement, Strong Agreement"
        for i in range(self.CODES):
            self._evaluations.append(set([eval.reference_method.id for eval in evaluations.filter(agreement_vote=i)]))

    @property
    def five_codes_evaluations(self):
        """
        Returns the whole set of evaluations with all the CODES (i.e., 5)
        """
        return self._evaluations

    @property
    def three_codes_evaluations(self):
        """
        Returns the set of evaluations corresponding to "only" three Codes,
        namely "No Coherence", "Don't Know", "Coherence".
        """
        evaluations = list()
        evaluations.append(self._evaluations[0].union(self._evaluations[1]))
        evaluations.append(self._evaluations[2])
        evaluations.append(self._evaluations[3].union(self._evaluations[4]))
        return evaluations

    @property
    def two_codes_evaluations(self):
        """
        Returns the set of evaluations corresponding to "only" two Codes,
        namely "No Coherence", "Coherence", respectively.
        """
        evaluations = list()
        evaluations.append(self._evaluations[0].union(self._evaluations[1]))
        evaluations.append(self._evaluations[3].union(self._evaluations[4]))
        return evaluations


def calculate_agreement_scores(judge1, judge2, k=3):
    """
    Calculate the Agreement Score Matrix according to the evaluations of the
    two input Judges.

    Parameters:
    -----------
    judge1 : The first judge (instance of Judge class)
    judge2 : The second judge (instance of Judge class)
    k : The number of codes to consider choosing from 3 or 5 codes.
        If None, other values, or no value will be given, the default number
        of codes will be considered, namely 3.

    Returns:
    --------
    J : the k x k agreement score matrix (to be passed to the cohens_kappa function)
    """

    if not k or not k in (2, 3, 5):
        k = 3

    if k == 2:
        j1_evals = judge1.two_codes_evaluations
        j2_evals = judge2.two_codes_evaluations
    elif k == 3:
        j1_evals = judge1.three_codes_evaluations
        j2_evals = judge2.three_codes_evaluations
    else:
        j1_evals = judge1.five_codes_evaluations
        j2_evals = judge2.five_codes_evaluations

    # Purge inconsistencies, i.e., evaluations appearing only in one judge set
    def merge_all_evaluations(evaluations):
        jall = set()
        for i in range(k):
            jall = jall.union(evaluations[i])
        return jall

    j1_all = merge_all_evaluations(j1_evals)
    j2_all = merge_all_evaluations(j2_evals)

    # Calculate Differences
    j1_j2_diff = j1_all.difference(j2_all)
    j2_j1_diff = j2_all.difference(j1_all)

    # Remove Differences
    if j1_j2_diff:
        j1_all = j1_all.difference(j1_j2_diff)
    if j2_j1_diff:
        j2_all = j2_all.difference(j2_j1_diff)

    assert j1_all == j2_all  # verify that so far the two sets are equal

    # Now iterate over all the sets for each code and purge all the methods not appearing in the
    # intersection among each score set and the global (i.e., "_all") one.
    for i in range(k):
        j1_evals[i] = j1_evals[i].intersection(j1_all)
        j2_evals[i] = j2_evals[i].intersection(j2_all)

    J = list()
    for i in range(k):
        J.append(list())
        for j in range(k):
            J[i].append(len(j1_evals[i].intersection(j2_evals[j])))

    J = np.array(J)
    return J


def cohens_kappa(J, weighted=False, log=False):
    """
    Calculate the Cohen's Kappa Index.

    $kappa = 1 - \frac{\sum W*X}{\sum W*M}$

    where $*$ indicates the element-wise matrix multiplication.

    $X$: Is the matrix of Observed Scores

    $M$: Is the matrix of Score Agreement by Chance

    $W$: Is the Weight Matrix.

    Note: So far, the implementation considers only two judges

    Parameters:
    -----------

    J : K^N numpy array containing the evaluation scores of Judges,
        where K is the number of codes and N (i.e., J.ndim) is the number of
        considered Judges.
        (see `calculate_agreement_scores` for further details).

    weighted : bool indicating if the Weighted Formulation of the Cohen's Kappa should be calculated
              (default=False and the unweighted index is computed.)
    """

    k = J.shape[0]  # number of codes
    n = J.ndim  # number of judges

    # Calculate X: The Matrix of Observed Scores
    X = J / np.sum(J)

    if log:
        print('J: \n', J)
        print('X: \n', X)

    # Determine the Weight Matrix
    if not weighted:
        # if unweighted, the Weight Matrix W has diagonal cells
        # containing zeros and all off-diagonal cells weights of one
        W = np.ones((k,k))
        np.fill_diagonal(W, 0)
    else:
        # In case Weighted Cohens' Kappa should be calculated, the
        # weighted matrix W is initialized according to the Stevens Scaling
        # Formula. In more details, in case of a 3x3 Matrix, W is euqal to
        # W = [[ 0.,  1.,  4.],
        #      [ 1.,  0.,  1.],
        #      [ 4.,  1.,  0.]]
        # In other words, zero elements on the diagonal, one off diagonal elements are 1^2,
        # those two off diagonal elements are 2^2
        W = np.zeros((k,k))
        for i in range(1, k):
            W += (np.eye(k, k, i) * (i**2)) + (np.eye(k, k, -i) * (i**2))
    if log:
        print('W: \n', W)

    # Calculate M: the Matrix of Agreement by Chance
    J_sum = np.zeros((n,k))
    AG = np.zeros((n,k))
    for i in range(n):
        AG[i] = np.sum(J, axis=1-i)
    J_sum = AG / np.sum(J)  # Calculate Total Probabilities

    if log:
        print('AG: \n', AG)
        print('J_sum: \n', J_sum)

    M = np.zeros((k,k))
    for i in range(k):
        M[i] = np.multiply(J_sum[0,i], J_sum[1])

    if log:
        print('M: \n', M)

    kappa = 1 - (np.sum(W*X)/np.sum(W*M))

    return kappa


def mean_precision(j1, j2):
    """
    Calculate the Precision Index among judges for a given set of evaluations.
    In particular, the function takes in input two sets of evaluations (one for each judges)
    referring to a single score for a given systems.
    For *each* judge, the function computes $P(ji) = P(j1 \cap j2) / P(ji)$, namely the precision 
    of evaluations of considered judge $ji$.
    The two values, $P(j1)$ and P(j2), are then combined by an Harmonic Mean, that is 
    finally returned by the function.
    
    Parameters:
    -----------
    
    j1: set of evaluations of the first judge, corresponding to a **single** score for a **single**
        system (e.g., "Method-Comment Aggreement" in Project "X")
    j2: set of evaluations of the second judge (see above for further details)
    
    Returns:
    --------
    
    pj1: Precision associated to the first judge
    pj2: Precision associated to the second judge
    F: Armonic Mean between pj1 and pj2
    """
    
    pj1 = 0.0 if not len(j1) else len(j1.intersection(j2))/len(j1)
    pj2 = 0.0 if not len(j2) else len(j2.intersection(j1))/len(j2)
    if (pj1 == pj2 == 0.0):
        return 0.0, 0.0, 0.0
    f = 2 * ((pj1 * pj2) / (pj1 + pj2))
    return pj1, pj2, f
