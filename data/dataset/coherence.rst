The Coherence dataset contains information about the 
coherence between the head comment and the implementation
of a source code methods.

=================   ==============
Classes                          2
Samples per class       (Pos) 1713
(Neg) 1168
Samples total           (Tot) 2881
Dimensionality                5642
Unique Terms                  2821
Features            real, positive
=================   ==============

Note:
-----
Since methods are gathered from different software projects, to ease data
analysis (e.g. slicing, splitting or extracting data of a single software project), 
data are stored according to classes and projects, respectively.
In particular, data are primarily grouped by class (all positive instances, first), and then
further organized per project.

So far, these are the distribution of examples per single project:
======================   ===========================
Project                  Positive | Negative | Total
CoffeeMaker (1.0)           27    |    20    |   47
JFreeChart (0.6.0)         406    |    55    |   461
JFreeChart (0.7.1)         520    |    68    |   588
JHotDraw (7.4.1)           760    |  1025    |  1785
======================   ===========================
        