Introduction
************

``PANCO2`` (*Pipeline for the Analysis of NIKA2 Cluster Observations 2*) is a Python software that aims at extracting the thermodynamical properties of the intra-cluster medium (ICM) of galaxy clusters observed with the NIKA2 camera.
This document aims at explaining how the code works, from the physics of the ICM probed by NIKA2 observations to the choices that have been made regarding the data analysis, as well as documenting the different functions of the code, and a guide to setup an analysis.
One of its main goals is also to be translated and to appear as a chapter in my PhD.

*Why a new version?*

While ``PANCO2``'s algorithm is strongly inspired from its precursor (PANCO), its implementation is vastly different (no lines of code were taken from the original).
The core ideas was to increase the speed of the extraction of thermodynamical properties, in order to make the analysis of a sample of clusters possible in a reasonable time.
Other benefits include:

* An increase in the code's readability (many unused functions have not been recoded)
* The switch to Python3, and the use of new features of some of the core libraries used in PANCO (*e.g.* ``emcee``).

Release history
===============

* v1.1-dec20: Possibility to consider correlated map pixels and to compute noise covariance matrices

* v1.0-oct20: First internal release inside the NIKA2 SZ Large Program team
