# Global-vegetation-physiology-under-drought

This repository contains two parts for a paper to be published as: Li, W., Pacheco-Labrador, J., Migliavacca, M., Miralles, D., Hoek van Dijke, A., Reichstein, M., Forkel, M., Zhang, W., Frankenberg, C., Panwar, A., Zhang, Q., Weber, U., Gentine, P., Orth, R. (2023). Widespread and complex drought effects on vegetation physiology inferred from space (Under review).

i) Demo codes to calculate vegetation physiological components;

ii) Codes required to reproduce main figures in the paper.

Note:

i) The raw data formatted as .npy required to calculate vegetation physiological components are stored at Zenodo https://doi.org/...;

ii) The raw datasets used to compute all analysis results are shared with public links in the paper. Processed data required to produce the final figures  are stored at Zenodo https://doi.org/....

We are happy to answer your questions! Contact: Wantong Li (wantong@bgc-jena.mpg.de)

* Conda environment installation:

Please use the enivronment.yml to set up the environment for runing provided codes. The Linux command for environment installation: conda env create -f enivronment.yml

* The guide of Demo_code_physiology.py:

To save the runtime, European domains are used instead of the global scale when calculating vegetation physiological anomalies.

* The guide of figure_codes:

i) All processed data are in NumPy array format for Python;

ii) Original data are all shared with public links in the paper.
