# Global-vegetation-physiology-under-drought

This repository contains two parts for a paper to be published as: Li, W., Pacheco-Labrador, J., Migliavacca, M., Miralles, D., Hoek van Dijke, A., Reichstein, M., Forkel, M., Zhang, W., Frankenberg, C., Panwar, A., Zhang, Q., Weber, U., Gentine, P., Orth, R. (2023). Widespread and complex drought effects on vegetation physiology inferred from space (Under review).

i) Demo codes to calculate vegetation physiological components;

ii) Codes required to reproduce main figures in the paper.

Note:

i) The raw data formatted as .npy required to calculate vegetation physiological components are stored at Zenodo https://doi.org/;

ii) The raw datasets used to compute all analysis results are shared with public links in the paper. Result data  required to produce the final figures  are stored at Zenodo https://doi.org/.

We are happy to answer your questions! Contact: Wantong Li (wantong@bgc-jena.mpg.de)
Conda environment installation
Please use the enivronment.yml to set up the environment for runing provided codes. The Linux command for environment installation: conda env create -f enivronment.yml

<!-- The guide of demo_of_sensitivity_analysis
i) To save the runtime, one satellite LAI product and one soil moisture reanalysis are used as an example, while for the paper analysis we calculate many times of sensitivity results using different satellite and land surface modelled LAI and soil moisture products;

ii) To save the runtime, European domains are used instead of the global scale when calculating sensitivity results.

The guide of figure_codes
i) All processed data are in NumPy array format for Python;

ii) Original data are all shared with public links in the paper.

References
i) Random forest modelling refers to:

Breiman, L. Random forests. Machine Learning 45, 5–32 (2001).

ii) SHAP values to interpret contributions of predictors to target variables refers to:

Lundberg, S. M. & Lee, S. I. A unified approach to interpreting model predictions. (2017);

Molnar, C. Interpretable Machine Learning: A Guide for Making Black Box Models Explainable (2021). [online: https://christophm.github.io/interpretable-ml-book/] -->
