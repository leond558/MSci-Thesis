# Masters-Thesis-ML-and-Diffusion-MRI

• Used Gaussian mixture models, generalised linear models and sklearn to analyse diffusion MRI data from COVID-19 patients.

• Tuned expectation-maximisation at the algorithmic level to draw abnormality maps 10x more sensitive than existing diagnostics
in locating brain regions severely affected by viral infection. 

• Summarised my conclusions in a clear and comprehensive paper awaiting publication with support from the Cavendish
Laboratory and the Cambridge Medical Research Council. 

Abstract:


Diffusion MRI (dMRI) leverages water molecule movement to give perspective on the microstructural organisation of brain tissue. Previous studies have established that dMRI biomarkers can be used to assess the severity of generalised abnormalities in central nervous function caused by acute COVID-19 viral infection. This investigation extends those studies by considering how Fractional Anisotropy and Mean Diffusivity are affected longitudinally. Analysis revealed diffusion restoration in white matter (WM) with post-hospitalisation treatment whilst grey matter (GM) exhibited a more heterogenous response with some persistency of microstructural disruption. The right side of the brain was more susceptible to COVID-19 with 9 of 13 significantly affected WM tracts on the right. Cognitive metric performance was then merged with diffusion data. The change in fractional anisotropy in the Middle Cingulate Gyrus, a region clinically known for responsibility in spatial processing, was discovered to be the best predictor of spatial reasoning score (p = 0.0275) when a generalised linear model was fit. Thus, dMRI data was seen to be an effective predictor of COVID-19 neuropathological response. Finally, fitting a three component Gaussian Mixture Model (GMM) allowed for voxel-based abnormality scores, providing a more nuanced picture of damage than averaged-out regional FA/MD. Indeed, abnormality score inclusion in the GLM resulted in a 45% reduction in AIC. Such abnormality scores are the best predictors of COVID-19 induced cognitive dysfunction and enable earlier diagnosis.
