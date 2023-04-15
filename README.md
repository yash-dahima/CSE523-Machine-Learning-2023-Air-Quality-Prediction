# CSE523-Machine-Learning-2023-Air-Quality-Prediction

Air Quality has been predicted over Ahmedabad, India using classical machine learning algorithms as a part of the course requirement of CSE523 - Machine Learning at Ahmedabad University, India.

The dataset containing various meteorological and air quality parameters was compiled using the CAMS GLobal Reanalysis Data - EAC4 (https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-reanalysis-eac4?tab=form). An Exploratory Data Analysis (EDA) was performed to visualize relationships between different features of the dataset. Correlation analysis & Principal Component Analysis (PCA) were applied on the dataset to select the most important features from the dataset. Various machine learning models such as linear models, kernel-based models, ensemble models were developed and evaluated via metrics such as r^2 score, mean absolute error, mean absolute percentage error, root mean square error using scikit-learn library of python. Gradient Boosting models were observed to perform best on the dataset.


![image](https://user-images.githubusercontent.com/46166118/232245303-14c1ba38-73e3-4879-b6d1-779fc68e6ef9.png)

References:

[1]	Kumar, K., Pande, B.P. Air pollution prediction with machine learning: a case study of Indian cities. Int. J. Environ. Sci. Technol. (2022). https://doi.org/10.1007/s13762-022-04241-5

[2]	Doreswamy HKS, Yogesh KM, Gad I (2020) Forecasting Air pollution particulate matter (PM2.5) using machine learning regression models. Procedia Comput Sci 171:2057–2066. https://doi.org/10.1016/j.procs.2020.04.221

[3]	Madhuri VM, Samyama GGH, Kamalapurkar S (2020) Air pollution prediction using machine learning supervised learning approach. Int J Sci Technol Res 9(4):118–123
