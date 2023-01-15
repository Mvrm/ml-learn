# model calibration 

**Table of Contents**

- Reliability Curves (diagnose calibration) using sklearn-calibration to identify whether or not our model is calibrated.

- Platt-scaling or sigmoid: The Platt Scaling is used for small datasets with a reliability curve in the sigmoid shape. It can be loosely understood as putting a sigmoid curve on top of the calibration plot to modify the predictive probabilities of the model.

- Isotonic calibration:  Isotonic Regression is a more complex approach and requires more data. The main advantage of Isotonic Regression is that it doesn’t require the reliability curve of the model to be S-shaped. However, this method is sensitive to outliers and works well for large datasets.

