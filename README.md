# **Maintenance Step Prediction Using Coordinate-Based Machine Learning**

Overview:
This project explores the use of supervised machine learning to predict aircraft maintenance
steps based on spatial coordinate data. The objective is to enhance augmented-reality (AR)
based maintenance instructions by automatically identifying the correct maintenance stage
from a technician’s tool position in 3D space.

Using an inverter assembly from a flight motion simulator as the case study, the project
demonstrates how coordinate-driven classification models can support intelligent maintenance
guidance systems in aerospace applications.

Dataset:
- Input features: X, Y, Z spatial coordinates
- Target variable: Maintenance step (13 discrete classes)
- Data stored and processed from CSV format

Methodology:
1. Data Processing
   - Loaded raw coordinate data into Pandas DataFrames
   - Prepared features and labels for supervised learning

2. Data Visualization
   - Statistical analysis and class-wise visualization
   - Explored spatial clustering of maintenance steps

3. Correlation Analysis
   - Pearson correlation used to assess feature influence
   - Evaluated how coordinate axes impact classification accuracy

4. Model Development
   - Implemented multiple classification algorithms using scikit-learn
   - Hyperparameter optimization using GridSearchCV
   - Additional optimization using RandomizedSearchCV

5. Model Evaluation
   - Compared models using accuracy, precision, and F1-score
   - Generated confusion matrices for detailed performance analysis

6. Model Stacking
   - Combined multiple classifiers using StackingClassifier
   - Evaluated performance gains from ensemble learning

7. Model Deployment
   - Best-performing model saved using Joblib
   - Model can predict maintenance steps from new coordinate inputs

Technologies Used:
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib

Key Outcomes:
- Successfully predicted maintenance steps with high classification accuracy
- Demonstrated the effectiveness of ensemble and stacked models
- Highlighted potential applications in AR-guided aerospace maintenance

Applications:
- Augmented reality maintenance systems
- Smart manufacturing and inspection workflows
- Aerospace maintenance automation

