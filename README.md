# HydroSense Kenya
## Scientific Computing Project

**Student Name:** Kenneth Mweu

**Registration Number:** SCT 211-0383/2024

**Lecturer:** Dr. Nderu

---

# Project Overview

HydroSense Kenya is a Scientific Computing project that demonstrates how numerical methods, simulation, optimization, and data analysis can be applied to precision agriculture.

The system models soil moisture dynamics using environmental data and automatically generates irrigation schedules that improve water-use efficiency while maintaining healthy crop conditions.

The project combines several areas of Scientific Computing including:

- Data preprocessing
- Data validation
- Numerical methods
- Vectorized computation
- Environmental simulation
- Optimization
- Automated testing

---

# Project Structure

```
HydroSense/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── Level_1_Data_Preprocessing.ipynb
│   ├── Level_2_Vectorization.ipynb
│   ├── Level_3_Numerical_Methods.ipynb
│   ├── Level_4_Data_Analysis.ipynb
│   ├── Level_5_Simulation_and_Optimization.ipynb
│   └── Level_6_Testing.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── validation.py
│   ├── numerical_methods.py
│   ├── simulation.py
│   └── optimization.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   ├── test_numerical_methods.py
│   ├── test_simulation.py
│   ├── test_optimization.py
│   ├── test_integration.py
│   └── run_tests.py
│
├── report/
│
├── requirements.txt
└── README.md
```

---

# Features

The HydroSense system includes:

- Agricultural data preprocessing
- Dataset validation
- Vectorized numerical computation
- Root-finding algorithms
- Numerical differentiation
- Numerical integration
- Gaussian Elimination
- Soil moisture simulation
- Euler Method simulation
- Fourth-Order Runge-Kutta (RK4) simulation
- Monte Carlo rainfall simulation
- Single-zone irrigation scheduling
- Multi-zone irrigation optimization
- Irrigation efficiency reporting
- Automated unit testing
- End-to-end integration testing

---

# Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- SciPy
- Jupyter Notebook

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
```

Navigate into the project directory:

```bash
cd HydroSense
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks in numerical order:

1. Level 1 – Data Preprocessing
2. Level 2 – Vectorization
3. Level 3 – Numerical Methods
4. Level 4 – Data Analysis
5. Level 5 – Simulation and Optimization
6. Level 6 – Testing

---

# Running the Test Suite

Run all tests:

```bash
python tests/run_tests.py
```

Or run an individual test:

```bash
python -m unittest tests.test_simulation
```

---

# Learning Outcomes

This project demonstrates practical applications of:

- Scientific Computing
- Numerical Methods
- Data Processing
- Environmental Simulation
- Precision Agriculture
- Optimization Algorithms
- Software Testing
- Python Programming

---

# References

Burden, R. L., & Faires, J. D. (2021). *Numerical Analysis* (11th ed.). Cengage Learning.

Chapra, S. C., & Canale, R. P. (2020). *Numerical Methods for Engineers* (8th ed.). McGraw-Hill.

McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly Media.

The NumPy Developers. *NumPy Documentation*. https://numpy.org/

The pandas Development Team. *pandas Documentation*. https://pandas.pydata.org/

The Matplotlib Development Team. *Matplotlib Documentation*. https://matplotlib.org/

---

# Author

Kenneth Mweu

SCT 211-0383/2024

Scientific Computing Project

HydroSense Kenya

