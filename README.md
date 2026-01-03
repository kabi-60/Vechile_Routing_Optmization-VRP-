# Vehicle Routing Optimization using Python

## 📌 Overview
This project solves the Vehicle Routing Problem (VRP) by optimizing routes using geographic distance calculation. It uses latitude and longitude data to minimize total travel distance.

---

## 🧠 Problem Statement
In logistics and delivery systems, manually planning vehicle routes leads to increased fuel cost and inefficient deliveries. An optimized routing system is required to improve efficiency.

---

## ✅ Solution
This project calculates distances between locations using the Haversine formula and applies clustering and heuristic techniques to generate optimized vehicle routes.

---

## 🚀 Features
- Distance calculation using Haversine formula
- CSV-based input for delivery locations
- Route optimization logic
- Scalable for large datasets
- Modular and clean Python code

---

## 🛠 Tech Stack
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

## 📂 Project Structure
vehicle-routing-optimization/
│── vrp/
│ ├── solver.py
│ ├── locations.csv
│── README.md



---

## ▶️ How to Run

1. Clone the repository
```bash
git clone https://github.com/kabi-60/vehicle-routing-optimization.git
cd vehicle-routing-optimization
Create and activate virtual environment

bash
Copy code
python -m venv venv
venv\Scripts\activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the program

bash
Copy code
python vrp/solver.py vrp/locations.csv true
📊 Output
The program clusters delivery locations and generates optimized vehicle routes based on distance minimization.

👨‍💻 Author
Kabi Krishna
Python | Optimization | Problem Solving