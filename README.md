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
```
vehicle-routing-optimization/
│── vrp/
│   ├── solver.py
│   ├── locations.csv
│── README.md
```

---

## ▶️ Installation & Usage 🔧

### Prerequisites
- **Python 3.10+**
- **Git**
- (Optional) **Docker** for containerized runs

### 1) Clone the repository
```bash
git clone https://github.com/kabi-60/vehicle-routing-optimization.git
cd vehicle-routing-optimization
```

### 2) Create & activate a virtual environment
- macOS / Linux
```bash
python -m venv venv
source venv/bin/activate
```

- Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) Install dependencies
If a `requirements.txt` file is present:
```bash
pip install -r requirements.txt
```
If not, install the core packages:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### 4) Run the solver
```bash
python vrp/solver.py vrp/locations.csv true
```
- **Arguments:**
  - `vrp/locations.csv` — path to the CSV containing locations (latitude, longitude, ...)
  - `true|false` — example flag (e.g., enable verbose output or plotting)

### Optional: Docker (quick run)
```bash
# build
docker build -t vrp .
# run (mount repo and run solver)
docker run --rm -v "$(pwd)/vrp:/app/vrp" vrp python vrp/solver.py vrp/locations.csv true
```

### Troubleshooting ⚠️
- PowerShell activation blocked? Run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force` (requires admin privileges)
- Missing packages? Re-run `pip install -r requirements.txt` or install packages manually

---

### Contributing 🤝
- Fork the repository, create a feature branch, add tests, and open a pull request.

---

### Author & License
**Author:** Kabi Krishna  
**License:** MIT (if applicable)

