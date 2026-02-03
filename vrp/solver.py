"""
Vehicle Routing Problem (VRP)
Clean & simple visualization using matplotlib
(author: louie + cleaned & simplified)
"""

import math
import itertools
import time
from collections import namedtuple

import numpy as np
import numpy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

NUM_VEHICLES = 2

# x = longitude, y = latitude
Warehouse = namedtuple("Warehouse", ["index", "x", "y"])
Customer  = namedtuple("Customer",  ["index", "x", "y"])
Vehicle   = namedtuple("Vehicle",   ["index", "customers"])

# =========================
# Distance (Haversine)
# =========================
def haversine(a, b):
    R = 6371
    lat1, lon1 = math.radians(a.y), math.radians(a.x)
    lat2, lon2 = math.radians(b.y), math.radians(b.x)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat/2)**2 +
        math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    )
    return 2 * R * math.asin(math.sqrt(h))

# =========================
# CSV Reader
# Format:
# latitude, longitude, is_customer
# =========================
def read_csv(file):
    df = pd.read_csv(file, header=None, names=["lat","lon","is_customer"])

    wh_row = df[df.is_customer == 0].iloc[0]
    warehouse = Warehouse(0, wh_row.lon, wh_row.lat)

    customers = []
    idx = 1
    for _, r in df[df.is_customer == 1].iterrows():
        customers.append(Customer(idx, r.lon, r.lat))
        idx += 1

    print("Warehouse:", warehouse)
    print("Customers:", len(customers))
    return warehouse, customers

# =========================
# Clustering customers
# =========================
def cluster(customers, k):
    pts = np.array([[c.x, c.y] for c in customers])
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = model.fit_predict(pts)

    vehicles = [Vehicle(i, []) for i in range(k)]
    for c, l in zip(customers, labels):
        vehicles[l].customers.append(c)

    return vehicles

# =========================
# Routing (Nearest Neighbor)
# =========================
def greedy_route(warehouse, customers):
    points = [warehouse] + customers
    coords = np.array([[p.x, p.y] for p in points])

    tour = [0]
    remaining = set(range(1, len(points)))

    while remaining:
        last = tour[-1]
        nxt = min(remaining, key=lambda i: LA.norm(coords[last] - coords[i]))
        tour.append(nxt)
        remaining.remove(nxt)

    # Return to warehouse
    tour.append(0)
    
    ordered = [points[i] for i in tour]
    # Sum distances between sequential points in the route
    dist = sum(haversine(ordered[i], ordered[i+1]) for i in range(len(ordered) - 1))
    return ordered, dist

# =========================
# CLEAN VISUALIZATION
# =========================
def plot_all_routes(warehouse, vehicle_routes):
    """
    Plots all vehicle routes on a single map with different colors.
    """
    plt.figure(figsize=(12, 8))
    
    
    
    # Plot warehouse
    plt.scatter(warehouse.x, warehouse.y, s=250, c="red", marker="s", 
                edgecolor="black", zorder=5, label="Warehouse")
    plt.text(warehouse.x, warehouse.y, "  Warehouse", fontsize=12, 
             fontweight='bold', va="center")

    for i, (v_idx, route) in enumerate(vehicle_routes):
        # Pick a distinct color for each vehicle
        color = plt.cm.tab10(i % 10)
        xs = [p.x for p in route]
        ys = [p.y for p in route]
        
        # Plot the route line
        plt.plot(xs, ys, "-", color=color, linewidth=2.5, alpha=0.8, 
                 label=f"Vehicle {v_idx}")
        
        # Plot customer points and numbers
        for j, p in enumerate(route):
            if j == 0: continue # Skip warehouse (already plotted)
            
            plt.scatter(p.x, p.y, s=100, color=color, edgecolor="black", zorder=4)
            plt.text(p.x, p.y, f" {j}", fontsize=9, fontweight='bold',
                     ha="left", va="bottom")

    plt.title("Optimized Vehicle Routing Plan", fontsize=16, pad=20)
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    
    # Set aspect ratio to equal to avoid geographical distortion
    plt.gca().set_aspect("equal", adjustable="datalim")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# =========================
# SOLVER
# =========================
def solve(csv_file, show_plot=True):
    warehouse, customers = read_csv(csv_file)
    vehicles = cluster(customers, NUM_VEHICLES)

    total_dist = 0
    all_routes = []

    for v in vehicles:
        if not v.customers:
            continue

        route, dist = greedy_route(warehouse, v.customers)
        total_dist += dist
        all_routes.append((v.index + 1, route))

        print(f"\nVehicle {v.index + 1}")
        print("Route:", " -> ".join(str(p.index) for p in route))
        print(f"Distance: {dist:.2f} km")

    baseline = sum(haversine(warehouse, c)*2 for c in customers)
    savings = baseline - total_dist
    savings_pct = (savings / baseline * 100) if baseline > 0 else 0

    print("\n" + "="*40)
    print("      DELIVERY OPTIMIZATION SUMMARY")
    print("="*40)
    print(f"Total Vehicles used    : {len(all_routes)}")
    print(f"Total Customers served : {len(customers)}")
    print("-" * 40)
    print(f"Actual (Unoptimized)   : {baseline:10.2f} km")
    print(f"Optimized Distance     : {total_dist:10.2f} km")
    print("-" * 40)
    print(f"DISTANCE SAVED         : {savings:10.2f} km")
    print(f"EFFICIENCY IMPROVEMENT : {savings_pct:10.2f} %")
    print("="*40)

    if show_plot and all_routes:
        print("\nOpening visualization... (Close the window to exit)")
        plot_all_routes(warehouse, all_routes)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python solver.py locations.csv")
    else:
        solve(sys.argv[1], show_plot=True)
