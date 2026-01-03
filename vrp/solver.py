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

    ordered = [points[i] for i in tour]
    dist = sum(haversine(ordered[i-1], ordered[i]) for i in range(len(ordered)))
    return ordered, dist

# =========================
# CLEAN VISUALIZATION
# =========================
def plot_route(vehicle_id, route):
    xs = [p.x for p in route]
    ys = [p.y for p in route]

    plt.figure(figsize=(10,3))
    plt.plot(xs, ys, "-o", linewidth=2)

    # Warehouse
    plt.scatter(xs[0], ys[0], s=200, c="red", marker="s", label="Warehouse")

    # Customer numbers
    for i in range(1, len(route)):
        plt.text(xs[i], ys[i], str(i), fontsize=10,
                 ha="center", va="bottom")

    plt.title(f"Vehicle {vehicle_id} Route")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # 🔥 KEY LINE (no distortion)
    plt.gca().set_aspect("equal", adjustable="box")

    plt.grid(True)
    plt.legend()
    plt.show()

# =========================
# SOLVER
# =========================
def solve(csv_file, show_plot=True):
    warehouse, customers = read_csv(csv_file)
    vehicles = cluster(customers, NUM_VEHICLES)

    total_dist = 0

    for v in vehicles:
        if not v.customers:
            continue

        route, dist = greedy_route(warehouse, v.customers)
        total_dist += dist

        print(f"\nVehicle {v.index + 1}")
        print("Route:", " -> ".join(str(p.index) for p in route))
        print(f"Distance: {dist:.2f} km")

        if show_plot:
            plot_route(v.index + 1, route)

    baseline = sum(haversine(warehouse, c)*2 for c in customers)

    print("\nSUMMARY")
    print(f"Baseline distance : {baseline:.2f} km")
    print(f"Optimized distance: {total_dist:.2f} km")
    print(f"Saved distance   : {baseline - total_dist:.2f} km")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python solver.py locations.csv")
    else:
        solve(sys.argv[1], show_plot=True)
