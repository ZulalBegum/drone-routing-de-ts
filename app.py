# User interface starter   ->   run with:  streamlit run app.py
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from hybrid import generate_instance, run, route_energy

st.title("Drone Routing – DE + Tabu Search")

# INPUTS (sidebar)

with st.sidebar:
    n = st.slider("Number of customers", 10, 100, 30, step=5)
    seed = st.number_input("Seed", value=7)
    budget = st.slider("Evaluation budget", 2000, 20000, 5000, step=1000)
    start = st.button("Run")


# RUN
if start:
    p = generate_instance(n, seed=int(seed))
    with st.spinner("Running..."):
        st.session_state["p"] = p
        st.session_state["DSDE"] = run(p, budget=budget, hybrid=False)
        st.session_state["DE-TS"] = run(p, budget=budget, hybrid=True)

if "p" not in st.session_state:
    st.info("Choose the parameters on the left and press Run.")
    st.stop()


# PARETO PLOT
fig, ax = plt.subplots()
for name, color in [("DSDE", "tab:red"), ("DE-TS", "tab:blue")]:
    F = [f for f, _ in st.session_state[name]]
    ax.scatter([f[0] for f in F], [f[1] for f in F], color=color, label=name)
ax.set_xlabel("f1 - setup cost")
ax.set_ylabel("f2 - energy (Wh)")
ax.legend()
st.pyplot(fig)


# SOLUTION SELECTION
st.subheader("Solution details")
algo = st.selectbox("Algorithm", ["DE-TS", "DSDE"])
archive = sorted(st.session_state[algo], key=lambda a: a[0][0])   # sort by f1
labels = [f"#{i}: f1={f[0]:.0f}, f2={f[1]:.0f} Wh, drones={len(sol)}"
          for i, (f, sol) in enumerate(archive)]
choice = st.selectbox("Pareto solution", range(len(archive)), format_func=lambda i: labels[i])
f, solution = archive[choice]
p = st.session_state["p"]

# ROUTE MAP
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(p.stations[:, 0], p.stations[:, 1], marker="s", s=80, color="lightgray", label="candidate station")
used = sorted({station for station, _ in solution})
ax.scatter(p.stations[used, 0], p.stations[used, 1], marker="s", s=120, color="black", label="open station")
ax.scatter(p.customers[:, 0], p.customers[:, 1], s=20, color="tab:blue", zorder=3, label="customer")
for k, (station, route) in enumerate(solution):
    points = np.vstack([p.stations[station], p.customers[route], p.stations[station]])
    ax.plot(points[:, 0], points[:, 1], color=plt.cm.tab20(k % 20))
ax.set_aspect("equal")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig)


# ROUTE TABLE
rows = []
for k, (station, route) in enumerate(solution):
    rows.append({
        "drone": k + 1,
        "station": station,
        "visit order": " -> ".join(str(c) for c in route),
        "load (kg)": round(float(p.demand[route].sum()), 2),
        "energy (Wh)": round(route_energy(p, station, route), 1),
    })
st.dataframe(rows, hide_index=True)

