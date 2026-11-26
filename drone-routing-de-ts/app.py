# User interface   ->   run with:  streamlit run app.py
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from hybrid import Problem, run, details

st.title("Two-Echelon Truck–Drone Routing – DE + Tabu Search")

# ---------------- 1) INPUTS (sidebar)
with st.sidebar:
    files = sorted(glob.glob("data/*.csv"), key=lambda f: (int(f.split(",")[-1][:-4]), f))
    names = [os.path.basename(f)[:-4] for f in files]
    instance = st.selectbox("Instance (article benchmark)", names, index=names.index("Ca1-2,3,15"))
    budget = st.slider("Evaluation budget", 500, 5000, 2000, step=500)
    seed = st.number_input("Seed", value=0)
    start = st.button("Run")

# ---------------- 2) RUN
if start:
    p = Problem(f"data/{instance}.csv")
    with st.spinner("Running DSDE and DE-TS..."):
        st.session_state["p"] = p
        st.session_state["DSDE"] = run(p, budget=budget, hybrid=False, seed=int(seed))
        st.session_state["DE-TS"] = run(p, budget=budget, hybrid=True, seed=int(seed))

if "p" not in st.session_state:
    st.info("Choose an instance on the left and press Run.")
    st.stop()
p = st.session_state["p"]
st.caption(f"{p.name}: {p.n} customers, {p.S} satellites, {p.P} depots")

# ---------------- 3) PARETO PLOT
fig, ax = plt.subplots()
for name, color in [("DSDE", "tab:red"), ("DE-TS", "tab:blue")]:
    F = np.array(sorted(f.tolist() for f, _ in st.session_state[name]))
    ax.step(F[:, 0], F[:, 1], where="post", color=color, alpha=0.4)
    ax.scatter(F[:, 0], F[:, 1], color=color, label=f"{name} ({len(F)} solutions)")
ax.set_xlabel("f1 - dispatch cost ($)")
ax.set_ylabel("f2 - operational cost ($)")
ax.legend()
st.pyplot(fig)

# ---------------- 4) SOLUTION SELECTION
st.subheader("Solution details")
algo = st.selectbox("Algorithm", ["DE-TS", "DSDE"])
archive = sorted(st.session_state[algo], key=lambda a: a[0][0])
labels = [f"#{i}: f1={f[0]:.2f}, f2={f[1]:.3f}, drones={len(sol)}" for i, (f, sol) in enumerate(archive)]
choice = st.selectbox("Pareto solution", range(len(archive)), format_func=lambda i: labels[i])
f, solution = archive[choice]
info = details(p, solution)

# ---------------- 5) ROUTE MAP
fig, ax = plt.subplots(figsize=(7, 7))
for d, order in info["truck_routes"]:                                   # 1st echelon: trucks
    pts = np.vstack([p.depots[d], p.sats[order], p.depots[d]])
    ax.plot(pts[:, 0], pts[:, 1], "k--", lw=2, alpha=0.6)
for k, dr in enumerate(info["drones"]):                                 # 2nd echelon: drones
    pts = np.vstack([p.sats[dr["sat"]], p.customers[dr["route"]], p.sats[dr["end_sat"]]])
    ax.plot(pts[:, 0], pts[:, 1], color=plt.cm.tab20(k % 20), lw=1.3)
ax.scatter(*p.depots.T, marker="^", s=150, color="tab:red", label="depot", zorder=4)
ax.scatter(*p.sats.T, marker="s", s=110, color="black", label="satellite", zorder=4)
ax.scatter(*p.customers[~p.pickup].T, s=30, color="tab:blue", label="delivery customer", zorder=3)
ax.scatter(*p.customers[p.pickup].T, s=30, color="tab:orange", label="pickup customer", zorder=3)
ax.set_aspect("equal")
ax.set_xlabel("km"); ax.set_ylabel("km")
ax.legend(loc="best", fontsize=8)
st.pyplot(fig)
st.caption(f"Dashed black: truck route(s), {info['truck_km']:.2f} km. Coloured lines: drone trips.")

# ---------------- 6) ROUTE TABLE
rows = []
for k, dr in enumerate(info["drones"]):
    rows.append({
        "drone": k + 1,
        "launch satellite": dr["sat"],
        "visit order": " -> ".join(str(c) for c in dr["route"]),
        "return satellite": dr["end_sat"],
        "max payload (kg)": round(dr["max_load"], 1),
        "energy (Wh)": round(dr["energy"], 1),
        "late (min)": round(dr["lateness"], 1),
    })
st.dataframe(rows, hide_index=True)
