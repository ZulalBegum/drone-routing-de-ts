# User interface starter   ->   run with:  streamlit run app.py
import matplotlib.pyplot as plt
import streamlit as st

from hybrid import generate_instance, run

st.title("Drone Routing – DE + Tabu Search")

# ---------------- 1) INPUTS (sidebar)
with st.sidebar:
    n = st.slider("Number of customers", 10, 100, 30, step=5)
    seed = st.number_input("Seed", value=7)
    budget = st.slider("Evaluation budget", 2000, 20000, 5000, step=1000)
    start = st.button("Run")

# ---------------- 2) RUN
if start:
    p = generate_instance(n, seed=int(seed))
    with st.spinner("Running..."):
        st.session_state["p"] = p
        st.session_state["DSDE"] = run(p, budget=budget, hybrid=False)
        st.session_state["DE-TS"] = run(p, budget=budget, hybrid=True)

if "p" not in st.session_state:
    st.info("Choose the parameters on the left and press Run.")
    st.stop()

# ---------------- 3) PARETO PLOT
fig, ax = plt.subplots()
for name, color in [("DSDE", "tab:red"), ("DE-TS", "tab:blue")]:
    F = [f for f, _ in st.session_state[name]]
    ax.scatter([f[0] for f in F], [f[1] for f in F], color=color, label=name)
ax.set_xlabel("f1 - setup cost")
ax.set_ylabel("f2 - energy (Wh)")
ax.legend()
st.pyplot(fig)

# ---------------- 4) TODO: solution selection, route map, route table
