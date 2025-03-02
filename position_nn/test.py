import cvxpy

# Data: Demands and undirected edges with costs.
demands = {"A": 2, "B": 2, "C": -3, "D": -7, "E": 6}
undirected_edges = {
    ("A", "B"): 6,
    ("A", "C"): 4,
    ("A", "D"): 3,
    ("B", "D"): 5,
    ("B", "C"): 2,
    ("B", "E"): 2,
    ("C", "E"): 7,
}


def solve_transshipment_integer(demands, undirected_edges):
    # Create directed edges (each undirected edge gives two directed edges)
    directed_edges = {}
    for (i, j), cost in undirected_edges.items():
        directed_edges[(i, j)] = cost
        directed_edges[(j, i)] = cost

    nodes = list(demands.keys())
    edges = list(directed_edges.keys())

    # Create an integer variable for each directed edge.
    x = {}
    for edge in edges:
        # Remove nonneg=True from the constructor to avoid conflict;
        # we add nonnegativity constraints later.
        x[edge] = cvxpy.Variable(integer=True)

    # Build the objective function term by term.
    cost_terms = []
    for edge in edges:
        cost_terms.append(directed_edges[edge] * x[edge])
    objective = cvxpy.Minimize(cvxpy.sum(cost_terms))

    # Build the flow-conservation constraints explicitly.
    constraints = []
    for v in nodes:
        inflow_terms = []
        outflow_terms = []
        for i, j in edges:
            if j == v:
                inflow_terms.append(x[(i, j)])
            if i == v:
                outflow_terms.append(x[(i, j)])
        inflow = cvxpy.sum(inflow_terms)
        outflow = cvxpy.sum(outflow_terms)
        constraints.append(inflow - outflow == demands[v])

    for edge in edges:
        constraints.append(x[edge] >= 0)

    # Formulate and solve the MILP using a MILP solver (e.g., GLPK_MI).
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver=cvxpy.ECOS_BB)

    print("Status:", prob.status)
    print("Optimal Objective Value (Total Cost):", prob.value)
    print("\nFlow on directed edges (only nonzero flows shown):")
    for edge in edges:
        flow_val = x[edge].value
        if flow_val is not None and flow_val > 1e-5:
            print(f"{edge[0]}->{edge[1]}: {flow_val:.0f}")


if __name__ == "__main__":
    solve_transshipment_integer(demands, undirected_edges)
