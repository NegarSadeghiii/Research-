# Research-
Soft Due-Date Extension — CAR-T Supply Chain Optimisation
This notebook replicates the original Pyomo formulation (i-SHIPMENT_Pyomo.ipynb) and adds patient-specific soft due dates.

For each patient p a nonneg lateness variable LATE[p] and a penalty weight PEN[p] are introduced. The due date DUE[p] is a mutable parameter defaulting to the global max turnaround time ND. The model stays feasible even when due dates are tight — lateness is allowed but penalised.

Requirements: Data200_profileA.dat in the working directory, Gurobi (or any MILP solver)
