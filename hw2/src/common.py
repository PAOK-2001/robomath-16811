import numpy as np


# Find an initial window by detecting sign changes in the function.
def initial_root_estimate(f: callable, pivot: float, search_range: int = 3, step: float =  0.001) -> float:
    lower_bound = pivot
    upper_bound = pivot

    prev_val =lower_bound
    while lower_bound > pivot - search_range:
        # Check if there is a sign change
        if f(lower_bound) * f(prev_val) < 0:
            break
        prev_val =lower_bound
        lower_bound -= step
    
    prev_val = upper_bound
    while upper_bound < pivot + search_range:
        # Check if there is a sign change
        if f(upper_bound) * f(prev_val) < 0:
            break
        prev_val = upper_bound
        upper_bound += step
    
    return lower_bound, upper_bound

def array_to_latex_table(c1: list, c2: list, labels: tuple) -> None:
    c1_label, c2_label = labels
    # create a latex table where c1 is the first column and c2 is the second column usign the labels provided
    table = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|c|c|}\n\\hline\n"
    table += f"{c1_label} & {c2_label} \\\\ \\hline\n"
    for i in range(len(c1)):
        table += f"{c1[i]} & {c2[i]} \\\\ \\hline\n"
    table += "\\end{tabular}\n\\end{table}"
    return table    