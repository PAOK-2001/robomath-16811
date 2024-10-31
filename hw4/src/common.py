import pandas as pd

def format_to_latex_table(data: list, headers: tuple) -> str:
    """
    Convert a list of lists to a latex table format.
    """
    table = "\\begin{table}[H]\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{|"
    for _ in headers:
        table += "c|"
    table += "}\n"
    table += "\\hline\n"
    table += " & ".join(headers) + " \\\\ \\hline\n"
    for row in data:
        table += " & ".join([str(x) for x in row]) + " \\\\ \\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    return table

def results_to_df(results: list, headers: tuple) -> pd.DataFrame:
    """
    Convert a list of lists to a pandas dataframe.
    """
    return pd.DataFrame(results, columns=headers)