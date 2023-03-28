"""Test functions for interacting with deck files."""
from pathlib import Path
from typing import Any

from rich import print
from sympy import Symbol
from sympy.abc import _clash
from sympy.parsing.sympy_parser import parse_expr


def load_deck(fpath: Path) -> list[str]:
    """
    Load a text file into a list of non-empty, non-commented lines.

    The function takes a Path object representing the path to a text file, reads
    its contents into a list of strings, removes empty lines and lines that begin
    with a hash symbol (#), and returns the resulting list.

    Args:
        fpath (Path): A Path object representing the path to the file to be loaded.

    Returns:
        list[str]: A list of non-empty, non-commented lines from the file.

    Example:
        Suppose we have a text file called "deck.txt" containing the following
        lines:

        # This is a comment
        Line 1
        Line 2

        To load this file, we can use the following code:

        >>> from pathlib import Path
        >>> deck_path = Path("deck.txt")
        >>> load_deck(deck_path)
        ["Line 1", "Line 2"]
    """
    with open(fpath, "r") as file:
        lines = file.readlines()
    lines = [
        line.strip()
        for line in lines
        if line.strip() != "" and not line.strip().startswith("#")
    ]
    return lines


def block_to_dict(block: list[str]) -> dict[str, str]:
    """Convert a block of text into a dictionary of key-value pairs.

    The function takes a list of strings as input, where each string is formatted
    as a key-value pair separated by either " = " or ":". The key and value are
    stripped of leading and trailing white space, and any comments following the
    value are removed.

    Args:
        block (list[str]): A list of strings, where each string represents a
        key-value pair.

    Returns:
        dict[str, str]: A dictionary of key-value pairs.

    Example:
        >>> block = ["key1 = value1 # comment",
        ...          "key2:value2"]
        >>> block_to_dict(block)
        {"key1": "value1", "key2": "value2"}
    """
    out = {}
    for b in block:
        b_split = b.split(" = ")
        if len(b_split) != 2:
            b_split = b.split(":")
            if len(b_split) != 2:
                continue
        key, value = [i.strip() for i in b_split]
        value = value.split("#")[0]
        out[key] = value
    return out


def split_to_blocks(lines_list: list[str]) -> dict[str, dict[str, str]]:
    """Split a list of lines into blocks of text.

    Each block is a dictionary of key-value pairs based on the lines between
    "begin:<block_name>" and "end:<block_name>". The block_name is used as
    the key in the returned dictionary.

    Args:
        lines_list (list[str]): A list of strings, each representing a line of text.

    Returns:
        dict[str, dict[str, str]]: A dictionary of blocks, where the keys are the
        block names and the values are dictionaries of key-value pairs based on the
        lines in the block.

    Raises:
        Exception: If no end:<block_name> line is found for a given begin:<block_name>
        line.

    Example:
        >>> lines_list = [
            "begin:block1",
            "key1:value1",
            "key2:value2",
            "end:block1",
            "begin:block2",
            "key3:value3",
            "key4:value4",
            "end:block2",
        ]
        >>> split_to_blocks(lines_list)
        {
            "block1": {
                "key1": "value1",
                "key2": "value2",
            },
            "block2": {
                "key3": "value3",
                "key4": "value4",
            },
        }
    """
    blocks = {}
    current_block_i = [0, 0]
    i = 0
    while i < len(lines_list):
        line = lines_list[i]
        if line.startswith("begin:"):
            block_name = line.split(":")[1]
            current_block_i[0] = i
            for j in range(i, len(lines_list)):
                inner_line = lines_list[j]
                if inner_line.startswith(f"end:{block_name}"):
                    current_block_i[1] = j
                    break
            if current_block_i[1] > current_block_i[0]:
                block_data = lines_list[current_block_i[0] + 1 : current_block_i[1]]
                blocks[block_name] = block_to_dict(block_data)
            else:
                raise Exception(f"No end:{block_name} found!")
            i = current_block_i[1] + 1
        else:
            i += 1

    return blocks


def parse_value(expr: str) -> Any:
    """
    Parse a mathematical expression and return the resulting SymPy object.

    The function takes a string containing a mathematical expression, replaces any
    "^" symbols with "**", parses the expression using SymPy's `parse_expr`
    function and a local dictionary `_clash`, and returns the resulting SymPy
    object.

    Args:
        expr (str): A string containing a mathematical expression.

    Returns:
        Any: A SymPy object representing the parsed mathematical expression.

    Example:
        To parse the expression "3*x**2 + 2*x + 1", we can use the following code:

        >>> parse_value("3*x^2 + 2*x + 1")
        3*x**2 + 2*x + 1
    """
    expr = expr.replace("^", "**")
    parsed = parse_expr(expr, local_dict=_clash)
    return parsed


def parse_constant_block(constant_block: dict[str, str]) -> dict[Symbol, float]:
    """
    Parse a dictionary of constant expressions and return a dictionary of SymPy symbols and their evaluated float values.

    The function takes a dictionary of constant expressions in the form of strings,
    uses the `parse_value` function to parse each expression into a SymPy object,
    evaluates each expression using a dictionary of predefined constants, and
    returns a dictionary mapping SymPy symbols to their float values.

    Args:
        constant_block (dict[str, str]): A dictionary mapping constant names to
            their expressions as strings.

    Returns:
        dict[Symbol, float]: A dictionary mapping SymPy symbols to their
            corresponding float values.

    Raises:
        ValueError: If an expression cannot be converted to a float, indicating
            that an undefined constant is being used.

    Example:
        Suppose we have a constant block in a deck file containing the following
        constants:

        g = 9.81
        pi = 3.14159

        To parse this block and return a dictionary of constants and their values,
        we can use the following code:

        >>> constant_block = {"g": "9.81", "pi": "3.14159"}
        >>> parse_constant_block(constant_block)
        {g: 9.81, pi: 3.14159}
    """
    var_dict = {
        Symbol("qe"): 1.60217663e-19,
        Symbol("mu0"): 1.25663706212e-6,
        Symbol("kb"): 1.380649e-23,
    }
    for var_name, var in constant_block.items():
        parsed_var = parse_value(var)
        eval_var = parsed_var.evalf(subs=var_dict)
        try:
            eval_var = float(eval_var)
        except TypeError:
            raise ValueError(
                f"Expression {var_name} = {eval_var} cannot be converted to float! is there an undefined constant?"
            )
        var_dict[Symbol(var_name)] = eval_var

    return var_dict


def get_deck_constants_sym(deck_path: Path) -> dict[Symbol, float]:
    """
    Parse a simulation input deck file and returns a dictionary of SymPy Symbols and their associated float values.

    Args:
        deck_path (Path): Path to the input deck file.

    Returns:
        dict[Symbol, float]: A dictionary mapping SymPy Symbols to their corresponding float values.

    Raises:
        IOError: If the input deck file cannot be found or opened.
        ValueError: If a constant in the input deck contains an undefined constant or cannot be converted to float.

    Example:
        >>> from pathlib import Path
        >>> deck_path = Path("input.deck")
        >>> constants = get_deck_constants_sym(deck_path)
        >>> print(constants)
        {mp: 1.67e-27, B0: 1e-08, n0: 10000000.0, wci: 1745.2416021588425, va: 3062.5403529371713, di: 1.7553509177693077, beta: 1.0, T0: 70211.80897863494, inflow: 1.005e+25, thBn: 40.0, ppc: 100.0, E0: 603.9289558066488, amp: 0.5, sigma: 0.4388372533379046}
    """
    deck_lines = load_deck(deck_path)
    blocks = split_to_blocks(deck_lines)
    constant = parse_constant_block(blocks["constant"])
    return constant


def get_deck_constants(deck_path: Path) -> dict[str, float]:
    """
    Parse a simulation input deck file and returns a dictionary of constant names and their associated float values.

    Args:
        deck_path (Path): Path to the input deck file.

    Returns:
        dict[str, float]: A dictionary mapping constant names to their corresponding float values.

    Raises:
        IOError: If the input deck file cannot be found or opened.
        SyntaxError: If the input deck contains invalid syntax.
        ValueError: If a constant in the input deck contains an undefined constant or cannot be converted to float.

    Example:
        >>> from pathlib import Path
        >>> deck_path = Path("input.deck")
        >>> constants = get_deck_constants(deck_path)
        >>> print(constants)
        {'mp': 1.67e-27, 'B0': 1e-08, 'n0': 10000000.0, 'wci': 1745.2416021588425, 'va': 3062.5403529371713, 'di': 1.7553509177693077, 'beta': 1.0, 'T0': 70211.80897863494, 'inflow': 1.005e+25, 'thBn': 40.0, 'ppc': 100.0, 'E0': 603.9289558066488, 'amp': 0.5, 'sigma': 0.4388372533379046}
    """
    constant = get_deck_constants_sym(deck_path)
    return {str(k): v for k, v in constant.items()}


if __name__ == "__main__":
    deck_path = Path("U6T40/input.deck")

    deck_lines = load_deck(deck_path)
    blocks = split_to_blocks(deck_lines)
    constant = parse_constant_block(blocks["constant"])
    print(constant)
