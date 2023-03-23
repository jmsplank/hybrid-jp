from pathlib import Path

from rich import print
from sympy import Expr, Symbol
from sympy.abc import _clash
from sympy.parsing.sympy_parser import parse_expr


def load_deck(fpath: Path) -> list[str]:
    """Read input.deck and return a list of lines with light parsing.
     - leading/trailing whitespace and newlines removed
     - comment lines (starting with #) removed
        - Comments within lines not removed

    Args:
        fpath (Path): path to input.deck

    Returns:
        list[str]: lines of input.deck, trimmed and comment lines removed.
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
    """Take a list of 'key = value' or 'key:value' strings
    and comvert to dict.

    Args:
        block (list[str]): the lines within the block

    Returns:
        dict[str, str]: 'key = value' parsed to {key:value}
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
    """Iterate over each line in deck and extract begin:name -> end:name blocks.
    'name' becomes the key of a dict and the lines between are converted to dict
    using block_to_dict()

    Args:
        lines_list (list[str]): Lines in input.deck, commonly from load_deck()

    Raises:
        Exception: Raised when no end:name found for a corresponding begin:name

    Returns:
        dict[str, dict[str, str]]: deck as {'block':{'var1':val1,'var2':val2}} format
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


def parse_value(expr: str) -> Expr:
    """Use sympy parse_expr() to parse string expression.
     - Replaces ^ exponent with ** exponent

    Args:
        expr (str): the expression to be parsed

    Returns:
        Expr: A Sympy expression
    """
    expr = expr.replace("^", "**")
    parsed = parse_expr(expr, local_dict=_clash)
    return parsed


def parse_constant_block(constant_block: dict[str, str]) -> dict[Symbol, float]:
    """Parse a dict of var:value where value is a string that can be transformed
    into an expression. Evaluate the expression as float.

    Args:
        constant_block (dict[str, str]): the Block to parse

    Raises:
        ValueError: Raised when undefined symbols remain in expression
                    that can't be substituted

    Returns:
        dict[Symbol, float]: uses sympy symbols as the key and python float as value
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
    """Wrapper fn that takes a path to an input.deck and returns
    the constants block as a dict of {sympy symbol:float value}.

    Args:
        deck_path (Path): pathlib path object pointing to input.deck

    Returns:
        dict[Symbol, float]: {sympy symbol:float value}
    """
    deck_lines = load_deck(deck_path)
    blocks = split_to_blocks(deck_lines)
    constant = parse_constant_block(blocks["constant"])
    return constant


def get_deck_constants(deck_path: Path) -> dict[str, float]:
    """calls get_deck_constants_sym() and converts sympy key to str key

    Args:
        deck_path (Path): path to input.deck

    Returns:
        dict[str, float]: {const: value}
    """
    constant = get_deck_constants_sym(deck_path)
    return {str(k): v for k, v in constant.items()}


if __name__ == "__main__":
    deck_path = Path("U6T40/input.deck")

    deck_lines = load_deck(deck_path)
    blocks = split_to_blocks(deck_lines)
    constant = parse_constant_block(blocks["constant"])
    print(constant)
