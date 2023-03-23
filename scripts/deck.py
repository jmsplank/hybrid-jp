from pathlib import Path
from typing import Any

from rich import print
from sympy import Symbol
from sympy.abc import _clash
from sympy.parsing.sympy_parser import parse_expr


def load_deck(fpath: Path) -> list[str]:
    with open(fpath, "r") as file:
        lines = file.readlines()
    lines = [
        line.strip()
        for line in lines
        if line.strip() != "" and not line.strip().startswith("#")
    ]
    return lines


def block_to_dict(block: list[str]) -> dict[str, str]:
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
    expr = expr.replace("^", "**")
    parsed = parse_expr(expr, local_dict=_clash)
    return parsed


def parse_constant_block(constant_block: dict[str, str]) -> dict[Symbol, float]:
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
    deck_lines = load_deck(deck_path)
    blocks = split_to_blocks(deck_lines)
    constant = parse_constant_block(blocks["constant"])
    return constant


def get_deck_constants(deck_path: Path) -> dict[str, float]:
    constant = get_deck_constants_sym(deck_path)
    return {str(k): v for k, v in constant.items()}


if __name__ == "__main__":
    deck_path = Path("U6T40/input.deck")

    deck_lines = load_deck(deck_path)
    blocks = split_to_blocks(deck_lines)
    constant = parse_constant_block(blocks["constant"])
    print(constant)
