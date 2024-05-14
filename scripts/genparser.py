import click

from jinja2 import Environment

NONTERMINALS = {
    "s",
    "program",
    "declaration_list",
    "declaration",
    "declaration_initial",
    "declaration_prime",
    "var_declaration_prime",
    "fun_declaration_prime",
    "type_specifier",
    "params",
    "param_list",
    "param",
    "param_prime",
    "compound_stmt",
    "statement_list",
    "statement",
    "expression_stmt",
    "selection_stmt",
    "else_stmt",
    "iteration_stmt",
    "return_stmt",
    "return_stmt_prime",
    "expression",
    "b",
    "h",
    "simple_expression_zegond",
    "simple_expression_prime",
    "c",
    "relop",
    "additive_expression",
    "additive_expression_prime",
    "additive_expression_zegond",
    "d",
    "addop",
    "term",
    "term_prime",
    "term_zegond",
    "g",
    "signed_factor",
    "signed_factor_prime",
    "signed_factor_zegond",
    "factor",
    "var_call_prime",
    "var_prime",
    "factor_prime",
    "factor_zegond",
    "args",
    "arg_list",
    "arg_list_prime",
}


class Statement:
    def __init__(self, is_terminal, value):
        self.is_terminal = is_terminal
        self.value = value

    @classmethod
    def from_value(cls, value):
        return cls(value not in NONTERMINALS, value)

    @classmethod
    def from_values(cls, values):
        return [Statement.from_value(value) for value in values]


class Nonterminal:
    def __init__(self, node_name, goes_to_epsilon, derivations):
        self.node_name = node_name
        self.goes_to_epsilon = goes_to_epsilon
        self.follows = []
        self.derivations = {
            condition: Statement.from_values(values)
            for condition, values in derivations.items()
        }


FOLLOW = {
    "program": ["EOF"],
    "declaration_list": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "for",
        "return",
        "+",
        "-",
        "EOF",
    ],
    "declaration": [
        "ID",
        ";",
        "NUM",
        "(",
        "int",
        "void",
        "{",
        "}",
        "break",
        "if",
        "for",
        "return",
        "+",
        "-",
        "EOF",
    ],
    "declaration_initial": [";", "[", "(", ")", ","],
    "declaration_prime": [
        "ID",
        ";",
        "NUM",
        "(",
        "int",
        "void",
        "{",
        "}",
        "break",
        "if",
        "for",
        "return",
        "+",
        "-",
        "EOF",
    ],
    "var_declaration_prime": [
        "ID",
        ";",
        "NUM",
        "(",
        "int",
        "void",
        "{",
        "}",
        "break",
        "if",
        "for",
        "return",
        "+",
        "-",
        "EOF",
    ],
    "fun_declaration_prime": [
        "ID",
        ";",
        "NUM",
        "(",
        "int",
        "void",
        "{",
        "}",
        "break",
        "if",
        "for",
        "return",
        "+",
        "-",
        "EOF",
    ],
    "type_specifier": ["ID"],
    "params": [")"],
    "param_list": [")"],
    "param": [")", ","],
    "param_prime": [")", ","],
    "compound_stmt": [
        "ID",
        ";",
        "NUM",
        "(",
        "int",
        "void",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
        "EOF",
    ],
    "statement_list": ["}"],
    "statement": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
    ],
    "expression_stmt": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
    ],
    "selection_stmt": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
    ],
    "else_stmt": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
    ],
    "iteration_stmt": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
    ],
    "return_stmt": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
    ],
    "return_stmt_prime": [
        "ID",
        ";",
        "NUM",
        "(",
        "{",
        "}",
        "break",
        "if",
        "endif",
        "else",
        "for",
        "return",
        "+",
        "-",
    ],
    "expression": [";", "]", ")", ","],
    "b": [";", "]", ")", ","],
    "h": [";", "]", ")", ","],
    "simple_expression_zegond": [";", "]", ")", ","],
    "simple_expression_prime": [";", "]", ")", ","],
    "c": [";", "]", ")", ","],
    "relop": ["ID", "NUM", "(", "+", "-"],
    "additive_expression": [";", "]", ")", ","],
    "additive_expression_prime": [";", "]", ")", ",", "<", "=="],
    "additive_expression_zegond": [";", "]", ")", ",", "<", "=="],
    "d": [";", "]", ")", ",", "<", "=="],
    "addop": ["ID", "NUM", "(", "+", "-"],
    "term": [";", "]", ")", ",", "<", "==", "+", "-"],
    "term_prime": [";", "]", ")", ",", "<", "==", "+", "-"],
    "term_zegond": [";", "]", ")", ",", "<", "==", "+", "-"],
    "g": [";", "]", ")", ",", "<", "==", "+", "-"],
    "signed_factor": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "signed_factor_prime": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "signed_factor_zegond": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "factor": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "var_call_prime": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "var_prime": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "factor_prime": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "factor_zegond": [";", "]", ")", ",", "<", "==", "+", "-", "*"],
    "args": [")"],
    "arg_list": [")"],
    "arg_list_prime": [")"],
}


PREDICTIVE_SET = {
    "program": Nonterminal(
        "Program", False, {("EOF", "int", "void"): ["declaration_list"]}
    ),
    "declaration_list": Nonterminal(
        "Declaration-list", True, {("int", "void"): ["declaration", "declaration_list"]}
    ),
    "declaration": Nonterminal(
        "Declaration",
        False,
        {("int", "void"): ["declaration_initial", "declaration_prime"]},
    ),
    "declaration_initial": Nonterminal(
        "Declaration-initial", False, {("int", "void"): ["type_specifier", "ID"]}
    ),
    "declaration_prime": Nonterminal(
        "Declaration-prime",
        False,
        {(";", "["): ["var_declaration_prime"], ("(",): ["fun_declaration_prime"]},
    ),
    "var_declaration_prime": Nonterminal(
        "Var-declaration-prime", False, {(";",): [";"], ("["): ["[", "NUM", "]", ";"]}
    ),
    "fun_declaration_prime": Nonterminal(
        "Fun-declaration-prime", False, {("(",): ["(", "params", ")", "compound_stmt"]}
    ),
    "type_specifier": Nonterminal(
        "Type-specifier", False, {("void",): ["void"], ("int",): ["int"]}
    ),
    "params": Nonterminal(
        "Params",
        False,
        {("void",): ["void"], ("int",): ["int", "ID", "param_prime", "param_list"]},
    ),
    "param_list": Nonterminal(
        "Param-list", True, {(",",): [",", "param", "param_list"]}
    ),
    "param": Nonterminal(
        "Param", False, {("int", "void"): ["declaration_initial", "param_prime"]}
    ),
    "param_prime": Nonterminal("Param-prime", True, {("["): ["[", "]"]}),
    "compound_stmt": Nonterminal(
        "Compound-stmt",
        False,
        {("{",): ["{", "declaration_list", "statement_list", "}"]},
    ),
    "statement_list": Nonterminal(
        "Statement-list",
        True,
        {
            ("ID", ";", "NUM", "(", "{", "break", "if", "for", "return", "+", "-"): [
                "statement",
                "statement_list",
            ]
        },
    ),
    "statement": Nonterminal(
        "Statement",
        False,
        {
            ("ID", ";", "NUM", "(", "break", "+", "-"): ["expression_stmt"],
            ("{",): ["compound_stmt"],
            ("if",): ["selection_stmt"],
            ("for",): ["iteration_stmt"],
            ("return",): ["return_stmt"],
        },
    ),
    "expression_stmt": Nonterminal(
        "Expression-stmt",
        False,
        {
            ("ID", "NUM", "(", "+", "-"): ["expression", ";"],
            (";",): [";"],
            ("break",): ["break", ";"],
        },
    ),
    "selection_stmt": Nonterminal(
        "Selection-stmt",
        False,
        {("if",): ["if", "(", "expression", ")", "statement", "else_stmt"]},
    ),
    "else_stmt": Nonterminal(
        "Else-stmt",
        False,
        {("endif",): ["endif"], ("else",): ["else", "statement", "endif"]},
    ),
    "iteration_stmt": Nonterminal(
        "Iteration-stmt",
        False,
        {
            ("for",): [
                "for",
                "(",
                "expression",
                ";",
                "expression",
                ";",
                "expression",
                ")",
                "statement",
            ]
        },
    ),
    "return_stmt": Nonterminal(
        "Return-stmt", False, {("return",): ["return", "return_stmt_prime"]}
    ),
    "return_stmt_prime": Nonterminal(
        "Return-stmt-prime",
        False,
        {("ID", "+", "-", "(", "NUM"): ["expression", ";"], (";",): [";"]},
    ),
    "expression": Nonterminal(
        "Expression",
        False,
        {("ID",): ["ID", "b"], ("NUM", "(", "+", "-"): ["simple_expression_zegond"]},
    ),
    "b": Nonterminal(
        "B",
        False,
        {
            (";", "]", "(", ")", ",", "<", "==", "+", "-", "*"): [
                "simple_expression_prime"
            ],
            ("[",): ["[", "expression", "]", "h"],
            ("=",): ["=", "expression"],
        },
    ),
    "h": Nonterminal(
        "H",
        False,
        {
            (";", "]", ")", ",", "<", "==", "+", "-", "*"): ["g", "d", "c"],
            ("=",): ["=", "expression"],
        },
    ),
    "simple_expression_zegond": Nonterminal(
        "Simple-expression-zegond",
        False,
        {("NUM", "(", "+", "-"): ["additive_expression_zegond", "c"]},
    ),
    "simple_expression_prime": Nonterminal(
        "Simple-expression-prime",
        False,
        {
            (";", "]", "(", ")", ",", "<", "==", "+", "-", "*"): [
                "additive_expression_prime",
                "c",
            ],
        },
    ),
    "c": Nonterminal("C", True, {("<", "=="): ["relop", "additive_expression"]}),
    "relop": Nonterminal("Relop", False, {("<",): ["<"], ("==",): ["=="]}),
    "additive_expression": Nonterminal(
        "Additive-expression", False, {("ID", "NUM", "(", "+", "-"): ["term", "d"]}
    ),
    "additive_expression_prime": Nonterminal(
        "Additive-expression-prime",
        False,
        {(";", "]", "(", ")", ",", "<", "==", "+", "-", "*"): ["term_prime", "d"]},
    ),
    "additive_expression_zegond": Nonterminal(
        "Additive-expression-zegond",
        False,
        {("NUM", "(", "+", "-"): ["term_zegond", "d"]},
    ),
    "d": Nonterminal("D", True, {("+", "-"): ["addop", "term", "d"]}),
    "addop": Nonterminal("Addop", False, {("+"): ["+"], ("-"): ["-"]}),
    "term": Nonterminal(
        "Term", False, {("+", "-", "(", "ID", "NUM"): ["signed_factor", "g"]}
    ),
    "term_prime": Nonterminal(
        "Term-prime",
        False,
        {
            ("+", "-", "(", ")", ";", "<", "==", "]", ",", "*"): [
                "signed_factor_prime",
                "g",
            ]
        },
    ),
    "term_zegond": Nonterminal(
        "Term-zegond", False, {("+", "-", "(", "NUM"): ["signed_factor_zegond", "g"]}
    ),
    "g": Nonterminal("G", True, {("*",): ["*", "signed_factor", "g"]}),
    "signed_factor": Nonterminal(
        "Signed-factor",
        False,
        {
            ("NUM", "(", "ID"): ["factor"],
            ("+",): ["+", "factor"],
            ("-",): ["-", "factor"],
        },
    ),
    "signed_factor_prime": Nonterminal(
        "Signed-factor-prime",
        False,
        {("(", ";", ",", ")", "<", "==", "+", "-", "*", "]"): ["factor_prime"]},
    ),
    "signed_factor_zegond": Nonterminal(
        "Signed-factor-zegond",
        False,
        {
            ("NUM", "("): ["factor_zegond"],
            ("+",): ["+", "factor"],
            ("-",): ["-", "factor"],
        },
    ),
    "factor": Nonterminal(
        "Factor",
        False,
        {
            ("(",): ["[", "expression", "]"],
            ("NUM",): ["NUM"],
            ("ID",): ["ID", "var_call_prime"],
        },
    ),
    "var_call_prime": Nonterminal(
        "Var-call-prime",
        False,
        {
            ("(",): ["(", "args", ")"],
            (";", ")", "+", "-", "<", "==", "*", "]", ",", "["): ["var_prime"],
        },
    ),
    "var_prime": Nonterminal("Var-prime", True, {("[",): ["[", "expression", "]"]}),
    "factor_prime": Nonterminal("Factor-prime", True, {("(",): ["(", "args", ")"]}),
    "factor_zegond": Nonterminal("Factor-zegond", False, {("NUM", "("): ["NUM"]}),
    "args": Nonterminal("Args", True, {("ID", "NUM", "(", "+", "-"): ["arg_list"]}),
    "arg_list": Nonterminal(
        "Arg-list",
        False,
        {("ID", "NUM", "(", "+", "-"): ["expression", "arg_list_prime"]},
    ),
    "arg_list_prime": Nonterminal(
        "Arg-list-prime", True, {(",",): [",", "expression", "arg_list_prime"]}
    ),
}


class Filters:
    @classmethod
    def add_filters(cls, environment, *filters):
        for filter in filters:
            environment.filters[filter] = getattr(cls, filter)

    @staticmethod
    def to_set(value):
        return "{" + ", ".join(f'"{item}"' for item in value) + "}"


def get_context():
    for name, nonterminal in PREDICTIVE_SET.items():
        nonterminal.follows = FOLLOW[name]

    return {"start": "program", "nonterminals": PREDICTIVE_SET}


@click.command()
@click.option(
    "--out",
    "-o",
    default="parser.py",
    show_default=True,
    type=click.File("w"),
    help="Path of the parser file to generate.",
)
@click.option(
    "--template",
    "-t",
    default="templates/parser.py.j2",
    show_default=True,
    type=click.File("r"),
    help="Path of the template file to generate parser from.",
)
def generate_parser(out, template):
    environment = Environment(trim_blocks=True, lstrip_blocks=True)
    Filters.add_filters(environment, "to_set")
    template = environment.from_string(template.read())
    context = get_context()
    parser = template.render(context)
    out.write(parser)


if __name__ == "__main__":
    generate_parser()
