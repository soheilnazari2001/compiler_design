from enum import Enum, auto


class StatementType(Enum):
    TERMINAL = auto()
    NONTERMINAL = auto()
    CODEGEN = auto()

    @classmethod
    def from_value(cls, value):
        match value:
            case value if value.startswith("#"):
                return cls.CODEGEN
            case value if value in NONTERMINALS:
                return cls.NONTERMINAL
            case _:
                return cls.TERMINAL


class Statement:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    @classmethod
    def from_value(cls, value):
        return cls(StatementType.from_value(value), value)

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
        "Declaration-initial",
        False,
        {
            ("int", "void"): [
                "type_specifier",
                "#save_type",
                "#set_force_declaration_flag",
                "ID",
                "#start_no_push",
                "#pid",
                "#end_no_push",
                "#unset_force_declaration_flag",
            ]
        },
    ),
    "declaration_prime": Nonterminal(
        "Declaration-prime",
        False,
        {
            (";", "["): [
                "var_declaration_prime",
                "#zero_initialize",
                "#void_check_throw",
            ],
            ("(",): ["fun_declaration_prime"],
        },
    ),
    "var_declaration_prime": Nonterminal(
        "Var-declaration-prime",
        False,
        {(";",): [";"], ("["): ["[", "NUM", "#pnum", "]", "#declare_array", ";"]},
    ),
    "fun_declaration_prime": Nonterminal(
        "Fun-declaration-prime",
        False,
        {
            ("(",): [
                "(",
                "#declare_function",
                "#open_scope",
                "#set_function_scope_flag",
                "params",
                ")",
                "compound_stmt",
                "#jump_back",
            ]
        },
    ),
    "type_specifier": Nonterminal(
        "Type-specifier", False, {("void",): ["void", "#void_check"], ("int",): ["int"]}
    ),
    "params": Nonterminal(
        "Params",
        False,
        {
            ("void",): ["void"],
            ("int",): [
                "int",
                "#save_type",
                "#set_force_declaration_flag",
                "ID",
                "#pid",
                "#unset_force_declaration_flag",
                "param_prime",
                "#pop_param",
                "param_list",
            ],
        },
    ),
    "param_list": Nonterminal(
        "Param-list", True, {(",",): [",", "param", "param_list"]}
    ),
    "param": Nonterminal(
        "Param",
        False,
        {("int", "void"): ["declaration_initial", "param_prime", "#pop_param"]},
    ),
    "param_prime": Nonterminal(
        "Param-prime", True, {("["): ["[", "]", "#array_param"]}
    ),
    "compound_stmt": Nonterminal(
        "Compound-stmt",
        False,
        {
            ("{",): [
                "#open_scope",
                "{",
                "declaration_list",
                "statement_list",
                "#close_scope",
                "}",
            ]
        },
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
            ("ID", "NUM", "(", "+", "-"): ["expression", "#pop", ";"],
            (";",): [";"],
            ("break",): ["break", "#break", ";"],
        },
    ),
    "selection_stmt": Nonterminal(
        "Selection-stmt",
        False,
        {
            ("if",): [
                "if",
                "(",
                "#start_rhs",
                "expression",
                "#end_rhs",
                ")",
                "#save",
                "statement",
                "else_stmt",
            ]
        },
    ),
    "else_stmt": Nonterminal(
        "Else-stmt",
        False,
        {
            ("endif",): ["endif", "#jpf_from_saved"],
            ("else",): [
                "else",
                "#save_and_jpf_from_last_save",
                "statement",
                "endif",
                "#jp_from_saved",
            ],
        },
    ),
    "iteration_stmt": Nonterminal(
        "Iteration-stmt",
        False,
        {
            ("for",): [
                "for",
                "(",
                "#start_rhs",
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
        "Return-stmt",
        False,
        {
            ("return",): [
                "return",
                "#start_rhs",
                "return_stmt_prime",
                "#end_rhs",
                "#jump_back",
            ]
        },
    ),
    "return_stmt_prime": Nonterminal(
        "Return-stmt-prime",
        False,
        {
            ("ID", "+", "-", "(", "NUM"): ["expression", "#set_return_value", ";"],
            (";",): [";"],
        },
    ),
    "expression": Nonterminal(
        "Expression",
        False,
        {
            ("ID",): [
                "ID",
                "#check_declaration",
                "#pid",
                "#uncheck_declaration",
                "#check_type",
                "b",
            ],
            ("NUM", "(", "+", "-"): ["simple_expression_zegond"],
        },
    ),
    "b": Nonterminal(
        "B",
        False,
        {
            (";", "]", "(", ")", ",", "<", "==", "+", "-", "*"): [
                "simple_expression_prime"
            ],
            ("[",): ["[", "#start_rhs", "expression", "#end_rhs", "]", "#array", "h"],
            ("=",): ["=", "#start_rhs", "expression", "#assign", "#end_rhs"],
        },
    ),
    "h": Nonterminal(
        "H",
        False,
        {
            (";", "]", ")", ",", "<", "==", "+", "-", "*"): ["g", "d", "c"],
            ("=",): ["=", "#start_rhs", "expression", "#assign", "#end_rhs"],
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
    "c": Nonterminal(
        "C", True, {("<", "=="): ["relop", "additive_expression", "#execute"]}
    ),
    "relop": Nonterminal(
        "Relop",
        False,
        {("<",): ["<", "#push_operation"], ("==",): ["==", "#push_operation"]},
    ),
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
    "d": Nonterminal("D", True, {("+", "-"): ["addop", "term", "#execute", "d"]}),
    "addop": Nonterminal(
        "Addop",
        False,
        {("+"): ["+", "#push_operation"], ("-"): ["-", "#push_operation"]},
    ),
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
    "g": Nonterminal(
        "G", True, {("*",): ["*", "#push_operation", "signed_factor", "#execute", "g"]}
    ),
    "signed_factor": Nonterminal(
        "Signed-factor",
        False,
        {
            ("NUM", "(", "ID"): ["factor"],
            ("+",): ["+", "factor"],
            ("-",): ["-", "factor", "#negate"],
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
            ("-",): ["-", "factor", "#negate"],
        },
    ),
    "factor": Nonterminal(
        "Factor",
        False,
        {
            ("(",): ["(", "#start_rhs", "expression", "#end_rhs", ")"],
            ("NUM",): ["NUM", "#pnum"],
            ("ID",): [
                "ID",
                "#check_declaration",
                "#pid",
                "#uncheck_declaration",
                "var_call_prime",
            ],
        },
    ),
    "var_call_prime": Nonterminal(
        "Var-call-prime",
        False,
        {
            ("(",): [
                "(",
                "#start_argument_list",
                "args",
                "#end_argument_list",
                ")",
                "#call",
            ],
            (";", ")", "+", "-", "<", "==", "*", "]", ",", "["): ["var_prime"],
        },
    ),
    "var_prime": Nonterminal(
        "Var-prime",
        True,
        {("[",): ["[", "#start_rhs", "expression", "#end_rhs", "]", "#array"]},
    ),
    "factor_prime": Nonterminal(
        "Factor-prime",
        True,
        {
            ("(",): [
                "(",
                "#start_argument_list",
                "args",
                "#end_argument_list",
                ")",
                "#call",
            ]
        },
    ),
    "factor_zegond": Nonterminal(
        "Factor-zegond",
        False,
        {
            ("NUM",): ["NUM", "#pnum"],
            ("(",): ["(", "#start_rhs", "expression", "#end_rhs", ")"],
        },
    ),
    "args": Nonterminal("Args", True, {("ID", "NUM", "(", "+", "-"): ["arg_list"]}),
    "arg_list": Nonterminal(
        "Arg-list",
        False,
        {
            ("ID", "NUM", "(", "+", "-"): [
                "expression",
                "#add_argument_count",
                "arg_list_prime",
            ]
        },
    ),
    "arg_list_prime": Nonterminal(
        "Arg-list-prime",
        True,
        {(",",): [",", "expression", "#add_argument_count", "arg_list_prime"]},
    ),
}
