import anytree
from anytree import RenderTree
from scanner import Scanner


class Node(anytree.Node):
    def __init__(self, type=None, value=None, parent=None, children=None, **kwargs):
        if value is not None:
            type = f"({type}, {value})"
        super().__init__(type, parent=parent, children=children, **kwargs)


class UnexpectedEOFException(Exception):
    def __init__(self):
        super().__init__("Unepected EOF")


class Parser:
    def __init__(self, input_file):
        self.scanner = Scanner(input_file)
        self.lookahead = None
        self.errors = []
        self.root = None
        self._terminal_to_node_map = {"EOF": "$"}
        self.next_token()
    
    def get_node_name_from_token(self, token):
        if token.terminal in self._terminal_to_node_map:
            return self._terminal_to_node_map[token.terminal]
        
        return f"({token.token_type}, {token.token_value})"

    def print_tree(self, output_file_path):
        tree = "\n".join(f"{pre}{node.name}" for pre, _, node in RenderTree(self.root))
        with open(output_file_path, "w") as output_file:
            output_file.write(tree)

    def print_errors(self, output_file_path):
        errors = "\n".join(self.errors) if self.errors else "There is no syntax error."
        with open(output_file_path, "w") as output_file:
            output_file.write(errors)

    def next_token(self):
        self.lookahead = self.scanner.get_next_token()

    def match(self, expected, parent=None):
        if self.lookahead.terminal == expected:
            node_name = self.get_node_name_from_token(self.lookahead)
            child = Node(node_name, parent=parent)
            self.next_token()
            return child
        
        if self.lookahead.terminal == "EOF":
            self.raise_eof_error()
            return

        self.raise_missing_error(expected, parent=parent)

    def raise_error(self, message):
        self.errors.append(f"#{self.scanner.reader.line_number} : syntax error, {message}")

    def raise_illegal_error(self, expected, actual, parent=None):
        self.raise_error(f"illegal {actual}")
        if actual != "EOF":
            self.next_token()

    def raise_missing_error(self, missing, parent=None):
        self.raise_error(f"missing {missing}")
    
    def raise_eof_error(self):
        self.scanner.reader.line_number += 1
        self.raise_error("Unexpected EOF")
        raise UnexpectedEOFException()

    def parse(self):
        try:
            self.parse_program(None)
        except UnexpectedEOFException:
            pass

    def epsilon(self, parent):
        node = Node("epsilon", parent=parent)
        return node

    def parse_program(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside program and seeing", terminal)
            if terminal in {"EOF", "int", "void"}:
                self.root = node = Node("Program", parent=parent)
                self.parse_declaration_list(node)
                Node("$", parent=node)
                return node
            if terminal in {"EOF"}:
                self.raise_missing_error("Program", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Program", terminal, parent=parent)

    def parse_declaration_list(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside declaration_list and seeing", terminal)
            if terminal in {"int", "void"}:
                node = Node("Declaration-list", parent=parent)
                self.parse_declaration(node)
                self.parse_declaration_list(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "for", "return", "+", "-", "EOF"}:
                node = Node("Declaration-list", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Declaration-list", terminal, parent=parent)

    def parse_declaration(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside declaration and seeing", terminal)
            if terminal in {"int", "void"}:
                node = Node("Declaration", parent=parent)
                self.parse_declaration_initial(node)
                self.parse_declaration_prime(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return", "+", "-", "EOF"}:
                self.raise_missing_error("Declaration", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Declaration", terminal, parent=parent)

    def parse_declaration_initial(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside declaration_initial and seeing", terminal)
            if terminal in {"int", "void"}:
                node = Node("Declaration-initial", parent=parent)
                self.parse_type_specifier(node)
                self.match("ID", parent=node)
                return node
            if terminal in {";", "[", "(", ")", ","}:
                self.raise_missing_error("Declaration-initial", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Declaration-initial", terminal, parent=parent)

    def parse_declaration_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside declaration_prime and seeing", terminal)
            if terminal in {";", "["}:
                node = Node("Declaration-prime", parent=parent)
                self.parse_var_declaration_prime(node)
                return node
            if terminal in {"("}:
                node = Node("Declaration-prime", parent=parent)
                self.parse_fun_declaration_prime(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return", "+", "-", "EOF"}:
                self.raise_missing_error("Declaration-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Declaration-prime", terminal, parent=parent)

    def parse_var_declaration_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside var_declaration_prime and seeing", terminal)
            if terminal in {";"}:
                node = Node("Var-declaration-prime", parent=parent)
                self.match(";", parent=node)
                return node
            if terminal in {"["}:
                node = Node("Var-declaration-prime", parent=parent)
                self.match("[", parent=node)
                self.match("NUM", parent=node)
                self.match("]", parent=node)
                self.match(";", parent=node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return", "+", "-", "EOF"}:
                self.raise_missing_error("Var-declaration-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Var-declaration-prime", terminal, parent=parent)

    def parse_fun_declaration_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside fun_declaration_prime and seeing", terminal)
            if terminal in {"("}:
                node = Node("Fun-declaration-prime", parent=parent)
                self.match("(", parent=node)
                self.parse_params(node)
                self.match(")", parent=node)
                self.parse_compound_stmt(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return", "+", "-", "EOF"}:
                self.raise_missing_error("Fun-declaration-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Fun-declaration-prime", terminal, parent=parent)

    def parse_type_specifier(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside type_specifier and seeing", terminal)
            if terminal in {"void"}:
                node = Node("Type-specifier", parent=parent)
                self.match("void", parent=node)
                return node
            if terminal in {"int"}:
                node = Node("Type-specifier", parent=parent)
                self.match("int", parent=node)
                return node
            if terminal in {"ID"}:
                self.raise_missing_error("Type-specifier", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Type-specifier", terminal, parent=parent)

    def parse_params(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside params and seeing", terminal)
            if terminal in {"void"}:
                node = Node("Params", parent=parent)
                self.match("void", parent=node)
                return node
            if terminal in {"int"}:
                node = Node("Params", parent=parent)
                self.match("int", parent=node)
                self.match("ID", parent=node)
                self.parse_param_prime(node)
                self.parse_param_list(node)
                return node
            if terminal in {")"}:
                self.raise_missing_error("Params", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Params", terminal, parent=parent)

    def parse_param_list(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside param_list and seeing", terminal)
            if terminal in {","}:
                node = Node("Param-list", parent=parent)
                self.match(",", parent=node)
                self.parse_param(node)
                self.parse_param_list(node)
                return node
            if terminal in {")"}:
                node = Node("Param-list", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Param-list", terminal, parent=parent)

    def parse_param(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside param and seeing", terminal)
            if terminal in {"int", "void"}:
                node = Node("Param", parent=parent)
                self.parse_declaration_initial(node)
                self.parse_param_prime(node)
                return node
            if terminal in {")", ","}:
                self.raise_missing_error("Param", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Param", terminal, parent=parent)

    def parse_param_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside param_prime and seeing", terminal)
            if terminal in {"["}:
                node = Node("Param-prime", parent=parent)
                self.match("[", parent=node)
                self.match("]", parent=node)
                return node
            if terminal in {")", ","}:
                node = Node("Param-prime", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Param-prime", terminal, parent=parent)

    def parse_compound_stmt(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside compound_stmt and seeing", terminal)
            if terminal in {"{"}:
                node = Node("Compound-stmt", parent=parent)
                self.match("{", parent=node)
                self.parse_declaration_list(node)
                self.parse_statement_list(node)
                self.match("}", parent=node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-", "EOF"}:
                self.raise_missing_error("Compound-stmt", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Compound-stmt", terminal, parent=parent)

    def parse_statement_list(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside statement_list and seeing", terminal)
            if terminal in {"ID", ";", "NUM", "(", "{", "break", "if", "for", "return", "+", "-"}:
                node = Node("Statement-list", parent=parent)
                self.parse_statement(node)
                self.parse_statement_list(node)
                return node
            if terminal in {"}"}:
                node = Node("Statement-list", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Statement-list", terminal, parent=parent)

    def parse_statement(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside statement and seeing", terminal)
            if terminal in {"ID", ";", "NUM", "(", "break", "+", "-"}:
                node = Node("Statement", parent=parent)
                self.parse_expression_stmt(node)
                return node
            if terminal in {"{"}:
                node = Node("Statement", parent=parent)
                self.parse_compound_stmt(node)
                return node
            if terminal in {"if"}:
                node = Node("Statement", parent=parent)
                self.parse_selection_stmt(node)
                return node
            if terminal in {"for"}:
                node = Node("Statement", parent=parent)
                self.parse_iteration_stmt(node)
                return node
            if terminal in {"return"}:
                node = Node("Statement", parent=parent)
                self.parse_return_stmt(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-"}:
                self.raise_missing_error("Statement", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Statement", terminal, parent=parent)

    def parse_expression_stmt(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside expression_stmt and seeing", terminal)
            if terminal in {"ID", "NUM", "(", "+", "-"}:
                node = Node("Expression-stmt", parent=parent)
                self.parse_expression(node)
                self.match(";", parent=node)
                return node
            if terminal in {";"}:
                node = Node("Expression-stmt", parent=parent)
                self.match(";", parent=node)
                return node
            if terminal in {"break"}:
                node = Node("Expression-stmt", parent=parent)
                self.match("break", parent=node)
                self.match(";", parent=node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-"}:
                self.raise_missing_error("Expression-stmt", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Expression-stmt", terminal, parent=parent)

    def parse_selection_stmt(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside selection_stmt and seeing", terminal)
            if terminal in {"if"}:
                node = Node("Selection-stmt", parent=parent)
                self.match("if", parent=node)
                self.match("(", parent=node)
                self.parse_expression(node)
                self.match(")", parent=node)
                self.parse_statement(node)
                self.parse_else_stmt(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-"}:
                self.raise_missing_error("Selection-stmt", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Selection-stmt", terminal, parent=parent)

    def parse_else_stmt(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside else_stmt and seeing", terminal)
            if terminal in {"endif"}:
                node = Node("Else-stmt", parent=parent)
                self.match("endif", parent=node)
                return node
            if terminal in {"else"}:
                node = Node("Else-stmt", parent=parent)
                self.match("else", parent=node)
                self.parse_statement(node)
                self.match("endif", parent=node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-"}:
                self.raise_missing_error("Else-stmt", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Else-stmt", terminal, parent=parent)

    def parse_iteration_stmt(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside iteration_stmt and seeing", terminal)
            if terminal in {"for"}:
                node = Node("Iteration-stmt", parent=parent)
                self.match("for", parent=node)
                self.match("(", parent=node)
                self.parse_expression(node)
                self.match(";", parent=node)
                self.parse_expression(node)
                self.match(";", parent=node)
                self.parse_expression(node)
                self.match(")", parent=node)
                self.parse_statement(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-"}:
                self.raise_missing_error("Iteration-stmt", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Iteration-stmt", terminal, parent=parent)

    def parse_return_stmt(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside return_stmt and seeing", terminal)
            if terminal in {"return"}:
                node = Node("Return-stmt", parent=parent)
                self.match("return", parent=node)
                self.parse_return_stmt_prime(node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-"}:
                self.raise_missing_error("Return-stmt", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Return-stmt", terminal, parent=parent)

    def parse_return_stmt_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside return_stmt_prime and seeing", terminal)
            if terminal in {"ID", "+", "-", "(", "NUM"}:
                node = Node("Return-stmt-prime", parent=parent)
                self.parse_expression(node)
                self.match(";", parent=node)
                return node
            if terminal in {";"}:
                node = Node("Return-stmt-prime", parent=parent)
                self.match(";", parent=node)
                return node
            if terminal in {"ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+", "-"}:
                self.raise_missing_error("Return-stmt-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Return-stmt-prime", terminal, parent=parent)

    def parse_expression(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside expression and seeing", terminal)
            if terminal in {"ID"}:
                node = Node("Expression", parent=parent)
                self.match("ID", parent=node)
                self.parse_b(node)
                return node
            if terminal in {"NUM", "(", "+", "-"}:
                node = Node("Expression", parent=parent)
                self.parse_simple_expression_zegond(node)
                return node
            if terminal in {";", "]", ")", ","}:
                self.raise_missing_error("Expression", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Expression", terminal, parent=parent)

    def parse_b(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside b and seeing", terminal)
            if terminal in {";", "]", "(", ")", ",", "<", "==", "+", "-", "*"}:
                node = Node("B", parent=parent)
                self.parse_simple_expression_prime(node)
                return node
            if terminal in {"["}:
                node = Node("B", parent=parent)
                self.match("[", parent=node)
                self.parse_expression(node)
                self.match("]", parent=node)
                self.parse_h(node)
                return node
            if terminal in {"="}:
                node = Node("B", parent=parent)
                self.match("=", parent=node)
                self.parse_expression(node)
                return node
            if terminal in {";", "]", ")", ","}:
                self.raise_missing_error("B", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("B", terminal, parent=parent)

    def parse_h(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside h and seeing", terminal)
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                node = Node("H", parent=parent)
                self.parse_g(node)
                self.parse_d(node)
                self.parse_c(node)
                return node
            if terminal in {"="}:
                node = Node("H", parent=parent)
                self.match("=", parent=node)
                self.parse_expression(node)
                return node
            if terminal in {";", "]", ")", ","}:
                self.raise_missing_error("H", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("H", terminal, parent=parent)

    def parse_simple_expression_zegond(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside simple_expression_zegond and seeing", terminal)
            if terminal in {"NUM", "(", "+", "-"}:
                node = Node("Simple-expression-zegond", parent=parent)
                self.parse_additive_expression_zegond(node)
                self.parse_c(node)
                return node
            if terminal in {";", "]", ")", ","}:
                self.raise_missing_error("Simple-expression-zegond", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Simple-expression-zegond", terminal, parent=parent)

    def parse_simple_expression_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside simple_expression_prime and seeing", terminal)
            if terminal in {";", "]", "(", ")", ",", "<", "==", "+", "-", "*"}:
                node = Node("Simple-expression-prime", parent=parent)
                self.parse_additive_expression_prime(node)
                self.parse_c(node)
                return node
            if terminal in {";", "]", ")", ","}:
                self.raise_missing_error("Simple-expression-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Simple-expression-prime", terminal, parent=parent)

    def parse_c(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside c and seeing", terminal)
            if terminal in {"<", "=="}:
                node = Node("C", parent=parent)
                self.parse_relop(node)
                self.parse_additive_expression(node)
                return node
            if terminal in {";", "]", ")", ","}:
                node = Node("C", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("C", terminal, parent=parent)

    def parse_relop(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside relop and seeing", terminal)
            if terminal in {"<"}:
                node = Node("Relop", parent=parent)
                self.match("<", parent=node)
                return node
            if terminal in {"=="}:
                node = Node("Relop", parent=parent)
                self.match("==", parent=node)
                return node
            if terminal in {"ID", "NUM", "(", "+", "-"}:
                self.raise_missing_error("Relop", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Relop", terminal, parent=parent)

    def parse_additive_expression(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside additive_expression and seeing", terminal)
            if terminal in {"ID", "NUM", "(", "+", "-"}:
                node = Node("Additive-expression", parent=parent)
                self.parse_term(node)
                self.parse_d(node)
                return node
            if terminal in {";", "]", ")", ","}:
                self.raise_missing_error("Additive-expression", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Additive-expression", terminal, parent=parent)

    def parse_additive_expression_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside additive_expression_prime and seeing", terminal)
            if terminal in {";", "]", "(", ")", ",", "<", "==", "+", "-", "*"}:
                node = Node("Additive-expression-prime", parent=parent)
                self.parse_term_prime(node)
                self.parse_d(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "=="}:
                self.raise_missing_error("Additive-expression-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Additive-expression-prime", terminal, parent=parent)

    def parse_additive_expression_zegond(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside additive_expression_zegond and seeing", terminal)
            if terminal in {"NUM", "(", "+", "-"}:
                node = Node("Additive-expression-zegond", parent=parent)
                self.parse_term_zegond(node)
                self.parse_d(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "=="}:
                self.raise_missing_error("Additive-expression-zegond", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Additive-expression-zegond", terminal, parent=parent)

    def parse_d(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside d and seeing", terminal)
            if terminal in {"+", "-"}:
                node = Node("D", parent=parent)
                self.parse_addop(node)
                self.parse_term(node)
                self.parse_d(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "=="}:
                node = Node("D", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("D", terminal, parent=parent)

    def parse_addop(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside addop and seeing", terminal)
            if terminal in {"+"}:
                node = Node("Addop", parent=parent)
                self.match("+", parent=node)
                return node
            if terminal in {"-"}:
                node = Node("Addop", parent=parent)
                self.match("-", parent=node)
                return node
            if terminal in {"ID", "NUM", "(", "+", "-"}:
                self.raise_missing_error("Addop", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Addop", terminal, parent=parent)

    def parse_term(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside term and seeing", terminal)
            if terminal in {"+", "-", "(", "ID", "NUM"}:
                node = Node("Term", parent=parent)
                self.parse_signed_factor(node)
                self.parse_g(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-"}:
                self.raise_missing_error("Term", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Term", terminal, parent=parent)

    def parse_term_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside term_prime and seeing", terminal)
            if terminal in {"+", "-", "(", ")", ";", "<", "==", "]", ",", "*"}:
                node = Node("Term-prime", parent=parent)
                self.parse_signed_factor_prime(node)
                self.parse_g(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-"}:
                self.raise_missing_error("Term-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Term-prime", terminal, parent=parent)

    def parse_term_zegond(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside term_zegond and seeing", terminal)
            if terminal in {"+", "-", "(", "NUM"}:
                node = Node("Term-zegond", parent=parent)
                self.parse_signed_factor_zegond(node)
                self.parse_g(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-"}:
                self.raise_missing_error("Term-zegond", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Term-zegond", terminal, parent=parent)

    def parse_g(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside g and seeing", terminal)
            if terminal in {"*"}:
                node = Node("G", parent=parent)
                self.match("*", parent=node)
                self.parse_signed_factor(node)
                self.parse_g(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-"}:
                node = Node("G", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("G", terminal, parent=parent)

    def parse_signed_factor(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside signed_factor and seeing", terminal)
            if terminal in {"NUM", "(", "ID"}:
                node = Node("Signed-factor", parent=parent)
                self.parse_factor(node)
                return node
            if terminal in {"+"}:
                node = Node("Signed-factor", parent=parent)
                self.match("+", parent=node)
                self.parse_factor(node)
                return node
            if terminal in {"-"}:
                node = Node("Signed-factor", parent=parent)
                self.match("-", parent=node)
                self.parse_factor(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                self.raise_missing_error("Signed-factor", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Signed-factor", terminal, parent=parent)

    def parse_signed_factor_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside signed_factor_prime and seeing", terminal)
            if terminal in {"(", ";", ",", ")", "<", "==", "+", "-", "*", "]"}:
                node = Node("Signed-factor-prime", parent=parent)
                self.parse_factor_prime(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                self.raise_missing_error("Signed-factor-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Signed-factor-prime", terminal, parent=parent)

    def parse_signed_factor_zegond(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside signed_factor_zegond and seeing", terminal)
            if terminal in {"NUM", "("}:
                node = Node("Signed-factor-zegond", parent=parent)
                self.parse_factor_zegond(node)
                return node
            if terminal in {"+"}:
                node = Node("Signed-factor-zegond", parent=parent)
                self.match("+", parent=node)
                self.parse_factor(node)
                return node
            if terminal in {"-"}:
                node = Node("Signed-factor-zegond", parent=parent)
                self.match("-", parent=node)
                self.parse_factor(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                self.raise_missing_error("Signed-factor-zegond", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Signed-factor-zegond", terminal, parent=parent)

    def parse_factor(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside factor and seeing", terminal)
            if terminal in {"("}:
                node = Node("Factor", parent=parent)
                self.match("(", parent=node)
                self.parse_expression(node)
                self.match(")", parent=node)
                return node
            if terminal in {"NUM"}:
                node = Node("Factor", parent=parent)
                self.match("NUM", parent=node)
                return node
            if terminal in {"ID"}:
                node = Node("Factor", parent=parent)
                self.match("ID", parent=node)
                self.parse_var_call_prime(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                self.raise_missing_error("Factor", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Factor", terminal, parent=parent)

    def parse_var_call_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside var_call_prime and seeing", terminal)
            if terminal in {"("}:
                node = Node("Var-call-prime", parent=parent)
                self.match("(", parent=node)
                self.parse_args(node)
                self.match(")", parent=node)
                return node
            if terminal in {";", ")", "+", "-", "<", "==", "*", "]", ",", "["}:
                node = Node("Var-call-prime", parent=parent)
                self.parse_var_prime(node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                self.raise_missing_error("Var-call-prime", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Var-call-prime", terminal, parent=parent)

    def parse_var_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside var_prime and seeing", terminal)
            if terminal in {"["}:
                node = Node("Var-prime", parent=parent)
                self.match("[", parent=node)
                self.parse_expression(node)
                self.match("]", parent=node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                node = Node("Var-prime", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Var-prime", terminal, parent=parent)

    def parse_factor_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside factor_prime and seeing", terminal)
            if terminal in {"("}:
                node = Node("Factor-prime", parent=parent)
                self.match("(", parent=node)
                self.parse_args(node)
                self.match(")", parent=node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                node = Node("Factor-prime", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Factor-prime", terminal, parent=parent)

    def parse_factor_zegond(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside factor_zegond and seeing", terminal)
            if terminal in {"NUM"}:
                node = Node("Factor-zegond", parent=parent)
                self.match("NUM", parent=node)
                return node
            if terminal in {"("}:
                node = Node("Factor-zegond", parent=parent)
                self.match("(", parent=node)
                self.parse_expression(node)
                self.match(")", parent=node)
                return node
            if terminal in {";", "]", ")", ",", "<", "==", "+", "-", "*"}:
                self.raise_missing_error("Factor-zegond", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Factor-zegond", terminal, parent=parent)

    def parse_args(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside args and seeing", terminal)
            if terminal in {"ID", "NUM", "(", "+", "-"}:
                node = Node("Args", parent=parent)
                self.parse_arg_list(node)
                return node
            if terminal in {")"}:
                node = Node("Args", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Args", terminal, parent=parent)

    def parse_arg_list(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside arg_list and seeing", terminal)
            if terminal in {"ID", "NUM", "(", "+", "-"}:
                node = Node("Arg-list", parent=parent)
                self.parse_expression(node)
                self.parse_arg_list_prime(node)
                return node
            if terminal in {")"}:
                self.raise_missing_error("Arg-list", parent=parent)
                return None
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Arg-list", terminal, parent=parent)

    def parse_arg_list_prime(self, parent):
        while True:
            terminal = self.lookahead.terminal
            print("inside arg_list_prime and seeing", terminal)
            if terminal in {","}:
                node = Node("Arg-list-prime", parent=parent)
                self.match(",", parent=node)
                self.parse_expression(node)
                self.parse_arg_list_prime(node)
                return node
            if terminal in {")"}:
                node = Node("Arg-list-prime", parent=parent)
                self.epsilon(node)
                return node
            if terminal == "EOF":
                self.raise_eof_error()
                return None
            self.raise_illegal_error("Arg-list-prime", terminal, parent=parent)
