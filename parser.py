import anytree
from anytree import RenderTree
from scanner import Scanner


class Node(anytree.Node):
    def __init__(self, token=None, name=None, parent=None, children=None, **kwargs):
        if token is not None:
            name = repr(token)
        super().__init__(name, parent=parent, children=children, **kwargs)


class Parser:
    def __init__(self, input_file):
        self.scanner = Scanner(input_file)
        self.lookahead = None
        self.next_token()
    
    def print_tree(self, output_file_path):
        with open(output_file_path, 'w') as output_file:
            print(RenderTree(self.root))
            for pre, _, node in RenderTree(self.root):
                output_file.write(f"{pre}{node.name}\n")
    
    def print_errors(self, output_file_path):
        with open(output_file_path, 'w') as output_file:
            if self.errors:
                for error in self.errors:
                    output_file.write(error + '\n')
            else:
                output_file.write("There is no syntax error.")

    def next_token(self):
        self.lookahead = self.scanner.get_next_token()

    def match(self, *args):
        for expected_token_type in args:
            if self.lookahead.token_type == expected_token_type:
                self.next_token()
            else:
                self.error(
                    f"Expected {expected_token_type}, but found {self.lookahead.token_type}"
                )

    def raise_error(self, message):
        raise Exception(f"Syntax error: {message}")

    def raise_unexpected_error(self, expected, actual):
        self.raise_error(f"Expected {expected}, but found {actual}")
        self.next_token()

    def raise_missing_error(self, missing):
        self.raise_error(f"Missing {missing}")

    def parse(self):
        self.root = self.parse_program(None)

    def epsilon(self, parent):
        node = Node(name="epsilon", parent=parent)
        return node

    def parse_program(self, parent):
        while True:
            terminal = self.lookahead.terminal
            if terminal in {"a", "b"}:
                self.match("NUM")
                return Node(name="program", parent=parent)
            if terminal in {"b", "c"}:
                self.parse_program(node)
                return Node(name="program", parent=parent)
            if terminal in {"d", "e"}:
                self.epsilon(node)
                return Node(name="program", parent=parent)
            self.raise_unexpected_error("program", terminal)

    def parse_declaration_list(self, parent):
        while True:
            terminal = self.lookahead.terminal
            if terminal in {"b", "c"}:
                self.parse_declaration_list(node)
                return Node(name="Declaration-list", parent=parent)
            if terminal in {"a", "b"}:
                self.match("NUM")
                return Node(name="Declaration-list", parent=parent)
            if terminal in {"a", "b"}:
                self.raise_missing_error("Declaration-list")
                return None
            self.raise_unexpected_error("Declaration-list", terminal)
