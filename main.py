# Soheil Nazari 99102412
# Soroush Sherafat 99105504
from enum import Enum, auto

from anytree import RenderTree, Node

non_terminals = ['S', 'Program', 'Declaration-list', 'Declaration', 'Declaration-initial', 'Declaration-prime',
                 'Var-declaration-prime', 'Fun-declaration-prime', 'Type-specifier', 'Params', 'Param-list', 'Param',
                 'Param-prime', 'Compound-stmt', 'Statement-list', 'Statement', 'Expression-stmt', 'Selection-stmt',
                 'Else-stmt', 'Iteration-stmt', 'Return-stmt', 'Return-stmt-prime', 'Expression', 'B', 'H',
                 'Simple-expression-zegond', 'Simple-expression-prime', 'C', 'Relop', 'Additive-expression',
                 'Additive-expression-prime', 'Additive-expression-zegond', 'D', 'Addop', 'Term', 'Term-prime',
                 'Term-zegond', 'G', 'Signed-factor', 'Signed-factor-prime', 'Signed-factor-zegond', 'Factor',
                 'Var-call-prime', 'Var-prime', 'Factor-prime', 'Factor-zegond', 'Args', 'Arg-list', 'Arg-list-prime']

rules = {'Program': [['Declaration-list']],
         'Declaration-list': [['Declaration', 'Declaration-list'], ['EPSILON']],
         'Declaration': [['Declaration-initial', 'Declaration-prime']],
         'Declaration-initial': [['Type-specifier', 'ID']],
         'Declaration-prime': [['Fun-declaration-prime'], ['Var-declaration-prime']],
         'Var-declaration-prime': [[';'], ['[', 'NUM', '];']],
         'Fun-declaration-prime': [['(', 'Params', ')', 'Compound-stmt']],
         'Type-specifier': [['int'], ['void']],
         'Params': [['int', 'ID', 'Param-prime', 'Param-list'], ['void']],
         'Param-list': [[',', 'Param', 'Param-list'], ['EPSILON']],
         'Param': [['Declaration-initial', 'Param-prime']],
         'Param-prime': [['[', ']'], ['EPSILON']],
         'Compound-stmt': [['{', 'Declaration-list', 'Statement-list', '}']],
         'Statement-list': [['Statement', 'Statement-list'], ['EPSILON']],
         'Statement': [['Expression-stmt'], ['Compound-stmt'], ['Selection-stmt'], ['Iteration-stmt'], ['Return-stmt']],
         'Expression-stmt': [['Expression', ';'], ['break', ';'], [';']],
         'Selection-stmt': [['if', '(', 'Expression', ')', 'Statement', 'Else-stmt']],
         'Else-stmt': [['endif'], ['else', 'Statement', 'endif']],
         'Iteration-stmt': [['for', '(Expression;', 'Expression;', 'Expression)', 'Statement']],
         'Return-stmt': [['return', 'Return-stmt-prime']],
         'Return-stmt-prime': [[';'], ['Expression', ';']],
         'Expression': [['Simple-expression-zegond'], ['ID', 'B']],
         'B': [['=', 'Expression'], ['Simple-expression-prime']],
         'H': [['=', 'Expression'], ['G', 'D', 'C']],
         'Simple-expression-zegond': [['Additive-expression-zegond', 'C']],
         'Simple-expression-prime': [['Additive-expression-prime', 'C']],
         'C': [['Relop', 'Additive-expression'], ['EPSILON']],
         'Relop': [['<'], ['==']],
         'Additive-expression': [['Term', 'D']],
         'Additive-expression-prime': [['Term-prime', 'D']],
         'Additive-expression-zegond': [['Term-zegond', 'D']],
         'D': [['Addop', 'Term'], ['D'], ['EPSILON']],
         'Addop': [['+'], ['-']],
         'Term': [['Signed-factor', 'G']],
         'Term-prime': [['Signed-factor-prime', 'G']],
         'Term-zegond': [['Signed-factor-zegond', 'G']],
         'G': [['*', 'Signed-factor', 'G'], ['EPSILON']],
         'Signed-factor': [['+', 'Factor'], ['-', 'Factor'], ['Factor']],
         'Signed-factor-prime': [['+', 'Factor-prime'], ['-', 'Factor-prime'], ['Factor-prime']],
         'Signed-factor-zegond': [['+', 'Factor'], ['-', 'Factor'], ['Factor-zegond']],
         'Factor': [['(', 'Expression', ')'], ['ID', 'Var-call-prime'], ['NUM']],
         'Var-call-prime': [['(', 'Args', ')'], ['Var-prime']],
         'Var-prime': [['[', 'Expression', ']'], ['EPSILON']],
         'Factor-prime': [['(', 'Args', ')'], ['EPSILON']],
         'Factor-zegond': [['(', 'Expression', ')'], ['NUM']],
         'Args': [['Arg-list'], ['EPSILON']],
         'Arg-list': [['Expression', 'Arg-list-prime']],
         'Arg-list-prime': [[',', 'Expression', 'Arg-list-prime'], ['EPSILON']]
         }

first_and_follow = {
    "Program": {"First": ["int", "void", "$"], "Follow": ["$"]},
    "Declaration-list": {"First": ["int", "void", "$"],
                         "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "for", "return", "+", "-", "$"]},
    "Declaration": {"First": ["int", "void"],
                    "Follow": ["ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return", "+", "-",
                               "$"]},
    "Declaration-initial": {"First": ["int", "void"], "Follow": [";", "[", "(", ")", ","]},
    "Declaration-prime": {"First": [";", "[", "("],
                          "Follow": ["ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return",
                                     "+", "-", "$"]},
    "Var-declaration-prime": {"First": [";", "["],
                              "Follow": ["ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return",
                                         "+", "-", "$"]},
    "Fun-declaration-prime": {"First": ["("],
                              "Follow": ["ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "for", "return",
                                         "+", "-", "$"]},
    "Type-specifier": {"First": ["int", "void"], "Follow": ["ID"]},
    "Params": {"First": ["int", "void"], "Follow": [")"]},
    "Param-list": {"First": [",", "$"], "Follow": [")"]},
    "Param": {"First": ["int", "void"], "Follow": [")", ","]},
    "Param-prime": {"First": ["[", "$"], "Follow": [")", ","]},
    "Compound-stmt": {"First": ["{"],
                      "Follow": ["ID", ";", "NUM", "(", "int", "void", "{", "}", "break", "if", "endif", "else", "for",
                                 "return", "+", "-", "$"]},
    "Statement-list": {"First": ["ID", ";", "NUM", "(", "{", "break", "if", "for", "return", "+", "-", "$"],
                       "Follow": ["}"]},
    "Statement": {"First": ["ID", ";", "NUM", "(", "{", "break", "if", "for", "return", "+", "-"],
                  "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+",
                             "-"]},
    "Expression-stmt": {"First": ["ID", ";", "NUM", "(", "break", "+", "-"],
                        "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return",
                                   "+", "-"]},
    "Selection-stmt": {"First": ["if"],
                       "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+",
                                  "-"]},
    "Else-stmt": {"First": ["endif", "else"],
                  "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+",
                             "-"]},
    "Iteration-stmt": {"First": ["for"],
                       "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+",
                                  "-"]},
    "Return-stmt": {"First": ["return"],
                    "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return", "+",
                               "-"]},
    "Return-stmt-prime": {"First": [";", "NUM", "(", "+", "-"],
                          "Follow": ["ID", ";", "NUM", "(", "{", "}", "break", "if", "endif", "else", "for", "return",
                                     "+", "-"]},
    "Expression": {"First": ["ID", "NUM", "(", "+", "-"], "Follow": [";", "]", ")", ","]},
    "B": {"First": ["[", "(", "=", "<", "==", "+", "-", "*", "$"], "Follow": [";", "]", ")", ","]},
    "H": {"First": ["=", "<", "==", "+", "-", "*", "$"], "Follow": [";", "]", ")", ","]},
    "Simple-expression-zegond": {"First": ["NUM", "(", "+", "-"], "Follow": [";", "]", ")", ","]},
    "Simple-expression-prime": {"First": [";", "]", "(", ")", ",", "<", "==", "+", "-", "*"],
                                "Follow": [";", "]", ")", ","]},
    "C": {"First": ["<", "==", "$"], "Follow": [";", "]", ")", ","]},
    "Relop": {"First": ["<", "=="], "Follow": ["ID", "NUM", "(", "+", "-"]},
    "Additive-expression": {"First": ["ID", "NUM", "(", "+", "-"], "Follow": [";", "]", ")", ","]},
    "Additive-expression-prime": {"First": [";", "]", "(", ")", ",", "<", "==", "+", "-", "*"],
                                  "Follow": [";", "]", ")", ",", "<", "=="]},
    "Additive-expression-zegond": {"First": ["NUM", "(", "+", "-"], "Follow": [";", "]", ")", ",", "<", "=="]},
    "D": {"First": ["+", "-", "$"], "Follow": [";", "]", ")", ",", "<", "=="]},
    "Addop": {"First": ["+", "-"], "Follow": ["ID", "NUM", "(", "+", "-"]},
    "Term": {"First": ["ID", "NUM", "(", "+", "-"], "Follow": [";", "]", ")", ",", "<", "==", "+", "-"]},
    "Term-prime": {"First": [";", "]", "(", ")", ",", "<", "==", "+", "-"],
                   "Follow": [";", "]", ")", ",", "<", "==", "+", "-"]},
    "Term-zegond": {"First": ["NUM", "(", "+", "-"], "Follow": [";", "]", ")", ",", "<", "==", "+", "-"]},
    "G": {"First": [";", "]", ")", ",", "<", "==", "+", "-"], "Follow": [";", "]", ")", ",", "<", "==", "+", "-"]},
    "Signed-factor": {"First": ["ID", "NUM", "(", "+", "-"], "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Signed-factor-prime": {"First": [";", "]", "(", ")", ",", "<", "==", "+", "-", "*"],
                            "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Signed-factor-zegond": {"First": ["NUM", "(", "+", "-"], "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Factor": {"First": ["ID", "NUM", "("], "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Var-call-prime": {"First": ["[", "(", "$"], "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Var-prime": {"First": [";", "[", "]", ")", ",", "<", "==", "+", "-", "*"],
                  "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Factor-prime": {"First": [";", "]", "(", ")", ",", "<", "==", "+", "-", "*"],
                     "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Factor-zegond": {"First": ["NUM", "("], "Follow": [";", "]", ")", ",", "<", "==", "+", "-", "*"]},
    "Args": {"First": ["ID", "NUM", "(", "+", "-", "$"], "Follow": [")"]},
    "Arg-list": {"First": ["ID", "NUM", "(", "+", "-"], "Follow": [")"]},
    "Arg-list-prime": {"First": [",", "$"], "Follow": [")"]}
}


class TokenType(Enum):
    NUM = auto()
    ID = auto()
    KEYWORD = auto()
    SYMBOL = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    START = auto()
    PANIC = auto()
    EOF = auto()


keywords = {'break', 'else', 'if', 'int', 'for', 'return', 'endif', 'void'}
single_symbols = set('+-<:;,()[]{}')
whitespaces = set(' \n\r\t\v\f')
digits = set('0123456789')
letters = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
legal_chars = single_symbols | {'/', '*'} | {'='} | whitespaces | digits | letters
illegal_chars = [chr(i) for i in range(256) if chr(i) not in legal_chars]
slash_symbol = ['/']
star_symbol = ['*']
equal_symbol = ['=']
hidden_tokens = [TokenType.COMMENT.name, TokenType.WHITESPACE.name, TokenType.START.name, TokenType.PANIC.name,
                 TokenType.EOF.name]


def line_number_str(line_number):
    return f"{str(line_number) + '.':<7}"


class SymbolTable:
    def __init__(self):
        self.symbols = {key: None for key in keywords}

    def add_symbol(self, symbol):
        if symbol not in self.symbols:
            self.symbols[symbol] = None

    def __str__(self):
        s = ''
        for i, symbol in enumerate(self.symbols):
            s += f'{line_number_str(i + 1)}{symbol}\n'
        return s


class Reader:
    def __init__(self, file):
        self.file = file
        self.line_number = 1
        self.char_index = 0
        self.line = self.file.readline()

    def get_char(self):
        while True:
            if self.char_index < len(self.line):
                char = self.line[self.char_index]
                self.char_index += 1
                return char
            else:
                self.line = self.file.readline()
                if self.line == '':
                    return None
                self.line_number += 1
                self.char_index = 0


class Token:
    def __init__(self, token_type, token_value=''):
        self.token_type = token_type
        self.token_value = token_value

    def __str__(self):
        return f'({self.token_type}, {self.token_value})'


class State:
    states = {}

    def __init__(self, state_id, state_type, is_final, is_star, error=''):
        self.id = state_id
        self.transitions = {}
        self.error = error
        self.state_type = state_type
        self.is_final = is_final
        self.is_star = is_star
        self.states[state_id] = self

    def add_transition(self, characters, destination):
        self.transitions.update(dict.fromkeys(characters, destination))
        return self

    def otherwise(self, destination):
        self.add_transition([chr(i) for i in range(256) if chr(i) not in self.transitions], destination)
        if None not in self.transitions:
            self.transitions[None] = destination

    def get_next_state(self, next_char):
        return self.transitions.get(next_char)


def setup_states():
    State(0, TokenType.START.name, is_final=False, is_star=False)

    State(10, TokenType.NUM.name, is_final=False, is_star=False)
    State(11, TokenType.NUM.name, is_final=True, is_star=True)

    State(20, TokenType.ID.name, is_final=False, is_star=False)
    State(21, TokenType.ID.name, is_final=True, is_star=True)

    State(30, TokenType.WHITESPACE.name, is_final=False, is_star=False)
    State(31, TokenType.WHITESPACE.name, is_final=True, is_star=True)

    State(40, TokenType.SYMBOL.name, is_final=True, is_star=False)  # Always-single symbols
    State(41, TokenType.SYMBOL.name, is_final=False, is_star=False)  # Equal symbol reached
    State(42, TokenType.SYMBOL.name, is_final=True, is_star=False)  # Double equal finished
    State(43, TokenType.SYMBOL.name, is_final=True, is_star=True)  # Reached other characters after single/double-symbol

    State(50, TokenType.COMMENT.name, is_final=False, is_star=False)  # '/' reached
    State(51, TokenType.COMMENT.name, is_final=False, is_star=False)  # '*' reached after '/' (comment)
    State(52, TokenType.COMMENT.name, is_final=False, is_star=False)  # '*' reached inside comment
    State(53, TokenType.COMMENT.name, is_final=True, is_star=False)  # '/' reached after '*' (comment finished)
    State(54, TokenType.COMMENT.name, is_final=False, is_star=False)  # '*' reached outside comment

    State(90, TokenType.PANIC.name, is_final=True, is_star=False, error='Invalid number')
    State(92, TokenType.PANIC.name, is_final=True, is_star=True, error='Unclosed comment')
    State(93, TokenType.PANIC.name, is_final=True, is_star=False, error='Invalid input')
    State(94, TokenType.PANIC.name, is_final=True, is_star=True, error='Invalid input')

    State(95, TokenType.PANIC.name, is_final=True, is_star=False, error='Unmatched comment')

    State(100, TokenType.EOF.name, is_final=True, is_star=True)

    State.states[0] \
        .add_transition(digits, State.states[10]) \
        .add_transition(letters, State.states[20]) \
        .add_transition(whitespaces, State.states[30]) \
        .add_transition([None], State.states[100]) \
        .add_transition(single_symbols, State.states[40]) \
        .add_transition(equal_symbol, State.states[41]) \
        .add_transition(slash_symbol, State.states[50]) \
        .add_transition(star_symbol, State.states[54]) \
        .add_transition(illegal_chars, State.states[93]) \
        .otherwise(State.states[93])

    State.states[10] \
        .add_transition(digits, State.states[10]) \
        .add_transition(letters, State.states[90]) \
        .otherwise(State.states[11])

    State.states[20] \
        .add_transition(digits | letters, State.states[20]) \
        .add_transition(illegal_chars, State.states[93]) \
        .otherwise(State.states[21])

    State.states[30] \
        .add_transition(whitespaces, State.states[30]) \
        .otherwise(State.states[31])

    State.states[41] \
        .add_transition(equal_symbol, State.states[42]) \
        .add_transition(illegal_chars, State.states[93]) \
        .otherwise(State.states[43])

    State.states[50] \
        .add_transition(legal_chars, State.states[94]) \
        .add_transition(star_symbol, State.states[51]) \
        .otherwise(State.states[93])

    State.states[51] \
        .add_transition(star_symbol, State.states[51]) \
        .add_transition([None], State.states[92]) \
        .add_transition(star_symbol, State.states[52]) \
        .otherwise(State.states[51])

    State.states[52] \
        .add_transition(slash_symbol, State.states[53]) \
        .add_transition([None], State.states[92]) \
        .otherwise(State.states[51])

    State.states[54] \
        .add_transition(slash_symbol, State.states[95]) \
        .add_transition(illegal_chars, State.states[93]) \
        .otherwise(State.states[43])


class Scanner:
    states = []

    def __init__(self, reader: Reader, start_state: State):
        self.reader = reader
        self.start_state: State = start_state
        self.current_state: State = start_state
        self.tokens = {}
        self.errors = {}
        self.symbol_table = SymbolTable()

    def get_next_token(self) -> Token:
        token_name = ''
        while not self.current_state.is_final:
            c = self.reader.get_char()
            self.current_state = self.current_state.get_next_state(next_char=c)
            if self.current_state.is_star:
                self.reader.char_index -= 1
            elif c in illegal_chars and token_name in {"/", "*", "="}:
                self.reader.char_index -= 1
                return Token(TokenType.SYMBOL.name, token_name)
            else:
                token_name += c

        if self.current_state.state_type == TokenType.ID.name:
            self.symbol_table.add_symbol(token_name)
            if token_name in keywords:
                return Token(TokenType.KEYWORD.name, token_name)
        elif self.current_state.state_type == TokenType.PANIC.name:

            token_name = token_name[:7] + '...' if len(token_name) > 7 else token_name
        return Token(self.current_state.state_type, token_name)

    def get_tokens(self):
        token = Token(token_type=TokenType.START.name)
        while token.token_type != TokenType.EOF.name:
            self.current_state = self.start_state
            line_number = self.reader.line_number
            token = self.get_next_token()
            if token.token_type == TokenType.PANIC.name:
                self.errors.setdefault(line_number, []) \
                    .append(Token(token.token_value, self.current_state.error))
            self.tokens.setdefault(line_number, []).append(token)

    def __str__(self):
        s = ''
        for line_number in self.tokens:
            line_tokens = ''
            for token in self.tokens[line_number]:
                if token.token_type not in hidden_tokens:
                    line_tokens += str(token) + ' '
            if line_tokens:
                s += str(line_number_str(line_number)) + line_tokens + '\n'
        return s

    def repr_lexical_errors(self):
        if not self.errors:
            return 'There is no lexical error.'
        return '\n'.join(map(lambda line_number:
                             line_number_str(line_number) + ' '.join(map(str, self.errors[line_number])), self.errors))


class Parser:
    def __init__(self, scanner):
        self.scanner = scanner
        self.lookahead = None
        self.next_token()
        self.errors = []
        self.root = Node("Program")
        self.follow = first_and_follow

    def next_token(self):
        self.lookahead = self.scanner.get_next_token()

    def match(self, expected_token_type, parent):
        if self.lookahead.token_type == expected_token_type:
            child = Node(expected_token_type, parent=parent, name=expected_token_type)
            self.next_token()
            return child
        else:
            self.error(f"Expected {expected_token_type}, but found {self.lookahead.token_type}", parent)
            return Node(f"error: expected {expected_token_type}", parent=parent,
                        name=f"error: expected {expected_token_type}")

    def error(self, message, parent):
        self.errors.append(f"Syntax error: {message} at line {self.lookahead.line_number}")
        self.panic_mode_recovery(parent)

    def panic_mode_recovery(self, parent):
        non_terminal = parent.name
        follow_set = self.follow[non_terminal]['Follow']
        while self.lookahead.token_type not in follow_set and self.lookahead.token_type != 'EOF':
            self.next_token()

    def parse(self):
        self.program(self.root)
        self.output_results()

    def program(self, parent):
        self.declaration_list(parent)

    def declaration_list(self, parent):
        child = Node("Declaration-list", parent=parent)
        if self.lookahead.token_type in self.follow['Declaration-list']['First']:
            self.declaration(child)
            self.declaration_list(child)
        elif 'EPSILON' in rules['Declaration-list'][1]:
            return

    def declaration(self, parent):
        child = Node("Declaration", parent=parent)
        self.declaration_initial(child)
        self.declaration_prime(child)

    def declaration_initial(self, parent):
        child = Node("Declaration-initial", parent=parent)
        self.type_specifier(child)
        self.match('ID', child)

    def declaration_prime(self, parent):
        child = Node("Declaration-prime", parent=parent)
        if self.lookahead.token_type == '(':
            self.fun_declaration_prime(child)
        else:
            self.var_declaration_prime(child)

    def var_declaration_prime(self, parent):
        child = Node("Var-declaration-prime", parent=parent)
        if self.lookahead.token_type == ';':
            self.match(';', child)
        else:
            self.match('[', child)
            self.match('NUM', child)
            self.match(']', child)

    def fun_declaration_prime(self, parent):
        child = Node("Fun-declaration-prime", parent=parent)
        self.match('(', child)
        self.params(child)
        self.match(')', child)
        self.compound_stmt(child)

    def type_specifier(self, parent):
        child = Node("Type-specifier", parent=parent)
        if self.lookahead.token_type in ['int', 'void']:
            self.match(self.lookahead.token_type, child)

    def params(self, parent):
        child = Node("Params", parent=parent)
        if self.lookahead.token_type == 'int':
            self.match('int', child)
            self.match('ID', child)
            self.param_prime(child)
            self.param_list(child)
        elif self.lookahead.token_type == 'void':
            self.match('void', child)

    def param_list(self, parent):
        child = Node("Param-list", parent=parent)
        if self.lookahead.token_type == ',':
            self.match(',', child)
            self.param(child)
            self.param_list(child)
        elif 'EPSILON' in rules['Param-list'][1]:
            return

    def param(self, parent):
        child = Node("Param", parent=parent)
        self.declaration_initial(child)
        self.param_prime(child)

    def param_prime(self, parent):
        child = Node("Param-prime", parent=parent)
        if self.lookahead.token_type == '[':
            self.match('[', child)
            self.match(']', child)

    def compound_stmt(self, parent):
        child = Node("Compound-stmt", parent=parent)
        self.match('{', child)
        self.declaration_list(child)
        self.statement_list(child)
        self.match('}', child)

    def statement_list(self, parent):
        child = Node("Statement-list", parent=parent)
        if self.lookahead.token_type in first_and_follow['Statement-list']['First']:
            self.statement(child)
            self.statement_list(child)
        elif 'EPSILON' in rules['Statement-list'][1]:
            return

    def d(self, parent):
        child = Node("D", parent=parent)
        if self.lookahead.token_type in first_and_follow['Addop']['First']:
            self.match(self.lookahead.token_type, child)  # '+' or '-'
            self.term(child)
        elif 'EPSILON' in rules['D'][1]:
            return

    def term(self, parent):
        child = Node("Term", parent=parent)
        self.signed_factor(child)
        self.g(child)

    def signed_factor(self, parent):
        child = Node("Signed-factor", parent=parent)
        if self.lookahead.token_type in ['+', '-']:
            self.match(self.lookahead.token_type, child)
        self.factor(child)

    def factor(self, parent):
        child = Node("Factor", parent=parent)
        if self.lookahead.token_type == '(':
            self.match('(', child)
            self.expression(child)
            self.match(')', child)
        elif self.lookahead.token_type == 'ID':
            self.match('ID', child)
            self.var_call_prime(child)
        elif self.lookahead.token_type == 'NUM':
            self.match('NUM', child)

    def var_call_prime(self, parent):
        child = Node("Var-call-prime", parent=parent)
        if self.lookahead.token_type == '(':
            self.match('(', child)
            self.args(child)
            self.match(')', child)
        elif self.lookahead.token_type == '[':
            self.match('[', child)
            self.expression(child)
            self.match(']', child)

    def g(self, parent):
        child = Node("G", parent=parent)
        if self.lookahead.token_type == '*':
            self.match('*', child)
            self.signed_factor(child)
            self.g(child)
        elif 'EPSILON' in rules['G'][1]:
            return

    def term_zegond(self, parent):
        child = Node("Term-zegond", parent=parent)
        self.signed_factor_zegond(child)
        self.g(child)

    def signed_factor_zegond(self, parent):
        child = Node("Signed-factor-zegond", parent=parent)
        if self.lookahead.token_type in ['+', '-']:
            self.match(self.lookahead.token_type, child)
        self.factor_zegond(child)

    def factor_zegond(self, parent):
        child = Node("Factor-zegond", parent=parent)
        if self.lookahead.token_type == '(':
            self.match('(', child)
            self.expression(child)
            self.match(')', child)
        elif self.lookahead.token_type == 'NUM':
            self.match('NUM', child)

    def statement(self, parent):
        child = Node("Statement", parent=parent)
        if self.lookahead.token_type in first_and_follow['Statement']['First']:
            if self.lookahead.token_type == 'if':
                self.selection_stmt(child)
            elif self.lookahead.token_type == 'for':
                self.iteration_stmt(child)
            elif self.lookahead.token_type == 'return':
                self.return_stmt(child)
            elif self.lookahead.token_type in first_and_follow['Expression-stmt']['First']:
                self.expression_stmt(child)
            elif self.lookahead.token_type == '{':
                self.compound_stmt(child)
        else:
            self.error("Invalid statement", child)

    def expression_stmt(self, parent):
        child = Node("Expression-stmt", parent=parent)
        if self.lookahead.token_type in ['+', 'ID', 'NUM']:  # Assuming '+' starts an expression
            self.expression(child)
            self.match(';', child)
        elif self.lookahead.token_type == 'break':
            self.match('break', child)
            self.match(';', child)
        else:
            self.match(';', child)

    def selection_stmt(self, parent):
        child = Node("Selection-stmt", parent=parent)
        self.match('if', child)
        self.match('(', child)
        self.expression(child)
        self.match(')', child)
        self.statement(child)
        self.else_stmt(child)

    def else_stmt(self, parent):
        child = Node("Else-stmt", parent=parent)
        if self.lookahead.token_type == 'else':
            self.match('else', child)
            self.statement(child)
            self.match('endif', child)
        else:
            self.match('endif', child)

    def iteration_stmt(self, parent):
        child = Node("Iteration-stmt", parent=parent)
        self.match('for', child)
        self.match('(', child)
        self.expression(child)
        self.match(';', child)
        self.expression(child)
        self.match(';', child)
        self.expression(child)
        self.match(')', child)
        self.statement(child)

    def return_stmt(self, parent):
        child = Node("Return-stmt", parent=parent)
        self.match('return', child)
        self.return_stmt_prime(child)

    def return_stmt_prime(self, parent):
        child = Node("Return-stmt-prime", parent=parent)
        if self.lookahead.token_type == ';':
            self.match(';', child)
        else:
            self.expression(child)
            self.match(';', child)

    def expression(self, parent):
        child = Node("Expression", parent=parent)
        if self.lookahead.token_type == 'ID' and self.scanner.lookahead(1) == '=':
            self.match('ID', child)
            self.match('=', child)
            self.expression(child)
        else:
            self.simple_expression_zegond(child)

    def simple_expression_zegond(self, parent):
        child = Node("Simple-expression-zegond", parent=parent)
        self.additive_expression_zegond(child)
        if self.lookahead.token_type in first_and_follow['C']['First']:
            self.c(child)

    def additive_expression_zegond(self, parent):
        child = Node("Additive-expression-zegond", parent=parent)
        self.term_zegond(child)
        self.d(child)

    def c(self, parent):
        child = Node("C", parent=parent)
        if self.lookahead.token_type in first_and_follow['Relop']['First']:
            self.relop(child)
            self.additive_expression(child)

    def args(self, parent):
        child = Node("Args", parent=parent)
        if self.lookahead.token_type in first_and_follow['Args']['First']:
            self.arg_list(child)
        elif 'EPSILON' in rules['Args'][1]:
            return

    def arg_list(self, parent):
        child = Node("Arg-list", parent=parent)
        self.expression(child)
        self.arg_list_prime(child)

    def arg_list_prime(self, parent):
        child = Node("Arg-list-prime", parent=parent)
        if self.lookahead.token_type == ',':
            self.match(',', child)
            self.expression(child)
            self.arg_list_prime(child)
        elif 'EPSILON' in rules['Arg-list-prime'][1]:
            return

    def relop(self, parent):
        child = Node("Relop", parent=parent)
        if self.lookahead.token_type in ['<', '==']:
            self.match(self.lookahead.token_type, child)

    def additive_expression(self, parent):
        child = Node("Additive-expression", parent=parent)
        self.term(child)
        self.d(child)

    def output_results(self):
        with open('parse_tree.txt', 'w') as tree_file:
            print(RenderTree(self.root))
            for pre, _, node in RenderTree(self.root):
                tree_file.write(f"{pre}{node.name}\n")

        with open('syntax_errors.txt', 'w') as error_file:
            if self.errors:
                for error in self.errors:
                    error_file.write(error + '\n')
            else:
                error_file.write("There is no syntax error.")


def main(file_name):
    with open(file_name, 'r') as input_file:
        setup_states()
        scanner = Scanner(reader=Reader(input_file), start_state=State.states[0])
        scanner.get_tokens()
        parser = Parser(scanner)
        parser.parse()
        with open('tokens.txt', 'w') as tokens_file:
            tokens_file.write(str(scanner))
        with open('symbol_table.txt', 'w') as symbols_file:
            symbols_file.write(str(scanner.symbol_table))


if __name__ == '__main__':
    main('input.txt')
