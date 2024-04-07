from collections import defaultdict
from enum import Enum, auto

keywords = {'break', 'else', 'if', 'int', 'repeat', 'return', 'until', 'void'}
single_symbols = {'+', '-', '<', ':', ';', ',', '(', ')', '[', ']', '{', '}'}
slash_symbol = {'/'}
star_symbol = {'*'}
equal_symbol = {'='}
whitespaces = {' ', '\n', '\r', '\t', '\v', '\f'}
digits = {chr(i) for i in range(48, 58)}
letters = {chr(i) for i in range(65, 91)}.union({chr(i) for i in range(97, 123)})


class StateType(Enum):
    NUM = auto()
    ID = auto()
    KEYWORD = auto()
    SYMBOL = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    START = auto()
    PANIC = auto()
    EOF = auto()


class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    SYMBOL = auto()
    EOF = auto()


hidden_tokens = {StateType.COMMENT, StateType.WHITESPACE, StateType.START, StateType.PANIC, StateType.EOF}


def line_number_str(line_number):
    return f"{str(line_number) + '.':<7} "


class SymbolTable:
    def __init__(self):
        self.symbols = set(keywords)

    def add_symbol(self, symbol):
        self.symbols.add(symbol)

    def __contains__(self, symbol):
        return symbol in self.symbols

    def __str__(self):
        return "\n".join(f'{line_number_str(i + 1)}{symbol}' for i, symbol in enumerate(sorted(self.symbols)))


class Reader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.line_number = 0
        self.file = None
        self.line = ""
        self.char_index = 0

    def __enter__(self):
        try:
            self.file = open(self.file_path, 'r')
        except IOError as e:
            print(f"Failed to open file: {e}")
            # Raise the exception to prevent further execution
            raise
        self.line = self.readline()
        self.char_index = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def get_char(self):
        if self.char_index >= len(self.line):
            self.line = self.readline()
            self.char_index = 0
            if self.line == "":
                return None
        char = self.line[self.char_index]
        self.char_index += 1
        return char

    def readline(self):
        line = self.file.readline()
        if line:
            self.line_number += 1
        return line.rstrip('\n')


class Token:
    def __init__(self, token_type, token_value=''):
        self.token_type = token_type
        self.token_value = token_value

    def __str__(self):
        return f'({self.token_type}, {self.token_value})'

    def __repr__(self):
        return f'Token({self.token_type!r}, {self.token_value!r})'


class State:
    states = {}

    def __init__(self, state_id, state_type, is_final=False, is_star=False, error=''):
        self.default_transition = None
        self.id = state_id
        self.transitions = {}
        self.error = error
        self.state_type = state_type
        self.is_final = is_final
        self.is_star = is_star
        State.states[state_id] = self

    @classmethod
    def create_state(cls, state_id, state_type, is_final=False, is_star=False, error=''):
        return cls(state_id, state_type, is_final, is_star, error)

    def add_transition(self, characters, destination_state_id):
        for char in characters:
            self.transitions[char] = State.states[destination_state_id]
        return self

    def add_default_transition(self, destination_state_id):
        self.default_transition = State.states[destination_state_id]
        return self

    def next_state(self, char):
        return self.transitions.get(char, self.default_transition)


class Scanner:
    def __init__(self, reader: Reader, start_state: State):
        self.symbol_table = SymbolTable()
        self.reader = reader
        self.start_state = start_state
        self.current_state = start_state
        self.tokens = defaultdict(list)
        self.lexical_errors = defaultdict(list)

    def get_next_token(self) -> Token:
        token_name = ''
        char = self.reader.get_char()
        while char is not None and not self.current_state.is_final:
            self.current_state = self.current_state.next_state(next_char=char)
            if not self.current_state.is_star:
                token_name += char
            char = self.reader.get_char() if not self.current_state.is_star else char

        if self.current_state.is_final:
            if self.current_state.state_type == 'ID' and token_name in keywords:
                return Token('KEYWORD', token_name)
            else:
                return Token(self.current_state.state_type, token_name)
        return Token('EOF', '')

    def get_tokens(self):
        while True:
            self.current_state = self.start_state
            line_number = self.reader.line_number
            token = self.get_next_token()

            if token.token_type == StateType.EOF:
                break

            if token.token_type == StateType.PANIC:
                self.lexical_errors[line_number].append(token)
            else:
                self.tokens[line_number].append(token)

    def __str__(self):
        output = ''
        for line_number, tokens in self.tokens.items():
            line_tokens = ' '.join(str(token) for token in tokens if token.token_type not in hidden_tokens)
            if line_tokens:
                output += f'{line_number_str(line_number)}{line_tokens}\n'
        return output

    def repr_lexical_errors(self):
        if not self.lexical_errors:
            return 'There are no lexical errors.'
        return '\n'.join(f'{line_number_str(line)}' + ' '.join(str(token) for token in tokens)
                         for line, tokens in self.lexical_errors.items())


def initialize_states():
    # Creating states
    State.create_state(0, StateType.START, is_final=False, is_star=False)

    State.create_state(10, StateType.NUM, is_final=False, is_star=False)
    State.create_state(11, StateType.NUM, is_final=True, is_star=True)

    State.create_state(20, StateType.ID, is_final=False, is_star=False)
    State.create_state(21, StateType.ID, is_final=True, is_star=True)

    State.create_state(30, StateType.WHITESPACE, is_final=False, is_star=False)
    State.create_state(31, StateType.WHITESPACE, is_final=True, is_star=True)

    State.create_state(40, StateType.SYMBOL, is_final=True, is_star=False)  # Always-single symbols
    State.create_state(41, StateType.SYMBOL, is_final=False, is_star=False)  # Equal symbol reached
    State.create_state(42, StateType.SYMBOL, is_final=True, is_star=False)  # Double equal finished
    State.create_state(43, StateType.SYMBOL, is_final=True,
                       is_star=True)  # Reached other characters after single/double-symbol

    State.create_state(50, StateType.COMMENT, is_final=False, is_star=False)  # '/' reached
    State.create_state(51, StateType.COMMENT, is_final=False, is_star=False)  # '*' reached after '/' (comment)
    State.create_state(52, StateType.COMMENT, is_final=False, is_star=False)  # '*' reached inside comment
    State.create_state(53, StateType.COMMENT, is_final=True, is_star=False)  # '/' reached after '*' (comment finished)
    State.create_state(54, StateType.COMMENT, is_final=False, is_star=False)  # '*' reached outside comment

    State.create_state(90, StateType.PANIC, is_final=True, is_star=False, error='Invalid number')
    State.create_state(92, StateType.PANIC, is_final=True, is_star=True, error='Unclosed comment')
    State.create_state(93, StateType.PANIC, is_final=True, is_star=False, error='Invalid input')
    State.create_state(94, StateType.PANIC, is_final=True, is_star=True, error='Invalid input')

    State.create_state(95, StateType.PANIC, is_final=True, is_star=False, error='Unmatched comment')

    State.create_state(100, StateType.EOF, is_final=True, is_star=True)


def main(file_name):
    initialize_states()
    try:
        with Reader(file_name) as reader:
            scanner = Scanner(reader, State.states[0])
            scanner.get_tokens()
        with open('tokens.txt', 'w') as output_file:
            output_file.write(str(scanner))
        with open('symbol_table.txt', 'w') as output_file:
            output_file.write(str(scanner.symbol_table))
        with open('lexical_errors.txt', 'w') as output_file:
            output_file.write(scanner.repr_lexical_errors())
    except FileNotFoundError:
        print(f"File {file_name} not found.")


if __name__ == '__main__':
    main('input.txt')
