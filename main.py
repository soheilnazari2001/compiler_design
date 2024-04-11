from enum import Enum, auto

keywords = ['break', 'else', 'if', 'int', 'repeat', 'return', 'until', 'void']
single_symbols = ['+', '-', '<', ':', ';', ',', '(', ')', '[', ']', '{', '}']
slash_symbol = ['/']
star_symbol = ['*']
equal_symbol = ['=']
whitespaces = [' ', '\n', '\r', '\t', '\v', '\f']
digits = [chr(i) for i in range(48, 58)]
letters = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
legal_chars = single_symbols + slash_symbol + star_symbol + equal_symbol + whitespaces + digits + letters + [None]
illegal_chars = [chr(i) for i in range(256) if chr(i) not in legal_chars]


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


hidden_tokens = [TokenType.COMMENT, TokenType.WHITESPACE, TokenType.START, TokenType.PANIC, TokenType.EOF]


def line_number_str(line_number):
    return f"{str(line_number) + '.':<7} "


class SymbolTable:
    def __init__(self):
        self.symbols = dict.fromkeys(keywords)

    def add_symbol(self, symbol):
        self.symbols[symbol] = None

    def __str__(self):
        s = ''
        for i, symbol in enumerate(self.symbols):
            s += f'{line_number_str(i + 1)}{symbol}\n'
        return s


class Reader:
    def __init__(self, file):
        self.line_number = 0
        self.file = file
        self.line = self.readline()
        self.char_index = 0

    def get_char(self):
        if self.char_index >= len(self.line):
            self.line = self.readline()
            self.char_index = 0
        if not self.line:
            return None
        c = self.line[self.char_index]
        self.char_index += 1
        return c

    def readline(self):
        line = self.file.readline()
        if line:
            self.line_number += 1
        return line


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

    def next_state(self, next_char):
        return self.transitions.get(next_char)  # returns None if char is not in transitions


class Scanner:
    def __init__(self, reader: Reader, start_state: State):
        self.symbol_table = SymbolTable()
        self.reader = reader
        self.start_state: State = start_state
        self.current_state: State = start_state
        self.tokens = {}  # line_number: list of tokens
        self.lexical_errors = {}  # line_number: list of errors

    def get_next_token(self) -> Token:
        token_name = ''
        while not self.current_state.is_final:
            c = self.reader.get_char()
            self.current_state = self.current_state.next_state(next_char=c)
            if self.current_state.is_star:
                self.reader.char_index -= 1
            else:
                token_name += c

        if self.current_state.state_type == TokenType.ID:
            self.symbol_table.add_symbol(token_name)
            if token_name in keywords:
                return Token(TokenType.KEYWORD, token_name)
        elif self.current_state.state_type == TokenType.PANIC:
            token_name = token_name[:7] + '...' if len(token_name) > 7 else token_name
        return Token(self.current_state.state_type, token_name)

    def get_tokens(self):
        token = Token(token_type=TokenType.START)
        while token.token_type != TokenType.EOF:
            self.current_state = self.start_state
            line_number = self.reader.line_number
            token = self.get_next_token()

            if token.token_type == TokenType.PANIC:
                self.lexical_errors.setdefault(line_number, []) \
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
                s += line_number_str(line_number) + line_tokens + '\n'
        return s

    def repr_lexical_errors(self):
        if not self.lexical_errors:
            return 'There is no lexical error.'
        s = ''
        for line_number in self.lexical_errors:
            s += line_number_str(line_number)
            for token in self.lexical_errors[line_number]:
                s += str(token) + ' '
            s += '\n'
        return s


def initialize_states():
    State(0, TokenType.START, is_final=False, is_star=False)

    State(10, TokenType.NUM, is_final=False, is_star=False)
    State(11, TokenType.NUM, is_final=True, is_star=True)

    State(20, TokenType.ID, is_final=False, is_star=False)
    State(21, TokenType.ID, is_final=True, is_star=True)

    State(30, TokenType.WHITESPACE, is_final=False, is_star=False)
    State(31, TokenType.WHITESPACE, is_final=True, is_star=True)

    State(40, TokenType.SYMBOL, is_final=True, is_star=False)  # Always-single symbols
    State(41, TokenType.SYMBOL, is_final=False, is_star=False)  # Equal symbol reached
    State(42, TokenType.SYMBOL, is_final=True, is_star=False)  # Double equal finished
    State(43, TokenType.SYMBOL, is_final=True, is_star=True)  # Reached other characters after single/double-symbol

    State(50, TokenType.COMMENT, is_final=False, is_star=False)  # '/' reached
    State(51, TokenType.COMMENT, is_final=False, is_star=False)  # '*' reached after '/' (comment)
    State(52, TokenType.COMMENT, is_final=False, is_star=False)  # '*' reached inside comment
    State(53, TokenType.COMMENT, is_final=True, is_star=False)  # '/' reached after '*' (comment finished)
    State(54, TokenType.COMMENT, is_final=False, is_star=False)  # '*' reached outside comment

    State(90, TokenType.PANIC, is_final=True, is_star=False, error='Invalid number')
    State(92, TokenType.PANIC, is_final=True, is_star=True, error='Unclosed comment')
    State(93, TokenType.PANIC, is_final=True, is_star=False, error='Invalid input')
    State(94, TokenType.PANIC, is_final=True, is_star=True, error='Invalid input')

    State(95, TokenType.PANIC, is_final=True, is_star=False, error='Unmatched comment')

    State(100, TokenType.EOF, is_final=True, is_star=True)

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
        .add_transition(digits + letters, State.states[20]) \
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


if __name__ == '__main__':
    file_name = "input.txt"
    initialize_states()

    with open(file_name, 'r') as input_file:
        scanner = Scanner(reader=Reader(input_file), start_state=State.states[0])
        scanner.get_tokens()
    with open('tokens.txt', 'w') as output_file:
        output_file.write(str(scanner))
    with open('symbol_table.txt', 'w') as output_file:
        output_file.write(str(scanner.symbol_table))
    with open('lexical_errors.txt', 'w') as output_file:
        output_file.write(scanner.repr_lexical_errors())
