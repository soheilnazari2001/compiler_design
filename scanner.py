from tokens import Token, TokenType

keywords = {"break", "else", "if", "int", "for", "return", "endif", "void"}
single_symbols = set("+-<:;,()[]{}")
whitespaces = set(" \n\r\t\v\f")
digits = set("0123456789")
letters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
legal_chars = single_symbols | {"/", "*"} | {"="} | whitespaces | digits | letters
illegal_chars = [chr(i) for i in range(256) if chr(i) not in legal_chars]
slash_symbol = ["/"]
star_symbol = ["*"]
equal_symbol = ["="]
hidden_tokens = [
    TokenType.COMMENT.name,
    TokenType.WHITESPACE.name,
    TokenType.START.name,
    TokenType.PANIC.name,
]


def line_number_str(line_number):
    return f"{str(line_number) + '.':<7}"


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

            self.line = self.file.readline()
            if self.line == "":
                return None
            self.line_number += 1
            self.char_index = 0


class State:
    states = {}

    def __init__(self, state_id, state_type, is_final, is_star, error=""):
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
        self.add_transition(
            [chr(i) for i in range(256) if chr(i) not in self.transitions], destination
        )
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

    State(
        40, TokenType.SYMBOL.name, is_final=True, is_star=False
    )  # Always-single symbols
    State(
        41, TokenType.SYMBOL.name, is_final=False, is_star=False
    )  # Equal symbol reached
    State(
        42, TokenType.SYMBOL.name, is_final=True, is_star=False
    )  # Double equal finished
    State(
        43, TokenType.SYMBOL.name, is_final=True, is_star=True
    )  # Reached other characters after single/double-symbol

    State(50, TokenType.COMMENT.name, is_final=False, is_star=False)  # '/' reached
    State(
        51, TokenType.COMMENT.name, is_final=False, is_star=False
    )  # '*' reached after '/' (comment)
    State(
        52, TokenType.COMMENT.name, is_final=False, is_star=False
    )  # '*' reached inside comment
    State(
        53, TokenType.COMMENT.name, is_final=True, is_star=False
    )  # '/' reached after '*' (comment finished)
    State(
        54, TokenType.COMMENT.name, is_final=False, is_star=False
    )  # '*' reached outside comment

    State(
        90, TokenType.PANIC.name, is_final=True, is_star=False, error="Invalid number"
    )
    State(
        92, TokenType.PANIC.name, is_final=True, is_star=True, error="Unclosed comment"
    )
    State(93, TokenType.PANIC.name, is_final=True, is_star=False, error="Invalid input")
    State(94, TokenType.PANIC.name, is_final=True, is_star=True, error="Invalid input")

    State(
        95,
        TokenType.PANIC.name,
        is_final=True,
        is_star=False,
        error="Unmatched comment",
    )

    State(100, TokenType.EOF.name, is_final=True, is_star=True)

    State.states[0].add_transition(digits, State.states[10]).add_transition(
        letters, State.states[20]
    ).add_transition(whitespaces, State.states[30]).add_transition(
        [None], State.states[100]
    ).add_transition(
        single_symbols, State.states[40]
    ).add_transition(
        equal_symbol, State.states[41]
    ).add_transition(
        slash_symbol, State.states[50]
    ).add_transition(
        star_symbol, State.states[54]
    ).add_transition(
        illegal_chars, State.states[93]
    ).otherwise(
        State.states[93]
    )

    State.states[10].add_transition(digits, State.states[10]).add_transition(
        letters, State.states[90]
    ).otherwise(State.states[11])

    State.states[20].add_transition(digits | letters, State.states[20]).add_transition(
        illegal_chars, State.states[93]
    ).otherwise(State.states[21])

    State.states[30].add_transition(whitespaces, State.states[30]).otherwise(
        State.states[31]
    )

    State.states[41].add_transition(equal_symbol, State.states[42]).add_transition(
        illegal_chars, State.states[93]
    ).otherwise(State.states[43])

    State.states[50].add_transition(legal_chars, State.states[94]).add_transition(
        star_symbol, State.states[51]
    ).otherwise(State.states[93])

    State.states[51].add_transition(star_symbol, State.states[51]).add_transition(
        [None], State.states[92]
    ).add_transition(star_symbol, State.states[52]).otherwise(State.states[51])

    State.states[52].add_transition(slash_symbol, State.states[53]).add_transition(
        [None], State.states[92]
    ).otherwise(State.states[51])

    State.states[54].add_transition(slash_symbol, State.states[95]).add_transition(
        illegal_chars, State.states[93]
    ).otherwise(State.states[43])


class Scanner:
    states = []

    def __init__(self, input_file):
        setup_states()
        start_state = State.states[0]
        self.reader = Reader(input_file)
        self.start_state: State = start_state
        self.current_state: State = start_state
        self.tokens = {}
        self.errors = {}

    def get_next_token(self) -> Token:
        while True:
            token = self._get_next_token()
            if token.type not in hidden_tokens:
                return token

    def _get_next_token(self) -> Token:
        self.current_state = self.start_state
        token_name = ""
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
            if token_name in keywords:
                return Token(TokenType.KEYWORD.name, token_name)
        elif self.current_state.state_type == TokenType.PANIC.name:

            token_name = token_name[:7] + "..." if len(token_name) > 7 else token_name
        return Token(self.current_state.state_type, token_name)

    def get_tokens(self):
        token = Token(type=TokenType.START.name)
        while token.type != TokenType.EOF.name:
            self.current_state = self.start_state
            line_number = self.reader.line_number
            token = self._get_next_token()

            if token.type == TokenType.PANIC.name:
                self.errors.setdefault(line_number, []).append(
                    Token(token.lexeme, self.current_state.error)
                )
            self.tokens.setdefault(line_number, []).append(token)

    def __str__(self):
        s = ""
        for line_number, line_tokens in self.tokens.items():
            line_tokens = ""
            for token in line_tokens:
                if token.type not in hidden_tokens:
                    line_tokens += str(token) + " "
            if line_tokens:
                s += str(line_number_str(line_number)) + line_tokens + "\n"
        return s

    def repr_lexical_errors(self):
        if not self.errors:
            return "There is no lexical error."
        return "\n".join(
            map(
                lambda line_number: line_number_str(line_number)
                + " ".join(map(str, self.errors[line_number])),
                self.errors,
            )
        )
