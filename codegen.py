from collections import OrderedDict

from parser import Parser


def indirect_address(address):
    if address.startswith("#"):
        return address[1:]

    return f"@{address}"


class Instruction:
    ARGS_COUNT = 3

    def __init__(self, opcode, *args):
        self.opcode = opcode
        self.args = args + [""] * (self.ARGS_COUNT - len(args))

    def __repr__(self):
        return f"{tuple(self.opcode, *self.args)}"

    @classmethod
    def add(cls, in1, in2, out):
        return cls("ADD", in1, in2, out)

    @classmethod
    def mult(cls, in1, in2, out):
        return cls("MULT", in1, in2, out)

    @classmethod
    def sub(cls, in1, in2, out):
        return cls("SUB", in1, in2, out)

    @classmethod
    def eq(cls, in1, in2, out):
        return cls("EQ", in1, in2, out)

    @classmethod
    def lt(cls, in1, in2, out):
        return cls("LT", in1, in2, out)

    @classmethod
    def assign(cls, left, right):
        return cls("ASSIGN", left, right)

    @classmethod
    def jpf(cls, condition, address):
        address = indirect_address(address)
        return cls("JPF", condition, address)

    @classmethod
    def jp(cls, address):
        address = indirect_address(address)
        return cls("JP", address)

    @classmethod
    def print(cls, value):
        return cls("PRINT", value)


class Symbol:
    def __init__(self, address, lexeme):
        self.address = address
        self.lexeme = lexeme


class ScopeStack:
    def __init__(self, code_generator):
        self.scopes: list[dict[str, Symbol]] = [{}]
        self.code_generator = code_generator

    def add_symbol(self, lexeme, symbol):
        self.scopes[-1][lexeme] = symbol

    def remove_symbol(self, lexeme):
        for scope in reversed(self.scopes):
            if lexeme in scope:
                del scope[lexeme]
                return

    def get_symbol_by_lexeme(self, lexeme):
        for scope in reversed(self.scopes):
            if lexeme in scope:
                return scope[lexeme]
        return None

    def get_address_by_lexeme(self, lexeme):
        symbol = self.get_symbol_by_lexeme(lexeme)
        return symbol.address if symbol else None


class RuntimeStack:
    def __init__(self, codegen: "CodeGenerator"):
        self.codegen = codegen

    def push(self, data):
        self.codegen.add_instructions(
            Instruction.sub(self.codegen.stack_pointer_address, f"#{self.codegen.WORD_SIZE}", self.codegen.stack_pointer_address),
            Instruction.assign(data, f"@{self.codegen.stack_pointer_address}"),
        )

    def pop(self, address):
        self.codegen.add_instructions(
            Instruction.assign(f"@{self.codegen.stack_pointer_address}", address),
            Instruction.add(self.codegen.stack_pointer_address, f"#{self.codegen.WORD_SIZE}", self.codegen.stack_pointer_address),
        )


class CodeGenerator:
    WORD_SIZE = 4

    def __init__(self, parser: Parser):
        self.parser = parser
        self.semantic_stack = []
        self.instruction_index = 0
        self.instructions: OrderedDict[int, Instruction] = OrderedDict()
        self.data_address = 20000
        self.temp_address = 60000
        self.stack_start_address = self.temp_address - WORD_SIZE
        self.return_address_address = self.get_next_data_address()
        self.return_value_address = self.get_next_data_address()
        self.stack_pointer_address = self.get_next_data_address()
        initialization_instructions = (
            Instruction.assign(f"#{self.stack_start_address}", f"@{self.stack_pointer_address}"),
            Instruction.assign("#0", f"@{self.return_address_address}"),
            Instruction.assign("#0", f"@{self.return_value_address}"),
        )
        self.add_instructions(*initialization_instructions)

    def add_instruction(self, instruction, index=None):
        if index is None:
            index = self.instruction_index
            self.instruction_index += 1

        self.instructions[index] = instruction

    def add_instructions(self, *instructions):
        for instruction in instructions:
            self.add_instruction(instruction)

    def get_next_data_address(self, size=WORD_SIZE):
        next_data_address = self.data_address
        self.data_address += size
        return next_data_address

    def get_next_temp_address(self, size=WORD_SIZE):
        next_temp_address = self.temp_address
        self.temp_address += size
        return next_temp_address

    def generate_implicit_output(self):


    def action_pnext(self):
        self.semantic_stack.append(self.parser.lookahead.token_value)
