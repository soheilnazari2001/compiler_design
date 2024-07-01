from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parser import Parser

SCOPE_SEMANTIC_ERROR = "Semantic Error! '{}' is not defined."
VOID_SEMANTIC_ERROR = "Semantic Error! Illegal type of void for '{}'."
ARG_COUNT_MISMATCH_SEMANTIC_ERROR = (
    "Semantic Error! Mismatch in numbers of arguments of '{}'."
)
BREAK_SEMANTIC_ERROR = "Semantic Error! No 'repeat ... until' found for 'break'."
OPERAND_TYPE_MISMATCH_SEMANTIC_ERROR = (
    "Semantic Error! Type mismatch in operands, Got {} instead of {}."
)
ARG_TYPE_MISMATCH_SEMANTIC_ERROR = "Semantic Error! Mismatch in type of argument {} of '{}'. Expected '{}' but got '{}' instead."


def indirect_address(address):
    if address.startswith("#"):
        return address[1:]

    return f"@{address}"


class Instruction:
    ARGS_COUNT = 3

    def __init__(self, opcode, *args):
        self.opcode = opcode
        self.args = args + ("",) * (self.ARGS_COUNT - len(args))

    def __repr__(self):
        return f"({self.opcode}, {', '.join(str(arg) for arg in self.args)})"

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
        return cls("PRSymbolType.INT.value", value)


class SemanticException(Exception):
    pass


class SymbolType(Enum):
    INT = "int"
    VOID = "void"
    ARRAY = "array"


class Symbol:
    def __init__(
        self, /, *, address=None, lexeme=None, type=None, size=0, param_count=0
    ):
        self.address = address
        self.lexeme = lexeme
        self.type = type
        self.size = size
        self.param_count = param_count
        self.param_symbols = []
        self.is_initialized = False
        self.is_function = False
        self.is_array = False


class ScopeStack:
    def __init__(self, codegen: "CodeGenerator"):
        self.scopes = [[]]
        self.codegen = codegen

    def add_symbol(self, lexeme, address):
        symbol = Symbol(lexeme=lexeme, address=address)
        self.scopes[-1].append(symbol)
        return symbol

    def remove_symbol(self, lexeme):
        scope = self.scopes[-1]
        for i, symbol in enumerate(scope):
            if symbol.lexeme == lexeme:
                scope.pop(i)
                return

    def find_address_by_lexeme(
        self, lexeme, check_declaration=False, force_declaration=False
    ):
        symbol = self.find_symbol_by_lexeme(
            lexeme, check_declaration, force_declaration
        )
        return symbol.address if symbol else None

    def find_symbol_by_lexeme(
        self,
        lexeme,
        check_declaration=False,
        force_declaration=False,
        prevent_add=False,
    ):
        if not force_declaration:
            for scope in reversed(self.scopes):
                for symbol in scope:
                    if symbol.lexeme == lexeme:
                        return symbol

        if not prevent_add:
            if check_declaration:
                raise SemanticException(SCOPE_SEMANTIC_ERROR.format(lexeme))
            address = self.codegen.get_next_data_address()
            return self.add_symbol(lexeme, address)

        return None

    def find_symbol_by_address(self, address):
        for scope in reversed(self.scopes):
            for symbol in scope:
                if symbol.address == address:
                    return symbol
        return None


class RuntimeStack:
    def __init__(self, codegen: "CodeGenerator"):
        self.codegen = codegen

    def push(self, data):
        self.codegen.add_instructions(
            Instruction.sub(
                self.codegen.stack_pointer_address,
                f"#{self.codegen.WORD_SIZE}",
                self.codegen.stack_pointer_address,
            ),
            Instruction.assign(data, f"@{self.codegen.stack_pointer_address}"),
        )

    def pop(self, address):
        self.codegen.add_instructions(
            Instruction.assign(f"@{self.codegen.stack_pointer_address}", address),
            Instruction.add(
                self.codegen.stack_pointer_address,
                f"#{self.codegen.WORD_SIZE}",
                self.codegen.stack_pointer_address,
            ),
        )


class Actor:
    INSTRUCTION_FROM_OPERATION = {
        "+": Instruction.add,
        "-": Instruction.sub,
        "<": Instruction.lt,
        "==": Instruction.eq,
        "*": Instruction.mult,
    }

    def __init__(self, codegen: "CodeGenerator", scope_stack: "ScopeStack"):
        self.codegen = codegen
        self.scope_stack = scope_stack
        self.argument_counts = []
        self.current_declared_function_symbol = None
        self.current_id = None
        self.is_rhs = False
        self.current_type = None
        self.called_functions = []
        self.no_push_flag = False
        self.check_declaration_flag = False
        self.function_scope_flag = False
        self.breaks = []
        self.has_reached_main = False
        self.force_declaration_flag = False
        self.current_id = ""
        self.void_flag = False
        self.found_arg_type_mismtach = []

    def raise_arg_type_mismatch_exception(self, index, lexeme, expected, got):
        if not self.found_arg_type_mismtach or not self.found_arg_type_mismtach[-1]:
            if len(self.found_arg_type_mismtach) == 0:
                self.found_arg_type_mismtach.append(True)
            self.found_arg_type_mismtach[-1] = True
            raise SemanticException(
                ARG_TYPE_MISMATCH_SEMANTIC_ERROR.format(index, lexeme, expected, got)
            )

    def pid(self, previous_token, current_token):
        self.current_id = previous_token.lexeme
        address = self.scope_stack.find_address_by_lexeme(
            previous_token.lexeme,
            self.check_declaration_flag,
            self.force_declaration_flag,
        )
        self.handle_main_function(previous_token)
        if not self.no_push_flag:
            self.codegen.semantic_stack.append(address)
        self.handle_operand_mismatch(current_token)
        self.handle_arg_mismatch(current_token, previous_token)

    def handle_main_function(self, previous_token):
        if previous_token.lexeme == "main":
            self.codegen.add_instruction(
                Instruction.jp(f"#{self.codegen.instruction_index}"),
                self.codegen.jump_to_main_address,
            )
            if not self.has_reached_main:
                for symbol in self.codegen.scope_stack.scopes[0]:
                    if not symbol.is_function:
                        self.codegen.add_instruction(
                            Instruction.assign("#0", symbol.address)
                        )
            self.has_reached_main = True

    def handle_operand_mismatch(self, current_token):
        if self.is_rhs:
            symbol = self.scope_stack.find_symbol_by_lexeme(
                self.current_id, prevent_add=True
            )
            if symbol.is_function:
                if symbol.type != SymbolType.INT.value:
                    raise SemanticException(
                        OPERAND_TYPE_MISMATCH_SEMANTIC_ERROR.format(
                            SymbolType.VOID.value, SymbolType.INT.value
                        )
                    )
            else:
                if symbol.is_array:
                    if current_token.lexeme != "[" and not self.argument_counts:
                        raise SemanticException(
                            OPERAND_TYPE_MISMATCH_SEMANTIC_ERROR.format(
                                SymbolType.ARRAY.value, SymbolType.INT.value
                            )
                        )

    def handle_arg_mismatch(self, current_token, previous_token):
        if len(self.argument_counts) > 0:
            index = self.argument_counts[-1]
            symbol: Symbol = self.scope_stack.find_symbol_by_lexeme(
                self.called_functions[-1], prevent_add=True
            )
            param_symbol: Symbol = symbol.param_symbols[index]
            current_symbol: Symbol = self.codegen.scope_stack.find_symbol_by_lexeme(
                previous_token.lexeme, prevent_add=True
            )
            if param_symbol.type == SymbolType.INT.value:
                if (
                    current_symbol.type == SymbolType.ARRAY.value
                    and current_token.lexeme != "["
                ):
                    self.raise_arg_type_mismatch_exception(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.INT.value,
                        SymbolType.ARRAY.value,
                    )
                if current_symbol.type == SymbolType.VOID.value:
                    self.raise_arg_type_mismatch_exception(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.INT.value,
                        SymbolType.VOID.value,
                    )
            if param_symbol.type == SymbolType.ARRAY.value:
                if current_symbol.type == SymbolType.INT.value:
                    self.raise_arg_type_mismatch_exception(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.ARRAY.value,
                        SymbolType.INT.value,
                    )
                if (
                    current_symbol.type == SymbolType.ARRAY.value
                    and current_token.lexeme == "["
                ):
                    self.raise_arg_type_mismatch_exception(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.ARRAY.value,
                        SymbolType.INT.value,
                    )
                if current_symbol.type == SymbolType.VOID.value:
                    self.raise_arg_type_mismatch_exception(
                        index + 1,
                        symbol.lexeme,
                        SymbolType.ARRAY.value,
                        SymbolType.VOID.value,
                    )

    def pnum(self, previous_token, current_token):
        num = f"#{previous_token.lexeme}"
        if not self.no_push_flag:
            self.codegen.semantic_stack.append(num)
        if len(self.argument_counts) > 0:
            index = self.argument_counts[-1]
            symbol: Symbol = self.scope_stack.find_symbol_by_lexeme(
                self.called_functions[-1], prevent_add=True
            )
            param_symbol: Symbol = symbol.param_symbols[index]
            if param_symbol.type == SymbolType.ARRAY.value:
                self.raise_arg_type_mismatch_exception(
                    index + 1,
                    symbol.lexeme,
                    SymbolType.ARRAY.value,
                    SymbolType.INT.value,
                )

    def label(self, previous_token, current_token):
        self.codegen.semantic_stack.append(f"#{self.codegen.instruction_index}")

    def save(self, previous_token, current_token):
        self.codegen.semantic_stack.append(f"#{self.codegen.instruction_index}")
        self.codegen.instruction_index += 1

    def push_operation(self, previous_token, current_token):
        self.codegen.semantic_stack.append(previous_token.lexeme)

    def execute(self, previous_token, current_token):
        temp_address = self.codegen.get_next_temp_address()
        operand2 = self.codegen.semantic_stack.pop()
        operation = self.codegen.semantic_stack.pop()
        operand1 = self.codegen.semantic_stack.pop()
        self.codegen.semantic_stack.append(temp_address)
        instruction = self.INSTRUCTION_FROM_OPERATION[operation](
            operand1, operand2, temp_address
        )
        self.codegen.add_instruction(instruction)

    def start_argument_list(self, previous_token, current_token):
        self.argument_counts.append(0)
        self.called_functions.append(self.current_id)
        self.found_arg_type_mismtach.append(False)

    def end_argument_list(self, previous_token, current_token):
        self.found_arg_type_mismtach.pop()

    def jp_from_saved(self, previous_token, current_token):
        instruction = Instruction.jp(f"#{self.codegen.instruction_index}")
        destination = self.codegen.semantic_stack.pop()
        self.codegen.add_instruction(instruction, destination)

    def jpf_from_saved(self, previous_token, current_token):
        destination = self.codegen.semantic_stack.pop()
        condition = self.codegen.semantic_stack.pop()
        instruction = Instruction.jpf(condition, f"#{self.codegen.instruction_index}")
        self.codegen.add_instruction(instruction, destination)

    def save_and_jpf_from_last_save(self, previous_token, current_token):
        destination = self.codegen.semantic_stack.pop()
        condition = self.codegen.semantic_stack.pop()
        instruction = Instruction.jpf(
            condition, f"#{self.codegen.instruction_index + 1}"
        )
        self.codegen.add_instruction(instruction, destination)
        self.codegen.semantic_stack.append(f"#{self.codegen.instruction_index}")
        self.codegen.instruction_index += 1

    def assign(self, previous_token, current_token):
        value = self.codegen.semantic_stack.pop()
        address = self.codegen.semantic_stack.pop()
        instruction = Instruction.assign(value, address)
        self.codegen.add_instruction(instruction)
        self.codegen.semantic_stack.append(value)
        symbol: Symbol = self.codegen.scope_stack.find_symbol_by_address(address)
        if symbol:
            symbol.is_initialized = True

    def start_no_push(self, previous_token, current_token):
        if not self.function_scope_flag:
            self.no_push_flag = True

    def end_no_push(self, previous_token, current_token):
        self.no_push_flag = False

    def declare_array(self, previous_token, current_token):
        # use [1:] to skip the '#'
        length = int(self.codegen.semantic_stack.pop()[1:])
        symbol: Symbol = self.codegen.scope_stack.scopes[-1][-1]
        symbol.is_array = True
        symbol.type = SymbolType.ARRAY.value
        size = length * self.codegen.WORD_SIZE
        array_start_address = self.codegen.get_next_data_address(size=size)
        self.codegen.add_instruction(
            Instruction.assign(f"#{array_start_address}", symbol.address)
        )
        if len(self.codegen.scope_stack.scopes) > 1:
            for address in range(
                array_start_address, array_start_address + size, self.codegen.WORD_SIZE
            ):
                self.codegen.add_instruction(Instruction.assign("#0", address))

    def array(self, previous_token, current_token):
        offset = self.codegen.semantic_stack.pop()
        temp = self.codegen.get_next_temp_address()
        array_start = self.codegen.semantic_stack.pop()
        instructions = [
            Instruction.mult(offset, f"#{self.codegen.WORD_SIZE}", temp),
            Instruction.add(temp, f"{array_start}", temp),
        ]
        self.codegen.add_instructions(instructions)
        self.codegen.semantic_stack.append(f"@{temp}")

    def until(self, previous_token, current_token):
        condition = self.codegen.semantic_stack.pop()
        destination = self.codegen.semantic_stack.pop()
        instruction = Instruction.jpf(condition, destination)
        self.codegen.add_instruction(instruction)

    def start_break_scope(self, previous_token, current_token):
        self.breaks.append([])

    def add_break(self, previous_token, current_token):
        if not self.breaks:
            raise SemanticException(BREAK_SEMANTIC_ERROR)
        self.breaks[-1].append(self.codegen.instruction_index)
        self.codegen.instruction_index += 1

    def handle_breaks(self, previous_token, current_token):
        for destination in self.breaks[-1]:
            instruction = Instruction.jp(f"#{self.codegen.instruction_index}")
            # insert method
            self.codegen.add_instruction(instruction, destination)
        self.breaks.pop()

    def pop(self, previous_token, current_token):
        self.codegen.semantic_stack.pop()

    def check_declaration(self, previous_token, current_token):
        self.check_declaration_flag = True

    def uncheck_declaration(self, previous_token, current_token):
        self.check_declaration_flag = False

    def set_function_scope_flag(self, previous_token, current_token):
        self.function_scope_flag = True

    def open_scope(self, previous_token, current_token):
        if not self.function_scope_flag:
            self.codegen.scope_stack.scopes.append([])
        self.function_scope_flag = False
        self.codegen.execution_flow_stack.append(
            (self.codegen.data_address, self.codegen.temp_address)
        )

    def close_scope(self, previous_token, current_token):
        self.codegen.scope_stack.scopes.pop()
        self.codegen.data_address, self.codegen.temp_address = (
            self.codegen.execution_flow_stack.pop()
        )

    def pop_param(self, previous_token, current_token):
        address = self.codegen.semantic_stack.pop()
        self.codegen.runtime_stack.pop(address)
        symbol: Symbol = self.codegen.scope_stack.find_symbol_by_address(address)
        symbol.type = self.current_type
        if previous_token and previous_token.lexeme == "]":
            symbol.type = SymbolType.ARRAY.value
            symbol.is_array = True
        self.current_declared_function_symbol.param_symbols.append(symbol)
        if symbol:
            symbol.is_initialized = True
            self.current_declared_function_symbol.param_count += 1

    def declare_function(self, previous_token, current_token):
        symbol: Symbol = self.codegen.scope_stack.scopes[-1][-1]
        symbol.address = f"#{self.codegen.instruction_index}"
        symbol.is_function = True
        symbol.type = self.current_type
        symbol.param_count = 0
        self.current_declared_function_symbol = symbol
        self.void_flag = False
        self.codegen.function_data_start_pointer = self.codegen.data_address
        self.codegen.function_temp_start_pointer = self.codegen.temp_address

    def call(self, previous_token, current_token):
        self.store_data_and_temp()
        self.codegen.push_addresses()

        arg_count = self.argument_counts.pop()
        self.codegen.save_return_address(arg_count)

        self.make_call(arg_count)

        self.codegen.pop_addresses()
        self.restore_data_and_temp()

        self.retrieve_return_value()

        function_name = self.called_functions.pop()
        symbol = self.codegen.scope_stack.find_symbol_by_lexeme(function_name)
        if symbol.param_count != arg_count:
            raise SemanticException(
                ARG_COUNT_MISMATCH_SEMANTIC_ERROR.format(function_name)
            )

    def retrieve_return_value(self):
        temp = self.codegen.get_next_temp_address()
        self.codegen.semantic_stack.append(temp)
        self.codegen.add_instruction(
            Instruction.assign(self.codegen.return_value_address, temp)
        )

    def restore_data_and_temp(self):
        for address in range(
            self.codegen.temp_address,
            self.codegen.function_temp_start_pointer,
            -self.codegen.WORD_SIZE,
        ):
            self.codegen.runtime_stack.pop(address - self.codegen.WORD_SIZE)
        for address in range(
            self.codegen.data_address,
            self.codegen.function_data_start_pointer,
            -self.codegen.WORD_SIZE,
        ):
            symbol: Symbol = self.codegen.scope_stack.find_symbol_by_address(
                address - self.codegen.WORD_SIZE
            )
            if symbol and symbol.is_initialized:
                self.codegen.runtime_stack.pop(address - self.codegen.WORD_SIZE)

    def make_call(self, arg_count):
        for _ in range(arg_count):
            data = self.codegen.semantic_stack.pop()
            self.codegen.runtime_stack.push(data)
        address = self.codegen.semantic_stack.pop()
        instruction = Instruction.jp(address)
        self.codegen.add_instruction(instruction)

    def store_data_and_temp(self):
        for address in range(
            self.codegen.function_data_start_pointer,
            self.codegen.data_address,
            self.codegen.WORD_SIZE,
        ):
            symbol: Symbol = self.codegen.scope_stack.find_symbol_by_address(address)
            if symbol and symbol.is_initialized:
                self.codegen.runtime_stack.push(address)
        for address in range(
            self.codegen.function_temp_start_pointer,
            self.codegen.temp_address,
            self.codegen.WORD_SIZE,
        ):
            self.codegen.runtime_stack.push(address)

    def set_return_value(self, previous_token, current_token):
        value = self.codegen.semantic_stack.pop()
        self.codegen.save_return_value(value)

    def jump_back(self, previous_token, current_token):
        if not self.has_reached_main:
            instruction = Instruction.jp(self.codegen.return_address_address)
            self.codegen.add_instruction(instruction)

    def add_argument_count(self, previous_token, current_token):
        self.found_arg_type_mismtach[-1] = False
        self.argument_counts[-1] += 1

    def zero_initialize(self, previous_token, current_token):
        if len(self.codegen.scope_stack.scopes) > 1:
            symbol: Symbol = self.codegen.scope_stack.scopes[-1][-1]
            if not symbol.is_array:
                symbol.type = SymbolType.INT.value
            self.codegen.add_instruction(Instruction.assign("#0", symbol.address))

    def array_param(self, previous_token, current_token):
        symbol: Symbol = self.codegen.scope_stack.scopes[-1][-1]
        symbol.is_array = True
        symbol.type = SymbolType.ARRAY.value

    def set_force_declaration_flag(self, previous_token, current_token):
        self.force_declaration_flag = True

    def unset_force_declaration_flag(self, previous_token, current_token):
        self.force_declaration_flag = False

    def void_check(self, previous_token, current_token):
        self.void_flag = True

    def void_check_throw(self, previous_token, current_token):
        if self.void_flag:
            self.void_flag = False
            self.codegen.scope_stack.remove_symbol(self.current_id)
            raise SemanticException(VOID_SEMANTIC_ERROR.format(self.current_id))

    def save_type(self, previous_token, current_token):
        self.current_type = previous_token.lexeme

    def start_rhs(self, previous_token, current_token):
        self.is_rhs = True

    def end_rhs(self, previous_token, current_token):
        self.is_rhs = False

    def check_type(self, previous_token, current_token):
        symbol = self.scope_stack.find_symbol_by_lexeme(
            self.current_id, prevent_add=True
        )
        if symbol.is_array:
            if current_token.lexeme != "[" and not self.argument_counts:
                raise SemanticException(
                    OPERAND_TYPE_MISMATCH_SEMANTIC_ERROR.format(
                        SymbolType.ARRAY.value, SymbolType.INT.value
                    )
                )

    def negate(self, previous_token, current_token):
        value = self.codegen.semantic_stack[-1]


class CodeGenerator:
    WORD_SIZE = 4

    def __init__(self, parser: "Parser"):
        self.parser = parser
        self.runtime_stack = RuntimeStack(self)
        self.scope_stack = ScopeStack(self)
        self.actor = Actor(self, self.scope_stack)
        self.semantic_stack = []
        self.execution_flow_stack = []
        self.instruction_index = 0
        self.instructions: OrderedDict[int, Instruction] = OrderedDict()
        self.data_address = 20000
        self.temp_address = 60000
        self.stack_start_address = self.temp_address - self.WORD_SIZE
        self.return_address_address = self.get_next_data_address()
        self.return_value_address = self.get_next_data_address()
        self.stack_pointer_address = self.get_next_data_address()
        self.add_instructions(
            Instruction.assign(
                f"#{self.stack_start_address}", f"@{self.stack_pointer_address}"
            ),
            Instruction.assign("#0", f"@{self.return_address_address}"),
            Instruction.assign("#0", f"@{self.return_value_address}"),
        )
        self.jump_to_main_address = len(self.instructions)
        self.instruction_index += 1
        self.generate_implicit_output()

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

    def push_addresses(self):
        self.runtime_stack.push(self.return_address_address)
        self.runtime_stack.push(self.stack_pointer_address)

    def pop_addresses(self):
        self.runtime_stack.pop(self.stack_pointer_address)
        self.runtime_stack.pop(self.return_address_address)

    def save_return_address(self, arg_count):
        self.add_instruction(
            Instruction.assign(
                f"#{self.instruction_index + 2 * (arg_count + 1)}",
                self.return_address_address,
            )
        )

    def save_return_value(self, value):
        self.add_instruction(Instruction.assign(value, self.return_value_address))

    def generate_implicit_output(self):
        pass  # TODO

    def do_action(self, identifier, previous_token, current_token):
        getattr(self.actor, identifier[1:])(previous_token, current_token)
