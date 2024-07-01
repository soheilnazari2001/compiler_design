from parser import Parser


def main(
    input_file_path, output_generated_code_file_path, output_semantic_errors_file_path
):
    with open(input_file_path, "r") as input_file:
        parser = Parser(input_file)
        parser.parse()
    parser.print_generated_code(output_generated_code_file_path)
    parser.print_semantic_errors(output_semantic_errors_file_path)


if __name__ == "__main__":
    main("input.txt", "output.txt", "semantic_errors.txt")
