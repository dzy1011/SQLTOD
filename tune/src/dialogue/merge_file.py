import argparse
import json

def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'w') as outfile:
        for file in input_files:
            with open(file, 'r') as infile:
                for line in infile:
                    outfile.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSONL files into one")
    parser.add_argument('input_files', nargs='+', help='List of input JSONL files to merge')
    parser.add_argument('--output_file', required=True, help='Output file name for the merged JSONL')

    args = parser.parse_args()

    merge_jsonl_files(args.input_files, args.output_file)
    print(f"Merge completed, result saved in {args.output_file}")
