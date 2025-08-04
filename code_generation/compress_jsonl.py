import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
args = parser.parse_args()

with open(args.input_file, 'r', encoding='utf-8') as infile, open(args.output_file, 'w', encoding='utf-8') as outfile:
    for line_num, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            continue
            
        try:
            data = json.loads(line)
            
            # Set problem.solutions and problem.test_cases to empty if they exist
            if 'problem' in data and isinstance(data['problem'], dict):
                if 'solutions' in data['problem']:
                    data['problem']['solutions'] = []
                if 'test_cases' in data['problem']:
                    data['problem']['test_cases'] = []
            
            # Write the compressed data
            outfile.write(json.dumps(data) + '\n')
        except json.JSONDecodeError as e:
            print(f"Error on line {line_num}: {e}")
            print(f"Line content: {line[:100]}...")

print(f"Compression complete. Output saved to {args.output_file}")