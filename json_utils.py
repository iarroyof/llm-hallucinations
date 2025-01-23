import json

class JSONLIterator:
    def __init__(self, file_path, keys, n_samples):
        """
        Initialize the JSONL iterator.

        Args:
            file_path (str): Path to the JSONL file.
            keys (list): List of keys to extract from each JSON object.
        """
        self.file_path = file_path
        self.keys = keys
        self.n_samples = n_samples

    def __iter__(self):
        """
        Iterate over the JSONL file and yield the specified keys as a tuple.
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for n, line in enumerate(file):
                if n < self.n_samples:
                    data = json.loads(line.strip())
                    yield tuple(data[key] for key in self.keys)

# Example usage
file_path = 'path/to/your/file.jsonl'
keys = ['model_input', 'model_output_text']

jsonl_iterator = JSONLIterator(file_path, keys)

# Iterate over the file and process each item
for model_input, model_output_text in jsonl_iterator:
    print(f"Input: {model_input}")
    print(f"Output: {model_output_text}")
    print("-" * 40)
