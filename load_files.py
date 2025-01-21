import json

def load_data_jsonl(path_file: str, labels: list = ['model_input','model_output_text']) -> dict:
    """
    Carga un archivo JSON Lines (.jsonl) y lo convierte en un dicionario de listas.

    Args:
        path_file (str): Ruta del archivo JSON Lines.

    Returns:
        dict: Diccionario de listas donde cada elemento de la lista corresponde al valor del dato en el archivo jsonl.
    """
    data_dict = {}
    for label in labels:
        data_dict[label] = []
    
    with open(path_file, 'r', encoding = 'utf-8') as file:
        for line in file:
            if line.strip():
                temp_dict = json.loads(line)
                for label in labels:
                    data_dict[label].append(temp_dict.get(label))
                
    return data_dict

if __name__ == "__main__":
    path = './val/mushroom.en-val.v2.jsonl'
    new_dict = load_data_jsonl(path)

    len_inputs = len(new_dict['model_input'])
    len_outputs = len(new_dict['model_output_text'])

    print(f'Labels: {new_dict.keys()}')
    print(f'Length inputs: {len_inputs}')
    print(f'Length outputs: {len_outputs}')