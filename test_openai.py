import os
from openai import OpenAI

# Usa la variable de entorno SOLAR_API_KEY para la clave de la API.
# Puedes definirla en tu entorno o reemplazar 'YOUR_DEFAULT_API_KEY' por tu clave.
SOLAR_API_KEY = os.environ.get('SOLAR_API_KEY', 'YOUR_DEFAULT_API_KEY')

# Configuración del cliente para la integración de NVIDIA con el modelo Solar.
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=SOLAR_API_KEY
)

def generate_limerick(prompt: str) -> str:
    """
    Envía un prompt al modelo upstage/solar-10.7b-instruct y devuelve el resultado.
    Se utiliza streaming para mostrar la respuesta a medida que llega.
    """
    messages = [{"role": "user", "content": prompt}]
    
    completion = client.chat.completions.create(
        model="upstage/solar-10.7b-instruct",
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        max_tokens=1024,
        stream=True
    )

    result = ""
    # Procesa el streaming: a medida que llegan "chunks", se imprime y se acumula en 'result'
    for chunk in completion:
        # chunk.choices es una lista y en cada chunk, el primer elemento suele contener el delta
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            result += content
    return result

if __name__ == "__main__":
    prompt = "Write a limerick about the wonders of GPU computing."
    print("Generating limerick:\n")
    limerick = generate_limerick(prompt)
    print("\n\nLimerick generated:")
    print(limerick)