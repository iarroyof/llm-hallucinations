import trafilatura
from ddgs import DDGS

def find_evidence(query, max_results=3, max_chars=2000):
    """
    Busca enlaces y extrae el contenido textual completo de las páginas.
    """
    ddgs = DDGS()
    evidence_texts = []
    
    # Excluir archivos PDF de los resultados
    refined_query = f"{query} -filetype:pdf"
    
    try:
        results = list(ddgs.text(refined_query, max_results=max_results))
        
        for result in results:
            url = result.get('href')
            title = result.get('title')
            fallback_snippet = result.get('body', '')
            
            content = None
            
            # Intentar descargar y extraer texto de la web
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    extracted = trafilatura.extract(downloaded)
                    if extracted:
                        # Limpiar saltos de linea y recortar
                        content = extracted.replace("\n", " ")[:max_chars]
            except Exception:
                pass
            
            # Si falló la extracción, usar el resumen de DDG
            if not content:
                content = fallback_snippet

            formatted_evidence = f"TITLE: {title}\nURL: {url}\nCONTENT: {content}"
            evidence_texts.append(formatted_evidence)
            
    except Exception as e:
        print(f"Error en el proceso de busqueda: {e}")

    return evidence_texts

#Ejemplo de uso 
if __name__ == "__main__":
    query = "Argumento Ninja Gaiden Sigma"
    data = find_evidence(query, max_results=2)
    
    for i, item in enumerate(data, 1):
        print(f"--- Evidence {i} ---")
        print(item)
        print("\n")