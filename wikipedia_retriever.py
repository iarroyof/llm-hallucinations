import wikipedia

class WikipediaSearch:
    def __init__(self, lang="en"):
        wikipedia.set_lang(lang)

    def get_background_knowledge(self, topic, max_chars=2000):

        print(f"Buscando en Wikipedia: '{topic}'...")
        
        try:
            search_results = wikipedia.search(topic, results=1)
            
            if not search_results:
                return None
            
            best_match = search_results[0]
            print(f" Mejor coincidencia encontrada: {best_match}")


            page = wikipedia.page(best_match, auto_suggest=False)
            

            full_text = page.content
            return full_text[:max_chars]

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"Ambigüedad detectada. Probando con: {e.options[0]}")
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return page.content[:max_chars]
            except:
                return None
                
        except wikipedia.exceptions.PageError:
            print("Página no encontrada.")
            return None
            
        except Exception as e:
            print(f"Error inesperado: {e}")
            return None

if __name__ == "__main__":
    wiki = WikipediaSearch()
    contexto = wiki.get_background_knowledge("Barack Obama birth place")
    
    if contexto:
        print("\n--- CONTEXTO RECUPERADO (Ground Truth) ---")
        print(contexto[:10000] + "...") 
    else:
        print("No se encontró información.")