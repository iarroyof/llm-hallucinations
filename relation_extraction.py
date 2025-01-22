import spacy
from spacy.language import Language
from spacy.tokens import Doc


class ExtractorDeRelaciones:
    def __init__(self, model="en_core_web_md"):
        # Inicializa spaCy y configura el pipeline
        self.nlp = spacy.load(model)
        self._register_extensions()
        self._add_pipelines()

    def _register_extensions(self):
        if not Doc.has_extension("relaciones"):
            Doc.set_extension("relaciones", default=[], force=True)

    def _add_pipelines(self):
        @Language.component("extraer_relaciones")
        def extraer_relaciones(doc):
            relaciones = []

            for token in doc:
                if token.pos_ == "VERB":
                    sujeto = None
                    for hijo in token.children:
                        if hijo.dep_ == "nsubj":
                            sujeto = hijo.text

                    for hijo in token.children:
                        if hijo.dep_ in ["dobj", "pobj", "obj"]:
                            if sujeto:
                                relaciones.append((sujeto, token.text, hijo.text))

            doc._.relaciones = relaciones
            return doc

        self.nlp.add_pipe("extraer_relaciones", after="ner")

    def extraer_relaciones(self, textos):
        relaciones_cadenas = []
        for texto in textos:
            doc = self.nlp(texto)
            relaciones = doc._.relaciones

            # Convierte las relaciones en cadenas de texto
            if relaciones:
                for relacion in relaciones:
                    cadena = " ".join(relacion)
                    relaciones_cadenas.append(cadena)
        return relaciones_cadenas
    
if __name__ == "__main__":
    textos = [
        "Lewis drinks coffee in her house.",
        "Juan eats a delicious apple.",
        "Pedro sent an email to Ana.",
    ]

    extractor = ExtractorRelaciones()
    resultados = extractor.extraer_relaciones(textos)
    print("Relaciones extraídas:", resultados)
