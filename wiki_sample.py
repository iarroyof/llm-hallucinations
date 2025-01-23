import wikipediaapi

class WikipediaBatchGenerator:
    def __init__(self):
        self.wiki_wiki = wikipediaapi.Wikipedia('en', user_agent="IarroyoF/0.0 (https://iarroyof.github.io/; iaf@gs.utm.mx) generic-package/0.0")
        self.themes = self._generate_themes()  # Generate at least 100 themes
        self.batches = self._generate_batches()

    def _generate_themes(self):
        """Generate a list of at least 100 themes, mixing intuitive and intricate topics."""
        intuitive_themes = [
            'Dog', 'Cat', 'Elephant', 'Lion', 'Tiger', 'Eagle', 'Shark', 'Dolphin', 'Penguin', 'Kangaroo',
            'Giraffe', 'Zebra', 'Bear', 'Wolf', 'Fox', 'Rabbit', 'Horse', 'Cow', 'Sheep', 'Goat',
            'Chicken', 'Duck', 'Swan', 'Owl', 'Parrot', 'Peacock', 'Flamingo', 'Panda', 'Koala', 'Gorilla',
            'Monkey', 'Cheetah', 'Leopard', 'Rhino', 'Hippo', 'Crocodile', 'Alligator', 'Turtle', 'Frog', 'Snake',
            'Whale', 'Octopus', 'Jellyfish', 'Starfish', 'Crab', 'Lobster', 'Butterfly', 'Bee', 'Ant', 'Spider'
        ]
        intricate_themes = [
            'Philosophy', 'Existentialism', 'Quantum mechanics', 'Epistemology', 'Metaphysics', 'Phenomenology',
            'Ethics', 'Aesthetics', 'Logic', 'Ontology', 'Determinism', 'Free will', 'Consciousness', 'Mind-body problem',
            'Rationalism', 'Empiricism', 'Idealism', 'Materialism', 'Dualism', 'Structuralism', 'Post-structuralism',
            'Deconstruction', 'Pragmatism', 'Utilitarianism', 'Kantianism', 'Nihilism', 'Stoicism', 'Humanism', 'Feminism',
            'Marxism', 'Capitalism', 'Socialism', 'Communism', 'Anarchism', 'Liberalism', 'Conservatism', 'Nationalism',
            'Globalization', 'Postmodernism', 'Relativism', 'Skepticism', 'Solipsism', 'Transhumanism', 'Existential risk',
            'Artificial intelligence', 'Machine learning', 'Quantum computing', 'String theory', 'Relativity', 'Big Bang',
            'Dark matter', 'Dark energy', 'Multiverse', 'Simulation hypothesis', 'Consciousness studies', 'Neurophilosophy',
            'Philosophy of mind', 'Philosophy of language', 'Philosophy of science', 'Philosophy of mathematics',
            'Philosophy of religion', 'Philosophy of history', 'Philosophy of art', 'Philosophy of technology',
            'Philosophy of education', 'Philosophy of law', 'Philosophy of politics', 'Philosophy of economics',
            'Philosophy of culture', 'Philosophy of time', 'Philosophy of space', 'Philosophy of logic', 'Philosophy of perception'
        ]
        # Combine and ensure at least 100 themes
        return intuitive_themes + intricate_themes[:100 - len(intuitive_themes)]

    def _get_first_paragraph(self, title):
        """Fetch the first paragraph of a Wikipedia article."""
        page = self.wiki_wiki.page(title)
        if page.exists():
            sections = page.sections
            if sections:
                return sections[0].text
        return None

    def _generate_batches(self):
        """Generate batches, where each batch corresponds to a theme."""
        batches = []
        for theme in self.themes:
            paragraph = self._get_first_paragraph(theme)
            if paragraph:
                batches.append([paragraph])  # Each batch is a list containing one paragraph
        return batches

    def get_batches(self):
        """Return the list of batches."""
        self.len = len(self.batches)
        return self.batches


# Example usage
if __name__ == "__main__":
    generator = WikipediaBatchGenerator()
    batches = generator.get_batches()

    # Print the total number of batches
    print(f"Total batches: {len(batches)}")

    # Print the first 100 characters of each batch
    for i, batch in enumerate(batches):
        print(f"Batch {i+1} (Theme: {generator.themes[i]}):")
        for j, paragraph in enumerate(batch):
            print(f"  {paragraph[:100]}...")  # Print first 100 characters of the paragraph
