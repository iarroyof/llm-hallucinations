import wikipedia
import Levenshtein

class WikipediaSearch:
    """
    # Example usage:
    searcher = WikipediaSearch(k=3)
    
    ## Search and get background knowledge based on titles:
    ```python
    background_title = searcher.get_background_knowledge("Artificial intelligence")
    print("--- Background based on titles ---")
    print(background_title)
    
    ## Search and get background knowledge based on summaries:
    ```python
    background_summary = searcher.get_background_knowledge("Artificial intelligence", use_summary=True)
    print("\n--- Background based on summaries ---")
    print(background_summary)
    
    ## Empty response handdling
    ```python
    background_custom = searcher.get_background_knowledge("NonExistentTopic", custom_message="No info available.")
    print("\n--- Custom message ---")
    print(background_custom)
    """
    def __init__(self, k=5, no_knowledge_message="No background knowledge for this query."):
        self.k = k
        self.no_knowledge_message = no_knowledge_message

    def morelike_ranked(self, topic, num_results=10, reverse=False):
        results = wikipedia.search(topic, results=num_results)
        ranked_results = []
        for page in results:
            try:
                summary = wikipedia.summary(page)
                distance_title = Levenshtein.distance(topic.lower(), page.lower())
                distance_summary = Levenshtein.distance(topic.lower(), summary.lower())
                combined_distance = distance_title * 0.8 + distance_summary * 0.2  # Example weighting
                combined_similarity = 1 / (1 + combined_distance)  # Convert distance to similarity
                ranked_results.append((page, combined_similarity))
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Disambiguation for {page}: {e.options}")
            except Exception as e:
                print(f"Error for {page}: {e}")

        ranked_results.sort(key=lambda x: x[1], reverse=reverse)  # Sort by similarity (descending)
        return [page for page, similarity in ranked_results]

    def morelike_ranked_by_summary(self, topic, num_results=10, reverse=False):
        results = wikipedia.search(topic, results=num_results)
        ranked_results = []
        for page in results:
            try:
                summary = wikipedia.summary(page)
                distance_summary = Levenshtein.distance(topic.lower(), summary.lower())
                combined_similarity = 1 / (1 + distance_summary)
                ranked_results.append((page, combined_similarity))
            except wikipedia.exceptions.DisambiguationError as e:
                print(f"Disambiguation for {page}: {e.options}")
            except Exception as e:
                print(f"Error for {page}: {e}")

        ranked_results.sort(key=lambda x: x[1], reverse=reverse)
        return [page for page, similarity in ranked_results]

    def get_background_knowledge(self, topic, custom_message=None, use_summary=False):  # Added use_summary parameter
        if use_summary:
            similar_topics = self.morelike_ranked_by_summary(topic)
        else:
            similar_topics = self.morelike_ranked(topic)

        if not similar_topics:
            return custom_message or self.no_knowledge_message

        k_most_similar = similar_topics[:min(self.k, len(similar_topics))]
        background_knowledge = []

        for similar_topic in k_most_similar:
            try:
                page = wikipedia.page(similar_topic)
                background_knowledge.append(f"## {similar_topic}\n\n{page.summary}\n\n")
            except wikipedia.exceptions.PageError:
                print(f"Page not found: {similar_topic}")
            except Exception as e:
                print(f"Error retrieving page for {similar_topic}: {e}")

        if not background_knowledge:
            return custom_message or self.no_knowledge_message

        return background_knowledge
