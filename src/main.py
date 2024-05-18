from collections import Counter
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

nlp = spacy.load('en_core_web_sm')

url = "https://devapi.beyondchats.com/api/get_message_with_sources"
threshold = 0.2
final_output = []


def preprocess_and_get_word_frequencies(text):
    try:
        doc = nlp(text.lower())
        words = [token.text for token in doc if token.is_alpha and not token.is_stop]
        return Counter(words)
    except Exception as e:
        print("An error occurred during preprocessing:", e)
        return Counter()


while url:
    # Make a GET request to the API
    response = requests.get(url)


    if response.status_code == 200:
       
        res = response.json()

       
        for pair in res['data']['data']:
            citations = []
            pair_response = pair['response']
            response_word_frequencies = preprocess_and_get_word_frequencies(
                pair_response)
            for src in pair['source']:
                source_word_frequencies = preprocess_and_get_word_frequencies(
                    src["context"])

                # Get the set of all unique words from both response and source
                unique_words = set(response_word_frequencies.keys()) | set(
                    source_word_frequencies.keys())

                # Create vectors of word frequencies for both response and source
                response_vector = np.array(
                    [response_word_frequencies[word] for word in unique_words])
                source_vector = np.array(
                    [source_word_frequencies[word] for word in unique_words])

                # Calculate cosine similarity
                similarity = cosine_similarity(response_vector.reshape(
                    1, -1), source_vector.reshape(1, -1))[0][0]
                
                # Check if similarity exceeds threshold
                if similarity >= threshold:
                    citations.append({
                        "id": src["id"],
                        "link": src["link"]})
            final_output.append(citations)

    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
    url = res['data']['next_page_url']


print(final_output)
