from scipy.spatial.distance import cosine
import numpy as np

with open('data.tsv', encoding='utf-8') as data:
    words = {}
    try:
        while True:
            row = next(data).split()
            word = row[0]
            vector = np.array([float(x) for x in row[1:]])
            words[word] = vector
    except Exception as e:
        pass

def distance(w1, w2):
    return cosine(w1, w2)

def closest_words(embedding):
    if isinstance(embedding, str):
        embedding = words[embedding]
    distances = {
        w: distance(embedding, words[w]) for w in words
    }
    return sorted(distances, key=lambda w: distances[w])[:10]

def closest_word(embedding):
    return closest_words(embedding)[0]
