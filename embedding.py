import numpy as np

def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# Convert sentence to average embedding
def sentence_to_embedding(sentence, embeddings, dim=100):
    words = sentence.split()
    valid_vectors = [embeddings[word] for word in words if word in embeddings]

    if not valid_vectors:
        return np.zeros(dim)
    
    return np.mean(valid_vectors, axis=0)
