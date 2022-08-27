import io
import re
import string
from tqdm import tqdm
import numpy as np
import tensorflow as tf

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed=SEED):
    targets, contexts, labels = [], [], []
    
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(size=vocab_size)

    for sequence in tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0
        )

        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(
                tf.constant([context_word], dtype='int64'), 1
            )
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed
            )
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype='int64')

            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
        
    return targets, contexts, labels

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, f'[{re.escape(string.punctuation)}]', '')

vocab_size = 4096
sequence_length = 10
num_ns = 4

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

vectorize_layer.adapt(text_ds.batch(1024))

inverse_vocab = vectorize_layer.get_vocabulary()

text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

sequences = list(text_vector_ds.as_numpy_iterator())

targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=num_ns,
    vocab_size=vocab_size,
    seed=SEED
)

targets = np.array(targets)
contexts = np.array(contexts)[:,:,0]
labels = np.array(labels)

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

class Word2Vec(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.target_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name='w2v_embedding'
        )
        self.context_embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns+1
        )
    
    def call(self, pair):
        target, context = pair
        
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        return dots

embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
word2vec.fit(dataset, epochs=20)

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

with io.open('vectors.tsv', 'w', encoding='utf-8') as out_v, io.open('metadata.tsv', 'w', encoding='utf-8') as out_m:
    for index, word in enumerate(vocab):
        if index == 0:
            continue
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + '\n')
        out_m.write(word + '\n')
