import numpy as np
from tqdm import tqdm


from sklearn.metrics.pairwise import cosine_similarity



# carga del modelo para usar fasttext
def cargar_modelo():
    print('cargando embeddings...')
    embeddings = {}
    f = open('./glove-sbwc.i25.vec', encoding='utf-8')
    for line in tqdm(f):
        datos = line.strip().rsplit(' ')
        token = datos[0]
        embedding = np.asarray(datos[1:], dtype='float32')
        embeddings[token] = embedding
    f.close()
    print('encontrados %s tokens' % len(embeddings))

    return embeddings


def obtener_vector(palabra, modelo):
    try:
        resultado = modelo[palabra]
    except KeyError:
        resultado = None
    return resultado

fasttext = cargar_modelo()

# ejemplo con elementos relacionados con agua y aves
palabras = ['agua', 'rio', 'embalse', 'mar', 'grifo', 'botella', 'pez', 'cocodrilo', 'gaviota', 'buitre']
tokens = [obtener_vector(token, fasttext) for token in palabras]
similitud = cosine_similarity(tokens)
rounded = np.round(similitud, decimals=3)
print(f'Comparacion de los embeddings de {palabras}.')
print(rounded)
print('\n\n')

# ejemplo con cánidos
palabras = ['beagle', 'labrador', 'retriever', 'perro', 'dálmata', 'lobo', 'hiena']
tokens = [obtener_vector(token, fasttext) for token in palabras]
similitud = cosine_similarity(tokens)
rounded = np.round(similitud, decimals=3)
print(f'Comparacion de los embeddings de {palabras}.')
print(rounded)

