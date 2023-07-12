from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


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

word2vec = cargar_modelo()

# ejemplo con términos de la realeza
palabras = ['rey', 'hombre', 'mujer', 'reina']
tokens = [obtener_vector(token, word2vec) for token in palabras]
print('Vectores de cada término:')
for cont in range(4):
    print(f'{palabras[cont]}: {tokens[cont]}.\n')

operacion = tokens[0] - tokens[1] + tokens[2]
comparar = [operacion, tokens[3]]

similitud = np.round(cosine_similarity(comparar), decimals=4)
print('Similitud de rey-hombre+mujer con reina.')
# puede observarse que el resultado de la operación es similar en un 77% a reina
print(f'{similitud}\n')

# ejemplo con animales
mascotas = ['perro', 'macho', 'hembra', 'perra']
tokens = [obtener_vector(token, word2vec) for token in mascotas]
operacion = tokens[0] - tokens[1] + tokens[2]
comparar = [operacion, tokens[3]]

similitud = np.round(cosine_similarity(comparar), decimals=4)
print('Similitud de perro-macho+hembra con perra.')
print(f'{similitud}\n')
