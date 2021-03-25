import utilidades

corpora = utilidades.Corpora()
util = utilidades.Util()

embedding_path_50 = 'pytorch_lstmcrf/data/glove_s50.txt'

embedding_path_100 = 'pytorch_lstmcrf/data/glove_s100.txt'


dim = 100

print('carregando os embeddings')

all_embeddings = corpora.load_embeddings(embedding_path_100)

print('carregando os embeddings dos datasets')

path_harem = 'pytorch_lstmcrf/data/harem_seg/full_data.txt'
path_geo = 'pytorch_lstmcrf/data/geocorpus/full_data.txt'
path_lener = 'pytorch_lstmcrf/data/lener_br/full_data.txt'
path_cojur = 'pytorch_lstmcrf/data/cojur/full_data.txt'

path_harem_aproximado = 'pytorch_lstmcrf/data/harem_to_geo_60.txt'
path_target_aproximado = 'pytorch_lstmcrf/data/pc_geocorpus_60.txt'

harem_embeddings = corpora.load_dataset_embeddings(all_embeddings,path_harem_aproximado,60)
target_embeddings = corpora.load_dataset_embeddings(all_embeddings,path_target_aproximado,60)

print('calculando as divergencias')

kl = util.divergencia_KL(harem_embeddings,target_embeddings)
centr = util.centroid_diff(harem_embeddings,target_embeddings,60)

print('KL: ' + str(kl))
print('Centroide: ' + str(centr))