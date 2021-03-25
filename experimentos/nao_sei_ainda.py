import utilidades
import nltk
util = utilidades.Corpora()

def write_to_file_harem(f,lista):

	for text in lista:
		tokenized_text = nltk.word_tokenize(text,language = 'portuguese')
		for token in tokenized_text:
			#print(token)
			split = token.split('-')
			if len(split) > 1:
				#se eu marquei com alguma coisa estranha garanta que a tag
				#sera 'O'
				if split[1] != 'B' and split[1] != 'I' and split[1] != 'O':
					split[1] = 'O'

				if split[1] == 'O':
					f.write(split[0] + ' ' + split[1] + '\n')
				else:
					f.write(split[0] + ' ' + split[1] + '-' + '\n')
			else:
				f.write(split[0] + ' ' + 'O')
		f.write('\n')



f_train = open('pytorch_lstmcrf/data/lener_br/train.txt','w')
f_test = open('pytorch_lstmcrf/data/lener_br/test.txt','w')
f_dev = open('pytorch_lstmcrf/data/lener_br/dev.txt','w')

canivete_suico = utilidades.Util()
particiona = canivete_suico.particiona
# harem = util.get_BIO_harem()

# harem_aux,harem_teste = particiona(harem,0.2)
# harem_treino,harem_dev = particiona(harem_aux,0.2)

f = open('/home/victor/TCC/experimentos/corpora/LeNER-Br/lener_br.txt','r')


lines = list(f.readlines())
geocorpus_aux,geocorpus_teste = particiona(lines,0.2)
geocorpus_treino,geocorpus_dev = particiona(geocorpus_aux,0.2)

for i in geocorpus_treino:
	f_train.write(i)
for i in geocorpus_teste:
	f_test.write(i)
for i in geocorpus_dev:
	f_dev.write(i)


# write_to_file(f_train,harem_treino)
# write_to_file(f_test,harem_teste)
# write_to_file(f_dev,harem_dev)