import os
from bs4 import BeautifulSoup
import string
import nltk
import math
from scipy import spatial
from scipy import stats
import numpy as np
import ast
from nltk.tokenize.treebank import TreebankWordDetokenizer
from config import Dados

path_corpora = Dados.path_corpora
path_embeddings = Dados.path_embeddings


class Corpora:

	# retorna o corpus harem no formato BIO. o formato eh
	# uma palavra por linha, com sua tag separada por um
	# espaco. Sentencas distintas estao separadas por uma
	# quebra de linha

	def get_BIO_harem(self,com_classe = False):

		util = Util()
		
		corpus_path_harem_tagged = path_corpora['harem']
		corpus_path_harem_docs = path_corpora['harem_docs']

		f_harem_tagged = open(corpus_path_harem_tagged,encoding = 'ISO-8859-1')
		f_harem_docs = open(corpus_path_harem_docs,encoding = 'ISO-8859-1')

		xt_harem_tagged =  BeautifulSoup(f_harem_tagged,'lxml')
		xt_harem_docs =  BeautifulSoup(f_harem_docs,'lxml')

		#documentos anotados
		docs_tagged = xt_harem_tagged.find_all('doc')
		doc_ids = [doc.get('docid') for doc in docs_tagged]
		all_docs = xt_harem_docs.find_all('doc')

		docs_non_tagged = []

		#get the non anotated doc version
		for doc_id in doc_ids:
			for doc in all_docs:
				if doc.get('docid') == doc_id:
					docs_non_tagged.append(doc)
		
		#paragraphs in both versions, tagged and non_tagged
		p_tagged = []
		p_non_tagged = []

		for i in range(len(docs_tagged)):

			p_tagged_curr = docs_tagged[i].find_all('p')
			p_non_tagged_curr = docs_non_tagged[i].find_all('p')

			#essa condicao exclui um dos 129 documentos do harem
			#por algum motivo o numero de paragrafos nao eh o mesmo
			#entao prefiro nao incluir do q deixar passar alguma
			#inconsistencia
			if len(p_tagged_curr) == len(p_non_tagged_curr):
				p_tagged += p_tagged_curr
				p_non_tagged = p_non_tagged + p_non_tagged_curr

		BIO_corpus = []

		for i in range(len(p_tagged)):

			list_em_text = [str(em.text) for em in p_tagged[i].find_all('em')]
			list_em_classes = [str(em.get('categ')) for em in p_tagged[i].find_all('em')]

			clean_text = str(p_non_tagged[i].text)
			#substring matching
			for i,em_text in enumerate(list_em_text):
				clean_text = util.substring_marking(em_text,clean_text,list_em_classes[i],com_classe)

			words_text = nltk.word_tokenize(clean_text,language='portuguese')
			for cont,word in enumerate(words_text):
				if len(word.split('-')) == 1:
					words_text[cont] = words_text[cont] + '-O'

			marked_text = TreebankWordDetokenizer().detokenize(words_text)
			BIO_corpus.append(marked_text)

		return BIO_corpus

	#recebe corpus no formato BIO e retorna
	#dict com formato palavra -> cont_palavra
	def conta_palavras(self,corpus_path):
		try:
			f = open(corpus_path,'r')
		except:
			print('nao foi possivel abrir o arquivo')		
			return
		cont_palavras = {}
		lines = f.readlines()
		for line in lines:
			palavra = line.split(' ')[0]
			if palavra not in cont_palavras:
				cont_palavras[palavra.lower()] = 1
			else:
				cont_palavras[palavra.lower()] += 1
		return cont_palavras

	def load_embeddings(self,embedding_path):
		f = open(embedding_path,'r')
		embedding_dict = {}
		for i,line in enumerate(f.readlines()):
			#ignore the first line
			if i != 0:
				split_line = line.split()
				word = split_line[0]
				embedding = split_line[1:]
				if word not in embedding_dict:
					try:
						embedding_dict[word.lower()] = np.array(embedding).astype(np.float)
					except:
						#one line on the file have a character invading the embedding
						embedding = embedding[1:]
						embedding_dict[word.lower()] = np.array(embedding).astype(np.float)

		return embedding_dict

	def load_dataset_embeddings(self,embeddings_dict,corpus_path,embedding_dim):
		corpus_embedding_dict = {}
		corpus_dict = self.conta_palavras(corpus_path)
		for palavra in corpus_dict:
			if palavra in embeddings_dict:
				corpus_embedding_dict[palavra] = embeddings_dict[palavra]
			else:
				corpus_embedding_dict[palavra] = np.random.rand(embedding_dim)
		return corpus_embedding_dict


class Util:

	#marca a substring da entidade nomeada no texto
	def substring_marking(self,subs,string,classe,com_classe):

		words_subs = nltk.word_tokenize(subs,language='portuguese')
		words_string = nltk.word_tokenize(string,language='portuguese')

		pos_subs = 0

		for cont,word_string in enumerate(words_string):

			if pos_subs == len(words_subs):

				pos_ini = cont-len(words_subs)
				off_set = len(words_subs)
				for j in range(pos_ini,pos_ini+off_set):
					if j == pos_ini:
						if com_classe:
							classe = classe.split('|')[0]
							words_string[j] = words_string[j] + '-B' + '-' + str(classe)
						else:
							words_string[j] = words_string[j] + '-B' 
					else:
						if com_classe:
							classe = classe.split('|')[0]
							words_string[j] = words_string[j] + '-I' + '-' + str(classe)
						else:
							words_string[j] = words_string[j] + '-I'

				pos_subs = 0

			if word_string == words_subs[pos_subs]:
				pos_subs = pos_subs + 1
			else:
				pos_subs = 0

		return TreebankWordDetokenizer().detokenize(words_string)


	#particiona o corpus em conjunto de treino e conjunto de testes. recebe o 
	#tamanho do conjunto de testes e uma lista, e retorna o conjunto de treino
	#e de teste de acordo com o parametro size_of_test, que eh uma proporcao eg: 0.25
	def particiona(self,list_en, size_of_test):
		size_of_corpus = len(list_en)
		range_of_test = int(size_of_test*size_of_corpus)
		test = list_en[:range_of_test]
		train = list_en[range_of_test:]
		return train,test

	def divergencia_KL(self,dict_s,dict_t):

		p_distribution_target,p_distribution_source = [],[]

		list_palavra_source = [palavra for palavra in dict_s]
		list_palavra_target = [palavra for palavra in dict_t]

		list_emb_source = [dict_s[palavra] for palavra in dict_s]
		list_emb_target = [dict_t[palavra] for palavra in dict_t]

		union = [palavra for palavra in dict_s]
		for palavra in dict_t:
			if palavra not in union:
				union.append(palavra)

		emb_union = []
		for palavra in union:
			if palavra in dict_s:
				emb_union.append(dict_s[palavra])
			else:
				emb_union.append(dict_t[palavra])

		epslon = 0.9
		num_sample = 100

		print(len(emb_union))

		for i,emb in enumerate(emb_union):

			cont_s,cont_t = 0,0

			amostra_s = np.random.choice(list_palavra_source,num_sample)
			amostra_t = np.random.choice(list_palavra_target,num_sample)

			for j in range(num_sample):
				if spatial.distance.cosine(emb,dict_s[amostra_s[j]]) <= epslon:
					cont_s += 1
				if spatial.distance.cosine(emb,dict_t[amostra_t[j]]) <= epslon:
					cont_t += 1

			p_source = cont_s/num_sample
			p_target = cont_t/num_sample

			print(p_source)
			print(p_target)

			p_distribution_source.append(p_source)
			p_distribution_target.append(p_target)

			if i % 100 == 0:
				print(str(round(i/len(emb_union)*100)) + ' % ' + 'completo')
				# print(spatial.distance.cosine(emb,dict_s[amostra_s[j]]))


		return stats.entropy(p_distribution_target,p_distribution_source)

	def calculate_mean(self,dict_emb,emb_dim):
		X = [dict_emb[palavra] for palavra in dict_emb if len(dict_emb[palavra]) == emb_dim]
		return np.mean(X,axis=0)

	def centroid_diff(self,dict_s,dict_t,emb_dim):
		mean_s = self.calculate_mean(dict_s,emb_dim)
		mean_t = self.calculate_mean(dict_t,emb_dim)
		return spatial.distance.euclidean(mean_s,mean_t)

	def lexical_triangular_diff(self,dict_s,dict_t):

		p_c1,p_c2,p_comum = 0,0,0

		larger,smaller = None,None

		if len(dict_s) >= len(dict_t):
			larger = dict_s
			smaller = dict_t
		else:
			larger = dict_t
			smaller = dict_s

		for palavra in larger:
			if palavra in smaller:
				p_comum += 1

		p_c1 = len(dict_s) - p_comum
		p_c2 = len(dict_t) - p_comum

		return (p_c1 + p_c2) - 2*p_comum