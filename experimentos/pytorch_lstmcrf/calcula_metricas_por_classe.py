#esse script recebe como entrada dois arquivos: O arquivo dos resultados
#e o arquivo de teste com as classes (nao apenas segmentando as entidades)

def read_file(file):
	l = []
	for line in file.readlines():
		if line != '\n':
			l.append(line)
	return l

def init_mapp(classe,mapp):
	mapp[classe] = {
		'Acertos':[0,0,0],
		'Total':0,
		'FN':[0,0,0]	
	}


f_resultados = open('data/harem_seg/results_harem_100.txt','r')
f_classes = open('data/harem_seg/test_com_classes.txt')

lines_resultados = read_file(f_resultados)
lines_classes = read_file(f_classes)

mapp = {}

for i in range(len(lines_resultados)):

	line_resultado = lines_resultados[i]
	line_classes = lines_classes[i]

	tag_result = line_resultado.split()[1]
	tag_teste = line_classes.split()[1]

	classe_atual = ''
	if tag_result != 'O':
		#tirando o '-'
		tag_result = line_resultado.split()[1][:-1]

	if tag_teste != 'O':
		tag_teste = line_classes.split()[1].split('-')[0]
		classe_atual = line_classes.split()[1].split('-')[1]
	else:
		classe_atual = 'FORA'
########################################

	if classe_atual not in mapp:
		init_mapp(classe_atual,mapp)

	mapp[classe_atual]['Total'] += 1

	if tag_result == tag_teste:
		if tag_teste == 'B':
			mapp[classe_atual]['Acertos'][0] += 1
		elif tag_teste == 'I':
			mapp[classe_atual]['Acertos'][1] += 1
		elif tag_teste == 'O':
			mapp[classe_atual]['Acertos'][2] += 1
	else:
		if tag_result == 'B':
			mapp[classe_atual]['FN'][0] += 1
		elif tag_result == 'I':
			mapp[classe_atual]['FN'][1] += 1
		elif tag_result == 'O':
			mapp[classe_atual]['FN'][2] += 1

for classe in mapp:
	print('classe: '+ classe)
	total = mapp[classe]['Total']
	total_acertos = sum(mapp[classe]['Acertos'])
	precisao = total_acertos/total
	total_fn = sum(mapp[classe]['FN'])
	recall = total_acertos/(total_acertos+total_fn)
	f_measure = (2*precisao*recall)/(precisao+recall)
	print('precisao: ' + str(round(precisao,2)))
	print('recall: ' + str(round(recall,2)))
	print('f-measure: ' + str(round(f_measure,2)))
	print()