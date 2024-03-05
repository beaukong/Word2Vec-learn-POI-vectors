# -*- coding: utf-8 -*-
#Visualization the learned vector
from matplotlib import pylab as plt
import pickle
from matplotlib.backends.backend_pdf import PdfPages


def plot(embeddings, labels, save_to_pdf='embed.pdf'):
	assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
	pp = PdfPages(save_to_pdf)
	plt.figure(figsize=(15,15))  # in inches
	for i, label in enumerate(labels):
		x, y = embeddings[i,:]
		plt.scatter(x, y)
		plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
						ha='right', va='bottom')
	
	plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签  kb
	plt.rcParams['axes.unicode_minus']=False #用来正常显示负号  kb
	plt.savefig(pp, format='pdf')
	#plt.show()
	plt.close('all')
	pp.close()

embedding_size_Lst=[200]#[50,100,150,200,250,300]#poi向量 的维度
vocabulary_size_Lst=[400]#[400,500,600,700,800]#poi类型数
for embedding_size_index in range(len(embedding_size_Lst)):
	embedding_size=embedding_size_Lst[embedding_size_index]
	for vocabulary_size_index in range(len(vocabulary_size_Lst)):
		vocabulary_size=vocabulary_size_Lst[vocabulary_size_index]

		method = 'cbow' 
		filename=r'\InputsOutputs\4_Word2VecPOIVec\Sort POI and Region'+'\\'+str(vocabulary_size)+'classes of poi'+'\\'+str(embedding_size)+'dimensions\\'+'2d_embedding_%s.pkl' % method  
		with open(filename, 'rb') as f:
			[two_d_embeddings, two_d_embeddings_2, reverse_dictionary] = pickle.load(f)

		num_points = len(two_d_embeddings)
		words = [reverse_dictionary[i] for i in range(1, num_points+1)]
		pdf1=r'\InputsOutputs\4_Word2VecPOIVec\Sort POI and Region'+'\\'+str(vocabulary_size)+'classes of poi'+'\\'+str(embedding_size)+'dimensions\\'+'two_d_embeddings_%s_NanShan_FuTian_PingShan_GuangMing_0819.pdf' % method
		pdf2=r'\InputsOutputs\4_Word2VecPOIVec\Sort POI and Region'+'\\'+str(vocabulary_size)+'classes of poi'+'\\'+str(embedding_size)+'dimensions\\'+'two_d_embeddings_2_%s_NanShan_FuTian_PingShan_GuangMing_0819.pdf' % method
		plot(two_d_embeddings, words, save_to_pdf=pdf1)
		plot(two_d_embeddings_2, words, save_to_pdf=pdf2)
