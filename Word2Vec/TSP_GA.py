# -*- encoding: utf-8 -*-
#Construct a POI corpus: disordered POI is composed into ordered data (analogous to a sequence of text data), words are categories of POI, sentences are POI within a block, and text is POI within multiple blocks.
# step1 Using the greedy algorithm, each POI is sorted according to the coordinates of the POI within each block
# step2 Use greedy algorithm to sort each block according to the coordinates of each block
# step3 According to the results of the previous two steps, get the sorted POI of all blocks, 
# and the POI category (word) corresponding to this sequence is the corpus
import numpy as np
import pandas as pd
from DW import Draw

class TSP(object):
    citys = np.array([])
    citys_name = np.array([])#kb store the type of POI
    pop_size = 50
    c_rate = 0.7
    m_rate = 0.05
    pop = np.array([])
    fitness = np.array([])
    city_size = -1
    ga_num = 200
    best_dist = 1
    best_gen = []
    dw = Draw()

    poiCount_A=0
    def __init__(self, c_rate, m_rate, pop_size, ga_num):
        self.fitness = np.zeros(self.pop_size)
        self.c_rate = c_rate
        self.m_rate = m_rate
        self.pop_size = pop_size
        self.ga_num = ga_num

    def init(self):
        tsp = self
        tsp.load_POIs()
        tsp.pop = tsp.creat_pop(tsp.pop_size)
        tsp.fitness = tsp.get_fitness(tsp.pop)#kb Gets the reciprocal of the distance of the pop path
        tsp.dw.bound_x = [np.min(tsp.citys[:, 0]), np.max(tsp.citys[:, 0])]
        tsp.dw.bound_y = [np.min(tsp.citys[:, 1]), np.max(tsp.citys[:, 1])]
        tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)

    # --------------------------------------
    def creat_pop(self, size):
        pop = []
        for i in range(size):
            gene = np.arange(self.citys.shape[0])
            np.random.shuffle(gene)
            pop.append(gene)

        return np.array(pop)

    def get_fitness(self, pop):
        d = np.array([])
        for i in range(pop.shape[0]):
            gen = pop[i]  
            dis = self.gen_distance(gen)#kb Gets the distance of the  path
            dis = self.best_dist / dis#kb Gets the reciprocal of the distance of the  path
            d = np.append(d, dis)  
        return d

    def get_local_fitness(self, gen, i):
        di = 0
        fi = 0
        if i == 0:
            di = self.ct_distance(self.citys[gen[0]], self.citys[gen[-1]])
        else:
            di = self.ct_distance(self.citys[gen[i]], self.citys[gen[i - 1]])
        od = []
        for j in range(self.city_size):
            if i != j:
                od.append(self.ct_distance(self.citys[gen[i]], self.citys[gen[i - 1]]))
        mind = np.min(od)
        fi = di - mind
        return fi

    def EO(self, gen):
        local_fitness = []
        for g in range(self.city_size):
            f = self.get_local_fitness(gen, g)
            local_fitness.append(f)
        max_city_i = np.argmax(local_fitness)
        maxgen = np.copy(gen)
        if 1 < max_city_i < self.city_size - 1:
            for j in range(max_city_i):
                maxgen = np.copy(gen)
                jj = max_city_i
                while jj < self.city_size:
                    gen1 = self.exechange_gen(maxgen, j, jj)
                    d = self.gen_distance(maxgen)
                    d1 = self.gen_distance(gen1)
                    if d > d1:
                        maxgen = gen1[:]
                    jj += 1
        gen = maxgen
        return gen

    # ------------------------------------select population
    def select_pop(self, pop):
        best_f_index = np.argmax(self.fitness)
        av = np.median(self.fitness, axis=0)
        for i in range(self.pop_size):
            if i != best_f_index and self.fitness[i] < av:
                pi = self.cross(pop[best_f_index], pop[i])
                pi = self.mutate(pi)
                pop[i, :] = pi[:]#kb 获取最短路径周期比下边if 判断方式 较长
                # d1 = self.distance(pi)
                # d2 = self.distance(pop[i])
                # if d1 < d2:

                # #kb 会加快获取最短路径，不过多了gen_distance，计算量加大
                # d1=self.gen_distance(pi)
                # d2=self.gen_distance(pop[i])
                # if d1<d2:
                #     pop[i]=pi
                #     pop[i,:]=pi[:]


        return pop

    def select_pop2(self, pop):
        probility = self.fitness / self.fitness.sum()
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=probility)
        n_pop = pop[idx, :]
        return n_pop

    def cross(self, parent1, parent2):
        """交叉"""
        if np.random.rand() > self.c_rate:
            return parent1
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        tempGene = parent2[index1:index2]  # 交叉的基因片段
        newGene = []
        p1len = 0
        for g in parent1:
            if p1len == index1:
                newGene.extend(tempGene)  # 插入基因片段
            if g not in tempGene:
                newGene.append(g)
            p1len += 1
        newGene = np.array(newGene)

        if newGene.shape[0] != self.city_size:
            print('c error')
            return self.creat_pop(1)
            # return parent1
        return newGene

    def mutate(self, gene):
        """突变"""
        if np.random.rand() > self.m_rate:
            return gene
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        newGene = self.reverse_gen(gene, index1, index2)
        if newGene.shape[0] != self.city_size:
            print('m error')
            return self.creat_pop(1)
        return newGene

    def reverse_gen(self, gen, i, j):
        if i >= j:
            return gen
        if j > self.city_size - 1:
            return gen
        parent1 = np.copy(gen)
        tempGene = parent1[i:j]
        newGene = []
        p1len = 0
        for g in parent1:
            if p1len == i:
                newGene.extend(tempGene[::-1])  # 插入基因片段
            if g not in tempGene:
                newGene.append(g)
            p1len += 1
        return np.array(newGene)

    def exechange_gen(self, gen, i, j):
        c = gen[j]
        gen[j] = gen[i]
        gen[i] = c
        return gen

    def evolution(self):
        tsp = self        
        for i in range(self.ga_num):
            best_f_index = np.argmax(tsp.fitness)
            worst_f_index = np.argmin(tsp.fitness)
            local_best_gen = tsp.pop[best_f_index]#kb 局部最优路径：这些点的序列
            local_best_dist = tsp.gen_distance(local_best_gen)
            if i == 0:
                tsp.best_gen = local_best_gen
                tsp.best_dist = tsp.gen_distance(local_best_gen)

            if local_best_dist < tsp.best_dist:
                tsp.best_dist = local_best_dist
                tsp.best_gen = local_best_gen
                # tsp.dw.ax.cla()
                # tsp.re_draw()
                # tsp.dw.plt.pause(0.001)
            else:
                tsp.pop[worst_f_index] = self.best_gen
            print('gen:%d evo,best dist :%s' % (i, self.best_dist))

            tsp.pop = tsp.select_pop(tsp.pop)
            tsp.fitness = tsp.get_fitness(tsp.pop)
            for j in range(self.pop_size):
                r = np.random.randint(0, self.pop_size - 1)
                if j != r:
                    tsp.pop[j] = tsp.cross(tsp.pop[j], tsp.pop[r])
                    tsp.pop[j] = tsp.mutate(tsp.pop[j])
            #self.best_gen = self.EO(self.best_gen)
            tsp.best_dist = tsp.gen_distance(self.best_gen)

    def load_Citys(self, file='南山区第二个商业区POI表.xls', delm=','):
        # 中国34城市经纬度
        #data = pd.read_csv(file, delimiter=delm, header=None).values#kb 3179个城市
        data=pd.read_excel(file)
        height,width= data.shape
        x = np.zeros((height,width))
        for i in range(0,height):
            for j in range(5,width+1): #遍历的实际下标，即excel第一行
                x[i][j-1] =500* data.ix[i,j-1]
        self.citys =x[:,4:]# data[:, 4:]
        #self.citys_name = data[data[:, 0] == '湖北省', 2]
        # self.citys = data[:, 4:]#kb 变成湖南省的135个城市
        # self.citys_name = data[:, 2]
        self.city_size = self.citys.shape[0]

    def load_Citys2(self, file='china.csv', delm=';'):
        # 中国34城市经纬度
        data = pd.read_csv(file, delimiter=delm, header=None).values
        self.citys = data[:, 1:]
        self.citys_name = data[:, 0]
        self.city_size = data.shape[0]

    def load_POIs(self, file='南山区第二个商业区POI表.xls', delm=','):
        RegionPOIInfo=r'InputsOutputs\2_BlockPOIInfo\AllRegionPOIInfo.txt'
        with open(RegionPOIInfo,encoding='utf-8') as f:
            for line in f:#kb 一行是一个block内poi的信息：每个poi的id 子类型 经度(单位米)*100 维度*100 其中不同poi信息用';'隔开，最后是block的id
                line=line.split(';') #kb 分割成列表 ，最后一个元素是block的id并带有\n
                poisSubType,poisCoordLst=[],[]#kb poi的子类型，坐标
                for i in range(len(line)-1):
                    onePoiInfo=line[i].split(',')#kb onePoiInfo=[poi的id, 子类型, 经度*100, 维度*100]
                    poisSubType.append(onePoiInfo[1])
                    poisCoordLst.append([float(onePoiInfo[2])/100,float(onePoiInfo[3])/100])#kb 添加经纬度
                    #self.citys_name.append(onePoiInfo[0])
                self.citys_name=np.array(poisSubType)
                self.citys=np.array(poisCoordLst)
                self.city_size=len(line)-1
                break#kb 先添加一个block的poi

    def gen_distance(self, gen):
        distance = 0.0
        for i in range(-1, len(self.citys) - 1):
            index1, index2 = gen[i], gen[i + 1]
            city1, city2 = self.citys[index1], self.citys[index2]
            distance += np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return distance

    def ct_distance(self, city1, city2):
        d = np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return d

    def draw_citys_way(self, gen):
        '''
        根据一条基因gen绘制一条旅行路线
        :param gen:
        :return:
        '''
        tsp = self
        dw = self.dw
        m = gen.shape[0]
        tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)#x 0:12680947.18  1:12681269.75  y 0:2573332.65  1:2573641.23
        for i in range(m):
            if i < m - 1:
                best_i = tsp.best_gen[i]
                next_best_i = tsp.best_gen[i + 1]
                best_icity = tsp.citys[best_i]
                next_best_icity = tsp.citys[next_best_i]
                dw.draw_arrow(best_icity, next_best_icity)
                #dw.draw_line(best_icity, next_best_icity)
        start = tsp.citys[tsp.best_gen[0]]
        end = tsp.citys[tsp.best_gen[-1]]
        #dw.draw_arrow(end, start)
        dw.draw_line(end, start)

    def draw_citys_name(self, gen, size=5):
        '''
        根据一条基因gen绘制对应城市名称
        :param gen:
        :param size: text size
        :return:
        '''
        tsp = self
        m = gen.shape[0]
        tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)
        for i in range(m):
            c = gen[i]
            best_icity = tsp.citys[c]
            tsp.dw.draw_text(best_icity[0], best_icity[1], tsp.citys_name[c], 10)

    def re_draw(self):
        tsp = self
        tsp.dw.draw_points(tsp.citys[:, 0], tsp.citys[:, 1])
        #tsp.draw_citys_name(tsp.pop[0], 8)
        tsp.draw_citys_way(self.best_gen)

    def EO_SequencedPOISubType(self,EOTxtPath):
        oneBlockSequencedPOISubType,sequencedPOISubType=[],[]
        for i in range(len(self.best_gen)):
            sequence=self.best_gen[i]
            oneBlockSequencedPOISubType+=[self.citys_name[sequence]]
        sequencedPOISubType.append(oneBlockSequencedPOISubType)
        np.savetxt(EOTxtPath, sequencedPOISubType, fmt="%s")
    def get_SequencedPOISubType(self): 
        oneBlockSequencedPOISubType_Str=''
        for i in range(len(self.best_gen)-1):
            sequence=self.best_gen[i]
            oneBlockSequencedPOISubType_Str+=self.citys_name[sequence]+','
        oneBlockSequencedPOISubType_Str+=self.citys_name[len(self.best_gen)-1]
        return oneBlockSequencedPOISubType_Str
def main():
    # # tsp = TSP(0.5, 0.1, 100, 5)#tsp = TSP(0.5, 0.1, 100, 500) c_rate, m_rate, pop_size, ga_num
    # #inputFilePath=r'\RegionPOIInfo.txt'
    t1,t2,t3,t4,t5,t6=0,0,0,0,0,0
    #step1 Sorts the poi within each block and outputs the id of the sorted poi+block within each block---------------------------------------------------
    RegionPOIInfo=r'InputsOutputs\2_BlockPOIInfo\AllRegionPOIInfo.txt' #AllRegionPOIInfo.txt'
    EOTxtPath='YY3GreedySequencedPartRegionPOI.txt'#'AllRegionsequencedPOISubType.txt'
    with open(RegionPOIInfo,encoding='utf-8') as f:
        AllRegionsequencedPOISubType=[]#block的id   输出的经过排序的poi和block
        for line in f:#kb One line is the information of poi within a block: id of each poi subtype longitude (in meters)*100 dimension *100 where different poi information is used with '; 'Separated, and finally the id of the block
            tsp = TSP(0.5, 0.1, 100, 5)#tsp = TSP(0.5, 0.1, 100, 500) c_rate, m_rate, pop_size, ga_num
            line=line.split('_') #kb Split into lists, the last element is the id of the block with \n
            poisSubType,poisCoordLst=[],[]#kb poi的子类型，坐标
            tsp.city_size=len(line)-1            
            for i in range(len(line)-1):
                onePoiInfo=line[i].split(',')#kb onePoiInfo=[poi的id, 子类型, 经度*100, 维度*100]
                if len(onePoiInfo)!=4:
                    tsp.city_size-=1
                    continue
                poisSubType.append(onePoiInfo[1])
                poisCoordLst.append([float(onePoiInfo[2])/100,float(onePoiInfo[3])/100])#kb 添加经纬度
                #self.citys_name.append(onePoiInfo[0])
            if tsp.city_size<3:
                continue
            tsp.citys_name=np.array(poisSubType)
            tsp.citys=np.array(poisCoordLst)
            
            tsp.pop = tsp.creat_pop(tsp.pop_size)
            tsp.fitness = tsp.get_fitness(tsp.pop)#kb 获取该pop个路径的距离的倒数
            tsp.dw.bound_x = [np.min(tsp.citys[:, 0]), np.max(tsp.citys[:, 0])]
            tsp.dw.bound_y = [np.min(tsp.citys[:, 1]), np.max(tsp.citys[:, 1])]
            tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)

            blockID_poiCountA= line[len(line)-1].strip('\n').split(',')#kb poiCount_A_Arr storage [ID of the block, number of POIs in the block book to grid area ratio]
            tsp.poiCount_A=int(blockID_poiCountA[1])
            # if tsp.poiCount_A<40:
            #     tsp.ga_num=500
            # elif tsp.poiCount_A<100:
            #     tsp.ga_num=1000
            # elif tsp.poiCount_A<150:
            #     tsp.ga_num=2000
            # elif tsp.poiCount_A<250:
            #     tsp.ga_num=3000
            # elif tsp.poiCount_A<350:
            #     tsp.ga_num=4000
            # else:
            #     tsp.ga_num=5000

            if tsp.poiCount_A<40:
                if t1<10:                    
                    t1+=1
                    tsp.ga_num=500
                else:
                    continue
            elif tsp.poiCount_A<100:
                if t2<10:
                    t2+=1
                    tsp.ga_num=1000
                else:
                    continue
            elif tsp.poiCount_A<150:
                if t3<10:
                    t3+=1
                    tsp.ga_num=2000
                else:
                    continue
            elif tsp.poiCount_A<250:
                if t4<10:
                    t4+=1
                    tsp.ga_num=3000
                else:
                    continue
            elif tsp.poiCount_A<350:
                if t5<5:
                    t5+=1
                    tsp.ga_num=4000
                else:
                    continue
            else:
                if t6<5:
                    t6+=1
                    tsp.ga_num=5000
                else:
                    continue

            tsp.evolution()
            tsp.re_draw()
            tsp.dw.plt.savefig(r'\fig\testblueline'+str(blockID_poiCountA[0])+'_'+str(tsp.city_size)+'.png')
            AllRegionsequencedPOISubType.append(tsp.get_SequencedPOISubType()+','+blockID_poiCountA[0])##blockID_poiCountA[0]是该block的id
        np.savetxt(EOTxtPath, AllRegionsequencedPOISubType, fmt="%s")


    #step2 sorts the blocks(region)------------------------------------------------------------------------------------------------------  
    RegionPOIInfo=r'InputsOutputs\2_BlockPOIInfo\AllRegionPOIInfo.txt'-----------------
    EOTxtPath='YY3GreedySequencedPartRegion.txt'
    with open(RegionPOIInfo,encoding='utf-8') as f:        
        sequencedRegions,gridsCentroidXY,gridsID=[],[],[]
        for line in f:#kb One line is the information of poi within a block: id of each poi subtype longitude (in meters)*100 dimension *100 where different poi information is used with '; 'Separated, and finally the id of the block, the ratio of the number of poi in the block to the area of the block, the x coordinate of the block centroid, and the y coordinate of the block centroid
            line=line.split('_') #kb Split into lists, the last element is the id of the block with \n
            gridInfo=line[len(line)-1].split(',')#The front part of the line is the poi information, and the last part of the line is the id of the block, the ratio of the number of poi in the block to the area of the block, the block centroid x coordinate, and the grid centroid y coordinate
            LenGridInfo = len(gridInfo)
            gridsCentroidXY.append([int(gridInfo[LenGridInfo-2]),int(gridInfo[LenGridInfo-1])])
            gridsID.append(gridInfo[0])
        tsp.ga_num=10000
        tsp.city_size=len(gridsCentroidXY)
        tsp.citys=np.array(gridsCentroidXY)
        tsp.citys_name=np.array(gridsID)
        tsp.pop = tsp.creat_pop(tsp.pop_size)
        tsp.fitness = tsp.get_fitness(tsp.pop)#kb 获取该pop个路径的距离的倒数
        tsp.dw.bound_x = [np.min(tsp.citys[:, 0]), np.max(tsp.citys[:, 0])]
        tsp.dw.bound_y = [np.min(tsp.citys[:, 1]), np.max(tsp.citys[:, 1])]
        tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)

        tsp.evolution()
        tsp.re_draw()
        tsp.dw.plt.show()
        sequencedRegions=tsp.get_SequencedPOISubType()
        sequencedRegions=sequencedRegions.split(',')
        np.savetxt(EOTxtPath, sequencedRegions, fmt="%s")

    #step3 The POI corpus was constructed according to the sorted regions(block) and the sorted pois in each region--------------------------------------------------------------------
    RegionInfo=r'\InputsOutputs\3_SequencedBlockPOI\Sort POI and Region\YY3GreedySequencedPartRegion.txt'
    AllRegionsequencedPOISubTypeTxt=r'\InputsOutputs\3_SequencedBlockPOI\Sort POI and Region\YY3GreedySequencedPartRegionPOI.txt'
    SequencedRegionAndPOITxt=r'\InputsOutputs\3_SequencedBlockPOI\Sort POI and Region\YY3SequencedRegionAndPOI.txt'
    regionID_POIs={}
    with open(AllRegionsequencedPOISubTypeTxt,encoding='utf-8') as f: 
        for line in f:
            line=line.strip('\n')
            line2=line
            line=line.split(',')
            regionID=line[len(line)-1]
            SequencedPOIs=line2.strip(","+regionID)#Obtain the pois type of the line (region) and remove the region ID
            SequencedPOIs=SequencedPOIs.replace(',',' ')
            regionID_POIs[regionID]=SequencedPOIs
    SequencedRegionAndPOI=[]
    with open(RegionInfo,encoding='utf-8') as f: 
        for line in f:
            line=line.strip('\n') 
            if regionID_POIs.__contains__(line):
                SequencedPOIs=regionID_POIs[line]           
                SequencedRegionAndPOI.append(SequencedPOIs)#Adds the sorted POIs of the region
    np.savetxt(SequencedRegionAndPOITxt,SequencedRegionAndPOI,fmt="%s")
        
if __name__ == '__main__':
    main()
