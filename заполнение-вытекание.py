# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:03:37 2017

@author: Аня
"""

import numpy as np
from random import uniform, choice
import time
import math
from scipy import spatial,sparse
from itertools import compress, chain
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
sns.set()

# =============================================================================
# Задаем параметры системы
# =============================================================================
L = 150.0 # Линейный размер куба
Rmean = 3.0 # Средний радиус поры
sigma = 0.5 # 
N = math.ceil(L/(2.0*(Rmean+3.5*sigma))) # Число ячеек в кубе
DSIGMA = 13/1000 # поверхностная энергия жидкости
SIGMA = 72/1000 # энергия границы раздела жидкость-поверхность пористого тела
T = 293 # температура

# =============================================================================
# Функция для определения количества соседних ячеек
# =============================================================================
def Where(g):
    t = [1 for i in g if (i == 0.0) or (i ==  L*(N-1)/N)]
    if len(t)==3:
        return 8
    elif len(t)==2:
        return 12
    elif len(t)==1:
        return 18
    else:
        return 27
    
# =============================================================================
# Функция объединения кластреров при их конфликте
# =============================================================================     
def Merging(lst):
    while True:
        merged_one = False
        sets = [lst[0]]
    
        for s in lst[1:]:
            in_super_set = False
            for st in sets:
                if s & st:
                    st |= s
                    merged_one = True
                    in_super_set = True
                    break
        
            if not in_super_set:
                sets.append(s)
            
        if not merged_one:
            break
    
        lst = sets[:]
    return lst

# =============================================================================
# Предствавим пространство пор в качесте совокупности пересекающихся сфер.
# Создадим класс сфер.
# =============================================================================
    
class Sphere:
    def __init__(self, r):
        self.center = [uniform(0.0, L), uniform(0.0, L), uniform(0.0, L)]
        self.cen = [L/2, L/2, L/2]
        self.rad = r
        self.mark = 0
        self.neighbors = set()
        self.AREA_s = []
        self.AREA_o = []
        self.mark_f = 0
        self.graf = []
    
    def __str__(self):
        return "{}, {}, {}".format(self.center, self.rad, self.mark)
    
    def __repr__(self):
        return "{}, {}, {}".format(self.center, self.rad, self.mark)
# =============================================================================
#     Функция определяющая пересечение сфер и их пересечения с гранями куба.
# =============================================================================
    def IsOverlap(self, other):
        try:
            return ((self.center[0] - other.center[0])*(self.center[0] - other.center[0]) + 
                    (self.center[1] - other.center[1])*(self.center[1] - other.center[1]) +
                    (self.center[2] - other.center[2])*(self.center[2] - other.center[2]))**(0.5) <= (self.rad + other.rad) 
        except AttributeError:
            if len(other)==2:
                return np.abs((self.center[other[1]] - other[0])) <= self.rad
            elif len(other)==3:
                return ((self.center[0] - other[0])*(self.center[0] - other[0]) + 
                    (self.center[1] - other[1])*(self.center[1] - other[1]) +
                    (self.center[2] - other[2])*(self.center[2] - other[2]))**(0.5) <= self.rad
            else:
                raise NameError("Либо пересечение с гранями, либо со сферой")
                
# =============================================================================
#      Функия определяющая номер ячейки в которую попадает сфера.           
# =============================================================================
    def Is_inside(self):
        return math.ceil(N*self.center[2]/L - 1) + N*math.ceil(N*self.center[0]/L - 1) + N*N*math.ceil(N*self.center[1]/L - 1)
    
# =============================================================================
#     Функция для генерации сфер, условие непересечения ядер.
# =============================================================================
    def Gen_Sph(self,other):
        k = ((self.center[0] - other.center[0])*(self.center[0] - other.center[0]) + 
            (self.center[1] - other.center[1])*(self.center[1] - other.center[1]) +
            (self.center[2] - other.center[2])*(self.center[2] - other.center[2]))**(0.5)
        return k <= (self.rad * 0.5 + other.rad) or k <= (other.rad * 0.5 + self.rad)
    
# =============================================================================
#     Функция для определения расстояния между центрами сфер.
# =============================================================================
    def dist(self, other):
        k = ((self.cen[0] - other.center[0])*(self.cen[0] - other.center[0]) + 
            (self.cen[1] - other.center[1])*(self.cen[1] - other.center[1]) +
            (self.cen[2] - other.center[2])*(self.cen[2] - other.center[2]))**(0.5)
        return k
    
# =============================================================================
#     Функция для определения площади поверхности мениска выбранной сферы            
# =============================================================================
    def area_meniscus_s(self,other):
        k = ((self.center[0] - other.center[0])*(self.center[0] - other.center[0]) + 
            (self.center[1] - other.center[1])*(self.center[1] - other.center[1]) +
            (self.center[2] - other.center[2])*(self.center[2] - other.center[2]))**(0.5)
        return 2.0 * math.pi * self.rad * (other.rad * other.rad - (k-self.rad)*(k-self.rad))/(2.0*k)
    
# =============================================================================
#    Функция для определения площади поверхности мениска сферы-соседа     
# =============================================================================
    def area_meniscus_o(self,other):
        k = ((self.center[0] - other.center[0])*(self.center[0] - other.center[0]) + 
            (self.center[1] - other.center[1])*(self.center[1] - other.center[1]) +
            (self.center[2] - other.center[2])*(self.center[2] - other.center[2]))**(0.5)
        return 2.0 * math.pi * other.rad * (self.rad * self.rad - (k-other.rad)*(k-other.rad))/(2.0*k)
    
# =============================================================================
#     Работа, необходимая для вытекания жидкости из поры.
# =============================================================================  
    def work(self,p):
        SM = np.sum([self.area_meniscus_s(spheres[elm]) for elm in 
                     self.neighbors if not spheres[elm].mark] ) #пустые мениски
        SMM = np.sum([self.area_meniscus_s(spheres[elm]) for ind, elm 
                      in enumerate(self.neighbors) ] ) # все мениски
        SMZ = np.sum([ self.area_meniscus_o(spheres[elm]) for elm in 
                      self.neighbors if spheres[elm].mark ] ) #заполненные мениски
        work = (p * 4.0 * 10**(-9) * math.pi * self.rad * self.rad * self.rad / 3.0 
                - (4.0 * math.pi * self.rad * self.rad - SMM) * DSIGMA 
                + SIGMA * (SMZ - SM) )
        if work < 0.0:
            return 1 
        else:
            return 0
            
# =============================================================================
#     Работа, необхоодимая для заполнения поры жидкостью.            
# =============================================================================
    def work_f(self,p):
        SM = np.sum([self.area_meniscus_s(spheres[elm]) for elm in 
                     self.neighbors if not spheres[elm].mark] ) 
        SMM = np.sum([self.area_meniscus_s(spheres[elm]) for ind, elm 
                      in enumerate(self.neighbors)] ) 
        SMZ = np.sum([ self.area_meniscus_o(spheres[elm]) for elm 
                      in self.neighbors if spheres[elm].mark ] )
        work = (- p * 4.0 * 10**(-9) * math.pi * self.rad * self.rad * self.rad / 3.0 
            + (4.0 * math.pi * self.rad * self.rad - SMM) * DSIGMA 
            + SIGMA * (SM - SMZ) )
        if work < 0.0:
            return 1 
        else:
            return 0
 
# =============================================================================
#     Площадь поверхности мениска.           
# =============================================================================
    def Segment_height(self,other):
        h = self.rad - abs((self.center[other[1]] - other[0]))
        return  math.pi * h * h *(self.rad - h / 3.0)
     
# =============================================================================
#     Расстояние между центрами сфер.
# =============================================================================
    def rast(self,other):
        return ((self.center[0] - other.center[0])*(self.center[0] - other.center[0]) + 
                (self.center[1] - other.center[1])*(self.center[1] - other.center[1]) +
                (self.center[2] - other.center[2])*(self.center[2] - other.center[2]))**(0.5)

# =============================================================================
# Функия для расчета пористости методом Монте-Карло.        
# =============================================================================
def Get_Porosity_Number(size,l):
    P = [(uniform(0,l), uniform(0,l), uniform(0,l)) for k in range(size)]
    inside = 0
    for point in P:
        wh = math.ceil(N*point[2]/L - 1) + N*math.ceil(N*point[0]/L - 1) + N*N*math.ceil(N*point[1]/L - 1)
        a = []
        for j in neigh[wh]:
            a += new_cell[j]       
        for k in a:
            if spheres[k].IsOverlap(point):
                inside += 1
                break            
    return inside / size
     
# =============================================================================
# Функция для формирования кластеров 
# =============================================================================
def FormClustersN3(input_sph):
    clust = [{0}]
    for ind1,elem1 in enumerate(input_sph):
        it_cls = [{ind1}] 
        a = [] 
        for j in neigh[sph_cell[ind1]]:
            a += new_cell[j]
        for ind2 in a:
            if ind1 != ind2 and elem1.IsOverlap(spheres[ind2]):
                it_cls[0].add(ind2)
                elem1.neighbors.add(ind2)
                elem1.AREA_s.append(elem1.area_meniscus_s(spheres[ind2]))
                elem1.AREA_o.append(elem1.area_meniscus_o(spheres[ind2]))
                elem1.graf.append((ind1, ind2, elem1.rast(spheres[ind2])))
        clust.extend(it_cls)
    clust = Merging(clust)       
    return clust    
    
    
def FormClustersN4(input_sph):
    clust = [{0}]
    for elem1 in input_sph:
        it_cls = [{elem1}] 
        a = [] 
        for j in neigh[sph_cell[elem1]]:
            a += new_cell[j]
        for ind2 in a:
            if elem1 != ind2 and spheres[elem1].IsOverlap(spheres[ind2]):
                it_cls[0].add(ind2)
        clust.extend(it_cls)
    clust = Merging(clust)  
    return clust    

 
sphh = [18000] 
spheres = []
sph_cell=[]
obj = Sphere(1.0)


num_n = np.linspace(0.0, L*(N-1)/N, N)
cell = np.vstack(np.meshgrid(num_n, num_n, num_n)).reshape(3,-1).T
tree = spatial.cKDTree(cell)
neigh = [ tree.query(p, k=Where(p))[1] for p in cell]
new_cell = [[] for i in range(len(neigh))]
porositys = []
sosedi = []
# =============================================================================
# Цикл генерации сфер
# =============================================================================
for s in sphh: 
    degree_filling = []       
    start2 = time.clock()
    while len(spheres) < s:
        test_sphere = Sphere(np.random.normal(Rmean, sigma))
        a = [] 
        for j in neigh[test_sphere.Is_inside()]:
            a += new_cell[j]
        if any([test_sphere.Gen_Sph(spheres[ind2]) for ind2 in a]):
            continue
        spheres.append(test_sphere)
        sph_cell.append(test_sphere.Is_inside())
        new_cell[test_sphere.Is_inside()].append(len(spheres)-1)
    print('test_sphere: {}'.format(time.clock() - start2))
   

# =============================================================================
# Формирование кластеров    
# =============================================================================
    start = time.clock()
    dd = FormClustersN3(spheres)
    print('FormClustersN3: {}'.format(time.clock() - start))
    

   
    start4 = time.clock()    
    grafs = []    
    for j in dd:
        G = nx.Graph()
        for elm in j:
            G.add_weighted_edges_from(spheres[elm].graf)
        grafs.append(G)
    print('FormGrafs: {}'.format(time.clock() - start4))
    
    vol = []
    start3 = time.clock()
    lower_bound = [ set(), set(), set() ]
    upper_bound = [ set(), set(), set() ]
    lower_x = [set()]
    upper_x = [set()]

    pressure1 = [x for x in range(36000000, -280000, -280000)]
    volume_seg = 0.0
    for ind,elem in enumerate(spheres):
        for i in range(3):
            if elem.IsOverlap([0.0, i]):
                lower_bound[i].add(ind)
                volume_seg += elem.Segment_height([0.0, i])
            if elem.IsOverlap([0.0, 0]):
                lower_x[0].add(ind)
                
            if elem.IsOverlap([L, i]):
                upper_bound[i].add(ind)
                volume_seg += elem.Segment_height([L, i])
            if elem.IsOverlap([L, 0]):
                upper_x[0].add(ind)    
    
    big_cls = tuple((item[0],len(item[1])) for item in enumerate(dd) if len(item[1]) > 1)
    large_cls = [big_cls[i][0] for i in range(len(big_cls)) if big_cls[i][1] == np.amax(big_cls, axis=0)[1]]
    large_cls_l = [big_cls[i][1] for i in range(len(big_cls)) if big_cls[i][1] == np.amax(big_cls, axis=1)[0]]
    volume_filled = 0
  
    pressure = [x for x in range(0, 36100000, 100000)]
    
    bound_bottom = set(chain.from_iterable(lower_bound))
    bound_top = set(chain.from_iterable(upper_bound))
    bound = bound_bottom.union(bound_top)
    
    bound_bottom_x = set(chain.from_iterable(lower_x))
    bound_top_x = set(chain.from_iterable(upper_x))
    bound_x = bound_bottom_x.union(bound_top_x)
    
    control = 0
    PR = False
    FIL = []
    radii_0 = [spheres[i].rad for i in range(s)]
    n = 0
    pp = []
    
# =============================================================================
#     Цикл по заполнению сфер
# =============================================================================
    for pr in pressure: 
        filling = set(elem for elem in bound if spheres[elem].work_f(pr) and not spheres[elem].mark)
        
        if len(filling) == 0:
            vol.append(volume_filled)
            degree_filling.append(n / s)
            continue
        
        for elm in filling:
            spheres[elm].mark = 1
            volume_filled += 4.0 * math.pi * spheres[elm].rad**3.0 / 3.0 
            n += 1
        
        while True:
            filling = set(element for j in dd for element in j if any([spheres[el].mark for el in             spheres[element].neighbors]) and not spheres[element].mark and spheres[element].work_f(pr))
            
            for elmm in filling:
                spheres[elmm].mark = 1
                volume_filled += 4.0 * math.pi * spheres[elmm].rad**3.0 / 3.0 
                n += 1
                          
            if len(filling) == 0:
                break 
            
        
        vol.append(volume_filled)
        radii_new = [spheres[i].rad for i in range(s) if spheres[i].mark]
        degree_filling.append(n / s)

        
        if radii_new:
            plt.hist(radii_0, alpha = 0.5)
            plt.hist(radii_new)
            plt.xlabel(u'Радиус поры')
            plt.legend((u'- распределение радиусов', u'- распределение заполненных радиусов'),  frameon=False,loc=2)
            plt.savefig('{}.pdf'.format(pr/101325))
            plt.show()
        
       
    plt.plot(pressure, degree_filling, 'o')
    plt.xlabel(u'Давление')
    plt.ylabel(u'Степень заполнения')
    plt.savefig('{}.pdf'.format('Степень заполнения от давления'))
    plt.show()
    
    print('заполнение: {}'.format(time.clock() - start3))

    volume_filled1 = volume_filled
    mx = volume_filled
    vol1 = []
    con = control
    start5 = time.clock()
    
    SPH = set(i for i in range(s) if spheres[i].mark)
    dd_new = FormClustersN4(SPH)
        
    grafs = []    
    for j in dd_new:
        G = nx.Graph()
        for elm in j:
            G.add_weighted_edges_from(spheres[elm].graf)
        grafs.append(G)
        
    rst = [1]    
    rs = 0
    
# =============================================================================
#     Цикл по вытеканию жидкости.
# =============================================================================
    for r in rst:
        op = 0
        for pr in pressure1:
            SP = set(i for i in range(s) if spheres[i].mark )
            SPH = set(i for i in range(s) if spheres[i].mark and obj.dist(spheres[i]) <= rs)
            dd_new = FormClustersN4(SP)
            bound_bottom_new = SP.intersection(bound_bottom)
            bound_top_new = SP.intersection(bound_top)
            bound = bound_bottom_new.union(bound_top_new) 
            outflow = [elem for elem in SPH if spheres[elem].work(pr)] 
            ln = len(outflow)
            ln1 = s 
        
            while ln != ln1: 
                distanse = []
                SP = set(i for i in range(s) if spheres[i].mark )
                SPH = set(i for i in range(s) if spheres[i].mark and obj.dist(spheres[i]) <= rs)
                dd_new = FormClustersN4(SP)
                bound_bottom_new = SP.intersection(bound_bottom)
                bound_top_new = SP.intersection(bound_top)
                bound = bound_bottom_new.union(bound_top_new)
                outflow = [elem for elem in SPH if spheres[elem].work(pr)] 
                ln = len(outflow)
                       
                if len(outflow) == 0:
                    break
            
                kk = 0    
                for ii in outflow:
                    for ind, el in enumerate(dd_new):
                        if not {ii}.isdisjoint(el):
                            number = ind
                            break
                
                    bound_new = bound.intersection(dd_new[number])
                
                   
                    if bound_new:
                        distanse.append((obj.dist(spheres[ii]), ii))
                        
                    
                if distanse:
                    distanse.sort()
                    spheres[distanse[0][1]].mark = 0
                    volume_filled1 -= 4.0 * math.pi * (spheres[distanse[0][1]].rad)**3.0 / 3.0 
                    con += 1
                    kk += 1
                    print(con,distanse[0][0])
                
                ln1 = len(outflow) - kk    
            
          
                if not bound:
                    break

            rs += 1
            vol1.append(volume_filled1)
            
    print('вытекание: {}'.format(time.clock() - start5))
        
vol00 = [x/mx for x in vol]
vol111 = [x/mx for x in vol1]
pressure = [x/101325 for x in pressure]
pressure1 = [x/101325 for x in pressure1]

a = 'вытек центр'

plt.plot(vol00, pressure, 'o')
plt.plot(vol111, pressure1,'o')
plt.xlabel(u'Относительный объем')
plt.ylabel(u'Давление, атм')
plt.savefig('{}.pdf'.format(a))
plt.show()

# =============================================================================
# Функция для отрисовки пересекающихся сфер.
# =============================================================================
def plot_implicit(fn, bbox=(0.0,L)):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 70) 
    B = np.linspace(xmin, xmax, 70) 
    A1,A2 = np.meshgrid(A,A) 

    for z in B: 
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z', cmap='RdGy', alpha = 0.1)


    for y in B:
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y',  cmap='RdGy', alpha = 0.1)

    for x in B: 
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x',  cmap='RdGy', alpha = 0.1)

    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)
    plt.savefig('{}.pdf'.format('упаковка'))
    plt.show()
    
def sph(x,y,z,rd):
    return x**2 + y**2 + z**2 - rd**2

def translate(fn,x,y,z,rd):
    return lambda a,b,c: fn(x-a,y-b,z-c,rd)
    
def union(*fns):
    return lambda x,y,z: np.min([fn(x,y,z) for fn in fns], 0)


SP = set(i for i in range(s) if spheres[i].mark )    
mas = []   
for k in SP:
    mas.append(translate(sph, spheres[k].center[0], spheres[k].center[1], spheres[k].center[2], spheres[k].rad))
plot_implicit(union(*mas), bbox=(0.0,L))
plot_implicit(union(*mas), bbox=(0.0,L))