import os
import time
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import matplotlib.image as mpimg
import scipy.cluster.hierarchy as sch
from scipy.stats import scoreatpercentile 
import sys
sys.path.insert(0, "/data/walker_Lab/miniconda3/envs/sca/lib/python3.6/site-packages/pysca")
from pysca import scaTools as sca
import pickle as pickle
from optparse import OptionParser

Dseq = list(); Dsca = list(); Dsect = list()
db2 = pickle.load(open('./PF00186_35.db', 'rb'))
Dseq.append(db2['sequence'])
Dsca.append(db2['sca'])
Dsect.append(db2['sector'])
N_alg = 1
AlgName = ['Manual']

ix = 1
plt.rcParams['figure.figsize'] = 9, 15
for k in range(N_alg):
    # List all elements above the diagonal (i<j):
    listS = [Dsca[k]['simMat'][i,j] for i in range(Dsca[k]['simMat'].shape[0]) \
             for j in range(i+1, Dsca[k]['simMat'].shape[1])]
    
    # Cluster the sequence similarity matrix
    Z = sch.linkage(Dsca[k]['simMat'],method = 'complete', metric = 'cityblock')
    R = sch.dendrogram(Z,no_plot = True)
    ind = R['leaves']
    
    #Plotting
    plt.rcParams['figure.figsize'] = 14, 4 
    plt.subplot(2,2,ix)
    ix += 1
    plt.hist(listS, int(round(Dseq[k]['Npos']/2)))
    plt.xlabel('Pairwise sequence identities', fontsize=14)
    plt.ylabel('Number', fontsize=14)
    plt.subplot(2,2,ix)
    ix += 1
    plt.imshow(Dsca[k]['simMat'][np.ix_(ind,ind)], vmin=0, vmax=1); plt.colorbar();   
    plt.tight_layout()
    plt.savefig("pairwise_seq_identity.pdf")
    

for a in range(N_alg):
    plt.rcParams['figure.figsize'] = 9, 4 
    hist0, bins = np.histogram(Dsca[a]['Lrand'].flatten(), bins=Dseq[a]['Npos'], \
                               range=(0,Dsect[a]['Lsca'].max()))
    hist1, bins = np.histogram(Dsect[a]['Lsca'], bins=Dseq[a]['Npos'],\
                               range=(0,Dsect[a]['Lsca'].max()))
    plt.subplot(2,1,a+1)
    plt.bar(bins[:-1], hist1, np.diff(bins),color='k')
    plt.plot(bins[:-1], hist0/Dsca[a]['Ntrials'], 'r', linewidth=3)
    plt.tick_params(labelsize=11)
    plt.xlabel('Eigenvalues', fontsize=18); plt.ylabel('Numbers', fontsize=18);
    print('Number of eigenmodes to keep is %i' %(Dsect[a]['kpos']))
plt.tight_layout()
plt.savefig("eigenvalues.pdf")

plt.rcParams['figure.figsize'] = 20, 5 
for a in range(N_alg):
    print("alignment: "+AlgName[a])
    for n,ipos in enumerate(Dsect[a]['ics']):
        sort_ipos = sorted(ipos.items)
        ats_ipos = ([Dseq[a]['ats'][s] for s in sort_ipos])
        ic_pymol = ('+'.join(ats_ipos))
        print('IC %i is composed of %i positions:' % (n+1,len(ats_ipos)))
        print(ic_pymol + "\n")

### Correct Population of Sectors ###
sectors = []
for a in range(N_alg):
    sec_groups = [k for k in range(Dsect[a]['kpos'])]
    sectors_alg = []
    s = sca.Unit()
    all_items = []
    all_Vp = []

    # Combine all ICs into a single sector and re-sort
    for i in range(Dsect[a]['kpos']):
        all_items += Dsect[a]['ics'][i].items
        tmp1 = Dsect[a]['Vpica'][Dsect[a]['ics'][i].items, :]
        all_Vp += list(tmp1[:, 0].T)
    
    svals = list(np.argsort(all_Vp))
    svals.reverse()
    s.items = [all_items[i] for i in svals]
    s.col = (1 / len(sec_groups)) * i
    sectors_alg.append(s)
    sectors.append(sectors_alg)

### Plotting the full SCA and reordered matrix side by side ###
for a in range(N_alg):
    # Set up a larger figure for side-by-side plotting
    plt.figure(figsize=(18, 9))  # 18x9 inches to accommodate both matrices side by side

    # Plot the full SCA matrix (left side)
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot (full matrix)
    plt.imshow(Dsca[a]['Csca'], vmin=0, vmax=2, interpolation='none', aspect='equal')
    plt.title('Full SCA Matrix (All Positions)')
    plt.colorbar()

    ### Plot the "reordered matrix" (right side) ###
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot (reordered matrix)
    plt.imshow(Dsca[a]['Csca'][np.ix_(Dsect[a]['sortedpos'], Dsect[a]['sortedpos'])], 
               vmin=0, vmax=2, interpolation='none', aspect='equal', 
               extent=[0, sum(Dsect[a]['icsize']), 0, sum(Dsect[a]['icsize'])])
    plt.title('Reordered Matrix')
    
    line_index = 0
    for i in range(Dsect[a]['kpos']):
        plt.plot([line_index + Dsect[a]['icsize'][i], line_index + Dsect[a]['icsize'][i]], 
                 [0, sum(Dsect[a]['icsize'])], 'w', linewidth=2)
        plt.plot([0, sum(Dsect[a]['icsize'])], [sum(Dsect[a]['icsize']) - line_index, 
                                                sum(Dsect[a]['icsize']) - line_index], 'w', linewidth=2)
        line_index += Dsect[a]['icsize'][i]
    
    plt.tight_layout()
    plt.savefig("sca_reordered_matrix_side_by_side.pdf")
    plt.close()

# Print sector information
for a in range(N_alg):
    print("Alignment: " + AlgName[a])
    for i,k in enumerate(sectors[a]):
        sort_ipos = sorted(k.items)
        ats_ipos = ([Dseq[a]['ats'][s] for s in sort_ipos])
        ic_pymol = ('+'.join(ats_ipos))
        print('Sector %i is composed of %i positions:' % (i+1,len(ats_ipos)))
        print(ic_pymol + "\n")
        
sca.writePymol('1RX2', sectors[1], Dsect[1]['ics'], Dseq[1]['ats'],\
               '../output/DHFR_PEPM3.pml','A', '../data/')
