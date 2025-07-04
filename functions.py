import math
import random
from networkx import grid_graph
import numpy as np

import networkx as nx

########################  - GRAPH INITIALIZATION     -    #####################
    
def initialize_graph_attributes(G,G0,beta,Yin,sourcenode,groundnode):
    #add the initial conductance
    
    for n in G.nodes():
        if G.nodes[n]['layer'] == 2:
            G.edges[groundnode, n]['Y'] = 100         #original=100
            #G.edges[groundnode, n]['deltaV']=0
        elif G.nodes[n]['layer'] == 98:
            G.edges[sourcenode, n]['Y'] = 100         #original=100
            #G.edges[groundnode, n]['deltaV']=0
    for a,b,c in G.edges(data=True):
        if a not in (sourcenode, groundnode) and b not in (sourcenode, groundnode):
    
            distance = abs(((G.nodes[a]['pos'][0] - G.nodes[b]['pos'][0])**2 +((G.nodes[a]['pos'][1] - G.nodes[b]['pos'][1])**2))**0.5)
            G[a][b]['distanceD']= distance
            G[a][b]['effectiveD']= abs(G[a][b]['distanceD']-0.35)
            G.edges[a,b]['z']= 0
            G.edges[a,b]['Y']= G0*np.exp(-beta*(G[a][b]['effectiveD']))
            G.edges[a,b]['R'] = 1/G.edges[a,b]['Y']

    for n in G.nodes():
        G.nodes[n]['pad']=False
        G.nodes[n]['source_node']= False
        G.nodes[n]['ground_node']= False
        
    return G

#################  - MODIFIED VOLTAGE NODE ANALYSIS     -    ##################

def mod_voltage_node_analysis(G, Vin, sourcenode, groundnode):
    ## MODIFIED VOlTAGE NODE ANALYSIS

    # definition of matrices
    matZ = np.zeros(shape=(G.number_of_nodes(), 1))         # a column matrix with row no = no of nodes
    matG = np.zeros(shape=(G.number_of_nodes()-1, G.number_of_nodes()-1))
    matB = np.zeros(shape=(G.number_of_nodes()-1, 1))
    matD = np.zeros(shape=(1, 1))

    # filling Z matrix
    matZ[-1] = Vin      #enter Vin in last element of the column

    # filling Y matrix as a combination of G B D in the form [(G B) ; (B' D)]

    # elements of G
    for k in range(0, G.number_of_nodes()):
        if k < groundnode:
            nodeneighbors = list(G.neighbors(k))  # list of neighbors nodes
            for m in range(0, len(nodeneighbors)):
                matG[k][k] = matG[k][k] + G[k][nodeneighbors[m]]['Y']  # divided by 1       this is diagonal elements of G: addition of nearest neighbor conductance
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]<groundnode:
                    matG[k][nodeneighbors[m]] = -G[k][nodeneighbors[m]]['Y']
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]>groundnode:
                    matG[k][nodeneighbors[m]-1] = -G[k][nodeneighbors[m]]['Y']
        if k > groundnode:
            nodeneighbors = list(G.neighbors(k))  # list of neighbors nodes
            for m in range(0, len(nodeneighbors)):
                matG[k-1][k-1] = matG[k-1][k-1] + G[k][nodeneighbors[m]]['Y']  # divided by 1
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]<groundnode:
                    matG[k - 1][nodeneighbors[m]] = -G[k][nodeneighbors[m]]['Y']
                if nodeneighbors[m] is not groundnode and nodeneighbors[m]>groundnode:
                    matG[k - 1][nodeneighbors[m]-1] = -G[k][nodeneighbors[m]]['Y']
    # matB
    if sourcenode < groundnode:
        matB[sourcenode] = 1
    elif sourcenode > groundnode:
        matB[sourcenode-1] = 1

    # matY
    submat1 = np.hstack((matG, matB))
    submat2 = np.hstack((np.transpose(matB), matD))
    matY = np.vstack((submat1, submat2))

    # solve X matrix from Yx = z
    invmatY = np.linalg.inv(matY)  # inverse of matY
    
    matX = np.matmul(invmatY, matZ)  # Ohm law

    # add voltage as a node attribute
    for n in G.nodes():
        if n == groundnode:
            G.nodes[n]['V'] = 0
        elif n < groundnode:
            G.nodes[n]['V'] = matX[n][0]
        elif n > groundnode:
            G.nodes[n]['V'] = matX[n - 1][0]
            
    ###DEFINE CURRENT DIRECTION

    # transform G to a direct graph H

    H = G.to_directed()  # transform G to a direct graph

    # add current as a node attribute

    for u, v in H.edges():
        H[u][v]['I'] = (H.nodes[u]['V'] - H.nodes[v]['V']) * H[u][v]['Y']
        H[u][v]['Irounded'] = np.round(H[u][v]['I'], 2)
        #print("H[u][v]['I']",H[u][v]['I'])

    # set current direction
    for u in H.nodes():  # select current direction
        for v in H.nodes():
            if H.has_edge(u, v) and H.has_edge(v, u):
                if H[u][v]['I'] < 0:
                    H.remove_edge(u, v)
                else:
                    H.remove_edge(v, u)

    return H


def update_edge_weigths(G,delta_t,G0,sourcenode,groundnode,beta,mu,k,IT,Icut,Gon,RF,noise_stddev):
    max_cur = 0
   
    for u,v in G.edges():
        if u not in (sourcenode, groundnode) and v not in (sourcenode, groundnode): 
        
            G[u][v]['deltaV']=abs(G.nodes[u]['V']-G.nodes[v]['V'])
            G[u][v]['Efield'] = G[u][v]['deltaV']/(G[u][v]['effectiveD']-G[u][v]['z'])
    noise = 0
    for u,v in G.edges():
        if u not in (sourcenode, groundnode) and v not in (sourcenode, groundnode):
        #print("flag:",G[u][v]['flag'], "u", u, "v ",v)     
            if not G[u][v]['flag']: 
                G[u][v]['z'] = abs((1-k*delta_t)*G[u][v]['z'] + k*delta_t*((((mu*G[u][v]['deltaV'])/(k*G[u][v]['effectiveD']**2))/(1-G[u][v]['z']))+noise))

                ### with dimensionless approach
                if G[u][v]['z'] >= 0.999:
                    G[u][v]['z']=0.999
                    G[u][v]['Y'] = Gon
                    G[u][v]['I'] = G[u][v]['deltaV']*G[u][v]['Y']
                    
                    if G[u][v]['I'] >= max_cur:
                        max_cur = G[u][v]['I']
                        
                    if G[u][v]['I'] >= Icut:
                        G[u][v]['flag'] = True
                else:
                    G[u][v]['Y'] = G0*np.exp(-beta*G[u][v]['effectiveD']*(1-G[u][v]['z']))
            else:
                G[u][v]['z'] = G[u][v]['z']*np.exp(-0.999/RF)
                G[u][v]['Y'] = G0*np.exp(-beta*G[u][v]['effectiveD']*(1-G[u][v]['z']))
                G[u][v]['I'] = G[u][v]['deltaV']*G[u][v]['Y']
                if G[u][v]['I'] <= Icut-IT:
                    G[u][v]['flag'] = False
    return max_cur, G

#######################  - CALCULATE V source     -    ########################
    
    
def calculate_Vsource(G, sourcenode):

    Vsource=G.nodes[sourcenode]['V']
    
    return Vsource


#######################  - CALCULATE z     -    ########################

def z(H, sourcenode):
    
    I_from_source=0
    for u,v in H.edges(sourcenode):
        a= H[u][v]['I']
        I_from_source=I_from_source+a
    
    return I_from_source

#######################  - CALCULATE I source     -    ########################

def calculate_Isource(H, sourcenode):
    
    I_from_source=0
    for u,v in H.edges(sourcenode):
        a= H[u][v]['I']
        I_from_source=I_from_source+a
    
    return I_from_source


#################  - CALCULATE  NETWORK RESISTANCE     -    ####################
    

def calculate_network_resistance(H, sourcenode):
    
    I_fromsource = 0
    for u,v in H.edges(sourcenode):
        a= H[u][v]['I']
        I_fromsource=I_fromsource+a

    
    Rnetwork=H.nodes[sourcenode]['V']/I_fromsource
    
    return Rnetwork


import sympy as sp


def calc_res_with_gaussian(G, nodeA, nodeB):
    N_nodes = G.number_of_nodes()
    # 2.1: Creating Gaussian Matrix
    gaussian_matrix = sp.zeros(
        N_nodes, N_nodes)  # sympy
    gaussian_matrix_shape = N_nodes
    for c_int, (node1, node2, data) in enumerate(G.edges(data=True)):
        gaussian_matrix[node1, node2] = round(
            G.edges[node1, node2]['conductance'], 25)  # C_function
        gaussian_matrix[node2, node1] = round(
            G.edges[node1, node2]['conductance'], 25)  # C_function
    # 2.2: Set Diagonal Elements
    for c_gaussian_matrix in range(0, gaussian_matrix_shape):
        # sympy
        S_minus_temp = 0
        
        for c2_gaussian_matrix in range(0, gaussian_matrix_shape):
            if c_gaussian_matrix == c2_gaussian_matrix:
                continue
            # sympy
            S_minus_temp += abs(
                gaussian_matrix[c_gaussian_matrix, c2_gaussian_matrix])
        # sympy
        gaussian_matrix[c_gaussian_matrix,
                              c_gaussian_matrix] = (-S_minus_temp)
    # 2.3: Adding vector components
    U_bias = 3  # [V]
    N_equations = N_nodes
    voltage_matrix = sp.Matrix(
        N_equations, 1, sp.symbols('U0:%d' % N_equations))
    ## Set Current vector
    current_matrix = sp.zeros(N_equations, 1)
    current_matrix[nodeA] = sp.symbols('I_start')
    current_matrix[nodeB] = sp.symbols('I_end')
        
    ## Set Voltage Bias
    voltage_matrix[nodeA] = U_bias
    voltage_matrix[nodeB] = 0
   
    # Solve of equations
    systemEq = []
    mulResult = gaussian_matrix*voltage_matrix        #this is the KCL
    for c_eq in range(0, N_equations):
        systemEq.append(mulResult[c_eq]-current_matrix[c_eq])
    result = sp.solve(systemEq)
    current_start = result[sp.symbols('I_start')]
    current_end = result[sp.symbols('I_end')]
    # Resistance of network
    R_ges_minus = U_bias / \
            abs(float(current_start))
    return R_ges_minus
