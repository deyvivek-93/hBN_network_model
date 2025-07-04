import matplotlib.pyplot as plt
import sys 
import gen_points
import networkx as nx
import numpy as np
import collections
from gen_points import *
from matplotlib.patches import Rectangle
import pandas as pd
import os
from scipy.optimize import curve_fit
import math

from function_points_Icut import initialize_graph_attributes, mod_voltage_node_analysis, calculate_network_resistance, calculate_Vsource, calculate_Isource, update_edge_weigths


#%% CHOOSE THE TYPE OF ANALYSIS

structural_analysis = 0 
functional_analysis = 1

#%% CHOOSE PLOTS

plt_network_adj_matrix = 0 #plot adjacency matrix

plot_network_1 = 0     
plot_network_pre = 0 
plot_network_current = 0  
plot_network_resistance = 0
plot_network_3 = 0     

plot_network_4 = 0     #plot degree of nodes histogram
plot_network_5 = 0    #plot connected components 
plot_network_6 = 0     #plot only the largest connected component
plot_network_7 = 0     #plot of the graph with highlighted the largest connected component
plot_network_8 = 0     #plot the electrical backbone (conneted component connecting sourcenode and graoundone)

plot_conductance = 0   #plot conductance vs stimulation voltage
voltage_distribution_map = 0
conductance_map = 0
information_centrality_map = 0
num_cluster_per_layer = 100
Lx_min , Lx_max = 0 , 10     #scale in nm
d0 = 0.2    #nm
L_x = Lx_max
L_y = 10    # nm

Y_min=0.1                                                                  
Y_max=Y_min*100
G0=1

beta= 100       
mu=0.346
k=0.038
           

import argparse

# Create an argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--Icut', type=float, help='Value of Icut')
parser.add_argument('--RF', type=float, help='Value of RF')
parser.add_argument('--Gon', type=float, help='Value of Gon')
parser.add_argument('--noise_stddev', type=float, help='Value of noise_stddev')
parser.add_argument('--cur_factor', type=float, help='Value of cur_factor')
parser.add_argument('--Vp', type=float, help='Value of Vp')

# Parse the command-line arguments
args = parser.parse_args()
Icut = args.Icut
RF= args.RF
Gon = args.Gon
noise_stddev = args.noise_stddev
cur_factor = args.cur_factor
Vp = args.Vp
IT= round(Icut*cur_factor,6)
#source and ground node 
sourcenode = 366                                            
groundnode = 365

#Electrical stimulation parameters
n=2000     ##number of pulses       
p=10     ##number of read points
l=1       ## length of pulse train


#Generate Input Voltage list
Vin_list=[0.001]
Vstim_list=[Vp,Vp,Vp,Vp,Vp,Vp,Vp,Vp,Vp,Vp]*n
Vread_list=[0.001]*p
print("IT",IT)


Vin_list.extend(Vstim_list)
Vin_list.extend(Vread_list)
Vin_list = Vin_list*l     #generated the input voltage pulse


#Generate Time list (should be consistent with the Voltage list)
tp = 0.01       # sampling time interval
delta_t=[0]
delta_t_stim=[tp,tp,tp,tp,tp,tp,tp,tp,tp,tp]*n
delta_t_stim=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]*n
delta_t_read=[0.01]*p

delta_t.extend(delta_t_stim)
delta_t.extend(delta_t_read)
delta_t = delta_t*l    #generated time list corresponding to the input voltage pulse

timesteps= len(delta_t)

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"Icut_{Icut}_RF_{RF}.csv"
plot_name = f"Icut_{Icut}_RF_{RF}"
file_name_structural = f"data_CL_{num_cluster_per_layer}_structural_analysis.csv"
output_folder = "output_data_percolation_02062025"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
file_path = os.path.join(output_folder, file_name)
file_path_structural = os.path.join(output_folder, file_name_structural)


#%%Logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


# Generate Ag in hBN network
cluster_dict = gen_points.generate_clusters(num_random_points_per_layer=num_cluster_per_layer,Lx_min = Lx_min ,Lx_max= Lx_max,d0 = d0,L_x = Lx_max,L_y = L_y)
                                         
                                        
gen_points.generate_coordinate_dict(cluster_dict)

gen_points.generate_node_and_edges(cluster_dict)

gen_points.generate_graph(cluster_dict)


###############################################################################

#%% GRAPH REPRESENTATION OF THE NETWORK

xpos=[x for x in cluster_dict['xc']]
ypos=[y for y in cluster_dict['yc']]
node_dict = cluster_dict['node_dict']
G = nx.Graph()

# Add nodes to the graph with their respective positions
for node_id, (x, y) in enumerate(zip(xpos, ypos)):
    node_data = node_dict[node_id]  # Get the node attributes from node_dict
    G.add_node(node_id, x=x, y=y)

edge_list = cluster_dict['edge_list']

for edge in edge_list:
    u, v = edge
    G.add_edge(u, v)

for n in G.nodes():
    
    G.nodes[n]['pos']=(xpos[n],ypos[n])
    G.nodes[n]['layer']=int(ypos[n]*10)

node_remove=[]
for u,v in G.edges():
    
    distance = abs(((G.nodes[u]['pos'][0] - G.nodes[v]['pos'][0])**2 +((G.nodes[u]['pos'][1] - G.nodes[v]['pos'][1])**2))**0.5)
    G[u][v]['distanceD']= distance
    if G[u][v]['distanceD'] <= 0.35:
        node_remove.append(u)
for node in node_remove:
    if G.has_node(node):
        # Remove the node
        G.remove_node(node)
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute='original_label')
       
   

for u,v in G.edges():
    G[u][v]['effectiveD'] = np.round(G[u][v]['distanceD'] - 0.35,4)
    
    if G[u][v]['effectiveD'] >= 0 and G[u][v]['effectiveD'] <= 0.35 :
        G.add_edge(u, v)
        G[u][v]['flag'] = False
    else:
        G.remove_edge(u,v)

for u in G.nodes():
    for v in G.nodes():
        if G.nodes[u]['layer'] == 2 and G.nodes[v]['layer']== -6:       ##ground
            G.add_edge(u,v)
        elif G.nodes[u]['layer'] == 98 and G.nodes[v]['layer']== 106:       ##source
            G.add_edge(u,v)  

if plot_network_pre:

    fig, ax = plt.subplots()
    
    pos=nx.get_node_attributes(G,'pos')

    
    Lx = cluster_dict['length_x']
    Ly = cluster_dict['length_y']
      
    ax.set_aspect(1) 
    ax.set_xlabel(r'x (nm)')
    ax.set_ylabel(r'y (nm)')
    ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax.axis([0*Lx,1*Lx,-0.1*Ly,Ly+1]) 
    ax.set_title('Ag cluster network structure - graph representation')
  
    nx.draw_networkx(G,pos, node_color='r',node_size=15, with_labels=True, font_size = 7)
    ax.grid()
    plt.show()

#%% ANALYSIS OF GRAPH FUNCTIONALITIES

if functional_analysis:

    logging.info('Functional analysis: started')
    
    # CHECK IF SOURCENODE AND GROUNDNODE ARE CONNECTED
    if nx.has_path(G, sourcenode, groundnode) is True:
        print('Source and ground node are connected!')
    else:
        print('Source and ground node are NOT connected!')
        sys.exit()
    
    #%% SELECTION OF THE CONNECTED COMPONENT (between sourcenode and groundnode) 
    
    ## Make a graph K with only nodes connected to source and ground nodes
    K=G.copy()  
    
    #remove nodes not connected to the groundnode (and sourcenode)
    removed_nodes= [n for n in G.nodes() if nx.has_path(G,n, groundnode) == False] 
    K.remove_nodes_from(removed_nodes)
    
    ## Make a graph M remapping of node names (for mod_voltage node analysis)
    M=K.copy() 
    mapping = dict(zip(M, range(0, K.number_of_nodes()))) 
    M = nx.relabel_nodes(M, mapping)
    node_list=list(M.nodes())
    if plot_network_pre:
        fig, ax = plt.subplots()
        pos=nx.get_node_attributes(M,'pos')
        Lx = cluster_dict['length_x']
        Ly = cluster_dict['length_y']
        ax.set_aspect(1) 
        ax.set_xlabel(r'x (nm)')
        ax.set_ylabel(r'y (nm)')
        ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
        ax.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
        ax.axis([0*Lx,1*Lx,-0.1*Ly,Ly+1]) 
        ax.set_title('Ag cluster network structure - graph representation')
        nx.draw_networkx(M,pos, node_color='r',node_size=15, with_labels=True, font_size = 7)
        ax.grid()
        plt.show()
    
    
    #%% DYNAMIC EVOLUTION OF THE GRAPH
    
    #Graph initialization
    M = initialize_graph_attributes(M,beta,G0,Y_min,sourcenode,groundnode)
    M.nodes[mapping[sourcenode]]['source_node']=True
    M.nodes[mapping[groundnode]]['ground_node']=True
    

    #Initialization of list over time      
    t_list=[[] for t in range(0,timesteps)]
    z_list=[[] for t in range(0,timesteps)]                                                           
    H_list=[[] for t in range(0,timesteps)] 
    I_list=[[] for t in range(0,timesteps)]
    V_list=[[] for t in range(0,timesteps)]                     
                                                     
    Rnetwork_list=[[] for t in range(0,timesteps)]
    Ynetwork_list=[[] for t in range(0,timesteps)]
    Shortest_path_length_network_list=[[] for t in range(0,timesteps)]
    
    
    #%% Pristine state                           
    t_list[0] = 0
    z_list[0] = 0   # hbn case
    H_list[0] = mod_voltage_node_analysis(M, Vin_list[0], mapping[sourcenode], mapping[groundnode])
    I_list[0] = calculate_Isource(H_list[0], mapping[sourcenode])
    V_list[0] = calculate_Vsource(H_list[0], mapping[sourcenode])

    
    nx.set_node_attributes(H_list[0], nx.information_centrality(M, weight='Y'), "information_centrality")
    
    Rnetwork_list[0] = calculate_network_resistance(H_list[0], mapping[sourcenode])
    Ynetwork_list[0] = 1/Rnetwork_list[0]
        
    #%% Evolution over time
    frequency = {}
    for i in range(1, int(timesteps)):
        
        t_list[i] = t_list[i-1]+delta_t[i] 
        current_max,M= update_edge_weigths(M,delta_t[i],G0,sourcenode,groundnode,beta,k,mu,IT,Icut,Gon,RF,noise_stddev)
        if current_max in frequency:
            frequency[current_max] += 1  
        else:
            frequency[current_max] = 1 
        H_list[i] = mod_voltage_node_analysis(M, Vin_list[i], mapping[sourcenode], mapping[groundnode])
        I_list[i] = calculate_Isource(H_list[i], mapping[sourcenode])
        V_list[i] = calculate_Vsource(H_list[i], mapping[sourcenode])
        nx.set_node_attributes(H_list[i], nx.information_centrality(M, weight='Y'), "information_centrality")
        Rnetwork_list[i]= nx.resistance_distance(M,mapping[sourcenode], mapping[groundnode], weight='Y', invert_weight=False)   
        Ynetwork_list[i] = 1/Rnetwork_list[i]
    logging.info('Functional analysis: finished')
    

### Plot 2 - Ag Network graph
def save_plot(filename):
    output_file = os.path.join(output_folder, filename)
    plt.savefig(output_file)
    plt.close()
values = list(frequency.keys())
num_freqs = list(frequency.values())

# Plot the histogram
plt.bar(values, num_freqs)
plt.xlabel('Current Max')
plt.ylabel('Frequency')
plt.title('Histogram of Current Max')
save_plot(f"{plot_name}_Histogram of Current Max.png")

if plot_network_2:

    fig, ax = plt.subplots()    
    pos=nx.get_node_attributes(G,'pos')
    Lx = cluster_dict['length_x']
    Ly = cluster_dict['length_y']
    ax.set_aspect(1) 
    ax.set_xlabel(r'x (nm)')
    ax.set_ylabel(r'y (nm)')
    ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax.axis([0*Lx,1*Lx,-0.1*Ly,Ly+1]) 
    ax.set_title('Ag cluster network structure - graph representation')
  
    nx.draw_networkx(G,pos, node_color='r',node_size=15, with_labels=True, font_size = 7)
    ax.grid()
    save_plot(f"{plot_name}_Ag cluster network structure.png")
    plt.show()
    

if plot_network_current:
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Current', color=color)
    ax1.plot(t_list, I_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.title('Current')
    save_plot(f"{plot_name}_Current.png")
    plt.show()

if plot_network_resistance:
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Resistance', color=color)
    ax1.plot(t_list, Rnetwork_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.title('Resistance')
    save_plot(f"{plot_name}_Resistance.png")

if plot_network_8:
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    
    pos=nx.get_node_attributes(G,'pos')
    pos_M = nx.get_node_attributes(M,'pos')
    
    Lx = cluster_dict['length_x']
    Ly = cluster_dict['length_y']
      
    ax.set_aspect(1) 
    ax.set_xlabel(r'x (nm)')
    ax.set_ylabel(r'y (nm)')
    ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax.axis([0*Lx,1*Lx,-0.1*Ly,Ly+1]) 
    ax.set_title(f"{file_name}_Electrical backbone")


    nx.draw_networkx(G,pos, node_color='lightgray',node_size=20, with_labels=False)     #removed hold
    nx.draw_networkx(M,pos_M, node_color='b',node_size=20, with_labels=False)       #removed hold
    
    nx.draw_networkx_nodes(G,pos,
                       nodelist=[sourcenode],
                       node_color='r',
                       node_size=300,
                   alpha=0.5)
    
    nx.draw_networkx_nodes(G,pos,
                       nodelist=[groundnode],
                       node_color='g',
                       node_size=300,
                   alpha=0.5)
    
    #ax.grid()
    save_plot('Electrical backbone')
    plt.show()
    

### Plot G-V characteristic

if plot_conductance:
    
    ### Plot 6 - G-V characteristic
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Input Voltage (V)', color=color)
    ax1.plot(t_list, V_list, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Conductance (a.u.)', color=color)  
    ax2.plot(t_list, Ynetwork_list, color=color)
    #color = 'tab:green'
    #ax2.plot(t_list, Ynetwork_list_gaussian, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    #fig.tight_layout()

    plt.title('Network conductance')
    save_plot(f"{plot_name}_Conductance.png")


## Plot voltage map

timestamp_map= len(t_list)-1   #select the timestamp

if voltage_distribution_map:

    L=H_list[timestamp_map].copy()
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    
    pos=nx.get_node_attributes(L,'pos')
    Lx = cluster_dict['length_x']
    Ly = cluster_dict['length_y']
      
    ax.set_aspect(1) 
    ax.set_xlabel(r'x (nm)')
    ax.set_ylabel(r'y (nm)')
    ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax.axis([0*Lx,1*Lx,-0.1*Ly,Ly+1]) 
    ax.set_title('Voltage distribution')

   
    nx.draw_networkx(L, pos, 
                 node_size=20,
                 node_color=[L.nodes[n]['V'] for n in L.nodes()],
                 cmap=plt.cm.plasma, 
                 arrows= False,
                 with_labels=False,font_size=6,
                 )
    
    nx.draw_networkx_nodes(L,pos,
                       nodelist=[mapping[sourcenode]],
                       node_color='r',
                       node_size=300,
                   alpha=0.5)
    
    nx.draw_networkx_nodes(L,pos,
                       nodelist=[mapping[groundnode]],
                       node_color='g',
                       node_size=300,
                   alpha=0.5)

    ##ax.grid()
    save_plot(f"{plot_name}_voltage_distribution_map.png")
    plt.show()
   
### Plot conductance map
    

timestamp_map=len(t_list)-1    #select the timestamp

if conductance_map:
    
    L=H_list[timestamp_map].copy()
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)
    
    pos=nx.get_node_attributes(L,'pos')
    Lx = cluster_dict['length_x']
    Ly = cluster_dict['length_y']
      
    ax.set_aspect(1) 
    ax.set_xlabel(r'x (nm)')
    ax.set_ylabel(r'y (nm)')
    ax.ticklabel_format(style='plain', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='plain', axis='y', scilimits=(0,0))
    ax.axis([0*Lx,1*Lx,-0.1*Ly,Ly+1 ]) 
    ax.set_title('Conductance distribution')

   
    nx.draw_networkx(L, pos, 
                 node_size=20,
                 node_color=[L.nodes[n]['V'] for n in L.nodes()],
                 cmap=plt.cm.Blues,
                 edge_color=[L[u][v]['Y'] for u,v in L.edges()],
                 width=2, 
                 edge_cmap=plt.cm.Reds, 
                 edge_vmin=Y_min,
                 edge_vmax=Y_max,
                 arrows= False,
                 with_labels=False,font_size=6,)
    
    nx.draw_networkx_nodes(L,pos,
                       nodelist=[mapping[sourcenode]],
                       node_color='r',
                       node_size=300,
                   alpha=0.5)
    
    nx.draw_networkx_nodes(L,pos,
                       nodelist=[mapping[groundnode]],
                       node_color='k',
                       node_size=300,
                   alpha=0.5)

    ##ax.grid()
    save_plot(f"{plot_name}_conductance_distribution_map.png")
    plt.show()

def fft_data(file_path):
    # Read data from CSV file
    data = pd.read_csv(file_path)
    time = data['Time'].values
    current = data['Current'].values

    # Compute the FFT
    fft_current = np.fft.fft(current)
    freq = np.fft.fftfreq(len(current), time[1] - time[0])
    real_fft = np.abs(np.real(fft_current))

    # Plot time vs. current data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(time, current)
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.title('Time vs. Current')

    # Plot the real part of the FFT in log-log scale
    plt.subplot(1, 2, 2)
    plt.loglog(freq, np.abs(np.real(fft_current)))
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Real Part of FFT (log-log scale)')

    plt.tight_layout()
    save_plot(f"{plot_name}_fft.png")
    plt.show()

    # Find the best fit parameters
    popt, pcov = curve_fit(power_law, freq, np.abs(np.real(fft_current)))
    exponent = popt[1]

    print('Exponent of 1/f noise:', exponent)
    return freq, real_fft

def calculate_psd_from_fft(file_path):

    data = pd.read_csv(file_path)
    time = data['Time'].values
    current = data['Current'].values
    # Calculate the sampling frequency (assuming evenly spaced time values)
    sampling_rate = 1 / np.mean(np.diff(time))

    # Perform the FFT on the current data
    fft_current = np.fft.fft(current)

    # Calculate the power spectrum
    #power_spectrum = np.abs(fft_current) ** 2
    power_spectrum = np.abs(fft_current)

    # Calculate the frequencies corresponding to each FFT bin
    frequencies = np.fft.fftfreq(len(current), d=1/sampling_rate)

    # Take only the positive frequencies (one-sided PSD)
    positive_frequencies = frequencies[:len(frequencies)//2]
    one_sided_psd = power_spectrum[:len(power_spectrum)//2]
    normalized_psd = one_sided_psd / np.max(one_sided_psd)

    return positive_frequencies, normalized_psd

# Fit to find the exponent of 1/f noise
def power_law(x, a, b):
    return a * np.power(x, -b)

def autocorrelation(file_path):
    data=pd.read_csv(file_path)
    current_series = data['Current']
    acf = np.correlate(current_series, current_series, mode='full')
    N = len(current_series)
    print("N",N)
    delta = 1.96 / np.sqrt(N)
    max_lag = int(0.5*N)
    lags = range(max_lag)
    acf_values = [current_series.autocorr(lag=lag) for lag in lags]
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
    # Plotting the ACF with confidence intervals
    ax1.stem(lags, acf_values)
    ax1.axhline(y=delta, color='r', linestyle='--', label='Upper Confidence Interval')
    ax1.axhline(y=-delta, color='r', linestyle='--', label='Lower Confidence Interval')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Autocorrelation Function with Confidence Intervals')
    ax1.legend()

    ax2.plot(lags, acf_values)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    plt.tight_layout()
    save_plot(f"{plot_name}_Autocorrelation.png")
    plt.show()
    return acf_values,lags,delta


df1 = pd.DataFrame({'Time': t_list})
df2 = pd.DataFrame({'pulse': Vin_list})
df3 = pd.DataFrame({'G': Ynetwork_list})
df4 = pd.DataFrame({'R': Rnetwork_list})
df5 = pd.DataFrame({'Current': I_list})
df6 = pd.DataFrame({'Cur_Hist_Values': values})
df7 = pd.DataFrame({'Cur_Hist_Freq': num_freqs})


# Combine DataFrames horizontally
combined_df = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=1)

# Save the combined DataFrame as a CSV file in the current folder
combined_df.to_csv(file_path, index=False)
acf_values,lags,delta = autocorrelation(file_path)
freq, real_fft = fft_data(file_path)
df8 = pd.DataFrame({'Lags': lags})
df9 = pd.DataFrame({'Autocorrelation': acf_values})
df10 = pd.DataFrame({'confidence_interval': delta},  index=[1])
df11 = pd.DataFrame({'freq': freq})
df12 = pd.DataFrame({'fft': real_fft})

combined_df_with_fft = pd.concat([combined_df, df8, df9, df10, df11, df12], axis=1)
combined_df_with_fft.to_csv(file_path, index=False)

freq_psd, psd = calculate_psd_from_fft(file_path)
df13 = pd.DataFrame({'freq_psd': freq_psd})
df14 = pd.DataFrame({'psd':psd})
combined_df_with_fft = pd.concat([combined_df_with_fft, df13, df14], axis=1)
combined_df_with_fft.to_csv(file_path, index=False)

print("Data saved to Excel file in folder")





