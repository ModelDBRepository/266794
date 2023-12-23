import argparse
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-a", "--gnmda", type=str,
                help="conductance of extrasynaptic NMDAR")

ap.add_argument("-b", "--stimfreq", type=str,
                help="frequency during stimulation phase")
ap.add_argument("-c", "--run", type=str,
                help="run")

args = vars(ap.parse_args())

import numpy as np
from brian2 import *
#import brian2genn
set_device('cpp_standalone', build_on_run=True, directory=args["gnmda"]+args["stimfreq"]+args["run"])  # Use fast "C++ standalone mode"
#prefs.codegen.target = 'numpy'
prefs.devices.cpp_standalone.openmp_threads = 4
seed(int(args["run"]))
################################################################################
# Model parameters
################################################################################
## Some metrics parameters needed to establish proper connections
size = 3.75*mmeter           # Length and width of the square lattice
distance = 50*umeter         # Distance between neurons
### Neuron parameters
C_m = 100*pF                 # Membrane capacitance
k_n = 0.7*nS/mV              # Constant 1/R
v_t = -50*mV                 # Instantaneous threshold potential
c_n = -60*mV                 # Potential reset value
d_n = 100*pA                 # outward minus inward current
a_n = 0.03/ms                # Recovery time constant
b_n = -2*nS                  # Constant 1/R
d_soma= 15e-4*cm             # Soma diameter
rad_soma =d_soma/2           # Soma radius
A_soma =(4*pi*rad_soma*rad_soma) # Soma surface
Vrest = -70*mV               # Resting potential
### Synapse parameters
Rm =0.7985e+8*kohm           # Input resistance of dendritic spine
tau_mem =50*ms               # Post synaptic time constant?
g_membrane = 1/Rm            # Post synaptic membrane conductance
rad_spine =0.3e-4*cm         # Post synaptic radius
vspine =((4/3)*pi*((rad_spine)*(rad_spine)*(rad_spine))) # Post synaptic volume
spine_area = 4*pi*(rad_spine)**2 # Post synaptic surface
spine_c_densitity = (tau_mem/Rm)/spine_area # Post synaptic capacitance density
ep_rest =0.1*umolar          # Post synaptic IP3 resting concentration
c_p_rest =0.1*umolar         # Post synaptic Ca2+ resting concentration
bt =200000*nmolar            # Endogenous buffer concentration,
                             # chosen so that endogenous buffer capacity is 20 (Tewari & Majumdar, 2012)
K_bt =10000*nmolar           # Ca2+ affinity of Endogenous buffer (Keller et al., 2008)
ks= (100e-3)/ms              # Calcium extrusion rate by PMCa (Keller et al., 2008)
frac_nmdar = 0.057           # Fraction of Ca2+ carried by NMDAR current
F_p=96487*coulomb/mole       # Faraday's constant
g_Ca = 15e-9*mS              # R-type channel conductance
E_Ca=27.4*mV                 # R-type reversal potential
alpha_ampa = 1.1/(mmolar*ms) # binding rate
beta_ampa = 0.19/ms          # unbinding rate
g_ampa_min= 0.35*nS          # Minimal AMPAR conductance
g_ampa_max= 1*nS             # Maximal AMPAR conductance
V_ampa =0*mV                 # AMPAR reversal potential
alpha_nmda = 0.072/(mmolar*ms)
beta_nmda = 0.0066/ms
all_g_nmda = [0.01e-6, 0.1e-6, 0.2e-6, 0.3e-6, 0.4e-6, 0.5e-6, 0.6e-6]*mS
g_nmda = all_g_nmda[int(args["gnmda"])] # selected extrasynaptic NMDAR conductance
g_nmda_min = all_g_nmda[0]   # Minimal NMDAR conductance, set for all synaptic NMDAR
g_nmda_max = all_g_nmda[-1]  # Maximal NMDAR conductance
Mg2 = 1                      # Mg2+ block (mM)
V_nmda = 0*mV                # NMDAR reversal potential
alpha_gaba= 5/(mmolar*ms)
beta_gaba= 0.18/ms
g_gaba=0.25*nS               # Synaptic and extrasynaptic GABAR conductance (fixed)
V_gaba = -70*mV              # GABAR reversal potential

### Astrocyte parameters
# ---  Calcium fluxes
O_P = 0.9*umolar/second      # Maximal Ca^2+ uptake rate by SERCAs
K_P = 0.05*umolar            # Ca2+ affinity of SERCAs
C_T = 2*umolar               # Total cell free Ca^2+ content
rho_A = 0.18                 # ER-to-cytoplasm volume ratio
Omega_C = 6/second           # Maximal rate of Ca^2+ release by IP_3Rs
Omega_L = 0.1/second         # Maximal rate of Ca^2+ leak from the ER
# --- IP_3R kinectics
d_1 = 0.13*umolar            # IP_3 binding affinity
d_2 = 1.05*umolar            # Ca^2+ inactivation dissociation constant
O_2 = 0.2/umolar/second      # IP_3R binding rate for Ca^2+ inhibition
d_3 = 0.9434*umolar          # IP_3 dissociation constant
d_5 = 0.08*umolar            # Ca^2+ activation dissociation constant
# --- IP_3 production
# --- Agonist-dependent IP_3 production
O_beta = 0.5*umolar/second   # Maximal rate of IP_3 production by PLCbeta
O_N = 0.3/umolar/second      # Agonist binding rate
Omega_N = 0.5/second         # Maximal inactivation rate
K_KC = 0.5*umolar            # Ca^2+ affinity of PKC
zeta = 10                    # Maximal reduction of receptor affinity by PKC
# --- Endogenous IP3 production
O_delta = 1.2*umolar/second  # Maximal rate of IP_3 production by PLCdelta
kappa_delta = 1.5*umolar     # Inhibition constant of PLC_delta by IP_3
K_delta = 0.1*umolar         # Ca^2+ affinity of PLCdelta
# --- IP_3 degradation
Omega_5P = 0.05/second       # Maximal rate of IP_3 degradation by IP-5P
K_D = 0.7*umolar             # Ca^2+ affinity of IP3-3K
K_3K = 1.0*umolar            # IP_3 affinity of IP_3-3K
O_3K = 4.5*umolar/second     # Maximal rate of IP_3 degradation by IP_3-3K
# --- IP_3 diffusion
F = 2*umolar/second          # GJC IP_3 permeability
I_Theta = 0.3*umolar         # Threshold gradient for IP_3 diffusion
omega_I = 0.05*umolar        # Scaling factor of diffusion

# --- IP_3 external production
F_ex = 0.09*umolar/second    # Maximal exogenous IP3 flow
I_Theta = 0.3*umolar         # Threshold gradient for IP_3 diffusion
omega_I = 0.05*umolar        # Scaling factor of diffusion
I_bias = 2*umolar
# --- Gliotransmitter release and time course
C_Theta = 0.19669*umolar     # Ca^2+ threshold for exocytosis
Omega_A = 1/(800*ms)         # Gliotransmitter recycling rate
U_A = 0.6                    # Gliotransmitter release probability
G_T = 250*mmolar             # Total vesicular gliotransmitter concentration
rho_e = 6.5e-4               # Astrocytic vesicle-to-extracellular volume ratio
Omega_e = 1/(100*ms)         # Gliotransmitter clearance rate
alpha = 0.0                  # Gliotransmission nature

########################
# Dendrite parameters
########################
N_postsyn = 20               # Number of incoming connections
dendrite_length = 0.5*mm     # Dendritic' compartment length
length_comp = dendrite_length/N_postsyn # length of a section of the dendritic compartment
dendrite_diam = 2e-4*cm      # Dendrite diameter
dend_area = 2*pi*dendrite_diam*dendrite_length # Dendrite surface
spine_density =2/um         # Number of synapses per micron
# Here we assume the synaptic density per section, in order to accumulate incoming dendritic spine current
# from several synaptic sources.
dendrite_spine_density = spine_density*length_comp/dend_area # Synaptic density
R_neck = 10*Gohm          # Spine' neck resistance
gna = 0.25*mS/cm2            # Sodium channel conductance
gks = 0.1*mS/cm2             # Potassium channel conductance
gka = 10*mS/cm2              # Potassium channel conductance
g_Leak = 0.3*mS/cm2          # Leak conductance
g_coupling = 0.2*mS/cm2      # Dendrite-soma coupling conductance
g_d = g_coupling/(1-0.1)     # Soma to dendrite coupling conductance
g_s = g_coupling/(0.1)       # Dendrite to soma coupling conductance
capacitance = 1*uF/cm2       # Dendritic' membrane capacitance


# In order to get a connectivity matrix with a fixed number of incoming connections,
# this function takes the id's of the presynaptic population,
# the id's of the postsynaptic population,
# and the desired number of income for the post population
# it returns the source and target id's
def get_connection_mat(pre_population, post_population, number):
    income = choice(pre_population, number.sum())
    post_income = np.repeat(post_population, number)
    return income, post_income

# Number of parallel pathway
nb_path = 1
# Number of Layer
nb_layer = 3
### General parameters
N_e = 80                 # Number of excitatory neurons
N_i = 20                 # Number of inhibitory neurons
N_sector = N_e+N_i       # Number of neuron per sector
ratio_exci = 0.8         # ratio of excitatory neuron
N_a = N_sector*nb_path*nb_layer # Number of astrocytes
All_n = N_sector*nb_path*nb_layer # All neurons
max_in = N_postsyn       # Number of incoming connections

# Stimulation
total_duration = 60      # Total simulation duration (second)
delta_t = 0.1*ms
first_freq = 0.1         # slower switching frequency, Hz
freqs = np.ones((10, 1))*first_freq # array for switching frequency
for ite in range(1, 10):
    freqs[ite] = freqs[ite-1]*2

#np.random.seed(int(args["run"])//10) # to get similar pattern across runs
stim_duration = 1000/freqs[int(args["stimfreq"])] # duration of one step signal
max_freq = 100           # the highest stimulation frequency in the signal
stim = np.random.random_sample((nb_path, int(1000*total_duration/stim_duration)))*max_freq # stimulation array
stim = np.repeat(stim, int(stim_duration/(delta_t/ms)), axis=1)

# This part is for similar stimulation across runs
# it simply adds some noise to the signal pattern
#noisy = np.random.randint(-10, 10, stim.shape)
#stim = stim+noisy
rolling = np.random.randint(0, int(stim_duration/(delta_t/ms)), nb_path)
stimu = np.zeros(stim.shape)
for r in range(nb_path):
    stimu[r, :] = np.roll(stim[r, :], rolling[r])
print(stimu.shape)
stimulus = TimedArray(stimu.T*Hz, dt=delta_t)

I_ex = ((200*uA/cm2)/delta_t)*ms       # External current for stimulation
I_ex2 = ((200*uA/cm2)/delta_t)*ms      # External current for noise

# Input neuron stimulation
# We only want to trigger spiking activity
# around the frequency given by the signal pattern
neuron_input_eqs = '''
index = 0 : integer (constant over dt)
du/dt = a_n*(b_n*(v-Vrest)-u)            : amp
# we got frequency closer to the one expected with stimulus(t, int(i/N_e))*delta_t*4)
dv/dt = (k_n*(v-Vrest)*(v-v_t)-u + I_ex*A_soma*(rand()<stimulus(t, int(i/N_e))*delta_t*4) )/C_m : volt
G = 1/(1+exp(-(v - 2*mV)/(5*mV))) : 1
is_ini = 0 : 1
# Neuron position in space
x : meter (constant)
y : meter (constant)
neuron_index : integer (constant)
'''
# Neuronal population base equations

neuron_eqs = '''
index : integer (constant)

# Summation of synaptic parameters
# Since Brian2 builds all objects with 
# similar properties and connectivities (ie, synapses),
# we need several variables to accumulate the activities 
# of several types of objects (ie, different types of synapses)
extra_a1 : 1
extra_a2 : 1
extra_ampar_average = (extra_a1 + extra_a2)/max_in : 1

# Dendrite part
r = 1/(1+exp(-(v_dend+57*mV)/5/mV)) : 1
I_NaP = gna * r**3 * (v_dend - (-55*mV)) : amp/metre**2
r2 = 1/(1+exp(-(v_dend+50*mV)/2/mV)) : 1
I_gks = gks * r2 * (v_dend - (-80*mV)) : amp/metre**2
a = 1/(1+exp(-(v_dend+45*mV)/6/mV)) : 1
b = 1/(1+exp(-(v_dend+56*mV)/15/mV)) : 1
I_ka = gka * a**3 * b * (v_dend - (-80*mV)) : amp/metre**2
I_leak = g_Leak*(v_dend- (-80*mV)) : amp/metre**2
I_ionic = -I_NaP -I_gks -I_ka -I_leak : amp/metre**2

# From previous layer
I_synapse1 : amp/metre**2
# From current layer excitatory
I_synapse2 : amp/metre**2
# From current layer inhibitory
I_synapse3 : amp/metre**2
I_synapse = I_synapse1 + I_synapse2 + I_synapse3  : amp/metre**2
in_soma = (g_d)*(v - v_dend) : amp/metre**2
to_soma = (-g_s) *  (v - v_dend) *A_soma : amp
# we got frequency closer to the one expected with ((1*Hz)*delta_t*4)
noise = I_ex2*int(rand()<((1*Hz)*delta_t*4)) : amp/metre**2 (constant over dt)
dv_dend/dt = ( I_synapse + in_soma +I_ionic)/capacitance : volt

du/dt = a_n*(b_n*(v-Vrest)-u)            : amp
dv/dt = (k_n*(v-Vrest)*(v-v_t)-u + to_soma +noise*A_soma)/C_m : volt
G = 1/(1+exp(-(v - 2*mV)/(5*mV))) : 1
# Neuron position in space
x : meter (constant)
y : meter (constant)
neuron_index : integer (constant)
'''

# Integration method
method_options  = None
method = 'euler'

# Input neuron constructor
P = NeuronGroup(N_e *nb_path,
                model=neuron_input_eqs,
                threshold='v>30*mV',
                reset='v=c_n \n u=u+100*pA ',
                events={'custom_event': 'v>-100*mV'},
                method=method,
                method_options=method_options,
                dt=delta_t)
P.v = Vrest

# Population neuron constructor
neurons = NeuronGroup(All_n,
                      model=neuron_eqs,
                      threshold='v>30*mV',
                      reset='v=c_n \n u=u+100*pA ',
                      events={'custom_event': 'v>-100*mV'},
                      method=method,
                      method_options=method_options,
                      dt=delta_t)

# Initial value setting
neurons.v_dend = Vrest
neurons.v = Vrest
neurons.I_synapse1 = 0*uamp/cm**2
neurons.I_synapse2 = 0*uamp/cm**2
neurons.I_synapse3 = 0*uamp/cm**2

# Identifications of inhibitory/excitatory neurons
# based on indice value
all_neurons = np.arange(All_n)
indices = np.arange(All_n)
neurons.neuron_index = np.arange(All_n)
np.random.shuffle(indices)
neurons.index = indices
exc_n_indice = indices[:int(ratio_exci*All_n)]
ini_n_indice = indices[int(ratio_exci*All_n):]

# Arrange neurons in a grid
N_rows = int( (N_e+N_i)*nb_path )
N_cols = nb_layer
grid_dist = (size / N_cols)
neurons.x = '(neuron_index // N_rows)*grid_dist - N_rows/2.0*grid_dist'
neurons.y = '(neuron_index % N_rows)*grid_dist - N_cols/2.0*grid_dist'
P.x = '(neuron_index // N_rows)*grid_dist - N_rows/2.0*grid_dist'
P.y = '(neuron_index % N_rows)*grid_dist - N_cols/2.0*grid_dist'

# binomial number generator
binomial_dis = BinomialFunction(n=6,p=0.52)
### Synapses
synapses_eqs = '''
neuron_index_syn : integer (constant)
# released synaptic neurotransmitter resources:
g                                                           : mmolar
# released synaptic neurotransmitter GABA:
gaba                                                        : mmolar
# gliotransmitter concentration in the extracellular space:
G_A_exc                                                         : mmolar
# which astrocyte covers this synapse ?
astrocyte_index : integer (constant)

# Potential of the dendritic compartment' membrane
v_dendrite = v_dend_post : volt
v_syn = v_dendrite : volt
# ratio of ionotropic receptors in "open" state
dm_ampa/dt = alpha_ampa * g * (1-m_ampa) - beta_ampa*m_ampa : 1 (clock-driven)
dm_nmda/dt = alpha_nmda * g * (1-m_nmda) - beta_nmda*m_nmda : 1 (clock-driven)
dm_gaba/dt = alpha_gaba * gaba * (1-m_gaba) - beta_gaba*m_gaba : 1 (clock-driven)
dm_nmdaextra/dt = alpha_nmda * G_A_exc * (1-m_nmdaextra) - beta_nmda*m_nmdaextra : 1 (clock-driven)

omega_ampa = 1 - (1/(0.4*sqrt(2*pi))*exp(-0.5 * (((c_p/umolar - 0.30)/0.25)** 2))) : 1
tau_ampa = (0.14)/(1.2 + (c_p/umolar)**0.61) : 1

dextra_ampar/dt = (1/(tau_ampa*second)) * (omega_ampa - extra_ampar) : 1 (clock-driven)

# g_ampa vary during simulation, thus it is estimated each time
# depending on extra_ampar value
g_ampa = g_ampa_min + (extra_ampar*(g_ampa_max-g_ampa_min)) : siemens
Iampa = (g_ampa * m_ampa * (v_syn- V_ampa)) : amp
Igaba = (g_gaba * m_gaba * (v_syn- V_gaba)) : amp


g_nmda_synapse : siemens
g_nmda_extrasynapse : siemens
BV1 = 1 / (1+exp(-0.062*v_dendrite/mV)*(Mg2/3.57)) : 1
BV2 = 1 / (1+exp(-0.062*v_syn/mV)*(Mg2/3.57)) : 1
Inmdaextra = (BV1 * g_nmda_extrasynapse * m_nmdaextra * (v_dendrite- V_nmda)) : amp
Inmda = (BV2 * g_nmda_synapse * m_nmda * (v_syn- V_nmda)) : amp
I_syn = (dendrite_spine_density)*(-Inmdaextra-Iampa-Inmda-Igaba) : amp/metre**2

# Postsynaptic calcium comcentration
theta=bt*K_bt/((K_bt+c_p_c)**2) : 1
noc = binomial_dis() : 1 (constant over dt)
i_Ca=g_Ca*noc*(v_syn  -E_Ca)*int(v_syn>=(-30*mV)) : amp
dc_p/dt= ( -frac_nmdar*Inmda/(2.0*F_p*vspine) -ks*(c_p_c-c_p_rest) -i_Ca/(2.0*F_p*vspine))/(1.0+theta)  : mmolar (clock-driven)
# clip the value to avoid anormal behavior
c_p_c = clip(c_p, 0*mmolar, 10*mmolar) : mmolar
'''

# Synaptic activity
# triggered by events from neuronal population
# (in this simulation, it's always triggered)
synapses_action = '''
g =  (1-int(index/(All_n*ratio_exci)))*G_pre*mmolar
gaba = (int(index/(All_n*ratio_exci)))* G_pre*mmolar
'''
Input_syn1 = '''
I_synapse1_post = I_syn : amp/metre**2 (summed)
extra_a1_post = extra_ampar : 1 (summed)
'''
Input_syn2 = '''
I_synapse2_post = I_syn : amp/metre**2 (summed)
extra_a2_post = extra_ampar : 1 (summed)
'''

# Synapses between Input neurons and first layer
exc_input= Synapses(P, neurons, model=synapses_eqs + Input_syn1,
                    on_pre=synapses_action,
                    on_event={'pre': 'custom_event'},
                    method=method,
                    method_options=method_options,
                    dt=delta_t)
# random attribution of 3 incoming source of connexion
# [P input from previous layer, P input from exc neuron, P input from inh neuron]
in_to_exc = np.random.multinomial(max_in, [0.8, 0.1, 0.1], N_sector)
in_to_exc2 = np.random.multinomial(max_in, [0.8, 0.1, 0.1], N_sector*2)
in_to_exc = np.concatenate((in_to_exc, in_to_exc2), axis=0)

# For inhibitory interneuron, connexion are only made within the same population
# [P input from exc neuron, P input from inh neuron]
in_to_ini = np.random.multinomial(max_in, [0.8, 0.2], N_sector*nb_path*nb_layer)
income = np.arange(N_e)
to_neuron = exc_n_indice[(exc_n_indice >= 0)  & (exc_n_indice < N_sector)]
preneuron, postneuron = get_connection_mat(income, to_neuron, in_to_exc[to_neuron, 0])

for i in range(1, nb_path):
    income = np.arange(N_e*i,N_e+N_e*i)
    to_neuron = exc_n_indice[(exc_n_indice >= (N_sector*i))  & (exc_n_indice < (N_sector+N_sector*i))]
    preneuron2, postneuron2 = get_connection_mat(income, to_neuron, in_to_exc[to_neuron, 0])
    preneuron, postneuron = np.concatenate((preneuron, preneuron2), axis=0), np.concatenate((postneuron, postneuron2), axis=0)

exc_input.connect(i=preneuron, j=postneuron)
exc_input.c_p = 0.1*umolar
exc_input.neuron_index_syn = postneuron
# with astrocytes
exc_input.g_nmda_synapse = g_nmda_max
exc_input.g_nmda_extrasynapse = g_nmda_max
# without astrocytes
# exc_input.g_nmda_synapse = 2*g_nmda_max
# exc_input.g_nmda_extrasynapse = 0*g_nmda_max
all_syn = Synapses(neurons, neurons, model=synapses_eqs + Input_syn2,
                   on_pre=synapses_action,
                   on_event={'pre': 'custom_event'},
                   method=method,
                   method_options=method_options,
                   dt=delta_t)
# exc syn to neurons by path/layer
for nlayer in range(nb_layer):
    for npath in range(nb_path):
        sector_start = N_sector * npath + N_sector * nb_path * nlayer
        sector_end = N_sector * npath + N_sector * nb_path * nlayer + N_sector
        # from previous layer
        if nlayer > 0:
            income = exc_n_indice[(exc_n_indice >= (sector_start - N_sector * nb_path )) & (
                exc_n_indice < (sector_end - N_sector * nb_path ))]
            to_neuron = exc_n_indice[(exc_n_indice >= sector_start) & (exc_n_indice < sector_end)]
            pr2, pst2 = get_connection_mat(income, to_neuron, in_to_exc[to_neuron, 0])

            pr, pst = np.concatenate((pr, pr2), axis=0), np.concatenate((pst, pst2), axis=0)


        # from exc, same layer
        to_neuron = all_neurons[(all_neurons >= sector_start) & (all_neurons < sector_end)]
        ne = exc_n_indice[(exc_n_indice >= sector_start) & (exc_n_indice < sector_end)]
        ni = ini_n_indice[(ini_n_indice >= sector_start) & (ini_n_indice < sector_end)]
        income = exc_n_indice[(exc_n_indice >= sector_start) & (exc_n_indice < sector_end)]
        pr2, pst2 = get_connection_mat(income, to_neuron,
                                     np.concatenate((in_to_exc[ne, 1],
                                                     in_to_ini[ni, 0]), axis=0))
        if nlayer==0 and npath ==0:
            pr, pst = pr2, pst2
        else:
            pr, pst = np.concatenate((pr, pr2), axis=0), np.concatenate((pst, pst2), axis=0)
        # from ini, same layer
        income = ini_n_indice[(ini_n_indice >= sector_start) & (ini_n_indice < sector_end)]
        pr2, pst2 = get_connection_mat(income, to_neuron,
                                     np.concatenate((in_to_exc[ne, 2],
                                                    in_to_ini[ni, 1]), axis=0))
        pr, pst = np.concatenate((pr, pr2), axis=0), np.concatenate((pst, pst2), axis=0)
all_syn.connect(i=pr, j=pst)
all_syn.c_p = 0.1*umolar
all_syn.neuron_index_syn = pst
# with astrocytes
all_syn.g_nmda_synapse = g_nmda_max
all_syn.g_nmda_extrasynapse = g_nmda_max
# without astrocytes
# all_syn.g_nmda_synapse = 2*g_nmda_max
# all_syn.g_nmda_extrasynapse = 0*g_nmda_max

# Connect excitatory synapses to an astrocyte depending on the position of the
# post-synaptic neuron
N_rows_a = int( (N_e+N_i)*nb_path )
N_cols_a = nb_layer
grid_dist = size / N_rows_a

### Astrocyte equations
astro_eqs = '''
# Fraction of activated astrocyte receptors:
dGamma_A/dt = O_N * Y_S * (1 - Gamma_A) -
              Omega_N*(1 + zeta * C/(C + K_KC)) * Gamma_A : 1

# IP_3 dynamics:
dI/dt = J_beta + J_delta - J_3K - J_5P + J_coupling     : mmolar
J_beta = O_beta * Gamma_A                         : mmolar/second
J_delta = O_delta/(1 + I/kappa_delta) *
                         C**2/(C**2 + K_delta**2) : mmolar/second
J_3K = O_3K * C**4/(C**4 + K_D**4) * I/(I + K_3K) : mmolar/second
J_5P = Omega_5P*I                                 : mmolar/second
delta_I_bias = I - I_bias : mmolar
J_ex = -F_ex/2*(1 + tanh((abs(delta_I_bias) - I_Theta)/omega_I)) *
                sign(delta_I_bias)                : mmolar/second
I_bias                                            : mmolar (constant)
# Diffusion between astrocytes:
J_coupling                                                       : mmolar/second

# Ca^2+-induced Ca^2+ release:
dC/dt = J_r + J_l - J_p                : mmolar
# IP3R de-inactivation probability
dh/dt = (h_inf - h_clipped)/tau_h *
        (1 + noise*xi*tau_h**0.5)      : 1
h_clipped = clip(h,0,1)                : 1
J_r = (Omega_C * m_inf**3 * h_clipped**3) *
      (C_T - (1 + rho_A)*C)            : mmolar/second
J_l = Omega_L * (C_T - (1 + rho_A)*C)  : mmolar/second
J_p = O_P * C**2/(C**2 + K_P**2)       : mmolar/second
m_inf = I/(I + d_1) * C/(C + d_5)      : 1
h_inf = Q_2/(Q_2 + C)                  : 1
tau_h = 1/(O_2 * (Q_2 + C))            : second
Q_2 = d_2 * (I + d_1)/(I + d_3)        : mmolar

# Fraction of gliotransmitter resources available for release:
dx_A/dt = Omega_A * (1 - x_A) : 1
# gliotransmitter concentration in the extracellular space:
dG_A/dt = -Omega_e*G_A        : mmolar

# Neurotransmitter concentration in the extracellular space
Y_S1     : mmolar
Y_S2     : mmolar
Y_S = Y_S1 + Y_S2 : mmolar
# Noise flag
noise   : 1 (constant)
# The astrocyte position in space
x : meter (constant)
y : meter (constant)
index1 : integer (constant)
neuron_index : integer (constant)
'''
glio_release = '''
G_A += rho_e * G_T * U_A * x_A
x_A -= U_A *  x_A 
'''
astrocytes = NeuronGroup(N_a, astro_eqs, method='milstein',
                         dt = delta_t,
                         threshold='C>C_Theta',
                         reset=glio_release)
# Arrange astrocytes in a grid
astrocytes.x = '(i // N_rows_a)*grid_dist - N_rows_a/2.0*grid_dist'
astrocytes.y = '(i % N_rows_a)*grid_dist - N_cols_a/2.0*grid_dist'
astrocytes.G_A = 0.001*mmolar
indices_a = astrocytes.i[:]
np.random.shuffle(indices_a)
astrocytes.index1 = indices_a

# Initialization
astrocytes.C = 0.1*umolar
astrocytes.h = 0.9
astrocytes.I = 0.16*umolar
astrocytes.x_A = 1.0
astrocytes.noise = 1
astrocytes.neuron_index[:] = np.arange(All_n)
array_a = preneuron
array_s = np.arange(postneuron.shape[0])

# With astro
#'''G_A_exc_post = G_A_pre : mmolar (summed)'''
# Without astro
#'''G_A_exc_post = 0*G_A_pre : mmolar (summed)'''
ecs_astro_to_inputsyn = Synapses(astrocytes, exc_input,
                                 '''G_A_exc_post = G_A_pre : mmolar (summed)''',
                                 method=method,
                                 method_options=method_options,
                                 dt=delta_t
                                 )
ecs_astro_to_inputsyn.connect('neuron_index_pre == neuron_index_syn_post')
inputsyn_to_astro = Synapses(exc_input, astrocytes,
                            'Y_S1_post = g_pre/max_in : mmolar (summed)',
                            method=method,
                            method_options=method_options,
                            dt=delta_t)
inputsyn_to_astro.connect('neuron_index_syn_pre == neuron_index_post')
#
ecs_astro_to_syn = Synapses(astrocytes, all_syn,
                                 '''G_A_exc_post = G_A_pre : mmolar (summed)''',
                            method=method,
                            method_options=method_options,
                            dt=delta_t)
ecs_astro_to_syn.connect('neuron_index_pre == neuron_index_syn_post')
ecs_syn_to_astro = Synapses(all_syn, astrocytes,
                            'Y_S2_post = g_pre/max_in : mmolar (summed)')
ecs_syn_to_astro.connect('neuron_index_syn_pre == neuron_index_post')
# Diffusion between astrocytes
astro_to_astro_eqs = '''
delta_I = I_post - I_pre            : mmolar
J_coupling_post = -(1 + tanh((abs(delta_I) - I_Theta)/omega_I))*
                  sign(delta_I)*F/2 : mmolar/second (summed)
'''
astro_to_astro = Synapses(astrocytes, astrocytes,
                          model=astro_to_astro_eqs,
                          method=method,
                          method_options=method_options,
                          dt=delta_t)
# Connect to all astrocytes less than 75um away
# (about 4 connections per astrocyte)
astro_to_astro.connect('neuron_index_pre != neuron_index_post and '
                       'sqrt((x_pre-x_post)**2 +'
                       '     (y_pre-y_post)**2) < 25*um')

# Monitor object for variables
C1 = StateMonitor(astrocytes, 'C', record=True)
vpre = StateMonitor(P, 'v', record=True)
v1 = StateMonitor(neurons, 'v', record=True)
ampar = StateMonitor(neurons, 'extra_ampar_average', record=True)
exc1 = exc_n_indice[(exc_n_indice>0*N_sector) & (exc_n_indice<1*N_e)]
exc2 = exc_n_indice[(exc_n_indice>1*N_sector) & (exc_n_indice<2*N_sector)]
exc3 = exc_n_indice[(exc_n_indice>2*N_sector) & (exc_n_indice<3*N_sector)]

ini1 = ini_n_indice[(ini_n_indice>0*N_sector) & (ini_n_indice<1*N_sector)]
ini2 = ini_n_indice[(ini_n_indice>1*N_sector) & (ini_n_indice<2*N_sector)]
ini3 = ini_n_indice[(ini_n_indice>2*N_sector) & (ini_n_indice<3*N_sector)]


pop0 = PopulationRateMonitor(P)
pop1 = PopulationRateMonitor(neurons[0 : N_sector])
pop2 = PopulationRateMonitor(neurons[N_sector : 2*N_sector])
pop3 = PopulationRateMonitor(neurons[2*N_sector : 3*N_sector])

M_sp = SpikeMonitor(neurons)
N_sp = SpikeMonitor(P)

M = SpikeMonitor(neurons)
# run simulation
run(total_duration*second, report='text')

# store results into npy arrays
# need to create the directory before
path = "test_noisy_noisemax/"+args["gnmda"]+"_"+args["stimfreq"]+"_"+args["run"]+"_"

# ignore the first 10 seconds, because astrocyte behavior is odd
cut = 0*second
all_dat = []

# Save Astrocytic Ca2+ level averaged by layer
all_dat.append(C1.t/second)
for layer in range(nb_layer):
    for sec in range(nb_path):
        all_dat.append(C1.C[N_sector*sec + N_sector*nb_path*layer : N_sector + N_sector*sec + N_sector*nb_path*layer, :].mean(axis=0)/nM)
all_dat = np.column_stack(all_dat)
np.save(path+'out_C', np.array(all_dat))
np.save(path+'C_brut', C1.C/nM)

all_dat = []
# Save excitatory neuron's membrane potential
all_dat.append(v1.t/second)
all_dat.append(v1.v[exc1, :].mean(axis=0)/mV)
all_dat.append(v1.v[exc2, :].mean(axis=0)/mV)
all_dat.append(v1.v[exc3, :].mean(axis=0)/mV)

all_dat = np.column_stack(all_dat)
np.save(path+'out_ve', all_dat)

all_dat = []
# Save inhibitory neuron's membrane potential
all_dat.append(v1.t/second)
all_dat.append(v1.v[ini1, :].mean(axis=0)/mV)
all_dat.append(v1.v[ini2, :].mean(axis=0)/mV)
all_dat.append(v1.v[ini3, :].mean(axis=0)/mV)
all_dat = np.column_stack(all_dat)
np.save(path+'out_vi', all_dat)

all_dat = []
# Save input neuron's membrane potential
all_dat.append(vpre.t/second)
all_dat.append(vpre.v.mean(axis=0)/mV)
all_dat = np.column_stack(all_dat)
np.save(path+'out_pre', all_dat)


all_spike = []
# Save spike time with neuron id
tim = np.concatenate((N_sp.t/ms, M_sp.t/ms), 0)
spik = np.concatenate((N_sp.i,M_sp.i+100), 0)
all_spike.append(tim)
all_spike.append(spik)
all_spike = np.column_stack(all_spike)
np.save(path+'spike', np.array(all_spike))

# Save exc and ini id
excdat = []
excdat.append(exc1+100)
excdat.append(exc2+100)
excdat.append(exc3+100)
inidat = []
inidat.append(ini1+100)
inidat.append(ini2+100)
inidat.append(ini3+100)
excdat = np.hstack(excdat)
np.save(path+'excdat', excdat)
inidat = np.hstack(inidat)
np.save(path+'inidat', inidat)

# Save smoothed firing rate of populations
pop0 = pop0.smooth_rate(window='flat', width=25*ms)/Hz
pop1 = pop1.smooth_rate(window='flat', width=25*ms)/Hz
pop2 = pop2.smooth_rate(window='flat', width=25*ms)/Hz
pop3 = pop3.smooth_rate(window='flat', width=25*ms)/Hz

all_dat = []
all_dat.append(pop0)
all_dat.append(pop1)
all_dat.append(pop2)
all_dat.append(pop3)
all_dat = np.column_stack(all_dat)
np.save(path+'rates', all_dat)

# Save average AMPAR level by population
all_dat = []
for layer in range(nb_layer):
    for sec in range(nb_path):
        all_dat.append(ampar.extra_ampar_average[N_sector*sec + N_sector*nb_path*layer : N_sector + N_sector*sec + N_sector*nb_path*layer, :].mean(axis=0))
all_dat = np.column_stack(all_dat)
np.save(path+'ampar', np.array(all_dat))

all_dat = []

# Same for excitatory neurons
all_dat.append(ampar.t/second)
all_dat.append(ampar.extra_ampar_average[exc1, :].mean(axis=0))
all_dat.append(ampar.extra_ampar_average[exc2, :].mean(axis=0))
all_dat.append(ampar.extra_ampar_average[exc3, :].mean(axis=0))

all_dat = np.column_stack(all_dat)
np.save(path+'ampare', all_dat)

all_dat = []
all_dat.append(ampar.t/second)
# Same for inhibitory neurons
all_dat.append(ampar.extra_ampar_average[ini1, :].mean(axis=0))
all_dat.append(ampar.extra_ampar_average[ini2, :].mean(axis=0))
all_dat.append(ampar.extra_ampar_average[ini3, :].mean(axis=0))
all_dat = np.column_stack(all_dat)
np.save(path+'ampari', all_dat)

#plt.plot(M.t/ms, M.i, '.')
# delete the compiled program (it takes a lot of useless memory if not erased)
device.delete(force=True)

# Save stimulation pattern
np.save(path+"stim", stimu)

#print(stim_to_plot.shape, all_dat[0].shape)
# all_dat = np.asarray(all_dat)
# for ls in range(nb_layer*nb_path):
#     plt.plot(all_dat[0], (all_dat[ls+1]-all_dat[ls+1].min())/(all_dat[ls+1].max()-all_dat[ls+1].min())+ls)
#     plt.plot(all_dat[0], stim_to_plot[ls, :]/stim_to_plot[ls, :].max()+ls)


#plt.show()
#plt.show()