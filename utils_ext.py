import pandas as pd
import numpy as np
import pickle
from sklearn.utils import shuffle
from tqdm import tqdm_notebook
from random import sample
from random import choices
from collections import defaultdict
import math
from random import random

import datetime as dt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def initiation_diffusion_no_age(pop,nakazy_sk, map_to_microregions, arr_all):
    first_infections=np.zeros(2926)
    for i in np.arange(nakazy_sk.shape[0]):
        first_infections[pop.munic==nakazy_sk.kod_obce.iloc[i]]=nakazy_sk.pocet.iloc[i]   

    #print("start sum", first_infections.sum())
    res=np.zeros(2926)
    for k in np.arange(2926):
        if (arr_all[:,:,2][:,k]).sum()>0:
            tmp=arr_all[:,:,2][:,k]
            res=res+((tmp/sum(tmp))*first_infections[k]*1.5 / 0.5)
    
    #print("diffused sum", res.sum())
    first_infections=res
    first_infections_tmp=np.concatenate([((np.tile([1.0],2926)) *( np.repeat(first_infections,1)) ),(np.repeat(0,261))])
    first_infections_micro=np.zeros(np.unique(map_to_microregions).shape[0])
    for i in np.unique(map_to_microregions):
        first_infections_micro[i]=first_infections_tmp[(np.array(map_to_microregions)==i)].sum()

    #print("merged sum before round", first_infections_micro.sum())
    first_infections_micro=np.array(list(map(probabilistic_round,first_infections_micro)))
    #print("merged sum after round", first_infections_micro.sum())
    return first_infections_micro

def probabilistic_round(x):
    return int(math.floor(x + random()))

def power_law(k_min, k_max, y, kappa):
    return ((k_max**(-kappa+1) - k_min**(-kappa+1))*y  + k_min**(-kappa+1.0))**(1.0/(-kappa + 1.0))

## Funkcie pre ziskavanie agregovanych vysledkov zo simulacie
def sumlist(x):
    tmp=x[0]
    if len(x)>1:
        for i in x[1:]:
            tmp=tmp+i
    return tmp/len(x)

def meanlist(x):
    tmp=x[0]
    if len(x)>1:
        for i in x[1:]:
            tmp=tmp+i
    return tmp/len(x)

def stdlist_inf(x):
    tmp=pd.DataFrame(x[0].inf)
    if len(x)>1:
        for i in x[1:]:
            tmp=pd.concat([tmp,pd.DataFrame(i.inf)],1)
    return tmp.apply(np.std,1)

def perclist_inf(x, p):
    tmp=pd.DataFrame(x[0].inf)
    if len(x)>1:
        for i in x[1:]:
            tmp=pd.concat([tmp,pd.DataFrame(i.inf)],1)
    return tmp.apply(lambda x: np.percentile(x, p), 1)

def perclist_exp(x, p):
    tmp=pd.DataFrame(x[0].inf)
    if len(x)>1:
        for i in x[1:]:
            tmp=pd.concat([tmp,pd.DataFrame(i.exp)],1)
    return tmp.apply(lambda x: np.percentile(x, p), 1)

def meanlist_sus(x):
    tmp=pd.DataFrame(x[0].sus)
    if len(x)>1:
        for i in x[1:]:
            tmp=pd.concat([tmp,pd.DataFrame(i.sus)],1)    
    return tmp.apply(np.mean,1)

def meanlist_inf(x):
    tmp=pd.DataFrame(x[0].inf)
    if len(x)>1:
        for i in x[1:]:
            tmp=pd.concat([tmp,pd.DataFrame(i.inf)],1)    
    return tmp.apply(np.mean,1)

def meanlist_exp(x):
    tmp=pd.DataFrame(x[0].inf)
    if len(x)>1:
        for i in x[1:]:
            tmp=pd.concat([tmp,pd.DataFrame(i.exp)],1)    
    return tmp.apply(np.mean,1)

def meanlist_rec(x):
    tmp=pd.DataFrame(x[0].rec)
    if len(x)>1:
        for i in x[1:]:
            tmp=pd.concat([tmp,pd.DataFrame(i.rec)],1)    
    return tmp.apply(np.mean,1)

def returning_infect(return_vec,N):
    tmp3=np.zeros(return_vec.shape[0])                                                                        
    tmp3[choices(np.arange(0,return_vec.shape[0]), weights=return_vec, k=N)]=1                                
    return tmp3

def load_population(fn='./zdroje int/munic_pop.xlsx'):
    pop = pd.read_excel(fn)
    pop_N = np.array(pop['popul'])
    N_popul = pop.popul.to_numpy()
    return pop, pop_N, N_popul

def OD_mat_load(path):
    with open(path,'rb') as f:
        OD=pickle.load(f)
        f.close()
    return OD

def OD_norm(OD,N_popul,warn=False):
    # copy here
    OD = np.array(OD)
    np.fill_diagonal(OD, 0)
    for j in range(OD.shape[0]):
        m = np.sum(OD[:, j]) / np.maximum(0.001, N_popul[j])
        if m > 1:
            if warn:
                print("bad column", j, "popul", N_popul[j], "OD sum", np.sum(OD[:,j]))
            OD[:, j] = OD[:, j] / m
    return OD 

def load_coords(fn, pop):
    df_coords=pd.read_excel(fn)
    def get_coors_long(x):
        return float(df_coords.loc[df_coords.IDN4.apply(str)==x,'long'])
    def get_coors_lat(x):
        return float(df_coords.loc[df_coords.IDN4.apply(str)==x,'lat'])
    
    data_i=pop
    data_i.loc[:,'long']=data_i.munic.apply(str).apply(get_coors_long)
    data_i.loc[:,'lat']=data_i.munic.apply(str).apply(get_coors_lat)
    df_coords=df_coords.loc[df_coords.IDN4.isin(data_i.munic),:]
    df_coords.index=np.arange(2926) 
    return df_coords, data_i

def load_first_infections(fn, locs_len, pop):
    nakazy_sk=pd.read_excel(fn, sheet_name = 'final')
    first_infections=np.zeros(locs_len)
    for i in np.arange(nakazy_sk.shape[0]):
        first_infections[pop.munic==nakazy_sk.kod_obce.iloc[i]]=nakazy_sk.pocet.iloc[i]
    return first_infections, nakazy_sk

def mrk_matrix_change(fn, pop, pop_N, data_i, OD, N_popul, alfa_to_mrk, alfa_from_mrk):
    mrk=pd.read_excel(fn)
    mrk.columns=['kod','coef_r0','pct']
    mrk.loc[:,'coef_r0']=mrk.coef_r0.apply(lambda x : (x+1)**(2.5)+1)
    mrk.loc[:,'pop_rmk']=0
    for k in np.arange(mrk.shape[0]):
        mrk.iloc[k,3]=(pop.loc[mrk.kod[k]==pop.munic,'popul']*mrk.pct[k]).iloc[0]

    mrk=mrk.loc[(mrk.pop_rmk>200),:]
    colonies=[]

    N_popul_mrk = N_popul.tolist()
    base_munic = [i for i in range(data_i.shape[0])]
    mrk_munic_list=[]
    for i in np.arange(data_i.shape[0]):
        if data_i.munic[i] in set(mrk.kod):
            colonies.append(data_i.munic[i])
            base_munic.append(i)
            mrk_munic_list.append(data_i.munic[i])
            OD=np.concatenate([OD,np.repeat(0,OD.shape[0]).reshape(OD.shape[0],1)],1)
            OD_pop_colony=(mrk.pct[mrk.kod==data_i.munic[i]].to_numpy()[0])*pop_N[i]
            OD[i,i]=(1-mrk.pct[mrk.kod==data_i.munic[i]].to_numpy()[0])*pop_N[i]
            OD=np.concatenate([OD,np.repeat(0,OD.shape[1]).reshape(1,OD.shape[1])],0)
            OD[-1,-1]=OD_pop_colony
            OD[-1,i]=OD_pop_colony*alfa_to_mrk
            OD[i,-1]=OD_pop_colony*alfa_from_mrk

            N_popul_mrk[i] -= OD_pop_colony
            N_popul_mrk.append(OD_pop_colony)

    ## Socio-ekonomicke faktory rizika
    R0_correction_demogr=[]
    for i in colonies:
        R0_correction_demogr.append(mrk.coef_r0[mrk.kod==i].iloc[0])

    R0_correction_demogr=np.array(R0_correction_demogr)
    beta_coef=np.repeat(1.0,2926+R0_correction_demogr.shape[0])
    beta_coef[2926:]=R0_correction_demogr
    print("added", len(colonies), "MRKs")
    return OD, beta_coef, np.array(N_popul_mrk), base_munic ,mrk_munic_list

def prepare_beta_list(U, k_min, k_max, kappa, gamma_factor, izol_beta_base):
    nodes_izol = round(U*1000000)
    nodes_neizol = 1000000 - nodes_izol
    beta_vec_neizol = np.zeros(nodes_neizol, float)
    for n in range(nodes_neizol):
        beta_vec_neizol[n] = power_law(k_min, k_max, np.random.uniform(0,1), kappa)
    
    # factor to skew gamma distribution towards low values with the same mean and smaller variance
    beta = izol_beta_base*gamma_factor
    scale = 1/gamma_factor

    beta_vec_izol = np.random.gamma(beta, scale, nodes_izol)

    # create joined vector from two distributions
    beta_unsorted = np.transpose([*np.transpose(beta_vec_izol),*np.transpose(beta_vec_neizol)])
    beta_vec = shuffle(beta_unsorted,random_state=0)
    return beta_vec.tolist(), beta_vec_izol, beta_vec_neizol

def adjust_beta(beta_new, beta_orig):
    m = beta_new / np.average(np.array(beta_orig))
    beta_list_new = m * np.array(beta_orig)
    #print('New beta average: ', np.average(beta_list_new))
    return beta_list_new.tolist()

def prob_rounding(x):
    integer_part = np.trunc(x)
    remainder = x - integer_part
    remainder_sampled = np.random.binomial(1, remainder)
    return integer_part + remainder_sampled

def simulation_SEIR(locs_len, N_popul, OD, alpha_vec, beta_mult, R0_correction_demogr, first_infections, beta_list, gamma,
               sigma, tau, T, return_vec_micro,E_start_ratio, new_infect_rounding_fn=lambda x: x):

    SEIR = np.zeros(shape=(locs_len, 4))
    SEIR[:, 0] = N_popul
    SEIR[:, 0] = SEIR[:, 0] - first_infections
    SEIR[:, 1] = first_infections * E_start_ratio
    SEIR[:, 2] = first_infections * (1 - E_start_ratio)

    row_sums = SEIR.sum(axis=1)
    SEIR_n = SEIR / row_sums[:, np.newaxis]
    gamma_vec = np.full(locs_len, gamma)
    SEIR_sim = SEIR.copy()
    SEIR_nsim = SEIR_n.copy()

    ## Simulacia
    susceptible_pop_norm = []
    exposed_pop_norm = []
    infected_pop_norm = []
    recovered_pop_norm = []

    # Pridanie 0-teho dna pre SEIR
    S = (N_popul.sum() - first_infections.sum()) / N_popul.sum()
    E = first_infections.sum() * E_start_ratio / N_popul.sum()
    I = first_infections.sum() * (1 - E_start_ratio) / N_popul.sum()
    R = 0 / N_popul.sum()
    susceptible_pop_norm.insert(0, S)
    exposed_pop_norm.insert(0, E)
    infected_pop_norm.insert(0, I)
    recovered_pop_norm.insert(0, R)

    ##############################################################################
    # Pridanie 0-teho dna pre obce
    SEIR_sim_arr = np.zeros([locs_len, 4, T + 1])
    SEIR_sim_arr[:, :, 0] = SEIR_sim
    w = 0
    # for time_step in tqdm_notebook(range(T)):
    # for time_step in tqdm.notebook.tqdm(range(T)):
    for time_step in tqdm_notebook(range(T)):
        alpha = alpha_vec[w]
        beta_vec = np.array(sample(beta_list, locs_len)) * R0_correction_demogr * beta_mult[w]

        # Pomer susceptible a infikovanych
        y = SEIR_sim[:, 0] / N_popul
        x = SEIR_sim[:, 2] / N_popul

        ## Clen 1
        outside_work = beta_vec * SEIR_sim[:, 0] * SEIR_sim[:, 2] / N_popul

        ## Clen 2
        during_work1 = np.zeros(locs_len)
        num_2 = np.zeros(locs_len)
        denominator_t = N_popul + OD.sum(axis=1) - OD.sum(axis=0)
        num_2 = (SEIR_sim[:, 0] - y * OD.sum(0)) * (
                    (x * beta_vec).dot(OD.T) + (SEIR_sim[:, 2] - x * OD.sum(0)) * beta_vec)
        ## Clen 3
        during_work1 = num_2 / denominator_t
        numerator_t = np.zeros(locs_len)
        numerator_t = ((SEIR_sim[:, 2] - x * OD.sum(0)) * beta_vec + (x * beta_vec).dot(OD.T))

        during_work2 = np.zeros(locs_len)
        during_work2 = y * np.sum(OD.transpose() * numerator_t / denominator_t, 1)

        # Assert, that everything is OK (this would indicate problem with OD matrix)
        assert np.min(outside_work) > -1e-8
        assert np.min(during_work1) > -1e-8
        assert np.min(during_work2) > -1e-8

        # And now fix numerical issues
        outside_work = np.maximum(0, outside_work)
        during_work1 = np.maximum(0, during_work1)
        during_work2 = np.maximum(0, during_work2)

#        print(np.sum(outside_work))
#        print(np.sum(during_work1))
#        print(np.sum(during_work2))

        # Total new exposed
        total_new_exposed = tau * outside_work + alpha * (1 - tau) * during_work1 + alpha * (1 - tau) * during_work2
        ## Nemoze byt viac exposed ako susceptible
        total_new_exposed = new_infect_rounding_fn(total_new_exposed) + returning_infect(return_vec_micro, 1)
        total_new_exposed = np.where(total_new_exposed > SEIR_sim[:, 0], SEIR_sim[:, 0], total_new_exposed)
        # New infected
        new_infected = sigma*SEIR_sim[:, 1]

        # Recovered
        new_recovered = gamma_vec * SEIR_sim[:, 2]

        ## Novoexposed odchadzaju z kategorie S
        SEIR_sim[:, 0] = SEIR_sim[:, 0] - total_new_exposed
        ## Novoexposed prichadzaju do kategorie E, odchadzaju prec ako infekcni
        SEIR_sim[:, 1] = SEIR_sim[:, 1] + total_new_exposed - new_infected
        ## Novoinfikovani prichadzaju do kategorie I (z E) a z nej odchadzaju vylieceni
        SEIR_sim[:, 2] = SEIR_sim[:, 2] + new_infected - new_recovered
        ## Vylieceni prichadzaju do kat. R
        SEIR_sim[:, 3] = SEIR_sim[:, 3] + new_recovered
        SEIR_sim = np.where(SEIR_sim < 0, 0, SEIR_sim)

        # Normalizacia
        row_sums = SEIR_sim.sum(axis=1)
        SEIR_nsim = SEIR_sim / row_sums[:, np.newaxis]

        SEIR_sim_arr[:, :, w] = SEIR_sim
        w = w + 1
        S = SEIR_sim[:, 0].sum() / N_popul.sum()
        E = SEIR_sim[:, 1].sum() / N_popul.sum()
        I = SEIR_sim[:, 2].sum() / N_popul.sum()
        R = SEIR_sim[:, 3].sum() / N_popul.sum()

        susceptible_pop_norm.append(S)
        exposed_pop_norm.append(E)
        infected_pop_norm.append(I)
        recovered_pop_norm.append(R)
#        break

    recovered_pop_norm = list(map(lambda x: x - (0) / sum(N_popul), recovered_pop_norm))
    SEIR_sim_arr[:, 3, :] = np.apply_along_axis(lambda x: x - (0), arr=SEIR_sim_arr[:, 3, :], axis=0)
    ## Vytvorenie konecnej matice
    res = pd.DataFrame(list(zip(susceptible_pop_norm, exposed_pop_norm, infected_pop_norm, recovered_pop_norm)),
                       columns=['sus', 'exp', 'inf', 'rec'])
    return res, SEIR_sim_arr

def simulation_SEIR_remove_age(locs_len, N_popul,N_popul_new, OD, alpha_vec, beta_mult, R0_correction_demogr,
                               first_infections, beta_list, gamma, sigma, tau, T, return_vec_micro,
                               E_start_ratio, new_infect_rounding_fn=lambda x: x):

    SEIR = np.zeros(shape=(locs_len, 4))
    SEIR[:, 0] = N_popul_new
    SEIR[:, 0] = SEIR[:, 0] - first_infections
    SEIR[:, 1] = first_infections * E_start_ratio
    SEIR[:, 2] = first_infections * (1 - E_start_ratio)
    SEIR[:, 3] = N_popul-N_popul_new

    row_sums = SEIR.sum(axis=1)
    SEIR_n = SEIR / row_sums[:, np.newaxis]
    gamma_vec = np.full(locs_len, gamma)
    SEIR_sim = SEIR.copy()
    SEIR_nsim = SEIR_n.copy()

    ## Simulacia
    susceptible_pop_norm = []
    exposed_pop_norm = []
    infected_pop_norm = []
    recovered_pop_norm = []

    # Pridanie 0-teho dna pre SEIR
    S = (N_popul_new.sum() - first_infections.sum()) / N_popul.sum()
    E = first_infections.sum() * E_start_ratio / N_popul.sum()
    I = first_infections.sum() * (1 - E_start_ratio) / N_popul.sum()
    R = (N_popul-N_popul_new).sum() /N_popul.sum()
    susceptible_pop_norm.insert(0, S)
    exposed_pop_norm.insert(0, E)
    infected_pop_norm.insert(0, I)
    recovered_pop_norm.insert(0, R)

    ##############################################################################
    # Pridanie 0-teho dna pre obce
    SEIR_sim_arr = np.zeros([locs_len, 4, T + 1])
    SEIR_sim_arr[:, :, 0] = SEIR_sim
    SEIR_sim_arr[:, 3, 0] = N_popul - N_popul_new

    w = 0
    # for time_step in tqdm_notebook(range(T)):
    # for time_step in tqdm.notebook.tqdm(range(T)):
    for time_step in tqdm_notebook(range(T)):
        alpha = alpha_vec[w]
        beta_vec = np.array(sample(beta_list, locs_len)) * R0_correction_demogr * beta_mult[w]

        # Pomer susceptible a infikovanych
        y = SEIR_sim[:, 0] / N_popul
        x = SEIR_sim[:, 2] / N_popul

        ## Clen 1
        outside_work = beta_vec * SEIR_sim[:, 0] * SEIR_sim[:, 2] / N_popul

        ## Clen 2
        during_work1 = np.zeros(locs_len)
        num_2 = np.zeros(locs_len)
        denominator_t = N_popul + OD.sum(axis=1) - OD.sum(axis=0)
        num_2 = (SEIR_sim[:, 0] - y * OD.sum(0)) * (
                (x * beta_vec).dot(OD.T) + (SEIR_sim[:, 2] - x * OD.sum(0)) * beta_vec)
        ## Clen 3
        during_work1 = num_2 / denominator_t
        numerator_t = np.zeros(locs_len)
        numerator_t = ((SEIR_sim[:, 2] - x * OD.sum(0)) * beta_vec + (x * beta_vec).dot(OD.T))

        during_work2 = np.zeros(locs_len)
        during_work2 = y * np.sum(OD.transpose() * numerator_t / denominator_t, 1)

        # Assert, that everything is OK (this would indicate problem with OD matrix)
        assert np.min(outside_work) > -1e-8
        assert np.min(during_work1) > -1e-8
        assert np.min(during_work2) > -1e-8

        # And now fix numerical issues
        outside_work = np.maximum(0, outside_work)
        during_work1 = np.maximum(0, during_work1)
        during_work2 = np.maximum(0, during_work2)

        #        print(np.sum(outside_work))
        #        print(np.sum(during_work1))
        #        print(np.sum(during_work2))

        # Total new exposed
        total_new_exposed = tau * outside_work + alpha * (1 - tau) * during_work1 + alpha * (1 - tau) * during_work2
        ## Nemoze byt viac exposed ako susceptible
        total_new_exposed = new_infect_rounding_fn(total_new_exposed) + returning_infect(return_vec_micro, 1)
        total_new_exposed = np.where(total_new_exposed > SEIR_sim[:, 0], SEIR_sim[:, 0], total_new_exposed)
        # New infected
        new_infected = sigma * SEIR_sim[:, 1]

        # Recovered
        new_recovered = gamma_vec * SEIR_sim[:, 2]

        ## Novoexposed odchadzaju z kategorie S
        SEIR_sim[:, 0] = SEIR_sim[:, 0] - total_new_exposed
        ## Novoexposed prichadzaju do kategorie E, odchadzaju prec ako infekcni
        SEIR_sim[:, 1] = SEIR_sim[:, 1] + total_new_exposed - new_infected
        ## Novoinfikovani prichadzaju do kategorie I (z E) a z nej odchadzaju vylieceni
        SEIR_sim[:, 2] = SEIR_sim[:, 2] + new_infected - new_recovered
        ## Vylieceni prichadzaju do kat. R
        SEIR_sim[:, 3] = SEIR_sim[:, 3] + new_recovered
        SEIR_sim = np.where(SEIR_sim < 0, 0, SEIR_sim)

        # Normalizacia
        row_sums = SEIR_sim.sum(axis=1)
        SEIR_nsim = SEIR_sim / row_sums[:, np.newaxis]

        SEIR_sim_arr[:, :, w] = SEIR_sim
        w = w + 1
        S = SEIR_sim[:, 0].sum() / N_popul.sum()
        E = SEIR_sim[:, 1].sum() / N_popul.sum()
        I = SEIR_sim[:, 2].sum() / N_popul.sum()
        R = SEIR_sim[:, 3].sum() / N_popul.sum()

        susceptible_pop_norm.append(S)
        exposed_pop_norm.append(E)
        infected_pop_norm.append(I)
        recovered_pop_norm.append(R)
    #        break

    #recovered_pop_norm = list(map(lambda x: x-(sum(N_popul)-sum(N_popul_new))/sum(N_popul), recovered_pop_norm))
    #SEIR_sim_arr[:, 3, :] = np.apply_along_axis(lambda x: x-(N_popul-N_popul_new), arr=SEIR_sim_arr[:, 3, :], axis=0)
    ## Vytvorenie konecnej matice
    res = pd.DataFrame(list(zip(susceptible_pop_norm, exposed_pop_norm, infected_pop_norm, recovered_pop_norm)),
                       columns=['sus', 'exp', 'inf', 'rec'])
    return res, SEIR_sim_arr

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    from math import radians, cos, sin, asin, sqrt
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def load_merges(fn, data_i, pop):
    # Preparing merge
    df_microregions = pd.read_excel(fn, 1)

    regions = defaultdict(list)
    munic_to_region = {}

    for x in df_microregions.itertuples():
        try:
            munic = int(x[8])
        except ValueError:   # tu su nejake mestske casti
            continue
        region = x[1]
        regions[region].append(munic)
        munic_to_region[munic] = region 

    # Add munic without region to closest region
    region_centers = {}

    for k, v in regions.items():
        lats = [data_i[data_i["munic"] == x]["lat"].values[0] for x in v]
        lons = [data_i[data_i["munic"] == x]["long"].values[0] for x in v]
        region_centers[k] = (np.mean(lats), np.mean(lons))

    for i in range(len(pop)):
        munic_id = pop.iloc[i]["munic"]
        if munic_id not in munic_to_region:
            lat = data_i[data_i["munic"] == munic_id]["lat"].values[0]
            lon = data_i[data_i["munic"] == munic_id]["long"].values[0]
            best_region = None
            best_dist = None
            for reg, (lat_r, lon_r) in region_centers.items():
                dist = haversine(lon_r, lat_r, lon, lat)
                if best_region is None or dist < best_dist:
                    best_region = reg
                    best_dist = dist
            print("Assigning", munic_id, "to", best_region)
            regions[best_region].append(munic_id)
            munic_to_region[munic_id] = best_region

    # Mergujeme podla mikroregionov
    merges = []
    for v in regions.values():
        merges.append(pop[pop.munic.isin(v)].index.values)

    return merges

def prepare_merge(merges, cur_size, expand_mult):
    with_cluster = set(x for cl in merges for x in cl)
    without_cluster = set()
    for i in range(cur_size):
        if i // expand_mult not in with_cluster:
            without_cluster.add(i // expand_mult)

    new_ids = [None for i in range(cur_size)]

    cur_cluster_id = 0

    for cl in merges:
        for x in cl:
            for j in range(expand_mult):
                new_ids[x*expand_mult+j] = cur_cluster_id*expand_mult + j
        cur_cluster_id += 1

    for x in without_cluster:
        for j in range(expand_mult):
            new_ids[x*expand_mult+j] = cur_cluster_id*expand_mult + j
        cur_cluster_id += 1

    assert all(x is not None for x in new_ids)
    return new_ids

def remove_age(N_popul,x,y,pct,data):
    y2=np.max([y,110])
    N_popul_new=N_popul-data.iloc[:,(x+5):(y2+5)].sum(1).to_numpy()*pct
    return N_popul_new

## Graf funkcia
def plot_peaks(dictagg,verzia_modelu, scen_vyber, legenda_vsetky_scenare,title,total_days,
               day_zero=dt.datetime.now() - dt.timedelta(days=0), SEIR = False):
    end = day_zero + dt.timedelta(days=total_days + 1)
    days = mdates.drange(day_zero, end, dt.timedelta(days=1))
    months = mdates.MonthLocator()
    sns.set(rc={'figure.figsize': (11, 4)})

    legenda = [legenda_vsetky_scenare[scen] for scen in scen_vyber]

    fig, ax = plt.subplots()
    clrs = ['red', 'purple', 'teal', 'blue', 'green', 'yellow', 'orange']

    with sns.axes_style("darkgrid"):
        for scen in scen_vyber:
            scenar = 'SIR_scenar' + str(scen)
            means = dictagg[scenar].inf
            p5 = dictagg[scenar].inf5
            p95 = dictagg[scenar].inf95
            if SEIR == True:
                means = means + dictagg[scenar].exp
                p5 = p5 + dictagg[scenar].exp5
                p95 = p95 + dictagg[scenar].exp95
            ax.plot(days, means, c=clrs[scen])
            ax.fill_between(days, p5, p95, alpha=0.3, facecolor=clrs[scen])
            # ax.set_xlim([dt.date(2020, 3, 29), dt.date(2021, 1, 8)])
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b' '%y'))
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel('Pomer nakazených')
    plt.legend(legenda)
    plt.tight_layout()

    path = './plots/' + str(verzia_modelu) + '.png'
    plt.savefig(path, dpi=300)
    plt.close

    i = 0
    for scen in scen_vyber:
        scenar = 'SIR_scenar' + str(scen)
        print('Scenar:', legenda[i], ', Peak:', np.round(np.max(dictagg[scenar].inf) * 100, 2), '%, Day: ',
              dictagg[scenar].inf.idxmax())
        i += 1


## Graf funkcia
def plot_peaks_log(dictagg,verzia_modelu, scen_vyber, legenda_vsetky_scenare,title,total_days,
                   day_zero=dt.datetime.now() - dt.timedelta(days=0), SEIR = True):

    end = day_zero + dt.timedelta(days=total_days + 1)
    days = mdates.drange(day_zero, end, dt.timedelta(days=1))
    months = mdates.MonthLocator()
    sns.set(rc={'figure.figsize': (11, 4)})

    legenda = [legenda_vsetky_scenare[scen] for scen in scen_vyber]

    fig, ax = plt.subplots()
    clrs = ['red', 'orange', 'purple', 'blue', 'green', 'yellow', 'teal']

    with sns.axes_style("darkgrid"):
        for scen in scen_vyber:
            scenar = 'SIR_scenar' + str(scen)
            means = dictagg[scenar].inf
            p5 = dictagg[scenar].inf5
            p95 = dictagg[scenar].inf95
            if SEIR == True:
                means = means + dictagg[scenar].exp
                p5 = p5 + dictagg[scenar].exp5
                p95 = p95 + dictagg[scenar].exp95
            ax.plot(days, means, c=clrs[scen])
            ax.fill_between(days, p5, p95, alpha=0.3, facecolor=clrs[scen])
            ax.set_ylim([0.00001, 0.1])
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b' '%y'))
            ax.set_yscale('log')
            formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
            ax.yaxis.set_major_formatter(formatter)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel('Pomer nakazených')
    plt.legend(legenda)
    plt.tight_layout()

    path = './plots/' + str(verzia_modelu) + '.png'
    plt.savefig(path, dpi=300)
    plt.close

    i = 0
    for scen in scen_vyber:
        scenar = 'SIR_scenar' + str(scen)
        print('Scenar:', legenda[i], ', Peak:', np.round(np.max(dictagg[scenar].inf) * 100, 2), '%, Day: ',
              dictagg[scenar].inf.idxmax())
        i += 1


## Graf funkcia lockdown
def plot_peaks_lockdown(dictagg,verzia_modelu, scen_vyber,legenda_vsetky_scenare,title,total_days,
                        day_zero=dt.datetime.now() - dt.timedelta(days=0), SEIR = True):
    end = day_zero + dt.timedelta(days=total_days + 1)
    days = mdates.drange(day_zero, end, dt.timedelta(days=1))
    months = mdates.MonthLocator()
    sns.set(rc={'figure.figsize': (11, 4)})

    legenda = [legenda_vsetky_scenare[scen] for scen in scen_vyber]

    fig, ax = plt.subplots()
    clrs = ['red', 'purple', 'teal', 'blue', 'green', 'yellow', 'orange']

    linest = '-'
    with sns.axes_style("darkgrid"):
        for scen in scen_vyber:
            if scen > 0:
                linest = '--'
            scenar = 'SIR_scenar' + str(scen)
            means = dictagg[scenar].inf
            p5 = dictagg[scenar].inf5
            p95 = dictagg[scenar].inf95

            if SEIR == True:
                means = means + dictagg[scenar].exp
                p5 = p5 + dictagg[scenar].exp5
                p95 = p95 + dictagg[scenar].exp95

            ax.plot(days, means, c=clrs[scen], linestyle = linest)
            ax.fill_between(days, p5, p95, alpha=0.3, facecolor=clrs[scen])
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b' '%y'))
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel('Pomer nakazených')
    plt.legend(legenda)
    plt.tight_layout()
    plt.axvline(x=day_zero + dt.timedelta(days=42), label='Lockdown 6 týždňov', color='purple')
    plt.axvline(x=day_zero + dt.timedelta(days=21), label='Lockdown 3 týždne', color='orange')

    path = './plots/' + str(verzia_modelu) + '.png'
    plt.savefig(path, dpi=300)
    plt.close

    i = 0
    for scen in scen_vyber:
        scenar = 'SIR_scenar' + str(scen)
        print('Scenar:', legenda[i], ', Peak:', np.round(np.max(dictagg[scenar].inf) * 100, 2), '%, Day: ',
              dictagg[scenar].inf.idxmax())
        i += 1
