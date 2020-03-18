#!/usr/bin/env python
# coding: utf-8

# ## Model 1: Dopady obmedzenia mobility na šírenie vírusu Covid-19

# In[18]:


## Nahratie balikov, pouzitie Anaconda Jupyter notebook + updatovane baliky.
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})


# In[19]:


## Funkcia pre výpočet priemeru zo simulácií
def sumlist(x):
    tmp=x[0]
    for i in x[1:]:
        tmp=tmp+i
    return tmp/len(x)


# In[20]:


## Údaje k počtu obyvateľov na obec 
pop = pd.read_excel('./zdroje/munic_pop.xlsx')
pop_N = np.array(pop['popul'])


# In[21]:


## Priradenie GPS suradnic pre kazdu obec
def get_coors_long(x):
    return float(df_coords.loc[df_coords.IDN4.apply(str)==x,'long'])

def get_coors_lat(x):
    return float(df_coords.loc[df_coords.IDN4.apply(str)==x,'lat'])
df_coords=pd.read_excel('./zdroje/obce1.xlsx')
data_i=pop
data_i.loc[:,'long']=data_i.munic.apply(str).apply(get_coors_long)
data_i.loc[:,'lat']=data_i.munic.apply(str).apply(get_coors_lat)


# In[22]:


## Otvorenie OD (origin-destination) matice, ktora popisuje migracne pohyby
## obyvatelstva na zaklade realnych dat.
with open('./vystupy_model/OD_final.pickle','rb') as f:
    OD=pickle.load(f)


# In[ ]:


## Stav nákazy 15.3.2020, priradenie poctov k jednotlivym obciam

#Bratislava (I.-V.) – 27
##Martin – 7
#Malacky (Kostolište, Vysoká pri Morave) – 4
#Nitra – 3
#Nové Mesto nad Váhom – 3
#Košice (I.-IV.) – 2
#Banská Bystrica – 2
#Trnava – 2
#Senec – 2
#Nové Zámky – 2
#Dunajská streda (Hviezdoslavov) – 1
#Svidník (Giraltovce) – 1
#Partizánske – 1
#Partizánske (Veľké Uherce) – 1
##Bytča – 1
#Trenčín – 1
#Spišská Nová Ves – 1

nakazy_sk=pd.DataFrame({'kod':[529346,529346,529320 , 512036,508063,500011,506338,598186,508438,506745,508217,503011,
                    501433,527106,505315,505315,517461,505820,526355],
                      'pocet':[4,3,25,7,4,3,3,2,2,2,2,2,1,1,1,1,1,1,1]})

first_infections=np.zeros(2926)
for i in np.arange(nakazy_sk.shape[0]):
    first_infections[pop.munic==nakazy_sk.kod.iloc[i]]=nakazy_sk.pocet.iloc[i]
    
first_infections_original=first_infections


# In[ ]:


## Definicia simulacie (na zaklade verejne dostupneho kodu k SIR modelu)
## Hlavnym parametrom je public_trans (alfa), ktory kontroluje level mobility populacie.

def simul(public_trans):   
    N_k = pop.popul.to_numpy()          # Populacia
    locs_len = len(N_k)                 # Pocet obci
    SIR = np.zeros(shape=(locs_len, 3)) # Dataframe pre S - susceptible, I - infected, R - recovered na kazdy den.
    SIR[:,0] = N_k                      # Inicializacia susceptible ako celej populacie (nikto nie je imunny)

    SIR[:, 0] = SIR[:, 0] - first_infections
    SIR[:, 1] = SIR[:, 1] + first_infections     # infikovani presunuti do I skupiny

    ## Standardizacia na pomer
    row_sums = SIR.sum(axis=1)
    SIR_n = SIR / row_sums[:, np.newaxis]

    ## Inicializacia parametrov
    beta = 0.4 # "Transmission rate"
    gamma = 0.10 # "Recovery rate"
    R0 = beta/gamma # Reprodukcne cislo ("Basic reproduction number") - pocitame skor s pesimistickym scenarom.
    gamma_vec = np.full(locs_len, gamma)
    public_trans_vec = np.full(locs_len, public_trans)

    ## Vytvarame kopie matic
    SIR_sim = SIR.copy()
    SIR_nsim = SIR_n.copy()
    
    ## Prebiehame simulaciu
    infected_pop_norm = []
    susceptible_pop_norm = []
    recovered_pop_norm = []
    SIR_sim_arr=np.zeros((SIR_sim.shape[0],SIR_sim.shape[1],200))
    j=0
    for time_step in tqdm_notebook(range(200)):
        ## Transmission rate je na kazdu obec ina, prvotne data su z gamma distribucie
        beta_vec = np.random.gamma(beta, 2, locs_len)
        # Matice infekcii
        infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
        OD_infected = np.round(OD*infected_mat)
        # Pocet infikovanych cestujucich do kazdej obce (vratane zotrvania vo vlastnej obci)
        inflow_infected = OD_infected.sum(axis=0)
        inflow_infected = np.round(inflow_infected*public_trans_vec)
        # Nove infekcie na zaklade rychlosti sirenia (beta), a novych nakaz,
        # standardizovane na podiel
        new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
        new_recovered = gamma_vec*SIR_sim[:, 1]
        new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
        ## Novoinfikovani odchadzaju z kategorie S
        SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
        ## Novoinfikovani prichadzaju do kategorie I a z nej odchadzaju vylieceni
        SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered
        ## Vylieceni prichadzaju do kat. R
        SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
        SIR_sim = np.where(SIR_sim<0,0,SIR_sim)
        # Normalizacia
        row_sums = SIR_sim.sum(axis=1)
        SIR_nsim = SIR_sim / row_sums[:, np.newaxis]
        SIR_sim_arr[:,:,j]=SIR_sim
        j=j+1
        S = SIR_sim[:,0].sum()/N_k.sum()
        I = SIR_sim[:,1].sum()/N_k.sum()
        R = SIR_sim[:,2].sum()/N_k.sum()
        infected_pop_norm.append(I)
        susceptible_pop_norm.append(S)
        recovered_pop_norm.append(R)
    ## Vytvor konecnu maticu
    res = pd.DataFrame(list(zip(infected_pop_norm, susceptible_pop_norm, recovered_pop_norm)), columns = ['inf','sus','rec'])
    return res,SIR_sim_arr


# In[ ]:


## Data k poctu seniorov na obec pre scenar o ziadnej mobilite pre tuto populaciu
data_senior=pd.read_excel('./zdroje/OD_IFP/senior.xlsx')
data_senior.loc[:,'munic']=data_senior.munic.apply(lambda x: x[-6:]).apply(int)
data_senior=data_senior.sort_values(by=['munic'])


# In[ ]:


## Simulacia pre seniorov - vypnuta mobilita pre tuto populaciu. 
def simul_senior(public_trans):
    # Znizenie populacie o seniorov, ktori nebudu migrovat v ramci obce podla tejto hypotezy
    N_k = pop.popul.to_numpy()-data_senior.senior.to_numpy()
    locs_len = len(N_k)                 
    SIR = np.zeros(shape=(locs_len, 3)) 
    SIR[:,0] = N_k                      

    SIR[:, 0] = SIR[:, 0] - first_infections
    SIR[:, 1] = SIR[:, 1] + first_infections

    row_sums = SIR.sum(axis=1)
    SIR_n = SIR / row_sums[:, np.newaxis]

    beta = 0.4
    gamma = 0.10
    R0 = beta/gamma
    gamma_vec = np.full(locs_len, gamma)
    public_trans_vec = np.full(locs_len, public_trans)

    SIR_sim = SIR.copy()
    SIR_nsim = SIR_n.copy()
    
    infected_pop_norm = []
    susceptible_pop_norm = []
    recovered_pop_norm = []
    SIR_sim_arr=np.zeros((SIR_sim.shape[0],SIR_sim.shape[1],200))
    j=0
    for time_step in tqdm_notebook(range(200)):
        beta_vec = np.random.gamma(beta, 2, locs_len)
        infected_mat = np.array([SIR_nsim[:,1],]*locs_len).transpose()
        OD_infected = np.round(OD*infected_mat)
        inflow_infected = OD_infected.sum(axis=0)
        inflow_infected = np.round(inflow_infected*public_trans_vec)
        new_infect = beta_vec*SIR_sim[:, 0]*inflow_infected/(N_k + OD.sum(axis=0))
        new_recovered = gamma_vec*SIR_sim[:, 1]
        new_infect = np.where(new_infect>SIR_sim[:, 0], SIR_sim[:, 0], new_infect)
        SIR_sim[:, 0] = SIR_sim[:, 0] - new_infect
        SIR_sim[:, 1] = SIR_sim[:, 1] + new_infect - new_recovered
        SIR_sim[:, 2] = SIR_sim[:, 2] + new_recovered
        SIR_sim = np.where(SIR_sim<0,0,SIR_sim)
        row_sums = SIR_sim.sum(axis=1)
        SIR_nsim = SIR_sim / row_sums[:, np.newaxis]
        SIR_sim_arr[:,:,j]=SIR_sim
        j=j+1
        ## Pridanie seniorov do celkovej populacie v tomto kroku pre spravny vypocet incidencie ochorenia
        S = SIR_sim[:,0].sum()/(N_k+data_senior.senior.to_numpy()).sum()
        I = SIR_sim[:,1].sum()/(N_k+data_senior.senior.to_numpy()).sum()
        R = SIR_sim[:,2].sum()/(N_k+data_senior.senior.to_numpy()).sum()
        infected_pop_norm.append(I)
        susceptible_pop_norm.append(S)
        recovered_pop_norm.append(R)
        
    res = pd.DataFrame(list(zip(infected_pop_norm, susceptible_pop_norm, recovered_pop_norm)), columns = ['inf','sus','rec'])
    return res,SIR_sim_arr


# In[ ]:


## Histogram R0 - distribucia Reproduction number
N_k = pop.popul.to_numpy()
locs_len = len(N_k) 
beta = 0.4
gamma = 0.10
R0 = beta/gamma
beta_vec = np.random.gamma(beta, 2, locs_len)
R0_vec = beta_vec / gamma

plt.hist(R0_vec, normed=True, bins=25)
plt.ylabel('Probability')
plt.savefig('./plots/Histogram_R0')


# In[ ]:


## Inicializacia zoznamov, ktore budu zaplnene v simulacii
percSIR_high=[]
percSIR_med=[]
percSIR_low=[]
percSIR_low_senior=[]

SIR_high=[]
SIR_med=[]
SIR_low=[]
SIR_low_senior=[]


# In[ ]:


for sim in np.arange(50):
    ## Uprava prvych infekcii na zaklade odhadhovaneho realnu poctu nakaz,
    ## sirsia diskusia v paperi. 
    first_infections=first_infections_original*6
    
    # Simulacia pre scenare vysoka mobilita (1), stredna mobilita (0.5) a nizka mobilita (0.3)
    # Posledny scenar pre nulovu mobilitu pre seniorov.
    a_high,b_high = simul(public_trans = 1)
    a_med,b_med = simul(public_trans = 0.5)
    a_low,b_low  = simul(public_trans = 0.3)
    a_low_senior,b_low_senior  = simul_senior(public_trans = 0.3)
    
    percSIR_high.append(a_high)
    SIR_high.append(b_high)
    percSIR_med.append(a_med)
    SIR_med.append(b_med)
    percSIR_low.append(a_low)
    SIR_low.append(b_low)
    percSIR_low_senior.append(a_low_senior)
    SIR_low_senior.append(b_low_senior)


# In[ ]:


## Ulozenie suboru simulacii
#with open('./vystupy_model/simulations.pickle','wb') as f:
#    pickle.dump([percSIR_high,percSIR_med,percSIR_low,SIR_high,SIR_med,SIR_low],f)


# In[24]:


## Otvorenie ulozeneho suboru simulacii
with open('./vystupy_model/simulations_17.3.2020.pickle','rb') as f:
    percSIR_high,percSIR_med,percSIR_low,SIR_high,SIR_med,SIR_low=pickle.load(f)
    f.close()


# In[26]:


## Porovnanie peakov podla mobility, relativne cisla graf
if True:
    x = np.arange(1,201)
    plt.rcParams['axes.facecolor']='white'
    for data in (percSIR_high):
        plt.plot(x,data.inf[0:200] ,c='red',alpha=0.7)
        plt.xlim((0, 200))
        plt.ylim((0, 0.5))
    for data in (percSIR_med):
        plt.plot(x,data.inf[0:200] ,c='orange',alpha=0.7)
        plt.xlim((0, 200))
        plt.ylim((0, 0.5))
    for data in (percSIR_low):
        plt.plot(x,data.inf[0:200] ,c='green',alpha=0.7)
        plt.xlim((0, 200))
        plt.ylim((0, 0.5))
    for data in (percSIR_low_senior):
        plt.plot(x,data.inf[0:200] ,c='blue',alpha=0.7)
        plt.xlim((0, 200))
        plt.ylim((0, 0.5))
        
    plt.title('Porovnanie peaku infekcie podľa mobility')
    plt.xlabel('Dni')
    plt.ylabel('Pomer nakazených')
    plt.savefig('./plots/plot_main_plus_senior2.png',dpi=300)
    plt.close


# In[ ]:


## Denny narast poctu infikovanych v absolutnych cislach - graf
if True:
    x = np.arange(1,200)
    plt.rcParams['axes.facecolor']='white'
    for data in [sumlist(SIR_high)[:,1,:].sum(0)]:
        plt.plot(x,data[1:100]-data[0:99] ,c='red',alpha=1,linewidth=3)
        plt.xlim((0, 100))
        plt.ylim((0, 550000))
    for data in [sumlist(SIR_med)[:,1,:].sum(0)]:
        plt.plot(x,data[1:100]-data[0:99] ,c='orange',alpha=1,linewidth=3)
        plt.xlim((0, 100))
        plt.ylim((0, 550000))
    for data in [sumlist(SIR_low)[:,1,:].sum(0)]:
        plt.plot(x,data[1:100]-data[0:99] ,c='green',alpha=1,linewidth=3)
        plt.xlim((0, 100))
        plt.ylim((0, 550000))
    plt.title('Denný nárast počtu infikovaných')
    plt.xlabel('Dni')
    plt.ylabel('Počet nových nákaz')

    plt.subplots_adjust(left = 0.155)
    plt.savefig('./plots/plot3_main.png',dpi=300)
    plt.close


# In[ ]:


## Graf pre vsetky vs. zachytene pripady virusu
data_uk=pd.DataFrame({'Všetky prípady':np.concatenate([np.array([1,3,5,7,7,10,21,32,44]),sumlist(SIR_low)[:,1,:].sum(0)]),
              'Zachytené prípady':np.concatenate([[0,0,0,0,0],np.array([1,3,5,7,7,10,21,32,44]),sumlist(SIR_low)[:,1,:].sum(0)[:-5]])})

x=np.arange(0,109)
plt.plot(x,data_uk['Zachytené prípady'] ,c='orange',alpha=1,linewidth=3)
plt.plot(x,data_uk['Všetky prípady'] ,c='red',alpha=1,linewidth=3)
plt.xlim(0,45)
plt.ylim(0,10000)
plt.xlabel('Dni')

plt.legend(['Zachytené prípady', 'Všetky prípady'])

plt.ylabel('Počet')
plt.savefig('./plots/zname_nezname.png',dpi=300)


# In[ ]:


## Ulozenie a export priemernej hodnoty zo 100 simulacii
pd.DataFrame(sumlist(SIR_low)[:,1,:]).to_csv('./results/I_low.csv')
pd.DataFrame(sumlist(SIR_low)[:,0,:]).to_csv('./results/S_low.csv')
pd.DataFrame(sumlist(SIR_low)[:,2,:]).to_csv('./results/R_low.csv')

pd.DataFrame(sumlist(SIR_med)[:,1,:]).to_csv('./results/I_med.csv')
pd.DataFrame(sumlist(SIR_med)[:,0,:]).to_csv('./results/S_med.csv')
pd.DataFrame(sumlist(SIR_med)[:,2,:]).to_csv('./results/R_med.csv')

pd.DataFrame(sumlist(SIR_high)[:,1,:]).to_csv('./results/I_high.csv')
pd.DataFrame(sumlist(SIR_high)[:,0,:]).to_csv('./results/S_high.csv')
pd.DataFrame(sumlist(SIR_high)[:,2,:]).to_csv('./results/R_high.csv')

percSIR_high_avg=sumlist(percSIR_high)
percSIR_med_avg=sumlist(percSIR_med)
percSIR_low_avg=sumlist(percSIR_low)


# In[ ]:


## Mapa sirenia infekcie po obciach (vysledkom je 100 obrazkov, z ktorych
## je mozne vytvorit animaciu)
## Scenar vysokej mobility
for jj in np.arange(100):
    day='day'+str(jj+1)
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = data_i.long,
        lat = data_i.lat,
        hoverinfo = 'text',
        text = 'Cases :'+(data_i.loc[:,day].apply(str)),
        mode = 'markers',
        marker = dict(
            size=np.log2(data_i.loc[:,day]+1),
            color = 'red',
            line = dict(
                width = 2,
                color = 'rgba(68, 68, 68, 0)'
            )
        )))

    fig.update_layout(
        title_text = day+'; infected=' + str(data_i.loc[:,day].sum()),
        showlegend = False,

        geo = go.layout.Geo(

            scope = 'europe',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
            lonaxis = go.layout.geo.Lonaxis(
                range= [ 16.5, 23 ]
            ),
            lataxis = go.layout.geo.Lataxis(
                range= [ 47.5, 50 ]
            ),
            domain = go.layout.geo.Domain(
                x = [ 0, 1 ],
                y = [ 0, 1 ]
            )
        )
    )
    fig.write_image("./gif/high_"+day+".png")


# In[ ]:


## Mapa sirenia infekcie po obciach
## Scenar strednej mobility

for i in np.arange(1,101):
    data_i[('day'+str(i))]=sumlist(SIR_med)[:,1,i-1]
data_i.iloc[:,4:]=data_i.iloc[:,4:].apply(np.floor)

for jj in np.arange(100):
    day='day'+str(jj+1)
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = data_i.long,
        lat = data_i.lat,
        hoverinfo = 'text',
        text = 'Cases :'+(data_i.loc[:,day].apply(str)),
        mode = 'markers',
        marker = dict(
            size=np.log2(data_i.loc[:,day]+1),
            color = 'orange',
            line = dict(
                width = 2,
                color = 'rgba(68, 68, 68, 0)'
            )
        )))

    fig.update_layout(
        title_text = day+'; infected=' + str(data_i.loc[:,day].sum()),
        showlegend = False,

        geo = go.layout.Geo(

            scope = 'europe',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
            lonaxis = go.layout.geo.Lonaxis(
                range= [ 16.5, 23 ]
            ),
            lataxis = go.layout.geo.Lataxis(
                range= [ 47.5, 50 ]
            ),
            domain = go.layout.geo.Domain(
                x = [ 0, 1 ],
                y = [ 0, 1 ]
            )
        )
    )
    fig.write_image("./gif/med_"+day+".png")




# In[ ]:


## Mapa sirenia infekcie po obciach
## Scenar strednej mobility
for i in np.arange(1,101):
    data_i[('day'+str(i))]=sumlist(SIR_low)[:,1,i-1]
data_i.iloc[:,4:]=data_i.iloc[:,4:].apply(np.floor)
for jj in np.arange(100):
    day='day'+str(jj+1)
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        locationmode = 'country names',
        lon = data_i.long,
        lat = data_i.lat,
        hoverinfo = 'text',
        text = 'Cases :'+(data_i.loc[:,day].apply(str)),
        mode = 'markers',
        marker = dict(
            size=np.log2(data_i.loc[:,day]+1),
            color = 'green',
            line = dict(
                width = 2,
                color = 'rgba(68, 68, 68, 0)'
            )
        )))


    fig.update_layout(
        title_text = day+'; infected=' + str(data_i.loc[:,day].sum()),
        showlegend = False,

        geo = go.layout.Geo(

            scope = 'europe',
            showland = True,
            landcolor = 'rgb(243, 243, 243)',
            countrycolor = 'rgb(204, 204, 204)',
            lonaxis = go.layout.geo.Lonaxis(
                range= [ 16.5, 23 ]
            ),
            lataxis = go.layout.geo.Lataxis(
                range= [ 47.5, 50 ]
            ),
            domain = go.layout.geo.Domain(
                x = [ 0, 1 ],
                y = [ 0, 1 ]
            )
        )
    )
    fig.write_image("./gif/low_"+day+".png")


# In[ ]:


## Tabulka s absolutnymi cislami o pocte infikovanych a vyliecenych pre vybrane dni
I_high=pd.read_csv('./results/I_high.csv').iloc[:,1:].sum(0)
I_med=pd.read_csv('./results/I_med.csv').iloc[:,1:].sum(0)
I_low=pd.read_csv('./results/I_low.csv').iloc[:,1:].sum(0)

R_high=pd.read_csv('./results/R_high.csv').iloc[:,1:].sum(0)
R_med=pd.read_csv('./results/R_med.csv').iloc[:,1:].sum(0)
R_low=pd.read_csv('./results/R_low.csv').iloc[:,1:].sum(0)

pd.DataFrame({'dni':np.arange(200),
              'I_high':I_high.to_numpy() ,
              'R_high':R_high.to_numpy(),
              'I_med':I_med.to_numpy() ,
              'R_med':R_med.to_numpy(),
              'I_low':I_low.to_numpy() ,
              'R_low':R_low.to_numpy()  
             }).to_excel('excel2.xlsx',engine='xlsxwriter')

I_high.iloc[[4,9,19,29,39,49,59,79,99,149,199]].to_numpy()


