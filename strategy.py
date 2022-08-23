import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


def Generate_Trajectory_BS(S_init, riskfree, div, vol, delta, matrice_alea, nbre_path = 10000):
    """
    Diffusion selon Black-Scholes
        Input:
            prix initial S_init
            riskfree
            div yield
            vol
            la variation de temps entre les dates de constatations
            la matrice aléatoire : matrice_alea : gauss * vol
            le nbre de scénario
            (le repo est nul pour la BCE)
    """
    
    diffusion_matrix = {}
    diffusion_init = np.zeros((nbre_path,1))
    diffusion_init[:] = S_init
    diffusion_init = pd.DataFrame(diffusion_init)
    diffusion_matrix[0] = diffusion_init
    
    int_nbre_date = matrice_alea.shape[1] + 1
    
    for j in range(1,int_nbre_date):
        #calcul de la diffusion
        diffusion_matrix[j] = diffusion_matrix[j-1] * ((riskfree[j] - div - 0.5*vol**2)*delta[j] + math.sqrt(delta[j])*pd.DataFrame(matrice_alea[:,j-1,:])).applymap(math.exp)
        
    return diffusion_matrix


def Generate_Trajectory_Heston(S_init,riskfree,div,timestep,V0,Eta,Theta,Kappa,Rho,nbre_path,Normal_Maxtrix_Spot,Normal_Maxtrix_Temp):
    """
    Diffusion selon le modèle d'Heston
        Dans le modèle, on simule la variance => modèle à volatilité stochastique
        
    Input:
        V0 = vol[0]**2
        Eta = vol de vol (ie) variance
        Theta = variance à l'infini, variance long terme
        Kappa = force de retour à la moyenne (ie) mean reversion
        Rho = corrélation entre les 2 lois normales (ie) le spot et la vol
        prix initial S_init
        riskfree
        div yield
        vol
        la variation de temps entre les dates de constatations
        la simulation aléatoire du spot matrice_alea
        la simulation aléatoire pour la vol
        le nbre de scénario
        (le repo est nul pour la BCE)
    
    """
    
    #timestep should contain the value 0 corresponding to the initial date
    dT=0.01 #pr une année qui contient 252 jours, cela correspon à ~2jours
    Dates=np.arange(0,timestep.iloc[-1],dT)
    Dates=sorted(list(set().union(Dates,timestep.values)));
    #Nb_Dates=length(Dates);

    #corrélation entre les 2 loi grâce à Rho
    Normal_Maxtrix_Vol=Rho*Normal_Maxtrix_Spot+math.sqrt(1-Rho**2)*Normal_Maxtrix_Temp;
    
    #diffusion spot de longueur len(Dates)
    diffusion_matrix = {}
    diffusion_init = np.zeros((nbre_path,1))
    diffusion_init[:] = S_init
    diffusion_init = pd.DataFrame(diffusion_init)
    diffusion_matrix[0] = diffusion_init
    
    #diffusion vol de longueur len(Dates)
    diffusion_var = {}
    diffusion_init_var = np.zeros((nbre_path,1))
    diffusion_init_var[:] = V0
    diffusion_init_var = pd.DataFrame(diffusion_init_var)
    diffusion_var[0] = diffusion_init_var

    Delta_t=pd.Series(Dates[1:]) - pd.Series(Dates[0:-1])
    
    #diffusion à retourner
    diffusion_matrix_return = {}
    diffusion_matrix_return[0] = diffusion_init
    diffusion_var_return = {}
    diffusion_var_return[0] = diffusion_init_var
    
    i=1;
    for j in range(1,len(Dates)):
        ##diffusion spot
        diffusion_matrix[j] = diffusion_matrix[j-1] * ((riskfree[j] - div - 0.5*diffusion_var[j-1])*Delta_t.iloc[j-1] + (Delta_t.iloc[j-1] * diffusion_var[j-1]).applymap(math.sqrt) * Normal_Maxtrix_Spot[:,j-1,:]).applymap(math.exp)
        
        ##diffusion vol
        diffusion_var[j] = diffusion_var[j-1] + Kappa * (Theta-diffusion_var[j-1])*Delta_t.iloc[j-1]+Eta*(diffusion_var[j-1]).applymap(math.sqrt)*Normal_Maxtrix_Vol[:,j-1,:]*math.sqrt(Delta_t.iloc[j-1])
        ispositive = diffusion_var[j] > 0
        diffusion_var[j] = diffusion_var[j] * ispositive + (~ispositive) * (diffusion_var[j-1] + Kappa * (Theta-diffusion_var[j-1])*Delta_t.iloc[j-1] + Eta*(diffusion_var[j-1]).applymap(math.sqrt)*(-Normal_Maxtrix_Vol[:,j-1,:])*math.sqrt(Delta_t.iloc[j-1]))

        
        if(Dates[j]==timestep[i]):
            diffusion_matrix_return[i]= diffusion_matrix[j]
            diffusion_var_return[i] = diffusion_var[j]
            i+=1
    
    dict_to_return = {}
    dict_to_return["diffusion"] = diffusion_matrix_return
    dict_to_return["variance"] = diffusion_var
    return dict_to_return


def Pricing_Anthemis(diffusion_matrix,coupon_du,strike,flt_barrier_bonus,DF,flt_bonus,capital_barrier,redemption_level):
    """
    Description:
        retourne le prix du Poruidt structuré
    
    """
    
    price = pd.DataFrame(np.zeros(diffusion_matrix[0].shape))

    #coupon dû (payé à maturité)
    coupon_infine = pd.DataFrame(np.zeros(diffusion_matrix[0].shape)) + coupon_du


    for t in range(1,len(diffusion_matrix)):
        #barrière de bonus
        isbonus = diffusion_matrix[t] >= strike*flt_barrier_bonus
        coupon_infine = coupon_infine + (isbonus)

        if t == (len(diffusion_matrix)-1): 
        #maturity case
            #payement des coupons dû à maturité
            price = price + DF[t] * coupon_infine * flt_bonus 
            #histo = histo + coupon_infine * flt_bonus

            #au dessus de la barrière de capital qd produit non expiré
            isabovecapital = diffusion_matrix[t] >= strike*capital_barrier
            price = price + DF[t] * redemption_level * (isabovecapital)
            #histo = histo + redemption_level * (isabovecapital)

            #en dessous du capital
            price = price + DF[t] * 100 * diffusion_matrix[t] / strike  * (~isabovecapital)
            #histo = histo+ 100 * diffusion_matrix[t] / strike * (~isabovecapital)

    return price.mean()[0]