import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

data = np.load('/home/kduplat/Documents/cours ML/TP_Filtrage/mydata_cluster.npz')

# Afficher les noms des tableaux stockés dans le fichier
print("Noms des tableaux dans le fichier :", data.files)

# Accéder à un tableau spécifique
tableau_psd = data['psd']
tableau_r = data['r']
tableau_y = data['y']
tableau_f = data['f']
print("Contenu du tableau :", tableau_psd)

# Fermer le fichier (optionnel mais recommandé pour de gros fichiers)
data.close()


# Colored noise

def generate_gaussian_data(n, a, b):
    return np.random.normal(loc=a, scale=b, size=n)

def generate_col_noise(TPSD):
    bbg = generate_gaussian_data(len(TPSD), 0, 0.001)
    TF_bbg = np.fft.fft(bbg) 
    TF_bbg_bis = TF_bbg * np.sqrt(TPSD)
    TF_inv_bbg = np.fft.ifft(TF_bbg_bis)
    
    return TF_inv_bbg.real


plt.figure(figsize=(10, 10))
plt.subplot(2,2,1)
plt.plot(tableau_psd)
plt.title('psd')
plt.subplot(2,2,2)

plt.plot(tableau_r)
plt.title('r')
plt.subplot(2,2,3)

plt.plot(tableau_y)
plt.title('y')
plt.subplot(2,2,4)
plt.plot(tableau_f)
plt.title('f')
plt.show()

plt.figure(figsize=(10, 10))
plt.scatter(tableau_r, tableau_y)
plt.xscale('log')
plt.show()


from numpy.linalg import inv

Cov = 0

nb_real = 10000
for i in range (nb_real):
    col_noise = generate_col_noise(tableau_psd)
    Cov += np.outer(col_noise.T, col_noise)
    
Cov = Cov/nb_real
Inv_Cov = inv(Cov)


a = 1.1
b = 5.5
c = 0.31


def gaussfct(x,A,x0,sigma):
    return A * np.exp(-(x - x0)**2/(2 * sigma**2))

def fct_densité(r, rp, rho_z, A, mu, sigma):
    
    gauss = gaussfct(r, A, mu, sigma)
    rho_r = rho_z/((r/rp)**c * (1 + (r/rp)**a)**((b-c)/a))
    return rho_r + gauss




A = 0.03
mu = 1500
sigma = 250
rho_z = 0.02
rp = 300

params = [rp, rho_z, A, mu, sigma]
gaus_data = gaussfct(tableau_r, A, mu, sigma)

signal = fct_densité(tableau_r, rp, rho_z, A, mu, sigma)

plt.figure(figsize=(10, 10))
plt.scatter(tableau_r, signal)
plt.xscale('log')
plt.show()


from scipy.optimize import curve_fit

popt, pcov = curve_fit(fct_densité, tableau_r,tableau_y, p0 = params)

rp_fit, rho_zfit, A_fit, mu_fit, sigma_fit = popt
print("rp = {}, rho_z = {}, A= {}, mu= {}, sigma= {}".format(rp_fit, rho_zfit, A_fit, mu_fit, sigma_fit))

y_fit = fct_densité(tableau_r, rp_fit, rho_zfit, A_fit, mu_fit, sigma_fit)
plt.figure(figsize=(8, 8))
plt.plot(tableau_r, tableau_y, 'o', label='Données')
plt.plot(tableau_r, y_fit, label='Modèle')
plt.xscale('log')
plt.legend()
plt.show()



from scipy.stats import chisquare

y_fit_normalized = y_fit * (sum(tableau_y) / sum(y_fit))

chi2 = chisquare(f_obs=tableau_y, f_exp=y_fit_normalized)

print(f"Chi2 statistic: {chi2.statistic}, p-value: {chi2.pvalue}")


chi2_2 = np.sum((tableau_y - y_fit)**2 / y_fit)

print ("Chi2 manual = ", chi2_2)


# CHI2 attendu

Chi2_mat = np.dot((tableau_y - y_fit).T, np.dot(Inv_Cov, (tableau_y - y_fit)))

print("Chi2 attendu = {} +/- {}".format(Chi2_mat, np.sqrt(2*295)))

# On s'attend àn avoir un Chi2 qui est égale à 300 - 5 (Nb de point - nb de degré de liberté)


vals = np.random.multivariate_normal(popt, pcov, 1000)


# print(vals)




from getdist import MCSamples, plots


names = ["rp", "rho_z", "A", "mu", "sigma"]
labels = ["rp", "rho_z", "A", "mu", "sigma"]

mcsamples = MCSamples(samples=vals, names=names, labels=labels)
mcsamples2 = MCSamples(samples=np.random.multivariate_normal(popt, pcov, 1000), names=names, labels=labels)


g = plots.get_subplot_plotter()

g.triangle_plot([mcsamples, mcsamples2], filled=True, legend_labels=["C1", "C2"])


#************ 1.2  MCMC  *************#

def generate_gaussian_data(a, b, n):
    return np.random.normal(loc=a, scale=b, size=n)

def params_def( teta, dteta):
    return np.random.normal(teta, dteta)

teta = [1,2,4,6,12]
dteta = [0.1,0.1,0.1,0.1,0.1]
teta_p = params_def(teta, dteta)

print(teta_p)



#rp = 901.106775343156, rho_z = 0.00939843995629463, A= 0.02110645881399912, mu= 1518.1927631585731, sigma= 204.88530355904697

def log_prior(params):
    rp, rho_z, A, mu, sigma = params
    if 0<rp and  0<rho_z  and 0 < A <0.05 and 1000 < mu < 3000 and 100 < sigma < 500:
        return 0
    else:
        return -np.inf

def log_likelihood(params):
    
    Vraisemblance = fct_densité(tableau_r, params[0], params[1], params[2], params[3], params[4])
    
    Chi2 = np.dot((tableau_y - Vraisemblance).T, np.dot(Inv_Cov, (tableau_y - Vraisemblance)))
    
    return (-1/2 * Chi2)

def log_acceptance(teta, teta_p):
    return log_likelihood(teta_p) + log_prior(teta_p) - log_likelihood(teta) - log_prior(teta)

def test_acceptance(teta, teta_p):
    if log_acceptance(teta, teta_p) >= np.log(np.random.uniform()):
        return teta_p
    else:
        return teta
    
    
def Metropolis_Hastings(teta_0, dteta_0, n_iter):
    teta = teta_0
    teta_list = [teta]
    for _ in range(n_iter):
        teta_p = params_def(teta, dteta_0)
        teta = test_acceptance(teta, teta_p)
        teta_list.append(teta)
    return teta_list



A = 0.03
mu = 1500
sigma = 250
rho_z = 0.02
rp = 300

teta_0 = [rp, rho_z, A, mu, sigma]
dteta_0 = np.array(teta_0) * 0.1
n_iter = 10000

teta_list = Metropolis_Hastings(teta_0, dteta_0, n_iter)




#*****************EN COURS*******************#

A = 0.03
mu = 1500
sigma = 250
rho_z = 0.02
rp = 300

teta_0 = [rp, rho_z, A, mu, sigma]
n_iter = 10000

n_Markov = 10

full_teta_list = np.zeros((n_Markov, n_iter+1, len(teta_0)))

plt.figure(figsize=(10, 10))
for i in range(n_Markov):
    
    teta_list = Metropolis_Hastings(teta_0, n_iter)
    teta_list_np = np.array(teta_list)
    full_teta_list[i] = teta_list_np
    plt.scatter(teta_list_np[:,0], teta_list_np[:,1], alpha = 0.2)
    
plt.xlabel(r'$r_p$', fontsize = 20)
plt.ylabel(r'$\rho _0$', fontsize = 20)
plt.show()


teta_list_np = np.array(teta_list)

x = np.arange(0, len(teta_list_np)/10, 0.1)
plt.figure(figsize=(10, 10))
plt.plot(x, teta_list_np[:,1])
plt.xlabel(r'Step', fontsize = 20)
plt.ylabel(r'$\rho _0$', fontsize = 20)
plt.show()


NCURVES = len(teta_list_np)
cmap = plt.cm.get_cmap('inferno')
colors = [cmap(i / NCURVES) for i in range(NCURVES)]

plt.figure(figsize=(10, 10))
plt.scatter(full_teta_list[0,:,0], full_teta_list[0,:,1], color = colors)
plt.xlabel(r'$r_p$', fontsize = 20)
plt.ylabel(r'$\rho _0$', fontsize = 20)
plt.show()

def log_probability(params): # log de la loi de probabilité (P(teta| d) = L(d|teta) * pi(teta))
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return log_likelihood(params) + log_prior(params)
    
    
    import emcee
ndim = len(teta_0)
nwalkers = 10
nstep = 1000
p0 = np.zeros((nwalkers, ndim))

A = 0.03
mu = 1500
sigma = 250
rho_z = 0.02
rp = 300
teta_0 = [rp, rho_z, A, mu, sigma]

dteta_0 = np.array(teta_0) * 0.1
# dteta_0 = [10, 0.0001, 0.001, 10, 1]

for i in range(nwalkers):
    p0[i] = params_def(teta_0, dteta_0)

print(p0)
    
# p0 = np.random.rand(nwalkers, ndim)

# print(p0)
# print(p0.shape)


sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

sampler.run_mcmc(p0, nstep)

chains = sampler.get_chain()
plt.plot(chains[:, :, 0])


def Gelman_rubin(Tab_chaine):
    """
    Tab_chaine : array de forme (N, M), où N est la longueur des chaînes et M est le nombre de chaînes.
    """
    N = Tab_chaine.shape[1] 
    M = Tab_chaine.shape[0] 
    mean_chaine = np.mean(Tab_chaine, axis=1)
    meanTab = np.mean(Tab_chaine)

    B = np.sum((mean_chaine - meanTab)**2) / (M - 1)
    W = np.mean(np.var(Tab_chaine, axis=0, ddof=1))

    # Estimation de la variance postulée
    Var_post = (1 - 1/N) * W + (1+1/M) * B

    # Calcul du facteur Gelman-Rubin
    R = Var_post / W
    
    return R

    
for i in range(len(teta_0)):
print("GR of {} = {}".format(names[i], Gelman_rubin(full_teta_list[:, 20:, i])))
    
    
    A = 0.03
mu = 1500
sigma = 250
rho_z = 0.02
rp = 300

teta_0 = [rp, rho_z, A, mu, sigma]
n_iter = 5000

n_Markov = 10

# full_teta_list = np.zeros((n_Markov, n_iter+1, len(teta_0)))

R=100

while(R > 1.03):
    n_iter *= 2
    full_teta_list = np.zeros((n_Markov, n_iter+1, len(teta_0)))
    
    for i in range(n_Markov):
    
    
        teta_list = Metropolis_Hastings(teta_0, n_iter)
        teta_list_np = np.array(teta_list)
        full_teta_list[i] = teta_list_np
        
    R = Gelman_rubin(full_teta_list[:, int(n_iter * 0.1):, 0])
    print("Gelman_rubin of {} = {}".format(names[0], Gelman_rubin(full_teta_list[:, int(n_iter * 0.1):, 0])))
        
print(n_iter)
        


