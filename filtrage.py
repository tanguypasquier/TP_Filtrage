import numpy as np
import matplotlib.pyplot as plt

d = np.random.uniform(200,400) # Distance aleatoire

t = np.linspace(0,3e-6, 1000) # Tableau echantillons temporels

ts = t[1] # Pas de temps du tableau

# Differentes constantes 
c=3e8
Ae=3e5
#Ae=2e5
Pe=np.square(Ae)
Ge=1
Gr=1
sigma=1
f=2e7
w=2e-7
lambd=c/f

print("Distance reelle = ", d, "m")

# Debug
#print("t = ",t)
#print("size = ", t.size)

def get_delta_t(dist):
	return (2*dist)/c

def get_distance_from_time(time):
	return c*(time/2)
	
def get_amplitude(dist):
	return np.sqrt((Pe*Ge*Gr*np.square(lambd)*sigma)/(np.power(4*np.pi,3)*np.power(dist,4)))
	
def get_template(A):
	time_template = np.arange(0,w,ts)
	template = A*np.sin(2*np.pi*f*time_template)
	return template

def get_received_pulse():
	received = np.zeros(1000)
	received_amp=get_amplitude(d)
	tr=get_delta_t(d)
	temp=get_template(received_amp)
	#tab_bool=(t>=tr)&(t<=tr+w) #tableau de bool de la taille de t vec True quand la condition est satisfaite
	#print(t,tr+w)
	#received[tab_bool]=get_template(received_amp) #on injecte le template quand 
	index_min=np.argmin(np.abs(t-tr))
	received[index_min:index_min+temp.size]=temp
	#plt.plot(received)
	#plt.show()
	return received
	#print(received)
	
def model_signal():
	return get_template(1)
	
	
noise = np.random.normal(0,1,1000)
sig = get_received_pulse()
data = noise + sig
plt.plot(data)
plt.show()

ref_signal=model_signal()


def correlation():
	Ng = ref_signal.size
	#tab_scal = []
	tau = [np.dot(ref_signal,data[i:i+Ng]) for i in range(1000-Ng)] # Tableau filtre
	index_sig = np.argmax(tau) # Indice du debut du signal
	noise_std = np.std(tau[:442]) # Je ne peux que avoir du bruit avant l'index 442 du tableau de temps (temps pour parcourir 200m aller-retour)
	#print("noise_std = ", noise_std)
	tab_SNR = tau / noise_std
	distance_signal = get_distance_from_time(t[index_sig])
	if(tab_SNR[index_sig] > 3):
		print("Signal found for distance = ", distance_signal, "m")
	else:
		print("No signal found")
	plt.plot(tab_SNR)
	plt.show()

	

correlation()
	


	
	


#get_received_pulse()
	
