import pyromat as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

'''
I know it is still a little bit crappy, because I didn't use classes, but I hope you like the code :D

All of the functions follow the same structure. First the states are calculated and saved into an array.
Then the work of the pump, the efficiency and so on are calculated and also saved
Then this arrays are saved in a dataframe, which is what is returned from each function.

Afterwards, there is a function that plots every cycle (with the water curve), 
ideal and with the irreversibilities (it is also a little bit crappy, but it works).

For the titles of the graphic representation the most unefficient thing occured to me, but again, it works. I added the
title of the cycle to the returning parameters of each function.
'''

plt.style.use('seaborn')
#PROBLEM DATA
fluid = pm.get('mp.H2O')
p1 = 8 #MPa
p2 = 0.008 #MPa

x1 = 1.
x3 = 0.

t1_oh = 723.15 #oh is for overheated
eta_t_real = 0.85 #performance, or efficiency of the turbine
eta_p_real = 0.85 #Same for the pump
W_cycle = 100*1e03 #kW


##############Ideal Cycle and with irreversibilities##################
def Rankine_cycle(fluid=fluid, p1=p1, p2=p2, x1=x1, x3=x3, 
	eta_t_real=eta_t_real, eta_p_real=eta_p_real, W_cycle=W_cycle):

	pm.config['unit_pressure'] = 'MPa' #Actually, it is possible to change units.

	#State 1:
	h1 = fluid.h(p=p1, x=x1)
	s1 = fluid.s(p=p1, x=x1)
	t1 = fluid.T(p=p1, x=x1)
	State_1 = np.array([t1, p1, h1, s1, x1])
	#State 2:
	s2 = s1 #Isoentropic process.
	t2, x2 = fluid.T_s(s=s2, p=p2, quality=True)
	h2 = fluid.h(p=p2, T=t2, x=x2)
	State_2 = np.array([t2, p2, h2, s2, x2])
	#Real State 2
	h2real = h1 - eta_t_real*(h1-h2)
	t2real, x2real = fluid.T_h(h=h2real, p=p2, quality=True)
	s2real = fluid.s(T=t2real, p=p2, x=x2real)
	State_2_real = np.array([t2real, p2, h2real, s2real, x2real])
	#State 3:
	t3 = t2 #Isothermic process
	p3 = fluid.p(T=t3, x=x3)
	s3 = fluid.s(T=t3, p=p3, x=x3)
	h3 = fluid.h(T=t3, p=p3, x=x3)
	State_3 = np.array([t3, p3, h3, s3, x3])
	H2O = pm.get('mp.H2O')
	#State 4:
	s4 = s3 #Isoentropic process
	p4 = p1 #Isobaric process
	t4, x4 = fluid.T_s(s=s4, p=p4, quality=True)
	h4 = fluid.h(p=p4, T=t4, x=x4)
	State_4 = np.array([t4, p4, h4, s4, x4])
	#Real State 4
	h4real=h3+(h4-h3)/eta_p_real
	t4real, x4real  = H2O.T_h(h=h4real,p=p4, quality=True)
	s4real  = H2O.s(T=t4real,p=p4,x=x4real)
	State_4_real = np.array([t4real, p4, h4real, s4real, x4real])

	data_ideal = [State_1, State_2, State_3, State_4]
	ideal_df = pd.DataFrame(data=data_ideal, index=['State 1', 'State 2', 'State 3', 'State 4'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])

	data_real = [State_1, State_2_real, State_3, State_4_real]
	real_df = pd.DataFrame(data=data_real, index=['State 1', 'State 2', 'State 3', 'State 4'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])

	#Ideal work and so on
	W_t = h1 - h2 #Work done by the turbine [kJ/kg]
	Q_out = h2-h3 #Heat out [kJ/kg]
	#Work from the compression
	W_p = h4-h3 #kJ/kg
	Q_in = h1-h4 #Absorbed heat Calor en la caldera. kJ/kg
	eta = abs((W_t-W_p)*100/Q_in) #ideal efficiency
	mass_flux = abs(W_cycle/(W_t-W_p)) #kg/s
	bwr = W_p / W_t *100 #%

	W_t *= mass_flux/1000 #[MW]
	W_p *= mass_flux/1000 #[MW]
	Q_in *= mass_flux/1000 #[MW]
	Q_out *= mass_flux/1000 #[MW]

	EnergyParams_ideal = {'W_t [MW]': W_t, 'Q_out [MW]':Q_out, 'W_p [MW]': W_p, 'Q_in [MW]': Q_in, 'bwr [%]': bwr,
					'eta [%]': eta, 'mass_flux [kg/s]': mass_flux}
	EnergyParams_ideal = pd.DataFrame(data=EnergyParams_ideal)

	# Real works and so on
	W_t_real = h1 - h2real #Work done by the turbine
	Q_out_real = h2real-h3 #Heat
	#Work from the compression
	W_p_real = h4real-h3
	Q_in_real = h1-h4real #Absorbed heat
	eta_real = abs((W_t_real-W_p_real)*100/Q_in_real) #ideal efficiency
	mass_flux_real = abs(W_cycle/(W_t_real-W_p_real))
	bwr_real = W_p_real / W_t_real *100
	W_t_real *= mass_flux_real/1000 #[MW]
	W_p_real *= mass_flux_real/1000 #[MW]
	Q_in_real *= mass_flux_real/1000 #[MW]
	Q_out_real *= mass_flux_real/1000 #[MW]


	EnergyParams_real = {'W_t [MW]':W_t_real,'Q_out [MW]': Q_out_real,'W_p [MW]': W_p_real,'Q_in [MW]': Q_in_real,'bwr [%]': bwr_real,
						 'eta [%]':eta_real,'mass_flux [kg/s]': mass_flux_real}
	EnergyParams_real = pd.DataFrame(data=EnergyParams_real)

	title = 'Rankine cycle'
	return ideal_df, EnergyParams_ideal, real_df, EnergyParams_real, title






def overheated_Rankine(fluid=fluid, p1=p1, p2=p2, x1=x1, x3=x3, 
    t1_oh=723.15, eta_t=eta_t_real, eta_p=eta_p_real,
    W_cycle=W_cycle):
	pm.config['unit_pressure'] = 'MPa' #Actually, it is possible to change units.
	#State 1:
	h1 = fluid.h(p=p1, x=x1)
	s1 = fluid.s(p=p1, x=x1)
	t1 = fluid.T(p=p1, x=x1)
	State_1 = np.array([t1, p1, h1, s1, x1])
	#Overheated state 1
	h1_oh = fluid.h(p=p1,T=t1_oh)
	s1_oh = fluid.s(p=p1,T=t1_oh)
	x1_oh = fluid.T_s(s=s1_oh, p=p1, quality=True)[1]
	State_1_oh = np.array([t1_oh, p1, h1_oh, s1_oh, x1_oh])
	#Overheated state 2 (ideal):
	s2_oh_id = s1_oh #Isoentropic process.
	t2_oh_id, x2_oh_id = fluid.T_s(s=s2_oh_id, p=p2, quality=True)
	h2_oh_id = fluid.h(p=p2, T=t2_oh_id, x=x2_oh_id)
	State_2_oh_id = np.array([t2_oh_id, p2, h2_oh_id, s2_oh_id, x2_oh_id])
	#Overheated state 2 (real)
	h2_oh = h1_oh - eta_t*(h1_oh-h2_oh_id)
	t2_oh, x2_oh = fluid.T_h(h=h2_oh, p=p2, quality=True)
	s2_oh = fluid.s(T=t2_oh, p=p2, x=x2_oh)
	W_r_oh = h2_oh - h1_oh
	State_2_oh = np.array([t2_oh, p2, h2_oh, s2_oh, x2_oh])
	#State 3:
	t3_oh = t2_oh #Isothermic process
	p3 = fluid.p(T=t3_oh, x=x3)
	s3_oh = fluid.s(T=t3_oh, p=p3, x=x3)
	h3_oh = fluid.h(T=t3_oh, p=p3, x=x3)
	State_3_oh = np.array([t3_oh, p3, h3_oh, s3_oh, x3])

	H2O = pm.get('mp.H2O')
	#State 4 (Ideal):
	s4_oh_id = s3_oh #Isoentropic process
	p4 = p1 #Isobaric process
	t4_oh_id, x4_oh_id = fluid.T_s(s=s4_oh_id, p=p4, quality=True)
	h4_oh_id = fluid.h(p=p4, T=t4_oh_id, x=x4_oh_id)
	State_4_oh_id = np.array([t4_oh_id, p4, h4_oh_id, s4_oh_id, x4_oh_id])

	#Real State 4
	h4_oh=h3_oh+(h4_oh_id-h3_oh)/eta_p
	t4_oh, x4_oh  = H2O.T_h(h=h4_oh,p=p4, quality=True)
	s4_oh  = H2O.s(T=t4_oh,p=p4,x=x4_oh)
	State_4_oh = np.array([t4_oh, p4, h4_oh, s4_oh, x4_oh])

	#Ideal oh
	data_ideal = [State_1, State_1_oh, State_2_oh_id, State_3_oh, State_4_oh_id]
	ideal_oh_df = pd.DataFrame(data=data_ideal, index=['State 1','State 1 oh', 'State 2', 'State 3', 'State 4'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])

	W_t_oh = h1_oh - h2_oh_id
	Q_out_oh = h2_oh_id - h3_oh #From the overheated state
	#Work from the compression
	#Overheated work
	W_p_oh = (h4_oh_id - h3_oh)
	Q_in_oh = h1_oh-h4_oh_id #Overheated heat
	eta_oh = abs((W_t_oh-W_p_oh)*100/Q_in_oh) #Overheated efficiency
	mass_flux = abs(W_cycle/(W_t_oh-W_p_oh))
	bwr = W_p_oh / W_t_oh *100
	W_t_oh *= mass_flux/1000 #[MW]
	W_p_oh *= mass_flux/1000 #[MW]
	Q_in_oh *= mass_flux/1000 #[MW]
	Q_out_oh *= mass_flux/1000 #[MW]
	EnergyParams_ideal = {'W_t [MW]':W_t_oh, 'Q_out [MW]':Q_out_oh,'W_p [MW]': W_p_oh,'Q_in [MW]': Q_in_oh,
						'bwr [%]': bwr,'eta [%]': eta_oh,'mass_flux [kg/s]': mass_flux}
	EnergyParams_ideal = pd.DataFrame(data=EnergyParams_ideal)


	data = [State_1, State_1_oh, State_2_oh, State_3_oh, State_4_oh]
	real_oh_df = pd.DataFrame(data=data, index=['State 1','State 1 oh', 'State 2', 'State 3', 'State 4'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])
	W_t_oh = h1_oh - h2_oh
	Q_out_oh = h2_oh - h3_oh #From the overheated state
	#Work from the compression
	#Overheated work
	W_p_oh = (h4_oh - h3_oh)
	Q_in_oh = h1_oh-h4_oh #Overheated heat
	eta_oh = abs((W_t_oh-W_p_oh)*100/Q_in_oh) #Overheated efficiency
	mass_flux = abs(W_cycle/(W_t_oh-W_p_oh))
	bwr = W_p_oh / W_t_oh *100
	W_t_oh *= mass_flux/1000 #[MW]
	W_p_oh *= mass_flux/1000 #[MW]
	Q_in_oh *= mass_flux/1000 #[MW]
	Q_out_oh *= mass_flux/1000 #[MW]

	EnergyParams_real = {'W_t [MW]':W_t_oh, 'Q_out [MW]':Q_out_oh,'W_p [MW]': W_p_oh,'Q_in [MW]': Q_in_oh,
						'bwr [%]': bwr,'eta [%]': eta_oh,'mass_flux [kg/s]': mass_flux}
	EnergyParams_real = pd.DataFrame(data=EnergyParams_real)
	title = 'Overheated Rankine cycle'
	return ideal_oh_df,EnergyParams_ideal, real_oh_df, EnergyParams_real, title


p1re = 10 #MPa
t1re = 500 +273.15 #K
W_cycle = 200*1e03 #kW
p2re = 0.8 #MPa
t3re = 460 + 273.15 #K
p3re = 0.8
p4re = 0.008
eta_t_re1 = 0.91
eta_t_re2 = 0.93
eta_p = 0.88

def reheat_Rankine(fluid=fluid, p1=p1re, p2=p2re, p3=p3re, x1=x1, t1=t1re,
    p4=p4re, t3=t3re, eta_t_re1=eta_t_re1, eta_t_re2=eta_t_re2, eta_p=eta_p,
    W_cycle=W_cycle):
	pm.config['unit_pressure'] = 'MPa' #Actually, it is possible to change units.
	p1re = p1
	#State 1
	h1re = fluid.h(p=p1re, T=t1re)
	s1re = fluid.s(p=p1re, T=t1re)
	State_1 = np.array([t1re, p1re, h1re, s1re, x1])
	#State 2 ideal
	p2re = p2
	s2re_id = s1re
	t2re_id, x2_id = fluid.T_s(s=s2re_id, p=p2re, quality=True)
	h2re_id = fluid.h(p=p2re, T=t2re_id, x=x2_id)
	State_2_id = np.array([t2re_id, p2re, h2re_id, s2re_id, x2_id])
	#State 2
	h2re = h1re - (eta_t_re1*(h1re-h2re_id))
	t2re, x2 = fluid.T_h(h=h2re, p=p2re, quality=True)
	s2re = fluid.s(T=t2re, p=p2re, x=x2)
	State_2 = np.array([t2re, p2re, h2re, s2re, x2])
	#State 3
	p3re = p3
	s3re = fluid.s(p=p3re, T=t3re)
	h3re = fluid.h(p=p3re, T=t3re)
	x3 = fluid.T_s(s=s3re, p=p3re, quality=True)[1]
	State_3 = np.array([t3re, p3re, h3re, s3re, x3])
	#State 4 ideal
	p4re = p4
	s4re_id = s3re
	t4re_id, x4_id = fluid.T_s(s=s4re_id, p=p4re, quality=True)
	h4re_id = fluid.h(p=p4re, T=t4re_id, x=x4_id)
	State_4_id = np.array([t4re_id, p4re, h4re_id, s4re_id, x4_id])
	#State 4
	h4re = h3re-(eta_t_re2*(h3re-h4re_id))
	t4re, x4 = fluid.T_h(h=h4re, p=p4re, quality=True)
	s4re = fluid.s(T=t4re, p=p4re, x=x4)
	State_4 = np.array([t4re, p4re, h4re, s4re, x4])
	#State 5
	t5re = t4re
	x5 = 0
	p5re = fluid.p(T=t5re, x=x5)
	s5re = fluid.s(p=p5re, T=t5re, x=x5)
	h5re = fluid.h(p=p5re, T=t5re, x=x5)
	State_5 = np.array([t5re, p5re, h5re, s5re, x5])
	#State 6 ideal
	s6re_id = s5re
	p6re = p1re
	t6re_id, x6_id = fluid.T_s(s=s6re_id, p=p6re, quality=True)
	h6re_id = fluid.h(p=p6re, T=t6re_id, x=x6_id)
	State_6_id = np.array([t6re_id, p6re, h6re_id, s6re_id, x6_id])
	#State 6
	h6re = h5re + (h6re_id-h5re)/eta_p
	t6re, x6 = fluid.T_h(h=h6re, p=p6re, quality=True)
	s6re = fluid.s(T=t6re, p=p6re, x=x6)
	State_6 = np.array([t6re, p6re, h6re, s6re, x6])

	data_id = [State_1, State_2_id, State_3, State_4_id, State_5, State_6_id]
	reheated_id_df = pd.DataFrame(data=data_id, index=['State 1', 'State 2', 'State 3', 'State 4', 'State 5', 'State 6'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])

	W_t = (h1re - h2re_id) + (h3re - h4re_id)
	Q_cald = h1re - h6re_id + (h3re - h2re_id)
	W_p = (h6re_id - h5re)
	Q_cond = h4re - h5re 
	eta = abs((W_t - W_p)*100/Q_cald) 
	mass_flux = abs(W_cycle/(W_t-W_p))
	bwr = W_p / W_t *100
	W_t *= mass_flux/1000 #[MW]
	W_p *= mass_flux/1000 #[MW]
	Q_cald *= mass_flux/1000 #[MW]
	Q_cond *= mass_flux/1000 #[MW]
	EnergyParams_id = {'W_t [MW]': W_t, 'Q_cald [MW]':Q_cald, 'W_p [MW]': W_p, 'Q_cond [MW]': Q_cond, 'bwr [%]': bwr,
					'eta [%]': eta, 'mass_flux [kg/s]': mass_flux}
	EnergyParams_id = pd.DataFrame(data=EnergyParams_id, index=['W_t [MW]', 
		'Q_cald [MW]', 'W_p [MW]', 'Q_cond [MW]','bwr [%]', 'eta [%]', 'mass_flux [kg/s]'])


	data = [State_1, State_2, State_3, State_4, State_5, State_6]
	reheated_df = pd.DataFrame(data=data, index=['State_1', 'State_2', 'State_3', 'State_4', 'State_5', 'State_6'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])

	W_t = (h1re - h2re) + (h3re - h4re)
	Q_cald = h1re - h6re + (h3re - h2re)
	W_p = (h6re - h5re)
	Q_cond = h4re - h5re 
	eta = abs((W_t - W_p)*100/Q_cald) 
	mass_flux = abs(W_cycle/(W_t-W_p))
	bwr = W_p / W_t *100
	W_t *= mass_flux/1000 #[MW]
	W_p *= mass_flux/1000 #[MW]
	Q_cald *= mass_flux/1000 #[MW]
	Q_cond *= mass_flux/1000 #[MW]
	EnergyParams = {'W_t [MW]': W_t, 'Q_cald [MW]':Q_cald, 'W_p [MW]': W_p, 'Q_cond [MW]': Q_cond, 'bwr [%]': bwr,
					'eta [%]': eta, 'mass_flux [kg/s]': mass_flux}
	EnergyParams = pd.DataFrame(data=EnergyParams)
	title = 'Reheated Rankine cycle'

	return reheated_id_df, EnergyParams_id, reheated_df, EnergyParams, title


def water_curve():
	#From the pyromat documentation we get the water curve:
	# Get the critical and triple point properties
	H2O = pm.get('mp.H2O')
	Tt,pt = H2O.triple()
	Tc,pc = H2O.critical()
	# Explore the temperatures between Tt and Tc in 5K increments
	T = np.linspace(Tt,Tc,1000)
	s0 = np.zeros(len(T))
	s1 = np.zeros(len(T))
	fig = plt.figure()
	for i in range(len(T)):
	    s0[i]=H2O.s(T[i],x=0)
	    s1[i]=H2O.s(T[i],x=1)
	return plt.plot(s0,T,'cyan',s1,T,'red',ls='--')



def plot_cycle(cycle):

	id_df = cycle[0]
	re_df = cycle[2]
	index = id_df.columns.values
	T_id = id_df[index[0]]
	s_id = id_df[index[3]]
	T_re = re_df[index[0]]
	s_re = re_df[index[3]]
	p_id = id_df[index[1]]
	p_re = re_df[index[1]]

	H2O = pm.get('mp.H2O')

	T_id_last = np.linspace(T_id[-1], T_id[0],100)
	p_id_last = p_id[-1] * np.ones(len(T_id_last))
	S_id_last = H2O.s(T=T_id_last,p=p_id_last)

	T_re_last = np.linspace(T_re[-1], T_re[0],100)
	p_re_last = p_re[-1] * np.ones(len(T_re_last))
	S_re_last = H2O.s(T=T_re_last,p=p_re_last)

	water_curve()
	plt.plot(S_id_last,T_id_last,'-g')
	plt.plot(s_id, T_id, '-og', label='Ideal')
	plt.plot(S_re_last,T_re_last,'-k')
	plt.plot(s_re, T_re, '-ok', label='Real')
	plt.legend()
	plt.title(cycle[-1])
	plt.xlabel('Entropy s [kJ/(kg K)]')
	plt.ylabel('Temperature T [K]')
	



#Ejercicios 1 y 2
print('Rankine cycle with and without irreversibilities')
print('Ideal states')
print(Rankine_cycle()[0])
print('Real states')
print(Rankine_cycle()[2])
print('-------------------')
print('Ideal work, heat, efficiency and other')
print(Rankine_cycle()[1])
print('Real work, heat, efficiency and other')
print(Rankine_cycle()[3])
#print(plot_cycle(Rankine_cycle()))


#Ejercicio 3
print('Overheated Rankine cycle with and without irreversibilities')
print('Ideal states')
print(overheated_Rankine()[0])
print('Real states')
print(overheated_Rankine()[2])
print('-------------------')
print('Ideal work, heat, efficiency and other')
print(overheated_Rankine()[1])
print('Real work, heat, efficiency and other')
print(overheated_Rankine()[3])
#print(plot_cycle(overheated_Rankine()))


#Ejercicio 4
print('Reheated Rankine cycle with and without irreversibilities')
print('Ideal states')
print(reheat_Rankine()[0])
print('Real states')
print(reheat_Rankine()[2])
print('-------------------')
print('Ideal work, heat, efficiency and other')
print(reheat_Rankine()[1])
print('Real work, heat, efficiency and other')
print(reheat_Rankine()[3])
#print(plot_cycle(reheat_Rankine()))




#Here I can plot the cycles
cycles = [Rankine_cycle(), overheated_Rankine(), reheat_Rankine()]
for cycle in cycles:
	print(plot_cycle(cycle=cycle))
#plt.show()



p_range = np.array([8., 10.,12., 15.,18., 20., 22.])
T_range = np.array([400., 450., 500., 550.]) + 273.15


#To run this function, remember to coment the line fig = plt.figure from the water curve function
'''
def plot_analysis(p_range=p_range, T_range=None):
	plt.figure()
	cols = 2
	if p_range is not None:
		rows = len(p_range)/2
		for i in range(len(p_range)):
			#plt.subplot(rows, cols, i+1)
			plot_cycle(reheat_Rankine(p1 = p_range[i]))
			#plt.text(3,400,'Hola')
	else:
		rows = len(T_range)/2
		for i in range(len(T_range)):
			#plt.subplot(rows, cols, i+1)
			plot_cycle(reheat_Rankine(t1 = T_range[i]))
			plt.title(str(T_range[i]))

	plt.show()
'''

eta_p_range = np.arange(0.7, 1.0, 0.05)


#This function compares the main values of the cycles by changing one of the parameters.

def analysis_changes(cycle, p_range, T_range=None, eta_p_range=None):
	efficiencies = []
	bwr = []
	mass_flux = []
	if p_range is not None:
		for p in p_range:
			data = cycle(p1=p)[3]
			efficiencies.append(data['eta [%]'][0])
			bwr.append(data['bwr [%]'][0])
			mass_flux.append(data['mass_flux [kg/s]'][0])
		data = {'P [MPa]': p_range, 'efficiency [%]': efficiencies, 'bwr [%]': bwr, 'mass_flux [kg/s]': mass_flux}
		df = pd.DataFrame(data=data)
	elif T_range is not None:
		for t in T_range:
			data = cycle(t1=t)[3]
			efficiencies.append(data['eta [%]'][0])
			bwr.append(data['bwr [%]'][0])
			mass_flux.append(data['mass_flux [kg/s]'][0])
		data = {'T [K]': T_range, 'efficiency [%]': efficiencies, 'bwr [%]': bwr, 'mass_flux [kg/s]': mass_flux}
		df = pd.DataFrame(data=data)
	else:
		for eta in eta_p_range:
			data = cycle(eta_p=eta)[3]
			efficiencies.append(data['eta [%]'][0])
			bwr.append(data['bwr [%]'][0])
			mass_flux.append(data['mass_flux [kg/s]'][0])
		data = {'eta_p [p.u]': eta_p_range, 'efficiency [%]': efficiencies, 'bwr [%]': bwr, 'mass_flux [kg/s]': mass_flux}
		df = pd.DataFrame(data=data)
	return df




#print(analysis_changes(cycle=reheat_Rankine, p_range=p_range))



















