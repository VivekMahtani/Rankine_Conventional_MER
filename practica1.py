import pyromat as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
I know it's a little crappy. You should separate the graphic representation functions
of the state's calculations. And it must be generalised for any
initial data... I know some times I calculate more than 1 time the same thing.
'''

plt.style.use('seaborn')
#PROBLEM DATA
fluid = pm.get('mp.H2O')
p1 = 8 #MPa
p2 = 0.008 #MPa

#p1 = P1*1e-5 #bar #PYROMAT WORKS WITH bar
#p2 = P2*1e-5 #bar

x1 = 1.
x3 = 0.

t1_oh = 723.15 #oh is for overheated
eta_t_real = 0.85 #performance, or efficiency of the turbine
eta_p_real = 0.85 #Same for the pump
W_cycle = 100*1e03 #kW

def ideal_Rankine(fluid=fluid, p1=p1, p2=p2, x1=x1, x3=x3, W_cycle=W_cycle):
	pm.config['unit_pressure'] = 'MPa' #Actually, it is possible to change units.
	#State 1:
	h1 = fluid.h(p=p1, x=x1)[0]
	s1 = fluid.s(p=p1, x=x1)[0]
	t1 = fluid.T(p=p1, x=x1)[0]
	State_1 = np.array([t1, p1, h1, s1, x1])
	#State 2:
	s2 = s1 #Isoentropic process.
	t2, x2 = fluid.T_s(s=s2, p=p2, quality = True)
	h2 = fluid.h(p=p2, T=t2, x=x2)
	State_2 = np.array([t2, p2, h2, s2, x2])
	#State 3:
	t3 = t2 #Isothermic process
	p3 = fluid.p(T=t3, x=x3)
	s3 = fluid.s(T=t3, p=p3, x=x3)
	h3 = fluid.h(T=t3, p=p3, x=x3)
	State_3 = np.array([t3, p3, h3, s3, x3])
	#State 4:
	s4 = s3 #Isoentropic process
	p4 = p1 #Isobaric process
	t4, x4 = fluid.T_s(s=s4, p=p4, quality=True)
	h4 = fluid.h(p=p4, T=t4, x=x4)
	State_4 = np.array([t4, p4, h4, s4, x4])
	data = [State_1, State_2, State_3, State_4]
	ideal_df = pd.DataFrame(data=data, index=['State_1', 'State_2', 'State_3', 'State_4'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])
	######## We have here all the states of the cycle ##################
	#Now we calculate the rest of the parameters.

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
	EnergyParams = [W_t, Q_out, W_p, Q_in, bwr, eta, mass_flux]

	EnergyParams = pd.DataFrame(data=EnergyParams, index=['W_t [MW]', 'Q_out [MW]', 'W_p [MW]', 'Q_in [MW]','bwr [%]', 'eta [%]', 'mass_flux [kg/s]'])
    ###############PLOTTING THE IDEAL CYCLE########################
	plt.figure(1)
	##########IDEAL CYCLE###########
	## 1--> 2
	T = np.array([t1, t2])
	S = np.array([s1, s2])
	plt.plot(S,T, '--ko', alpha=1)
	## 2-->3
	T = np.array([t2, t3])
	S = np.array([s2, s3])
	plt.plot(S,T, '--ko', alpha=1)
	## 3--> 4
	T = np.array([t3, t4])
	S = np.array([s3, s4])
	plt.plot(S,T, '--ko', alpha=1)
	##4-->1
	H2O = pm.get('mp.H2O')
	T = np.linspace(t4, t1,100)
	p = p4 * np.ones(len(T))
	S = H2O.s(T=T,p=p)
	plt.plot(S,T,'--k', alpha=1)
	#From the pyromat documentation we get the water curve:
	# Get the critical and triple point properties
	Tt,pt = H2O.triple()
	Tc,pc = H2O.critical()
	# Explore the temperatures between Tt and Tc in 5K increments
	T = np.linspace(Tt,Tc,1000)
	s0 = np.zeros(len(T))
	s1 = np.zeros(len(T))
	for i in range(len(T)):
	    s0[i]=H2O.s(T[i],x=0)
	    s1[i]=H2O.s(T[i],x=1)
	plt.plot(s0,T,'cyan',s1,T,'red',ls='--')
	#plt.show()
	return ideal_df, EnergyParams

	##############Real Cycle##################
def real_Rankine(fluid=fluid, p1=p1, p2=p2, x1=x1, x3=x3, 
    t1_oh=723.15, eta_t_real=eta_t_real, eta_p_real=eta_p_real,
    W_cycle=W_cycle):
	pm.config['unit_pressure'] = 'MPa' #Actually, it is possible to change units.
	# #Actally, the only states that change are 2 and 4
	#state 1:
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
	print(s2real)
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
	data = [State_1, State_2_real, State_3, State_4_real]
	real_df = pd.DataFrame(data=data, index=['State_1', 'State_2', 'State_3', 'State_4'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])

	W_t = h1 - h2real #Work done by the turbine
	Q_out = h2real-h3 #Heat
	#Work from the compression
	W_p = h4real-h3
	Q_in = h1-h4real #Absorbed heat
	eta = abs((W_t-W_p)*100/Q_in) #ideal efficiency
	mass_flux = abs(W_cycle/(W_t-W_p))
	bwr = W_p / W_t *100
	W_t *= mass_flux/1000 #[MW]
	W_p *= mass_flux/1000 #[MW]
	Q_in *= mass_flux/1000 #[MW]
	Q_out *= mass_flux/1000 #[MW]
	EnergyParams = [W_t, Q_out, W_p, Q_in, bwr, eta, mass_flux]
	EnergyParams = pd.DataFrame(data=EnergyParams, index=['W_t [MW]', 'Q_out [MW]', 'W_p [MW]', 'Q_in [MW]','bwr [%]', 'eta [%]', 'mass_flux [kg/s]'])

    ###############PLOTTING THE REAL CYCLE########################
	plt.figure(2)
	## 1--> 2
	T = np.array([t1, t2real])
	S = np.array([s1, s2real])
	plt.plot(S,T, '--ko', alpha=1)
	## 2-->3
	T = np.array([t2real, t3])
	S = np.array([s2real, s3])
	plt.plot(S,T, '--ko', alpha=1)
	## 3--> 4
	T = np.array([t3, t4real])
	S = np.array([s3, s4real])
	plt.plot(S,T, '--ko', alpha=1)
	##4-->1
	H2O = pm.get('mp.H2O')
	T = np.linspace(t4real, t1,100)
	p = p4 * np.ones(len(T))
	S = H2O.s(T=T,p=p)
	plt.plot(S,T,'--k', alpha=1)
	#From the pyromat documentation we get the water curve:
	# Get the critical and triple point properties
	Tt,pt = H2O.triple()
	Tc,pc = H2O.critical()
	# Explore the temperatures between Tt and Tc in 5K increments
	T = np.linspace(Tt,Tc,1000)
	s0 = np.zeros(len(T))
	s1 = np.zeros(len(T))
	for i in range(len(T)):
	    s0[i]=H2O.s(T[i],x=0)
	    s1[i]=H2O.s(T[i],x=1)
	plt.plot(s0,T,'cyan',s1,T,'red',ls='--')
	#plt.show()
	return real_df, EnergyParams


def overheated_Rankine(fluid=fluid, p1=p1, p2=p2, x1=x1, x3=x3, 
    t1_oh=723.15, eta_t_real=eta_t_real, eta_p_real=eta_p_real,
    W_cycle=W_cycle):
	pm.config['unit_pressure'] = 'MPa' #Actually, it is possible to change units.
	#state 1:
	h1 = fluid.h(p=p1, x=x1)
	s1 = fluid.s(p=p1, x=x1)
	t1 = fluid.T(p=p1, x=x1)
	State_1 = np.array([t1, p1, h1, s1, x1])
	#Overheated state 1
	h1_oh = fluid.h(p=p1,T=t1_oh)
	s1_oh = fluid.s(p=p1,T=t1_oh)
	x1_oh = x1
	State_1_oh = np.array([t1_oh, p1, h1_oh, s1_oh, x1_oh])
	#State 2:
	s2 = s1_oh #Isoentropic process.
	t2, x2 = fluid.T_s(s=s2, p=p2, quality=True)
	h2 = fluid.h(p=p2, T=t2, x=x2)
	State_2 = np.array([t2, p2, h2, s2, x2])
	#Real State 2
	h2real = h1 - eta_t_real*(h1-h2)
	t2real, x2real = fluid.T_h(h=h2real, p=p2, quality=True)
	s2real = fluid.s(T=t2real, p=p2, x=x2real)
	State_2_real = np.array([t2real, p2, h2real, s2real, x2real])
	#Overheated state 2
	s2_oh = s1_oh
	t2_oh, x2_oh = fluid.T_s(s=s2_oh, p=p2, quality=True)
	s2_oh = fluid.s(T=t2_oh, p=p2, x=x2_oh)
	h2 = fluid.h(T=t2_oh, x=x2_oh)
	h2_oh = h1_oh - eta_t_real*(h1_oh-h2)
	W_r_oh = h2_oh - h1_oh
	State_2_oh = np.array([t2_oh, p2, h2_oh, s2_oh, x2_oh])
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
	data = [State_1_oh, State_2_oh, State_3, State_4_real]
	real_oh_df = pd.DataFrame(data=data, index=['State_1', 'State_2', 'State_3', 'State_4'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])
	W_t_oh = h1_oh - h2_oh
	Q_out_oh = h2_oh - h3 #From the overheated state
	#Work from the compression
	#Overheated work
	W_p_oh = (h4real - h3)
	Q_in_oh = h1_oh-h4real #Overheated heat
	eta_oh = abs((W_t_oh-W_p_oh)*100/Q_in_oh) #Overheated efficiency
	mass_flux = abs(W_cycle/(W_t_oh-W_p_oh))
	bwr = W_p_oh / W_t_oh *100
	W_t_oh *= mass_flux/1000 #[MW]
	W_p_oh *= mass_flux/1000 #[MW]
	Q_in_oh *= mass_flux/1000 #[MW]
	Q_out_oh *= mass_flux/1000 #[MW]
	EnergyParams = [W_t_oh, Q_out_oh, W_p_oh, Q_in_oh, bwr, eta_oh, mass_flux]
	EnergyParams = pd.DataFrame(data=EnergyParams, index=['W_t [MW]', 'Q_out [MW]', 'W_p [MW]', 'Q_in [MW]','bwr [%]', 'eta [%]', 'mass_flux [kg/s]'])
    ###############PLOTTING THE REAL CYCLE########################
	plt.figure(3)
	#1 --> 1'
	T = np.array([t1, t1_oh])
	S = np.array([s1, s1_oh])
	plt.plot(S,T, '--ko', alpha=1)
	## 1'--> 2
	T = np.array([t1_oh, t2_oh])
	S = np.array([s1_oh, s2_oh])
	plt.plot(S,T, '--ko', alpha=1)
	## 2-->3
	T = np.array([t2_oh, t3])
	S = np.array([s2_oh, s3])
	plt.plot(S,T, '--ko', alpha=1)
	## 3--> 4
	T = np.array([t3, t4real])
	S = np.array([s3, s4real])
	plt.plot(S,T, '--ko', alpha=1)
	##4-->1
	H2O = pm.get('mp.H2O')
	T = np.linspace(t4real, t1,100)
	p = p4 * np.ones(len(T))
	S = H2O.s(T=T,p=p)
	plt.plot(S,T,'--k', alpha=1)
	#From the pyromat documentation we get the water curve:
	# Get the critical and triple point properties
	Tt,pt = H2O.triple()
	Tc,pc = H2O.critical()
	# Explore the temperatures between Tt and Tc in 5K increments
	T = np.linspace(Tt,Tc,1000)
	s0 = np.zeros(len(T))
	s1 = np.zeros(len(T))
	for i in range(len(T)):
	    s0[i]=H2O.s(T[i],x=0)
	    s1[i]=H2O.s(T[i],x=1)
	plt.plot(s0,T,'cyan',s1,T,'red',ls='--')
	plt.show()
	return real_oh_df, EnergyParams




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

def reheat_Rankine(fluid=fluid, p1re=p1re, p2re=p2re, p3re=p3re, x1=x1, t1re=t1re,
    p4re=p4re, t3re=t3re, eta_t_re1=eta_t_re1, eta_t_re2=eta_t_re2, eta_p=eta_p,
    W_cycle=W_cycle):
	pm.config['unit_pressure'] = 'MPa' #Actually, it is possible to change units.

	#State 1
	h1re = fluid.h(p=p1re, T=t1re)
	s1re = fluid.s(p=p1re, T=t1re)
	State_1 = np.array([t1re, p1re, h1re, s1re, x1])
	#State 2
	s2re_id = s1re
	t2re_id, x2 = fluid.T_s(s=s2re_id, p=p2re, quality=True)
	h2re_id = fluid.h(p=p2re, T=t2re_id, x=x2)

	h2re = h1re - (eta_t_re1*(h1re-h2re_id))
	t2re, x2 = fluid.T_h(h=h2re, p=p2re, quality=True)
	s2re = fluid.s(T=t2re, p=p2re, x=x2)
	State_2 = np.array([t2re, p2re, h2re, s2re, x2])
	#State 3
	s3re = fluid.s(p=p3re, T=t3re)
	h3re = fluid.h(p=p3re, T=t3re)
	x3 = fluid.T_s(s=s3re, p=p3re, quality=True)[1]
	State_3 = np.array([t3re, p3re, h3re, s3re, x3])
	#State 4 
	s4re_id = s3re
	t4re_id, x4 = fluid.T_s(s=s4re_id, p=p4re, quality=True)
	h4re_id = fluid.h(p=p4re, T=t4re_id, x=x4)

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
	#State 6
	s6re_id = s5re
	p6re = p1re
	T6re_id, x6_id = fluid.T_s(s=s6re_id, p=p6re, quality=True)
	h6re_id = fluid.h(p=p6re, T=T6re_id, x=x6_id)

	h6re = h5re + (h6re_id-h5re)/eta_p
	t6re, x6 = fluid.T_h(h=h6re, p=p6re, quality=True)
	s6re = fluid.h(T=t6re, p=p6re, x=x6)
	State_6 = np.array([t6re, p6re, h6re, s6re, x6])

	data = [State_1, State_2, State_3, State_4, State_5, State_6]
	re_df = pd.DataFrame(data=data, index=['State_1', 'State_2', 'State_3', 'State_4', 'State_5', 'State_6'],
		columns=['T [K]','P [MPa]', 'h [kJ/kg]', 's [kJ/(kg K)]','x [p.u]'])
	return re_df

print(reheat_Rankine())



    
#print(ideal_Rankine())
#print(real_Rankine())
#print(overheated_Rankine())