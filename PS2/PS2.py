import sys
from scipy import optimize
import numpy as np
import math
import itertools
import scipy.stats as stats
import scipy.integrate as integrate
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
mpl.style.use('seaborn')
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
import ipywidgets as widgets
from ipywidgets import interact, interact_manual


logger = logging.getLogger(__name__)


class TwoAgent_GE:
	"""
	TwoAgent_GE defines the general equilibrium of a 2-period model with two types
	of agents: Entrepreneurs and consumers. The '.plots' provides an overview of 
	the built-in plots included in the class, along with a description. 
	"""

	def __init__(self,name="",**kwargs):
		self.name = name
		self.base_par()
		self.upd_par(kwargs)

	def base_par(self):

		self.n = 0.01
		self.lambda_ = 0.05
		self.e = 5
		self.alpha = 0.5
		self.tau = 0
		self.max_ = 10**3 #ad hoc limit to graphs
		self.makegrids()

		self.plots = {'plot_interactive_ds': 'Plots demand for capital on grid of R. Includes interactive control of n and lambda.',
					  'plot_interactive_cequi': 'Plots equilibrium capital on grid of n. Includes interactive control of lambda.',
					  'plot_u_ce_tau': 'Plots the utility for consumers/entrepreneurs when varying a lump-sum transfer from consumers to entrepreneurs.'}

	def makegrids(self):
		self.ps = dict()
		self.ps['Rmin'] = 0.1
		self.ps['Rmax'] = 2
		self.ps['R_n']  = 100

		self.ps['nmin'] = 0
		self.ps['nmax'] = 0.5
		self.ps['n_n']  = 25
		self.ps['nbase'] = min(12,self.ps['n_n'])

		self.ps['lambda_min'] = 0
		self.ps['lambda_max'] = 0.9
		self.ps['lambda__n']  = 25
		self.ps['lambda_base'] = min(12,self.ps['lambda__n'])

		self.ps['taumin'] = 0
		self.ps['taumax'] = min(0.2,self.e)
		self.ps['tau_n'] = 100
		self.ps['tau_base'] = min(12,self.ps['tau_n'])
		
		self.grids = dict()
		self.grids['R'] = np.linspace(self.ps['Rmin'],self.ps['Rmax'],self.ps['R_n'])
		self.grids['n'] = np.round(np.linspace(self.ps['nmin'],self.ps['nmax'],self.ps['n_n']),2)
		self.grids['lambda_'] = np.round(np.linspace(self.ps['lambda_min'],self.ps['lambda_max'],self.ps['lambda__n']),2)
		self.grids['tau'] = np.round(np.linspace(self.ps['taumin'],self.ps['taumax'],self.ps['tau_n']),2)

	def upd_par(self,kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)
		self.makegrids()

	def dsgrid(self):
		def aux_sol(x):
			par = {'n': x[0], 'lambda_': x[1]}
			self.upd_par(par)
			return {'d': self.cap_d(self,self.grids['R']), 's': self.cap_s(self,self.grids['R'])}
		self.ds_grid = {x: aux_sol(x) for x in list(itertools.product(*[self.grids['n'],self.grids['lambda_']]))}
		self.ds_base = self.ds_grid[self.grids['n'][self.ps['nbase']],self.grids['lambda_'][self.ps['lambda_base']]]
		self.base_par() # reset parameter settings

	def capequigrid(self):
		def aux_sol(x):
			par = {'lambda_': x}
			self.upd_par(par)
			return {'E': self.cap_equi(self,self.grids['n'],self.tau)}
		self.c_grid = {x: aux_sol(x) for x in self.grids['lambda_']}
		self.c_base = self.c_grid[self.grids['lambda_'][self.ps['lambda_base']]]
		self.base_par() # reset parameter settings

	def plot_u_ce_tau(self):
		uc_grid = self.uc_equi(self.grids['tau'])
		ue_grid = self.ue_equi(self.grids['tau'])
		fig, axes = plt.subplots(1,2,figsize=(16,6))
		plt.subplot(1,2,1)
		plt.plot(self.grids['tau'],uc_grid)
		plt.xlabel('$\\tau$')
		plt.ylabel('$u_c$')
		plt.title('Utility of consumers',fontweight='bold')
		plt.subplot(1,2,2)
		plt.plot(self.grids['tau'],ue_grid)
		plt.xlabel('$\\tau$')
		plt.ylabel('$u_e$')
		plt.title('Utility of entrepreneurs', fontweight='bold')
		fig.tight_layout()

	def plot_interactive_cequi(self):
		try:
			getattr(self,"c_grid")
		except AttributeError:
			self.capequigrid()
		def plot_from_dict(lambda_):
			TwoAgent_GE.plot_c_instance(self.c_grid[lambda_]['E'],self.c_base['E'],self.grids['n'])
		lambda_slider = widgets.SelectionSlider(
				description = "Borrowing constr., $\\lambda$",
				options = self.grids['lambda_'],
				style = {'description_width': 'initial'})
		widgets.interact(plot_from_dict,
			lambda_=lambda_slider)

	def plot_interactive_ds(self):
		try:
			getattr(self,"ds_grid")
		except AttributeError:
			self.dsgrid()
		def plot_from_dict(n,lambda_):
			TwoAgent_GE.plot_ds_instance(self.ds_grid[n,lambda_]['d'],self.ds_base['d'],self.ds_grid[n,lambda_]['s'],self.ds_base['s'],self.grids['R'])
		nslider = widgets.SelectionSlider(
				description = "Endowment, $n$",
				options = self.grids['n'],
				style = {'description_width': 'initial'})
		lambda_slider = widgets.SelectionSlider(
				description = "Borrowing constr., $\\lambda$",
				options = self.grids['lambda_'],
				style = {'description_width': 'initial'})
		widgets.interact(plot_from_dict,
			n = nslider,
			lambda_ = lambda_slider)

	@staticmethod
	def plot_c_instance(c,cbase,ngrid):
		fig = plt.figure(frameon=False,figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(ngrid,cbase)
		ax.plot(ngrid,c)
		ax.set_ylabel('$k$ in equilibrium')
		ax.set_xlabel('Endowment $n$')
		plt.legend(('Baseline', 'Adjusted $\\lambda$'))
		plt.title('Equilibrium for varying parameter values')
		fig.tight_layout()

	@staticmethod
	def plot_ds_instance(d,dbase,s,sbase,Rgrid):
		fig = plt.figure(frameon=False,figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(Rgrid,dbase,linestyle='--',c=colors[0])
		ax.plot(Rgrid,d,c=colors[0])
		ax.plot(Rgrid,s,c=colors[1])
		ax.set_xlabel('$R$')
		ax.set_ylabel('Capital demand / interest rate function')
		ax.set_ylim([0, 5])
		plt.legend(('Demand (base)', 'Demand (adjusted)', 'Interest rate function'))
		plt.title('Demand and interest rate functions')
		fig.tight_layout()

	@staticmethod
	def cap_d(self,R):
		return (R>1)*(self.lambda_*R<1)*((self.n+self.tau)/(1-self.lambda_*R+sys.float_info.epsilon))+(self.lambda_*R>1)*self.max_

	@staticmethod
	def cap_s(self,R):
		return (self.alpha/R)**(1/(1-self.alpha))

	@staticmethod
	def cap_equi(self,n,tau):
		return ((n+tau)<(1-self.lambda_)*self.alpha**(1/(1-self.alpha)))*self.cap_constr(self,(n+tau))+((n+tau)>=(1-self.lambda_)*self.alpha**(1/(1-self.alpha)))*self.cap_star(self,(n+tau))

	@staticmethod
	def cap_constr(self,n):
		sol = np.zeros((n.size))
		for i in range(0,n.size):
			sol[i] = optimize.newton(lambda x: x-self.alpha*self.lambda_*x**(self.alpha)-(n[i]+self.tau),1)
		return sol

	@staticmethod 
	def cap_star(self,n):
		return self.alpha**(1/(1-self.alpha))*np.ones(n.size)

	@staticmethod
	def u_c(e,tau,alpha,k):
		return 2*e-tau+(1-alpha)*k**(alpha)

	def uc_equi(self,tau):
		return self.u_c(self.e,tau,self.alpha,self.cap_equi(self,self.n,tau))

	@staticmethod
	def u_e(n,tau,alpha,k):
		return n+tau+alpha*k**(alpha)-k

	def ue_equi(self,tau):
		return self.u_e(self.e,tau,self.alpha,self.cap_equi(self,self.n,tau))





class OptDebt:
	"""
	OptDebt defines the model of optimal debt with costly state verification.
	"""

	def __init__(self,name="",**kwargs):
		self.name = name
		self.base_par()
		self.makegrids()
		self.upd_par(kwargs)
		self.ftype()

	def base_par(self):
		self.I = 1
		self.R = 2
		self.sDelta = 1
		self.kappa = 0.1
		self.sdist = 'uniform'

	def makegrids(self):
		self.ps = dict()
		self.ps['sbarmin'] = 1-self.sDelta
		self.ps['sbarmax'] = 1+self.sDelta
		self.ps['sbar_n'] = 100

		self.ps['Imin'] = 0
		self.ps['Imax'] = 3
		self.ps['I_n'] = 100
		self.ps['Ibase'] = min(50,self.ps['I_n'])
		
		self.grids = dict()
		self.grids['sbar'] = np.linspace(self.ps['sbarmin'],self.ps['sbarmax'],self.ps['sbar_n'])
		self.grids['I'] = np.round(np.linspace(self.ps['Imin'], self.ps['Imax'], self.ps['I_n']),2)

	def upd_par(self,kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)
		self.makegrids()
		self.ftype()

	def ftype(self):
		"""
		NB: It is asssumed that the expected value of s is always 1.
		Thus the user can only choose a level of variance for the
		stochastic variable.
		"""
		if self.sdist == 'uniform':
			self.s_pdf = lambda s: (1/(2*self.sDelta))*(s>(1-self.sDelta))*(s<(1+self.sDelta))
			self.s_cdf = lambda s: ((s-(1-self.sDelta))/(2*self.sDelta))*(s>(1-self.sDelta))*(s<(1+self.sDelta))
		else:
			raise ValueError("Only 'uniform' distribution currently supported.")

	@np.vectorize
	def exp_profit(self,sbar):
		return integrate.quad(lambda s: self.s_pdf(s)*self.R*(s-sbar), sbar, 1+self.sDelta)[0]

	def vec_profit(self):
		return self.exp_profit(self,self.grids['sbar'])

	@np.vectorize
	def participation_constr(self,sbar):
		return integrate.quad(lambda s: self.s_pdf(s)*self.R*sbar, sbar,1+self.sDelta)[0]+integrate.quad(lambda s: self.s_pdf(s)*(self.R*s-self.kappa), 1-self.sDelta,sbar)[0]-self.I

	def vec_part(self):
		return self.participation_constr(self,self.grids['sbar'])

	def plt_profit_participationconstr(self):
		fig = plt.figure(frameon=False,figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(self.grids['sbar'],self.vec_profit())
		ax.plot(self.grids['sbar'],self.vec_part())
		ax.axhline(linestyle='--',linewidth=1,c='k')
		ax.set_xlabel('$sbar$')
		ax.set_ylabel('$E[\pi],(u(p)-u(np))$')
		plt.legend(('Expected profits','Participation Constraint'))
		plt.title('Expected profits and participation constraint',fontweight='bold')
		fig.tight_layout()

	def pc_grid_of_I(self):
		def aux_sol(x):
			par = {'I': x}
			self.upd_par(par)
			return {'I': self.vec_part()}
		self.I_grid = {x: aux_sol(x) for x in self.grids['I']}
		self.I_base = self.I_grid[self.grids['I'][self.ps['Ibase']]]
		self.base_par()

	@staticmethod
	def plot_pc_instance(I,Ibase,sgrid):
		fig = plt.figure(frameon=False,figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(sgrid,Ibase)
		ax.plot(sgrid,I)
		ax.axhline(linestyle='--',linewidth=1.5, c='k')
		ax.set_ylabel('Expected gain from participation')
		ax.set_xlabel('Audit threshold $s$')
		ax.set_ylim([-4, 2])
		plt.legend(('Baseline', 'Adjusted $I$'))
		plt.title('Participation constraint for various $I$', fontweight='bold')
		fig.tight_layout()

	def plot_interactive_pc(self):
		try:
			getattr(self,"I_grid")
		except AttributeError:
			self.pc_grid_of_I()
		def plot_from_dict(I):
			OptDebt.plot_pc_instance(self.I_grid[I]['I'],self.I_base['I'],self.grids['I'])
		I_slider = widgets.SelectionSlider(
				description = "Investment cost, $I$",
				options = self.grids['I'],
				style = {'description_width': 'initial'})
		widgets.interact(plot_from_dict,
			I=I_slider)
