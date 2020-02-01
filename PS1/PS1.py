from scipy import optimize
import numpy as np
import itertools
import math
import scipy.stats as stats
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


class contInvest:
	"""
	contInvest defines the continous investment model with decreasing returns to scale
	as outlined in the The Theory of Corporate Finance, exercise 3.5. It includes a 
	number of plotting features for teaching purposes, such as interactive plots using
	widgets.
	"""

	def __init__(self,name="",**kwargs):
		self.name = name
		self.base_par()
		self.upd_par(kwargs)
		self.set_ftype()

	def base_par(self):

		self.Rtype='FlexiblePower' # Type of revenue function of investment.
		self.p = 0.3 # If Rtype='FlexiblePower', self.p indicates shape of revenue function, R = I^p.
		self.pH = 0.8 # probability of success, high
		self.pL = 0.5 # probability of success, low
		self.B = 0.5 # Linear benefits on investment when shirking

		self.plots = {'plot_eu': 'Plots the utility of entrepreneur on a grid of assets when behaving/not behaving.',
					  'plot_interactive_sol': 'Plots the equilibrium outcome, the unconstrained solution and the binding IC constraint on a grid of assets.'}

		self.plot_settings = dict()
		self.plot_settings['Amin'] = 0
		self.plot_settings['Amax'] = 2
		self.plot_settings['A_n'] = 25
		self.plot_settings['Imin'] = 0.1
		self.plot_settings['Imax'] = 2
		self.plot_settings['I_n'] = 25
		
		self.plot_settings['pHmin'] = self.pL+0.1
		self.plot_settings['pHmax'] = min(1,self.pL+0.4)
		self.plot_settings['pH_n'] = 25
		self.plot_settings['pHbase'] = min(12,self.plot_settings['pH_n']) # index between 0,pH_n

		self.plot_settings['Bmin'] = min(self.B,0.5)
		self.plot_settings['Bmax'] = max(self.B,2)
		self.plot_settings['B_n'] = 25
		self.plot_settings['Bbase'] = min(12,self.plot_settings['B_n']) # index between 0,pH_n

		self.grids = dict()
		self.grids['A'] = np.linspace(self.plot_settings['Amin'],self.plot_settings['Amax'],self.plot_settings['A_n'])
		self.grids['I'] = np.linspace(self.plot_settings['Imin'],self.plot_settings['Imax'],self.plot_settings['I_n'])
		self.grids['pH'] = np.round(np.linspace(self.plot_settings['pHmin'],self.plot_settings['pHmax'],self.plot_settings['pH_n']),2)
		self.grids['B'] = np.round(np.linspace(self.plot_settings['Bmin'],self.plot_settings['Bmax'],self.plot_settings['B_n']),2)

	def upd_par(self,kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)
		self.set_ftype()

	def set_ftype(self):
		if self.Rtype=='FlexiblePower':
			self.Rf = lambda I: np.power(I,self.p)
			self.Rfgrad = lambda I: self.p*np.power(I,self.p-1)
			self.Rsol = (self.p*self.pH)**(1/(1-self.p))
		elif self.Rtype=='ln':
			self.Rf = lambda I: np.log(I)
			self.Rfgrad = lambda I: 1/I
			self.Rsol = self.pH
		else:
			raise ValueError("Unknown return-on-investment function (Rtype)")

	def plot_eu(self):
		eu_diff_I = (self.pH-self.pL)*self.Rf(self.grids['I'])-self.B*self.grids['I']
		fig, axes = plt.subplots(1,1,figsize=(8,6))
		plt.subplot(1,1,1)
		plt.plot(self.grids['I'],eu_diff_I)
		plt.axhline(linestyle='--',linewidth=1,c='k')
		plt.xlabel('Investment')
		plt.ylabel('$u_e(b)-u_e(nb)$')
		plt.title('The entrepreneurs incentive to behave',fontweight='bold')
		fig.tight_layout()

	def solve(self,print_='Yes'):
		self.IR_constraint(print_)
		self.soleq()

	def IR_constraint(self,print_='Yes'):
		def f(x):
			return self.pH*(self.Rf(x)-self.B*x/(self.pH-self.pL))-(x-self.grids['A'])
		def grad(x):
			return np.diag(self.pH*(self.Rfgrad(x)-self.B/(self.pH-self.pL))-1)
		x0 = np.ones((self.grids['A'].size))
		self.IR,info,self.IR_ier,msg = optimize.fsolve(f,x0,fprime=grad,full_output=True)
		if print_=='Yes':
			return print(msg)
		elif self.IR_ier != 1:
			return print(msg)

	def soleq(self):
		self.sol_I = np.minimum(self.IR,self.Rsol*np.ones(self.grids['A'].size))

	def solgrid(self):
		def aux_sol(x):
			par = {'pH': x[0], 'B': x[1]}
			self.upd_par(par)
			self.set_ftype()
			self.solve('No')
			return {'IR': self.IR, 'Istar': np.ones(self.IR.size)*self.Rsol, 'Sol': self.sol_I}
		self.sol_grid = {x: aux_sol(x) for x in list(itertools.product(*[self.grids['pH'], self.grids['B']]))}
		self.sol_base = self.sol_grid[self.grids['pH'][self.plot_settings['pHbase']],self.grids['B'][self.plot_settings['Bbase']]]
		self.sol_grid_ylim = [contInvest.round_down(min([min([min(self.sol_grid[x][y]) for x in self.sol_grid.keys()]) for y in ['IR','Istar','Sol']]),1),contInvest.round_up(max([max([max(self.sol_grid[x][y]) for x in self.sol_grid.keys()]) for y in ['IR','Istar','Sol']]),1)]

	def plot_interactive_sol(self):
		try:
			getattr(self,"sol_grid")
		except AttributeError:
			self.solgrid()
		def plot_from_dict(ph,B):
			contInvest.plot_instance(self.sol_grid[ph,B]['IR'],self.sol_base['IR'],self.sol_grid[ph,B]['Istar'],self.sol_base['Istar'],self.sol_grid[ph,B]['Sol'],self.sol_base['Sol'],self.grids['A'],self.sol_grid_ylim)
		prob = widgets.SelectionSlider(
				description = "Probability, $p_H$",
				options = self.grids['pH'],
				style = {'description_width': 'initial'})
		benefit = widgets.SelectionSlider(
				description = "Private benefit, $B$",
				options = self.grids['B'],
				style = {'description_width': 'initial'})
		widgets.interact(plot_from_dict,
			ph = prob,
			B = benefit)

	@staticmethod
	def plot_instance(IR,IRbase,Istar,Istarbase,Sol,Solbase,Agrid,ylim=[]):
		fig = plt.figure(frameon=False,figsize=(8,6))
		ax = fig.add_subplot(1,1,1)
		ax.plot(Agrid,IR,linestyle='--',c='b')
		ax.plot(Agrid,Istar,linestyle='--',c='r')
		ax.plot(Agrid,Sol,linewidth=2,c='g')
		ax.plot(Agrid,IRbase,linestyle='--',c='gray',alpha=0.1)
		ax.plot(Agrid,Istarbase,linestyle='--',c='gray',alpha=0.1)
		ax.plot(Agrid,Solbase,linewidth=2,c='gray',alpha=0.1)
		if ylim:
			ax.set_ylim(ylim)
		ax.set_xlabel('Assets')
		ax.set_ylabel('Investment')
		plt.legend(('Binding IC constraint','Unconstrained solution', 'Equilibrium'), loc='upper left')
		fig.tight_layout()

	@staticmethod
	def round_up(n, decimals=0): 
		multiplier = 10 ** decimals 
		return math.ceil(n * multiplier) / multiplier

	@staticmethod
	def round_down(n, decimals=0): 
		multiplier = 10 ** decimals 
		return math.floor(n * multiplier) / multiplier

		
class poolingCredit:
	"""
	poolingCredit sets up the model with a continuum of agents of varying quality.
	"""

	def __init__(self,name="",**kwargs):
		self.name = name
		self.base_par()
		self.upd_par(kwargs)
		self.set_distr()

	def base_par(self):

		self.pH = 0.8
		self.pL = 0.5
		self.R = 10
		self.I = 2
		self.Lower = 0
		self.Upper = 10

		self.plots = {'plot_distr': 'Plots the pdf and cdf for the continuum of entrepreneurs'}

		self.plot_settings = dict()
		self.plot_settings['B_n'] = 100

		self.grids = dict()
		self.grids['B'] = np.round(np.linspace(self.Lower,self.Upper,self.plot_settings['B_n']),2)

	def upd_par(self,kwargs):
		for key,value in kwargs.items():
			setattr(self,key,value)		
		self.set_distr()

	def set_distr(self):
		self.distr = stats.uniform(loc=self.Lower, scale=self.Upper-self.Lower)

	def plot_distr(self):
		fig, axes = plt.subplots(1,2, figsize=(16,6))
		plt.subplot(1,2,1) 
		plt.plot(self.grids['B'],self.distr.pdf(self.grids['B']))
		plt.xlabel('$B$')
		plt.ylabel('$dH(\cdot)/dB$')
		plt.title('Density (pdf)')
		plt.ylim([0, 1])
		plt.subplot(1,2,2)
		plt.plot(self.grids['B'],self.distr.cdf(self.grids['B']))
		plt.xlabel('$B$')
		plt.ylabel('$H(\cdot)$')
		plt.title('Cumulative density (cdf)')
		plt.ylim([0, 1])
		fig.tight_layout()

	def plot_exp_profits(self):
		fig, axes = plt.subplots(1,1,figsize=(8,6))
		plt.subplot(1,1,1)
		profit,zeroprofit,ier = self.expected_profit(self.distr,self.pH,self.pL,self.R,self.I,1)
		plt.plot(self.grids['B'],profit(self.grids['B']))
		if ier ==1:
			plt.axvline(x=zeroprofit,color='k',linestyle='--')
		plt.axhline(y=0,color='k')
		plt.xlabel('$B$')
		plt.ylabel('$E[\pi(B)]$')
		plt.title('Expected profit on grid of $B$')
		if ier==1:
			plt.legend(('Expected profits','Level of $B$ implying zero profits'))
		fig.tight_layout()

	@staticmethod
	def expected_profit(distr,pH,pL,R,I,dim):
		profit = lambda B: distr.cdf(B)*pH*(R-B)+(1-distr.cdf(B))*pL*(R-B)-I
		zeroprofit,info,ier,msg = optimize.fsolve(profit,np.ones(dim),full_output=True)
		zeroprofit = np.round(zeroprofit,2)
		return profit,zeroprofit,ier