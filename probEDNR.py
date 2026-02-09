from enum import Enum
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib as mpl
import matplotlib.pyplot as plt  
from dataclasses import dataclass, field
from types import SimpleNamespace
from copy import deepcopy
import types
from IPython.display import clear_output
# from distfit import distfit
import scipy.stats as stats 
import yaml
# import pandas as pd
# import xarray as xr
from datetime import datetime
import logging
import sys

from contextlib import contextmanager
import time
# from scipy.special import erfinv
# from sklearn.model_selection import train_test_split
# from sklearn.utils import resample


if (__name__=="__main__"):
    print("Hello world!")

@contextmanager
def timed(title:str,logger:logging.Logger,show=True): # with timed(): ...
    start_time = time.time()
    if show: logger.info(f"====== starting: {title} ======")
    yield # returns a generator!
    end_time = time.time()
    diff=end_time-start_time
    if show: logger.warning(f'====== {title} took: {diff*1000:.2f} ms ======')

def vectoprint(arr,n=1,perc=False):
    s="%" if perc else "f"
    return [f"[{i}]{el:.{n}{s}}" for i,el in enumerate(arr)]    

## Generator Class
@dataclass
class Generator:
    type: str
    rate_copkwh: float
    power_perunit_kw: list
    power_kw: float = field(init=False)
    gen_curve_pu: list = field(kw_only=True,default_factory=lambda: [])
    intermittent: bool = field(kw_only=True,default=False)
    dispatchable: bool = field(kw_only=True,default=True)
    IBR: int = field(kw_only=True,default=0)
    UR: float = field(kw_only=True,default=0)
    DR: float = field(kw_only=True,default=0)
    T: int = field(kw_only=True,default=24)
    def __post_init__(self):
        if len(self.power_perunit_kw)==0:
            raise Exception("useless generator")
        else:
            self.power_kw=sum(self.power_perunit_kw)
        match self.IBR:
            case 0: # Regular syn Gen, not an IBR. Does ED+R
                self.dispatchable=True
                self.intermittent=False
            case 1: # Fully P-controllable IBR, fixed bounds, like synGens. Does ED+R
                self.dispatchable=True
                self.intermittent=False
            case 2: # Non P-controllable IBR, e.g. MPPT SPV. Does ED, no R, random DeltaPout
                self.dispatchable=False
                self.intermittent=True
            case 3: # Short-term P-controllable IBR from intermittent source. Does ED+R.
                self.dispatchable=True
                self.intermittent=True
            case _:
                raise Exception("undefined IBR type")

        if not self.intermittent:
            self.gen_curve_pu=np.ones(self.T)
        elif not len(self.gen_curve_pu)==self.T:
            raise Exception(f"curve needs {self.T} periods")
        elif max(self.gen_curve_pu)>1:
            raise Exception("curve is in p.u., <=1")
        if self.UR==0:
            self.UR=10*self.power_kw
        if self.DR==0:
            self.DR=10*self.power_kw

## BESS Class
@dataclass
class BESS:
    nmods: int = field(init=True)
    # default is Pylontech US5000, eff considering BMS/CC,inverter
    ch_eff: float = 0.86
    dc_eff: float = 0.83
    lifecycle: float = 8000 
    capcost_copkwh: float = field(default=1.2e6,repr=False) 
    modulecap_kwh: float = field(default=0.95*4.8,repr=False) #0.95 DoD
    modulepower_kw: float = field(default=4.8,repr=False)
    cap_kwh: float = field(init=False)
    power_kw: float = field(init=False)
    power_kw_ch: float = field(default=None)
    power_kw_dc: float = field(default=None)
    syscost_cop: float = field(init=False)
    throughput_kwh: float = field(init=False)
    dischargecostperkwh: float = field(init=False)

    def __post_init__(self):
        n=self.nmods
        self.cap_kwh=self.modulecap_kwh*n
        cap=self.cap_kwh
        self.power_kw=self.modulepower_kw*n
        self.power_kw_ch=self.power_kw if self.power_kw_ch is None else self.power_kw_ch # used for constr bounds
        self.power_kw_dc=self.power_kw if self.power_kw_dc is None else self.power_kw_dc # used for constr bounds
        self.syscost_cop=self.capcost_copkwh*cap
        self.throughput_kwh=self.lifecycle*cap
        self.dischargecostperkwh=self.syscost_cop/self.throughput_kwh

## Microgrid Class
@dataclass
class MG:
    peak_demand_kw: float = field(init=True)
    demand_curve_pu: list = field(init=True)
    demand_curve_kw: list = field(init=False,default_factory=lambda: [])
    has_bess: bool = field(default=False)
    BESS: object = field(default=None)
    Gens: tuple = field(default=())
    def __post_init__(self):
        if not len(self.demand_curve_pu)==24:
            raise Exception("bruh, 24 hours")
        elif max(self.demand_curve_pu)>1:
            raise Exception("in p.u. means at most 1.0")
        else:
            self.demand_curve_pu=np.array(self.demand_curve_pu)
            self.demand_curve_kw=self.demand_curve_pu*self.peak_demand_kw
        if len(self.Gens)==0:
            raise Exception("No Generators defined")
        else:
            s=w=0
            for g in self.Gens:
                if g.IBR==2: # MPPT
                    if g.type in {'SPV','PV','Solar'}:
                        s+=1
                        if s>1: raise Exception("ONLY ONE TYPE 2 SOLAR GEN PER MG (represents aggregate)")
                    elif g.type in {'WP','WT','Wind'}:
                        w+=1
                        if w>1: raise Exception("ONLY ONE TYPE 2 WIND GEN PER MG (represents aggregate)")
                    else:
                        raise Exception("Undefined Type 2 Gen")
        if self.has_bess and self.BESS is None:
            print("please specify BESS")

class genericStochasticProgram:
    def __init__(self,logger_scope:int,logger_level:int):
        self.logger = logging.getLogger(__name__)
        fmt_strs=["> %(message)s",
                 "[%(lineno)2s - %(funcName)2s] %(message)s",
                 "[%(lineno)2s - %(levelname)-4s - %(funcName)2s] %(message)s"]
        logging.basicConfig(format=fmt_strs[logger_scope],stream=sys.stdout)
        self.logger.setLevel(logger_level)
        self.hasBeenSolved=False
        self.showtimed=False
        """
        Generic program optimizing J(x,xi). Takes a training sample {Xi_in},
        makes a decision x_dec=argmin(SP), with expected in-sample cost Jis=min(SP).
        Can be tested on points xi_out belonging to a testing sample {Xi_out},
        to obtain point performance Joos_i=J(x_dec,xi_out) and average performance E[Joos]=avg(Joos_i).
        [CC testing] point performance test also returns constraint violation freq PrViol.
        """
    def debug(self,name:str,element,endl=True):
        """Just a quick debugger on self.logger
        Args:
            name (str): description of thingy
            element (_type_): thingy to debug
        """
        self.logger.debug(f"===== {name}: =====\n{element}{"\n" if endl else ""}")

    def Joos_i(self,x_dec,testSample,**kwargs):
        """
        Get out of sample cost/performance Joos_i=J(x_dec,xi_i) and PrViol of decision x_dec on a test sample point xi_i.
        On some problems point performance test can only return PrViol in {0,1},
        on ED it considers all daily subsamples so PrViol in [0,1]. 
        """
    def Joos(self,x_dec,testSampleSet,**kwargs):
        """
        Get AVERAGE performance of decision x_dec: out of sample cost Joos=E[J(x_dec,xi)] and PrViol, over a test sample set Xitest={xi}.
        """
        #repacking in case several PrViol
        Joos_,*PrViol_=zip(*[self.Joos_i(x_dec,samp,**kwargs) for samp in testSampleSet])
        self.logger.info(f"Joos_: {Joos_}")
        self.logger.info(f"PrViol_: {PrViol_}")
        Joos,PrViol=np.mean(Joos_),list(np.mean(PrViol_,axis=1))
        self.logger.info(f"Joos,PrViol: {Joos,PrViol}")
        return Joos,PrViol
    def solve(self,trainSampleSet,**kwargs):
        """
        Solves program with params, using training sample set. Of course, D,S,R,DR specifics may use one training sample, average, whole Set, etc.
        Returns decision x_dec, and in-sample performance Jis.     
        """
        raise NotImplementedError("solve() must be implemented in child class")

    def resolve(self,trainSampleSet):
        """
        Update and resolve model with new training sample set and/or new ADMM constants. Implement in child class.
        """
        raise NotImplementedError("resolve() must be implemented in child class")
        
    def solveTest(self,trainSampleSet,testSampleSet,**kwargs):
        """
        Calls solve(), gets decisions x_dec from trainSampleSet, tests its performance with Joos() over testSampleSet.
        Returns (x_dec, Jis, Joos, reliability, PrViol) point [corresponding to given trainSampleSet Xi_hat]
        """

        if not self.hasBeenSolved:
            with timed("solveTest calling solve()",self.logger,self.showtimed):
                x_dec,Jis=self.solve(trainSampleSet,**kwargs)
        else:
            with timed("solveTest calling resolve()",self.logger,self.showtimed):
                x_dec,Jis=self.resolve(trainSampleSet,**kwargs)
        
        # Whether solved or resolved, test the decision
        with timed("solveTest calling Joos()",self.logger,self.showtimed):
            Joos,PrViol=self.Joos(x_dec,testSampleSet)
        reliability=float((Joos<=Jis))
        self.logger.info(f"Jis: {Jis}")
        self.logger.info(f"Joos: {Joos}")
        self.logger.info(f"rel: {reliability}")
        self.logger.info(f"prvio: {PrViol}")
        #  PrViol is always a vector
        return x_dec,Jis,Joos,reliability,PrViol
    
    ###################
    ### INTERFACES FOR DOING
    # for s in trainsets: for p in params: solvetest(p,s)
    # results=(Nexperiments,Nparams)

    def iter_params(self,trainSampleSet,testSampleSet,paramRange):
        """
        Meant to be used by simulate(), NOT DIRECTLY.
        Generator for iterating over solveTest() while varying parameters over a given range.
        """       
        # self.logger.critical(f"paramrange: {paramRange}")
        dontupdateTrainSample=False
        for i,params in enumerate(paramRange):
            # yields a tuple ( , , , []) for every p
            self.logger.warning(f"=== Params[{i}] : {params} ===")
            if i>0: dontupdateTrainSample=True
            yield self.solveTest(trainSampleSet,testSampleSet,dontupdateTrainSample=dontupdateTrainSample,**params)

    def simulate(self,trainSampleSet,testSampleSet,paramRange):
        """
        Iteratively do solveTest() with trainSampleSet and testSampleSet over a parameter range using iter_params().
        Returns x_dec, Jis, Joos, rel as reorganized independent vectors.
        """

        # [(x,y,z) for x,y,z in iter_params()] builds result tuples for every iteration
        # zip returns a tuple iterator, which fills up output vectors iter by iter
        # so zip(*[]) basically reshapes tuples into separate same-sized vectors   
        x_dec,Jis,Joos,rel,prviol = zip(*[(_x, _ji, _jo, _r, _pv) for _x, _ji, _jo, _r, _pv in self.iter_params(trainSampleSet,testSampleSet,paramRange)])
        Jis=list(Jis)
        Joos=list(Joos)
        rel=list(rel)
        prviol=list(prviol)            
        self.logger.info(f"Joos: {Joos}")
        self.logger.info(f"Jis: {Jis}")
        self.logger.info(f"prviol: {prviol}")
        self.logger.info(f"rel: {rel}")
        return x_dec,Jis,Joos,rel,prviol
    
    def iter_traindata_overparams(self,trainSampleSets,testSampleSet,paramRange):
        """
        Meant to be used by runSimulations(), NOT DIRECTLY.
        Generator for iterating over simulate() while varying the training sample set used used to solve, over a given set of datasets.
        for trainsets: for params: solvetest()
        """
        # self.logger.critical(f"paramrange: {paramRange}")

        for i,dataset in enumerate(trainSampleSets):
            self.logger.error(f"=== Experiment Set #{i} ===")
            yield self.simulate(dataset,testSampleSet,paramRange)
    ###################


    ###################
    ### INTERFACES FOR DOING
    # for p in params: for s in trainsets: solvetest(p,s)
    # results=(Nexperiments,Nparams) # SAME SHIT
    def iter_samples(self,trainSampleSets,testSampleSet,params):
        """
        Meant to be used by simulate(), NOT DIRECTLY.
        Generator for iterating over solveTest() while varying parameters over a given range.
        """       
        # self.logger.critical(f"paramrange: {paramRange}")

        for i,dataset in enumerate(trainSampleSets):
            # yields a tuple ( , , , []) for every p
            self.logger.warning(f"=== Experiment Set #{i} ===")
            yield self.solveTest(dataset,testSampleSet,**params)
    
    def simulate_over_samples(self,trainSampleSets,testSampleSet,params):
        x_dec,Jis,Joos,rel,prviol = zip(*[(_x, _ji, _jo, _r, _pv) for _x, _ji, _jo, _r, _pv in self.iter_samples(trainSampleSets,testSampleSet,params)])
        Jis=list(Jis)
        Joos=list(Joos)
        rel=list(rel)
        prviol=list(prviol)
        self.logger.info(f"Joos: {Joos}")
        self.logger.info(f"Jis: {Jis}")
        self.logger.info(f"prviol: {prviol}")
        self.logger.info(f"rel: {rel}")
        return x_dec,Jis,Joos,rel,prviol
    
    def iter_params_overtraindata(self,trainSampleSets,testSampleSet,paramRange):
        """
        Meant to be used by runSimulations(), NOT DIRECTLY.
        Generator for iterating over simulate_over_samples() while varying the params used to solve, over a given set of params.
        for params: for trainsets: solvetest()
        """
        for i,params in enumerate(paramRange):
            # yields a tuple ( , , , []) for every p
            self.logger.error(f"=== Params[{i}] : {params} ===")
            yield self.simulate_over_samples(trainSampleSets,testSampleSet,params)

    ###################


    def runSimulations(self,trainSampleSets,testSampleSet,paramRange:list=[{}],big_for_is_params=False):
        """
        Iteratively do experiments with simulate() with a set of different trainSample input sets using iter_traindata(),
        in order to obtain robust measurements of performance for the stochastic program, over a given parameter range.
        Obtains E[x_dec], E[Jis], E[Joos], Q20[Joos], Q80[Joos], E[rel] vectors, calculated over the different training experiments.
        """
        # self.logger.critical(f"paramrange: {paramRange}")

        if big_for_is_params:
            x_dec,Jis,Joos,rel,prviol=zip(*self.iter_params_overtraindata(trainSampleSets,testSampleSet,paramRange))
            Jis=np.transpose(list(Jis))
            Joos=np.transpose(list(Joos))
            rel=np.transpose(list(rel))
            prviol=np.transpose(list(prviol),axes=(1,0,2))

        else:
            x_dec,Jis,Joos,rel,prviol=zip(*self.iter_traindata_overparams(trainSampleSets,testSampleSet,paramRange))
            Jis=np.array(list(Jis))
            Joos=np.array(list(Joos))
            rel=np.array(list(rel))
            prviol=np.array(list(prviol))

        self.logger.info(f"Jis: {Jis}")
        self.logger.info(f"prvio: {prviol}")
        self.logger.info(f"rel: {rel}")
        # jis and joos are [ [] [] ] of Nexperiments x Nparams  
        # results statistics are of len Nparams 
        self.avgJis=np.mean(Jis,axis=0)
        self.q25Jis=np.quantile(Jis,0.25,axis=0)
        self.q75Jis=np.quantile(Jis,0.75,axis=0)

        self.avgJoos=np.mean(Joos,axis=0)
        self.q25Joos=np.quantile(Joos,0.25,axis=0)
        self.q75Joos=np.quantile(Joos,0.75,axis=0)
        
        self.avgRel=np.mean(rel,axis=0)
        # in case several PrViol result is Npar x 2 e.g.
        self.avgPrViol=np.mean(prviol,axis=0)
        self.q25Prviol=np.quantile(prviol,0.25,axis=0)
        self.q75Prviol=np.quantile(prviol,0.75,axis=0)

        ## Define avg( decision ) in Child
        # self.avgDec=np.mean(x_dec,axis=0)
        return x_dec,Jis,Joos,rel,prviol

# How to use genericStochasticProgram:
# class ProgExampleChild(genericStochasticProgram):
#     def __init__(self,...,**kwargs):
#     #####
#     def solve(self,trainSampleSet,**kwargs):
#     # Metodo para optimizar/decidir x_dec con base en trainSampleSet
#     #####
#     x_dec=None
#     return x_dec
    
#     def Joos_i(self, x_dec, testSamplePoint,**kwargs):
#     # Calcular costo de x_dec en el escenario testSamplePoint
#     Joos_i=None
#     return Joos_i
    
#     def generateSamples(self,N,**kwargs):
#     SampSet=[None for i in range(N)]
#     return SampSe
    
# instanceparams1={...}
# Child1=ProgExampleChild(instanceparams1)
# trainSet=Child1.generateSamples(N=50,...)
# testSet=Child1.generateSamples(N=1000,...)
# (sol1,Jis1,Joos1,rel1,PrViol1)=Child1.solveTest(trainSet,testSet)
# #####
# paramRange=[{alpha=1},{alpha=2},...]
# (sols,Jises,Jooses,rels,PrViols)=Child1.simulate(paramRange,trainSet,testSet)
# #####
# trainSets=[Child1.generateSamples(N=50,...) for i in range(200)]
# Child1.runSimulations(paramRange,trainSets,testSet)
# Results=(Child1.avgJis,Child1.avgJoos,Child1.q25Joos,Child1.q75Joos,Child1.avgRel,Child1.avgPrViol)

#Wrapper class for Economic Dispatch with Reserve. Inherits genericStochasticProgram test methods.


class EDnR(genericStochasticProgram):
    """
    Generic Economic Dispatch with Reserve Wrapper Class. Sample generation functions and misc helper functions.
    """
    def __init__(self,MG:MG,subperiods:int=30,strictlycircularbess:bool=True,BESS_SOE_init=0.0,
                 plotcolors=None,seed=None,grb_verbose:bool=False,logger_scope:int=1,
                 logger_level:int=logging.CRITICAL,LastInstance=None,model_name=None,needMPO:bool=False,**kwargs):
        """
        :param MG:
            (MG). Microgrid to use.
        :param subperiods:
            (int, optional) Defaults to 30. Number of Operation intrahour subperiods.
        :param strictlycircularbess:
            (bool, optional) Defaults to True. BESS in ED has circular dynamics (0:00=24:00). Generator UR/DR are always treated as circular.
        :param BESS_SOE_init:
            (float, optional) Defaults to 0.0. If BESS is not circular, starting SOE (in %)
        :param plotcolors:
            (string, optional) Defaults to 'Set1'. One of matplotlib.colormaps (name string)
        :param seed:
            (Any, optional) Defaults to None. Rng seed.
        :param grb_verbose:
            (bool, optional) Defaults to False. Show Gurobi output.
        :param logger_scope:
            (int, optional) Defaults to 1. Show function name in logger string.
        :param logger_level:
            (int, optional) Defaults to logging.CRITICAL.
        :param LastInstance:
            (string, optional) Defaults to None. Name of Last Instance Gen.
        :param model_name:
            (string, optional) Defaults to DateHour. nName of Gurobi Model for setup.
        :param needMPO:
            (bool, optional) Defaults to False. Set True to be able to calculate MPO in solve().
        :param transactive:
            (bool, optional) Defaults to 0. Set True for ADMM.
        :param rho:
            (float, optional) Defaults to 1. For ADMM.
        :param neighbors:
            (list, optional) Defaults to empty list. For ADMM.
        :param lineCapacities:
            (list, optional) Defaults to list of 0 of len=lenneighbors. For ADMM
        :param lineCosts:
            (float, optional). For ADMM
        :param z_PC_k,z_PV_k,lam_C_k,lam_V_k:
            (float, optional) Defaults to list of 0 of len=lenneighbors. 0th iteration For ADMM.
        :param xi_hat:
            (NDarray, optional) Defaults to dummy zeros matrix. Input training sample (Nsamples x Dimension).
        """
        super().__init__(logger_scope=logger_scope,logger_level=logger_level)
        self.MG=deepcopy(MG) # deepcopy to avoid modifying original MG
        self.T=24
        self.subperiods=subperiods
        if plotcolors is not None:
            assert plotcolors in mpl.colormaps(), f"plotcolors must be a matplotlib.colormaps. Available: {mpl.colormaps()}"
            self.plotcolors=mpl.colormaps[plotcolors].colors
        else:    
            self.plotcolors=mpl.colormaps['Set1'].colors
        self.rng=np.random.default_rng(seed)
        self.grb_verbose=grb_verbose
        self.strictlycircularbess=strictlycircularbess
        assert BESS_SOE_init>=0 and BESS_SOE_init<=1, "BESS_SOE_init not in [0,1]"
        self.BESS_SOE_init=BESS_SOE_init
        
        # MG Nominal Demand
        self.peak_demand_kw=self.MG.peak_demand_kw #IT WILL BE STATIC
        self.demand_curve_pu=self.MG.demand_curve_pu
        # sample generation needs to use nominal (stable) data
        self.peak_demand_kw_smpgen=deepcopy(self.MG.peak_demand_kw) # SO THIS IS NOT NECCESARY BUT WHATEV
        self.demand_curve_pu_smpgen=deepcopy(self.MG.demand_curve_pu)
        
        self.demand_curve_kw=self.MG.demand_curve_kw
        if not len(self.demand_curve_pu)==self.T:
            raise Exception("MG's demand curve length is not 24")
        else:
            for g in self.MG.Gens:
                if g.intermittent and not len(g.gen_curve_pu)==self.T:
                    raise Exception(f"{g.type} gen curve length doesn't match options.T")
        # MG.Gens Params
        self.Ngen=len(self.MG.Gens)
        self.UR_kwhr=np.array([g.UR for g in self.MG.Gens])
        self.DR_kwhr=np.array([g.DR for g in self.MG.Gens])
        self.c_gen_copkwh=np.array([g.rate_copkwh for g in self.MG.Gens])
        self.gennames=[g.type for g in self.MG.Gens]
        # Check if Type 2 (MPPT, random) generation available
        self.solarMPPTavailable=self.windMPPTavailable=False
        for i,g in enumerate(self.MG.Gens):
            if g.IBR==2:
                match g.type:
                    case 'SPV'|'PV'|'Solar':
                        # ONLY ONE TYPE 2 SOLAR GEN PER MG (represents aggregate)
                        self.idx_solarMPPT=i 
                        self.solarMPPTavailable=True
                        self.solar_gen_curve_kw_smpgen=deepcopy(np.array(g.gen_curve_pu)*g.power_kw)
                        self.solar_powernom_kw_smpgen=deepcopy(g.power_kw)
                    case 'WP'|'WT'|'Wind':
                        # ONLY ONE TYPE 2 WIND GEN PER MG (represents aggregate)
                        self.idx_windMPPT=i 
                        self.windMPPTavailable=True
                        self.wind_gen_curve_kw_smpgen=deepcopy(np.array(g.gen_curve_pu)*g.power_kw)
                        self.N100kWT_smpgen=len(g.power_perunit_kw)
                    case _:
                        raise Exception("Undefined Type 2 Gen")

        # Add BESS Params
        self.HAS_BESS=self.MG.has_bess
        if(self.HAS_BESS):
            self.deltat_h=24/self.T
            self.p_dc_max_bess_kw=self.MG.BESS.power_kw_dc
            self.p_ch_max_bess_kw=self.MG.BESS.power_kw_ch
            self.CE_bess_kwh=self.MG.BESS.cap_kwh
            self.eta_ch=self.MG.BESS.ch_eff
            self.eta_dc=self.MG.BESS.dc_eff
            self.c_chdc_copkwh=self.MG.BESS.dischargecostperkwh

        if LastInstance is not None:
            assert not LastInstance=='BESS', "BESS not allowed as Last Instance"
            # Get the Last Instance Gen
            for g in self.MG.Gens:
                if g.type==LastInstance:
                    assert g.dispatchable, "Last Instance Not Dispatchable"
                    self.LastInstance=LastInstance
                    break
            else: #for-break-else is pretty cool
                raise Exception(f"Last Instance Not Found: {LastInstance}")

        self.needMPO=needMPO

        self.transactive=kwargs.get("transactive",False)
        # When inititating in TRANSACTIVE MODE
        # Instance must initiate with Neighbors, linecaps, linecosts, rho
        if self.transactive:
            assert "neighbors" in kwargs, "neighbors not set"
            assert "rho" in kwargs, "rho not set"
            assert "lineCapacities" in kwargs, "lineCapacities not set"
            assert "lineCosts" in kwargs, "lineCosts not set"
            self.neighbors=kwargs.get("neighbors",[])
            self.lenneighbors=len(self.neighbors)
            if self.lenneighbors==0:
                raise Exception("transactive EDnR requires neighbors list")
            self.rho=kwargs.get("rho") # ADMM penalty param
            self.lineCapacities=kwargs.get("lineCapacities") # Max line capacity with each neighbor (kW)
            self.lineCosts=kwargs.get("lineCosts") # Cost of energy transacted with neighbors ($/kWh)
            # Z AND LAMBDAS ARE SET ON SOLVE/RESOLVE
            self.logger.warning(f"Instance set for transactive EDnR. neighbors: rho: {self.rho}, linecaps: {self.lineCapacities}, linecosts: {self.lineCosts}")
        else:
            self.logger.warning("Instance set for non-transactive EDnR")   
            
        if "xi_hat" not in kwargs:
            self.logger.warning("No training sample provided for Model. Set when solving (non det-nominal).")
        else:
            raise NotImplementedError("For now, just set the trainsample when solving")
            # self.createTrainSampleSetVar(kwargs.get("xi_hat"))
        
        # Create Gurobi Model
        self.M=self.CreateModel(model_name,**kwargs)
        self.sol_time=[]
        # PDF info are init attributes    
        with open('SolarPDF.yaml','r') as f:    
            self.solarinfo=yaml.safe_load(f)

        with open('WindPDF.yaml','r') as f:    
            self.windinfo=yaml.safe_load(f)
    

    def CreateModel(self,model_name=None,**kwargs):
        '''
        Create Economic Dispatch with Reserve Base Gurobi model.
        Makes variables/constraint handlers accessible as self.attributes
        '''
        ### CREATE GUROBI MODEL
        model_name=kwargs.get("model_name",datetime.strftime(datetime.today(),"%b%d_%H%M%S"))
        grb_licence_params=kwargs.get("grb_licence_params",None)
        env=gp.Env(params=grb_licence_params)
        M=gp.Model(f"EDnR_{model_name}",env=env)
        M.params.LogToConsole=self.grb_verbose
        self.logger.info(f"Model \"{model_name}\" created")

        ### MAIN VARIABLE
        self.pG=M.addMVar(shape=(self.Ngen,self.T),name="pG",vtype=GRB.CONTINUOUS)
        
        ### BASE OBJECTIVE FUNCTION
        self.fobj=gp.LinExpr()
        for t in range(self.T):
            self.fobj+=self.c_gen_copkwh @ self.pG[:,t]
        # SET OBJECTIVE FUNCTION
        M.setObjective(self.fobj,GRB.MINIMIZE)   
        M.update()
        return M

    def createXiScenarioSetVar(self,XiScenarioSet):
        """
        Input training sample as param in Gurobi model, in case used.
        """
        #xihat_dim may be a list
        xihat_num,xihat_dim=XiScenarioSet.shape
        self.xihat_var=self.M.addMVar(shape=(xihat_num,xihat_dim),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY,ub=GRB.INFINITY)
        self.samplectrt=self.M.addConstr(self.xihat_var==XiScenarioSet,name="samplectrt")
        self.M.update()
        self.logger.info("training sample constraint built")

    def solveOrResolve(self,trainSampleSet,dontupdateTrainSample=False,**kwargs):
        """
        If hasbeensolved, and given sampleset is same size as current OR currently no sample yet
        (i.e. only det-nominal has been run) calls resolve(), else createmodel() and calls solve().
        """
        fullretry=kwargs.get("fullretry",False)
        if fullretry: self.logger.warning("===== Doing a fullretry =====")
        self.logger.info(f"kwargs passed: {kwargs}")

        if not fullretry and not self.hasBeenSolved:
            self.logger.warning("Solving for the 1st time.")
            self.logger.info(f"Currently: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            with timed("solveOrResolve calling solve(), first time",self.logger,self.showtimed):
                x_dec,Jis=self.solve(trainSampleSet,**kwargs)
            self.logger.info(f"After 1st Solve: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")

        elif not fullretry and not hasattr(self,'samplectrt'):
            self.logger.warning("Resolving. No sample ctrt.")
            self.logger.info(f"Currently: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            with timed("solveOrResolve calling resolve(), no sampl ctrt",self.logger,self.showtimed):
                x_dec,Jis=self.resolve(False,trainSampleSet,**kwargs)
            self.logger.info(f"After resolve: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")

        elif not fullretry and (self.samplectrt.RHS.shape[0]==len(trainSampleSet)):
            self.logger.debug(f"samplctrt is={self.samplectrt.RHS.shape}, given set len={len(trainSampleSet)}")
            if dontupdateTrainSample: self.logger.warning("Not updating trainSample, only other params")
            else: self.logger.warning("Resolving, updating samplectrt and/or other params.")
            self.logger.info(f"Currently: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            with timed("solveOrResolve calling resolve(), updating sampl ctrt",self.logger,self.showtimed):
                x_dec,Jis=self.resolve(True,trainSampleSet,dontupdateTrainSample=dontupdateTrainSample,**kwargs) 
            self.logger.info(f"After resolve: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
               
        else:
            if hasattr(self,'samplectrt'):
                self.logger.debug(f"samplctrt={self.samplectrt.RHS.shape}, given set={len(trainSampleSet)}")
            self.logger.warning("Recreating whole model and solving.")
            self.logger.info(f"Currently: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            # Recreate Gurobi Model
            with timed("solveOrResolve recreating model and calling solve()",self.logger,self.showtimed):
            
                self.M.remove(self.M.getVars())
                self.M.remove(self.M.getConstrs())
                self.M.remove(self.M.getGenConstrs())
                self.M.update()
                self.logger.info(f"After deletion: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
                
                self.M=self.CreateModel(**kwargs)
                self.M.update()
                self.logger.info(f"After creation+update: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
                
                # self.sol_time=[]
                x_dec,Jis=self.solve(trainSampleSet,**kwargs)
              
            self.logger.info(f"After new solve: {self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")

            ## Remember i shouldnt do:
            # self.M.remove(self.samplectrt)
            # self.createXiScenarioSetVar(XiScenarioSet)  
            # self.logger.critical("training sample dimension changed. recreated. Fobj 2ndStg costs is most likely incoherent because Nscenarios changed.")

        if Jis==-1:
            with timed("solveOrResolve failed, calling solveOrResolve()",self.logger):

                self.logger.critical(f"===== FAILED =====\nCurrent kwargs: {kwargs}")
                if self.HAS_BESS:
                    if max(self.T_RB_ch_h,self.T_RB_dc_h,self.fpart_bess)<0.01:
                        raise Exception(f"Retry with different params. kwargs: {kwargs}")
                    kwargs["T_RB_dc_h"]=np.round(self.T_RB_dc_h*0.8,3)
                    kwargs["T_RB_ch_h"]=np.round(self.T_RB_ch_h*0.8,3)
                    kwargs["fpart_bess"]=np.round(self.fpart_bess*0.8,3)
                
                cresrv=self.customReserve
                if min(cresrv["up"],cresrv["down"])>self.peak_demand_kw*4:
                    raise Exception(f"UNFEASIBLE. Retry with different params. Currently: {kwargs} ")
                    # self.logger.critical(f"===== UNFEASIBLE =====\nCurrent kwargs: {kwargs}")
                    # return None,-1
                    
                kwargs["customReserve"]={"up":np.round(cresrv["up"]*1.2,0),
                                            "down":np.round(cresrv["down"]*1.2,0)}
                kwargs["fullretry"]=True
                self.logger.critical(f"RETRYING WITH KWARGS: {kwargs}")
                x_dec,Jis=self.solveOrResolve(trainSampleSet,**kwargs)

        self.logger.info(f"Finally, Jis: {Jis:,.0f}")
        return x_dec,Jis


    def solveTest(self,trainSampleSet,testSampleSet,**kwargs):
        """
        Calls solveOrResolve(), gets decisions x_dec from trainSampleSet, tests its performance with Joos() over testSampleSet.
        Returns (x_dec, Jis, Joos, reliability, PrViol) point [corresponding to given trainSampleSet Xi_hat]
        """
        with timed("solveTest calling solveOrResolve()",self.logger,True):
            x_dec,Jis=self.solveOrResolve(trainSampleSet,**kwargs)
        # Whether solved or resolved, test the decision
        if x_dec is not None:
            with timed("solveTest calling Joos()",self.logger,True):
                Joos,PrViol=self.Joos(x_dec,testSampleSet,**kwargs)
            reliability=float(Joos<=Jis)
            self.logger.info(f"Jis: {Jis}")
            self.logger.info(f"prvio: {PrViol}")
            self.logger.info(f"rel: {reliability}")
            #  PrViol is always a vector
            return x_dec,Jis,Joos,reliability,PrViol
        else:
            return None,-1,-1,0,1 # will never happen btw, solveOrResolve always retries or raises Exception
    
    def resolve(self,hassamplectrt:bool,trainSampleSet=None,Q=None,rwass=None,
                       lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,dontupdateTrainSample=False,**kwargs):
        """
        If given, actualiza trainsampleset (if samesize).
        If given, actualiza ADMM vars.
        Then, does M.optimize(), gets result
        
        """
        if not self.hasBeenSolved: # must never happen btw
            raise Exception("Instance must be solved once with solve() before resolve() can be called") 
        

        # si estoy seteando o no puedo sacarla de smplctrt
            # set xiscen
            # si estoy seteando seteala
        # si puedo sacarla y no estoy seteando
            # sacala

        # New average day from sample set
        if (trainSampleSet is not None and not dontupdateTrainSample) or not hassamplectrt:
            self.logger.info("updating nominal rnwgen and demand with trainSampleSet passed to resolve()")
            self.updateNominaltoSampleSetMean(trainSampleSet)

            # Reset Pmax constraint with new nominal gen, the one set by heuristic_reserve
            # Yes, sample set Pmax bounds for P_ED of Type2/MPPT gens, excess generated after PED is accounted for in Operation as eff Dem
            self.pmaxCtrt.RHS=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
            
            # Reset balance constraint RHS with new nominal demand curve
            self.loadbalanceCtrt.RHS=self.demand_curve_kw 

            XiScenarioSet=self.GetVarScenariosFromSampleSet(trainSampleSet)
            if hassamplectrt:
                self.samplectrt.RHS=XiScenarioSet
                self.M.update()
        else:
            XiScenarioSet=self.samplectrt.RHS
        
        # Update MAX MIN xi ctrt RHS
        if Q is not None:
            if not hasattr(self,'ximaxCtrt'):
                raise Exception("Cannot update Q.")
            self.logger.info(f"updating Q: {Q}")
            Ximax,Ximin=self.BoundsFromSet(XiScenarioSet,Q)                
            self.logger.info(f"Prev Ximax :{self.ximaxCtrt.RHS}")
            self.logger.info(f"Prev Ximin :{self.ximinCtrt.RHS}")
            self.ximaxCtrt.RHS=Ximax
            self.ximinCtrt.RHS=Ximin
            self.M.update()
            self.logger.info(f"updated Ximax to :{self.ximaxCtrt.RHS}")
            self.logger.info(f"updated Ximin to :{self.ximinCtrt.RHS}")
        
        if rwass is not None:
            if not hasattr(self,'rwass_ctrt'):
                raise Exception("Cannot update R_wass.")
            self.logger.info(f"updating rW: {rwass}")
            self.rwass_ctrt.RHS=rwass
        
        if self.transactive:
            assert lambdas_C is not None and lambdas_V is not None and z_PC is not None and z_PV is not None, "All z and lambdas must be set on every resolve"
            assert not self.neighbors==[], "Instance has no neighbors, cannot pass ADMM parameters"
            assert self.lenneighbors==len(lambdas_C)==len(lambdas_V)==len(z_PC)==len(z_PV),"ADMM parameter lists must have same length as number of neighbors"
            
            self.logger.info("updating z and lambda values (RHS) with ADMM parameters passed to resolve()")
            self.z_PC_k=z_PC
            self.z_PV_k=z_PV
            self.lam_C_k=lambdas_C
            self.lam_V_k=lambdas_V
            # Update ADMM parameters RHSs
            self.z_PC_ctrt.RHS=self.z_PC_k
            self.z_PV_ctrt.RHS=self.z_PV_k
            self.lam_C_ctrt.RHS=self.lam_C_k
            self.lam_V_ctrt.RHS=self.lam_V_k
    
        # Resolve Model
        self.logger.info("solving...")
        try:
            self.M.optimize()
        except gp.GurobiError as e:
            self.logger.error(f"Uhhh something happened: {e}")
            self.logger.error(f"{self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            self.M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
            return None,-1
        self.sol_time.append(self.M.Runtime)
        if self.M.status==GRB.OPTIMAL:
            self.logger.info(f"ED solved optimally in {self.M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.error(f"ED not solved optimally. Status: {self.M.status}")
            try:
                self.M.computeIIS()
                self.M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.error("Irreducible Inconsistent Subsystem written to .ilp")
                return None,-1
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
                return None,-1
        
        # Build Result Data
        if hasattr(self,'ReserveResult'): # Only det has it
            result=SimpleNamespace(**self.ReserveResult.__dict__)                 
        else:
            result=SimpleNamespace()
            result.fpart=self.fpart.X
            result.H_p=self.Rp.X
            result.H_n=self.Rn.X
            result.ResT_p=np.sum(self.Rp.X,axis=0)
            result.ResT_n=np.sum(self.Rn.X,axis=0)
            if hasattr(self,'kappa'):
                result.kappa=self.kappa.X
        
        result.ObjVal=self.M.ObjVal
        # GETTING THE DUALS MAY BE A BIT HARDER IN DROW
        if self.needMPO:
            try:
                result.MPO=self.loadbalanceCtrt.Pi
            except gp.GurobiError as e:
                self.logger.warning(f"Couldnt get LMP/MPO as Pi - {e}, trying from fixed model")
                fixedmodel=self.M.fixed()
                fixedmodel.optimize()
                fixedctrs_=[fixedmodel.getConstrByName(n) for n in self.loadbalanceCtrt.ConstrName]
                result.MPO=np.array(list(map(lambda c: c.Pi, fixedctrs_)))
                assert len(result.MPO)==self.T and (result.MPO>0).all(),f"Invalid LMP/MPOs: {result.MPO}"

        result.Pgen=self.pG.X
        # ADMM variables
        if self.transactive:
            result.P_C=self.P_C.X
            result.P_V=self.P_V.X
        if self.HAS_BESS:
            pCH_res=self.pCH.X
            pDC_res=self.pDC.X
            SOE_res=self.SOE.X
            result.PCH=pCH_res
            result.PDC=pDC_res
            result.PBESS=pDC_res-pCH_res
            result.SOE=SOE_res
            for t in range(self.T):
                if(pCH_res[t] > 1e-2 and pDC_res[t]>1e-2) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    self.logger.critical(f"t={t}")
                    self.logger.critical(f"pCH_res: {pCH_res[:t+1]}")
                    self.logger.critical(f"pDC_res: {pDC_res[:t+1]}")
                    self.logger.critical(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
        
        c_gen_copkwh_w_bess=self.c_gen_copkwh
        if self.HAS_BESS: c_gen_copkwh_w_bess=np.append(c_gen_copkwh_w_bess,self.c_chdc_copkwh)   
        
        ## CALCULATE REAL COST OF ED+R [D-1] PLANNING: ED, R, PURCHASES AND SALES
        Cost_of_EDnR=0
        for t in range(self.T):
            # P_ED
            Cost_of_EDnR+=np.sum(self.c_gen_copkwh*result.Pgen[:,t])
            # P^DC_ED - only pay BESS per kwh cycled aka discharged, "they" pay for the charge up
            if self.HAS_BESS:
                Cost_of_EDnR+=self.c_chdc_copkwh*pDC_res[t]
            # R+ and R-
            if not hasattr(self,'ReserveResult'): #non Det
                Cost_of_EDnR+=self.reservecost_wrt_gencost*np.sum(c_gen_copkwh_w_bess*(result.H_p[:,t]+result.H_n[:,t]))
            # buying and selling
            if self.transactive:
                for j in range(self.lenneighbors):
                    # if actually buying, add (LAM+lineCost) * P_C
                    if result.P_C[j,t]>1e-2:
                        Cost_of_EDnR+=(self.lam_C_k[j,t]+self.lineCosts)*result.P_C[j,t]
                    # if actually selling, sub LAM * P_V
                    if result.P_V[j,t]>1e-2:
                        Cost_of_EDnR-=self.lam_V_k[j,t]*result.P_V[j,t]        
        
        if hasattr(self,'ReserveResult'): #just for det
            result.EDcost=Cost_of_EDnR
            result.EDnRcost=result.EDcost+result.Rcost
            Jis=result.EDnRcost
        else:
            Jis=result.ObjVal
            result.EDnRcost=Cost_of_EDnR
    
        if plot_ED:
            self.plotED(result,**kwargs)            
        # return result,result.EDnRcost #x_dec,Jis
        return result,Jis

    # For plotting SOE logically
    def SOEvectoplot(self,vec):
        if self.strictlycircularbess:
            return np.append(np.roll(vec,1),vec[-1])
        else:
            return np.array([self.BESS_SOE_init*self.CE_bess_kwh]+list(vec))
    
    # def randomizeDemand(self,sigma_day=0.01,sigma_period=0.01,sigma_fast=0.02,window=12,**kwargs):
    # def randomizeDemand(self,sigma_day=0.08,sigma_period=0.08,sigma_fast=0.08,window=12,**kwargs):
    def randomizeDemand(self,sigma_day=0.12,sigma_period=0.12,sigma_fast=0.15,window=12,**kwargs):
        # ###### Remember, #P(-2*sigma<dev<+2*sigma)=95%, 3sigma~99.7%
        # ADDITIVE RANDOMNESS INSTEAD OF MULTIPLICATIVE; MAY DO LATER
        # CURRENT WORKS, BUT MAY SHIFT SAMPLE MEAN A BIT TOO MUCH BC GEOMETRIC RANDOMNESS
        # Daily variation
        realPeak=self.peak_demand_kw_smpgen*(dayDev:=self.rng.normal(1,sigma_day))
        # Period variation
        realDemCurve_kw=realPeak*self.demand_curve_pu_smpgen*self.rng.normal(1, sigma_period, self.T)
        # Fast signal with fillers
        fastDemCurve=np.concatenate([periodDem*self.rng.normal(1,sigma_fast,self.subperiods) for periodDem in realDemCurve_kw])
        startfiller=self.rng.normal(1,sigma_fast,window-window//2-1)*realDemCurve_kw[-1]
        endfiller=self.rng.normal(1,sigma_fast,window//2)*realDemCurve_kw[0]
        fastDemCurve=np.concatenate([startfiller,fastDemCurve,endfiller])
        # Low Pass Filter aka Moving Average
        LPFimpResp=np.ones(window)/window
        fastDemCurve_LPF=np.convolve(fastDemCurve,LPFimpResp,mode='valid') #LPF aka promedio movil
        return fastDemCurve_LPF,dayDev
    
    ## RENEWABLE POWER SAMPLE GENERATION
    def SPVPower(self,GTI_Wm2,PnomAtSTC_kw):  # Po[kW]=Pstc[kW]*GTI[W/m2]/1000Wm2
        if hasattr(GTI_Wm2,"__len__"):
            return [self.SPVPower(gti,PnomAtSTC_kw) for gti in GTI_Wm2]
        else:
            return GTI_Wm2*PnomAtSTC_kw/1000
    def SolarRand(self,PnomAtSTC_kw,window=7,subsubperiods=2,roll_hours=0.2):
        """Pout y Delta Pout con respecto al promedio.
        Con base en Puerto Carreño (Solcast).
        SoLO IMPLEMENTADO A T=24"""
        def GTI_Wm2_SampleCutAvg(pdf_obj,rng,params,Nsamples,min_,max_):
            sample=np.array(pdf_obj.rvs(*params,size=Nsamples,random_state=rng))
            sample=sample[(min_<=sample)&(sample<=max_)]
            cutavg=sum(sample)/len(sample) if len(sample)>0 else min_
            return cutavg
        fastGTICurve=np.zeros(self.T*self.subperiods)
        # Fill fastGTICurve
        for t in range(self.T):
            pdf_t=self.solarinfo['solar'][t]
            max_=pdf_t['max']
            min_=pdf_t['min']
            params=[float(i) for i in pdf_t['params']]
            pdf_obj=getattr(stats,pdf_t['pdfname'])
            sample_t=[GTI_Wm2_SampleCutAvg(pdf_obj,self.rng,params,subsubperiods,min_,max_) for _ in range(self.subperiods)]
            fastGTICurve[t*self.subperiods:(t+1)*self.subperiods]=sample_t
        #add fillers to fastCurve
        startfiller=np.zeros((window-window//2-1))
        endfiller=np.zeros(window//2)
        fastGTICurve=np.concatenate([startfiller,fastGTICurve,endfiller])
        # Low Pass Filter aka Moving Average
        LPFimpResp=np.ones(window)/window
        fastGTICurve_LPF=np.convolve(fastGTICurve,LPFimpResp,mode='valid')
        fastGTICurve_LPF=np.roll(fastGTICurve_LPF,int(roll_hours*self.subperiods))
        fastPoutCurve_LPF=np.array(self.SPVPower(fastGTICurve_LPF,PnomAtSTC_kw))
        return fastPoutCurve_LPF

    def WTPower(self,vw_meas:'m/s',r:'m'=9,startVw:'m/s'=3,cutVw:'m/s'=25,pomax_unit:'kW'=100,H_meas:'m'=50,H_WT:'m'=50,airdensity:'kg/m3'=1.22)->'kW':
        """Usando Turbinas convencionales de eje horizontal de 100kW pico."""
        if hasattr(vw_meas,"__len__"):
            wtpow_=[self.WTPower(r,vw,startVw,cutVw,pomax_unit,H_meas,H_WT) for vw in vw_meas]
            return wtpow_
        else:
            vw_WT=vw_meas if H_meas==H_WT else vw_meas*(H_WT/H_meas)**0.17
            WindPower=1/2*airdensity*(np.pi*r**2)*vw_WT**3
            BetzCoeff=16/27
            NonIdeal=0.85
            if vw_meas<=cutVw and vw_meas>=startVw:
                ##### c_p as function of v_w
                decay=2 # smaller means max eff range is wider (~TSR-control is better)
                vw_maxeff=8 # peak efficiency wind speed
                nonideal_func=NonIdeal*2.7*(vw_meas/vw_maxeff)**decay*np.exp(-(vw_meas/vw_maxeff)**decay)+0.03*vw_meas # to move up the end tail a bit
                po=nonideal_func*BetzCoeff*WindPower/1e3
                ##### c_p static (perfect TSR-control)
                # xx=NonIdeal*BetzCoeff*WindPower/1e3
                return min(po,pomax_unit)
            else:
                return 0
    def WindRand(self,N_100kw_units,window=9,subsubperiods=1):
        """Pout y DeltaPout con respecto al promedio.
        Con base en Sardinata (PDF de NASA Power
        renormalizado al promedio global de GlobalWindAtlas).
        SoLO IMPLEMENTADO A T=24"""    
        
        def windSampleCutAvg(pdf_obj,rng,params,Nsamples,min_,max_):
            sample=np.array(pdf_obj.rvs(*params,size=Nsamples,random_state=rng))
            sample=sample[(min_<=sample)&(sample<=max_)]
            cutavg=sum(sample)/len(sample) if len(sample)>0 else min_
            return cutavg
        fastWSCurve=np.zeros(self.T*self.subperiods)
        # self.logger.debug(f"size: {fastWSCurve.shape}")
        # Fill fastWSCurve
        for t in range(self.T):
            pdf_t=self.windinfo['wind'][t]
            max_=pdf_t['max']
            min_=pdf_t['min']
            params=[float(i) for i in pdf_t['params']]
            pdf_obj=getattr(stats,pdf_t['pdfname'])
            sample_t=[windSampleCutAvg(pdf_obj,self.rng,params,subsubperiods,min_,max_) for _ in range(self.subperiods)]
            fastWSCurve[t*self.subperiods:(t+1)*self.subperiods]=sample_t
        #add fillers to fastCurve
        startfiller=fastWSCurve[-(window-window//2-1):]
        endfiller=fastWSCurve[:window//2]
        fastWSCurve=np.concatenate([startfiller,fastWSCurve,endfiller])
        #LPF aka promedio movil
        LPFimpResp=np.ones(window)/window
        fastWSCurve_LPF=np.convolve(fastWSCurve,LPFimpResp,mode='valid')
        fastPoutCurve_LPF=np.array(self.WTPower(fastWSCurve_LPF))*N_100kw_units
        return fastPoutCurve_LPF

    def generateDaySample(self,**kwargs):
        """Generate one Sample containing random Demand and Random Type 2 (MPPT) Solar and Wind Generation, if available."""
        xi_i=SimpleNamespace()
        xi_i.subperiods=self.subperiods
        xi_i.fastDemandCurve,xi_i.dayDev=self.randomizeDemand(**kwargs)
        if self.solarMPPTavailable:
            xi_i.fastPSolar=self.SolarRand(PnomAtSTC_kw=self.solar_powernom_kw_smpgen,**kwargs)
            # self.logger.debug(f"size: {xi_i.fastPSolar.shape}")
        if self.windMPPTavailable:
            xi_i.fastPWind=self.WindRand(N_100kw_units=self.N100kWT_smpgen,**kwargs)
            # self.logger.debug(f"size: {xi_i.fastPWind.shape}")
        return xi_i
    
    def generateSampleSet(self,Nsamples,**kwargs):
        """Generate set of (day) Samples, either for training or testing."""
        return [self.generateDaySample(**kwargs) for _ in range(Nsamples)]
            
    ## RECOURSE ACTION
    # Microgrid Operation - Reserve Execution 
    def MGOperation(self,EDnRres,day_sample,/,Type2HasReconCost=True,asymm_Rec=True,asymm_uses_MPO_not_multiplier=False,RecMultiplier=1.2,plot_op=False,plotFastEffDemand=True,plotDeltaSOE=True,plot_DeltaPrnw=False,**kwargs):
        # clear_output()
        assert day_sample.subperiods==self.subperiods, "Sample subperiods do not match"
        assert hasattr(self,"LastInstance"), "Last Instance Gen name not specified"
        assert not self.LastInstance=='BESS', "BESS not allowed as Last Instance"
        assert not asymm_uses_MPO_not_multiplier or self.needMPO, "Instance must be init with needMPO to use it"
        # Get the Last Instance Gen
        for iLI,gLI in enumerate(self.MG.Gens):
            if gLI.type==self.LastInstance:
                assert gLI.dispatchable, "Last Instance Not Dispatchable"
                break
        else: #for-break-else is pretty cool
            raise Exception("Last Instance Not Found")
        
        # with timed("MGOperation --- creating fastEffDem",self.logger):

        # Get Randomized Demand from sample
        fastDemCurve=day_sample.fastDemandCurve
        if self.transactive:
            Pbought=EDnRres.P_C
            Psold=EDnRres.P_V
            P_incoming=np.zeros(self.T)
            for j in range(self.lenneighbors):
                P_incoming+=Pbought[j]-Psold[j]

            for t in range(self.subperiods):
                fastDemCurve[t] -= P_incoming[t//self.T]

            demandcurvekw=self.demand_curve_kw-P_incoming
            self.logger.debug(f"Pbought:{Pbought.astype(int)}")
            self.logger.debug(f"Psold:{Psold.astype(int)}")
            self.logger.debug(f"P_incoming:{P_incoming.astype(int)}")
        else:
            demandcurvekw=self.demand_curve_kw

        self.logger.debug(f"fastDemcurve:{fastDemCurve.astype(int)}")
        # Get Randomized (Max) Generation (for Intermittents T2) from sample
        fastDeltaGenCurve=np.zeros(self.T*self.subperiods)
        if self.solarMPPTavailable:
            fastPSolar=day_sample.fastPSolar
            SolarEDCurve_kw=EDnRres.Pgen[self.idx_solarMPPT,:]
            fastDeltaPSolar=np.array([fastpo-SolarEDCurve_kw[i//self.subperiods] for i,fastpo in enumerate(fastPSolar)])
            fastDeltaGenCurve+=fastDeltaPSolar
        if self.windMPPTavailable:
            fastPWind=day_sample.fastPWind
            WindEDCurve_kw=EDnRres.Pgen[self.idx_windMPPT,:]
            fastDeltaPWind=np.array([fastpo-WindEDCurve_kw[i//self.subperiods] for i,fastpo in enumerate(fastPWind)])
            fastDeltaGenCurve+=fastDeltaPWind
        # Effective Demand = realDemand - DeltaGeneration
        fastEffDem=deepcopy(fastDemCurve)
        if(self.solarMPPTavailable|self.windMPPTavailable):
            fastEffDem-=fastDeltaGenCurve

        # with timed("MGOperation --- Executing Recourse",self.logger):

        ### Execute ED+Recourse Actions
        Nsources=self.Ngen+self.HAS_BESS
        DeltaT=24/(self.T*self.subperiods) # en horas
        DeltaPG=np.zeros((Nsources,self.T))
        Nundermargin=Novermargin=0
        if self.HAS_BESS:
            Noverdischarge=Novercharge=0
            # calculate DeltaSOE
            DeltaSOE=np.zeros(self.T)
            SOE_real=np.zeros(self.T)
            # self.logger.debug(f"EDnRres.SOE: {vectoprint(EDnRres.SOE)}")
            
        if (not hasattr(EDnRres.ResT_p,"__len__")) or len(EDnRres.ResT_p)==1:
            EDnRres.fpart=np.vstack([EDnRres.fpart for _ in range(self.T)]).T
            EDnRres.H_p=np.vstack([EDnRres.H_p for _ in range(self.T)]).T
            EDnRres.H_n=np.vstack([EDnRres.H_n for _ in range(self.T)]).T
            EDnRres.ResT_p=[EDnRres.ResT_p for _ in range(self.T)]
            EDnRres.ResT_n=[EDnRres.ResT_n for _ in range(self.T)]
        # self.logger.debug(EDnRres.fpart)
        # self.logger.debug(EDnRres.H_n)
        # self.logger.debug(EDnRres.ResT_p)


        for tf,dem in enumerate(fastEffDem):
            tper,tmod=divmod(tf,self.subperiods)
            periodForecastedDem=demandcurvekw[tper]
            DeltaPL=dem-periodForecastedDem
            # self.logger.debug(f"tf:{tf}, DeltaPL:{DeltaPL}\n")
            # self.logger.debug(f"DeltaPG[:,tper]: {DeltaPG[:,tper]}\n")
            if DeltaPL==0: # No Recourse
                # DeltaPG[:,tper] += np.zeros(Nsources,1)
                continue
            elif DeltaPL>0: # Positive Recourse
                #overmargin above assigned reserve
                # self.logger.debug(f"DeltaPL-EDnRres.ResT_p[tper]: {DeltaPL-EDnRres.ResT_p[tper]}")
                if (overmargin:=DeltaPL-EDnRres.ResT_p[tper])>=0:
                    Novermargin+=1
                    # Recourse = (Holguras Totales + Overmargin del Last Instance) * DeltaT
                    DeltaPG[:,tper] += EDnRres.H_p[:,tper]*DeltaT 
                    DeltaPG[iLI,tper] += overmargin*DeltaT
                else:
                    # Recourse = fpart * DeltaPL * DeltaT
                    DeltaPG[:,tper] += EDnRres.fpart[:,tper]*DeltaPL*DeltaT 
            elif DeltaPL<0: # Negative Recourse
                if (undermargin:=-DeltaPL-EDnRres.ResT_n[tper])>=0:
                    Nundermargin+=1
                    # Recourse = (Holguras Totales + Overmargin del Last Instance) * DeltaT
                    DeltaPG[:,tper] -= EDnRres.H_n[:,tper]*DeltaT 
                    DeltaPG[iLI,tper] -= undermargin*DeltaT
                else:
                    # Recourse = fpart * DeltaPL * DeltaT
                    DeltaPG[:,tper] += EDnRres.fpart[:,tper]*DeltaPL*DeltaT
                # Recourse = (Holguras Totales + Overmargin) del Last Instance * DeltaT
            # self.logger.debug(f"DeltaPG[:,tper]: {DeltaPG[:,tper]}\n")
            
            # Al final del periodo (tfast+1)%subperiods=0
            if (tmod-(self.subperiods-1))==0:
                if self.HAS_BESS:
                # Actualizar SOE wrt Recourse actions
                # SOE[2] = SOE(t=2:59), al final del periodo
                # and if SOE nin [0,CE] anular over(dis)charge trasladando AGC a LI
                    # is battery net charging? (ED + recourse) 
                    charging_at_period = (EDnRres.PBESS[tper]+DeltaPG[-1,tper])<0
                    # then SOE at period increases with eta_ch
                    DeltaSOE[tper] = - DeltaPG[-1,tper] * (24/self.T) * (charging_at_period*self.eta_ch + (not charging_at_period)/self.eta_dc)
                        # can recourse actions make BESS ch and dc within same period? sure
                        # but meh it's rare enough and (eta_ch-eta_dc are similar enough)
                        # to ignore subperiod efficiency changes
                        
                    # then acumulate/cumsum DeltaSOE up to t:59
                    # on top of EDnRres.SOE to get SOE_real
                    SOE_real[tper] = EDnRres.SOE[tper] + np.sum(DeltaSOE)
                    # (it doesn't matter if circular or not, SOE(0:00) shouldn't change)
                    # self.logger.debug(f"EDnRres.SOE:  {vectoprint(EDnRres.SOE)}")
                    # self.logger.debug(f"DeltaSOE:     {vectoprint(DeltaSOE)}")            
                    # self.logger.debug(f"SOE_real t={tper}: {vectoprint(SOE_real)}\n")
                    if (overcharge_kwh:=SOE_real[tper] - self.CE_bess_kwh)>=0:
                        # if charging_at_period: self.logger.debug(f"not ch, weird t=[{tper}]")
                        # not really weird, been at max for a while
                        Novercharge+=1
                        SOE_real[tper] = self.CE_bess_kwh
                        DeltaSOE[tper] -= overcharge_kwh
                        overchargingpower_kw = overcharge_kwh  / ((24/self.T) * self.eta_ch)
                        DeltaPG[-1,tper] += overchargingpower_kw
                        DeltaPG[iLI,tper] -= overchargingpower_kw
                    elif (overdischarge_kwh:= 0 - SOE_real[tper])>=0:
                        # if not charging_at_period: self.logger.debug(f"not dc, weird t=[{tper}]")
                        # not really weird, been at 0 for a while
                        Noverdischarge+=1
                        SOE_real[tper] = 0
                        DeltaSOE[tper] += overdischarge_kwh
                        overdischargingpower_kw = overdischarge_kwh  / ((24/self.T) * self.eta_ch)
                        DeltaPG[-1,tper] -= overdischargingpower_kw
                        DeltaPG[iLI,tper] += overdischargingpower_kw
        
        self.logger.debug(f"DeltaPG: {DeltaPG.astype(int)}")
        # self.logger.debug(f"realEffDemand: {realEffDemand.astype(int)}")
        DeltaPG_for_costs=deepcopy(DeltaPG)

        # normally, reconcile w the MPPTs over DeltaPgen
        if Type2HasReconCost:
            if self.solarMPPTavailable:
                for t in range(self.T):
                    DeltaPG_for_costs[self.idx_solarMPPT,t]=np.sum(fastDeltaPSolar[t*self.subperiods:(t+1)*self.subperiods])
            if self.windMPPTavailable:
                for t in range(self.T):
                    DeltaPG_for_costs[self.idx_windMPPT,t]=np.sum(fastDeltaPWind[t*self.subperiods:(t+1)*self.subperiods])
        # OFC this will add to DeltaPG although it was already considered in effective demand on sample average calculations
        # hence why we touch it afterwards, only to obtain the correct c_gens_tiled
        self.logger.debug(f"DeltaPG_for_costs: {DeltaPG_for_costs.astype(int)}")

        disp_c_gens_copkwh=deepcopy(self.c_gen_copkwh)
        c_gens_tiled=np.tile(disp_c_gens_copkwh,(self.T,1)).T
        if asymm_Rec:
            # Not quite CREG 64/2000 bc no "ED ideal" but asymmetric rate is important
            # self.logger.info(f"DeltaPG: {DeltaPG.astype(int)}")
            # watch this logical array selection magic
            isPgeq0=DeltaPG_for_costs[:self.Ngen,:]>=0
            isPlessthan=~isPgeq0
            # self.logger.debug(f"c_gens_tiled:{c_gens_tiled}\n")
            # self.logger.debug(f"isPgeq0:{isPgeq0}\n")
            # self.logger.debug(f"isPlessthan:{isPlessthan}\n")
            # EXCEPT THE BESS, IT SHOULD BE PAID THE SAME OFC        
            # self.logger.debug(f"disp_c_gens_copkwh:{disp_c_gens_copkwh} ==== EDnRres.MPO;{EDnRres.MPO}")
            # self.logger.debug(f"disp_c_gens_copkwh:{disp_c_gens_copkwh.shape} ==== EDnRres.MPO;{EDnRres.MPO.shape}")
            # self.logger.debug(f"c_gens_tiled:{c_gens_tiled}")
            # self.logger.debug(f"c_gens_tiled:{c_gens_tiled.shape}")
            # self.logger.debug(f"mpo_tiled:{mpo_tiled}")
            # self.logger.debug(f"mpo_tiled:{mpo_tiled.shape}")
            # self.logger.debug(f"_cost_agc_auxmtx:{_cost_agc_auxmtx}")
            # self.logger.debug(f"_cost_agc_auxmtx:{_cost_agc_auxmtx.shape}")
            # self.debug("cost_agc_copkwh_max",cost_agc_copkwh_max)
            # self.debug("cost_agc_copkwh_min",cost_agc_copkwh_min)

                # asymmRec=True
                # RecMultiplier=1


            if asymm_uses_MPO_not_multiplier:
                ### Cost for DeltaPgen>0 is max(c_i,LMP), for DeltaPgen<0 is min(c_i,LMP) [asymmetric]. 
                mpo_tiled=np.tile(EDnRres.MPO,(len(disp_c_gens_copkwh),1))
                _cost_agc_auxmtx=np.stack([c_gens_tiled,mpo_tiled])
                cost_agc_copkwh_max=np.max(_cost_agc_auxmtx,axis=0)
                cost_agc_copkwh_min=np.min(_cost_agc_auxmtx,axis=0)
            
            else:
                cost_agc_copkwh_max=c_gens_tiled*RecMultiplier
                cost_agc_copkwh_min=c_gens_tiled/RecMultiplier

            # huzzah correct asymmetric agc prices according to logical indexing
            c_gens_tiled=isPgeq0*cost_agc_copkwh_max + isPlessthan*cost_agc_copkwh_min
        
        # self.debug("c_gens_tiled",c_gens_tiled)
        
        # get idxdispatchable just in case
        idxdispatchable=[i for i,g in enumerate(self.MG.Gens) if g.dispatchable]
        # i only plot agc service of non-T2's even if they get paid for it
        disp_gennames=deepcopy(self.gennames)
        if self.HAS_BESS:
            idxdispatchable.append(-1) # do get last one
            disp_gennames.append('BESS')
            # concat Bess agc cost to c tiled
            c_gens_tiled=np.concatenate([c_gens_tiled,np.tile(self.c_chdc_copkwh,(1,24))],axis=0)
        self.debug("idxdispatchable",idxdispatchable)
        self.debug("disp_gennames",disp_gennames)
        self.debug("c_gens_tiled",c_gens_tiled)


        # get op cost mtx, includes MPPT
        op_cost=c_gens_tiled*DeltaPG_for_costs
        self.logger.debug(f"op_cost: {op_cost.astype(int)}")
        # self.logger.info(f"DeltaPG: {DeltaPG.astype(int)}")

        # Calculate period real (effective) demand before touching DeltaPG
        realEffDemand=demandcurvekw+np.sum(DeltaPG,axis=0)
        self.logger.debug(f"realEffDemand: {realEffDemand.astype(int)}")
        
        # normally they get paid for agc as rec, as incentive to operator to predict better
        if Type2HasReconCost:
            # calculate sum
            total_op_cost=np.sum(op_cost)
            if not plot_DeltaPrnw:
                op_cost=op_cost[idxdispatchable,:]
                disp_gennames=[n for i,n in enumerate(self.gennames) if i in idxdispatchable]+self.HAS_BESS*['BESS']
        else:
            # se recortan sus idx de la matriz de costos
            op_cost=op_cost[idxdispatchable,:]
            # y del display
            disp_gennames=[n for i,n in enumerate(self.gennames) if i in idxdispatchable]+self.HAS_BESS*['BESS']
            # then calculate total op cost
            total_op_cost=np.sum(op_cost)

        # Plot
        if plot_op:
            periods=np.arange(0,24+1/self.T,24/self.T)
            fig,ax0=plt.subplots(figsize=(14,8))
            # Expected Demand (hourly)
            ax0.step(periods,np.append(demandcurvekw,demandcurvekw[-1]),where='post',lw=2,label='Dem. esperada [D-1]')
            # Effective Demand served with ED+R (hourly)
            ax0.step(periods,np.append(realEffDemand,realEffDemand[-1]),where='post',lw=2,label='Dem. atendida (ED+R) [D]')
            # Real Demand (fast)
            subpidx=np.arange(0,24*self.subperiods,24/self.T)/self.subperiods
            ax0.plot(subpidx,fastDemCurve,label="Dem. real (D)",alpha=0.4)
            # Real Effective Demand (fast)
            if plotFastEffDemand:
                ax0.plot(subpidx,fastEffDem,label=r"Dem. efectiva (D)",alpha=0.4)
            # Type 2 Gen
            if self.solarMPPTavailable:
                ax0.plot(subpidx,fastPSolar,label=r"$P_{spv}$",alpha=0.4)
            if self.windMPPTavailable:
                ax0.plot(subpidx,fastPWind,label=r"$P_{wp}$",alpha=0.4)
            ax0.grid(True, linestyle='--', alpha=0.3)
            ax0.set_xticks(periods)
            ax0.set_xlim([0,24])
            ax0.set_ylabel('Potencia (kW)')
            ax0.set_xlabel('Tiempo (h)')
            ylim=ax0.get_ylim()
            ax0.set_ylim((ylim[0]*0.4,ylim[1]*1))
            # AGC Cost
            ax1=ax0.twinx()
            ax1.stackplot(periods,np.concatenate((op_cost,np.array([op_cost[:,-1]]).T),axis=1),labels=disp_gennames,step='post',alpha=0.8,colors=self.plotcolors)
            ax1.set_ylabel('Costo por Desviaciones (COP)')
            ax1.yaxis.set_major_formatter('${x:,.0f}')
            # ylim=EDnRres.EDnRcost/self.T
            # ylim_vis=2.5
            # pos=0.2
            # ax1.set_ylim([-pos*2*ylim/ylim_vis,(1-pos)*2*ylim/ylim_vis])
            ylim=ax1.get_ylim()
            ax1.set_ylim((ylim[0] , ylim[1] + (ylim[1]-ylim[0])*1.5))
            h0,l0=ax0.get_legend_handles_labels()
            h1,l1=ax1.get_legend_handles_labels()
            leg0,leg1=(0.07,0.97),(0.91,0.97)
            #Plot DeltaSOE
            if plotDeltaSOE and self.HAS_BESS:
                ax2=ax0.twinx()
                ax2.spines['left'].set_position(('axes',-0.1))
                ax2.yaxis.set_label_position('left')
                ax2.yaxis.set_ticks_position('left')
                ax2.plot(periods,self.SOEvectoplot(EDnRres.SOE),lw=1,label='SOE esperado [D-1]',alpha=0.8)
                ax2.plot(periods,np.append(EDnRres.SOE[0],SOE_real),lw=1,label='SOE real [D]',alpha=0.8)
                ax1.text(0.81,0.1,f"RestricSOE={Noverdischarge+Novercharge}/24",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none'))
                ax2.set_ylabel("SOE (kWh)")
                # ylim=ax2.get_ylim()
                # ax2.set_ylim((ylim[0]-(ylim[1]-ylim[0])*0.1,(ylim[1]-ylim[0])*1.6))
                ax2.set_ylim((0,self.CE_bess_kwh*1.5))
                SOE_tick_labels=ax2.get_yticklabels()
                SOE_tick_labels = [f"{lab.get_text()} - {float(lab.get_text().replace('−','-'))/self.CE_bess_kwh:.0%}" for lab in SOE_tick_labels]
                ax2.set_yticklabels(SOE_tick_labels)
                h_,l_=ax2.get_legend_handles_labels()
                h0,l0=h_+h0,l_+l0
                leg0=(0.18,0.97)
            fig.legend(h0,l0,loc='upper left', bbox_to_anchor=leg0, frameon=True)
            fig.legend(h1,l1,loc='upper right', bbox_to_anchor=leg1, frameon=True)
            # ax1.text(0.01,0.02,f"Desv. del pico diario={dayDev-1:.2%}",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none'))
            ax1.text(0.01,0.06,f"Rec. RSF=${total_op_cost:,.0f} COP",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none'))
            ax1.text(0.81,0.06,f"Fuera de margen={Nundermargin+Novermargin}/{self.subperiods*24}",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none'))
            if self.transactive:
                ax1.text(0.01,0.02,"[\"Dem\"=Dem+PVen-PCompr]",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none'))     
            # ax1.text(0.81,0.02,f"Sobremarg={Novermargin}/{self.subperiods*24}",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none'))
            # ax1.text(0.81,0.06,f"Submarg={Nundermargin}/{self.subperiods*24}",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.7, edgecolor='none'))
            fig.set_dpi(500)
            plt.tight_layout()
            plt.show()

        operationResult=SimpleNamespace()
        operationResult.RecCost=total_op_cost
        operationResult.Nover=Novermargin
        operationResult.Nunder=Nundermargin
        if self.HAS_BESS: 
            operationResult.Noverdc=Noverdischarge
            operationResult.Noverch=Novercharge
        return operationResult

    def Joos_i(self, x_dec, testSample,**kwargs):
        """Calls MGOperation() with one (day) test sample to get (instance) out of sample cost.
        
        **MGOp kwargs**:
            Type2HasReconCost=True,plot_op=False,plotFastEffDemand=True,plotDeltaSOE=True
        """
        attr={"EDnRcost","fpart","ResT_p","ResT_n"}
        for a in attr: 
            if not hasattr(x_dec,a):
                raise Exception("ED+R decision misspecified")
        if not hasattr(testSample,"fastDemandCurve"):
            raise Exception("test Sample misspecified")
        op=self.MGOperation(x_dec,testSample,**kwargs)
        Joos_i=op.RecCost+x_dec.EDnRcost
        ProbResViol=(op.Nover+op.Nunder)/(self.subperiods*self.T)
        if self.HAS_BESS:
            ProbBCapViol=(op.Noverdc+op.Noverch)/self.T
            return Joos_i,ProbResViol,ProbBCapViol
        return Joos_i,ProbResViol
            
    
    def GetExpectedFromSampleSet(self,DaySamplesSet):
        """Returns expected Demand and Type 2 (MPPT) generation (if available) from Sample.
        For use in ED (24 periods)."""
        Nsamples=len(DaySamplesSet)
        avgDaySample={}
        # Demand
        avgDemCurve=np.zeros(self.T)
        for samp in DaySamplesSet:
            avgDemCurve+=[sum(samp.fastDemandCurve[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
        avgDemCurve/=Nsamples
        avgDaySample['demand_curve_kw']=np.round(avgDemCurve,0)
        avgDaySample['demand_curve_pu']=avgDaySample['demand_curve_kw']/self.peak_demand_kw

        # Solar Type2
        if self.solarMPPTavailable:
            avgPsolarCurve=np.zeros(self.T)
            for samp in DaySamplesSet:
                avgPsolarCurve+=[sum(samp.fastPSolar[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
            avgPsolarCurve/=Nsamples
            avgDaySample['Psolar_kw']=np.round(avgPsolarCurve,0)
        # Wind Type2
        if self.windMPPTavailable:
            avgPwCurve=np.zeros(self.T)
            for samp in DaySamplesSet:
                avgPwCurve+=[sum(samp.fastPWind[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
            avgPwCurve/=Nsamples
            avgDaySample['Pwind_kw']=np.round(avgPwCurve,0)
        return avgDaySample # E[xi]
    
    def updateNominaltoSampleSetMean(self,DaySamplesSet):
        """
        Update nominal demand {peak_demand_kw,demand_curve_kw,demand_curve_pu} and Type 2 (MPPT) generation {g.gen_curve_pu} (if available) with expected from SampleSet.
        """
        nominalScenario={'peak_demand_kw':self.peak_demand_kw, 'demand_curve_kw':self.demand_curve_kw,'demand_curve_pu':self.demand_curve_pu}
        if self.solarMPPTavailable: 
            g=self.MG.Gens[self.idx_solarMPPT]
            nominalScenario|={'Psolar_kw':g.gen_curve_pu*g.power_kw}
        if self.windMPPTavailable:
            g=self.MG.Gens[self.idx_windMPPT]
            nominalScenario|={'Pwind_kw':g.gen_curve_pu*g.power_kw}
        self.logger.debug(f"previous nominal Day: {nominalScenario}")
            
        meanScenario=self.GetExpectedFromSampleSet(DaySamplesSet)
        
        # update nominal demand 
        # I SHOULDNT UPDATE PEAK ONLY THE CURVE
        # self.peak_demand_kw=max(meanScenario.demand_curve_kw)
        # need the deepcopy, to ensure deletion of meanScenario (i think)
        self.demand_curve_kw=deepcopy(meanScenario['demand_curve_kw']) 
        self.demand_curve_pu=self.demand_curve_kw/self.peak_demand_kw

        self.logger.debug(f"expected, new nominal Day: {meanScenario}")

        ## update solar nominal gen curve
        if self.solarMPPTavailable: 
            g=self.MG.Gens[self.idx_solarMPPT]
            g.gen_curve_pu=meanScenario['Psolar_kw']/g.power_kw
        ## update wind nominal gen curve
        if self.windMPPTavailable:
            g=self.MG.Gens[self.idx_windMPPT]
            g.gen_curve_pu=meanScenario['Pwind_kw']/g.power_kw
        del nominalScenario,meanScenario

        # Get Xihat scenarios from sample set, considering E[Xi]=0
    def GetVarScenariosFromSampleSet(self,DaySamplesSet):
        """Returns days/scenarios sample set as sample set of effective variable generation aka negative effective load (Xi),
        with respect to nominal currently in instance."""
        Nsamples=len(DaySamplesSet)
        self.subperiods=DaySamplesSet[0].subperiods
        ScenarioSet=np.zeros((Nsamples,self.T))
        for n,samp in enumerate(DaySamplesSet):
            ScenarioSet[n,:] = self.demand_curve_kw - [sum(samp.fastDemandCurve[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
            if self.solarMPPTavailable:
                g=self.MG.Gens[self.idx_solarMPPT]
                ScenarioSet[n,:] += [sum(samp.fastPSolar[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
                ScenarioSet[n,:] -= g.gen_curve_pu*g.power_kw
            if self.windMPPTavailable:
                g=self.MG.Gens[self.idx_windMPPT]
                ScenarioSet[n,:] += [sum(samp.fastPWind[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
                ScenarioSet[n,:] -= g.gen_curve_pu*g.power_kw
        # self.logger.debug(f"ScenarioSet[3]: {ScenarioSet[3,:]}")
        return ScenarioSet
        
    def BoundsFromSet(self,XiScenarioSet,Q):
        """"returns Ximax,Ximin. Q>0. Can be >100"""
        # self.logger.debug(f"======\nXiScenarioSet : {XiScenarioSet}\n======\n")
        if Q<=0:
            raise Exception("Q>0.")
        if Q<100:
            Ximax=np.percentile(XiScenarioSet,50+float(Q)/2,axis=0) # mu + Q/2
            Ximin=np.percentile(XiScenarioSet,50-float(Q)/2,axis=0) # mu - Q/2
        else: 
            Ximax=np.max(XiScenarioSet,axis=0)
            Ximin=np.min(XiScenarioSet,axis=0)
            self.logger.debug(f"Actual max: {Ximax}")
            self.logger.debug(f"Actual min: {Ximin}")
            P50support=np.percentile(XiScenarioSet,75,axis=0) - np.percentile(XiScenarioSet,25,axis=0)
            Ximax += P50support*(float(Q)-100)/100 # max + (P50support)*overQ/100
            Ximin -= P50support*(float(Q)-100)/100 # min - (P50support)*overQ/100
        self.logger.info(f"Ximax finally: {Ximax}")
        self.logger.info(f"Ximin finally: {Ximin}")
        return Ximax,Ximin
 
    def heuristic_reserve(self,customReserve=None,customReg=None,peakSupportedDisconn=0.4,
                          f_hi=62,f_low=58.5,fpart_bess=None,Type3GenFactor=0.5,
                        Res_verbose=False,T_RB_dc_h=2,T_RB_ch_h=2,**kwargs):
        """
            :param customReserve:
              ({'up'=float,'down':float}, optional) Defaults to None. Custom Total MG Reserve.
            :param customReg:
              (list, optional) Defaults to None. List of custom Regulation constants [Hz/kW] list of Ngens.
            :param peakSupportedDisconn:
              (float, optional) Defaults to 0.4. % of Peak Demand disconnected supported to calculate ResT+.
            :param f_hi:
              (float, optional) Defaults to 62.
            :param f_low:
              (float, optional) Defaults to 58.5.
            :param fpart_bess:
              (float, optional) Defaults to None. (Initial) Participation Factor used for BESS. If None, rule of thumb proportional to CapB/PeakDem cut btwn [0.1,0.5].
            :param Type3GenFactor:
              (float, optional) Defaults to 0.5. Intermittent Dispatchable gens get assigned [Type3GenFactor] times less Reserve than Type0/1 (rescaled).
            :param Res_verbose:
              (bool, optional) Defaults to False. Log results. Logger level should be <=WARNING
            :param T_RB_dc_h:
              (int, optional) Defaults to 2. Time to calculate SOE min.
            :param T_RB_ch_h:
              (int, optional) Defaults to 2. Time to calculate SOE max.
        
        Returns    
            Reserve decision object (SimpleNamespace) with fpart,ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,Rcost
        """        
        ### 0. CRITERIO N-1 PARA RESERVA TOTAL 
        if customReserve is None:
            # Carga mas grande desconectable, asumiendo un % de peak demand
            ResdownTotal=peakSupportedDisconn*self.peak_demand_kw
            # Capacidad mas grande de unidad de generacion (N-1)
            ResupTotal=max([max(g.power_perunit_kw) for g in self.MG.Gens]+[peakSupportedDisconn*self.peak_demand_kw])
        elif isinstance(customReserve,dict):
            # A menos que se estipule lo contrario
            ResupTotal=customReserve['up']
            ResdownTotal=customReserve['down']
        self.MaxResT_p=ResupTotal
        self.MaxResT_n=ResdownTotal
        # 1. Definir estatismo total del sistema
        R_MGup=(f_hi-60)/ResdownTotal #Hz/kW
        R_MGlo=(60-f_low)/ResupTotal #Hz/kW
        # 2. Elegir la constante de regulacion menor (mas robusto)
        R_MG_max=min(R_MGup,R_MGlo) #Hz/kW'
        if Res_verbose:
            if (self.logger.level>logging.WARNING):
                self.logger.critical("logging level too high, Reserve info not shown")
            self.logger.warning(f"Total Rsrv: +{ResupTotal/1000:.2f}MW, -{ResdownTotal/1000:.2f}MW")
            self.logger.warning(f"Freq Limits: {f_low-60}Hz, {f_hi-60}Hz, ")
            self.logger.warning(f"R_MG: {R_MG_max*1000:.2f}Hz/MW, {R_MG_max*self.peak_demand_kw/60:.3%} puHz/puMW")
        # 3. Definir factores de participacion, constantes de regulacion y holguras
        fpart=[0]*(self.Ngen+self.HAS_BESS) # % 
        R=[0]*(self.Ngen+self.HAS_BESS) # Hz/kW

        if self.HAS_BESS and fpart_bess is None:
            # idk a very ad hoc rule of thumb
            fpart_bess=np.round(min(max((self.CE_bess_kwh/(self.peak_demand_kw*3.5))+0.1,0.1),0.5),2)
            self.logger.warning(f"fpart_bess not given; set to: {fpart_bess:.1%}")
        while True: 
            retry=False
            # 3.1. Definir fpart y ctes de Reg
            # 3.1.a. Usar constantes Reg custom, si hay
            if customReg is not None:
                if not len(customReg)==self.Ngen+self.HAS_BESS:
                    raise Exception("wrong length for customReg")
                else:
                    R=customReg
                    if min(R[np.nonzero(R)])<=R_MG_max:
                        raise Exception("R_MG larger than customReg elements! Change MG reserve margins")
                    fpart=[R_MG_max/Ri for Ri in R]
                    # Funcionalidad futura: Permitir algunos Ri custom y setear demas en consecuencia.
                    # Por ahora, relying on proper Rcustom definition
            else:
                # 3.1.b.0. BESS (si hay) contribuye (f_part_bess)% del trabajo
                if self.HAS_BESS:
                    fpart[-1]=fpart_bess #%
                    R[-1]=R_MG_max/fpart_bess #Hz/kW
                    R_MG_remaining=R_MG_max/(1-fpart_bess) #Hz/kW
                else:
                    R_MG_remaining=R_MG_max #Hz/kW
                
                # 3.1.b.1. Calc P despachable y definir cte regulacion igual en todos (gens no BESS)
                CapTotDispatch=sum(g.power_kw for g in self.MG.Gens if g.dispatchable)
                r=R_MG_remaining*CapTotDispatch/60 # igual para todos, en puHz/pukW
                if Res_verbose:
                    self.logger.warning(f"Equal droops r: {r:.2%} puHz/puMW")
                # 3.1.b.2. Si hay intermitentes no despachables (Tipo 2 o MPPT) asignarles r->+inf puHz/pukW
                rTipo3=1000 # aka no participan en regulacion
                R[:self.Ngen]=[r*60/g.power_kw if g.dispatchable else rTipo3*60/g.power_kw for g in self.MG.Gens] # Hz/kW
                # 3.1.b.3. Calcular factores de participacion (%) en RSF
                fpart[:self.Ngen]=[R_MG_max/Ri for Ri in R[:self.Ngen]] # %
                if sum(g.intermittent and g.dispatchable for g in self.MG.Gens)>0:
                # 3.1.b.4. Si hay despachables intermitentes (Tipo 3) ...
                    if Res_verbose:
                        self.logger.warning(f"Rescaling by {Type3GenFactor:.2f} for Type 3")
                    # ... se reducen su fpart_i  por (Type3GenFactor, e.g. 0.5) w.r.t. Tipo 1...
                    fpart[:self.Ngen]=[Type3GenFactor*f if self.MG.Gens[i].intermittent else f for i,f in enumerate(fpart[:self.Ngen])]
                    s=sum(f for f in fpart[:self.Ngen])# suma (de los non-bess)
                    # ... y se renormaliza para que los fpart sumen 1
                    fpart[:self.Ngen]=[f*(1-fpart_bess*self.HAS_BESS)/s for f in fpart[:self.Ngen]]
                    R[:self.Ngen]=[R_MG_max/f for f in fpart[:self.Ngen]]

            # 3.2. Definir Holguras/Margenes de Reserva rounded a 1kW (para eliminar Holgura Tipo 2)
            H_n=np.round([ResdownTotal*f for f in fpart],0)
            H_p=np.round([ResupTotal*f for f in fpart],0)
            for i,g in enumerate(self.MG.Gens):
                Gcap = sum(g.power_perunit_kw)
                assert Gcap > H_p[i], f"Unfeasible R+: {H_p[i]} for {g.type}, Pmax: {Gcap}"
                assert Gcap > H_n[i], f"Unfeasible R-: {H_n[i]} for {g.type}, Pmax: {Gcap}"
            # Y recalcular ResupTotal y ResdownTotal en consecuencia
            ResupTotal=np.sum(H_p)
            ResdownTotal=np.sum(H_n)
            if self.HAS_BESS:
                # 3.3. Se calculan las cotas de carga y descarga de BESS
                p_dc_max_bess_kw=max(self.p_dc_max_bess_kw-H_p[-1],0)
                p_ch_max_bess_kw=max(self.p_ch_max_bess_kw-H_n[-1],0)
                if (0 in {p_dc_max_bess_kw , p_ch_max_bess_kw}):
                    self.logger.critical(f"Batt cannot charge and/or discharge!: Pdc <= {p_dc_max_bess_kw:.0f}kW, Pch <= {p_ch_max_bess_kw:.0f}kW.")
                    retry=True

                # 3.4. Se calculan Cotas de SOE
                # Debe poder atender H+ por (T_RB_dc_h) horas
                minSOE_perc=min(T_RB_dc_h*H_p[-1]/self.CE_bess_kwh,1)
                # Debe poder absorber H- por (T_RB_ch_h) horas
                maxSOE_perc=max(1-T_RB_ch_h*H_n[-1]/self.CE_bess_kwh,0)
                if(minSOE_perc>=maxSOE_perc):
                    self.logger.critical(f"Batt SOE bounds infeasible: {minSOE_perc:.1%} <= SOE <= {maxSOE_perc:.1%}.")
                    retry=True

                if(not self.strictlycircularbess and maxSOE_perc<=self.BESS_SOE_init):
                    self.logger.critical(f"Max SOE {maxSOE_perc:.1%} is lower than SOE init {self.BESS_SOE_init:.1%}.")
                    retry=True
                
            # 3.5. Si asignacion de BESS es infactible, intentar de nuevo...
            if retry:
                if customReg is not None:
                    raise Exception("Custom Regulation forced unfeasible BESS bounds")
                fpart_bess=fpart_bess*0.8 #...con menor participation del BESS
                self.logger.critical(f"Retrying with lower fpart_bess={fpart_bess:.1%}")
                continue
            # makeshift do-while
            else:
                self.customReserve={"up":ResupTotal,"down":ResdownTotal}
                self.fpart_bess = fpart_bess
                self.T_RB_ch_h = T_RB_ch_h
                self.T_RB_dc_h = T_RB_dc_h
                break

        # 4. Se recalculan las constantes de regulacion finales
        R_pu=[Ri*self.MG.Gens[i].power_kw/60 for i,Ri in enumerate(R[:self.Ngen])]
        if self.HAS_BESS:
            R_pu += [R[-1]*self.MG.BESS.power_kw/60]
        R_HzMw=[Ri*1000 for Ri in R]      
        
        # 5. Usar holguras para definir cotas de generacion para ED
        p_gmax_kw_mtx=np.vstack([np.array(g.gen_curve_pu)*g.power_kw-H_p[i] for i,g in enumerate(self.MG.Gens)])
        p_gmin_kw_mtx=np.vstack([[H_n[i]]*self.T for i,g in enumerate(self.MG.Gens)])
        paramsForED={'p_gmax_kw_mtx':p_gmax_kw_mtx,
                'p_gmin_kw_mtx':p_gmin_kw_mtx}
        if self.HAS_BESS:
            paramsForED=paramsForED|{'p_dc_max_bess_kw':p_dc_max_bess_kw,
                    'p_ch_max_bess_kw':p_ch_max_bess_kw,
                    'minSOE_perc':minSOE_perc,
                    'maxSOE_perc':maxSOE_perc}
            if Res_verbose: self.logger.warning(f"{minSOE_perc:.2%} <= SOE <= {maxSOE_perc:.2%}")
                    
        if Res_verbose:
            self.logger.warning(f"For {[g.type for g in self.MG.Gens]+self.HAS_BESS*["BESS"]}")
            self.logger.warning(f"fpart: [{", ".join([f"{f:.1%}" for f in fpart])}]")
            self.logger.warning(f"R: [{", ".join(f"{r:.1%}" for r in R_pu)}]% puHz/puMW")
            self.logger.warning(f"R: [{", ".join(f"{r:.1f}" for r in R_HzMw)}] Hz/MW")
            self.logger.warning(f"H+: {H_p}kW")
            self.logger.warning(f"H-: {H_n}kW")
            self.logger.warning(f"ResTotal: +{ResupTotal}kW,-{ResdownTotal}kW")
            if Res_verbose>1:
                self.logger.warning(f"params passed to BaseED: {paramsForED}")
            

        ##### fpart,ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n        
        ReserveResult=SimpleNamespace()
        ReserveResult.fpart=np.array(fpart)
        ReserveResult.ResT_p=ResupTotal
        ReserveResult.ResT_n=ResdownTotal
        ReserveResult.R_HzMw=np.array(R_HzMw)
        ReserveResult.R_pu=np.array(R_pu)
        ReserveResult.H_p=H_p
        ReserveResult.H_n=H_n
        return paramsForED,ReserveResult
 
    def plotED(self,EDResult,plot_R=False,plotcolors=None,EDplotstyle='stack',stackalpha=0.7,**kwargs):
        # self.logger.debug(f"Result passed:{EDResult}")
        # Graficar demanda y generacion
        if plotcolors is not None:
            assert plotcolors in mpl.colormaps(), f"plotcolors must be a matplotlib.colormaps. Available: {mpl.colormaps()}"
            self.plotcolors=mpl.colormaps[plotcolors].colors
        else:    
            self.plotcolors=mpl.colormaps['Set1'].colors
        genlabels=self.gennames 
        T=self.T
        if self.HAS_BESS:
            genlabels+=['BESS']
            cP_mtx=np.transpose(np.tile(np.concatenate((self.c_gen_copkwh,[self.c_chdc_copkwh])),(T,1)))
            P_sys=np.concatenate((EDResult.Pgen,[EDResult.PBESS]),axis=0)
            P_cost=np.concatenate((EDResult.Pgen,[EDResult.PDC]),axis=0)
        else:
            cP_mtx=np.transpose(np.tile(self.c_gen_copkwh,(T,1)))
            P_sys=EDResult.Pgen
            P_cost=EDResult.Pgen
            
        cP_res=P_cost*cP_mtx # costo de despacho de cada generador/bess por periodo, Ngen+hasbess x T
        cop_hr=np.sum(cP_res,axis=0) # costo de despacho por periodo, 1 x T
        cop_kwh=cop_hr/(self.demand_curve_kw) # costo unitario de despacho operacion por periodo, 1 x T 
        cop_kwh_avg=np.sum(cop_hr)/np.sum(self.demand_curve_kw) # costo unitario promedio diario
        

        periods = np.arange(0, 24+1/T, 24/T)
        fig, ax0 = plt.subplots(figsize=(14,8))
        ax0.step(periods,np.append(self.demand_curve_kw,self.demand_curve_kw[-1]),color="teal",linestyle='--',label='Demanda',where='post')
        ax0.set_xticks(periods)
        ax0.set_xlim((0,24))
        match EDplotstyle:
            case 'line'|'step'|'plot':
                for i,p in enumerate(P_sys):
                    ax0.step(periods, np.append(p,p[0]), where='post', label=genlabels[i], alpha=0.8,color=self.plotcolors[i]) #Clearer
            case 'bars'|'bar'|'barplot':
                n_sources = self.Ngen+self.HAS_BESS
                width=0.8/n_sources
                for i in range(n_sources):
                    ax0.bar(periods[:-1] + (i+1)/(n_sources+1), P_sys[i], width, label=genlabels[i], color=self.plotcolors[i % len(self.plotcolors)])
                ax0.tick_params(axis='x',bottom=True,top=False,labelbottom=True,labeltop=False,direction='out')        
                axx=ax0.twiny()
                axx.spines['bottom'].set_position('zero')
                axx.set_xlim(ax0.get_xlim())
                axx.set_xticks(ax0.get_xticks())
                axx.tick_params(axis='x',bottom=True,top=False,labelbottom=False,labeltop=False,direction='out')
            case 'stack'|'stackplot'|'stacked':
                #Prettier
                ax0.stackplot(periods, np.concatenate((P_sys,np.array([P_sys[:,-1]]).T),axis=1), labels=genlabels, alpha=stackalpha,step='post',colors=self.plotcolors)
            case _:
                raise Exception("Undefined ED plot style")
        ax0.set_xlabel('Tiempo (h)')
        ax0.set_ylabel('Potencia (kW)')
        h, l = ax0.get_legend_handles_labels()
        bbox1=(0.07,0.96)
        if self.HAS_BESS:
            # Segundo eje izquierda para el SOE
            ax1 = ax0.twinx()
            ax1.spines['left'].set_position(('axes', -0.12))  # Mover a la izquierda de ax0
            ax1.yaxis.set_label_position('left')
            ax1.yaxis.set_ticks_position('left')
            # SOE[2] es SOE(t=2:59), al final del periodo
            # se hace roll para mostrar el SOE acorde a la hora del punto
            ax1.plot(periods, self.SOEvectoplot(EDResult.SOE), color='gold', linestyle='solid', label='SOE [t:00]')
            ax1.set_ylabel('SOE (kWh)', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.locator_params(nbins=12,axis='y')
            ax1.set_ylim((0,self.CE_bess_kwh*1.5))
            SOE_tick_labels=ax1.get_yticklabels()
            SOE_tick_labels = [f"{l.get_text()} - {float(l.get_text().replace('−','-'))/self.CE_bess_kwh:.0%}" for l in SOE_tick_labels]
            ax1.set_yticklabels(SOE_tick_labels)
            h2, l2 = ax1.get_legend_handles_labels()
            h+=h2
            l+=l2
            bbox1=(0.19,0.96)
        fig.legend(h, l, loc='upper left',bbox_to_anchor=bbox1, frameon=True)

        # Primer eje derecho (COP/kWh)
        ax2 = ax0.twinx()
        ax2.step(periods, np.append(cop_kwh,cop_kwh[-1]), where='post', color='black', linestyle='--', label='Costo unitario')
        ax2.set_ylabel('Costo unitario ($COP/kWh)')
        ax2.set_ylim((0,np.round(max(cop_kwh)*1.5/200,0)*200))
        ax2.tick_params(axis='y', labelcolor='black')

        # Segundo eje derecho (COP totales)
        ax3 = ax0.twinx()
        ax3.spines['right'].set_position(('axes', 1.1))  # Mueve el eje un poco mas a la derecha
        ax3.step(periods, np.append(cop_hr,cop_hr[-1]), where='post', color='red', linestyle=':', label='Costo horario')
        ax3.set_ylabel('Costo horario ($COP/h)')
        ax3.set_ylim((0,np.round(max(cop_hr)*1.5/2.5e5,0)*2.5e5))
        ax3.tick_params(axis='y', labelcolor='black')
        ax3.yaxis.set_major_formatter('${x:,.0f}')

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        fig.legend(h1 + h2, l1 + l2, loc='upper right',
                bbox_to_anchor=(0.83, 0.96),  # coordenadas dentro del grafico
                frameon=True)

        ax0.text(0.02, 0.08,f"Promedio: ${cop_kwh_avg:,.0f} COP/kWh",transform=ax0.transAxes,fontsize=10,color='black',bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
        just_ED_no_R = not hasattr(EDResult,'fpart')
        cost_of_ed=EDResult.EDcost if just_ED_no_R else EDResult.EDnRcost
        ax0.text(0.02, 0.04,f"Costo Diario [ED{"" if just_ED_no_R else "+R"}]: ${cost_of_ed:,.0f} COP",transform=ax0.transAxes,fontsize=10,color='black',bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
        
        # plt.title("Despacho acumulado con costo/demanda")
        fig.set_dpi(500)
        plt.tight_layout()
        plt.show()

        if plot_R and not just_ED_no_R:
            self.plotR(EDResult,stackalpha=stackalpha)

    def plotR(self,EDresult,stackalpha):

        genlabels=self.gennames 
        T=self.T
        if self.HAS_BESS:
            genlabels+=['BESS']

        Hp=EDresult.H_p
        Hn=EDresult.H_n
        RTp=EDresult.ResT_p
        RTn=EDresult.ResT_n
        if len(Hp.shape)==1 or len(Hn.shape)==1:
            #in case detED
            Hp=np.tile(Hp,(T,1)).T
            Hn=np.tile(Hn,(T,1)).T
            RTp=np.tile(RTp,T)
            RTn=np.tile(RTn,T)

        periods = np.arange(0, 24+1/T, 24/T)
        fig, ax = plt.subplots(figsize=(14,8))
        ax.stackplot(periods,np.concatenate((Hp,np.array([Hp[:,-1]]).T),axis=1),labels=genlabels, alpha=stackalpha,step='post',colors=self.plotcolors)
        ax.stackplot(periods,-np.concatenate((Hn,np.array([Hn[:,-1]]).T),axis=1), alpha=stackalpha,step='post',colors=self.plotcolors)
        
        ax.step(periods,np.append(RTp,RTp[-1]),color="teal",linestyle='--',label=r"$R^+_T$",where='post')
        ax.step(periods,-np.append(RTn,RTn[-1]),color="teal",linestyle='--',label=r"$R^-_T$",where='post')

        ax.axhline(0,color='black',lw=1)
        ax.set_xticks(periods)
        ax.set_xlim((0,T-1))
        ax.set_xlabel('Tiempo (h)')
        ax.set_ylabel("Holgura (kW)")
        fig.legend(loc='upper left',bbox_to_anchor=(0.07,0.97),frameon=True)
        fig.set_dpi(500)
        plt.tight_layout()
        plt.show()
    
class detEDnR(EDnR):
    def __init__(self,MG:MG,subperiods:int=30,strictlycircularbess:bool=True,BESS_SOE_init=0.0,
                plotcolors=None,seed=None,
                grb_verbose:bool=False,logger_scope:int=1,logger_level:int=logging.CRITICAL,LastInstance=None,model_name=None,**kwargs):
        
        super().__init__(MG,subperiods,strictlycircularbess,BESS_SOE_init,
                            plotcolors,seed,grb_verbose,logger_scope,logger_level,LastInstance,model_name,**kwargs)

    def solve(self,TrainSampleSet=None,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,reservecost_wrt_gencost=0.4,**kwargs):
        """Solve deterministic EDnR. Does heuristic_reserve() then BaseDetED(), but
        takes an input training sample set to calculate avg/exp day, to be used as nominal day,
        **updating instance parameters (demand and Type 3 generation).**
        Returns decision x_dec=EDnRresult object with {Pgen[GxT],PBESS[T],SOE[T],fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
                    
        **R kwargs:**
            customReserve=None,customReg=None,peakSupportedDisconn=0.4,f_hi=62,f_low=58.5,
            reservecost_wrt_gencost=0.4,fpart_bess=0.6,Type3GenFactor=0.5,
            Res_verbose=False,T_RB_dc_h=0.5,T_RB_ch_h=0.5
        
        **ED kwargs:**
            plot_ED=False,EDplotstyle='stack',stackalpha=0.7,grb_verbose=None,BESS_SOE_init=None
        """
        self.reservecost_wrt_gencost=reservecost_wrt_gencost
        if TrainSampleSet is not None:
            # Update nominal day with expected from sample set
            self.updateNominaltoSampleSetMean(TrainSampleSet)
        
        # Realizar heuristica de reserva y obtener cotas de generacion
        EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
        self.logger.debug(f"edparams: {EDparams}")
        # Guardar para que resolve() sepa si es det
        self.ReserveResult=ReserveResult
        # Calcular costo de R segun heuristica
        c_gen_copkwh_w_bess=self.c_gen_copkwh
        if self.HAS_BESS: c_gen_copkwh_w_bess=np.append(c_gen_copkwh_w_bess,self.c_chdc_copkwh)   
        ReserveResult.Rcost=reservecost_wrt_gencost*np.sum(c_gen_copkwh_w_bess*(ReserveResult.H_p+ReserveResult.H_n))*self.T
        
        # Realizar ED con cotas definidas y obtener costos (D-1) de Despacho y Reserva
        EDResult=self.detED(EDparams,lambdas_C,lambdas_V,z_PC,z_PV,**kwargs)
        if EDResult==-1: return None,-1

        # Combinar resultados de ED + R
        x_dec=SimpleNamespace(**EDResult.__dict__,**ReserveResult.__dict__)                 
        x_dec.EDnRcost=x_dec.EDcost+x_dec.Rcost
        if plot_ED:
            self.plotED(x_dec,**kwargs)   
        J_is=x_dec.EDnRcost # ObjVal is just EDcost in det
        return x_dec,J_is
    
    def detED(self,params={},lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,grb_verbose=None,BESS_SOE_init=None,**kwargs):
        """Meant to be called after heuristic_reserve(). Solves deterministic ED with params (can be {}) and
        instance creation parameters. Returns object with {Pgen[GxT],PBESS[
            T],SOE[T],EDcost}."""
        T=self.T
        pG=self.pG
        M=self.M
        # Overwrites __init__ self.grb_verbose
        if grb_verbose is not None: self.grb_verbose=grb_verbose
        M.params.LogToConsole=self.grb_verbose
        # Overwrites __init__ self.BESS_SOE_init
        if BESS_SOE_init is not None: self.BESS_SOE_init=BESS_SOE_init
        # Parse params
        p_gmax_kw_mtx=params.get('p_gmax_kw_mtx',np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens]))
        p_gmin_kw_mtx=params.get('p_gmin_kw_mtx',np.vstack([np.zeros(T) for _ in self.MG.Gens]))
        ### GENERATOR CONSTRAINTS
        self.pmaxCtrt=M.addConstr(pG<=p_gmax_kw_mtx,"pmax")
        pmin=M.addConstrs((pG[:,i]>=p_gmin_kw_mtx[:,i] for i in range(T)),"pmin")
        upramp=M.addConstrs((pG[:,i+1]-pG[:,i]<=self.UR_kwhr for i in range(T-1)),"upramp")
        upramp_last=M.addConstr(pG[:,0]-pG[:,T-1]<=self.UR_kwhr,"upramp_last")
        downramp=M.addConstrs((pG[:,i+1]-pG[:,i]>=-self.DR_kwhr for i in range(T-1)),"downramp")
        downramp_last=M.addConstr(pG[:,0]-pG[:,T-1]>=-self.DR_kwhr,"downramp_last")    
                    
        GenBal=gp.GenExpr()
        GenBal=pG.sum(axis=0)
    
        # BESS VARIABLES AND CONSTRAINTS
        if self.HAS_BESS:
            # Parse params
            minSOE_perc=params.get('minSOE_perc',0)
            maxSOE_perc=params.get('maxSOE_perc',1)
            p_ch_max_bess_kw=params.get('p_ch_max_bess_kw',self.p_ch_max_bess_kw)
            p_dc_max_bess_kw=params.get('p_dc_max_bess_kw',self.p_dc_max_bess_kw)
            # Add BESS Vars
            pCH=M.addMVar(shape=(T),lb=0,ub=p_ch_max_bess_kw)
            pDC=M.addMVar(shape=(T),lb=0,ub=p_dc_max_bess_kw)
            SOE=M.addMVar(shape=(T),lb=minSOE_perc*self.CE_bess_kwh,ub=maxSOE_perc*self.CE_bess_kwh)
            # Add BESS [cRT*pDC] TO OBJECTIVE FUNCTION
            for t in range(T):
                self.fobj+=self.c_chdc_copkwh*(pDC[t]+1/10*pCH[t]) # 1/20 is to slightly penalize charging
            # Add BESS Constraints
            sumOfChDc=M.addConstrs((pCH[t]+pDC[t]<=min(p_ch_max_bess_kw,p_dc_max_bess_kw) for t in range(T)),"sumOfChDcCutPlane")
            SOEdynamics=M.addConstrs((SOE[t+1]==SOE[t]+self.deltat_h*(pCH[t+1]*self.eta_ch-pDC[t+1]/self.eta_dc) for t in range(T-1)),"SOEdynamics")
            if self.strictlycircularbess:
                SOEcircular=M.addConstr(SOE[0]==SOE[T-1]+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEcircular")
            else:    
                SOEstartcond=M.addConstr(SOE[0]==self.BESS_SOE_init*self.CE_bess_kwh+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEstartcond")
                SOEendcond=M.addConstr(SOE[T-1]>=self.BESS_SOE_init*self.CE_bess_kwh,"SOEendcond")
                etacutplane=(self.eta_ch+1/self.eta_dc)/2
                SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range(T-1)),"SOEdynamics")
            # make var/ctrt handlers accessible as attributes   
            self.pCH=pCH
            self.pDC=pDC
            self.SOE=SOE   
            GenBal+=pDC-pCH 
                        
        # FOR ADMM MODE, THE RESOLVE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
        if self.transactive:
            if self.neighbors==[]:
                raise Exception("transactive EDnR requires neighbors list")
            else:
                # Z AND LAMBDAS MUST BE GIVEN 
                assert z_PC is not None, f"z_PC is None"
                assert z_PV is not None, f"z_PV is None"
                assert lambdas_C is not None, f"lambdas_C is None"
                assert lambdas_V is not None, f"lambdas_V is None"
                self.z_PC_k=z_PC
                self.z_PV_k=z_PV
                self.lam_C_k=lambdas_C
                self.lam_V_k=lambdas_V
                # Add P compras and P ventas variables to model
                P_C=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_C") # P compras
                P_V=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_V") # P ventas
                
                # Add ADMM params as variables == constant RHS to update in resolve()
                z_PC_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                z_PV_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas
                
                self.z_PC_ctrt=M.addConstr(z_PC_var==self.z_PC_k,"z_PC_const")
                self.z_PV_ctrt=M.addConstr(z_PV_var==self.z_PV_k,"z_PV_const")
                
                lam_C_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                lam_V_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                self.lam_C_ctrt=M.addConstr(lam_C_var==self.lam_C_k,"lam_C_const")
                self.lam_V_ctrt=M.addConstr(lam_V_var==self.lam_V_k,"lam_V_const")
                
                for t in range(T):
                    for j in range(self.lenneighbors): #list of MG indices
                        ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                        self.fobj+=(lam_C_var[j,t]+self.lineCosts)*P_C[j,t]-lam_C_var[j,t]*P_V[j,t]
                        self.fobj+=(self.rho/2)*((P_C[j,t]-z_PC_var[j,t])*(P_C[j,t]-z_PC_var[j,t])+(P_V[j,t]-z_PV_var[j,t])*(P_V[j,t]-z_PV_var[j,t]))
                self.P_C=P_C
                self.P_V=P_V
            ### METER INTERCAMBIOS EN BALANCE DE CARGA
            GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)
        
        ### LOAD BALANCE CONSTRAINT
        self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")
        
        M.Params.QCPDual=1
        M.setObjective(self.fobj, GRB.MINIMIZE)
        self.logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            self.logger.warning(f"Uhhh something happened: {e}")
            self.logger.warning(f"{self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
            return -1
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            self.logger.info(f"ED solved optimally in {self.M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {self.M.status}")
            try:
                M.computeIIS()
                M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
                return -1
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
                return -1

        # Build Result Object
        result=SimpleNamespace()
        result.ObjVal=M.ObjVal
        if self.needMPO: result.MPO=self.loadbalanceCtrt.Pi
        result.Pgen=pG.X

        if self.transactive:
            result.P_C=P_C.X
            result.P_V=P_V.X
        
        if self.HAS_BESS:
            pCH_res=pCH.X
            pDC_res=pDC.X
            SOE_res=SOE.X
            result.PCH=pCH_res
            result.PDC=pDC_res
            result.PBESS=pDC_res-pCH_res
            result.SOE=SOE_res
            for t in range(T):
                if(pCH_res[t] > 1e-2 and pDC_res[t]>1e-2) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    self.logger.warning(f"t={t}")
                    self.logger.warning(f"pCH_res: {pCH_res[:t+1]}")
                    self.logger.warning(f"pDC_res: {pDC_res[:t+1]}")
                    self.logger.warning(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
        
        ## CALCULATE REAL COST OF ED+R [D-1] PLANNING: ED, R, PURCHASES AND SALES
        Cost_of_ED=0
        for t in range(T):
            # P_ED
            Cost_of_ED+=np.sum(self.c_gen_copkwh*result.Pgen[:,t])
            # P^DC_ED - only pay BESS per kwh cycled aka discharged, "they" pay for the charge up
            if self.HAS_BESS:
                Cost_of_ED+=self.c_chdc_copkwh*pDC_res[t]
            # buying and selling
            if self.transactive:
                for j in range(self.lenneighbors):
                    # if actually buying, add (LAM+lineCost) * P_C
                    if result.P_C[j,t]>1e-2:
                        Cost_of_ED+=(self.lam_C_k[j,t]+self.lineCosts)*result.P_C[j,t]
                    # if actually selling, sub LAM * P_V
                    if result.P_V[j,t]>1e-2:
                        Cost_of_ED-=self.lam_V_k[j,t]*result.P_V[j,t]        
        result.EDcost=Cost_of_ED
        # Si se corre desde solve, se le suma Rcost para obtener EDnRcost
        return result
        
            
#ED+R decision taken with SAA 
class SEDnR(EDnR):
    def __init__(self,MG:MG,subperiods:int=30,strictlycircularbess:bool=True,BESS_SOE_init=0.0,
                 plotcolors=None,seed=None,grb_verbose:bool=False,logger_scope:int=1,logger_level:int=logging.CRITICAL,LastInstance=None,model_name=None,**kwargs):
        super().__init__(MG,subperiods,strictlycircularbess,BESS_SOE_init,plotcolors,seed,grb_verbose,logger_scope,logger_level,LastInstance,model_name,**kwargs)
        
    def solve(self,TrainSampleSet,reservecost_wrt_gencost=0.4,Q=100,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,**kwargs):
        """Solve Stochastic EDnR. Returns decision x_dec=EDnRresult object with {Pgen[GxT],PBESS[T],SOE[T],fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
                    
        **R kwargs:**
            customReserve=None,customReg=None,peakSupportedDisconn=0.4,f_hi=62,f_low=58.5,
            reservecost_wrt_gencost=0.2,fpart_bess=0.6,Type3GenFactor=0.5,
            Res_verbose=False,T_RB_dc_h=0.5,T_RB_ch_h=0.5
        
        **ED kwargs:**
            plot_ED=False,EDplotstyle='stack',stackalpha=0.7,grb_verbose=None,BESS_SOE_init=None
        """ 
        self.reservecost_wrt_gencost=reservecost_wrt_gencost
        # Update nominal day with expected from sample set
        self.updateNominaltoSampleSetMean(TrainSampleSet)
                
        # Se obtienen las reservas nominales a partir de la heuristica determinista
        # Estas se usaran como RESERVAS MAXIMAS en el ED estocastico
        EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
        SEDnRparams={'Hp_max':ReserveResult.H_p,
                    'Hn_max':ReserveResult.H_n}    
        if self.HAS_BESS:
            SEDnRparams=SEDnRparams|{'minSOE_perc':EDparams['minSOE_perc'],
                    'maxSOE_perc':EDparams['maxSOE_perc']}

        # Toma el conjunto muestral y lo convierte en variaciones a partir del nominal
        XiScenarioSet=self.GetVarScenariosFromSampleSet(TrainSampleSet) # Scenarios of effective variable generation or negative effective load
        # Se calculan los escenarios de cuantiles maximos/minimos de variacion **por cada periodo**
        Ximax,Ximin=self.BoundsFromSet(XiScenarioSet,Q)

        # Llama solve_stochastic con las reservas maximas y los escenarios de variacion
        x_dec=self.solve_stochastic(XiScenarioSet,Ximax,Ximin,SEDnRparams,reservecost_wrt_gencost,Q,lambdas_C,lambdas_V,z_PC,z_PV,**kwargs) ## ED AND R RESULTS
        if x_dec==-1: return None,-1
        if plot_ED:
            self.plotED(x_dec,**kwargs)  
        # convierte el resultado Jis
        # J_is=x_dec.EDnRcost #nope, this is for Joos
        J_is=x_dec.ObjVal 
        return x_dec,J_is
    
    def solve_stochastic(self,XiScenarioSet,Ximax,Ximin,params,reservecost_wrt_gencost,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,grb_verbose=None,BESS_SOE_init=None,**kwargs):
        """
        Meant to be called by solve(). Solves Stochastic ED (SAA) with params (can be {}) and
        instance creation parameters. Returns EDnR object with {Pgen[GxT],PBESS[T],SOE[T],EDcost,fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,Rcost,EDnRcost}.
        """   

        T=self.T
        pG=self.pG
        M=self.M
        # Overwrites __init__ self.grb_verbose
        if grb_verbose is not None: self.grb_verbose=grb_verbose
        M.params.LogToConsole=self.grb_verbose
        # Overwrites __init__ self.BESS_SOE_init
        if BESS_SOE_init is not None: self.BESS_SOE_init=BESS_SOE_init
        # Parse params
        minSOE_perc=params.get('minSOE_perc',0)
        maxSOE_perc=params.get('maxSOE_perc',1)
        Hp_max=params.get('Hp_max',0)
        Hn_max=params.get('Hn_max',0)
        
        ### GENERATOR CONSTRAINTS
        p_gmax_kw_mtx=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
        p_gmin_kw_mtx=np.vstack([np.zeros(T) for _ in self.MG.Gens])
        self.pmaxCtrt=M.addConstr(pG<=p_gmax_kw_mtx,"pmax")
        pmin=M.addConstrs((pG[:,i]>=p_gmin_kw_mtx[:,i] for i in range(T)),"pmin")
        upramp=M.addConstrs((pG[:,i+1]-pG[:,i]<=self.UR_kwhr for i in range(T-1)),"upramp")
        upramp_last=M.addConstr(pG[:,0]-pG[:,T-1]<=self.UR_kwhr,"upramp_last")
        downramp=M.addConstrs((pG[:,i+1]-pG[:,i]>=-self.DR_kwhr for i in range(T-1)),"downramp")
        downramp_last=M.addConstr(pG[:,0]-pG[:,T-1]>=-self.DR_kwhr,"downramp_last")    
                    
        GenBal=gp.LinExpr()
        GenBal=pG.sum(axis=0)
        
        ### RESERVE VARIABLES
        Rp=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.vstack([Hp_max for _ in range(T)]).T,name="Rp") # upward reserve
        Rn=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.vstack([Hn_max for _ in range(T)]).T,name="Rn") # downward reserve
        fpart=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.ones(((self.Ngen+self.HAS_BESS),T)),name="fpart") # participation factor in frequency regulation
        self.createXiScenarioSetVar(XiScenarioSet) #xihat_var,samplectrt
        Nscen=XiScenarioSet.shape[0]
        ximax_v=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="ximax_v")
        ximin_v=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="ximin_v")
        
        self.Rp=Rp
        self.Rn=Rn
        self.fpart=fpart
        
        ## RESERVE CONSTRAINTS
        # self.logger.debug(f"ximin: {Ximin}, ximax={Ximax}")
        self.ximinCtrt=M.addConstr(ximin_v==Ximin,"ximinctrt")
        self.ximaxCtrt=M.addConstr(ximax_v==Ximax,"ximaxctrt")
        fpartis1=M.addConstrs((fpart[:,t].sum()==1 for t in range(T)),"fpartis1") # sum of participation factors is 1 at each t
        RpandPGenLim=M.addConstrs((p_gmin_kw_mtx[:,t]<=pG[:,t]-Rn[:self.Ngen,t] for t in range(T)),"RpandPGenLim")
        RnandPGenLim=M.addConstrs((pG[:,t]+Rp[:self.Ngen,t]<=p_gmax_kw_mtx[:,t] for t in range(T)),"RnandPGenLim")
        fpartximax=M.addConstrs((fpart[:,t]*ximax_v[t]<=Rn[:,t] for t in range(T)),"fpartximax")
        fpartximin=M.addConstrs((-fpart[:,t]*ximin_v[t]<=Rp[:,t] for t in range(T)),"fpartximin")
        
        # Full costs vector Gens+BESS
        c_gen_copkwh_w_bess=self.c_gen_copkwh

        # BESS VARIABLES AND CONSTRAINTS
        if self.HAS_BESS:
            # Add BESS Vars
            pCH=M.addMVar(shape=(T),lb=0,ub=GRB.INFINITY)
            pDC=M.addMVar(shape=(T),lb=0,ub=GRB.INFINITY)
            SOE=M.addMVar(shape=(T),lb=minSOE_perc*self.CE_bess_kwh,ub=maxSOE_perc*self.CE_bess_kwh)
            
            # Ideally pDCmax=M.addConstrs((pDC[t]<=max(self.p_dc_max_bess_kw-Rp[-1,t],0)) for t in range(T)), same for pCH
            p_ch_max_clip=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            p_dc_max_clip=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _aux_ch=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _aux_dc=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _off_ch=M.addConstrs((_aux_ch[t]==self.p_ch_max_bess_kw-Rn[-1,t]) for t in range(T))
            _off_dc=M.addConstrs((_aux_dc[t]==self.p_dc_max_bess_kw-Rp[-1,t]) for t in range(T))
            _clip_ch=M.addConstrs((p_ch_max_clip[t]==gp.max_(_aux_ch[t],0) for t in range(T)))
            _clip_dc=M.addConstrs((p_dc_max_clip[t]==gp.max_(_aux_dc[t],0) for t in range(T)))
            pCHmax=M.addConstr(pCH<=p_ch_max_clip)
            pDCmax=M.addConstr(pDC<=p_dc_max_clip)            
            
            # Add BESS [cRT*pDC] TO OBJECTIVE FUNCTION
            for t in range(T):
                self.fobj+=self.c_chdc_copkwh*(pDC[t]+1/10*pCH[t]) # 1/10 is to slightly penalize charging
                # BESS RESERVE COST
                self.fobj+=reservecost_wrt_gencost*self.c_chdc_copkwh*(Rp[-1,t]+Rn[-1,t])
                        
            # Add BESS Constraints
            sumOfChDc=M.addConstrs((pCH[t]+pDC[t]<=min(self.p_ch_max_bess_kw,self.p_dc_max_bess_kw) for t in range(T)),"sumOfChDcCutPlane")
            SOEdynamics=M.addConstrs((SOE[t+1]==SOE[t]+self.deltat_h*(pCH[t+1]*self.eta_ch-pDC[t+1]/self.eta_dc) for t in range(T-1)),"SOEdynamics")
            if self.strictlycircularbess:
                SOEcircular=M.addConstr(SOE[0]==SOE[T-1]+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEcircular")
            else:    
                SOEstartcond=M.addConstr(SOE[0]==self.BESS_SOE_init*self.CE_bess_kwh+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEstartcond")
                SOEendcond=M.addConstr(SOE[T-1]>=self.BESS_SOE_init*self.CE_bess_kwh,"SOEendcond")
                etacutplane=(self.eta_ch+1/self.eta_dc)/2
                SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range(T-1)),"SOEdynamics")
            # make var/ctrt handlers accessible as attributes
            self.pCH=pCH
            self.pDC=pDC
            self.SOE=SOE   
            GenBal+=pDC-pCH 
            c_gen_copkwh_w_bess=np.append(c_gen_copkwh_w_bess,self.c_chdc_copkwh)
    
        ## ADD RESERVE ASSIGNMENT COSTS TO OBJECTIVE FUNCTION
        self.fobj+=reservecost_wrt_gencost*c_gen_copkwh_w_bess@(Rp.sum(axis=1)+Rn.sum(axis=1))
        
        ## ADD SCENARIO COSTS TO OBJECTIVE FUNCTION
        # fobj add SUM -c fpart  xiscen_i
        ScenCost=gp.LinExpr()
        ScenCost=-c_gen_copkwh_w_bess@fpart@self.xihat_var.sum(axis=0)/Nscen
        self.fobj+=ScenCost

        # FOR ADMM MODE, THE RESOLVE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
        if self.transactive:
            if self.neighbors==[]:
                raise Exception("transactive EDnR requires neighbors list")
            else:
                # Z AND LAMBDAS MUST BE SET
                assert z_PC is not None, f"z_PC is None"
                assert z_PV is not None, f"z_PV is None"
                assert lambdas_C is not None, f"lambdas_C is None"
                assert lambdas_V is not None, f"lambdas_V is None"
                self.z_PC_k=z_PC
                self.z_PV_k=z_PV
                self.lam_C_k=lambdas_C
                self.lam_V_k=lambdas_V
                # Add P compras and P ventas variables to model
                P_C=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_C") # P compras
                P_V=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_V") # P ventas
                
                # Add ADMM params as variables == constant RHS to update in resolve()
                z_PC_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                z_PV_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas
                
                self.z_PC_ctrt=M.addConstr(z_PC_var==self.z_PC_k,"z_PC_const")
                self.z_PV_ctrt=M.addConstr(z_PV_var==self.z_PV_k,"z_PV_const")
                
                lam_C_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                lam_V_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                self.lam_C_ctrt=M.addConstr(lam_C_var==self.lam_C_k,"lam_C_const")
                self.lam_V_ctrt=M.addConstr(lam_V_var==self.lam_V_k,"lam_V_const")
                
                for t in range(T):
                    for j in range(self.lenneighbors): #list of MG indices
                        ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                        self.fobj+=(lam_C_var[j,t]+self.lineCosts)*P_C[j,t]-lam_C_var[j,t]*P_V[j,t]
                        self.fobj+=(self.rho/2)*((P_C[j,t]-z_PC_var[j,t])*(P_C[j,t]-z_PC_var[j,t])+(P_V[j,t]-z_PV_var[j,t])*(P_V[j,t]-z_PV_var[j,t]))
                self.P_C=P_C
                self.P_V=P_V
            ### METER INTERCAMBIOS EN BALANCE DE CARGA
            GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)
        
        ### LOAD BALANCE CONSTRAINT
        self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")
        
        M.Params.QCPDual=1        
        M.setObjective(self.fobj, GRB.MINIMIZE)
        self.logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            self.logger.warning(f"Uhhh something happened: {e}")
            self.logger.warning(f"{self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
            return -1
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            self.logger.info(f"ED solved optimally in {self.M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {self.M.status}")
            try:
                M.computeIIS()
                M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
                return -1
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
                return -1

        # Build Result Object
        result=SimpleNamespace()
        result.ObjVal=M.ObjVal
        # GETTING THE DUALS MAY BE A BIT HARDER WITH BATTERIES
        if self.needMPO:
            try:
                result.MPO=self.loadbalanceCtrt.Pi
            except gp.GurobiError as e:
                self.logger.critical(f"Couldnt get LMP/MPO as Pi - {e}, trying from fixed model")
                fixedmodel=M.fixed()
                fixedmodel.optimize()
                fixedctrs_=[fixedmodel.getConstrByName(n) for n in self.loadbalanceCtrt.ConstrName]
                result.MPO=np.array(list(map(lambda c: c.Pi, fixedctrs_)))
                assert len(result.MPO)==T and (result.MPO>0).all(),f"Invalid LMP/MPOs: {result.MPO}"

        result.Pgen=pG.X
        result.fpart=fpart.X
        result.H_p=Rp.X
        result.H_n=Rn.X
        result.ResT_p=np.sum(Rp.X,axis=0)
        result.ResT_n=np.sum(Rn.X,axis=0)

        if self.transactive:
            result.P_C=P_C.X
            result.P_V=P_V.X
        
        if self.HAS_BESS:
            pCH_res=pCH.X
            pDC_res=pDC.X
            SOE_res=SOE.X
            result.PCH=pCH_res
            result.PDC=pDC_res
            result.PBESS=pDC_res-pCH_res
            result.SOE=SOE_res
            for t in range(T):
                if(pCH_res[t] > 1e-2 and pDC_res[t]>1e-2) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    self.logger.warning(f"t={t}")
                    self.logger.warning(f"pCH_res: {pCH_res[:t+1]}")
                    self.logger.warning(f"pDC_res: {pDC_res[:t+1]}")
                    self.logger.warning(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")

        ## CALCULATE REAL COST OF ED+R [D-1] PLANNING: ED, R, PURCHASES AND SALES
        Cost_of_EDnR=0
        for t in range(T):
            # P_ED
            Cost_of_EDnR+=np.sum(self.c_gen_copkwh*result.Pgen[:,t])
            # P^DC_ED - only pay BESS per kwh cycled aka discharged, "they" pay for the charge up
            if self.HAS_BESS:
                Cost_of_EDnR+=self.c_chdc_copkwh*pDC_res[t]
            # R+ and R-
            Cost_of_EDnR+=reservecost_wrt_gencost*np.sum(c_gen_copkwh_w_bess*(result.H_p[:,t]+result.H_n[:,t]))
            # buying and selling
            if self.transactive:
                for j in range(self.lenneighbors):
                    # if actually buying, add (LAM+lineCost) * P_C
                    if result.P_C[j,t]>1e-2:
                        Cost_of_EDnR+=(self.lam_C_k[j,t]+self.lineCosts)*result.P_C[j,t]
                    # if actually selling, sub LAM * P_V
                    if result.P_V[j,t]>1e-2:
                        Cost_of_EDnR-=self.lam_V_k[j,t]*result.P_V[j,t]
        result.EDnRcost=Cost_of_EDnR
        return result
    
#ED+R decision taken with P(Q) from xihat itself, no assuming distribution
class REDnR(EDnR):
    def __init__(self,MG:MG,subperiods:int=30,strictlycircularbess:bool=True,BESS_SOE_init=0.0,
                 plotcolors=None,seed=None,
                 grb_verbose:bool=False,logger_scope:int=1,logger_level:int=logging.CRITICAL,LastInstance=None,model_name=None,**kwargs):
        super().__init__(MG,subperiods,strictlycircularbess,BESS_SOE_init,plotcolors,seed,grb_verbose,logger_scope,logger_level,LastInstance,model_name=None,**kwargs)
    def solve(self,TrainSampleSet,Q=100,reservecost_wrt_gencost=0.4,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,**kwargs):
        """
        Solve Robust EDnR. Returns decision x_dec=EDnRresult object with {Pgen[GxT],PBESS[T],SOE[T],fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
        """
        self.reservecost_wrt_gencost=reservecost_wrt_gencost

        # Update nominal day with expected from sample set
        self.updateNominaltoSampleSetMean(TrainSampleSet)
                
        # Se obtienen las reservas nominales a partir de la heuristica determinista
        # Estas se usaran como RESERVAS MAXIMAS en el ED estocastico
        EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
        REDnRparams={'Hp_max':ReserveResult.H_p,
                    'Hn_max':ReserveResult.H_n}    
        if self.HAS_BESS:
            REDnRparams=REDnRparams|{'minSOE_perc':EDparams['minSOE_perc'],
                    'maxSOE_perc':EDparams['maxSOE_perc']}

        # Toma el conjunto muestral y lo convierte en variaciones a partir del nominal
        XiScenarioSet=self.GetVarScenariosFromSampleSet(TrainSampleSet) # Scenarios of effective variable generation or negative effective load

        # Se calculan los escenarios de cuantiles maximos/minimos de variacion **por cada periodo**
        Ximax,Ximin=self.BoundsFromSet(XiScenarioSet,Q)
                
        # Llama solve_robust con las reservas maximas y los escenarios de variacion extremos
        x_dec=self.solve_robust(Ximax,Ximin,REDnRparams,reservecost_wrt_gencost,lambdas_C,lambdas_V,z_PC,z_PV,**kwargs) ## ED AND R RESULTS
        if x_dec==-1: return None,-1
        if plot_ED:
            self.plotED(x_dec,**kwargs)  
        # convierte el resultado Jis
        # J_is=x_dec.EDnRcost #nope, this is for Joos
        J_is=x_dec.ObjVal
        return x_dec,J_is     
       
    def solve_robust(self,Ximax,Ximin,params,reservecost_wrt_gencost,lambdas_C,lambdas_V,z_PC,z_PV,grb_verbose=None,BESS_SOE_init=None,**kwargs):
        """
        Meant to be called by solve(). Solves Robust EDnR with Quantile edge samples (can be {}) and
        instance creation parameters. Returns EDnR object with {Pgen[GxT],PBESS[T],SOE[T],EDcost,fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,Rcost,EDnRcost}.
        """           
        
        T=self.T
        pG=self.pG
        M=self.M
        # Overwrites __init__ self.grb_verbose
        if grb_verbose is not None: self.grb_verbose=grb_verbose
        M.params.LogToConsole=self.grb_verbose
        # Overwrites __init__ self.BESS_SOE_init
        if BESS_SOE_init is not None: self.BESS_SOE_init=BESS_SOE_init
        # Parse params
        minSOE_perc=params.get('minSOE_perc',0)
        maxSOE_perc=params.get('maxSOE_perc',1)
        Hp_max=params.get('Hp_max',0)
        Hn_max=params.get('Hn_max',0)
        
        ### GENERATOR CONSTRAINTS
        p_gmax_kw_mtx=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
        p_gmin_kw_mtx=np.vstack([np.zeros(T) for _ in self.MG.Gens])
        self.pmaxCtrt=M.addConstr(pG<=p_gmax_kw_mtx,"pmax")
        pmin=M.addConstrs((pG[:,i]>=p_gmin_kw_mtx[:,i] for i in range(T)),"pmin")
        upramp=M.addConstrs((pG[:,i+1]-pG[:,i]<=self.UR_kwhr for i in range(T-1)),"upramp")
        upramp_last=M.addConstr(pG[:,0]-pG[:,T-1]<=self.UR_kwhr,"upramp_last")
        downramp=M.addConstrs((pG[:,i+1]-pG[:,i]>=-self.DR_kwhr for i in range(T-1)),"downramp")
        downramp_last=M.addConstr(pG[:,0]-pG[:,T-1]>=-self.DR_kwhr,"downramp_last")    
                    
        GenBal=gp.LinExpr()
        GenBal=pG.sum(axis=0)
        
        ### RESERVE VARIABLES
        Rp=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.vstack([Hp_max for _ in range(T)]).T,name="Rp") # upward reserve
        Rn=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.vstack([Hn_max for _ in range(T)]).T,name="Rn") # downward reserve
        fpart=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.ones(((self.Ngen+self.HAS_BESS),T)),name="fpart") # participation factor in frequency regulation
        ximax_v=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="ximax_v")
        ximin_v=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="ximin_v")
        
        self.Rp=Rp
        self.Rn=Rn
        self.fpart=fpart
        
        ## RESERVE CONSTRAINTS
        # self.logger.debug(f"ximin: {Ximin}, ximax={Ximax}")
        self.ximinCtrt=M.addConstr(ximin_v==Ximin,"ximinctrt")
        self.ximaxCtrt=M.addConstr(ximax_v==Ximax,"ximaxctrt")
        fpartis1=M.addConstrs((fpart[:,t].sum()==1 for t in range(T)),"fpartis1") # sum of participation factors is 1 at each t
        RpandPGenLim=M.addConstrs((p_gmin_kw_mtx[:,t]<=pG[:,t]-Rn[:self.Ngen,t] for t in range(T)),"RpandPGenLim")
        RnandPGenLim=M.addConstrs((pG[:,t]+Rp[:self.Ngen,t]<=p_gmax_kw_mtx[:,t] for t in range(T)),"RnandPGenLim")
        fpartximax=M.addConstrs((fpart[:,t]*ximax_v[t]<=Rn[:,t] for t in range(T)),"fpartximax")
        fpartximin=M.addConstrs((-fpart[:,t]*ximin_v[t]<=Rp[:,t] for t in range(T)),"fpartximin")
        
        # Full costs vector Gens+BESS
        c_gen_copkwh_w_bess=self.c_gen_copkwh

        # BESS VARIABLES AND CONSTRAINTS
        if self.HAS_BESS:
            # Add BESS Vars
            pCH=M.addMVar(shape=(T),lb=0,ub=GRB.INFINITY)
            pDC=M.addMVar(shape=(T),lb=0,ub=GRB.INFINITY)
            SOE=M.addMVar(shape=(T),lb=minSOE_perc*self.CE_bess_kwh,ub=maxSOE_perc*self.CE_bess_kwh)
            
            # Ideally pDCmax=M.addConstrs((pDC[t]<=max(self.p_dc_max_bess_kw-Rp[-1,t],0)) for t in range(T)), same for pCH
            p_ch_max_clip=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            p_dc_max_clip=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _aux_ch=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _aux_dc=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _off_ch=M.addConstrs((_aux_ch[t]==self.p_ch_max_bess_kw-Rn[-1,t]) for t in range(T))
            _off_dc=M.addConstrs((_aux_dc[t]==self.p_dc_max_bess_kw-Rp[-1,t]) for t in range(T))
            _clip_ch=M.addConstrs((p_ch_max_clip[t]==gp.max_(_aux_ch[t],0) for t in range(T)))
            _clip_dc=M.addConstrs((p_dc_max_clip[t]==gp.max_(_aux_dc[t],0) for t in range(T)))
            pCHmax=M.addConstr(pCH<=p_ch_max_clip)
            pDCmax=M.addConstr(pDC<=p_dc_max_clip)            
            
            # Add BESS [cRT*pDC] TO OBJECTIVE FUNCTION
            for t in range(T):
                self.fobj+=self.c_chdc_copkwh*(pDC[t]+1/10*pCH[t]) # 1/10 is to slightly penalize charging
                # BESS RESERVE COST
                self.fobj+=reservecost_wrt_gencost*self.c_chdc_copkwh*(Rp[-1,t]+Rn[-1,t])
                        
            # Add BESS Constraints
            sumOfChDc=M.addConstrs((pCH[t]+pDC[t]<=min(self.p_ch_max_bess_kw,self.p_dc_max_bess_kw) for t in range(T)),"sumOfChDcCutPlane")
            SOEdynamics=M.addConstrs((SOE[t+1]==SOE[t]+self.deltat_h*(pCH[t+1]*self.eta_ch-pDC[t+1]/self.eta_dc) for t in range(T-1)),"SOEdynamics")
            if self.strictlycircularbess:
                SOEcircular=M.addConstr(SOE[0]==SOE[T-1]+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEcircular")
            else:    
                SOEstartcond=M.addConstr(SOE[0]==self.BESS_SOE_init*self.CE_bess_kwh+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEstartcond")
                SOEendcond=M.addConstr(SOE[T-1]>=self.BESS_SOE_init*self.CE_bess_kwh,"SOEendcond")
                etacutplane=(self.eta_ch+1/self.eta_dc)/2
                SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range(T-1)),"SOEdynamics")
            # make var/ctrt handlers accessible as attributes
            self.pCH=pCH
            self.pDC=pDC
            self.SOE=SOE   
            GenBal+=pDC-pCH 
            c_gen_copkwh_w_bess=np.append(c_gen_copkwh_w_bess,self.c_chdc_copkwh)
    
        ## ADD RESERVE ASSIGNMENT COSTS TO OBJECTIVE FUNCTION
        self.fobj+=reservecost_wrt_gencost*c_gen_copkwh_w_bess@(Rp.sum(axis=1)+Rn.sum(axis=1))
        
        ## ADD WORSTCASE COSTS TO OBJECTIVE FUNCTION
        # fobj add -c fpart  ximin
        WCcost=gp.LinExpr()
        WCcost= -c_gen_copkwh_w_bess@fpart@ximin_v
        self.fobj+=WCcost
    
        # FOR ADMM MODE, THE RESOLVE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
        if self.transactive:
            if self.neighbors==[]:
                raise Exception("transactive EDnR requires neighbors list")
            else:
                # Z AND LAMBDAS MUST BE SET
                assert z_PC is not None, f"z_PC is None"
                assert z_PV is not None, f"z_PV is None"
                assert lambdas_C is not None, f"lambdas_C is None"
                assert lambdas_V is not None, f"lambdas_V is None"
                self.z_PC_k=z_PC
                self.z_PV_k=z_PV
                self.lam_C_k=lambdas_C
                self.lam_V_k=lambdas_V
                # Add P compras and P ventas variables to model
                P_C=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_C") # P compras
                P_V=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_V") # P ventas
                
                # Add ADMM params as variables == constant RHS to update in resolve()
                z_PC_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                z_PV_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas
                
                self.z_PC_ctrt=M.addConstr(z_PC_var==self.z_PC_k,"z_PC_const")
                self.z_PV_ctrt=M.addConstr(z_PV_var==self.z_PV_k,"z_PV_const")
                
                lam_C_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                lam_V_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                self.lam_C_ctrt=M.addConstr(lam_C_var==self.lam_C_k,"lam_C_const")
                self.lam_V_ctrt=M.addConstr(lam_V_var==self.lam_V_k,"lam_V_const")
                
                for t in range(T):
                    for j in range(self.lenneighbors): #list of MG indices
                        ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                        self.fobj+=(lam_C_var[j,t]+self.lineCosts)*P_C[j,t]-lam_C_var[j,t]*P_V[j,t]
                        self.fobj+=(self.rho/2)*((P_C[j,t]-z_PC_var[j,t])*(P_C[j,t]-z_PC_var[j,t])+(P_V[j,t]-z_PV_var[j,t])*(P_V[j,t]-z_PV_var[j,t]))
                self.P_C=P_C
                self.P_V=P_V
            ### METER INTERCAMBIOS EN BALANCE DE CARGA
            GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)
                  
        ### LOAD BALANCE CONSTRAINT
        self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")

        M.Params.QCPDual=1
        M.setObjective(self.fobj, GRB.MINIMIZE)
        self.logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            self.logger.warning(f"Uhhh something happened: {e}")
            self.logger.warning(f"{self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
            return -1
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            self.logger.info(f"ED solved optimally in {self.M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {self.M.status}")
            try:
                M.computeIIS()
                M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
                return -1
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
                return -1

        # Build Result Object
        result=SimpleNamespace()
        result.ObjVal=M.ObjVal
        # GETTING THE DUALS MAY BE A BIT HARDER WITH BATTERIES
        if self.needMPO:
            try:
                result.MPO=self.loadbalanceCtrt.Pi
            except gp.GurobiError as e:
                self.logger.critical(f"Couldnt get LMP/MPO as Pi - {e}, trying from fixed model")
                fixedmodel=M.fixed()
                fixedmodel.optimize()
                fixedctrs_=[fixedmodel.getConstrByName(n) for n in self.loadbalanceCtrt.ConstrName]
                result.MPO=np.array(list(map(lambda c: c.Pi, fixedctrs_)))
                assert len(result.MPO)==T and (result.MPO>0).all(),f"Invalid LMP/MPOs: {result.MPO}"

        result.Pgen=pG.X
        result.fpart=fpart.X
        result.H_p=Rp.X
        result.H_n=Rn.X
        result.ResT_p=np.sum(Rp.X,axis=0)
        result.ResT_n=np.sum(Rn.X,axis=0)

        if self.transactive:
            result.P_C=P_C.X
            result.P_V=P_V.X
        
        if self.HAS_BESS:
            pCH_res=pCH.X
            pDC_res=pDC.X
            SOE_res=SOE.X
            result.PCH=pCH_res
            result.PDC=pDC_res
            result.PBESS=pDC_res-pCH_res
            result.SOE=SOE_res
            for t in range(T):
                if(pCH_res[t] > 1e-2 and pDC_res[t]>1e-2) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    self.logger.warning(f"t={t}")
                    self.logger.warning(f"pCH_res: {pCH_res[:t+1]}")
                    self.logger.warning(f"pDC_res: {pDC_res[:t+1]}")
                    self.logger.warning(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
        
        ## CALCULATE REAL COST OF ED+R [D-1] PLANNING: ED, R, PURCHASES AND SALES
        Cost_of_EDnR=0
        for t in range(T):
            # P_ED
            Cost_of_EDnR+=np.sum(self.c_gen_copkwh*result.Pgen[:,t])
            # P^DC_ED - only pay BESS per kwh cycled aka discharged, "they" pay for the charge up
            if self.HAS_BESS:
                Cost_of_EDnR+=self.c_chdc_copkwh*pDC_res[t]
            # R+ and R-
            Cost_of_EDnR+=reservecost_wrt_gencost*np.sum(c_gen_copkwh_w_bess*(result.H_p[:,t]+result.H_n[:,t]))
            # buying and selling
            if self.transactive:
                for j in range(self.lenneighbors):
                    # if actually buying, add (LAM+lineCost) * P_C
                    if result.P_C[j,t]>1e-2:
                        Cost_of_EDnR+=(self.lam_C_k[j,t]+self.lineCosts)*result.P_C[j,t]
                    # if actually selling, sub LAM * P_V
                    if result.P_V[j,t]>1e-2:
                        Cost_of_EDnR-=self.lam_V_k[j,t]*result.P_V[j,t]        
        result.EDnRcost=Cost_of_EDnR
        return result        
    
class DRWEDnR(EDnR):
    def __init__(self,MG:MG,subperiods:int=30,strictlycircularbess:bool=True,BESS_SOE_init=0.0,
                 plotcolors=None,seed=None,
                 grb_verbose:bool=False,logger_scope:int=1,logger_level:int=logging.CRITICAL,LastInstance=None,model_name=None,**kwargs):
        super().__init__(MG,subperiods,strictlycircularbess,BESS_SOE_init,plotcolors,seed,grb_verbose,logger_scope,logger_level,LastInstance,model_name,**kwargs)
        
    def solve(self,TrainSampleSet,rwass=0.001,Q=100,reservecost_wrt_gencost=0.4,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,**kwargs):
        """
        Solve Distributionally Robust Wasserstein EDnR. Returns decision x_dec=EDnRresult object
        with {Pgen[GxT],PBESS[T],SOE[T],fpart,ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
        """
        self.reservecost_wrt_gencost=reservecost_wrt_gencost

        # Update nominal day with expected from sample set
        self.updateNominaltoSampleSetMean(TrainSampleSet)
                
        # Se obtienen las reservas nominales a partir de la heuristica determinista
        # Estas se usaran como RESERVAS MAXIMAS en el ED estocastico
        EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
        DRWEDnRparams={'Hp_max':ReserveResult.H_p,
                    'Hn_max':ReserveResult.H_n}    
        if self.HAS_BESS:
            DRWEDnRparams=DRWEDnRparams|{'minSOE_perc':EDparams['minSOE_perc'],
                    'maxSOE_perc':EDparams['maxSOE_perc']}

        # Toma el conjunto muestral y lo convierte en variaciones a partir del nominal
        XiScenarioSet=self.GetVarScenariosFromSampleSet(TrainSampleSet) # Scenarios of effective variable generation or negative effective load
        # Se calculan los escenarios de cuantiles maximos/minimos de variacion **por cada periodo**
        Ximax,Ximin=self.BoundsFromSet(XiScenarioSet,Q)
        
        # Llama solve_dro con las reservas maximas y los escenarios de variacion extremos
        x_dec=self.solve_drow(XiScenarioSet,Ximax,Ximin,rwass,DRWEDnRparams,reservecost_wrt_gencost,lambdas_C,lambdas_V,z_PC,z_PV,**kwargs) ## ED AND R RESULTS
        if x_dec==-1: return None,-1
        # convierte el resultado Jis
        # J_is=x_dec.EDnRcost #nope, this is for Joos
        J_is=x_dec.ObjVal
        if plot_ED:
            self.plotED(x_dec,**kwargs)  
        return x_dec,J_is
    
    def solve_drow(self,XiScenarioSet,Ximax,Ximin,rwass,params,reservecost_wrt_gencost,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,grb_verbose=None,BESS_SOE_init=None,**kwargs):
        """
        Meant to be called by solve(). Solves Wasserstein Distributionally Robust EDnR with params (can be {}) and
        instance creation parameters. Returns EDnR object with {Pgen[GxT],PBESS[T],SOE[T],EDcost,fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,Rcost,EDnRcost}.
        """   

        T=self.T
        pG=self.pG
        M=self.M
        # Overwrites __init__ self.grb_verbose
        if grb_verbose is not None: self.grb_verbose=grb_verbose
        M.params.LogToConsole=self.grb_verbose
        # Overwrites __init__ self.BESS_SOE_init
        if BESS_SOE_init is not None: self.BESS_SOE_init=BESS_SOE_init
        # Parse params
        minSOE_perc=params.get('minSOE_perc',0)
        maxSOE_perc=params.get('maxSOE_perc',1)
        Hp_max=params.get('Hp_max',0)
        Hn_max=params.get('Hn_max',0)
        
        ### GENERATOR CONSTRAINTS
        p_gmax_kw_mtx=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
        p_gmin_kw_mtx=np.vstack([np.zeros(T) for _ in self.MG.Gens])
        self.pmaxCtrt=M.addConstr(pG<=p_gmax_kw_mtx,"pmax")
        pmin=M.addConstrs((pG[:,i]>=p_gmin_kw_mtx[:,i] for i in range(T)),"pmin")
        upramp=M.addConstrs((pG[:,i+1]-pG[:,i]<=self.UR_kwhr for i in range(T-1)),"upramp")
        upramp_last=M.addConstr(pG[:,0]-pG[:,T-1]<=self.UR_kwhr,"upramp_last")
        downramp=M.addConstrs((pG[:,i+1]-pG[:,i]>=-self.DR_kwhr for i in range(T-1)),"downramp")
        downramp_last=M.addConstr(pG[:,0]-pG[:,T-1]>=-self.DR_kwhr,"downramp_last")    
                    
        GenBal=gp.LinExpr()
        GenBal=pG.sum(axis=0)

        ### RESERVE VARIABLES
        Rp=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.vstack([Hp_max for _ in range(T)]).T,name="Rp") # upward reserve
        Rn=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.vstack([Hn_max for _ in range(T)]).T,name="Rn") # downward reserve
        fpart=M.addMVar(shape=((self.Ngen+self.HAS_BESS),T),lb=0,ub=np.ones(((self.Ngen+self.HAS_BESS),T)),name="fpart") # participation factor in frequency regulation
        self.createXiScenarioSetVar(XiScenarioSet) #xihat_var,samplectrt
        Nscen=XiScenarioSet.shape[0]
        ximax_v=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="ximax_v")
        ximin_v=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="ximin_v")

        self.Rp=Rp
        self.Rn=Rn
        self.fpart=fpart
        
        ## RESERVE CONSTRAINTS
        # self.logger.debug(f"ximin: {Ximin}, ximax={Ximax}")
        self.ximinCtrt=M.addConstr(ximin_v==Ximin,"ximinctrt")
        self.ximaxCtrt=M.addConstr(ximax_v==Ximax,"ximaxctrt")
        fpartis1=M.addConstrs((fpart[:,t].sum()==1 for t in range(T)),"fpartis1") # sum of participation factors is 1 at each t
        RpandPGenLim=M.addConstrs((p_gmin_kw_mtx[:,t]<=pG[:,t]-Rn[:self.Ngen,t] for t in range(T)),"RpandPGenLim")
        RnandPGenLim=M.addConstrs((pG[:,t]+Rp[:self.Ngen,t]<=p_gmax_kw_mtx[:,t] for t in range(T)),"RnandPGenLim")
        fpartximax=M.addConstrs((fpart[:,t]*ximax_v[t]<=Rn[:,t] for t in range(T)),"fpartximax")
        fpartximin=M.addConstrs((-fpart[:,t]*ximin_v[t]<=Rp[:,t] for t in range(T)),"fpartximin")
 
        # Full costs vector Gens+BESS
        c_gen_copkwh_w_bess=self.c_gen_copkwh

        # BESS VARIABLES AND CONSTRAINTS
        if self.HAS_BESS:
            # Add BESS Vars
            pCH=M.addMVar(shape=(T),lb=0,ub=GRB.INFINITY)
            pDC=M.addMVar(shape=(T),lb=0,ub=GRB.INFINITY)
            SOE=M.addMVar(shape=(T),lb=minSOE_perc*self.CE_bess_kwh,ub=maxSOE_perc*self.CE_bess_kwh)
            
            # Ideally pDCmax=M.addConstrs((pDC[t]<=max(self.p_dc_max_bess_kw-Rp[-1,t],0)) for t in range(T)), same for pCH
            p_ch_max_clip=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            p_dc_max_clip=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _aux_ch=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _aux_dc=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY)
            _off_ch=M.addConstrs((_aux_ch[t]==self.p_ch_max_bess_kw-Rn[-1,t]) for t in range(T))
            _off_dc=M.addConstrs((_aux_dc[t]==self.p_dc_max_bess_kw-Rp[-1,t]) for t in range(T))
            _clip_ch=M.addConstrs((p_ch_max_clip[t]==gp.max_(_aux_ch[t],0) for t in range(T)))
            _clip_dc=M.addConstrs((p_dc_max_clip[t]==gp.max_(_aux_dc[t],0) for t in range(T)))
            pCHmax=M.addConstr(pCH<=p_ch_max_clip)
            pDCmax=M.addConstr(pDC<=p_dc_max_clip)            
            
            # Add BESS [cRT*pDC] TO OBJECTIVE FUNCTION
            for t in range(T):
                self.fobj+=self.c_chdc_copkwh*(pDC[t]+1/10*pCH[t]) # 1/10 is to slightly penalize charging
                # BESS RESERVE COST
                self.fobj+=reservecost_wrt_gencost*self.c_chdc_copkwh*(Rp[-1,t]+Rn[-1,t])
                        
            # Add BESS Constraints
            sumOfChDc=M.addConstrs((pCH[t]+pDC[t]<=min(self.p_ch_max_bess_kw,self.p_dc_max_bess_kw) for t in range(T)),"sumOfChDcCutPlane")
            SOEdynamics=M.addConstrs((SOE[t+1]==SOE[t]+self.deltat_h*(pCH[t+1]*self.eta_ch-pDC[t+1]/self.eta_dc) for t in range(T-1)),"SOEdynamics")
            if self.strictlycircularbess:
                SOEcircular=M.addConstr(SOE[0]==SOE[T-1]+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEcircular")
            else:    
                SOEstartcond=M.addConstr(SOE[0]==self.BESS_SOE_init*self.CE_bess_kwh+self.deltat_h*(pCH[0]*self.eta_ch-pDC[0]/self.eta_dc),"SOEstartcond")
                SOEendcond=M.addConstr(SOE[T-1]>=self.BESS_SOE_init*self.CE_bess_kwh,"SOEendcond")
                etacutplane=(self.eta_ch+1/self.eta_dc)/2
                SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range(T-1)),"SOEdynamics")
            # make var/ctrt handlers accessible as attributes
            self.pCH=pCH
            self.pDC=pDC
            self.SOE=SOE   
            GenBal+=pDC-pCH 
            c_gen_copkwh_w_bess=np.append(c_gen_copkwh_w_bess,self.c_chdc_copkwh)
    
        ## ADD RESERVE ASSIGNMENT COSTS TO OBJECTIVE FUNCTION
        self.fobj+=reservecost_wrt_gencost*c_gen_copkwh_w_bess@(Rp.sum(axis=1)+Rn.sum(axis=1))
        
        ## ADD DROW VARIABLES
        kappa=M.addVar(ub=GRB.INFINITY,name="kappa")
        recourse_norm=M.addVar(ub=GRB.INFINITY,name="recourse_norm")
        recourse_aux=M.addMVar(shape=(T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="recourse_aux")
        rwass_var=M.addVar(lb=0,ub=GRB.INFINITY,name="rwass")
        self.rwass_ctrt=M.addConstr(rwass_var==rwass,"rwass_ctrt")
        self.kappa=kappa
        

        ## ADD DROW CONSTRAINT
        #|| fpart.T @ c ||inf<=kappa
        auxdef=M.addConstr(recourse_aux==fpart.T@c_gen_copkwh_w_bess,"auxdef")
        auxnorm=M.addConstr(recourse_norm==gp.norm(recourse_aux,GRB.INFINITY),"auxnorm")
        normleqkappa=M.addConstr(recourse_norm<=kappa,"normleqkappa")
        
        # ADD COSTS TO OBJECTIVE FUNCTION
        # fobj add SUM -c fpart  xiscen_i
        WscenCost=gp.LinExpr()
        WscenCost=kappa*rwass_var-c_gen_copkwh_w_bess@fpart@self.xihat_var.sum(axis=0)/Nscen
        self.fobj+=WscenCost

        # FOR ADMM MODE, THE RESOLVE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
        if self.transactive:
            if self.neighbors==[]:
                raise Exception("transactive EDnR requires neighbors list")
            else:
                # Z AND LAMBDAS MUST BE SET
                assert z_PC is not None, f"z_PC is None"
                assert z_PV is not None, f"z_PV is None"
                assert lambdas_C is not None, f"lambdas_C is None"
                assert lambdas_V is not None, f"lambdas_V is None"
                self.z_PC_k=z_PC
                self.z_PV_k=z_PV
                self.lam_C_k=lambdas_C
                self.lam_V_k=lambdas_V
                # Add P compras and P ventas variables to model
                P_C=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_C") # P compras
                P_V=M.addMVar(shape=((self.lenneighbors,T)),lb=0,ub=np.vstack([self.lineCapacities for _ in range(T)]).T,name="P_V") # P ventas
                
                # Add ADMM params as variables == constant RHS to update in resolve()
                z_PC_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                z_PV_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas

                self.z_PC_ctrt=M.addConstr(z_PC_var==self.z_PC_k,"z_PC_const")
                self.z_PV_ctrt=M.addConstr(z_PV_var==self.z_PV_k,"z_PV_const")
                
                lam_C_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                lam_V_var=M.addMVar(shape=((self.lenneighbors,T)),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                self.lam_C_ctrt=M.addConstr(lam_C_var==self.lam_C_k,"lam_C_const")
                self.lam_V_ctrt=M.addConstr(lam_V_var==self.lam_V_k,"lam_V_const")

                for t in range(T):
                    for j in range(self.lenneighbors): #list of MG indices
                        ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                        self.fobj+=(lam_C_var[j,t]+self.lineCosts)*P_C[j,t]-lam_C_var[j,t]*P_V[j,t]
                        self.fobj+=(self.rho/2)*((P_C[j,t]-z_PC_var[j,t])*(P_C[j,t]-z_PC_var[j,t])+(P_V[j,t]-z_PV_var[j,t])*(P_V[j,t]-z_PV_var[j,t]))
                self.P_C=P_C
                self.P_V=P_V
            ### METER INTERCAMBIOS EN BALANCE DE CARGA
            GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)
                  
        ### LOAD BALANCE CONSTRAINT
        self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")

        M.Params.QCPDual=1
        M.setObjective(self.fobj, GRB.MINIMIZE)
        self.logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            self.logger.warning(f"Uhhh something happened: {e}")
            self.logger.warning(f"{self.M.NumVars} Vars, {self.M.NumNZs} Num NZs, {self.M.NumConstrs} Constraints, {self.M.NumQConstrs} QConstrts, {self.M.NumGenConstrs} GenCtrts, {self.M.NumBinVars} BinVars, {self.M.NumSOS} SOSCtrts")
            M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
            return -1
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            self.logger.info(f"ED solved optimally in {self.M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {self.M.status}")
            try:
                M.computeIIS()
                M.write(f"GRB_IIS/model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
                return -1
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
                return -1

        # Build Result Object
        result=SimpleNamespace()
        result.ObjVal=M.ObjVal
        # GETTING THE DUALS MAY BE A BIT HARDER IN DROW
        if self.needMPO:
            try:
                result.MPO=self.loadbalanceCtrt.Pi
            except gp.GurobiError as e:
                self.logger.critical(f"Couldnt get LMP/MPO as Pi - {e}, trying from fixed model")
                fixedmodel=M.fixed()
                fixedmodel.optimize()
                fixedctrs_=[fixedmodel.getConstrByName(n) for n in self.loadbalanceCtrt.ConstrName]
                result.MPO=np.array(list(map(lambda c: c.Pi, fixedctrs_)))
                assert len(result.MPO)==T and (result.MPO>0).all(),f"Invalid LMP/MPOs: {result.MPO}"

        result.Pgen=pG.X
        result.fpart=fpart.X
        result.H_p=Rp.X
        result.H_n=Rn.X
        result.ResT_p=np.sum(Rp.X,axis=0)
        result.ResT_n=np.sum(Rn.X,axis=0)
        result.kappa=kappa.X

        if self.transactive:
            result.P_C=P_C.X
            result.P_V=P_V.X
        
        if self.HAS_BESS:
            pCH_res=pCH.X
            pDC_res=pDC.X
            SOE_res=SOE.X
            result.PCH=pCH_res
            result.PDC=pDC_res
            result.PBESS=pDC_res-pCH_res
            result.SOE=SOE_res
            for t in range(T):
                if(pCH_res[t] > 1e-2 and pDC_res[t]>1e-2) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    self.logger.warning(f"t={t}")
                    self.logger.warning(f"pCH_res: {pCH_res[:t+1]}")
                    self.logger.warning(f"pDC_res: {pDC_res[:t+1]}")
                    self.logger.warning(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")

        ## CALCULATE REAL COST OF ED+R [D-1] PLANNING: ED, R, PURCHASES AND SALES
        Cost_of_EDnR=0
        for t in range(T):
            # P_ED
            Cost_of_EDnR+=np.sum(self.c_gen_copkwh*result.Pgen[:,t])
            # P^DC_ED - only pay BESS per kwh cycled aka discharged, "they" pay for the charge up
            if self.HAS_BESS:
                Cost_of_EDnR+=self.c_chdc_copkwh*pDC_res[t]
            # R+ and R-
            Cost_of_EDnR+=reservecost_wrt_gencost*np.sum(c_gen_copkwh_w_bess*(result.H_p[:,t]+result.H_n[:,t]))
            # buying and selling
            if self.transactive:
                for j in range(self.lenneighbors):
                    # if actually buying, add (LAM+lineCost) * P_C
                    if result.P_C[j,t]>1e-2:
                        Cost_of_EDnR+=(self.lam_C_k[j,t]+self.lineCosts)*result.P_C[j,t]
                    # if actually selling, sub LAM * P_V
                    if result.P_V[j,t]>1e-2:
                        Cost_of_EDnR-=self.lam_V_k[j,t]*result.P_V[j,t]        
        result.EDnRcost=Cost_of_EDnR
        return result           

# EN MODO ADMM (transactive=true at init):
# AT xEDnR() INIT se *deben* setear building params:
# rho, neighbors, linecaps, linecosts
# AT SOLVE/RESOLVE se *deben* setear z,lam iter params (sea k=0 o no):
# NO SE PUEDE RESETEAR RHO

# option 3: init ADMM to config
class ADMMexchange:
    def __init__(self,MGs:list[MG],methods:list[str],trainingsamplesets:list,ednrparams:list[dict],
                neighbors:dict[int,list[int]],linecosts:float,lineCaps:dict[tuple[int,int],float|int],
                rho:float|int,max_iters:int=1000,error_threshold=1E-6,
                P_C={}, P_V={},     
                z_PC={}, z_PV={},   
                lam_C={}, lam_V={}, 
                state_init={},
                x_0=0,z_0=0,lam_0=0,
                logger_level=logging.CRITICAL):
        """
            :param MGs:
              (list[MG]) list of MGs in the MMG. len=N
            :param methods:
              (list[str]): List 'd', 's', 'r', 'drw' to select methods used. Must be len N.
            :param trainingsamplesets:
              (list): List of TrainingSet Objects() for each MG. Must be len N.
            :param ednrparams:
              (list[dict]): List of dicts with init and solve params for each solver. Must be len N.
            :param neighbors:
              (dict[int,list[int]]): Adjacency Map of MMG, e.g. {1:[2,3],2:[4]}. Must be len N.
            :param linecosts:
              (float): Unit cost of transporting energy in $/kWh. Same for all lines.
            :param lineCaps:
              (dict[tuple[int,int],float | int]): Map of max P to transport in kW for each line. E.g. {(2,3):2000,(1,2):1000}
            :param rho:
              (float | int): ADMM parameter for convergence tuning.
            :param max_iters:
              (int, optional): Defaults to 1000.
            :param error_threshold:
              (float, optional): Defaults to 1E-6. Convergence threshold for the sum of N residuals.
            :param P_C,P_V:
              (dict, optional): Defaults to {}. Initial values for exchanges (x0).
            :param z_PC,z_PV:
              (dict, optional): Defaults to {}. Initial values for consensus variables (z0).
            :param lam_C,lam_V:
              (dict, optional): Defaults to {}. Initial values for dual variables (lambda0).
            :param state_init:
              (dict, optional): Defaults to {}. Initial state for warm start of MG solvers. Not implemented.
            :param x_0,z_0,lam_0:
              (int, optional): Defaults to 0. Initial values for (x0,z0,lambda0) to initialize uniformly.
            :param logger_level:
              (Literal, optional): Defaults to logging.CRITICAL.'


        """      

        ## INIT LOGGER
        admmlogger=logging.getLogger(__name__)
        logging.basicConfig(format="[%(lineno)2s - ADMM] %(message)s",force=True,stream=sys.stdout)
        admmlogger.setLevel(logger_level)
        self.admmlogger=admmlogger
        T=24

        ## ====== INPUT PARSING AND ASSERTS ===========
        assert len(MGs)==len(methods)==len(trainingsamplesets)==len(ednrparams)==len(neighbors), "Diff lengths"

        methods_avail={"d":detEDnR,"s":SEDnR,"r":REDnR,"drw":DRWEDnR}
        for m in methods: assert m in methods_avail, "method should be one of 'd', 's', 'r', 'drw'"
        # SELECT SOLVER CLASS
        xEDnR_callers=[methods_avail[m] for m in methods]

        # SMALL TEST RUN
        admmlogger.warning("Testing detEDnR with MGs")
        for i,mg in enumerate(MGs):
            x,j=detEDnR(mg,**ednrparams[i]).solve(**ednrparams[i])
            assert j!=-1, f"====TEST FAILED FOR MG{i}==="
        logging.basicConfig(format="[%(lineno)2s - ADMM] %(message)s",force=True,stream=sys.stdout)
        admmlogger.warning("===Test successfull. Initializing MMG===")


        ## PARSING NEIGHBORS dict-of-lists into sets per line
        agents_idx={i+1 for i in range(len(MGs))}
        assert agents_idx==set(neighbors.keys()), "neighbors keys should be 1,2,...,N_MGs"
        # sets de 2|E| aristas dirigidas y de |E| aristas no dirigidas 
        # dir pa construir init x,lams por agente por vecino
        # undir pa actualizar z una sola vez por par (i,j)
        directed_edges, undirected_edges = set(),set()
        for i in agents_idx:
            for j in neighbors.get(i, []):
                directed_edges.add((i,j))
                assert i != j, f"MG {i} cannot be neighbor to itself"
                undirected_edges.add(tuple(sorted((i,j))))
        
        ## PARSING ADMM INIT PARAMS 
        ## Inicializar x por vecino por agente
        if P_C=={} or P_V=={}:
            assert P_C==P_V, "both P_C and P_V should both be set (or not)"
            for (i,j) in directed_edges:
                P_C[(i,j)]=np.zeros(T)
                P_V[(i,j)]=np.zeros(T)
        else:
            for (i,j) in directed_edges:
                assert P_C.get((i,j)) is not None, f"P_C doesn't have {(i,j)}"
                assert P_V.get((i,j)) is not None, f"P_V doesn't have {(i,j)}"

        ## Inicializar z y lambdas por vecino por agente
        if z_PC=={} or z_PV=={} or lam_C=={} or lam_V=={}:
            assert z_PC==z_PV==lam_C==lam_V, "initial z and lambdas should all be set (or not)"
            # admmlogger.debug(f"dual if: {[(i,j) for j in neighbors.get(i) for i in agents]} ")
            # dual if doesnt work, just does cartesian product!
            for (i,j) in directed_edges:
                z_PC[(i,j)]=np.zeros(T)
                z_PV[(i,j)]=np.zeros(T)
                lam_C[(i,j)]=np.zeros(T)
                lam_V[(i,j)]=np.zeros(T)
        else:
            for (i,j) in directed_edges:
                assert z_PC.get((i,j)) is not None, f"z_PC doesn't have {(i,j)}"
                assert z_PV.get((i,j)) is not None, f"z_PV doesn't have {(i,j)}"
                assert lam_C.get((i,j)) is not None, f"lam_C doesn't have {(i,j)}"
                assert lam_V.get((i,j)) is not None, f"lam_V doesn't have {(i,j)}"

        # INITIAL STATE (e.g. WARM START PARAMS)
        state = {i: {} for i in agents_idx} if state_init=={} else state_init

        # Printing inputs
        admmlogger.debug(f"state_init has: {set(state_init.keys())}")
        admmlogger.debug(f"P_C: {P_C}")
        admmlogger.debug(f"P_V: {P_V}")
        admmlogger.debug(f"z_PC: {z_PC}")
        admmlogger.debug(f"z_PV: {z_PV}")
        admmlogger.debug(f"lam_C: {lam_C}")
        admmlogger.debug(f"lam_V: {lam_V}")

        ## SOLVER ADMM PARAMS CONFIG
        lineCapsDup={e:cap for (i,j), cap in lineCaps.items() for e in ((i,j),(j,i))} # Both directions appear
        lineCapsPerAgent={i:[lineCapsDup.get((i,j),0.0) for j in neighbors.get(i)] for i in agents_idx} # list per agentidx, caps ordered based on neighbors[i] order
        ADMMsolverinit_global={'transactive':True,'rho':rho,'lineCosts':linecosts}
        ADMMsolverinit={i:{'neighbors':neighbors[i], 'lineCapacities':lineCapsPerAgent[i]}
                        | ADMMsolverinit_global  for i in agents_idx}
        ## ====== BUILDING DICT OF AGENTS ======
        agents={}
        solvers=[]
        # Train_Sets=[]
        for i,mg in enumerate(MGs):
            agentidx=i+1
            admmlogger.info(f"Setting up solver for MG{agentidx}")
            logging.basicConfig(format=f"[%(lineno)2s - MG{agentidx} - %(funcName)2s] %(message)s",force=True,stream=sys.stdout)
            solvers.append(xEDnR_callers[i](mg,**ednrparams[i]|ADMMsolverinit[agentidx],
                                            model_name=f"MG{agentidx}_{methods[i]}"))
            # Train_Sets.append(solvers[i].generateSampleSet(Nsamples))
            agents[agentidx]={'solver':solvers[i],
                         'solverparams': {'trainSampleSet':trainingsamplesets[i]}|ednrparams[i] }

            admmlogger.info(f"mg.neighbors= {solvers[i].neighbors}")
        self.solvers=solvers
        logging.basicConfig(format="[%(lineno)2s - ADMM] %(message)s",force=True,stream=sys.stdout)

        # ====== INIT HISTORIAS ======

        # DICT OF LISTS, HARDER TO BUILD, EASIER TO PARSE
        # Residuo Primal ||lambda_k+1 - lambda_k|| per k>=2
        # Residuo Dual ||z_k+1 - z_k|| per k>=2
        primal_hist = {idx:{"linf":[], 'l2':[]} for idx in agents_idx|{'sum'}}
        dual_hist = {idx:{"linf":[], 'l2':[]} for idx in undirected_edges|{'sum'}}
        P_C_hist = {(i,j):[] for (i,j) in directed_edges}
        P_V_hist = {(i,j):[] for (i,j) in directed_edges}
        lam_C_hist = {(i,j):[] for (i,j) in directed_edges}
        lam_V_hist = {(i,j):[] for (i,j) in directed_edges}
        # IN THIS CASE LIST OF DICTS
        # statehist=[{i:{},j:{}} , {...}] guarda copia del state por agente per iter
        x_state_hist = [] 
        R_prim,R_dual={},{}

        convergenceCondition=False
        k=0
        admmlogger.warning("Initialized. Starting ADMM")

        while(k<max_iters and not convergenceCondition):
            admmlogger.warning(f"===== k: {k}=====")
            admmlogger.debug(f"Prev P_C: {P_C}")
            admmlogger.debug(f"Prev P_V: {P_V}")
            admmlogger.info("====Updating x====")

            # ---------- (1) ACTUALIZACIÓN x^k+1: usa z^k y lam^k (secuencial) ----------
            for i,config in agents.items():
                admmlogger.info(f"i: {i}")
                admmlogger.debug(f"n_i:{neighbors.get(i)}")
                # x=[(i,j) for j in neighbors.get(i)]
                # admmlogger.debug(x)
                z_PC_i = np.array([z_PC.get((i, j)) for j in neighbors.get(i)])
                z_PV_i = np.array([z_PV.get((i, j)) for j in neighbors.get(i)])
                lam_C_i = np.array([lam_C.get((i, j)) for j in neighbors.get(i)])
                lam_V_i = np.array([lam_V.get((i, j)) for j in neighbors.get(i)])
                solver=config.get('solver')
                params=config.get('solverparams') # |state[i]['param_x'] # may be useful for warm start 
                
                solver.logger.debug(f"z_PC_i: {z_PC_i}")
                solver.logger.debug(f"z_PV_i: {z_PV_i}")
                solver.logger.debug(f"lam_C_i: {lam_C_i}")
                solver.logger.debug(f"lam_V_i: {lam_V_i}")

                logging.basicConfig(format=f"[%(lineno)2s - MG{i} - %(funcName)2s] %(message)s",force=True,stream=sys.stdout)
                xdec,jis = solver.solveOrResolve(**params,z_PC=z_PC_i,z_PV=z_PV_i,lambdas_C=lam_C_i,lambdas_V=lam_V_i) 
                res={"P_C":{(i,j): xdec.P_C[neigh_idx] for neigh_idx,j in enumerate(neighbors.get(i, []))},
                    "P_V":{(i,j): xdec.P_V[neigh_idx] for neigh_idx,j in enumerate(neighbors.get(i, []))},
                    "state":{'xdec':{"Full EDnResult(...)"},'jis':jis}} # Que guardar para warm start
                # For history
                logging.basicConfig(format="[%(lineno)2s - ADMM] %(message)s",force=True,stream=sys.stdout)
                solver.logger.debug(f"res: {res}")
                # Just for the final
                res["state"]["xdec"]=xdec
                # Actualiza P_C y P_V locales del agente i
                for tup, val in res.get("P_C").items():
                    P_C[tup] = deepcopy(val)
                    # Guardar hist
                    P_C_hist[tup].append(val)
                for tup, val in res.get("P_V").items():
                    P_V[tup] = deepcopy(val)
                    # Guardar hist
                    P_V_hist[tup].append(val)

                # Warm start: guarda el nuevo estado del agente i
                state[i] = deepcopy(res.get("state"))
                del res
                
            admmlogger.debug("----x results----")
            admmlogger.debug(f"P_C: {P_C}")
            admmlogger.debug(f"P_V: {P_V}")

            x_state_hist.append( {i:deepcopy(state[i]["jis"]) for i in agents} )

            # ---------- (2) ACTUALIZACIÓN z^k+1: usa x^{k+1} y lam^k ----------
            # Guarda z^k para residuales de consenso por agente
            admmlogger.info("----Previous z----")
            admmlogger.debug(f"z_PC: {z_PC}")
            admmlogger.debug(f"z_PV: {z_PV}")
            admmlogger.info("====Updating z====")
            z_PC_prev = deepcopy(z_PC)
            z_PV_prev = deepcopy(z_PV)

            for (i, j) in undirected_edges:
                # Actualiza consensos una vez por linea
                PVi_j  = P_V[(i, j)]
                PCj_i  = P_C[(j, i)]
                PCi_j  = P_C[(i, j)]
                PVj_i  = P_V[(j, i)]
                lVi_j  = lam_V[(i, j)]
                lCj_i  = lam_C[(j, i)]
                lCi_j  = lam_C[(i, j)]
                lVj_i  = lam_V[(j, i)]

                tPV_ij = 0.5 * (PVi_j + PCj_i) + (lVi_j + lCj_i) / (2.0 * rho)
                z_PV[(i, j)] = tPV_ij
                z_PC[(j, i)] = tPV_ij

                tPC_ij = 0.5 * (PCi_j + PVj_i) + (lCi_j + lVj_i) / (2.0 * rho)
                z_PC[(i, j)] = tPC_ij
                z_PV[(j, i)] = tPC_ij

            admmlogger.debug(f"z_PC: {z_PC}")
            admmlogger.debug(f"z_PV: {z_PV}")
            # ---------- (3) ACTUALIZACIÓN lam^k+1: usa P^{k+1} y z^{k+1} ----------
            admmlogger.info("----Previous lam----")
            admmlogger.debug(f"lam_C: {lam_C}")
            admmlogger.debug(f"lam_V: {lam_V}")
            lam_C_prev = deepcopy(lam_C)
            lam_V_prev = deepcopy(lam_V)

            admmlogger.info("====Updating lambda====")
            for i in agents:
                for j in neighbors.get(i, []):
                    lam_V[(i, j)] += rho * (P_V[(i, j)] - z_PV[(i, j)])
                    lam_C[(i, j)] += rho * (P_C[(i, j)] - z_PC[(i, j)])
            admmlogger.debug(f"lam_C: {lam_C}")
            admmlogger.debug(f"lam_V: {lam_V}")
            for tup, val in lam_C.items():
                # Guardar hist
                lam_C_hist[tup].append(val)
            for tup, val in lam_V.items():
                # Guardar hist
                lam_V_hist[tup].append(val)

            # ---------- (4) RESIDUALES DE CONSENSO POR AGENTE y HISTORIA DE P ----------
            # Residuals primal y dual, de L2 y Linf de cada agente/line, y suma
            # Primal (d_lambda) es per agent, dual (d_z) es per line
            admmlogger.info("====Calculating Residuals====")
            # Prev_R_prim=deepcopy(R_prim)
            # Prev_R_dual=deepcopy(R_dual)


            # PRIMAL RES: ||lambda_k+1 - lambda_k||
            linf_t=0.0
            l2_t=0.0
            for i in agents:
                linf_i = 0.0
                l2_i   = 0.0
                for j in neighbors.get(i, []):
                    d_lam_c = lam_C[(i, j)] - lam_C_prev.get((i, j)) 
                    d_lam_v = lam_V[(i, j)] - lam_V_prev.get((i, j)) 
                    linf_i = max( linf_i , np.abs(d_lam_c).max() , np.abs(d_lam_c).max() )
                    l2_i  += (d_lam_c**2).sum() + (d_lam_v**2).sum()
                primal_hist[i]["linf"].append(linf_i)
                primal_hist[i]["l2"].append(l2_i**0.5)
                R_prim[i]={"linf":linf_i,"l2":l2_i**0.5}
                linf_t+=linf_i
                l2_t+=l2_i**0.5
            primal_hist['sum']["linf"].append(linf_t)
            primal_hist['sum']["l2"].append(l2_t)
            R_prim['sum']={"linf":linf_t,"l2":l2_t}
            admmlogger.debug(f"R_prim: {R_prim}")

            # DUAL RES: ||z_k+1 - z_k||
            linf_t=0.0
            l2_t=0.0
            for (i,j) in undirected_edges:
                d_pc = z_PC[(i, j)] - z_PC_prev.get((i, j)) 
                d_pv = z_PV[(j, i)] - z_PV_prev.get((j, i)) 
                linf_ij = max( np.abs(d_pc).max() , np.abs(d_pv).max() )
                l2_ij  = (d_lam_c**2).sum() + (d_lam_v**2).sum()
                dual_hist[(i,j)]["linf"].append(linf_ij)
                dual_hist[(i,j)]["l2"].append(l2_ij**0.5)
                R_dual[(i,j)]={"linf":linf_ij,"l2":l2_ij**0.5}
                linf_t += linf_ij
                l2_t += l2_ij**0.5
            dual_hist['sum']["linf"].append(linf_t)
            dual_hist['sum']["l2"].append(l2_t)
            R_dual['sum']={"linf":linf_t,"l2":l2_t}
            admmlogger.debug(f"R_prim: {R_dual}")

            
            # Update convergence cond
            # convergenceCondition=False # Añadir condición para e.g. norma de residuo
            convergenceCondition = (max(R_prim['sum']["l2"] , R_prim['sum']["linf"]) <= error_threshold)
            k+=1

        self.result = {
            # Estado final
            "P_C": P_C,"P_V": P_V,
            "z_PC": z_PC,"z_PV": z_PV,
            "lam_C": lam_C,"lam_V": lam_V,
            "state": state,
            # Historias
            'primal_hist':primal_hist,'dual_hist':dual_hist,
            "P_C_hist": P_C_hist,"P_V_hist": P_V_hist,
            "lam_C_hist":lam_C_hist,"lam_V_hist":lam_V_hist,
            "x_state_hist": x_state_hist,      # dict i -> lista de estados (warm-start friendly)
        }
    def get_result(self):
        return self.result