import logging
logger = logging.getLogger(__name__)
fmt_str = "[%(filename)2s:%(lineno)2s - %(funcName)2s()] %(message)s"
logging.basicConfig(format=fmt_str)
logger.setLevel(logging.DEBUG)
logger.debug("Hello world!")

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt  
from dataclasses import dataclass, field
from copy import deepcopy
import types
from IPython.display import clear_output
from distfit import distfit
import scipy.stats as stats 
import yaml
import pandas as pd
import xarray as xr
from datetime import datetime
# from scipy.special import erfinv
# from sklearn.model_selection import train_test_split
# from sklearn.utils import resample

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
    capcost_copkwh: float = field(default=1.3e6,repr=False) 
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
        if self.has_bess and self.BESS==None:
            print("please specify BESS")

class genericStochasticProgram:
    def __init__(self):
        """
        Generic program optimizing J(x,xi). Takes a training sample {Xi_in},
        makes a decision x_dec=argmin(SP), with expected in-sample cost Jis=min(SP).
        Can be tested on points xi_out belonging to a testing sample {Xi_out},
        to obtain point performance Joos_i=J(x_dec,xi_out) and average performance E[Joos]=avg(Joos_i).
        [CC testing] point performance test also returns constraint violation freq PrViol.
        """
    def Joos_i(self,x_dec,testSample):
        """
        Get out of sample cost/performance Joos_i=J(x_dec,xi_i) and PrViol of decision x_dec on a test sample point xi_i.
        On some problems point performance test can only return PrViol in {0,1},
        on ED it considers all daily subsamples so PrViol in [0,1]. 
        """
    def Joos(self,x_dec,testSampleSet):
        """
        Get AVERAGE performance of decision x_dec: out of sample cost Joos=E[J(x_dec,xi)] and PrViol, over a test sample set Xitest={xi}.
        """
        Joos_,PrViol_=zip(*[self.Joos_i(x_dec,samp) for samp in testSampleSet])
        logger.debug(f"Joos_: {Joos_}")
        logger.debug(f"PrViol_: {PrViol_}")
        Joos,PrViol=np.mean(Joos_),np.mean(PrViol_)
        return Joos,PrViol
    def solve(self,trainSampleSet,**kwargs):
        """
        Solves program with params, using training sample set. Of course, D,S,R,DR specifics may use one training sample, average, whole Set, etc.
        Returns decision x_dec, and in-sample performance Jis.     
        """
    def solveTest(self,trainSampleSet,testSampleSet,**kwargs):
        """
        Calls solve(), gets decisions x_dec from trainSampleSet, tests its performance with Joos() over testSampleSet.
        Returns (x_dec, Jis, Joos, reliability, PrViol) point [corresponding to given trainSampleSet Xi_hat]
        """
        if self.hasBeenSolved:
            x_dec,Jis=self.resolve(trainSampleSet,**kwargs)
        else:
            x_dec,Jis=self.solve(trainSampleSet,**kwargs)
        Joos,PrViol=self.Joos(x_dec,testSampleSet)
        reliability=(Joos<=Jis)
        return x_dec,Jis,Joos,reliability,PrViol
    def iter_params(self,trainSampleSet,testSampleSet):
        """
        Meant to be used by simulate(), NOT DIRECTLY.
        Generator for iterating over solveTest() while varying parameters over a given range.
        """        
        for par in self.paramRange:
            yield self.solveTest(trainSampleSet,testSampleSet,par)
    def simulate(self,paramRange,trainSampleSet,testSampleSet):
        """
        Iteratively do solveTest() with trainSampleSet and testSampleSet over a parameter range using iter_params().
        Returns x_dec, Jis, Joos, rel as reorganized independent vectors.
        """
        self.paramRange=paramRange
        # [(x,y,z) for x,y,z in iter_params()] builds result tuples for every iteration
        # zip returns a tuple iterator, which fills up output vectors iter by iter
        # so zip(*[]) basically reshapes tuples into separate same-sized vectors   
        x_dec,Jis,Joos,rel,prviol = zip(*[(_x, _ji, _jo, _r, _pv) for _x, _ji, _jo, _r, _pv in self.iter_params(trainSampleSet,testSampleSet)])
        return x_dec,Jis,Joos,rel,prviol
    def iter_traindata(self,trainSampleSets,testSampleSet):
        """
        Meant to be used by runSimulations(), NOT DIRECTLY.
        Generator for iterating over simulate() while varying the training sample set used used to solve, over a given set of datasets.
        """
        for dataset in trainSampleSets:
            yield self.simulate(self,self.paramRange,dataset,testSampleSet)
    def runSimulations(self,paramRange,trainSampleSets,testSampleSet):
        """
        Iteratively do experiments with simulate() with a set of different trainSample input sets using iter_traindata(),
        in order to obtain robust measurements of performance for the stochastic program, over a given parameter range.
        Obtains E[x_dec], E[Jis], E[Joos], Q20[Joos], Q80[Joos], E[rel] vectors, calculated over the different training experiments.
        """
        self.paramRange=paramRange
        x_dec,Jis,Joos,rel,prviol=zip(*self.iter_traindata(trainSampleSets,testSampleSet))
        self.avgJis=np.mean(Jis,axis=0)
        self.avgJoos=np.mean(Joos,axis=0)
        self.q25Joos=np.quantile(Joos,0.25,axis=0)
        self.q75Joos=np.quantile(Joos,0.75,axis=0)
        self.avgRel=np.mean(rel,axis=0)
        self.avgPrViol=np.mean(prviol,axis=0)
        ## Define avg( decision ) in Child
        # self.avgDec=np.mean(x_dec,axis=0)
        
        
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
    def __init__(self,MG,**kwargs):
        """
        Generic Economic Dispatch with Reserve Wrapper Class Init.
        
        :param quirk:
            [DateHour] string. Custom name for specific setup.
        :param grb_verbose:
            False. Show Gurobi output for ED
        :param strictlycircularbess:
            True. BESS in ED has circular dynamics (0:00=24:00). Generator UR/DR are always treated as circular.
        :param BESS_SOE_init:
            0. If BESS is not circular, starting SOE (in %)
        :param xi_hat: 
            Dummy Numpy zeros 2x2. Input training sample (Nsamples x Dimension)
        :param seed:
            None. Rng seed.
        :param subperiods:
            30. Number of Operation intrahour subperiods.
        :param plotcolors:
            plt.cm.Set1.colors. Generator pyplot palette.
        """
        self.MG=deepcopy(MG) # deepcopy to avoid modifying original MG
        self.T=24
        self.subperiods=kwargs.get("subperiods",30)
        self.plotcolors=kwargs.get("plotcolors",plt.cm.Set1.colors)
        self.rng=np.random.default_rng(kwargs.get("seed",None))
        self.grb_verbose=kwargs.get("grb_verbose",False)
        self.strictlycircularbess=kwargs.get("strictlycircularbess",True)
        self.BESS_SOE_init=kwargs.get("BESS_SOE_init",0)
        self.hasBeenSolved=False
        
        # MG Nominal Demand
        self.peak_demand_kw=self.MG.peak_demand_kw
        self.demand_curve_pu=self.MG.demand_curve_pu
        # sample generation needs to use stable (nominal) data
        self.peak_demand_kw_smpgen=deepcopy(self.MG.peak_demand_kw)
        self.demand_curve_pu_smpgen=deepcopy(self.MG.demand_curve_pu)
        
        self.demand_curve_kw=self.MG.demand_curve_kw
        if not len(self.demand_curve_pu)==self.T:
            raise Exception("MG's demand curve length is not 24")
        else:
            for g in self.MG.Gens:
                if g.intermittent==True and not len(g.gen_curve_pu)==self.T:
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
            
        self.transactive=kwargs.get("transactive",False)
        # When inititatin in TRANSACTIVE MODE
        # Instance must initiate with Neighbors, linecaps, linecosts, rho, and 0th iteration z and lambdas
        if self.transactive:
            logger.debug("Instance set for transactive EDnR")
            self.neighborhs=kwargs.get("neighbors",[])
            self.lenneighbors=len(self.neighborhs)
            if self.lenneighbors==0:
                raise Exception("transactive EDnR requires neighbors list")
            self.rho=kwargs.get("rho",0) # ADMM penalty param
            self.lineCapacities=kwargs.get("lineCapacities",[0]*self.lenneighbors) # Max line capacity with each neighbor (kW)
            self.lineCosts=kwargs.get("lineCosts",1000) # Cost of energy transacted with neighbors ($/kWh)
            self.z_PC_k=kwargs.get("z_PC_k",[0]*self.lenneighbors)
            self.z_PV_k=kwargs.get("z_PV_k",[0]*self.lenneighbors)
            self.lam_C_k=kwargs.get("lam_C_k",[0]*self.lenneighbors)
            self.lam_V_k=kwargs.get("lam_V_k",[0]*self.lenneighbors)  
        else:
            logger.debug("Instance set for non-transactive EDnR")   
            
        # Create Gurobi Model
        self.M=self.CreateModel(**kwargs)
        self.sol_time=[]
        if hasattr(kwargs,"xi_hat"):
            self.createTrainSampleSetVar(kwargs.get("xi_hat"))
        else:
            logging.warning("No training sample provided for Model. Update using updateTrainSampleSet(trainSampleSet) if needed.")
            # PDF info are init attributes    
        with open('SolarPDF.yaml','r') as f:    
            self.solarinfo=yaml.safe_load(f)

        with open('WindPDF.yaml','r') as f:    
            self.windinfo=yaml.safe_load(f)

    def CreateModel(self,**kwargs):
        '''
        Create Economic Dispatch with Reserve Base Gurobi model.
        Makes variables/constraint handlers accessible as self.attributes
        '''
        ### CREATE GUROBI MODEL
        quirk=kwargs.get("quirk",datetime.strftime(datetime.today(),"%b%d_%H%M%S"))
        M=gp.Model(f"EDnR_{quirk}")
        M.params.LogToConsole=self.grb_verbose
        logger.debug(f"Model \"{quirk}\" created")

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
        xihat_num,*xihat_dim=XiScenarioSet.shape
        self.xihat_var=self.M.addMVar(shape=(xihat_num,*xihat_dim),vtype=GRB.CONTINUOUS,lb=-GRB.INFINITY)
        self.samplectrt=self.M.addConstr(self.xihat_var==XiScenarioSet,name="samplectrt")
        self.M.update()
        logger.debug("training sample constraint built")

    def updateXiScenarioSetVar(self,XiScenarioSet):
        """
        Update training sample param in Gurobi model.
        """
        if (XiScenarioSet.shape==self.samplectrt.RHS.shape):
            self.samplectrt.RHS=XiScenarioSet
            self.M.update()
            logger.debug("training sample updated")
        else:
            self.M.remove(self.samplectrt)
            self.createXiScenarioSetVar(XiScenarioSet)  
            logger.debug("training sample dimension changed. recreated. Fobj 2ndStg costs is most likely incoherent because Nscenarios changed.")
            
    def resolve(self,tranSampleSet):
        """
        Update and resolve model with new training sample set and/or new ADMM constants. Implement in child class.
        """
        raise NotImplementedError("resolve() must be implemented in child class")
        
    # For logically plotting SOE
    @staticmethod
    def SOEvectoplot(vec):
        return np.append(np.roll(vec,1),vec[-1])
    
    ## DEMAND SAMPLE GENERATION
    def randomizeDemand(self,sigma_day=0.12,sigma_period=0.12,sigma_fast=0.15,window=12,**kwargs):
        ###### Remember, +-2sigma~95%, 3sigma~99.7%
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
    def SPVPower(self,GTI:'W/m2',PnomAtSTC_kw:'kW'):  # Po[kW]=Pstc[kW]*GTI[W/m2]/1000Wm2
        if hasattr(GTI,"__len__"):
            return [self.SPVPower(gti,PnomAtSTC_kw) for gti in GTI]
        else:
            return GTI*PnomAtSTC_kw/1000
    def SolarRand(self,PnomAtSTC_kw,window=9,subsubperiods=3,roll_hours=0.2):
        """Pout y Delta Pout con respecto al promedio.
        Con base en Puerto Carreño (Solcast).
        SÓLO IMPLEMENTADO A T=24"""
        def solarSampleCutAvg(pdf_obj,rng,params,Nsamples,min_,max_):
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
            sample_t=[solarSampleCutAvg(pdf_obj,self.rng,params,subsubperiods,min_,max_) for _ in range(self.subperiods)]
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
    def WindRand(self,N_100kw_units,window=5,subsubperiods=1):
        """Pout y DeltaPout con respecto al promedio.
        Con base en Sardinata (PDF de NASA Power
        renormalizado al promedio global de GlobalWindAtlas).
        SÓLO IMPLEMENTADO A T=24"""    
        
        def windSampleCutAvg(pdf_obj,rng,params,Nsamples,min_,max_):
            sample=np.array(pdf_obj.rvs(*params,size=Nsamples,random_state=rng))
            sample=sample[(min_<=sample)&(sample<=max_)]
            cutavg=sum(sample)/len(sample) if len(sample)>0 else min_
            return cutavg
        fastWSCurve=np.zeros(self.T*self.subperiods)
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
        xi_i=types.SimpleNamespace()
        xi_i.subperiods=self.subperiods
        xi_i.fastDemandCurve,xi_i.dayDev=self.randomizeDemand(**kwargs)
        if self.solarMPPTavailable:
            xi_i.fastPSolar=self.SolarRand(PnomAtSTC_kw=self.solar_powernom_kw_smpgen)
        if self.windMPPTavailable:
            xi_i.fastPWind=self.WindRand(N_100kw_units=self.N100kWT_smpgen)
        return xi_i
    
    def generateSampleSet(self,Nsamples,**kwargs):
        """Generate set of (day) Samples, either for training or testing."""
        return [self.generateDaySample(**kwargs) for _ in range(Nsamples)]
            
    ## RECOURSE ACTION
    # Microgrid Operation - Reserve Execution 
    def MGOperation(self,EDnRres,day_sample,/,Type3HasReconCost=True,plot_op=False,plotFastEffDemand=True,plotDeltaSOE=True):
        if not hasattr(self,"LastInstance"):
            raise Exception("Last Instance Gen name not specified")
        if not day_sample.subperiods==self.subperiods:
            raise Exception("Sample subperiods do not match")
        # Get Randomized Demand from sample
        fastDemCurve,dayDev=day_sample.fastDemandCurve,day_sample.dayDev
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
        # Get Last Instance Gen's index
        if self.LastInstance=='BESS': iLI=-1
        else:
            for iLI,gLI in enumerate(self.MG.Gens):
                if gLI.type==self.LastInstance:
                    break
            else:
                raise Exception("Wrong Last Instance Name")
        ### Execute ED+Recourse Actions
        Nsources=self.Ngen+self.HAS_BESS
        DeltaT=24/(self.T*self.subperiods) # en horas
        DeltaPG=np.zeros([Nsources,self.T])
        Nundermargin=Novermargin=0
        
        if (not hasattr(EDnRres.ResT_p,"__len__")) or len(EDnRres.ResT_p)==1:
            EDnRres.fpart=np.vstack([EDnRres.fpart for _ in range(self.T)]).T
            EDnRres.H_p=np.vstack([EDnRres.fpart for _ in range(self.T)]).T
            EDnRres.H_n=np.vstack([EDnRres.fpart for _ in range(self.T)]).T
            EDnRres.ResT_p=[EDnRres.ResT_p for _ in range(self.T)]
            EDnRres.ResT_n=[EDnRres.ResT_n for _ in range(self.T)]
        # logger.debug(EDnRres.fpart)
        # logger.debug(EDnRres.H_n)
        # logger.debug(EDnRres.ResT_p)
        for tf,dem in enumerate(fastEffDem):
            tper=tf//self.subperiods
            periodForecastedDem=self.demand_curve_kw[tper]
            DeltaPL=dem-periodForecastedDem
            if DeltaPL==0: # No Recourse
                # DeltaPG[:,tper] += np.zeros(Nsources,1)
                continue
            elif DeltaPL>0: # Positive Recourse
                if overmargin:=(DeltaPL-EDnRres.ResT_p[tper])>=0:
                    Novermargin+=1
                    # Recourse = (Holguras Totales + Overmargin) del Last Instance * DeltaT
                    DeltaPG[:,tper] += EDnRres.H_p[:,tper]*DeltaT 
                    DeltaPG[iLI,tper] += overmargin*DeltaT
                else:
                    # Recourse = fpart * DeltaPL * DeltaT
                    DeltaPG[:,tper] += EDnRres.fpart[:,tper]*DeltaPL*DeltaT 
            elif DeltaPL<0: # Negative Recourse
                if undermargin:=(-DeltaPL-EDnRres.ResT_n[tper])>=0:
                    Nundermargin+=1
                    # Recourse = (Holguras Totales + Overmargin) del Last Instance * DeltaT
                    DeltaPG[:,tper] -= EDnRres.H_n[:,tper]*DeltaT 
                    DeltaPG[iLI,tper] -= undermargin*DeltaT
                else:
                    # Recourse = fpart * DeltaPL * DeltaT
                    DeltaPG[:,tper] += EDnRres.fpart[:,tper]*DeltaPL*DeltaT
                # Recourse = (Holguras Totales + Overmargin) del Last Instance * DeltaT
        # Execute BESS Recourse actions
        ## SOE[2] = SOE(t=2:59), al final del período
        if self.HAS_BESS:
            # calculate DeltaSOE
            DeltaSOE=np.zeros(self.T)
            for period,deltaPB in enumerate(DeltaPG[-1,:]):
                # can recourse actions make BESS ch and dc within same period? sure
                # but it's rare enough and (eta_ch-eta_dc are similar enough)
                # to ignore subperiod efficiency changes
                charging_at_period = (EDnRres.PBESS[period]+deltaPB<0)
                DeltaSOE[period] = -DeltaPG[-1,period]*24/self.T*(charging_at_period*self.eta_ch + (not charging_at_period)/self.eta_dc)
            # then acumulate to see cumsum(DeltaSOE) up to 23:59
            cumDeltaSOE=np.cumsum(DeltaSOE)
            # add to EDnRres.SOE to get SOE_real
            # # (it doesn't matter if circular or not, SOE(0:00) shouldn't change)
            SOE_real=np.sum([EDnRres.SOE,cumDeltaSOE],axis=0)
        
        # Calculate cost for t for all dispatchables, including BESS
        idxdispatchable=[i for i,g in enumerate(self.MG.Gens) if g.dispatchable]
        disp_gennames=[self.MG.Gens[i].type for i in idxdispatchable]
        disp_c_gens_copkwh=[self.MG.Gens[i].rate_copkwh for i in idxdispatchable]
        if self.HAS_BESS:
            idxdispatchable.append(-1)
            disp_gennames.append('BESS')
            disp_c_gens_copkwh.append(self.c_chdc_copkwh)

        c_gens_tiled=np.tile(disp_c_gens_copkwh,(self.T,1)).T
        op_cost=c_gens_tiled*DeltaPG[idxdispatchable,:]
        total_op_cost=np.sum(op_cost)
        # If Type 2s play in recourse cost, add it to opcost
        if Type3HasReconCost:
            if self.solarMPPTavailable:
                total_op_cost+=sum(self.MG.Gens[self.idx_solarMPPT].rate_copkwh*fastDeltaPSolar)
            if self.windMPPTavailable:
                total_op_cost+=sum(self.MG.Gens[self.idx_windMPPT].rate_copkwh*fastDeltaPWind)
        # Calculate period real (effective) demand
        realEffDemand=self.demand_curve_kw+np.sum(DeltaPG,axis=0)
        
        # Plot
        if plot_op:
            periods=np.arange(0,24+1/self.T,24/self.T)
            fig,ax0=plt.subplots(figsize=(14,8))
            # Expected Demand (hourly)
            ax0.step(periods,np.append(self.demand_curve_kw,self.demand_curve_kw[-1]),where='post',lw=2,label='Demanda esperada [D-1]')
            # Effective Demand served with ED+R (hourly)
            ax0.step(periods,np.append(realEffDemand,realEffDemand[-1]),where='post',lw=2,label='Demanda atendida (ED+R) [D]')
            # Real Demand (fast)
            subpidx=np.arange(0,24*self.subperiods,24/self.T)/self.subperiods
            ax0.plot(subpidx,fastDemCurve,label=f"Demanda real (D)",alpha=0.4)
            # Real Effective Demand (fast)
            if plotFastEffDemand: ax0.plot(subpidx,fastEffDem,label=rf"Demanda efectiva (D)$",alpha=0.4)
            # Type 2 Gen
            if self.solarMPPTavailable: ax0.plot(subpidx,fastPSolar,label=rf"$P_{{spv}}$",alpha=0.4)
            if self.windMPPTavailable: ax0.plot(subpidx,fastPWind,label=rf"$P_{{wp}}$",alpha=0.4)
            ax0.grid(True, linestyle='--', alpha=0.3)
            ax0.set_xticks(periods)
            ax0.set_xlim([0,24])
            ax0.set_ylabel('Demanda (kW)')
            ax0.set_xlabel('Tiempo (h)')
            ax0.set_ylim([ax0.get_ylim()[0]*0.4,ax0.get_ylim()[1]*1])
            # AGC Cost
            ax1=ax0.twinx()
            ax1.stackplot(periods,np.concatenate((op_cost,np.array([op_cost[:,-1]]).T),axis=1),labels=disp_gennames,step='post',alpha=0.8,colors=self.plotcolors)
            ax1.set_ylabel('Costo por Desviaciones (COP)')
            ax1.yaxis.set_major_formatter('${x:,.0f}')
            ylim=EDnRres.EDnRcost/self.T
            ylim_vis=2
            pos=0.2
            ax1.set_ylim([-pos*2*ylim/ylim_vis,(1-pos)*2*ylim/ylim_vis])
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
                ax2.plot(periods,self.SOEvectoplot(SOE_real),lw=1,label='SOE real [D]',alpha=0.8)
                ax2.set_ylabel("SOE (kWh)")
                h,l=ax2.get_legend_handles_labels()
                h0,l0=h+h0,l+l0
                leg0=(0.14,0.97)
            fig.legend(h0,l0,loc='upper left', bbox_to_anchor=leg0, frameon=True)
            fig.legend(h1,l1,loc='upper right', bbox_to_anchor=leg1, frameon=True)
            ax1.text(0.01,0.02,f"Desv. del pico diario={dayDev-1:.2%}",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
            ax1.text(0.01,0.06,f"Recon. Total RSF=${total_op_cost:,.0f} COP",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
            ax1.text(0.81,0.02,f"#Sobremargen={Novermargin}",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
            ax1.text(0.81,0.06,f"#Submargen={Nundermargin}",transform=ax0.transAxes,fontsize=10,bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
            fig.set_dpi(500)
            plt.tight_layout()
            plt.show()

        operationResult=types.SimpleNamespace()
        operationResult.RecCost=total_op_cost
        operationResult.Nover=Novermargin
        operationResult.Nunder=Nundermargin
        return operationResult

    def Joos_i(self, x_dec, testSample,**kwargs):
        """Calls MGOperation() with one (day) test sample to get (instance) out of sample cost.
        
        **MGOp kwargs**:
            Type3HasReconCost=True,plot_op=False,plotFastEffDemand=True,plotDeltaSOE=True
        """
        attr={"EDnRcost","fpart","ResT_p","ResT_n"}
        for a in attr: 
            if not hasattr(x_dec,a): raise Exception("ED+R decision misspecified")
        if not hasattr(testSample,"fastDemandCurve"): raise Exception("test Sample misspecified")
        op=self.MGOperation(x_dec,testSample,**kwargs)
        Joos_i=op.RecCost+x_dec.EDnRcost
        ProbResViol=(op.Nover+op.Nunder)/(self.subperiods*self.T)
        return Joos_i,ProbResViol
    
    def GetExpectedFromSampleSet(self,DaySamplesSet):
        """Returns expected Demand and Type 2 (MPPT) generation (if available) from Sample.
        For use in ED (24 periods)."""
        Nsamples=len(DaySamplesSet)
        avgDaySample=types.SimpleNamespace()
        # Demand
        avgDemCurve=np.zeros(self.T)
        for samp in DaySamplesSet:
            avgDemCurve+=[sum(samp.fastDemandCurve[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
        avgDemCurve/=Nsamples
        avgDaySample.demand_curve_kw=np.round(avgDemCurve,0)
        # Solar Type2
        if self.solarMPPTavailable:
            avgPsolarCurve=np.zeros(self.T)
            for samp in DaySamplesSet:
                avgPsolarCurve+=[sum(samp.fastPSolar[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
            avgPsolarCurve/=Nsamples
            avgDaySample.Psolar_kw=np.round(avgPsolarCurve,0)
        # Wind Type2
        if self.windMPPTavailable:
            avgPwCurve=np.zeros(self.T)
            for samp in DaySamplesSet:
                avgPwCurve+=[sum(samp.fastPWind[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
            avgPwCurve/=Nsamples
            avgDaySample.Pwind_kw=np.round(avgPwCurve,0)
        return avgDaySample # E[xi]
    
    def updateNominalwithExpectFromSampleSet(self,DaySamplesSet):
        """
        Update nominal demand {peak_demand_kw,demand_curve_kw,demand_curve_pu} and Type 2 (MPPT) generation {g.gen_curve_pu} (if available) with expected from SampleSet.
        """
        prevnominal={'peak_demand_kw':self.peak_demand_kw, 'demand_curve_kw':self.demand_curve_kw,'demand_curve_pu':self.demand_curve_pu}
        if self.solarMPPTavailable: 
            g=self.MG.Gens[self.idx_solarMPPT]
            prevnominal|={'Psolar_kw':g.gen_curve_pu*g.power_kw}
        if self.windMPPTavailable:
            g=self.MG.Gens[self.idx_windMPPT]
            prevnominal|={'Pwind_kw':g.gen_curve_pu*g.power_kw}
        logger.debug(f"previous nominal Day: {prevnominal}")
            
        nominalDay=deepcopy(self.GetExpectedFromSampleSet(DaySamplesSet))
        logger.debug(f"expected, new nominal Day: {nominalDay}")
        
        # update nominal demand 
        self.peak_demand_kw=max(nominalDay.demand_curve_kw)
        self.demand_curve_kw=nominalDay.demand_curve_kw
        self.demand_curve_pu=self.demand_curve_kw/self.peak_demand_kw

        ## update solar nominal gen curve
        if self.solarMPPTavailable: 
            g=self.MG.Gens[self.idx_solarMPPT]
            g.gen_curve_pu=nominalDay.Psolar_kw/g.power_kw
        ## update wind nominal gen curve
        if self.windMPPTavailable:
            g=self.MG.Gens[self.idx_windMPPT]
            g.gen_curve_pu=nominalDay.Pwind_kw/g.power_kw
    
        # Get Xihat scenarios from sample set, E[Xi]=0

    def GetVarScenariosFromSampleSet(self,DaySamplesSet):
        """Returns Xi scenarios of effective variable generation aka negative effective load,
        with respect to nominal currently in instance
        For use in SED (24 periods)."""
        Nsamples=len(DaySamplesSet)
        self.subperiods=DaySamplesSet[0].subperiods
        # Should I use solar and wind nominal gen curves as xi or instead sample average curves?
        # avgDaySample=self.GetExpectedFromSampleSet(DaySamplesSet)
        # In theory SO already considers randomness within objective func to determine ED, so not needed
        ScenarioSet=np.zeros((Nsamples,self.T))
        for n,samp in enumerate(DaySamplesSet):
            ScenarioSet[n,:]=self.demand_curve_kw-[sum(samp.fastDemandCurve[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
            if self.solarMPPTavailable:
                g=self.MG.Gens[self.idx_solarMPPT]
                ScenarioSet[n,:]+=[sum(samp.fastPSolar[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
                ScenarioSet[n,:]-=g.gen_curve_pu*g.power_kw
            if self.windMPPTavailable:
                g=self.MG.Gens[self.idx_windMPPT]
                ScenarioSet[n,:]+=[sum(samp.fastPWind[i*self.subperiods:(i+1)*self.subperiods])/self.subperiods for i in range(self.T)]
                ScenarioSet[n,:]-=g.gen_curve_pu*g.power_kw
        logger.debug(f"ScenarioSet (first 3): {ScenarioSet[:3,:]}")
        return ScenarioSet
 
    def heuristic_reserve(self,customReserve=None,customReg=None,peakSupportedDisconn=0.5,f_hi=62,f_low=58.5,
                        reservecost_wrt_gencost=0.2,fpart_bess=0.4,Type3GenFactor=0.5,
                        Res_verbose=False,SOEmin_h=0.5,SOEmax_h=0.5,**kwargs):
        """Solve deterministic EDnR. Returns Reserve decision object
        with {fpart,ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,Rcost}.
        
        **R kwargs:**
            customReserve=None,customReg=None,peakSupportedDisconn=0.4,f_hi=62,f_low=58.5,
            reservecost_wrt_gencost=0.2,fpart_bess=0.6,Type3GenFactor=0.5,
            Res_verbose=False,SOEmin_h=0.5,SOEmax_h=0.5
        
        **ED kwargs:**
            plot_ED=False,EDplotstyle='stack',stackalpha=0.7,grb_verbose=None,BESS_SOE_init=None
            """
        
        ### 0. CRITERIO N-1 PARA RESERVA TOTAL 
        if customReserve is None:
            # Capacidad más grande de unidad de generación (N-1)
            ResupTotal=max([max(g.power_perunit_kw) for g in self.MG.Gens])
            # Carga más grande desconectable (N-1), asumiendo 40% de peak demand
            ResdownTotal=peakSupportedDisconn*self.peak_demand_kw
        elif isinstance(customReserve,dict):
            # A menos que se estipule lo contrario
            ResupTotal=customReserve['up']
            ResdownTotal=customReserve['down']
        
        # 1. Definir estatismo total del sistema
        R_MGup=(f_hi-60)/ResdownTotal #Hz/kW
        R_MGlo=(60-f_low)/ResupTotal #Hz/kW
        # 2. Elegir la constante de regulación menor (más robusto)
        R_MG_max=min(R_MGup,R_MGlo) #Hz/kW'
        if Res_verbose:
                logger.debug(f"Total Rsrv: +{ResupTotal/1000:.2f}MW, -{ResdownTotal/1000:.2f}MW")
                logger.debug(f"Freq Limits: {f_low-60}Hz, {f_hi-60}Hz, ")
                logger.debug(f"R_MG: {R_MG_max*1000:.2f}Hz/MW, {R_MG_max*self.peak_demand_kw/60:.3%} puHz/puMW")
        # 3. Definir factores de participación/constantes de regulación
        fpart=[0]*(self.Ngen+self.HAS_BESS) # % 
        R=[0]*(self.Ngen+self.HAS_BESS) # Hz/kW
        # Usar constantes Reg custom, si hay
        if customReg is not None:
            if not len(customReg)==self.Ngen+self.HAS_BESS:
                raise Exception("wrong length for customReg")
            else:
                R=customReg
                if min(R[np.nonzero(R)])<=R_MG_max:
                    raise Exception("R_MG larger than customReg elements! Change MG reserve margins")
                fpart=[R_MG_max/Ri for Ri in R]
                # Funcionalidad futura: Permitir algunos Ri custom y setear demás en consecuencia.
                # Por ahora, relying on proper Rcustom definition
        else:
            # BESS (si hay) hace la mayor parte del trabajo (f_part_bess)
            R_MG_remaining=R_MG_max #Hz/kW
            if self.HAS_BESS:
                fpart[-1]=fpart_bess #%
                R[-1]=R_MG_max/fpart_bess #Hz/kW
                R_MG_remaining=R_MG_max/(1-fpart_bess) #Hz/kW

            CapTotDispatch=sum(g.power_kw for g in self.MG.Gens if g.dispatchable)
            # Si no hay despachables intermitentes (Tipo 3), se define cte regulación [pu] igual en todos (los que no son BESS)
            r=R_MG_remaining*CapTotDispatch/60 # igual para todos, en puHz/pukW
            if Res_verbose: logger.debug(f"Equal droops r: {r:.2%} puHz/puMW")
            # Si hay intermitentes no despachables (Tipo 2 o MPPT) asignar R=100000% puHz/pukW
            rTipo3=1000 # aka 100000%
            R[:self.Ngen]=[r*60/g.power_kw if g.dispatchable else rTipo3*60/g.power_kw for g in self.MG.Gens] # Hz/kW
            fpart[:self.Ngen]=[R_MG_max/Ri for Ri in R[:self.Ngen]] # %
            if sum(g.intermittent&g.dispatchable for g in self.MG.Gens)>0:
            # Si hay despachables intermitentes (Tipo 3)
                if Res_verbose: logger.debug(f"Rescaling by {Type3GenFactor:.2f} for Type 3")
                # su fpart_i será (Type3GenFactor, e.g. 0.5)x veces menor que los Tipo 1 presentes
                fpart[:self.Ngen]=[Type3GenFactor*f if self.MG.Gens[i].intermittent else f for i,f in enumerate(fpart[:self.Ngen])]
                s=sum(f for f in fpart[:self.Ngen])# suma (de los non-bess)
                fpart[:self.Ngen]=[f*(1-fpart_bess*self.HAS_BESS)/s for f in fpart[:self.Ngen]] # renormalizar para que los fpart sumen 1
                R[:self.Ngen]=[R_MG_max/f for f in fpart[:self.Ngen]]

        # 4. Definir Holguras/Margenes de Reserva rounded a 1kW (para eliminar Holgura Tipo 2)
        H_n=np.round([ResdownTotal*f for f in fpart],0)
        H_p=np.round([ResupTotal*f for f in fpart],0)
        # Y se redefinen ResupTotal y ResdownTotal en consecuencia
        ResupTotal=np.sum(H_p)
        ResdownTotal=np.sum(H_n)
        # Mostrar resultados de asignación de RSF
        R_pu=[Ri*self.MG.Gens[i].power_kw/60 for i,Ri in enumerate(R[:self.Ngen])]
        if self.HAS_BESS: R_pu += [R[-1]*self.MG.BESS.power_kw/60]
        R_HzMw=[Ri*1000 for Ri in R]
        if Res_verbose:
            logger.debug(f"For {[g.type for g in self.MG.Gens]+self.HAS_BESS*["BESS"]}")
            logger.debug(f"fpart: [{", ".join([f"{f:.2%}" for f in fpart])}]")
            logger.debug(f"R: [{", ".join(f"{r:.2%}" for r in R_pu)}]% puHz/puMW")
            logger.debug(f"R: [{", ".join(f"{r:.2f}" for r in R_HzMw)}] Hz/MW")
            logger.debug(f"H+: {H_p}kW")
            logger.debug(f"H-: {H_n}kW")
        
        # 5. Usar holguras para definir cotas de generación
        p_gmax_kw_mtx=np.vstack([np.array(g.gen_curve_pu)*g.power_kw-H_p[i] for i,g in enumerate(self.MG.Gens)])
        p_gmin_kw_mtx=np.vstack([[H_n[i]]*self.T for i,g in enumerate(self.MG.Gens)])
        paramsForED={'p_gmax_kw_mtx':p_gmax_kw_mtx,
                'p_gmin_kw_mtx':p_gmin_kw_mtx}
        if self.HAS_BESS:
            # Y las cotas de carga y descarga de BESS
            p_dc_max_bess_kw=max(self.p_dc_max_bess_kw-H_p[-1],0)
            p_ch_max_bess_kw=max(self.p_ch_max_bess_kw-H_n[-1],0)
            # Cotas de SOE
            # Debe poder atender H+ por (SOEmin_h) horas
            minSOE_perc=min(SOEmin_h*H_p[-1]/self.CE_bess_kwh,1)
            # Debe poder absorber H- por (SOEmax_h) horas
            maxSOE_perc=max(1-SOEmax_h*H_n[-1]/self.CE_bess_kwh,0)

            if Res_verbose: logger.debug(f"{minSOE_perc:.2%} <= SOE <= {maxSOE_perc:.2%}")
            paramsForED=paramsForED|{'p_dc_max_bess_kw':p_dc_max_bess_kw,
                    'p_ch_max_bess_kw':p_ch_max_bess_kw,
                    'minSOE_perc':minSOE_perc,
                    'maxSOE_perc':maxSOE_perc}
                    ##### fpart,ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,
        if Res_verbose==2:logger.debug(f"params passed to BaseED: {paramsForED}")
        
        ReserveResult=types.SimpleNamespace()
        ReserveResult.fpart=np.array(fpart)
        ReserveResult.ResT_p=ResupTotal
        ReserveResult.ResT_n=ResdownTotal
        ReserveResult.R_HzMw=np.array(R_HzMw)
        ReserveResult.R_pu=np.array(R_pu)
        ReserveResult.H_p=H_p
        ReserveResult.H_n=H_n
        
        # 6. Calcular costo de reserva como fracción del costo de generación
        c_gen_copkwh_w_bess=self.c_gen_copkwh
        if self.HAS_BESS: c_gen_copkwh_w_bess=np.append(c_gen_copkwh_w_bess,self.c_chdc_copkwh)
        ReserveResult.Rcost=reservecost_wrt_gencost*np.sum((H_p+H_n)*c_gen_copkwh_w_bess)   
        
        return paramsForED,ReserveResult
 
    def plotED(self,EDResult,EDplotstyle='stack',stackalpha=0.7,**kwargs):
        # Graficar demanda y generación
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
            # SOE[2] es SOE(t=2:59), al final del período
            # se hace roll para mostrar el SOE acorde a la hora del punto
            ax1.plot(periods, self.SOEvectoplot(EDResult.SOE), color='gold', linestyle='solid', label='SOE @ t:00')
            ax1.set_ylabel('SOE (MWh)', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.locator_params(nbins=12,axis='y')
            ax1.set_ylim((0,self.CE_bess_kwh*1.5))
            SOE_tick_labels=ax1.get_yticklabels()
            SOE_tick_labels = [f"{l.get_text()} - {float(l.get_text())/self.CE_bess_kwh:.0%}" for i,l in enumerate(SOE_tick_labels)]
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
        ax3.spines['right'].set_position(('axes', 1.1))  # Mueve el eje un poco más a la derecha
        ax3.step(periods, np.append(cop_hr,cop_hr[-1]), where='post', color='red', linestyle=':', label='Costo horario')
        ax3.set_ylabel('Costo horario ($COP/h)')
        ax3.set_ylim((0,np.round(max(cop_hr)*1.5/2.5e5,0)*2.5e5))
        ax3.tick_params(axis='y', labelcolor='black')
        ax3.yaxis.set_major_formatter('${x:,.0f}')

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax3.get_legend_handles_labels()
        fig.legend(h1 + h2, l1 + l2, loc='upper right',
                bbox_to_anchor=(0.83, 0.96),  # coordenadas dentro del gráfico
                frameon=True)

        ax0.text(0.02, 0.08,f"Promedio: ${cop_kwh_avg:,.0f} COP/kWh",transform=ax0.transAxes,fontsize=10,color='black',bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
        ax0.text(0.02, 0.04,f"Costo Diario: ${EDResult.EDnRcost:,.0f} COP",transform=ax0.transAxes,fontsize=10,color='black',bbox=dict(facecolor='lightgray', alpha=0.8, edgecolor='none'))
        
        # plt.title("Despacho acumulado con costo/demanda")
        fig.set_dpi(500)
        plt.tight_layout()
        plt.show()
            
    
class detEDnR(EDnR):
        def __init__(self, MG, **kwargs):
            super().__init__(MG, **kwargs)

        def solve(self,TrainSampleSet,plot_ED=False,**kwargs):
            """Solve deterministic EDnR. Does heuristic_reserve() then BaseDetED(), but
            takes an input training sample set to calculate avg/exp day, to be used as nominal day,
            **updating instance parameters (demand and Type 3 generation).**
            Returns decision x_dec=EDnRresult object with {Pgen[GxT],PBESS[T],SOE[T],fpart,
            ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
                     
            **R kwargs:**
                customReserve=None,customReg=None,peakSupportedDisconn=0.4,f_hi=62,f_low=58.5,
                reservecost_wrt_gencost=0.2,fpart_bess=0.6,Type3GenFactor=0.5,
                Res_verbose=False,SOEmin_h=0.5,SOEmax_h=0.5
            
            **ED kwargs:**
                plot_ED=False,EDplotstyle='stack',stackalpha=0.7,grb_verbose=None,BESS_SOE_init=None
            """

            # Update nominal day with expected from sample set
            self.updateNominalwithExpectFromSampleSet(TrainSampleSet)
            
            # Realizar heurística de reserva y obtener cotas de generación
            EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
            # Save for resolve()
            self.ReserveResult=ReserveResult
            
            # Realizar ED con cotas definidas y obtener costos (D-1) de Despacho y Reserva
            EDResult=self.detED(params=EDparams,**kwargs)

            # Combinar resultados de ED + R
            x_dec=types.SimpleNamespace(**EDResult.__dict__,**ReserveResult.__dict__)                 
            x_dec.EDnRcost=x_dec.EDcost+x_dec.Rcost
            if plot_ED:
                self.plotED(x_dec,**kwargs)   
            J_is=x_dec.EDnRcost
            return x_dec,J_is
        
        def resolve(self,trainSampleSet=None,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,**kwargs):
            if self.hasBeenSolved==False:
                raise Exception("Instance must be solved once with solve() before resolve() can be called") 
         
            # New average day from sample set
            if trainSampleSet is not None:
                logger.debug("updating nominal rnwgen,demand with trainSampleSet passed to resolve()")
                self.updateNominalwithExpectFromSampleSet(trainSampleSet)
       
                # Reset Pmax constraint with new nominal gen
                self.pmaxCtrt.RHS=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
                
                # Reset balance constraint RHS with new nominal demand curve
                self.loadbalanceCtrt.RHS=self.demand_curve_kw 
       
            if lambdas_C is not None and lambdas_V is not None and z_PC is not None and z_PV is not None:
                logger.debug("updating z and lambda values (RHS) with ADMM parameters passed to resolve()")
                if not self.transactive:
                    raise Exception("Instance not set for transactive EDnR, cannot pass ADMM parameters")
                if self.neighborhs==[]:
                    raise Exception("Instance has no neighbors, cannot pass ADMM parameters")
                if not (len(lambdas_C)==self.lenneighbors and len(lambdas_V)==self.lenneighbors and len(z_PC)==self.lenneighbors and len(z_PV)==self.lenneighbors):
                    raise Exception("ADMM parameter lists must have same length as number of neighbors")
                
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
            logger.debug("solving...")
            self.M.optimize()
            self.sol_time.append(self.M.Runtime)
            if self.M.status==GRB.OPTIMAL:
                logger.debug(f"ED solved optimally in {self.M.Runtime:.2f} seconds")
            else:
                logging.warning(f"ED not solved optimally. Status: {self.M.status}")
                try:
                    self.M.computeIIS()
                    self.M.write(f"model_{self.M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                    logging.warning("Irreducible Inconsistent Subsystem written to model.ilp")
                except gp.GurobiError as e:
                    logging.error(f"Error reported when computing IIS: {e}")
            # Get Result Data
            pG_res=self.pG.X
            # Build Result Object
            detEDresult=types.SimpleNamespace()
            # Pgen[GxT],PBESS[T],SOE[T],EDcost
            detEDresult.Pgen=pG_res
            detEDresult.EDcost=self.M.ObjVal # == np.sum(cop_hr)
            
            if self.HAS_BESS:
                pCH_res=self.pCH.X
                pDC_res=self.pDC.X
                SOE_res=self.SOE.X
                detEDresult.PDC=pCH_res
                detEDresult.PCH=pDC_res
                detEDresult.PBESS=pDC_res-pCH_res
                detEDresult.SOE=SOE_res
                for t in range(self.T):
                    if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                        logger.debug(f"t={t}")
                        logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                        logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                        logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                        raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
                
            # Combine ED + R results, return new solution
            x_dec=types.SimpleNamespace(**detEDresult.__dict__,**self.ReserveResult.__dict__)   
            x_dec.EDnRcost=x_dec.EDcost+x_dec.Rcost
            if plot_ED:
                self.plotED(x_dec,**kwargs)   
            J_is=x_dec.EDnRcost
            return x_dec,J_is #x_dec,Jis
        
        def detED(self,params={},grb_verbose=None,BESS_SOE_init=None,**kwargs):
            """Meant to be called after heuristic_reserve(). Solves deterministic ED with params (can be {}) and
            instance creation parameters. Returns object with {Pgen[GxT],PBESS[T],SOE[T],EDcost}."""
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
                    SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range[T-1]),"SOEdynamics")
                # make var/ctrt handlers accessible as attributes
                self.pCH=pCH
                self.pDC=pDC
                self.SOE=SOE   
                GenBal+=pDC-pCH 
                          
            # FOR ADMM MODE, THE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
            if self.transactive:
                if self.neighborhs==[]:
                    raise Exception("transactive EDnR requires neighbors list")
                else:
                    # Add P compras and P ventas variables to model
                    P_C=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_C") # P compras
                    P_V=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_V") # P ventas
                    
                    # Add ADMM params as variables == constant RHS to update in resolve()
                    z_PC=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                    z_PV=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas
                    self.z_PC_ctrt=M.addConstr(z_PC==self.z_PC_k,"z_PC_const")
                    self.z_PV_ctrt=M.addConstr(z_PV==self.z_PV_k,"z_PV_const")
                    
                    lam_C=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                    lam_V=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                    self.lam_C_ctrt=M.addConstr(lam_C==self.lam_C_k,"lam_C_const")
                    self.lam_V_ctrt=M.addConstr(lam_V==self.lam_V_k,"lam_V_const")
                    
                    for j in range(self.lenneighbors): #list of MG indices
                        ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                        self.fobj+=(lam_C[j]+self.lineCosts)*P_C[j]-lam_V[j]*P_V[j]+(self.rho/2)*((P_C[j]-z_PC[j])@(P_C[j]-z_PC[j])+(P_V[j]-z_PV[j])@(P_V[j]-z_PV[j]))
                ### METER INTERCAMBIOS EN BALANCE DE CARGA
                GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)
            
            ### LOAD BALANCE CONSTRAINT
            self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")
                     
            M.setObjective(self.fobj, GRB.MINIMIZE)
            logger.debug("solving...")
            M.optimize()
            self.hasBeenSolved=True
            self.sol_time.append(M.Runtime)
            if M.status==GRB.OPTIMAL:
                logger.debug(f"ED solved optimally in {M.Runtime:.2f} seconds")
            else:
                logging.warning(f"ED not solved optimally. Status: {M.status}")
                try:
                    M.computeIIS()
                    M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                    logging.warning("Irreducible Inconsistent Subsystem written to model.ilp")
                except gp.GurobiError as e:
                    logging.error(f"Error reported when computing IIS: {e}")
            # Get Result Data
            pG_res=pG.X
            # Build Result Object
            detEDresult=types.SimpleNamespace()
            # Pgen[GxT],PBESS[T],SOE[T],EDcost
            detEDresult.Pgen=pG_res
            detEDresult.EDcost=M.ObjVal # == np.sum(cop_hr)
            
            if self.HAS_BESS:
                pCH_res=pCH.X
                pDC_res=pDC.X
                SOE_res=SOE.X
                detEDresult.PDC=pCH_res
                detEDresult.PCH=pDC_res
                detEDresult.PBESS=pDC_res-pCH_res
                detEDresult.SOE=SOE_res
                for t in range(T):
                    if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                        logger.debug(f"t={t}")
                        logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                        logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                        logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                        raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
            
            return detEDresult
        
            
#ED+R decision taken with SAA 
class SEDnR(EDnR):
    def __init__(self, MG, **kwargs):
        super().__init__(MG, **kwargs)
        
    def solve(self,TrainSampleSet,plot_ED=False,**kwargs):
        """Solve Stochastic EDnR. Returns decision x_dec=EDnRresult object with {Pgen[GxT],PBESS[T],SOE[T],fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
                    
        **R kwargs:**
            customReserve=None,customReg=None,peakSupportedDisconn=0.4,f_hi=62,f_low=58.5,
            reservecost_wrt_gencost=0.2,fpart_bess=0.6,Type3GenFactor=0.5,
            Res_verbose=False,SOEmin_h=0.5,SOEmax_h=0.5
        
        **ED kwargs:**
            plot_ED=False,EDplotstyle='stack',stackalpha=0.7,grb_verbose=None,BESS_SOE_init=None
        """ 
        # Update nominal day with expected from sample set
        self.updateNominalwithExpectFromSampleSet(TrainSampleSet)
                
        # Se obtienen las reservas nominales a partir de la heuristica determinista
        # Estas se usarán como RESERVAS MAXIMAS en el ED estocástico
        EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
        SEDnRparams={'minSOE_perc':EDparams['minSOE_perc'],
                    'maxSOE_perc':EDparams['maxSOE_perc'],'Hp_max':ReserveResult.H_p,
                    'Hn_max':ReserveResult.H_n}    

        # Toma el conjunto muestral y lo convierte en variaciones a partir del nominal
        XiScenarioSet=self.GetVarScenariosFromSampleSet(TrainSampleSet) # Scenarios of effective variable generation or negative effective load

        # Llama solve_stochastic con las reservas maximas y los escenarios de variacion
        x_dec=self.solve_stochastic(XiScenarioSet,SEDnRparams,**kwargs) ## ED AND R RESULTS
        if plot_ED:
            self.plotED(x_dec,**kwargs)  
        # convierte el resultado Jis
        J_is=x_dec.EDnRcost
        return x_dec,J_is
    
    def resolve(self,trainSampleSet=None,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,**kwargs):
        if self.hasBeenSolved==False:
            raise Exception("Instance must be solved once with solve() before resolve() can be called") 
        
        # New average day from sample set
        if trainSampleSet is not None:
            logger.debug("updating nominal rnwgen,demand with trainSampleSet passed to resolve()")
            self.updateNominalwithExpectFromSampleSet(trainSampleSet)
    
            # Reset Pmax constraint with new nominal gen
            self.pmaxCtrt.RHS=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
            
            # Reset balance constraint RHS with new nominal demand curve
            self.loadbalanceCtrt.RHS=self.demand_curve_kw 

            # if new sample set passed
            # Update SAMPLE CTRT RHS
            XiScenarioSet=self.GetVarScenariosFromSampleSet(trainSampleSet)
            self.updateXiScenarioSetVar(XiScenarioSet)
        
            # Update MAX MIN xi ctrt RHS
            Ximax=np.percentile(XiScenarioSet,float(80),axis=0)
            Ximin=np.percentile(XiScenarioSet,float(20),axis=0)
            self.ximinCtrt.RHS=Ximax
            self.ximaxCtrt.RHS=Ximin
            
        if lambdas_C is not None and lambdas_V is not None and z_PC is not None and z_PV is not None:
            logger.debug("updating z and lambda values (RHS) with ADMM parameters passed to resolve()")
            if not self.transactive:
                raise Exception("Instance not set for transactive EDnR, cannot pass ADMM parameters")
            if self.neighborhs==[]:
                raise Exception("Instance has no neighbors, cannot pass ADMM parameters")
            if not (len(lambdas_C)==self.lenneighbors and len(lambdas_V)==self.lenneighbors and len(z_PC)==self.lenneighbors and len(z_PV)==self.lenneighbors):
                raise Exception("ADMM parameter lists must have same length as number of neighbors")
            
            self.z_PC_k=z_PC
            self.z_PV_k=z_PV
            self.lam_C_k=lambdas_C
            self.lam_V_k=lambdas_V
            # Update ADMM parameters RHSs
            self.z_PC_ctrt.RHS=self.z_PC_k
            self.z_PV_ctrt.RHS=self.z_PV_k
            self.lam_C_ctrt.RHS=self.lam_C_k
            self.lam_V_ctrt.RHS=self.lam_V_k
            
        M=self.M
        logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            logger.debug(f"Uhhh something happened: {e}")
            logger.debug(f"{M.NumVars} Vars, {M.NumNZs} Num NZs, {M.NumConstrs} Constraints, {M.NumQConstrs} QConstrts, {M.NumGenConstrs} GenCtrts, {M.NumBinVars} BinVars, {M.NumSOS} SOSCtrts")
            M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            logger.debug(f"ED solved optimally in {M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {M.status}")
            try:
                M.computeIIS()
                M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
        # Get Result Data
        pG_res=self.pG.X
        # Build Result Object
        SEDnRresult=types.SimpleNamespace()
        # Pgen[GxT],PBESS[T],SOE[T],EDcost
        SEDnRresult.Pgen=pG_res
        SEDnRresult.EDnRcost=M.ObjVal
        SEDnRresult.fpart=self.fpart.X
        SEDnRresult.H_p=self.Rp.X
        SEDnRresult.H_n=self.Rn.X
        SEDnRresult.ResT_p=np.sum(self.Rp.X,axis=0)
        SEDnRresult.ResT_n=np.sum(self.Rn.X,axis=0)
        
        if self.HAS_BESS:
            pCH_res=self.pCH.X
            pDC_res=self.pDC.X
            SOE_res=self.SOE.X
            SEDnRresult.PDC=pCH_res
            SEDnRresult.PCH=pDC_res
            SEDnRresult.PBESS=pDC_res-pCH_res
            SEDnRresult.SOE=SOE_res
            for t in range(self.T):
                if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    logger.debug(f"t={t}")
                    logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                    logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                    logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
        if plot_ED:
            self.plotED(SEDnRresult,**kwargs) 

        # EDnR results, new solution
        x_dec=SEDnRresult
        J_is=x_dec.EDnRcost
        return x_dec,J_is #x_dec,Jis
        
    def solve_stochastic(self,XiScenarioSet,params={},reservecost_wrt_gencost=0.2,grb_verbose=None,BESS_SOE_init=None,**kwargs):
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
        # Ximax=np.max(XiScenarioSet,axis=0)
        # Ximin=np.min(XiScenarioSet,axis=0)
        Ximax=np.percentile(XiScenarioSet,float(80),axis=0)
        Ximin=np.percentile(XiScenarioSet,float(20),axis=0)
        
        self.ximinCtrt=M.addConstr(ximin_v==Ximin,"ximinctrt")
        self.ximaxCtrt=M.addConstr(ximax_v==Ximax,"ximaxctrt")
        fpartis1=M.addConstrs((fpart[:,t].sum()==1 for t in range(T)),"fpartis1") # sum of participation factors is 1 at each t
        RpandPGenLim=M.addConstrs((p_gmin_kw_mtx[:,t]<=pG[:,t]-Rn[:self.Ngen,t] for t in range(T)),"RpandPGenLim")
        RnandPGenLim=M.addConstrs((pG[:,t]+Rp[:self.Ngen,t]<=p_gmax_kw_mtx[:,t] for t in range(T)),"RnandPGenLim")
        fpartximax=M.addConstr(-Rn+fpart*Ximax<=np.zeros(((self.Ngen+self.HAS_BESS),T)),"fpartximax")
        fpartximin=M.addConstr(Rp+fpart*Ximin>=np.zeros(((self.Ngen+self.HAS_BESS),T)),"fpartximin")
        
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
                SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range[T-1]),"SOEdynamics")
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
    
        # FOR ADMM MODE, THE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
        if self.transactive:
            if self.neighborhs==[]:
                raise Exception("transactive EDnR requires neighbors list")
            else:
                # Add P compras and P ventas variables to model
                P_C=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_C") # P compras
                P_V=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_V") # P ventas
                
                # Add ADMM params as variables == constant RHS to update in resolve()
                z_PC=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                z_PV=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas
                self.z_PC_ctrt=M.addConstr(z_PC==self.z_PC_k,"z_PC_const")
                self.z_PV_ctrt=M.addConstr(z_PV==self.z_PV_k,"z_PV_const")
                
                lam_C=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                lam_V=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                self.lam_C_ctrt=M.addConstr(lam_C==self.lam_C_k,"lam_C_const")
                self.lam_V_ctrt=M.addConstr(lam_V==self.lam_V_k,"lam_V_const")
                
                for j in range(self.lenneighbors): #list of MG indices
                    ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                    self.fobj+=(lam_C[j]+self.lineCosts)*P_C[j]-lam_V[j]*P_V[j]+(self.rho/2)*((P_C[j]-z_PC[j])@(P_C[j]-z_PC[j])+(P_V[j]-z_PV[j])@(P_V[j]-z_PV[j]))
            ### METER INTERCAMBIOS EN BALANCE DE CARGA
            GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)
        
        ### LOAD BALANCE CONSTRAINT
        self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")
                     
        M.setObjective(self.fobj, GRB.MINIMIZE)
        logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            logger.debug(f"Uhhh something happened: {e}")
            logger.debug(f"{M.NumVars} Vars, {M.NumNZs} Num NZs, {M.NumConstrs} Constraints, {M.NumQConstrs} QConstrts, {M.NumGenConstrs} GenCtrts, {M.NumBinVars} BinVars, {M.NumSOS} SOSCtrts")
            M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            logger.debug(f"ED solved optimally in {M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {M.status}")
            try:
                M.computeIIS()
                M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
        # Get Result Data
        pG_res=pG.X
        # Build Result Object
        SEDnRresult=types.SimpleNamespace()
        # Pgen[GxT],PBESS[T],SOE[T],EDcost
        SEDnRresult.Pgen=pG_res
        SEDnRresult.EDnRcost=M.ObjVal
        SEDnRresult.fpart=fpart.X
        SEDnRresult.H_p=Rp.X
        SEDnRresult.H_n=Rn.X
        SEDnRresult.ResT_p=np.sum(Rp.X,axis=0)
        SEDnRresult.ResT_n=np.sum(Rn.X,axis=0)
        
        if self.HAS_BESS:
            pCH_res=pCH.X
            pDC_res=pDC.X
            SOE_res=SOE.X
            SEDnRresult.PDC=pCH_res
            SEDnRresult.PCH=pDC_res
            SEDnRresult.PBESS=pDC_res-pCH_res
            SEDnRresult.SOE=SOE_res
            for t in range(T):
                if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    logger.debug(f"t={t}")
                    logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                    logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                    logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
            
        return SEDnRresult
    
#ED+R decision taken with P(Q) from xihat itself, no assuming distribution
class REDnR(EDnR):
    def __init__(self, MG, **kwargs):
        super().__init__(MG, **kwargs)
    def solve(self,TrainSampleSet,Q=95,plot_ED=False,**kwargs):
        """
        Solve Robust EDnR. Returns decision x_dec=EDnRresult object with {Pgen[GxT],PBESS[T],SOE[T],fpart,
        ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
        """
        # Update nominal day with expected from sample set
        self.updateNominalwithExpectFromSampleSet(TrainSampleSet)
                
        # Se obtienen las reservas nominales a partir de la heuristica determinista
        # Estas se usarán como RESERVAS MAXIMAS en el ED estocástico
        EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
        REDnRparams={'minSOE_perc':EDparams['minSOE_perc'],
                    'maxSOE_perc':EDparams['maxSOE_perc'],'Hp_max':ReserveResult.H_p,
                    'Hn_max':ReserveResult.H_n}    

        # Toma el conjunto muestral y lo convierte en variaciones a partir del nominal
        XiScenarioSet=self.GetVarScenariosFromSampleSet(TrainSampleSet) # Scenarios of effective variable generation or negative effective load

        # Se calculan los escenarios de cuantiles máximos/mínimos de variación **por cada período**
        Ximax=np.percentile(XiScenarioSet,Q,axis=0)
        Ximin=np.percentile(XiScenarioSet,Q,axis=0)
        
        # Llama solve_robust con las reservas maximas y los escenarios de variacion extremos
        x_dec=self.solve_robust(Ximax,Ximin,REDnRparams,**kwargs) ## ED AND R RESULTS
        if plot_ED:
            self.plotED(x_dec,**kwargs)  
        # convierte el resultado Jis
        J_is=x_dec.EDnRcost
        return x_dec,J_is     
       
    def resolve(self,trainSampleSet=None,Q=95,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,**kwargs):
        if self.hasBeenSolved==False:
            raise Exception("Instance must be solved once with solve() before resolve() can be called") 
        
        # New average day from sample set
        if trainSampleSet is not None:
            logger.debug("updating nominal rnwgen,demand with trainSampleSet passed to resolve()")
            self.updateNominalwithExpectFromSampleSet(trainSampleSet)
    
            # Reset Pmax constraint with new nominal gen
            self.pmaxCtrt.RHS=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
            
            # Reset balance constraint RHS with new nominal demand curve
            self.loadbalanceCtrt.RHS=self.demand_curve_kw 

            # if new sample set passed
            # Update MAX MIN xi ctrt RHS
            XiScenarioSet=self.GetVarScenariosFromSampleSet(trainSampleSet)
            Ximax=np.percentile(XiScenarioSet,float(Q),axis=0)
            Ximin=np.percentile(XiScenarioSet,float(100-Q),axis=0)
            self.ximinCtrt.RHS=Ximax
            self.ximaxCtrt.RHS=Ximin
            
        if lambdas_C is not None and lambdas_V is not None and z_PC is not None and z_PV is not None:
            logger.debug("updating z and lambda values (RHS) with ADMM parameters passed to resolve()")
            if not self.transactive:
                raise Exception("Instance not set for transactive EDnR, cannot pass ADMM parameters")
            if self.neighborhs==[]:
                raise Exception("Instance has no neighbors, cannot pass ADMM parameters")
            if not (len(lambdas_C)==self.lenneighbors and len(lambdas_V)==self.lenneighbors and len(z_PC)==self.lenneighbors and len(z_PV)==self.lenneighbors):
                raise Exception("ADMM parameter lists must have same length as number of neighbors")
            
            self.z_PC_k=z_PC
            self.z_PV_k=z_PV
            self.lam_C_k=lambdas_C
            self.lam_V_k=lambdas_V
            # Update ADMM parameters RHSs
            self.z_PC_ctrt.RHS=self.z_PC_k
            self.z_PV_ctrt.RHS=self.z_PV_k
            self.lam_C_ctrt.RHS=self.lam_C_k
            self.lam_V_ctrt.RHS=self.lam_V_k
            
        M=self.M
        logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            logger.debug(f"Uhhh something happened: {e}")
            logger.debug(f"{M.NumVars} Vars, {M.NumNZs} Num NZs, {M.NumConstrs} Constraints, {M.NumQConstrs} QConstrts, {M.NumGenConstrs} GenCtrts, {M.NumBinVars} BinVars, {M.NumSOS} SOSCtrts")
            M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            logger.debug(f"ED solved optimally in {M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {M.status}")
            try:
                M.computeIIS()
                M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
        pG_res=self.pG.X
        # Build Result Object
        REDnRresult=types.SimpleNamespace()
        # Pgen[GxT],PBESS[T],SOE[T],EDcost
        REDnRresult.Pgen=pG_res
        REDnRresult.EDnRcost=M.ObjVal
        REDnRresult.fpart=self.fpart.X
        REDnRresult.H_p=self.Rp.X
        REDnRresult.H_n=self.Rn.X
        REDnRresult.ResT_p=np.sum(self.Rp.X,axis=0)
        REDnRresult.ResT_n=np.sum(self.Rn.X,axis=0)
        
        if self.HAS_BESS:
            pCH_res=self.pCH.X
            pDC_res=self.pDC.X
            SOE_res=self.SOE.X
            REDnRresult.PDC=pCH_res
            REDnRresult.PCH=pDC_res
            REDnRresult.PBESS=pDC_res-pCH_res
            REDnRresult.SOE=SOE_res
            for t in range(self.T):
                if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    logger.debug(f"t={t}")
                    logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                    logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                    logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
        if plot_ED:
            self.plotED(REDnRresult,**kwargs) 

        # EDnR results, new solution
        x_dec=REDnRresult
        J_is=x_dec.EDnRcost
        return x_dec,J_is #x_dec,Jis

    def solve_robust(self,Ximax,Ximin,params={},reservecost_wrt_gencost=0.2,plot_ED=False,grb_verbose=None,BESS_SOE_init=None,**kwargs):
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
        self.ximinCtrt=M.addConstr(ximin_v==Ximin,"ximinctrt")
        self.ximaxCtrt=M.addConstr(ximax_v==Ximax,"ximaxctrt")
        fpartis1=M.addConstrs((fpart[:,t].sum()==1 for t in range(T)),"fpartis1") # sum of participation factors is 1 at each t
        RpandPGenLim=M.addConstrs((p_gmin_kw_mtx[:,t]<=pG[:,t]-Rn[:self.Ngen,t] for t in range(T)),"RpandPGenLim")
        RnandPGenLim=M.addConstrs((pG[:,t]+Rp[:self.Ngen,t]<=p_gmax_kw_mtx[:,t] for t in range(T)),"RnandPGenLim")
        fpartximax=M.addConstr(-Rn+fpart*Ximax<=np.zeros(((self.Ngen+self.HAS_BESS),T)),"fpartximax")
        fpartximin=M.addConstr(Rp+fpart*Ximin>=np.zeros(((self.Ngen+self.HAS_BESS),T)),"fpartximin")
        
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
                SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range[T-1]),"SOEdynamics")
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
        WCcost=-c_gen_copkwh_w_bess@fpart@ximin_v
        self.fobj+=WCcost
    
        # FOR ADMM MODE, THE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
        if self.transactive:
            if self.neighborhs==[]:
                raise Exception("transactive EDnR requires neighbors list")
            else:
                # Add P compras and P ventas variables to model
                P_C=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_C") # P compras
                P_V=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_V") # P ventas
                
                # Add ADMM params as variables == constant RHS to update in resolve()
                z_PC=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                z_PV=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas
                self.z_PC_ctrt=M.addConstr(z_PC==self.z_PC_k,"z_PC_const")
                self.z_PV_ctrt=M.addConstr(z_PV==self.z_PV_k,"z_PV_const")
                
                lam_C=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                lam_V=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                self.lam_C_ctrt=M.addConstr(lam_C==self.lam_C_k,"lam_C_const")
                self.lam_V_ctrt=M.addConstr(lam_V==self.lam_V_k,"lam_V_const")
                
                for j in range(self.lenneighbors): #list of MG indices
                    ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                    self.fobj+=(lam_C[j]+self.lineCosts)*P_C[j]-lam_V[j]*P_V[j]+(self.rho/2)*((P_C[j]-z_PC[j])@(P_C[j]-z_PC[j])+(P_V[j]-z_PV[j])@(P_V[j]-z_PV[j]))
            ### METER INTERCAMBIOS EN BALANCE DE CARGA
            GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)  
                  
        ### LOAD BALANCE CONSTRAINT
        self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")
                     
        M.setObjective(self.fobj, GRB.MINIMIZE)
        logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            logger.debug(f"Uhhh something happened: {e}")
            logger.debug(f"{M.NumVars} Vars, {M.NumNZs} Num NZs, {M.NumConstrs} Constraints, {M.NumQConstrs} QConstrts, {M.NumGenConstrs} GenCtrts, {M.NumBinVars} BinVars, {M.NumSOS} SOSCtrts")
            M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            logger.debug(f"ED solved optimally in {M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {M.status}")
            try:
                M.computeIIS()
                M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
        # Get Result Data
        pG_res=pG.X
        # Build Result Object
        REDnRresult=types.SimpleNamespace()
        # Pgen[GxT],PBESS[T],SOE[T],EDcost
        REDnRresult.Pgen=pG_res
        REDnRresult.EDnRcost=M.ObjVal
        REDnRresult.fpart=fpart.X
        REDnRresult.H_p=Rp.X
        REDnRresult.H_n=Rn.X
        REDnRresult.ResT_p=np.sum(Rp.X,axis=0)
        REDnRresult.ResT_n=np.sum(Rn.X,axis=0)
        
        if self.HAS_BESS:
            pCH_res=pCH.X
            pDC_res=pDC.X
            SOE_res=SOE.X
            REDnRresult.PDC=pCH_res
            REDnRresult.PCH=pDC_res
            REDnRresult.PBESS=pDC_res-pCH_res
            REDnRresult.SOE=SOE_res
            for t in range(T):
                if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    logger.debug(f"t={t}")
                    logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                    logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                    logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
            
        return REDnRresult        
        
    
class DRWEDnR(EDnR):
    def __init__(self, MG, **kwargs):
        super().__init__(MG, **kwargs)
    def solve(self,TrainSampleSet,rwass=0.1,Q=95,plot_ED=False,**kwargs):
        """
        Solve Distributionally Robust Wasserstein EDnR. Returns decision x_dec=EDnRresult object
        with {Pgen[GxT],PBESS[T],SOE[T],fpart,ResT_p,ResT_n,R_HzMw,R_pu,H_p,H_n,EDcost,Rcost}, and in-sample performance Jis=ED+R cost [D-1].
        """
        # Update nominal day with expected from sample set
        self.updateNominalwithExpectFromSampleSet(TrainSampleSet)
                
        # Se obtienen las reservas nominales a partir de la heuristica determinista
        # Estas se usarán como RESERVAS MAXIMAS en el ED estocástico
        EDparams,ReserveResult=self.heuristic_reserve(**kwargs)
        DRWEDnRparams={'minSOE_perc':EDparams['minSOE_perc'],
                    'maxSOE_perc':EDparams['maxSOE_perc'],'Hp_max':ReserveResult.H_p,
                    'Hn_max':ReserveResult.H_n}    

        # Toma el conjunto muestral y lo convierte en variaciones a partir del nominal
        XiScenarioSet=self.GetVarScenariosFromSampleSet(TrainSampleSet) # Scenarios of effective variable generation or negative effective load

        # Se calculan los escenarios de cuantiles máximos/mínimos de variación **por cada período**
        Ximax=np.percentile(XiScenarioSet,Q,axis=0)
        Ximin=np.percentile(XiScenarioSet,Q,axis=0)
        
        # Llama solve_dro con las reservas maximas y los escenarios de variacion extremos
        x_dec=self.solve_drow(XiScenarioSet,Ximax,Ximin,rwass,DRWEDnRparams,**kwargs) ## ED AND R RESULTS
        # convierte el resultado Jis
        J_is=x_dec.EDnRcost
        if plot_ED:
            self.plotED(x_dec,**kwargs)  
        return x_dec,J_is
        
    def resolve(self,trainSampleSet=None,rwass=None,Q=95,lambdas_C=None,lambdas_V=None,z_PC=None,z_PV=None,plot_ED=False,**kwargs):
        if self.hasBeenSolved==False:
            raise Exception("Instance must be solved once with solve() before resolve() can be called") 
        
        # New average day from sample set
        if trainSampleSet is not None:
            logger.debug("updating nominal rnwgen,demand with trainSampleSet passed to resolve()")
            self.updateNominalwithExpectFromSampleSet(trainSampleSet)
    
            # Reset Pmax constraint with new nominal gen
            self.pmaxCtrt.RHS=np.vstack([np.array(g.gen_curve_pu)*g.power_kw for g in self.MG.Gens])
            
            # Reset balance constraint RHS with new nominal demand curve
            self.loadbalanceCtrt.RHS=self.demand_curve_kw 

            # if new sample set passed
            # Update SAMPLE CTRT RHS
            XiScenarioSet=self.GetVarScenariosFromSampleSet(trainSampleSet)
            self.updateXiScenarioSetVar(XiScenarioSet)    
                
            Ximax=np.percentile(XiScenarioSet,float(Q),axis=0)
            Ximin=np.percentile(XiScenarioSet,float(100-Q),axis=0)
            self.ximinCtrt.RHS=Ximax
            self.ximaxCtrt.RHS=Ximin
        
        if rwass is not None:
            self.rwass_ctrt.RHS=rwass
 
        if lambdas_C is not None and lambdas_V is not None and z_PC is not None and z_PV is not None:
            logger.debug("updating z and lambda values (RHS) with ADMM parameters passed to resolve()")
            if not self.transactive:
                raise Exception("Instance not set for transactive EDnR, cannot pass ADMM parameters")
            if self.neighborhs==[]:
                raise Exception("Instance has no neighbors, cannot pass ADMM parameters")
            if not (len(lambdas_C)==self.lenneighbors and len(lambdas_V)==self.lenneighbors and len(z_PC)==self.lenneighbors and len(z_PV)==self.lenneighbors):
                raise Exception("ADMM parameter lists must have same length as number of neighbors")
            
            self.z_PC_k=z_PC
            self.z_PV_k=z_PV
            self.lam_C_k=lambdas_C
            self.lam_V_k=lambdas_V
            # Update ADMM parameters RHSs
            self.z_PC_ctrt.RHS=self.z_PC_k
            self.z_PV_ctrt.RHS=self.z_PV_k
            self.lam_C_ctrt.RHS=self.lam_C_k
            self.lam_V_ctrt.RHS=self.lam_V_k
            
        M=self.M
        logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            logger.debug(f"Uhhh something happened: {e}")
            logger.debug(f"{M.NumVars} Vars, {M.NumNZs} Num NZs, {M.NumConstrs} Constraints, {M.NumQConstrs} QConstrts, {M.NumGenConstrs} GenCtrts, {M.NumBinVars} BinVars, {M.NumSOS} SOSCtrts")
            M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            logger.debug(f"ED solved optimally in {M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {M.status}")
            try:
                M.computeIIS()
                M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
        pG_res=self.pG.X
        # Build Result Object
        DRWEDnRresult=types.SimpleNamespace()
        # Pgen[GxT],PBESS[T],SOE[T],EDcost
        DRWEDnRresult.Pgen=pG_res
        DRWEDnRresult.EDnRcost=M.ObjVal
        DRWEDnRresult.fpart=self.fpart.X
        DRWEDnRresult.H_p=self.Rp.X
        DRWEDnRresult.H_n=self.Rn.X
        DRWEDnRresult.ResT_p=np.sum(self.Rp.X,axis=0)
        DRWEDnRresult.ResT_n=np.sum(self.Rn.X,axis=0)
        DRWEDnRresult.kappa=self.kappa.X
        
        if self.HAS_BESS:
            pCH_res=self.pCH.X
            pDC_res=self.pDC.X
            SOE_res=self.SOE.X
            DRWEDnRresult.PDC=pCH_res
            DRWEDnRresult.PCH=pDC_res
            DRWEDnRresult.PBESS=pDC_res-pCH_res
            DRWEDnRresult.SOE=SOE_res
            for t in range(self.T):
                if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    logger.debug(f"t={t}")
                    logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                    logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                    logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
        if plot_ED:
            self.plotED(DRWEDnRresult,**kwargs) 

        # EDnR results, new solution
        x_dec=DRWEDnRresult
        J_is=x_dec.EDnRcost
        return x_dec,J_is #x_dec,Jis        
        
    
    def solve_drow(self,XiScenarioSet,Ximax,Ximin,rwass,params={},reservecost_wrt_gencost=0.2,plot_ED=False,grb_verbose=None,BESS_SOE_init=None,**kwargs):
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
        self.ximinCtrt=M.addConstr(ximin_v==Ximin,"ximinctrt")
        self.ximaxCtrt=M.addConstr(ximax_v==Ximax,"ximaxctrt")
        fpartis1=M.addConstrs((fpart[:,t].sum()==1 for t in range(T)),"fpartis1") # sum of participation factors is 1 at each t
        RpandPGenLim=M.addConstrs((p_gmin_kw_mtx[:,t]<=pG[:,t]-Rn[:self.Ngen,t] for t in range(T)),"RpandPGenLim")
        RnandPGenLim=M.addConstrs((pG[:,t]+Rp[:self.Ngen,t]<=p_gmax_kw_mtx[:,t] for t in range(T)),"RnandPGenLim")
        fpartximax=M.addConstr(-Rn+fpart*Ximax<=np.zeros(((self.Ngen+self.HAS_BESS),T)),"fpartximax")
        fpartximin=M.addConstr(Rp+fpart*Ximin>=np.zeros(((self.Ngen+self.HAS_BESS),T)),"fpartximin")
 
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
                SOEcutplane=M.addConstrs((self.BESS_SOE_init+etacutplane*self.deltat_h*gp.quicksum(pCH[k]-pDC[k] for k in range(t-1))<=maxSOE_perc*self.CE_bess_kwh for t in range[T-1]),"SOEdynamics")
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

        # FOR ADMM MODE, THE WRAPPER UPDATES RHS of z_PV_ctrt, z_PC_ctrt, lam_C_ctrt, lam_V_ctrt
        if self.transactive:
            if self.neighborhs==[]:
                raise Exception("transactive EDnR requires neighbors list")
            else:
                # Add P compras and P ventas variables to model
                P_C=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_C") # P compras
                P_V=M.addMVar(shape=(self.lenneighbors,T),lb=0,ub=self.lineCapacities,name="P_V") # P ventas
                
                # Add ADMM params as variables == constant RHS to update in resolve()
                z_PC=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PC") # consensus P compras
                z_PV=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="z_PV") # consensus P ventas
                self.z_PC_ctrt=M.addConstr(z_PC==self.z_PC_k,"z_PC_const")
                self.z_PV_ctrt=M.addConstr(z_PV==self.z_PV_k,"z_PV_const")
                
                lam_C=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_C") # lambda compras
                lam_V=M.addMVar(shape=(self.lenneighbors,T),lb=-GRB.INFINITY,ub=GRB.INFINITY,name="lam_V") # lambda ventas   
                self.lam_C_ctrt=M.addConstr(lam_C==self.lam_C_k,"lam_C_const")
                self.lam_V_ctrt=M.addConstr(lam_V==self.lam_V_k,"lam_V_const")
                
                for j in range(self.lenneighbors): #list of MG indices
                    ### METER INTERCAMBIOS EN FUNCION OBJETIVO
                    self.fobj+=(lam_C[j]+self.lineCosts)*P_C[j]-lam_V[j]*P_V[j]+(self.rho/2)*((P_C[j]-z_PC[j])@(P_C[j]-z_PC[j])+(P_V[j]-z_PV[j])@(P_V[j]-z_PV[j]))
            ### METER INTERCAMBIOS EN BALANCE DE CARGA
            GenBal+=P_C.sum(axis=0)-P_V.sum(axis=0)  
                  
        ### LOAD BALANCE CONSTRAINT
        self.loadbalanceCtrt=M.addConstr(GenBal==self.demand_curve_kw,"loadbalance")
                     
        M.setObjective(self.fobj, GRB.MINIMIZE)
        logger.debug("solving...")
        try:
            M.optimize()
        except gp.GurobiError as e:
            logger.debug(f"Uhhh something happened: {e}")
            logger.debug(f"{M.NumVars} Vars, {M.NumNZs} Num NZs, {M.NumConstrs} Constraints, {M.NumQConstrs} QConstrts, {M.NumGenConstrs} GenCtrts, {M.NumBinVars} BinVars, {M.NumSOS} SOSCtrts")
            M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.lp")
        self.sol_time.append(M.Runtime)
        if M.status==GRB.OPTIMAL:
            logger.debug(f"ED solved optimally in {M.Runtime:.2f} seconds")
            self.hasBeenSolved=True
        else:
            logging.warning(f"ED not solved optimally. Status: {M.status}")
            try:
                M.computeIIS()
                M.write(f"model_{M.ModelName}_{datetime.strftime(datetime.today(),"%H%M%S")}.ilp")
                logging.warning("Irreducible Inconsistent Subsystem written to .ilp")
            except gp.GurobiError as e:
                logging.error(f"Error reported when computing IIS: {e}")
        # Get Result Data
        pG_res=pG.X
        # Build Result Object
        DRWEDnRresult=types.SimpleNamespace()
        # Pgen[GxT],PBESS[T],SOE[T],EDcost
        DRWEDnRresult.Pgen=pG_res
        DRWEDnRresult.EDnRcost=M.ObjVal
        DRWEDnRresult.fpart=fpart.X
        DRWEDnRresult.H_p=Rp.X
        DRWEDnRresult.H_n=Rn.X
        DRWEDnRresult.ResT_p=np.sum(Rp.X,axis=0)
        DRWEDnRresult.ResT_n=np.sum(Rn.X,axis=0)
        DRWEDnRresult.kappa=kappa.X
        
        if self.HAS_BESS:
            pCH_res=pCH.X
            pDC_res=pDC.X
            SOE_res=SOE.X
            DRWEDnRresult.PDC=pCH_res
            DRWEDnRresult.PCH=pDC_res
            DRWEDnRresult.PBESS=pDC_res-pCH_res
            DRWEDnRresult.SOE=SOE_res
            for t in range(T):
                if(pCH_res[t]>0 and pDC_res[t]>0) and (pCH_res[t]/pDC_res[t]>0.33 and pCH_res[t]/pDC_res[t]<3):
                    logger.debug(f"t={t}")
                    logger.debug(f"pCH_res: {pCH_res[:t+1]}")
                    logger.debug(f"pDC_res: {pDC_res[:t+1]}")
                    logger.debug(f"SOE_res: {SOE_res[:t+1]}")
                    raise Exception(f"Battery is charging and discharging at same time t={t} for some reason")
            
        return DRWEDnRresult             