import ROOT
from cmath import pi
from time import time 
import yaml
  
def timer_func(func): 
    # This function shows the execution time of  
    # the function object passed 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 

files = './test/run_*_reduced.root'
sps_cebra_settings = './spsevb_cebra/scripts/SPS_CeBrA_settings_templete.yaml'

ROOT.EnableImplicitMT()

def histo1d(df, name:str, column:str, bins:int, range:tuple[int,int], fig_list:list=None, draw=False):
    
    histo = df.Histo1D((f"{name}",f"{name}",bins, range[0], range[1]),column)
    # histo.SetTitle(f'; {column}; Counts')
    
    if fig_list is not None:
        fig_list.append(histo)
    
    if draw:
        histo.Draw()
        input("Enter to continue")
    
    return histo

def histo2d(df, name:str, x_column:str, y_column:str, bins:tuple[int,int], range:list[tuple[int,int],tuple[int,int]], fig_list:list=None, draw=False):

    histo = df.Histo2D((f"{name}",f"{name}",bins[0], range[0][0], range[0][1], bins[1], range[1][0], range[1][1]), x_column, y_column)
    # histo.SetTitle(f'; {x_column}; {y_column}')
    
    if fig_list is not None:
        fig_list.append(histo)
    
    if draw:
        histo.Draw('colz')
        input("Enter to continue")
        
    return histo

df = ROOT.RDataFrame('SPSTree',files)

@timer_func
def SPSPlots(df, Settings=None):
    
    if Settings is not None:
        
        with open(Settings, 'r') as file:
            settings = yaml.safe_load(file)
                
        SPS_Settings = settings.get('SPS',[])
        for component in SPS_Settings:
            column = component.get('ColumnName')
            ecal = [component['energy_calibration'].get('a'), component['energy_calibration'].get('b'),component['energy_calibration'].get('c')]
            df = df.Define(f'{column}_EnergyCalibration', f'{ecal[0]}*{column}*{column} + {ecal[1]}*{column} + {ecal[2]}')
                
    figures = []
    
    df = df.Define('DelayFrontAverageEnergy', '(DelayFrontRightEnergy + DelayFrontLeftEnergy)/2')\
        .Define('DelayBackAverageEnergy', '(DelayBackRightEnergy + DelayBackLeftEnergy)/2')\
            
    histo1d(df=df, name=f'X1', column=f'X1', bins=600, range=[-300,300], fig_list=figures)
    histo1d(df=df, name=f'X2', column=f'X2', bins=600, range=[-300,300], fig_list=figures)    
    histo2d(df=df, name=f'X2_X1', x_column=f'X1', y_column=f'X2', bins=[600, 600], range=[[-300,300], [-300,300]], fig_list=figures)
            
    histo2d(df=df, name=f'DelayBackRightEnergy_X1', x_column=f'X1', y_column=f'DelayBackRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayBackLeftEnergy_X1', x_column=f'X1', y_column=f'DelayBackLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontRightEnergy_X1', x_column=f'X1', y_column=f'DelayFrontRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontLeftEnergy_X1', x_column=f'X1', y_column=f'DelayFrontLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'DelayBackRightEnergy_X2', x_column=f'X2', y_column=f'DelayBackRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayBackLeftEnergy_X2', x_column=f'X2', y_column=f'DelayBackLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontRightEnergy_X2', x_column=f'X2', y_column=f'DelayFrontRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontLeftEnergy_X2', x_column=f'X2', y_column=f'DelayFrontLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'DelayBackRightEnergy_Xavg', x_column=f'Xavg', y_column=f'DelayBackRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayBackLeftEnergy_Xavg', x_column=f'Xavg', y_column=f'DelayBackLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontRightEnergy_Xavg', x_column=f'Xavg', y_column=f'DelayFrontRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontLeftEnergy_Xavg', x_column=f'Xavg', y_column=f'DelayFrontLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'DelayFrontAverageEnergy_X1', x_column=f'X1', y_column=f'DelayFrontAverageEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayBackAverageEnergy_X1', x_column=f'X1', y_column=f'DelayBackAverageEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontAverageEnergy_X2', x_column=f'X2', y_column=f'DelayFrontAverageEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayBackAverageEnergy_X2', x_column=f'X2', y_column=f'DelayBackAverageEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayFrontAverageEnergy_Xavg', x_column=f'Xavg', y_column=f'DelayFrontAverageEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'DelayBackAverageEnergy_Xavg', x_column=f'Xavg', y_column=f'DelayBackAverageEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)


    pid = histo2d(df=df, name=f'AnodeBackEnergy_ScintLeftEnergy', x_column=f'ScintLeftEnergy', y_column=f'AnodeBackEnergy', bins=[512, 512], range=[[0,4096], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'AnodeFrontEnergy_ScintLeftEnergy', x_column=f'ScintLeftEnergy', y_column=f'AnodeFrontEnergy', bins=[512, 512], range=[[0,4096], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'CathodeEnergy_ScintLeftEnergy', x_column=f'ScintLeftEnergy', y_column=f'CathodeEnergy', bins=[512, 512], range=[[0,4096], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'AnodeBackEnergy_ScintRightEnergy', x_column=f'ScintRightEnergy', y_column=f'AnodeBackEnergy', bins=[512, 512], range=[[0,4096], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'AnodeFrontEnergy_ScintRightEnergy', x_column=f'ScintRightEnergy', y_column=f'AnodeFrontEnergy', bins=[512, 512], range=[[0,4096], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'CathodeEnergy_ScintRightEnergy', x_column=f'ScintRightEnergy', y_column=f'CathodeEnergy', bins=[512, 512], range=[[0,4096], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'ScintLeftEnergy_X1', x_column=f'X1', y_column=f'ScintLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'ScintLeftEnergy_X2', x_column=f'X2', y_column=f'ScintLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'ScintLeftEnergy_Xavg', x_column=f'Xavg', y_column=f'ScintLeftEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'ScintRightEnergy_X1', x_column=f'X1', y_column=f'ScintRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'ScintRightEnergy_X2', x_column=f'X2', y_column=f'ScintRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'ScintRightEnergy_Xavg', x_column=f'Xavg', y_column=f'ScintRightEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    
    histo2d(df=df, name=f'AnodeBackEnergy_X1', x_column=f'X1', y_column=f'AnodeBackEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'AnodeBackEnergy_X2', x_column=f'X2', y_column=f'AnodeBackEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    anode_xavg = histo2d(df=df, name=f'AnodeBackEnergy_Xavg', x_column=f'Xavg', y_column=f'AnodeBackEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'AnodeFrontEnergy_X1', x_column=f'X1', y_column=f'AnodeFrontEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'AnodeFrontEnergy_X2', x_column=f'X2', y_column=f'AnodeFrontEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'AnodeFrontEnergy_Xavg', x_column=f'Xavg', y_column=f'AnodeFrontEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)

    histo2d(df=df, name=f'CathodeEnergy_X1', x_column=f'X1', y_column=f'CathodeEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'CathodeEnergy_X2', x_column=f'X2', y_column=f'CathodeEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
    histo2d(df=df, name=f'CathodeEnergy_Xavg', x_column=f'Xavg', y_column=f'CathodeEnergy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)

    '''Both planes'''
    
    df_bothplanes = df.Filter('X1 != -1e6 && X2 != -1e6')\
        .Define('DelayFrontLeftTime_AnodeFrontTime', 'DelayFrontLeftTime - AnodeFrontTime')\
        .Define('DelayFrontRightTime_AnodeFrontTime', 'DelayFrontRightTime - AnodeFrontTime')\
        .Define('DelayBackLeftTime_AnodeBackTime', 'DelayBackLeftTime - AnodeBackTime')\
        .Define('DelayBackRightTime_AnodeBackTime', 'DelayBackRightTime - AnodeBackTime')\
    
    xavg = histo1d(df=df_bothplanes, name=f'Xavg_bothplanes', column=f'Xavg', bins=600, range=[-300,300], fig_list=figures)
    x1 = histo1d(df=df_bothplanes, name=f'X1_bothplanes', column=f'X1', bins=600, range=[-300,300], fig_list=figures)
    x2 = histo1d(df=df_bothplanes, name=f'X2_bothplanes', column=f'X2', bins=600, range=[-300,300], fig_list=figures)
    theta = histo2d(df=df_bothplanes, name=f'Theta_Xavg_bothplanes', x_column=f'Xavg', y_column=f'Theta', bins=[600, 300], range=[[-300,300], [0,pi/2]], fig_list=figures)

    histo1d(df=df_bothplanes, name=f'DelayFrontLeftTime_relTo_AnodeFrontTime_bothplanes', column=f'DelayFrontLeftTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_bothplanes, name=f'DelayFrontRightTime_relTo_AnodeFrontTime_bothplanes', column=f'DelayFrontRightTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_bothplanes, name=f'DelayBackLeftTime_relTo_AnodeBackTime_bothplanes', column=f'DelayBackLeftTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_bothplanes, name=f'DelayBackRightTime_relTo_AnodeBackTime_bothplanes', column=f'DelayBackRightTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    
    '''Only 1 plane: X1'''
    df_onlyX1plane = df.Filter('X1 != -1e6 && X2 == -1e6')\
        .Define('DelayFrontLeftTime_AnodeFrontTime', 'DelayFrontLeftTime - AnodeFrontTime')\
        .Define('DelayFrontRightTime_AnodeFrontTime', 'DelayFrontRightTime - AnodeFrontTime')\
        .Define('DelayBackLeftTime_AnodeFrontTime', 'DelayBackLeftTime - AnodeFrontTime')\
        .Define('DelayBackRightTime_AnodeFrontTime', 'DelayBackRightTime - AnodeFrontTime')\
        .Define('DelayFrontLeftTime_AnodeBackTime', 'DelayFrontLeftTime - AnodeBackTime')\
        .Define('DelayFrontRightTime_AnodeBackTime', 'DelayFrontRightTime - AnodeBackTime')\
        .Define('DelayBackLeftTime_AnodeBackTime', 'DelayBackLeftTime - AnodeBackTime')\
        .Define('DelayBackRightTime_AnodeBackTime', 'DelayBackRightTime - AnodeBackTime')\
    
    x1_only1plane = histo1d(df=df_onlyX1plane, name=f'X1_only1plane', column=f'X1', bins=600, range=[-300,300], fig_list=figures)
    
    histo1d(df=df_onlyX1plane, name=f'DelayFrontLeftTime_relTo_AnodeFrontTime_noX2', column=f'DelayFrontLeftTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX1plane, name=f'DelayFrontRightTime_relTo_AnodeFrontTime_noX2', column=f'DelayFrontRightTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX1plane, name=f'DelayBackLeftTime_relTo_AnodeFrontTime_noX2', column=f'DelayBackLeftTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX1plane, name=f'DelayBackRightTime_relTo_AnodeFrontTime_noX2', column=f'DelayBackRightTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    
    histo1d(df=df_onlyX1plane, name=f'DelayFrontLeftTime_relTo_AnodeBackTime_noX2', column=f'DelayFrontLeftTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX1plane, name=f'DelayFrontRightTime_relTo_AnodeBackTime_noX2', column=f'DelayFrontRightTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX1plane, name=f'DelayBackLeftTime_relTo_AnodeBackTime_noX2', column=f'DelayBackLeftTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX1plane, name=f'DelayBackRightTime_relTo_AnodeBackTime_noX2', column=f'DelayBackRightTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)

    '''Only 1 plane: X2'''
    df_onlyX2plane = df.Filter('X1 == -1e6 && X2 != -1e6')\
        .Define('DelayFrontLeftTime_AnodeFrontTime', 'DelayFrontLeftTime - AnodeFrontTime')\
        .Define('DelayFrontRightTime_AnodeFrontTime', 'DelayFrontRightTime - AnodeFrontTime')\
        .Define('DelayBackLeftTime_AnodeFrontTime', 'DelayBackLeftTime - AnodeFrontTime')\
        .Define('DelayBackRightTime_AnodeFrontTime', 'DelayBackRightTime - AnodeFrontTime')\
        .Define('DelayFrontLeftTime_AnodeBackTime', 'DelayFrontLeftTime - AnodeBackTime')\
        .Define('DelayFrontRightTime_AnodeBackTime', 'DelayFrontRightTime - AnodeBackTime')\
        .Define('DelayBackLeftTime_AnodeBackTime', 'DelayBackLeftTime - AnodeBackTime')\
        .Define('DelayBackRightTime_AnodeBackTime', 'DelayBackRightTime - AnodeBackTime')\
    
    x2_only1plane = histo1d(df=df_onlyX2plane, name=f'X2_only1plane', column=f'X2', bins=600, range=[-300,300], fig_list=figures)
    
    histo1d(df=df_onlyX2plane, name=f'DelayFrontLeftTime_relTo_AnodeFrontTime_noX1', column=f'DelayFrontLeftTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX2plane, name=f'DelayFrontRightTime_relTo_AnodeFrontTime_noX1', column=f'DelayFrontRightTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX2plane, name=f'DelayBackLeftTime_relTo_AnodeFrontTime_noX1', column=f'DelayBackLeftTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX2plane, name=f'DelayBackRightTime_relTo_AnodeFrontTime_noX1', column=f'DelayBackRightTime_AnodeFrontTime', bins=8000, range=[-4000,4000], fig_list=figures)
    
    histo1d(df=df_onlyX2plane, name=f'DelayFrontLeftTime_relTo_AnodeBackTime_noX1', column=f'DelayFrontLeftTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX2plane, name=f'DelayFrontRightTime_relTo_AnodeBackTime_noX1', column=f'DelayFrontRightTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX2plane, name=f'DelayBackLeftTime_relTo_AnodeBackTime_noX1', column=f'DelayBackLeftTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)
    histo1d(df=df_onlyX2plane, name=f'DelayBackRightTime_relTo_AnodeBackTime_noX1', column=f'DelayBackRightTime_AnodeBackTime', bins=8000, range=[-4000,4000], fig_list=figures)

    
    '''Time relative to Back Anode'''
    
    df_TimeRelBackAnode = df.Filter('AnodeBackTime != -1e6 && ScintLeftTime != -1e6')\
        .Define('AnodeFrontTime_AnodeBackTime', 'AnodeFrontTime - AnodeBackTime')\
        .Define('AnodeBackTime_AnodeFrontTime', 'AnodeBackTime - AnodeFrontTime')\
        .Define('AnodeFrontTime_ScintLeftTime', 'AnodeFrontTime - ScintLeftTime')\
        .Define('AnodeBackTime_ScintLeftTime', 'AnodeBackTime - ScintLeftTime')\
        .Define('DelayFrontLeftTime_ScintLeftTime', 'DelayFrontLeftTime - ScintLeftTime')\
        .Define('DelayFrontRightTime_ScintLeftTime', 'DelayFrontRightTime - ScintLeftTime')\
        .Define('DelayBackLeftTime_ScintLeftTime', 'DelayBackLeftTime - ScintLeftTime')\
        .Define('DelayBackRightTime_ScintLeftTime', 'DelayBackRightTime - ScintLeftTime')\
        .Define('ScintRightTime_ScintLeftTime', 'ScintRightTime - ScintLeftTime')
        
    histo1d(df=df_TimeRelBackAnode, name=f'AnodeFrontTime_AnodeBackTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'AnodeBackTime_AnodeFrontTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'AnodeFrontTime_ScintLeftTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'AnodeBackTime_ScintLeftTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'DelayFrontLeftTime_ScintLeftTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'DelayFrontRightTime_ScintLeftTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'DelayBackLeftTime_ScintLeftTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'DelayBackRightTime_ScintLeftTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo1d(df=df_TimeRelBackAnode, name=f'ScintRightTime_ScintLeftTime', column=f'AnodeFrontTime_AnodeBackTime', bins=1000, range=[-3000,3000], fig_list=figures)
    histo2d(df=df_TimeRelBackAnode, name=f'ScintTimeDif_Xavg', x_column=f'Xavg', y_column=f'ScintRightTime_ScintLeftTime', bins=[600, 12800], range=[[-300,300],[-3200,3200]], fig_list=figures)
    
    
    '''TCanvas to show the most important plots'''
    ROOT.gROOT.SetBatch()    
    c = ROOT.TCanvas('SE-SPS Plots','SE-SPS Plots',2000,1000)
    c.Divide(3,2)

    c.cd(1)
    pid.Draw('colz')
    
    c.cd(2)
    anode_xavg.Draw('colz')
    
    c.cd(3)
    theta.Draw('colz')
    
    c.cd(4)
    xavg.Draw()
    
    c.cd(5)
    x1.Draw()
    x1_only1plane.SetLineColor(2) #red
    x1_only1plane.Draw('Same')
    
    c.cd(6)
    
    x2.Draw()
    x2_only1plane.SetLineColor(2) #red
    x2_only1plane.Draw("Same")
    
    figures.insert(0, c)
    
    return figures

@timer_func
def CeBrAPlots(df, NumDetectors, SPS=False, ADCShifts=None, TimeGates=None):
    
    figures = []
    
    CeBrAEnergy_ADCShift_figures = []
    
    Xavg_timecut_figures = []
    CeBrAEnergy_ADCShift_TimeCut_figures = []
    Xavg_CeBrAEnergy_ADCShift_TimeCut_figures = []
    X1_CeBrAEnergy_ADCShift_TimeCut_figures = []
    
    for det in range(NumDetectors):

        histo1d(df=df, name=f'Cebra{det}Energy', column=f'Cebra{det}Energy', bins=512, range=[0,4096],  fig_list=figures)
            
        if ADCShifts is not None:
            m,b = ADCShifts[det]
            df = df.Define(f'Cebra{det}Energy_ADCShift', f'{m}*Cebra{det}Energy + {b}')
            histo1d(df=df, name=f'Cebra{det}Energy_ADCShift', column=f'Cebra{det}Energy_ADCShift', bins=512, range=[0,4096],  fig_list=figures)
            
            CebraEnergy_ADCShift_i = histo1d(df=df, name=f'CeBrAEnergy_ADCShift', column=f'Cebra{det}Energy_ADCShift', bins=512, range=[0,4096])
            CeBrAEnergy_ADCShift_figures.append(CebraEnergy_ADCShift_i)
            
        if SPS:
                    
            df_cebr_time = df.Filter('AnodeBackTime != -1e6 && ScintLeftTime !=-1e6').Define(f"Cebra{det}TimeToScint", f"Cebra{det}Time - ScintLeftTime")
        
            histo1d(df=df_cebr_time, name=f'Cebra{det}TimeToScint', column=f'Cebra{det}TimeToScint', bins=6400, range=[-3200,3200], fig_list=figures)
            histo2d(df=df_cebr_time, name=f'Theta_Cebra{det}TimeToScint', x_column=f'Cebra{det}TimeToScint', y_column=f'Theta', bins=[6400, 300], range=[[-3200,3200], [0,pi/2]], fig_list=figures)
            histo2d(df=df_cebr_time, name=f'Cebra{det}Energy_Xavg', x_column=f'Xavg', y_column=f'Cebra{det}Energy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
            histo2d(df=df_cebr_time, name=f'Cebra{det}Energy_X1', x_column=f'X1', y_column=f'Cebra{det}Energy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
            histo2d(df=df_cebr_time, name=f'Cebra{det}Energy_X2', x_column=f'X2', y_column=f'Cebra{det}Energy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
            histo2d(df=df_cebr_time, name=f'Cebra{det}TimeToScint_Xavg', x_column=f'Xavg', y_column=f'Cebra{det}TimeToScint', bins=[600, 6400], range=[[-300,300], [-3200,3200]], fig_list=figures)
            
            if TimeGates is not None:
                
                TimeGate = TimeGates[det]
                df_timecut = df_cebr_time.Filter(f'Cebra{det}TimeToScint >= {TimeGate[0]} && Cebra{det}TimeToScint <= {TimeGate[1]}')
                
                histo1d(df=df_timecut, name=f'Cebra{det}TimeToScint_TimeCut', column=f'Cebra{det}TimeToScint', bins=6400, range=[-3200,3200], fig_list=figures)
                histo2d(df=df_timecut, name=f'Cebra{det}TimeToScint_Theta_TimeCut', x_column=f'Cebra{det}TimeToScint', y_column=f'Theta', bins=[6400, 300], range=[[-3200,3200], [0,pi/2]])
                histo2d(df=df_timecut, name=f'Cebra{det}Energy_Xavg_TimeCut', x_column=f'Xavg', y_column=f'Cebra{det}Energy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
                histo2d(df=df_timecut, name=f'Cebra{det}Energy_X1_TimeCut', x_column=f'X1', y_column=f'Cebra{det}Energy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
                histo2d(df=df_timecut, name=f'Cebra{det}Energy_X2_TimeCut', x_column=f'X2', y_column=f'Cebra{det}Energy', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
                histo1d(df=df_timecut, name=f'Xavg_TimeCut{det}', column=f'Xavg', bins=600, range=[-300,300], fig_list=figures)
                xavg_TimeCut_i = histo1d(df=df_timecut, name=f'Xavg_TimeCuts', column=f'Xavg', bins=600, range=[-300,300])
                Xavg_timecut_figures.append(xavg_TimeCut_i)

            if ADCShifts is not None:
                histo1d(df=df_timecut, name=f'Cebra{det}Energy_ADCShift_TimeCut', column=f'Cebra{det}Energy_ADCShift', bins=512, range=[0,4096], fig_list=figures)
                CebrEnergy_ADCShift_TimeCut_i = histo1d(df=df_timecut, name=f'CeBrAEnergy_ADCShift_TimeCut', column=f'Cebra{det}Energy_ADCShift', bins=512, range=[0,4096])
                CeBrAEnergy_ADCShift_TimeCut_figures.append(CebrEnergy_ADCShift_TimeCut_i)
                
                histo2d(df=df_timecut, name=f'Cebra{det}Energy_ADCShift_Xavg_TimeCut', x_column=f'Xavg', y_column=f'Cebra{det}Energy_ADCShift', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
                Xavg_CeBrAEnergy_ADCShift_TimeCut_i = histo2d(df=df_timecut, name=f'CeBrAEnergy_ADCShift_Xavg_TimeCut', x_column=f'Xavg', y_column=f'Cebra{det}Energy_ADCShift', bins=[600, 512], range=[[-300,300], [0,4096]])
                Xavg_CeBrAEnergy_ADCShift_TimeCut_figures.append(Xavg_CeBrAEnergy_ADCShift_TimeCut_i)
                
                histo2d(df=df_timecut, name=f'Cebra{det}Energy_ADCShift_X1_TimeCut', x_column=f'X1', y_column=f'Cebra{det}Energy_ADCShift', bins=[600, 512], range=[[-300,300], [0,4096]], fig_list=figures)
                X1_CeBrAEnergy_ADCShift_TimeCut_i = histo2d(df=df_timecut, name=f'CeBrAEnergy_ADCShift_X1_TimeCut', x_column=f'X1', y_column=f'Cebra{det}Energy_ADCShift', bins=[600, 512], range=[[-300,300], [0,4096]])
                X1_CeBrAEnergy_ADCShift_TimeCut_figures.append(X1_CeBrAEnergy_ADCShift_TimeCut_i)

    '''For summing histograms together'''
    if ADCShifts is not None:
        for det in range(NumDetectors):
            if det==0:
                CeBrAEnergy_ADCShift = CeBrAEnergy_ADCShift_figures[det]
            else:
                CeBrAEnergy_ADCShift.Add(CeBrAEnergy_ADCShift_figures[det].GetPtr())
        CeBrAEnergy_ADCShift.SetTitle(f'; CeBrAEnergy_ADCShifts; Counts')
        
        figures.append(CeBrAEnergy_ADCShift)
                
        if SPS:
            for det in range(NumDetectors):
                if det==0:
                    Xavg_TimeCuts = Xavg_timecut_figures[det]
                    CeBrAEnergy_ADCShift_TimeCuts = CeBrAEnergy_ADCShift_TimeCut_figures[det]
                    Xavg_CeBrA = Xavg_CeBrAEnergy_ADCShift_TimeCut_figures[det]
                    X1_CeBrA = X1_CeBrAEnergy_ADCShift_TimeCut_figures[det]
                else:
                    Xavg_TimeCuts.Add(Xavg_timecut_figures[det].GetPtr())
                    CeBrAEnergy_ADCShift_TimeCuts.Add(CeBrAEnergy_ADCShift_TimeCut_figures[det].GetPtr())
                    Xavg_CeBrA.Add(Xavg_CeBrAEnergy_ADCShift_TimeCut_figures[det].GetPtr())
                    X1_CeBrA.Add(X1_CeBrAEnergy_ADCShift_TimeCut_figures[det].GetPtr())
                
            Xavg_TimeCuts.SetTitle(f'; Xavg_TimeCuts; Counts')
            CeBrAEnergy_ADCShift_TimeCuts.SetTitle(f'; CeBrAEnergy_ADCShifts_TimeCuts; Counts')
            Xavg_CeBrA.SetTitle(f';Xavg; CeBrA')
            X1_CeBrA.SetTitle(f'; X1; CeBrA')
            
            figures.append(Xavg_TimeCuts)
            figures.append(CeBrAEnergy_ADCShift_TimeCuts)
            figures.append(Xavg_CeBrA)
            figures.append(X1_CeBrA)

    return figures

SESPS_figures = SPSPlots(df=df, Settings=sps_cebra_settings)

TimeGates = [ [-1158,-1152],
            [-1157,-1151],
            [-1157,-1151],
            [-1156,-1150],
            [-1126,-1120] ]

ADCShifts = [[1.7551059351549314, -12.273506897222896],
             [1.9510278378962256, -16.0245754973971],
             [1.917190081718234, 16.430212777833802],
             [1.6931918955746692, 12.021258506937766],
             [1.6373533248536343, 13.091030061910748] ]

CeBrA_figures = CeBrAPlots(df=df, NumDetectors=5, SPS=True, ADCShifts=ADCShifts, TimeGates=TimeGates)
    
SPS_CeBrA_Figures = SESPS_figures + CeBrA_figures
    
output = ROOT.TFile.Open(f"./test/53Cr.root", "RECREATE")
output.cd()


for fig in SPS_CeBrA_Figures:
    fig.Write()
output.Close()