import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from histogrammer import Histogrammer
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

@timer_func
def SPS(df):
        
    h = Histogrammer()
    
    df = df.with_columns( (pl.col('DelayFrontRightEnergy') + pl.col('DelayFrontLeftEnergy')/2).alias('DelayFrontAverageEnergy'),
                         (pl.col('DelayBackRightEnergy') + pl.col('DelayBackLeftEnergy')/2).alias('DelayBackAverageEnergy') )
    
    h.add_fill_hist1d(name=f'X1', data=df[f'X1'], bins=600, range=[-300,300])
    h.add_fill_hist1d(name=f'X2', data=df[f'X2'], bins=600, range=[-300,300])    
    h.add_fill_hist2d(name=f'X2_X1', x_data=df[f'X1'], y_data=df[f'X2'], bins=[600, 600], ranges=[[-300,300], [-300,300]])
    h.add_fill_hist2d(name=f'DelayBackRightEnergy_X1', x_data=df[f'X1'], y_data=df[f'DelayBackRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackLeftEnergy_X1', x_data=df[f'X1'], y_data=df[f'DelayBackLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontRightEnergy_X1', x_data=df[f'X1'], y_data=df[f'DelayFrontRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontLeftEnergy_X1', x_data=df[f'X1'], y_data=df[f'DelayFrontLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackRightEnergy_X2', x_data=df[f'X2'], y_data=df[f'DelayBackRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackLeftEnergy_X2', x_data=df[f'X2'], y_data=df[f'DelayBackLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontRightEnergy_X2', x_data=df[f'X2'], y_data=df[f'DelayFrontRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontLeftEnergy_X2', x_data=df[f'X2'], y_data=df[f'DelayFrontLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackRightEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'DelayBackRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackLeftEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'DelayBackLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontRightEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'DelayFrontRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontLeftEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'DelayFrontLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontAverageEnergy_X1', x_data=df[f'X1'], y_data=df[f'DelayFrontAverageEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackAverageEnergy_X1', x_data=df[f'X1'], y_data=df[f'DelayBackAverageEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontAverageEnergy_X2', x_data=df[f'X2'], y_data=df[f'DelayFrontAverageEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackAverageEnergy_X2', x_data=df[f'X2'], y_data=df[f'DelayBackAverageEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayFrontAverageEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'DelayFrontAverageEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'DelayBackAverageEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'DelayBackAverageEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeBackEnergy_ScintLeftEnergy', x_data=df[f'ScintLeftEnergy'], y_data=df[f'AnodeBackEnergy'], bins=[512, 512], ranges=[[0,4096], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeFrontEnergy_ScintLeftEnergy', x_data=df[f'ScintLeftEnergy'], y_data=df[f'AnodeFrontEnergy'], bins=[512, 512], ranges=[[0,4096], [0,4096]])
    h.add_fill_hist2d(name=f'CathodeEnergy_ScintLeftEnergy', x_data=df[f'ScintLeftEnergy'], y_data=df[f'CathodeEnergy'], bins=[512, 512], ranges=[[0,4096], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeBackEnergy_ScintRightEnergy', x_data=df[f'ScintRightEnergy'], y_data=df[f'AnodeBackEnergy'], bins=[512, 512], ranges=[[0,4096], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeFrontEnergy_ScintRightEnergy', x_data=df[f'ScintRightEnergy'], y_data=df[f'AnodeFrontEnergy'], bins=[512, 512], ranges=[[0,4096], [0,4096]])
    h.add_fill_hist2d(name=f'CathodeEnergy_ScintRightEnergy', x_data=df[f'ScintRightEnergy'], y_data=df[f'CathodeEnergy'], bins=[512, 512], ranges=[[0,4096], [0,4096]])
    h.add_fill_hist2d(name=f'ScintLeftEnergy_X1', x_data=df[f'X1'], y_data=df[f'ScintLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'ScintLeftEnergy_X2', x_data=df[f'X2'], y_data=df[f'ScintLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'ScintLeftEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'ScintLeftEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'ScintRightEnergy_X1', x_data=df[f'X1'], y_data=df[f'ScintRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'ScintRightEnergy_X2', x_data=df[f'X2'], y_data=df[f'ScintRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'ScintRightEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'ScintRightEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeBackEnergy_X1', x_data=df[f'X1'], y_data=df[f'AnodeBackEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeBackEnergy_X2', x_data=df[f'X2'], y_data=df[f'AnodeBackEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeBackEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'AnodeBackEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeFrontEnergy_X1', x_data=df[f'X1'], y_data=df[f'AnodeFrontEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeFrontEnergy_X2', x_data=df[f'X2'], y_data=df[f'AnodeFrontEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'AnodeFrontEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'AnodeFrontEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'CathodeEnergy_X1', x_data=df[f'X1'], y_data=df[f'CathodeEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'CathodeEnergy_X2', x_data=df[f'X2'], y_data=df[f'CathodeEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
    h.add_fill_hist2d(name=f'CathodeEnergy_Xavg', x_data=df[f'Xavg'], y_data=df[f'CathodeEnergy'], bins=[600, 512], ranges=[[-300,300], [0,4096]])
             
    '''Both planes'''
             
    df_bothplanes = df.filter(pl.col('X1') != -1e6).filter(pl.col('X2') != -1e6).with_columns(
        (pl.col('DelayFrontLeftTime') - pl.col('AnodeFrontTime')).alias('DelayFrontLeftTime_AnodeFrontTime'),
        (pl.col('DelayFrontRightTime') - pl.col('AnodeFrontTime')).alias('DelayFrontRightTime_AnodeFrontTime'),
        (pl.col('DelayBackLeftTime') - pl.col('AnodeBackTime')).alias('DelayBackLeftTime_AnodeBackTime'),
        (pl.col('DelayBackRightTime') - pl.col('AnodeBackTime')).alias('DelayBackRightTime_AnodeBackTime'),
    )
        
    h.add_fill_hist1d(name=f'Xavg_bothplanes', data=df_bothplanes[f'Xavg'], bins=600, range=[-300,300])
    h.add_fill_hist1d(name=f'X1_bothplanes', data=df_bothplanes[f'X1'], bins=600, range=[-300,300])
    h.add_fill_hist1d(name=f'X2_bothplanes', data=df_bothplanes[f'X2'], bins=600, range=[-300,300])
    h.add_fill_hist2d(name=f'Theta_Xavg_bothplanes', x_data=df_bothplanes[f'Xavg'], y_data=df_bothplanes[f'Theta'], bins=[600, 300], ranges=[[-300,300], [0,pi/2]])
    h.add_fill_hist1d(name=f'DelayFrontLeftTime_relTo_AnodeFrontTime_bothplanes', data=df_bothplanes[f'DelayFrontLeftTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayFrontRightTime_relTo_AnodeFrontTime_bothplanes', data=df_bothplanes[f'DelayFrontRightTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackLeftTime_relTo_AnodeBackTime_bothplanes', data=df_bothplanes[f'DelayBackLeftTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackRightTime_relTo_AnodeBackTime_bothplanes', data=df_bothplanes[f'DelayBackRightTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    
    '''Only 1 plane: X1'''

    df_onlyX1plane = df.filter(pl.col('X1') != -1e6).filter(pl.col('X2') == -1e6).with_columns(
        (pl.col('DelayFrontLeftTime') - pl.col('AnodeFrontTime')).alias('DelayFrontLeftTime_AnodeFrontTime'),
        (pl.col('DelayFrontRightTime') - pl.col('AnodeFrontTime')).alias('DelayFrontRightTime_AnodeFrontTime'),
        (pl.col('DelayBackLeftTime') - pl.col('AnodeFrontTime')).alias('DelayBackLeftTime_AnodeFrontTime'),
        (pl.col('DelayBackRightTime') - pl.col('AnodeFrontTime')).alias('DelayBackRightTime_AnodeFrontTime'),
        (pl.col('DelayFrontLeftTime') - pl.col('AnodeBackTime')).alias('DelayFrontLeftTime_AnodeBackTime'),
        (pl.col('DelayFrontRightTime') - pl.col('AnodeBackTime')).alias('DelayFrontRightTime_AnodeBackTime'),
        (pl.col('DelayBackLeftTime') - pl.col('AnodeBackTime')).alias('DelayBackLeftTime_AnodeBackTime'),
        (pl.col('DelayBackRightTime') - pl.col('AnodeBackTime')).alias('DelayBackRightTime_AnodeBackTime'),
    )
            
    h.add_fill_hist1d(name=f'X1_only1plane', data=df_onlyX1plane[f'X1'], bins=600, range=[-300,300])
    h.add_fill_hist1d(name=f'DelayFrontLeftTime_relTo_AnodeFrontTime_noX2', data=df_onlyX1plane[f'DelayFrontLeftTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayFrontRightTime_relTo_AnodeFrontTime_noX2', data=df_onlyX1plane[f'DelayFrontRightTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackLeftTime_relTo_AnodeFrontTime_noX2', data=df_onlyX1plane[f'DelayBackLeftTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackRightTime_relTo_AnodeFrontTime_noX2', data=df_onlyX1plane[f'DelayBackRightTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayFrontLeftTime_relTo_AnodeBackTime_noX2', data=df_onlyX1plane[f'DelayFrontLeftTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayFrontRightTime_relTo_AnodeBackTime_noX2', data=df_onlyX1plane[f'DelayFrontRightTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackLeftTime_relTo_AnodeBackTime_noX2', data=df_onlyX1plane[f'DelayBackLeftTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackRightTime_relTo_AnodeBackTime_noX2', data=df_onlyX1plane[f'DelayBackRightTime_AnodeBackTime'], bins=8000, range=[-4000,4000])

    '''Only 1 plane: X2'''

    df_onlyX2plane = df.filter(pl.col('X1') == -1e6).filter(pl.col('X2') != -1e6).with_columns(
        (pl.col('DelayFrontLeftTime') - pl.col('AnodeFrontTime')).alias('DelayFrontLeftTime_AnodeFrontTime'),
        (pl.col('DelayFrontRightTime') - pl.col('AnodeFrontTime')).alias('DelayFrontRightTime_AnodeFrontTime'),
        (pl.col('DelayBackLeftTime') - pl.col('AnodeFrontTime')).alias('DelayBackLeftTime_AnodeFrontTime'),
        (pl.col('DelayBackRightTime') - pl.col('AnodeFrontTime')).alias('DelayBackRightTime_AnodeFrontTime'),
        (pl.col('DelayFrontLeftTime') - pl.col('AnodeBackTime')).alias('DelayFrontLeftTime_AnodeBackTime'),
        (pl.col('DelayFrontRightTime') - pl.col('AnodeBackTime')).alias('DelayFrontRightTime_AnodeBackTime'),
        (pl.col('DelayBackLeftTime') - pl.col('AnodeBackTime')).alias('DelayBackLeftTime_AnodeBackTime'),
        (pl.col('DelayBackRightTime') - pl.col('AnodeBackTime')).alias('DelayBackRightTime_AnodeBackTime'),
    )

    h.add_fill_hist1d(name=f'X2_only1plane', data=df_onlyX2plane[f'X1'], bins=600, range=[-300,300])
    h.add_fill_hist1d(name=f'DelayFrontLeftTime_relTo_AnodeFrontTime_noX1', data=df_onlyX2plane[f'DelayFrontLeftTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayFrontRightTime_relTo_AnodeFrontTime_noX1', data=df_onlyX2plane[f'DelayFrontRightTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackLeftTime_relTo_AnodeFrontTime_noX1', data=df_onlyX2plane[f'DelayBackLeftTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackRightTime_relTo_AnodeFrontTime_noX1', data=df_onlyX2plane[f'DelayBackRightTime_AnodeFrontTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayFrontLeftTime_relTo_AnodeBackTime_noX1', data=df_onlyX2plane[f'DelayFrontLeftTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayFrontRightTime_relTo_AnodeBackTime_noX1', data=df_onlyX2plane[f'DelayFrontRightTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackLeftTime_relTo_AnodeBackTime_noX1', data=df_onlyX2plane[f'DelayBackLeftTime_AnodeBackTime'], bins=8000, range=[-4000,4000])
    h.add_fill_hist1d(name=f'DelayBackRightTime_relTo_AnodeBackTime_noX1', data=df_onlyX2plane[f'DelayBackRightTime_AnodeBackTime'], bins=8000, range=[-4000,4000])


    '''Time relative to Back Anode'''
    
    df_TimeRelBackAnode = df.filter(pl.col('AnodeBackTime') != -1e6).filter(pl.col('ScintLeftTime') != -1e6).with_columns(
        (pl.col('AnodeFrontTime') - pl.col('AnodeBackTime')).alias('AnodeFrontTime_AnodeBackTime'),
        (pl.col('AnodeBackTime') - pl.col('AnodeFrontTime')).alias('AnodeBackTime_AnodeFrontTime'),
        (pl.col('AnodeFrontTime') - pl.col('ScintLeftTime')).alias('AnodeFrontTime_ScintLeftTime'),
        (pl.col('AnodeBackTime') - pl.col('ScintLeftTime')).alias('AnodeBackTime_ScintLeftTime'),
        (pl.col('DelayFrontLeftTime') - pl.col('ScintLeftTime')).alias('DelayFrontLeftTime_ScintLeftTime'),
        (pl.col('DelayFrontRightTime') - pl.col('ScintLeftTime')).alias('DelayFrontRightTime_ScintLeftTime'),
        (pl.col('DelayBackLeftTime') - pl.col('ScintLeftTime')).alias('DelayBackLeftTime_ScintLeftTime'),
        (pl.col('DelayBackRightTime') - pl.col('ScintLeftTime')).alias('DelayBackRightTime_ScintLeftTime'),
        (pl.col('ScintRightTime') - pl.col('ScintLeftTime')).alias('ScintRightTime_ScintLeftTime'),
    )
                                                                            
    h.add_fill_hist1d(name=f'AnodeFrontTime_AnodeBackTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'AnodeBackTime_AnodeFrontTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'AnodeFrontTime_ScintLeftTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'AnodeBackTime_ScintLeftTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'DelayFrontLeftTime_ScintLeftTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'DelayFrontRightTime_ScintLeftTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'DelayBackLeftTime_ScintLeftTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'DelayBackRightTime_ScintLeftTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist1d(name=f'ScintRightTime_ScintLeftTime', data=df_TimeRelBackAnode[f'AnodeFrontTime_AnodeBackTime'], bins=1000, range=[-3000,3000])
    h.add_fill_hist2d(name=f'ScintTimeDif_Xavg', x_data=df_TimeRelBackAnode[f'Xavg'], y_data=df_TimeRelBackAnode[f'ScintRightTime_ScintLeftTime'], bins=[600, 12800], ranges=[[-300,300],[-3200,3200]])
    
    return h

@timer_func
def CeBrA(df:pl.DataFrame, input_file:yaml, SPS=True):
    
    h = Histogrammer()
    
    with open(input_file, 'r') as file:
        data = yaml.safe_load(file)
        
    detectors_data = data.get('CeBr3_Detectors', [])
    
    Cebra_Ecal_para = [500,0,6000]
    
    '''Summed plots'''    
    h.add_hist1d(name='CeBrAEnergyGainMatched', bins=512, range=(0,4096))
    h.add_hist1d(name='CeBrAEnergyCalibrated', bins=Cebra_Ecal_para[0], range=[Cebra_Ecal_para[1],Cebra_Ecal_para[2]])
    
    h.add_hist1d(name='CeBrATimeToScintShifted', bins=100, range=(-50,50))
    h.add_hist2d(name='CeBrATimeToScint_Xavg', bins=(600,100), ranges=[(-300,300), (-50,50)])
    
    h.add_hist1d(name='CeBrATimeToScintShifted_TimeCut', bins=100, range=(-50,50))
    h.add_hist1d(name='Xavg_TimeCut', bins=600, range=(-300,300))
    h.add_hist1d(name='CeBrAEnergyGainMatched_TimeCut', bins=512, range=(0,4096))
    h.add_hist1d(name='CeBrAEnergyCalibrated_TimeCut', bins=Cebra_Ecal_para[0], range=[Cebra_Ecal_para[1],Cebra_Ecal_para[2]])
    
    h.add_hist2d(name='CeBrAEnergyGainMatched_X1', bins=(600,512), ranges=[(-300,300), (0,4096)])
    h.add_hist2d(name='CeBrAEnergyGainMatched_Xavg', bins=(600,512), ranges=[(-300,300), (0,4096)])
    h.add_hist2d(name='CeBrAEnergyCalibrated_Xavg', bins=(600,Cebra_Ecal_para[0]), ranges=[(-300,300), (Cebra_Ecal_para[1],Cebra_Ecal_para[2])])
    
    for detector in detectors_data:
        det_num = detector['detector_number']
        
        det_df = df.filter(pl.col(f"Cebra{det_num}Energy") != -1e6)
        h.add_fill_hist1d(name=f"Cebra{det_num}Energy", data=det_df[f"Cebra{det_num}Energy"], bins=512, range=[0,4096])
        
        det_df = det_df.with_columns( (( pl.col(f"Cebra{det_num}Time") - pl.col("ScintLeftTime") )).alias(f"Cebra{det_num}TimeToScint") )
        h.add_fill_hist1d(name=f"Cebra{det_num}TimeToScint", data=det_df[f"Cebra{det_num}TimeToScint"], bins=6400, range=[-3200,3200])
        
        det_df = det_df.with_columns( (( detector['gain_shift'].get('m') * pl.col(f"Cebra{det_num}Energy") + detector['gain_shift'].get('b') )).alias(f"Cebra{det_num}EnergyGainMatched") )
        h.add_fill_hist1d(name=f"Cebra{det_num}EnergyGainMatched", data=det_df[f"Cebra{det_num}EnergyGainMatched"], bins=512, range=[0,4096])
        h.fill_hist1d(name=f"CeBrAEnergyGainMatched", data=det_df[f"Cebra{det_num}EnergyGainMatched"])
        
        det_df = det_df.with_columns( (( detector['energy_calibration'].get('a')*pl.col(f"Cebra{det_num}EnergyGainMatched")*pl.col(f"Cebra{det_num}EnergyGainMatched")
                                + detector['energy_calibration'].get('b')*pl.col(f"Cebra{det_num}EnergyGainMatched")
                                + detector['energy_calibration'].get('c') ))
                             .alias(f"Cebra{det_num}EnergyCalibrated") )
        
        h.add_fill_hist1d(name=f"Cebra{det_num}EnergyCalibrated", data=det_df[f"Cebra{det_num}EnergyCalibrated"], bins=Cebra_Ecal_para[0], range=[Cebra_Ecal_para[1],Cebra_Ecal_para[2]])
        h.fill_hist1d(name=f"CeBrAEnergyCalibrated", data=det_df[f"Cebra{det_num}EnergyCalibrated"])
        
        centroid = detector['time_cut'].get('centroid', None)
        t1 = detector['time_cut'].get('lower_gate', None)
        t2 = detector['time_cut'].get('upper_gate', None)
        
        if all(coeff is not None for coeff in [centroid, t1, t2]):
            # Calculate the new column '<detector_name>EnergyCalibrated' and add it to the DataFrame
            det_df = det_df.with_columns( (( pl.col(f"Cebra{det_num}TimeToScint") - centroid )).alias(f"Cebra{det_num}TimeToScintShifted") )
            h.add_fill_hist1d(name=f"Cebra{det_num}TimeToScintShifted", data=det_df[f"Cebra{det_num}TimeToScintShifted"], bins=6400, range=[-3200,3200])
            h.fill_hist1d(name='CeBrATimeToScintShifted',data=det_df[f"Cebra{det_num}TimeToScintShifted"])
            h.fill_hist2d(name='CeBrATimeToScint_Xavg', x_data=det_df['Xavg'], y_data=det_df[f"Cebra{det_num}TimeToScintShifted"])
            
            det_df_timecut = det_df.filter( pl.col(f"Cebra{det_num}TimeToScint") >= t1).filter( pl.col(f"Cebra{det_num}TimeToScint") <= t2 )
            h.add_fill_hist1d(name=f"Cebra{det_num}TimeToScint_TimeCut", data=det_df_timecut[f"Cebra{det_num}TimeToScint"], bins=6400, range=[-3200,3200])
            h.add_fill_hist1d(name=f"Cebra{det_num}TimeToScintShifted_TimeCut", data=det_df_timecut[f"Cebra{det_num}TimeToScintShifted"], bins=6400, range=[-3200,3200])
            h.add_fill_hist1d(name=f"Cebra{det_num}EnergyGainMatched_TimeCut", data=det_df_timecut[f"Cebra{det_num}EnergyGainMatched"], bins=512, range=[0,4096])
            h.add_fill_hist1d(name=f"Cebra{det_num}EnergyCalibrated_TimeCut", data=det_df_timecut[f"Cebra{det_num}EnergyCalibrated"], bins=Cebra_Ecal_para[0], range=[Cebra_Ecal_para[1],Cebra_Ecal_para[2]])
            h.add_fill_hist2d(name=f"Cebra{det_num}EnergyGainMatched_X1", x_data=det_df_timecut['X1'], y_data=det_df_timecut[f"Cebra{det_num}EnergyGainMatched"], bins=[600,512], ranges=[[-300,300],[0,4096]])
            h.add_fill_hist2d(name=f"Cebra{det_num}EnergyGainMatched_Xavg", x_data=det_df_timecut['Xavg'], y_data=det_df_timecut[f"Cebra{det_num}EnergyGainMatched"], bins=[600,512], ranges=[[-300,300],[0,4096]])
            h.add_fill_hist2d(name=f"Cebra{det_num}EnergyCalibrated_Xavg", x_data=det_df_timecut['Xavg'], y_data=det_df_timecut[f"Cebra{det_num}EnergyCalibrated"], bins=[600,Cebra_Ecal_para[0]], ranges=[[-300,300],[Cebra_Ecal_para[1],Cebra_Ecal_para[2]]])
           
            h.fill_hist1d(name='Xavg_TimeCut',data=det_df_timecut[f"Xavg"])
            h.fill_hist1d(name='CeBrATimeToScintShifted_TimeCut',data=det_df_timecut[f"Cebra{det_num}TimeToScintShifted"])
            h.fill_hist1d(name=f"CeBrAEnergyGainMatched_TimeCut", data=det_df_timecut[f"Cebra{det_num}EnergyGainMatched"])
            h.fill_hist1d(name=f"CeBrAEnergyCalibrated_TimeCut", data=det_df_timecut[f"Cebra{det_num}EnergyCalibrated"])
            h.fill_hist2d(name=f"CeBrAEnergyGainMatched_X1", x_data=det_df_timecut['X1'], y_data=det_df_timecut[f"Cebra{det_num}EnergyGainMatched"])
            h.fill_hist2d(name=f"CeBrAEnergyGainMatched_Xavg", x_data=det_df_timecut['Xavg'], y_data=det_df_timecut[f"Cebra{det_num}EnergyGainMatched"])
            h.fill_hist2d(name=f"CeBrAEnergyCalibrated_Xavg", x_data=det_df_timecut['Xavg'], y_data=det_df_timecut[f"Cebra{det_num}EnergyCalibrated"])
           
    return h

df = pl.read_parquet("../../52Cr_July2023_REU_CeBrA/analysis/run_83_112_gainmatched.parquet")

SPS_histograms = SPS(df=df)

# fig, ax = plt.subplots(2,2)
# ax = ax.flatten()
# SPS_histograms.draw_hist2d(name='AnodeBackEnergy_ScintLeftEnergy', axis=ax[0])
# SPS_histograms.draw_hist2d(name='AnodeBackEnergy_Xavg', axis=ax[1])
# SPS_histograms.draw_hist1d(name='Xavg_bothplanes', axis=ax[2])
# SPS_histograms.draw_hist1d(name='X1_only1plane', axis=ax[2])
# SPS_histograms.draw_hist1d(name='X2_only1plane', axis=ax[2])
# SPS_histograms.draw_hist2d(name='Theta_Xavg_bothplanes', axis=ax[3])

CeBrA_histograms = CeBrA(df=df, input_file='./SPS_CeBrA_settings_templete.yaml')

# '''Gets the CebraTime - ScintLeftTime histograms'''
# CeBrATimeToScint_fig, CeBrATimeToScint_ax = plt.subplots(2,3,figsize=(10,6))
# CeBrATimeToScint_ax = CeBrATimeToScint_ax.flatten()
# for det in range(5):
#     CeBrA_histograms.draw_hist1d(name=f'Cebra{det}TimeToScint', axis=CeBrATimeToScint_ax[det], display_stats=False)
#     CeBrA_histograms.draw_hist1d(name=f'Cebra{det}TimeToScint_TimeCut', axis=CeBrATimeToScint_ax[det], color='green')  
# CeBrA_histograms.draw_hist1d(name=f'CeBrATimeToScintShifted', axis=CeBrATimeToScint_ax[5], display_stats=False)
# CeBrA_histograms.draw_hist1d(name=f'CeBrATimeToScintShifted_TimeCut', axis=CeBrATimeToScint_ax[5], color='green')
# CeBrATimeToScint_fig.tight_layout()

# '''Gets the CebraEnergyGainMatched compared to CebraEnergy'''
# CeBrAEnergy_fig, CeBrAEnergy_ax = plt.subplots(2,3,figsize=(10,6))
# CeBrAEnergy_ax = CeBrAEnergy_ax.flatten()
# for det in range(5):
#     CeBrA_histograms.draw_hist1d(name=f'Cebra{det}Energy', axis=CeBrAEnergy_ax[det])
#     CeBrA_histograms.draw_hist1d(name=f'Cebra{det}EnergyGainMatched', axis=CeBrAEnergy_ax[det])  
# CeBrA_histograms.draw_hist1d(name=f'CeBrAEnergyGainMatched', axis=CeBrAEnergy_ax[5])

# '''Gets the CebraEnergyGainMatched compared to CebraEnergy'''
# CeBrAEnergyCalibrated_fig, CeBrAEnergyCalibrated_ax = plt.subplots(2,3,figsize=(10,6))
# CeBrAEnergyCalibrated_ax = CeBrAEnergyCalibrated_ax.flatten()
# for det in range(5):
#     CeBrA_histograms.draw_hist1d(name=f'Cebra{det}EnergyCalibrated', axis=CeBrAEnergyCalibrated_ax[det])  
#     CeBrA_histograms.draw_hist1d(name=f'Cebra{det}EnergyCalibrated', axis=CeBrAEnergyCalibrated_ax[5])  
# CeBrA_histograms.draw_hist1d(name=f'CeBrAEnergyCalibrated', axis=CeBrAEnergyCalibrated_ax[5])

'''Particle-gamma coincidence matricies'''
# pg_fig, pg_ax = plt.subplots(1,2,figsize=(10,6))
# # pg_ax = pg_ax.flatten()
# # for det in range(5):
# #     CeBrA_histograms.draw_hist2d(name=f'Cebra{det}EnergyCalibrated_Xavg', axis=pg_ax[det])

# CeBrA_histograms.draw_hist1d(name='Xavg_TimeCut', axis=pg_ax[0])
# CeBrA_histograms.draw_hist2d(name=f'CeBrAEnergyCalibrated_Xavg', axis=pg_ax[1])


# pg_fig, pg_ax = plt.subplots(1,1,figsize=(12,6))
CeBrA_histograms.draw_hist2d(name=f'CeBrAEnergyCalibrated_Xavg', axis=pg_ax, display_stats=False)
# SPS_histograms.draw_hist1d(name=f'Xavg_bothplanes', axis=pg_ax, display_stats=True)
# CeBrA_histograms.draw_hist1d(name=f'Xavg_TimeCut', axis=pg_ax[1], display_stats=True)

plt.show()