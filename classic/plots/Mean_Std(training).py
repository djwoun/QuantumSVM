# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
Noise = [0,5,10,15,20,25,30]
SplitRatios = [80,70,60,50]

version = "Wbc"

df2 = pd.read_csv (filepath_or_buffer='../testResult/'+version+'.csv')
xRan = ['Time Count','Training Error','Testing Error',"Number of SV",'Objective Function']
unit=''
multiplier = 1
for Xvar in xRan:
    
    if Xvar == 'Time Count':
        unit ='(ms)'
        multiplier = 1000
    elif (Xvar == 'Training Error' or Xvar == 'Testing Error' ):
        unit ='(%)'
        multiplier = 100
    elif (Xvar == 'Objective Function'):
        multiplier = 1
        unit = ''
    elif (Xvar == 'Number of SV'):
        multiplier = 1
        unit = ''
    df5 = pd.DataFrame()
    df =pd.DataFrame()
    for x in Noise:
        df3 = pd.DataFrame()
        S8N0 = df2[df2['Training'] ==50][df2['Noise'] ==x ]
        S8N5 = df2[df2['Training'] ==60][df2['Noise'] ==x ]
        S8N10 = df2[df2['Training'] ==70][df2['Noise'] ==x ]
        S8N15 = df2[df2['Training'] ==80][df2['Noise'] ==x ]
    
        a = (S8N0)[Xvar].to_numpy()
        S8N0 = pd.DataFrame(a)
        
        a = (S8N5)[Xvar].to_numpy()
        S8N5 = pd.DataFrame(a)
        
        a = (S8N10)[Xvar].to_numpy()
        S8N10 = pd.DataFrame(a)
        
        a = (S8N15)[Xvar].to_numpy()
        S8N15 = pd.DataFrame(a)
    
        
       
        
        #df3['0'] = df3['0'].append(S8N0, ignore_index = True)
        df3['50'] = S8N0*multiplier
        df3['60'] = S8N5*multiplier
        df3['70'] = S8N10*multiplier
        df3['80'] = S8N15*multiplier
        df3['g'] = [x] * 30
        df = pd.concat([df, df3])
        
        #print("STD of S8N0" ,S8N0.std())
        
        df4 = pd.DataFrame({'Mean'+'50': S8N0.mean(),
                            'STD'+'50': S8N0.std(),
                            'Mean'+'60': S8N5.mean(),
                            'STD'+'60': S8N5.std(),
                            'Mean'+'70': S8N10.mean(),
                            'STD'+'70': S8N10.std(),
                            'Mean'+'80': S8N15.mean(),
                            'STD'+'80': S8N15.std(),
                            'g':x,
                            })
        df5 = pd.concat([df5, df4])
    if Xvar == "Testing Error":  
        df5.to_csv(path_or_buf='../Mean&STD/Trained Sets vs'+ Xvar+version+' MEAN&STD.csv')
        
    dfm = df.melt(id_vars='g',var_name='Trained Sets', value_name=Xvar)
    
    p = sns.pointplot(data=dfm, x='Trained Sets', y=Xvar, hue='g', ci=68, dodge=0.25)
    p.set(xlabel='Trained(%)', ylabel=Xvar+unit)
    p.legend(title='Noise(%)', bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig('Trained Sets vs ' +Xvar+version+'.png', bbox_inches='tight', dpi=250)
    plt.clf()

