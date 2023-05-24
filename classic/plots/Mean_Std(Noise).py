# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
version = "Syn80.001"
SplitRatios = [80,70,60,50]

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
    
    for x in SplitRatios:
        df3 = pd.DataFrame()
        S8N0 = df2[df2['Training'] ==x][df2['Noise'] ==0 ]
        S8N5 = df2[df2['Training'] ==x][df2['Noise'] ==5 ]
        S8N10 = df2[df2['Training'] ==x][df2['Noise'] ==10 ]
        S8N15 = df2[df2['Training'] ==x][df2['Noise'] ==15 ]
        S8N20 = df2[df2['Training'] ==x][df2['Noise'] ==20 ]
        S8N25 = df2[df2['Training'] ==x][df2['Noise'] ==25 ]
        S8N30 = df2[df2['Training'] ==x][df2['Noise'] ==30 ]
    
        a = (S8N0)[Xvar].to_numpy()
        S8N0 = pd.DataFrame(a)
        
        a = (S8N5)[Xvar].to_numpy()
        S8N5 = pd.DataFrame(a)
        
        a = (S8N10)[Xvar].to_numpy()
        S8N10 = pd.DataFrame(a)
        
        a = (S8N15)[Xvar].to_numpy()
        S8N15 = pd.DataFrame(a)
        
        a = (S8N20)[Xvar].to_numpy()
        S8N20 = pd.DataFrame(a)
        
        a = (S8N25)[Xvar].to_numpy()
        S8N25 = pd.DataFrame(a)
        
        a = (S8N30)[Xvar].to_numpy()
        S8N30 = pd.DataFrame(a)
        
       
        
        #df3['0'] = df3['0'].append(S8N0, ignore_index = True)
        df3['0'] = S8N0*multiplier
        df3['5'] = S8N5*multiplier
        df3['10'] = S8N10*multiplier
        df3['15'] = S8N15*multiplier
        df3['20'] = S8N20*multiplier
        df3['25'] = S8N25*multiplier
        df3['30'] = S8N30*multiplier
        #df3['g'] = [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
        df3['g'] = [x] * 630
        #df3['g'] = [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
        df = df.append(df3, ignore_index = True)
        #df = df*multiplier
        #print("STD of S8N0" ,S8N0.std())
        
        df4 = pd.DataFrame({'Mean'+'0': S8N0.mean(),
                            'STD'+'0': S8N0.std(),
                            'Mean'+'5': S8N5.mean(),
                            'STD'+'5': S8N5.std(),
                            'Mean'+'10': S8N10.mean(),
                            'STD'+'10': S8N10.std(),
                            'Mean'+'15': S8N15.mean(),
                            'STD'+'15': S8N15.std(),
                            'Mean'+'20': S8N20.mean(),
                            'STD'+'20': S8N20.std(),
                            'Mean'+'25': S8N25.mean(),
                            'STD'+'25': S8N25.std(),
                            'Mean'+'30': S8N30.mean(),
                            'STD'+'30': S8N30.std(),
                            'g':x,
                            })
        #df4 = df4*multiplier
        df5 = pd.concat([df5, df4])
    if Xvar == "Time Count":    
        df5.to_csv(path_or_buf='../Mean&STD/Noise vs ' +Xvar+version+' MEAN&STD.csv')
        
    dfm = df.melt(id_vars='g',var_name='Noise', value_name=Xvar)
    
    #p.legend(title='Percent Trained', bbox_to_anchor=(0.7, 0.5), loc='upper left') testing error
    
    
    p = sns.pointplot(data=dfm, x='Noise', y=Xvar, hue='g', ci=68, dodge=0.5)
    p.legend(title='Trained(%)', bbox_to_anchor=(1, 1), loc='upper left')
    
    p.set(xlabel='Noise(%)', ylabel=Xvar+unit)
    
    #plt.savefig('Noise vs ' +Xvar+version+'.png', bbox_inches='tight', dpi=250)
    plt.clf()


