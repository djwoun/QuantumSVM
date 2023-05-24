# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
version = "syn81"
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
        S8N0 = df2[df2['Training'] ==x][df2['Features'] ==2 ]
        S8N5 = df2[df2['Training'] ==x][df2['Features'] ==4 ]
        S8N10 = df2[df2['Training'] ==x][df2['Features'] ==8 ]
        S8N15 = df2[df2['Training'] ==x][df2['Features'] ==16 ]
        S8N20 = df2[df2['Training'] ==x][df2['Features'] ==32 ]
        S8N25 = df2[df2['Training'] ==x][df2['Features'] ==64 ]
        S8N30 = df2[df2['Training'] ==x][df2['Features'] ==128 ]
        S8N35 = df2[df2['Training'] ==x][df2['Features'] ==256 ]
        S8N40 = df2[df2['Training'] ==x][df2['Features'] ==512 ]
        S8N45 = df2[df2['Training'] ==x][df2['Features'] ==1024 ]
        S8N50 = df2[df2['Training'] ==x][df2['Features'] ==2048 ]
        S8N55 = df2[df2['Training'] ==x][df2['Features'] ==4096 ]
        S8N60 = df2[df2['Training'] ==x][df2['Features'] ==8192 ]
        
        S8N65 = df2[df2['Training'] ==x][df2['Features'] ==16384 ]
        S8N70 = df2[df2['Training'] ==x][df2['Features'] ==32768 ]
        S8N75 = df2[df2['Training'] ==x][df2['Features'] ==65536 ]
        S8N80 = df2[df2['Training'] ==x][df2['Features'] ==131072 ]
        S8N85 = df2[df2['Training'] ==x][df2['Features'] ==262144 ]
        S8N90 = df2[df2['Training'] ==x][df2['Features'] ==524288 ]
        S8N95 = df2[df2['Training'] ==x][df2['Features'] ==1048576 ]
        S8N100 = df2[df2['Training'] ==x][df2['Features'] ==2097152 ]
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
        
        a = (S8N35)[Xvar].to_numpy()
        S8N35 = pd.DataFrame(a)
        
        a = (S8N40)[Xvar].to_numpy()
        S8N40 = pd.DataFrame(a)
        
        a = (S8N45)[Xvar].to_numpy()
        S8N45 = pd.DataFrame(a)
        
        a = (S8N50)[Xvar].to_numpy()
        S8N50 = pd.DataFrame(a)
        
        a = (S8N55)[Xvar].to_numpy()
        S8N55 = pd.DataFrame(a)

        a = (S8N60)[Xvar].to_numpy()
        S8N60 = pd.DataFrame(a)

        a = (S8N65)[Xvar].to_numpy()
        S8N65 = pd.DataFrame(a)
        
        a = (S8N70)[Xvar].to_numpy()
        S8N70 = pd.DataFrame(a)
        
        a = (S8N75)[Xvar].to_numpy()
        S8N75 = pd.DataFrame(a)
        
        a = (S8N80)[Xvar].to_numpy()
        S8N80 = pd.DataFrame(a)
        
        a = (S8N85)[Xvar].to_numpy()
        S8N85 = pd.DataFrame(a)
        
        a = (S8N90)[Xvar].to_numpy()
        S8N90 = pd.DataFrame(a)
        
        a = (S8N95)[Xvar].to_numpy()
        S8N95 = pd.DataFrame(a)
        
        a = (S8N100)[Xvar].to_numpy()
        S8N100 = pd.DataFrame(a)
       
        
        #df3['0'] = df3['0'].append(S8N0, ignore_index = True)
        df3['2^1'] = S8N0*multiplier
        df3['2^2'] = S8N5*multiplier
        df3['2^3'] = S8N10*multiplier
        df3['2^4'] = S8N15*multiplier
        df3['2^5'] = S8N20*multiplier
        df3['2^6'] = S8N25*multiplier
        df3['2^7'] = S8N30*multiplier
        df3['2^8'] = S8N35*multiplier
        df3['2^9'] = S8N40*multiplier
        df3['2^10'] = S8N45*multiplier
        df3['2^11'] = S8N50*multiplier
        df3['2^12'] = S8N55*multiplier
        df3['2^13'] = S8N60*multiplier
        
        
        df3['2^14'] = S8N65*multiplier
        df3['2^15'] = S8N70*multiplier
        df3['2^16'] = S8N75*multiplier
        df3['2^17'] = S8N80*multiplier
        df3['2^18'] = S8N85*multiplier
        df3['2^19'] = S8N90*multiplier
        df3['2^20'] = S8N95*multiplier
        df3['2^21'] = S8N100*multiplier
        #df3['g'] = [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
        df3['g'] = [x] * 30
        #df3['g'] = [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
        df = df.append(df3, ignore_index = True)
        #df = df*multiplier
        #print("STD of S8N0" ,S8N0.std())
        
        df4 = pd.DataFrame({#'Mean'+'0': S8N0.mean(),
                            'STD'+'0': S8N0.std(),
                            #'Mean'+'5': S8N5.mean(),
                            'STD'+'5': S8N5.std(),
                            #'Mean'+'10': S8N10.mean(),
                            'STD'+'10': S8N10.std(),
                            #'Mean'+'15': S8N15.mean(),
                            'STD'+'15': S8N15.std(),
                            #'Mean'+'20': S8N20.mean(),
                            'STD'+'20': S8N20.std(),
                            #'Mean'+'25': S8N25.mean(),
                            'STD'+'25': S8N25.std(),
                            #'Mean'+'30': S8N30.mean(),
                            'STD'+'30': S8N30.std(),
                            #'Mean'+'35': S8N35.mean(),
                            'STD'+'35': S8N35.std(),
                            #'Mean'+'40': S8N40.mean(),
                            'STD'+'40': S8N40.std(),
                            #'Mean'+'45': S8N45.mean(),
                            'STD'+'45': S8N45.std(),
                            #'Mean'+'50': S8N50.mean(),
                            'STD'+'50': S8N50.std(),
                            #'Mean'+'55': S8N55.mean(),
                            'STD'+'55': S8N55.std(),
                            #'Mean'+'60': S8N60.mean(),
                            'STD'+'60': S8N60.std(),
                            #'Mean'+'65': S8N65.mean(),
                            'STD'+'65': S8N65.std(),
                            #'Mean'+'70': S8N70.mean(),
                            'STD'+'70': S8N70.std(),
                            'STD'+'75': S8N75.std(),
                            #'Mean'+'75': S8N75.mean(),
                            'STD'+'80': S8N80.std(),
                            #'Mean'+'80': S8N80.mean(),
                            'STD'+'85': S8N85.std(),
                            #'Mean'+'85': S8N85.mean(),
                            'STD'+'90': S8N90.std(),
                            #'Mean'+'90': S8N90.mean(),
                            'STD'+'95': S8N95.std(),
                            #'Mean'+'95': S8N95.mean(),
                            'g':x,
                            })
        
        #df4 = df4*multiplier
        
        df5 = pd.concat([df5, df4])
        df5 = df5.round(5)
    if Xvar == "Time Count":  
        
        df5.to_csv(path_or_buf='../Mean&STD/Noise vs ' +Xvar+version+' std.csv')
        
    dfm = df.melt(id_vars='g',var_name='Features', value_name=Xvar)
    
    #p.legend(title='Percent Trained', bbox_to_anchor=(0.7, 0.5), loc='upper left') testing error
    
    
    p = sns.pointplot(data=dfm, x='Features', y=Xvar, hue='g', ci=68, dodge=0.5)
    p.legend(title='Trained(%)', bbox_to_anchor=(1, 1), loc='upper left')
    p.set_xticklabels(p.get_xticklabels(),rotation=40,ha="right")
    p.set(xlabel='Features', ylabel=Xvar+unit)
    
    #plt.savefig('Features vs ' +Xvar+version+'.png', bbox_inches='tight', dpi=250)
    plt.clf()


