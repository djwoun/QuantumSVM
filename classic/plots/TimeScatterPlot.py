# -*- coding: utf-8 -*-
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
Noise = [0,5,10,15,20,25,30]
SplitRatios = [80,70,60,50]
df =pd.DataFrame()
df2 = pd.read_csv (filepath_or_buffer='../testResult/2.csv')
df5 = pd.DataFrame()
df3 = pd.DataFrame()
Yvar = 'Time Count'
Xvar = 'Noise '
#Xvar = 'Training  '
df4 = pd.DataFrame()
    
S8N0 = df2[df2['Training  '] ==50]
#S8N0 = df2[df2['Noise '] ==0 ]
a = (S8N0)[Yvar].to_numpy()

b = (S8N0)[Xvar].to_numpy()
S8N0 = pd.DataFrame(a,b)
    
plt.scatter(b,a)
print(S8N0)
   
    
    #df3['0'] = df3['0'].append(S8N0, ignore_index = True)
    #df3['0'] = S8N0
    #df3['g'] = [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]

    #df = df.append(df3, ignore_index = True)
    
    #print("STD of S8N0" ,S8N0.std())
    
   #df4 = pd.DataFrame({'Mean'+'50': S8N0.mean(),
   #                     'STD'+'50': S8N0.std(),
                        #'g':x,
   #                     })
    #df5 = pd.concat([df5, df4])
    
#df5.to_csv(path_or_buf='../Mean&STD/Trained Sets vs'+ Xvar+' MEAN&STD.csv')
    
#dfm = df.melt(id_vars='g',var_name='Noise', value_name=Xvar)

#p = sns.pointplot(data=dfm, x='Noise', y=Xvar, hue='g', ci=68, dodge=0.5)
#p.set_title('Noise vs '+Xvar)
#p.legend(title='Trained Sets', bbox_to_anchor=(0.85, 0.8), loc='upper left')

#plt.savefig('Trained Sets vs'+ Xvar+'.png')


