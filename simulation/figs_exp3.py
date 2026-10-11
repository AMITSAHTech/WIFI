import pickle, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from runlib import summarize
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif","Times New Roman","Liberation Serif"],"font.size":8,"axes.linewidth":0.6,"savefig.dpi":600})
O="fig/figs2/"
os.makedirs(O, exist_ok=True)
o=pickle.load(open('exp3.pkl','rb')); Ms=sorted(o)
sty={'Native':('0.55','-','o'),'Greedy':('0.4','--','s'),'EpsSticky':('0.0',':','^'),'TSStat':('0.5','-.','v'),'AIZen':('0.0','-','D'),'Central oracle':('0.75','-',None)}
lab={'Native':'Native OS','Greedy':'Greedy (EWMA)','EpsSticky':r'$\epsilon$-greedy+stick.','TSStat':'Thompson (stationary)','AIZen':'Wi-Fi AI Zen','Central oracle':'Central oracle'}
fig,ax=plt.subplots(1,3,figsize=(7.16,2.3))
for n,(c,l,mk) in sty.items():
    for a,k in zip(ax,('tput','ho_per_h','herd')):
        v=[summarize([m[k] for m in o[M][n]]) for M in Ms]
        a.errorbar(Ms,[x for x,_ in v],yerr=[y for _,y in v],color=c,ls=l,marker=mk,ms=3,lw=1.0,capsize=1.5,elinewidth=0.5,label=lab[n] if k=='tput' else None)
ax[0].set_ylabel('Per-client throughput (Mbps)'); ax[1].set_ylabel('Handoffs per client-hour'); ax[1].set_yscale('log'); ax[2].set_ylabel('Share on busiest BSSID')
ax[0].legend(frameon=False,fontsize=6.2,loc='upper right')
for a,t in zip(ax,['(a)','(b)','(c)']):
    a.set_xlabel('Number of clients'); a.set_xticks(Ms); a.set_title(t,fontsize=8,loc='left')
    for s in ('top','right'): a.spines[s].set_visible(False)
    a.tick_params(labelsize=7,width=0.5,length=2.5)
fig.tight_layout(pad=0.4); fig.savefig(O+'multi.png'); plt.close(fig); print('ok')
