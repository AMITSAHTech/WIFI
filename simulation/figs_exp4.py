import pickle, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from runlib import summarize
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif","Times New Roman","Liberation Serif"],"font.size":8,"axes.linewidth":0.6,"savefig.dpi":600})
O="fig/figs2/"
os.makedirs(O, exist_ok=True)
o=pickle.load(open('exp4.pkl','rb'))
def m(v,k): return summarize([x[k] for x in v])
ls=sorted(o['load_scale']); sty={'Native':('0.55','-','o'),'KV':('0.35','--','s'),'EpsSticky':('0.0',':','^'),'AIZen':('0.0','-','D'),'Oracle':('0.75','-',None)}
lab={'EpsSticky':r'$\epsilon$-greedy+stick.','KV':'802.11k/v/r','AIZen':'Wi-Fi AI Zen','Native':'Native OS','Oracle':'Oracle'}
fig,ax=plt.subplots(1,3,figsize=(7.16,2.35))
for n,(c,l,mk) in sty.items():
    mu=[m(o['load_scale'][v][n],'tput') for v in ls]; ax[0].errorbar(ls,[a for a,_ in mu],yerr=[b for _,b in mu],color=c,ls=l,marker=mk,ms=3,lw=1.0,capsize=1.5,elinewidth=0.5,label=lab[n])
    p=[m(o['load_scale'][v][n],'rtt_p99') for v in ls]; ax[1].plot(ls,[a for a,_ in p],color=c,ls=l,marker=mk,ms=3,lw=1.0)
ax[0].set_xlabel('Load scale (1 = light, 3 = default)'); ax[0].set_ylabel('Throughput (Mbps)'); ax[0].legend(frameon=False,fontsize=6.2,loc='lower left')
ax[1].set_xlabel('Load scale'); ax[1].set_ylabel('p99 RTT (ms)'); ax[1].set_yscale('log')
gs=sorted(o['gamma']); xs=[m(o['gamma'][g]['AIZen'],'ho_per_h')[0] for g in gs]; ys=[m(o['gamma'][g]['AIZen'],'tput')[0] for g in gs]
ax[2].plot(xs,ys,color='0',marker='D',ms=3.5,lw=1.0)
for g,x,y in zip(gs,xs,ys): ax[2].annotate(r'$\gamma$=%s'%('none' if g==1.0 else g),(x,y),textcoords='offset points',xytext=(4,3),fontsize=6.3)
ax[2].set_xscale('log'); ax[2].set_xlabel('Handoffs per hour'); ax[2].set_ylabel('Throughput (Mbps)')
for a,t in zip(ax,['(a)','(b)','(c)']):
    a.set_title(t,fontsize=8,loc='left')
    for s in ('top','right'): a.spines[s].set_visible(False)
    a.tick_params(labelsize=7,width=0.5,length=2.5)
fig.tight_layout(pad=0.4); fig.savefig(O+'sens.png'); plt.close(fig); print('ok')
print({v:(round(m(o['p_mbb'][v]['AIZen'],'sess_ho_day')[0],2),round(m(o['p_mbb'][v]['AIZen'],'disr_s_day')[0],1)) for v in o['p_mbb']})
