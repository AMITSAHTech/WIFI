import pickle, numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env import *; from policies import *; from runlib import *
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif","Times New Roman","Liberation Serif"],"font.size":8,"axes.linewidth":0.6,"savefig.dpi":600,"mathtext.fontset":"stix"})
O="fig/figs2/"
os.makedirs(O, exist_ok=True)
d=pickle.load(open('exp1.pkl','rb')); res=d['res']; pool=d['pool']
sty={'Native':('0.55','-',1.0),'KV':('0.35','--',1.0),'EpsSticky':('0.0',':',1.3),'TSStat':('0.45','-.',1.0),'AIZen':('0.0','-',1.6),'Oracle':('0.7','-',0.9)}
lab={'EpsSticky':r'$\epsilon$-greedy+stickiness','KV':'802.11k/v/r-assisted','AIZen':'Wi-Fi AI Zen','TSStat':'Thompson (stationary)','Native':'Native OS','Oracle':'Oracle'}
# ---- Fig 4: CDFs
fig,ax=plt.subplots(1,2,figsize=(7.16,2.45))
for n,(c,ls,lw) in sty.items():
    x=np.sort(pool[n]['T']); y=np.arange(1,len(x)+1)/len(x)
    ax[0].plot(x[::max(1,len(x)//800)],y[::max(1,len(x)//800)],color=c,ls=ls,lw=lw,label=lab[n])
    x=np.sort(pool[n]['R']); y=np.arange(1,len(x)+1)/len(x)
    ax[1].plot(x[::max(1,len(x)//800)],y[::max(1,len(x)//800)],color=c,ls=ls,lw=lw)
ax[0].set_xlabel('Effective throughput (Mbps)'); ax[0].set_ylabel('CDF'); ax[0].set_title('(a)',fontsize=8,loc='left')
ax[1].set_xscale('log'); ax[1].set_xlabel('Round-trip time (ms)'); ax[1].set_title('(b)',fontsize=8,loc='left'); ax[1].set_xlim(3,400)
ax[0].legend(frameon=False,fontsize=6.5,loc='upper left')
for a in ax:
    for s in ('top','right'): a.spines[s].set_visible(False)
    a.tick_params(labelsize=7,width=0.5,length=2.5)
fig.tight_layout(pad=0.4); fig.savefig(O+'cdf.png'); plt.close(fig)
# ---- Fig 5: ablation (paired differences vs full system)
names=[n for n in res if n not in ('Oracle','Native','RSSI-best','KV','EpsSticky','DUCB','SWUCB','EpsSticky+MBB','TSStat','SWTS','AIZen')]
def pd(n,k): return summarize([m[k]-f[k] for m,f in zip(res[n],res['AIZen'])])
fig,ax=plt.subplots(1,3,figsize=(7.16,2.3),sharey=True); y=np.arange(len(names))[::-1]
for a_,k,xl in ((ax[0],'tput',r'$\Delta$ throughput (Mbps)'),(ax[1],'rtt_p99',r'$\Delta$ p99 RTT (ms)')):
    v=[pd(n,k) for n in names]
    a_.barh(y,[m for m,_ in v],xerr=[c for _,c in v],color='0.6',ecolor='k',error_kw=dict(lw=0.6),height=0.6); a_.axvline(0,color='k',lw=0.6); a_.set_xlabel(xl)
ho=[summarize([m['ho_per_h'] for m in res[n]]) for n in names]
ax[2].barh(y,[m for m,_ in ho],xerr=[c for _,c in ho],color='0.6',ecolor='k',error_kw=dict(lw=0.6),height=0.6); ax[2].set_xscale('log'); ax[2].set_xlabel('Handoffs per hour')
ax[2].axvline(summarize([m['ho_per_h'] for m in res['AIZen']])[0],color='k',ls=':',lw=0.8)
ax[0].set_yticks(y); ax[0].set_yticklabels(names,fontsize=6.5)
for a_,t in zip(ax,['(a)','(b)','(c)']): a_.set_title(t,fontsize=8,loc='left')
for a_ in ax:
    for sp in ('top','right'): a_.spines[sp].set_visible(False)
    a_.tick_params(labelsize=7,width=0.5,length=2.5)
fig.tight_layout(pad=0.4); fig.savefig(O+'ablation.png'); plt.close(fig)
# ---- Fig 3: real posterior snapshots (seed 0, tuned AI Zen)
from scipy.stats import beta as Bd
B=tuned(); env=make_env(0,days=3)
snap={}
class Rec(AIZen):
    def act(s,t,cur):
        if t in (int(40*120+0), int(60*120+0)) or t in (int(35*120),int(45*120)): snap[t]=(s.a.copy(),s.b.copy(),s._score(t).copy())
        return super().act(t,cur)
hrs=[(1*24+12),(1*24+21)]   # day 2 noon and 9 pm
steps=[int(h*120) for h in hrs]
class Rec2(AIZen):
    def act(s,t,cur):
        if t in steps: snap[t]=(s.a.copy(),s.b.copy(),s._score(t).copy())
        return super().act(t,cur)
p=Rec2(seed=0,**B['AIZen']); run_assoc(env,p)
fig,ax=plt.subplots(1,2,figsize=(3.45*2,2.2),sharey=True); x=np.linspace(0.001,0.999,500)
cols=['0.0','0.0','0.5','0.5']; lss=['-','--','-','--']; nm=['AP1 2.4 GHz','AP1 5 GHz','AP2 2.4 GHz','AP2 5 GHz']
for k,(t,a_) in enumerate(zip(steps,ax)):
    A,Bb,sc=snap[t]; kap=B['AIZen']['kappa']
    for i in range(4):
        a_.plot(x,Bd.pdf(x,A[i]+kap*sc[i],Bb[i]+kap*(1-sc[i])),color=cols[i],ls=lss[i],lw=1.1,label=nm[i])
        tr=reward(env['Ttrue'][t,i],env['rtt_mean'][t,i],100*env['ploss'][t,i],env['P']['tcap'])
        a_.axvline(tr,color=cols[i],ls=lss[i],lw=0.6,alpha=0.8)
    a_.set_xlabel(r'Expected reward $\theta$'); a_.set_title(['(a) Day 2, 12:00','(b) Day 2, 21:00'][k],fontsize=8,loc='left'); a_.set_xlim(0.4,1.0)
    for s in ('top','right'): a_.spines[s].set_visible(False)
    a_.tick_params(labelsize=7,width=0.5,length=2.5)
ax[0].set_ylabel('Posterior density'); ax[0].legend(frameon=False,fontsize=6.3,loc='upper left')
fig.tight_layout(pad=0.4); fig.savefig(O+'posterior.png'); plt.close(fig)
print('figs ok')
