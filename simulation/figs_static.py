import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif","Times New Roman","Liberation Serif"],"font.size":8,"mathtext.fontset":"stix","axes.linewidth":0.6,"savefig.dpi":600})
O="fig/figs2/"
os.makedirs(O, exist_ok=True)
# ---- FSM
fig,ax=plt.subplots(figsize=(7.16,2.75)); ax.set_xlim(0,100); ax.set_ylim(-5,40); ax.axis("off")
def box(x,y,w,h,title,lines,fc="0.95"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.3,rounding_size=1.2",fc=fc,ec="0.1",lw=0.7,zorder=2))
    ax.text(x+w/2,y+h-2.3,title,ha="center",va="center",fontsize=7.6,weight="bold",zorder=3)
    ax.text(x+w/2,y+h-4.9,"\n".join(lines),ha="center",va="top",fontsize=6.7,linespacing=1.25,zorder=3)
box(1,19,18,14,"S1: Steady",["Monitor active link","EWMA anomaly test","Sample posteriors"])
box(27,19,22,14,"S2: Safety gate",["Inhibit if call or bulk transfer","is active (unless link degraded)","or dwell timer not expired"])
box(55,19,19,14,"S3: Make",["Associate target BSSID","on a second interface","Verify gateway reachability"])
box(82,19,17,14,"S4: Break",["Release old BSSID","Flush ARP and routes","Hand traffic to new link"])
box(82,1,17,12,"S5: Meta-learning gate",["Measure realized change","Adapt gate $G$"],fc="0.88")
box(55,1,19,12,"Abort / failover",["Keep prior BSSID","Backoff (dwell restart)"],fc="0.88")
def seg(pts,label=None,lp=None,ha="center"):
    for a,b in zip(pts[:-2],pts[1:-1]): ax.plot([a[0],b[0]],[a[1],b[1]],color="0.05",lw=0.8,zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2],pts[-1],arrowstyle="-|>",mutation_scale=7,lw=0.8,color="0.05",shrinkA=0,shrinkB=0,zorder=1))
    if label: ax.text(*lp,label,fontsize=6.5,ha=ha,va="center",style="italic",bbox=dict(fc="w",ec="none",pad=0.6),zorder=4)
seg([(19.4,26),(26.6,26)],r"$\hat\Delta > G$",(23,29.4))
seg([(49.4,26),(54.6,26)],"approved",(52,29.4))
seg([(74.4,26),(81.6,26)],"verified",(78,29.4))
seg([(90.5,18.6),(90.5,13.4)])
seg([(64.5,18.6),(64.5,13.4)],"verify fails",(65.6,16),ha="left")
seg([(38,33.4),(38,37.5),(10,37.5),(10,33.6)],"inhibited",(24,37.5))
seg([(54.6,7),(14,7),(14,18.6)],"revert",(34,7))
seg([(90.5,0.6),(90.5,-3),(6,-3),(6,18.6)],"gate updated",(48,-3))
fig.subplots_adjust(0.003,0.003,0.997,0.997); fig.savefig(O+"fsm.png"); plt.close(fig)
