"""Plot saved band-resolved and fixed-field orbital-source energy comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



def plot_source_channels(args,data):
    rows=data['channels']
    if not args.experiment_beta or not args.experiment_sigma:
        raise ValueError('Multichannel comparison requires both figure-derived experimental datasets')
    beta=json.loads(args.experiment_beta.read_text())['panels']
    sigma=json.loads(args.experiment_sigma.read_text())['panels']
    labels=args.channel_labels or [p['label'] for p in beta]
    if len(rows)!=len(labels) or len(beta)!=len(rows) or len(sigma)!=len(rows):
        raise ValueError('Calculated and experimental channel counts must match')
    def experimental(panels,label):
        p=next(p for p in panels if p['label']==label)['points']
        return np.array([r['x'] for r in p]),np.array([r['y'] for r in p])
    def interpolate(x,y):
        if not x[0]<=args.reference_photon_energy<=x[-1]:
            raise ValueError('Reference energy outside compared data')
        return np.interp(args.reference_photon_energy,x,y)
    names=['Dyson','canonical_same_norm']
    norm={name:sum(interpolate(np.asarray(r['photon_energies_ev']),r['results'][name]['cross_sections_Mb']) for r in rows) for name in names}
    norm_exp=sum(interpolate(*experimental(sigma,label)) for label in labels)
    if args.observable=='beta':
        fig,axes=plt.subplots((len(rows)+1)//2,2,figsize=(7.2,2.5*((len(rows)+1)//2)),sharex=True,constrained_layout=True)
        for ax,row,label in zip(np.asarray(axes).ravel(),rows,labels):
            for name,color,style in zip(names,['#2166ac','#b2182b'],['-','--']):
                ax.plot(row['photon_energies_ev'],row['results'][name]['beta'],color=color,ls=style,
                        label='Dyson' if name=='Dyson' else 'Canonical, same norm')
            xe,ye=experimental(beta,label);ax.plot(xe,ye,'o',color='.2',ms=3,label='Experiment')
            ax.set_title(label,loc='right',fontsize=10);ax.set_ylabel('Anisotropy $\\beta$')
            ax.spines[['top','right']].set_visible(False);ax.set_xlim(15,55)
        for ax in np.asarray(axes).ravel()[len(rows):]:ax.set_visible(False)
        for ax in np.asarray(axes).reshape(-1,2)[-1]:ax.set_xlabel('Photon energy / eV')
        np.asarray(axes).ravel()[0].legend(frameon=False,fontsize=8,loc='lower right')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(args.output.with_suffix('.pdf'));fig.savefig(args.output.with_suffix('.png'),dpi=180)
        plt.close(fig)
        return
    fig,axes=plt.subplots(len(rows),2,figsize=(7.2,2.0*len(rows)),sharex=True,constrained_layout=True)
    for i,(row,label) in enumerate(zip(rows,labels)):
        x=np.array(row['photon_energies_ev'])
        for name,color,style in zip(names,['#2166ac','#b2182b'],['-','--']):
            r=row['results'][name]
            axes[i,0].plot(x,np.asarray(r['cross_sections_Mb'])/norm[name],color=color,ls=style,label='Dyson' if name=='Dyson' else 'Canonical, same norm')
            axes[i,1].plot(x,r['beta'],color=color,ls=style)
        xe,ye=experimental(sigma,label);axes[i,0].plot(xe,ye/norm_exp,'o',color='.2',ms=2.5,label='Experiment')
        xe,ye=experimental(beta,label);axes[i,1].plot(xe,ye,'o',color='.2',ms=2.5)
        for j,ax in enumerate(axes[i]):
            ax.set_title(label,loc='right',fontsize=10)
            ax.spines[['top','right']].set_visible(False);ax.set_xlim(15,55)
        axes[i,0].set_ylabel('Relative partial strength')
        axes[i,1].set_ylabel('Anisotropy $\\beta$')
    axes[0,0].legend(frameon=False,fontsize=7,loc='upper center')
    for ax in axes[-1]:ax.set_xlabel('Photon energy / eV')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.pdf'));fig.savefig(args.output.with_suffix('.png'),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--band-observables',type=Path)
    p.add_argument('--source-comparison',type=Path)
    p.add_argument('--observable',choices=['both','beta'],default='both')
    p.add_argument('--experiment-beta',type=Path)
    p.add_argument('--experiment-sigma',type=Path)
    p.add_argument('--channel-labels',nargs='+')
    p.add_argument('--reference-photon-energy',type=float,default=32.5)
    p.add_argument('--bands',nargs='+',default=['100','010','001'])
    p.add_argument('--case',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if (args.band_observables is None)==(args.source_comparison is None):
        p.error('Supply exactly one saved comparison')
    if args.source_comparison:
        data=json.loads(args.source_comparison.read_text())
        if 'channels' in data:
            plot_source_channels(args,data)
            return
    fig,axes=plt.subplots(2,1,figsize=(6.6,5.4),sharex=True,constrained_layout=True)
    if args.band_observables:
        data=json.loads(args.band_observables.read_text());x=data['photon_energies_ev']
        row=data['cases'][args.case]
        models=list(row['models']);colors=['#2166ac','#b2182b','#16866a','#986bb5']
        for i,model in enumerate(models):
            style=['--','-',':','-.'][i%4]
            for band,color in zip(args.bands,colors):
                line=row['models'][model]['lines'][band]
                axes[0].plot(x,100*np.array(line['ratio_to_reference']),ls=style,color=color,lw=1.5)
                axes[1].plot(x,line['beta'],ls=style,color=color,lw=1.5)
            axes[1].plot(x,row['models'][model]['lines']['reference']['beta'],ls=style,color='.3',lw=1.4)
        handles=[Line2D([],[],color=c,label=b) for b,c in zip(args.bands,colors)]
        handles.append(Line2D([],[],color='.3',label='Origin (lower panel)'))
        handles += [Line2D([],[],color='.1',ls=['--','-',':','-.'][i%4],label=model.replace('ZAFF_DVR','ZAFF/DVR')) for i,model in enumerate(models)]
        axes[0].legend(handles=handles,frameon=False,fontsize=8,ncol=2,loc='center right')
        axes[0].set_yscale('log');axes[0].set_ylabel('Integrated strength / % of origin')
        axes[1].set_ylabel('Photoelectron anisotropy $\\beta$')
    else:
        data=json.loads(args.source_comparison.read_text());x=data['electron_energies_hartree']
        x=np.array(x)*27.211386245988
        styles=[('#2166ac','-'),('#b2182b','--'),('#777777',':')]
        labels={'Dyson':'Correlated Dyson','canonical_same_norm':'Canonical, same norm','canonical_unit_norm':'Canonical, unit norm'}
        for (name,row),(color,style) in zip(data['results'].items(),styles):
            axes[0].plot(x,row['cross_sections_Mb'],color=color,ls=style,lw=1.5,label=labels.get(name,name))
            axes[1].plot(x,row['beta'],color=color,ls=style,lw=1.5)
        axes[0].legend(frameon=False,fontsize=8)
        axes[0].set_ylabel('Fixed-nuclei cross section / Mb')
        axes[1].set_ylabel('Photoelectron anisotropy $\\beta$')
    for ax,label in zip(axes,['(a)','(b)']):
        ax.text(-.08,1.02,label,transform=ax.transAxes,fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    axes[1].set_xlabel('Photon energy / eV' if args.band_observables else 'Electron kinetic energy / eV')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.pdf'));fig.savefig(args.output.with_suffix('.png'),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
