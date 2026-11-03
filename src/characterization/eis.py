import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents
                 if (parent / "styles" / "manuscript.mplstyle").is_file())
EIS_DATA = REPO_ROOT / "data" / "characterization" / "EIS"
STYLE = REPO_ROOT / "styles" / "manuscript.mplstyle"
EIS_SCREEN = 5.0  # Descriptive THD/NSD screen.
SELECTED_SCANS = {"BL": 5, "BO86": 6}


def load_eis_plot_data(data_dir=EIS_DATA):
    """Load point timing, scan metrics and the two matched recovery spectra."""
    data_dir = Path(data_dir)
    frequency = pd.read_csv(data_dir / "frequency_measurements.csv")
    metrics = pd.read_csv(data_dir / "scan_quality_metrics.csv")
    selected = {}
    for condition, scan in SELECTED_SCANS.items():
        path = data_dir / "numerical" / f"{condition}_07_GEIS.csv.gz"
        spectrum = pd.read_csv(path)
        spectrum = spectrum.loc[spectrum["scan"] == scan].sort_values("freq_Hz").copy()
        if len(spectrum) != 31:
            raise ValueError(f"{condition}: expected 31 points in scan {scan}")
        if not np.allclose(spectrum.ReZ_cell_ohm,
                           spectrum.ReZ_ohm + spectrum.ReZ_ce_ohm,
                           rtol=1e-5, atol=.05):
            raise ValueError(f"{condition}: full-cell real impedance sum fails")
        selected[condition] = spectrum
    return frequency, metrics, selected


def _condition_color(condition):
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return palette[0 if condition == "BL" else 1]


def eis_style(ax,xlabel,ylabel,title=None):
    ax.set_xlabel(xlabel,fontsize=7,labelpad=3)
    ax.set_ylabel(ylabel,fontsize=7,labelpad=3)
    if title:ax.set_title(title,fontsize=7,pad=5)
    ax.tick_params(labelsize=5,width=.5,pad=2)
    ax.set_box_aspect(1);ax.grid(False)
    for spine in ax.spines.values():spine.set_linewidth(.5)

def eis_shared_legend(fig,y=.02):
    fig.legend(handles=[Line2D([],[],color=_condition_color(c),marker='o',lw=.6,
                              markersize=3,label=(r'Pareto front condition \#86' if plt.rcParams['text.usetex'] else 'Pareto front condition #86') if c == 'BO86' else 'Baseline condition') for c in ['BL','BO86']],
               loc='lower center',bbox_to_anchor=(.5,y),ncol=2,
               frameon=False,fontsize=5,handlelength=1.7)

def eis_frequency_points(frequency,hz):
    d=frequency.loc[(frequency.condition.isin(['BL','BO86'])) &
                    (frequency.stage!='pre') & np.isclose(frequency.nominal_frequency_Hz,hz)].copy()
    return d.sort_values(['condition','time_after_cp_min'])

def eis_recovery(ax,data,column):
    for condition in ['BL','BO86']:
        d=data.loc[data.condition==condition];color=_condition_color(condition)
        ax.plot(d.time_after_cp_min,d[column]/1000,'o-',color=color,
                lw=.6,ms=3,mec=color,mfc=color,mew=.6,zorder=2)


def plot_eis_recovery_comparison(frequency, metrics, selected):
    fig,axes=plt.subplots(2,2,figsize=(6.5,6.5),dpi=600)
    fig.subplots_adjust(left=.10,right=.91,bottom=.13,top=.98,wspace=.25,hspace=.20)
    one=eis_frequency_points(frequency,1.0)
    eis_recovery(axes[0,0],one,'Zabs_ohm')
    eis_style(axes[0,0],'Time after current interruption (min)',
              r'$|Z|$ (k$\Omega$)','Working electrode, 1 Hz')
    axes[0,0].set_xlim(0,65);axes[0,0].set_ylim(bottom=0)
    for condition in ['BL','BO86']:
        d=one.loc[one.condition==condition]
        axes[0,1].plot(d.time_after_cp_min,d.Emean_V,'o-',
                        color=_condition_color(condition),ms=3,lw=.6)
    eis_style(axes[0,1],'Time after current interruption (min)',
              'WE potential vs. Pt (V)','Potential at each 1 Hz measurement')
    axes[0,1].set_xlim(0,65)
    for condition in ['BL','BO86']:
        d=selected[condition].loc[lambda x:x.freq_Hz>=.999]
        color=_condition_color(condition)
        axes[1,0].plot(d.ReZ_ohm/1000,d.minusImZ_ohm/1000,'o-',color=color,lw=.6,ms=2.3)
        axes[1,1].loglog(d.freq_Hz,d.Zabs_ohm/1000,color=color,lw=.7)
    eis_style(axes[1,0],r"$Z'$ (k$\Omega$)",r"$-Z''$ (k$\Omega$)",'Working electrode, near 50 min')
    eis_style(axes[1,1],'Frequency (Hz)',r'$|Z|$ (k$\Omega$)','Working electrode, near 50 min')
    phase=axes[1,1].twinx();phase.set_box_aspect(1)
    axes[1,1].yaxis.set_minor_formatter(NullFormatter())
    for condition in ['BL','BO86']:
        d=selected[condition].loc[lambda x:x.freq_Hz>=.999]
        phase.semilogx(d.freq_Hz,d.phase_deg,color=_condition_color(condition),ls='--',lw=.7)
    phase.set_ylabel('Phase (degrees)',fontsize=7,labelpad=3)
    phase.tick_params(labelsize=5,width=.5);phase.set_ylim(-90,10)
    axes[1,1].text(.05,.06,'Solid: magnitude\nDashed: phase',transform=axes[1,1].transAxes,fontsize=5,va='bottom')
    # Zoom contains both formulations at the same frequencies, not a separate fit.
    zoom=axes[1,0].inset_axes([.47,.10,.49,.37])
    for condition in ['BL','BO86']:
        d=selected[condition].loc[lambda x:x.freq_Hz>=.999]
        zoom.plot(d.ReZ_ohm/1000,d.minusImZ_ohm/1000,'o-',
                  color=_condition_color(condition),lw=.55,ms=1.8)
    pareto=selected['BO86'].loc[lambda x:x.freq_Hz>=.999]
    x=pareto.ReZ_ohm.to_numpy()/1000;y=pareto.minusImZ_ohm.to_numpy()/1000
    dx=max(np.ptp(x),.1);dy=max(np.ptp(y),.1)
    # Shared limits retain the baseline high-frequency segment as well as #86.
    zoom.set_xlim(1.35,max(3.2,x.max()+.16*dx))
    zoom.set_ylim(-.10,max(1.3,y.max()+.20*dy))
    # Inset uses the same kOhm units as the parent axes; no title.
    zoom.tick_params(labelsize=4,width=.4,length=2,pad=1)
    from matplotlib.ticker import MaxNLocator
    zoom.xaxis.set_major_locator(MaxNLocator(3));zoom.yaxis.set_major_locator(MaxNLocator(3))
    for spine in zoom.spines.values():spine.set_linewidth(.4)
    axes[1,0].indicate_inset_zoom(zoom,edgecolor='.45',alpha=.65,linewidth=.5)
    # Highlight the exact 1 Hz points corresponding to the detailed spectra.
    for condition,scan in [('BL',5),('BO86',6)]:
        chosen=one.loc[(one.condition==condition)&(one.stage=='late')&(one['scan']==scan)]
        if len(chosen)!=1:raise ValueError('Selected 1 Hz observation is not unique')
        p=chosen.iloc[0];color=_condition_color(condition)
        for ax,value in [(axes[0,0],p.Zabs_ohm/1000),(axes[0,1],p.Emean_V)]:
            ax.plot(p.time_after_cp_min,value,marker='D',ls='none',
                    ms=5,mfc=color,mec='black',mew=.65,zorder=5)
    for ax in axes.flat:ax.set_title('')
    zoom.set_title('')
    eis_shared_legend(fig,y=.04)
    return fig,axes






def plot_eis_quality_context(frequency,metrics,selected):
    fig,axes=plt.subplots(2,2,figsize=(6.5,6.5),dpi=600)
    fig.subplots_adjust(left=.11,right=.96,bottom=.14,top=.94,wspace=.48,hspace=.52)
    one=eis_frequency_points(frequency,1.0)
    for condition in ['BL','BO86']:
        color=_condition_color(condition)
        m=metrics.loc[(metrics.condition==condition)&(metrics.stage!='pre')].copy()
        time=(m.start_after_cp_s+m.end_after_cp_s)/120
        axes[0,0].plot(time,m.HF_median/1000,'o-',color=color,ms=3,lw=.6)
        d=one.loc[one.condition==condition]
        axes[0,1].plot(d.time_after_cp_min,d.THD_E_pct,'o-',color=color,ms=3,lw=.6)
        axes[0,1].plot(d.time_after_cp_min,d.NSD_E_pct,'s--',color=color,ms=2.5,lw=.6)
        axes[1,0].plot(time,m.Emean_span_mV,'o-',color=color,ms=3,lw=.6)
        scan=selected[condition]
        axes[1,1].loglog(scan.freq_Hz,scan.Zabs_ohm/1000,'o-',color=color,ms=2,lw=.6)
        flagged=(scan.THD_E_pct>EIS_SCREEN)|(scan.NSD_E_pct>EIS_SCREEN)
        axes[1,1].loglog(scan.loc[flagged,'freq_Hz'],scan.loc[flagged,'Zabs_ohm']/1000,
                         'o',mfc='white',mec=color,ms=3,mew=.6)
    eis_style(axes[0,0],'Scan midpoint after interruption (min)',r'High-frequency proxy (k$\Omega$)','Working electrode')
    eis_style(axes[0,1],'Time after current interruption (min)','Voltage THD / NSD (%)','Quality indicators at 1 Hz')
    axes[0,1].axhline(EIS_SCREEN,color='.5',ls=':',lw=.5)
    axes[0,1].text(.55,.94,'Solid: THD\nDashed: NSD',transform=axes[0,1].transAxes,fontsize=5,va='top')
    eis_style(axes[1,0],'Scan midpoint after interruption (min)','WE potential span during scan (mV)','Within-scan potential variation')
    eis_style(axes[1,1],'Frequency (Hz)',r'$|Z|$ (k$\Omega$)','Full measured frequency range, near 50 min')
    for ax in axes.flat[:3]:ax.set_xlim(0,65);ax.set_ylim(bottom=0)
    for ax in axes.flat:ax.set_title('')
    eis_shared_legend(fig)
    return fig,axes




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Display both recovery figures.")
    parser.add_argument("--no-tex", action="store_true", help="Use Matplotlib math text.")
    args = parser.parse_args()
    data = load_eis_plot_data()
    one_hz = eis_frequency_points(data[0], 1.0)
    print(one_hz[["condition", "time_after_cp_min", "Zabs_ohm", "phase_deg", "Emean_V"]]
          .to_string(index=False))
    if args.plot:
        with plt.style.context(STYLE):
            if args.no_tex:
                plt.rcParams["text.usetex"] = False
            plot_eis_recovery_comparison(*data)
            plot_eis_quality_context(*data)
            plt.show()


if __name__ == "__main__":
    main()
