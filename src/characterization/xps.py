from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

SAMPLES = ("BL-1", "BO-1")
REGIONS = ("F 1s", "C 1s", "O 1s", "B 1s", "N 1s", "Li 1s")
REGION_LIMITS = {"F 1s":(693,677),"C 1s":(297,278),"O 1s":(542,524),
                 "B 1s":(198.753,182),"N 1s":(408,390),"Li 1s":(65,49)}

@dataclass
class XPSData:
    spectra: dict
    elemental: pd.DataFrame
    fit_parameters: pd.DataFrame
    positions: pd.DataFrame
    inventory: pd.DataFrame


def repository_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate/"data").is_dir() and (candidate/"styles").is_dir():return candidate
    raise FileNotFoundError("Cannot locate repository data/ and styles/ directories.")


def _columns(table,required,label):
    missing=set(required)-set(table.columns)
    if missing:raise ValueError(f"{label}: missing columns {sorted(missing)}")


def load_xps(root=None):
    """Reconstruct original wide CasaXPS frames from repository long tables."""
    root=Path(root).resolve() if root is not None else repository_root()
    directory=root/"data/characterization/XPS"
    def read(name):return pd.read_csv(directory/name,float_precision="round_trip")
    raw=read("spectra.csv");components=read("fitted_components.csv")
    params=read("fit_parameters.csv");elemental=read("elemental_quantification.csv")
    positions=read("positions.csv");inventory=read("spectral_inventory.csv")
    # Fixed scope: unselected positions do not enter checks, analysis or plots.
    raw=raw.loc[raw['sample'].isin(SAMPLES)].copy()
    components=components.loc[components['sample'].isin(SAMPLES)].copy()
    params=params.loc[params['sample'].isin(SAMPLES)].copy()
    elemental=elemental.loc[elemental['Location ID'].isin(SAMPLES)].copy()
    positions=positions.loc[positions['Location ID'].isin(SAMPLES)].copy()
    inventory=inventory.loc[inventory['sample'].isin(SAMPLES)].copy()
    samples=SAMPLES
    _columns(raw,["sample","region","source_row_1based","binding_energy_eV","intensity_CPS"],"Spectra")
    _columns(components,["sample","region","source_row_1based","binding_energy_eV","series","value_CPS"],"Fit curves")
    _columns(params,["sample","region","component_label","position_eV","exported_area"],"Fit parameters")
    _columns(elemental,["Region","Location ID","Reported atomic concentration (%)"],"Elemental analysis")
    if len(set(samples))!=len(samples) or not samples:raise ValueError("Supply distinct sample IDs.")
    missing=set(samples)-set(raw['sample'])
    if missing:raise ValueError(f"Samples absent from repository spectra: {sorted(missing)}. Both selected samples are required.")
    if raw.duplicated(['sample','region','source_row_1based']).any():raise ValueError('Duplicate spectral rows.')
    if components.duplicated(['sample','region','series','source_row_1based']).any():raise ValueError('Duplicate fit-curve rows.')
    spectra={}
    for sample in samples:
        for region in (*REGIONS,'Survey'):
            frame=raw.loc[raw['sample'].eq(sample)&raw['region'].eq(region)].sort_values('source_row_1based')
            if frame.empty:raise ValueError(f'Missing spectrum: {sample}, {region}.')
            wide=frame[['binding_energy_eV','intensity_CPS']].reset_index(drop=True).copy()
            if not np.isfinite(wide.to_numpy()).all() or not np.all(np.diff(wide.binding_energy_eV)<0):
                raise ValueError(f'Expected finite descending energies: {sample}, {region}.')
            fitted=components.loc[components['sample'].eq(sample)&components['region'].eq(region)]
            for series in fitted['series'].drop_duplicates():
                values=fitted.loc[fitted['series'].eq(series)].sort_values('source_row_1based')
                if not np.array_equal(values.source_row_1based.to_numpy(),frame.source_row_1based.to_numpy()) or not np.array_equal(values.binding_energy_eV.to_numpy(),wide.binding_energy_eV.to_numpy()):
                    raise ValueError(f'Misaligned fit grid: {sample}, {region}, {series}.')
                wide[series]=values.value_CPS.to_numpy()
            if not np.isfinite(wide.to_numpy()).all():raise ValueError('Nonfinite fit values.')
            peak_positions={}
            metadata=params.loc[params['sample'].eq(sample)&params['region'].eq(region)]
            if metadata.component_label.duplicated().any():raise ValueError('Duplicate component metadata.')
            for row in metadata.itertuples(index=False):
                matches=[s for s in wide.columns[2:] if s.rsplit(':',1)[-1]==row.component_label]
                if len(matches)!=1:raise ValueError(f'Cannot match component metadata: {sample}, {region}, {row.component_label}.')
                if not np.isfinite(row.position_eV):raise ValueError('Nonfinite component position.')
                peak_positions[matches[0]]=float(row.position_eV)
            if set(wide.columns[2:])-{'Background','Envelope'}!=set(peak_positions):
                raise ValueError(f'Missing component positions: {sample}, {region}.')
            wide.attrs['component_positions']=peak_positions
            spectra[sample,region]=wide
    if elemental.duplicated(['Region','Location ID']).any():raise ValueError('Duplicate elemental entries.')
    values=elemental['Reported atomic concentration (%)'].dropna()
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid reported atomic percentages.')
    pivot=elemental.pivot(index='Region',columns='Location ID',values='Reported atomic concentration (%)').reindex(columns=list(samples))
    pivot.index.name='region';pivot.columns.name='sample'
    return XPSData(spectra,pivot,params.loc[params['sample'].isin(samples)].copy(),
                   positions.loc[positions['Location ID'].isin(samples)].copy(),
                   inventory.loc[inventory['sample'].isin(samples)].copy())


def elemental_ratios(data):
    """Original F/C, O/C and B/C arithmetic, without altering atomic percentages."""
    values=data.elemental.loc[['C 1s','O 1s','F 1s','B 1s'],list(SAMPLES)]
    if not np.isfinite(values.to_numpy()).all() or (values.loc['C 1s']<=0).any():
        raise ValueError('Finite elemental values and positive carbon values are required.')
    return values.loc[['F 1s','O 1s','B 1s']].div(values.loc['C 1s'])


def component_area_fractions(data):
    """Recompute fractions from reported component areas; these are not atomic %."""
    result=data.fit_parameters.loc[data.fit_parameters['sample'].isin(SAMPLES)].copy()
    area=result['exported_area'].to_numpy(float)
    if not np.isfinite(area).all() or (area<0).any():raise ValueError('Invalid component areas.')
    totals=result.groupby(['sample','region'])['exported_area'].transform('sum')
    if (totals<=0).any():raise ValueError('Nonpositive region area sum.')
    result['Recalculated region area']=totals
    result['Recalculated component fraction']=result['exported_area']/totals
    if 'fraction_of_existing_fit_area' in result:
        np.testing.assert_allclose(result['Recalculated component fraction'],result['fraction_of_existing_fit_area'],rtol=1e-12,atol=1e-12)
    return result


def spectral_quality(data):
    """Report grid sizes and closure of exported envelope/component/background curves."""
    records=[]
    for (sample,region),d in data.spectra.items():
        if sample not in SAMPLES:continue
        curves=[x for x in d.columns[2:] if x not in ('Background','Envelope')]
        closure=np.nan
        if curves and 'Background' in d and 'Envelope' in d:
            calculated=d[curves].sum(axis=1)-(len(curves)-1)*d['Background']
            closure=float(np.max(np.abs(calculated-d['Envelope'])))
        records.append({'sample':sample,'region':region,'spectral_points':len(d),
                        'components':len(curves),'envelope_closure_max_CPS':closure})
    return pd.DataFrame(records)


def run_xps(root=None,*,output_dir=None):
    """Load and summarize data; export numerical tables on request."""
    data=load_xps(root)
    tables={'reported_atomic_percent':data.elemental.reset_index(),
            'atomic_ratios':elemental_ratios(data).reset_index(),
            'component_area_fractions':component_area_fractions(data),
            'spectral_quality':spectral_quality(data)}
    if output_dir is not None:
        destination=Path(output_dir)
        if not destination.is_absolute():destination=(Path(root) if root is not None else repository_root())/destination
        destination.mkdir(parents=True,exist_ok=True)
        for name,table in tables.items():table.to_csv(destination/(name+'.csv'),index=False)
    return data,tables


class _XPSPlotter:
    """Original notebook artists with figure state supplied explicitly."""
    def __init__(self):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MaxNLocator
        self.plt=plt;self.Line2D=Line2D;self.MaxNLocator=MaxNLocator
        palette=plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.colors={s:palette[i] for i,s in enumerate(SAMPLES)}
        self.names={'BL-1':'Baseline condition','BO-1':r'Pareto front condition \#86' if plt.rcParams['text.usetex'] else 'Pareto front condition #86'}
        self.limits=REGION_LIMITS

    def finish_axis(self, ax, region):
        ax.set_xlim(*self.limits[region])
        ax.set_title(region, fontsize=7, pad=4)
        ax.set_xlabel("Binding energy (eV)", fontsize=7, labelpad=3)
        ax.set_ylabel(r"Intensity (counts s$^{-1}$)", fontsize=7, labelpad=3)
        ax.tick_params(labelsize=5, width=.5, pad=2)
        ax.xaxis.set_major_locator(self.MaxNLocator(4))
        ax.yaxis.set_major_locator(self.MaxNLocator(4))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize(5)
        ax.grid(False)
        for spine in ax.spines.values(): spine.set_linewidth(.5)

    def plot_region(self, ax, spectra, region, samples, fits=True, annotate=False):
        """Original proposed assignments; exported peak positions shown as dotted lines."""
        assignments={
            'C 1s':[(285,'C--C / C--H',(.64,.88)),(287,'C--O',(.43,.49))],
            'O 1s':[(533,'O--C',(.50,.87))],
            'F 1s':[(685,'LiF',(.56,.49)),(687,r'B--F / BF$_4$-derived',(.40,.88))],
            'Li 1s':[(56.18, 'LiF?', (.72, .82)), (57.24, 'Salt-associated Li', (.40, .94))],
            'B 1s':[(196,r'B--F / BF$_4$-derived',(.30,.88)),(193,'B--O',(.42,.39))],
            'N 1s':[(402,r'NH$_4^{+}$-like N?',(.34,.86))],
        }
        lo,hi=sorted(self.limits[region]);labels=set()
        for sample in samples:
            d=spectra[sample,region];x=d.binding_energy_eV;color=self.colors[sample]
            ax.plot(x,d.intensity_CPS,color=color,lw=.45,alpha=.65,zorder=2)
            positions=d.attrs.get('component_positions',{})
            for position in sorted(set(positions.values())):
                if lo<=position<=hi:
                    ax.axvline(position,color=color,ls=':',lw=.5,alpha=.45,zorder=0)
            if fits and 'Envelope' in d:
                mask=~np.isclose(d.Background,d.intensity_CPS,rtol=1e-5,atol=.001) if 'Background' in d else np.ones(len(d),bool)
                ax.plot(x,d.Envelope.where(mask),color=color,lw=.8,zorder=3)
                for col in d.columns:
                    if col in ['binding_energy_eV','intensity_CPS','Envelope']:continue
                    ax.plot(x,d[col].where(mask),color=color,lw=.45,
                            ls=':' if col=='Background' else '--',alpha=.8,zorder=1)
            if annotate:
                available=set(positions)
                for target,label,location in assignments.get(region,[]):
                    if region == 'Li 1s' and ((label == 'LiF?' and not sample.startswith('BL')) or (label == 'Salt-associated Li' and not sample.startswith('BO'))):
                        continue
                    matches=[name for name in available if abs(positions[name]-target)<=.9]
                    if not matches:continue
                    match=min(matches,key=lambda name:abs(positions[name]-target))
                    available.remove(match)
                    if label not in labels:
                        ax.text(*location,label,transform=ax.transAxes,fontsize=5,ha='center',va='center')
                        labels.add(label)
        self.finish_axis(ax,region)
        values=[]
        for sample in samples:
            d=spectra[sample,region]
            values.extend(d.loc[d.binding_energy_eV.between(lo,hi),'intensity_CPS'])
        low,high=min(values),max(values);span=max(high-low,1)
        ax.set_ylim(low-.08*span,high+.38*span)

    def color_legend(self, fig, y=.035):
        fig.legend(handles=[self.Line2D([],[],color=self.colors[s],lw=.8,label=self.names[s])
                            for s in ('BL-1','BO-1')],
                   loc='lower center',bbox_to_anchor=(.5,y),ncol=2,
                   frameon=False,fontsize=5,handlelength=1.7)

    def upper_regions(self, spectra):
        fig,axes=self.plt.subplots(2,3,figsize=(6.5,6.5*2/3),dpi=600)
        fig.subplots_adjust(left=.09,right=.985,bottom=.15,top=.94,wspace=.38,hspace=.32)
        for ax,region in zip(axes.flat,REGIONS):
            self.plot_region(ax,spectra,region,('BL-1','BO-1'),annotate=True)
        self.color_legend(fig)
        return fig,axes


    def survey(self, spectra, axes=None, legend=True):
        if axes is None:
            fig,axes=self.plt.subplots(1,1,figsize=(6.5,3.6),dpi=600,squeeze=False)
            axes=axes.ravel()
            fig.subplots_adjust(left=.10,right=.985,bottom=.23,top=.94)
        else:
            fig=axes[0].figure
        for ax,pair,name in zip(axes,[SAMPLES],['Upper regions']):
            for s in pair:
                d=spectra[s,'Survey'];ax.plot(d.binding_energy_eV,d.intensity_CPS,color=self.colors[s],lw=.5)
            ax.set_title(name,fontsize=7,pad=5)
            ax.set_ylabel(r'Intensity (counts s$^{-1}$)',fontsize=7)
            ax.tick_params(labelsize=5,width=.5)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
            ax.yaxis.get_offset_text().set_fontsize(5)
            ax.grid(False);ax.set_xlim(1200,0);ax.margins(y=.25)
            for energy,label in [(685,'F 1s'),(533,'O 1s'),(402,'N 1s'),(285,'C 1s'),(196,'B 1s'),(56,'Li 1s'),(153,'Si 2s'),(102,'Si 2p'),(32,'F 2s')]:
                ax.axvline(energy,color='.6',ls=':',lw=.4,zorder=0)
                ax.text(energy,({'Si 2s':.72,'Si 2p':.43,'F 2s':.27}.get(label,.88 if label not in ('B 1s','Li 1s') else .58)),label,
                        transform=ax.get_xaxis_transform(),fontsize=5,ha='center',va='top')
        axes[-1].set_xlabel('Binding energy (eV)',fontsize=7)
        if legend: self.color_legend(fig)
        return fig,axes

    def reported_composition(self, elemental, axes=None, legend=True):
        samples=('BL-1','BO-1');regions=('C 1s','O 1s','F 1s','B 1s')
        values=elemental.loc[list(regions),list(samples)]
        if not np.isfinite(values.to_numpy()).all():raise ValueError('Missing elemental values')
        ratios=values.loc[['F 1s','O 1s','B 1s']].div(values.loc['C 1s'])
        if axes is None:
            fig,axes=self.plt.subplots(1,2,figsize=(6.5,2.7),dpi=600)
            fig.subplots_adjust(left=.10,right=.98,bottom=.25,top=.94,wspace=.40)
        else:
            fig=axes[0].figure
        for k,s in enumerate(samples):
            axes[0].bar(np.arange(4)+(k-.5)*.34,values[s],width=.34,color=self.colors[s])
            axes[1].bar(np.arange(3)+(k-.5)*.34,ratios[s],width=.34,color=self.colors[s])
        axes[0].set_xticks(range(4),['C','O','F','B'])
        axes[1].set_xticks(range(3),['F/C','O/C','B/C'])
        axes[0].set_ylabel(r'Reported atomic concentration (\%)' if self.plt.rcParams['text.usetex'] else 'Reported atomic concentration (%)',fontsize=7)
        axes[1].set_ylabel('Atomic ratio',fontsize=7)
        for ax in axes:
            ax.tick_params(labelsize=5,width=.5);ax.set_ylim(bottom=0);ax.grid(False)
        if legend: self.color_legend(fig)
        return fig,axes

    def survey_composition(self, spectra, elemental):
        fig=self.plt.figure(figsize=(6.5,6.5),dpi=600)
        survey_axes=np.array([fig.add_axes([.10,.57,.885,.36])])
        composition_axes=np.array([
            fig.add_axes([.10,.14,.37,.37]),
            fig.add_axes([.615,.14,.37,.37]),
        ])
        self.survey(spectra,axes=survey_axes,legend=False)
        survey_axes[0].set_title('')
        self.reported_composition(elemental,axes=composition_axes,legend=False)
        for ax in composition_axes:
            ax.set_box_aspect(1)
        self.color_legend(fig,y=.085)
        return fig,{'survey':survey_axes,'composition':composition_axes}


def plot_xps(data,root=None,*,figure="core",use_tex=None):
    import matplotlib.pyplot as plt
    root=Path(root).resolve() if root is not None else repository_root()
    style=root/'styles/manuscript.mplstyle'
    if not style.is_file():raise FileNotFoundError(style)
    if figure not in ('core','summary'):raise ValueError('figure must be core or summary.')
    with plt.style.context(str(style)),plt.rc_context({} if use_tex is None else {'text.usetex':bool(use_tex)}):
        plotter=_XPSPlotter()
        if figure=='core':return plotter.upper_regions(data.spectra)
        return plotter.survey_composition(data.spectra,data.elemental)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--plot',action='store_true',help='Display the two selected-position manuscript figures.')
    parser.add_argument('--no-tex',action='store_true',help='Explicitly disable the style LaTeX renderer.')
    args=parser.parse_args(argv)
    data,tables=run_xps(args.root,output_dir=args.output_dir)
    print('Reported atomic percentages (missing values remain NaN):')
    print(tables['reported_atomic_percent'].to_string(index=False))
    print('\nReported atomic ratios:')
    print(tables['atomic_ratios'].to_string(index=False))
    print('\nSelected upper positions: one electrode per formulation. Existing fits are reused.')
    if args.plot:
        for figure in ['core','summary']:plot_xps(data,args.root,figure=figure,use_tex=False if args.no_tex else None)
        import matplotlib.pyplot as plt
        plt.show()
    return 0

if __name__=='__main__':raise SystemExit(main())
