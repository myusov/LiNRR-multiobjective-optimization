from __future__ import annotations
import argparse
from dataclasses import dataclass, fields
from pathlib import Path
import numpy as np
import pandas as pd

PARAMETER_UNITS = {
    "eta_thf":"mPa s", "rho_1m":"g/cm^3", "nu_1m_main":"cm^2/s", "nu_1m_si":"cm^2/s",
    "n2_thf":"mM", "n2_1m_low":"mM", "n2_1m_central":"mM", "n2_1m_high":"mM",
    "reference_salt":"M", "comparison_salt":"M", "temperature":"K",
}

@dataclass(frozen=True)
class TransportInputs:
    eta_thf: float
    rho_1m: float
    nu_1m_main: float
    nu_1m_si: float
    n2_thf: float
    n2_1m_low: float
    n2_1m_central: float
    n2_1m_high: float
    reference_salt: float
    comparison_salt: float
    temperature: float

    def __post_init__(self):
        values=np.array([getattr(self,f.name) for f in fields(self)],dtype=float)
        if not np.isfinite(values).all() or (values<=0).any():
            raise ValueError("Transport inputs must be positive and finite.")
        if not self.n2_1m_low<=self.n2_1m_central<=self.n2_1m_high:
            raise ValueError("Central solubility must lie between the two endpoints.")
        if self.comparison_salt>self.reference_salt:
            raise ValueError("Comparison concentration must lie in the interpolation range.")

    @property
    def eta_endpoints(self):
        return 100*self.rho_1m*np.array([self.nu_1m_main,self.nu_1m_si])

    @property
    def solubility_endpoints(self):
        return np.array([self.n2_1m_low,self.n2_1m_high])


def repository_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate/"data").is_dir() and (candidate/"styles").is_dir():return candidate
    raise FileNotFoundError("Cannot locate repository data/ and styles/ directories.")


def load_transport_inputs(root=None, *, path=None):
    root=Path(root).resolve() if root is not None else repository_root()
    source=Path(path) if path is not None else root/"data/literature/transport_reference_values.csv"
    if path is not None and not source.is_absolute():source=root/source
    table=pd.read_csv(source)
    needed={"Parameter","Value","Unit"}
    if not needed.issubset(table.columns):raise ValueError(f"Missing input columns: {sorted(needed-set(table.columns))}")
    if table.Parameter.duplicated().any():raise ValueError("Duplicate transport input parameters.")
    indexed=table.set_index("Parameter")
    missing=set(PARAMETER_UNITS)-set(indexed.index)
    if missing:raise ValueError(f"Missing transport parameters: {sorted(missing)}")
    for name,unit in PARAMETER_UNITS.items():
        if indexed.loc[name,"Unit"]!=unit:raise ValueError(f"{name}: expected unit {unit}.")
    return TransportInputs(**{name:float(indexed.loc[name,"Value"]) for name in PARAMETER_UNITS})


def _concentration(c, inputs):
    values=np.asarray(c,dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>inputs.reference_salt).any():
        raise ValueError("Concentration must be finite and within the interpolation endpoints.")
    return values/inputs.reference_salt


def transport_eta(c, endpoint, inputs):
    """Linear interpolation of dynamic viscosity in mPa s."""
    return inputs.eta_thf+_concentration(c,inputs)*(endpoint-inputs.eta_thf)


def transport_solubility(c, endpoint, inputs):
    """Linear interpolation of dissolved nitrogen concentration in mM."""
    return inputs.n2_thf+_concentration(c,inputs)*(endpoint-inputs.n2_thf)


def calculate_transport(inputs, *, grid_n=501):
    """Return grid, four endpoint scenarios, paired ratios and selected values.

    Stokes-Einstein-type scaling assumes equal T and effective molecular radius.
    Relative supply scales with D*C only for unchanged diffusion-layer thickness
    and negligible electrode-surface N2 concentration. No FE/rate change is predicted.
    """
    if isinstance(grid_n,bool) or not isinstance(grid_n,(int,np.integer)) or grid_n<2:
        raise ValueError("grid_n must be an integer of at least two.")
    salt=np.linspace(0,inputs.reference_salt,grid_n)
    eta_ends=inputs.eta_endpoints
    sol_ends=inputs.solubility_endpoints
    eta_midpoint=eta_ends.mean()
    eta_bounds=np.array([transport_eta(salt,e,inputs) for e in eta_ends])
    eta_mid=transport_eta(salt,eta_midpoint,inputs)
    sol_bounds=np.array([transport_solubility(salt,e,inputs) for e in sol_ends])
    sol_mid=transport_solubility(salt,inputs.n2_1m_central,inputs)
    dc_scenarios=np.array([
        inputs.eta_thf/transport_eta(salt,eta,inputs)*transport_solubility(salt,sol,inputs)/inputs.n2_thf
        for eta in eta_ends for sol in sol_ends])
    dc_mid=inputs.eta_thf/eta_mid*sol_mid/inputs.n2_thf
    dc_low,dc_high=dc_scenarios.min(axis=0),dc_scenarios.max(axis=0)
    np.testing.assert_allclose(dc_scenarios[:,0],1)
    if not ((dc_mid>=dc_low-1e-12)&(dc_mid<=dc_high+1e-12)).all():
        raise RuntimeError("Central transport curve is outside the scenario envelope.")
    grid=pd.DataFrame({"LiBF4 concentration (M)":salt,
        "Viscosity central (mPa s)":eta_mid,"Viscosity low (mPa s)":eta_bounds.min(axis=0),
        "Viscosity high (mPa s)":eta_bounds.max(axis=0),
        "N2 solubility central (mM)":sol_mid,"N2 solubility low (mM)":sol_bounds.min(axis=0),
        "N2 solubility high (mM)":sol_bounds.max(axis=0),
        "DC relative to THF central":dc_mid,"DC relative to THF low":dc_low,"DC relative to THF high":dc_high})
    scenarios=[];comparisons=[]
    for eta_index,eta in enumerate(eta_ends):
        for sol_index,sol in enumerate(sol_ends):
            key=eta_index*len(sol_ends)+sol_index
            scenarios.append(pd.DataFrame({"Scenario":key+1,"LiBF4 concentration (M)":salt,
                "Viscosity reference endpoint (mPa s)":eta,"Solubility reference endpoint (mM)":sol,
                "DC relative to THF":dc_scenarios[key]}))
            # Same endpoint in numerator and denominator; do not divide unrelated extrema.
            d_ratio=eta/transport_eta(inputs.comparison_salt,eta,inputs)
            c_ratio=transport_solubility(inputs.comparison_salt,sol,inputs)/sol
            ratio=eta/transport_eta(inputs.comparison_salt,eta,inputs)*transport_solubility(inputs.comparison_salt,sol,inputs)/sol
            comparisons.append({"Scenario":key+1,"Viscosity reference endpoint (mPa s)":eta,
                "Solubility reference endpoint (mM)":sol,"D comparison/reference":d_ratio,
                "C comparison/reference":c_ratio,"DC comparison/reference":ratio})
    central_ratios=np.array([eta/transport_eta(inputs.comparison_salt,eta,inputs)
        *transport_solubility(inputs.comparison_salt,inputs.n2_1m_central,inputs)/inputs.n2_1m_central for eta in eta_ends])
    comparison=pd.DataFrame(comparisons)
    selected=[]
    for concentration in [0.,inputs.comparison_salt,inputs.reference_salt]:
        etas=np.array([transport_eta(concentration,e,inputs) for e in eta_ends])
        sols=np.array([transport_solubility(concentration,e,inputs) for e in sol_ends])
        dc=np.array([inputs.eta_thf/transport_eta(concentration,e,inputs)*transport_solubility(concentration,s,inputs)/inputs.n2_thf for e in eta_ends for s in sol_ends])
        selected.append({"LiBF4 concentration (M)":concentration,
            "Viscosity low (mPa s)":etas.min(),"Viscosity high (mPa s)":etas.max(),
            "N2 solubility low (mM)":sols.min(),"N2 solubility high (mM)":sols.max(),
            "DC relative to THF low":dc.min(),"DC relative to THF high":dc.max()})
    ratio_summary=pd.DataFrame([{"Comparison salt (M)":inputs.comparison_salt,"Reference salt (M)":inputs.reference_salt,
        "D ratio low":comparison["D comparison/reference"].min(),"D ratio high":comparison["D comparison/reference"].max(),
        "Central-solubility DC ratio low":central_ratios.min(),"Central-solubility DC ratio high":central_ratios.max(),
        "Full-scenario DC ratio low":comparison["DC comparison/reference"].min(),
        "Full-scenario DC ratio high":comparison["DC comparison/reference"].max()}])
    return {"transport_grid":grid,"transport_scenarios":pd.concat(scenarios,ignore_index=True),
            "paired_comparisons":comparison,"selected_concentrations":pd.DataFrame(selected),"ratio_summary":ratio_summary}


def run_transport(root=None, *, input_path=None, output_dir=None):
    inputs=load_transport_inputs(root,path=input_path)
    tables=calculate_transport(inputs)
    if output_dir is not None:
        destination=Path(output_dir)
        if not destination.is_absolute():destination=(Path(root) if root is not None else repository_root())/destination
        destination.mkdir(parents=True,exist_ok=True)
        for name,table in tables.items():table.to_csv(destination/(name+".csv"),index=False)
    return tables


def plot_transport(tables, root=None, *, inputs=None, use_tex=None):
    import matplotlib.pyplot as plt
    root=Path(root).resolve() if root is not None else repository_root()
    if inputs is None:inputs=load_transport_inputs(root)
    style=root/"styles/manuscript.mplstyle"
    if not style.is_file():raise FileNotFoundError(style)
    data=tables["transport_grid"]
    salt=data["LiBF4 concentration (M)"].to_numpy()
    if not np.isclose(salt[-1],inputs.reference_salt):raise ValueError("Grid and input reference concentrations differ.")
    settings={'figure.autolayout':False,'axes.linewidth':.5,'axes.labelweight':'normal',
        'axes.labelsize':7,'xtick.labelsize':5,'ytick.labelsize':5,'xtick.major.width':.5,
        'ytick.major.width':.5,'lines.linewidth':.5,'lines.markersize':3}
    if use_tex is not None:settings['text.usetex']=bool(use_tex)
    with plt.style.context(str(style)),plt.rc_context(settings):
        color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        fig,axes=plt.subplots(1,3,figsize=(6.5,3.25),dpi=600)
        panels=[('Viscosity',' (mPa s)','Dynamic viscosity (mPa s)'),
                ('N2 solubility',' (mM)',r'N$_2$ solubility (mM)'),
                ('DC relative to THF','',r'$(DC)_c/(DC)_{\mathrm{THF}}$')]
        for ax,(stem,unit,ylabel) in zip(axes,panels):
            central=data[stem+' central'+unit].to_numpy();low=data[stem+' low'+unit].to_numpy();high=data[stem+' high'+unit].to_numpy()
            ax.fill_between(salt,low,high,color=color,linewidth=0,zorder=1,alpha=.2)
            ax.plot(salt,central,color=color,linewidth=.5,zorder=2)
            ax.plot(0,central[0],'o',color=color,markersize=3,zorder=4)
            ax.errorbar(inputs.reference_salt,central[-1],yerr=np.array([[central[-1]-low[-1]],[high[-1]-central[-1]]]),
                fmt='o',color=color,ms=3,elinewidth=.5,capsize=2,capthick=.5,markeredgewidth=.5,zorder=4)
            ax.plot(inputs.comparison_salt,np.interp(inputs.comparison_salt,salt,central),'D',mfc='white',mec=color,mew=.5,ms=3.5,zorder=5)
            ax.axvline(inputs.comparison_salt,color='black',ls=':',lw=.5,zorder=3)
            ax.set_xlabel(r'LiBF$_4$ concentration (M)',fontsize=7,labelpad=2)
            ax.set_ylabel(ylabel,fontsize=7,labelpad=2)
            ax.set_xlim(-.04*inputs.reference_salt,1.04*inputs.reference_salt)
            ax.set_xticks(np.linspace(0,inputs.reference_salt,6))
            ax.tick_params(axis='both',labelsize=5,width=.5,pad=1)
            ax.minorticks_off();ax.set_box_aspect(1);ax.grid(False)
            for spine in ax.spines.values():spine.set_linewidth(.5)
        axes[0].set_ylim(.4,1.25);axes[1].set_ylim(1.6,6.6);axes[2].set_ylim(0,1.05)
        axes[2].set_yticks([0,.2,.4,.6,.8,1])
        fig.tight_layout(pad=.6,w_pad=.7)
    return fig,axes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--input-path',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--plot',action='store_true',help='Display the manuscript plot without saving it.')
    parser.add_argument('--no-tex',action='store_true',help='Explicitly disable LaTeX for plotting.')
    args=parser.parse_args(argv)
    tables=run_transport(args.root,input_path=args.input_path,output_dir=args.output_dir)
    print(tables['selected_concentrations'].to_string(index=False))
    print('\nPaired comparison scenarios:')
    print(tables['ratio_summary'].to_string(index=False))
    print('\nScenario spans are not confidence intervals; alcohol and interpolation-law uncertainties are excluded.')
    if args.plot:
        plot_transport(tables,args.root,inputs=load_transport_inputs(args.root,path=args.input_path),use_tex=False if args.no_tex else None)
        import matplotlib.pyplot as plt
        plt.show()
    return 0

if __name__=='__main__':raise SystemExit(main())
