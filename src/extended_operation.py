from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import linregress

EXPERIMENT_ID = "LAB-8-54-FLOW"
FARADAY_CONSTANT = 96485.33212


def repository_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate/"data").is_dir() and (candidate/"styles").is_dir():return candidate
    raise FileNotFoundError("Cannot locate repository data/ and styles/.")


def read_inputs(root=None):
    """Read repository inputs; preserve CSV floating-point round trips."""
    root=Path(root).resolve() if root is not None else repository_root()
    directory=root/"data/validation/extended_operation"
    config=json.loads((root/"configs/experiments/five_hour.json").read_text(encoding="utf8"))
    def read(name):return pd.read_csv(directory/name,float_precision="round_trip")
    calibration=read("five_hour_calibrations.csv")
    assays=pd.concat([read("five_hour_aliquot_assays.csv"),read("five_hour_endpoint_assays.csv")],ignore_index=True)
    blanks=read("five_hour_water_blanks.csv")
    potential=read("five_hour_potential.csv")
    metadata=pd.read_csv(root/"data/validation/validation_run_metadata.csv",float_precision="round_trip")
    selected=metadata.loc[metadata["Experiment ID"].eq(EXPERIMENT_ID)]
    if len(selected)!=1:raise ValueError("Expected one five-hour experiment metadata row.")
    for column,key in [("Current density magnitude (mA/cm^2)","current_density_mA_cm2"),
                       ("Working electrode geometric area (cm^2)","area_cm2"),
                       ("Electrolysis duration (min)","duration_min"),
                       ("Electrolyte circulation (mL/min)","flow_mL_min")]:
        if not np.isclose(float(selected.iloc[0][column]),config[key],rtol=0,atol=1e-12):
            raise ValueError(f"Configuration and experiment metadata disagree: {key}.")
    for table in [assays,potential]:
        if not table["Experiment ID"].eq(EXPERIMENT_ID).all():raise ValueError("Unexpected experiment ID.")
    return config,calibration,assays,blanks,potential


def fit_calibrations(points):
    """Refit both free-intercept unweighted calibrations from deposited points."""
    curves={}
    for key,identifier,expected in [("low","FLOW-CAL-LOW",np.arange(7)),
                                     ("high","FLOW-CAL-HIGH",np.arange(0,61,10))]:
        data=points.loc[points["Calibration ID"].eq(identifier)].sort_values("Concentration (uM)")
        x=data["Concentration (uM)"].to_numpy(float);y=data["Blank-corrected delta absorbance"].to_numpy(float)
        if not np.array_equal(x,expected) or not np.isfinite(y).all():raise ValueError("Invalid calibration points: "+key)
        fit=linregress(x,y)
        if fit.slope<=0:raise ValueError("Nonpositive calibration slope.")
        curves[key]=dict(slope=fit.slope,intercept=fit.intercept,r2=fit.rvalue**2,
                         slope_stderr=fit.stderr,data=np.column_stack([x,y]).tolist())
    return curves


def blank_consistency(blanks,cfg):
    """Recompute the original 450-700 nm offset-corrected shape and delta checks."""
    spectra=[];deltas=[]
    for name in cfg['blank_files']:
        data=blanks.loc[blanks['Blank file'].eq(name)].sort_values('Wavelength (nm)')
        x=data['Wavelength (nm)'].to_numpy(float);y=data['Absorbance'].to_numpy(float)
        if len(x)<2 or not np.isfinite(np.column_stack([x,y])).all() or not np.all(np.diff(x)>0):
            raise ValueError('Invalid blank spectrum: '+name)
        if x.min()>450 or x.max()<700:raise ValueError('Blank does not span assay window: '+name)
        spectra.append(np.interp(np.arange(450.,701.),x,y))
        deltas.append(float(y[np.argmin(abs(x-650))])-float(y[np.argmin(abs(x-475))]))
    ys=np.array(spectra);deltas=np.array(deltas)
    centered=ys-ys.mean(axis=1,keepdims=True)
    rms=np.sqrt(np.mean((centered-centered.mean(axis=0))**2,axis=1))
    consistent=(np.ptp(deltas)<=cfg['blank_delta_tolerance_AU'] and max(rms)<=cfg['blank_shape_rms_tolerance_AU'])
    return deltas,bool(consistent),pd.DataFrame({'Blank file':cfg['blank_files'],'Delta absorbance':deltas,'Centered shape RMS':rms})


def analyze_extended_operation(cfg,calibration_points,assay_readings,blank_spectra,potential):
    for key in ['aliquot_mL','aliquot_diluted_mL','final_diluted_mL','duration_min','area_cm2','current_density_mA_cm2']:
        if not np.isfinite(cfg[key]) or cfg[key]<=0:raise ValueError('Invalid configuration: '+key)
    curves=fit_calibrations(calibration_points)
    deltas,consistent,blank_checks=blank_consistency(blank_spectra,cfg)
    required=['file','compartment','time_min','final','dilution','label_override','blank','signal_AU']
    missing=set(required)-set(assay_readings.columns)
    if missing:raise ValueError(f'Missing assay columns: {sorted(missing)}')
    if assay_readings.duplicated(['file','dilution']).any():raise ValueError('Duplicate sample/dilution rows.')
    records=[]
    for name,group in assay_readings.groupby('file',sort=True):
        if group['final'].nunique()!=1 or group['compartment'].nunique()!=1:raise ValueError('Inconsistent sample metadata: '+name)
        low=curves['low']
        signals=group.signal_AU.to_numpy(float)
        if not np.isfinite(signals).all():raise ValueError('Nonfinite assay signal.')
        low_concentrations=(signals-low['intercept'])/low['slope']
        which='high' if any(c>6 for c in low_concentrations) else 'low'
        fit=curves[which]
        for row in group.itertuples(index=False):
            if not np.isfinite(row.dilution) or row.dilution<=0:raise ValueError('Invalid dilution factor.')
            concentration=(row.signal_AU-fit['intercept'])/fit['slope']
            valid=0<=concentration<=(6 if which=='low' else 60)
            flag='in_range' if valid else ('below_calibration' if concentration<0 else 'above_calibration')
            factor=row.dilution*(cfg['aliquot_diluted_mL']/cfg['aliquot_mL'] if not row.final else 1)
            records.append(dict(file=name,compartment=row.compartment,time_min=row.time_min,
                final=bool(row.final),dilution=row.dilution,label_override=bool(row.label_override),
                blank=row.blank,signal_AU=row.signal_AU,fc_calibration=which,assay_uM=concentration,
                concentration_uM=concentration*factor,valid=valid,flag=flag,
                external_blank_span_uM=np.ptp(deltas)/fit['slope']*factor if row.blank!='internal water' else 0))
    readings=pd.DataFrame(records)
    grouped = []
    for name, g in readings.groupby('file', sort=False):
        accepted = g[g.valid]
        if accepted.empty:
            raise ValueError(f'No in-range dilution for {name}; inspect the dilution readings')
        grouped.append(dict(file=name, compartment=g.compartment.iloc[0], final=bool(g.final.iloc[0]),
                            time_min=g.time_min.iloc[0], concentration_uM=accepted.concentration_uM.mean(),
                            dilution_SD_uM=accepted.concentration_uM.std(ddof=1), n_dilutions=len(accepted)))
    samples = pd.DataFrame(grouped)
    # Error bars are one across-dilution sample SD (ddof=1), not SEM.
    # Propagation assumes independent sample measurements; calibration and
    # shared dilution-preparation systematic uncertainties are not included.
    def propagated_sd(coefficients, sample_sd):
        coefficients = np.asarray(coefficients)
        sample_sd = np.asarray(sample_sd)
        active = coefficients != 0
        return float(np.sqrt(np.sum((coefficients[active] * sample_sd[active]) ** 2)))

    time_rows, recovery_rows = [], []
    baseline = {}
    for kind in ['e', 'a']:
        ts = samples[(samples.compartment==kind) & ~samples.final].sort_values('time_min')
        if not np.array_equal(ts.time_min.to_numpy(), np.arange(0, cfg['duration_min']+1, 30)):
            raise ValueError(f'Missing or unexpected aliquot times for {kind}')
        v0 = cfg['initial_volume_mL'][kind]
        baseline[kind] = ts.concentration_uM.iloc[0]*v0/1000
        sample_sd = ts.dilution_SD_uM.to_numpy()
        removed = 0.
        for k, row in enumerate(ts.itertuples()):
            volume = v0-k*cfg['aliquot_mL']
            inventory = row.concentration_uM*volume/1000
            coefficients = np.zeros(len(ts))
            coefficients[:k] = cfg['aliquot_mL'] / 1000
            coefficients[k] = volume / 1000
            coefficients[0] -= v0 / 1000
            net_sd = propagated_sd(coefficients, sample_sd)
            time_rows.append(dict(compartment=kind, time_min=row.time_min,
                                  volume_before_sampling_mL=volume,
                                  concentration_uM=row.concentration_uM,
                                  reservoir_umol=inventory, previous_aliquots_umol=removed,
                                  cumulative_gross_umol=inventory+removed,
                                  cumulative_net_umol=inventory+removed-baseline[kind],
                                  cumulative_net_SD_umol=net_sd))
            removed += row.concentration_uM*cfg['aliquot_mL']/1000
        final = samples[(samples.compartment==kind) & samples.final].iloc[0]
        final_amount = final.concentration_uM*cfg['final_diluted_mL']/1000
        coefficients = np.full(len(ts), cfg['aliquot_mL'] / 1000)
        coefficients[0] -= v0 / 1000
        recovery_sd = np.hypot(
            propagated_sd(coefficients, sample_sd),
            final.dilution_SD_uM * cfg['final_diluted_mL'] / 1000)
        recovery_rows.append(dict(compartment=kind, final_collection_umol=final_amount,
                                  all_aliquots_umol=removed, initial_background_umol=baseline[kind],
                                  recovered_net_umol=final_amount+removed-baseline[kind],
                                  recovered_net_SD_umol=recovery_sd))
    deposit = samples[(samples.compartment=='d') & samples.final].iloc[0]
    deposit_amount = deposit.concentration_uM*cfg['final_diluted_mL']/1000
    recovery_rows.append(dict(compartment='d', final_collection_umol=deposit_amount,
                              all_aliquots_umol=0., initial_background_umol=0., recovered_net_umol=deposit_amount,
                              recovered_net_SD_umol=deposit.dilution_SD_uM * cfg['final_diluted_mL'] / 1000))
    timeline = pd.DataFrame(time_rows)
    current_A = cfg['current_density_mA_cm2']*cfg['area_cm2']/1000
    total = timeline.groupby('time_min').cumulative_net_umol.sum().reset_index()
    total['cumulative_net_SD_umol'] = timeline.groupby('time_min').cumulative_net_SD_umol.agg(
        lambda sd: np.sqrt(np.sum(sd.to_numpy() ** 2))).to_numpy()
    total['charge_C'] = current_A*total.time_min*60
    total['FE_percent_reservoirs_only'] = np.where(total.charge_C>0, 3*96485.33212*total.cumulative_net_umol*1e-6/total.charge_C*100, np.nan)
    total['interval_rate_umol_h'] = total.cumulative_net_umol.diff()/total.time_min.diff()*60
    recovery = pd.DataFrame(recovery_rows)
    net = float(recovery.recovered_net_umol.sum())
    net_sd = float(np.sqrt(np.sum(recovery.recovered_net_SD_umol.to_numpy() ** 2)))
    charge = current_A*cfg['duration_min']*60
    summary = dict(current_mA=current_A*1000, charge_C=charge, recovered_net_umol=net,
                   endpoint_FE_percent=3*96485.33212*net*1e-6/charge*100,
                   average_rate_nmol_s_cm2=net*1000/(cfg['duration_min']*60*cfg['area_cm2']),
                   endpoint_FE_SD_percent=3*96485.33212*net_sd*1e-6/charge*100,
                   average_rate_SD_nmol_s_cm2=net_sd*1000/(cfg['duration_min']*60*cfg['area_cm2']),
                   water_blanks_shape_consistent=bool(consistent), blank_delta_span_AU=float(np.ptp(deltas)),
                   calibration_fits=curves)
    qc = []
    for row in samples.itertuples():
        if row.n_dilutions > 1 and row.dilution_SD_uM > 0.1*abs(row.concentration_uM):
            qc.append(dict(check='dilution_disagreement', sample=row.file,
                           detail='Across-dilution SD exceeds 10% of mean; not independent replicate uncertainty.'))
    for kind in ['e', 'a']:
        last = timeline[timeline.compartment==kind].sort_values('time_min').iloc[-1]
        predicted = last.concentration_uM*(last.volume_before_sampling_mL-cfg['aliquot_mL'])/1000
        observed = recovery[recovery.compartment==kind].final_collection_umol.iloc[0]
        qc.append(dict(check='endpoint_mass_balance', sample=kind,
                       detail=f'Final collection {observed:.6g} umol; last aliquot predicts {predicted:.6g} umol remaining. Ratio={observed/predicted:.4g}'))
    for row in recovery.itertuples():
        if row.recovered_net_umol < 0:
            qc.append(dict(check='negative_net_recovery', sample=row.compartment,
                           detail='Recovered amount is below starting background; retained without clipping.'))
    qc.append(dict(check='calibration_slope_ratio', sample='low/high',
                   detail=f"Slope ratio {curves['low']['slope']/curves['high']['slope']:.4g}; both standards are from other experiment dates."))
    qc.append(dict(check='dilution_header_override', sample='a210min.csv', detail='2z interpreted as 2x; verify against lab notes.'))
    cp=potential[['time/s','<Ewe/V>']].apply(pd.to_numeric,errors='raise').dropna()
    if cp.empty or not np.isfinite(cp.to_numpy()).all() or not np.all(np.diff(cp['time/s'])>0):
        raise ValueError('Potential timestamps must be finite and strictly increasing.')
    return dict(cp=cp,total=total,recovery=recovery,summary=summary,timeline=timeline,
                readings=readings,samples=samples,qc=pd.DataFrame(qc),config=cfg,blank_checks=blank_checks)


def fc_evaporation_sensitivity(results):
    cfg=results['config']
    inference=[]
    profiles=[]
    for kind in ['e','a']:
        aliquots=results['samples'].loc[(results['samples']['compartment']==kind)
            & ~results['samples']['final']].sort_values('time_min')
        recovery=results['recovery'].set_index('compartment').loc[kind]
        last=aliquots.iloc[-1]
        c_last=float(last['concentration_uM'])
        expected_final=cfg['initial_volume_mL'][kind]-len(aliquots)*cfg['aliquot_mL']
        inferred_final=float(recovery['final_collection_umol'])/c_last*1000
        extra_loss=expected_final-inferred_final
        inference.append({'Compartment':kind,
            'Initial inventory (mL)':cfg['initial_volume_mL'][kind],
            'Aliquots withdrawn (mL)':len(aliquots)*cfg['aliquot_mL'],
            'Expected final inventory without evaporation (mL)':expected_final,
            'Last aliquot concentration (uM)':c_last,
            'Final collected ammonia (umol)':recovery['final_collection_umol'],
            'Inferred final inventory (mL)':inferred_final,
            'Additional volume loss needed (mL)':extra_loss,
            'Additional loss / expected final volume (%)':100*extra_loss/expected_final,
            'Mean additional loss (mL/h)':extra_loss/(cfg['duration_min']/60),
        })
    inference=pd.DataFrame(inference)
    electrolyte_loss=float(inference.set_index('Compartment').loc['e','Additional volume loss needed (mL)'])
    for name,power,loss in [('No evaporation',1.,0.),('Early volume loss',.5,electrolyte_loss),
                           ('Linear volume loss',1.,electrolyte_loss),('Late volume loss',2.,electrolyte_loss)]:
        compartment_frames=[]
        for kind in ['e','a']:
            aliquots=results['samples'].loc[(results['samples']['compartment']==kind)
                & ~results['samples']['final']].sort_values('time_min')
            times=aliquots['time_min'].to_numpy()
            concentration=aliquots['concentration_uM'].to_numpy()
            removed=np.r_[0.,np.cumsum(concentration[:-1]*cfg['aliquot_mL']/1000)]
            volume=cfg['initial_volume_mL'][kind]-np.arange(len(times))*cfg['aliquot_mL']
            if kind=='e': volume=volume-loss*(times/cfg['duration_min'])**power
            assert (volume>0).all()
            background=concentration[0]*cfg['initial_volume_mL'][kind]/1000
            amount=concentration*volume/1000+removed-background
            compartment_frames.append(pd.DataFrame({'time_min':times,'net_NH3_umol':amount,
                'compartment':kind,'volume_before_sampling_mL':volume,'scenario':name}))
        profiles.append(pd.concat(compartment_frames,ignore_index=True))
    components=pd.concat(profiles,ignore_index=True)
    combined=components.groupby(['scenario','time_min'],as_index=False)['net_NH3_umol'].sum()
    combined['FE_reservoirs_only_percent']=3*96485.33212*combined['net_NH3_umol']*1e-6/(
        results['summary']['current_mA']/1000*combined['time_min'].replace(0,np.nan)*60)*100
    # A diagnostic of whether uniform net electrolyte accumulation could fit
    # the data under evaporation alone. A negative implied loss is incompatible
    # with evaporation-only volume shrinkage under the other stated assumptions.
    aliquots=results['samples'].loc[(results['samples']['compartment']=='e')
        & ~results['samples']['final']].sort_values('time_min')
    times=aliquots['time_min'].to_numpy()
    concentration=aliquots['concentration_uM'].to_numpy()
    removed=np.r_[0.,np.cumsum(concentration[:-1]*cfg['aliquot_mL']/1000)]
    initial=cfg['initial_volume_mL']['e']
    endpoint_e=results['recovery'].set_index('compartment').loc['e','recovered_net_umol']
    target=endpoint_e*times/cfg['duration_min']
    required_volume=(target+concentration[0]*initial/1000-removed)/concentration*1000
    nominal=initial-np.arange(len(times))*cfg['aliquot_mL']
    uniform_check=pd.DataFrame({'time_min':times,'No-evaporation volume (mL)':nominal,
        'Volume required for uniform electrolyte NH3 accumulation (mL)':required_volume,
        'Implied additional loss (mL)':nominal-required_volume})
    return dict(inference=inference,components=components,combined=combined,
                uniform_accumulation_check=uniform_check)

def _plot_fc_manuscript(results, colors):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    cp = results["cp"]
    total = results["total"]
    recovery = results["recovery"]
    summary = results["summary"]
    cfg = results["config"]
    blue, orange, green = colors[:3]
    use_tex = plt.rcParams["text.usetex"]
    fe_label = r"Faradaic efficiency (\%)" if use_tex else "Faradaic efficiency (%)"
    potential_label = "Potential (V)"
    with plt.rc_context({"axes.linewidth": .5, "axes.labelweight": "normal",
                         "figure.autolayout": False}):
        fig, (ax_time, ax_end) = plt.subplots(1, 2, figsize=(6.5, 3.25), dpi=600)
        fig.subplots_adjust(
            left=.09, right=.91, bottom=.18, top=.92, wspace=.45
        )
        ax_potential = ax_time.twinx()
        elapsed_h = (cp["time/s"].to_numpy() - cp["time/s"].iloc[0]) / 3600
        ax_potential.plot(elapsed_h, cp["<Ewe/V>"].to_numpy(), color="black",
                          alpha=.5, linewidth=.5, zorder=1)
        ax_time.errorbar(total["time_min"].to_numpy()/60,
                          total["cumulative_net_umol"].to_numpy(),
                          yerr=total["cumulative_net_SD_umol"].to_numpy(),
                          fmt="o-", color=blue, linewidth=.5, markersize=3,
                          markeredgecolor="black", markeredgewidth=.4,
                          elinewidth=.5, capsize=2, capthick=.5, zorder=4)
        ax_time.set_zorder(ax_potential.get_zorder()+1)
        ax_time.patch.set_visible(False)
        ax_time.set_xlim(0, cfg["duration_min"]/60)
        ax_time.set_xlabel("Elapsed time (h)", fontsize=7)
        ax_time.set_ylabel(r"Cumulative NH$_3$ ($\mu$mol)", fontsize=7, color=blue)
        ax_potential.set_ylabel(potential_label, fontsize=7, labelpad=3)
        ax_time.tick_params(axis="y", colors=blue)
        ax_time.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_potential.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_time.legend(handles=[
            Line2D([],[],color=blue,marker="o",markersize=3,linewidth=.5,
                   markeredgecolor="black",markeredgewidth=.4,label=r"Cumulative NH$_3$"),
            Line2D([],[],color="black",linewidth=.5,label="Potential"),
        ], loc="upper left", frameon=False, fontsize=5, handletextpad=.4)

        endpoint = recovery.set_index("compartment").loc[["e","a","d"]]
        fe_per_umol = 3*96485.33212*1e-6/summary["charge_C"]*100
        rate_per_umol = 1000/(cfg["duration_min"]*60*cfg["area_cm2"])
        rate_per_fe = rate_per_umol/fe_per_umol
        means = endpoint["recovered_net_umol"].to_numpy()*fe_per_umol
        sd = endpoint["recovered_net_SD_umol"].to_numpy()*fe_per_umol
        if not np.isfinite(sd).all():
            raise ValueError("Missing endpoint analytical SD; inspect accepted dilution counts.")
        ax_end.bar(np.arange(3), means, width=.65, color=[blue,orange,green],
                   alpha=.9, edgecolor="none", zorder=2)
        ax_end.errorbar(np.arange(3),means,yerr=sd,fmt="none",ecolor="black",
                        elinewidth=.5,capsize=2,capthick=.5,zorder=3)
        ax_end.axhline(0,color="black",linewidth=.5,zorder=1)
        ax_end.set_xticks(np.arange(3),["Electrolyte","Acid trap","Deposits"])
        ax_end.set_ylabel(fe_label,fontsize=7,labelpad=3)
        ax_end.yaxis.set_major_locator(MaxNLocator(nbins=5))
        low=min(0.,float(np.min(means-sd)))
        high=max(0.,float(np.max(means+sd)),summary["endpoint_FE_percent"])
        span=max(high-low,1e-8)
        ax_end.set_ylim(low-.05*span,high+.45*span)
        ax_rate=ax_end.secondary_yaxis("right",functions=(lambda fe:fe*rate_per_fe,
                                                         lambda rate:rate/rate_per_fe))
        ax_rate.set_ylabel(r"Rate ($\mathrm{nmol\,s^{-1}\,cm^{-2}}$)",fontsize=7,labelpad=3)
        ax_rate.yaxis.set_major_locator(MaxNLocator(nbins=5))
        pct=r"\%" if use_tex else "%"
        annotation=(f"Total FE: {summary['endpoint_FE_percent']:.2f} "
                    + r"$\pm$" + f" {summary['endpoint_FE_SD_percent']:.2f}{pct}\n"
                    + f"Total rate: {summary['average_rate_nmol_s_cm2']:.2f} "
                    + r"$\pm$" + f" {summary['average_rate_SD_nmol_s_cm2']:.2f} "
                    + r"$\mathrm{nmol\,s^{-1}\,cm^{-2}}$")
        ax_end.text(.04,.97,annotation,transform=ax_end.transAxes,
                    ha="left",va="top",fontsize=5)
        for axis in [ax_time,ax_potential,ax_end,ax_rate]:
            axis.tick_params(axis="both",labelsize=5,width=.5)
            for spine in axis.spines.values(): spine.set_linewidth(.5)
        for axis in [ax_time,ax_potential,ax_end]:
            axis.set_box_aspect(1)
            axis.grid(False)
        return fig


def run_extended_operation(root=None, *, exploratory=False, output_dir=None):
    result=analyze_extended_operation(*read_inputs(root))
    if exploratory:
        sensitivity=fc_evaporation_sensitivity(result)
    if output_dir is not None:
        destination=Path(output_dir)
        if not destination.is_absolute():destination=(Path(root) if root is not None else repository_root())/destination
        destination.mkdir(parents=True,exist_ok=True)
        for name,value in result.items():
            if isinstance(value,pd.DataFrame):value.to_csv(destination/(name+'.csv'),index=False)
        public_summary={key:value for key,value in result['summary'].items() if key!='calibration_fits'}
        pd.DataFrame([public_summary]).to_csv(destination/'summary.csv',index=False)
        if exploratory:
            for name,value in result['exploratory'].items():value.to_csv(destination/('exploratory_'+name+'.csv'),index=False)
    return result


def plot_extended_operation(result,root=None,*,use_tex=None):
    import matplotlib.pyplot as plt
    root=Path(root).resolve() if root is not None else repository_root()
    style=root/'styles/manuscript.mplstyle'
    if not style.is_file():raise FileNotFoundError(style)
    with plt.style.context(str(style)),plt.rc_context({} if use_tex is None else {'text.usetex':bool(use_tex)}):
        colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        return _plot_fc_manuscript(result,colors)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--exploratory',action='store_true',help='Conditional volume-loss scenarios; no endpoint correction.')
    parser.add_argument('--plot',action='store_true',help='Display the manuscript figure without saving it.')
    parser.add_argument('--no-tex',action='store_true')
    args=parser.parse_args(argv)
    result=run_extended_operation(args.root,exploratory=args.exploratory,output_dir=args.output_dir)
    s=result['summary']
    print(f"Five-hour experiment: one independent electrolysis\nNet NH3: {s['recovered_net_umol']:.6f} umol")
    print(f"FE: {s['endpoint_FE_percent']:.6f} +/- {s['endpoint_FE_SD_percent']:.6f}%")
    print(f"Average rate: {s['average_rate_nmol_s_cm2']:.6f} +/- {s['average_rate_SD_nmol_s_cm2']:.6f} nmol s^-1 cm^-2")
    print('Uncertainties are propagated analytical SDs, not variation across electrolyses.')
    if args.exploratory:print(result['exploratory']['inference'].to_string(index=False))
    if args.plot:
        plot_extended_operation(result,args.root,use_tex=False if args.no_tex else None)
        import matplotlib.pyplot as plt
        plt.show()
    return 0

if __name__=='__main__':raise SystemExit(main())
