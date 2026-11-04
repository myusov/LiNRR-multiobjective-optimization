from __future__ import annotations
import argparse
from dataclasses import dataclass,field
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

FIGURE_NAMES={'independent':'SI independent pairs','spatial':'SI additional fields','main':'Main Figure 6'}

@dataclass
class SEMFrame:
    pixels: np.ndarray
    um_per_pixel: float
    metadata: dict
    cropped_pixels_sha256: str

@dataclass
class SEMData:
    root: Path
    panels: pd.DataFrame
    metadata: pd.DataFrame
    _frames: dict=field(default_factory=dict,repr=False)


def repository_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate/'data').is_dir() and (candidate/'styles').is_dir():return candidate
    raise FileNotFoundError('Cannot locate repository data/ and styles/.')


def _columns(table,required,label):
    missing=set(required)-set(table.columns)
    if missing:raise ValueError(f'{label}: missing columns {sorted(missing)}')


def load_sem_data(root=None):
    """Read panel selection and calibrated image metadata; no images modified."""
    root=Path(root).resolve() if root is not None else repository_root()
    folder=root/'data/characterization/SEM'
    panels=pd.read_csv(folder/'panel_selection.csv',float_precision='round_trip')
    metadata=pd.read_csv(folder/'image_metadata.csv',float_precision='round_trip')
    _columns(panels,['Panel ID','Figure','Selection array row','Selection array column','Image ID',
                    'Specimen ID','Scale bar (um)','Panel row','Panel column','Packaged image'], 'Panels')
    _columns(metadata,['Image ID','session','filename','Packaged image','image_height_without_footer_px',
                      'pixel_width_nm','hfw_um','sha256'],'Image metadata')
    if panels['Panel ID'].duplicated().any() or metadata['Image ID'].duplicated().any():
        raise ValueError('Duplicate panel or image metadata IDs.')
    if panels[['Panel ID','Figure','Image ID','Packaged image']].isna().any().any() or metadata.isna().any().any():
        raise ValueError('Missing panel/image metadata.')
    for name in ['image_height_without_footer_px','pixel_width_nm','hfw_um','session']:
        values=metadata[name].to_numpy(float)
        if not np.isfinite(values).all() or (values<=0).any():raise ValueError('Invalid metadata: '+name)
    heights=metadata['image_height_without_footer_px'].to_numpy(float)
    if not np.equal(heights,np.floor(heights)).all():raise ValueError('Crop heights must be integers.')
    for name in ['Selection array row','Selection array column','Panel row','Panel column']:
        values=panels[name].to_numpy(float)
        if not np.isfinite(values).all() or (values<1).any() or not np.equal(values,np.floor(values)).all():
            raise ValueError('Panel coordinates must be positive integers.')
    if not np.isfinite(panels['Scale bar (um)']).all() or (panels['Scale bar (um)']<=0).any():
        raise ValueError('Scale bars must be positive and finite.')
    if panels.duplicated(['Figure','Selection array row','Selection array column']).any():
        raise ValueError('Duplicate figure selection positions.')
    lookup=metadata.set_index('Image ID')
    for row in panels.to_dict('records'):
        if row['Image ID'] not in lookup.index:raise ValueError('Missing calibration for '+row['Image ID'])
        if lookup.loc[row['Image ID'],'Packaged image']!=row['Packaged image']:
            raise ValueError('Panel and calibration image paths disagree.')
    return SEMData(root,panels,metadata)


def load_sem_frame(data,image_id):
    if image_id in data._frames:return data._frames[image_id]
    matched=data.metadata.loc[data.metadata['Image ID'].eq(image_id)]
    if len(matched)!=1:raise ValueError('No unique image metadata for '+image_id)
    record=matched.iloc[0].to_dict()
    folder=(data.root/'data/characterization/SEM').resolve()
    path=(folder/record['Packaged image']).resolve()
    if not path.is_relative_to(folder):raise ValueError('Image path is outside the SEM data folder.')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:
        raise ValueError('Source hash does not match metadata: '+image_id)
    # Open the same bytes that were hashed, avoiding a separate mutable file read.
    from io import BytesIO
    with Image.open(BytesIO(raw)) as image:pixels=np.array(image)
    height=int(record['image_height_without_footer_px'])
    if height>pixels.shape[0] or pixels.ndim not in (2,3):raise ValueError('Invalid image crop height or dimensions.')
    nm_per_pixel=float(record['pixel_width_nm']);hfw=float(record['hfw_um'])
    if not np.isclose(pixels.shape[1]*nm_per_pixel/1000,hfw,rtol=.002):
        raise ValueError('Inconsistent horizontal calibration: '+image_id)
    cropped=pixels[:height]
    digest=hashlib.sha256(cropped.tobytes()).hexdigest()
    cropped.setflags(write=False)
    frame=SEMFrame(cropped,nm_per_pixel/1000,record,digest)
    data._frames[image_id]=frame
    return frame


def draw_sem_frame(ax,frame,bar_um):
    pixels=frame.pixels;um_per_pixel=frame.um_per_pixel;h,w=pixels.shape[:2]
    if not np.isfinite(bar_um) or bar_um<=0:raise ValueError('Scale bar must be positive.')
    if pixels.ndim==2:
        if np.issubdtype(pixels.dtype,np.integer):top=np.iinfo(pixels.dtype).max
        else:raise ValueError('Float SEM image: define acquisition intensity range explicitly.')
        ax.imshow(pixels,cmap='gray',vmin=0,vmax=top,interpolation='none')
    else:ax.imshow(pixels,interpolation='none')
    length=bar_um/um_per_pixel
    if length>.7*w:raise ValueError('Scale bar too long for this field')
    right=.94*w;bottom=.92*h
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((right-length-.025*w,bottom-.10*h),length+.05*w,.15*h,
                           facecolor='black',edgecolor='none',alpha=.75,zorder=3))
    ax.plot([right-length,right],[bottom,bottom],color='white',lw=1.4,
            solid_capstyle='butt',zorder=4)
    ax.text(right-length/2,bottom-.02*h,f'{bar_um:g} \N{MICRO SIGN}m',
            ha='center',va='bottom',color='white',fontsize=6,zorder=4,
            fontfamily='Arial',usetex=False)
    ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.set_axis_off()
    return frame.metadata


def draw_sem_panel(ax,data,panel_id):
    rows=data.panels.loc[data.panels['Panel ID'].eq(panel_id)]
    if len(rows)!=1:raise ValueError('Expected a unique panel ID: '+panel_id)
    row=rows.iloc[0];frame=load_sem_frame(data,row['Image ID'])
    return draw_sem_frame(ax,frame,float(row['Scale bar (um)']))


def plot_sem_pairs(data,*,spatial=False,use_tex=None):
    import matplotlib.pyplot as plt
    style=data.root/'styles/manuscript.mplstyle'
    if not style.is_file():raise FileNotFoundError(style)
    name=FIGURE_NAMES['spatial' if spatial else 'independent']
    chosen=data.panels.loc[data.panels['Figure'].eq(name)].sort_values(['Selection array row','Selection array column'])
    expected={(row,col) for row in range(1,4) for col in range(1,3)}
    if set(zip(chosen['Selection array row'],chosen['Selection array column']))!=expected:
        raise ValueError('Expected a complete 3x2 selection for '+name)
    with plt.style.context(str(style)),plt.rc_context({} if use_tex is None else {'text.usetex':bool(use_tex)}):
        fig,axes=plt.subplots(3,2,figsize=(6.5,8.0),dpi=600)
        fig.subplots_adjust(left=.10,right=.99,bottom=.025,top=.955,wspace=.035,hspace=.10)
        titles=['Baseline condition','Pareto front condition '+(r'\#86' if plt.rcParams['text.usetex'] else '#86')]
        for ax,title in zip(axes[0],titles):ax.set_title(title,fontsize=7,pad=7,color='black')
        provenance=[]
        for row in chosen.to_dict('records'):
            frame=load_sem_frame(data,row['Image ID']);bar=float(row['Scale bar (um)'])
            r=int(row['Selection array row']);c=int(row['Selection array column'])
            record=draw_sem_frame(axes[r-1,c-1],frame,bar)
            provenance.append({'panel':f'row {r}, column {c}','session':int(record['session']),
                'filename':record['filename'],'field_width_um':float(record['hfw_um']),
                'scale_bar_um':bar,'sha256':record['sha256']})
    return fig,axes,provenance


def run_sem(root=None,*,output_dir=None):
    data=load_sem_data(root);records=[]
    for row in data.panels.to_dict('records'):
        frame=load_sem_frame(data,row['Image ID']);h,w=frame.pixels.shape[:2]
        bar=float(row['Scale bar (um)']);length=bar/frame.um_per_pixel
        if length>.7*w:raise ValueError('Scale bar too long for '+row['Panel ID'])
        records.append({**row,'session':int(frame.metadata['session']),
            'Displayed width (px)':w,'Displayed height (px)':h,'Pixel dtype':str(frame.pixels.dtype),
            'um per pixel':frame.um_per_pixel,'Scale bar length (px)':length,
            'Source SHA256':frame.metadata['sha256'],'Cropped pixels SHA256':frame.cropped_pixels_sha256})
    table=pd.DataFrame(records)
    if output_dir is not None:
        destination=Path(output_dir)
        if not destination.is_absolute():destination=data.root/destination
        destination.mkdir(parents=True,exist_ok=True)
        table.to_csv(destination/'sem_panel_provenance.csv',index=False)
    return data,table


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--plot',action='store_true',help='Display both SEM figures without saving them.')
    parser.add_argument('--no-tex',action='store_true')
    args=parser.parse_args(argv)
    data,table=run_sem(args.root,output_dir=args.output_dir)
    print(f"Verified {table['Image ID'].nunique()} source images across {len(table)} panel uses.")
    print('Three independently electrolyzed electrodes per formulation; additional fields reuse those electrodes.')
    print(table[['Panel ID','Image ID','Scale bar (um)','Scale bar length (px)']].to_string(index=False))
    if args.plot:
        for spatial in [False,True]:plot_sem_pairs(data,spatial=spatial,use_tex=False if args.no_tex else None)
        import matplotlib.pyplot as plt
        plt.show()
    return 0

if __name__=='__main__':raise SystemExit(main())
