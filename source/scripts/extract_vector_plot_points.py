"""Recover circular data symbols from a vector PDF with declared axis calibration.

This is figure-derived data, not an author's original numerical dataset.
The JSON specification declares marker diameter, colour, panel bounds and
linear/logarithmic tick coordinates in PDF points. No raster recognition is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specification',type=Path,required=True)
    parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import pymupdf
    spec=json.loads(args.specification.read_text())
    page=pymupdf.open(args.pdf)[spec.get('page',0)]
    drawings=page.get_drawings()
    def coordinate(pixel,axis):
        ticks=np.asarray(axis['ticks'],float)
        values=ticks[:,1]
        if axis.get('log10',False):values=np.log10(values)
        a,b=np.linalg.lstsq(np.column_stack((ticks[:,0],np.ones(len(ticks)))),values,rcond=None)[0]
        result=a*pixel+b
        return float(10**result if axis.get('log10',False) else result)
    panels=[]
    for panel in spec['panels']:
        rows=[];xmin,ymin,xmax,ymax=panel['bounds']
        for index,d in enumerate(drawings):
            r=d['rect'];colour=d['fill']
            if (colour is None or not np.allclose(colour,spec['marker_colour'],atol=1e-4)
                    or abs(r.width-spec['marker_diameter'])>spec.get('diameter_tolerance',.03)
                    or abs(r.height-spec['marker_diameter'])>spec.get('diameter_tolerance',.03)
                    or len(d['items'])!=4 or any(item[0]!='c' for item in d['items'])):continue
            x=(r.x0+r.x1)/2;y=(r.y0+r.y1)/2
            if not (xmin<x<xmax and ymin<y<ymax):continue
            row=dict(x=coordinate(x,panel['x']),y=coordinate(y,panel['y']),
                     pdf_center=[x,y],drawing_index=index)
            # Preserve visible vertical error bars when separately drawn.
            bars=[]
            for other in drawings:
                q=other['rect']
                if (other['type']=='s' and q.width<.01 and q.height>spec['marker_diameter']
                        and abs(q.x0-x)<.2 and q.y0<y<q.y1
                        and other['color'] is not None
                        and np.allclose(other['color'],spec['marker_colour'],atol=1e-4)):
                    bars.append(q)
            if bars:
                q=min(bars,key=lambda q:q.height)
                row['y_lower']=min(coordinate(q.y0,panel['y']),coordinate(q.y1,panel['y']))
                row['y_upper']=max(coordinate(q.y0,panel['y']),coordinate(q.y1,panel['y']))
            rows.append(row)
        rows.sort(key=lambda r:r['x'])
        panels.append(dict(label=panel['label'],points=rows))
    result=dict(schema='vector_plot_points.v1',source=spec['source'],
                source_pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
                provenance='Symbols recovered from published vector figure; not original numerical data',
                calibration=spec,panels=panels)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print([(p['label'],len(p['points'])) for p in panels])


if __name__=='__main__':main()
