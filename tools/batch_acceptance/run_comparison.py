"""Run two existing candidate methods with auditable shared fusion stages."""
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'src'))

import numpy as np
from PIL import Image
import torch

from gpt_animeimage_denoise import fusion_pipeline as pipeline
from gpt_animeimage_denoise.filters import validate_unit_interval
from gpt_animeimage_denoise.fusion import _policy
from gpt_animeimage_denoise.fusion_models import FusionModels
from gpt_animeimage_denoise.pipeline import _read_source,_sha,_json

METHODS=('l0','esrgan-l0')
PARTS=('skin','face','hair','eyes','mouth','outfit')
SHARED=('person_labels','raw_labels','objects','unknown','conflict','consensus','protected','source_map',
        *('module_'+part for part in PARTS))
CODE_FILES = (
    'src/gpt_animeimage_denoise/fusion_pipeline.py',
    'src/gpt_animeimage_denoise/fusion_models.py',
    'src/gpt_animeimage_denoise/fusion.py',
    'src/gpt_animeimage_denoise/semantic_repair.py',
    'src/gpt_animeimage_denoise/auto_sam.py',
    'src/gpt_animeimage_denoise/corrections.py',
    'src/gpt_animeimage_denoise/filters.py',
    'src/gpt_animeimage_denoise/grouping.py',
    'src/gpt_animeimage_denoise/pipeline.py',
    'src/gpt_animeimage_denoise/denoisers.py',
    'src/denoise.py',
    'src/l0_cuda.py',
    'src/tiled_inference.py',
    'tools/batch_acceptance/run_comparison.py',
)


def now():return datetime.now(timezone.utc).isoformat()


def verify_pair(source,output_root,reports):
    rgb,alpha,icc=_read_source(source);digest=_sha(source);root=Path(output_root)
    if set(reports)!=set(METHODS):raise ValueError('Both comparison methods are required')
    first=None;shared=None;weight_first=None
    for method,record in reports.items():
        if record['status']!='complete' or record['source_sha256']!=digest:
            raise ValueError('Incomplete or wrong-source comparison')
        parameters=record['parameters']
        if parameters['profile']!='fusion' or parameters['denoiser']!=method:
            raise ValueError('Comparison requires the requested fusion methods')
        common={k:parameters.get(k) for k in ('profile','lambda','sigma_at_2048','strength',
                    'device','models','implementation','policy','l0_implementation')}
        common.update({key:validate_unit_interval(parameters.get(key,1.),key)
                       for key in ('structure_protection','restoration_gain')})
        if first is not None and common!=first:raise ValueError('Comparison parameters differ')
        first=common;folder=root/record['folder']
        for name,expected in record['output_hashes'].items():
            if _sha(folder/name)!=expected:raise ValueError('Output hash differs: '+name)
        arrays={name:np.asarray(Image.open(folder/(name+'.png'))) for name in SHARED}
        if shared is not None and any(not np.array_equal(arrays[k],shared[k]) for k in SHARED):
            raise ValueError('The two methods did not use identical masks')
        shared=arrays;weight=np.load(folder/'weight.npy',allow_pickle=False)
        if (weight.dtype!=np.float32 or weight.shape!=rgb.shape[:2] or not np.isfinite(weight).all()
                or np.any((weight<0)|(weight>1))):raise ValueError('Invalid float weight')
        if weight_first is not None and not np.array_equal(weight_first,weight):
            raise ValueError('The two methods did not use identical float weights')
        weight_first=weight
        labels=arrays['person_labels'];objects=arrays['objects']>0;unknown=arrays['unknown']>0
        protected=arrays['protected']>0
        if not np.array_equal(unknown,labels==255) or np.any(labels[objects]!=0):
            raise ValueError('Unknown/object labels are inconsistent')
        for index,part in enumerate(PARTS,1):
            if not np.array_equal(arrays['module_'+part]>0,labels==index):
                raise ValueError('Module does not come from actual person labels: '+part)
        if weight[objects|unknown|protected].any():raise ValueError('Protected weights are nonzero')
        with Image.open(folder/'result.png') as image:
            result=np.asarray(image.convert('RGB'))
            if image.size!=(rgb.shape[1],rgb.shape[0]) or image.info.get('icc_profile')!=icc:
                raise ValueError('Output size/ICC changed')
            if alpha is not None and not np.array_equal(np.asarray(image.getchannel('A')),alpha):
                raise ValueError('Original alpha changed')
        if not np.array_equal(result[weight==0],rgb[weight==0]):raise ValueError('W=0 pixels changed')
        if alpha is not None and weight[alpha==0].any():raise ValueError('Transparent protection changed')
    if _sha(source)!=digest:raise ValueError('Source changed while verifying')
    return {'same_masks':True,'same_weight':True,'source_unchanged':True,
        'protected_unchanged':True,'modules_from_actual_labels':True,'output_hashes_verified':True,
        'alpha':'preserved' if alpha is not None else 'not_present',
        'icc':'preserved' if icc is not None else 'not_present'}


@contextmanager
def capture_stages(root,context):
    """Save actual executed stages, calling original implementations unchanged."""
    old_fuse=pipeline.fuse_evidence;old_refine=pipeline.refine_regions;old_weight=pipeline.build_weight
    def stage_folder(stage):
        return root/'trace'/context['id']/context['method']/stage
    def save_regions(stage,result):
        folder=stage_folder(stage);folder.mkdir(parents=True,exist_ok=True)
        for name in ('person_labels','objects','unknown','conflict','source_map'):
            value=result['person_labels']==255 if name=='unknown' else result[name]
            if value.dtype==bool:value=value.astype(np.uint8)*255
            Image.fromarray(value).save(folder/(name+'.png'))
        info={k:v for k,v in result.items() if not isinstance(v,np.ndarray) and not k.startswith('_')}
        info.update(stage=stage,captured_at=now(),kind='actual-execution-snapshot',
            source_sha256=context['source_sha256'],coordinate_grid='EXIF-oriented-original',
            artifacts={p.name:_sha(p) for p in folder.glob('*.png')})
        _json(folder/'stage.json',info)
    def fuse(*args,**kwargs):
        result=old_fuse(*args,**kwargs);save_regions('pre-sam',result);return result
    def refine(*args,**kwargs):
        result=old_refine(*args,**kwargs);save_regions('post-sam',result);return result
    def weight(allowed,protected,rgb,*args,**kwargs):
        result=old_weight(allowed,protected,rgb,*args,**kwargs)
        folder=stage_folder('weight');folder.mkdir(parents=True,exist_ok=True)
        for name,mask in [('allowed',allowed),('protected',protected)]:
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/(name+'.png'))
        np.save(folder/'weight_unscaled.npy',result,allow_pickle=False)
        _json(folder/'stage.json',{'stage':'weight','kind':'actual-execution-snapshot',
            'captured_at':now(),'source_sha256':context['source_sha256'],'parameters':kwargs,
            'note':'Native weight.npy is final after strength and threshold; this is the build_weight return.',
            'artifacts':{p.name:_sha(p) for p in folder.iterdir() if p.name!='stage.json'}})
        return result
    pipeline.fuse_evidence=fuse;pipeline.refine_regions=refine;pipeline.build_weight=weight
    try:yield
    finally:
        pipeline.fuse_evidence=old_fuse;pipeline.refine_regions=old_refine;pipeline.build_weight=old_weight


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--l0',type=float,default=.0005)
    parser.add_argument('--downscale',type=float,default=.6);parser.add_argument('--sigma',type=float,default=12)
    parser.add_argument('--strength',type=float,default=1)
    parser.add_argument('--structure-protection',type=float,default=pipeline.DEFAULT_STRUCTURE_PROTECTION)
    parser.add_argument('--restoration-gain',type=float,default=pipeline.DEFAULT_RESTORATION_GAIN)
    args=parser.parse_args()
    if not math.isfinite(args.l0) or args.l0<=0:raise ValueError('L0 must be finite positive')
    finish={key:validate_unit_interval(getattr(args,key),key) for key in ('structure_protection','restoration_gain')}
    source=args.input.resolve();root=args.out.resolve()
    if not source.is_dir() or root.is_relative_to(source):raise ValueError('Output must be outside source directory')
    files=sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'))
    if not files:raise ValueError('No images found')
    items=[]
    for index,path in enumerate(files,1):
        rgb,alpha,icc=_read_source(path)
        with Image.open(path) as image:
            item={'id':f'batch_{index:04d}','path':str(path),'sha256':_sha(path),'title':path.stem,
                'width':rgb.shape[1],'height':rgb.shape[0],'stored_size':list(image.size),'mode':image.mode,
                'exif_orientation':image.getexif().get(274,1),'alpha_present':alpha is not None,
                'icc_sha256':hashlib.sha256(icc).hexdigest() if icc else None}
        items.append(item)
    root.mkdir(parents=True,exist_ok=True);log=root/'log';log.mkdir(exist_ok=True)
    manifest={'schema_version':1,'source':str(source),'items':items}
    manifest_path=root/'input-manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=manifest:raise ValueError('Input inventory changed')
    else:_json(manifest_path,manifest)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; this run requests GPU')
    torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats();models=FusionModels(device='cuda')
    code={name:_sha(PROJECT/name) for name in CODE_FILES}
    config={'schema_version':1,'run_id':root.name,'created_at_utc':now(),
        'input_manifest_sha256':_sha(manifest_path),'policy':_policy(),'model_signature':models.signature,
        'model_hashes':models.identities,'code_hashes':code,
        'parameters':{'profile':'fusion','methods':list(METHODS),'lambda':args.l0,
            'downscale':args.downscale,'sigma_at_2048':args.sigma,'strength':args.strength,**finish},
        'runtime':{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'tf32':models._tf32_flags()},
        'hardware':{'gpu':torch.cuda.get_device_name(0),'gpu_bytes':torch.cuda.get_device_properties(0).total_memory},
        'instrumentation':'Snapshot wrappers call production fuse/refine/build_weight unchanged; no stage reconstructed.'}
    for package in ('numpy','scipy','Pillow','opencv-python','transformers','huggingface-hub','safetensors','spandrel'):
        try:config['runtime'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:config['runtime'][package]='not-in-distribution-metadata'
    config_path=log/'run-config.json'
    if config_path.exists():
        old=json.loads(config_path.read_text(encoding='utf-8'))
        for key in ('input_manifest_sha256','policy','model_signature','code_hashes','parameters'):
            if old[key]!=config[key]:raise ValueError('Run identity changed: '+key)
        config=old
    else:_json(config_path,config)
    events=log/'events.jsonl';native=root/'native';records=[];completed=[];errors=[]
    def event(kind,**value):
        with events.open('a',encoding='utf-8') as f:f.write(json.dumps({'time':now(),'event':kind,**value},ensure_ascii=False)+'\n')
    context={}
    try:
        with capture_stages(root,context):
            for item in items:
                pair={};entry={'id':item['id'],'methods':{}}
                for method in METHODS:
                    context.update(id=item['id'],method=method,source_sha256=item['sha256'])
                    event('start',id=item['id'],method=method);start=time.perf_counter()
                    print('START',item['id'],method,flush=True)
                    try:
                        record=pipeline.process_image(item['path'],native,models,profile='fusion',
                            case_id=item['id']+'--'+method,denoiser=method,lam=args.l0,
                            downscale=args.downscale,sigma=args.sigma,strength=args.strength,
                            **finish,
                            _input_paths=[manifest_path,config_path,*map(lambda x:Path(x['path']),items)])
                        pair[method]=record;records.append(record)
                        entry['methods'][method]={'status':record['status'],
                            'report':str((native/record['folder']/'report.json').relative_to(root)),
                            'fingerprint':record['fingerprint']}
                        event('complete',id=item['id'],method=method,seconds=time.perf_counter()-start,
                              fingerprint=record['fingerprint'],cache_hits=record['cache_hits'],resumed=record['resumed'])
                    except Exception as error:
                        message=type(error).__name__+': '+str(error);errors.append({'id':item['id'],'method':method,'error':message})
                        entry['methods'][method]={'status':'error','error':message};event('error',**errors[-1])
                        print('ERROR',message,flush=True)
                    pipeline.write_records(native,records,input_paths=[manifest_path,config_path,*[x['path'] for x in items]])
                if len(pair)==2:
                    entry['checks']=verify_pair(item['path'],native,pair)
                    if _sha(item['path'])!=item['sha256']:raise RuntimeError('Source SHA changed')
                completed.append(entry)
                _json(root/'run-manifest.json',{'schema_version':1,'run_id':root.name,'cases':completed,'errors':errors})
                print('PAIR_DONE',item['id'],flush=True)
        if errors:raise RuntimeError('One or more methods failed; inspect events and run-manifest')
        for item in items:
            if _sha(item['path'])!=item['sha256']:raise RuntimeError('Source changed during batch')
        if any(_sha(PROJECT/name)!=digest for name,digest in code.items()):raise RuntimeError('Source code changed during run')
        resume=pipeline.process_image(items[0]['path'],native,models,profile='fusion',
            case_id=items[0]['id']+'--l0',denoiser='l0',lam=args.l0,downscale=args.downscale,
            sigma=args.sigma,strength=args.strength,**finish)
        if not resume['resumed']:raise RuntimeError('Unchanged resume did not reuse its completed result')
        _json(root/'validation.json',{'schema_version':1,'case_count':len(items),'method_runs':len(records),
            'cases':completed,'source_unchanged':True,'same_masks_and_weights':True,'resume_passed':True,
            'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'errors':errors,
            'quality_accepted_by_user':False,'finished_at_utc':now()})
        event('finished',cases=len(items),method_runs=len(records));print('ALL_PAIRS_AND_TRACE_VERIFIED',flush=True)
    finally:models.release()


if __name__=='__main__':main()
