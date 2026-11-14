"""Five bounded restoration/structure ablations from exact saved candidates.

No model inference or production-policy update occurs. The experiments keep
the same labels, actual allowed mask, semantic protection, feather and strength.
"""
import argparse
import json
import math
from numbers import Real
from pathlib import Path
import re
import sys

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
if str(PROJECT/'src') not in sys.path:sys.path.insert(0,str(PROJECT/'src'))

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from gpt_animeimage_denoise.filters import (_original_rgb,_float_array,_bool_mask,_radius,
    blend,build_weight)
from gpt_animeimage_denoise.pipeline import _assert_inputs_safe,_case_folder,_read_source,_save_rgb,_sha
from tools.batch_acceptance.build_review import METHODS,load_exact_candidate,load_method,replay_final,validate_pair

VARIANTS={
    'current':{'restoration_gain':None,'weight':'current'},
    'no-restore':{'restoration_gain':0.,'weight':'current'},
    'restore25':{'restoration_gain':.25,'weight':'current'},
    'no-structure-no-restore':{'restoration_gain':0.,'weight':'soft'},
    'no-structure-restore25':{'restoration_gain':.25,'weight':'soft'},
}


def _gain(value,name='restoration gain'):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError(name+' must be finite in0..1')
    return float(value)


def _correction(original,candidate,sigma):
    original=_original_rgb(original);candidate=_float_array(candidate,original.shape,'candidate')
    sigma=_radius(sigma,'sigma',strictly_positive=True)
    residual=original.astype(np.float32)/255-candidate
    return candidate,gaussian_filter(residual,sigma=(sigma,sigma,0),mode='reflect')


def restore_candidate(original,candidate,sigma,gain):
    """Compute clip(D+gain*G(I-D)) before any clipping or PNG quantization."""
    gain=_gain(gain)
    candidate,correction=_correction(original,candidate,sigma)
    return np.clip(candidate+np.float32(gain)*correction,0,1).astype(np.float32)


def make_soft_weight(original,allowed,hard_protected,feather_radius,strength):
    """Cancel only structural veto; retain actual allowed/hard/feather rules."""
    original=_original_rgb(original)
    allowed=_bool_mask(allowed,original.shape[:2],'allowed')
    hard_protected=_bool_mask(hard_protected,original.shape[:2],'hard_protected')
    strength=_gain(strength,'strength')
    feather_radius=_radius(feather_radius,'feather radius')
    weight=build_weight(allowed,hard_protected,original,feather_radius=feather_radius,
        structure_radius=0,gradient_threshold=4)
    weight*=np.float32(strength);weight[weight<1/255]=0
    if weight[~allowed|hard_protected].any():raise AssertionError('Soft experiment expanded hard scope')
    return weight


def make_variants(original,candidate,current_weight,soft_weight,sigma,hard_protected,*,current_gain=1.):
    original=_original_rgb(original);shape=original.shape[:2]
    hard_protected=_bool_mask(hard_protected,shape,'hard_protected')
    current_weight=_float_array(current_weight,shape,'current weight')
    soft_weight=_float_array(soft_weight,shape,'soft weight')
    if current_weight[hard_protected].any() or soft_weight[hard_protected].any():
        raise ValueError('A hard-protected pixel received experiment weight')
    current_gain=_gain(current_gain)
    candidate,correction=_correction(original,candidate,sigma)
    values={}
    for name,setting in VARIANTS.items():
        gamma=current_gain if setting['restoration_gain'] is None else setting['restoration_gain']
        restored=np.clip(candidate+np.float32(gamma)*correction,0,1).astype(np.float32)
        weight=current_weight if setting['weight']=='current' else soft_weight
        result=blend(original,restored,weight)
        if (not np.array_equal(result[weight==0],original[weight==0])
                or not np.array_equal(result[hard_protected],original[hard_protected])):
            raise AssertionError('A zero-weight or hard-protected pixel changed')
        values[name]=result
    return values


def write_results(output_root,values,alpha,icc,*,input_paths=()):
    output=Path(output_root).resolve()
    if not values or any(name not in VARIANTS for name in values):raise ValueError('Unknown ablation variant')
    targets=[output/(name+'.png') for name in values]
    _assert_inputs_safe(targets,input_paths)
    output.mkdir(parents=True,exist_ok=True)
    for name,value in values.items():
        value=_original_rgb(value)
        if alpha is not None and (alpha.dtype!=np.uint8 or alpha.shape!=value.shape[:2]):
            raise ValueError('Original alpha does not match ablation grid')
        _save_rgb(output/(name+'.png'),value,alpha,icc)


def _local(root,value,*,directory=False):
    if not isinstance(value,str) or value.startswith(('/','\\')) or re.match(r'^[a-z][a-z0-9+.-]*:',value,re.I):
        raise ValueError('Expected a local relative cohort path')
    path=(root/value.replace('\\','/')).resolve()
    if not path.is_relative_to(root) or (not path.is_dir() if directory else not path.is_file()):
        raise ValueError('Cohort path is missing or escapes the run')
    return path


def _component_for(run,case_id):
    aggregate=run/'aggregate-manifest.json'
    if not aggregate.exists():return run
    document=json.loads(aggregate.read_text(encoding='utf-8'))
    matches=[c for c in document.get('components',[]) if case_id in c.get('case_ids',[])]
    if len(matches)!=1:raise ValueError('Need one actual producer cohort for '+case_id)
    prefix=matches[0].get('prefix','')
    return _local(run,prefix,directory=True) if prefix else run


def _weight_inputs(component,case,pair,rgb,alpha):
    allowed=None;inputs=[];parameters=None
    for method in METHODS:
        root=component/'trace'/case['id']/method/'weight';metadata=root/'stage.json'
        record=json.loads(metadata.read_text(encoding='utf-8'))
        if (record.get('kind')!='actual-execution-snapshot' or record.get('stage')!='weight'
                or record.get('source_sha256')!=case['source_sha256']):
            raise ValueError('Actual weight-hook metadata is required for an isolated ablation')
        inputs.append(metadata)
        for name,digest in record.get('artifacts',{}).items():
            path=_local(root,name)
            if _sha(path)!=digest:raise ValueError('Actual weight-hook artifact changed: '+name)
            inputs.append(path)
        for name in ('allowed.png','protected.png','weight_unscaled.npy'):
            if name not in record.get('artifacts',{}):raise ValueError('Weight hook lacks '+name)
        with Image.open(root/'allowed.png') as image:actual_allowed=np.array(image)>0
        with Image.open(root/'protected.png') as image:actual_protected=np.array(image)>0
        native_protected=np.array(Image.open(pair[method]['folder']/'protected.png'))>0
        if (actual_allowed.shape!=rgb.shape[:2] or not np.array_equal(actual_protected,native_protected)
                or (actual_allowed&actual_protected).any()
                or pair[method]['weight'][~actual_allowed|actual_protected].any()):
            raise ValueError('Actual allowed/protection arguments differ from native results')
        if (actual_allowed&~np.isin(pair[method]['labels'],[1,2,3])).any():
            raise ValueError('Ablation may not enable eyes, mouth, clothes, unknown or background')
        if allowed is not None and (not np.array_equal(allowed,actual_allowed) or parameters!=record['parameters']):
            raise ValueError('Two algorithms did not use identical weight inputs')
        allowed=actual_allowed;parameters=record['parameters']
        unscaled=np.load(root/'weight_unscaled.npy',allow_pickle=False)
        unscaled=_float_array(unscaled,rgb.shape[:2],'recorded unscaled weight').copy()
        strength=_gain(pair[method]['report']['parameters']['strength'],'strength')
        unscaled*=np.float32(strength);unscaled[unscaled<1/255]=0
        if not np.array_equal(unscaled,pair[method]['weight']):raise ValueError('Hook weight does not reproduce native strength/threshold')
    scale=min(rgb.shape[:2])/2048
    if not math.isclose(float(parameters['feather_radius']),16*scale,rel_tol=0,abs_tol=1e-8):
        raise ValueError('This bounded experiment retains the actual16@2048 feather')
    hard=np.array(Image.open(pair['l0']['folder']/'protected.png'))>0
    hard|=~np.isin(pair['l0']['labels'],[1,2,3])|pair['l0']['objects']|pair['l0']['unknown']
    if alpha is not None:hard|=alpha==0
    if (allowed&hard).any():raise ValueError('Actual allowed would release a required hard protection')
    return allowed,hard,parameters,inputs


def _prepare_case(run,case):
    component=_component_for(run,case['id'])
    folders={method:_case_folder(component/'native',case['id']+'--'+method) for method in METHODS}
    report_paths={m:folders[m]/'report.json' for m in METHODS}
    initial=json.loads(report_paths['l0'].read_text(encoding='utf-8'))
    source=Path(initial['source']).resolve()
    if _sha(source)!=case['source_sha256']:raise ValueError('Persistent source SHA differs')
    rgb,alpha,icc=_read_source(source)
    pair={m:load_method(folders[m],case['source_sha256'],rgb,alpha,icc,m) for m in METHODS}
    validate_pair(pair)
    for method in METHODS:
        expected=case['methods'][method]['report_fingerprint']
        if pair[method]['report']['fingerprint']!=expected:raise ValueError('Viewer refers to another native result')
        copied=_local(run,case['methods'][method]['report'])
        if _sha(copied)!=_sha(report_paths[method]):raise ValueError('Viewer/native copied report differs')
    allowed,hard,parameters,hook_inputs=_weight_inputs(component,case,pair,rgb,alpha)
    inputs=[source,*hook_inputs,*[p for method in METHODS for p in pair[method]['input_paths']]]
    inputs += [_local(run,case['methods'][m]['report']) for m in METHODS]
    return {'source':source,'component':component,'pair':pair,'rgb':rgb,'alpha':alpha,'icc':icc,
        'allowed':allowed,'hard':hard,'weight_parameters':parameters,'inputs':list(dict.fromkeys(inputs))}


def _targets(output,case_id):
    root=_case_folder(output,case_id)
    result=[root/name for name in ('original.png','allowed.png','hard-protected.png','weight-current.npy',
                                  'weight-soft.npy','weight-current.png','weight-soft.png','trace.json')]
    for method in METHODS:
        result += [root/method/(name+'.png') for name in VARIANTS]
        result.append(root/method/'trace.json')
    return root,result


def _write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)


def export_case(run,output,case):
    loaded=_prepare_case(run,case);root,targets=_targets(output,case['id'])
    _assert_inputs_safe([p for x in targets for p in (x,x.with_suffix(x.suffix+'.tmp'))],loaded['inputs'])
    snapshots={str(p):_sha(p) for p in loaded['inputs']}
    rgb,alpha,icc=loaded['rgb'],loaded['alpha'],loaded['icc'];pair=loaded['pair']
    current=pair['l0']['weight'];parameters=loaded['weight_parameters']
    strength=pair['l0']['report']['parameters']['strength']
    soft=make_soft_weight(rgb,loaded['allowed'],loaded['hard'],parameters['feather_radius'],strength)
    root.mkdir(parents=True,exist_ok=True);relative=lambda p:p.relative_to(output).as_posix()
    _save_rgb(root/'original.png',rgb,alpha,icc)
    for name,array in [('allowed',loaded['allowed']),('hard-protected',loaded['hard'])]:
        Image.fromarray(array.astype(np.uint8)*255).save(root/(name+'.png'))
    for name,array in [('weight-current',current),('weight-soft',soft)]:
        np.save(root/(name+'.npy'),array,allow_pickle=False)
        Image.fromarray(np.ceil(array*255).astype(np.uint8)).save(root/(name+'.png'))
    methods={}
    for method in METHODS:
        value=pair[method];report=value['report']
        candidate=load_exact_candidate(value['folder'].parent/'.cache',report,rgb.shape)
        gain=_gain(report['parameters'].get('restoration_gain',1.))
        structure=_gain(report['parameters'].get('structure_protection',1.),'structure protection')
        replay_final(rgb,candidate,current,report['effective_sigma'],value['final'],restoration_gain=gain)
        results=make_variants(rgb,candidate,current,soft,report['effective_sigma'],loaded['hard'],current_gain=gain)
        if not np.array_equal(results['current'],value['final']):raise AssertionError('Current ablation failed exact replay')
        method_root=root/method;write_results(method_root,results,alpha,icc,input_paths=loaded['inputs'])
        trace={'method':method,'report':str(value['folder']/'report.json'),'report_sha256':_sha(value['folder']/'report.json'),
            'report_fingerprint':report['fingerprint'],'candidate_cache_identity':report['candidate_cache_identity'],
            'candidate_cache_stage':report['candidate_cache_stage'],'effective_sigma':report['effective_sigma'],
            'same_float_candidate_for_all_variants':True,'variants':{},'production_default_changed':False}
        for name,image in results.items():
            path=method_root/(name+'.png');setting=VARIANTS[name]
            without_structure=setting['weight']=='soft';weight=soft if without_structure else current
            with Image.open(path) as saved:
                if saved.size!=(rgb.shape[1],rgb.shape[0]) or saved.info.get('icc_profile')!=icc:
                    raise AssertionError('Experiment output size/ICC changed')
                if alpha is not None and not np.array_equal(np.array(saved.getchannel('A')),alpha):
                    raise AssertionError('Experiment original alpha changed')
                if not np.array_equal(np.array(saved.convert('RGB')),image):raise AssertionError('Saved output RGB differs')
            trace['variants'][name]={**setting,'image':relative(path),'sha256':_sha(path),
                'restoration_gain':gain if setting['restoration_gain'] is None else setting['restoration_gain'],
                'structure_protection':0. if without_structure else structure,
                'changed_pixels':int(np.any(image!=rgb,axis=2).sum()),'weight_active_pixels':int((weight>0).sum()),
                'own_w0_and_hard_protection_preserved':True}
        _write_json(method_root/'trace.json',trace);methods[method]=trace
    trace={'id':case['id'],'title':case.get('title',case['id']),'source':str(loaded['source']),
        'source_sha256':case['source_sha256'],'width':rgb.shape[1],'height':rgb.shape[0],
        'original':relative(root/'original.png'),'component_root':str(loaded['component']),
        'methods':methods,'weight_inputs':'actual execution-hook allowed/protected and unscaled weight',
        'original_weight_parameters':parameters,
        'soft_weight_parameters':{**parameters,'structure_radius':0,'structure_protection':0.},
        'strength':strength,'weight_zero_threshold':1/255,
        'weights':{'current':relative(root/'weight-current.npy'),'soft':relative(root/'weight-soft.npy')},
        'hard_protected':relative(root/'hard-protected.png'),'allowed':relative(root/'allowed.png'),
        'alpha_status':'verified' if alpha is not None else 'not_present',
        'icc_status':'verified' if icc is not None else 'not_present',
        'experimental_warning':'取消结构保护可能损伤线稿；仅实验，未切换生产默认。',
        'w0_scope':'每个实验自身W=0及硬保护不变；soft实验可改变原结构保护造成的W=0，但不解除语义范围。',
        'input_sha256':snapshots,'production_default_changed':False}
    if any(_sha(p)!=digest for p,digest in snapshots.items()):raise RuntimeError('An original producer input changed during ablation')
    _write_json(root/'trace.json',trace)
    return trace


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ids',nargs='+',required=True,help='One to three case IDs from the acceptance manifest')
    args=parser.parse_args(argv);run=args.run.resolve();output=args.out.resolve()
    if output==run or output.is_relative_to(run):raise ValueError('Experimental outputs must be separate from the original acceptance run')
    if not 1<=len(args.ids)<=3 or len(set(args.ids))!=len(args.ids):raise ValueError('This bounded experiment accepts1..3 distinct cases')
    path=run/'manifest.json';document=json.loads(path.read_text(encoding='utf-8'))
    by_id={c['id']:c for c in document['cases']}
    if any(cid not in by_id for cid in args.ids):raise ValueError('A requested case is not in the acceptance run')
    cases=[by_id[cid] for cid in args.ids]
    target=output/'ablation-manifest.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf-8'))
        if previous.get('kind')!='bounded-denoise-effect-ablation' or previous.get('source_manifest_sha256')!=_sha(path) or previous.get('case_ids')!=args.ids:
            raise FileExistsError('Keep the prior experiment intact; choose another output directory')
    inputs=[path];targets=[target,target.with_suffix('.json.tmp')]
    for case in cases:
        prepared=_prepare_case(run,case);inputs+=prepared['inputs']
        targets += [p for x in _targets(output,case['id'])[1] for p in (x,x.with_suffix(x.suffix+'.tmp'))]
        del prepared
    _assert_inputs_safe(targets,inputs);snapshots={str(p):_sha(p) for p in inputs}
    output.mkdir(parents=True,exist_ok=True);entries=[]
    for case in cases:
        print('ABLATE',case['id'],flush=True);entries.append(export_case(run,output,case))
    if any(_sha(p)!=digest for p,digest in snapshots.items()):raise RuntimeError('Original run changed during experiment')
    result={'schema_version':1,'kind':'bounded-denoise-effect-ablation','source_run_id':document['run_id'],
        'source_manifest':str(path),'source_manifest_sha256':_sha(path),'case_ids':args.ids,
        'variants':VARIANTS,'cases':entries,'methods':list(METHODS),
        'variant_note':'A null restoration gain uses the actual per-method report setting.',
        'formula':'C_gamma=clip(D+gamma*Gaussian(I-D),0,1); gain is applied before clipping; blend is linear RGB',
        'same_candidate_masks_feather_strength':True,'only_structure_veto_and_restoration_gain_varied':True,
        'production_default_changed':False,'new_model_inference':False,'ground_truth_accuracy_claimed':False,
        'code_sha256':_sha(Path(__file__)),'input_sha256':snapshots}
    _write_json(target,result);print('EXPORTED_BOUNDED_ABLATION',len(entries)*len(METHODS)*len(VARIANTS),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
