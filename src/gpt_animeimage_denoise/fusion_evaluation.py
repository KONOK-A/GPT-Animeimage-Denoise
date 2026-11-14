"""Reference review and ROI-scoped V2.1 quality metrics; no implicit approval."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_erosion, distance_transform_edt
from .pipeline import _assert_inputs_safe,_case_folder

NAMES = ('background', 'skin', 'face', 'hair', 'eyes', 'mouth', 'outfit')
COLORS = np.array([[245,245,245],[255,220,180],[100,150,255],[255,60,60],
                   [0,220,230],[255,0,150],[180,0,255],[255,125,20]], np.uint8)
KNOWN_CASES = ('img_0001','img_0002','img_0003','img_0017','img_0023',
               'img_0029','img_0035','img_0043','img_0047','img_0051')
# Display names for exact saved candidates, including retired methods in old reports.
DENOISER_NAMES = {'l0':'pure L0','esrgan':'ESRGAN','esrgan-l0':'ESRGAN + L0',
                  'cugan':'CUGAN','cugan-l0':'CUGAN + L0','apisr':'APISR'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_grid():
    for primary in (.60,.75,.90):
        for auxiliary in (0,1,2):
            yield {'m2f_threshold':primary,'see_threshold':auxiliary,'margin':.15}


def require_approved_references(cases, split):
    cases = list(cases)
    if not cases or any(c.get('approved') is not True for c in cases):
        raise ValueError('User reference approval required before quality calibration/evaluation')
    if any(c.get('split') != split for c in cases):
        raise ValueError('Reference split mismatch; never use holdout for calibration')
    return cases


def select_review_cases(manifest, count=12):
    eligible = {i['id']:i for i in manifest['items']
                if i.get('status') == 'eligible' and i.get('split') == 'tune'}
    selected = [eligible[i] for i in KNOWN_CASES if i in eligible][:count]
    remaining = [eligible[i] for i in sorted(eligible) if i not in {s['id'] for s in selected}]
    # Farthest dHash diversity augments known regressions without adjusting rules
    # to any source image's coordinates. Same input manifest -> same selection.
    while remaining and len(selected) < count:
        def separation(item):
            value = int(item.get('dhash','0'),16)
            return min(((value ^ int(s.get('dhash','0'),16)).bit_count() for s in selected),default=64)
        chosen = max(remaining, key=lambda i:(separation(i),i['id']))
        selected.append(chosen);remaining.remove(chosen)
    return selected


def segmentation_metrics(reference, prediction, objects, valid_mask=None):
    reference = np.asarray(reference);prediction = np.asarray(prediction);objects = np.asarray(objects)
    if reference.ndim != 2 or prediction.shape != reference.shape or objects.shape != reference.shape or objects.dtype != bool:
        raise ValueError('Expected aligned HxW references/predictions and bool objects')
    if not np.isin(reference,[0,1,2,3,4,5,6,7,255]).all() or not np.isin(prediction,[0,1,2,3,4,5,6,255]).all():
        raise ValueError('Invalid label IDs')
    valid = reference != 255
    if valid_mask is not None:
        if np.asarray(valid_mask).dtype != bool or np.asarray(valid_mask).shape != reference.shape:
            raise ValueError('valid_mask must be aligned bool')
        valid &= valid_mask
    people = valid & np.isin(reference,range(1,7));external = valid & (reference == 7)
    modules = {}
    radius = max(1,round(min(reference.shape)/2048*2))
    for i,name in enumerate(NAMES[1:],1):
        gold=(reference==i)&valid;pred=(prediction==i)&valid
        tp=int((gold&pred).sum());fp=int((~gold&pred&valid).sum());fn=int((gold&~pred).sum())
        # Explicitly use a 2px@shortedge2048 inside-boundary band. Do not form
        # artificial boundary scores at the edge of the reviewed ROI itself.
        gb=gold&~binary_erosion(gold,iterations=radius,border_value=0)
        pb=pred&~binary_erosion(pred,iterations=radius,border_value=0)
        interior=binary_erosion(valid,iterations=radius,border_value=0)
        gb &= interior;pb &= interior
        boundary_union=int((gb|pb).sum())
        modules[name]={'tp':tp,'fp':fp,'fn':fn,'support':int(gold.sum()),
                       'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
                       'precision':tp/(tp+fp) if tp+fp else None,
                       'recall':tp/(tp+fn) if tp+fn else None,
                       'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                       'boundary_intersection':int((gb&pb).sum()),'boundary_union':boundary_union,
                       'boundary_iou':int((gb&pb).sum())/boundary_union if boundary_union else None}
    present=[m['f1'] for m in modules.values() if m['support'] and m['f1'] is not None]
    person_count=int(people.sum());object_count=int(external.sum())
    return {'scope':'approved reference ROI pixels only','valid_pixels':int(valid.sum()),
            'unknown_pixels':int(((prediction==255)&valid).sum()),
            'person_pixels':person_count,'object_pixels':object_count,
            'person_false_excluded_pixels':int((objects&people).sum()),
            'outfit_object_leakage_pixels':int(((prediction==6)&external).sum()),
            'unknown_on_objects_pixels':int(((prediction==255)&external).sum()),
            'external_failure_pixels':int(((prediction!=0)&external).sum()),
            'person_false_exclusion':int((objects&people).sum())/person_count if person_count else None,
            'outfit_object_leakage':int(((prediction==6)&external).sum())/object_count if object_count else None,
            'external_failure_rate':int(((prediction!=0)&external).sum())/object_count if object_count else None,
            'macro_f1':float(np.mean(present)) if present else None,'modules':modules}


def aggregate_metrics(results):
    results=list(results)
    if not results:raise ValueError('No reviewed metrics')
    total={key:sum(r[key] for r in results) for key in ['valid_pixels','unknown_pixels','person_pixels','object_pixels',
             'person_false_excluded_pixels','outfit_object_leakage_pixels','unknown_on_objects_pixels','external_failure_pixels']}
    total['modules']={}
    for name in NAMES[1:]:
        row={key:sum(r['modules'][name][key] for r in results) for key in
             ['tp','fp','fn','support','boundary_intersection','boundary_union']}
        tp,fp,fn=row['tp'],row['fp'],row['fn']
        row.update(iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                   precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                   f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                   boundary_iou=row['boundary_intersection']/row['boundary_union'] if row['boundary_union'] else None)
        total['modules'][name]=row
    f1=[r['f1'] for r in total['modules'].values() if r['support'] and r['f1'] is not None]
    total.update(scope='approved reference ROI pixels only',macro_f1=float(np.mean(f1)) if f1 else None,
        person_false_exclusion=total['person_false_excluded_pixels']/total['person_pixels'] if total['person_pixels'] else None,
        outfit_object_leakage=total['outfit_object_leakage_pixels']/total['object_pixels'] if total['object_pixels'] else None,
        external_failure_rate=total['external_failure_pixels']/total['object_pixels'] if total['object_pixels'] else None)
    return total


def release_gate(baseline, candidate):
    reasons=[]
    if not candidate['person_pixels'] or not candidate['object_pixels'] or any(
            not candidate['modules'][name]['support'] or not baseline['modules'][name]['support'] for name in NAMES[1:]):
        reasons.append('insufficient_reference')
    if candidate['person_false_exclusion'] is not None and candidate['person_false_exclusion']>.01:
        reasons.append('person_false_exclusion_above_1_percent')
    old=baseline['outfit_object_leakage'];new=candidate['outfit_object_leakage']
    reduction=(old-new)/old if old is not None and old>0 and new is not None else None
    if reduction is None or reduction < .25:reasons.append('object_leakage_reduction_below_25_percent_or_undefined')
    effective_old=baseline['external_failure_rate'];effective_new=candidate['external_failure_rate']
    effective_reduction=(effective_old-effective_new)/effective_old if effective_old is not None and effective_old>0 and effective_new is not None else None
    if effective_reduction is None or effective_reduction<.25:
        reasons.append('unresolved_or_misclassified_objects_not_improved_25_percent')
    if candidate['macro_f1'] is not None and baseline['macro_f1'] is not None and candidate['macro_f1']+1e-12<baseline['macro_f1']:
        reasons.append('macro_f1_decreased')
    for name in NAMES[1:]:
        a=baseline['modules'][name]['f1'];b=candidate['modules'][name]['f1']
        if a is not None and b is not None and a-b>.02+1e-12:reasons.append(name+'_f1_decreased_over_2pp')
    return {'accepted':not reasons,'reasons':reasons,'object_leakage_relative_reduction':reduction,
            'external_failure_relative_reduction':effective_reduction}


def _font(size):
    path=Path('C:/Windows/Fonts/msyh.ttc')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default()


def prepare_review(manifest_path, predictions_root, output, count=12):
    """Create editable model prelabels. Every case starts explicitly unapproved."""
    manifest_path=Path(manifest_path);root=Path(predictions_root).resolve();out=Path(output).resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (out/'review_manifest.json').exists():raise FileExistsError('Keep existing review/approval state; choose a new output')
    chosen=select_review_cases(manifest,count)
    chosen += [i for i in manifest['items'] if i.get('status')=='eligible' and i.get('split')=='holdout']
    inputs=[manifest_path,*[Path(i['path']) for i in manifest['items']]]
    targets=[out/'review_manifest.json',out/'README.txt']
    for item in chosen:
        folder=_case_folder(out,item['id'])
        targets += [folder/name for name in ['reference_labels.png','proposed_overlay.png','review.png']]
        targets += [folder/(n+suffix+'.png') for n in [*NAMES[1:],'objects'] for suffix in ['_original','_proposed']]
    _assert_inputs_safe(targets,inputs)
    out.mkdir(parents=True,exist_ok=True);cases=[]
    for item in chosen:
        source=Path(item['path'])
        if _sha(source)!=item['sha256']:raise ValueError('Source differs from frozen dataset')
        folder=_case_folder(out,item['id']);folder.mkdir(exist_ok=True)
        with Image.open(source) as stored:
            oriented=ImageOps.exif_transpose(stored).convert('RGBA')
            visible=Image.new('RGBA',oriented.size,'white');visible.alpha_composite(oriented)
            rgb=np.array(visible.convert('RGB'))
        source_folder=root/item['id']
        label_path=source_folder/'person_labels.png'
        if not label_path.exists():label_path=source_folder/'labels.png'
        if label_path.exists():
            labels=np.array(Image.open(label_path).convert('L'))
            if labels.shape!=rgb.shape[:2] or not np.isin(labels,[0,1,2,3,4,5,6,255]).all():
                raise ValueError('Invalid prelabel image: '+str(label_path))
            for name in ['object_exclusion.png','objects.png']:
                if (source_folder/name).exists():
                    objects=np.array(Image.open(source_folder/name).convert('L'))>0
                    if objects.shape!=labels.shape:raise ValueError('Objects shape mismatch')
                    labels[objects]=7;break
        else:
            labels=np.full(rgb.shape[:2],255,np.uint8)
        Image.fromarray(labels).save(folder/'reference_labels.png')
        colors=np.full_like(rgb,185)
        known=labels<8;colors[known]=COLORS[labels[known]]
        overlay=np.rint(rgb*.5+colors*.5).astype(np.uint8)
        Image.fromarray(overlay).save(folder/'proposed_overlay.png')
        rois=[]
        board=Image.new('RGB',(960,650),'#edf1f5');draw=ImageDraw.Draw(board)
        for col,(name,array) in enumerate([('原图',rgb),('模型预标注：尚未确认',overlay)]):
            view=Image.fromarray(array);view.thumbnail((450,570),Image.Resampling.LANCZOS)
            board.paste(view,(col*480+(480-view.width)//2,55));draw.text((col*480+15,12),name,font=_font(22),fill='#263747')
        board.save(folder/'review.png')
        for value,name in list(enumerate(NAMES[1:],1))+[(7,'objects')]:
            part=labels==value
            if not part.any():continue
            field=distance_transform_edt(part);cy,cx=np.unravel_index(field.argmax(),field.shape)
            side=min(384,rgb.shape[0],rgb.shape[1]);x=max(0,min(int(cx)-side//2,rgb.shape[1]-side));y=max(0,min(int(cy)-side//2,rgb.shape[0]-side))
            box=[x,y,x+side,y+side];Image.fromarray(rgb).crop(box).save(folder/(name+'_original.png'))
            Image.fromarray(overlay).crop(box).save(folder/(name+'_proposed.png'))
            rois.append({'name':name,'xyxy':box,'restoration_judgment':'pending',
                         'texture_kind':'unclassified','note':'Review original; do not treat model prelabels as truth'})
        cases.append({'id':item['id'],'split':item['split'],'group_id':item.get('group_id'),
                      'source':str(source),'source_sha256':item['sha256'],'size':[rgb.shape[1],rgb.shape[0]],
                      'reference_labels':item['id']+'/reference_labels.png','approved':False,'rois':rois,
                      'prelabel_available':bool(label_path.exists()),'approval_note':''})
    document={'schema_version':1,'source_manifest':str(manifest_path.resolve()),'source_manifest_sha256':_sha(manifest_path),
              'label_ids':{str(i):name for i,name in enumerate(NAMES)}|{'7':'external_objects','255':'ignore_ambiguous_reference'},
              'prediction_unknown_counts_as_error':True,'cases':cases,'approval_required':True,
              'instructions':'Correct reference_labels.png and ROI boxes against original. Set approved=true only after user review. Gold255 is ignored; predicted255 is incomplete. Do not calibrate on holdout.'}
    path=out/'review_manifest.json';path.write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README.txt').write_text('模型预标注不是真值。请核对review.png与原图，修正reference_labels.png。标签0背景，1皮肤，2脸，3头发，4眼睛，5嘴，6服饰，7外物，255模糊/忽略。仅确认ROI覆盖的像素用于评测。核对后将对应approved改为true；不要把待确认预测排除出指标。保留集只用于冻结后的验收。\n',encoding='utf-8')
    return document


def load_reviewed_cases(review_manifest, split):
    path=Path(review_manifest);document=json.loads(path.read_text(encoding='utf-8'))
    selected=[c for c in document['cases'] if c['split']==split]
    require_approved_references(selected,split)
    _validate_review_document(document)
    loaded=[]
    for case in selected:
        if _sha(case['source'])!=case['source_sha256']:raise ValueError('Reviewed source hash mismatch')
        with Image.open(path.parent/case['reference_labels']) as stored:
            if list(stored.size)!=case['size']:raise ValueError('Reviewed label dimensions mismatch')
            labels=np.array(stored.convert('L'))
        valid=np.zeros(labels.shape,bool)
        for roi in case['rois']:
            x0,y0,x1,y1=roi['xyxy']
            if not(0<=x0<x1<=labels.shape[1] and 0<=y0<y1<=labels.shape[0]):raise ValueError('Reviewed ROI out of bounds')
            valid[y0:y1,x0:x1]=True
        if not valid.any():raise ValueError('No reviewed ROI coverage')
        loaded.append((case,labels,valid))
    return loaded


def _validate_review_document(document):
    frozen=Path(document.get('source_manifest',''))
    if not frozen.is_file() or _sha(frozen)!=document.get('source_manifest_sha256'):
        raise ValueError('Frozen source manifest is missing or its hash changed')
    inventory=json.loads(frozen.read_text(encoding='utf-8'))
    items=inventory['items'];by_id={i['id']:i for i in items}
    if len(by_id)!=len(items):raise ValueError('Duplicate dataset IDs')
    group_splits={};source_splits={}
    for item in items:
        if item.get('status')!='eligible':continue
        for key,mapping in [(item.get('group_id'),group_splits),(item.get('sha256'),source_splits)]:
            if not key:raise ValueError('Frozen dataset lacks group/source identity')
            mapping.setdefault(key,set()).add(item['split'])
    if any(len(v)>1 for mapping in [group_splits,source_splits] for v in mapping.values()):
        raise ValueError('Source/group crosses frozen tune/holdout split')
    seen=set();groups={};sources={}
    for case in document['cases']:
        _case_folder(frozen.parent,case['id'])
        if case['id'] in seen:raise ValueError('Duplicate reference case ID')
        seen.add(case['id']);item=by_id.get(case['id'])
        if item is None or item.get('status')!='eligible':raise ValueError('Reference is not an eligible frozen case')
        if (case['split']!=item['split'] or case.get('group_id')!=item.get('group_id')
                or case['source_sha256']!=item['sha256'] or Path(case['source']).resolve()!=Path(item['path']).resolve()):
            raise ValueError('Reference source/group/split differs from frozen manifest')
        if case['size']!=[item['width'],item['height']]:raise ValueError('Reference size differs from frozen manifest')
        groups.setdefault(case['group_id'],set()).add(case['split'])
        sources.setdefault(case['source_sha256'],set()).add(case['split'])
    if any(len(v)>1 for mapping in [groups,sources] for v in mapping.values()):
        raise ValueError('Reference group/source crosses split')


def review_input_paths(review_manifest):
    path=Path(review_manifest).resolve();doc=json.loads(path.read_text(encoding='utf-8'))
    result=[path,Path(doc['source_manifest'])]
    for case in doc['cases']:
        result.extend([Path(case['source']),path.parent/case['reference_labels']])
    return result


def safe_evaluation_output(output, review_manifest, *, policy=None, prediction_roots=(),additional_outputs=()):
    inputs=review_input_paths(review_manifest)
    if policy is not None:inputs.append(Path(policy))
    for root in prediction_roots:
        root=Path(root)
        if root.exists():
            # Only files are read here; no directory links are followed/changed.
            inputs.extend(p for p in root.rglob('*') if p.is_file())
            # Resolve the cache explicitly: rglob need not follow a junction.
            cache=root/'.cache'
            if cache.exists():inputs.extend(p for p in cache.glob('*') if p.is_file())
    path=Path(output)
    _assert_inputs_safe([path,path.with_suffix(path.suffix+'.tmp'),*additional_outputs],inputs)
    return inputs


def _validated_prediction(folder, case, expected_profile):
    folder=Path(folder)
    report=json.loads((folder/'report.json').read_text(encoding='utf-8'))
    if report.get('status')!='complete':raise ValueError('Predictions must be complete before quality evaluation')
    if report.get('profile')!=expected_profile:raise ValueError('Prediction profile differs from required '+expected_profile)
    if report.get('source_sha256')!=case['source_sha256']:raise ValueError('Prediction source identity differs')
    if expected_profile=='reference' and (report.get('edits') or report.get('sam_records')):
        raise ValueError('Reference baseline must have no per-image corrections')
    hashes=report.get('output_hashes',{})
    object_name='objects.png' if (folder/'objects.png').exists() else 'object_exclusion.png'
    if not {'person_labels.png',object_name}.issubset(hashes):raise ValueError('Prediction output hashes missing')
    for name,digest in hashes.items():
        if not isinstance(name,str) or any(c in name for c in '/\\:') or name in ['.','..']:
            raise ValueError('Invalid prediction artifact name')
        file=folder/name
        if not file.is_file() or _sha(file)!=digest:raise ValueError('Prediction artifact hash differs: '+name)
    pred=np.array(Image.open(folder/'person_labels.png'))
    objects=np.array(Image.open(folder/object_name))>0
    return report,pred,objects


def evaluate_runs(review_manifest, baseline_root, candidate_root, *, split='holdout'):
    records=[]
    for case,gt,valid in load_reviewed_cases(review_manifest,split):
        pair=[]
        for root,profile in [(baseline_root,'reference'),(candidate_root,'fusion')]:
            folder=Path(root)/case['id']
            _,pred,objects=_validated_prediction(folder,case,profile)
            pair.append(segmentation_metrics(gt,pred,objects,valid))
        records.append({'id':case['id'],'baseline':pair[0],'candidate':pair[1]})
    baseline=aggregate_metrics([r['baseline'] for r in records]);candidate=aggregate_metrics([r['candidate'] for r in records])
    return {'split':split,'reference_approved':True,'baseline':baseline,'candidate':candidate,
            'gate':release_gate(baseline,candidate),'cases':records,'note':'ROI-scoped metrics, not full-image ground truth; denoising judgments require separate review'}


def reference_digest(review_manifest, split='tune'):
    path=Path(review_manifest);cases=load_reviewed_cases(path,split)
    identity=[{'id':case['id'],'source':case['source_sha256'],'labels':_sha(path.parent/case['reference_labels']),
               'rois':case['rois'],'group_id':case.get('group_id')} for case,_,_ in cases]
    return hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def calibrate_policies(review_manifest, candidate_provider, baseline_provider, *, output=None):
    """Select from exactly nine dev-only policies, gated before any model call.

    Providers return final region dicts with person_labels and objects. They may
    use pipeline caches; this function never reads or selects on holdout labels.
    """
    cases=load_reviewed_cases(review_manifest,'tune')
    if output is not None:
        target=Path(output)
        if (target/'calibration.json').exists() or (target/'frozen_policy.json').exists():
            raise FileExistsError('Keep frozen calibration immutable; choose a new output')
        _assert_inputs_safe([target/'calibration.json',target/'frozen_policy.json'],review_input_paths(review_manifest))
    initial_reference=reference_digest(review_manifest,'tune')
    from .fusion import DEFAULT_POLICY
    from copy import deepcopy
    policies=[]
    for values in threshold_grid():
        policy=deepcopy(DEFAULT_POLICY)
        policy.update(m2f=values['m2f_threshold'],see=values['see_threshold'],margin=values['margin'])
        policies.append(policy)
    baseline_metrics=[];candidate_metrics=[[] for _ in policies]
    for case,gt,valid in cases:
        base=baseline_provider(case)
        baseline_metrics.append(segmentation_metrics(gt,base['person_labels'],base['objects'],valid))
        for i,policy in enumerate(policies):
            result=candidate_provider(case,policy)
            candidate_metrics[i].append(segmentation_metrics(gt,result['person_labels'],result['objects'],valid))
    baseline=aggregate_metrics(baseline_metrics);rows=[];eligible=[]
    for policy,metrics in zip(policies,candidate_metrics):
        score=aggregate_metrics(metrics)
        safe=score['person_false_exclusion'] is not None and score['person_false_exclusion']<=.01
        safe &= score['macro_f1'] is not None and baseline['macro_f1'] is not None and score['macro_f1']+1e-12>=baseline['macro_f1']
        safe &= all(score['modules'][n]['support'] and baseline['modules'][n]['support'] and
                    score['modules'][n]['f1']+0.02+1e-12>=baseline['modules'][n]['f1'] for n in NAMES[1:])
        safe &= score['object_pixels']>0
        row={'policy':policy,'metrics':score,'eligible':bool(safe)};rows.append(row)
        if safe:eligible.append(row)
    selected=min(eligible,key=lambda r:(r['metrics']['external_failure_rate'],
                r['metrics']['outfit_object_leakage'],-r['metrics']['macro_f1'],r['metrics']['unknown_pixels'],
                r['policy']['m2f'],r['policy']['see'])) if eligible else None
    if reference_digest(review_manifest,'tune')!=initial_reference:
        raise RuntimeError('Reviewed references changed during calibration')
    frozen=None
    if selected is not None:
        frozen=deepcopy(selected['policy'])
        frozen.update(provisional=False,calibrated=True,release_validated=False,
                      calibration_reference_sha256=initial_reference)
    report={'schema_version':1,'status':'calibrated' if frozen else 'no_safe_policy',
            'reference_approved':True,'split':'tune','baseline':baseline,'candidates':rows,
            'selected_policy':frozen,'holdout_used':False,
            'note':'Calibration does not imply held-out quality acceptance.'}
    if output is not None:
        target=Path(output);target.mkdir(parents=True,exist_ok=True)
        (target/'calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if frozen is not None:
            (target/'frozen_policy.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def evaluate_release(review_manifest, baseline_root, candidate_root, policy_path):
    from .fusion import _policy
    policy=json.loads(Path(policy_path).read_text(encoding='utf-8'))
    if policy.get('provisional',True) or not policy.get('calibrated'):
        raise ValueError('A calibrated frozen policy is required for holdout acceptance')
    if policy.get('calibration_reference_sha256')!=reference_digest(review_manifest,'tune'):
        raise ValueError('Calibration reference changed; frozen policy is stale')
    expected_policy=_policy(policy)
    reviewed=load_reviewed_cases(review_manifest,'holdout')
    for case,_,_ in reviewed:
        record,_,_=_validated_prediction(Path(candidate_root)/case['id'],case,'fusion')
        actual=record.get('parameters',{}).get('policy',{})
        if _policy(actual)!=expected_policy:raise ValueError('Holdout predictions do not match the frozen policy')
    result=evaluate_runs(review_manifest,baseline_root,candidate_root,split='holdout')
    result['frozen_policy_sha256']=_sha(policy_path)
    return result


def _review_candidate(cache_root, report, shape, *, replay=None):
    """Read the report's exact float candidate cache, never an exported image."""
    from .fusion_pipeline import _payload_hash,_unpack,_code_hash
    from .pipeline import _digest
    parameters=report['parameters'];denoiser=parameters.get('denoiser','l0')
    if denoiser not in DENOISER_NAMES:raise ValueError('Unknown review denoiser')
    uses_l0=denoiser=='l0' or denoiser.endswith('-l0')
    identity=report.get('candidate_cache_identity');stage=report.get('candidate_cache_stage')
    if identity is not None or stage is not None or 'denoiser' in parameters:
        required={'source_sha256','lambda','device','denoiser','denoiser_config','code','callable','precision'}
        if not isinstance(identity,dict) or not required.issubset(identity):
            raise ValueError('Exact candidate cache identity must be declared by the prediction report')
        if stage not in ['candidate','l0'] or (denoiser!='l0' and stage!='candidate'):
            raise ValueError('Invalid candidate cache stage; original-grid float cache required')
        expected_lambda=parameters['lambda'] if uses_l0 else None
        if (identity['source_sha256']!=report['source_sha256'] or identity['lambda']!=expected_lambda
                or identity['denoiser']!=denoiser or identity['denoiser_config']!=parameters.get('denoiser_config',{})
                or identity['callable']!=parameters['l0_implementation']
                or identity['device']!=parameters.get('device',identity['device'])
                or identity['precision']!='float32-cache'):
            raise ValueError('Prediction candidate cache identity differs from report parameters')
        path=Path(cache_root)/(stage+'-'+_digest(identity)+'.npz')
        if not path.is_file():
            raise ValueError('Exact float candidate cache unavailable for review; no PNG reconstruction is allowed')
        with np.load(path,allow_pickle=False) as saved:
            meta=json.loads(str(saved['metadata'].item()))
            if meta['identity']!=identity:raise ValueError('Candidate review cache identity mismatch')
            arrays={name:saved[name] for name in saved.files if name!='metadata'}
            if meta['payload_hash']!=_payload_hash(meta['tree'],arrays):
                raise ValueError('Candidate review cache hash mismatch')
            value=_unpack(meta['tree'],arrays)
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=shape
                or not np.isfinite(value).all() or np.any((value<0)|(value>1))):
            raise ValueError('Invalid float review candidate')
        return value
    # Historical reports retain the original current-code restriction. A
    # matching replay is only an additional consistency check: masked or
    # quantized output cannot prove the identity of a different-code candidate.
    expected_code=_code_hash(['../denoise.py','../l0_cuda.py','fusion_pipeline.py'])
    selected=None;matched_identity=False
    for path in sorted(Path(cache_root).glob('l0-*.npz')):
        with np.load(path,allow_pickle=False) as saved:
            meta=json.loads(str(saved['metadata'].item()));identity=meta['identity']
            if ('denoiser' in identity or identity.get('source_sha256')!=report['source_sha256']
                    or identity.get('lambda')!=parameters['lambda'] or identity.get('device')!=parameters.get('device')
                    or identity.get('callable')!=parameters['l0_implementation']
                    or identity.get('code')!=expected_code):continue
            matched_identity=True
            arrays={name:saved[name] for name in saved.files if name!='metadata'}
            if meta['payload_hash']!=_payload_hash(meta['tree'],arrays):raise ValueError('L0 review cache hash mismatch')
            value=_unpack(meta['tree'],arrays)
        if (not isinstance(value,np.ndarray) or value.dtype!=np.float32 or value.shape!=shape
                or not np.isfinite(value).all() or np.any((value<0)|(value>1))):
            raise ValueError('Invalid float L0 review candidate')
        if replay is None:return value
        if selected is not None and np.array_equal(value,selected):continue
        from .filters import blend,gradient_preserving_candidate
        restored=gradient_preserving_candidate(replay['original'],value,replay['effective_sigma'],
            restoration_gain=parameters.get('restoration_gain',1.))
        if not np.array_equal(blend(replay['original'],restored,replay['weight']),replay['saved_result']):continue
        if selected is not None:
            raise ValueError('Ambiguous float L0 candidates reproduce the saved result')
        selected=value
    if selected is not None:return selected
    if replay is not None and matched_identity:
        raise ValueError('No float L0 candidate reproduces the saved result; no PNG reconstruction is allowed')
    raise ValueError('Exact float L0 cache unavailable for review; rerun processing to generate a report with '
                     'candidate_cache_identity; replay cannot identify a historical candidate and no PNG reconstruction is allowed')


def _review_l0(cache_root, report, shape, *, replay=None):
    """Compatibility entry point for old callers; the report selects the engine."""
    return _review_candidate(cache_root,report,shape,replay=replay)


def render_denoise_review(review_manifest,predictions_root):
    """Fixed-mask randomized comparisons, with no neural inference or approval."""
    from .filters import blend,gradient_preserving_candidate
    from .pipeline import _read_source
    path=Path(review_manifest).resolve();document=json.loads(path.read_text(encoding='utf-8'))
    if any(c.get('approved') for c in document['cases']):
        raise ValueError('Do not modify a review bundle with approved references')
    _validate_review_document(document)
    output=path.parent/'denoise_review.json';key_path=path.parent/'blind_key.json'
    if output.exists():raise FileExistsError('Denoising review already exists')
    targets=[key_path]
    for case in document['cases']:
        folder=_case_folder(path.parent,case['id'])
        targets += [folder/f'denoise_roi_{i:02d}.png' for i,roi in enumerate(case['rois'])
                    if roi.get('name') in ['skin','face','hair','outfit']]
    # Preflight every write against every consumed reference/prediction/cache,
    # before a single comparison or key can overwrite an aliased input.
    inputs=safe_evaluation_output(output,path,prediction_roots=[predictions_root],additional_outputs=targets)
    cases=[];keys={}
    for case in document['cases']:
        folder=Path(predictions_root)/case['id'];entry={'id':case['id'],'comparisons':[],'approved':False}
        report_path=folder/'report.json'
        if not report_path.exists():
            entry['unavailable']='prediction missing';cases.append(entry);continue
        report=json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('status')!='complete' or report.get('source_sha256')!=case['source_sha256']:
            raise ValueError('Incomplete or wrong-source denoise review prediction')
        rgb,alpha,icc=_read_source(case['source'])
        if _sha(case['source'])!=case['source_sha256']:raise ValueError('Review source changed')
        if _sha(folder/'weight.npy')!=report['output_hashes']['weight.npy']:
            raise ValueError('Review weight changed')
        weight=np.load(folder/'weight.npy',allow_pickle=False)
        if _sha(folder/'result.png')!=report['output_hashes']['result.png']:
            raise ValueError('Review final result changed')
        with Image.open(folder/'result.png') as final:
            saved_result=np.array(final.convert('RGB'))
        pure=_review_candidate(folder.parent/'.cache',report,rgb.shape,
            replay={'original':rgb,'weight':weight,'effective_sigma':report['effective_sigma'],'saved_result':saved_result})
        parameters=report['parameters']
        denoiser=parameters.get('denoiser','l0')
        raw_candidate='pure_l0' if denoiser=='l0' else denoiser.replace('-','_')
        # Historical reports used full restoration and line protection.
        restoration_gain=parameters.get('restoration_gain',1.)
        entry.update(denoiser=denoiser,raw_candidate=raw_candidate,restoration_gain=restoration_gain,
                     structure_protection=parameters.get('structure_protection',1.))
        restored=gradient_preserving_candidate(rgb,pure,report['effective_sigma'],restoration_gain=restoration_gain)
        values={'original':rgb,raw_candidate:blend(rgb,pure,weight),
                'restored':blend(rgb,restored,weight),'half_strength':blend(rgb,restored,weight*.5)}
        if not np.array_equal(values['restored'],saved_result):
            raise ValueError('Float-candidate review does not reproduce the saved result')
        order=list(values);random.Random(int(case['source_sha256'][:16],16)).shuffle(order)
        keys[case['id']]={chr(65+i):name for i,name in enumerate(order)}
        for index,roi in enumerate(case['rois']):
            if roi.get('name') not in ['skin','face','hair','outfit']:continue
            x0,y0,x1,y1=roi['xyxy'];side=min(256,x1-x0,y1-y0)
            cx=(x0+x1)//2;cy=(y0+y1)//2;box=(cx-side//2,cy-side//2,cx-side//2+side,cy-side//2+side)
            target=path.parent/case['id']/f'denoise_roi_{index:02d}.png'
            _assert_inputs_safe([target],inputs)
            board=Image.new('RGB',(4*(side+12),side+55),'#edf1f5');draw=ImageDraw.Draw(board)
            for col,name in enumerate(order):
                draw.text((col*(side+12)+8,8),chr(65+col),font=_font(20),fill='#263747')
                view=values[name]
                if alpha is not None:
                    a=alpha[...,None].astype(np.float32)/255
                    view=np.rint(view*a+255*(1-a)).astype(np.uint8)
                board.paste(Image.fromarray(view).crop(box),(col*(side+12),43))
            board.save(target,**({'icc_profile':icc} if icc else {}))
            entry['comparisons'].append({'roi':roi['name'],'xyxy':list(box),'image':str(target.relative_to(path.parent)),
                'preferred':'pending','artifact_reduction':'pending','detail_damage':'pending'})
        entry['float_candidate_reproduced_saved_result']=True;cases.append(entry)
    methods=sorted({c['denoiser'] for c in cases if 'denoiser' in c})
    candidate_names=[DENOISER_NAMES[method] for method in methods]
    result={'schema_version':1,'cases':cases,'method':'same original-grid float candidate ('+', '.join(candidate_names)+
            ') and same final weight; original/raw candidate/Gaussian-restored/half strength randomized',
            'original_pixel_crops_resampled':False,'approval_required':True,'quality_accepted':False}
    key_path.write_text(json.dumps(keys,ensure_ascii=False,indent=2),encoding='utf-8')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
