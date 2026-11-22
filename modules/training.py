"""Thí nghiệm bước 5: CV theo nhóm trên raw train, không nhận dữ liệu test."""
from pathlib import Path
import hashlib, json, platform, time, warnings
import numpy as np
import pandas as pd
import joblib, sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .preprocessing import build_preprocessor, normalize_config, encode_target, FEATURE_COLUMNS, _audit_fit
from .splitting import frame_fingerprint, make_group_ids

SEED=42
METRICS=['f1','precision','recall','accuracy','macro_f1','roc_auc']
CV_CONFIG={'n_splits':3,'shuffle':True,'random_state':SEED}

def candidate_configs():
    rows=[{'id':'dummy_majority','family':'Dummy','scope':'reference','preprocessing':{},'model':{}}]
    for family,prefix in [('LogisticRegression','lr'),('LinearSVC','svm')]:
        for c,tag in [(0.1,'01'),(1.0,'1'),(10.0,'10')]:
            rows.append({'id':f'{prefix}_c{tag}','family':family,'scope':'model','preprocessing':{},'model':{'C':c}})
    for tag,depth,leaf in [('full_leaf1',None,1),('full_leaf3',None,3),('depth20_leaf3',20,3)]:
        rows.append({'id':'rf_'+tag,'family':'RandomForest','scope':'model','preprocessing':{},
                     'model':{'n_estimators':150,'max_depth':depth,'min_samples_leaf':leaf}})
    for tag,layers,alpha in [('64',[64],0.0001),('64_32',[64,32],0.001)]:
        rows.append({'id':'mlp_'+tag,'family':'MLP','scope':'model','preprocessing':{},
                     'model':{'hidden_layer_sizes':layers,'alpha':alpha}})
    for tag,scope,config in [
        ('missing_constant','preprocessing',{'categorical_imputation':'constant'}),
        ('ordinal','preprocessing',{'encoding':'ordinal'}),
        ('minmax01','preprocessing',{'scaling':'minmax','feature_range':[0,1]}),
        ('minmax11','preprocessing',{'scaling':'minmax','feature_range':[-1,1]}),
        ('pca90','pca',{'pca_variance':0.90}),
        ('pca95','pca',{'pca_variance':0.95}),
    ]:
        rows.append({'id':'lr_'+tag,'family':'LogisticRegression','scope':scope,'preprocessing':config,'model':{'C':1.0}})
    # Bổ sung sau pilot 18 cấu hình: chỉ dựa vào cảnh báo tối ưu trong train/CV,
    # chưa xem validation hay test; giữ nguyên và báo cáo cả hai MLP ban đầu.
    for tag,layers,alpha in [('32_reg',[32],0.01),('64_reg',[64],0.1)]:
        rows.append({'id':'mlp_'+tag,'family':'MLP','scope':'model','preprocessing':{},
            'model':{'hidden_layer_sizes':layers,'alpha':alpha,'tol':0.001,'n_iter_no_change':10}})
    return rows

def build_model(candidate):
    c=candidate['model']; family=candidate['family']
    if family=='Dummy':return DummyClassifier(strategy='most_frequent')
    if family=='LogisticRegression':
        return LogisticRegression(C=c['C'],solver='lbfgs',max_iter=2000,tol=1e-5,random_state=SEED)
    if family=='LinearSVC':
        return LinearSVC(C=c['C'],dual=False,max_iter=10000,tol=1e-5,random_state=SEED)
    if family=='RandomForest':
        return RandomForestClassifier(**c,max_features='sqrt',bootstrap=True,n_jobs=1,random_state=SEED)
    if family=='MLP':
        return MLPClassifier(hidden_layer_sizes=tuple(c['hidden_layer_sizes']),alpha=c['alpha'],
            activation='relu',solver='adam',learning_rate_init=0.001,batch_size=256,
            max_iter=300,tol=c.get('tol',1e-4),n_iter_no_change=c.get('n_iter_no_change',20),early_stopping=False,random_state=SEED)
    raise ValueError('Mô hình không được hỗ trợ: '+str(family))

def build_training_pipeline(candidate):
    return Pipeline([('preprocessing',build_preprocessor(candidate['preprocessing'])),('model',build_model(candidate))])

def score_vector(pipeline,X):
    if hasattr(pipeline,'decision_function'):
        return np.asarray(pipeline.decision_function(X),dtype=float)
    classes=pipeline.named_steps['model'].classes_
    column=int(np.flatnonzero(classes==1)[0])
    return np.asarray(pipeline.predict_proba(X)[:,column],dtype=float)

def compute_metrics(y,pred,score):
    return {'f1':float(f1_score(y,pred,pos_label=1,zero_division=0)),
        'precision':float(precision_score(y,pred,pos_label=1,zero_division=0)),
        'recall':float(recall_score(y,pred,pos_label=1,zero_division=0)),
        'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
        'roc_auc':float(roc_auc_score(y,score))}

def _json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()

def frozen_cv(train,metadata,path,config=None):
    config=dict(CV_CONFIG if config is None else config)
    if train.columns.tolist()!=FEATURE_COLUMNS+['income']:
        raise ValueError('Train phải giữ đúng schema 14 thuộc tính và income.')
    groups=metadata['group_id'].astype(str).to_numpy()
    if len(groups)!=len(train) or metadata['source_id'].duplicated().any():
        raise ValueError('Metadata train không hợp lệ.')
    np.testing.assert_array_equal(groups,np.asarray(make_group_ids(train)))
    expected={'data_sha256':frame_fingerprint(train),'source_ids_sha256':_digest(metadata['source_id'].astype(str).tolist()),
        'groups_sha256':_digest(groups.tolist()),'config':config}
    path=Path(path)
    if path.exists():
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest['identity']!=expected:raise ValueError('CV manifest khác dữ liệu/metadata/cấu hình; không tự chia lại.')
    else:
        splitter=StratifiedGroupKFold(**config)
        manifest={'identity':expected,'generator_sklearn':sklearn.__version__,
            'validation_indices':[va.tolist() for _,va in splitter.split(train,encode_target(train),groups)]}
        _json_write(path,manifest)
    folds=[];seen=np.zeros(len(train),dtype=int)
    if len(manifest['validation_indices'])!=config['n_splits']:raise ValueError('Sai số fold.')
    all_indices=np.arange(len(train))
    for indices in manifest['validation_indices']:
        va=np.asarray(indices,dtype=np.int64)
        if len(va)==0 or len(set(va))!=len(va) or (va<0).any() or (va>=len(train)).any():
            raise ValueError('Chỉ số fold không hợp lệ.')
        tr=np.setdiff1d(all_indices,va)
        if set(groups[tr])&set(groups[va]):raise ValueError('Nhóm trùng xuất hiện hai phía fold.')
        if len(np.unique(encode_target(train.iloc[va])))!=2:raise ValueError('Fold cần đủ hai lớp.')
        seen[va]+=1;folds.append((tr,va))
    if not np.all(seen==1):raise ValueError('Các fold không phủ mỗi dòng đúng một lần.')
    return folds,manifest

def _fit(pipeline,X,y,candidate):
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipeline.fit(X,y)
    elapsed=time.perf_counter()-start
    convergence=[str(w.message) for w in caught if issubclass(w.category,ConvergenceWarning)]
    notices=[{'type':w.category.__name__,'message':str(w.message)} for w in caught]
    prep=pipeline.named_steps['preprocessing']
    audit=_audit_fit(prep,X,normalize_config(candidate['preprocessing']))
    if prep.named_steps['input'].fit_rows_!=len(X):raise AssertionError('Fit sai số dòng train.')
    return elapsed,convergence,notices,audit

def run_cv(train,metadata,project_dir='.',candidates=None,cv_config=None,reuse_completed=True):
    root=Path(project_dir);out=root/'results/step_05';out.mkdir(parents=True,exist_ok=True)
    candidates=candidate_configs() if candidates is None else candidates
    if len({c['id'] for c in candidates})!=len(candidates):raise ValueError('Candidate ID bị lặp.')
    folds,cv_manifest=frozen_cv(train,metadata,out/'cv_indices.json',cv_config)
    _json_write(root/'configs/experiments.json',{'seed':SEED,'cv':cv_manifest['identity']['config'],
        'selection':'maximum mean CV F1 of >50K; tie: smaller std then candidate ID',
        'threshold':'native predict: probability 0.5, LinearSVC decision 0',
        'class_weight':None,'sampling':'none','candidates':candidates})
    cache_identity={'train':frame_fingerprint(train),'cv':_digest(cv_manifest),'candidates':_digest(candidates),
        'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'python':platform.python_version(),
        'source_hashes':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                         for name in ['training.py','preprocessing.py','splitting.py','data_loader.py']}}
    cache_path=out/'cv_complete.json'
    if reuse_completed and cache_path.exists():
        saved=json.loads(cache_path.read_text(encoding='utf-8'))
        if saved['identity']!=cache_identity:raise ValueError('Cache CV khác dữ liệu/code/phiên bản; tắt reuse_completed để chạy lại.')
        for rel,sha in saved['files'].items():
            if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=sha:raise ValueError('Cache CV đã thay đổi: '+rel)
        print('Tái sử dụng CV đã hoàn tất, khớp dữ liệu, fold, cấu hình, mã nguồn, phiên bản và SHA-256.',flush=True)
        return pd.read_csv(out/'cv_summary.csv'),json.loads((out/'selection_cv.json').read_text(encoding='utf-8'))
    fold_summary=[]
    for i,(tr,va) in enumerate(folds):
        fold_summary.append({'fold':i,'fit_rows':len(tr),'assessment_rows':len(va),
            'fit_positive_percent':float(encode_target(train.iloc[tr]).mean()*100),
            'assessment_positive_percent':float(encode_target(train.iloc[va]).mean()*100),
            'group_overlap':0})
    pd.DataFrame(fold_summary).to_csv(out/'cv_fold_summary.csv',index=False)
    X=train[FEATURE_COLUMNS].copy();y=encode_target(train);snapshot=train.copy(deep=True)
    scores=[];audits=[]
    with threadpool_limits(limits=2):
        for ci,c in enumerate(candidates,1):
            print(f"[{ci}/{len(candidates)}] {c['id']}",flush=True)
            for fold,(tr,va) in enumerate(folds):
                pipeline=build_training_pipeline(c)
                elapsed,conv,notices,audit=_fit(pipeline,X.iloc[tr],y[tr],c)
                before=joblib.hash(pipeline)
                start=time.perf_counter();pred=pipeline.predict(X.iloc[va]);score=score_vector(pipeline,X.iloc[va])
                predict_seconds=time.perf_counter()-start
                if joblib.hash(pipeline)!=before:raise AssertionError('Predict thay đổi trạng thái fitted.')
                if not np.isfinite(score).all():raise AssertionError('Điểm dự đoán không hữu hạn.')
                row={'candidate_id':c['id'],'family':c['family'],'scope':c['scope'],'fold':fold,
                    **compute_metrics(y[va],pred,score),'fit_seconds':elapsed,'predict_seconds':predict_seconds,
                    'feature_count':len(pipeline.named_steps['preprocessing'].get_feature_names_out()),
                    'fit_rows':len(tr),'assessment_rows':len(va),'convergence_warning_count':len(conv)}
                scores.append(row)
                audits.append({'candidate_id':c['id'],'fold':fold,'fit_audit':audit,'warnings':notices,
                    'model_n_iter':np.asarray(getattr(pipeline.named_steps['model'],'n_iter_',[])).tolist(),
                    'prediction_state_unchanged':True})
                pd.DataFrame(scores).to_csv(out/'cv_folds.csv',index=False)
                _json_write(out/'fit_audits.json',audits)
                print(f"  fold {fold+1}: F1={row['f1']:.4f}, fit={elapsed:.1f}s, convergence warnings={len(conv)}",flush=True)
    pd.testing.assert_frame_equal(train,snapshot)
    summary=[];df=pd.DataFrame(scores)
    for c in candidates:
        group=df[df['candidate_id']==c['id']]
        row={'candidate_id':c['id'],'family':c['family'],'scope':c['scope'],'folds':len(group)}
        for metric in METRICS:
            row[metric+'_mean']=float(group[metric].mean());row[metric+'_std']=float(group[metric].std(ddof=1))
        row.update({'fit_seconds_mean':float(group['fit_seconds'].mean()),
            'convergence_warning_count':int(group['convergence_warning_count'].sum()),
            'preprocessing_json':json.dumps(normalize_config(c['preprocessing']),sort_keys=True),
            'model_json':json.dumps(c['model'],sort_keys=True)})
        summary.append(row)
    summary=pd.DataFrame(summary).sort_values(['f1_mean','f1_std','candidate_id'],ascending=[False,True,True]).reset_index(drop=True)
    summary.to_csv(out/'cv_summary.csv',index=False)
    # Convergence is reported, not hidden; do not lock an unconverged candidate.
    eligible=summary[(summary.family!='Dummy')&(summary.convergence_warning_count==0)]
    if eligible.empty:raise RuntimeError('Không có cấu hình hội tụ để chọn; cần khảo sát lại và ghi rõ thay đổi.')
    selection={'cv_manifest_sha256':_digest(cv_manifest),'candidates_sha256':_digest(candidates),
        'training_data_sha256':frame_fingerprint(train),'selection_metric':'f1_mean',
        'tie_breakers':['f1_std ascending','candidate_id ascending'],
        'winner_id':str(eligible.iloc[0].candidate_id),
        'family_winners':{f:str(g.iloc[0].candidate_id) for f,g in eligible.groupby('family',sort=True)},
        'convergence_policy':'Candidates with any convergence warning are reported but excluded from locking.',
        'validation_used_for_selection':False,'test_accessed_by_experiment':False,
        'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__,
                    'pandas':pd.__version__,'joblib':joblib.__version__},
        'candidate_count':len(candidates),'fold_count':len(folds),'cv_fit_count':len(scores),
        'note':'CV mean selected from this search is not an unbiased final performance estimate.'}
    _json_write(out/'selection_cv.json',selection)
    paths=[out/name for name in ['cv_folds.csv','cv_summary.csv','cv_fold_summary.csv','fit_audits.json','selection_cv.json']]
    _json_write(cache_path,{'identity':cache_identity,'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    return summary,selection

def refit_and_validate(train,train_meta,validation,validation_meta,project_dir='.'):
    """Khóa lựa chọn bằng CV trước; validation chỉ đối chiếu, không chọn lại."""
    root=Path(project_dir);out=root/'results/step_05';art=root/'artifacts/models';art.mkdir(parents=True,exist_ok=True)
    selection=json.loads((out/'selection_cv.json').read_text(encoding='utf-8'))
    configs=json.loads((root/'configs/experiments.json').read_text(encoding='utf-8'))
    candidates={c['id']:c for c in configs['candidates']}
    if selection['training_data_sha256']!=frame_fingerprint(train) or selection['candidates_sha256']!=_digest(configs['candidates']):
        raise ValueError('Dữ liệu/cấu hình khác giai đoạn CV đã khóa.')
    if set(train_meta['group_id'])&set(validation_meta['group_id']):raise ValueError('Train/validation trùng nhóm.')
    if set(train_meta['source_id'])&set(validation_meta['source_id']):raise ValueError('Train/validation trùng ID.')
    snapshots=[train.copy(deep=True),validation.copy(deep=True)]
    X=train[FEATURE_COLUMNS]; y=encode_target(train);Xv=validation[FEATURE_COLUMNS];yv=encode_target(validation)
    ids=list(selection['family_winners'].values())+['dummy_majority']
    rows=[];fitted={};predictions={};logs=[]
    with threadpool_limits(limits=2):
        for ident in ids:
            c=candidates[ident];p=build_training_pipeline(c)
            print('Refit + validation:',ident,flush=True)
            seconds,conv,notices,audit=_fit(p,X,y,c)
            if conv:raise RuntimeError('Refit chưa hội tụ: '+ident+'; không tự đổi siêu tham số.')
            state=joblib.hash(p);pred=p.predict(Xv);score=score_vector(p,Xv)
            if joblib.hash(p)!=state:raise AssertionError('Validation thay đổi pipeline.')
            path=art/f'{ident}.joblib';joblib.dump(p,path,compress=3)
            fitted[ident]=path.relative_to(root).as_posix()
            rows.append({'candidate_id':ident,'family':c['family'],**compute_metrics(yv,pred,score),
                         'fit_seconds':seconds,'fit_rows':len(train),'validation_rows':len(validation),
                         'selected_by_cv':ident==selection['winner_id']})
            predictions[ident]=pd.DataFrame({'source_id':validation_meta['source_id'].to_numpy(),
                'y_true':yv,'y_pred':pred,'score':score})
            predictions[ident].to_csv(out/f'validation_predictions_{ident}.csv',index=False)
            logs.append({'candidate_id':ident,'fit_audit':audit,'warnings':notices,
                         'model_n_iter':np.asarray(getattr(p.named_steps['model'],'n_iter_',[])).tolist()})
            if c['family']=='MLP':
                pd.DataFrame({'epoch':np.arange(1,len(p.named_steps['model'].loss_curve_)+1),
                    'training_loss':p.named_steps['model'].loss_curve_}).to_csv(out/'mlp_training_loss.csv',index=False)
    for frame,snapshot in zip([train,validation],snapshots):pd.testing.assert_frame_equal(frame,snapshot)
    table=pd.DataFrame(rows);table.to_csv(out/'validation_summary.csv',index=False)
    _json_write(out/'refit_audits.json',logs)
    selection.update({'fitted_models':fitted,'selected_model_path':fitted[selection['winner_id']],
        'validation_data_sha256':frame_fingerprint(validation),'refit_rows':len(train),
        'validation_rows':len(validation),'refit_fit_count':len(ids),
        'threshold_tuned':False,'refit_on_train_plus_validation':False,
        'test_model_metrics_computed':False})
    _json_write(out/'selected_model.json',selection)
    return table,selection

def plot_experiments(result_dir='results/step_05'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(result_dir);summary=pd.read_csv(root/'cv_summary.csv')
    def plot(table,labels,stem,title):
        fig,ax=plt.subplots(figsize=(8,4.7))
        bars=ax.bar(np.arange(len(table)),table.f1_mean,yerr=table.f1_std,capsize=4,color='#2563EB',alpha=.85)
        ax.set_xticks(np.arange(len(table)),labels,rotation=15,ha='right')
        ax.set(title=title,ylabel='F1 lớp >50K — mean ± SD (3 fold)',ylim=(0,1))
        for bar,value in zip(bars,table.f1_mean):ax.text(bar.get_x()+bar.get_width()/2,value+.025,f'{value:.3f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(root/f'{stem}.{ext}',dpi=160,bbox_inches='tight')
        plt.close(fig)
    base=summary[summary.scope.isin(['model','reference'])].sort_values(['f1_mean','f1_std','candidate_id'],ascending=[False,True,True])
    base=base.groupby('family',sort=True).head(1).sort_values('family')
    plot(base,base.family.tolist(),'cv_models','So sánh mô hình trên cùng tiền xử lý cơ sở')
    ids=['lr_c1','lr_missing_constant','lr_ordinal','lr_minmax01','lr_minmax11']
    t=summary.set_index('candidate_id').loc[ids].reset_index()
    plot(t,['Cơ sở','Nhóm thiếu','Ordinal','MinMax [0,1]','MinMax [-1,1]'],'cv_preprocessing','Thay từng lựa chọn tiền xử lý — Logistic Regression C=1')
    t=summary.set_index('candidate_id').loc[['lr_c1','lr_pca90','lr_pca95']].reset_index()
    plot(t,['Không PCA','PCA 90%','PCA 95%'],'cv_pca','Ảnh hưởng của PCA — Logistic Regression C=1')
    return ['cv_models','cv_preprocessing','cv_pca']
