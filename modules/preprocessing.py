"""Pipeline tiền xử lý cấu hình được; fit chỉ trên train, không chứa bộ phân loại."""
from pathlib import Path
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from .data_loader import COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

FEATURE_COLUMNS=COLUMNS[:-1]
TARGET_MAPPING={'<=50K':0,'>50K':1}
DEFAULT_CONFIG={
    'numeric_imputation':'median',
    'categorical_imputation':'most_frequent',
    'missing_token':'__MISSING__',
    'encoding':'onehot',
    'scaling':'standard',
    'feature_range':[0.0,1.0],
    'pca_variance':None,
}
EXPORT_CONFIGS={
    'baseline':{},
    'pca_90':{'pca_variance':0.90},
    'pca_95':{'pca_variance':0.95},
}


class AdultInputAdapter(TransformerMixin,BaseEstimator):
    """Kiểm tra schema và chuyển pd.NA thành np.nan cho scikit-learn."""
    def fit(self,X,y=None):
        self.feature_names_in_=np.array(FEATURE_COLUMNS,dtype=object)
        self.n_features_in_=len(FEATURE_COLUMNS)
        self.fit_rows_=len(X)
        self.transform(X)
        return self

    def transform(self,X):
        check_is_fitted(self,'feature_names_in_')
        if not isinstance(X,pd.DataFrame) or X.columns.has_duplicates:
            raise ValueError('Đầu vào phải là DataFrame với tên cột duy nhất.')
        if set(X.columns)!=set(FEATURE_COLUMNS):
            raise ValueError('Pipeline chỉ nhận đúng 14 thuộc tính, không nhận income hoặc cột metadata.')
        out=X.loc[:,FEATURE_COLUMNS].copy()
        for column in NUMERIC_COLUMNS:
            out[column]=pd.to_numeric(out[column],errors='raise').astype('float64')
        for column in CATEGORICAL_COLUMNS:
            values=out[column].astype('object')
            out[column]=values.where(pd.notna(values),np.nan)
        return out

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'feature_names_in_')
        return self.feature_names_in_.copy()


def normalize_config(config=None):
    overrides=config or {}
    unknown=set(overrides)-set(DEFAULT_CONFIG)
    if unknown:raise ValueError(f'Tham số không được hỗ trợ: {sorted(unknown)}')
    result={**DEFAULT_CONFIG,**overrides}
    if result['numeric_imputation'] not in ['median','mean']:
        raise ValueError('numeric_imputation phải là median hoặc mean.')
    if result['categorical_imputation'] not in ['most_frequent','constant']:
        raise ValueError('categorical_imputation phải là most_frequent hoặc constant.')
    if result['encoding'] not in ['onehot','ordinal']:
        raise ValueError('encoding phải là onehot hoặc ordinal.')
    if result['scaling'] not in ['standard','minmax','none']:
        raise ValueError('scaling phải là standard, minmax hoặc none.')
    bounds=result['feature_range']
    if len(bounds)!=2 or not all(np.isfinite(float(x)) for x in bounds) or not float(bounds[0])<float(bounds[1]):
        raise ValueError('feature_range phải gồm hai số hữu hạn tăng dần.')
    result['feature_range']=[float(x) for x in bounds]
    if not isinstance(result['missing_token'],str) or not result['missing_token']:
        raise ValueError('missing_token phải là chuỗi không rỗng.')
    pca=result['pca_variance']
    if pca is not None:
        if isinstance(pca,bool) or not isinstance(pca,(float,np.floating)) or not 0<float(pca)<1:
            raise ValueError('pca_variance phải là None hoặc số thực trong khoảng (0,1).')
        result['pca_variance']=float(pca)
    return result


def build_preprocessor(config=None):
    """Trả về estimator CHƯA FIT, có thể clone và đặt trong CV pipeline."""
    c=normalize_config(config)
    scaler={
        'standard':lambda:StandardScaler(),
        'minmax':lambda:MinMaxScaler(feature_range=tuple(c['feature_range']),clip=False),
        'none':lambda:'passthrough',
    }[c['scaling']]()
    numeric=Pipeline([
        ('imputer',SimpleImputer(strategy=c['numeric_imputation'],keep_empty_features=True)),
        ('scaler',scaler),
    ])
    cat_imputer=SimpleImputer(
        strategy=c['categorical_imputation'],fill_value=c['missing_token'],
        keep_empty_features=True,
    )
    encoder=(OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32)
        if c['encoding']=='onehot' else
        OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,dtype=np.float32))
    categorical=Pipeline([('imputer',cat_imputer),('encoder',encoder)])
    columns=ColumnTransformer([
        ('numeric',numeric,NUMERIC_COLUMNS),
        ('categorical',categorical,CATEGORICAL_COLUMNS),
    ],remainder='drop',sparse_threshold=0,verbose_feature_names_out=True)
    pca='passthrough' if c['pca_variance'] is None else PCA(n_components=c['pca_variance'],svd_solver='full',whiten=False)
    return Pipeline([
        ('input',AdultInputAdapter()),('columns',columns),('pca',pca),
        ('float32',FunctionTransformer(np.asarray,kw_args={'dtype':np.float32},feature_names_out='one-to-one')),
    ])


def encode_target(frame):
    if frame['income'].isna().any() or not frame['income'].isin(TARGET_MAPPING).all():
        raise ValueError('Nhãn ngoài hai giá trị Adult hoặc bị thiếu.')
    return frame['income'].map(TARGET_MAPPING).to_numpy(dtype=np.int8)


def _sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _audit_fit(pipeline,X_train,config):
    """Đối chiếu các thống kê fitted với dữ liệu train, không dựa vào test."""
    adapted=pipeline.named_steps['input'].transform(X_train)
    col=pipeline.named_steps['columns']
    numeric=col.named_transformers_['numeric']
    imputed=numeric.named_steps['imputer'].transform(adapted[NUMERIC_COLUMNS])
    expected=(adapted[NUMERIC_COLUMNS].median() if config['numeric_imputation']=='median' else adapted[NUMERIC_COLUMNS].mean()).to_numpy()
    expected=np.where(np.isnan(expected),0,expected)
    np.testing.assert_allclose(numeric.named_steps['imputer'].statistics_,expected,rtol=1e-12,atol=1e-12)
    scaler=numeric.named_steps['scaler']
    if isinstance(scaler,StandardScaler):
        np.testing.assert_allclose(scaler.mean_,imputed.mean(axis=0),rtol=1e-12,atol=1e-12)
    elif isinstance(scaler,MinMaxScaler):
        np.testing.assert_allclose(scaler.data_min_,imputed.min(axis=0))
        np.testing.assert_allclose(scaler.data_max_,imputed.max(axis=0))
    cat=col.named_transformers_['categorical']
    filled=cat.named_steps['imputer'].transform(adapted[CATEGORICAL_COLUMNS])
    for j,categories in enumerate(cat.named_steps['encoder'].categories_):
        if set(categories)!=set(filled[:,j]):raise AssertionError('Miền encoder khác miền sau điền thiếu trên train.')
    pca=pipeline.named_steps['pca']
    if isinstance(pca,PCA):
        assert pca.n_samples_==len(X_train)
        before_pca=col.transform(adapted)
        np.testing.assert_allclose(pca.mean_,before_pca.mean(axis=0),rtol=1e-12,atol=1e-12)
    return {
        'fit_rows':len(X_train),
        'numeric_imputer_matches_train':True,
        'numeric_scaler_matches_train':True,
        'encoder_categories_match_train':True,
        'pca_samples_match_train':True if isinstance(pca,PCA) else None,
    }


def export_features(frames,metadata,project_dir='.',configs=None):
    """Fit từng cấu hình trên train; xuất X/y/source_row, pipeline và manifest."""
    if set(frames)!=set(['train','validation','test']) or set(metadata)!=set(frames):
        raise ValueError('Cần đủ ba tập và metadata tương ứng.')
    root=Path(project_dir); features=root/'features';results=root/'results/step_04'
    artifacts=root/'artifacts/preprocessors'
    for directory in [features,results,artifacts]:directory.mkdir(parents=True,exist_ok=True)
    snapshots={name:frame.copy(deep=True) for name,frame in frames.items()}
    X={name:frame[FEATURE_COLUMNS].copy() for name,frame in frames.items()}
    common_files={}
    for name in ['train','validation','test']:
        if len(frames[name])!=len(metadata[name]):raise ValueError('Metadata lệch số mẫu.')
        for prefix,array in [('y',encode_target(frames[name])),
            ('source_row',metadata[name]['source_row'].to_numpy(dtype=np.int64))]:
            path=features/f'{prefix}_{name}.npy';np.save(path,array,allow_pickle=False)
            common_files[path.relative_to(root).as_posix()]={'shape':list(array.shape),'dtype':str(array.dtype),'sha256':_sha_file(path)}
    candidates=configs or EXPORT_CONFIGS
    all_manifests={};shape_rows=[]
    for name,overrides in candidates.items():
        if not name.replace('_','').isalnum():raise ValueError('Tên cấu hình chỉ dùng chữ, số và dấu gạch dưới.')
        config=normalize_config(overrides)
        pipeline=build_preprocessor(config)
        with threadpool_limits(limits=2):
            pipeline.fit(X['train'])
            fit_audit=_audit_fit(pipeline,X['train'],config)
            before_transform=joblib.hash(pipeline)
            transformed={split:pipeline.transform(values) for split,values in X.items()}
            after_transform=joblib.hash(pipeline)
        if before_transform!=after_transform:raise AssertionError('Transform làm thay đổi trạng thái fitted.')
        names=list(map(str,pipeline.get_feature_names_out()))
        directory=features/name;directory.mkdir(parents=True,exist_ok=True)
        files={}
        for split,array in transformed.items():
            if array.dtype!=np.float32 or not np.isfinite(array).all():
                raise AssertionError('Đặc trưng phải float32, hữu hạn và không NaN.')
            if array.shape!=(len(frames[split]),len(names)):
                raise AssertionError('Đặc trưng bị lệch dòng hoặc số cột.')
            path=directory/f'X_{split}.npy';np.save(path,array,allow_pickle=False)
            files[path.relative_to(root).as_posix()]={'shape':list(array.shape),'dtype':str(array.dtype),'sha256':_sha_file(path)}
        (directory/'feature_names.json').write_text(json.dumps(names,ensure_ascii=False,indent=2),encoding='utf-8')
        fitted_path=artifacts/f'{name}.joblib';joblib.dump(pipeline,fitted_path,compress=3)
        pca=pipeline.named_steps['pca']
        retained=float(pca.explained_variance_ratio_.sum()) if isinstance(pca,PCA) else None
        entry={
            'config':config,'feature_count':len(names),'fit_split':'train',
            'fit_audit':fit_audit,'transform_did_not_change_fitted_state':True,
            'pca_retained_variance':retained,'arrays':files,
            'pipeline_path':fitted_path.relative_to(root).as_posix(),
            'feature_names_path':(directory/'feature_names.json').relative_to(root).as_posix(),
        }
        (directory/'manifest.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2),encoding='utf-8')
        all_manifests[name]=entry
        for split,array in transformed.items():
            shape_rows.append({'configuration':name,'split':split,'rows':array.shape[0],'features':array.shape[1],
                'dtype':str(array.dtype),'pca_retained_variance':retained,'all_values_finite':True})
        if isinstance(pca,PCA):
            pd.DataFrame({'component':np.arange(1,len(pca.explained_variance_ratio_)+1),
                'explained_variance_ratio':pca.explained_variance_ratio_,
                'cumulative_variance_ratio':np.cumsum(pca.explained_variance_ratio_)}).to_csv(results/f'{name}_variance.csv',index=False,encoding='utf-8')
        del transformed
    for name in frames:pd.testing.assert_frame_equal(frames[name],snapshots[name])
    shape_table=pd.DataFrame(shape_rows);shape_table.to_csv(results/'feature_shapes.csv',index=False,encoding='utf-8')
    versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__}
    train_ids=metadata['train']['source_id'].astype(str).tolist()
    manifest={
        'step':4,'fit_split':'train','fit_rows':len(frames['train']),
        'input_feature_columns':FEATURE_COLUMNS,'numeric_columns':NUMERIC_COLUMNS,
        'categorical_columns':CATEGORICAL_COLUMNS,'target_mapping':TARGET_MAPPING,
        'source_ids_file':'data/splits/{split}_metadata.csv',
        'training_source_ids_sha256':hashlib.sha256('\n'.join(train_ids).encode('utf-8')).hexdigest(),
        'common_arrays':common_files,'configurations':all_manifests,'versions':versions,
        'source_data_mutated':False,'classifier_trained':False,
        'cv_rule':'Use raw train data and fit a fresh complete pipeline inside each group CV fold; do not CV on these globally transformed X arrays.',
        'test_model_metrics_computed':False,
    }
    (features/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (results/'preprocessing_summary.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (results/'available_options.json').write_text(json.dumps({
        'numeric_imputation':['median','mean'],'categorical_imputation':['most_frequent','constant'],
        'encoding':['onehot','ordinal'],'scaling':['standard','minmax','none'],
        'feature_range':'[lower, upper], used for minmax','pca_variance':[None,0.90,0.95],
    },ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest


def plot_preprocessing(result_dir='results/step_04'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    root=Path(result_dir)
    fig,ax=plt.subplots(figsize=(10,9));ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
    def box(x,y,w,h,text,color='#E8F0FE'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=color,edgecolor='#2563EB'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,color='#333333'))
    box(2.0,8.7,6,0.8,'14 thuộc tính đầu vào, tách riêng nhãn income')
    box(.5,6.3,4,1.5,'6 biến số\nĐiền median / mean\nStandard / MinMax / không scaling')
    box(5.5,6.3,4,1.5,'8 biến phân loại\nĐiền mode / nhóm thiếu\nOne-hot / ordinal')
    box(2.5,4.7,5,.8,'Ghép hai nhánh đặc trưng')
    box(2.5,3.1,5,.8,'PCA tùy chọn: bỏ qua / 90% / 95%')
    box(2.5,1.5,5,.8,'X float32, không NaN, lưu .npy')
    arrow((3.2,8.7),(2.5,7.8));arrow((6.8,8.7),(7.5,7.8))
    arrow((2.5,6.3),(4,5.5));arrow((7.5,6.3),(6,5.5))
    arrow((5,4.7),(5,3.9));arrow((5,3.1),(5,2.3))
    ax.text(5,.5,'FIT: chỉ train    |    TRANSFORM: train, validation, test',ha='center',fontsize=11,fontweight='bold')
    fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(root/f'preprocessing_pipeline.{ext}',bbox_inches='tight',dpi=170)
    plt.close(fig)
    pca=pd.read_csv(root/'pca_95_variance.csv')
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(pca['component'],pca['cumulative_variance_ratio']*100,color='#2563EB',marker='o',markersize=3)
    for level,color in [(90,'#EA8A23'),(95,'#64748B')]:ax.axhline(level,color=color,linestyle='--',label=f'{level}%')
    ax.set(title='Phương sai tích lũy của các thành phần giữ trong PCA 95%',xlabel='Số thành phần',ylabel='Phương sai tích lũy (%)',ylim=(0,100))
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    for ext in ['pdf','png']:fig.savefig(root/f'pca_cumulative_variance.{ext}',bbox_inches='tight',dpi=170)
    plt.close(fig)


def update_latex(result_dir='results/step_04',report_root='reports/latex'):
    import shutil
    result=Path(result_dir);report=Path(report_root)
    summary=json.loads((result/'preprocessing_summary.json').read_text(encoding='utf-8'))
    entries=summary['configurations']
    baseline=entries['baseline'];p90=entries['pca_90'];p95=entries['pca_95']
    def write(rel,text):
        path=report/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    def number(x):return f'{int(x):,}'.replace(',','.')
    def percent(x):return f'{float(x)*100:.2f}'.replace('.',',')
    for stem,folder in [('preprocessing_pipeline','4_1'),('pca_cumulative_variance','4_4')]:
        target=report/'Images/chapter_4'/folder;target.mkdir(parents=True,exist_ok=True)
        for ext in ['pdf','png']:shutil.copyfile(result/f'{stem}.{ext}',target/f'{stem}.{ext}')

    write('chapters/chapter_2/2_1_bai_toan_phan_loai_nhi_phan.tex',r'''\section{Bài toán phân loại nhị phân}
\label{sec:2_1}
Trong học có giám sát, mỗi mẫu gồm vector thuộc tính $\mathbf{x}$ và nhãn $y$. Với bài toán phân loại nhị phân, $y\in\{0,1\}$. Mô hình học từ các mẫu train để dự đoán nhãn hoặc điểm cho một mẫu mới.

Trong đề tài này, nhãn \texttt{<=50K} được ánh xạ thành 0 và \texttt{>50K} thành 1. Nhãn được tách riêng trước khi fit mọi bộ biến đổi đầu vào. Mã hóa nhãn chỉ là quy ước biểu diễn; nó không bổ sung thông tin vào các đặc trưng. Việc chọn ngưỡng dự đoán và đánh giá chất lượng mô hình thuộc các bước thực nghiệm sau.
''')
    write('chapters/chapter_2/2_2_cac_ky_thuat_tien_xu_ly_du_lieu_bang.tex',r'''\section{Các kỹ thuật tiền xử lý dữ liệu bảng}
\label{sec:2_2}
Dữ liệu dạng bảng có thể gồm các biến số, biến phân loại và giá trị thiếu. Các bước biến đổi được tổ chức thành pipeline để cùng một quy tắc đã học trên train được áp dụng cho dữ liệu mới. Khi đánh giá bằng cross-validation, pipeline phải được fit lại bên trong từng fold~\cite{sklearn-pitfalls}.

\input{chapters/chapter_2/2_2_1_dien_gia_tri_thieu}
\input{chapters/chapter_2/2_2_2_ma_hoa_bien_phan_loai}
\input{chapters/chapter_2/2_2_3_chuan_hoa_bien_so}
''')
    write('chapters/chapter_2/2_2_1_dien_gia_tri_thieu.tex',r'''\subsection{Điền giá trị thiếu}
\label{subsec:2_2_1}
Imputation thay thế giá trị thiếu bằng giá trị được xác định theo một quy tắc. Với biến số, hai lựa chọn thông dụng là trung bình và trung vị của các giá trị quan sát trên train. Trung vị ít bị ảnh hưởng bởi các giá trị cực trị hơn trung bình. Với biến phân loại, có thể điền mức xuất hiện nhiều nhất hoặc tạo một mức riêng biểu diễn trạng thái thiếu~\cite{sklearn-imputer}.

Các giá trị điền được học khi gọi \texttt{fit}. Khi gọi \texttt{transform} trên validation hoặc test, phải tái sử dụng các giá trị đó; không tính lại thống kê điền từ từng tập đánh giá. Trong hiện thực, cột hoàn toàn thiếu ở một lần fit vẫn được giữ để tránh mất vị trí cột giữa các lần biến đổi.
''')
    write('chapters/chapter_2/2_2_2_ma_hoa_bien_phan_loai.tex',r'''\subsection{Mã hóa biến phân loại}
\label{subsec:2_2_2}
One-hot encoding biểu diễn mỗi mức phân loại bằng một cột chỉ báo. Cách này cho phép mô hình sử dụng các mức của một biến mà không áp đặt thứ tự số học giữa chúng. Tập mức và thứ tự các cột được xác định từ train. Với mức chưa thấy khi fit, cấu hình \texttt{handle\_unknown=ignore} tạo các giá trị 0 cho nhánh one-hot của biến đó~\cite{sklearn-onehot}.

Ordinal encoding gán một mã số cho mỗi mức. Khi mức phân loại không có thứ tự tự nhiên, khoảng cách giữa các mã không mang sẵn ý nghĩa về dữ liệu; vì vậy cần xem nó là một lựa chọn để thực nghiệm, không mặc định tốt hơn one-hot. Trong pipeline của đề tài, mã $-1$ được dành cho mức chưa thấy khi dùng ordinal. Cả hai encoder đều được học sau bước điền thiếu và chỉ từ train.
''')
    write('chapters/chapter_2/2_2_3_chuan_hoa_bien_so.tex',r'''\subsection{Chuẩn hóa biến số}
\label{subsec:2_2_3}
Standard scaling biến đổi một biến theo công thức
\begin{equation}
z=\frac{x-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}},
\end{equation}
trong đó trung bình và độ lệch chuẩn được ước lượng trên train sau khi điền thiếu. Trong scikit-learn, độ lệch chuẩn dùng phương sai với mẫu số $n$. Với cột hằng, thư viện xử lý hệ số tỷ lệ để tránh chia cho 0.

Min-max scaling ánh xạ dựa trên hai cực trị train vào khoảng cấu hình $[a,b]$:
\begin{equation}
z=a+\frac{x-x_{\min,\mathrm{train}}}{x_{\max,\mathrm{train}}-x_{\min,\mathrm{train}}}(b-a).
\end{equation}
Khi không bật clipping, một giá trị validation hoặc test nằm ngoài miền train có thể cho kết quả ngoài $[a,b]$~\cite{sklearn-preprocessing}. Đây không phải lỗi tính toán. Trong đề tài, scaling chỉ áp dụng cho nhánh sáu biến số; các cột one-hot giữ giá trị chỉ báo 0/1.
''')
    write('chapters/chapter_2/2_3_giam_chieu_bang_pca.tex',r'''\section{Giảm chiều bằng PCA}
\label{sec:2_3}
PCA là phương pháp giảm chiều không sử dụng nhãn, tìm các hướng trực giao giữ nhiều phương sai của dữ liệu. Với ma trận đặc trưng đã xử lý $X\in\mathbb{R}^{n\times d}$, đặt $X_c=X-\boldsymbol{\mu}$ và ma trận hiệp phương sai
\begin{equation}
S=\frac{1}{n-1}X_c^\top X_c.
\end{equation}
Các vector riêng của $S$ tương ứng với trị riêng giảm dần tạo nên những hướng thành phần chính. Chiếu lên $k$ hướng đầu cho biểu diễn $Z=X_cW_k$. Tỷ lệ phương sai được giữ lại là tổng các trị riêng được chọn chia cho tổng trị riêng.

Pipeline cho phép chọn mức phương sai 90\% hoặc 95\%. Với \texttt{svd\_solver=full}, scikit-learn chọn số thành phần sao cho tổng tỷ lệ phương sai vượt mức yêu cầu~\cite{sklearn-pca}. PCA phải được fit trên train và tái sử dụng tâm cùng các hướng chiếu cho validation/test. Giữ nhiều phương sai hơn không tự bảo đảm chất lượng phân loại tốt hơn; điều đó phải được kiểm tra ở các thí nghiệm mô hình.
''')
    write('chapters/chapter_4/4_1_so_do_tong_the_cua_pipeline.tex',r'''\section{Sơ đồ tổng thể của pipeline}
\label{sec:4_1}
Sau khi khóa cách chia tập, dữ liệu đầu vào được tách thành sáu biến số và tám biến phân loại. Hai nhánh xử lý được ghép lại, có thể đi qua PCA, rồi chuyển thành ma trận \texttt{float32}. Nhãn \texttt{income} và các cột metadata không tham gia fit đặc trưng.

\begin{figure}[htbp]
\centering
\includegraphics[width=.88\textwidth,height=.76\textheight,keepaspectratio]{Images/chapter_4/4_1/preprocessing_pipeline.pdf}
\caption{Cấu trúc pipeline tiền xử lý; tất cả bước học tham số chỉ fit trên train}
\label{fig:preprocessing-pipeline}
\end{figure}

Bước 4 chỉ chuẩn bị đặc trưng và chưa huấn luyện bộ phân loại. Ở bước thực nghiệm tiếp theo, bộ tiền xử lý chưa fit sẽ được ghép với từng mô hình trong một pipeline hoàn chỉnh, để fit đúng phạm vi dữ liệu của mỗi fold.
''')
    write('chapters/chapter_4/4_3_tien_xu_ly_du_lieu.tex',r'''\section{Tiền xử lý dữ liệu}
\label{sec:4_3}
Hiện thực sử dụng \texttt{Pipeline} và \texttt{ColumnTransformer}. Một lớp thích nghi kiểm tra đúng tên 14 thuộc tính, giữ thứ tự cột và chuyển các giá trị thiếu sang dạng tương thích với scikit-learn. Nếu có nhãn hoặc cột metadata trong đầu vào, pipeline báo lỗi thay vì đưa chúng vào đặc trưng.

Tất cả 14 thuộc tính được giữ trong bước này, bao gồm \texttt{fnlwgt} như một thuộc tính đầu vào; nó không được dùng làm trọng số mẫu. Hai trường \texttt{education} và \texttt{education-num} cùng mô tả học vấn nhưng được giữ theo schema ban đầu. Chưa thực hiện loại thuộc tính hoặc chọn đặc trưng theo nhãn.

\input{chapters/chapter_4/4_3_1_xu_ly_gia_tri_thieu}
\input{chapters/chapter_4/4_3_2_ma_hoa_bien_phan_loai}
\input{chapters/chapter_4/4_3_3_chuan_hoa_bien_so}
''')
    write('chapters/chapter_4/4_3_1_xu_ly_gia_tri_thieu.tex',r'''\subsection{Xử lý giá trị thiếu}
\label{subsec:4_3_1}
Cấu hình cơ sở dùng trung vị cho nhánh số và mức xuất hiện nhiều nhất cho nhánh phân loại. Các lựa chọn khác là trung bình cho biến số hoặc giá trị hằng \texttt{\_\_MISSING\_\_} cho biến phân loại. Dữ liệu thiếu được điền mà không xóa dòng.

Chỉ gọi fit trên tập train đã tách ở bước 3. Các thống kê fitted được đối chiếu lại với train và lưu cùng thông tin phiên bản pipeline. Cột hoàn toàn thiếu được giữ; theo cách xử lý của \texttt{SimpleImputer}, giá trị thay thế cho cột rỗng là 0 khi dùng chiến lược thống kê, hoặc hằng cấu hình khi dùng constant~\cite{sklearn-imputer}.
''')
    write('chapters/chapter_4/4_3_2_ma_hoa_bien_phan_loai.tex',r'''\subsection{Mã hóa biến phân loại}
\label{subsec:4_3_2}
Cấu hình cơ sở dùng one-hot cho cả tám biến phân loại, không bỏ một mức tham chiếu và cho phép mức chưa thấy khi transform. Số cột đầu ra được xác định duy nhất từ những mức có trong train sau khi điền thiếu. Validation và test không bổ sung mức mới vào encoder.

Pipeline cũng hỗ trợ ordinal với mã $-1$ cho mức chưa thấy. Lựa chọn encoding được truyền qua cấu hình để các thí nghiệm sau có thể thay đúng một yếu tố mà giữ các bước còn lại.
''')
    write('chapters/chapter_4/4_3_3_chuan_hoa_bien_so.tex',r'''\subsection{Chuẩn hóa biến số}
\label{subsec:4_3_3}
Nhánh số dùng StandardScaler trong cấu hình cơ sở. Có thể chuyển sang MinMaxScaler với \texttt{feature\_range} cấu hình được hoặc bỏ scaling. Việc chuẩn hóa diễn ra sau điền thiếu và chỉ áp dụng cho sáu biến số.

MinMaxScaler không bật clipping; do đó các giá trị đánh giá nằm ngoài miền train vẫn được giữ đúng theo phép biến đổi đã fit. Chưa loại các điểm bị gắn cờ IQR ở EDA. Các quyết định về hiệu quả của từng cách chuẩn hóa được dành cho bước thí nghiệm mô hình.
''')
    text=r'''\section{Giảm chiều đặc trưng bằng PCA}
\label{sec:4_4}
PCA được đặt sau khi ghép hai nhánh. Bước này tự center đầu vào đã xử lý và không whitening; không có bước scale bổ sung cho từng cột one-hot. Các hướng chiếu và số thành phần được học từ train, dùng bộ giải SVD đầy đủ.
'''
    text+=f"\nCấu hình cơ sở có {number(baseline['feature_count'])} đặc trưng. PCA 90\\% giữ {number(p90['feature_count'])} thành phần và {percent(p90['pca_retained_variance'])}\\% phương sai train; PCA 95\\% giữ {number(p95['feature_count'])} thành phần và {percent(p95['pca_retained_variance'])}\\% phương sai. Đây là kết quả kiểm tra đặc trưng, chưa phải so sánh chất lượng phân loại.\n"
    text+=r'''
\begin{figure}[htbp]
\centering
\includegraphics[width=.9\textwidth]{Images/chapter_4/4_4/pca_cumulative_variance.pdf}
\caption{Phương sai tích lũy của những thành phần được giữ trong cấu hình PCA 95\%}
\label{fig:pca-retained-variance}
\end{figure}
'''
    write('chapters/chapter_4/4_4_giam_chieu_dac_trung_bang_pca.tex',text)
    write('chapters/chapter_4/4_7_cau_hinh_cac_buoc_cua_pipeline.tex',r'''\section{Cấu hình các bước của pipeline}
\label{sec:4_7}
Hàm \texttt{build\_preprocessor(config)} trả về một pipeline chưa fit. Các tham số được kiểm tra trước khi tạo đối tượng. Cấu hình được lưu trong \texttt{configs/preprocessing.json} và các manifest đầu ra.

\begin{table}[htbp]
\centering\small
\caption{Các lựa chọn tiền xử lý được hỗ trợ}\label{tab:preprocessing-options}
\begin{tabular}{p{.31\textwidth}p{.57\textwidth}}
\toprule
Thành phần & Lựa chọn \\
\midrule
Điền thiếu biến số & median hoặc mean \\
Điền thiếu phân loại & most\_frequent hoặc constant với hằng cấu hình \\
Mã hóa & onehot hoặc ordinal \\
Scaling nhánh số & standard, minmax hoặc none \\
Khoảng MinMax & Hai cận tăng dần, mặc định $[0,1]$ \\
PCA & Bỏ qua, giữ 90\% hoặc 95\% phương sai \\
\bottomrule
\end{tabular}
\end{table}

Ba bộ đặc trưng được xuất ở bước 4 là baseline, pca\_90 và pca\_95. Các lựa chọn khác được hỗ trợ để dùng trong thí nghiệm, chưa có kết luận về hiệu quả dự đoán. Khi cross-validation, phải tạo pipeline mới trong từng fold và dùng dữ liệu train thô cùng mã nhóm; không dùng trực tiếp các ma trận đã fit từ toàn bộ train để tính điểm CV~\cite{sklearn-pitfalls}.
''')
    text=r'''\section{Lưu đặc trưng và thông tin đi kèm}
\label{sec:4_8}
Mỗi cấu hình có một thư mục \texttt{features/<cau-hinh>/} chứa \texttt{X\_train.npy}, \texttt{X\_validation.npy}, \texttt{X\_test.npy}, tên cột và manifest. Các ma trận có kiểu \texttt{float32}, cùng số cột trong một cấu hình, không NaN và không có giá trị vô hạn.

\begin{table}[htbp]
\centering\small
\caption{Kích thước đặc trưng đã xuất ở bước 4}\label{tab:feature-shapes}
\begin{tabular}{lrrr}
\toprule
Cấu hình & Train & Validation & Test \\
\midrule
'''
    for name,e in entries.items():
        arrays=e['arrays'];dims={split:arrays[f'features/{name}/X_{split}.npy']['shape'] for split in ['train','validation','test']}
        title=name.replace('_',r'\_')
        text+=title+' & '+' & '.join(f"${number(dims[split][0])}\\times {dims[split][1]}$" for split in ['train','validation','test'])+r' \\'+'\n'
    text+=r'''\bottomrule
\end{tabular}
\end{table}

Các nhãn \texttt{y\_<tap>.npy} dùng kiểu \texttt{int8}, với \texttt{<=50K}=0 và \texttt{>50K}=1. Các tệp \texttt{source\_row\_<tap>.npy} dùng kiểu \texttt{int64} và khớp thứ tự dòng với metadata đã khóa. Manifest ghi cấu hình, tên đặc trưng, ánh xạ nhãn, phiên bản thư viện, mã SHA-256 và nguồn train đã dùng để fit. Các pipeline fitted được lưu trong \texttt{artifacts/preprocessors/} để tái sử dụng trong cùng môi trường tương thích.

Kiểm tra đã đối chiếu các thống kê imputer, scaler, miền encoder và tâm PCA với train. Trạng thái fitted không đổi sau transform validation/test. Dữ liệu gốc và phân chia bước 3 được giữ nguyên. Các nhãn test được mã hóa để lưu tệp, không dùng để fit đặc trưng, chọn cấu hình hoặc tính chỉ số mô hình.
'''
    write('chapters/chapter_4/4_8_luu_dac_trung_va_thong_tin_di_kem.tex',text)
    write('backmatter/appendix_a/A_3_danh_sach_tep_dac_trung_va_cach_su_dung.tex',r'''\section{Danh sách tệp đặc trưng và cách sử dụng}
\label{sec:A_3}
Ba thư mục đặc trưng là \texttt{features/baseline/}, \texttt{features/pca\_90/} và \texttt{features/pca\_95/}. Mỗi thư mục chứa ba ma trận X cùng tên cột và manifest. Nhãn và chỉ số dòng dùng chung được đặt ngay trong \texttt{features/}.

Đọc tệp bằng \texttt{numpy.load(..., allow\_pickle=False)}. Ghép X với y và source\_row của đúng tập, giữ nguyên thứ tự. Tra cứu mã nhóm và source\_id trong metadata CSV ở \texttt{data/splits/}. Không ghép cột giữa các cấu hình PCA và không đưa nhãn hoặc metadata vào X.

Các ma trận này đã được fit tiền xử lý trên toàn bộ train, phù hợp với quy trình holdout đã khóa. Để cross-validation và tìm siêu tham số, dùng dữ liệu train thô và ghép \texttt{build\_preprocessor} vào pipeline mô hình để fit lại trong mỗi fold. Sau khi đổi cấu hình hoặc fit lại trên train cùng validation ở giai đoạn cuối, cần xuất lại đặc trưng và cập nhật manifest.
''')
    write('backmatter/appendix_a/A_1_chay_toan_bo_notebook_tren_google_colab.tex',r'''\section{Chạy toàn bộ notebook trên Google Colab}
\label{sec:A_1}
Tải \texttt{notebooks/Adult\_BTL.ipynb} lên Colab và chọn \texttt{Runtime > Run all}. Notebook tự chuẩn bị module, tải dữ liệu công khai, EDA, khôi phục cách chia đã khóa, fit tiền xử lý trên train và xuất các bộ đặc trưng. Nguồn LaTeX và ZIP báo cáo được cập nhật từ chính kết quả vừa chạy.

Các tệp đặc trưng và manifest nằm trong \texttt{features/}; pipeline fitted nằm trong \texttt{artifacts/preprocessors/}; kiểm tra bước 4 nằm trong \texttt{results/step\_04/}. Notebook dựng lại nguồn báo cáo từ khung nhúng, nên các sửa đổi tự thực hiện cần được sao lưu trước khi chạy lại.

Các ô Python đã được kiểm tra cục bộ trên dữ liệu UCI gốc. Chưa xác nhận chạy trực tiếp trên dịch vụ Colab hoặc biên dịch toàn bộ LaTeX tại đây. Bước 4 chưa huấn luyện bộ phân loại hoặc báo cáo điểm mô hình.
''')
    reference=report/'backmatter/references.tex';text=reference.read_text(encoding='utf-8')
    refs={
        'sklearn-imputer':('SimpleImputer','https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html'),
        'sklearn-onehot':('OneHotEncoder','https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html'),
        'sklearn-preprocessing':('Preprocessing data','https://scikit-learn.org/stable/modules/preprocessing.html'),
        'sklearn-pca':('PCA','https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'),
        'sklearn-pitfalls':('Common pitfalls and recommended practices','https://scikit-learn.org/stable/common_pitfalls.html'),
    }
    for key,(title,url) in refs.items():
        if f'\\bibitem{{{key}}}' not in text:
            text=text.replace(r'\end{thebibliography}',f'\\bibitem{{{key}}}\nScikit-learn developers, \\textit{{{title}}}, tài liệu chính thức.\n\\url{{{url}}}.\n'+r'\end{thebibliography}')
    reference.write_text(text,encoding='utf-8')
    write('TIEN_DO.md','# Tiến độ sau bước 4\n\nGiữ chương 1 và 3, phần chia tập của bước 3. Đã có lý thuyết phân loại, tiền xử lý, PCA; phần hiện thực tiền xử lý và lưu đặc trưng ở chương 4. Các mục mô hình và kết quả phân loại sẽ bổ sung ở bước 5–6. Chưa biên dịch toàn bộ LaTeX tại đây.\n')
    return report
