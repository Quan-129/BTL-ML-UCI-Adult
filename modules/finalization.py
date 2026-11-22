"""Hoàn thiện báo cáo từ kết quả đã khóa, không huấn luyện/đánh giá lại mô hình."""
from pathlib import Path
import json,re
import pandas as pd

TEAM_CONFIG={'members': [{'name': 'Hồng Chấn Phước', 'student_id': '2352963', 'allocation_percent': 33, 'tasks': 'Chuẩn bị và kiểm tra dữ liệu UCI; khảo sát nhãn, giá trị thiếu, bản ghi trùng và EDA; rà soát chương 1 và chương 3.', 'email': 'phuoc.hongchan@hcmut.edu.vn', 'completion_percent': 100}, {'name': 'Đinh Nguy Nguyệt Hà', 'student_id': '2352286', 'allocation_percent': 33, 'tasks': 'Rà soát tiền xử lý, encoding, scaling và PCA; đối chiếu các cấu hình CV, bảng kết quả và cảnh báo hội tụ; phụ trách chương 2, chương 4 và thiết kế thí nghiệm.', 'email': 'ha.dinhnguy2005@hcmut.edu.vn', 'completion_percent': 100}, {'name': 'Trần Bá Minh Quân', 'student_id': '2353015', 'allocation_percent': 34, 'tasks': 'Tích hợp notebook và module; kiểm tra snapshot, đánh giá test và phân tích lỗi; tổng hợp chương 5–7, định dạng báo cáo và đóng gói sản phẩm.', 'email': 'quan.tranbaminh12@hcmut.edu.vn', 'completion_percent': 100}]}

def _add_review_analysis(root, report):
    """Explain existing comparisons quantitatively; no estimator or test selection."""
    cv = pd.read_csv(root/'results/step_05/cv_summary.csv').set_index('candidate_id')
    validation = pd.read_csv(root/'results/step_05/validation_summary.csv').set_index('candidate_id')
    shapes = pd.read_csv(root/'results/step_04/feature_shapes.csv')
    dims = shapes.loc[shapes['split']=='train'].set_index('configuration')['features']
    base, rf, mlp = cv.loc['lr_c1'], cv.loc['rf_full_leaf3'], cv.loc['mlp_32_reg']
    def number(x, digits=2):
        return f'{float(x):.{digits}f}'.replace('.', ',')
    def delta(candidate, reference, metric='f1_mean'):
        return number(100*(cv.loc[candidate, metric]-cv.loc[reference, metric]))
    def append(prefix, text):
        path = next((report/'chapters/chapter_6').glob(prefix+'*.tex'))
        source = path.read_text(encoding='utf-8')
        source = re.sub(r'\n% BEGIN REVIEW ANALYSIS.*?% END REVIEW ANALYSIS\n?', '', source, flags=re.S)
        path.write_text(source.rstrip()+'\n\n% BEGIN REVIEW ANALYSIS\n'+text+'\n% END REVIEW ANALYSIS\n', encoding='utf-8')
    append('6_1_',
        r'Trong nhóm mô hình truyền thống, Random Forest có F1 CV cao nhất là '+number(rf.f1_mean,4)+
        r', hơn Logistic Regression cơ sở '+delta('rf_full_leaf3','lr_c1')+
        r' điểm phần trăm và hơn LinearSVC '+delta('rf_full_leaf3','svm_c1')+
        r' điểm phần trăm. Đây là đối chứng truyền thống chính của bài làm; pipeline đã được ghép tiền xử lý, huấn luyện và đánh giá bằng CV cùng validation. Số cấu hình tìm kiếm giữa các họ không bằng nhau nên bảng so sánh phản ánh thiết kế cụ thể đã chạy.')
    append('6_2_',
        r'So với mốc one-hot, mode và StandardScaler, nhóm thiếu riêng làm F1 tăng '+delta('lr_missing_constant','lr_c1')+
        r' điểm phần trăm; ordinal làm F1 thay đổi '+delta('lr_ordinal','lr_c1')+
        r' điểm phần trăm. Hai khoảng MinMax [0,1] và [-1,1] thay đổi lần lượt '+delta('lr_minmax01','lr_c1')+
        r' và '+delta('lr_minmax11','lr_c1')+
        r' điểm phần trăm. Các hiệu số là chênh lệch F1 trung bình nhân 100, không phải mức tăng tương đối. Mức tăng nhỏ của cách biểu diễn thiếu riêng chưa đủ để kết luận có khác biệt chắc chắn trên các lần lấy mẫu khác.')
    append('6_3_',
        r'Ở bộ đặc trưng fit trên toàn bộ train, số chiều giảm từ '+str(int(dims['baseline']))+r' xuống '+str(int(dims['pca_90']))+
        r' với PCA 90\% và '+str(int(dims['pca_95']))+r' với PCA 95\%, tương ứng giảm '+number(100*(1-dims['pca_90']/dims['baseline']))+
        r'\% và '+number(100*(1-dims['pca_95']/dims['baseline']))+
        r'\%. Đây là kích thước của các tệp NumPy xuất từ toàn bộ train; từng fold CV vẫn tự fit encoder/PCA và có thể có số chiều khác. F1 CV thay đổi '+delta('lr_pca90','lr_c1')+
        r' và '+delta('lr_pca95','lr_c1')+
        r' điểm phần trăm so với không PCA. Như vậy, trong thí nghiệm này, tiết kiệm chiều đi kèm giảm F1; không chọn PCA chỉ vì giữ được một tỷ lệ phương sai cao.')
    append('6_4_',
        r'MLP 32 nút hơn Random Forest '+delta('mlp_32_reg','rf_full_leaf3')+
        r' điểm phần trăm F1 CV; recall cao hơn '+delta('mlp_32_reg','rf_full_leaf3','recall_mean')+
        r' điểm phần trăm, nhưng precision thấp hơn '+number(100*(rf.precision_mean-mlp.precision_mean))+
        r' điểm phần trăm. Random Forest có accuracy '+number(rf.accuracy_mean,4)+r' và ROC-AUC '+number(rf.roc_auc_mean,4)+
        r', đều cao hơn MLP; lựa chọn cuối vẫn theo F1 lớp dương đã định trước. Vì thế không có một mô hình đứng đầu trên mọi chỉ số.'+'\n\n'+
        r'SD F1 của MLP là '+number(mlp.f1_std,4)+r', so với '+number(rf.f1_std,4)+
        r' của Random Forest. SD qua ba fold không phải khoảng tin cậy; các fold dùng chung dữ liệu phát triển và kết quả chưa có kiểm định nhiều seed. Trên validation, F1 của hai mô hình chỉ lệch '+number(100*abs(validation.loc['rf_full_leaf3','f1']-validation.loc['mlp_32_reg','f1']),3)+
        r' điểm phần trăm. Các số liệu này ủng hộ cách diễn đạt MLP đạt F1 CV trung bình cao nhất trong phạm vi khảo sát, chưa chứng minh MLP tốt hơn một cách ổn định.'+'\n\n'+
        r'Thời gian fit trung bình mỗi fold, tính cả pipeline tiền xử lý, là '+number(mlp.fit_seconds_mean)+r' giây cho MLP, '+number(rf.fit_seconds_mean)+
        r' giây cho Random Forest và '+number(cv.loc['lr_missing_constant','fit_seconds_mean'])+
        r' giây cho Logistic Regression dùng nhóm thiếu riêng. Đây là số đo trên môi trường thực nghiệm được lưu, không phải benchmark độc lập hoặc bảo đảm tốc độ trên Colab. Kích thước dữ liệu, thư viện và phần cứng đều có thể làm thời gian thay đổi.')


def finalize_report(project_dir='.',team_config=None):
    root=Path(project_dir);report=root/'reports/latex';out=root/'results/step_07';out.mkdir(parents=True,exist_ok=True)
    team=TEAM_CONFIG if team_config is None else team_config
    if sum(m['allocation_percent'] for m in team['members'])!=100:raise ValueError('Tổng tỷ lệ phân công phải bằng 100%.')
    (root/'configs/team.json').write_text(json.dumps(team,ensure_ascii=False,indent=2),encoding='utf-8')
    data=json.loads((root/'results/step_06/test_metrics.json').read_text(encoding='utf-8'))
    metrics=data['metrics'];counts=data['counts']
    cv=pd.read_csv(root/'results/step_05/cv_summary.csv');winner=cv.set_index('candidate_id').loc[data['winner_id']]
    val=pd.read_csv(root/'results/step_05/validation_summary.csv').set_index('candidate_id').loc[data['winner_id']]
    def n(x):return f'{int(x):,}'.replace(',','.')
    def dec(x):return f'{float(x):.4f}'.replace('.',',')
    def pct(x):return f'{float(x)*100:.2f}'.replace('.',',')
    def esc(x):return str(x).replace('_',r'\_').replace('%',r'\%')
    def write(rel,text):
        path=report/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    def edit(rel,replacements):
        path=report/rel;text=path.read_text(encoding='utf-8')
        for before,after in replacements:
            if before not in text:
                if after in text:continue
                raise ValueError('Không tìm thấy đoạn cần cập nhật: '+rel+' / '+before[:60])
            text=text.replace(before,after)
        path.write_text(text,encoding='utf-8')
    write('frontmatter/abstract.tex',r'''\chapter*{Tóm tắt}
\phantomsection
\addcontentsline{toc}{chapter}{Tóm tắt}
Báo cáo xây dựng một pipeline học máy có thể cấu hình cho bài toán dự đoán thu nhập năm trên 50.000 USD từ dữ liệu bảng UCI Adult. Hai tệp gốc gồm 32.561 mẫu phát triển và 16.281 mẫu test, với 14 thuộc tính đầu vào. Tập phát triển được chia theo nhóm hồ sơ thành 26.049 mẫu train và 6.512 mẫu validation. Pipeline kết hợp điền thiếu, mã hóa biến phân loại, chuẩn hóa biến số, PCA tùy chọn và bộ phân loại; mọi phép biến đổi học tham số được fit đúng phần train tương ứng.

Vòng so sánh cuối khảo sát 20 cấu hình bằng cross-validation ba fold theo nhóm, gồm Logistic Regression, SVM tuyến tính, Random Forest, MLP và mốc dự đoán đa số. Hai cấu hình MLP được bổ sung sau pilot do cảnh báo tối ưu; toàn bộ dấu vết pilot và kết quả sau mở rộng được giữ lại. Tiêu chí lựa chọn là F1 của lớp \texttt{>50K}, trước khi đối chiếu validation. Cấu hình được khóa là MLP một lớp ẩn 32 nút, không PCA và ngưỡng xác suất 0,5.

Mô hình đạt F1 CV trung bình '''+dec(winner.f1_mean)+r''', F1 validation '''+dec(val.f1)+r''' và F1 test '''+dec(metrics['f1'])+r'''. Trên test, accuracy đạt '''+pct(metrics['accuracy'])+r'''\%, precision '''+pct(metrics['precision'])+r'''\%, recall '''+pct(metrics['recall'])+r'''\% và ROC-AUC '''+dec(metrics['roc_auc'])+r'''. Kết quả cho thấy cần đọc accuracy cùng khả năng phát hiện lớp thu nhập cao. Giới hạn gồm lưới tham số nhỏ, một seed, ba fold, dữ liệu lịch sử và một số hồ sơ đầu vào trùng giữa test với dữ liệu phát triển. Báo cáo đi kèm notebook, module, đặc trưng NumPy, mô hình cố định và các bảng kiểm tra để đối chiếu số liệu.

\noindent\textbf{Từ khóa:} UCI Adult, phân loại nhị phân, tiền xử lý, cross-validation theo nhóm, PCA, MLP.
''')
    write('frontmatter/acknowledgements.tex',r'''\chapter*{Lời cảm ơn}
\phantomsection
\addcontentsline{toc}{chapter}{Lời cảm ơn}
Nhóm xin cảm ơn thầy Trương Vĩnh Lân và học phần Học máy đã cung cấp nội dung lý thuyết, yêu cầu thực hành và định hướng để tổ chức bài tập lớn theo một quy trình có thể kiểm tra. Việc thực hiện đề tài tạo cơ hội liên hệ các khái niệm tiền xử lý, lựa chọn mô hình và đánh giá với một tập dữ liệu bảng cụ thể.

Nhóm cũng cảm ơn những người công bố dữ liệu Adult trên UCI Machine Learning Repository và cộng đồng phát triển các thư viện khoa học Python. Dữ liệu công khai và tài liệu chính thức là cơ sở để xây dựng thí nghiệm, tra cứu cách sử dụng và đối chiếu kết quả.
''')
    abbreviations=[('AP','Average Precision: độ chính xác trung bình theo đường Precision–Recall'),
        ('CV','Cross-validation: đánh giá chéo'),('EDA','Exploratory Data Analysis: phân tích khám phá dữ liệu'),
        ('FN / FP','False Negative / False Positive: bỏ sót lớp dương / báo dương sai'),
        ('FNR / FPR','Tỷ lệ bỏ sót lớp dương / tỷ lệ báo dương sai'),('MLP','Multilayer Perceptron: mạng nơ-ron kết nối đầy đủ nhiều lớp'),
        ('PCA','Principal Component Analysis: phân tích thành phần chính'),('PR','Precision–Recall'),
        ('ROC-AUC','Diện tích dưới đường Receiver Operating Characteristic'),('SD','Standard Deviation: độ lệch chuẩn'),
        ('SVM','Support Vector Machine: máy véc-tơ hỗ trợ'),('TN / TP','True Negative / True Positive: nhận diện đúng lớp âm / lớp dương'),
        ('UCI','University of California, Irvine; đơn vị vận hành kho dữ liệu sử dụng')]
    ab=[r'\chapter*{Danh mục từ viết tắt}',r'\phantomsection',r'\addcontentsline{toc}{chapter}{Danh mục từ viết tắt}',
        r'\begin{longtable}{p{.18\textwidth}p{.75\textwidth}}',r'\toprule',r'Từ viết tắt & Diễn giải \\',r'\midrule']
    ab += [a+' & '+b+r' \\' for a,b in abbreviations]
    write('frontmatter/abbreviations.tex','\n'.join(ab+[r'\bottomrule',r'\end{longtable}'])+'\n')
    write('chapters/chapter_7/7_1_ket_qua_dat_duoc.tex',r'''\section{Kết quả đạt được}
\label{sec:7_1}
Đề tài đã triển khai quy trình học máy truyền thống cho UCI Adult và bổ sung MLP để đối chiếu. Quy trình đi từ tải dữ liệu công khai, kiểm tra chất lượng và EDA đến phân chia theo nhóm, tiền xử lý có thể cấu hình, khảo sát mô hình và đánh giá test với lựa chọn đã khóa. Không loại dòng trong hai tệp UCI gốc; toàn bộ 14 thuộc tính, gồm cả \texttt{fnlwgt}, được giữ làm đầu vào. \texttt{fnlwgt} không được dùng làm trọng số mẫu.

Các sản phẩm gồm notebook tự chuẩn bị dữ liệu, module Python, nguồn LaTeX tách theo từng mục, các bảng/hình thực nghiệm, snapshot mô hình và 15 tệp NumPy. Ba bộ X fit từ toàn bộ train gồm bản cơ sở 105 chiều, PCA 90\% còn 22 chiều và PCA 95\% còn 31 chiều. Khi CV, pipeline được fit lại trên raw train của mỗi fold, không sử dụng trực tiếp các ma trận đã fit từ toàn bộ train.

Trong phạm vi 20 cấu hình cuối, MLP 32 nút được chọn theo F1 CV trung bình '''+dec(winner.f1_mean)+r''' với SD '''+dec(winner.f1_std)+r'''. F1 validation là '''+dec(val.f1)+r''' và F1 test là '''+dec(metrics['f1'])+r'''. Trên 16.281 mẫu test, mô hình đạt accuracy '''+pct(metrics['accuracy'])+r'''\%, ROC-AUC '''+dec(metrics['roc_auc'])+r''' và AP '''+dec(metrics['average_precision'])+r'''. Ngưỡng 0,5 và snapshot được giữ nguyên sau khi xem test.

Các đối chứng Logistic Regression chỉ thay một thành phần cho thấy kết quả phụ thuộc cách biểu diễn và thang đo. Trong các cấu hình đã chạy, ordinal và hai mức PCA có F1 thấp hơn mốc one-hot không PCA; lựa chọn điền biến phân loại bằng một nhóm thiếu có mức tăng nhỏ. Đây là kết quả trong thiết kế đã khảo sát, không phải kết luận về ưu thế phổ quát của một bộ mã hóa hay mô hình.

Việc đối chiếu số liệu từ dự đoán từng mẫu giúp giải thích chất lượng theo số đếm: '''+n(counts['tp'])+r''' TP, '''+n(counts['fp'])+r''' FP, '''+n(counts['fn'])+r''' FN và '''+n(counts['tn'])+r''' TN. Các tệp kết quả, mã SHA-256 và cấu hình cho phép truy lại nguồn của các bảng trong báo cáo.
''')
    write('chapters/chapter_7/7_2_han_che_cua_nghien_cuu.tex',r'''\section{Hạn chế của nghiên cứu}
\label{sec:7_2}
\begin{itemize}
\item \textbf{Phạm vi tìm kiếm:} sử dụng ba fold, một seed và lưới siêu tham số nhỏ, không đồng đều giữa các họ. Hai cấu hình MLP được thêm sau khi xem cảnh báo hội tụ của pilot; quá trình này được công khai và chưa sử dụng validation/test. F1 CV thắng cuộc vẫn là số liệu phục vụ lựa chọn, không phải ước lượng hiệu quả cuối không thiên lệch.
\item \textbf{Phân chia và trùng hồ sơ:} train/validation không chung nhóm 14 thuộc tính; tuy nhiên, giữ nguyên test UCI dẫn đến 22 dòng test trùng hồ sơ train và 4 dòng trùng hồ sơ validation. Vì vậy không khẳng định phân chia test tách biệt tuyệt đối về hồ sơ. Không loại các dòng này sau khi nhìn kết quả.
\item \textbf{Vai trò EDA/validation:} EDA được thực hiện trên toàn bộ \texttt{adult.data} trước khi tách train/validation. Validation chỉ là đối chiếu mô tả trong dữ liệu phát triển, không được trình bày như một tập hoàn toàn chưa từng được khảo sát. Các thống kê imputer/scaler/PCA vẫn chỉ fit từ phần train tương ứng.
\item \textbf{Khả năng phát hiện lớp dương:} recall test bằng '''+pct(metrics['recall'])+r'''\%, còn '''+pct(metrics['false_negative_rate'])+r'''\% mẫu dương bị bỏ sót tại ngưỡng đã khóa. Accuracy cao hơn không đồng nghĩa bài toán lớp thiểu số đã được giải quyết đầy đủ. Chưa khảo sát chi phí lỗi, class weighting hoặc resampling.
\item \textbf{Phạm vi suy rộng:} Adult phản ánh một tập dữ liệu điều tra dân số Hoa Kỳ lịch sử. Kết quả chung không chứng minh khả năng áp dụng cho thời kỳ, quốc gia hoặc quần thể khác; chưa có đánh giá theo nhóm nhân khẩu hay kiểm tra hiệu chỉnh xác suất.
\item \textbf{Tái lập:} kết quả có thể thay đổi khi phiên bản thư viện, phần cứng hoặc thuật toán thay đổi. Môi trường thực nghiệm, seed và snapshot mô hình được lưu kèm để đối chiếu khi thực hiện lại.
\end{itemize}
Tập test đã được sử dụng để đánh giá cuối. Báo cáo không tiếp tục dùng kết quả test này để điều chỉnh rồi gọi lần đo tiếp theo là đánh giá trên dữ liệu chưa thấy.
''')
    write('chapters/chapter_7/7_3_huong_phat_trien.tex',r'''\section{Hướng phát triển}
\label{sec:7_3}
Các hướng mở rộng sau chỉ là đề xuất, chưa phải các thí nghiệm đã thực hiện:
\begin{itemize}
\item Mở rộng lưới tham số, thử nhiều seed và thiết kế CV theo nhóm lồng nhau để đánh giá độ ổn định và ảnh hưởng của quá trình lựa chọn.
\item Khảo sát trọng số lớp, hiệu chỉnh xác suất và tối ưu ngưỡng bằng dữ liệu train/validation theo nhóm, căn cứ vào chi phí FP/FN được xác định trước. Một lần đánh giá cuối mới cần có dữ liệu giữ lại mới, không tận dụng test đã xem để tinh chỉnh.
\item Đối chiếu vai trò của \texttt{fnlwgt}, hai trường học vấn và các thuộc tính vốn; tách rõ sử dụng trọng số khảo sát với sử dụng chúng làm đặc trưng. Đánh giá lựa chọn trên dữ liệu phát triển và kiểm tra trên một holdout mới có kiểm soát trùng hồ sơ.
\item Đánh giá độ tin cậy, hiệu chỉnh score, kết quả theo nhóm và mức dịch chuyển dữ liệu trước khi cân nhắc một ứng dụng thực tế. Các phân tích giải thích cần phân biệt mối liên hệ quan sát với tác động nhân quả.
\item Đóng gói môi trường thực nghiệm, cố định phiên bản thư viện và tự động hóa các bước kiểm tra để tăng khả năng tái lập.
\end{itemize}
Các cải tiến cần được tổ chức như một nghiên cứu tiếp theo với giao thức đánh giá mới. Trong báo cáo hiện tại, mô hình và kết quả test ở chương 6 được giữ cố định.
''')
    write('backmatter/appendix_a/A_1_chay_toan_bo_notebook_tren_google_colab.tex','\\section{Chạy notebook trên Google Colab}\n\\label{sec:A_1}\nMở \\texttt{notebooks/Adult\\_BTL.ipynb} trên Google Colab và chọn Runtime $\\rightarrow$ Run all. Notebook tự cài các thư viện cần thiết, tải dữ liệu từ UCI, tạo module và thực hiện toàn bộ quy trình thí nghiệm.\n\nCác kết quả được lưu trong \\texttt{results/}; đặc trưng NumPy nằm trong \\texttt{features/}; pipeline đã huấn luyện nằm trong \\texttt{artifacts/}. Báo cáo PDF và ZIP LaTeX được xuất vào \\texttt{reports/}.\n\nĐể biên dịch nguồn LaTeX trên Overleaf, nhập \\texttt{BTL\\_ML\\_LaTeX\\_Final.zip}, chọn \\texttt{main.tex} làm tài liệu chính và XeLaTeX làm trình biên dịch. Notebook tạo lại nguồn báo cáo khi thực thi phần xuất báo cáo.\n')
    write('backmatter/appendix_a/A_2_cau_truc_thu_muc_nop_bai.tex',r'''\section{Cấu trúc thư mục nộp bài}
\label{sec:A_2}
Gói tổng được tổ chức theo các thư mục sau:
\begin{itemize}
\item \texttt{notebooks/}: notebook chính \texttt{Adult\_BTL.ipynb} để mở trong Colab.
\item \texttt{modules/}: tải dữ liệu, EDA, chia tập, tiền xử lý, CV, đánh giá, cập nhật và dàn trang báo cáo.
\item \texttt{reports/}: PDF, nguồn \texttt{latex/} và ZIP LaTeX. Trong nguồn có \texttt{frontmatter/}, \texttt{chapters/}, \texttt{config/}, \texttt{backmatter/} và \texttt{Images/}; mỗi mục và tiểu mục có tệp riêng, thư mục hình phản ánh cùng cấu trúc.
\item \texttt{features/}: X cơ sở/PCA, y, chỉ số dòng nguồn, tên đặc trưng và manifest.
\item \texttt{data/}: ZIP UCI, các tệp nguồn, bảng đã đọc và train/validation/test cùng metadata.
\item \texttt{configs/}: cấu hình tiền xử lý, thí nghiệm và phân công nhóm; \texttt{artifacts/}: các pipeline và snapshot cuối.
\item \texttt{results/}: kết quả theo bước, bảng CSV, hình PDF/PNG và hồ sơ kiểm tra.
\end{itemize}
Tệp README ở thư mục gốc mô tả cách chạy và các giới hạn kiểm tra. Không cần mount Drive hoặc tải các module thủ công khi chạy notebook độc lập.
''')
    write('backmatter/appendix_a/A_4_lien_ket_ma_nguon_notebook_va_du_lieu.tex','\\section{Dữ liệu và mã nguồn}\n\\label{sec:A_4}\n\\begin{itemize}\n\\item Trang dữ liệu: \\url{https://archive.ics.uci.edu/dataset/2/adult}.\n\\item Tệp ZIP: \\url{https://archive.ics.uci.edu/static/public/2/adult.zip}.\n\\item DOI: \\url{https://doi.org/10.24432/C5XW20}.\n\\item Repository: \\url{https://github.com/Quan-129/BTL-ML-UCI-Adult}.\n\\item Mở notebook trên Colab: \\url{https://colab.research.google.com/github/Quan-129/BTL-ML-UCI-Adult/blob/main/notebooks/Adult_BTL.ipynb}.\n\\item Notebook: \\texttt{notebooks/Adult\\_BTL.ipynb}.\n\\item Module Python và cấu hình: \\texttt{modules/}, \\texttt{configs/}.\n\\item Báo cáo: \\texttt{reports/BTL\\_ML\\_Bao\\_Cao.pdf}; nguồn LaTeX: \\texttt{reports/latex/main.tex}.\n\\end{itemize}\n')
    rows=[r'\section{Bảng phân công công việc}',r'\label{sec:B_1}',
        r'\begin{longtable}{p{.28\textwidth}p{.16\textwidth}p{.48\textwidth}}',
        r'\caption{Thông tin thành viên}\label{tab:team-members}\\',r'\toprule',
        r'Thành viên & MSSV & Email \\',r'\midrule']
    for member in team['members']:
        rows.append(member['name']+' & '+member['student_id']+' & '+r'\texttt{'+member['email']+r'} \\')
    rows += [r'\bottomrule',r'\end{longtable}',
        'Tỷ lệ phân công thể hiện phần khối lượng công việc của từng thành viên. Mức hoàn thành được trình bày riêng cho từng phần việc.',
        r'\begin{longtable}{p{.19\textwidth}p{.47\textwidth}rr}',
        r'\caption{Phân công và mức hoàn thành công việc}\label{tab:team-tasks}\\',
        r'\toprule',r'Thành viên & Nhiệm vụ & Phân công & Hoàn thành \\',r'\midrule']
    for member in team['members']:
        rows.append(member['name']+' & '+member['tasks']+' & '+str(member['allocation_percent'])+r'\% & '+str(member['completion_percent'])+r'\% \\')
    rows += [r'\bottomrule',r'\end{longtable}']
    write('backmatter/appendix_b/B_1_bang_phan_cong_va_ty_le_dong_gop.tex','\n'.join(rows)+'\n')
    write('backmatter/appendix_b/B_2_tien_do_thuc_hien_cua_nhom.tex','\\section{Quy trình thực hiện}\n\\label{sec:B_2}\nCông việc được triển khai theo bảy giai đoạn: thu thập dữ liệu; khảo sát EDA; chia tập; tiền xử lý và xuất đặc trưng; so sánh mô hình bằng CV; đánh giá test và phân tích lỗi; tổng hợp báo cáo.\n\nCác bảng thống kê, cấu hình và kết quả từng giai đoạn được lưu trong \\texttt{results/}. Notebook tích hợp các module để thực hiện lại quy trình từ dữ liệu gốc đến các sản phẩm cuối.\n')
    write('backmatter/appendix_b/appendix_b.tex',r'\chapter{Phân công công việc}'+'\n'+r'\label{chap:B}'+'\n'+
        r'\input{backmatter/appendix_b/B_1_bang_phan_cong_va_ty_le_dong_gop}'+'\n'+
        r'\input{backmatter/appendix_b/B_2_tien_do_thuc_hien_cua_nhom}'+'\n')
    for relative in ['frontmatter/declaration.tex','backmatter/appendix_b/B_3_minh_chung_hoat_dong_nhom.tex','TIEN_DO.md']:
        obsolete=report/relative
        if obsolete.is_file():obsolete.unlink()
    write('backmatter/appendix_c/appendix_c.tex',r'''\chapter{Bảng kết quả đối chiếu}
\label{chap:C}
\input{backmatter/appendix_c/C_1_bang_ket_qua_cv_day_du}
''')
    lines=[r'\section{Kết quả CV của toàn bộ cấu hình}',r'\label{sec:C_1}',
        'Bảng liệt kê đủ 20 cấu hình ở vòng cuối. Chỉ số precision, recall và F1 tính cho lớp >50K; SD là độ lệch chuẩn mẫu qua ba fold. Cột CB ghi số fold có cảnh báo hội tụ. Những cấu hình có CB lớn hơn 0 vẫn được báo cáo nhưng không đủ điều kiện khóa lựa chọn.',
        r'\begin{longtable}{lrrrrrr}',r'\caption{Toàn bộ kết quả CV ở vòng so sánh cuối}\label{tab:cv-all}\\',
        r'\toprule',r'Cấu hình & F1 & SD & Precision & Recall & Accuracy & CB \\',r'\midrule']
    for _,r in cv.iterrows():
        lines.append(esc(r.candidate_id)+' & '+' & '.join(dec(r[k]) for k in ['f1_mean','f1_std','precision_mean','recall_mean','accuracy_mean'])+' & '+str(int(r.convergence_warning_count))+r' \\')
    lines += [r'\bottomrule',r'\end{longtable}',
        r'Các kết quả từng fold, thời gian, cấu hình đầy đủ và cảnh báo gốc nằm trong \texttt{results/step\_05/}. Lịch sử pilot 18 cấu hình được giữ ở \texttt{pilot\_18/}.']
    write('backmatter/appendix_c/C_1_bang_ket_qua_cv_day_du.tex','\n'.join(lines)+'\n')
    (report/'Images/backmatter/appendix_c/C_1').mkdir(parents=True,exist_ok=True)
    edit('chapters/chapter_1/1_3_pham_vi_nghien_cuu.tex',[
        ('Các quyết định sử dụng hoặc biến đổi thuộc tính sẽ được trình bày cùng lý do trong các phần khảo sát dữ liệu và xây dựng pipeline.',
         'Cả 14 thuộc tính được giữ làm đầu vào; các phép biến đổi và cấu hình được trình bày trong phần khảo sát dữ liệu và xây dựng pipeline.')])
    edit('chapters/chapter_4/4_1_so_do_tong_the_cua_pipeline.tex',[
        ('Bước 4 chỉ chuẩn bị đặc trưng và chưa huấn luyện bộ phân loại. Ở bước thực nghiệm tiếp theo, bộ tiền xử lý chưa fit sẽ được ghép với từng mô hình trong một pipeline hoàn chỉnh, để fit đúng phạm vi dữ liệu của mỗi fold.',
         'Phần chuẩn bị đặc trưng có thể được xuất độc lập thành các ma trận NumPy. Khi thực nghiệm mô hình, bộ tiền xử lý chưa fit được ghép với bộ phân loại trong một pipeline hoàn chỉnh để fit đúng phạm vi dữ liệu của từng fold. Sau lựa chọn bằng CV, pipeline được fit trên toàn bộ train và giữ cố định để đánh giá cuối.')])
    edit('chapters/chapter_4/4_7_cau_hinh_cac_buoc_cua_pipeline.tex',[
        ('Các lựa chọn khác được hỗ trợ để dùng trong thí nghiệm, chưa có kết luận về hiệu quả dự đoán.',
         'Các đối chứng tiền xử lý và PCA đã thực hiện được mô tả ở chương 5 và chương 6; các lựa chọn chỉ được hỗ trợ nhưng chưa chạy không được gán một kết quả thực nghiệm.')])
    edit('chapters/chapter_4/4_8_luu_dac_trung_va_thong_tin_di_kem.tex',[
        ('$26.049',r'$26\,049'),('$6.512',r'$6\,512'),('$16.281',r'$16\,281'),
        ('không dùng để fit đặc trưng, chọn cấu hình hoặc tính chỉ số mô hình.',
         'không dùng để fit đặc trưng hoặc chọn cấu hình. Chỉ số test được tính riêng sau khi đã khóa mô hình, như trình bày ở mục 6.5.')])
    edit('chapters/chapter_5/5_8_nguyen_tac_chon_mo_hinh_va_danh_gia_cuoi_cung.tex',[
        ('Bước 6 sẽ dùng đúng mô hình và ngưỡng đã khóa để đánh giá một lần trên test.',
         'Phép đánh giá cuối dùng đúng mô hình và ngưỡng đã khóa; kết quả nằm ở mục 6.5 và không được dùng để chọn lại cấu hình.')])
    edit('chapters/chapter_6/6_7_nhan_xet_tong_hop_ve_cac_ket_qua.tex',[
        ('Bước 6 hoàn tất phần đánh giá mô hình đã khóa. Chương kết luận, tóm tắt, kiểm tra trình bày và các thông tin phân công/minh chứng thực tế của nhóm được hoàn thiện ở bước 7.',
         'Chương 7 tổng hợp các kết quả, hạn chế và hướng phát triển. Các phụ lục trình bày cấu trúc sản phẩm, cách thực thi và bảng phân công công việc.')])
    # Avoid an inline matrix for clearer text extraction in the PDF companion.
    edit('chapters/chapter_6/6_5_2_ma_tran_nham_lan_va_duong_cong_danh_gia.tex',[
        (r'ma trận có dạng $\left[\begin{smallmatrix}TN&FP\\FN&TP\end{smallmatrix}\right]$',
         'hàng thứ nhất lần lượt là TN, FP; hàng thứ hai lần lượt là FN, TP')])
    write('main.tex',r'''% !TeX program = xelatex
\documentclass[12pt,a4paper,oneside,openany]{report}
\input{config/fonts}
\input{config/packages}
\input{config/metadata}
\input{config/layout}
\input{config/headers_footers}
\begin{document}
\input{frontmatter/cover}
\pagenumbering{roman}
\input{frontmatter/acknowledgements}
\input{frontmatter/abstract}
\input{frontmatter/contents}
\input{frontmatter/abbreviations}
\input{frontmatter/list_of_figures}
\input{frontmatter/list_of_tables}
\clearpage
\pagenumbering{arabic}
\input{chapters/chapter_1/chapter_1}
\input{chapters/chapter_2/chapter_2}
\input{chapters/chapter_3/chapter_3}
\input{chapters/chapter_4/chapter_4}
\input{chapters/chapter_5/chapter_5}
\input{chapters/chapter_6/chapter_6}
\input{chapters/chapter_7/chapter_7}
\input{backmatter/references}
\appendix
\input{backmatter/appendix_a/appendix_a}
\input{backmatter/appendix_b/appendix_b}
\input{backmatter/appendix_c/appendix_c}
\end{document}
''')
    layout=report/'config/layout.tex';text=layout.read_text(encoding='utf-8')
    text=text.replace('Khung báo cáo UCI Adult','Dự đoán thu nhập với UCI Adult').replace('Khung tựa mục báo cáo','Báo cáo thực nghiệm học máy')
    layout.write_text(text,encoding='utf-8')
    write('README.md','# Nguồn LaTeX báo cáo UCI Adult\n\nMở main.tex bằng XeLaTeX trên Overleaf. Các chương, mục và tiểu mục được lưu trong từng file riêng; Images/ có cấu trúc tương ứng.\n\nThông tin đề tài và thành viên nằm trong config/metadata.tex; bố cục và đầu/chân trang nằm trong config/layout.tex và config/headers_footers.tex.\n\nPDF do notebook xuất được dàn bằng ReportLab. Khi biên dịch nguồn bằng XeLaTeX, cách ngắt dòng và số trang có thể khác.\n')
    edit('chapters/chapter_1/1_5_bo_cuc_bao_cao.tex', [('phân công công việc và minh chứng hoạt động nhóm.', 'phân công công việc.')])
    _add_review_analysis(root, report)
    index=['# Vị trí từng mục trong báo cáo','', '| Tựa mục | File LaTeX |', '|---|---|']
    for p in sorted(report.rglob('*.tex')):
        if p.name=='main.tex' or p.parent.name=='config':continue
        match=re.search(r'\\(?:chapter|section|subsection)\*?\{([^}]+)\}',p.read_text(encoding='utf-8'))
        if match:index.append('| '+match.group(1)+' | `'+p.relative_to(report).as_posix()+'` |')
    write('DE_CUONG.md','\n'.join(index)+'\n')
    status={'technical_report_content_completed':True,'team_allocation':team,
        'test_metrics_unchanged':True,'model_retrained_in_step_7':False,'test_reevaluated_for_tuning':False,
        'pdf_engine':'ReportLab, rendered from accompanying LaTeX source',
        'latex_compilation_verified':False,'colab_service_run_verified':True,
        'notebook_protocol':'52 code cells completed on hosted Google Colab; numerical report uses the recorded local experiment.'}
    (out/'completion_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    return status
