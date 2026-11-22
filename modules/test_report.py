"""Bổ sung báo cáo từ kết quả test cố định, không tính hoặc chọn lại mô hình."""
from pathlib import Path
import json,shutil
import pandas as pd

def update_test_report(result_dir='results/step_06',report_root='reports/latex'):
    result=Path(result_dir);report=Path(report_root)
    data=json.loads((result/'test_metrics.json').read_text(encoding='utf-8'))
    overlap=json.loads((result/'test_profile_overlap.json').read_text(encoding='utf-8'))
    metric=data['metrics'];count=data['counts']
    def n(value):return f'{int(value):,}'.replace(',','.')
    def dec(value):return f'{float(value):.4f}'.replace('.',',')
    def pct(value):return f'{float(value)*100:.2f}'.replace('.',',')
    def esc(value):return str(value).replace('_',r'\_').replace('%',r'\%')
    def write(folder,prefix,text):
        matches=list((report/folder).glob(prefix+'*.tex'))
        if len(matches)!=1:raise ValueError((folder,prefix,matches))
        matches[0].write_text(text,encoding='utf-8')
    c6='chapters/chapter_6'
    dest=report/'Images/chapter_6/6_5/6_5_2';dest.mkdir(parents=True,exist_ok=True)
    for stem in ['test_confusion_matrix','test_roc','test_precision_recall']:
        for ext in ['png','pdf']:shutil.copyfile(result/f'{stem}.{ext}',dest/f'{stem}.{ext}')
    write(c6,'6_5_danh_',r'''\section{Đánh giá mô hình đã chọn trên tập kiểm tra}
\label{sec:6_5}
Mô hình được đánh giá là \texttt{'''+esc(data['winner_id'])+r'''}, MLP một lớp ẩn 32 nút với $\alpha=0{,}01$, đã được chọn bằng F1 CV ở bước 5. Đầu vào dùng median, mode, one-hot và StandardScaler trên sáu biến số, không PCA. Mô hình đã fit trên 26.049 mẫu train; không fit thêm bằng validation hoặc test. Ngưỡng xác suất lớp \texttt{>50K} giữ nguyên ở 0,5.

Trước khi đánh giá, pipeline và thông tin lựa chọn được sao lưu cố định ở \texttt{artifacts/final/}. SHA-256 và trạng thái fitted được kiểm tra trước/sau dự đoán. Module đánh giá không gọi fit, không tìm ngưỡng tối ưu và không so sánh thêm các mô hình trên test để đổi lựa chọn. Lần chạy lại với cùng snapshot và dữ liệu đọc kết quả đã xác minh thay vì mở một thí nghiệm lựa chọn mới.

Tập test là toàn bộ 16.281 dòng \texttt{adult.test} gốc sau khi chuẩn hóa định dạng như bước 1, gồm '''+n(count['negative_rows'])+r''' mẫu \texttt{<=50K} và '''+n(count['positive_rows'])+r''' mẫu \texttt{>50K}. Mã hóa nhãn không thay số dòng hay thứ tự. Các mục dưới đây trình bày kết quả test; không gộp chúng với điểm CV/validation.

\input{chapters/chapter_6/6_5_1_cac_chi_so_danh_gia}
\input{chapters/chapter_6/6_5_2_ma_tran_nham_lan_va_duong_cong_danh_gia}
''')
    table=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Các chỉ số trên toàn bộ tập test với ngưỡng đã khóa}\label{tab:test-metrics}',
           r'\begin{tabular}{lr}',r'\toprule',r'Chỉ số & Giá trị \\',r'\midrule']
    for title,key in [('Accuracy','accuracy'),('Precision lớp >50K','precision'),('Recall lớp >50K','recall'),('F1 lớp >50K','f1'),
                      ('Macro-F1','macro_f1'),('ROC-AUC','roc_auc'),('Average precision (AP)','average_precision'),
                      ('Specificity','specificity'),('False positive rate (FPR)','false_positive_rate'),
                      ('False negative rate (FNR)','false_negative_rate'),('Balanced accuracy','balanced_accuracy')]:
        table.append(title+' & '+dec(metric[key])+r' \\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    perclass=pd.read_csv(result/'classification_report.csv')
    class_lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Kết quả test theo từng lớp}\label{tab:test-per-class}',
                 r'\begin{tabular}{lrrrr}',r'\toprule',r'Lớp & Precision & Recall & F1 & Số mẫu \\',r'\midrule']
    for _,r in perclass.iterrows():
        class_lines.append(r'\texttt{'+r['class']+'} & '+' & '.join(dec(r[k]) for k in ['precision','recall','f1-score'])+' & '+n(r.support)+r' \\')
    class_lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write(c6,'6_5_1_',r'''\subsection{Các chỉ số đánh giá}
\label{subsec:6_5_1}
Các chỉ số được tính từ nhãn test và dự đoán cố định; lớp dương là \texttt{>50K}. Accuracy đạt '''+pct(metric['accuracy'])+r'''\%, nhưng recall lớp dương chỉ đạt '''+pct(metric['recall'])+r'''\%. Vì vậy, tỷ lệ dự đoán đúng tổng thể không đủ để mô tả khả năng nhận diện nhóm thu nhập cao.

'''+ '\n'.join(table)+'\n'+ '\n'.join(class_lines)+r'''
Precision '''+dec(metric['precision'])+r''' và recall '''+dec(metric['recall'])+r''' cho thấy tại ngưỡng 0,5 mô hình vẫn bỏ sót một phần mẫu dương. Specificity bằng '''+dec(metric['specificity'])+r''' mô tả tỷ lệ nhận diện đúng lớp âm; FPR bằng '''+dec(metric['false_positive_rate'])+r''' là tỷ lệ mẫu âm bị dự đoán thành dương. Macro-F1 và balanced accuracy cho mỗi lớp vai trò cân bằng hơn so với accuracy tổng thể.

ROC-AUC và AP sử dụng score lớp dương để đánh giá thứ hạng qua nhiều ngưỡng. AP là tổng có trọng số theo bước tăng recall, không được đồng nhất với diện tích PR tính bằng quy tắc hình thang~\cite{sklearn-ap}. Không dùng các đường cong test để chọn lại ngưỡng.
''')
    write(c6,'6_5_2_',r'''\subsection{Ma trận nhầm lẫn và đường cong đánh giá}
\label{subsec:6_5_2}
Hàng của ma trận là nhãn thật, cột là nhãn dự đoán; thứ tự lớp là \texttt{<=50K}, \texttt{>50K}. Với quy ước này, ma trận có dạng $\left[\begin{smallmatrix}TN&FP\\FN&TP\end{smallmatrix}\right]$~\cite{sklearn-cm}. Kết quả gồm TN='''+n(count['tn'])+', FP='+n(count['fp'])+', FN='+n(count['fn'])+' và TP='+n(count['tp'])+r'''.

\begin{figure}[htbp]
\centering
\includegraphics[width=.98\textwidth]{Images/chapter_6/6_5/6_5_2/test_confusion_matrix.pdf}
\caption{Ma trận nhầm lẫn: số mẫu và tỷ lệ chuẩn hóa theo hàng}\label{fig:test-cm}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{minipage}{.49\textwidth}\centering
\includegraphics[width=\linewidth]{Images/chapter_6/6_5/6_5_2/test_roc.pdf}
\end{minipage}\hfill
\begin{minipage}{.49\textwidth}\centering
\includegraphics[width=\linewidth]{Images/chapter_6/6_5/6_5_2/test_precision_recall.pdf}
\end{minipage}
\caption{Đường ROC và Precision--Recall trên test; chấm đánh dấu là ngưỡng 0,5 đã khóa}\label{fig:test-curves}
\end{figure}

Mẫu bỏ sót FN chiếm '''+pct(metric['false_negative_rate'])+r'''\% số mẫu dương, trong khi FP chiếm '''+pct(metric['false_positive_rate'])+r'''\% số mẫu âm. Tỷ lệ lớp dương trên test là '''+pct(metric['positive_prevalence'])+r'''\%, được vẽ làm đường tham chiếu trên biểu đồ PR. Các đường cong mô tả sự đánh đổi nếu thay ngưỡng; bài làm giữ nguyên điểm vận hành đã chọn trước khi xem test.
''')
    examples=pd.read_csv(result/'error_examples.csv')
    lines=[r'\begin{table}[htbp]',r'\centering\scriptsize',r'\caption{Một số bản ghi dự đoán sai được chọn theo score}\label{tab:test-error-examples}',
           r'\begin{tabular}{lrrrrr}',r'\toprule',r'Dòng test & Loại & Tuổi & education-num & Giờ/tuần & Score >50K \\',r'\midrule']
    for _,r in examples.iterrows():
        lines.append(str(int(r.source_row)+1)+' & '+r.outcome+' & '+str(int(r.age))+' & '+str(int(r['education-num']))+' & '+str(int(r['hours-per-week']))+' & '+dec(r.score)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    profile=pd.read_csv(result/'outcome_numeric_profiles.csv').set_index('outcome')
    profiles=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Trung vị một số thuộc tính theo nhóm kết quả test}\label{tab:test-error-profiles}',
              r'\begin{tabular}{lrrrr}',r'\toprule',r'Nhóm & Số mẫu & Tuổi & education-num & Giờ/tuần \\',r'\midrule']
    for name in ['TN','FP','FN','TP']:
        r=profile.loc[name]
        profiles.append(name+' & '+n(r['rows'])+' & '+' & '.join(f'{r[k]:g}'.replace('.',',') for k in ['age_median','education-num_median','hours-per-week_median'])+r' \\')
    profiles += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write(c6,'6_6_',r'''\section{Phân tích các trường hợp dự đoán sai}
\label{sec:6_6}
Mô hình dự đoán sai '''+n(count['error_rows'])+r''' mẫu, tương đương '''+pct(metric['error_rate'])+r'''\% tập test, gồm '''+n(count['fp'])+r''' FP và '''+n(count['fn'])+r''' FN. FP là nhãn thật \texttt{<=50K} nhưng dự đoán \texttt{>50K}; FN là trường hợp ngược lại. Toàn bộ các bản ghi sai và 14 thuộc tính gốc được lưu trong \texttt{results/step\_06/test\_errors.csv}.

Để xem trường hợp cụ thể, chọn tối đa năm FP có score lớn nhất và năm FN có score nhỏ nhất, phá hòa bằng chỉ số dòng. Đây là các ví dụ được chọn có chủ đích theo score, không đại diện ngẫu nhiên cho toàn bộ lỗi. Cột ``Dòng test'' đếm từ 1 trong dữ liệu đã parse, không tính dòng chú thích và dòng trống của tệp gốc.

'''+ '\n'.join(lines)+'\n'+ '\n'.join(profiles)+r'''
Các trung vị chỉ mô tả nhóm kết quả, không chứng minh một thuộc tính gây ra lỗi và không thay thế phân tích tầm quan trọng đặc trưng. Score từ MLP chưa được kiểm định hiệu chỉnh xác suất, nên một giá trị lớn/nhỏ không được diễn giải thành mức chắc chắn đã được đảm bảo. Không thay đổi mô hình, tiền xử lý hoặc ngưỡng dựa trên các ví dụ test này.
''')
    cv=pd.read_csv(result.parent/'step_05/cv_summary.csv').set_index('candidate_id').loc[data['winner_id']]
    val=pd.read_csv(result.parent/'step_05/validation_summary.csv').set_index('candidate_id').loc[data['winner_id']]
    write(c6,'6_7_',r'''\section{Nhận xét tổng hợp về các kết quả}
\label{sec:6_7}
Trong phạm vi tìm kiếm đã thực hiện, MLP 32 nút được khóa bằng F1 CV trung bình '''+dec(cv.f1_mean)+r''' với SD '''+dec(cv.f1_std)+r'''. F1 validation là '''+dec(val.f1)+r''' và F1 test là '''+dec(metric['f1'])+r'''. Đây là ba giai đoạn có vai trò khác nhau; không chọn giá trị lớn nhất giữa chúng để gọi là kết quả cuối. Test đạt accuracy '''+pct(metric['accuracy'])+r'''\% và ROC-AUC '''+dec(metric['roc_auc'])+r''', trong khi recall lớp thu nhập cao '''+pct(metric['recall'])+r'''\% vẫn là hạn chế tại ngưỡng 0,5.

Sau khi đã khóa và đánh giá, đối chiếu 14 thuộc tính đầu vào phát hiện '''+n(overlap['test_rows_matching_train_profile'])+r''' dòng test có hồ sơ trùng train và '''+n(overlap['test_rows_matching_validation_profile'])+r''' dòng test có hồ sơ trùng validation. Nhãn income không tham gia tạo khóa nhóm. Việc trùng thuộc tính không khẳng định chắc chắn cùng một người thực. Toàn bộ 16.281 mẫu test UCI vẫn được giữ trong phép đánh giá; không loại các dòng trùng sau khi thấy kết quả và không báo cáo tập test như đã được bảo đảm tách biệt tuyệt đối theo hồ sơ đầu vào.

Các kết quả dựa trên lưới siêu tham số giới hạn, ba fold và một seed; tìm kiếm MLP được mở rộng sau pilot do cảnh báo tối ưu và đã được ghi lại. Chưa có kiểm định nhiều seed, khoảng tin cậy, đánh giá hiệu chỉnh xác suất hoặc bảo đảm tổng quát hóa sang quần thể khác. Tập test đã được sử dụng ở bước 6 nên không tiếp tục dùng nó để tinh chỉnh cấu hình rồi trình bày như dữ liệu đánh giá chưa thấy.

Bước 6 hoàn tất phần đánh giá mô hình đã khóa. Chương kết luận, tóm tắt, kiểm tra trình bày và các thông tin phân công/minh chứng thực tế của nhóm được hoàn thiện ở bước 7.
''')
    theory=report/'chapters/chapter_2/2_6_cac_chi_so_danh_gia_mo_hinh.tex'
    text=theory.read_text(encoding='utf-8')
    if 'sec:metric-specificity-ap' not in text:
        text+=r'''
\paragraph{Các chỉ số bổ sung khi đánh giá test.}\label{sec:metric-specificity-ap}
Specificity $=TN/(TN+FP)$, FPR $=FP/(TN+FP)$ và FNR $=FN/(TP+FN)$ lần lượt mô tả khả năng nhận diện lớp âm, tỷ lệ báo dương sai và tỷ lệ bỏ sót lớp dương. Balanced accuracy là trung bình recall của hai lớp. Average precision (AP) tóm tắt đường Precision--Recall bằng tổng precision có trọng số theo phần tăng recall; khác cách lấy diện tích hình thang~\cite{sklearn-ap}. Các chỉ số này được tính từ dự đoán test đã khóa, không dùng làm tiêu chí chọn lại mô hình.
'''
        theory.write_text(text,encoding='utf-8')
    write('backmatter/appendix_a','A_1_',r'''\section{Chạy toàn bộ notebook trên Google Colab}
\label{sec:A_1}
Tải \texttt{notebooks/Adult\_BTL.ipynb} lên Colab rồi chọn Runtime $\rightarrow$ Run all. Notebook tự tải ZIP UCI, tạo module và chạy lần lượt các bước 1--6, không mount Drive. Bước 5 thực hiện CV và khóa lựa chọn trước khi bước 6 đánh giá test. Trong gói hiện tại, snapshot đánh giá là bản sao đúng pipeline bước 5; trên môi trường Colab mới, notebook huấn luyện lại theo giao thức trước khi tạo snapshot mới nên kết quả có thể khác theo phiên bản thư viện.

Kết quả test nằm trong \texttt{results/step\_06/}; snapshot và SHA-256 nằm trong \texttt{artifacts/final/}. Chạy lại cùng dữ liệu và snapshot sử dụng kết quả đã xác minh. Nếu fingerprint khác, notebook từ chối tự thay kết quả đánh giá. Không xóa khóa để dùng lại test như tập chọn cấu hình. Nguồn báo cáo ở \texttt{reports/latex/}; cuối notebook tạo ZIP LaTeX bước 6. Sao lưu sửa tay trước Run all vì notebook dựng lại nguồn.
''')
    write('backmatter/appendix_b','B_2_',r'''\section{Tiến độ thực hiện của nhóm}
\label{sec:B_2}
Đến bước 6, đã chuẩn bị dữ liệu, EDA, phân chia theo nhóm, xuất đặc trưng, CV, khóa mô hình và đánh giá test. Có bảng chỉ số, ma trận nhầm lẫn, ROC/PR, dự đoán từng mẫu và phân tích lỗi. Bước 7 còn hoàn thiện báo cáo và gói nộp. Phân công, tỷ lệ đóng góp và minh chứng hoạt động phải được điền theo công việc thực tế của các thành viên; không tự suy ra từ kết quả chạy mã.
''')
    refs=report/'backmatter/references.tex';text=refs.read_text(encoding='utf-8');new=[]
    for key,title,url in [
        ('sklearn-cm','Confusion matrix','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html'),
        ('sklearn-ap','Average precision','https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html')]:
        if '\\bibitem{'+key+'}' not in text:new.append('\\bibitem{'+key+'} scikit-learn developers. '+title+'. \\url{'+url+'}.')
    refs.write_text(text.replace(r'\end{thebibliography}','\n'.join(new)+'\n'+r'\end{thebibliography}'),encoding='utf-8')
