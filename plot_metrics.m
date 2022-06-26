array_precision=[];
array_accuracy=[];
array_F1=[];

for i=1:size(name_array,2)
    s=stat_array{1, i};
    array_precision(i)=s.macroAVG(5);
    array_accuracy(i)=s.macroAVG(8);
    array_F1(i)=s.macroAVG(9);
end

array_n_bins = [100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 300, 300, 300, 300, 300]'
array_n_top_Points = [50, 100, 150, 200,500, 50, 100, 150, 200,500, 50, 100, 150, 200,500]'

figure;
PrecisionBar = reshape(array_precision,[5,3]);
bar3(PrecisionBar');
set(gca,'YTickLabel',[100 200 300]);
set(gca,'XTickLabel',[50 100 150 200 500]);
xlabel('Bins'); ylabel('Top-KeyPoints');
zlabel('Precision');

figure;
AccuracyBar = reshape(array_accuracy,[5,3]);
bar3(AccuracyBar');
set(gca,'YTickLabel',[100 200 300]);
set(gca,'XTickLabel',[50 100 150 200 500]);
xlabel('Bins'); ylabel('Top-KeyPoints');
zlabel('Accuracy');

figure;
F1Bar = reshape(array_F1,[5,3]);
bar3(F1Bar');
set(gca,'YTickLabel',[100 200 300]);
set(gca,'XTickLabel',[50 100 150 200 500]);
xlabel('Bins'); ylabel('Top-KeyPoints');
zlabel('F1');