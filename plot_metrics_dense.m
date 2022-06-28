array_precision=[];
array_accuracy=[];
array_F1=[];

for i=1:size(name_array,2)
    s=stat_array{1, i};
    array_precision(i)=s.macroAVG(5);
    array_accuracy(i)=s.macroAVG(8);
    array_F1(i)=s.macroAVG(9);
end

figure;
PrecisionBar = reshape(array_precision,[2,3]);
bar3(PrecisionBar');
set(gca,'YTickLabel',[100 200 300]);
set(gca,'XTickLabel',[32 64]);
xlabel('Bins'); ylabel('Grid-size');
zlabel('Precision');

figure;
AccuracyBar = reshape(array_accuracy,[2,3]);
bar3(AccuracyBar');
set(gca,'YTickLabel',[100 200 300]);
set(gca,'XTickLabel',[32 64]);
xlabel('Bins'); ylabel('Grid-size');
zlabel('Accuracy');

figure;
F1Bar = reshape(array_F1,[2,3]);
bar3(F1Bar');
set(gca,'YTickLabel',[100 200 300]);
set(gca,'XTickLabel',[32 64]);
xlabel('Bins'); ylabel('Grid-size');
zlabel('F1');