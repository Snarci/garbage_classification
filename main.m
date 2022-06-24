%% inizializzazione hyps
onGpu = 0;
n_clusters = 125;
max_iter_k_means = 500;
need_to_extract_features = 1;
number_of_bins = 100;
%% creazione del ds e splits
ds_path="garbage_classification";
[train_set,test_set] = split_ds_80_20(ds_path);
[train_set2,test_set2] = split_ds_80_20(ds_path);

%% estrazione keypoints e features per il train set
%https://www.mathworks.com/help/vision/feature-detection-and-extraction.html
if need_to_extract_features
   train_table2 = extraction_LBP(train_set2);
   test_table2 = extraction_LBP(test_set2);
   train_table = features_extraction(train_set,"SIFT");
   test_table = features_extraction(test_set,"SIFT");
end
%% creazione vocabolario
train_features=train_table.train_features;
tic; % Start stopwatch timer
if onGpu
    gpuDevice(1)
    train_featuresGpu=gpuArray(train_features);
    
    stream = RandStream('mlfg6331_64');  % Random number stream
    options = statset('UseParallel',1,'UseSubstreams',1,...
    'Streams',stream);
    
    [idx,C,sumd,D] = kmeans(train_featuresGpu,n_clusters,'Options',options,'MaxIter',...
        max_iter_k_means,'Display','final');
else
    stream = RandStream('mlfg6331_64');  % Random number stream
    options = statset('UseParallel',1,'UseSubstreams',1,...
    'Streams',stream);
    [idx,C,sumd,D] = kmeans(train_features,n_clusters,'Options',options,'MaxIter',...
        max_iter_k_means,'Display','final');
end
toc % Terminate stopwatch timer
%da fare TFIDF per normalizzare i dati e parole comuni quindi prima si
%calcola l'histogramma https://www.youtube.com/watch?v=a4cFONdc6nc&ab_channel=CyrillStachniss
%consigliato anche usare la cosine distance

%% 
u_filenames = train_set.Files;
if onGpu
    idx = gather(idx);
end

%% Hists Extraction

idx_train=idx(1:size(train_features));
idx_test=idx((size(train_features)+1):size(idx,1));
%train
train_hist_construction_table = table(train_table.train_reference,train_table.train_labels,idx_train);
train_table_hist = hists_extraction(train_hist_construction_table,u_filenames_train,number_of_bins);
%test
test_hist_construction_table = table(test_table.train_reference,test_table.train_labels,idx_test);
test_table_hist = hists_extraction(test_hist_construction_table,u_filenames_test,number_of_bins);


% TF-IDF

hists_train = tf_idf(train_table_hist.array_hists,number_of_bins);
train_table_hist_TF_IDF=table(train_table_hist.array_hists_names,train_table_hist.array_hists_classes,hists_train);

hists_test = tf_idf(test_table_hist.array_hists,number_of_bins);
test_table_hist_TF_IDF=table(test_table_hist.array_hists_names,test_table_hist.array_hists_classes,hists_test);

%% Test_Model_without_LBP
% use train_table_hist_TF_IDF to train the model with machine learning tool

yfit = trainedModel.predictFcn(test_table_hist_TF_IDF);

label = test_table_hist_TF_IDF.Var2;

cm = confusionmat(label,yfit);

stats_without = computeStats(cm);

%% FINAL-TABLE_with_LBP

concatenated_table_trainSet= train_table_hist_TF_IDF;
concatenated_table_trainSet.Var3=horzcat(train_table_hist_TF_IDF.Var3,LBP_train_set.train_features);
concatenated_table_testSet= test_table_hist_TF_IDF;
concatenated_table_testSet.Var3=horzcat(test_table_hist_TF_IDF.Var3,LBP_test_set.train_features);

%% Test_Model_whith_LBP
% Use concatenated_table_trainSet to train the model

yfit = trainedModel.predictFcn(concatenated_table_testSet);

label = concatenated_table_testSet.Var2;

cm = confusionmat(label,yfit);

stats_with_BLP = computeStats(cm);
