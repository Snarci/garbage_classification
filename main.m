%% inizializzazione hyps
onGpu = 1;
n_clusters = 500;
max_iter_k_means = 250;
need_to_extract_features = 1;
number_of_bins = 150;
%[200, 400, 600]
%[50,100,150,200]
%% creazione del ds e splits
ds_path="garbage_classification";
[train_set,val_set,test_set] = split_ds(ds_path);
[train_set2,val_set2,test_set2] = split_ds(ds_path);
%per velocizzare
%[train_set,~]=splitEachLabel(train_set,0.02);
%[train_set2,swdwd]=splitEachLabel(train_set2,0.02);
%% estrazione keypoints e features per il train set
%https://www.mathworks.com/help/vision/feature-detection-and-extraction.html
if need_to_extract_features
   train_table2 = extraction_LBP(train_set2);
   train_table = features_extraction(train_set,"SIFT");
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


%%
u_filenames = train_set.Files;
if onGpu
    idx = gather(idx);
end

%% Hists Extraction

train_hist_construction_table = table(train_table.train_reference,train_table.train_labels,idx);
train_table_hist = hists_extraction(train_hist_construction_table,u_filenames,number_of_bins);


%% TF-IDF

hists = tf_idf(train_table_hist.array_hists,number_of_bins);

train_table_hist_TF_IDF=table(train_table_hist.array_hists_names,train_table_hist.array_hists_classes,hists);

%% FINAL-TABLE

concatenated_table= train_table_hist_TF_IDF;

concatenated_table.Var3=horzcat(concatenated_table.Var3,train_table2.train_features);
