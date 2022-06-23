%% inizializzazione hyps
onGpu = 0;
n_clusters = 125;
max_iter_k_means = 500;
need_to_extract_features = 1;
%% creazione del ds e splits
ds_path="garbage_classification";
[train_set,val_set,test_set] = split_ds(ds_path);

%% estrazione keypoints e features per il train set
%https://www.mathworks.com/help/vision/feature-detection-and-extraction.html
if need_to_extract_features
   train_table=features_extraction(train_set,"SIFT")
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
train_hist_construction_table = table(train_table.train_reference,train_table.train_labels,idx);
train_table_hist = hists_extraction()

%% TF-IDF

hists = tf_idf(train_table_hist.array_hists,n_clusters);

train_table_hist_TF_IDF=table(array_hists_names,array_hists_classes,hists);
