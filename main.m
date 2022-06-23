%% inizializzazione hyps
onGpu = 0;
n_clusters = 125;
max_iter_k_means = 1000;
need_to_extract_features = 1;

%% creazione del ds e splits
ds_path="garbage_classification";
[train_set,val_set,test_set] = split_ds(ds_path);

%% estrazione keypoints e features per il train set
%https://www.mathworks.com/help/vision/feature-detection-and-extraction.html
if need_to_extract_features
    train_features=[];
    train_labels=[];
    train_reference={};
    index=1;
    while hasdata(train_set)
        fprintf("Estrazione train features immagine: %d\n",index);
        [img,info] = read(train_set);
        [features, valid_points] = img_features_extraction_V2(img,"BRISK");
        train_features = vertcat(train_features,features);
        %devo tener conto di che immagine sono partito, classe è opzionale
        for i=1:size(features,1)
            train_labels = vertcat(train_labels,info.Label);
            train_reference = vertcat(train_reference,info.Filename);
        end
        index=index+1;
    end
    train_table=table(train_reference,train_labels,train_features);
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

%%
i=1;
j=1;
train_hist_construction_table = table(train_table.train_reference,train_table.train_labels,idx);
cont_clusters=[];
array_hists=[];
array_hists_names=[];
array_hists_classes=[];
while i ~= (size(u_filenames,1)+1) && j ~= (size(train_hist_construction_table,1)+1) %&& i~=30
    fprintf("Estrazione train hists immagine: %d\n",i);

    current_filename = string(train_hist_construction_table(j,1).Var1);
    if string(u_filenames(i)) ~= current_filename
        current_hist = hist(cont_clusters,n_clusters);
        array_hists = vertcat(array_hists,current_hist);
        %j-1 per tornare nello spazio della classe se fa missmatch scatat
        %comunque di uno e la classe è sbagliata
        array_hists_classes = vertcat(array_hists_classes,categorical(string(train_hist_construction_table(j-1,2).Var2)));
        array_hists_names = vertcat(array_hists_names,string(u_filenames(i)));
        cont_clusters=[];
        i=i+1;
    else
        cont_clusters = horzcat(cont_clusters,train_hist_construction_table(j,3).Var3);
        j=j+1;
    end
    


end
train_table_hist=table(array_hists_names,array_hists_classes,array_hists);

%% TF-IDF

hists = train_table_hist.array_hists;
for i=1:size(hists,1)
    for j=1:n_clusters
        hists(i,j) = (hists(i,j)/sum(hists(i,:)))*log(size(hists,1)/(sum(hists(:,j)>0)));
    end
end
train_table_hist_TF_IDF=table(array_hists_names,array_hists_classes,hists);
