%% inizializzazione hyps
onGpu = 1;
n_clusters = 100;
max_iter_k_means = 500;
%% creazione del ds e splits
ds_path="garbage_classification";
[train_set,val_set,test_set] = split_ds(ds_path);

%% estrazione keypoints e features
%https://www.mathworks.com/help/vision/feature-detection-and-extraction.html
train_features=[];
index=1;
while hasdata(train_set)
    fprintf("Estrazioen features immagine: %d\n",index);
    [img,info] = read(train_set);
    [features, valid_points] = img_features_extraction(img);
    train_features = vertcat(train_features,features);
    index=index+1;
end
%% creazione vocabolario

tic; % Start stopwatch timer
if onGpu
    device=gpuDevice(1);
    train_featuresGpu=gpuArray(train_features);
    [idx,C,sumd,D] = kmeans(train_featuresGpu,n_clusters,'MaxIter',...
        max_iter_k_means,'Display','final');
else
    [idx,C,sumd,D] = kmeans(train_features,n_clusters,'MaxIter',...
        max_iter_k_means,'Display','final');
end
toc % Terminate stopwatch timer
%da fare TFIDF per normalizzare i dati e parole comuni quindi prima si
%calcola l'histogramma https://www.youtube.com/watch?v=a4cFONdc6nc&ab_channel=CyrillStachniss
%consigliato anche usare la cosine distance

