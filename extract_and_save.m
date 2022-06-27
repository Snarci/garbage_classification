function [] = extract_and_save(onGpu,n_clusters,max_iter_k_means,need_to_extract_features,number_of_bins,n_top)

%creazione del ds e splits
ds_path="garbage_classification";
[train_set,test_set] = split_ds8020(ds_path);
%per velocizzare
%[train_set,~]=splitEachLabel(train_set,0.02);
%[test_set,~]=splitEachLabel(test_set,0.02);
% estrazione keypoints e features per il train set
%https://www.mathworks.com/help/vision/feature-detection-and-extraction.html
if need_to_extract_features
   train_table = features_extraction(train_set,"SIFT",n_top);
   test_table = features_extraction(test_set,"SIFT",n_top);
end
% creazione vocabolario
train_features=train_table.train_features;
test_features=test_table.train_features;
tot_features=vertcat(train_features,test_features);
tic; % Start stopwatch timer
if onGpu
    gpuDevice(1)
    tot_featuresGpu=gpuArray(tot_features);

    stream = RandStream('mlfg6331_64');  % Random number stream
    options = statset('UseParallel',1,'UseSubstreams',1,...
        'Streams',stream);

    [idx,C,sumd,D] = kmeans(tot_featuresGpu,n_clusters,'Options',options,'MaxIter',...
        max_iter_k_means,'Display','final');
else

    [idx,C,sumd,D] = kmeans(tot_features,n_clusters,'MaxIter',...
        max_iter_k_means,'Display','final');
end
toc % Terminate stopwatch timer


%
u_filenames_train = train_set.Files;
u_filenames_test = test_set.Files;
if onGpu
    idx = gather(idx);
end
idx_train=idx(1:size(train_features));
idx_test=idx((size(train_features)+1):size(idx,1));
% Hists Extraction
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
endpath=strcat("n_clusters_",num2str(n_clusters),"_number_of_bins_",num2str(number_of_bins),"_n_top_",num2str(n_top));
path_train=strcat("E:\CV\garbage_classification/computate_di_notte/train/table_",endpath);
path_test=strcat("E:\CV\garbage_classification/computate_di_notte/test/table_",endpath);
save(path_train,'train_table_hist_TF_IDF');
save(path_test,'test_table_hist_TF_IDF');

end

