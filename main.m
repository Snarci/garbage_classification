ds_path="garbage_classification";
%% creazione del ds e splits

[train_set,val_set,test_set] = split_ds(ds_path);

%% estrazione keypoints 
%https://www.mathworks.com/help/vision/feature-detection-and-extraction.html

[img,info] = read(train_set);
[features, valid_points] = img_features_extraction(img);
[img1,info1] = read(train_set);
[features1, valid_points1] = img_features_extraction(img1);
%plot(pp)