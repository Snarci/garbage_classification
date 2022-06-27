function [train_table] = features_extraction(train_set,size_patch)
%FEATURES_EXTRACTION Summary of this function goes here
train_features=[];
train_labels=[];
train_reference={};  
index=1;
while hasdata(train_set)
    fprintf("Estrazione train features immagine: %d\n",index);
    [img,info] = read(train_set);
    [features] = img_features_extraction_V2_dense(img,size_patch);
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


