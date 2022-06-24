function [train_table] = extraction_LBP(train_set)
%EXTRACTION_LBP Summary of this function goes here
train_features=[];
train_labels=[];
train_reference={};  
index=1;
while hasdata(train_set)
    fprintf("Estrazione train features immagine: %d\n",index);
    [img,info] = read(train_set);
    % img = imadjust(to_gray(img));
    features = extractLBPFeatures(to_gray(img));
    train_features = vertcat(train_features,features);
    for i=1:size(features,1)
        train_labels = vertcat(train_labels,info.Label);
        train_reference = vertcat(train_reference,info.Filename);
    end
    index=index+1;
end
    train_table=table(train_reference,train_labels,train_features);
end

