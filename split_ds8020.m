function [train_set,test_set] = split_ds8020(ds_path)
%SPLIT_DS Divide il dataset in porzioni
imds = imageDatastore(ds_path,"IncludeSubfolders",true,"LabelSource","foldernames");
seed = 42;
rng(seed)
imds = shuffle(imds);
[train_set,test_set] = splitEachLabel(imds,0.8);
end

