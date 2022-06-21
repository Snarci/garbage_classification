function [train_set,val_set,test_set] = split_ds(ds_path)
%SPLIT_DS Divide il dataset in porzioni
imds = imageDatastore(ds_path,"IncludeSubfolders",true,"LabelSource","foldernames");
seed = 42;
rng(seed)
imds = shuffle(imds);
[train_set,tmp] = splitEachLabel(imds,0.6);
[val_set,test_set] = splitEachLabel(tmp,0.5);

end

