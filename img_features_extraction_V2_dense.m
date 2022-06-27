function [features] = img_features_extraction_V2_dense(img,size_patch)
%IMG_FEATURES_EXTRACTION Summary of this function goes here

img = to_gray(img);

%points = detectSIFTFeatures(img);
[sift_arr, ~, ~] = sp_dense_sift(img,size_patch,size_patch);
features = reshape(sift_arr,[size(sift_arr,1)*size(sift_arr,2),128]);


end

