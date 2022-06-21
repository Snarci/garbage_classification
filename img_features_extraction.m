function [features, valid_points] = img_features_extraction(img)
%IMG_FEATURES_EXTRACTION Summary of this function goes here

points = extract_keypoints_from_image(img);
show_top_keypoints(img, points, 10);

[features, valid_points] = extractFeatures(to_gray(img), points,"Method","SIFT");

end

