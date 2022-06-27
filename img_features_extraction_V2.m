function [features, valid_points] = img_features_extraction_V2(img,point_extractor_and_descript,n_top)
%IMG_FEATURES_EXTRACTION Summary of this function goes here

img = to_gray(img);
switch point_extractor_and_descript
    case "SIFT"
        points = detectSIFTFeatures(img);
    case "SURF"
        points = detectSURFFeatures(img);
    case "ORB"
        points = detectORBFeatures(img);
    case "BRISK"
        points = detectBRISKFeatures(img);
    case "FREAK"
        points = detectFREAKFeatures(img);
    case "KAZE"
        points = detectKAZEFeatures(img);
end
%points = detectSIFTFeatures(img);
points=selectStrongest(points,n_top);
[features, valid_points] = extractFeatures(to_gray(img), points,"Method",point_extractor_and_descript);

if isa(features,'binaryFeatures')
    features=features.Features;
end
end

