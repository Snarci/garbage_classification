function [points] = extract_keypoints_from_image(img)
%EXTRACT_TOP_KEYPOINTS Summary of this function goes here

%nel caso ci siano immagini non rgb le portiamo in grayscale
img = to_gray(img);

points = detectSIFTFeatures(img);

end

