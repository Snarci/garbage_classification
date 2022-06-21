function [img] = to_gray(img)
%TO_GRAY Summary of this function goes here

if size(img,3)==3
    img = im2double(rgb2gray(img));
else
    img = im2double(img);
end

end

