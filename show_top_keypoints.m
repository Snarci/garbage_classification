function show_top_keypoints(img, points, number_of_points)
%SHOW_TOP_KEYPOINTS Summary of this function goes here

imshow(img);
hold on;
pp=points.selectStrongest(number_of_points);
plot(pp)

%pp_points=pp.Location;
%plot(pp_points(:,1),pp_points(:,2),"+")
end

