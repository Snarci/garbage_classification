I=imread("cameraman.tif");
[sift_arr, grid_x, grid_y] = sp_dense_sift(I,8,8);