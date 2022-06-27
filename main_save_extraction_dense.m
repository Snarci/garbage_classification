
array_k=    [400];
array_bins= [200, 300];
array_top=  [64,32];

for a=1:length(array_k)

    for b=1:length(array_bins)

        for c=1:length(array_top)
            onGpu = 1;
            need_to_extract_features = 1;
            max_iter_k_means = 250;

            n_clusters = array_k(a);
            n_top=array_top(c);
            number_of_bins = array_bins(b);

            extract_and_save_dense(onGpu,n_clusters,max_iter_k_means,need_to_extract_features,number_of_bins,n_top)

        end
    end
end