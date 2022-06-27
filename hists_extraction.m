function [train_table_hist] = hists_extraction(train_hist_construction_table,u_filenames,n_bins)
%HISTS_EXTRACTION Summary of this function goes here

i=1;
j=1;
cont_clusters=[];
array_hists=[];
array_hists_names=[];
array_hists_classes=[];
while i <= (size(u_filenames,1)) && j <= (size(train_hist_construction_table,1)) %&& i~=30
    fprintf("Estrazione train hists immagine: %d\n",i);

    current_filename = string(train_hist_construction_table(j,1).Var1);
    if string(u_filenames(i)) ~= current_filename
        current_hist = hist(cont_clusters,n_bins);
        array_hists = vertcat(array_hists,current_hist);
        %j-1 per tornare nello spazio della classe se fa missmatch scatat
        %comunque di uno e la classe è sbagliata
        array_hists_classes = vertcat(array_hists_classes,categorical(string(train_hist_construction_table(j-1,2).Var2)));
        array_hists_names = vertcat(array_hists_names,string(u_filenames(i)));
        cont_clusters=[];
        i=i+1;
    else
        cont_clusters = horzcat(cont_clusters,train_hist_construction_table(j,3).Var3);
        j=j+1;
    end
  if j > (size(train_hist_construction_table,1))  
        current_hist = hist(cont_clusters,n_bins);
        array_hists = vertcat(array_hists,current_hist);
        array_hists_classes = vertcat(array_hists_classes,categorical(string(train_hist_construction_table(j-1,2).Var2)));
        array_hists_names = vertcat(array_hists_names,string(u_filenames(i)));
  end


end
train_table_hist=table(array_hists_names,array_hists_classes,array_hists);

end

