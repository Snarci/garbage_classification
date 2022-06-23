function [hists] = tf_idf(hists, number_of_words)
%TF_IDF Summary of this function goes here
for i=1:size(hists,1)
    for j=1:number_of_words
        hists(i,j) = (hists(i,j)/sum(hists(i,:)))*log(size(hists,1)/(sum(hists(:,j)>0)));
    end
end
end

