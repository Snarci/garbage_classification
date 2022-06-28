%% With SIFT Without LBP to train the model
load('C:\Users\massi\Documents\GitHub\garbage_classification\computate_di_notte\test\table_n_clusters_400_number_of_bins_100_n_top_500.mat');
load('C:\Users\massi\Documents\GitHub\garbage_classification\computate_di_notte\train\table_n_clusters_400_number_of_bins_100_n_top_500.mat');

%% Test_model imported

%Use train_table_hist_TF_IDF to train the model

load('C:\Users\massi\Documents\GitHub\garbage_classification\computate_di_notte\model\trainedModelSift.mat')

yfit = trainedModelSift.predictFcn(test_table_hist_TF_IDF);

label = test_table_hist_TF_IDF.Var2;

cm_Sift = confusionmat(label,yfit);

stats_Sift = computeStats(cm_Sift);

%% Adding LBP
load('C:\Users\massi\Documents\GitHub\garbage_classification\computate_di_notte\LBP\LBP_train.mat');
load('C:\Users\massi\Documents\GitHub\garbage_classification\computate_di_notte\LBP\LBP_test.mat');

concatenated_table_trainSet= train_table_hist_TF_IDF;
concatenated_table_trainSet.Var3=horzcat(train_table_hist_TF_IDF.Var3,train_table2.train_features);
concatenated_table_testSet= test_table_hist_TF_IDF;
concatenated_table_testSet.Var3=horzcat(test_table_hist_TF_IDF.Var3,test_table2.train_features);

%% Test_model imported

%Use concatenated_table_trainSet to train the model

load('C:\Users\massi\Documents\GitHub\garbage_classification\computate_di_notte\model\trainedModelSift_LBP.mat')

yfit2 = trainedModelSift_LBP.predictFcn(concatenated_table_testSet);

label = concatenated_table_testSet.Var2;

cm_Sift_LBP = confusionmat(label,yfit2);

stats_Sift_LBP = computeStats(cm_Sift_LBP);

%% Latex Results

Name = stats_Sift.name;
Classes = stats_Sift.classes;
Macro = stats_Sift.macroAVG;
Micro = stats_Sift.microAVG;

Final = horzcat(Name,Classes,Macro,Micro);

latex_table = latex(str2sym(Final))

Name2 = stats_Sift_LBP.name;
Classes2 = stats_Sift_LBP.classes;
Macro2 = stats_Sift_LBP.macroAVG;
Micro2 = stats_Sift_LBP.microAVG;

Final2 = horzcat(Name2,Classes2,Macro2,Micro2);

latex_table2 = latex(str2sym(Final))

