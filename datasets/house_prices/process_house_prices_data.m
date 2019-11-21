function [output_train, output_factor, input_train, input_test, feature_names] = process_house_prices_data()
data_train = readcell([pwd, '/datasets/house_prices/train.csv']);
data_test = readcell([pwd, '/datasets/house_prices/test.csv']);
data =[data_train(:, 2:size(data_train,2)-1); data_test(2:size(data_test,1), 2:size(data_test,2))];
output_factor = 1;
output_train = cell2mat(data_train(2:size(data_train,1), size(data_train,2)))'/output_factor;
num_train = size(data_train, 1)-1;
num_test = size(data_test, 1)-1;
%data_train = [data_train;data_test]
m = size(data, 1)-1;
n = size(data,2)-1;
% Separate categorical features from numerical features
type_col = NaN*zeros(n,1);

for k = 1:length(type_col)
    if isa(data{2,k}, 'double')
        type_col(k) = 1;
    else
        type_col(k) = 0;
    end
end

data_numeric = NaN*zeros(m, sum(type_col));
data_categorical = cell(m, sum(type_col==0));
names_numeric = cell(1,size(data_numeric,2));
names_categorical = cell(1,size(data_categorical,2));
j_num = 1; j_cat = 1;

for k = 1:length(type_col)
    if type_col(k) == 1
        names_numeric{j_num} = data{1,k};
        for i = 1:m
            if strcmp(data{i+1,k},'NA')==0
                data_numeric(i,j_num) = cell2mat(data(i+1,k));
            else
                data_numeric(i,j_num) = 0;
            end
        end
        j_num = j_num + 1;
    else
        names_categorical{j_cat} = data{1,k};
        if strcmp('MSZoning',names_categorical{j_cat})
            for i= 1:m
                if strcmp(data{i,k},'C (all)')
                    data_categorical{i,j_cat} = 'C';
                else
                    data_categorical{i,j_cat} = data{i,k};
                end
            end
        elseif strcmp('HouseStyle',names_categorical{j_cat})
            for i = 1:m
                data_categorical{i,j_cat} = char(join(split(data{i,k},'.'),'_'));
            end
        elseif strcmp('RoofMatl',names_categorical{j_cat})
            for i = 1:m
                data_categorical{i,j_cat} = char(join(split(data{i,k},'&'),'_'));
            end
        elseif strcmp('Exterior1st',names_categorical{j_cat})||strcmp('Exterior2nd',names_categorical{j_cat})
            for i = 1:m
                data_categorical{i,j_cat} = char(join(split(data{i,k},' '),'_'));
            end
        else
            data_categorical(:,j_cat) = data(2:m+1,k);
        end
        j_cat = j_cat + 1;
    end
end

% put all the numeric input between -1 and 1
normalize=true;
if normalize==true
    for col = 1:size(data_numeric,2)
        min_val = min(data_numeric(:, col));
        max_val = max(data_numeric(:, col));
        data_numeric(:, col) = 2*(data_numeric(:, col) - min_val)/(max_val-min_val)-1;
    end
end
% one hot encoding of categorical variables
data_categorical = cell2table(data_categorical);
data_categorical.Properties.VariableNames = names_categorical;
utils = UtilitiesIDL;

for name = 1:length(names_categorical)
    data_categorical = utils.createOneHotEncoding(data_categorical,names_categorical{name});
end

names_categorical = data_categorical.Properties.VariableNames;
feature_names = [names_numeric,names_categorical];
data = [data_numeric,table2array(data_categorical)];
input_train = data(1:num_train,:)';
input_test = data(num_train+1:num_train+num_test,:)';
end