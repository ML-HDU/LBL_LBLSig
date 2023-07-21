function CIFAR10_my_formatting(data_path)


train_label = [];
train_data = [];

for k = 1 : 5
    load(fullfile(data_path,['data_batch_' num2str(k) '.mat']));

    display(batch_label);

    train_data = [train_data;data];

    train_label = [train_label;labels];

end

clear batch_label data k labels;

load(fullfile(data_path,['test_batch' '.mat']));
test_data = data;
test_label = labels;

clear batch_label data k labels;

P = double(train_data)'/255;
T = double(train_label);

TV_P = double(test_data)'/255;
TV_T = double(test_label);

clear train_* test_*;

save(fullfile( '.' , 'train_data.mat') , 'P');
save(fullfile( '.' , 'train_label.mat'),'T');
save(fullfile( '.' , 'test_data.mat'),'TV_P');
save(fullfile( '.' , 'test_label.mat'),'TV_T');

clear P T TV_*;





end
