clear, close all, clc
%https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

%% read in data and information

% folder names which will also be used as classification labels
labels = {'COVID', 'Normal'};
path = 'covid/COVID-19_Radiography_Dataset/';

num_cov = 3590;
num_nor = 10100;


% initialize arrays to store the data
images = zeros(256, 256, num_cov + num_nor, 'uint8');
image_labels = cell([num_cov + num_nor, 1]);

% load COVID images and remove hidden files
cov_files = dir([path labels{1} '/images']);
cov_files = cov_files(4:end);

for i = 1:num_cov
    temp = imread([cov_files(i).folder '/' cov_files(i).name]);
    images(:,:,i) = imresize(temp, [256 256]);
    image_labels{i} = labels{1}; 
end


% load normal images and remove hidden files
nor_files = dir([path labels{2} '/images']);
nor_files = nor_files(4:end);

for i = num_cov+1:num_cov+num_nor 
    temp = imread([nor_files(i - num_cov).folder '/' nor_files(i - num_cov).name]); % Adjust the index
    images(:,:,i) = imresize(temp, [256 256]);
    image_labels{i} = labels{2}; 
end

%% randomize data

rng(13)

% Randomize the order of images and labels
perm = randperm(num_cov + num_nor);
images = images(:, :, perm);
image_labels = image_labels(perm);


%% split train and test data

train_percentage = 0.8;

% Split the data into training and testing sets
num_train = round(train_percentage * (num_cov + num_nor));
train_images = reshape(images(:, :, 1:num_train), [256 256 1 num_train]);
train_labels = categorical(image_labels(1:num_train));

val_images = images(:, :, num_train+1:end);
val_images = reshape(val_images, [256 256 1 size(val_images, 3)]);
val_labels = categorical(image_labels(num_train+1:end));


%% create network and parameters

% define the CNN architecture
layers = [
    imageInputLayer([256 256 1]) 
    convolution2dLayer(3, 16, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(3, 32, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(2) % Number of classes 
    softmaxLayer
    classificationLayer
];


% training options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', {val_images, val_labels}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress');


% Train the network
net = trainNetwork(train_images, train_labels, layers, options);


%% save so that i dont have to spend another 2 hours waiting
save('trained_network.mat', 'net');

%% presentation slide 2 - a look at the data

images_to_show = randperm(num_train, 4); % indicies: 1858,10517,6150,4853

figure
for i = 1:4

    subplot(2,2,i), imshow(train_images(:,:,:,images_to_show(i)))
    title(train_labels(images_to_show(i)), 'FontSize', 18)

end


%% presentation slide 2 - a look at the network

lgraph = layerGraph(layers);
plot(lgraph)

%% classify images model hasnt seen before - slide 7

test_files = dir([path 'test']);
test_files = test_files(4:end);

figure('Position', [10, 200, 1300, 700]),

for i = 1:3

    % take first 3 unseen covid images
    im = imread([test_files(i).folder '/' test_files(i).name]);
    im = imresize(im, [256 256]);
    subplot(3,2,i)

    prob = predict(net, im);
    pred = classify(net,im);

    title_text = sprintf('Truth: COVID\nClassification: %.2f%% %s', max(prob)*100, pred);
    imshow(im, [], 'InitialMagnification',150), title(title_text)


    % take last 3 unseen normal image
    im = imread([test_files(end-i+1).folder '/' test_files(end-i+1).name]);
    im = imresize(im, [256 256]);
    subplot(3,2,6-i+1)

    prob = predict(net, im);
    pred = classify(net,im);

    title_text = sprintf('Truth: Normal\nClassification: %.2f%% %s', max(prob)*100, pred);
    imshow(im, [], 'InitialMagnification',150), title(title_text)

end



%% accuracy on all 118 images - slide 7

test_files = dir([path 'test']);
test_files = test_files(3:end);

truth_labels = cell(numel(test_files), 1);
predicted_labels = cell(numel(test_files), 1);

% read each unseen image in one by one and get all predictions
for i = 1:numel(test_files)

    temp = imread([test_files(i).folder '/' test_files(i).name]);
    im = imresize(temp, [256 256]);

    % use file names to find the truth
    truth_labels{i} = regexp(test_files(i).name, '.*(?=-)', 'match', 'once'); 

    predictions = predict(net, im);
    [~, predicted_class] = max(predictions);
    predicted_labels{i} = labels{predicted_class};

end

accuracy = sum(strcmp(truth_labels, predicted_labels)) / numel(test_files);












