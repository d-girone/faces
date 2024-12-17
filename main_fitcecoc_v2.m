% Classification into several classes with HPC improvements
% This script trains a facial recognition model using parallelization.
delete(gcp('nocreate'))

targetSize = [128,128];
k = 30;                                  % Number of features to consider
location = fullfile("C:\Users\dylan\Downloads\lfw");

% Initialize Parallel Pool
disp('Initializing parallel pool...');
parpool;

disp('Creating image datastore...');
imds0 = imageDatastore(location, 'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames', ...
    'ReadFcn', @(filename) imresize(im2gray(imread(filename)), targetSize));

% Select persons with 10 to 40 images
disp('Creating subset of several persons...');
tbl = countEachLabel(imds0);
mask = tbl{:,2} >= 8 & tbl{:,2} <= 40;
disp(['Number of images: ', num2str(sum(tbl{mask,2}))]);
persons = unique(tbl{mask,1});

[lia, locb] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

% Display the total number of people the model is trained to recognize
numPersons = numel(persons);
disp(['The model is trained to recognize ', num2str(numPersons), ' people.']);

% Display montage of images
t = tiledlayout('flow');
nexttile(t);
montage(imds);

% Parallelized Image Reading
disp('Reading all images in parallel...');
numImages = numel(imds.Files);
B = zeros(prod(targetSize), numImages, 'single');

tic;
parfor i = 1:numImages
    img = imresize(im2gray(imread(imds.Files{i})), targetSize);
    B(:, i) = single(img(:)) ./ 256; % Normalize as we read
end
toc;

disp('Normalizing data...');
[B, C, SD] = normalize(B);

% Perform SVD in Parallel (on CPU)
disp('Performing SVD...');
tic;
[U, S, V] = svd(B, 'econ'); % SVD operation
toc;

% Get eigenfaces and display top 16
disp('Generating eigenfaces...');
Eigenfaces = arrayfun(@(j) reshape((U(:,j) - min(U(:,j))) ./ (max(U(:,j)) - min(U(:,j))), targetSize), ...
    1:size(U,2), 'uni', false);

nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

% Keep first K features
k = min(size(V,2), k);
W = S * V'; % Transform V to weights
W = W(1:k, :); % Keep first K weights
U = U(:, 1:k); % Keep K eigenfaces

% Create feature vectors and labels
X = W';
Y = categorical(imds.Labels, persons);

% Assign colors for visualization
cm = [1,0,0; 0,0,1; 0,1,0];
c = cm(1 + mod(uint8(Y), size(cm,1)), :);

disp('Training Support Vector Machine...');
options = statset('UseParallel', true); % Enable parallel SVM training
tic;

Mdl = fitcecoc(X, Y,'Verbose', 2,'Learners','svm',...
               'Options',options);

% Mdl = fitcecoc(X, Y, 'Verbose', 2, 'Learners', 'svm', ...
%                'Options', options, ...
%                'OptimizeHyperparameters', 'all', ...
%                'HyperparameterOptimizationOptions', struct('UseParallel', true));
toc;

% Plot feature space using top predictors
disp('Generating feature plots...');
nexttile(t);
scatter3(X(:,1), X(:,2), X(:,3), 50, c);
title('A top 3-predictor plot');
xlabel('x1'); ylabel('x2'); zlabel('x3');

nexttile(t);
scatter3(X(:,4), X(:,5), X(:,6), 50, c);
title('A next 3-predictor plot');
xlabel('x4'); ylabel('x5'); zlabel('x6');

% Prediction and performance metrics
disp('Evaluating model...');
[YPred, Score, Cost] = resubPredict(Mdl);

% Plot ROC metrics
disp('Plotting ROC metrics...');
rm = rocmetrics(imds.Labels, Score, persons);
nexttile(t);
plot(rm);

% Confusion matrix
disp('Plotting confusion matrix...');
nexttile(t);
confusionchart(Y, YPred);
title(['Number of features: ', num2str(k)]);

% Save the trained model
disp('Saving model...');
save('model', 'Mdl', 'persons', 'U', 'targetSize');

disp('Script completed successfully.');