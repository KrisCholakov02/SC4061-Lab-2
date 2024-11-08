%% 3.1 Image Segmentation

%% a) Otsu's Global Thresholding with Multiple Thresholds

% Path to the images folder
imgFolder = 'img/';

% List of images and their ground truths
imageFiles = {'document01.bmp', 'document02.bmp', 'document03.bmp', 'document04.bmp'};
gtFiles = {'document01-GT.tiff', 'document02-GT.tiff', 'document03-GT.tiff', 'document04-GT.tiff'};

% Range of thresholds to test
thresholdValues = 0:0.05:1;  % From 0 to 1 in steps of 0.05

% Process each image
for i = 1:length(imageFiles)
    % Read the image and its ground truth
    imgPath = fullfile(imgFolder, imageFiles{i});
    gtPath = fullfile(imgFolder, gtFiles{i});
    img = imread(imgPath);
    gt = imread(gtPath);

    % Convert to grayscale if necessary
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    if size(gt, 3) == 3
        gt = rgb2gray(gt);
    end

    % Ensure ground truth is binary
    if ~islogical(gt)
        gt = imbinarize(gt);
    end

    % Apply Otsu's thresholding
    otsuLevel = graythresh(img);
    binaryImgOtsu = imbinarize(img, otsuLevel);
    otsuDiff = xor(binaryImgOtsu, gt);
    otsuDiffSum = sum(otsuDiff(:));

    % Initialize variables to find the best threshold
    minDiffSum = inf;
    bestThreshold = otsuLevel;
    bestBinaryImg = binaryImgOtsu;

    % Store sum differences for each threshold
    thresholdDiffs = zeros(length(thresholdValues), 2);

    % Test each threshold value
    for j = 1:length(thresholdValues)
        t = thresholdValues(j);
        
        % Apply thresholding
        binaryImg = imbinarize(img, t);
        
        % Compute difference with ground truth
        diffImg = xor(binaryImg, gt);
        diffSum = sum(diffImg(:));

        % Store results
        thresholdDiffs(j, :) = [t, diffSum];

        % Update best threshold if necessary
        if diffSum < minDiffSum
            minDiffSum = diffSum;
            bestThreshold = t;
            bestBinaryImg = binaryImg;
        end
    end

    % Display results
    fprintf('Image: %s\n', imageFiles{i});
    fprintf('Otsu Threshold: %.2f, Difference Sum (Otsu): %d\n', otsuLevel, otsuDiffSum);
    fprintf('Best Threshold: %.2f, Difference Sum (Best): %d\n\n', bestThreshold, minDiffSum);

    % Show threshold differences as a table
    thresholdTable = array2table(thresholdDiffs, 'VariableNames', {'Threshold', 'DifferenceSum'});
    disp(thresholdTable);

    % Plot sum differences for each threshold
    figure('Name', sprintf('Threshold Differences for %s', imageFiles{i}), 'NumberTitle', 'off');
    hold on;
    plot(thresholdDiffs(:, 1), thresholdDiffs(:, 2), 'b-o');
    plot(otsuLevel, otsuDiffSum, 'ro', 'MarkerSize', 8, 'LineWidth', 1.5);
    xlabel('Threshold');
    ylabel('Sum of Differences');
    title(sprintf('Difference Sums for Thresholds - %s', imageFiles{i}));
    legend('Thresholds', 'Otsu Threshold', 'Location', 'best');
    hold off;

    % Show images
    figure('Name', sprintf('Global Thresholding Results for %s', imageFiles{i}), 'NumberTitle', 'off');
    subplot(2, 2, 1); imshow(img); title('Original');
    subplot(2, 2, 2); imshow(binaryImgOtsu); title(sprintf('Otsu Threshold (%.2f)', otsuLevel));
    subplot(2, 2, 3); imshow(bestBinaryImg); title(sprintf('Best Threshold (%.2f)', bestThreshold));
    subplot(2, 2, 4); imshow(gt); title('Ground Truth');

    % Show difference images
    figure('Name', sprintf('Difference Images for %s', imageFiles{i}), 'NumberTitle', 'off');
    subplot(1, 2, 1); imshow(otsuDiff); title('Difference (Otsu)');
    subplot(1, 2, 2); imshow(xor(bestBinaryImg, gt)); title('Difference (Best)');
end

%% b) Niblack's Local Thresholding with Bayesian Optimization

% Search ranges for k and window size
kRange = [-3.5, 3.5];
windowSizeRange = [3, 300];

% Initialize results summary
resultsSummary = struct([]);

% Process each image
for i = 1:length(imageFiles)
    % Read the image and its ground truth
    imgPath = fullfile(imgFolder, imageFiles{i});
    gtPath = fullfile(imgFolder, gtFiles{i});
    img = imread(imgPath);
    gt = imread(gtPath);

    % Convert to grayscale if necessary
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    if size(gt, 3) == 3
        gt = rgb2gray(gt);
    end

    % Ensure ground truth is binary
    if ~islogical(gt)
        gt = imbinarize(gt);
    end

    % Define the objective function for optimization
    objectiveFunction = @(params) niblackObjective(img, gt, params.k, params.windowSize);

    % Define optimization variables
    kVar = optimizableVariable('k', kRange, 'Transform', 'none');
    windowSizeVar = optimizableVariable('windowSize', windowSizeRange, 'Type', 'integer');

    % Run Bayesian optimization
    results = bayesopt(objectiveFunction, [windowSizeVar, kVar], ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 250, ...
        'IsObjectiveDeterministic', true, ...
        'ExplorationRatio', 0.4, ...
        'PlotFcn', {@plotObjectiveModel});

    % Extract best parameters
    bestK = results.XAtMinObjective.k;
    bestWindowSize = results.XAtMinObjective.windowSize;
    minDifference = results.MinObjective;

    % Generate best binary image
    bestBinaryImg = niblackThreshold(img, bestWindowSize, bestK);

    % Display results
    fprintf('Image: %s\n', imageFiles{i});
    fprintf('Best k (Niblack): %0.2f\n', bestK);
    fprintf('Best Window Size: %d\n', bestWindowSize);
    fprintf('Difference Sum (Best): %d\n\n', minDifference);

    % Show images
    figure('Name', sprintf('Best Niblack Segmentation for %s', imageFiles{i}), 'NumberTitle', 'off');
    subplot(2, 2, 1); imshow(img); title('Original');
    subplot(2, 2, 2); imshow(bestBinaryImg); title(sprintf('Best Binary (k = %0.2f, Window = %d)', bestK, bestWindowSize));
    subplot(2, 2, 3); imshow(gt); title('Ground Truth');
    subplot(2, 2, 4); imshow(xor(bestBinaryImg, gt)); title('Difference Image');

    % Store results
    resultsSummary(i).imageName = imageFiles{i};
    resultsSummary(i).bestK = bestK;
    resultsSummary(i).bestWindowSize = bestWindowSize;
    resultsSummary(i).minDifference = minDifference;
end

% Summary of best results
fprintf('\n=== Summary of Best Results ===\n');
for i = 1:length(resultsSummary)
    fprintf('Image: %s\n', resultsSummary(i).imageName);
    fprintf('Best k (Niblack): %0.2f\n', resultsSummary(i).bestK);
    fprintf('Best Window Size: %d\n', resultsSummary(i).bestWindowSize);
    fprintf('Difference Sum (Best): %d\n\n', resultsSummary(i).minDifference);
end

% Objective function for Bayesian optimization (Niblack)
function diffSum = niblackObjective(img, gt, k, windowSize)
    binaryImg = niblackThreshold(img, windowSize, k);
    diffImg = xor(binaryImg, gt);
    diffSum = sum(diffImg(:));
end

% Niblack's thresholding function
function binaryImg = niblackThreshold(img, windowSize, k)
    img = double(img);
    halfWin = floor(windowSize / 2);
    padImg = padarray(img, [halfWin, halfWin], 'symmetric');

    meanFilter = fspecial('average', windowSize);
    localMean = imfilter(padImg, meanFilter, 'replicate');
    localMean = localMean(halfWin+1:end-halfWin, halfWin+1:end-halfWin);

    localSqMean = imfilter(padImg.^2, meanFilter, 'replicate');
    localSqMean = localSqMean(halfWin+1:end-halfWin, halfWin+1:end-halfWin);

    localVariance = localSqMean - localMean.^2;
    localStdDev = sqrt(localVariance);

    threshold = localMean + k * localStdDev;
    binaryImg = img > threshold;
end

%% c.1) Bayesian Optimization for Sauvola's Local Thresholding

% Process each image
for i = 1:length(imageFiles)
    % Read the image and its ground truth
    imgPath = fullfile(imgFolder, imageFiles{i});
    gtPath = fullfile(imgFolder, gtFiles{i});
    img = imread(imgPath);
    gt = imread(gtPath);

    % Convert to grayscale if necessary
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    if size(gt, 3) == 3
        gt = rgb2gray(gt);
    end

    % Ensure ground truth is binary
    if ~islogical(gt)
        gt = imbinarize(gt);
    end

    % Define the objective function
    objectiveFunction = @(params) sauvolaObjective(img, gt, params.windowSize, params.k, params.R);

    % Define optimization variables
    windowSizeVar = optimizableVariable('windowSize', [3, 41], 'Type', 'integer');
    kVar = optimizableVariable('k', [0.1, 0.7]);
    RVar = optimizableVariable('R', [32, 192], 'Type', 'integer');

    % Run Bayesian optimization
    results = bayesopt(objectiveFunction, [windowSizeVar, kVar, RVar], ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'MaxObjectiveEvaluations', 250, ...
        'IsObjectiveDeterministic', true, ...
        'PlotFcn', {@plotObjectiveModel, @plotMinObjective});

    % Extract best parameters
    bestWindowSize = results.XAtMinObjective.windowSize;
    bestK = results.XAtMinObjective.k;
    bestR = results.XAtMinObjective.R;
    minDifference = results.MinObjective;

    % Generate best binary image
    bestBinaryImg = sauvolaThreshold(img, bestWindowSize, bestK, bestR);

    % Display results
    fprintf('Image: %s\n', imageFiles{i});
    fprintf('Best Window Size (Sauvola): %d\n', bestWindowSize);
    fprintf('Best k (Sauvola): %.2f\n', bestK);
    fprintf('Best R (Sauvola): %d\n', bestR);
    fprintf('Difference Sum (Best): %d\n\n', minDifference);

    % Show images
    figure('Name', sprintf('Best Sauvola Segmentation for %s', imageFiles{i}), 'NumberTitle', 'off');
    subplot(2, 2, 1); imshow(img); title('Original');
    subplot(2, 2, 2); imshow(bestBinaryImg); title(sprintf('Best Binary (w=%d, k=%.2f, R=%d)', bestWindowSize, bestK, bestR));
    subplot(2, 2, 3); imshow(gt); title('Ground Truth');
    subplot(2, 2, 4); imshow(xor(bestBinaryImg, gt)); title('Difference Image');
end

% Objective function for Bayesian optimization (Sauvola)
function diffSum = sauvolaObjective(img, gt, windowSize, k, R)
    binaryImg = sauvolaThreshold(img, windowSize, k, R);
    diffImg = xor(binaryImg, gt);
    diffSum = sum(diffImg(:));
end

% Sauvola's thresholding function
function binaryImg = sauvolaThreshold(img, windowSize, k, R)
    img = double(img);
    halfWin = floor(windowSize / 2);
    padImg = padarray(img, [halfWin, halfWin], 'symmetric');

    meanFilter = fspecial('average', windowSize);
    localMean = imfilter(padImg, meanFilter, 'replicate');
    localMean = localMean(halfWin+1:end-halfWin, halfWin+1:end-halfWin);

    localSqMean = imfilter(padImg.^2, meanFilter, 'replicate');
    localSqMean = localSqMean(halfWin+1:end-halfWin, halfWin+1:end-halfWin);

    localVariance = max(0, localSqMean - localMean.^2);
    localStdDev = sqrt(localVariance);
    localStdDev(localStdDev == 0) = eps;

    threshold = localMean .* (1 + k * ((localStdDev / (R + eps)) - 1));
    threshold = min(max(threshold, 0), 255);

    binaryImg = img > threshold;
end

%% c.2) Filter Bank with Smoothed Features and K-Means Clustering

% Parameters for Gabor filters and clustering
numOrientationsList = [4, 6, 8, 10];
kClusterValues = [2, 3, 4, 5, 6];
gaborWavelengths = [2, 4, 8, 12];
poolingScales = [1];

% Process each image
for i = 1:length(imageFiles)
    % Read the image and its ground truth
    imgPath = fullfile(imgFolder, imageFiles{i});
    gtPath = fullfile(imgFolder, gtFiles{i});
    img = imread(imgPath);
    gt = imread(gtPath);

    % Convert to grayscale if necessary
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    if size(gt, 3) == 3
        gt = rgb2gray(gt);
    end

    % Ensure ground truth is binary
    if ~islogical(gt)
        gt = imbinarize(gt);
    end

    % Variables to store best results
    minDifference = inf;
    bestParams = struct('numOrientations', 0, 'kClusters', 0, 'wavelength', 0, 'poolingScale', 0);
    bestSegmentedImg = [];

    % Loop over parameters
    for numOrientations = numOrientationsList
        for wavelength = gaborWavelengths
            for poolingScale = poolingScales
                fprintf('Processing %s with %d orientations, wavelength %d, pooling %.1f\n', ...
                        imageFiles{i}, numOrientations, wavelength, poolingScale);

                % Create Gabor filters
                angles = linspace(0, 180, numOrientations + 1);
                angles = angles(1:end-1);
                gaborArray = gabor(wavelength, angles);
                gaborMag = imgaborfilt(img, gaborArray);

                % Pool features with smoothing
                pooledFeatures = arrayfun(@(j) imgaussfilt(imresize(gaborMag(:,:,j), poolingScale), 1), ...
                                          1:numel(gaborArray), 'UniformOutput', false);
                pooledSize = size(pooledFeatures{1});
                featureSet = reshape(cat(3, pooledFeatures{:}), [], numel(gaborArray));

                % Add intensity and gradient magnitude
                imgResized = imresize(img, poolingScale);
                imgResized = double(imgResized);
                [Gmag, ~] = imgradient(imgResized);
                featureSet = [featureSet, imgResized(:), Gmag(:)];

                % Normalize features
                featureSet = (featureSet - mean(featureSet, 1)) ./ (std(featureSet, [], 1) + eps);

                % K-means clustering
                for kClusters = kClusterValues
                    fprintf('  Running K-means with %d clusters...\n', kClusters);

                    opts = statset('MaxIter', 500, 'UseParallel', true, 'TolFun', 1e-4);
                    clusterIdx = kmeans(featureSet, kClusters, ...
                                        'Replicates', 3, 'Options', opts, 'Start', 'plus');

                    % Reshape to image size
                    segmentedImg = reshape(clusterIdx, pooledSize);

                    % Map clusters to binary based on mean intensity
                    clusterMeans = arrayfun(@(c) mean(imgResized(segmentedImg == c)), 1:kClusters);
                    [~, textCluster] = min(clusterMeans);

                    binaryImg = segmentedImg == textCluster;
                    binaryImg = ~binaryImg;
                    binaryImg = imresize(binaryImg, size(img), 'nearest');

                    % Compute difference with ground truth
                    diffImg = xor(binaryImg, gt);
                    diffSum = sum(diffImg(:));

                    fprintf('    Difference Sum for orientations=%d, k=%d, λ=%d, pooling=%.1f: %d\n', ...
                            numOrientations, kClusters, wavelength, poolingScale, diffSum);

                    if diffSum < minDifference
                        minDifference = diffSum;
                        bestParams.numOrientations = numOrientations;
                        bestParams.kClusters = kClusters;
                        bestParams.wavelength = wavelength;
                        bestParams.poolingScale = poolingScale;
                        bestSegmentedImg = binaryImg;
                        fprintf('    New best result with difference sum: %d\n', minDifference);
                    end
                end
            end
        end
    end

    % Display best result
    fprintf('Image: %s\n', imageFiles{i});
    fprintf('Best orientations: %d\n', bestParams.numOrientations);
    fprintf('Best k clusters: %d\n', bestParams.kClusters);
    fprintf('Best wavelength: %d\n', bestParams.wavelength);
    fprintf('Best pooling scale: %.1f\n', bestParams.poolingScale);
    fprintf('Difference Sum: %d\n', minDifference);

    % Show images
    figure('Name', sprintf('Best Segmentation for %s', imageFiles{i}), 'NumberTitle', 'off');
    subplot(2, 2, 1); imshow(img); title('Original');
    subplot(2, 2, 2); imshow(bestSegmentedImg); title(sprintf('Best Segmentation\nOrients=%d, k=%d, λ=%d, Pool=%.1f', ...
        bestParams.numOrientations, bestParams.kClusters, bestParams.wavelength, bestParams.poolingScale));
    subplot(2, 2, 3); imshow(gt); title('Ground Truth');
    subplot(2, 2, 4); imshow(xor(bestSegmentedImg, gt)); title('Difference Image');
end

%% 3.2 3D Stereo Vision

% Parameters
maxDisparity = 15;
templateSize = [11, 11];

%% b) Load and Preprocess Synthetic Stereo Images

leftImgPath = 'img/corridorl.jpg';
rightImgPath = 'img/corridorr.jpg';

leftImg = imread(leftImgPath);
rightImg = imread(rightImgPath);

if size(leftImg, 3) == 3
    leftImgGray = rgb2gray(leftImg);
else
    leftImgGray = leftImg;
end

if size(rightImg, 3) == 3
    rightImgGray = rgb2gray(rightImg);
else
    rightImgGray = rightImg;
end

%% c) Compute Disparity Map for Synthetic Images

D = computeDisparityMapSSD(leftImgGray, rightImgGray, templateSize, maxDisparity);

figure('Name', 'Disparity Map - Synthetic Images', 'NumberTitle', 'off');
imshow(D, [-15 15]);
title('Disparity Map (Synthetic Images)');
colormap("gray");
colorbar;

%% d) Compute Disparity Map for Real Stereo Images

leftImgRealPath = 'img/triclopsi2l.jpg';
rightImgRealPath = 'img/triclopsi2r.jpg';

leftImgReal = imread(leftImgRealPath);
rightImgReal = imread(rightImgRealPath);

if size(leftImgReal, 3) == 3
    leftImgRealGray = rgb2gray(leftImgReal);
else
    leftImgRealGray = leftImgReal;
end

if size(rightImgReal, 3) == 3
    rightImgRealGray = rgb2gray(rightImgReal);
else
    rightImgRealGray = rightImgReal;
end

D_real = computeDisparityMapSSD(leftImgRealGray, rightImgRealGray, templateSize, maxDisparity);

figure('Name', 'Disparity Map - Real Images', 'NumberTitle', 'off');
imshow(D_real, [-15 15]);
title('Disparity Map (Real Images)');
colormap("gray");
colorbar;

%% a) Function to Compute Disparity Map using SSD
function disparityMap = computeDisparityMapSSD(leftImg, rightImg, templateSize, maxDisparity)
    % Convert images to double for calculation precision
    leftImg = double(leftImg);
    rightImg = double(rightImg);
    [rows, cols] = size(leftImg);
    
    % Calculate half of template dimensions for easier indexing
    halfTempH = floor(templateSize(1) / 2);
    halfTempW = floor(templateSize(2) / 2);
    
    % Initialize disparity map with size adjusted for template borders
    disparityMap = zeros(rows - templateSize(1) + 1, cols - templateSize(2) + 1);

    % Loop through each pixel within the valid template area
    for i = 1 + halfTempH : rows - halfTempH
        for j = 1 + halfTempW : cols - halfTempW
            minCost = Inf;  % Initialize min cost for SSD
            bestDisparity = 0;  % Initialize best disparity to zero

            % Extract the template from the left image
            templateLeft = leftImg(i - halfTempH : i + halfTempH, j - halfTempW : j + halfTempW);

            % Search for matching template in the right image within bidirectional maxDisparity
            for d = max(1, j - maxDisparity) : min(cols, j + maxDisparity)
                % Ensure template does not go out of right image bounds
                if (d - halfTempW >= 1) && (d + halfTempW <= cols)
                    % Extract the template from the right image
                    templateRight = rightImg(i - halfTempH : i + halfTempH, d - halfTempW : d + halfTempW);
                    
                    % Calculate SSD (Sum of Squared Differences) between templates
                    cost = sum(sum((templateLeft - templateRight) .^ 2));

                    % Update best disparity if current cost is lower
                    if cost < minCost
                        minCost = cost;
                        bestDisparity = j - d;  % Calculate disparity
                    end
                end
            end

            % Store the best disparity for the current pixel
            disparityMap(i - halfTempH, j - halfTempW) = bestDisparity;
        end
    end
end

%% e) Compute Disparity Map using ZNCC and NCC

% Smooth images
leftImgGraySmooth = imgaussfilt(leftImgGray, 1);
rightImgGraySmooth = imgaussfilt(rightImgGray, 1);

% Disparity Map using ZNCC for Synthetic Images
D_ZNCC = computeDisparityMapNCC(leftImgGraySmooth, rightImgGraySmooth, templateSize, maxDisparity, true);

figure('Name', 'Disparity Map using ZNCC - Synthetic Images', 'NumberTitle', 'off');
imshow(D_ZNCC, [-15 15]);
title('Disparity Map using ZNCC (Synthetic Images)');
colormap("gray");
colorbar;

% Disparity Map using NCC for Synthetic Images
D_NCC = computeDisparityMapNCC(leftImgGraySmooth, rightImgGraySmooth, templateSize, maxDisparity, false);

figure('Name', 'Disparity Map using NCC - Synthetic Images', 'NumberTitle', 'off');
imshow(D_NCC, [-15 15]);
title('Disparity Map using NCC (Synthetic Images)');
colormap("gray");
colorbar;

% Smooth real images
leftImgRealGraySmooth = imgaussfilt(leftImgRealGray, 1);
rightImgRealGraySmooth = imgaussfilt(rightImgRealGray, 1);

% Disparity Map using ZNCC for Real Images
D_real_ZNCC = computeDisparityMapNCC(leftImgRealGraySmooth, rightImgRealGraySmooth, templateSize, maxDisparity, true);

figure('Name', 'Disparity Map using ZNCC - Real Images', 'NumberTitle', 'off');
imshow(D_real_ZNCC, [-15 15]);
title('Disparity Map using ZNCC (Real Images)');
colormap("gray");
colorbar;

% Disparity Map using NCC for Real Images
D_real_NCC = computeDisparityMapNCC(leftImgRealGraySmooth, rightImgRealGraySmooth, templateSize, maxDisparity, false);

figure('Name', 'Disparity Map using NCC - Real Images', 'NumberTitle', 'off');
imshow(D_real_NCC, [-15 15]);
title('Disparity Map using NCC (Real Images)');
colormap("gray");
colorbar;

% Function to Compute Disparity Map using NCC/ZNCC with Bidirectional Disparity
function disparityMap = computeDisparityMapNCC(leftImg, rightImg, templateSize, maxDisparity, zeroMean)
    leftImg = double(leftImg);
    rightImg = double(rightImg);
    [rows, cols] = size(leftImg);
    halfTempH = floor(templateSize(1) / 2);
    halfTempW = floor(templateSize(2) / 2);
    disparityMap = zeros(rows - 2 * halfTempH, cols - 2 * halfTempW);

    for i = 1 + halfTempH : rows - halfTempH
        for j = 1 + halfTempW : cols - halfTempW
            maxScore = -Inf;
            bestDisparity = 0;

            % Extract template from the left image
            templateLeft = leftImg(i - halfTempH : i + halfTempH, j - halfTempW : j + halfTempW);
            if zeroMean
                templateLeft = templateLeft - mean(templateLeft(:));
            end
            denomLeft = sqrt(sum(templateLeft(:) .^ 2));

            % Bidirectional search for matching template in the right image
            for d = -maxDisparity : maxDisparity
                % Ensure the template does not go out of bounds in the right image
                if (j - d - halfTempW >= 1) && (j - d + halfTempW <= cols)
                    % Extract the corresponding template from the right image
                    templateRight = rightImg(i - halfTempH : i + halfTempH, j - d - halfTempW : j - d + halfTempW);
                    if zeroMean
                        templateRight = templateRight - mean(templateRight(:));
                    end
                    denomRight = sqrt(sum(templateRight(:) .^ 2));

                    % Compute NCC/ZNCC score
                    if denomLeft == 0 || denomRight == 0
                        score = 0;
                    else
                        score = sum(sum(templateLeft .* templateRight)) / (denomLeft * denomRight);
                    end

                    % Update the best disparity if the current score is higher
                    if score > maxScore
                        maxScore = score;
                        bestDisparity = d;
                    end
                end
            end

            % Store the best disparity for the current pixel
            disparityMap(i - halfTempH, j - halfTempW) = bestDisparity;
        end
    end
end


%% f) Compute Disparity Map using SAD

% Disparity Map using SAD for Synthetic Images
D_SAD = computeDisparityMapSAD(leftImgGraySmooth, rightImgGraySmooth, templateSize, maxDisparity);

figure('Name', 'Disparity Map using SAD - Synthetic Images', 'NumberTitle', 'off');
imshow(D_SAD, [-15 15]);
title('Disparity Map using SAD (Synthetic Images)');
colormap("gray");
colorbar;

% Disparity Map using SAD for Real Images
D_real_SAD = computeDisparityMapSAD(leftImgRealGraySmooth, rightImgRealGraySmooth, templateSize, maxDisparity);

figure('Name', 'Disparity Map using SAD - Real Images', 'NumberTitle', 'off');
imshow(D_real_SAD, [-15 15]);
title('Disparity Map using SAD (Real Images)');
colormap("gray");
colorbar;

% Function to Compute Disparity Map using SAD with Bidirectional Disparity
function disparityMap = computeDisparityMapSAD(leftImg, rightImg, templateSize, maxDisparity)
    leftImg = double(leftImg);
    rightImg = double(rightImg);
    [rows, cols] = size(leftImg);
    halfTempH = floor(templateSize(1) / 2);
    halfTempW = floor(templateSize(2) / 2);
    disparityMap = zeros(rows - 2 * halfTempH, cols - 2 * halfTempW);

    for i = 1 + halfTempH : rows - halfTempH
        for j = 1 + halfTempW : cols - halfTempW
            minSAD = Inf;
            bestDisparity = 0;

            % Extract the template from the left image
            templateLeft = leftImg(i - halfTempH : i + halfTempH, j - halfTempW : j + halfTempW);

            % Bidirectional search for matching template in the right image
            for d = -maxDisparity : maxDisparity
                % Ensure the template does not go out of bounds in the right image
                if (j - d - halfTempW >= 1) && (j - d + halfTempW <= cols)
                    % Extract the corresponding template from the right image
                    templateRight = rightImg(i - halfTempH : i + halfTempH, j - d - halfTempW : j - d + halfTempW);
                    
                    % Calculate the Sum of Absolute Differences (SAD)
                    sad = sum(sum(abs(templateLeft - templateRight)));

                    % Update the best disparity if the current SAD is lower
                    if sad < minSAD
                        minSAD = sad;
                        bestDisparity = d;
                    end
                end
            end

            % Store the best disparity for the current pixel
            disparityMap(i - halfTempH, j - halfTempW) = bestDisparity;
        end
    end
end
