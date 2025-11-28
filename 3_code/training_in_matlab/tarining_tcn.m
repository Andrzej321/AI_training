% train_tcn_manual.m
% Single-model TCN training with Python-style per-epoch validation,
% conditional best checkpoint saving, optional early stopping.
%
% This implementation avoids sequence folding/unfolding (unsupported in dlnetwork)
% by representing the time axis as a spatial dimension (H=T) and using 2D conv
% with width=1. Residual connections include 1x1 projection when channels differ.
%
% Data expectation:
%   Each CSV: columns 1..F are features, last column is target (scalar).
% Feature selection supported (keep/drop or input_id mapping).
%
% Output:
%   - trainedTCN_best.mat (best validation model: assembled DAG/Series with regression layer)
%   - trainedTCN_history.mat (loss curves + config)
%
% Author: (Andrzej Skrodzki, TCN variant)

clear; clc;

%% ================= USER CONFIG =================
% --- High-level training config ---
seqLen         = 50;      % sequence length (window length)
maxEpochs      = 100;
miniBatchSize  = 12;
initialLR      = 1e-4;
patience       = 5;        % set Inf to disable early stopping
gradientClip   = 5.0;

% --- TCN architecture config ---
% Dilation schedule for residual blocks, e.g. [1 2 4 8] or [1 2 4 8 16 32]
dilationSchedule   = [1 2 4 8];
numResidualBlocks  = numel(dilationSchedule);

% Channels per layer:
% (1) scalar: same channels in all blocks, e.g. 64
% (2) vector of length numResidualBlocks: [32 64 64 128]
channelsPerLayer   = 64;

% Number of Conv layers per residual block
convsPerBlock      = 2;

% Temporal kernel size (applied along H = time)
kernelSize         = 4;  % causal: pad top by (kernelSize-1)*dilation

% Dropout inside residual blocks
dropoutRate        = 0.1;

% --- Data paths & sequence extraction ---
trainDir = "C:\work\AI_training\1_data\i7\it_1\it_1_100\1_training";
valDir   = "C:\work\AI_training\1_data\i7\it_1\it_1_100\2_testing";
stepSize = 5;              % sliding window stride

featureSelectionMode = "keep";   % "keep" | "drop" | "none"
selectedFeatureCols  = [1, 2, 6, 7, 8, 9, 10, 11, 13, 15 ,17, 19];  % EDIT if needed
dropFeatureCols      = [];

useInputIdMapping = false;
input_id = 2;     % only used if useInputIdMapping=true

saveBestPath    = "C:\work\AI_training\3_code\training_in_matlab\trained_models\TCN\trainedTCN_best.mat";
saveHistoryPath = "C:\work\AI_training\3_code\training_in_matlab\trained_models\TCN\trainedTCN_history.mat";

rng(42);  % reproducibility
%% ===============================================

fprintf("Loading training data...\n");
[XTrain, YTrain, inputSizeTrain] = loadDirAsSequences(trainDir, seqLen, stepSize, ...
    featureSelectionMode, selectedFeatureCols, dropFeatureCols, useInputIdMapping, input_id);
fprintf("Training sequences: %d | Features: %d | SeqLen=%d\n", numel(XTrain), inputSizeTrain, size(XTrain{1},2));

fprintf("Loading validation data...\n");
[XVal, YVal, inputSizeVal] = loadDirAsSequences(valDir, seqLen, stepSize, ...
    featureSelectionMode, selectedFeatureCols, dropFeatureCols, useInputIdMapping, input_id);
assert(inputSizeVal == inputSizeTrain, "Train/Val feature mismatch.");
inputSize = inputSizeTrain;

% Ensure responses numeric vectors
if iscell(YTrain), YTrain = cell2mat(YTrain); end
if iscell(YVal),   YVal   = cell2mat(YVal);   end
assert(size(YTrain,2)==1 && size(YVal,2)==1, "Expect scalar targets.");
assert(all(isfinite(YTrain)) && all(isfinite(YVal)), "Targets contain NaN/Inf.");

%% Normalize channelsPerLayer to vector form
if isscalar(channelsPerLayer)
    channelsVec = repmat(channelsPerLayer, 1, numResidualBlocks);
else
    assert(numel(channelsPerLayer)==numResidualBlocks, ...
        "channelsPerLayer must be scalar or length numResidualBlocks.");
    channelsVec = channelsPerLayer(:)'; % row
end

%% Build TCN as 2D-conv over [H=T, W=1, C=features]
% Input: imageInputLayer([H W C]) with H=seqLen, W=1, C=inputSize
layers = [
    imageInputLayer([seqLen 1 inputSize], "Name","input", "Normalization","none")
];

% Keep track of last layer name per block for residual wiring
blockLastOut = strings(numResidualBlocks,1);

for b = 1:numResidualBlocks
    dilation = dilationSchedule(b);
    nCh      = channelsVec(b);
    blockName = sprintf("b%d",b);

    for c = 1:convsPerBlock
        convName = sprintf("%s_conv%d",blockName,c);
        bnName   = sprintf("%s_bn%d",blockName,c);
        reluName = sprintf("%s_relu%d",blockName,c);
        dropName = sprintf("%s_drop%d",blockName,c);

        % Causal padding along H (top=pad, bottom=0). No padding along W.
        padTop = (kernelSize-1)*dilation;

        layers(end+1) = convolution2dLayer([kernelSize 1], nCh, ...
            "Name",convName, ...
            "DilationFactor",[dilation 1], ...
            "Padding",[padTop 0 0 0]); %#ok<AGROW>
        layers(end+1) = batchNormalizationLayer("Name",bnName); %#ok<AGROW>
        layers(end+1) = reluLayer("Name",reluName); %#ok<AGROW>
        if dropoutRate > 0
            layers(end+1) = dropoutLayer(dropoutRate,"Name",dropName); %#ok<AGROW>
            lastName = dropName;
        else
            lastName = reluName;
        end
    end

    blockLastOut(b) = string(lastName);
end

% Final temporal pooling to scalar per sequence: global average over H (W=1)
layers(end+1) = globalAveragePooling2dLayer("Name","gap"); %#ok<AGROW>
layers(end+1) = fullyConnectedLayer(1,"Name","fc"); %#ok<AGROW>

lg = layerGraph(layers);

%% Wire residual connections and projections. Route add -> next block (or gap)
currentChannels = inputSize;

for b = 1:numResidualBlocks
    blockName   = sprintf("b%d",b);
    nCh         = channelsVec(b);
    addName     = sprintf("%s_add", blockName);

    % Names for first and last layers inside the block
    firstConvName = sprintf("%s_conv1", blockName);
    lastOutName   = blockLastOut(b);

    % Determine block input source
    if b == 1
        blockInputName     = "input";
    else
        blockInputName     = sprintf("b%d_add", b-1);
    end

    % Addition layer for residual sum
    lg = addLayers(lg, additionLayer(2,"Name",addName));

    % Skip/projection branch
    if currentChannels ~= nCh
        % 1x1 projection to match channels
        projName = sprintf("%s_skip_conv1x1", blockName);
        lg = addLayers(lg, convolution2dLayer([1 1], nCh, "Padding","same", "Name", projName));
        lg = connectLayers(lg, blockInputName, projName);
        lg = connectLayers(lg, projName, addName + "/in1");
    else
        % Direct skip
        lg = connectLayers(lg, blockInputName, addName + "/in1");
    end

    % Main path output into adder
    lg = connectLayers(lg, lastOutName, addName + "/in2");

    % Route forward: adder output feeds next block's first conv or GAP
    if b < numResidualBlocks
        nextFirst = sprintf("b%d_conv1", b+1);
        % Replace default connection lastOut -> nextFirst with add -> nextFirst
        lg = safeDisconnect(lg, char(lastOutName), nextFirst);
        lg = connectLayers(lg, addName, nextFirst);
    else
        % Last block: send to GAP
        lg = safeDisconnect(lg, char(lastOutName), "gap");
        lg = connectLayers(lg, addName, "gap");
    end

    currentChannels = nCh;
end

% Create dlnetwork
dlnet = dlnetwork(lg);

%% Training state
bestValLoss      = inf;
bestEpoch        = 0;
earlyStopCounter = 0;

trainLossHistory = zeros(maxEpochs,1);
valLossHistory   = zeros(maxEpochs,1);

% Adam accumulators
avgGrad   = [];
avgSqGrad = [];
beta1 = 0.9;
beta2 = 0.999;

%% Mini-batch index preparation
numTrainSeq = numel(XTrain);
numValSeq   = numel(XVal);
iteration   = 0;  % increments per mini-batch

fprintf("Starting TCN training...\n");
for epoch = 1:maxEpochs
    fprintf("\nEpoch %d / %d\n", epoch, maxEpochs);

    order = randperm(numTrainSeq);
    epochTrainLoss = 0;
    numTrainBatches = 0;

    for startIdx = 1:miniBatchSize:numTrainSeq
        batchIdx = order(startIdx:min(startIdx+miniBatchSize-1, numTrainSeq));
        iteration = iteration + 1;

        % Prepare batch: (F x T) → dlarray 'SSCB' as [T x 1 x F x B]; targets row vector [1 x B] (CB)
        [dlX, dlY] = makeBatch(XTrain(batchIdx), YTrain(batchIdx));

        % Forward + gradients
        [gradients, lossValue] = dlfeval(@modelGradients_tcn, dlnet, dlX, dlY);
        epochTrainLoss   = epochTrainLoss + double(lossValue);
        numTrainBatches  = numTrainBatches + 1;

        % Gradient clipping across gradients table
        gradients = dlupdate(@(g) clipGrad(g, gradientClip), gradients);

        % Adam update
        [dlnet, avgGrad, avgSqGrad] = adamupdate( ...
            dlnet, gradients, avgGrad, avgSqGrad, iteration, initialLR, beta1, beta2);
    end

    avgTrainLoss = epochTrainLoss / max(1,numTrainBatches);
    trainLossHistory(epoch) = avgTrainLoss;

    % --------- Validation pass ---------
    valLossAccum = 0;
    valBatches   = 0;
    for startIdx = 1:miniBatchSize:numValSeq
        batchIdx = startIdx:min(startIdx+miniBatchSize-1, numValSeq);
        [dlXv, dlYv] = makeBatch(XVal(batchIdx), YVal(batchIdx));
        yValPred = forward(dlnet, dlXv);        % expected 'CB' [1 x B]
        lossVal  = mse(yValPred, dlYv);         % both 'CB' [1 x B]
        valLossAccum = valLossAccum + double(lossVal);
        valBatches = valBatches + 1;
    end
    avgValLoss = valLossAccum / max(1,valBatches);
    valLossHistory(epoch) = avgValLoss;

    fprintf("TrainLoss: %.6f | ValLoss: %.6f\n", avgTrainLoss, avgValLoss);

    % Checkpoint if improved
    if avgValLoss < bestValLoss
        bestValLoss = avgValLoss;
        bestEpoch   = epoch;
        earlyStopCounter = 0;

        % Add regression layer at save time and assemble
        netBest = assembleForSave(dlnet); %#ok<NASGU>
        save(saveBestPath, 'netBest', 'seqLen', 'inputSize', ...
            'dilationSchedule','channelsPerLayer','kernelSize','convsPerBlock', ...
            'dropoutRate','bestValLoss','bestEpoch');
        fprintf("  >> Improved. Saved best TCN model to %s\n", saveBestPath);
    else
        earlyStopCounter = earlyStopCounter + 1;
        fprintf("  No improvement (%d / %d patience)\n", earlyStopCounter, patience);
        if earlyStopCounter >= patience
            fprintf("Early stopping triggered.\n");
            break;
        end
    end
end

%% Save history
save(saveHistoryPath, 'trainLossHistory', 'valLossHistory', 'bestValLoss', 'bestEpoch', ...
    'seqLen','inputSize','dilationSchedule','channelsPerLayer', ...
    'kernelSize','convsPerBlock','dropoutRate','patience','maxEpochs');
fprintf("\nTCN training complete. Best val loss %.6f at epoch %d\n", bestValLoss, bestEpoch);
fprintf("History saved to %s\n", saveHistoryPath);

%% ============== Helper Functions ==============

function netOut = assembleForSave(dlnet)
    % Add regression layer, connect 'fc'->'regression', assemble to DAG/Series.
    % fc outputs 'CB' for GAP2D input, which regressionLayer expects.
    lgSave = layerGraph(dlnet);
    hasReg = any(strcmp({lgSave.Layers.Name}, 'regression'));
    if ~hasReg
        reg = regressionLayer('Name','regression');
        lgSave = addLayers(lgSave, reg);
        lgSave = connectLayers(lgSave, 'fc', 'regression');
    end
    netOut = assembleNetwork(lgSave);
end

function [dlX, dlY] = makeBatch(XCell, YVec)
% XCell: cell array of sequences (F x T)
% YVec: numeric vector (batch x 1)
    batchSize = numel(XCell);
    F = size(XCell{1},1);
    T = size(XCell{1},2);
    X = zeros(T, 1, F, batchSize, 'single'); % H=T, W=1, C=F, B=batch
    for i = 1:batchSize
        Xi = single(XCell{i});   % F x T
        X(:,1,:,i) = reshape(permute(Xi, [2 1]), [T 1 F]); % -> T x 1 x F
    end
    dlX = dlarray(X, 'SSCB');                 % H, W, C, B
    dlY = dlarray(single(YVec(:))', 'CB');    % row vector [1 x B]
end

function [gradients, loss] = modelGradients_tcn(dlnet, dlX, dlYrow)
    yPred = forward(dlnet, dlX); % expected 'CB' [1 x B]
    loss  = mse(yPred, dlYrow);
    gradients = dlgradient(loss, dlnet.Learnables);
end

function g = clipGrad(g, clipVal)
    if isempty(g); return; end
    n = sqrt(sum(g(:).^2));
    if n > clipVal
        g = g * (clipVal / max(n, eps('like',g)));
    end
end

function lg = safeDisconnect(lg, src, dst)
    conn = lg.Connections;
    if any(strcmp(conn.Source, src) & strcmp(conn.Destination, dst))
        lg = disconnectLayers(lg, src, dst);
    end
end

% ================= Data loading (copied from training_gru.m) ==============
function [XCell, YVec, F] = loadDirAsSequences(dirPath, seqLen, stepSize, ...
        featureSelectionMode, selectedCols, dropCols, useInputIdMapping, input_id)

    files = dir(fullfile(dirPath,"*.csv"));
    assert(~isempty(files), "No CSV files in %s", dirPath);

    if useInputIdMapping
        selectedCols = selectColumnsForInputId(input_id);
        featureSelectionMode = "keep";
    end

    XCell = {};
    YRaw  = [];

    for k = 1:numel(files)
        P = fullfile(files(k).folder, files(k).name);
        M = readmatrix(P);
        if isempty(M) || size(M,2) < 2
            continue
        end
        feats = M(:,1:end-1);
        targ  = M(:,end);

        feats = applyFeatureSelection(feats, featureSelectionMode, selectedCols, dropCols);

        [Xs, Ys] = makeWindows(feats, targ, seqLen, stepSize);
        if isempty(Xs), continue; end

        XCell = [XCell; Xs]; %#ok<AGROW>
        YRaw  = [YRaw; Ys]; %#ok<AGROW>
    end
    assert(~isempty(XCell), "No sequences produced from %s", dirPath);

    % Convert targets to numeric and remove non-finite
    YVec = YRaw;
    mask = isfinite(YVec);
    if ~all(mask)
        warning("Removing %d sequences with non-finite targets.", sum(~mask));
        YVec = YVec(mask);
        XCell = XCell(mask);
    end
    F = size(XCell{1},1); % sequences stored as (F x T)
end

function feats = applyFeatureSelection(feats, mode, keepCols, dropCols)
    switch string(mode)
        case "keep"
            assert(~isempty(keepCols),"Mode 'keep' requires selectedFeatureCols.");
            feats = feats(:, keepCols);
        case "drop"
            if ~isempty(dropCols)
                feats = feats(:, setdiff(1:size(feats,2), dropCols));
            end
        otherwise
            % none
    end
end

function [Xseq, Yseq] = makeWindows(X, y, T, step)
    N = size(X,1);
    if N < T
        Xseq = {};
        Yseq = [];
        return
    end
    starts = 1:step:(N - T + 1);
    Xseq = cell(numel(starts),1);
    Yseq = zeros(numel(starts),1);
    w = 0;
    for s = starts
        e = s + T - 1;
        winX = X(s:e,:);
        targetY = y(e,1);
        if any(~isfinite(winX),'all') || ~isfinite(targetY)
            continue
        end
        w = w + 1;
        Xseq{w} = winX';   % (F x T)
        Yseq(w) = targetY;
    end
    Xseq = Xseq(1:w);
    Yseq = Yseq(1:w);
end

function cols = selectColumnsForInputId(input_id)
    switch input_id
        case 1
            cols = [1 2 3 5 7];
        case 2
            cols = [1 4 6 8 10 12]; % EDIT to match Python mapping
        case 3
            cols = [2 3 9 11];
        otherwise
            error("Unknown input_id=%d.", input_id);
    end
end