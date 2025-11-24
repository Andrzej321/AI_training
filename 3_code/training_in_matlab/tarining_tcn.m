% train_tcn_manual.m
% Single-model TCN training with Python-style per-epoch validation,
% conditional best checkpoint saving, optional early stopping.
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
seqLen     = 150;      % sequence length (window length)
maxEpochs      = 100;
miniBatchSize  = 12;
initialLR      = 1e-4;
patience       = 5;        % set Inf to disable early stopping
gradientClip   = 5.0;

% --- TCN architecture config ---
% Dilation schedule for residual blocks, e.g. [1 2 4 8] or [1 2 4 8 16 32]
dilationSchedule = [1 2 4 8];
numResidualBlocks = numel(dilationSchedule);

% Channels per layer:
% (1) scalar: same channels in all blocks, e.g. 64
% (2) vector of length numResidualBlocks: [32 64 64 128]
channelsPerLayer = 64;

% Number of Conv1D layers per residual block
convsPerBlock = 2;

% Temporal kernel size
kernelSize = 4;  % must be odd for symmetric padding in this implementation

% Dropout inside residual blocks
dropoutRate = 0.1;

% --- Data paths & sequence extraction ---
trainDir = "C:\my files\thesis\AI_training\1_data\i7\it_1\it_1_100\1_training";
valDir   = "C:\my files\thesis\AI_training\1_data\i7\it_1\it_1_100\2_testing";
stepSize = 5;              % sliding window stride

featureSelectionMode = "keep";   % "keep" | "drop" | "none"
selectedFeatureCols  = [1, 2, 6, 7, 8, 9, 10, 11, 13, 15 ,17, 19];  % EDIT if needed
dropFeatureCols      = [];

useInputIdMapping = false;
input_id = 2;     % only used if useInputIdMapping=true

saveBestPath    = "C:\my files\thesis\AI_training\3_code\training_in_matlab\trained_models\TCN\trainedTCN_best.mat";
saveHistoryPath = "C:\my files\thesis\AI_training\3_code\training_in_matlab\trained_models\TCN\trainedTCN_history.mat";

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

%% Build TCN (last time-step output only)
layers = [ sequenceInputLayer(inputSize,"Name","input","Normalization","none") ];

prevChannels = inputSize;
for b = 1:numResidualBlocks
    dilation = dilationSchedule(b);
    nCh      = channelsVec(b);

    % Block naming prefix
    blockName = sprintf("b%d",b);

    for c = 1:convsPerBlock
        convName = sprintf("%s_conv%d",blockName,c);
        bnName   = sprintf("%s_bn%d",blockName,c);
        reluName = sprintf("%s_relu%d",blockName,c);
        dropName = sprintf("%s_drop%d",blockName,c);

        % Causal padding: we pad only on the left by (kernelSize-1)*dilation
        pad = (kernelSize-1)*dilation;

        layers(end+1) = sequenceFoldingLayer("Name",convName+"_fold"); %#ok<AGROW>
        layers(end+1) = convolution1dLayer(kernelSize, nCh, ...
            "Name",convName, ...
            "DilationFactor",dilation, ...
            "Padding",[pad 0]); %#ok<AGROW>
        layers(end+1) = batchNormalizationLayer("Name",bnName); %#ok<AGROW>
        layers(end+1) = reluLayer("Name",reluName); %#ok<AGROW>
        if dropoutRate > 0
            layers(end+1) = dropoutLayer(dropoutRate,"Name",dropName); %#ok<AGROW>
        end
        layers(end+1) = sequenceUnfoldingLayer("Name",convName+"_unfold"); %#ok<AGROW>

        prevChannels = nCh; % after first conv
    end

    % Residual connection from block input to block output.
    % We will wire these in layerGraph below.
end

% Final temporal pooling to scalar per sequence: use global average pooling over time.
layers(end+1) = globalAveragePooling1dLayer("Name","gap"); %#ok<AGROW>
layers(end+1) = fullyConnectedLayer(1,"Name","fc"); %#ok<AGROW>

% Note: No regressionLayer in the training dlnetwork graph (we compute loss manually).
lg = layerGraph(layers);

% Wire residual connections between the appropriate layers.
% For simplicity, we connect:
%   - Input of block b (which is output of previous block or 'input')
%   - To final unfolded output of that block via addition layer.
for b = 1:numResidualBlocks
    blockName = sprintf("b%d",b);

    % Identify last unfold of this block
    lastUnfoldName = sprintf("%s_conv%d_unfold",blockName,convsPerBlock);
    addName        = sprintf("%s_add",blockName);

    % Add addition layer
    lg = addLayers(lg, additionLayer(2,"Name",addName));

    if b == 1
        blockInputName = "input";
    else
        prevAddName = sprintf("b%d_add",b-1);
        blockInputName = prevAddName;
    end

    % Connect block input and block output to adder
    lg = connectLayers(lg, blockInputName, sprintf('%s/in1', addName));
    lg = connectLayers(lg, lastUnfoldName, sprintf('%s/in2', addName));
end

% Connect final add to global average pooling
lastAddName = sprintf("b%d_add",numResidualBlocks);
lg = connectLayers(lg, lastAddName, "gap");

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

        % Prepare batch: (F x T) → dlarray 'CTB'; targets row vector [1 x B]
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
        dlOutVal = forward(dlnet, dlXv);   % [1 x B]
        lossVal = mse(dlOutVal, dlYv);     % both are [1 x B]
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
    X = zeros(F, T, batchSize, 'single');
    for i = 1:batchSize
        Xi = XCell{i};
        X(:,:,i) = single(Xi);
    end
    dlX = dlarray(X, 'CTB');           % C=features, T=time, B=batch
    dlY = dlarray(single(YVec(:)'));   % row vector [1 x B]
end

function [gradients, loss] = modelGradients_tcn(dlnet, dlX, dlYrow)
    dlOut = forward(dlnet, dlX); % [1 x batch]
    loss = mse(dlOut, dlYrow);
    gradients = dlgradient(loss, dlnet.Learnables);
end

function g = clipGrad(g, clipVal)
    if isempty(g); return; end
    n = sqrt(sum(g(:).^2));
    if n > clipVal
        g = g * (clipVal / max(n, eps('like',g)));
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