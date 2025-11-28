% train_gru_manual.m
% Single-model GRU training with Python-style per-epoch validation,
% conditional best checkpoint saving, optional early stopping.
%
% Data expectation:
%   Each CSV: columns 1..F are features, last column is target (scalar).
% Feature selection supported (keep/drop or input_id mapping).
%
% Output:
%   - trainedGRU_best.mat (best validation model: assembled DAG/Series with regression layer)
%   - trainedGRU_history.mat (loss curves + config)
%
% Author: (Andrzej Skrodzki)

clear; clc;

%% ================= USER CONFIG =================
seqLen     = 90;      % sequence length
hiddenSize = 208;     % GRU hidden size
numLayers  = 3;       % number of GRU layers

maxEpochs      = 100;
miniBatchSize  = 12;
initialLR      = 1e-5;
patience       = 5;        % set Inf to disable early stopping
gradientClip   = 5.0;

trainDir = "C:\work\AI_training\1_data\i7\it_1\it_1_100\1_training";
valDir   = "C:\work\AI_training\1_data\i7\it_1\it_1_100\2_testing";
stepSize = 5;              % sliding window stride

featureSelectionMode = "keep";   % "keep" | "drop" | "none"
selectedFeatureCols  = [1, 2, 6, 7, 8, 9, 10, 11, 13, 15 ,17, 19];  % EDIT to match Python subset
dropFeatureCols      = [];

useInputIdMapping = false;
input_id = 2;     % only used if useInputIdMapping=true

saveBestPath    = "C:\work\AI_training\3_code\training_in_matlab\trained_models\GRU\trainedGRU_best.mat";
saveHistoryPath = "C:\work\AI_training\3_code\training_in_matlab\trained_models\GRU\trainedGRU_history.mat";

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

%% Build GRU (last output only)
layers = [ sequenceInputLayer(inputSize,"Name","input") ];
for L = 1:numLayers
    if L < numLayers
        layers(end+1) = gruLayer(hiddenSize,"OutputMode","sequence","Name",sprintf("gru_%d",L)); %#ok<AGROW>
    else
        layers(end+1) = gruLayer(hiddenSize,"OutputMode","last","Name","gru_last"); %#ok<AGROW>
    end
end
layers(end+1) = fullyConnectedLayer(1,"Name","fc"); %#ok<AGROW>

% Note: No regressionLayer in the training dlnetwork graph (we compute loss manually).
lg = layerGraph(layers);
dlnet = dlnetwork(lg);

%% Training state
bestValLoss      = inf;
bestEpoch        = 0;
earlyStopCounter = 0;

trainLossHistory = zeros(maxEpochs,1);
valLossHistory   = zeros(maxEpochs,1);

% Adam accumulators (managed by built-in adamupdate)
avgGrad   = [];
avgSqGrad = [];
beta1 = 0.9;
beta2 = 0.999;

%% Mini-batch index preparation
numTrainSeq = numel(XTrain);
numValSeq   = numel(XVal);
iteration   = 0;  % increments per mini-batch

fprintf("Starting training...\n");
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
        [gradients, lossValue] = dlfeval(@modelGradients, dlnet, dlX, dlY);
        epochTrainLoss   = epochTrainLoss + double(lossValue);
        numTrainBatches  = numTrainBatches + 1;

        % Gradient clipping across gradients table
        gradients = dlupdate(@(g) clipGrad(g, gradientClip), gradients);

        % Built-in Adam update (no manual table indexing)
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

        % Convert dlnet to a valid assembled network for prediction/codegen by
        % adding a regression layer connected to 'fc' only at save time.
        netBest = assembleForSave(dlnet); %#ok<NASGU>
        save(saveBestPath, 'netBest', 'seqLen', 'inputSize', 'hiddenSize', 'numLayers', ...
            'bestValLoss', 'bestEpoch');
        fprintf("  >> Improved. Saved best model to %s\n", saveBestPath);
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
    'seqLen','inputSize','hiddenSize','numLayers','patience','maxEpochs');
fprintf("\nTraining complete. Best val loss %.6f at epoch %d\n", bestValLoss, bestEpoch);
fprintf("History saved to %s\n", saveHistoryPath);

%% ============== Helper Functions ==============

function netOut = assembleForSave(dlnet)
    % Create a copy of the current graph, add a regression layer, connect fc->regression,
    % and assemble to a DAG/Series network suitable for predict and codegen.
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

function [gradients, loss] = modelGradients(dlnet, dlX, dlYrow)
    % Forward
    dlOut = forward(dlnet, dlX); % [1 x batch]
    % Ensure shapes align: both [1 x B]
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