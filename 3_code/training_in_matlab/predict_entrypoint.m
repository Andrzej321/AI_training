function y = predict_entrypoint(X)
% predict_entrypoint.m
% Entry point for C code generation with best saved GRU (trainedGRU_best.mat).
% X must be [features x seqLen] to match training orientation (F x T).
%
% Example (if F=12, T=100):
%   codegen predict_entrypoint -args {zeros(12,100,'single')} -report

persistent net seqLen inputSize
if isempty(net)
    S = load('trainedGRU_best.mat','netBest','seqLen','inputSize');
    if ~isfield(S,'netBest')
        error('trainedGRU_best.mat does not contain netBest.');
    end
    net = S.netBest;
    if isfield(S,'seqLen'),    seqLen = S.seqLen;    end
    if isfield(S,'inputSize'), inputSize = S.inputSize; end
end

assert(isequal(size(X,1), inputSize) && isequal(size(X,2), seqLen), ...
    'Input X must be [%d x %d] (features x seqLen).', inputSize, seqLen);

% Series/DAG network accepts (features x time) directly for sequenceInputLayer
y = predict(net, X);
end