%% Simple demo of "Learning to See in the Dark" pretrained network
clear; clc; close all;

%% 1) Download pretrained network (only once)
dataDir = fullfile(tempdir, "Sony2025");
if ~exist(dataDir, "dir")
    mkdir(dataDir);
end

modelFile = fullfile(dataDir, "trainedLowLightCameraPipelineNet.mat");
if ~isfile(modelFile)
    fprintf("Downloading pretrained low-light model...\n");

    modelURL = "https://ssd.mathworks.com/supportfiles/" + ...
        "vision/data/trainedLowLightCameraPipelineDlnetwork.zip";

    zipFile = fullfile(dataDir, "trainedLowLightCameraPipelineDlnetwork.zip");

    websave(zipFile, modelURL);
    unzip(zipFile, dataDir);

    fprintf("Download complete. Model saved in:\n%s\n", dataDir);
end

%% 2) Load pretrained network
load(modelFile, "netTrained");

%% 3) Read your image (edit this line)
I = imread("Example_03.png");   
I = im2single(imresize(I, [512 512]));

%% 4) Simulate dark + noisy image
darkFactor = 0.05;          % <-- CAMBIO REALIZADO (antes 0.03)
I_dark = I * darkFactor;
I_dark_noisy = imnoise(I_dark, "gaussian", 0, 0.002);

%% 5) Build simple 4-channel fake RAW
I_gray = rgb2gray(I_dark_noisy);
rawFake = repmat(I_gray, 1, 1, 4);

%% 6) Wrap into dlarray
input = dlarray(rawFake, "SSCB");

if canUseGPU
    input = gpuArray(input);
end

%% 7) Run pretrained network
out = predict(netTrained, input);

%% 8) Convert to image
out = gather(extractdata(out));
out = squeeze(out);
out = im2uint8(out);

%% 9) Show results
figure;
subplot(1,3,1); imshow(I);               title("Original RGB");
subplot(1,3,2); imshow(I_dark_noisy);    title("Simulated low light");
subplot(1,3,3); imshow(out);             title("Enhanced image");
