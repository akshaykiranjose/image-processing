% load the images from the AT&T-faces dataset
close all; clear; clc;

dataset_loc = 'C:\Users\Akshay\Desktop\Compressed Sensing\datasets';
all_image_files = dir(fullfile(dataset_loc, 'att_faces', '**', '*.pgm'));

att_faces = cell(length(all_image_files), 1);
for i = 1:length(att_faces)
    att_faces{i,1} = imread(fullfile(all_image_files(i).folder, all_image_files(i).name));
end

figure;
img = att_faces{9,1};
imshow(img, 'InitialMagnification', 250)

figure;
h_w_range = [19 19 59 79];
crop_img = imcrop(img, h_w_range); % size of cropped image [80 60]
imshow(crop_img, 'InitialMagnification', 250)

figure;
resize_img = imresize(crop_img, [64, 64]);
imshow(resize_img, 'InitialMagnification', 250);

%% Sparsity comparison between the two domains

% Find the DCT and equate low-magnitude 
% coefficients to zero and check the reconstruction quality

K = resize_img;
J = dct2(K);
figure;
subplot(2,1,1);
x = K(:);
h1 = histogram(x, 60, 'Normalization', 'probability', ...
               'FaceAlpha', 0.6, 'FaceColor', [0.2 0.6 0.9]);
legend({'Standard basis'},'Location','best');

grid on;
subplot(2,1,2);
s = J(:);
h2 = histogram(s, 60, 'Normalization', 'probability', ...
               'FaceAlpha', 0.6, 'FaceColor', [0.9 0.4 0.2]);
grid on;

legend({'DCT basis'},'Location','best');
set(gca,'FontSize',11);

%% 


% We need a basis for representing our images of size (64 x 64), 
% already sparse in the DCT domain, needs a representation of the 
% form x = \Phi s where s is a sparse vector and x is the image 
% stretched to make a single column (note the major order of transforming 
% the image matrix into a vector.


[rows, cols] = size(K);

row_dctmtx = dctmtx(rows);
col_dctmtx = dctmtx(cols);

Phi = kron(col_dctmtx, row_dctmtx);

[P, M] = size(K);
IMG_SIZE = M*P;

NUM_SAMPLES = [500, 1000, 1500, 2000, 2500, 3000];
P_SNRS_DCT = [];
SSIMS_DCT = [];

for i = 1:length(NUM_SAMPLES)
    SAMPLES = randi(IMG_SIZE, [NUM_SAMPLES(i), 1]);

    sprintf("============%d============", i)

    x = K(:);
    
    % measured x
    y = double(x(SAMPLES));
    Theta = Phi(SAMPLES, :);
    
    cvx_begin quiet
        variable s_l1(IMG_SIZE);
        minimize( norm(s_l1, 1) );
        subject to
            Theta*s_l1 == y;
    cvx_end;

    x_l1_recon = Phi * s_l1;
    K_l1_recon = uint8(reshape(x_l1_recon, [rows, cols]));

    P_SNRS_DCT = [P_SNRS_DCT psnr(K_l1_recon, K)];
    SSIMS_DCT = [SSIMS_DCT multissim(K_l1_recon, K)];

end


%% SVD reconstruction

load 'U_yalefaces.mat'

Phi = U;

NUM_SAMPLES = [500, 1000, 1500, 2000, 2500, 3000];
P_SNRS_SVD = [];
SSIMS_SVD = [];

for i = 1:length(NUM_SAMPLES)
    SAMPLES = randi(IMG_SIZE, [NUM_SAMPLES(i), 1]);

    sprintf("============%d============", i)

    x = K(:);
    
    % measured x
    y = double(x(SAMPLES));
    Theta = Phi(SAMPLES, :);
    
    cvx_begin quiet
        variable s_l1(IMG_SIZE);
        minimize( norm(s_l1, 1) );
        subject to
            Theta*s_l1 == y;
    cvx_end;

    x_l1_recon = Phi * s_l1;
    K_l1_recon = uint8(reshape(x_l1_recon, [rows, cols]));

    P_SNRS_SVD = [P_SNRS_SVD psnr(K_l1_recon, K)];
    SSIMS_SVD = [SSIMS_SVD multissim(K_l1_recon, K)];

end

%%

figure;
plot(NUM_SAMPLES, SSIMS_DCT, 'r-o', 'LineWidth', 1.5);  % Red line with circles
hold on;
plot(NUM_SAMPLES, SSIMS_SVD, 'b-s', 'LineWidth', 1.5);  % Blue line with squares
hold off;

% Label and legend
xlabel('Number of samples in observation');
ylabel('SSIM');
title('Variation of SSIM with observation size');
legend('DCT', 'SVD', 'Location', 'best');
grid on;


figure;
plot(NUM_SAMPLES, P_SNRS_DCT, 'r-o', 'LineWidth', 1.5);  % Red line with circles
hold on;
plot(NUM_SAMPLES, P_SNRS_SVD, 'b-s', 'LineWidth', 1.5);  % Blue line with squares
hold off;

% Label and legend
xlabel('Number of samples in observation');
ylabel('SNR (dB)');
title('Variation of Peak-SNR with observation size');
legend('DCT', 'SVD', 'Location', 'best');
grid on;


%%
montage({K_l1_recon, K});