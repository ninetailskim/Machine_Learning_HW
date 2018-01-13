clear;
close all;

folder = dir('./fl3_10000_10000_test/images/*-outputs.png');
disp(length(folder));
data = './fl3_10000_10000_test/images/';
i = 0;
fl3_10t_10t_test = [];
for s = 1:length(folder)
    filename = folder(s, 1).name;
    pre_filename = filename(1:length(filename) - 11);
    outputs_path = [data, filename];
    targets_path = [data, pre_filename, 'targets.png'];
    %outputs_path
    %targets_path
    outputs_image = imread(outputs_path);
    targets_image = imread(targets_path);
%     size(outputs_image)
%     size(targets_image)
%     figure;
%     imshow(outputs_image);
%     figure;
%     imshow(targets_image);
    %disp(pre_filename);
    tempfile.name = pre_filename;
    tempfile.psnr = psnr(outputs_image, targets_image);
    tempfile.ssim = ssim(outputs_image, targets_image);
    
    fl3_10t_10t_test = [fl3_10t_10t_test; tempfile];
    i = i + 1
end
