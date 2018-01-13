clear;
close all;

load('test_100.mat');
load('test_1000.mat');
load('test_10000.mat');
load('only_result_1000.mat');
load('v2_result_1000.mat');
load('onlyon_test_1000.mat');
load('fl_10_result.mat');
load('fl_100_result.mat');
load('origin.mat');
load('fl_10t_10t_test.mat');
load('fl3_10_10t_test.mat');
load('fl3_10t_10t_test.mat');

psnr = 0;
ssim = 0;
for index = 1:length(fl_10_result)
    psnr = psnr + fl_10_result(index).psnr;
    ssim = ssim + fl_10_result(index).ssim;
end
disp(psnr / 400);
disp(ssim / 400);

disp('-------------------------------')


psnr = 0;
ssim = 0;
for index = 1:length(fl_100_result)
    psnr = psnr + fl_100_result(index).psnr;
    ssim = ssim + fl_100_result(index).ssim;
end
disp(psnr / 400);
disp(ssim / 400);

disp('-------------------------------')

psnr = 0;
ssim = 0;
count = 0;
for index = 1:length(origin)
    if origin(index).psnr == Inf
        count = count + 1;
    else
        psnr = psnr + origin(index).psnr;
    end
    ssim = ssim + origin(index).ssim;
end
disp(psnr / (400 - count));
disp(ssim / 400);

disp('-------------------------------')


psnr = 0;
ssim = 0;
for index = 1:length(fl3_10_10t_test)
    psnr = psnr + fl3_10_10t_test(index).psnr;
    ssim = ssim + fl3_10_10t_test(index).ssim;
end
disp(psnr / 400);
disp(ssim / 400);

disp('-------------------------------')


psnr = 0;
ssim = 0;
for index = 1:length(fl3_10t_10t_test)
    psnr = psnr + fl3_10t_10t_test(index).psnr;
    ssim = ssim + fl3_10t_10t_test(index).ssim;
end
disp(psnr / 400);
disp(ssim / 400);

disp('-------------------------------')

psnr = 0;
ssim = 0;
for index = 1:length(fl_10t_10t_test)
    psnr = psnr + fl_10t_10t_test(index).psnr;
    ssim = ssim + fl_10t_10t_test(index).ssim;
end
disp(psnr / 400);
disp(ssim / 400);

disp('-------------------------------')
% psnr = 0;
% ssim = 0;
% for index = 1:length(result)
%     psnr = psnr + result(index).psnr;
%     ssim = ssim + result(index).ssim;
% end
% disp(psnr / 400);
% disp(ssim / 400);
% 
% disp('-------------------------------')
% 
% 
% psnr = 0;
% ssim = 0;
% for index = 1:length(result_1000)
%     psnr = psnr + result_1000(index).psnr;
%     ssim = ssim + result_1000(index).ssim;
% end
% disp(psnr / 400);
% disp(ssim / 400);
% 
% disp('-------------------------------')
% 
% 
% psnr = 0;
% ssim = 0;
% for index = 1:length(result_10000)
%     psnr = psnr + result_10000(index).psnr;
%     ssim = ssim + result_10000(index).ssim;
% end
% disp(psnr / 400);
% disp(ssim / 400);
% 
% disp('-------------------------------')
% 
% 
% 
% psnr = 0;
% ssim = 0;
% for index = 1:length(v2_result_1000)
%     psnr = psnr + v2_result_1000(index).psnr;
%     ssim = ssim + v2_result_1000(index).ssim;
% end
% disp(psnr / 2905);
% disp(ssim / 2905);
% 
% disp('-------------------------------')
% 
% 
% 
% psnr = 0;
% ssim = 0;
% for index = 1:length(only_result_1000)
%     psnr = psnr + only_result_1000(index).psnr;
%     ssim = ssim + only_result_1000(index).ssim;
% end
% disp(psnr / 2905);
% disp(ssim / 2905);
% 
% disp('-------------------------------')
% 
% 
% psnr = 0;
% ssim = 0;
% for index = 1:length(onlyon_result_1000)
%     psnr = psnr + onlyon_result_1000(index).psnr;
%     ssim = ssim + onlyon_result_1000(index).ssim;
% end
% disp(psnr / 2905);
% disp(ssim / 2905);
% 
% disp('-------------------------------')
