close all;
clear variables; 

addpath('matfiles');

set(0,'defaultAxesFontSize',18);
set(0,'defaultTextFontSize',18);

% Load IR data
folderName = "S1-M3969_mat/"; % "S32-M441_mat/"; % 
[posMic, posSrc, ir] = loadIR(folderName);

% Sampling rate
samplerate = 48000;

% Select IR data
srcIdx = 1;
micIdx = 1;
fprintf("Source position (m): %f\n", posSrc(srcIdx));
fprintf("Mic position (m): %f\n", posMic(micIdx));

% Plot geometry
figure;
hold on;
plot3(posMic(:,1),posMic(:,2),posMic(:,3),'.')
plot3(posSrc(:,1),posSrc(:,2),posSrc(:,3),'*')
hold off;
view(30, 30);
xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');

% IR plots
ir_plt = squeeze(ir(srcIdx, micIdx, :));
t = (0:length(ir_plt)-1)/samplerate;

figure;
plot(t,ir_plt);

% Extract xy-plane at z=0
[posMic_z, ir_z] = extract_plane(posMic, ir, 0);
posMicX = unique(posMic_z(:,1));
posMicY = unique(posMic_z(:,2));
numXY = [length(posMicX), length(posMicY)];
[posMicXY, irXY, ~] = sortIR(posMic_z, ir_z, numXY, posMicX, posMicY);

% Lowpass filter
irXY_lp = zeros(size(irXY));
for xx = 1:size(irXY,2)
    for yy = 1:size(irXY,3)
        irXY_lp(:,xx,yy,:) = lowpass(squeeze(irXY(:,xx,yy,:)).', 600, samplerate).';
    end
end

% Wave image
[~, idx] = max(abs(irXY_lp(:)));
[~, ~, ~, tIdx] = ind2sub(size(irXY_lp), idx);

x = [min(posMicX), max(posMicX)];
y = [min(posMicY), max(posMicY)];
%z_range = [-0.2, 0.2];

figure;
imagesc(x, y, squeeze(irXY_lp(srcIdx, :, :, tIdx+100)));
axis tight;
axis equal;
%caxis(z_range);
colormap(pink);
colorbar;
set(gca,'Ydir','normal');
xlabel('x (m)'); ylabel('y (m)');
