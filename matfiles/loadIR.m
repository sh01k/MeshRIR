function [pos_mic, pos_src, fullIR] = loadIR(folderName)
    load(append(folderName, "pos_mic.mat"),'pos_mic');
    load(append(folderName, "pos_src.mat"),'pos_src');

    numMic = size(pos_mic, 1);

    load(append(folderName, "ir_0.mat"), 'ir');
    numSrc = size(ir, 1);
    irLen = size(ir, 2);

    fileList = dir(folderName);

    allIR = zeros(numMic, numSrc, irLen);
    irIndices = zeros(numMic, 1);
    disp('Loading IR files...')
    idx = 1;
    for i = 1:size(fileList, 1)
       if strncmp(fileList(i).name, 'ir_', 3)
           %disp(fileList(i).name)
           load(append(folderName, fileList(i).name),'ir');
           allIR(idx, :, :) = ir;
           irIndices(idx) = str2double(extractBetween(fileList(i).name,'ir_','.mat'));
           idx = idx + 1;
       end
    end

    fullIR = zeros(numSrc, numMic, irLen);
    for i = 1:numMic
        fullIR(:,irIndices(i)+1,:) = allIR(i,:,:);
    end
end