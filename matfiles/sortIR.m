function [sortedPos, sortedIR, sortIdx] = sortIR(pos, ir, numXY, posX, posY)
    numSrc = size(ir, 1);
    irLen = size(ir, 3);
    sortIdx = zeros(numXY(1), numXY(2));
    
    for i = 1:numXY(2)
        xIdx = find(abs(pos(:,2)-posY(i)) < eps);
        [~, sorter] = sort(pos(xIdx,1));
        xIdxSort = xIdx(sorter);

        sortIdx(:,i) = xIdxSort;
    end

    sortedPos = reshape(pos(sortIdx, :), [numXY,3]);
    sortedIR = reshape(ir(:, sortIdx, :), [numSrc, numXY, irLen]);
end