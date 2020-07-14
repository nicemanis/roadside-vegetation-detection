clc;	
clear;	
close all;

files = dir('*.jpg');
N = length(files);
for i = 1:N
    thisfile = files(i).name
    
    [rgbIm, storedColorMap] = imread(thisfile);
    [rows, columns, numberOfColorBands] = size(rgbIm); 
    figure,imshow(rgbIm);
    title('Input Image');
    
    hsvIm = rgb2hsv(rgbIm);
    figure,imshow(hsvIm);
    title('HSV Image');
    figure();
    hIm = hsvIm(:,:,1);
    subplot(2,3,1);imshow(hIm);title('H Image');
    sIm = hsvIm(:,:,2);
    subplot(2,3,2);imshow(sIm);title('S Image');
    vIm = hsvIm(:,:,3);
    subplot(2,3,3);imshow(vIm);title('V Image');

    [hueCounts, hueBinValues] = imhist(hIm);
    maxHueBinValue = find(hueCounts > 0, 1, 'last');
    maxCountHue = max(hueCounts);
    hHuePlot = subplot(2, 3, 4);
    area(hueBinValues, hueCounts, 'FaceColor', 'g');
    grid on; 
    xlabel('Hue Value'); 
    ylabel('Pixel Count'); 
    title('Histogram of Hue Image');

    [saturationCounts, saturationBinValues] = imhist(sIm); 
    maxSaturationBinValue = find(saturationCounts > 0,1,'last');

    maxCountSaturation = max(saturationCounts);

    SaturationPlot = subplot(2, 3, 5);
    area(saturationBinValues, saturationCounts, 'FaceColor', 'g'); 
    grid on; 
    xlabel('Saturation Value'); 
    ylabel('Pixel Count'); 
    title('Saturation');

    [valueCounts, valueBinValues] = imhist(vIm); 
    maxValueBinValue = find(valueCounts > 0, 1, 'last'); 
    maxCountValue = max(valueCounts);
    ValuePlot = subplot(2, 3, 6);
    area(valueBinValues, valueCounts, 'FaceColor', 'b'); 
    grid on; 
    xlabel('Value Value'); 
    ylabel('Pixel Count'); 
    title('Value');

    %%setting threshold
    hueThresholdLow = 0.15;
    hueThresholdHigh = 0.60;
    saturationThresholdLow = 0.36;
    saturationThresholdHigh = 1;
    valueThresholdLow = 0;
    valueThresholdHigh = 0.8;

    %%Green colour detection
    hueMask = (hIm >= hueThresholdLow) & (hIm <= hueThresholdHigh);
    saturationMask = (sIm >= saturationThresholdLow) & (sIm <= saturationThresholdHigh);
    valueMask = (vIm >= valueThresholdLow) & (vIm <= valueThresholdHigh);

    figure();
    subplot(1, 3, 1);
    imshow(hueMask, []);
    title('Hue Mask');
    subplot(1, 3, 2);
    imshow(saturationMask, []);
    title('Saturation Mask');
    subplot(1, 3, 3);
    imshow(valueMask, []);
    title('Value Mask');

    %%Set all axes to be the same width and height.
    maxCount = max([maxCountHue,  maxCountSaturation, maxCountValue]); 
    axis([hHuePlot SaturationPlot ValuePlot], [0 1 0 maxCount]);

    %%Plot all histograms in one histogram
    figure(); 
    plot(hueBinValues, hueCounts, 'b', 'LineWidth', 2); 
    grid on; 
    xlabel('Values'); 
    ylabel('Pixel Count'); 
    hold on; 
    plot(saturationBinValues, saturationCounts, 'g', 'LineWidth', 2); 
    plot(valueBinValues, valueCounts, 'r', 'LineWidth', 2); 
    title('Histogram of All Bands'); 
    maxGrayLevel = max([maxHueBinValue, maxSaturationBinValue, maxValueBinValue]);
    xlim([0 1])

    % Combine the masks to find where all 3 are true
    coloredObjectsMask = uint8(hueMask & saturationMask & valueMask);
    figure,imshow(coloredObjectsMask,[]);

    structuringElement = strel('disk', 2);
    coloredObjectsMask = imclose(coloredObjectsMask, structuringElement);
    figure,imshow(coloredObjectsMask,[]);
    coloredObjectsMask = imfill(logical(coloredObjectsMask), 'holes');
    figure,imshow(coloredObjectsMask,[]);
    coloredObjectsMask = cast(coloredObjectsMask, 'like', rgbIm);

    maskedImR = coloredObjectsMask .* rgbIm(:,:,1);
    maskedImG = coloredObjectsMask .* rgbIm(:,:,2);
    maskedImB = coloredObjectsMask .* rgbIm(:,:,3);

    maskedRGBIm = cat(3, maskedImR, maskedImG, maskedImB);
    figure,imshow(maskedRGBIm);
end
