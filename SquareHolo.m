close all; clear; % Clear any variables previously stored in memory

N = 1920; M = 1080;
% DMD resolution

[X, Y] = sl.Grid(N,M,8e-6,8e-6);
% Create a specialised meshgrid which takes into account the spacing
% between the micromirrors using the custom Grid class

U = SquareAp(N, M, 0.3); % setting len = 0.3 makes the square take up 30%
                         % of the vertical height

H = sl.DMD_Hol(U, X, Y, 5e3, 5.2e3, 0, 1); % Applying the hologram grating

H = H'; % Transpose matrix to make it horizontal

figure(1); imagesc(H); colormap gray; axis image off; % Plot the mask matrix

sl.Fullscreen(H, 2); % Send mask to DMD screen

function y = SquareAp(N, M, len)
% This function generates a square aperture, with sides of length 
% len*(smallest dimension of array), thus it is always between 0 and 1.
% Note that matlab functions must be placed at the end of the code.

    mask = zeros(M,N);

    a = floor(min(N,M)*len*0.5);

    mask(floor(M/2-a):floor(M/2+a),floor(N/2-a):floor(N/2+a))=1;

    y = mask;

end
