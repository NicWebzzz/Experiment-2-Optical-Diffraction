close all; clear;

N = 1920; M = 1080;
[X, Y] = sl.Grid(N,M,8e-6,8e-6);
[TH, R] = cart2pol(Y, X);

mag=-2;
ast = mag*sl.Zernike(2,-2,R,TH,M*8e-6);

tref = 0*sl.Zernike(4,2,R,TH,M*8e-6);

test=0*sl.Zernike(4,0,R,TH,M*8e-6);

U = sl.LG(R,TH,1e-3,3,3);

U = U .* exp(1i*ast).* exp(1i*tref).* exp(1i*test);

H = sl.DMD_Hol(U, X, Y, 5e3, 5.2e3, 0, 1);

figure(1); imagesc(abs(H')); colormap gray; axis image off;

% sl.Fullscreen(H', 2);