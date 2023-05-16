close all; clear;

N = 1920; M = 1080;
[X, Y] = sl.Grid(N,M,8e-6,8e-6);
[TH, R] = cart2pol(Y, X);

% for i = 0:15
%     U = CircAp(N, M, 0.3, 0.3, i*pi/2);
% 
%     H = sl.DMD_Hol(U, X, Y, 5e3, 5.2e3, 0, 1);
%     H = H';
% 
%     H = padarray(H,[2000,2000]);
% 
%     Uf = fft2(H);
%     If = abs(fftshift(Uf));
% 
%     figure(1); imagesc(If(2680:2820,3120:3280)); colormap turbo; axis image off;
% 
%     exportgraphics(figure(1),'Images/PhaseChange.gif','Append',true);
%     i
% end

i=0;

U = CircAp(N, M, 0.3, 0.3, i*pi/4);

H = sl.DMD_Hol(U, X, Y, 5e3, 5.2e3, 0, 1);
H = H';

sl.Fullscreen(H, 2);

function y = CircAp(N, M, d, rad, theta)
% Note that N, M is resolution,
% d is distance between apertures (between 0 and 1), and rad is
% radial size (between 0 and 1 [see square aperture code for details])

    c = zeros(M,N);
    secsz = N*d;
    x=(size(c(:,1:secsz)));
    y=x(1);
    x=x(2);
    [X, Y] = meshgrid(-x/2:x/2-1,-y/2:y/2-1);
    [TH, R] = cart2pol(Y, X);
    ap=zeros(size(R));
    ap(R<y*rad/2)=1;
    
    c(:,(N/2-secsz):(N/2-1))=sign(c(:,(N/2-secsz):(N/2-1)) + ap);
    c(:,(N/2):(N/2+secsz-1))=sign(c(:,(N/2):(N/2+secsz-1)) + ap).*exp(1j*theta);

    y = c;

end