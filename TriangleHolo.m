
close all; clear;

N = 1920; M = 1080;
[X, Y] = sl.Grid(N,M,8e-6,8e-6);

U = TriAp(N, M, 0.3);
H = sl.DMD_Hol(U, X, Y, 5e3, 5.2e3, 0, 1);
H = H';
    
U_f = fft2(U);
Int_f = abs(fftshift(Uf));
% Fourier transform of field U

figure(1); imagesc(H); colormap gray; axis image off;

sl.Fullscreen(H, 2);

function y = TriAp(N, M, l)
    U = zeros(M,N);

    for b = 0:M*l
        a = round( (1/sqrt(3))*b );
        for a = 0:a
            i = round(b + M/2 - M*l/2);
            U(i, round(N/2+a))=1;
            U(i, round(N/2-a))=1;
        end 
    end

    y = U;
end
