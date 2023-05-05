close all; clear;

N = 1920; M = 1080;
[X, Y] = sl.Grid(N,M,8e-6,8e-6);
[TH, R] = cart2pol(Y, X);

U = CircAp(N, M, 1, 0.3);

H = sl.DMD_Hol(U, X, Y, 5e3, 5.2e3, 0, 1);
H = H';

figure(1); imagesc(H); colormap gray; axis image off;

sl.Fullscreen(H, 2);

function y = CircAp(N, M, n, rad)

    c = zeros(M,N);
    x=(size(c(:,1:N/n)));
    y=x(1);
    x=x(2);
    [X, Y] = meshgrid(-x/2:x/2-1,-y/2:y/2-1);
    [TH, R] = cart2pol(Y, X);
    ap=zeros(size(R));
    ap(R<y*rad/2*n)=1;
    apend = size(ap);
    apend = apend(2);
    
    for i = 1:n
        st=1+((i-1)*apend);
        stp=i*apend;
        c(:,st:stp)=ap;
    end
    y = c;

end