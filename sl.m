classdef sl
    methods (Static = true)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate X-Y coordinate arrays%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [X,Y] = Grid(N, M, dn, dm)
            x = -(N*dn)/2:dn:(N*dn)/2 - dn;
            y = -(M*dm)/2:dm:(M*dm)/2 - dm;
            [X, Y] = meshgrid(x, y);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate the Laguerre polynomial%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function v = LaguerreL(varargin)
            % Function inputs:
            if (nargin == 2)     % Evaluate classical Laguerre Polynomials
                n=varargin{1};
                k=0;
                x=varargin{2};
            elseif (nargin == 3) % Evaluate generalized Laguerre Polynomials
                n=varargin{1};
                k=varargin{2};
                x=varargin{3};
            else 
                error('Usage: >> LaguerreL(n:int,k:int,x:array)');
            end
            % Verify inputs
            if rem(n,1)~=0, error('n must be an integer.'); end
            if rem(k,1)~=0, error('k must be an integer.'); end
            if n < 0, error('n must be positive integer.'); end
            if k < 0, error('k must be positive integer.'); end
            % Initialize solution array
            v = zeros(size(x));
            % Compute Laguerre Polynomials
            GL = zeros( numel(x), n+1 );
            if n==0
                v(:) = 1.0;         % GL(0,k,x)
            elseif n==1
                v(:) = 1+k-x(:);    % GL(1,k,x)
            elseif n>1
                GL(:,1) = 1;        % GL(0,k,x)
                GL(:,2) = 1+k-x(:); % GL(1,k,x)
                for i = 2:n
                    GL(:,i+1) = ( (k+2*i-1-x(:)).*GL(:,i) + (-k-i+1).*GL(:,i-1) )/i;
                end
                v(:) = GL(:,i+1);   % GL(n,k,x)
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate Laguerre Gaussian mode at z=0%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [out] = LG(R,TH,w0,l,p)
            Norm = sqrt((2*factorial(p))/(pi*factorial(abs(l)+p)*w0^2));
            A = ((sqrt(2).*sqrt(R.^2))./(w0)).^(abs(l));
            L = sl.LaguerreL(p,abs(l),(2.*(R.^2))./w0^2);
            G = exp(-(sqrt(R.^2)./w0).^2);
            PHI = exp(-1i.*l.*TH);
            LG = Norm.*A.*L.*G.*PHI;
            [out]=LG;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate binary DMD hologram%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [out] = DMD_Hol(U, X, Y, gx, gy, rot, weight)
            if rot == 0
                A = abs(U); % correction factors
                A = (A*weight)./max(A(:));
                A = asin(A)/pi; % amplitude term for hologram
                Phi = angle(U); Phi = Phi/(2*pi); % phase term for hologram
                h = 0.5+0.5*sign(cos(2*pi*(gx*X+gy*Y)+2*pi*Phi)-cos(pi*A)); % generates hologram
                h(h==0.5) = 0; % binarises hologram
                normh = h/max(h(:));
                normh = normh';
            else
                U0 = U;
                U = imrotate(U,rot);
                U = U(size(U,1)/2 - size(U0,1)/2:size(U,1)/2 + size(U0,1)/2 - 1, ...
                      size(U,2)/2 - size(U0,2)/2:size(U,2)/2 + size(U0,2)/2 - 1);
                A = abs(U); % correction factors
                A = (A*weight)./max(A(:));
                A = asin(A)/pi; % amplitude term for hologram
                Phi = angle(U); Phi = Phi/pi; % phase term for hologram
                h = 0.5+0.5*sign(cos(2*pi*(gx*X+gy*Y)+pi*Phi)-cos(pi*A)); % generates hologram
                h(h==0.5) = 0; % binarises hologram
                normh = h/max(h(:));
                normh = normh';
            end
            out = normh;
    
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to display images on external display%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Fullscreen(image, device_number)
            ge = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment();
            gds = ge.getScreenDevices();
            height = gds(device_number).getDisplayMode().getHeight();
            width = gds(device_number).getDisplayMode().getWidth();

            if ~isequal(size(image,1),height)
                error(['Image must have vertical resolution of ' num2str(height)]);
            elseif ~isequal(size(image,2),width)
                error(['Image must have horizontal resolution of ' num2str(width)]);
            end

            try
                imwrite(image,[tempdir 'display.bmp']);
            catch
                error('Image must be compatible with imwrite()');
            end

            buff_image = javax.imageio.ImageIO.read(java.io.File([tempdir 'display.bmp']));

            if ~exist('fullscreenData','var')
                global fullscreenData;
            end

            if (length(fullscreenData) >= device_number)
                frame_java = fullscreenData(device_number).frame_java;
                icon_java = fullscreenData(device_number).icon_java;
                device_number_java = fullscreenData(device_number).device_number_java;
            else
                frame_java = {};
                icon_java = {};
                device_number_java = {};
            end

            if ~isequal(device_number_java, device_number)
                try frame_java.dispose(); end
                frame_java = [];
                device_number_java = device_number;
            end

            if ~isequal(class(frame_java), 'javax.swing.JFrame')
                frame_java = javax.swing.JFrame(gds(device_number).getDefaultConfiguration());
                bounds = frame_java.getBounds(); 
                frame_java.setUndecorated(true);
                frame_java.setAlwaysOnTop(true); % MC: should stop minimizing on lost focus. See: http://stackoverflow.com/questions/32048428/keep-the-jframe-open-on-a-dual-monitor-configuration-in-java
                icon_java = javax.swing.ImageIcon(buff_image); 
                label = javax.swing.JLabel(icon_java); 
                frame_java.getContentPane.add(label);
                %gds(device_number).setFullScreenWindow(frame_java); % MC: this is a problem
                frame_java.setSize(width, height);
                frame_java.setLocation( bounds.x, bounds.y ); 
            else
                icon_java.setImage(buff_image);
            end
            frame_java.pack
            frame_java.repaint
            frame_java.show

            fullscreenData(device_number).frame_java = frame_java;
            fullscreenData(device_number).icon_java = icon_java;
            fullscreenData(device_number).device_number_java = device_number_java;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate Zernike polynomials on disk of rmax%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Z] = Zernike(n,m,rho,theta,rho_max)
            rho = rho./max(max(rho_max));
            R = zeros(size(rho));

            m1 = abs(m);

            if mod(n-m1,2) == 1
                R = zeros(size(rho));
            else
                for k = 0:((n-m1)/2)
                    R = R + (-1)^k*factorial(n-k)/(factorial(k)*factorial((n+m1)/2-k)*factorial((n-m1)/2-k)).*rho.^(n-2*k); 
                end
            end

            if m >= 0
                Z = sqrt(2*(n+1)).*R.*cos(m.*theta);
%                   Z = R.*cos(m.*theta);
            else
                Z = sqrt(2*(n+1)).*R.*sin(m.*theta);
%                   Z = R.*sin(m.*theta);
            end

            Z(rho>1) = 0;

        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate a FFT Kolmogorov turbulence screen%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function y = Turb_Gen(Nx,Ny,D,Dr,pixel)
            % SCREEN: produce 2D grid of values for the phase on an SLM to simulate turbulence
            % the output is complex valued: the real and imaginary parts two separate 
            % to avoid complications with the FFT, the function is calculated on a square grid that fits 
            % inside the SLM screen grid
            % units are in m
            % 
            % Nx,Ny = number of points along x and y
            % r0 = Fried parameter [r0 = 0.185*(lambda^2/Cn2/z)^(3/5)]
            % D is the hard aperture of the field
            % Dr is the desired D/r0 
            % pixel = pixel size [micron]  ---- (typically 8 micron)

            % compute number of points for square area    
            r0 = D./(Dr);
            tal = min(Nx,Ny);
            Getal=tal;
            Delta = 1/pixel/Getal; % increment size for x and y

            % put zero (origin) between samples to avoid singularity
            [nx,ny] = meshgrid((1:Getal)-Getal/2-1/2);
            Modgrid = real(exp(-1i*pi*(nx+ny)));
            rr = (nx.*nx+ny.*ny)*Delta^2;

            % Square root of the Kolmogorov spectrum:
            qKol = 0.1517*Delta/r0^(5/6)*rr.^(-11/12);%

            f0 = (randn(Getal)+1i*randn(Getal)).*qKol/sqrt(2);
            f1 = Modgrid.*fft2(f0);
            % f1 = exp(1i.*real(f1));
            % subgrids 
            % coordinates and increment sizes (relative to Delta)
            ary = [-0.25,-0.25,-0.25,-0.125,-0.125,-0.125,0,0,0,0,0.125,0.125,0.125,0.25,0.25,0.25];
            bry = [-0.25,0,0.25,-0.125,0,0.125,-0.25,-0.125,0.125,0.25,-0.125,0,0.125,-0.25,0,0.25];
            dary = [0.25,0.25,0.25,0.125,0.125,0.125,0.25,0.125,0.125,0.25,0.125,0.125,0.125,0.25,0.25,0.25];
            dbry = [0.25,0.25,0.25,0.125,0.125,0.125,0.25,0.125,0.125,0.25,0.125,0.125,0.125,0.25,0.25,0.25];
            ss = (ary.*ary+bry.*bry)*Delta^2;
            qsKol = 0.1516*Delta/r0^(5/6)*ss.^(-11/12);
            f0 = (randn(1,16)+1i*randn(1,16)).*qsKol/sqrt(2);
            fn = f1; % zeros(Getal);
            for pp = 1:16
              eks = exp(1i*2*pi*(nx*ary(pp)+ny*bry(pp))/Getal);
              fn = fn + f0(pp)*eks*dary(pp)*dbry(pp);
            end
            y = zeros(Ny,Nx);
            y((Ny/2-Getal/2+1):(Ny/2+Getal/2),(Nx/2-Getal/2+1):(Nx/2+Getal/2))=real(fn);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to center an image by cropping around a user click%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Out] = Center(U, N, M)
            imagesc(U);
            [cx, cy] = ginput(1);
            Out = U(cx - N/2: cx + N/2, ...
                  cy - M/2: cy + M/2);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to crop an array about a point%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Out] = Crop(cx, cy, U, N, M)
            Out = U(cx - N/2: cx + N/2 - 1, ...
                  cy - M/2: cy + M/2 - 1); 
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to perform a Fourier transform using the discreet%
        % Fourier transform matrix                                  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [out] = DFT(U,f,dx,wl)
            % Computes Fourier transform matrix but scales to keep spatial coordinates
            % Outputs Fourier matrix and its inverse, then FT = F'*mode*F;

            L = size(U,1)*dx; % physical side length of image
            x = -L/2:dx:L/2-dx; % x coordinate system

            Lk = wl*f/dx; % side "length" at Fourier plane (inverse length)
            dk = wl*f/L; % sample "size" at Fourier plane (inverse length)
            k = -Lk/2:dk:Lk/2-dk; % spatial frequency coordinate system
            k = k./dk^2; %empirical

            F = exp(-1i*2*pi/size(U,1)).^(k'*x); %Fourier transform matrix: exp(-1i 2pi/d)^(k x')
            out = F'*U*F;
    %         Finv = conj(F); %Inverse Fourier transform matrix: exp(1i 2pi/d)^(k x')
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to perform an inverse Fourier transform using the%
        % discreet inverse Fourier transform matrix                 %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [out] = IDFT(U,f,dx,wl)
            % Computes Fourier transform matrix but scales to keep spatial coordinates
            % Outputs Fourier matrix and its inverse, then FT = F'*mode*F;

            L = size(U,1)*dx; % physical side length of image
            x = -L/2:dx:L/2-dx; % x coordinate system

            Lk = wl*f/dx; % side "length" at Fourier plane (inverse length)
            dk = wl*f/L; % sample "size" at Fourier plane (inverse length)
            k = -Lk/2:dk:Lk/2-dk; % spatial frequency coordinate system
            k = k./dk^2; %empirical

            IF = conj(exp(-1i*2*pi/H).^(k'*x)); %Fourier transform matrix: exp(-1i 2pi/d)^(k x')
            out = IF*U*IF';
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to encode an image into a discretized concurrence basis %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [out] = Im2C(Filename, Base)
                I = imread(Filename);
                I = im2bw(I, 0.5);
                I = imresize(I, [2^8, 2^8]);

                for row = 1:2^8
                    Bits = squeeze(I(row, :));
                    count = 1
                    for Byte = 1:(2^8)/Base
                        Chunk = Bits(Byte:Byte+Base - 1);
                        Dec = 0;
                        for Element = 1:Base
                            Dec = Dec + Chunk(Element)*(2^(Base-Element));
                        end
                        Cdx(row, count) = Dec + 1;
                        count = count + 1;
                    end
                end
                out = Cdx;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate an optical frozen wave %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Psi] = FrozenWave(R, TH, k, L, Q, z, Fz, ldx, w0)

        Psi = zeros(size(R));
        z_range = 0:L/(size(Fz,2)):L - L/(size(Fz,2));

        N = round(abs((k-Q)*L/(2*pi)));
        if N > 30
            N = 30;
        end
        n_range = -N:1:N;

        for n = n_range
            krn = sqrt((k)^2 - (Q + 2*pi*n/L)^2);
            An = (1/L)*sum(sum(Fz.*exp(-1i*2*pi*n*z_range/L)));
    %         Psi = Psi + exp(1i*Q*z).*An.*besselj(abs(ldx), krn*R).* ...
    %                     exp(1i*ldx*TH).*exp(1i*2*pi*n*z/L);

            Psi = Psi + exp(1i*Q*z).*An.*exp(-1i.*krn.*R).* ...
                        exp(1i*ldx*TH).*exp(1i*2*pi*n*z/L).*exp(-R.^2/w0^2);

        end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function to generate Arizzon type 3 SLM holograms %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Hol] = SLM_Hol(U, X, Y, gx, gy)  
            A=abs(U); % Amplitude
            A=A./max(A(:)); % Amplitude normalization
            A=round(A*500)/500; % Amplitude bit depth
            Phi=angle(U); % Phase
            f=zeros(size(A)); % Amplitude envelope preallocation 
            load fx;
            for m=1:size(A,1)
                for n=1:size(A,2)
                    temp=A(m,n);
                    f(m,n)=fx(round(temp/0.002+1)); % Calling inverted Bessel
                end
            end
            Hol=f.*sin(Phi+2*pi*(gx*X+gy*Y)); 
        end
    end
end
