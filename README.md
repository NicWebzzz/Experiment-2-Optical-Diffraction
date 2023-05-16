# Experiment-2-Optical-Diffraction

Here you will find all the code needed to perform the experiment. Feel free to explore some of the other holograms in the custom sl.m class!

Note that it is not expected of you to completely understand how the custom sl class functions work, just so long as you can use them to generate the holograms needed.

The sqare hologram code is annotated fully, however the format for generating the rest of the apertures is exactly the same.

ImageData.py is useful for analysing beam properties in detail (including analysing image cross-sections).

sl.m written by Keshaan Singh (1106433@students.wits.ac.za)

def circfraun(r,a,wl,f):
    
    k = 2*pi/wl
    
    return ( jv(1,k*r*a/f)/(k*r*a/f) )**2


def squarefraun(X,Y,a,wl,f):
    
    k=2*pi/wl
    
    return ( sin(k*X*a/f)*sin(k*Y*a/f)/(k*X*a/f)/(k*Y*a/f) )**2


squareU = squarefraun(X,Y,a,wl,f)


circU = circfraun(R,a,wl,f)
