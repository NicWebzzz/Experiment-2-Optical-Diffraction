# Experiment-2-Optical-Diffraction

Here you will find all the code needed to perform the experiment. Feel free to explore some of the other holograms in the custom sl.m class!

Note that it is not expected of you to completely understand how the custom sl class functions work, just so long as you can use them to generate the holograms needed.

The sqare hologram code is annotated fully, however the format for generating the rest of the apertures is exactly the same.

ImageData.py is useful for analysing beam properties in detail (including analysing image cross-sections).

sl.m written by Keshaan Singh (1106433@students.wits.ac.za)


d = 11.9*sin(arctan(1080/1920)) ### Physical height of DMD, 
                                ### diagonal is specified as 11.9 mm

wl = 632.8e-6

f = 500

a = 0.3/2*d ### Acts as radius in the case of circle, and
            ### half the legnth of each side for the square aperture
