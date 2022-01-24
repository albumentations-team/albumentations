function rgb = lab_to_rgb(lab)
cform = makecform('lab2srgb');
rgb = applycform(lab,cform);