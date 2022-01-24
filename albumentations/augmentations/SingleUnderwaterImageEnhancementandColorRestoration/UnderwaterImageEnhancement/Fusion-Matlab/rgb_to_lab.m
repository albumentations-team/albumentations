function lab = rgb_to_lab(rgb)
cform = makecform('srgb2lab');
lab = applycform(rgb,cform);