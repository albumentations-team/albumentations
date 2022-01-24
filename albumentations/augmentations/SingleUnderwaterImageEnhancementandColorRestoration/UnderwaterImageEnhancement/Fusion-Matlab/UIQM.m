function uiqm = UIQM(img)

c1 = 0.0282;
c2 = 0.2953;
c3 = 3.5753;

uicm = UICM(img);
uism = UISM(img);
uiconm = UIConM(img);

uiqm = c1 * uicm + c2 * uism + c3 * uiconm;
