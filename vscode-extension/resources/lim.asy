// lim.svg -- Asymptote version of the LIM logo icon
// Compile with: asy -f svg lim.asy
// or asy -f png lim.asy

settings.tex="xelatex";

size(128,128);
import fontsize;
import roundedpath;

texpreamble("\usepackage{xcolor}");
texpreamble("\definecolor{blue}{RGB}{121,192,255}");
texpreamble("\definecolor{orange}{RGB}{206,145,120}");
texpreamble("\definecolor{green}{RGB}{181,206,168}");
texpreamble("\usepackage{fontspec}");
texpreamble("\newfontfamily\DVSBold{DejaVu Sans Bold}");
texpreamble("\newfontfamily\DVSMMono{DejaVu Sans Mono Bold}");
texpreamble("\newcommand\limtext{\DVSBold\textcolor{blue}{L}\textcolor{orange}{I}\textcolor{green}{M}}");

filldraw(roundedpath(unitsquare, 0.09), rgb("#1e1e1e"));

label("\limtext", (0.5,0.6), fontsize(58pt));

label("\textcolor{blue}{\DVSMMono >>> /undo}", (0.5,0.25), fontsize(19.8pt));
