load "graphsettings.gnuh"
#set terminal epslatex size 3,2 color colortext
set output "numdist_insert.tex"

set yrange [0:6]
set ytics 0,1,5

#plot "nodes.dat" using ($5/($2*$2)):xtic(1) lc rgb "#00ff00" lt 1,\
     #"nodes.dat" using ($6/($2*$2)):xtic(1) lc rgb "#ff0000" lt 1

set ylabel "number of distance comparisons\nin tree \\emph\{construction\} only\n(normalized by the original cover tree)"
#divided by\n number of distance comparisons in original cover tree"
#set key center top outside

#tinyimages ancestor_train really is 85002878

set label  "19.1" at screen 0.400,screen 0.85 #6,5.5
set arrow from screen 0.47, screen 0.735 to screen 0.515, screen 0.785 nohead front lc rgb "#ffffff" lw 8pt
set arrow from screen 0.47, screen 0.75 to screen 0.515, screen 0.8 nohead front
set arrow from screen 0.47, screen 0.72 to screen 0.515, screen 0.77 nohead front

set arrow from 0,1 to 9,1 nohead lt 3 lc rgb "#999999"

plot "nodes.dat" using (1*($5)/($5)):xtic(1) lc rgb "#770000" lt 1 title "original",\
     "nodes.dat" using (1*($7)/($5)):xtic(1) lc rgb "#007700" lt 1 title "simplified",\
     "nodes.dat" using (1*($13)/($5)):xtic(1) lc rgb "#0000ff" lt 1 title "ancestor"

