load "graphsettings.gnuh"
set output "numdist_query.tex"

set yrange [0:1.2]
set ytics autofreq 0 0.2

#plot "nodes.dat" using ($5/($2*$2)):xtic(1) lc rgb "#00ff00" lt 1,\
     #"nodes.dat" using ($6/($2*$2)):xtic(1) lc rgb "#ff0000" lt 1

set ylabel "number of distance comparisons divided by\n number of distance comparisons in original cover tree"
#set key center top outside


plot "nodes.dat" using (1*($6)/($6)):xtic(1) lc rgb "#ff0000" lt 1 title "original",\
     "nodes.dat" using (1*($8)/($6)):xtic(1) lc rgb "#00ff00" lt 1 title "simplified",\
     "nodes.dat" using (1*($14)/($6)):xtic(1) lc rgb "#0000ff" lt 1 title "ancestor"

