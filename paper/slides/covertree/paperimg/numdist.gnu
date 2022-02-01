load "graphsettings.gnuh"
set output "numdist.tex"

set yrange [0:]
set ylabel "number of distance comparisons\nin tree \\emph\{construction\} and \\emph\{query\}\n(normalized by $n^2$)" offset 2
set key center top outside
unset key

set arrow from 0,1 to 9,1 nohead lt 3 lc rgb "#999999"

plot "nodes.dat" using (1*($5+$6)/($2*$2)):xtic(1) lc rgb "#770000" lt 1 title "original",\
     "nodes.dat" using (1*($7+$8)/($2*$2)):xtic(1) lc rgb "#007700" lt 1 title "simplified",\
     "nodes.dat" using (1*($13+$14)/($2*$2)):xtic(1) lc rgb "#0000ff" lt 1 title "ancestor"

set output "numdist-each.tex"
plot "nodes.dat" using (1*($5+$6)):xtic(1) lc rgb "#ff0000" lt 1 title "original",\
     "nodes.dat" using (1*($7+$8)):xtic(1) lc rgb "#00ff00" lt 1 title "simplified",\
     "nodes.dat" using (1*($13+$14)):xtic(1) lc rgb "#0000ff" lt 1 title "ancestor"

#removed from nodes.dat
#1000000x10  1000000     10      xxx         xxx             xxx             xxx             xxx         65716079    27717039612 xxx             xxx             xxx             xxx            xxx
#calhousing      20640       8       27072       1035256         18048650        1002794         5881191     817331      3823598     1437417         5289380         835059          4170253        2009944
