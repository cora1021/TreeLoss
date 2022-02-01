load "graphsettings.gnuh"
set output 'nodes.tex'


set ylabel "fraction of nodes in the original cover tree \n required for the simplified cover tree"

set yrange [0:1]
set ytics autofreq 1 0.1

plot "nodes.dat" using ($2/$4):xtic(1) lc rgb "#0000ff" lt 1

