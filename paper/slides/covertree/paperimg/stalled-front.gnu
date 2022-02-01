load "graphsettings.gnuh"
set output "stalled-front.tex"

set ylabel "stalled CPU cycle rate\n(stalled cycles / total cycles)"

set yrange [0:1]
set ytics autofreq 1 0.2

plot "cache.dat" using ($2/100):xtic(1) lc rgb "#663300" lt 1,\
     "cache.dat" using ($8/100):xtic(1) lc rgb "#cc9900" lt 1,\
     "cache.dat" using ($11/100):xtic(1) lc rgb "#0000ff" lt 1,\
     "cache.dat" using ($5/100):xtic(1) lc rgb "#7777ff" lt 1



