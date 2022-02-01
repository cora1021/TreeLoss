load "graphsettings.gnuh"

set output "cache-twit.tex"

set yrange [0:100]
set ylabel "percent"
set key outside

plot "cache-twit.dat" using ($2):xtic(1) lc rgb "#ff0000" lt 1,\
     "cache-twit.dat" using ($3):xtic(1) lc rgb "#00ff00" lt 1,\
     "cache-twit.dat" using ($4):xtic(1) lc rgb "#0000ff" lt 1
     #"cache-twit.dat" using ($5):xtic(1) lc rgb "#00ffff" lt 1

