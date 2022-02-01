load "graphsettings.gnuh"
set output "parallel-ancestor-build.tex"

#set terminal postscript enhanced "Times-Roman" 25
#set output "parallel-ancestor-build.pdf"
#set xtic rotate by 0

set ylabel "normalized tree \\emph\{construction\} time"

set logscale y 2
set yrange [0.03125:4]
set ytics ("$2^{+1}$" 2, "$2^{+0}$" 1, "$2^{-1}$" 0.5, "$2^{-2}$" 0.25, "$2^{-3}$" 0.125, "$2^{-4}$" 0.0625)
set grid ytics lt 0 lw 1 lc rgb "#bbbbbb"

set arrow from graph 0.20, graph 0.87 to graph 0.17, graph 0.78 lc rgb "#000000" lw 0.5 lt 1 arrowstyle 1

set label "\\tiny 1" at graph 0.1475, graph 0.75
set label "\\tiny 2" at graph 0.1775, graph 0.68
set label "\\tiny 4" at graph 0.205, graph 0.52
set label "\\tiny 8" at graph 0.2325, graph 0.445
set label "\\tiny 16" at graph 0.26, graph 0.405
set label "number of processors" at graph 0.15, graph 0.9

#set arrow from 1,0.1 to 2,5

plot "parallel-ancestor-build.dat" using ($2/$2):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-ancestor-build.dat" using ($3/$2):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-ancestor-build.dat" using ($4/$2):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-ancestor-build.dat" using ($5/$2):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-ancestor-build.dat" using ($6/$2):xtic(1) lc rgb "#0000ff" lt 1


#corel           4.0         2.9         1.8         1.2         1.2         1.6
#artificial4     0.7         0.6         0.45        0.5         0.46                            modified
#calhousing      1.84        1.02        0.65        0.36        0.35
#faces           0.3         0.19        0.145       0.10        0.11
