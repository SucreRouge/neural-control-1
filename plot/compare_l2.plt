reset

set terminal pdf

cd "data"
set output "../figures/compare_l2.pdf"
set ylabel "loss"
set xlabel "P"
set key top right
plot "compare_l2.txt" u 1:2 t "analytic" w l lw 2, "" u 1:3 t "dt=0.1" w l  lw 2, "" u 1:4 t "dt=0.01" w l lw 2 lc 2 dt "-"

cd ".."
set output "/dev/null"
