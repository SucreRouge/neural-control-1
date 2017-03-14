reset

set terminal pdf

set xlabel "time steps"
set ylabel "system state"
set key top left

set output "figures/compare_rect.pdf"
set xrange [5000 to 7000]
plot "data/compare_rect.txt" u 0:1 w l lw 2 t "PID", "" u 0:7 w l lw 2 t "Resilient", "" u 0:13 w l lw 2 t "Taylor"

set output "figures/compare_weights.pdf"
set yrange [-0.5 to 5]
set xrange [0 to 10100]
set arrow from 5000, graph 0 to 5000, graph 1 nohead dt "."
set arrow from 7000, graph 0 to 7000, graph 1 nohead dt "."
plot "data/compare_rect.txt" u 0:10 w l lw 2 lc 1 t "P", "" u 0:16 w l lw 2 lc 1 dt "-" t "", \
                     "" u 0:12 w l lw 2 lc 3 t "D", "" u 0:18 w l lw 2 lc 3 dt "-" t ""

set output "/dev/null"