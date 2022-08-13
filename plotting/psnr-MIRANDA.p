set term post eps enh "Arial" 32 color
set output "psnr-MIRANDA.eps"
set datafile missing "-"
set key inside bottom right Left reverse
#set nokey

#set auto x
set xrange [0:1.5]
set yrange [0:100]
#set grid y

set style line 1 lt 1 lc rgb "black" lw 5
set style line 2 lt 2 lc rgb "red" lw 5
set style line 3 lt 3 lc rgb "blue" lw 5
set style line 4 lt 4 lc rgb "green" lw 5
set style line 5 lt 5 lc rgb "purple" lw 5

set xlabel "Bit Rate"
set ylabel "PSNR"

set style data lines
set boxwidth 0.9
#set xtic rotate by -45
plot 'MIRANDA.txt' using 1:2 ti col ls 2, '' u 3:4 ti col ls 3, '' u 5:6 ti col ls 4, '' u 7:8 ti col ls 5, '' u 9:10 ti col ls 1
