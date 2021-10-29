#!/bin/bash


# dims, m_sizes=8,16,..., rk_orders='RK1','RK2', dg_orders=1,2
# python main.py "taylor_tests" $m_size $dg_order $rk_order

#echo $m_size $dg_order $rk_order 
#python main.py "elastic_wave_taylor_test" #$m_size $dg_order $rk_order

#echo $m_size $dg_order $rk_order 
#python main.py "LOH1_taylor_test" #$m_size $dg_order $rk_order



for x in 0 8 16 24 32 40 48 56
do
        for y in 0 8 16 24 32
        do
               
        python main.py "LOH1_ic_tests" $x $y
			
        done
done

for m_size in 8
do
        for dg_order in 1
        do
                for rk_order in 1
                do
			
			 
			#echo $m_size $dg_order $rk_order 
	                #python main.py "LOH1_taylor_test" $m_size $dg_order $rk_order
			
                done
        done
done

