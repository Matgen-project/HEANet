cd hea_4k
for i in `ls initial_p`;do
  cat gibbs_eng.csv | grep $i | head
  mkdir -p ../examples/initial_p/$i  ../examples/relaxed_p/$i
  for str in `cat gibbs_eng.csv | grep $i | head`;do
    cp "initial_p/$i/"`awk -F',' '{print $1}' <<< "$str"`".cif" ../examples/initial_p/$i
    cp "relaxed_p/$i/"`awk -F',' '{print $1}' <<< "$str"`".cif" ../examples/relaxed_p/$i
  done
done
