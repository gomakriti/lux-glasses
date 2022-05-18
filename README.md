This software can be used to acquire and process data produced by the luxottica glasses with integrated sensors.
To begin the acquisition run from the src folder the following command
```bash
python isee_decode_sz.py --mac {MAC_address} --duration {acquisition duration} > ../dataset/out.csv
```

Then in order to process the produced files with the parsed data run
```bash
python glasses_data_from_CSV.py 
```
 
This will produce two files,
* A fast file, found in dataset/fast, containing measures that change quickly (like those from the accelerometer, gyroscope, etc)
* A slow file, found in dataset/slow, containing measurements that change slowly (such as the temperature)

The script interactive_cutting.py can be used for labeling purposes, in particular it lets you load a file, cut the signal in pieces and produce
a pickle file as output (remember to omit file extension)

```bash
python interactive_cutting.py -i {input_file} -o {output_file}
```

The input file name should be the same you used when calling isee_decode_sz.py without the .csv extension

The remaning scripts are used to process the data and show how to train a neural network to solve a possible problem
 
