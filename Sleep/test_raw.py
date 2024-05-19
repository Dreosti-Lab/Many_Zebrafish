# Read the ".raw" file to find light ON and OFF transistions

raw_path = "/run/media/kampff/Crucial X9/Zebrafish/Sleep/220919_10_11_gria3xpo7/20220919-190935.raw"

#file = open(raw_path, "r", encoding="latin-1")
file = open(raw_path, "rb")

data = file.read(4000)
#data = file.read(4)

print(data)

file.close()

#FIN
