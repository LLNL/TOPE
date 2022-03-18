from argparse import Namespace
tails = open("tail.txt")
ns_str = tails.read()
ns = eval(ns_str)
d = vars(ns)
if d["delta"] >= 1:
    delta = str(int(d["delta"]))
else:
    delta = str(d["delta"]).replace('.','')
if d["tau"] >= 1:
    tau = str(int(d["tau"]))
else:
    tau = str(d["tau"]).replace('.','')
if d["mu"] >= 1:
    mu = str(int(d["mu"]))
else:
    mu = str(d["mu"]).replace('.','')
RH = str(d["filter_radius"]).replace('.','')
Ny = str(d["Ny"])
filename = "optimized_design_" +"Ny"+Ny+"delta"+delta+"tau"+tau+"mu"+mu+"RH"+RH +".png"
print(filename)
