#!/usr/bin/python
# sample perl script to create SGE jobs (sun grid engine)
#  for scanning a parameter space

import os

jobname = "F1_" # should be short
name = "" + jobname # name of shell scripts
res = "result_f1-EQLDIV"

submitfile = "submit_" + name + ".sh"
SUBMIT = open(submitfile,'w')
SUBMIT.write("#/bin/bash\n")

pwd=os.getcwd()

#number of epochs
e=10000
regstart = e/4
regend = e-e/20

i = 0
for l1 in [10**(-l1exp/10.0) for l1exp in range(35,60)]:
  for l_n in [2,3]:
    for normal in ([True] if i > 0 else [False, True]):
      epochs = e if  normal else 1
      result = res + "/" if normal else res + "test/"
      base_cmd = ["python src/mlfg_final.py -i ", str(i),
                  " -f ", result,
                  " -d data/f1-n-10k-1.dat.gz",
                  " --extrapol=data/f1-n-5k-1-test.dat.gz",
                  " --extrapol=data/f1-n-5k-1-2-test.dat.gz",
                  " --epochs=", str(epochs),
                  " --l1=", str(l1),
                  " --layers=", str(l_n),
                  " --iterNum=1",
                  " --reg_start=", str(regstart),
                  " --reg_end=", str(regend),
                  " -o",
                  ]
      cmd= "".join(base_cmd)

      script_fname = ((name + str(i)) if normal else name + "000_test") + ".sh"
      if normal:
        SUBMIT.write("./" + script_fname + "\n")
      with open(script_fname, 'w') as FILE:
        FILE.write("""\
#!/bin/bash
# %(jobname)s%(i)d
cd %(pwd)s

export OMP_NUM_THREADS=1
export PATH=${HOME}/bin:/usr/bin:${PATH}

if %(cmd)s; then
    rm -f %(script_fname)s
else
    touch %(script_fname)s.dead
fi
""" % locals())
      os.chmod(script_fname,0755)
    i += 1

SUBMIT.close()
os.chmod(submitfile,0755)
print "Jobs:" , i

with open("finished_" + name + ".sh",'w') as FINISHED:
  FINISHED.write("#!/bin/bash\nset -e\n" +
                 'grep "#" $(ls ' + res + '/*.res -1 | head -n 1) >'  + res + '/all.dat\n' +
                 "cat " + res + '/*.res | grep -v "#" >>' + res + '/all.dat\n' +
                 "cp " + __file__ + ' ' + res + '/\n' +
                 "rm finished_" + name+ '.sh ' + submitfile + '\n')
os.chmod("finished_" + name + ".sh",0755)
