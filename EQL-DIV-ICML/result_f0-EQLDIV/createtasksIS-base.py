#!/usr/bin/python
# sample perl script to create SGE jobs (sun grid engine)
#  for scanning a parameter space

import os

jobname = "FG1_" # should be short
name = "" + jobname # name of shell scripts
res = "result_fg1a-fg"

mem = "2000"
#maxtime = "4:00:00"

submitfile = "submit_" + name + ".sh"
SUBMIT = open(submitfile,'w')
SUBMIT.write("#/bin/bash\n")

pwd=os.getcwd()

e=4000

i = 0
for iter in (range(0,1)):
  for lr in [0.01]:
    for l_n in [3]:
      for n_n in [10]:
        for l1 in [10**(-l1exp/10.0) for l1exp in range(35,60)]:
          for l2 in [0]:
            for c in [""]:
              for regstart in [500]:
                regend = 3500
                for batch in [20]:
                  for normal in ([True] if i > 0 else [False, True]):
                    epochs = e if  normal else 1
                    result = res + "/" if normal else res + "test/"
                    base_cmd = ["python src/mlfg_final.py -i ", str(i),
                                " -f ", result,
                                " -d data/fg1a-n-10k-1.dat.gz",
                                " --extrapol=data/fg1a-n-5k-1-test.dat.gz",
                                " --extrapol=data/fg1a-n-5k-1-1_5-test.dat.gz",
                                " --extrapol=data/fg1a-n-5k-1-2-test.dat.gz",
                                " --epochs=", str(epochs),
                                " --lr=", str(lr),
                                " --l1=", str(l1),
                                " --l2=", str(l2),
                                " --layers=", str(l_n),
                                " --nodes=", str(n_n),
                                " --batchsize=", str(batch),
                                " --iterNum=", str(iter),
                                c,
                                " --reg_start=", str(regstart),
                                " --reg_end=", str(regend)
                                ]
                    if iter==1:
                        base_cmd.append(" -o")
                    cmd= "".join(base_cmd)

                    script_fname = ((name + str(i)) if normal else name + "000_test") + ".sh"
                    job_fname = ((name + str(i)) if normal else name + "000_test") + ".sh.sub"

                    if normal:
                        SUBMIT.write("condor_submit " + job_fname + "\n")
                    with open(script_fname, 'w') as FILE:
                        FILE.write("""\
#!/bin/bash
# %(jobname)s%(i)d
cd %(pwd)s

export OMP_NUM_THREADS=1
export PATH=${HOME}/bin:/usr/bin:${PATH}

if %(cmd)s; then
    rm -f %(script_fname)s
    rm -f %(job_fname)s
else
    touch %(job_fname)s.dead
fi
""" % locals())
                    with open(job_fname, 'w') as FILE:
                        FILE.write("""\
executable = %(pwd)s/%(script_fname)s
error = %(script_fname)s.err
output = %(script_fname)s.out
log = %(script_fname)s.log
request_memory=%(mem)s
request_cpus=1
queue
""" % locals())
                    os.chmod(script_fname,0755)
                    i += 1

SUBMIT.close()
os.chmod(submitfile,0755)
print "Jobs:" , i

with open("re" + submitfile,'w') as RESUBMIT:
  RESUBMIT.write("for F in " + name + "*.sh; do qsub $F; done;\n")

with open("finished_" + name + ".sh",'w') as FINISHED:
  FINISHED.write("#!/bin/bash\nset -e\n" +
                 "cat " + res + '/*.res >' + res + '/all.dat\n' +
                 "mv " + name + "*.* " + res + '/\n' +
                 "cp " + __file__ + ' ' + res + '/\n' +
                 "rm finished_" + name+ '.sh\n' +
                 "rm re" + submitfile + ' \n')
os.chmod("finished_" + name + ".sh",0755)
