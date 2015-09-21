#!/usr/bin/env python

import subprocess
import glob
def run():
    eps_files=glob.glob("*.eps")

    for fullname in eps_files:
        print(" ")
        file_base=fullname[:-4]  #strip extension
        
        pdf_name=file_base+'_psf2pdf.pdf'
        ps2pdf= "ps2pdf -dEPSFitPage "+file_base+'.eps ' + pdf_name
        print("command: %s" % ps2pdf)
        subprocess.call(ps2pdf, shell=True) #calls a unix shell
        pdfcrop= "pdfcrop "+ pdf_name
        print("command: %s" % pdfcrop)
        subprocess.call(pdfcrop, shell=True) #calls a unix shell
        remove_extra_pdf= "rm " + pdf_name
        subprocess.call(remove_extra_pdf, shell=True) #calls a unix shell

    print(" commands run!")

#relocated code for adding a question between commands:
   #print(" (n terminates, anything else continues")
    #s=raw_input()
    #if s=="n":
    #    break
    #print(" ")
    
