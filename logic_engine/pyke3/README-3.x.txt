This is the source directory for the version of Pyke that is ready for
Python3.x.

To use this:

Make a clone of the pre_2to3_r1 repository from mercurial:

    $ hg clone \
      http://pyke.hg.sourceforge.net:8000/hgroot/pyke/pre_2to3_r1 \
      pyke3

Run the provided script which runs 2to3 on the sources to convert them to
Python3.x.  (Note: the 2to3 program comes with both Python2.6 and Python3.x
and should be installed on your system when you install either of these
versions of Python):

    $ cd pyke3
    $ ./run_2to3 > /dev/null

Then you can either put pyke3 on your PYTHONPATH, or install it.  But note
that putting pyke3 on your PYTHONPATH also allows you to more easily run the
examples!

Putting the converted pyke3 on your PYTHONPATH:

    $ pwd > ~/.local/lib/python3.1/site-packages/pyke3.pth

Or, to install Pyke:

    $ python3.1 setup.py install

Please do _not_ commit any changes from pyke3, as 2to3 has changed nearly all
of the source files! 

Enjoy!
