##########
imgblender
##########

A Python package for blending image data.


***********************
Why did you write this?
***********************
I've been working on some code to procedurally generate images and
video. It is getting pretty bloated, and the blending modes seemed
like something reasonable to peel off. I have no idea if its useful
to anyone else, but here it is.


**********************
How do I run the code?
**********************
You can clone the repository to your local system and play with it.
It's intended to be used as a module by other python applications. You
can find examples of how to use it in the examples directory.


***************
Is it portable?
***************
It should be. It's pure Python, and the main library it uses is
numpy.


***************************************
Can I install this package from pipenv?
***************************************
Yes, but imgblender is not currently available through PyPI. You
will need to clone the repository to the system you want to install
imgblender on and run the following::

    pip install path/to/local/copy

Replace `path/to/local/copy` with the path for your local clone of
this repository.


*********************************************************
How do I install opencv-python or numpy on macOS Big Sur?
*********************************************************
If you are getting an error when trying to install either `opencv-python`
or `numpy` when running on macOS Big Sur, try running the following
before the install::

    export SYSTEM_VERSION_COMPAT=1

Big Sur changed the major version number for macOS from ten to eleven,
which seems to cause a problem for some versions of `pip` and `numpy`.
Setting the `SYSTEM_VERSION_COMPAT` environment variable to one tells
macOS to report its version as `10.16` rather than `11.0`.


***********************
How do I run the tests?
***********************
The `precommit.py` script in the root of the repository will run the
unit tests and a few other tests beside. Otherwise, the unit tests
are written with the standard unittest module, so you can run the
tests with::

    python -m unittest discover tests


********************
How do I contribute?
********************
At this time, this is code is really just me exploring and learning.
I've made it available in case it helps anyone else, but I'm not really
intending to turn this into anything other than a personal project.

That said, if other people do find it useful and start using it, I'll
reconsider. If you do use it and see something you want changed or
added, go ahead and open an issue. If anyone ever does that, I'll
figure out how to handle it.
