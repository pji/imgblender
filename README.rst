##########
imgblender
##########

A Python package for blending image data.


Why did you write this?
=======================
I've been working on some code to procedurally generate images and
video. It is getting pretty bloated, and the blending modes seemed
like something reasonable to peel off. I have no idea if its useful
to anyone else, but here it is.


How do I run the code?
======================
You can clone the repository to your local system and play with it.
It's intended to be used as a module by other python applications. You
can find examples of how to use it in the examples directory.


Is it portable?
===============
It should be. It's pure Python, and the main library it uses is
`numpy`.


Can I install this package from pipenv?
=======================================
Yes, but imgblender is not currently available through PyPI. You
will need to clone the repository to the system you want to install
`imgblender` on and run the following::

    pip install path/to/local/copy

Replace `path/to/local/copy` with the path for your local clone of
this repository.


How do I run the tests?
=======================
The `Makefile` in the root of the repository is set up to simplify
testing. To run just the unit tests::

    make test

To run the verbose version of the unit tests::

    make testv

To run the full suite of tests and checks::

    make pre


How do I contribute?
====================
At this time, this is code is really just me exploring and learning.
I've made it available in case it helps anyone else, but I'm not really
intending to turn this into anything other than a personal project.

That said, if other people do find it useful and start using it, I'll
reconsider. If you do use it and see something you want changed or
added, go ahead and open an issue. If anyone ever does that, I'll
figure out how to handle it.
