#######################
imgblender Requirements
#######################

The purpose of this document is to detail the requirements for
imgblender, a Python module procedurally create image data. This is an
initial take for the purposes of planning. There may be additional
requirements or non-required features added in the future that are
not covered in this document.


*******
Purpose
*******
The purpose of imgblender is to merge two sets of image data together
using a blending algorithm. It can be used on either still image data
or video.


***********************
Functional Requirements
***********************
The following are the functional requirements for imgblender:

*   imgblender can perform common image blending operations.


**********************
Technical Requirements
**********************
The following are the technical requirements for imgblender:

*   imgeaser accepts "image data", which are three-dimensional array-
    like objects of floating-point numbers from 0 to 1, inclusive.
*   imgeaser outputs image data in array-like objects that can be
    saved by the imgwriter package.
