# Python > 3.10 needed (swich statement)
# Format time matrix
# [timestep, info] info=coord, velocity, acceleration, ...
# FIRST LINE / Path can be changed

"""
There are two ways to plot strain rates.
Set STRFLG to 1 using *DATABASE_EXTENT_BINARY so that strains are written directly to the output databases. Write very high resolution output to D3THDT or ELOUT for a few elements of interest. Select these elements using *DATABAE_HISTORY_... . Set N3THDT=1 in *DATABASE_EXTENT_BINARY to minimize output. After you run the job, read d3thdt into LS-Prepost and plot a strain time history. Choose "Oper" in the time history window and select "differentiate" (you may have to toggle off and then back on "differentiate" to get it to activate), and then select "Apply".  This approach allows you to choose a thru-thickness location in shells (lower, middle, upper).
Use Fcomp > SRate to produce a fringe plot of strain rate computed from nodal displacements. Next, plot a time history of strain rate using History > Scalar. This strain rate corresponds to the midsurface of the shell. Method 1 must be used to get strain rates at other through-thickness locations.
Using either method, the accuracy of the strain rate is dependent upon the resolution of the output. The units of strain rate are strain per time unit used in the model."""

layer=0, 1 or 2