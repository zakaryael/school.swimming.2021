import os

#-------------------CFG FILE--------------#
# change just the json, geo and location of the results
def write_cfg(path,Dim):
    title='three_sphere_swimmer.cfg'

    template = """directory=toolboxes/fluid/moving_body/q-learning/three_sphere_swimmer
case.dimension={Dim}
[fluid]
filename=$cfgdir/three_sphere_swimmer.json 
mesh.filename=$cfgdir/three_sphere_swimmer.geo 
gmsh.hsize=3
#solver=Oseen #Oseen,Picard,Newton
ksp-monitor=true
pc-type=lu
ksp-type=preonly
#reuse-prec=1
ksp-maxit-reuse=20
snes-monitor=true
snes-maxit=100
define-pressure-cst=true
#define-pressure-cst.method=lagrange-multiplier#algebraic
verbose_solvertimer=1
#body.articulation.method=p-matrix
[fluid.alemesh]
pc-type=lu
[fluid.bdf]
order=2
[ts]
time-step=0.1#23
time-final=1
#restart=true
restart.at-last-save=true
time-initial=-0.1#0#-0.1
#save.freq=2"""
    context = {"Dim":Dim}
    with  open(os.path.join(path,title),'w') as myfile:
        myfile.write(template.format(**context))

#-------------------JSON FILE--------------#
# repeats the material section and body section for each new sphere
# the articulations are attached to the sphere on the right of the shrinking link
# !!! for the moment, the bounday conditions have not been tuned !!!
# indices are from 0 to N-1
def write_json(action,path,Dim):  
    title='three_sphere_swimmer.json'
    template="""{{
    "Name": "three_sphere_swimmer",
    "ShortName":"three_sphere_swimmer",
    "Models":
    {{
        "equations":"Stokes"
    }},
    "Materials":
    {{
        "Fluid":{{
            "physics":"fluid",
            "rho":"1",
            "mu":"1"
        }},
        "CirLeft":{{
            "physics":"body",
            "rho":1e-1
        }},
        "CirCenter":{{
            "physics":"body",
            "rho":1e-1
        }},
        "CirRight":{{
            "physics":"body",
            "rho":1e-1
        }}
    }},
    "Parameters":
    {{
        "eps":1e-10
    }},
    "BoundaryConditions":
    {{
        "velocity":
        {{
            "Dirichlet":
            {{
                "BoxWalls":
                {{
                    "expr":"{{0,0}}"
                }}
            }}
        }},
        "fluid":
        {{
            "body":
            {{
                "CircleCenter":
                {{
                    "markers":["CircleCenter"],
                    "materials":
                    {{
                        "names":["CirCenter"]
                    }}
                }},"""
    if action == "retract_right_arm" :
        velocityLeftSphere = "0*pulse(t,0,1-eps,1):t:eps"
        velocityRightSphere = "4*pulse(t,0,1-eps,1):t:eps"
    elif action == "extend_right_arm" :
        velocityLeftSphere = "0*pulse(t,0,1-eps,1):t:eps"
        velocityRightSphere = "-4*pulse(t,0,1-eps,1):t:eps"
    elif action == "retract_left_arm" :
        velocityLeftSphere = "4*pulse(t,0,1-eps,1):t:eps"
        velocityRightSphere = "0*pulse(t,0,1-eps,1):t:eps"
    elif action == "extend_left_arm" :
        velocityLeftSphere = "-4*pulse(t,0,1-eps,1):t:eps"
        velocityRightSphere = "0*pulse(t,0,1-eps,1):t:eps"


    template = template + """
                "CircleRight":
                {{
                    "markers":["CircleRight"],
                    "materials":
                    {{
                        "names":["CirRight"]
                    }},
                    "articulation":
                    {{
                        "body":"CircleCenter",
                        "translational-velocity":"+"""+str(velocityRightSphere)+""""
                    }}
                }},
                "CircleLeft":
                {{
                    "markers":["CircleLeft"],
                    "materials":
                    {{
                        "names":["CirLeft"]
                    }},
                    "articulation":
                    {{
                        "body":"CircleCenter",
                        "translational-velocity":"+"""+str(velocityLeftSphere)+""""
                    }}
                }}
            }}
        }}
    }},
    "PostProcess":
    {{
        "Measures":
        {{
            "Quantities":
            {{
                "names":"all"
            }}
        }}
    }}
}}"""

    context = {"action":action}
    with  open(os.path.join(path,title),'w') as myfile:
        myfile.write(template.format(**context))

#-------------------GEO FILE--------------#
# it repeats the same pattern for the creation of a new sphere, adding i*100 to get the new indices,
# while the indices of the bounding box are N*100
# the spheres are numbered from the left to the right, from 1 to N
# the flags are "BoxWalls" and "Fluid"
# "Spherei", for i from 0 to N-1 for the surface tag
# "Sphi", for i from 0 to N-1 for the volume tag
def write_geo(state,path,Dim):
    title = 'three_sphere_swimmer.geo'
    template = """
    RCircle = 1.;
    RDom = 10;

    h = 3;
    lcCircle = h/10;
    lcDom = h;

    Cx = 0;
    Cy = 0;

    Point(9) = {Cx,Cy,0,lcCircle};
    Point(10) = {Cx+RCircle,Cy,0,lcCircle};
    Point(11) = {Cx-RCircle,Cy,0,lcCircle};
    Circle(7) = {10,9,11};
    Circle(8) = {11,9,10};

    """
    if state[0] :
        template = template + """leftCenter = Cx-10*RCircle;"""
    else :
        template = template + """leftCenter = Cx-6*RCircle;"""
        
    template = template + """

    Point(1) = {leftCenter,Cy,0,lcCircle};
    Point(2) = {leftCenter+RCircle,Cy,0,lcCircle};
    Point(3) = {leftCenter,RCircle+Cy,0,lcCircle};
    Point(4) = {leftCenter,-RCircle+Cy,0,lcCircle};
    Circle(1) = {2,1,3};
    Circle(2) = {4,1,2};
    Circle(3) = {3,1,4};

    """
    if state[1] :
        template = template + """RightCenter = Cx+10*RCircle;"""
    else :
        template = template + """RightCenter = Cx+6*RCircle;"""    
    template = template + """

    Point(5) = {RightCenter,Cy,0,lcCircle};
    Point(6) = {RightCenter-RCircle,Cy,0,lcCircle};
    Point(7) = {RightCenter,RCircle+Cy,0,lcCircle};
    Point(8) = {RightCenter,-RCircle+Cy,0,lcCircle};
    Circle(4) = {7,5,6};
    Circle(5) = {6,5,8};
    Circle(6) = {8,5,7};

    Line Loop(7) = {1,2,3};
    Plane Surface(8) = {7};
    Line Loop(9) = {4,5,6};
    Plane Surface(10) = {9};
    Line Loop(11) = {7,8};
    Plane Surface(11) = {11};

    Line(12) = {11, 2};
    Line(13) = {10, 6};

    HalfSide = 50;
    HalfHeight = 20;

    Point(12) = {-HalfSide,-HalfHeight,0,lcDom};
    Point(13) = {HalfSide,-HalfHeight,0,lcDom};
    Point(14) = {HalfSide,HalfHeight,0,lcDom};
    Point(15) = {-HalfSide,HalfHeight,0,lcDom};

    Line(16) = {12, 13};
    Line(17) = {13, 14};
    Line(18) = {14, 15};
    Line(19) = {15, 12};
    Line Loop(20) = {16,17,18,19};

    Plane Surface(21) = {20, 7, 9, 11};


    Physical Curve("CircleLeft") = {1, 3, 2};
    Physical Curve("CircleCenter") = {7, 8};
    Physical Curve("CircleRight") = {4, 5, 6};
    Physical Curve("BoxWalls") = {16, 17, 18, 19};

    Physical Surface("CirLeft") = {8};
    Physical Surface("CirCenter") = {11};
    Physical Surface("CirRight") = {10};
    Physical Surface("Fluid") = {21};
        """
    context = {"state":state, "Dim":Dim}
    with  open(os.path.join(path,title),'w') as myfile:
        myfile.write(template)
    #else:
        #print('Dimension not coded')

def write_preconditioner(path,Dim):
    title="three_sphere_swimmer_preconditioner.cfg"
    template="""[fluid]
    preconditioner.attach-pmm=1
    #pc-view=1
    [fluid]
    ksp-type=fgmres#gcr#fgmres
    fgmres-restart=100
    # fieldsplit
    pc-type=fieldsplit
    fieldsplit-type=schur #additive, multiplicative, schur
    fieldsplit-schur-fact-type=upper#full#upper#full #diag, lower, upper, full
    fieldsplit-schur-precondition=self
    fieldsplit-fields=0->(0,2,3,4,5,6,7,8,9),1->(1)
    # block velocity
    [fluid.fieldsplit-0]
    ksp-type=preonly#gmres#preonly
    ksp-maxit=2#5#
    pc-type=fieldsplit
    fieldsplit-fields=0->(0),1->(2,3,4,5,6,7,8,9)
    fieldsplit-type=multiplicative #additive
    [fluid.fieldsplit-0.fieldsplit-0]
    ksp-type=gmres#preonly
    ksp-rtol=1e-3
    pc-type=gamg#gasm#lu#bjacobi#lu#gasm#boomeramg#gamg#lu
    #pc-gamg-nsmooths=2
    ksp-maxit=5#30#5#30
    #pc-factor-mat-solver-package-type=mumps
    #pc-factor-mumps.icntl-14=60
    [fluid.fieldsplit-0.fieldsplit-1]
    ksp-type=preonly
    pc-type=lu
    pc-factor-mat-solver-package-type=mumps
    [fluid.fieldsplit-1]
    ksp-type=gmres#preonly#gmres
    ksp-maxit=1#15
    ksp-rtol=1e-3
    pc-type=pmm
    pmm.pc-type=gamg#lu
    pmm.ksp-type=preonly#gmres
    #pmm.ksp-view=1
    #pmm.ksp-monitor=1
    #pc-factor-mat-solver-package-type=mumps
    #pc-factor-mumps.icntl-14=60
    
    """
    with  open(os.path.join(path,title),'w') as myfile:
        myfile.write(template)

