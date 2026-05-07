Franka Arm Dynamics Simulation
==============================

This folder contains a compact Franka arm dynamics simulation script:

    franka_cleaned_sim.py

The script provides two standard demos:

1. Free-space PD demo
   - No wall or contact constraint is enabled.
   - A Cartesian end-effector velocity command is converted to joint velocity by resolved-rate IK.
   - The integrated joint reference is tracked by the joint-space PD controller through the dynamics model.
   - The default free-space motion is a longer piecewise Cartesian trajectory. The motion stays close to the original initial configuration and uses small velocities to avoid singular regions.

2. Wall-contact HPFC demo
   - The robot starts from the same initial configuration as the original contact demo.
   - Before contact, the robot approaches the wall using resolved-rate IK + joint-space PD.
   - During contact, the controller switches to Hybrid Force/Position Control (HPFC): normal force is controlled while tangential motion follows the wall surface.
   - The wall is the curved/bump wall used in the original contact demo.


Directory Layout
----------------

Recommended location:

    E:\git\Franka_arm_dynamics\CPBox\Franka_arm

Expected files after setup:

    Franka_arm/
        franka_cleaned_sim.py
        README_franka_sim.txt
        results/                    # generated automatically when GIFs are saved
            free_space_pd.gif
            wall_hpfc.gif

The results folder is created automatically if it does not already exist.


Requirements
------------

Python packages:

    numpy
    scipy
    matplotlib
    pillow

The wall-contact demo is designed to use the CPBox contact solver:

    CPBox.Solvers.CP_Judice_Child.CPJudice

When franka_cleaned_sim.py is placed inside:

    <repo_root>\CPBox\Franka_arm

it automatically adds <repo_root> to sys.path so the CPBox import works when the script is launched directly from PyCharm or from the command line.

If CPBox is not available, the script contains a small fallback contact solver so the demo can still run. The fallback is only for basic testing; the intended solver for the repository is CPJudice.


Run
---

From the target folder:

    cd /d E:\git\Franka_arm_dynamics\CPBox\Franka_arm
    python franka_cleaned_sim.py

Default behavior:

    python franka_cleaned_sim.py

runs both demos and saves GIF files to:

    E:\git\Franka_arm_dynamics\CPBox\Franka_arm\results\free_space_pd.gif
    E:\git\Franka_arm_dynamics\CPBox\Franka_arm\results\wall_hpfc.gif

Run only one demo:

    python franka_cleaned_sim.py --demo free
    python franka_cleaned_sim.py --demo wall

Run both demos without saving GIFs:

    python franka_cleaned_sim.py --demo both --no-save-gif

Show the Matplotlib animation window:

    python franka_cleaned_sim.py --demo wall --show

Save GIFs to a custom folder:

    python franka_cleaned_sim.py --demo both --results-dir results

The default output folder is based on the script location, not the current PyCharm project root. If the script is run from E:\download, the GIFs will be saved in E:\download\results. Put the script in E:\git\Franka_arm_dynamics\CPBox\Franka_arm to save outputs in the repository results folder.


Initial Configuration
---------------------

Both demos use the same initial joint configuration:

    q0 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571 + pi / 2, 0.0]

This is defined in:

    default_initial_state()

Change the initial joint configuration there if a new starting pose is needed.


Free-space Demo Settings
------------------------

Free-space settings are in:

    run_free_space_test()

Current default:

    model.enable_wall_contact = False
    model.steps_total = 420
    model.free_space_velocity_segments = [
        (140, np.array([0.000, 0.030, 0.000])),
        (280, np.array([0.035, 0.000, 0.000])),
        (420, np.array([0.000, 0.000, -0.020])),
    ]

The free-space control pipeline is:

    desired Cartesian EE velocity
        -> solve_ik_velocity()
        -> integrate theta_ref
        -> pd_joint_torque()
        -> joint dynamics integration

This means the arm is not moved by directly overwriting joint states. IK creates the joint reference, and the joint-space PD controller drives the simulated dynamics.

To test a different free-space movement, edit model.free_space_velocity_segments. Each tuple has this format:

    (end_step, np.array([vx, vy, vz]))

For example:

    model.free_space_velocity_segments = [
        (150, np.array([0.020, 0.000, 0.000])),
        (300, np.array([0.000, 0.020, 0.000])),
        (450, np.array([0.000, 0.000, -0.015])),
    ]

Use small Cartesian velocities and avoid large movements that stretch the arm into near-singular poses.


Wall-contact Demo Settings
--------------------------

Wall-contact settings are in:

    run_wall_hpfc_test()

Current default:

    model.enable_wall_contact = True
    model.steps_total = 692
    model.free_space_velocity = np.array([0.05, 0.0, 0.0])
    model.contact_tangent_velocity = np.array([0.0, 0.0, -0.07])
    model.desired_normal_force = 2.5

The wall geometry is:

    model.x_wall = 0.80
    model.wall_bump_amp = -0.12
    model.wall_bump_center_z = 0.73
    model.wall_bump_sigma_z = 0.035

The wall surface is modeled as:

    x_surface = x_wall + wall_height(y, z)

Set the bump amplitude to zero for a flat wall:

    model.wall_bump_amp = 0.0

Relevant wall/contact functions:

    wall_height()
    wall_height_gradient()
    wall_signed_distance()
    contact_frame_at_point()
    update_contact_state()


Changing Contact Number and Contact Position
--------------------------------------------

The contact point definition is in:

    run_wall_hpfc_test()

Current default:

    model.contact_points_local = np.array([[0.0, 0.0, 0.0]], dtype=float)

The contact points are expressed in the end-effector/flange local frame.

The shape is:

    (N, 3)

where N is the number of contact points. One row means one contact point.

Single contact point at the EE/TCP:

    model.contact_points_local = np.array([
        [0.0, 0.0, 0.0],
    ], dtype=float)

Two contact points around the tool:

    model.contact_points_local = np.array([
        [0.0, -0.05, 0.10],
        [0.0,  0.05, 0.10],
    ], dtype=float)

After changing contact_points_local, the script automatically resizes the internal contact state arrays during simulation.

The contact point list is used by:

    get_contact_points_world()
    wall_signed_distance()
    update_contact_state()
    contact_point_linear_jacobian()
    contact_jacobian_at_config()
    contact_regularizer()
    solve_contact_force()

For N active contacts, the stacked contact force vector has 3N entries:

    [lambda_n_1, lambda_t1_1, lambda_t2_1,
     lambda_n_2, lambda_t1_2, lambda_t2_2,
     ...]


Controller Summary
------------------

Free-space mode:

    resolved-rate IK + joint-space PD

Contact mode:

    Hybrid Force/Position Control + contact force solve

Before contact, the robot approaches the wall using the same IK + PD pipeline as the free-space case. Once contact is detected, HPFC controls the wall-normal direction by force and the tangent direction by motion.

Main controller-related functions:

    solve_ik_velocity()
    pd_joint_torque()
    hpfc_projection_matrix()
    hpfc_torque()
    step_dynamics()
    simulate_episode()


Generated Outputs
-----------------

The script prints a short summary for each demo, including:

    initial joint configuration
    end-effector start position
    end-effector end position
    end-effector movement
    wall distance phi for the contact case
    final contact debug information

GIF outputs are saved by default:

    results/free_space_pd.gif
    results/wall_hpfc.gif

The GIF saving uses the full simulation trajectory. There is no frame skipping or GIF stride option in the current script.


Troubleshooting
---------------

1. CPBox import error

   Place franka_cleaned_sim.py in:

       <repo_root>\CPBox\Franka_arm

   and run it from there. The script will add <repo_root> to sys.path automatically.

2. GIF saved in the wrong folder

   The output folder is relative to the script file location. Move the script to the intended repository folder or pass --results-dir.

3. Contact does not happen

   Check these values in run_wall_hpfc_test():

       model.x_wall
       model.wall_bump_amp
       model.free_space_velocity
       model.steps_total
       model.contact_points_local

   Also check the printed initial wall distance phi. Positive phi means the contact point is outside the wall. Zero or negative phi means contact or penetration.

4. Motion becomes unstable

   Reduce Cartesian velocity, reduce steps_total, or move the initial configuration away from singular regions. For contact tuning, start with small changes to desired_normal_force, force_kp, force_ki, task_kp, and task_kd.
