### Directional Distance Field for Modeling the Difference between 3D Point Clouds [[OpenReview](https://openreview.net/forum?id=lEkFq4RUCX&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2024%2FConference%2FAuthors%23your-submissions))]

Here we provide the examples for three tasks, including **shape reconstruction**, **rigid registration** and **scene flow estimation.**

##### Requirements

```
pytorch             #1.10.0+cu111
pytorch3d           #0.6.2
open3d
trimesh
point-cloud-utils
```

##### Shape Reconstruction

`python exp_shape_reconstruction.py`

The reconstructed point cloud and trained model is saved at [example/shape_reconstruction](example/shape_reconstruction).

##### Rigid Registration

run `python exp_registration.py` to get the RE and TE.

*(Note the rotation matrix here is the transpose of the GT rotation matrix.*)

##### Scene Flow Estimation

run `python exp_scene_flow.py` to get the EPE, ACC-0.05, ACC-0.1, and Outliers.
