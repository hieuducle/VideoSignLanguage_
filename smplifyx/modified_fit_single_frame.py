import time
import pickle

import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

import cv2

from smplifyx.optimizers import optim_factory
from smplifyx import fitting

import smplifyx.fitting
# from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

from collections import namedtuple

ImgSize = namedtuple('ImgSize', ['height', 'width'])

ImgSize.__new__.__defaults__ = (None,) * len(ImgSize._fields)


def fit_single_frame(keypoints,
                     body_model,
                     camera,
                     joint_weights,
                     body_pose_prior,
                     jaw_prior,
                     left_hand_prior,
                     right_hand_prior,
                     shape_prior,
                     expr_prior,
                     angle_prior,
                     img_size:ImgSize,
                     result_fn='out.pkl',
                     loss_type='smplify',
                     use_cuda=True,
                     init_joints_idxs=(9, 12, 2, 5),
                     use_face=True,
                     use_hands=True,
                     data_weights=None,
                     body_pose_prior_weights=None,
                     hand_pose_prior_weights=None,
                     jaw_pose_prior_weights=None,
                     shape_weights=None,
                     expr_weights=None,
                     hand_joints_weights=None,
                     face_joints_weights=None,
                     depth_loss_weight=1e2,
                     interpenetration=False,
                     coll_loss_weights=None,
                     focal_length=5000.,
                     side_view_thsh=25.,
                     rho=100,
                     vposer_latent_dim=32,
                     vposer_ckpt='',
                     use_joints_conf=False,
                     interactive=True,
                     visualize=False,
                     degrees=None,
                     batch_size=1,
                     dtype=torch.float32,
                     left_shoulder_idx=2,
                     right_shoulder_idx=5,
                     **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

    interpenetration = False
    visualize = False

    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    if degrees is None:
        degrees = [0, 90, 180, 270]

    if data_weights is None:
        data_weights = [1, ] * 5

    if body_pose_prior_weights is None:
        body_pose_prior_weights = [4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78]

    msg = (
        'Number of Body pose prior weights {}'.format(
            len(body_pose_prior_weights)) +
        ' does not match the number of data term weights {}'.format(
            len(data_weights)))
    assert (len(data_weights) ==
            len(body_pose_prior_weights)), msg

    if use_hands:
        if hand_pose_prior_weights is None:
            hand_pose_prior_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of hand pose prior weights')
        assert (len(hand_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg
        if hand_joints_weights is None:
            hand_joints_weights = [0.0, 0.0, 0.0, 1.0]
            msg = ('Number of Body pose prior weights does not match the' +
                   ' number of hand joint distance weights')
            assert (len(hand_joints_weights) ==
                    len(body_pose_prior_weights)), msg

    if shape_weights is None:
        shape_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
    msg = ('Number of Body pose prior weights = {} does not match the' +
           ' number of Shape prior weights = {}')
    assert (len(shape_weights) ==
            len(body_pose_prior_weights)), msg.format(
                len(shape_weights),
                len(body_pose_prior_weights))

    if use_face:
        if jaw_pose_prior_weights is None:
            jaw_pose_prior_weights = [[x] * 3 for x in shape_weights]
        else:
            jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                         jaw_pose_prior_weights)
            jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of jaw pose prior weights')
        assert (len(jaw_pose_prior_weights) ==
                len(body_pose_prior_weights)), msg

        if expr_weights is None:
            expr_weights = [1e2, 5 * 1e1, 1e1, .5 * 1e1]
        msg = ('Number of Body pose prior weights = {} does not match the' +
               ' number of Expression prior weights = {}')
        assert (len(expr_weights) ==
                len(body_pose_prior_weights)), msg.format(
                    len(body_pose_prior_weights),
                    len(expr_weights))

        if face_joints_weights is None:
            face_joints_weights = [0.0, 0.0, 0.0, 1.0]
        msg = ('Number of Body pose prior weights does not match the' +
               ' number of face joint distance weights')
        assert (len(face_joints_weights) ==
                len(body_pose_prior_weights)), msg

    if coll_loss_weights is None:
        coll_loss_weights = [0.0] * len(body_pose_prior_weights)
    msg = ('Number of Body pose prior weights does not match the' +
           ' number of collision loss weights')
    assert (len(coll_loss_weights) ==
            len(body_pose_prior_weights)), msg

    use_vposer = kwargs.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        # vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer, _ = load_model(vposer_ckpt, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
        vposer = vposer.to(device=device)
        vposer.eval()

    if use_vposer:
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    if use_joints_conf:
        joints_conf = keypoint_data[:, :, 2].reshape(1, -1)

    # Transfer the data to the correct device
    gt_joints = gt_joints.to(device=device, dtype=dtype)
    if use_joints_conf:
        joints_conf = joints_conf.to(device=device, dtype=dtype)

    # Create the search tree
    search_tree = None
    pen_distance = None
    filter_faces = None

    # Weights used for the pose prior and the shape prior
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights}
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    # The indices of the joints used for the initialization of the camera
    init_joints_idxs = torch.tensor(init_joints_idxs, device=device)

    edge_indices = [(5, 12), (2,9)]
    init_t = fitting.guess_init(body_model, gt_joints, edge_indices,
                                use_vposer=use_vposer, vposer=vposer,
                                pose_embedding=pose_embedding,
                                model_type=kwargs.get('model_type', 'smpl'),
                                focal_length=focal_length, dtype=dtype)

    camera_loss = fitting.create_loss('camera_init',
                                      trans_estimation=init_t,
                                      init_joints_idxs=init_joints_idxs,
                                      depth_loss_weight=depth_loss_weight,
                                      dtype=dtype).to(device=device)
    camera_loss.trans_estimation[:] = init_t

    loss = fitting.create_loss(loss_type=loss_type,
                               joint_weights=joint_weights,
                               rho=rho,
                               use_joints_conf=use_joints_conf,
                               use_face=use_face, use_hands=use_hands,
                               vposer=vposer,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               interpenetration=interpenetration,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    with fitting.FittingMonitor(
            batch_size=batch_size, visualize=visualize, **kwargs) as monitor:

        H = img_size.height
        W = img_size.width

        data_weight = 1000 / H
        # The closure passed to the optimizer
        camera_loss.reset_loss_weights({'data_weight': data_weight})

        # Reset the parameters to estimate the initial translation of the
        # body model
        body_model.reset_params(body_pose=body_mean_pose)

        # If the distance between the 2D shoulders is smaller than a
        # predefined threshold then try 2 fits, the initial one and a 180
        # degree rotation
        shoulder_dist = torch.dist(gt_joints[:, left_shoulder_idx],
                                   gt_joints[:, right_shoulder_idx])
        try_both_orient = shoulder_dist.item() < side_view_thsh

        # Update the value of the translation of the camera as well as
        # the image center.
        with torch.no_grad():
            camera.translation[:] = init_t.view_as(camera.translation)
            camera.center[:] = torch.tensor([W, H], dtype=dtype) * 0.5

        # Re-enable gradient calculation for the camera translation
        camera.translation.requires_grad = True

        camera_opt_params = [camera.translation, body_model.global_orient]

        camera_optimizer, camera_create_graph = optim_factory.create_optimizer(
            camera_opt_params,
            **kwargs)

        # The closure passed to the optimizer
        fit_camera = monitor.create_fitting_closure(
            camera_optimizer, body_model, camera, gt_joints,
            camera_loss, create_graph=camera_create_graph,
            use_vposer=use_vposer, vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False, return_verts=False)

        # Step 1: Optimize over the torso joints the camera translation
        # Initialize the computational graph by feeding the initial translation
        # of the camera and the initial pose of the body model.
        camera_init_start = time.time()
        cam_init_loss_val = monitor.run_fitting(camera_optimizer,
                                                fit_camera,
                                                camera_opt_params, body_model,
                                                use_vposer=use_vposer,
                                                pose_embedding=pose_embedding,
                                                vposer=vposer)

        if interactive:
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            tqdm.write('Camera initialization done after {:.4f}'.format(
                time.time() - camera_init_start))
            tqdm.write('Camera initialization final loss {:.4f}'.format(
                cam_init_loss_val))

        # If the 2D detections/positions of the shoulder joints are too
        # close the rotate the body by 180 degrees and also fit to that
        # orientation
        if try_both_orient:
            body_orient = body_model.global_orient.detach().cpu().numpy()
            flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
                cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
            flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()

            flipped_orient = torch.tensor(flipped_orient,
                                          dtype=dtype,
                                          device=device).unsqueeze(dim=0)
            orientations = [body_orient, flipped_orient]
        else:
            orientations = [body_model.global_orient.detach().cpu().numpy()]

        # store here the final error for both orientations,
        # and pick the orientation resulting in the lowest error
        results = []

        # Step 2: Optimize the full model
        final_loss_val = 0
        for or_idx, orient in enumerate(tqdm(orientations, desc='Orientation')):
            opt_start = time.time()

            new_params = defaultdict(global_orient=orient,
                                     body_pose=body_mean_pose)
            body_model.reset_params(**new_params)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):

                body_params = list(body_model.parameters())

                final_params = list(
                    filter(lambda x: x.requires_grad, body_params))

                if use_vposer:
                    final_params.append(pose_embedding)

                body_optimizer, body_create_graph = optim_factory.create_optimizer(
                    final_params,
                    **kwargs)
                body_optimizer.zero_grad()

                curr_weights['data_weight'] = data_weight
                curr_weights['bending_prior_weight'] = (
                    3.17 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 25:67] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 67:] = curr_weights['face_weight']
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    camera=camera, gt_joints=gt_joints,
                    joints_conf=joints_conf,
                    joint_weights=joint_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    stage_start = time.time()
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer)

                if interactive:
                    if use_cuda and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.time() - stage_start
                    if interactive:
                        tqdm.write('Stage {:03d} done after {:.4f} seconds'.format(
                            opt_idx, elapsed))

            if interactive:
                if use_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - opt_start
                tqdm.write(
                    'Body fitting Orientation {} done after {:.4f} seconds'.format(
                        or_idx, elapsed))
                tqdm.write('Body final loss val = {:.5f}'.format(
                    final_loss_val))

            # Get the result of the fitting process
            # Store in it the errors list in order to compare multiple
            # orientations, if they exist
            result = {'camera_' + str(key): val.detach().cpu().numpy()
                      for key, val in camera.named_parameters()}
            # body_model_forward = body_model.forward(body_pose=vposer.decode(pose_embedding, output_type='aa').reshape((1,-1)) if use_vposer else None)
            body_model_forward = body_model.forward(body_pose=vposer.decode(pose_embedding).get( 'pose_body').reshape((1,-1)) if use_vposer else None)
            if use_vposer:
                result['body_pose'] = pose_embedding.detach().cpu().numpy()
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model_forward.items() if val is not None})
            result.update({key: val.detach().cpu().numpy()
                           for key, val in body_model.named_parameters() if key not in result})

            results.append({'loss': final_loss_val,
                            'result': result})

        with open(result_fn, 'wb') as result_file:
            if len(results) > 1:
                min_idx = (0 if results[0]['loss'] < results[1]['loss']
                           else 1)
            else:
                min_idx = 0
            pickle.dump(results[min_idx]['result'], result_file, protocol=2)
        
        return results, result_fn




# Mano --------------------------------------------------update




