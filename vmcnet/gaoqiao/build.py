from typing import Optional, Tuple, Union
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.networks as networks
import vmcnet.gaoqiao.dp as dp
import vmcnet.gaoqiao.open_feature_layer as open_feature_layer
import vmcnet.gaoqiao.jastrows as jastrows
import jax
import logging

def _get_activation(name: str):
    if name == "tanh":
        return jax.nn.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "silu":
        return jax.nn.silu
    raise ValueError(
        f"Unsupported activation {name!r}; expected one of ['tanh', 'relu', 'silu']."
    )

def build_network(
    n,
    charges,
    nspins: Tuple[int, ...],
    key,
    ndet,
    depth,
    h1,
    h2,
    nh,
    do_complex,
    gq_type: str = "ef",
    envelope_type: str = "ds_hz",
    feat_params: Optional[dict] = None,
    det_mode: str = "det",
    jastrow_type: str = "mlp",
    jastrow_mlp_nlayer: int = 4,
    jastrow_mlp_ndim: int = 32,
    activation_type: str = "silu",
    jastrow_activation_type: str = "tanh",
    dp_type: str = "full",
):
    activation_fn = _get_activation(activation_type)
    jastrow_activation_fn = _get_activation(jastrow_activation_type)

    hidden_dims = tuple((h1, h2) for _ in range(depth))
    ndim = 3
    natom = len(charges)
    full_det = True
    bias_orbitals = False
    make_envelope_kwargs = {"hiddens": [] if nh == 0 else [nh]}
    mes = dp.ManyElectronSystem(charges, nspins, dp_type)
    feat_params = {} if feat_params is None else dict(feat_params)

    feature_layer = open_feature_layer.make_open_features_ef(
        n,
        ndim,
        **feat_params,
    )

    if envelope_type == "ds_hz":
        envelope = envelopes.make_ds_hz_envelope(**make_envelope_kwargs,activation_fn=activation_fn)
    elif envelope_type == "ds_hz_pos":
        envelope = envelopes.make_ds_hz_envelope_positive(**make_envelope_kwargs,activation_fn=activation_fn)
    elif envelope_type == "iso":
        envelope = envelopes.make_isotropic_envelope()
    elif envelope_type == "ds_hz_deg":
        envelope = envelopes.make_ds_hz_degenerate_envelope(**make_envelope_kwargs,activation_fn=activation_fn)
    else:
        raise ValueError("envelope_type should be in ['ds_hz', 'ds_hz_pos', 'iso']")

    if jastrow_type == "mlp_res":
        jastrow = jastrows.make_mlp_jastrow(
            nspins=nspins,
            hiddenlayers_num=jastrow_mlp_nlayer,
            hiddenlayers_size=jastrow_mlp_ndim,
            activation_fn=jastrow_activation_fn,
        )
    elif jastrow_type == "simple_ee":
        jastrow = jastrows.make_simple_ee_jastrow(nspins=nspins)
    elif jastrow_type == "null":
        jastrow = jastrows.make_null_jastrow()
    else:
        raise ValueError("jastrow_type should be in ['mlp_res', 'simple_ee', 'null']")

    logging.info("wfn type: %s", gq_type)
    if gq_type == "ef_shrd_sym":
        ferminet_model = networks.make_fermi_net_model_ef_shrd_sym(
            n,
            ndim,
            nspins,
            feature_layer,
            hidden_dims,
            mes=mes,
            activation_fn=activation_fn,
        )
    elif gq_type == "ef_shrd_sym_ablation":
        ferminet_model = networks.make_fermi_net_model_ef_shrd_sym_ablation(
            n,
            ndim,
            nspins,
            feature_layer,
            hidden_dims,
            mes=mes,
            activation_fn=activation_fn,
        )
    else:
        raise ValueError(f"Unknown wfn_type: {gq_type!r}")

    # Original make_fermi_net path stays untouched for all old network types.
    network_init, signed_network, det_fn, network_options, orbitals, hz = (
        networks.make_fermi_net(
            n,
            ndim,
            nspins,
            charges,
            envelope=envelope,
            feature_layer=feature_layer,
            ferminet_model=ferminet_model,
            bias_orbitals=bias_orbitals,
            full_det=full_det,
            hidden_dims=hidden_dims,
            determinants=ndet,
            do_complex=do_complex,
            mes=mes,
            det_mode=det_mode,
            jastrow=jastrow,
        )
    )

    key, subkey = jax.random.split(key)
    params = network_init(subkey)

    return params, signed_network, det_fn, orbitals, hz
