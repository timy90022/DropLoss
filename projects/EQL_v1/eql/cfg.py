def add_eql_config(cfg):
    """
    Add config for EQL.
    """
    cfg.MODEL.ROI_HEADS.LAMBDA = 0.0010062
    cfg.MODEL.ROI_HEADS.PRIOR_PROB = 0.001

    # 0.0010061576850323984
    # 0.0010162192618827222
    # legacy cfg key (make model compatible with previous ckpt)
    cfg.MODEL.ROI_HEADS.FREQ_INFO = ""
