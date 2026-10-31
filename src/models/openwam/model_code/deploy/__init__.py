# wam_interp: upstream re-exported engine/model_loader/server here (eager imports of the full model
# stack). The vendored package keeps no re-exports so obs_preprocess imports without the model;
# import deploy.engine / deploy.model_loader / deploy.denoise_schedule directly.
