import eagerpy as ep
from typing import Union, Tuple, Optional, Any
from foolbox.tensorboard import TensorBoard
from foolbox.attacks import BoundaryAttack

class BoundaryStep(BoundaryAttack):
    def __init__(self, *args, **kwargs):
        super(BoundaryStep, self).__init__(*args, **kwargs)
        
    def reset(self, model, substitute, inputs, criterion, misterion, starting_points, early_stop: Optional[float] = None):
        self.model = model
        self.sub = substitute
        self.originals, self.restore_type = ep.astensor_(inputs)
        del inputs
        self.tb = TensorBoard(logdir=self.tensorboard)
        self.early_stop = early_stop
        self.starting_points = starting_points
        self.iter = 0
        self.done = False

        self.criterion = self.get_criterion(criterion) 
        self.misterion = self.get_criterion(misterion)
        self.is_adversarial = self.get_is_adversarial(self.criterion, self.model)
        self.sub_adversarial = self.get_is_adversarial(self.criterion, self.sub)
        self.mis_adversarial = self.get_is_adversarial(self.misterion, self.model)

        if starting_points is None:
            raise ValueError("no starting_point provided")
        else:
            self.best_advs = ep.astensor(self.starting_points)

        is_adv = self.is_adversarial(self.best_advs)
        if not is_adv:
            raise ValueError("starting_point is not adversarial")
        del starting_points

        self.N = len(self.originals)
        self.ndim = self.originals.ndim
        self.spherical_steps = ep.ones(self.originals, self.N) * self.spherical_step
        self.source_steps = ep.ones(self.originals, self.N) * self.source_step

        # tb.scalar("batchsize", N, 0)

        # create two queues for each sample to track success rates
        # (used to update the hyper parameters)
        self.stats_spherical_adversarial = self.ArrayQueue(maxlen=100, N=self.N)
        self.stats_step_adversarial = self.ArrayQueue(maxlen=30, N=self.N)

        self.bounds = self.model.bounds

    def step(self, actionID):
        self.iter += 1
        self.converged = self.source_steps < self.source_step_convergance
        if self.converged or self.iter > self.steps:
            self.tb.close()
            return True, self.best_advs  # pragma: no cover
        self.converged = self.atleast_kd(self.converged, self.ndim)

        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(self.flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / self.atleast_kd(self.source_norms, self.ndim)

        # only check spherical candidates every k steps
        self.check_spherical_and_update_stats = self.iter % self.update_stats_every_k == 0

        if not self.check_spherical_and_update_stats:
            self.candidates, self.spherical_candidates = self.draw_proposals(
                self.bounds,
                self.originals,
                self.best_advs,
                self.unnormalized_source_directions,
                self.source_directions,
                self.source_norms,
                self.spherical_steps,
                self.source_steps,
                )
            self.is_adv = self.switch(actionID, self.candidates)
        else:
            assert self.spherical_candidates is not None
            self.spherical_is_adv = self.is_adversarial(self.spherical_candidates)
            self.stats_spherical_adversarial.append(self.spherical_is_adv)
            # TODO: algorithm: the original implementation ignores those samples
            # for which spherical is not adversarial and continues with the
            # next iteration -> we estimate different probabilities (conditional vs. unconditional)
            # TODO: thoughts: should we always track this because we compute it anyway
            self.stats_step_adversarial.append(self.is_adv)
            self.update_stats()
            return False, self.spherical_candidates
        
        # in theory, we are closer per construction
        # but limited numerical precision might break this
        self.distances = ep.norms.l2(self.flatten(self.originals - self.candidates), axis=-1)
        self.closer = self.distances < self.source_norms
        is_best_adv = ep.logical_and(self.is_adv, self.closer)
        is_best_adv = self.atleast_kd(is_best_adv, self.ndim)

        cond = self.converged.logical_not().logical_and(is_best_adv)
        self.best_advs = ep.where(cond, self.candidates, self.best_advs)

        self.tb.probability("converged", self.converged, self.iter)
        self.tb.scalar("updated_stats", self.check_spherical_and_update_stats, self.iter)
        self.tb.histogram("norms", self.source_norms, self.iter)
        self.tb.probability("is_adv", self.is_adv, self.iter)
        self.tb.histogram("candidates/distances", self.distances, self.iter)
        self.tb.probability("candidates/closer", self.closer, self.iter)
        self.tb.probability("candidates/is_best_adv", is_best_adv, self.iter)
        self.tb.probability("new_best_adv_including_converged", is_best_adv, self.iter)
        self.tb.probability("new_best_adv", cond, self.iter)

        self.tb.histogram("spherical_step", self.spherical_steps, self.iter)
        self.tb.histogram("source_step", self.source_steps, self.iter)
        
        return False, self.candidates
    
    def switch(self, actionID, candidates):
        if actionID == 0:
            is_adv = self.is_adversarial(candidates)
        if actionID == 1:
            is_adv = self.sub_adversarial(candidates)
        if actionID == 2:
            is_adv = self.mis_adversarial(candidates)
        return is_adv
    
    def update_stats(self):
        self.tb.probability("spherical_is_adv", self.spherical_is_adv, self.iter)
        full = self.stats_spherical_adversarial.isfull()
        self.tb.probability("spherical_stats/full", full, self.iter)
        if full.any():
            probs = self.stats_spherical_adversarial.mean()
            cond1 = ep.logical_and(probs > 0.5, full)
            self.spherical_steps = ep.where(cond1, self.spherical_steps * self.step_adaptation, self.spherical_steps)
            self.source_steps = ep.where(cond1, self.source_steps * self.step_adaptation, self.source_steps)
            cond2 = ep.logical_and(probs < 0.2, full)
            self.spherical_steps = ep.where(cond2, self.spherical_steps / self.step_adaptation, self.spherical_steps)
            self.source_steps = ep.where(cond2, self.source_steps / self.step_adaptation, self.source_steps)
            self.stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))
            self.tb.conditional_mean("spherical_stats/isfull/success_rate/mean", probs, full, self.iter)
            self.tb.probability_ratio("spherical_stats/isfull/too_linear", cond1, full, self.iter)
            self.tb.probability_ratio("spherical_stats/isfull/too_nonlinear", cond2, full, self.iter)

        full = self.stats_step_adversarial.isfull()
        self.tb.probability("step_stats/full", full, self.iter)
        if full.any():
            probs = self.stats_step_adversarial.mean()
            # TODO: algorithm: changed the two values because we are currently tracking p(source_step_sucess)
            # instead of p(source_step_success | spherical_step_sucess) that was tracked before
            cond1 = ep.logical_and(probs > 0.25, full)
            self.source_steps = ep.where(cond1, self.source_steps * self.step_adaptation, self.source_steps)
            cond2 = ep.logical_and(probs < 0.1, full)
            self.source_steps = ep.where(cond2, self.source_steps / self.step_adaptation, self.source_steps)
            self.stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
            self.tb.conditional_mean("step_stats/isfull/success_rate/mean", probs, full, self.iter)
            self.tb.probability_ratio("step_stats/isfull/success_rate_too_high", cond1, full, self.iter)
            self.tb.probability_ratio("step_stats/isfull/success_rate_too_low", cond2, full, self.iter)
