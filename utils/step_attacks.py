import eagerpy as ep
from typing import Union, Tuple, Optional, Any
from tensorboard import TensorBoard
from foolbox.attacks import BoundaryAttack, HopSkipJump

class BoundaryStep(BoundaryAttack):
    def __init__(self, *args, **kwargs):
        super(BoundaryStep, self).__init__(*args, **kwargs)
        
    def reset(self, model, substitute, inputs, criterion, starting_points, early_stop: Optional[float] = None):
        self.model = model
        self.sub = substitute
        self.originals, self.restore_type = ep.astensor_(inputs)
        del inputs
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
            return True, self.best_advs  # pragma: no cover
        self.converged = self.atleast_kd(self.converged, self.ndim)

        self.unnormalized_source_directions = self.originals - self.best_advs
        self.source_norms = ep.norms.l2(self.flatten(self.unnormalized_source_directions), axis=-1)
        self.source_directions = self.unnormalized_source_directions / self.atleast_kd(self.source_norms, self.ndim)

        # only check spherical candidates every k steps
        self.check_spherical_and_update_stats = self.iter % self.update_stats_every_k == 0

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

        # put the conditional at the beginning, calling step will check if its a turn
        # to check spherical or normal candidate
        if self.check_spherical_and_update_stats:
            self.spherical_is_adv = self.is_adversarial(self.spherical_candidates)
            self.stats_spherical_adversarial.append(self.spherical_is_adv)
            # TODO: algorithm: the original implementation ignores those samples
            # for which spherical is not adversarial and continues with the
            # next iteration -> we estimate different probabilities (conditional vs. unconditional)
            # TODO: thoughts: should we always track this because we compute it anyway
            self.stats_step_adversarial.append(self.is_adv)
            # HERE
            return False, self.spherical_candidates
        else:
            spherical_is_adv = None
            
        # track stats based same for normal and spherical calls, in the object's attributes
        
        # in theory, we are closer per construction
        # but limited numerical precision might break this
        distances = ep.norms.l2(flatten(originals - candidates), axis=-1)
        closer = distances < source_norms
        is_best_adv = ep.logical_and(is_adv, closer)
        is_best_adv = atleast_kd(is_best_adv, ndim)

        cond = converged.logical_not().logical_and(is_best_adv)
        best_advs = ep.where(cond, candidates, best_advs)

        tb.probability("converged", converged, step)
        tb.scalar("updated_stats", check_spherical_and_update_stats, step)
        tb.histogram("norms", source_norms, step)
        tb.probability("is_adv", is_adv, step)
        if spherical_is_adv is not None:
            tb.probability("spherical_is_adv", spherical_is_adv, step)
        tb.histogram("candidates/distances", distances, step)
        tb.probability("candidates/closer", closer, step)
        tb.probability("candidates/is_best_adv", is_best_adv, step)
        tb.probability("new_best_adv_including_converged", is_best_adv, step)
        tb.probability("new_best_adv", cond, step)

        if check_spherical_and_update_stats:
            full = stats_spherical_adversarial.isfull()
            tb.probability("spherical_stats/full", full, step)
            if full.any():
                probs = stats_spherical_adversarial.mean()
                cond1 = ep.logical_and(probs > 0.5, full)
                spherical_steps = ep.where(
                    cond1, spherical_steps * self.step_adaptation, spherical_steps
                )
                source_steps = ep.where(
                    cond1, source_steps * self.step_adaptation, source_steps
                )
                cond2 = ep.logical_and(probs < 0.2, full)
                spherical_steps = ep.where(
                    cond2, spherical_steps / self.step_adaptation, spherical_steps
                )
                source_steps = ep.where(
                    cond2, source_steps / self.step_adaptation, source_steps
                )
                stats_spherical_adversarial.clear(ep.logical_or(cond1, cond2))
                tb.conditional_mean(
                    "spherical_stats/isfull/success_rate/mean", probs, full, step
                )
                tb.probability_ratio(
                    "spherical_stats/isfull/too_linear", cond1, full, step
                )
                tb.probability_ratio(
                    "spherical_stats/isfull/too_nonlinear", cond2, full, step
                )

            full = stats_step_adversarial.isfull()
            tb.probability("step_stats/full", full, step)
            if full.any():
                probs = stats_step_adversarial.mean()
                # TODO: algorithm: changed the two values because we are currently tracking p(source_step_sucess)
                # instead of p(source_step_success | spherical_step_sucess) that was tracked before
                cond1 = ep.logical_and(probs > 0.25, full)
                source_steps = ep.where(
                    cond1, source_steps * self.step_adaptation, source_steps
                )
                cond2 = ep.logical_and(probs < 0.1, full)
                source_steps = ep.where(
                    cond2, source_steps / self.step_adaptation, source_steps
                )
                stats_step_adversarial.clear(ep.logical_or(cond1, cond2))
                tb.conditional_mean(
                    "step_stats/isfull/success_rate/mean", probs, full, step
                )
                tb.probability_ratio(
                    "step_stats/isfull/success_rate_too_high", cond1, full, step
                )
                tb.probability_ratio(
                    "step_stats/isfull/success_rate_too_low", cond2, full, step
                )

        tb.histogram("spherical_step", spherical_steps, step)
        tb.histogram("source_step", source_steps, step)
        tb.close()
        return restore_type(best_advs)
        
        return done, self.candidates
    
    def switch(self, actionID, candidates):
        if actionID == 0:
            is_adv = self.is_adversarial(candidates)
        if actionID == 1:
            is_adv = self.sub_adversarial(candidates)
        if actionID == 2:
            is_adv = self.mis_adversarial(candidates)
        return is_adv