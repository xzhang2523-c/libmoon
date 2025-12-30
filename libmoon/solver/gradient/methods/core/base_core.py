class BaseCore:
    core_name: str = "BaseCore"

    def get_alpha(self, Jacobian, losses, idx=None):
        """
        Compute the objective weights alpha.
        Args:
            Jacobian: torch.Tensor, shape (n_obj, n_var)
            losses: torch.Tensor, shape (n_obj,)
            idx: int, preference index (optional)
        Returns:
            alpha: torch.Tensor, shape (n_obj,)
        """
        raise NotImplementedError("get_alpha must be implemented in subclass.")

    def get_alpha_array(self, Jacobian_array, losses_array, *args, **kwargs):
        """
        Batch compute objective weights alpha.
        Args:
            Jacobian_array: torch.Tensor, shape (n_prob, n_obj, n_var)
            losses_array: torch.Tensor, shape (n_prob, n_obj)
        Returns:
            alpha_array: torch.Tensor, shape (n_prob, n_obj)
        """
        raise NotImplementedError("get_alpha_array must be implemented in subclass.")