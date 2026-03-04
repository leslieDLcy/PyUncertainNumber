# * ---------------- to put into test
# # --- Example usage ---
# if __name__ == "__main__":
#     xs = [0, 2, 5]
#     ws = [0.5, 0.3, 0.2]
#     r = 0.6

#     xg, F, Gu, Gl = wasserstein_w1_cdf_envelope(
#         xs, ws, r, x_grid=np.linspace(-1, 6, 50)
#     )

#     # Print a few sample points
#     for idx in [0, 10, 20, 30, 40, 49]:
#         print(
#             f"x={xg[idx]:.2f}  F={F[idx]:.3f}  lower={Gl[idx]:.3f}  upper={Gu[idx]:.3f}"
#         )
