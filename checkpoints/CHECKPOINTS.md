# Pre-trained checkpoints for UltraSharp
# =========================================
# Oral Presentation, IEEE ISBI 2026
# "UltraSharp: Beltrami Transformers for Ultrasound Super-Resolution"
#
# Pre-trained model weights will be released here after ISBI 2026.
#
# Expected files (to be released):
#
#   checkpoints/
#       ultrasharp-t_x2.pth     Tiny variant,  scale x2
#       ultrasharp-t_x4.pth     Tiny variant,  scale x4
#       ultrasharp-s_x2.pth     Small variant, scale x2
#       ultrasharp-s_x4.pth     Small variant, scale x4
#       ultrasharp-b_x2.pth     Base variant,  scale x2  (paper default)
#       ultrasharp-b_x4.pth     Base variant,  scale x4  (paper default)
#       ultrasharp-l_x4.pth     Large variant, scale x4
#
# Loading a checkpoint (after release):
#
#   from models.builder import build_ultrasharp
#   model = build_ultrasharp("ultrasharp-b", scale=4,
#                             checkpoint="checkpoints/ultrasharp-b_x4.pth")
#
# See the repository README for the release announcement.
