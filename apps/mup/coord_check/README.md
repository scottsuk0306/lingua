- prev_mup
- prev_mup_v2
    - attention scaling
    - logit scaling before norm

- prev_mup_v3
    - zero init for lm head

- prev_mup_v4
    - no attention scaling

- ~~prev_mup_v5~~
    - ~~no logit scaling after norm~~ go back to logit scaling before norm
    - logit explodes


- mupv6
 - change to global depth - not works