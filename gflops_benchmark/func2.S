#ifndef KERNEL_NAME
#define KERNEL_NAME func2 
#endif

.text
.align 5
.global KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:
.loop2:
    fmla v0.4s, v0.4s, v0.4s
    fmla v1.4s, v1.4s, v1.4s
    fmla v2.4s, v2.4s, v2.4s
    fmla v3.4s, v3.4s, v3.4s

    subs x0, x0, #1

    fmla v4.4s, v4.4s, v4.4s
    fmla v5.4s, v5.4s, v5.4s
    fmla v6.4s, v6.4s, v6.4s
    fmla v7.4s, v7.4s, v7.4s
    bne .loop2
    ret
