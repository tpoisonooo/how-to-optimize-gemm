.text
.align 5
.global func2 

func2:
.loop2:
    fmla v0.4s, v0.4s, v0.4s
    fmla v1.4s, v1.4s, v1.4s
    fmla v2.4s, v2.4s, v2.4s
    fmla v3.4s, v3.4s, v3.4s

    fmla v4.4s, v4.4s, v4.4s
    fmla v5.4s, v5.4s, v5.4s
    fmla v6.4s, v6.4s, v6.4s
    fmla v7.4s, v7.4s, v7.4s

    fmla v8.4s, v8.4s, v8.4s
    fmla v9.4s, v9.4s, v9.4s

    subs x0, x0, #1
    bne .loop2
    ret
