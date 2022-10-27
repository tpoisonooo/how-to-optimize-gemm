.text
.align 5
.global func1 

func1:
.loop1:
    fmla v0.4s, v0.4s, v16.s[0]
    fmla v1.4s, v1.4s, v16.s[1]
    fmla v2.4s, v2.4s, v16.s[2]
    fmla v3.4s, v3.4s, v16.s[3]

    subs x0, x0, #1

    fmla v4.4s, v4.4s, v17.s[0]
    fmla v5.4s, v5.4s, v17.s[1]
    fmla v6.4s, v6.4s, v17.s[2]
    fmla v7.4s, v7.4s, v17.s[3]

    fmla v8.4s, v8.4s, v18.s[0]
    fmla v9.4s, v9.4s, v18.s[1]
    bne .loop1
    ret
