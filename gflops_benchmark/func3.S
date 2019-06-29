.text
.align 5
.global func3

func3:
.loop2:
    smlal v10.8h, v0.8b, v0.8b
    smlal v11.8h, v1.8b, v1.8b
    smlal v12.8h, v2.8b, v2.8b
    smlal v13.8h, v3.8b, v3.8b

    smlal v14.8h, v4.8b, v4.8b
    smlal v15.8h, v5.8b, v5.8b
    smlal v16.8h, v6.8b, v6.8b
    smlal v17.8h, v7.8b, v7.8b

    smlal v18.8h, v8.8b, v8.8b
    smlal v19.8h, v9.8b, v9.8b

    subs x0, x0, #1
    bne .loop2
    ret
