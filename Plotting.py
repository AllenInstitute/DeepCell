import plotly.graph_objects as go


def plot_loss(train_loss, val_loss, best_epoch, early_stopping):
    fig = go.Figure()

    train_loss = train_loss[:best_epoch + early_stopping]
    val_loss = val_loss[:best_epoch + early_stopping]

    fig.add_trace(go.Scatter(
        x=list(range(best_epoch + early_stopping)),
        y=train_loss,
        name='train'
    ))
    fig.add_trace(go.Scatter(
        x=list(range(best_epoch + early_stopping)),
        y=val_loss,
        name='val'
    ))

    y0 = min(train_loss[best_epoch], val_loss[best_epoch])
    y1 = max(train_loss[best_epoch], val_loss[best_epoch])

    fig.add_shape(type="line",
                  x0=best_epoch, y0=y0 - .05, x1=best_epoch, y1=y1 + .05,
                  line=dict(
                      color="red",
                      width=4,
                      dash="dot"
                  )
                  )
    return fig
