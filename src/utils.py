
def restore_model(saver, session, loadfrom):
    saver.restore(session, loadfrom)
    print "Model restored from " + loadfrom

def store_model(saver, session, saveto):
    # Tricky closure
    print saveto
    def save(sess=session):
        if saveto is not None:
            save_path = saver.save(sess, saveto)
            print "Model saved to " + save_path
    return save