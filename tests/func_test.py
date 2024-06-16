import ex1 as hw
import numpy as np
def test_hwfunc1():
    model, R2 = hw.homework()
    assert np.allclose(model.coef_, 1.193033644189594, atol=1e-4)
    assert np.allclose(model.intercept_, -3.89578087831185, atol=1e-4)


def test_hwfunc2():
    model, R2 = hw.homework()
    assert np.allclose(R2, 0.7020315537841397, atol=1e-4)
  
def test_hwfunc3():
    model, R2 = hw.homework()
    assert np.allclose(model.predict([[7.0]]), [4.4554546], atol=1e-4)
    assert np.allclose(model.predict([[10.0]]), [8.03455556], atol=1e-4)
