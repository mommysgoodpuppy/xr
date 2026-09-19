import { Matrix4, Quaternion, Vector3 } from 'three'
import { HandleTransformState } from '../state.js'
import { HandleOptions } from '../store.js'
import {
  addSpaceFromTransformOptions,
  BaseHandleStoreData,
  computeHandleTransformState,
  projectOntoSpace,
} from './utils.js'

const vectorHelper1 = new Vector3()
const vectorHelper2 = new Vector3()
const vectorHelper3 = new Vector3()

const axisHelper = new Vector3()
const crossHelper = new Vector3()
const radiusHelper = new Vector3()

/** Length of `v` with any component along `axis` removed; the plain length when there is no axis. */
function radiusAboutAxis(v: Vector3, axis: Vector3 | undefined): number {
  if (axis == null) return v.length()
  return radiusHelper.copy(v).addScaledVector(axis, -v.dot(axis)).length()
}

const deltaHelper1 = new Vector3()
const deltaHelper2 = new Vector3()

const qHelper1 = new Quaternion()
const qHelper2 = new Quaternion()

const matrixHelper = new Matrix4()

const scaleHelper = new Vector3()

const space: Array<Vector3> = []

export type TranslateAsHandlePointerData = {
  pointerWorldPoint: Vector3
  pointerWorldDirection: Vector3 | undefined
  pointerWorldQuaternion: Quaternion
  pointerWorldOrigin: Vector3
  prevPointerWorldQuaternion: Quaternion
  initialPointerWorldPoint: Vector3
  initialPointerWorldDirection: Vector3 | undefined
}

export type TranslateAsHandleStoreData = {
  initialTargetPosition: Vector3
  initialTargetQuaternion: Quaternion
  initialTargetScale: Vector3
  initialTargetParentWorldMatrix: Matrix4 | undefined
  prevTranslateAsDeltaRotation: Quaternion | undefined
  /** Absolute extra rotation on top of the pointer-derived one, e.g. a joystick nudge. */
  pendingRotation: Quaternion | undefined
} & BaseHandleStoreData

export function computeTranslateAsHandleTransformState(
  time: number,
  /** Pointers actually down, not just the one driving the rotation (a second may be pinching). */
  pointerAmount: number,
  pointerData: TranslateAsHandlePointerData,
  storeData: TranslateAsHandleStoreData,
  targetWorldMatrix: Matrix4,
  targetParentWorldMatrix: Matrix4 | undefined,
  options: HandleOptions<any> & { translate?: 'as-rotate' | 'as-scale' | 'as-rotate-and-scale' },
): HandleTransformState {
  //compute target parent world quaternion
  if (targetParentWorldMatrix == null) {
    qHelper1.identity()
  } else {
    targetParentWorldMatrix.decompose(vectorHelper1, qHelper1, vectorHelper2)
  }
  //compute space
  space.length = 0
  if (options.translate === 'as-scale') {
    addSpaceFromTransformOptions(space, qHelper1, storeData.initialTargetRotation, options.scale ?? true, 'scale')
  }
  if (options.translate != 'as-scale') {
    addSpaceFromTransformOptions(space, qHelper1, storeData.initialTargetRotation, options.rotate ?? true, 'rotate')
  }

  matrixHelper.makeTranslation(storeData.initialTargetPosition)
  if (storeData.initialTargetParentWorldMatrix != null) {
    matrixHelper.premultiply(storeData.initialTargetParentWorldMatrix)
  }

  //compute initial delta between point and target projected on space
  deltaHelper1.setFromMatrixPosition(matrixHelper)
  projectOntoSpace(
    options.projectRays,
    space,
    pointerData.initialPointerWorldPoint,
    pointerData.pointerWorldOrigin,
    deltaHelper1,
    undefined,
  )
  deltaHelper1.negate().add(pointerData.initialPointerWorldPoint)

  //compute current delta between point and target projected on space
  deltaHelper2.setFromMatrixPosition(targetWorldMatrix)
  projectOntoSpace(
    options.projectRays,
    space,
    pointerData.initialPointerWorldPoint,
    pointerData.pointerWorldOrigin,
    deltaHelper2,
    undefined,
  )
  projectOntoSpace(
    options.projectRays,
    space,
    pointerData.initialPointerWorldPoint,
    pointerData.pointerWorldOrigin,
    vectorHelper2.copy(pointerData.pointerWorldPoint),
    pointerData.pointerWorldDirection,
  )
  deltaHelper2.negate().add(vectorHelper2)

  // The single rotation axis, when the rotation is constrained to one (a hinge). Its sign does not
  // matter: it only ever projects components out of a vector.
  const rotationAxis = space.length === 2
    ? axisHelper.crossVectors(
      (space as [Vector3, Vector3])[0],
      (space as [Vector3, Vector3])[1],
    ).normalize()
    : undefined

  //compute delta rotation
  if (options.translate === 'as-scale') {
    qHelper1.copy(storeData.initialTargetQuaternion)
  } else {
    vectorHelper1.copy(deltaHelper1)
    if (storeData.prevTranslateAsDeltaRotation != null) {
      vectorHelper1.applyQuaternion(storeData.prevTranslateAsDeltaRotation)
    }
    if (rotationAxis != null) {
      // A single-axis rotation (a hinge) measures the angle *about that axis*. The minimal
      // rotation between two grab offsets instead tilts with any offset along the axis, which
      // scales the joint by r²/(r²+dx²): grabbing half a radius off-centre already halves the
      // effect. Dropping the axial component first makes the same hand movement turn the joint by
      // the same angle wherever it is grabbed.
      vectorHelper1.addScaledVector(rotationAxis, -vectorHelper1.dot(rotationAxis))
      vectorHelper2.copy(deltaHelper2).addScaledVector(rotationAxis, -deltaHelper2.dot(rotationAxis))
      // atan2 with the signed axis keeps the original direction for any orientation of `space`.
      qHelper1.setFromAxisAngle(
        rotationAxis,
        Math.atan2(
          rotationAxis.dot(crossHelper.crossVectors(vectorHelper1, vectorHelper2)),
          vectorHelper1.dot(vectorHelper2),
        ),
      )
    } else {
      vectorHelper1.normalize()
      vectorHelper2.copy(deltaHelper2).normalize()
      qHelper1.setFromUnitVectors(vectorHelper1, vectorHelper2)
    }
    if (storeData.prevTranslateAsDeltaRotation == null) {
      storeData.prevTranslateAsDeltaRotation = new Quaternion()
    } else {
      qHelper1.multiply(storeData.prevTranslateAsDeltaRotation)
    }
    storeData.prevTranslateAsDeltaRotation.copy(qHelper1)

    if (storeData.initialTargetParentWorldMatrix != null) {
      qHelper2.setFromRotationMatrix(storeData.initialTargetParentWorldMatrix)
      qHelper1.multiply(qHelper2.normalize())
      qHelper1.premultiply(qHelper2.invert())
    }
    // In the target's own frame, so it composes with (and is independent of) the axis-locked
    // pointer rotation the same way an extra turn of the axis would.
    if (storeData.pendingRotation != null) {
      qHelper1.multiply(storeData.pendingRotation)
    }
    qHelper1.multiply(storeData.initialTargetQuaternion)
  }

  //compute delta scale
  if (options.translate === 'as-rotate') {
    scaleHelper.set(1, 1, 1)
  } else if (typeof options.scale === 'object' && (options.scale.uniform ?? false)) {
    // A uniform scale here is the pointer's distance from the rotation pivot over its initial
    // distance: how one hand resizes a hinged element, with the hinge standing in for the second
    // hand. The radius is measured from the axis (matching the rotation) so an offset along the
    // axis cannot weaken the gesture off-centre.
    const radius1 = radiusAboutAxis(deltaHelper1, rotationAxis)
    const radius2 = radiusAboutAxis(deltaHelper2, rotationAxis)
    scaleHelper.setScalar(radius1 === 0 ? 1 : radius2 / radius1)
  } else if (options.translate === 'as-rotate-and-scale') {
    //compute the initial world quaternion and initial world scale
    matrixHelper.compose(
      storeData.initialTargetPosition,
      storeData.initialTargetQuaternion,
      storeData.initialTargetScale,
    )
    if (storeData.initialTargetParentWorldMatrix != null) {
      matrixHelper.premultiply(storeData.initialTargetParentWorldMatrix)
    }
    matrixHelper.decompose(vectorHelper2, qHelper2, vectorHelper3)
    //compute the initial scale axis
    vectorHelper1.copy(deltaHelper1).applyQuaternion(qHelper2.invert()).divide(vectorHelper3)
    vectorHelper1.x = Math.abs(vectorHelper1.x)
    vectorHelper1.y = Math.abs(vectorHelper1.y)
    vectorHelper1.z = Math.abs(vectorHelper1.z)
    const maxCompInitialDelta = Math.max(...vectorHelper1.toArray())
    vectorHelper1.divideScalar(maxCompInitialDelta)
    scaleHelper.set(1, 1, 1)
    scaleHelper.addScaledVector(vectorHelper1, deltaHelper2.length() / deltaHelper1.length() - 1)
  } else {
    //as scale
    if (storeData.initialTargetParentWorldMatrix != null) {
      storeData.initialTargetParentWorldMatrix.decompose(vectorHelper1, qHelper2, vectorHelper2)
      qHelper2.multiply(storeData.initialTargetQuaternion)
    } else {
      qHelper2.copy(storeData.initialTargetQuaternion)
    }
    vectorHelper1.copy(deltaHelper1).applyQuaternion(qHelper2.invert())

    if (targetParentWorldMatrix != null) {
      targetParentWorldMatrix.decompose(vectorHelper2, qHelper2, vectorHelper3)
      qHelper2.multiply(storeData.initialTargetQuaternion)
    } else {
      qHelper2.copy(storeData.initialTargetQuaternion)
    }
    vectorHelper2.copy(deltaHelper2).applyQuaternion(qHelper2.invert())

    scaleHelper.x = Math.abs(vectorHelper1.x) < 0.001 ? 1 : Math.abs(vectorHelper2.x / vectorHelper1.x)
    scaleHelper.y = Math.abs(vectorHelper1.y) < 0.001 ? 1 : Math.abs(vectorHelper2.y / vectorHelper1.y)
    scaleHelper.z = Math.abs(vectorHelper1.z) < 0.001 ? 1 : Math.abs(vectorHelper2.z / vectorHelper1.z)
  }

  scaleHelper.multiply(storeData.initialTargetScale)

  matrixHelper.compose(storeData.initialTargetPosition, qHelper1, scaleHelper)

  //we pass targetParentWorldMatrix as undefined, because we calculated matrixHelper1 in local target space
  return computeHandleTransformState(time, pointerAmount, matrixHelper, storeData, undefined, options)
}
