import { Euler, Matrix4, Object3D, Quaternion, Vector2Tuple, Vector3, Vector3Tuple } from 'three'
import {
  computeTranslateAsHandleTransformState,
  TranslateAsHandlePointerData,
  TranslateAsHandleStoreData,
} from './computations/index.js'
import {
  computeOnePointerHandleTransformState,
  OnePointerHandlePointerData,
  OnePointerHandleStoreData,
} from './computations/one-pointer.js'
import {
  computeTwoPointerHandleTransformState,
  TwoPointerHandlePointerData,
  TwoPointerHandleStoreData,
} from './computations/two-pointer.js'
import { Axis, HandleState, HandleStateImpl, HandleTransformState } from './state.js'
import { getWorldDirection } from './utils.js'
import type { PointerEvent } from '@pmndrs/pointer-events'

export type HandleOptions<T> = {
  /**
   * function that allows to modify and apply the state to the target
   * @default (state, target) => {target.position.copy(state.current.position);target.quaternion.copy(state.current.quaternion);target.scale.copy(state.current.scale);}
   */
  apply?: (state: HandleState<T>, target: Object3D) => T
  /**
   * @default false
   * necassary if the relative space (e.g. when using the default relativeTo="local") changes often (e.g. every frame)
   */
  alwaysUpdate?: boolean
  ///**
  // * @default false
  // */
  //TODO rubberband?: boolean
  /**
   * @default true
   */
  multitouch?: boolean
  /**
   * @default true
   */
  rotate?: HandleTransformOptions
  /**
   * allows to configure whether rays from input devices should be projected onto the interaction space (3D plane or 3D Line).
   * @default true
   */
  projectRays?: boolean
  /**
   * @default true
   */
  scale?: HandleTransformOptions & {
    /**
     * @default false
     */
    uniform?: boolean
  }
  /**
   * Largest grab offset (metres between the pointer and the point it grabbed) that may drive a
   * one-hand scale under `translate: 'as-rotate-and-scale'`.
   *
   * A direct grab has the hand at the element, so the offset is a few centimetres and a radial pull
   * is deliberate. A laser's grab point sits a whole grab-distance away, so the same pull reacts to
   * pointing wobble and the panel slides in size. `undefined` allows any offset. Never applies while
   * two pointers are down — a pinch always resizes.
   *
   * Measured from geometry rather than the pointer or intersection type: a controller may expose
   * its ray as a 'grab' pointer, and a near-hand sphere can still fire at arm's length.
   */
  oneHandScaleMaxGrabOffset?: number
  /**
   * Filter interaction. Return false to ignore the event.
   */
  filter?: (event: PointerEvent) => boolean
  /**
   * @default true
   */
  stopPropagation?: boolean
} & (
  | {
      /**
       * @default true
       */
      translate?: HandleTransformOptions
    }
  | {
      /**
       * @default true
       */
      translate?: 'as-rotate' | 'as-scale' | 'as-rotate-and-scale'
    }
)

export type HandleTransformOptions =
  | {
      x?: boolean | Vector2Tuple
      y?: boolean | Vector2Tuple
      z?: boolean | Vector2Tuple
    }
  | boolean
  | Axis
  | Array<Vector3Tuple | Vector3>

const vectorHelper = new Vector3()
const axisHelper = new Vector3()
const rotationHelper = new Quaternion()

export class HandleStore<T>
  implements OnePointerHandleStoreData, TwoPointerHandleStoreData, TranslateAsHandleStoreData
{
  //internal out state (will be used to output the state)
  protected outputState: HandleStateImpl<T>
  protected latestMoveEvent: PointerEvent | undefined

  //internal in state (will be written on save)
  readonly inputState = new Map<
    number,
    OnePointerHandlePointerData & TwoPointerHandlePointerData & TranslateAsHandlePointerData
  >()
  readonly capturedObjects = new Map<number, Object3D>()
  /** Prevent duplicate runtime devices for one physical hand from becoming fake multitouch. */
  readonly capturedPointerGroups = new Map<number, string>()
  readonly initialTargetPosition = new Vector3()
  readonly initialTargetQuaternion = new Quaternion()
  readonly initialTargetRotation = new Euler()
  readonly initialTargetScale = new Vector3()
  initialTargetParentWorldMatrix: Matrix4 | undefined

  //prev state
  prevTwoPointerDeltaRotation: Quaternion | undefined
  prevTranslateAsDeltaRotation: Quaternion | undefined
  prevAngle: number | undefined
  /** Extra rotation applied on top of the pointer-derived transform, for the current grab. */
  pendingRotation: Quaternion | undefined

  public readonly handlers = {
    onPointerDown: this.onPointerDown.bind(this),
    onPointerMove: this.onPointerMove.bind(this),
    onPointerUp: this.onPointerUp.bind(this),
    onPointerCancel: this.onPointerCancel.bind(this),
  }

  constructor(
    private readonly target: Object3D | { current?: Object3D | null },
    public readonly getOptions: () => HandleOptions<T> = () => ({}),
  ) {
    this.outputState = new HandleStateImpl<T>(this.cancel.bind(this))
  }

  /**
   * @requires that the pointerId is in this.capturedSet
   */
  private firstOnPointer(event: PointerEvent): void {
    const target = this.getTarget()
    if (target == null) {
      return
    }
    const pointerWorldDirection = getWorldDirection(event, vectorHelper) ? vectorHelper.clone() : undefined

    event.intersection.details.type
    this.inputState.set(event.pointerId, {
      pointerWorldDirection,
      pointerWorldPoint: event.point,
      pointerWorldOrigin: event.pointerPosition,
      pointerWorldQuaternion: event.pointerQuaternion,
      initialPointerWorldPoint: event.point.clone(),
      initialPointerWorldDirection: pointerWorldDirection?.clone(),
      initialPointerWorldQuaternion: event.pointerQuaternion.clone(),
      prevPointerWorldQuaternion: event.pointerQuaternion,
    })
    this.save()
    if (this.inputState.size === 1) {
      this.outputState.start(event, {
        pointerAmount: 1,
        time: event.timeStamp,
        position: this.initialTargetPosition.clone(),
        quaternion: this.initialTargetQuaternion.clone(),
        rotation: this.initialTargetRotation.clone(),
        scale: this.initialTargetScale.clone(),
      })
    }
    this.outputState.memo = this.apply(target)
  }

  private onPointerDown(event: PointerEvent): void {
    if (this.getOptions().filter?.(event) === false) {
      return
    }
    const pointerGroup = getExclusivePointerGroup(event)
    if (pointerGroup != null && [...this.capturedPointerGroups.values()].includes(pointerGroup)) {
      return
    }
    this.stopPropagation(event)
    if (!this.capturePointer(event.pointerId, event.object)) {
      return
    }
    if (pointerGroup != null) {
      this.capturedPointerGroups.set(event.pointerId, pointerGroup)
    }
    this.firstOnPointer(event)
  }

  private onPointerMove(event: PointerEvent): void {
    if (!this.capturedObjects.has(event.pointerId)) {
      return
    }
    this.stopPropagation(event)
    const entry = this.inputState.get(event.pointerId)
    if (entry == null) {
      this.firstOnPointer(event)
      return
    }
    this.latestMoveEvent = event
    entry.pointerWorldPoint = event.point
    entry.prevPointerWorldQuaternion = entry.pointerWorldQuaternion
    entry.pointerWorldQuaternion = event.pointerQuaternion
    entry.pointerWorldOrigin = event.pointerPosition
    if (entry.pointerWorldDirection != null) {
      getWorldDirection(event, entry.pointerWorldDirection)
    }
  }

  public cancel(): void {
    if (this.capturedObjects.size === 0) {
      return
    }
    for (const [pointerId, object] of this.capturedObjects) {
      object.releasePointerCapture(pointerId)
    }
    this.capturedObjects.clear()
    this.capturedPointerGroups.clear()
    this.inputState.clear()
    this.outputState.end(undefined)
    const target = this.getTarget()
    if (target != null) {
      this.apply(target)
    }
  }

  private onPointerUp(event: PointerEvent): void {
    if (!this.capturedObjects.has(event.pointerId)) {
      return
    }
    this.stopPropagation(event)
    this.releasePointer(event.pointerId, event.object, event)
  }

  private onPointerCancel(event: PointerEvent): void {
    const capturedObject = this.capturedObjects.get(event.pointerId)
    if (capturedObject == null) {
      return
    }
    this.stopPropagation(event)
    this.releasePointer(event.pointerId, capturedObject, event)
  }

  update(time: number, force: boolean = false) {
    const target = this.getTarget()
    if (
      target == null ||
      this.inputState.size === 0 ||
      (!force && this.latestMoveEvent == null && (this.getOptions().alwaysUpdate ?? false) === false)
    ) {
      return
    }

    const options = this.getOptions()
    let transformState: HandleTransformState

    if (
      options.translate === 'as-rotate' ||
      options.translate === 'as-rotate-and-scale' ||
      options.translate === 'as-scale'
    ) {
      this.prevTwoPointerDeltaRotation = undefined
      this.prevAngle = undefined
      const [p1, p2] = this.inputState.values()
      const matrixWorld = target.matrixWorld
      const parentMatrixWorld = target.parent?.matrixWorld
      transformState = computeTranslateAsHandleTransformState(
        time,
        this.inputState.size,
        p1,
        this,
        matrixWorld,
        parentMatrixWorld,
        options,
      )
      // A constrained rotation would otherwise be single-pointer only, so a second hand could not
      // resize the element. The rotation stays the first pointer's ('as-rotate'); the second only
      // contributes the pinch scale.
      if (p2 != null && options.scale !== false) {
        const pinch = computeTwoPointerHandleTransformState(
          time,
          p1,
          p2,
          this,
          parentMatrixWorld,
          { ...options, translate: false, rotate: false },
        )
        transformState = { ...transformState, scale: pinch.scale }
      } else if (
        options.oneHandScaleMaxGrabOffset != null &&
        p1.pointerWorldPoint.distanceTo(p1.pointerWorldOrigin) > options.oneHandScaleMaxGrabOffset
      ) {
        // A grab out at arm's length keeps the rotation only; the radial pull would be a laser's
        // twitchy lever rather than a deliberate resize.
        transformState = { ...transformState, scale: this.initialTargetScale.clone() }
      }
    } else if (this.inputState.size === 1) {
      this.prevTwoPointerDeltaRotation = undefined
      this.prevAngle = undefined
      this.prevTranslateAsDeltaRotation = undefined
      const [p1] = this.inputState.values()
      transformState = computeOnePointerHandleTransformState(time, p1, this, target.parent?.matrixWorld, options)
    } else {
      this.prevTranslateAsDeltaRotation = undefined
      const [p1, p2] = this.inputState.values()
      transformState = computeTwoPointerHandleTransformState(time, p1, p2, this, target.parent?.matrixWorld, options)
    }

    this.outputState.update(this.latestMoveEvent, transformState)
    this.outputState.memo = this.apply(target)
    this.latestMoveEvent = undefined
  }

  /** Move an active grab along its pointer ray and apply the transform immediately. */
  translateAlongPointerRay(pointerId: number, distance: number): boolean {
    const pointer = this.inputState.get(pointerId)
    const direction = pointer?.pointerWorldDirection
    if (pointer == null || direction == null || !Number.isFinite(distance) || distance === 0) {
      return false
    }
    // A string `translate` ('as-rotate'/'as-scale') routes translation into rotation or scale, so
    // the handle has no free translation to move. Shifting the pointer baseline anyway would feed
    // a bogus term into the derived transform and corrupt the in-progress grab.
    const translateOption = this.getOptions().translate
    if (translateOption === false || typeof translateOption === 'string') {
      return false
    }
    pointer.initialPointerWorldPoint.addScaledVector(direction, -distance)
    this.update(this.outputState.current.time + 1 / 60, true)
    return true
  }

  /**
   * Hand an active grab's constrained rotation over to an absolute target-frame angle, e.g. a
   * joystick.
   *
   * Rebasing first (`save`) collapses the pointer's accumulated delta to identity, so while the
   * caller keeps calling this the hand no longer turns the target at all — the two inputs can no
   * longer fight. When the caller stops, the next pointer move rotates on from the angle left here.
   * `angle` is absolute (not a delta), so the caller can clamp it against the rotation limits
   * without a wind-up dead zone.
   */
  setTargetAxisAngle(pointerId: number, axis: Axis, angle: number): boolean {
    if (!this.inputState.has(pointerId) || !Number.isFinite(angle)) {
      return false
    }
    this.save()
    this.pendingRotation = rotationHelper.setFromAxisAngle(
      axisHelper.set(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0),
      angle - this.initialTargetRotation[axis],
    )
    this.update(this.outputState.current.time + 1 / 60, true)
    return true
  }

  protected getTarget() {
    return this.target instanceof Object3D ? this.target : this.target?.current
  }

  private capturePointer(pointerId: number, object: Object3D): boolean {
    if (this.capturedObjects.has(pointerId)) {
      return false
    }
    const { multitouch, translate } = this.getOptions()
    // A string `translate` is single-pointer for the *transform*, but a second pointer may still
    // join to pinch-scale when scaling is enabled.
    const singlePointerTransform = (multitouch ?? true) === false ||
      (typeof translate === 'string' && this.getOptions().scale === false)
    if (singlePointerTransform && this.capturedObjects.size === 1) {
      return false
    }
    this.capturedObjects.set(pointerId, object)
    object.setPointerCapture(pointerId)
    return true
  }

  private releasePointer(pointerId: number, object: Object3D, event: PointerEvent | undefined): void {
    const target = this.getTarget()
    if (target == null || !this.capturedObjects.delete(pointerId)) {
      return
    }
    this.inputState.delete(pointerId)
    this.capturedPointerGroups.delete(pointerId)
    object.releasePointerCapture(pointerId)
    if (this.inputState.size > 0) {
      this.save()
      return
    }
    this.outputState.end(event)
    this.apply(target)
  }

  private stopPropagation(event: PointerEvent | undefined) {
    if (event == null || !(this.getOptions()?.stopPropagation ?? true)) {
      return
    }
    event.stopPropagation()
  }

  protected apply(target: Object3D): T {
    const apply = this.getOptions().apply ?? defaultApply
    return apply(this.outputState, target)
  }

  getState(): HandleState<T> | undefined {
    return this.inputState.size === 0 ? undefined : this.outputState
  }

  save(): void {
    const target = this.getTarget()
    if (target == null) {
      return
    }
    target.updateWorldMatrix(true, false)
    //reset prev
    this.prevAngle = undefined
    this.prevTwoPointerDeltaRotation = undefined
    this.prevTranslateAsDeltaRotation = undefined
    this.pendingRotation = undefined
    //update initial
    this.initialTargetParentWorldMatrix = target.parent?.matrixWorld.clone()
    if (target.matrixAutoUpdate) {
      this.initialTargetPosition.copy(target.position)
      this.initialTargetQuaternion.copy(target.quaternion)
      this.initialTargetRotation.copy(target.rotation)
      this.initialTargetScale.copy(target.scale)
    } else {
      target.matrix.decompose(this.initialTargetPosition, this.initialTargetQuaternion, this.initialTargetScale)
      this.initialTargetRotation.setFromQuaternion(this.initialTargetQuaternion, target.rotation.order)
    }
    for (const data of this.inputState.values()) {
      if (data.pointerWorldDirection != null) {
        data.initialPointerWorldDirection?.copy(data.pointerWorldDirection)
      }
      data.initialPointerWorldPoint.copy(data.pointerWorldPoint)
      data.initialPointerWorldQuaternion.copy(data.pointerWorldQuaternion)
    }
  }

  bind(handle: Object3D): () => void {
    const { onPointerDown, onPointerMove, onPointerUp, onPointerCancel } = this.handlers
    handle.addEventListener('pointerdown', onPointerDown)
    handle.addEventListener('pointermove', onPointerMove)
    handle.addEventListener('pointerup', onPointerUp)
    handle.addEventListener('pointercancel', onPointerCancel)
    return () => {
      handle.removeEventListener('pointerdown', onPointerDown)
      handle.removeEventListener('pointermove', onPointerMove)
      handle.removeEventListener('pointerup', onPointerUp)
      handle.removeEventListener('pointercancel', onPointerCancel)
      this.cancel()
    }
  }

  capture(pointerId: number, object: Object3D): () => void {
    if (!this.capturePointer(pointerId, object)) {
      return noop
    }
    return () => this.releasePointer(pointerId, object, undefined)
  }
}

function noop() {}

function getExclusivePointerGroup(event: PointerEvent): string | undefined {
  if (event.pointerType !== 'grab') {
    return undefined
  }
  const handedness = (event.pointerState as { inputSource?: { handedness?: string } } | null)?.inputSource?.handedness
  return handedness === 'left' || handedness === 'right' ? `grab:${handedness}` : undefined
}

export function defaultApply(state: HandleState<unknown>, target: Object3D): any {
  target.position.copy(state.current.position)
  target.rotation.order = state.current.rotation.order
  target.quaternion.copy(state.current.quaternion)
  target.scale.copy(state.current.scale)
}
