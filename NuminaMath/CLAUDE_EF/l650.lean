import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_reals_covered_l650_65014

-- Define the sequence of rational numbers
def rational_sequence : ℕ → ℚ := sorry

-- Define the open interval for each rational number
def open_interval (n : ℕ) : Set ℝ :=
  {x : ℝ | ∃ (center : ℝ), |x - center| < 1 / (2 * n)}

-- Define the union of all open intervals
def union_of_intervals : Set ℝ :=
  ⋃ (n : ℕ), open_interval n

-- Theorem statement
theorem not_all_reals_covered :
  ∃ (x : ℝ), x ∉ union_of_intervals := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_reals_covered_l650_65014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_eq_pi_sufficient_not_necessary_l650_65006

noncomputable def curve (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

def passes_through_origin (φ : ℝ) : Prop := ∃ x, curve x φ = 0

theorem phi_eq_pi_sufficient_not_necessary :
  (∀ φ, φ = Real.pi → passes_through_origin φ) ∧
  ¬(∀ φ, passes_through_origin φ → φ = Real.pi) := by
  sorry

#check phi_eq_pi_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_eq_pi_sufficient_not_necessary_l650_65006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lead_isotope_ratio_l650_65011

/-- Represents the atomic weights of lead isotopes -/
def LeadIsotope := Fin 3 → ℝ

/-- The average atomic weight of naturally occurring lead -/
def averageAtomicWeight : ℝ := 207.2

/-- The atomic weights of the three lead isotopes -/
def isotopesWeights : LeadIsotope :=
  fun i => match i with
    | 0 => 206
    | 1 => 207
    | 2 => 208

/-- The ratio of lead isotopes (206:207:208) -/
def leadRatio : Fin 3 → ℕ :=
  fun i => match i with
    | 0 => 3
    | 1 => 2
    | 2 => 5

/-- Theorem stating that the given ratio satisfies the conditions -/
theorem lead_isotope_ratio :
  let totalAtoms : ℝ := (leadRatio 0 : ℝ) + (leadRatio 1 : ℝ) + (leadRatio 2 : ℝ)
  let weightedSum : ℝ := (leadRatio 0 : ℝ) * isotopesWeights 0 +
                         (leadRatio 1 : ℝ) * isotopesWeights 1 +
                         (leadRatio 2 : ℝ) * isotopesWeights 2
  abs ((weightedSum / totalAtoms) - averageAtomicWeight) < 0.1 ∧
  abs ((leadRatio 2 : ℝ) - ((leadRatio 0 : ℝ) + (leadRatio 1 : ℝ))) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lead_isotope_ratio_l650_65011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_plane_l650_65079

-- Define a structure for a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a structure for a line in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a structure for a plane in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define membership for Point3D in Line3D
def Point3D.mem (p : Point3D) (l : Line3D) : Prop :=
  ∃ t : ℝ, p = Point3D.mk 
    (l.point.x + t * l.direction.x)
    (l.point.y + t * l.direction.y)
    (l.point.z + t * l.direction.z)

instance : Membership Point3D Line3D where
  mem := Point3D.mem

-- Define membership for Point3D in Plane3D
def Point3D.memPlane (p : Point3D) (plane : Plane3D) : Prop :=
  (p.x - plane.point.x) * plane.normal.x +
  (p.y - plane.point.y) * plane.normal.y +
  (p.z - plane.point.z) * plane.normal.z = 0

instance : Membership Point3D Plane3D where
  mem := Point3D.memPlane

-- Main theorem statement
theorem fixed_point_plane 
  (l1 l2 l3 : Line3D) 
  (A B C : Point3D) 
  (α β γ : ℝ) :
  (∀ M N L : Point3D, 
    M ∈ l1 ∧ N ∈ l2 ∧ L ∈ l3 →
    (∃ k : ℝ, α * distance A M + β * distance B N + γ * distance C L = k)) →
  (∃ P : Point3D, ∀ M N L : Point3D, 
    M ∈ l1 ∧ N ∈ l2 ∧ L ∈ l3 →
    ∃ plane : Plane3D, M ∈ plane ∧ N ∈ plane ∧ L ∈ plane ∧ P ∈ plane) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_plane_l650_65079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l650_65083

def sequenceA (n : ℕ) : ℤ := (-1)^(n+1) * (4*n - 1)

theorem sequence_correct : ∀ n : ℕ, 
  (n = 1 → sequenceA n = 3) ∧ 
  (n = 2 → sequenceA n = -7) ∧ 
  (n = 3 → sequenceA n = 11) ∧ 
  (n = 4 → sequenceA n = -15) ∧
  (n > 4 → sequenceA n = (-1)^(n+1) * (4*n - 1)) :=
by
  intro n
  constructor
  · intro h; simp [sequenceA, h]
  constructor
  · intro h; simp [sequenceA, h]
  constructor
  · intro h; simp [sequenceA, h]
  constructor
  · intro h; simp [sequenceA, h]
  · intro h; simp [sequenceA]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_correct_l650_65083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l650_65093

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the side length
noncomputable def side_length (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the angle measure in radians
noncomputable def angle_measure (p q r : ℝ × ℝ) : ℝ :=
  Real.arccos ((side_length p q)^2 + (side_length p r)^2 - (side_length q r)^2) / (2 * side_length p q * side_length p r)

theorem triangle_angle_measure (t : Triangle) :
  let a := side_length t.B t.C
  let b := side_length t.A t.C
  let c := side_length t.A t.B
  a^2 + c^2 - b^2 = a * c →
  angle_measure t.A t.B t.C = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_measure_l650_65093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_circles_l650_65012

noncomputable def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

def line_l (m n x y : ℝ) : Prop := m * x + n * y = 1

noncomputable def triangle_area (h : ℝ) : ℝ := Real.sqrt (h^2 - h^4)

theorem max_triangle_area_on_circles :
  ∃ (n : ℝ),
    circle_C (1/2) n ∧
    (∃ (A B : ℝ × ℝ),
      A ≠ B ∧
      circle_O A.1 A.2 ∧
      circle_O B.1 B.2 ∧
      line_l (1/2) n A.1 A.2 ∧
      line_l (1/2) n B.1 B.2 ∧
      triangle_area (1 / Real.sqrt 2) = 1/2 ∧
      ∀ (h : ℝ), triangle_area h ≤ 1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_on_circles_l650_65012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_special_numbers_l650_65022

theorem ordering_of_special_numbers :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := Real.log 3 / Real.log π
  let c : ℝ := -Real.log 3 / Real.log 2
  c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_special_numbers_l650_65022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_equality_unique_solution_l650_65035

open Nat

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- The main theorem: 101 in base 18 equals 11011 in base 4 --/
theorem base_conversion_equality :
  to_decimal [1, 0, 1] 18 = to_decimal [1, 1, 0, 1, 1] 4 := by
  -- Evaluate left side
  have h1 : to_decimal [1, 0, 1] 18 = 325 := by
    rfl
  
  -- Evaluate right side
  have h2 : to_decimal [1, 1, 0, 1, 1] 4 = 325 := by
    rfl
  
  -- Show equality
  rw [h1, h2]

/-- Uniqueness of the solution --/
theorem unique_solution (n k : Nat) :
  n^2 + 1 = k^4 + k^3 + k + 1 → n = 18 ∧ k = 4 := by
  sorry  -- Full proof omitted for brevity

#eval to_decimal [1, 0, 1] 18  -- Should output 325
#eval to_decimal [1, 1, 0, 1, 1] 4  -- Should also output 325

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_equality_unique_solution_l650_65035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_pressure_change_l650_65074

/-- Represents the pressure-volume relationship for a gas at constant temperature -/
structure GasState where
  pressure : ℝ
  volume : ℝ

/-- The constant k in the pressure-volume relationship -/
def pressureVolumeConstant (state : GasState) : ℝ :=
  state.pressure * state.volume

theorem gas_pressure_change
  (initialState finalState : GasState)
  (h1 : initialState.pressure = 7)
  (h2 : initialState.volume = 3.4)
  (h3 : finalState.volume = 4.25)
  (h4 : pressureVolumeConstant initialState = pressureVolumeConstant finalState) :
  ∃ ε > 0, |finalState.pressure - 5.6| < ε := by
  sorry

#check gas_pressure_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_pressure_change_l650_65074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_centroid_l650_65059

/-- Function to calculate the area of a triangle given its vertices -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Given a triangle PQR with vertices P(4, 9), Q(2, -3), and R(7, 2),
    if point S(x, y) is chosen such that the areas of triangles PQS, PRS, and QRS are equal,
    then 8x + y = 37 1/3 -/
theorem equal_area_centroid (x y : ℝ) :
  let P : ℝ × ℝ := (4, 9)
  let Q : ℝ × ℝ := (2, -3)
  let R : ℝ × ℝ := (7, 2)
  let S : ℝ × ℝ := (x, y)
  (area_triangle P Q S = area_triangle P R S) ∧
  (area_triangle P R S = area_triangle Q R S) →
  8 * x + y = 37 + 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_centroid_l650_65059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_charge_is_28_l650_65061

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  fixed_charge : ℚ
  call_charge : ℚ

/-- Calculates the total bill amount -/
def total_bill (bill : MonthlyBill) : ℚ :=
  bill.fixed_charge + bill.call_charge

theorem fixed_charge_is_28
  (january : MonthlyBill)
  (february : MonthlyBill)
  (h1 : total_bill january = 52)
  (h2 : total_bill february = 76)
  (h3 : february.call_charge = 2 * january.call_charge)
  (h4 : january.fixed_charge = february.fixed_charge) :
  january.fixed_charge = 28 := by
  -- Proof steps would go here
  sorry

#check fixed_charge_is_28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_charge_is_28_l650_65061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_reduction_l650_65077

/-- The total percentage reduction after discounts and subtraction -/
noncomputable def total_reduction (P : ℝ) : ℝ :=
  (0.44 + 50 / P) * 100

/-- Theorem stating the total percentage reduction after discounts and subtraction -/
theorem discount_reduction (P : ℝ) (h : P > 0) :
  let F := 0.56 * P - 50
  (1 - F / P) * 100 = total_reduction P :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_reduction_l650_65077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l650_65009

noncomputable def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_specific : 
  triangle_area 4 (-1) 2 7 4 4 = 5 := by
  -- Unfold the definition of triangle_area
  unfold triangle_area
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_specific_l650_65009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_implies_angle_difference_l650_65029

theorem cosine_sum_zero_implies_angle_difference (α β γ : ℝ) :
  0 < α ∧ α < β ∧ β < γ ∧ γ < 2 * Real.pi →
  (∀ x : ℝ, Real.cos (x + α) + Real.cos (x + β) + Real.cos (x + γ) = 0) →
  γ - α = 4 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_zero_implies_angle_difference_l650_65029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_literature_not_math_proof_l650_65018

/-- This theorem represents the fact that the given problem is not a mathematical proof. -/
theorem literature_not_math_proof : True := by
  trivial

#print axioms literature_not_math_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_literature_not_math_proof_l650_65018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l650_65064

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then x + 2
  else if x < 2 then x^2
  else 2 * x

-- Theorem statement
theorem f_properties :
  (f (-3) = -1) ∧
  (f (f (-3)) = 1) ∧
  (∀ a : ℝ, f a = 8 ↔ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l650_65064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_f_l650_65002

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := x + 4 / x

-- Theorem stating the extreme points of f(x)
theorem extreme_points_of_f :
  ∀ x y : ℝ, (x ≠ 0 ∧ f x = y ∧ ∀ z : ℝ, z ≠ 0 → f z ≤ f x) ∨ 
             (x ≠ 0 ∧ f x = y ∧ ∀ z : ℝ, z ≠ 0 → f z ≥ f x) →
  (x = 2 ∧ y = 4) ∨ (x = -2 ∧ y = -4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_of_f_l650_65002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_ratio_l650_65008

/-- Given a circle with radius r and two inscribed semicircles with radii r/2,
    the ratio of the combined areas of the semicircles to the area of the circle is 1/4 -/
theorem semicircle_area_ratio (r : ℝ) (hr : r > 0) :
  (2 * (π * (r/2)^2 / 2)) / (π * r^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_area_ratio_l650_65008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l650_65007

/-- Polar curve C -/
noncomputable def C (p θ : ℝ) : Prop := p^2 = 12 / (2 + Real.cos θ)

/-- Line l -/
noncomputable def L (p θ : ℝ) : Prop := 2 * p * Real.cos (θ - Real.pi/6) = Real.sqrt 3

/-- Distance between intersection points -/
noncomputable def distance_AB : ℝ := 4 * Real.sqrt 10 / 3

/-- Theorem stating that the distance between intersection points is 4√10/3 -/
theorem intersection_distance : 
  ∃ (p₁ θ₁ p₂ θ₂ : ℝ), 
    C p₁ θ₁ ∧ L p₁ θ₁ ∧ 
    C p₂ θ₂ ∧ L p₂ θ₂ ∧ 
    p₁ ≠ p₂ ∧
    Real.sqrt ((p₁ * Real.cos θ₁ - p₂ * Real.cos θ₂)^2 + 
               (p₁ * Real.sin θ₁ - p₂ * Real.sin θ₂)^2) = distance_AB :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l650_65007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l650_65015

/-- Represents a hyperbola with equation mx^2 - y^2 = 1 -/
structure Hyperbola where
  m : ℝ
  eq : ∀ x y : ℝ, m * x^2 - y^2 = 1

/-- The distance between the vertices of a hyperbola -/
noncomputable def vertexDistance (h : Hyperbola) : ℝ := 2 * Real.sqrt (1 / h.m)

/-- The equation of the asymptotes of a hyperbola -/
def asymptoticEquation (h : Hyperbola) : Set (ℝ × ℝ) :=
  {(x, y) | y = x * Real.sqrt h.m ∨ y = -x * Real.sqrt h.m}

theorem hyperbola_asymptotes 
    (h : Hyperbola) 
    (h_vertex_distance : vertexDistance h = 4) :
  asymptoticEquation h = {(x, y) | y = x / 2 ∨ y = -x / 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l650_65015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_three_ways_l650_65039

/-- Represents the attributes of a block -/
structure BlockAttributes where
  material : Fin 3
  size : Fin 3
  color : Fin 4
  shape : Fin 4
  texture : Fin 2
deriving Fintype, DecidableEq

/-- The reference block: 'plastic medium red circle smooth' -/
def referenceBlock : BlockAttributes := {
  material := 0,
  size := 1,
  color := 2,
  shape := 0,
  texture := 0
}

/-- Counts the number of differences between two BlockAttributes -/
def countDifferences (b1 b2 : BlockAttributes) : Nat :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.texture ≠ b2.texture then 1 else 0)

/-- The main theorem stating that 34 blocks differ in exactly three ways from the reference block -/
theorem blocks_differing_in_three_ways :
  (Finset.filter (fun b => countDifferences referenceBlock b = 3) (Finset.univ : Finset BlockAttributes)).card = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blocks_differing_in_three_ways_l650_65039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_difference_of_bounds_l650_65005

noncomputable def arithmeticGeometricSequence (n : ℕ) : ℝ := 2 * (-1/3)^(n-1)

noncomputable def sumOfTerms (n : ℕ) : ℝ := 
  (3/2) - (3/2) * (-1/3)^n

noncomputable def lowerBound (S : ℝ) : ℝ := 3 * S - 1 / S

noncomputable def upperBound (S : ℝ) : ℝ := 3 * S - 1 / S

theorem minimum_difference_of_bounds :
  ∃ (A B : ℝ), ∀ (n : ℕ), 
    A ≤ lowerBound (sumOfTerms n) ∧ 
    upperBound (sumOfTerms n) ≤ B ∧
    B - A = 9/4 ∧
    ∀ (A' B' : ℝ), (∀ (n : ℕ), 
      A' ≤ lowerBound (sumOfTerms n) ∧ 
      upperBound (sumOfTerms n) ≤ B') → 
    B' - A' ≥ 9/4 :=
by
  sorry

#check minimum_difference_of_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_difference_of_bounds_l650_65005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sums_of_squares_l650_65078

theorem unique_sums_of_squares (n : ℕ+) : 
  (∃ a b : ℕ+, n = a^2 + b^2 ∧ 
   Nat.Coprime a b ∧ 
   ∀ p : ℕ, Nat.Prime p → p ≤ Real.sqrt (n : ℝ) → p ∣ (a * b)) ↔ 
  n = 2 ∨ n = 5 ∨ n = 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_sums_of_squares_l650_65078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_pair_l650_65075

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Check if a point is inside or on a rectangle -/
def Point.inRectangle (p : Point) (r : Rectangle) : Prop :=
  0 ≤ p.x ∧ p.x ≤ r.width ∧ 0 ≤ p.y ∧ p.y ≤ r.height

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The main theorem -/
theorem exists_close_pair (points : Finset Point) (r : Rectangle) : 
  r.width = 2 ∧ r.height = 1 →
  points.card = 8 →
  (∀ p, p ∈ points → p.inRectangle r) →
  ∃ p1 p2, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧ distance p1 p2 ≤ Real.sqrt 5 / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_close_pair_l650_65075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l650_65020

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define point A
noncomputable def point_A : ℝ × ℝ := (1, 2 * Real.sqrt 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem max_distance_sum :
  ∃ (max : ℝ), max = 4 + 2 * Real.sqrt 3 ∧
  ∀ (P : ℝ × ℝ), ellipse P.1 P.2 →
    distance P point_A + distance P right_focus ≤ max := by
  sorry

#check max_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l650_65020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_outcome_l650_65092

/-- Represents a player in the game -/
inductive Player : Type
| First : Player
| Second : Player

/-- The game state, including the current player and the assigned signs -/
structure GameState :=
  (currentPlayer : Player)
  (assignedSigns : Fin 20 → Option Bool)

/-- The optimal strategy for both players -/
def optimalStrategy : GameState → Fin 20 → Bool → GameState :=
  sorry

/-- The final score of the game after all moves are made -/
def finalScore (initialState : GameState) : ℤ :=
  sorry

/-- The main theorem stating the outcome of the game -/
theorem game_outcome :
  ∀ (initialState : GameState),
  initialState.currentPlayer = Player.First →
  (∀ n : Fin 20, initialState.assignedSigns n = none) →
  |finalScore initialState| = 30 :=
by
  sorry

#check game_outcome

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_outcome_l650_65092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_speed_is_30_l650_65071

/-- Represents the chase scenario between a zebra and a tiger -/
structure ChaseScenario where
  zebra_delay : ℚ  -- Time delay before zebra starts chasing (in hours)
  chase_duration : ℚ  -- Time taken by zebra to catch tiger (in hours)
  zebra_speed : ℚ  -- Average speed of zebra (in km/h)

/-- Calculates the average speed of the tiger given a chase scenario -/
def tiger_speed (scenario : ChaseScenario) : ℚ :=
  (scenario.chase_duration * scenario.zebra_speed) / (scenario.zebra_delay + scenario.chase_duration)

/-- Theorem stating that for the given scenario, the tiger's speed is 30 km/h -/
theorem tiger_speed_is_30 :
  let scenario := ChaseScenario.mk 5 6 55
  tiger_speed scenario = 30 := by
  -- The proof goes here
  sorry

#eval tiger_speed (ChaseScenario.mk 5 6 55)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiger_speed_is_30_l650_65071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l650_65085

/-- The number of days it takes for two workers to complete a job together,
    given the efficiency ratio between them and the time for one worker to complete the job. -/
noncomputable def days_to_complete (efficiency_ratio : ℚ) (days_for_p : ℚ) : ℚ :=
  let q_efficiency := 1 / (1 + efficiency_ratio)
  let p_efficiency := 1
  let combined_efficiency := p_efficiency + q_efficiency
  days_for_p / combined_efficiency

/-- Theorem stating that if p is 40% more efficient than q and can complete the work in 24 days,
    then p and q working together will complete the same work in 15 days. -/
theorem work_completion_time :
  days_to_complete (2/5) 24 = 15 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l650_65085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_positive_l650_65031

theorem negation_of_universal_sin_positive :
  (¬ ∀ x : ℝ, Real.sin x > 0) ↔ (∃ x : ℝ, Real.sin x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_positive_l650_65031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unbounded_above_l650_65098

/-- The function to be maximized -/
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2)

/-- Theorem stating that f is unbounded above -/
theorem f_unbounded_above : ∀ M : ℝ, ∃ x : ℝ, f x > M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unbounded_above_l650_65098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l650_65019

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - Real.cos x ^ 2

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (x : ℝ), f x ≤ 1) ∧
  (∃ (x : ℝ), f x = 1) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l650_65019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_lambda_bound_l650_65000

def a (n : ℕ) (lambda : ℝ) : ℝ := -2 * (n^2 : ℝ) + lambda * n

theorem decreasing_sequence_lambda_bound (lambda : ℝ) :
  (∀ n : ℕ, n > 0 → a n lambda > a (n + 1) lambda) → lambda < 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_sequence_lambda_bound_l650_65000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_not_divisible_by_51_l650_65050

theorem largest_product_not_divisible_by_51 : 
  ∃ n : ℕ, (n = 16 ∧ 
    (∀ k : ℕ, k ≤ n → (Finset.range k).prod (λ i => i + 1) % 51 ≠ 0) ∧
    (Finset.range (n + 1)).prod (λ i => i + 1) % 51 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_product_not_divisible_by_51_l650_65050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonSimilar300PointedStars_noStarsFor346_l650_65016

/-- A regular n-pointed star -/
structure RegularStar (n : ℕ) where
  m : ℕ
  coprime : Nat.Coprime m n
  skip : m < n

/-- The number of non-similar regular n-pointed stars -/
def nonSimilarStarCount (n : ℕ) : ℕ :=
  (Nat.totient n - 2) / 2

/-- Theorem: The number of non-similar regular 300-pointed stars is 39 -/
theorem nonSimilar300PointedStars :
  nonSimilarStarCount 300 = 39 := by
  sorry

/-- No regular 3, 4, or 6-pointed stars exist -/
theorem noStarsFor346 :
  ∀ n, n ∈ ({3, 4, 6} : Set ℕ) → ¬∃ (star : RegularStar n), True := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonSimilar300PointedStars_noStarsFor346_l650_65016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_probability_l650_65044

def S : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def T : Finset Char := {'a', 'b', 'c'}

theorem subset_probability :
  (Finset.card (Finset.powerset S) : ℚ) / (Finset.card (Finset.powerset (S.filter (λ x => x ∈ T)))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_probability_l650_65044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_share_money_l650_65052

theorem equal_share_money (total_amount : ℚ) (num_people : ℕ) (share : ℚ) :
  total_amount = 3.75 →
  num_people = 3 →
  share = total_amount / num_people →
  share = 1.25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_share_money_l650_65052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_rotating_parallel_lines_l650_65056

/-- Two points in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a 2D plane -/
structure Line where
  point : Point
  direction : ℝ × ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Two parallel lines that rotate around fixed points -/
structure RotatingParallelLines where
  l₁ : Line
  l₂ : Line
  parallel : l₁.direction = l₂.direction

theorem max_distance_between_rotating_parallel_lines :
  let P : Point := ⟨-1, 2⟩
  let Q : Point := ⟨2, -3⟩
  let lines : RotatingParallelLines := ⟨⟨P, (0, 0)⟩, ⟨Q, (0, 0)⟩, rfl⟩
  (distance P Q) = Real.sqrt 34 ∧
  ∀ (θ : ℝ), distance P Q ≥ 
    (let newDir := (Real.cos θ, Real.sin θ)
     let newLines : RotatingParallelLines := ⟨⟨P, newDir⟩, ⟨Q, newDir⟩, rfl⟩
     abs (newLines.l₁.point.x * newDir.2 - newLines.l₁.point.y * newDir.1
        - (newLines.l₂.point.x * newDir.2 - newLines.l₂.point.y * newDir.1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_rotating_parallel_lines_l650_65056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_l650_65070

def vector_parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

def plane_perpendicular (u v : Fin 3 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) + (u 2) * (v 2) = 0

theorem parallel_and_perpendicular :
  (vector_parallel (![2, 3, -1]) (![(-1), (-3/2), 1/2])) ∧
  (plane_perpendicular (![2, 2, -1]) (![(-3), 4, 2])) := by
  sorry

#check parallel_and_perpendicular

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_l650_65070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_500_l650_65062

/-- Represents the race between Biff and Kenneth -/
structure Race where
  biff_speed : ℝ  -- Biff's rowing speed in yards per minute
  kenneth_speed : ℝ  -- Kenneth's rowing speed in yards per minute
  kenneth_extra_distance : ℝ  -- Extra distance Kenneth rows past the finish line

/-- The length of the race in yards -/
noncomputable def race_length (r : Race) : ℝ :=
  (r.kenneth_speed * r.kenneth_extra_distance) / (r.kenneth_speed - r.biff_speed)

/-- Theorem stating the length of the race given the specific conditions -/
theorem race_length_is_500 (r : Race) 
  (h1 : r.biff_speed = 50)
  (h2 : r.kenneth_speed = 51)
  (h3 : r.kenneth_extra_distance = 10) : 
  race_length r = 500 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_length_is_500_l650_65062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equality_l650_65090

/-- Given sets U, A, and B, prove that the union of their complements is equal to the specified set -/
theorem complement_union_equality (U A B : Set ℕ) 
  (hU : U = {1,2,3,4,5,6,7})
  (hA : A = {2,4,5,7})
  (hB : B = {3,4,5}) :
  (U \ A) ∪ (U \ B) = {1,2,3,6,7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_equality_l650_65090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_eccentricity_for_trisection_l650_65082

/-- An ellipse with semi-major axis a and semi-minor axis b. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on an ellipse. -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The property that a point P on the ellipse trisects ∠APB with PF₁ and PF₂. -/
def trisects_angle (e : Ellipse) (P : PointOnEllipse e) : Prop :=
  ∃ (A B F₁ F₂ : ℝ × ℝ),
    let d₁ := Real.sqrt ((P.x - F₁.1)^2 + (P.y - F₁.2)^2)
    let d₂ := Real.sqrt ((P.x - F₂.1)^2 + (P.y - F₂.2)^2)
    let d₃ := Real.sqrt ((P.x - A.1)^2 + (P.y - A.2)^2)
    let d₄ := Real.sqrt ((P.x - B.1)^2 + (P.y - B.2)^2)
    d₁ / d₃ = d₂ / d₄ ∧ d₁ / d₃ = (d₁ + d₂) / (d₃ + d₄)

/-- The main theorem: there exists exactly one eccentricity satisfying the trisection condition. -/
theorem unique_eccentricity_for_trisection (e : Ellipse) :
  ∃! ecc : ℝ, ecc ∈ Set.Ioo 0 1 ∧
    ∃ P : PointOnEllipse e, trisects_angle e P ∧ eccentricity e = ecc := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_eccentricity_for_trisection_l650_65082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorics_identity_derivative_of_f_l650_65034

-- Part I
theorem combinatorics_identity : (Nat.choose 5 2 + Nat.choose 5 3) / (5 * 4 * 3 : ℚ) = 1 / 3 := by sorry

-- Part II
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * Real.sin (2 * x)

theorem derivative_of_f (x : ℝ) : deriv f x = Real.exp (-x) * (-Real.sin (2 * x) + 2 * Real.cos (2 * x)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combinatorics_identity_derivative_of_f_l650_65034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_x_axis_l650_65032

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- A point is on the x-axis if its y and z coordinates are zero -/
def onXAxis (p : Point3D) : Prop :=
  p.y = 0 ∧ p.z = 0

theorem equidistant_point_on_x_axis :
  let A : Point3D := ⟨3, 2, 0⟩
  let B : Point3D := ⟨2, -1, 2⟩
  let M : Point3D := ⟨2, 0, 0⟩
  onXAxis M ∧ distance M A = distance M B ∧
  ∀ M' : Point3D, onXAxis M' ∧ distance M' A = distance M' B → M' = M :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_on_x_axis_l650_65032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_four_l650_65081

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x + 8) / (x^2 - 8*x + 16)

def has_vertical_asymptote (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ M > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x| > M

theorem vertical_asymptote_at_four :
  has_vertical_asymptote f 4 := by
  sorry

#check vertical_asymptote_at_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_four_l650_65081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l650_65025

noncomputable section

open Real

theorem triangle_tangent_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Ensure angles are valid
  A + B + C = π →  -- Sum of angles in a triangle
  a * sin B = b * sin A →  -- Sine rule
  c * sin A = a * sin C →  -- Sine rule
  c * sin B = b * sin C →  -- Sine rule
  a * cos B - b * cos A = (3/5) * c →  -- Given condition
  tan A / tan B = 4 := by
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_ratio_l650_65025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_real_roots_and_S_equals_two_l650_65099

noncomputable def equation (k x : ℝ) : ℝ := (k - 1) * x^2 + 2 * k * x + 2

noncomputable def discriminant (k : ℝ) : ℝ := (2 * k)^2 - 4 * (k - 1) * 2

noncomputable def S (x1 x2 : ℝ) : ℝ := x2 / x1 + x1 / x2 + x1 + x2

theorem equation_real_roots_and_S_equals_two :
  (∀ k : ℝ, ∃ x1 x2 : ℝ, equation k x1 = 0 ∧ equation k x2 = 0) ∧
  (∃ k : ℝ, k = 2 ∧ 
    ∃ x1 x2 : ℝ, equation k x1 = 0 ∧ equation k x2 = 0 ∧ S x1 x2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_real_roots_and_S_equals_two_l650_65099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pc_length_is_four_l650_65067

/-- A triangle with a right angle and a special interior point -/
structure RightTriangleWithPoint where
  -- Points of the triangle
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Interior point
  P : ℝ × ℝ
  -- Right angle at B
  right_angle_at_B : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0
  -- PA = 8
  PA_length : Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 8
  -- PB = 4
  PB_length : Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 4
  -- ∠APB = ∠BPC = ∠CPA
  equal_angles : 
    let angle (X Y Z : ℝ × ℝ) := 
      Real.arccos (((X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2)) / 
        (Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) * 
         Real.sqrt ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2)))
    angle A P B = angle B P C ∧ angle B P C = angle C P A

/-- The length of PC in a right triangle with a special interior point is 4 -/
theorem pc_length_is_four (t : RightTriangleWithPoint) : 
  Real.sqrt ((t.P.1 - t.C.1)^2 + (t.P.2 - t.C.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pc_length_is_four_l650_65067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_20_degrees_l650_65055

theorem cos_20_degrees (k : ℝ) (h : Real.sin (10 * π / 180) = k) :
  Real.cos (20 * π / 180) = 1 - 2 * k^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_20_degrees_l650_65055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l650_65095

/-- The function f(x) = (x-1)e^x - (3/2)x^2 -/
noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x - (3/2) * x^2

/-- Theorem stating that for all x₁ ∈ ℝ and x₂ > 0, f(x₁ + x₂) - f(x₁ - x₂) > -2x₂ -/
theorem f_inequality (x₁ x₂ : ℝ) (h : x₂ > 0) : f (x₁ + x₂) - f (x₁ - x₂) > -2 * x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l650_65095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_basket_count_l650_65073

theorem fruit_basket_count 
  (oranges apples bananas peaches : ℕ) : 
  oranges = 6 →
  apples = oranges - 2 →
  bananas = 3 * apples →
  peaches = bananas / 2 →
  oranges + apples + bananas + peaches = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_basket_count_l650_65073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_two_sevenths_l650_65023

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

theorem tan_theta_is_two_sevenths
  (θ : ℝ) (k : ℝ) (hk : k > 0)
  (h : rotation_matrix θ * dilation_matrix k = !![7, -2; 2, 7]) :
  Real.tan θ = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_is_two_sevenths_l650_65023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_max_area_l650_65024

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition given in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + Real.sqrt 3 * t.c * Real.sin t.B

/-- Area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  1 / 2 * t.a * t.c * Real.sin t.B

theorem triangle_angle_and_max_area (t : Triangle) 
  (h : triangle_condition t) (h_b : t.b = 1) :
  t.B = π / 6 ∧ 
  (∀ s : Triangle, triangle_condition s → s.b = 1 → 
    area s ≤ (2 + Real.sqrt 3) / 4) ∧
  area t = (2 + Real.sqrt 3) / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_max_area_l650_65024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_l650_65026

theorem cos_sin_sum (α : ℝ) (h1 : Real.cos (75 * Real.pi / 180 + α) = 1/2) 
  (h2 : Real.pi < α ∧ α < 3*Real.pi/2) : 
  Real.cos (105 * Real.pi / 180 - α) + Real.sin (α - 105 * Real.pi / 180) = 1/2 + Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_l650_65026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_hyperbola_l650_65060

-- Define the hyperbola and circle
def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 / 3 - y^2 / b^2 = 1
def hyperbola_circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

-- Define the asymptote of the hyperbola
def asymptote (b : ℝ) (x y : ℝ) : Prop := y = (b / Real.sqrt 3) * x

-- Define the intersection points M and N
def intersection_points (b : ℝ) (M N : ℝ × ℝ) : Prop :=
  asymptote b M.1 M.2 ∧ hyperbola_circle M.1 M.2 ∧
  asymptote b N.1 N.2 ∧ hyperbola_circle N.1 N.2

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem focal_distance_of_hyperbola (b : ℝ) (M N : ℝ × ℝ) :
  intersection_points b M N → distance M N = 2 → 
  ∃ (F₁ F₂ : ℝ × ℝ), hyperbola b F₁.1 F₁.2 ∧ hyperbola b F₂.1 F₂.2 ∧ distance F₁ F₂ = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_focal_distance_of_hyperbola_l650_65060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_eq_g_2004_l650_65001

/-- A function satisfying the given properties -/
noncomputable def g : ℝ → ℝ := sorry

/-- The first property of g -/
axiom g_scale (x : ℝ) (h : x > 0) : g (4 * x) = 4 * g x

/-- The second property of g -/
axiom g_def (x : ℝ) (h : 2 ≤ x ∧ x ≤ 6) : g x = 2 - |x - 3|

/-- The theorem stating the smallest x for which g(x) = g(2004) -/
theorem smallest_x_eq_g_2004 : 
  ∀ x : ℝ, x > 0 → g x = g 2004 → x ≥ 2048 ∧ g 2048 = g 2004 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_eq_g_2004_l650_65001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l650_65063

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 3) - Real.sqrt 3

theorem omega_range_for_three_zeros :
  ∀ ω : ℝ, (∀ x ∈ Set.Icc 0 (Real.pi / 2), f ω x = 0) ∧ 
           (∃! (x₁ x₂ x₃ : ℝ), x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
                                x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
                                x₃ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
                                x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
                                f ω x₁ = 0 ∧ f ω x₂ = 0 ∧ f ω x₃ = 0) ↔ 
  ω ∈ Set.Icc 4 (14 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_for_three_zeros_l650_65063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_m_l650_65003

def m : ℕ := 2^5 * 3^6 * 5^2 * 10^3

theorem number_of_factors_m : (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 378 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_m_l650_65003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_sum_l650_65049

-- Define the function as noncomputable
noncomputable def f (D E F : ℤ) (x : ℝ) : ℝ := x / (x^3 + D*x^2 + E*x + F)

-- State the theorem
theorem asymptotes_sum (D E F : ℤ) :
  (∀ x ∈ ({-3, 0, 4} : Set ℝ), ¬ ∃ y, f D E F x = y) →
  D + E + F = -13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_sum_l650_65049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_cut_ratio_for_10cm_square_l650_65068

/-- The ratio of the perimeter of a triangle resulting from folding and cutting a square
    to the perimeter of the original square -/
noncomputable def fold_cut_ratio (side_length : ℝ) : ℝ :=
  let half_side := side_length / 2
  let hypotenuse := Real.sqrt (side_length ^ 2 + half_side ^ 2)
  let triangle_perimeter := side_length + half_side + hypotenuse
  let square_perimeter := 4 * side_length
  triangle_perimeter / square_perimeter

/-- Theorem stating that for a square with side length 10 cm, the fold_cut_ratio
    is equal to (15 + √125) / 40 -/
theorem fold_cut_ratio_for_10cm_square :
  fold_cut_ratio 10 = (15 + Real.sqrt 125) / 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fold_cut_ratio_for_10cm_square_l650_65068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_constraint_implies_a_value_l650_65036

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^4

-- Define the domain
def domain : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem slope_constraint_implies_a_value (a : ℝ) :
  (∀ x y, x ∈ domain → y ∈ domain → x ≠ y → 
    (1/2 : ℝ) ≤ (f a y - f a x) / (y - x) ∧ (f a y - f a x) / (y - x) ≤ 4) →
  a = 9/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_constraint_implies_a_value_l650_65036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l650_65094

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := x * Real.exp (x + 1)
def g (k x : ℝ) : ℝ := k * Real.log x + k * (x + 1)
def h (k x : ℝ) : ℝ := f x - g k x

-- State the theorem
theorem function_properties (k : ℝ) (hk : k > 0) :
  (∃! x : ℝ, x > 0 ∧ f x = k) ∧
  (∀ x > 0, h k x ≥ 0 ↔ k ≤ Real.exp 1) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l650_65094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_exists_l650_65045

-- Define a rectangle by its left, right, bottom, and top coordinates
structure Rectangle where
  left : ℝ
  right : ℝ
  bottom : ℝ
  top : ℝ
  h1 : left ≤ right
  h2 : bottom ≤ top

-- Define the set of rectangles
def RectangleSet : Set Rectangle := sorry

-- Condition: Any two rectangles intersect
axiom rectangles_intersect : 
  ∀ (r1 r2 : Rectangle), r1 ∈ RectangleSet → r2 ∈ RectangleSet → 
    ∃ (x y : ℝ), r1.left ≤ x ∧ x ≤ r1.right ∧ r1.bottom ≤ y ∧ y ≤ r1.top ∧
                  r2.left ≤ x ∧ x ≤ r2.right ∧ r2.bottom ≤ y ∧ y ≤ r2.top

-- Theorem: There exists a point that belongs to all rectangles
theorem common_point_exists : ∃ (x y : ℝ), ∀ (r : Rectangle), r ∈ RectangleSet → 
  r.left ≤ x ∧ x ≤ r.right ∧ r.bottom ≤ y ∧ y ≤ r.top := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_point_exists_l650_65045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_exam_time_l650_65096

/-- Proves that given the conditions of Joan's exams, the time required for the Math exam is 1.5 hours -/
theorem joan_exam_time (english_questions math_questions : ℕ) (english_time extra_time_per_question : ℝ) :
  english_questions = 30 →
  math_questions = 15 →
  english_time = 1 →
  extra_time_per_question = 4/60 →
  let time_per_english_question := english_time / english_questions
  let time_per_math_question := time_per_english_question + extra_time_per_question
  let total_math_time := time_per_math_question * math_questions
  total_math_time = 1.5 := by
  sorry

#check joan_exam_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joan_exam_time_l650_65096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_c_incorrect_l650_65042

-- Define the space of points
variable {P : Type*}

-- Define the loci and conditions
variable (Locus1 Locus2 : Set P)
variable (X Y : P → Prop)

-- Define the intersection of Locus1 and Locus2
def Intersection (Locus1 Locus2 : Set P) : Set P := Locus1 ∩ Locus2

-- Define the statement C
def StatementC (Locus1 Locus2 : Set P) (X Y : P → Prop) : Prop :=
  (∀ p, p ∈ Intersection Locus1 Locus2 → X p ∧ Y p) ∧
  ¬(∀ p, X p ∧ Y p → p ∈ Intersection Locus1 Locus2)

-- Theorem: Statement C is incorrect for defining the intersection
theorem statement_c_incorrect (Locus1 Locus2 : Set P) (X Y : P → Prop) :
  ¬(StatementC Locus1 Locus2 X Y ↔ (∀ p, p ∈ Intersection Locus1 Locus2 ↔ X p ∧ Y p)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_c_incorrect_l650_65042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l650_65028

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - a

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ), 
    (f a 0 = -Real.sqrt 3) ∧ 
    (a = Real.sqrt 3) ∧
    (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) →
      f (Real.sqrt 3) x ∈ Set.Icc (-Real.sqrt 3) 2) ∧
    (∃ x₁ x₂, x₁ ∈ Set.Icc 0 (Real.pi / 2) ∧ 
              x₂ ∈ Set.Icc 0 (Real.pi / 2) ∧
              f (Real.sqrt 3) x₁ = -Real.sqrt 3 ∧ 
              f (Real.sqrt 3) x₂ = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l650_65028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_increase_l650_65089

/-- Given a line where an increase of 4 x-units results in an increase of 9 y-units,
    prove that an increase of 12 x-units results in an increase of 27 y-units. -/
theorem line_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 9) :
  ∀ x, f (x + 12) - f x = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_increase_l650_65089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l650_65072

-- Problem 1
theorem problem_1 : (2013 : ℝ)^0 + Real.sqrt 8 - (1/2)^(-1 : ℤ) + |2 - Real.sqrt 2| = 1 + Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 3) :
  (1 + 2 / (x + 1)) / ((x^2 - 9) / (x^2 + 2*x + 1)) = (x + 1) / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l650_65072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l650_65057

noncomputable def f (x : ℝ) := Real.tan (2 * x - Real.pi / 4)

theorem symmetry_center_of_tan :
  ∃ (k : ℤ), ((-Real.pi / 8 : ℝ), 0) = ((2 * k + 1) * Real.pi / 8, 0) ∧
  ∀ (x : ℝ), f ((-Real.pi / 8) + x) = -f ((-Real.pi / 8) - x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_tan_l650_65057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l650_65004

/-- Represents the fuel efficiency of a car in miles per gallon. -/
noncomputable def fuel_efficiency (distance : ℚ) (fuel_used : ℚ) : ℚ :=
  distance / fuel_used

/-- Theorem: A car that travels 100 miles using 5 gallons of gas has a fuel efficiency of 20 miles per gallon. -/
theorem car_fuel_efficiency :
  fuel_efficiency 100 5 = 20 := by
  unfold fuel_efficiency
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l650_65004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_possible_l650_65066

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := 1 - a / b

-- State the theorem
theorem arithmetic_operations_possible :
  ∃ (add sub mul div : ℝ → ℝ → ℝ),
    (∀ a b, add a b = a + b) ∧
    (∀ a b, sub a b = a - b) ∧
    (∀ a b, mul a b = a * b) ∧
    (∀ a b, div a b = a / b) ∧
    (∀ a b, add a b = a - (star b 0 + 0)) ∧
    (∀ a b, sub a b = star b a + a) ∧
    (∀ a b, mul a b = div a (div 1 b)) ∧
    (∀ a b, div a b = star (star a b) 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_operations_possible_l650_65066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l650_65051

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 4)^2 + (y + 3)^2 = 25

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop := 2*x + 3*y + 1 = 0

-- Theorem statement
theorem circle_equation_proof :
  -- The circle passes through the origin (0, 0)
  my_circle 0 0 ∧
  -- The circle passes through the point (1, 1)
  my_circle 1 1 ∧
  -- The center of the circle (4, -3) lies on the line 2x + 3y + 1 = 0
  center_line 4 (-3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_proof_l650_65051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_proportion_in_recipe_l650_65043

/-- Given a recipe with three ingredients in a ratio of 1:2:3 and a total of 6 cups,
    prove that the ingredient with the smallest proportion is 1 cup. -/
theorem smallest_proportion_in_recipe (a b c : ℚ) (total : ℚ) :
  a + b + c = 6 →
  (a : ℚ) / (b : ℚ) = 1 / 2 →
  (a : ℚ) / (c : ℚ) = 1 / 3 →
  total = 6 →
  a = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_proportion_in_recipe_l650_65043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l650_65088

theorem negation_of_exists_sin_greater_than_one :
  (¬ ∃ x : ℝ, Real.sin x > 1) ↔ (∀ x : ℝ, Real.sin x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exists_sin_greater_than_one_l650_65088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_thrown_percentage_l650_65047

/-- Calculates the percentage of apples thrown away over four days given the selling and discarding percentages for each day. -/
noncomputable def applesThrown (initialApples : ℝ) (sell1 sell2 sell3 sell4 : ℝ) (throw1 throw2 throw3 throw4 : ℝ) : ℝ :=
  let day1Remain := initialApples * (1 - sell1)
  let day1Thrown := day1Remain * throw1
  let day2Start := day1Remain - day1Thrown
  let day2Remain := day2Start * (1 - sell2)
  let day2Thrown := day2Remain * throw2
  let day3Start := day2Remain - day2Thrown
  let day3Remain := day3Start * (1 - sell3)
  let day3Thrown := day3Remain * throw3
  let day4Start := day3Remain - day3Thrown
  let day4Remain := day4Start * (1 - sell4)
  let day4Thrown := day4Remain * throw4
  (day1Thrown + day2Thrown + day3Thrown + day4Thrown) / initialApples * 100

/-- Theorem stating that the percentage of apples thrown away is approximately 22.53% -/
theorem apples_thrown_percentage : 
  ∀ (initialApples : ℝ), initialApples > 0 →
  |applesThrown initialApples 0.555 0.47 0.625 0.287 (1/3) 0.35 0.5 0.201 - 22.53| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apples_thrown_percentage_l650_65047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_six_eq_43_over_16_l650_65076

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x - 9

noncomputable def g (x : ℝ) : ℝ := 
  let y := (x + 9) / 4
  3 * y^2 + 4 * y - 2

-- State the theorem
theorem g_of_negative_six_eq_43_over_16 : g (-6) = 43 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_negative_six_eq_43_over_16_l650_65076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l650_65054

-- Define the rounding function R as noncomputable
noncomputable def R (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

-- Define the theorem
theorem range_of_x (x : ℝ) (h : R (R (x + 1) / 2) = 5) : 
  7.5 ≤ x ∧ x < 9.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l650_65054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l650_65027

/-- Given two vectors in ℝ³, prove that the magnitude of a specific linear combination is √17 -/
theorem magnitude_of_vector_combination (a b : ℝ × ℝ × ℝ) : 
  a = (1, 1, 0) → b = (-1, 0, 2) → 
  ‖(2 * a.1 - b.1, 2 * a.2.1 - b.2.1, 2 * a.2.2 - b.2.2)‖ = Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_combination_l650_65027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_eccentricity_product_l650_65087

/-- The eccentricity of a hyperbola with asymptotes y = ± k x -/
noncomputable def hyperbola_eccentricity (k : ℝ) : ℝ := Real.sqrt (1 + k^2)

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 -/
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (min a b / max a b)^2)

theorem hyperbola_ellipse_eccentricity_product (b : ℝ) :
  b > 0 →
  (hyperbola_eccentricity (3/4) * ellipse_eccentricity 2 b = 1) →
  (b = 6/5 ∨ b = 10/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_ellipse_eccentricity_product_l650_65087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l650_65048

def fibonacci_seq : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => fibonacci_seq n + fibonacci_seq (n + 1)

def floor (x : ℚ) : ℤ := Int.floor x

def frac (x : ℚ) : ℚ := x - ↑(floor x)

theorem fibonacci_product_theorem :
  (floor (fibonacci_seq 1 / fibonacci_seq 0 : ℚ)) *
  (Finset.prod (Finset.range 96) (fun k => frac (fibonacci_seq (k + 2) / fibonacci_seq (k + 1) : ℚ))) *
  (floor (fibonacci_seq 97 / fibonacci_seq 1 : ℚ)) = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l650_65048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l650_65069

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ × ℝ := 
  (2 * Real.sqrt 2 * Real.cos θ * Real.cos θ, 2 * Real.sqrt 2 * Real.cos θ * Real.sin θ)

-- Define point A
def A : ℝ × ℝ := (1, 0)

-- Define the locus of P
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := 
  (3 - Real.sqrt 2 + 2 * Real.cos θ, 2 * Real.sin θ)

-- State the theorem
theorem no_intersection :
  ∀ θ₁ θ₂ : ℝ, C θ₁ ≠ C₁ θ₂ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l650_65069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_is_linear_in_two_variables_l650_65084

/-- Represents a polynomial equation in two variables -/
structure PolynomialEquation (α : Type*) [Field α] where
  lhs : MvPolynomial (Fin 2) α
  rhs : MvPolynomial (Fin 2) α

/-- Checks if a polynomial equation is linear in two variables -/
def isLinearInTwoVariables (eq : PolynomialEquation ℚ) : Prop :=
  (eq.lhs.vars.card + eq.rhs.vars.card ≤ 2) ∧
  (eq.lhs.totalDegree ≤ 1) ∧
  (eq.rhs.totalDegree ≤ 1)

/-- The equation 4x = (y-2)/4 represented as a polynomial equation -/
noncomputable def equation : PolynomialEquation ℚ := {
  lhs := 16 * MvPolynomial.X 0
  rhs := MvPolynomial.X 1 - 2
}

/-- Theorem stating that the given equation is linear in two variables -/
theorem equation_is_linear_in_two_variables :
  isLinearInTwoVariables equation := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_is_linear_in_two_variables_l650_65084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_11pi_12_l650_65033

theorem cos_alpha_minus_11pi_12 (α : ℝ) 
  (h : Real.sin (7 * π / 12 + α) = 2 / 3) : 
  Real.cos (α - 11 * π / 12) = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_11pi_12_l650_65033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_to_the_power_of_y_equals_216_l650_65040

-- Define the ⊘ operation
noncomputable def circledSlash (a b : ℝ) : ℝ := a * b - a / b

-- Define x and y
noncomputable def x : ℝ := circledSlash 4 2
noncomputable def y : ℝ := circledSlash 2 2

-- Theorem statement
theorem x_to_the_power_of_y_equals_216 : x ^ y = 216 := by
  -- Expand the definitions of x and y
  have hx : x = 6 := by
    unfold x circledSlash
    ring
  have hy : y = 3 := by
    unfold y circledSlash
    ring
  -- Rewrite x and y with their values
  rw [hx, hy]
  -- Evaluate 6^3
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_to_the_power_of_y_equals_216_l650_65040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_concyclicity_l650_65065

-- Define the parabolas
def parabola1 (x : ℝ) : ℝ := x^2
def parabola2 (x : ℝ) : ℝ := 2009 * x^2

-- Define a point on a parabola
structure PointOnParabola (f : ℝ → ℝ) where
  x : ℝ
  y : ℝ
  on_parabola : y = f x

-- Define concyclicity
def are_concyclic (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  let (x4, y4) := p4
  Matrix.det !![x1, y1, x1^2 + y1^2, 1;
                x2, y2, x2^2 + y2^2, 1;
                x3, y3, x3^2 + y3^2, 1;
                x4, y4, x4^2 + y4^2, 1] = 0

theorem parabola_concyclicity 
  (A1 A2 A3 A4 : PointOnParabola parabola1)
  (B1 B2 B3 B4 : PointOnParabola parabola2)
  (h_x1 : A1.x = B1.x) (h_x2 : A2.x = B2.x) (h_x3 : A3.x = B3.x) (h_x4 : A4.x = B4.x)
  (h_concyclic : are_concyclic (A1.x, A1.y) (A2.x, A2.y) (A3.x, A3.y) (A4.x, A4.y)) :
  are_concyclic (B1.x, B1.y) (B2.x, B2.y) (B3.x, B3.y) (B4.x, B4.y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_concyclicity_l650_65065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l650_65091

theorem unique_solution : ∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ a^3 - a^2*b + a^2 + 2*a + 2*b + 1 = 0 ∧ a = 5 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l650_65091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_perpendicular_points_sum_inverse_squares_l650_65030

/-- The curve C defined by x²/4 + y² = 1 -/
def CurveC (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The distance from the origin to a point (x, y) -/
noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- Two vectors are perpendicular if their dot product is zero -/
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

theorem curve_c_perpendicular_points_sum_inverse_squares :
  ∀ (x1 y1 x2 y2 : ℝ),
    CurveC x1 y1 → CurveC x2 y2 → perpendicular x1 y1 x2 y2 →
    1 / (distance_from_origin x1 y1)^2 + 1 / (distance_from_origin x2 y2)^2 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_c_perpendicular_points_sum_inverse_squares_l650_65030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l650_65097

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

-- Theorem statement
theorem function_inequality_range (x : ℝ) :
  f x + f (x - 1/2) > 1 ↔ x > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_range_l650_65097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l650_65086

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 1)

theorem f_range : Set.range f = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l650_65086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l650_65010

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions and theorems
noncomputable def area (t : Triangle) : ℝ := (1/2) * t.a * t.b * Real.sin t.C

theorem part1 (t : Triangle) 
  (h1 : area t = Real.sqrt 3 / 2)
  (h2 : t.c = 2)
  (h3 : t.A = π / 3) : 
  t.a = Real.sqrt 3 ∧ t.b = 1 := by sorry

theorem part2 (t : Triangle)
  (h1 : t.c = t.a * Real.cos t.B)
  (h2 : (t.a + t.b + t.c) * (t.a + t.b - t.c) = (2 + Real.sqrt 2) * t.a * t.b) :
  t.A = π / 2 ∧ t.B = π / 4 ∧ t.C = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l650_65010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_for_triple_f_l650_65038

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x)

theorem three_solutions_for_triple_f :
  ∃! (s : Finset ℝ), s.card = 3 ∧ 
    (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 ∧ f (f (f x)) = f x) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 2 ∧ f (f (f x)) = f x → x ∈ s) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_for_triple_f_l650_65038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l650_65080

noncomputable section

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the asymptote of the hyperbola
def asymptote (a b : ℝ) (x y : ℝ) : Prop := y = (b/a)*x

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2/a^2)

theorem hyperbola_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hA : ∃ x y : ℝ, parabola p x y ∧ asymptote a b x y) 
  (hd : ∀ x y : ℝ, parabola p x y ∧ asymptote a b x y → x + p/2 = p) :
  eccentricity a b = Real.sqrt 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l650_65080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l650_65013

/-- Represents the structure of an athletic track -/
structure AthleticTrack where
  innerRadius : ℝ
  straightLength : ℝ

/-- Calculates the length of a path around the track at a given offset from the inner edge -/
noncomputable def pathLength (track : AthleticTrack) (offset : ℝ) : ℝ :=
  2 * track.straightLength + 2 * Real.pi * (track.innerRadius + offset)

/-- Theorem: Given the specified track conditions, the walker's speed is π/3 m/s -/
theorem walker_speed (track : AthleticTrack) (speed : ℝ) 
    (h1 : pathLength track 8 / speed = pathLength track 0 / speed + 48)
    (h2 : speed > 0) : speed = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walker_speed_l650_65013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_origin_to_line_l650_65053

/-- The minimum distance from the origin to the line 2x - y + 5 = 0 is √5 -/
theorem min_distance_origin_to_line : 
  let line := {P : ℝ × ℝ | 2 * P.fst - P.snd + 5 = 0}
  ∃ d : ℝ, d = Real.sqrt 5 ∧ ∀ P ∈ line, d ≤ Real.sqrt (P.fst^2 + P.snd^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_origin_to_line_l650_65053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l650_65021

noncomputable section

-- Define the line l
def line_l (x : ℝ) : ℝ := Real.sqrt 3 * x

-- Define the curve C
def curve_C (x y : ℝ) : Prop := (x - 2)^2 + (y - Real.sqrt 3)^2 = 3

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
  line_l A.1 = A.2 ∧ line_l B.1 = B.2

-- Define the distance from origin to a point
noncomputable def distance_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1^2 + p.2^2)

-- Theorem statement
theorem intersection_product (A B : ℝ × ℝ) :
  intersection_points A B →
  distance_from_origin A * distance_from_origin B = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l650_65021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rotation_theorem_l650_65058

/-- Represents the state of the blackboard after rotations -/
def BlackboardState := Fin 12 → ℕ

/-- The sum of all numbers on the clock face -/
def clockSum : ℕ := 78

/-- Rotates the clock and updates the blackboard state -/
def rotate (state : BlackboardState) (angle : ℕ) : BlackboardState :=
  λ i ↦ state i + angle / 30 * ((i.val : ℕ) + 1)

/-- Checks if all numbers on the blackboard are equal to the target -/
def allEqual (state : BlackboardState) (target : ℕ) : Prop :=
  ∀ i, state i = target

/-- Counts how many numbers on the blackboard are equal to the target -/
def countEqual (state : BlackboardState) (target : ℕ) : ℕ :=
  (Finset.univ.filter (λ i ↦ state i = target)).card

/-- The main theorem to be proved -/
theorem clock_rotation_theorem :
  (¬ ∃ (n : ℕ) (state : BlackboardState), 
    (∀ i, state i = 0) ∧ allEqual (n.iterate (λ s ↦ rotate s 30) state) 1984) ∧
  (¬ ∃ (n : ℕ) (state : BlackboardState), 
    (∀ i, state i = 0) ∧ countEqual (n.iterate (λ s ↦ rotate s 30) state) 1984 = 11) ∧
  (∃ (n : ℕ) (state : BlackboardState), 
    (∀ i, state i = 0) ∧ countEqual (n.iterate (λ s ↦ rotate s 30) state) 1984 = 10) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_rotation_theorem_l650_65058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l650_65017

/-- Represents the time it takes to fill a tank with two pipes -/
noncomputable def fill_time (pipe_a_time : ℝ) (pipe_b_multiplier : ℝ) : ℝ :=
  1 / (1 / pipe_a_time + pipe_b_multiplier * (1 / pipe_a_time))

/-- Theorem stating that with given conditions, the tank will be filled in 2 minutes -/
theorem tank_fill_time :
  let pipe_a_time : ℝ := 6
  let pipe_b_multiplier : ℝ := 2
  fill_time pipe_a_time pipe_b_multiplier = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_fill_time_l650_65017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_padic_solutions_l650_65046

/-- A p-adic number -/
structure PAdicNumber where
  value : ℚ -- We'll use rationals as a simplified representation for p-adic numbers

/-- Specific p-adic numbers mentioned in the problem -/
noncomputable def d : PAdicNumber := ⟨90625/100000⟩
noncomputable def e : PAdicNumber := ⟨9376/100000⟩

instance : HMul PAdicNumber PAdicNumber PAdicNumber where
  hMul a b := ⟨a.value * b.value⟩

instance : OfNat PAdicNumber n where
  ofNat := ⟨n⟩

/-- Function to check if a p-adic number satisfies x^2 = x -/
def satisfiesEquation (x : PAdicNumber) : Prop := x * x = x

/-- The main theorem stating that there are exactly four p-adic solutions to x^2 = x -/
theorem four_padic_solutions :
  ∃! (s : Finset PAdicNumber), s.card = 4 ∧ 
  (∀ x : PAdicNumber, x ∈ s ↔ satisfiesEquation x) ∧
  (0 : PAdicNumber) ∈ s ∧ (1 : PAdicNumber) ∈ s ∧ d ∈ s ∧ e ∈ s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_padic_solutions_l650_65046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_pairing_l650_65037

theorem remainder_pairing (n : ℕ) (a₁ a₂ a₃ a₄ : ℤ) 
  (h_n : n ≥ 2)
  (h_coprime : ∀ i : Fin 4, Int.gcd n (List.get [a₁, a₂, a₃, a₄] i) = 1)
  (h_sum : ∀ k ∈ Finset.range (n-1), 
    (k * a₁ % n) + (k * a₂ % n) + (k * a₃ % n) + (k * a₄ % n) = 2 * n) :
  ∃ (i j k l : Fin 4), i ≠ j ∧ k ≠ l ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧
    (List.get [a₁, a₂, a₃, a₄] i % n) + (List.get [a₁, a₂, a₃, a₄] j % n) = n ∧
    (List.get [a₁, a₂, a₃, a₄] k % n) + (List.get [a₁, a₂, a₃, a₄] l % n) = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_pairing_l650_65037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l650_65041

/-- The number of bricks required to pave a rectangular courtyard -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℕ :=
  (courtyard_length * 100 * courtyard_width * 100 / (brick_length * brick_width)).floor.toNat

/-- Theorem stating the number of bricks required for the given courtyard and brick dimensions -/
theorem courtyard_paving : 
  bricks_required 25 16 20 10 = 20000 := by
  sorry

#eval bricks_required 25 16 20 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courtyard_paving_l650_65041
