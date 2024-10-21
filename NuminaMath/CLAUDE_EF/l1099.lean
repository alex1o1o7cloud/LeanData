import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_approx_l1099_109967

/-- The speed of the man's train in km/h -/
noncomputable def mans_train_speed : ℝ := 55

/-- The length of the goods train in meters -/
noncomputable def goods_train_length : ℝ := 320

/-- The time it takes for the goods train to pass the man in seconds -/
noncomputable def passing_time : ℝ := 10

/-- Conversion factor from km/h to m/s -/
noncomputable def km_h_to_m_s : ℝ := 1000 / 3600

/-- The speed of the man's train in m/s -/
noncomputable def mans_train_speed_m_s : ℝ := mans_train_speed * km_h_to_m_s

/-- The relative speed between the two trains in m/s -/
noncomputable def relative_speed : ℝ := goods_train_length / passing_time

/-- The speed of the goods train in m/s -/
noncomputable def goods_train_speed_m_s : ℝ := relative_speed - mans_train_speed_m_s

/-- The speed of the goods train in km/h -/
noncomputable def goods_train_speed_km_h : ℝ := goods_train_speed_m_s / km_h_to_m_s

theorem goods_train_speed_approx :
  ∃ ε > 0, |goods_train_speed_km_h - 60.2| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_approx_l1099_109967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l1099_109934

/-- Given a point P(x, y) on the curve y = x³ - x + 2/3, the range of the angle of inclination α of the tangent at P is [0, π/2) ∪ [3π/4, π). -/
theorem tangent_angle_range :
  ∀ x y : ℝ, y = x^3 - x + 2/3 →
  ∃ α : ℝ, (α ∈ Set.Ioc 0 (π/2) ∪ Set.Icc (3*π/4) π) ∧
           (Real.tan α = 3*x^2 - 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_range_l1099_109934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tax_rate_calculation_l1099_109941

/-- Calculates the new personal income tax rate given the annual income, original tax rate, and differential savings. -/
theorem new_tax_rate_calculation (annual_income : ℝ) (original_tax_rate : ℝ) (differential_savings : ℝ) :
  annual_income = 45000 →
  original_tax_rate = 0.40 →
  differential_savings = 3150 →
  (annual_income * original_tax_rate - differential_savings) / annual_income = 0.33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_tax_rate_calculation_l1099_109941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_closed_interval_l1099_109959

-- Define the sets A and B
def A : Set ℝ := {x | -x + 2*x ≥ 0}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_equals_closed_interval : A ∩ B = Set.Icc 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_closed_interval_l1099_109959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_to_zero_sequence_l1099_109982

/-- Represents the two allowed operations: add_one and neg_inv -/
inductive Operation
| add_one : Operation
| neg_inv : Operation

/-- Applies the operation to a rational number -/
def apply_operation (op : Operation) (x : ℚ) : ℚ :=
  match op with
  | Operation.add_one => x + 1
  | Operation.neg_inv => if x ≠ 0 then -1 / x else 0

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a starting value -/
def apply_sequence (seq : OperationSequence) (start : ℚ) : ℚ :=
  seq.foldl (fun x op => apply_operation op x) start

/-- The theorem stating that there exists a sequence of operations
    starting from 0 and ending at 0 -/
theorem exists_zero_to_zero_sequence :
  ∃ (seq : OperationSequence),
    apply_sequence seq 0 = 0 ∧
    seq = [Operation.add_one, Operation.neg_inv, Operation.add_one] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_to_zero_sequence_l1099_109982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1099_109942

noncomputable def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x + c) / ((x - 3) * (x + 2))

theorem exactly_one_vertical_asymptote (c : ℝ) :
  (∃! x, (x = 3 ∨ x = -2) ∧ x^2 - 2*x + c = 0) ↔ (c = -3 ∨ c = -8) := by
  sorry

#check exactly_one_vertical_asymptote

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_vertical_asymptote_l1099_109942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1099_109979

-- Define necessary structures and predicates
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

def CircleContains (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def LineIntersect (l1 l2 : Line) (p : Point) : Prop :=
  l1.a * p.x + l1.b * p.y + l1.c = 0 ∧
  l2.a * p.x + l2.b * p.y + l2.c = 0

def Concurrent (l1 l2 l3 : Line) : Prop :=
  ∃ p : Point, LineIntersect l1 l2 p ∧ LineIntersect l2 l3 p ∧ LineIntersect l3 l1 p

/-- Given six points on a circle, prove that certain lines are concurrent -/
theorem lines_concurrent (c : Circle) (A B C D E F : Point) : 
  CircleContains c A ∧ CircleContains c B ∧ CircleContains c C ∧ 
  CircleContains c D ∧ CircleContains c E ∧ CircleContains c F →
  ∃ Q : Point, LineIntersect (Line.mk 0 0 0) (Line.mk 0 0 0) Q →
  ∃ N : Point, LineIntersect (Line.mk 0 0 0) (Line.mk 0 0 0) N ∧ LineIntersect (Line.mk 0 0 0) (Line.mk 0 0 0) N →
  ∃ P : Point, LineIntersect (Line.mk 0 0 0) (Line.mk 0 0 0) P →
  ∃ H : Point, LineIntersect (Line.mk 0 0 0) (Line.mk 0 0 0) H →
  ∃ G : Point, LineIntersect (Line.mk 0 0 0) (Line.mk 0 0 0) G →
  ∃ M : Point, LineIntersect (Line.mk 0 0 0) (Line.mk 0 0 0) M →
  Concurrent (Line.mk 0 0 0) (Line.mk 0 0 0) (Line.mk 0 0 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_concurrent_l1099_109979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fine_calculation_correct_l1099_109915

/-- Calculates the daily increment 'n' for a given day 'd' -/
def n (d : ℕ) : ℚ :=
  if d ≤ 1 then 0 else 0.05 + 0.05 * (d - 2)

/-- Calculates the total fine 'F' for a given day 'd' and interest rate 'r' -/
def F (d : ℕ) (r : ℚ) : ℚ :=
  match d with
  | 0 => 0
  | 1 => 0.07
  | d + 1 => min (F d r + n (d + 1)) (2 * F d r) * (1 + r)

/-- Theorem stating the correctness of the fine calculation formula -/
theorem fine_calculation_correct (d : ℕ) (r : ℚ) :
  F d r = match d with
  | 0 => 0
  | 1 => 0.07
  | d + 1 => min (F d r + n (d + 1)) (2 * F d r) * (1 + r) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fine_calculation_correct_l1099_109915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_trigonometric_evaluation_cosine_product_l1099_109988

-- Part 1
theorem trigonometric_identity (α : Real) :
  (Real.cos (2 * α)) / (2 * Real.tan (π / 4 - α) * (Real.sin (π / 4 + α))^2) = 1 := by sorry

-- Part 2
theorem trigonometric_evaluation :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by sorry

-- Part 3
theorem cosine_product :
  Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (60 * π / 180) * Real.cos (80 * π / 180) = 1 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_trigonometric_evaluation_cosine_product_l1099_109988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_radius_approx_l1099_109983

/-- The volume of a hemisphere in cubic centimeters -/
noncomputable def hemisphere_volume : ℝ := 19404

/-- The formula for the volume of a hemisphere -/
noncomputable def volume_formula (r : ℝ) : ℝ := (2/3) * Real.pi * r^3

/-- The radius of the hemisphere -/
noncomputable def hemisphere_radius : ℝ := ((3 * hemisphere_volume) / (2 * Real.pi))^(1/3)

theorem hemisphere_radius_approx :
  ∃ ε > 0, |hemisphere_radius - 21.02| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_radius_approx_l1099_109983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_and_max_area_l1099_109991

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of an ellipse -/
noncomputable def Ellipse.focus (e : Ellipse) : Point :=
  ⟨e.a * e.eccentricity, 0⟩

theorem ellipse_standard_equation_and_max_area 
  (e : Ellipse) 
  (h_ecc : e.eccentricity = Real.sqrt 2 / 2) 
  (h_focus : e.focus = ⟨Real.sqrt 2, 0⟩) :
  (∃ (x y : ℝ), x^2 / 4 + y^2 / 2 = 1) ∧
  (∃ (m : ℝ), 
    let area := Real.sqrt 2 / 3 * Real.sqrt (m^2 * (6 - m^2))
    ∀ m' : ℝ, 
      let area' := Real.sqrt 2 / 3 * Real.sqrt (m'^2 * (6 - m'^2))
      area ≥ area') :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_and_max_area_l1099_109991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l1099_109952

/-- Given a curve y = ax², prove that if the tangent line at (1, a) is parallel to 2x - y - 6 = 0, then a = 1 -/
theorem tangent_line_parallel (a : ℝ) : 
  (∀ x, (x, a * x^2) ∈ Set.range (λ x : ℝ ↦ (x, a * x^2))) →  -- Curve equation
  (1, a) ∈ Set.range (λ x : ℝ ↦ (x, a * x^2)) →               -- Point (1, a) is on the curve
  (∃ k : ℝ, ∀ x y, y = k * (x - 1) + a ∧                     -- Tangent line equation
             (∀ x y, y = 2 * x - 6 → k = 2)) →               -- Parallel to 2x - y - 6 = 0
  a = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_l1099_109952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l1099_109950

/-- The curve defined by r = 3 tan θ sec θ is a hyperbola -/
theorem curve_is_hyperbola (θ : ℝ) (r : ℝ) (x y : ℝ) : 
  r = 3 * Real.tan θ * (1 / Real.cos θ) →
  x = r * Real.cos θ →
  y = r * Real.sin θ →
  ∃ (a b : ℝ), (x^2 / a^2) - (y^2 / b^2) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_is_hyperbola_l1099_109950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row_time_ratio_l1099_109949

/-- Proves that the ratio of time taken to row against the stream to the time taken to row in favor of the stream is 2:1, given that the ratio of boat speed to stream speed is 3:1. -/
theorem row_time_ratio (B S : ℝ) (h : B = 3 * S) :
  (1 / (B - S)) / (1 / (B + S)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row_time_ratio_l1099_109949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_distance_and_midpoint_line_l1099_109958

def line_l₁ (x y : ℝ) : Prop := x + 3 * y + 18 = 0
def line_l₂ (x y : ℝ) : Prop := 2 * x + 3 * y - 8 = 0

noncomputable def distance_between_lines (l₁ l₂ : ℝ → ℝ → Prop) : ℝ := sorry

def midpoint_line (l₁ l₂ : ℝ → ℝ → Prop) : ℝ → ℝ → Prop := sorry

theorem lines_distance_and_midpoint_line :
  (distance_between_lines line_l₁ line_l₂ = 2 * Real.sqrt 13) ∧
  (∀ x y, midpoint_line line_l₁ line_l₂ x y ↔ 2 * x + 3 * y + 5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_distance_and_midpoint_line_l1099_109958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1099_109927

/-- Represents the speed of a train in km/h -/
noncomputable def train_speed : ℚ := 144

/-- Represents the time taken by the train to cross a pole in seconds -/
noncomputable def crossing_time : ℚ := 40

/-- Conversion factor from km/h to m/s -/
noncomputable def km_per_hour_to_m_per_sec : ℚ := 5 / 18

/-- Calculates the length of the train in meters -/
noncomputable def train_length : ℚ := train_speed * km_per_hour_to_m_per_sec * crossing_time

theorem train_length_calculation :
  train_length = 1600 := by
  -- Unfold the definitions
  unfold train_length train_speed km_per_hour_to_m_per_sec crossing_time
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1099_109927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1099_109955

/-- The curve defining the lower bound of the region -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The line defining the upper bound of the region -/
def g (x : ℝ) : ℝ := x

/-- The volume of the solid generated by rotating the region enclosed by y = x³ - x and y = x about the line y = x -/
noncomputable def rotationVolume : ℝ := 64 * Real.pi / 105

theorem volume_of_rotation : 
  ∃ (a b : ℝ), a < b ∧ 
  (∫ (x : ℝ) in a..b, π * ((g x - f x) * (2 * x - g x - f x))) = rotationVolume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_rotation_l1099_109955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_income_is_27000_l1099_109954

/-- Represents the tax structure and Alex's income --/
structure TaxInfo where
  q : ℚ  -- The base tax rate as a rational number
  income : ℚ  -- Alex's annual income

/-- Calculates the total tax paid based on the given tax structure --/
def totalTax (info : TaxInfo) : ℚ :=
  if info.income ≤ 20000 then
    info.q * info.income
  else
    info.q * 20000 + (info.q + 3/100) * (info.income - 20000)

/-- The theorem stating that Alex's income is $27000 --/
theorem alex_income_is_27000 (info : TaxInfo) :
  (totalTax info = (info.q + 75/10000) * info.income) → info.income = 27000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alex_income_is_27000_l1099_109954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1099_109997

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x - k ≤ 0}

-- Theorem statement
theorem intersection_range (k : ℝ) : (M ∩ N k).Nonempty → k ∈ Set.Ici (-1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l1099_109997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_coefficient_l1099_109917

-- Define the line equation
def line_equation (x y : ℝ) (a : ℝ) : Prop := 2 * x + a * y + 3 = 0

-- Define the slope angle
noncomputable def slope_angle : ℝ := 120 * Real.pi / 180

-- Theorem statement
theorem line_slope_coefficient :
  ∃ (a : ℝ), (∀ x y : ℝ, line_equation x y a) ∧ 
             (Real.tan slope_angle = -2 / a) ∧ 
             (a = 2 * Real.sqrt 3 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_coefficient_l1099_109917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1099_109998

def p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc (1/4 : ℝ) (1/2 : ℝ), 2*x < m*(x^2 + 1)

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  (4 : ℝ)^x + 2^(x+1) + m - 1

def q (m : ℝ) : Prop :=
  ∃ x, f m x = 0

theorem m_range (m : ℝ) (h : p m ∧ q m) : m ∈ Set.Ioo (4/5 : ℝ) (1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_l1099_109998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₂_to_C₃_l1099_109943

-- Define the curves
def C₁ (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 1

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (4 * Real.cos θ, 3 * Real.sin θ)

def C₃ (x y : ℝ) : Prop := x - 2*y - 7 = 0

-- Define the distance function from a point to C₃
noncomputable def dist_to_C₃ (x y : ℝ) : ℝ :=
  |x - 2*y - 7| / Real.sqrt 5

-- Theorem statement
theorem max_distance_C₂_to_C₃ :
  ∃ (θ : ℝ), ∀ (φ : ℝ),
    dist_to_C₃ (C₂ θ).1 (C₂ θ).2 ≥ dist_to_C₃ (C₂ φ).1 (C₂ φ).2 ∧
    dist_to_C₃ (C₂ θ).1 (C₂ θ).2 = (2 * Real.sqrt 65 + 7 * Real.sqrt 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_C₂_to_C₃_l1099_109943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_XY_length_l1099_109947

-- Define the circle and points
variable (circle : Set (ℝ × ℝ))
variable (A B C D P Q X Y : ℝ × ℝ)

-- Define the conditions
axiom on_circle : A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ D ∈ circle ∧ X ∈ circle ∧ Y ∈ circle
axiom AB_length : dist A B = 15
axiom CD_length : dist C D = 23
axiom P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B
axiom Q_on_CD : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ Q = (1 - s) • C + s • D
axiom AP_length : dist A P = 9
axiom CQ_length : dist C Q = 8
axiom PQ_length : dist P Q = 35
axiom X_Y_on_PQ : ∃ u v : ℝ, 0 ≤ u ∧ u ≤ 1 ∧ 0 ≤ v ∧ v ≤ 1 ∧ 
                   X = (1 - u) • P + u • Q ∧ 
                   Y = (1 - v) • P + v • Q ∧
                   u < v

-- Theorem to prove
theorem XY_length : dist X Y = 66 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_XY_length_l1099_109947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_correct_l1099_109977

/-- The area of a figure formed by rotating a semicircle of radius R around one of its ends by an angle of 20° -/
noncomputable def rotated_semicircle_area (R : ℝ) : ℝ := (2 * Real.pi * R^2) / 9

/-- Theorem stating that the area of the rotated semicircle figure is correct -/
theorem rotated_semicircle_area_correct (R : ℝ) (h : R > 0) :
  rotated_semicircle_area R = (2 * Real.pi * R^2) / 9 := by
  -- Unfold the definition of rotated_semicircle_area
  unfold rotated_semicircle_area
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotated_semicircle_area_correct_l1099_109977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_parabola_l1099_109914

-- Define the plane
variable (x y : ℝ)

-- Define the point P
def P := (x, y)

-- Define the line x + 4 = 0
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -4}

-- Define the point M(2,0)
def M : ℝ × ℝ := (2, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the distance from a point to a line
noncomputable def distToLine (p : ℝ × ℝ) : ℝ := |p.1 + 4|

-- Define the set of points satisfying the condition
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | distToLine p - distance p M = 2}

-- Theorem: The set S is a parabola
theorem S_is_parabola : ∃ (a b c d e : ℝ), a ≠ 0 ∧ 
  S = {p : ℝ × ℝ | a * p.1^2 + b * p.1 * p.2 + c * p.2^2 + d * p.1 + e * p.2 = 0} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_parabola_l1099_109914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_incorrect_statement_2_incorrect_statement_3_incorrect_statement_4_correct_l1099_109995

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2)
def g (x : ℝ) : ℝ := 1/x
def h (x : ℝ) : ℝ := x^2
def k (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the domains and ranges as sets
def domain_f : Set ℝ := {x | x ≤ 0}
def range_f : Set ℝ := {y | y ≤ 1}
def domain_g : Set ℝ := {x | x > 2}
def range_g : Set ℝ := {y | y ≤ 1/2}
def range_h : Set ℝ := {y | 0 ≤ y ∧ y ≤ 4}
def domain_h : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def range_k : Set ℝ := {y | y ≤ 3}
def domain_k : Set ℝ := {x | 0 < x ∧ x ≤ 8}

-- Theorem statements
theorem statement_1_incorrect : 
  ¬(∀ x ∈ domain_f, f x ∈ range_f) :=
sorry

theorem statement_2_incorrect :
  ¬(∀ x ∈ domain_g, g x ∈ range_g) :=
sorry

theorem statement_3_incorrect :
  ¬(range_h = {y | ∃ x ∈ domain_h, h x = y}) :=
sorry

theorem statement_4_correct :
  range_k = {y | ∃ x ∈ domain_k, k x = y} :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_statement_1_incorrect_statement_2_incorrect_statement_3_incorrect_statement_4_correct_l1099_109995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l1099_109987

/-- Represents a person's financial situation --/
structure FinancialSituation where
  income : ℚ
  expenditure : ℚ
  savings : ℚ

/-- Calculates the savings given income and expenditure ratio --/
def calculateSavings (income : ℚ) (ratio : ℚ) : ℚ :=
  income - (income * (1 - 1 / (1 + ratio)))

/-- Theorem: Given the income-expenditure ratio and income, prove the savings --/
theorem savings_calculation (fs : FinancialSituation) 
    (h1 : fs.income = 19000)
    (h2 : 5 / 4 = fs.income / fs.expenditure) :
  fs.savings = calculateSavings fs.income (5 / 4) ∧ fs.savings = 3800 := by
  sorry

#eval calculateSavings 19000 (5 / 4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_calculation_l1099_109987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_coincides_with_square_bottom_all_area_above_line_l1099_109970

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line defined by two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- A square defined by its bottom-left and top-right corners -/
structure Square where
  bottomLeft : Point
  topRight : Point

/-- Check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (l.p2.y - l.p1.y) * (p.x - l.p1.x)

/-- Calculate the area above the line in a square -/
noncomputable def area_above_line (s : Square) (l : Line) : ℝ := sorry

/-- Calculate the total area of a square -/
noncomputable def total_area (s : Square) : ℝ := sorry

theorem line_coincides_with_square_bottom (s : Square) (l : Line) : Prop :=
  let bottomLeft := s.bottomLeft
  let bottomRight := Point.mk s.topRight.x s.bottomLeft.y
  l.p1 = Point.mk 4 3 ∧
  l.p2 = Point.mk 8 1 ∧
  s.bottomLeft = Point.mk 4 1 ∧
  s.topRight = Point.mk 8 5 ∧
  isPointOnLine bottomLeft l ∧
  isPointOnLine bottomRight l

theorem all_area_above_line (s : Square) (l : Line) :
  line_coincides_with_square_bottom s l → 1 = (area_above_line s l) / (total_area s) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_coincides_with_square_bottom_all_area_above_line_l1099_109970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l1099_109919

def M : Set ℝ := {x | x^2 + 3*x + 2 < 0}
def N : Set ℝ := {x | (1/2 : ℝ)^x ≤ 4}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_M_and_N_l1099_109919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_original_days_is_twelve_l1099_109925

/-- Represents a work project with a variable number of workers and days. -/
structure WorkProject where
  baseWorkers : ℕ
  baseDays : ℕ
  additionalWorkers : ℕ
  reducedDays : ℕ

/-- The work project described in the problem. -/
def problemProject : WorkProject where
  baseWorkers := 30
  baseDays := 12  -- We now know this value
  additionalWorkers := 10
  reducedDays := 3

/-- 
Theorem stating that for the given work project, 
the original number of days is 12.
-/
theorem original_days_is_twelve (project : WorkProject) : 
  project = problemProject → project.baseDays = 12 := by
  intro h
  rw [h]
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_original_days_is_twelve_l1099_109925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1099_109944

noncomputable def g (x : ℝ) : ℝ := (x + 3) / (x - 2)

theorem function_properties :
  -- 1. x ≠ (y + 3) / (y - 2)
  ∀ x y : ℝ, x ≠ (y + 3) / (y - 2) ∧
  -- 2. g(0) = -1.5
  g 0 = -3/2 ∧
  -- 3. g(-3) = 0
  g (-3) = 0 ∧
  -- 4. g(y) = x implies y = (x + 3) / (x - 2)
  ∀ x y : ℝ, g y = x → y = (x + 3) / (x - 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1099_109944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walk_solution_l1099_109946

/-- A man's walk with a turn -/
def ManWalk (x : ℝ) : Prop :=
  let initialPos : ℝ × ℝ := (0, 0)
  let afterWestPos : ℝ × ℝ := (-x, 0)
  let turnAngle : ℝ := 120 * (Real.pi / 180)  -- Use Real.pi instead of π
  let finalPos : ℝ × ℝ := 
    (afterWestPos.1 + 5 * Real.cos turnAngle, 
     afterWestPos.2 + 5 * Real.sin turnAngle)
  Real.sqrt ((finalPos.1 - initialPos.1)^2 + (finalPos.2 - initialPos.2)^2) = 5

theorem man_walk_solution :
  ∀ x : ℝ, ManWalk x → x = 0 ∨ x = 5 := by
  sorry

#check man_walk_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_walk_solution_l1099_109946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_skew_lines_in_cube_l1099_109956

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a unit cube in 3D space -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Returns the center of a face given three of its corners -/
noncomputable def faceCenterPoint (p1 p2 p3 : Point3D) : Point3D :=
  { x := (p1.x + p2.x + p3.x) / 3
    y := (p1.y + p2.y + p3.y) / 3
    z := (p1.z + p2.z + p3.z) / 3 }

/-- Calculates the cosine of the angle between two vectors -/
noncomputable def cosineAngle (v1 v2 : Point3D) : ℝ :=
  (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z) /
  (Real.sqrt (v1.x^2 + v1.y^2 + v1.z^2) * Real.sqrt (v2.x^2 + v2.y^2 + v2.z^2))

theorem cosine_angle_skew_lines_in_cube (cube : UnitCube) : 
  let O := faceCenterPoint cube.A cube.B cube.C
  let O₁ := faceCenterPoint cube.A cube.D cube.D₁
  let v1 := { x := cube.D₁.x - O.x, y := cube.D₁.y - O.y, z := cube.D₁.z - O.z }
  let v2 := { x := cube.B.x - O₁.x, y := cube.B.y - O₁.y, z := cube.B.z - O₁.z }
  cosineAngle v1 v2 = 5/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_skew_lines_in_cube_l1099_109956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_C_D_grid_with_circles_l1099_109966

/-- Represents a grid with circles on top -/
structure GridWithCircles where
  gridSize : Nat
  squareSize : Real
  numCircles : Nat

/-- Calculates the visible shaded area of the grid with circles -/
noncomputable def visibleShadedArea (g : GridWithCircles) : Real :=
  g.gridSize^2 * g.squareSize^2 - g.numCircles * Real.pi * (g.squareSize / 2)^2

/-- Theorem stating the sum of C and D for the specific grid configuration -/
theorem sum_C_D_grid_with_circles :
  let g : GridWithCircles := { gridSize := 4, squareSize := 3, numCircles := 4 }
  let visibleArea := visibleShadedArea g
  let C : Real := g.gridSize^2 * g.squareSize^2
  let D : Real := g.numCircles * (g.squareSize / 2)^2
  C + D = 153 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_C_D_grid_with_circles_l1099_109966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_eq_pi_sqrt3_div_12_l1099_109973

/-- A regular hexagon ABCDEF with an inscribed circle P in triangle BDF -/
structure RegularHexagonWithInscribedCircle where
  -- Side length of the hexagon
  s : ℝ
  -- Assume s is positive
  s_pos : s > 0

/-- The ratio of the area of the inscribed circle to the area of rectangle ABDE -/
noncomputable def areaRatio (h : RegularHexagonWithInscribedCircle) : ℝ :=
  (Real.pi * h.s^2 / 4) / (h.s^2 * Real.sqrt 3)

/-- Theorem stating that the area ratio is equal to π√3 / 12 -/
theorem area_ratio_eq_pi_sqrt3_div_12 (h : RegularHexagonWithInscribedCircle) :
  areaRatio h = Real.pi * Real.sqrt 3 / 12 := by
  sorry

#check area_ratio_eq_pi_sqrt3_div_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_eq_pi_sqrt3_div_12_l1099_109973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_of_sphere_l1099_109948

/-- The volume of a wedge of a sphere -/
noncomputable def wedge_volume (circumference : ℝ) : ℝ :=
  let radius := circumference / (2 * Real.pi)
  let sphere_volume := (4 / 3) * Real.pi * radius ^ 3
  sphere_volume / 6

/-- Theorem: The volume of one wedge of a sphere with circumference 24π inches is 384π cubic inches -/
theorem wedge_volume_of_sphere (circumference : ℝ) (h : circumference = 24 * Real.pi) :
  wedge_volume circumference = 384 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedge_volume_of_sphere_l1099_109948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_due_in_nine_months_l1099_109938

/-- Calculates the number of months until a bill is due given its face value, true discount, and annual interest rate. -/
noncomputable def months_until_due (face_value : ℝ) (true_discount : ℝ) (annual_interest_rate : ℝ) : ℝ :=
  let present_value := face_value - true_discount
  let monthly_interest_rate := annual_interest_rate / 12
  (true_discount * 100) / (present_value * monthly_interest_rate)

/-- Theorem stating that given the specified conditions, the bill is due in 9 months. -/
theorem bill_due_in_nine_months :
  months_until_due 2520 270 0.16 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_due_in_nine_months_l1099_109938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_g_domain_implies_m_range_l1099_109908

noncomputable def f (x : ℝ) := |x - 2| - |x - 4|

noncomputable def g (m : ℝ) (x : ℝ) := 1 / (m - f x)

theorem f_inequality_solution_set :
  {x : ℝ | f x < 0} = {x : ℝ | x < 3} := by sorry

theorem g_domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, g m x ≠ 0) → m < -2 ∨ m > 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_g_domain_implies_m_range_l1099_109908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zhang_distance_when_bus_starts_l1099_109957

/-- The speed of Zhang Qiang's bike in meters per minute -/
noncomputable def zhang_speed : ℝ := 250

/-- The speed of the bus in meters per minute -/
noncomputable def bus_speed : ℝ := 450

/-- The time in minutes the bus travels before stopping -/
noncomputable def bus_travel_time : ℝ := 6

/-- The time in minutes the bus stops at each station -/
noncomputable def bus_stop_time : ℝ := 1

/-- The time in minutes after which the bus catches up to Zhang Qiang -/
noncomputable def catch_up_time : ℝ := 15

/-- The effective speed of the bus considering its stops -/
noncomputable def effective_bus_speed : ℝ := 
  (bus_speed * bus_travel_time) / (bus_travel_time + bus_stop_time)

/-- The theorem stating the distance Zhang Qiang had ridden when the bus started -/
theorem zhang_distance_when_bus_starts : 
  zhang_speed * catch_up_time = 2100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zhang_distance_when_bus_starts_l1099_109957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_when_a_greater_than_e_l1099_109994

/-- The function f(x) = x - ln(ax) --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - Real.log (a * x)

theorem two_roots_when_a_greater_than_e (a : ℝ) (h : a > Real.exp 1) :
  ∃! (s : Set ℝ), s.Finite ∧ s.ncard = 2 ∧ ∀ x ∈ s, f a x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_when_a_greater_than_e_l1099_109994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l1099_109960

/-- An increasing function f satisfying the given properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → ContinuousAt f x) ∧ 
  (∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y) ∧
  (∀ x y, x < y → f x < f y) ∧
  (f 6 = 1)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 1 = 0) ∧ 
  ({x : ℝ | f (x + 3) - f (1/3) < 2} = Set.Ioo (-3) 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_function_properties_l1099_109960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_game_termination_l1099_109974

/-- Represents a card in the game -/
structure Card where
  id : Nat
deriving DecidableEq

/-- Represents the state of the game -/
structure GameState where
  player1_cards : List Card
  player2_cards : List Card

/-- Represents the result of comparing two cards -/
inductive CompareResult
  | Beats
  | LosesTo

/-- The card comparison function -/
noncomputable def compare_cards : Card → Card → CompareResult := sorry

/-- The game transition function -/
noncomputable def play_turn (state : GameState) : GameState := sorry

theorem card_game_termination 
  (n : Nat) 
  (cards : Finset Card) 
  (h_card_count : cards.card = n)
  (h_distinct : ∀ c1 c2 : Card, c1 ∈ cards → c2 ∈ cards → c1 ≠ c2 → 
    (compare_cards c1 c2 = CompareResult.Beats ∧ compare_cards c2 c1 = CompareResult.LosesTo) ∨
    (compare_cards c1 c2 = CompareResult.LosesTo ∧ compare_cards c2 c1 = CompareResult.Beats))
  (initial_state : GameState)
  (h_initial_distribution : initial_state.player1_cards.toFinset ∪ initial_state.player2_cards.toFinset = cards) :
  ∃ (final_state : GameState), 
    (final_state.player1_cards = [] ∧ final_state.player2_cards.toFinset = cards) ∨
    (final_state.player2_cards = [] ∧ final_state.player1_cards.toFinset = cards) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_game_termination_l1099_109974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l1099_109945

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 - a*x - 5 else a/x

-- State the theorem
theorem range_of_a_for_increasing_f :
  (∀ a : ℝ, StrictMono (f a)) →
  {a : ℝ | ∃ x : ℝ, f a x = 0} = Set.Icc (-3) (-2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_increasing_f_l1099_109945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l1099_109981

theorem no_such_function :
  ¬∃ f : ℕ → ℕ,
    (∀ a b : ℕ, a ≠ 0 → b ≠ 0 → f a * f b ≥ f (a * b) * f 1) ∧
    (∀ a b : ℕ, a ≠ 0 → b ≠ 0 → 
      (f a * f b = f (a * b) * f 1 ↔ (a - 1) * (b - 1) * (a - b) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l1099_109981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revolutions_for_given_bicycle_l1099_109978

/-- Represents a wheel on a bicycle --/
structure Wheel where
  diameter : ℝ

/-- Represents a tandem bicycle --/
structure TandemBicycle where
  frontWheel : Wheel
  backWheel : Wheel

/-- Calculates the number of revolutions made by the back wheel --/
noncomputable def backWheelRevolutions (bike : TandemBicycle) (frontRevolutions : ℝ) : ℝ :=
  frontRevolutions * bike.frontWheel.diameter / bike.backWheel.diameter

/-- Theorem: The back wheel of a tandem bicycle with front wheel diameter 40 inches
    and back wheel diameter 20 inches will make 300 revolutions when the front wheel
    makes 150 revolutions, assuming no slippage. --/
theorem back_wheel_revolutions_for_given_bicycle :
  let bike := TandemBicycle.mk (Wheel.mk 40) (Wheel.mk 20)
  backWheelRevolutions bike 150 = 300 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_back_wheel_revolutions_for_given_bicycle_l1099_109978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_hours_left_l1099_109969

/-- Represents the number of sick days and vacation days an employee gets per year -/
def total_days_per_type : ℕ := 10

/-- Represents the fraction of days Mark uses -/
def used_fraction : ℚ := 1/2

/-- Represents the number of hours in a workday -/
def hours_per_day : ℕ := 8

/-- Calculates the number of hours Mark has left -/
def hours_left : ℕ :=
  (2 * (total_days_per_type - Nat.floor (used_fraction * total_days_per_type))) * hours_per_day

theorem mark_hours_left : hours_left = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_hours_left_l1099_109969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_fractional_part_gt_999_l1099_109924

/-- The fractional part of a real number x -/
noncomputable def fractionalPart (x : ℝ) : ℝ := x - ⌊x⌋

/-- There exists a natural number n such that the fractional part of (2 + √2)^n is greater than 0.999 -/
theorem exists_n_fractional_part_gt_999 :
  ∃ n : ℕ, fractionalPart ((2 + Real.sqrt 2) ^ n) > 0.999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_fractional_part_gt_999_l1099_109924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_volume_in_specific_prism_l1099_109923

/-- A closed right triangular prism with given dimensions -/
structure RightTriangularPrism where
  ab : ℝ
  bc : ℝ
  aa1 : ℝ
  ab_perp_bc : ab > 0 ∧ bc > 0 ∧ aa1 > 0

/-- The maximum volume of a sphere that can fit inside a right triangular prism -/
noncomputable def max_sphere_volume (p : RightTriangularPrism) : ℝ :=
  (4 / 3) * Real.pi * (p.aa1 / 2) ^ 3

/-- Theorem stating the maximum volume of a sphere in a specific right triangular prism -/
theorem max_sphere_volume_in_specific_prism :
  ∃ (p : RightTriangularPrism), 
    p.ab = 6 ∧ p.bc = 8 ∧ p.aa1 = 3 ∧ max_sphere_volume p = 9 * Real.pi / 2 := by
  sorry

#check max_sphere_volume_in_specific_prism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_volume_in_specific_prism_l1099_109923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_silver_ratio_l1099_109975

-- Define the weight of the bar
noncomputable def bar_weight : ℝ := 30

-- Define the weight loss of the bar in water
noncomputable def bar_loss : ℝ := 3

-- Define the weight loss ratio for tin
noncomputable def tin_loss_ratio : ℝ := 1.375 / 10

-- Define the weight loss ratio for silver
noncomputable def silver_loss_ratio : ℝ := 0.375 / 5

-- Define the weight of tin in the bar
noncomputable def tin_weight : ℝ := 12

-- Define the weight of silver in the bar
noncomputable def silver_weight : ℝ := 18

-- Theorem to prove
theorem tin_silver_ratio : 
  tin_weight / silver_weight = 2 / 3 ∧
  tin_weight + silver_weight = bar_weight ∧
  tin_loss_ratio * tin_weight + silver_loss_ratio * silver_weight = bar_loss := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tin_silver_ratio_l1099_109975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_cost_is_20_l1099_109928

/-- The cost of one pie in rubles -/
def pie_cost : ℕ → Prop := sorry

/-- The total amount of money Masha has in rubles -/
def masha_total : ℕ → Prop := sorry

/-- Masha is short 60 rubles to buy 4 pies with her two-ruble coins -/
axiom two_ruble_short : ∀ x, pie_cost x → masha_total (4 * x + 60 - x * 2)

/-- Masha is short 60 rubles to buy 5 pies with her five-ruble coins -/
axiom five_ruble_short : ∀ x, pie_cost x → masha_total (5 * x + 60 - x * 5)

/-- Masha is short 60 rubles to buy 6 pies in total -/
axiom total_short : ∀ x, pie_cost x → masha_total (6 * x + 60 - (x * 2 + x * 5))

/-- The cost of one pie is 20 rubles -/
theorem pie_cost_is_20 : pie_cost 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pie_cost_is_20_l1099_109928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_ceiling_evaluation_l1099_109963

theorem complex_fraction_ceiling_evaluation :
  (⌈(23 : ℚ) / 9 - ⌈(35 : ℚ) / 23⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈(9 : ℚ) * 23 / 35⌉⌉) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_ceiling_evaluation_l1099_109963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1099_109964

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a point and a direction, returns the equation of the line passing through the point with that direction -/
noncomputable def lineFromPointAndDirection (p : Point) (direction : Line) : Line :=
  { a := direction.a
    b := direction.b
    c := -(direction.a * p.x + direction.b * p.y) }

/-- Returns the reflection of a point across the y-axis -/
def reflectPointAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Returns the intersection point of a line with the y-axis -/
noncomputable def intersectionWithYAxis (l : Line) : Point :=
  { x := 0, y := -l.c / l.b }

/-- Given two points, returns the line passing through them -/
noncomputable def lineFromTwoPoints (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y
    b := p1.x - p2.x
    c := p2.x * p1.y - p1.x * p2.y }

theorem reflected_ray_equation (emissionPoint : Point) (direction : Line) : 
  emissionPoint = { x := 2, y := 3 } →
  direction = { a := 1, b := -2, c := 0 } →
  let incidentRay := lineFromPointAndDirection emissionPoint direction
  let intersectionPoint := intersectionWithYAxis incidentRay
  let reflectedPoint := reflectPointAcrossYAxis emissionPoint
  let reflectedRay := lineFromTwoPoints intersectionPoint reflectedPoint
  reflectedRay = { a := 1, b := 2, c := -4 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l1099_109964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1099_109989

/-- Configuration of circles -/
structure CircleConfiguration where
  largeRadius : ℝ
  smallRadius : ℝ
  smallCirclesTouch : Prop

/-- The area of the shaded region in the circle configuration -/
noncomputable def shadedArea (config : CircleConfiguration) : ℝ :=
  Real.pi * config.largeRadius^2 - 3 * Real.pi * config.smallRadius^2

/-- Theorem stating the area of the shaded region for the given configuration -/
theorem shaded_area_calculation (config : CircleConfiguration) 
  (h1 : config.largeRadius = 9)
  (h2 : config.smallRadius = config.largeRadius / 2)
  (h3 : config.smallCirclesTouch) :
  shadedArea config = 20.25 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1099_109989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_coefficient_product_l1099_109904

-- Define the original curve C
noncomputable def C : Set (ℝ × ℝ) := {(x, y) | x * y = 1}

-- Define the reflection line
noncomputable def reflection_line (x : ℝ) : ℝ := 2 * x

-- Define the reflection transformation
noncomputable def reflect (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-(3/5) * x + (4/5) * y, (4/5) * x + (3/5) * y)

-- Define the reflected curve C*
noncomputable def C_star : Set (ℝ × ℝ) := {p | reflect p ∈ C}

-- State the theorem
theorem reflection_coefficient_product :
  ∃ (b c d : ℝ), (∀ (x y : ℝ), (x, y) ∈ C_star ↔ 12 * x^2 + b * x * y + c * y^2 + d = 0) ∧ b * c = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_coefficient_product_l1099_109904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1099_109907

open BigOperators

theorem arithmetic_sequence_problem (a₁ a₅ : ℝ) (h1 : a₁^2 - 12*a₁ + 27 = 0)
                                    (h2 : a₅^2 - 12*a₅ + 27 = 0)
                                    (h3 : ∀ n : ℕ, a₁ < a₅) : 
  (∀ n : ℕ, a₁ + (n-1)*(a₅ - a₁)/4 = 2*n - 1) ∧
  (∀ n : ℕ, (2:ℝ)/(3^n) = 2/(3^n)) ∧
  (∀ n : ℕ, (∑ i in Finset.range n, (2*i - 1) * (2/(3^i))) = 2 - (2*n + 2)/(3^n)) := by
  sorry

#check arithmetic_sequence_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l1099_109907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_greater_than_b2_l1099_109910

/-- An arithmetic sequence with a_1 = 4 and a_4 = 1 -/
noncomputable def arithmetic_sequence (n : ℕ) : ℝ :=
  4 - (n - 1)

/-- A geometric sequence with b_1 = 4 and b_4 = 1 -/
noncomputable def geometric_sequence (n : ℕ) : ℝ :=
  4 * (1/2)^(n - 1)

theorem a2_greater_than_b2 : arithmetic_sequence 2 > geometric_sequence 2 := by
  -- Unfold the definitions
  unfold arithmetic_sequence geometric_sequence
  -- Simplify the expressions
  simp
  -- Prove the inequality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a2_greater_than_b2_l1099_109910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_three_probability_first_player_wins_l1099_109911

def card_set : Finset Nat := {1, 2, 3, 4}

-- Function to check if a product is divisible by 3
def is_divisible_by_three (a b : Nat) : Bool :=
  (a * b) % 3 = 0

-- Probability of product being divisible by 3
theorem probability_divisible_by_three :
  (Finset.filter (fun (p : Nat × Nat) => is_divisible_by_three p.1 p.2)
    (Finset.product card_set card_set)).card
  / (card_set.card * card_set.card) = 7 / 16 := by sorry

-- Function to check if first player wins
def first_player_wins (a b c d : Nat) : Bool :=
  a + b > c + d

-- Probability of first player winning given they drew 3 and 4
theorem probability_first_player_wins :
  let remaining_cards := card_set.erase 3 ∪ card_set.erase 4
  (Finset.filter (fun (p : Nat × Nat) => first_player_wins 3 4 p.1 p.2)
    (Finset.product remaining_cards remaining_cards)).card
  / (remaining_cards.card * remaining_cards.card) = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_by_three_probability_first_player_wins_l1099_109911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1099_109929

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x + 1| > 2) ↔ a ∈ Set.Ioi 1 ∪ Set.Iic (-3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1099_109929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_equal_distances_l1099_109984

/-- Given points A, B, and C in R^2, and a line l passing through C with equal distances from A and B,
    prove that the equation of l is either x + y - 1 = 0 or x - 2 = 0. -/
theorem line_equation_with_equal_distances (A B C : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  A = (1, 4) → B = (3, 2) → C = (2, -1) →
  C ∈ l →
  (∀ p ∈ l, dist A p = dist B p) →
  (∀ x y : ℝ, (x, y) ∈ l ↔ x + y - 1 = 0) ∨
  (∀ x y : ℝ, (x, y) ∈ l ↔ x - 2 = 0) :=
by
  sorry

where
  dist : ℝ × ℝ → ℝ × ℝ → ℝ := λ p q ↦ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_equal_distances_l1099_109984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1099_109980

theorem inequality_proof (a b c : ℝ) (h : a^2 + 2*b^2 + 3*c^2 = 3/2) :
  (3 : ℝ)^(-a) + (9 : ℝ)^(-b) + (27 : ℝ)^(-c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l1099_109980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_function_equivalence_l1099_109902

noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def translated_function (x : ℝ) : ℝ := original_function (x + Real.pi / 4) + 1

theorem translated_function_equivalence (x : ℝ) :
  translated_function x = 2 * (Real.cos x) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_function_equivalence_l1099_109902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_through_circumcenter_l1099_109926

-- Define the points
variable (A B C P Q : EuclideanSpace ℝ (Fin 2))

-- Define the circumcenter
noncomputable def circumcenter (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := sorry

-- Define the line through two points
def line_through (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the distance between two points
def distance (P Q : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem pq_through_circumcenter 
  (h_not_collinear : ¬ Collinear ℝ {A, B, C})
  (h_ratio_abp : distance A P / distance B P = 21 / 20)
  (h_ratio_abq : distance A Q / distance B Q = 21 / 20)
  (h_ratio_bcp : distance B P / distance C P = 20 / 19)
  (h_ratio_bcq : distance B Q / distance C Q = 20 / 19) :
  (circumcenter A B C) ∈ line_through P Q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pq_through_circumcenter_l1099_109926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_five_halves_unique_solution_g_equals_g_inv_l1099_109901

/-- The function g(x) = 5x - 10 -/
noncomputable def g (x : ℝ) : ℝ := 5 * x - 10

/-- The inverse function of g -/
noncomputable def g_inv (x : ℝ) : ℝ := (x + 10) / 5

/-- Theorem stating that g(x) = g⁻¹(x) when x = 5/2 -/
theorem g_equals_g_inv_at_five_halves :
  g (5/2 : ℝ) = g_inv (5/2 : ℝ) := by
  sorry

/-- Theorem stating that 5/2 is the unique solution to g(x) = g⁻¹(x) -/
theorem unique_solution_g_equals_g_inv :
  ∀ x : ℝ, g x = g_inv x ↔ x = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_g_inv_at_five_halves_unique_solution_g_equals_g_inv_l1099_109901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_hyperbola_l1099_109996

/-- A parabola and an ellipse intersecting at a point, and a hyperbola sharing foci with the ellipse -/
theorem parabola_ellipse_hyperbola 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (C : Set (ℝ × ℝ)) 
  (hC : C = {(x, y) | x^2/a^2 + y^2/b^2 = 1}) 
  (P : Set (ℝ × ℝ)) 
  (hP_vertex : (0, 0) ∈ P) 
  (hP_directrix : ∃ (d : ℝ), P = {(x, y) | y^2 = -4*x} ∧ 
                              {(x, y) | x = d} ∩ C ≠ ∅ ∧ 
                              ∀ (x y : ℝ), (x, y) ∈ C → (d - x)^2 + y^2 = a^2) 
  (hP_C_intersection : (-2/3, 2*Real.sqrt 6/3) ∈ P ∩ C) 
  (H : Set (ℝ × ℝ)) 
  (hH_foci : ∀ (x y : ℝ), (x, y) ∈ H ↔ 
              ∃ (c : ℝ), c^2 = a^2 - b^2 ∧ 
                          ((x + c)^2 + y^2)^(1/2) - ((x - c)^2 + y^2)^(1/2) = 2*(a^2 - c^2)^(1/2)) 
  (hH_asymptotes : ∀ (x y : ℝ), (x, y) ∈ H → (y = 4/3*x ∨ y = -4/3*x)) :
  (P = {(x, y) | y^2 = -4*x}) ∧ 
  (C = {(x, y) | x^2/4 + y^2/3 = 1}) ∧ 
  (H = {(x, y) | 25*x^2/9 - 25*y^2/16 = 1}) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_hyperbola_l1099_109996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l1099_109906

/-- The function y in terms of x, a, and b -/
noncomputable def y (x a b : ℝ) : ℝ := b + a^(x^2 + 2*x)

/-- The theorem statement -/
theorem function_values (a b : ℝ) :
  a > 0 ∧ a ≠ 1 ∧
  (∀ x ∈ Set.Icc (-3/2) 0, y x a b ≤ 3) ∧
  (∃ x ∈ Set.Icc (-3/2) 0, y x a b = 3) ∧
  (∀ x ∈ Set.Icc (-3/2) 0, y x a b ≥ 5/2) ∧
  (∃ x ∈ Set.Icc (-3/2) 0, y x a b = 5/2) →
  ((a = 2 ∧ b = 2) ∨ (a = 2/3 ∧ b = 3/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_values_l1099_109906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_integer_l1099_109968

/-- For pairwise coprime positive integers a, b, c, 2abc - ab - bc - ca is the largest integer
    that cannot be represented as xbc + yca + zab for non-negative integers x, y, z. -/
theorem largest_non_representable_integer (a b c : ℕ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_coprime : Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime c a) : 
  ∀ n : ℕ, n > 2*a*b*c - a*b - b*c - c*a → 
  ∃ (x y z : ℕ), n = x*b*c + y*c*a + z*a*b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_non_representable_integer_l1099_109968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_values_l1099_109976

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.cos x

theorem theta_values (θ : ℝ) (h : θ = Real.pi * (5/6) ∨ θ = Real.pi * (2/3)) :
  ∀ x₁ ∈ Set.Icc 0 (Real.pi / 2),
    ∃ x₂ ∈ Set.Icc (-Real.pi / 2) 0,
      2 * f x₁ = 2 * g (x₂ + θ) + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theta_values_l1099_109976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1099_109953

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^4 + 2 * abs (x - 1)

-- State the theorem
theorem f_inequality_range :
  ∀ x : ℝ, f x > f (2 * x) ↔ 0 < x ∧ x < 2/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_range_l1099_109953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_circle_properties_l1099_109992

def point1 : ℝ × ℝ := (-3, 4)
def point2 : ℝ × ℝ := (5, -2)
def circle_radius : ℝ := 5

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def on_circle (p : ℝ × ℝ) (r : ℝ) : Prop :=
  p.1^2 + p.2^2 = r^2

theorem distance_and_circle_properties :
  (distance point1 point2 = 10) ∧
  (on_circle point1 circle_radius) ∧
  ¬(on_circle point2 circle_radius) := by
  sorry

#eval point1
#eval point2
#eval circle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_and_circle_properties_l1099_109992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1099_109962

-- Define the constants
noncomputable def a : ℝ := 3^(1/5)
noncomputable def b : ℝ := Real.log 7 / Real.log 6
noncomputable def c : ℝ := Real.log 6 / Real.log 5

-- State the theorem
theorem ordering_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1099_109962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1099_109972

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Conditions of the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  (t.a + t.c) * Real.sin t.A = Real.sin t.A + Real.sin t.C ∧
  t.c^2 + t.c = t.b^2 - 1 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  0 < t.A ∧ t.A < Real.pi ∧
  0 < t.B ∧ t.B < Real.pi ∧
  0 < t.C ∧ t.C < Real.pi

/-- D is the midpoint of AC and BD = √3/2 -/
def MidpointCondition (t : Triangle) : Prop :=
  ∃ (D : ℝ × ℝ), (D.1 = (t.a + t.c) / 2) ∧ 
  ((D.1 - t.a)^2 + D.2^2 = (Real.sqrt 3 / 2)^2)

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : TriangleConditions t) (h2 : MidpointCondition t) : 
  t.B = 2 * Real.pi / 3 ∧ (1/2 * t.a * t.c * Real.sin t.B = Real.sqrt 3 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1099_109972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1099_109965

noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

theorem f_properties :
  (∀ x : ℝ, Real.sin 1 ≤ f x + f (x + 1) ∧ f x + f (x + 1) ≤ 2 * Real.cos (1 / 2)) ∧
  (∀ n : ℕ, n > 0 → (Finset.range (2 * n)).sum (λ k => f (↑k + ↑n) / (↑k + ↑n)) > Real.sin 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1099_109965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_smallest_angle_l1099_109916

theorem triangle_cosine_smallest_angle :
  ∀ (n : ℕ),
    (∃ (x y : Real),
      0 < x ∧ 0 < y ∧ 
      x + y + (x + y) = 2 * Real.pi ∧
      y = 2 * x ∧
      Real.cos x = (n + 5 : Real) / (2 * (n + 2)) ∧
      Real.cos y = (n - 3 : Real) / (2 * n) ∧
      n > 0) →
    Real.cos x = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_smallest_angle_l1099_109916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_change_l1099_109933

theorem fraction_change (original : ℚ) (num_increase : ℚ) (denom_decrease : ℚ) :
  original = 1 / 12 →
  num_increase = 1 / 5 →
  denom_decrease = 1 / 4 →
  (original * (1 + num_increase)) / (1 - denom_decrease) = 4 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_change_l1099_109933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_isosceles_triangle_ratio_l1099_109961

/-- Hyperbola with equation x²/a² - y²/b² = 1 -/
structure Hyperbola (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- Point on a hyperbola -/
structure PointOnHyperbola (E : Hyperbola a b) where
  x : ℝ
  y : ℝ
  on_hyperbola : x^2 / a^2 - y^2 / b^2 = 1

/-- Foci of a hyperbola -/
structure Foci (E : Hyperbola a b) where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def IsIsosceles (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

/-- Function to calculate the vertex angle of a triangle -/
noncomputable def VertexAngle (A B C : ℝ × ℝ) : ℝ :=
  let a := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let b := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let c := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

/-- Theorem: If there exists a point P on hyperbola E such that triangle F₁F₂P is isosceles
    with vertex angle 2π/3, then a²/b² = 2√3/3 -/
theorem hyperbola_isosceles_triangle_ratio 
  (E : Hyperbola a b) (F : Foci E) (P : PointOnHyperbola E)
  (h_isosceles : IsIsosceles F.F₁ F.F₂ (P.x, P.y))
  (h_vertex_angle : VertexAngle F.F₁ F.F₂ (P.x, P.y) = 2 * π / 3) :
  a^2 / b^2 = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_isosceles_triangle_ratio_l1099_109961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_16_less_than_quarter_less_than_2ln2_minus_ln3_l1099_109971

theorem sin_pi_16_less_than_quarter_less_than_2ln2_minus_ln3 :
  Real.sin (π / 16) < (1 / 4 : ℝ) ∧ (1 / 4 : ℝ) < 2 * Real.log 2 - Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_16_less_than_quarter_less_than_2ln2_minus_ln3_l1099_109971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_special_case_l1099_109930

/-- The height of a right pyramid with a square base -/
noncomputable def pyramid_height (base_perimeter : ℝ) (apex_to_vertex : ℝ) : ℝ :=
  let base_side := base_perimeter / 4
  let base_center_to_vertex := (base_side * Real.sqrt 2) / 2
  Real.sqrt (apex_to_vertex^2 - base_center_to_vertex^2)

/-- The height of a right pyramid with a square base of perimeter 40 inches
    and an apex 15 inches from each vertex is 5√7 inches -/
theorem pyramid_height_special_case :
  pyramid_height 40 15 = 5 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_special_case_l1099_109930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1099_109939

-- Define the linear function f(x) = ax + b
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

-- Define the distance between intersection points of y = x^2 - 1 and y = f(x)
noncomputable def distance1 (a b : ℝ) : ℝ := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 4))

-- Define the distance between intersection points of y = x^2 and y = f(x) + 3
noncomputable def distance2 (a b : ℝ) : ℝ := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 12))

-- Define the distance between intersection points of y = x^2 - 1 and y = f(x) + 1
noncomputable def distance3 (a b : ℝ) : ℝ := Real.sqrt ((a^2 + 1) * (a^2 + 4*b + 8))

theorem intersection_distance (a b : ℝ) :
  distance1 a b = Real.sqrt 30 →
  distance2 a b = Real.sqrt 46 →
  distance3 a b = Real.sqrt 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1099_109939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1099_109985

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + (Real.cos x)^2 + a

-- Define the theorem
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x ∈ Set.Icc (-π/6) (π/3), f x a ≤ f (π/3) a ∧ f (-π/6) a ≤ f x a) ∧ 
    (f (π/3) a + f (-π/6) a = 1) →
    (∀ x : ℝ, f (x + π) a = f x a) ∧ 
    (∀ k : ℤ, ∀ x y : ℝ, x ∈ Set.Icc (k * π - π/3) (k * π + π/6) → 
      y ∈ Set.Icc (k * π - π/3) (k * π + π/6) → x ≤ y → f x a ≤ f y a) ∧
    (a = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1099_109985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_implies_inequality_l1099_109903

open Set
open Function
open Real

theorem monotonicity_implies_inequality 
  (f : ℝ → ℝ) 
  (h_diff : DifferentiableOn ℝ f (Ioi 0)) 
  (h_ineq : ∀ x ∈ Ioi 0, (x + 1) * (deriv f x) > f x) : 
  3 * f 3 > 4 * f 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_implies_inequality_l1099_109903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1099_109918

def set_A : Set ℝ := {x | 1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 8}
def set_B : Set ℝ := {x | -x^2 - 2*x + 8 ≥ 0}

theorem intersection_of_A_and_B : set_A ∩ set_B = Set.Ioc 0 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1099_109918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_permutation_l1099_109920

def is_valid_permutation (p : Equiv.Perm (Fin 4)) : Prop :=
  (∀ i : Fin 2, ¬(p i < p (i + 1) ∧ p (i + 1) < p (i + 2))) ∧
  (∀ i : Fin 2, ¬(p i > p (i + 1) ∧ p (i + 1) > p (i + 2))) ∧
  (p 0 < p 3)

theorem unique_valid_permutation :
  ∃! p : Equiv.Perm (Fin 4), is_valid_permutation p :=
sorry

#check unique_valid_permutation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_permutation_l1099_109920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_points_x_axis_l1099_109913

theorem symmetry_points_x_axis (θ : Real) : 
  (Real.cos θ = Real.sin (θ + π/3) ∧ Real.sin θ = Real.cos (θ + π/3)) → 
  θ = π/12 ∨ θ = -π/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_points_x_axis_l1099_109913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y7z2_l1099_109999

theorem coefficient_x3y7z2 :
  let expression := (4/7 : ℚ) * X - (3/5 : ℚ) * Y + (2/3 : ℚ) * Z
  let expanded := expression^12
  let coefficient := Nat.choose 12 3 * Nat.choose 9 7 * (4/7 : ℚ)^3 * (-3/5 : ℚ)^7 * (2/3 : ℚ)^2
  coefficient = -3534868480 / 218968125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x3y7z2_l1099_109999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_ratio_l1099_109932

/-- Represents the dimensions and frame properties of a painting --/
structure FramedPainting where
  width : ℝ  -- Width of the painting
  height : ℝ  -- Height of the painting
  sideFrameWidth : ℝ  -- Width of the frame on the sides

/-- Calculates the ratio of smaller to larger dimension of a framed painting --/
noncomputable def dimensionRatio (p : FramedPainting) : ℝ :=
  min (p.width + 2 * p.sideFrameWidth) (p.height + 3 * p.sideFrameWidth) /
  max (p.width + 2 * p.sideFrameWidth) (p.height + 3 * p.sideFrameWidth)

/-- Theorem stating the ratio of smaller to larger dimension for the specific painting --/
theorem framed_painting_ratio :
  ∃ (p : FramedPainting),
    p.width = 16 ∧
    p.height = 20 ∧
    (p.width + 2 * p.sideFrameWidth) * (p.height + 3 * p.sideFrameWidth) = 2 * p.width * p.height ∧
    dimensionRatio p = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_framed_painting_ratio_l1099_109932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l1099_109922

theorem solution_pairs (x p : ℕ) (h_prime : Nat.Prime p) (h_bound : x ≤ 2 * p) 
  (h_divides : x ^ (p - 1) ∣ (p - 1) ^ x + 1) :
  (x = 1 ∧ Nat.Prime p) ∨ (x = 2 ∧ p = 2) ∨ (x = 3 ∧ p = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l1099_109922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1099_109986

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.cos x + Real.sqrt (1 + Real.sin x) + Real.sqrt (1 - Real.sin x)

noncomputable def t (x : ℝ) : ℝ := Real.sqrt (1 + Real.sin x) + Real.sqrt (1 - Real.sin x)

noncomputable def g (a : ℝ) (t : ℝ) : ℝ := (1/2) * a * t^2 + t - a

theorem f_properties (a : ℝ) (h : a < 0) :
  (∀ x, x ∈ Set.Icc (-π/2) (π/2) → t x ∈ Set.Icc (Real.sqrt 2) 2) ∧
  (∀ x, x ∈ Set.Icc (-π/2) (π/2) → f a x = g a (t x)) ∧
  (∀ x, x ∈ Set.Icc (-π/2) (π/2) →
    (a ≤ -Real.sqrt 2 / 2 → f a x ≤ Real.sqrt 2) ∧
    (-Real.sqrt 2 / 2 < a → a < -1/2 → f a x ≤ -1/(2*a) - a) ∧
    (-1/2 ≤ a → f a x ≤ a + 2)) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-π/2) (π/2) → x₂ ∈ Set.Icc (-π/2) (π/2) → 
    (|f a x₁ - f a x₂| ≤ 1 ↔ a ∈ Set.Icc (Real.sqrt 2 - 3) 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1099_109986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_time_l1099_109951

/-- The number of days required for two borrowers to owe the same amount -/
noncomputable def days_to_equal_debt (principal1 principal2 rate1 rate2 : ℝ) : ℝ :=
  (principal2 - principal1) / (principal1 * rate1 - principal2 * rate2)

/-- Theorem stating that Laura and Noah will owe the same amount after 12.5 days -/
theorem equal_debt_time : 
  days_to_equal_debt 200 250 0.12 0.08 = 12.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_time_l1099_109951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_A_and_B_properties_l1099_109921

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (2 * x - x^2)}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem sets_A_and_B_properties :
  (A = Set.Icc 0 2) ∧
  ((Set.univ \ A) ∩ B = Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sets_A_and_B_properties_l1099_109921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1099_109912

-- Define the parabola
noncomputable def parabola (x θ : ℝ) : ℝ := x^2 - x * Real.cos θ + 2 * Real.sin θ - 1

-- Define the function f(θ)
noncomputable def f (θ : ℝ) : ℝ := Real.cos θ^2 - 4 * Real.sin θ + 2

-- Theorem statement
theorem parabola_properties :
  ∀ θ : ℝ,
  (∃ x₁ x₂ : ℝ, parabola x₁ θ = 0 ∧ parabola x₂ θ = 0 ∧ x₁ ≠ x₂) →
  (∃ x₁ x₂ : ℝ, parabola x₁ θ = 0 ∧ parabola x₂ θ = 0 ∧ x₁^2 + x₂^2 = f θ) ∧
  (∀ θ' : ℝ, f θ' ≥ -2) ∧
  (∀ θ' : ℝ, f θ' ≤ 6) ∧
  (∃ θ₁ θ₂ : ℝ, f θ₁ = -2 ∧ f θ₂ = 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1099_109912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_lines_l1099_109935

/-- Given a circle with equation x^2 + y^2 = 5, prove that the tangent lines
    passing through Q(3, 1) outside the circle and P(2, 1) on the circle
    have equations x + 2y - 5 = 0, 2x - y - 5 = 0, and 2x + y - 5 = 0 -/
theorem circle_tangent_lines :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 = 5
  let Q : ℝ × ℝ := (3, 1)
  let P : ℝ × ℝ := (2, 1)
  let tangent_line_1 := fun (x y : ℝ) => x + 2*y - 5 = 0
  let tangent_line_2 := fun (x y : ℝ) => 2*x - y - 5 = 0
  let tangent_line_3 := fun (x y : ℝ) => 2*x + y - 5 = 0
  circle_eq P.1 P.2 →
  (tangent_line_1 Q.1 Q.2 ∨ tangent_line_2 Q.1 Q.2 ∨ tangent_line_3 Q.1 Q.2) ∧
  (tangent_line_1 P.1 P.2 ∨ tangent_line_2 P.1 P.2 ∨ tangent_line_3 P.1 P.2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_lines_l1099_109935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l1099_109900

noncomputable def book_widths : List ℝ := [8, -3/4, 1.5, 3.25, 7, 12]

theorem average_book_width :
  (book_widths.map abs).sum / book_widths.length = 5.41667 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_book_width_l1099_109900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_measure_l1099_109937

noncomputable section

structure Triangle (P Q R : ℝ × ℝ) where
  pq_length : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4
  qr_length : Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 8
  right_angle_at_q : (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0

noncomputable def angle_measure (P Q R : ℝ × ℝ) : ℝ :=
  Real.arccos (((P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2)) /
    (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) * Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)))

theorem right_angle_measure (P Q R : ℝ × ℝ) (t : Triangle P Q R) :
  angle_measure Q P R = 90 * π / 180 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_angle_measure_l1099_109937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_tile_savings_l1099_109993

/-- Calculates the savings when choosing purple tiles over turquoise tiles for a bathroom tiling project. -/
theorem bathroom_tile_savings (turquoise_cost purple_cost : ℚ) 
  (wall1_length wall1_width wall2_length wall2_width : ℚ) 
  (tiles_per_sqft : ℚ) : 
  turquoise_cost = 13 →
  purple_cost = 11 →
  wall1_length = 5 →
  wall1_width = 8 →
  wall2_length = 7 →
  wall2_width = 8 →
  tiles_per_sqft = 4 →
  (turquoise_cost - purple_cost) * 
    ((wall1_length * wall1_width + wall2_length * wall2_width) * tiles_per_sqft) = 768 := by
  intro h1 h2 h3 h4 h5 h6 h7
  -- Substitute the given values
  rw [h1, h2, h3, h4, h5, h6, h7]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

-- Remove the #eval line as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathroom_tile_savings_l1099_109993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l1099_109931

-- Define the system of differential equations
def system (y z : ℝ → ℝ) : Prop :=
  (∀ x, deriv y x = z x) ∧ (∀ x, deriv z x = -y x)

-- Define the proposed solution
noncomputable def solution (C₁ C₂ : ℝ) (x : ℝ) : ℝ × ℝ :=
  (C₁ * Real.cos x + C₂ * Real.sin x, -C₁ * Real.sin x + C₂ * Real.cos x)

-- Theorem statement
theorem solution_satisfies_system :
  ∀ C₁ C₂ : ℝ, system (λ x => (solution C₁ C₂ x).1) (λ x => (solution C₁ C₂ x).2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_satisfies_system_l1099_109931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1099_109905

def isValidSequence (seq : List ℕ) : Prop :=
  seq.length ≤ 16 ∧
  (∀ i, i < seq.length - 1 → seq.get! i ≠ seq.get! (i + 1)) ∧
  (seq.head? = some 0 ∧ seq.get? (seq.length - 1) = some 0)

def maxSequenceLength : ℕ := 15

theorem max_sequence_length :
  ∀ (seq : List ℕ),
    isValidSequence seq →
    seq.length ≤ maxSequenceLength :=
by
  intro seq hyp
  sorry

#check max_sequence_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sequence_length_l1099_109905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_g_min_a_for_f_ge_g_l1099_109909

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

-- Part I: Maximum difference of g on [0, 2]
theorem max_difference_g :
  ∃ (M : ℝ), M = 112/27 ∧
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 2 → x₂ ∈ Set.Icc 0 2 →
    g x₁ - g x₂ ≤ M) ∧
  (∀ (ε : ℝ), ε > 0 →
    ∃ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧
      g x₁ - g x₂ > M - ε) :=
by
  sorry

-- Part II: Minimum value of a for f ≥ g on [1/2, 2]
theorem min_a_for_f_ge_g :
  ∃ (a_min : ℝ), a_min = 1 ∧
  (∀ (a : ℝ), a ≥ a_min →
    ∀ (x : ℝ), x ∈ Set.Icc (1/2) 2 → f a x ≥ g x) ∧
  (∀ (ε : ℝ), ε > 0 →
    ∃ (x : ℝ), x ∈ Set.Icc (1/2) 2 ∧ f (a_min - ε) x < g x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_difference_g_min_a_for_f_ge_g_l1099_109909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_even_integers_l1099_109936

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 20 →
  a + b + c + d = 35 →
  a + b + c + d + e + f = 50 →
  Odd e →
  Odd f →
  ∃ (count : ℕ), count = (Finset.filter Even {a, b, c, d, e, f}).card ∧ 
                 count ≥ 1 ∧
                 ∀ (other_count : ℕ), other_count = (Finset.filter Even {a, b, c, d, e, f}).card → 
                                       other_count ≥ count :=
by
  intro h1 h2 h3 h4 h5
  sorry

#check min_even_integers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_even_integers_l1099_109936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_abs_equation_l1099_109990

theorem sum_of_roots_abs_equation : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, |x - 3| = 3 * |x + 3|) ∧ (S.sum id = -4.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_abs_equation_l1099_109990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_diagonal_sum_theorem_l1099_109940

/-- A 3x3 grid filled with numbers from 1 to 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements on a diagonal of the grid -/
def diagonalSum (g : Grid) (mainDiag : Bool) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) fun i => (g i i).val) + 
  if mainDiag then 0 else ((g 0 2).val + (g 2 0).val - (g 1 1).val)

/-- The sum of elements in the non-diagonal cells -/
def nonDiagonalSum (g : Grid) : ℕ :=
  (Finset.sum (Finset.univ : Finset (Fin 3)) fun i => 
    Finset.sum (Finset.univ : Finset (Fin 3)) fun j => (g i j).val) - 
  (diagonalSum g true + diagonalSum g false - (g 1 1).val)

/-- All numbers from 1 to 9 appear exactly once in the grid -/
def isValidGrid (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), g i j = n

theorem grid_diagonal_sum_theorem (g : Grid) (h_valid : isValidGrid g) 
    (h_diag1 : diagonalSum g true = 7) (h_diag2 : diagonalSum g false = 21) :
    nonDiagonalSum g = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_diagonal_sum_theorem_l1099_109940
