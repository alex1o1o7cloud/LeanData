import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l300_30095

/-- An arithmetic sequence with a maximum sum -/
structure ArithSeqMaxSum where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h1 : ∀ n, a (n + 1) = a n + d  -- Arithmetic sequence property
  h2 : d < 0  -- Negative common difference (for maximum sum)
  h3 : a 11 / a 10 + 1 < 0  -- Given condition

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithSeqMaxSum) (n : ℕ) : ℝ :=
  (n : ℝ) * (seq.a 1 + seq.a n) / 2

/-- The theorem to be proved -/
theorem max_positive_sum (seq : ArithSeqMaxSum) :
  (∀ n ≤ 19, S seq n > 0) ∧ S seq 20 ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l300_30095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_problem_l300_30057

structure Triangle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  A : α
  B : α
  C : α

/-- The length of a line segment between two points -/
def distance {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (p q : α) : ℝ :=
  ‖p - q‖

/-- The angle bisector of ∠BAC intersects BC at point L -/
noncomputable def angle_bisector_A {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- The angle bisector of ∠ABC intersects AC at point K -/
noncomputable def angle_bisector_B {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- The foot of the perpendicular from C to BK -/
noncomputable def foot_M {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (t : Triangle α) : α :=
  sorry

/-- The foot of the perpendicular from C to AL -/
noncomputable def foot_N {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] (t : Triangle α) : α :=
  sorry

theorem triangle_bisector_problem {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α] 
  (t : Triangle α) :
  distance t.A t.B = 130 →
  distance t.A t.C = 123 →
  distance t.B t.C = 126 →
  distance (foot_M t) (foot_N t) = 59.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_bisector_problem_l300_30057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_interior_angle_regular_nonagon_interior_angle_proof_l300_30055

/-- The measure of one interior angle of a regular nonagon is 140 degrees. -/
theorem regular_nonagon_interior_angle : ℚ :=
140

/-- A regular nonagon has 9 sides. -/
def regular_nonagon_sides : ℕ := 9

/-- The sum of interior angles of a polygon with n sides is 180(n-2) degrees. -/
def sum_of_interior_angles (n : ℕ) : ℚ := 180 * (n - 2)

/-- The measure of one interior angle of a regular polygon is the sum of interior angles divided by the number of sides. -/
def interior_angle_measure (n : ℕ) : ℚ :=
  (sum_of_interior_angles n) / n

/-- Proof that the measure of one interior angle of a regular nonagon is 140 degrees. -/
theorem regular_nonagon_interior_angle_proof :
  interior_angle_measure regular_nonagon_sides = regular_nonagon_interior_angle :=
by
  -- Expand the definitions
  unfold interior_angle_measure
  unfold sum_of_interior_angles
  unfold regular_nonagon_sides
  -- Simplify the expression
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_interior_angle_regular_nonagon_interior_angle_proof_l300_30055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l300_30041

/-- Calculates the time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_speed_kmh : ℝ) (train_length : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train with given specifications takes 26 seconds to cross the platform -/
theorem train_crossing_time :
  let train_speed_kmh : ℝ := 72
  let train_length : ℝ := 270
  let platform_length : ℝ := 250
  time_to_cross_platform train_speed_kmh train_length platform_length = 26 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_cross_platform 72 270 250

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l300_30041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_second_quadrant_l300_30082

theorem sin_2α_second_quadrant (α : Real) :
  α ∈ Set.Ioo (π / 2) π →
  Real.tan α = -5 / 12 →
  Real.sin (2 * α) = -120 / 169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2α_second_quadrant_l300_30082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_2_l300_30001

noncomputable def sequenceA (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => 2 * sequenceA a n * (1 - sequenceA a n)

def is_periodic_with_period_2 (f : ℕ → ℝ) : Prop :=
  ∀ n, f (n + 2) = f n

theorem sequence_not_periodic_2 (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  ¬ is_periodic_with_period_2 (sequenceA a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_periodic_2_l300_30001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l300_30010

/-- A triangular pyramid with two equilateral faces and two isosceles right triangle faces -/
structure SpecialPyramid where
  /-- Side length of the equilateral triangular faces -/
  a : ℝ
  /-- Assumption that a is positive -/
  a_pos : 0 < a

/-- The radius of the inscribed sphere in the special pyramid -/
noncomputable def inscribed_sphere_radius (p : SpecialPyramid) : ℝ :=
  (p.a * Real.sqrt 2) / (2 * (2 + Real.sqrt 3))

/-- Theorem stating that the radius of the inscribed sphere in the special pyramid
    is equal to a√2 / (2(2 + √3)) -/
theorem inscribed_sphere_radius_formula (p : SpecialPyramid) :
  inscribed_sphere_radius p = (p.a * Real.sqrt 2) / (2 * (2 + Real.sqrt 3)) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_radius_formula_l300_30010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l300_30025

/-- Represents the speed of the train in kilometers per hour -/
noncomputable def train_speed_kmh : ℝ := 54

/-- Represents the length of the train in meters -/
noncomputable def train_length_m : ℝ := 100

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

/-- Calculates the time in seconds for the train to cross an electric pole -/
noncomputable def time_to_cross (speed_ms : ℝ) (length_m : ℝ) : ℝ := length_m / speed_ms

/-- Theorem stating that a train 100 m long traveling at 54 km/hr will take approximately 6.67 seconds to cross an electric pole -/
theorem train_crossing_time :
  ∃ ε > 0, abs (time_to_cross (kmh_to_ms train_speed_kmh) train_length_m - 6.67) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l300_30025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_equals_inverse_of_y1_l300_30090

def y : ℕ → ℕ
  | 0 => 135  -- Add case for 0
  | 1 => 135
  | (n + 2) => 2 * (y (n + 1))^2 + y (n + 1)

theorem sum_of_series_equals_inverse_of_y1 :
  ∑' n, 1 / (y n + 1 : ℝ) = 1 / (y 1 : ℝ) := by
  sorry

#eval y 1  -- Add this line to check if y 1 evaluates correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_series_equals_inverse_of_y1_l300_30090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_two_zero_otimes_cyclic_otimes_not_distributive_l300_30031

-- Define the operation ⊗
def otimes : ℝ → ℝ → ℝ := sorry

-- Axioms for ⊗
axiom otimes_zero (a : ℝ) : otimes 0 a = a
axiom otimes_comm (a b : ℝ) : otimes a b = otimes b a
axiom otimes_assoc (a b c : ℝ) : otimes (otimes a b) c = otimes c (a * b) + otimes a c + otimes b c - 2 * c

-- Theorems to prove
theorem otimes_two_zero : otimes (otimes 2 0) (otimes 2 0) = 8 := by sorry

theorem otimes_cyclic (a b c : ℝ) : otimes a (otimes b c) = otimes b (otimes c a) := by sorry

theorem otimes_not_distributive : ∃ a b c : ℝ, otimes (a + b) c ≠ otimes a c + otimes b c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_two_zero_otimes_cyclic_otimes_not_distributive_l300_30031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_radius_l300_30062

-- Define the triangle OAB
def O : ℝ × ℝ := (0, 0)
noncomputable def A : ℝ × ℝ := (4, 4 * Real.sqrt 3)
def B : ℝ × ℝ := (8, 0)

-- Define the incircle center I (we don't know its exact coordinates, so we leave it abstract)
noncomputable def I : ℝ × ℝ := sorry

-- Define circle C (we don't know its center coordinates, so we leave it abstract)
def C : (ℝ × ℝ) → Prop := sorry

-- Define points P and Q (we don't know their exact coordinates, so we leave them abstract)
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry

-- Theorem statement
theorem circle_C_radius :
  -- C passes through A and B
  C A ∧ C B →
  -- P and Q are on both C and the incircle
  C P ∧ C Q →
  -- The tangents at P and Q are perpendicular
  -- (We can't express this condition precisely without more definitions, so we assume it)
  True →
  -- The radius of C is 2√7
  ∃ (center : ℝ × ℝ), C center ∧ 
    Real.sqrt ((center.1 - A.1)^2 + (center.2 - A.2)^2) = 2 * Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_radius_l300_30062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l300_30058

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + Real.sqrt (1 - x)

-- State the theorem
theorem f_range : Set.range f = Set.Iic (5/4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l300_30058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evaluation_l300_30006

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x + 4 else 6 - 3 * x

theorem f_evaluation :
  f (-2) = 0 ∧ f 4 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_evaluation_l300_30006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l300_30048

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →
  (1 / 2 * a 3 = 1 / 2 * (3 * a 1 + 2 * a 2)) →
  (a 8 + a 9) / (a 6 + a 7) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l300_30048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l300_30007

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 - 2*x else -x^2 - 2*x

theorem odd_function_extension (x : ℝ) : 
  (∀ y, f (-y) = -f y) → 
  (∀ z ≤ 0, f z = z^2 - 2*z) → 
  x > 0 → 
  f x = -x^2 - 2*x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_extension_l300_30007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l300_30014

/-- The point where the mouse starts getting farther from the cheese -/
noncomputable def turning_point : ℝ × ℝ := (-3/5, 153/20)

/-- The sum of coordinates of the turning point -/
noncomputable def coordinate_sum : ℝ := 7.05

theorem mouse_cheese_problem :
  let cheese : ℝ × ℝ := (15, 12)
  let mouse_path (x : ℝ) : ℝ := -4 * x + 9
  turning_point.1 + turning_point.2 = coordinate_sum ∧
  ∀ t : ℝ, 
    let mouse : ℝ × ℝ := (t, mouse_path t)
    let dist_to_cheese := Real.sqrt ((mouse.1 - cheese.1)^2 + (mouse.2 - cheese.2)^2)
    let dist_to_turning_point := Real.sqrt ((mouse.1 - turning_point.1)^2 + (mouse.2 - turning_point.2)^2)
    t < turning_point.1 → dist_to_cheese > dist_to_turning_point ∧
    t > turning_point.1 → dist_to_cheese < dist_to_turning_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l300_30014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l300_30034

/-- An increasing function f on ℝ with f(-1) = -4 and f(2) = 2 -/
noncomputable def f : ℝ → ℝ := sorry

/-- The set P parameterized by t -/
def P (t : ℝ) : Set ℝ := {x | f (x + t) < 2}

/-- The set Q -/
def Q : Set ℝ := {x | f x < -4}

variable (t : ℝ)

/-- f is an increasing function -/
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y

/-- f(-1) = -4 -/
axiom f_neg_one : f (-1) = -4

/-- f(2) = 2 -/
axiom f_two : f 2 = 2

/-- P is a proper subset of Q -/
axiom P_subset_Q : P t ⊂ Q

/-- The range of t for which P is a proper subset of Q -/
theorem range_of_t : P t ⊂ Q ↔ t > 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l300_30034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_proof_l300_30040

open Set

theorem set_equality_proof {R : Type} [LinearOrderedField R] : 
  let M := {x : R | -3 < x ∧ x < 1}
  let N := {x : R | x ≤ 0}
  {x : R | x ≥ 1} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_proof_l300_30040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l300_30087

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

/-- The problem statement -/
theorem dilation_problem : 
  dilation (2 - 3*I) 3 (-1 + I) = -7 + 9*I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l300_30087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_inscribed_sphere_radius_l300_30075

/-- A triangular pyramid with special properties -/
structure SpecialPyramid where
  a : ℝ
  x : ℝ
  y : ℝ
  a_positive : 0 < a
  x_positive : 0 < x
  y_positive : 0 < y
  sum_condition : a = x + y
  perpendicular : True  -- This represents the pairwise perpendicularity condition

/-- The radius of the inscribed sphere for a special pyramid -/
noncomputable def inscribed_sphere_radius (p : SpecialPyramid) : ℝ := p.a / 2

/-- The theorem stating that the radius of the inscribed sphere is a/2 -/
theorem special_pyramid_inscribed_sphere_radius (p : SpecialPyramid) :
  inscribed_sphere_radius p = p.a / 2 := by
  -- Unfold the definition of inscribed_sphere_radius
  unfold inscribed_sphere_radius
  -- The result follows directly from the definition
  rfl

#check special_pyramid_inscribed_sphere_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_pyramid_inscribed_sphere_radius_l300_30075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_implies_k_value_l300_30028

theorem function_difference_implies_k_value :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 2 * x + 4
  let g : ℝ → ℝ := λ x => x^2 - (-18) * x - 6
  f 10 - g 10 = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_difference_implies_k_value_l300_30028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_multiples_of_five_l300_30024

def digits : List Nat := [1, 1, 1, 2, 0]

def is_valid_arrangement (arr : List Nat) : Bool :=
  arr.length = 4 && arr.head? != some 0 && arr.getLast? == some 0

def count_valid_arrangements : Nat :=
  (digits.permutations.filter is_valid_arrangement).length

theorem four_digit_multiples_of_five : count_valid_arrangements = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_multiples_of_five_l300_30024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l300_30016

/-- Given vectors a, b, and c in ℝ², prove that if lambda * a + b is collinear with c, then lambda = -1. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (lambda : ℝ) 
  (h1 : a = (1, 2)) 
  (h2 : b = (2, 0)) 
  (h3 : c = (1, -2)) 
  (h4 : ∃ (k : ℝ), k ≠ 0 ∧ lambda • a + b = k • c) : 
  lambda = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_collinearity_l300_30016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_proof_l300_30021

def complex_number_location (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_number_proof :
  let i : ℂ := Complex.I
  let z : ℂ := (-1 - 2*i) * i
  z = 2 - i ∧ complex_number_location z := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_proof_l300_30021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_theorem_not_parallel_to_planes_l300_30078

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (not_parallel : Line → Line → Prop)
variable (not_subset : Line → Plane → Prop)
variable (not_parallel_plane : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_planes_theorem 
  (α β : Plane) (m : Line) 
  (h1 : perpendicular m α) 
  (h2 : subset m β) : 
  perpendicular_planes α β := by sorry

-- Theorem 2
theorem not_parallel_to_planes 
  (α β : Plane) (m n : Line)
  (h1 : intersect α β m)
  (h2 : not_parallel n m)
  (h3 : not_subset n α)
  (h4 : not_subset n β) :
  not_parallel_plane n α ∧ not_parallel_plane n β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_theorem_not_parallel_to_planes_l300_30078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_perpendicular_line_l300_30094

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x - 2*y + 10 = 0

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 100

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the intersection point B
def B : ℝ × ℝ := (6, 8)

-- Define the perpendicular line
def perp_line (x : ℝ) : ℝ := -2*x + 20

-- State the theorem
theorem y_intercept_of_perpendicular_line : 
  ∃ (x y : ℝ), 
    l₁ x y ∧ 
    circle_eq x y ∧ 
    first_quadrant x y ∧
    x = B.1 ∧ 
    y = B.2 ∧
    perp_line 0 = 20 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_intercept_of_perpendicular_line_l300_30094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l300_30061

theorem cos_double_angle (x : ℝ) (h : Real.cos x = 3/4) : Real.cos (2*x) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l300_30061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_equals_b_l300_30029

theorem max_x_equals_b (a b c x : ℕ) : 
  2^a + 3^b = 2^c + 3^x → 
  x ≤ b ∧ ∃ (y : ℕ), y = b ∧ 2^a + 3^y = 2^c + 3^y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_equals_b_l300_30029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l300_30074

theorem inequality_proof (a b : ℝ) (ha : a ≠ 1) (hb : b ≠ 1) 
  (hab : a > b) (hb_pos : b > 0) (hab_gt1 : a * b > 1) : 
  (3 : ℝ)^a + (3 : ℝ)^b > 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l300_30074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_on_circle_l300_30036

/-- Two parabolas in the xy-plane -/
structure Parabolas where
  p1 : ℝ → ℝ
  p2 : ℝ → ℝ
  h1 : ∀ x, p1 x = (x - 2)^2
  h2 : ∀ y, p2 y + 6 = (y - 5)^2

/-- The intersection points of the parabolas -/
def intersection_points (p : Parabolas) : Set (ℝ × ℝ) :=
  {(x, y) | p.p1 x = y ∧ p.p2 y = x}

/-- The circle on which the intersection points lie -/
def intersection_circle (center : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | (x - center.1)^2 + (y - center.2)^2 = r^2}

/-- The main theorem -/
theorem parabolas_intersection_on_circle (p : Parabolas) :
  ∃ (center : ℝ × ℝ) (r : ℝ), 
    (intersection_points p).ncard = 4 ∧
    (intersection_points p) ⊆ intersection_circle center r ∧
    r^2 = 123/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabolas_intersection_on_circle_l300_30036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_price_l300_30063

/-- The price per kg of a rice mixture -/
noncomputable def mixture_price (price1 price2 : ℝ) (amount1 amount2 : ℝ) : ℝ :=
  (price1 * amount1 + price2 * amount2) / (amount1 + amount2)

/-- Theorem: The price per kg of the rice mixture is 8.20 Rs/kg -/
theorem rice_mixture_price :
  let price1 := (6.60 : ℝ)
  let price2 := (9.60 : ℝ)
  let amount1 := (49 : ℝ)
  let amount2 := (56 : ℝ)
  mixture_price price1 price2 amount1 amount2 = 8.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rice_mixture_price_l300_30063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_forty_one_is_prime_factor_largest_prime_factor_is_41_l300_30015

theorem largest_prime_factor_of_expression (p : Nat) :
  (Nat.Prime p ∧ p ∣ (20^3 + 15^4 - 10^5)) →
  p ≤ 41 :=
by
  sorry

theorem forty_one_is_prime_factor :
  Nat.Prime 41 ∧ 41 ∣ (20^3 + 15^4 - 10^5) :=
by
  sorry

theorem largest_prime_factor_is_41 :
  (∃ (p : Nat), Nat.Prime p ∧ p ∣ (20^3 + 15^4 - 10^5) ∧ p = 41) ∧
  (∀ (q : Nat), Nat.Prime q ∧ q ∣ (20^3 + 15^4 - 10^5) → q ≤ 41) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_forty_one_is_prime_factor_largest_prime_factor_is_41_l300_30015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_perimeter_identity_radii_sum_identity_l300_30030

/-- Right triangle with incircle and excircles -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_right : a^2 + b^2 = c^2
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- Semi-perimeter of the triangle -/
noncomputable def semi_perimeter (t : RightTriangle) : ℝ := (t.a + t.b + t.c) / 2

/-- Inradius of the triangle -/
noncomputable def inradius (t : RightTriangle) : ℝ := sorry

/-- Exradius opposite to side a -/
noncomputable def exradius_a (t : RightTriangle) : ℝ := sorry

/-- Exradius opposite to side b -/
noncomputable def exradius_b (t : RightTriangle) : ℝ := sorry

/-- Exradius opposite to side c -/
noncomputable def exradius_c (t : RightTriangle) : ℝ := sorry

/-- Theorem 1: p(p - c) = (p - a)(p - b) -/
theorem semi_perimeter_identity (t : RightTriangle) :
  let p := semi_perimeter t
  p * (p - t.c) = (p - t.a) * (p - t.b) := by sorry

/-- Theorem 2: r + r_A + r_B + r_C = 2p -/
theorem radii_sum_identity (t : RightTriangle) :
  let p := semi_perimeter t
  inradius t + exradius_a t + exradius_b t + exradius_c t = 2 * p := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semi_perimeter_identity_radii_sum_identity_l300_30030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l300_30097

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 20 cm and 18 cm, 
    and a distance of 12 cm between them, is equal to 228 cm² -/
theorem trapezium_area_example : trapezium_area 20 18 12 = 228 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Simplify the arithmetic
  simp [mul_add, mul_div_assoc]
  -- Check that the result is equal to 228
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l300_30097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_minimizing_S_l300_30033

-- Define the type for ball arrangements
def BallArrangement := List Nat

-- Function to calculate S for a given arrangement
def calculateS (arrangement : BallArrangement) : Nat :=
  sorry

-- Function to check if an arrangement minimizes S
def minimizesS (arrangement : BallArrangement) : Prop :=
  sorry

-- Total number of unique arrangements considering rotations and reflections
def totalArrangements : Nat := 20160 -- 8! / 2

-- Number of arrangements that minimize S
def minimizingArrangements : Nat := 64

-- Theorem stating the probability of minimizing S
theorem probability_of_minimizing_S :
  (minimizingArrangements : Rat) / totalArrangements = 1 / 315 := by
  sorry

#eval (minimizingArrangements : Rat) / totalArrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_minimizing_S_l300_30033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_die_probabilities_l300_30012

noncomputable def P (p : ℕ) : ℝ := Real.log (p + 2) - Real.log p

theorem prime_die_probabilities :
  P 5 + P 11 + P 13 = 2 * P 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_die_probabilities_l300_30012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_locus_of_perpendiculars_l300_30050

/-- Given a parabola with equation y^2 = 2px, the locus of the feet of perpendiculars
    drawn from its focus to its normals is described by the equation y^2 = (p/2)x. -/
theorem parabola_locus_of_perpendiculars (p : ℝ) (x y : ℝ → ℝ) :
  (∀ t, (y t)^2 = 2 * p * (x t)) →
  ∃ f : ℝ → ℝ, (∀ t, (f t)^2 = (p / 2) * (x t - p / 2)) ∧
       (∀ t, ∃ s, x s = x t - p / 2 ∧ y s = f t) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_locus_of_perpendiculars_l300_30050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l300_30083

noncomputable def f (x : ℝ) : ℝ := -2 / x

theorem inverse_proportion_properties :
  (f (-1) = 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f x₁ > f x₂) ∧
  (∀ x : ℝ, x ≠ 0 → (x > 0 ∧ f x < 0) ∨ (x < 0 ∧ f x > 0)) ∧
  (∀ x : ℝ, x > 1 → -2 < f x ∧ f x < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_properties_l300_30083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_monotonicity_l300_30071

/-- A power function with exponent (1-3m)/5 -/
noncomputable def f (m : ℤ) (x : ℝ) : ℝ := x^((1 - 3*m)/5 : ℝ)

/-- The maximum integer value of m for which f is monotonically decreasing on (-∞, 0) 
    and monotonically increasing on (0, +∞) -/
theorem max_m_for_monotonicity : 
  (∀ m : ℤ, (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 0 → x₂ < 0 → f m x₁ > f m x₂) ∧ 
             (∀ x₁ x₂ : ℝ, x₁ < x₂ → 0 < x₁ → 0 < x₂ → f m x₁ < f m x₂)) →
  (∃ max_m : ℤ, max_m = -1 ∧ 
    ∀ m : ℤ, (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ < 0 → x₂ < 0 → f m x₁ > f m x₂) ∧ 
             (∀ x₁ x₂ : ℝ, x₁ < x₂ → 0 < x₁ → 0 < x₂ → f m x₁ < f m x₂) →
    m ≤ max_m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_monotonicity_l300_30071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l300_30068

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / (x - 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x | x > -1 ∧ x ≠ 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l300_30068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fraction_proof_l300_30085

def is_simplified_proper_fraction (n : ℕ) (d : ℕ) : Prop :=
  0 < n ∧ n < d ∧ Nat.Coprime n d

theorem largest_fraction_proof 
  (a b c : ℕ) 
  (ha : is_simplified_proper_fraction a 6)
  (hb : is_simplified_proper_fraction b 15)
  (hc : is_simplified_proper_fraction c 20)
  (h_product : (a : ℚ) / 6 * (b : ℚ) / 15 * (c : ℚ) / 20 = 1 / 30) :
  (a : ℚ) / 6 = 5 / 6 ∧ 
  (a : ℚ) / 6 ≥ (b : ℚ) / 15 ∧ 
  (a : ℚ) / 6 ≥ (c : ℚ) / 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_fraction_proof_l300_30085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_minimum_length_l300_30019

theorem tangent_line_minimum_length :
  ∀ (a b : ℝ), 
    a^2 + b^2 = 1 →
    let A := (1/a : ℝ)
    let B := (1/b : ℝ)
    (Real.sqrt (A^2 + B^2) : ℝ) ≥ 2 ∧ 
    ∃ (a₀ b₀ : ℝ), a₀^2 + b₀^2 = 1 ∧ 
      let A₀ := (1/a₀ : ℝ)
      let B₀ := (1/b₀ : ℝ)
      (Real.sqrt (A₀^2 + B₀^2) : ℝ) = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_minimum_length_l300_30019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solution_l300_30092

theorem cubic_root_equation_solution : 
  ∃! x : ℚ, (5 - x : ℝ)^(1/3 : ℝ) = -3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solution_l300_30092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisible_l300_30064

def digits (x : ℕ+) : List ℕ :=
  Nat.digits 10 x.val

def digit_sum (x : ℕ+) : ℕ :=
  (digits x).sum

theorem digit_sum_divisible (n : ℕ) (h : n > 1) :
  ∃ x : ℕ+, 
    (∀ d : ℕ, d ∈ digits x → d ≠ 0) ∧ 
    (Nat.digits 10 x.val).length = n ∧
    x.val % (digit_sum x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_divisible_l300_30064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_for_even_numbers_l300_30018

-- Define the unknown fraction as a parameter
variable (f : ℚ)

-- Define the bracket function
def bracket (m : ℕ) : ℚ :=
  if m % 2 = 1 then 3 * m else f * m

-- Define the conditions
axiom odd_case (m : ℕ) (h : m % 2 = 1) : bracket f m = 3 * m
axiom even_case (m : ℕ) (h : m % 2 = 0) : bracket f m = f * m
axiom product_equality : bracket f 9 * bracket f 10 = 45

-- Theorem to prove
theorem fraction_for_even_numbers : f = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_for_even_numbers_l300_30018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleToMakeAllRed_l300_30072

/-- Represents a point in the pentagon configuration -/
structure Point where
  isBlue : Bool

/-- Represents the pentagon configuration -/
structure PentagonConfig where
  innerVertices : List Point
  otherPoints : List Point

/-- An operation that can be performed on the pentagon configuration -/
def performOperation (config : PentagonConfig) : PentagonConfig :=
  sorry

/-- Counts the number of blue inner vertices -/
def countBlueInnerVertices (config : PentagonConfig) : Nat :=
  config.innerVertices.filter (·.isBlue) |>.length

/-- The main theorem stating that it's impossible to make all points red -/
theorem impossibleToMakeAllRed (initialConfig : PentagonConfig) 
  (h1 : ∀ p ∈ initialConfig.innerVertices ++ initialConfig.otherPoints, p.isBlue = true)
  (h2 : ∀ c, countBlueInnerVertices (performOperation c) = countBlueInnerVertices c ∨ 
             countBlueInnerVertices (performOperation c) = countBlueInnerVertices c + 2 ∨
             countBlueInnerVertices (performOperation c) = countBlueInnerVertices c - 2)
  (h3 : initialConfig.innerVertices.length = 5) :
  ¬ ∃ finalConfig, (∃ operations : List (PentagonConfig → PentagonConfig), 
    finalConfig = operations.foldl (λ acc f => f acc) initialConfig ∧
    ∀ p ∈ finalConfig.innerVertices ++ finalConfig.otherPoints, p.isBlue = false) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleToMakeAllRed_l300_30072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_work_units_l300_30066

/-- The amount of work done by a group of women given the number of women, days, hours per day, and total work units -/
noncomputable def work_done (women : ℕ) (days : ℕ) (hours_per_day : ℕ) (total_units : ℝ) : ℝ :=
  (women * days * hours_per_day : ℝ) / total_units

theorem first_group_work_units : ∃ (w : ℝ),
  work_done 6 8 5 w = work_done 4 3 8 30 ∧ w = 75 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_work_units_l300_30066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l300_30076

theorem inequality_proof (m n p q : ℝ) (hm : m > 0) (hn : n > 0) (hp : p > 0) (hq : q > 0) :
  let t := (m + n + p + q) / 2
  (m / (t + n + p + q)) + (n / (t + p + q + m)) + (p / (t + q + m + n)) + (q / (t + m + n + p)) ≥ 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l300_30076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_population_ratio_l300_30054

/-- The ratio of whales this year to last year -/
theorem whale_population_ratio : ℚ := by
  /- Number of whales last year -/
  let whales_last_year : ℕ := 4000
  /- Predicted increase for next year -/
  let predicted_increase : ℕ := 800
  /- Predicted number of whales next year -/
  let whales_next_year : ℕ := 8800
  /- The total number of whales this year is a multiple of last year's number -/
  have whales_this_year_multiple : ∃ k : ℕ, k * whales_last_year = whales_next_year - predicted_increase := by sorry
  /- Relationship between this year and next year -/
  have next_year_prediction : whales_next_year = (whales_next_year - predicted_increase) + predicted_increase := by sorry
  /- The ratio of whales this year to last year is 2:1 -/
  have ratio_is_two : ((whales_next_year - predicted_increase) : ℚ) / whales_last_year = 2 := by sorry
  exact 2


end NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_population_ratio_l300_30054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_five_l300_30049

/-- Definition of the sequence u_n -/
def u : ℕ → ℚ
  | 0 => 5  -- Add this case to handle n = 0
  | 1 => 5
  | n + 1 => u n + 3 + 4 * (n - 1)

/-- u_n is a polynomial in n -/
axiom u_is_polynomial : ∃ (a b c : ℚ), ∀ n : ℕ, u n = a * n^2 + b * n + c

theorem sum_of_coefficients_is_five :
  ∃ (a b c : ℚ), (∀ n : ℕ, u n = a * n^2 + b * n + c) ∧ a + b + c = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_five_l300_30049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_cyclic_two_digit_numbers_l300_30037

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem gcd_of_cyclic_two_digit_numbers (a b c : ℕ) 
  (ha : is_digit a) (hb : is_digit b) (hc : is_digit c)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
  (hX : is_two_digit (10 * a + b))
  (hY : is_two_digit (10 * b + c))
  (hZ : is_two_digit (10 * c + a)) :
  Nat.gcd (10 * a + b) (Nat.gcd (10 * b + c) (10 * c + a)) ∈ ({1, 2, 3, 4, 7, 13, 14} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_cyclic_two_digit_numbers_l300_30037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_translated_function_pi_sixth_is_solution_l300_30093

/-- The original function -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x + Real.pi / 2)

/-- The translated function -/
noncomputable def g (x : ℝ) : ℝ := -3 * Real.sin (2 * x - Real.pi / 3)

/-- Theorem stating that (π/6, 0) is a symmetry center of the translated function -/
theorem symmetry_center_of_translated_function :
  ∀ x : ℝ, g (Real.pi / 6 + x) = g (Real.pi / 6 - x) :=
by
  sorry

/-- Theorem stating that (π/6, 0) is indeed a solution -/
theorem pi_sixth_is_solution :
  ∃ k : ℤ, Real.pi / 6 = k * Real.pi / 2 + Real.pi / 6 :=
by
  use 0
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_of_translated_function_pi_sixth_is_solution_l300_30093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l300_30080

theorem exponential_equation_solution :
  ∃ x : ℚ, (3 : ℝ) ^ ((2 : ℝ) * x^2 - 6*x + 2) = 3 ^ ((2 : ℝ) * x^2 + 8*x - 4) ∧ x = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l300_30080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_otimes_three_l300_30039

/-- The custom binary operation ⊗ -/
noncomputable def otimes (a b : ℝ) : ℝ := a^2 + (4 * a) / (3 * b)

/-- Theorem: 9 ⊗ 3 = 85 -/
theorem nine_otimes_three : otimes 9 3 = 85 := by
  -- Unfold the definition of otimes
  unfold otimes
  -- Simplify the expression
  simp [pow_two]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_otimes_three_l300_30039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l300_30027

noncomputable section

-- Define the angle in radians
def angle_radians : ℝ := 9 * Real.pi / 4

-- Define the equivalent angle in degrees
def angle_degrees (k : ℤ) : ℝ := k * 360 - 315

-- Theorem statement
theorem same_terminal_side :
  ∀ k : ℤ, ∃ n : ℤ, angle_radians = (angle_degrees k + n * 360) * (Real.pi / 180) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_terminal_side_l300_30027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_inequalities_l300_30056

theorem relationship_between_inequalities (a b c d : ℝ) (h : c < d) :
  (∀ a b c d : ℝ, c < d → a - c < b - d → a < b) ∧
  (∃ a b c d : ℝ, c < d ∧ a < b ∧ ¬(a - c < b - d)) :=
by
  constructor
  · -- Prove necessity
    intros a b c d hcd hineq
    linarith
  · -- Prove not sufficient
    use 2, 3, 0, 1
    constructor
    · linarith -- c < d
    constructor
    · linarith -- a < b
    · linarith -- ¬(a - c < b - d)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_between_inequalities_l300_30056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_newspaper_sales_l300_30000

/-- Represents the daily newspaper sales scenario --/
structure NewspaperSales where
  buy_price : ℚ
  sell_price : ℚ
  return_price : ℚ
  total_days : ℕ
  high_demand_days : ℕ
  low_demand_days : ℕ
  high_demand_sales : ℕ
  low_demand_sales : ℕ

/-- Calculates the monthly profit for a given number of copies bought daily --/
def monthly_profit (s : NewspaperSales) (copies : ℕ) : ℚ :=
  let high_revenue := s.high_demand_days * s.sell_price * (min copies s.high_demand_sales : ℚ)
  let low_revenue := s.low_demand_days * s.sell_price * (min copies s.low_demand_sales : ℚ)
  let return_revenue := s.return_price * ((copies : ℚ) - s.low_demand_sales) * s.low_demand_days
  let cost := s.buy_price * (copies : ℚ) * s.total_days
  high_revenue + low_revenue + return_revenue - cost

/-- The main theorem stating the optimal number of copies and maximum profit --/
theorem optimal_newspaper_sales (s : NewspaperSales) 
    (h1 : s.buy_price = 1/5)
    (h2 : s.sell_price = 3/10)
    (h3 : s.return_price = 1/20)
    (h4 : s.total_days = 30)
    (h5 : s.high_demand_days = 20)
    (h6 : s.low_demand_days = 10)
    (h7 : s.high_demand_sales = 400)
    (h8 : s.low_demand_sales = 250) :
    ∃ (optimal_copies : ℕ) (max_profit : ℚ),
      optimal_copies = 400 ∧
      max_profit = 825 ∧
      ∀ (x : ℕ), monthly_profit s x ≤ max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_newspaper_sales_l300_30000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l300_30069

/-- Given an ellipse C with eccentricity e = √6/3 and points A and B on C with N(3,1) as the midpoint of AB,
    if the circle with diameter AB is tangent to the line √2x + y + 1 = 0,
    then the equation of the ellipse is (x^2/24) + (y^2/8) = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (C : Set (ℝ × ℝ)) 
  (hC : C = {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1})
  (e : ℝ) (he : e = Real.sqrt 6/3)
  (A B : ℝ × ℝ) (hA : A ∈ C) (hB : B ∈ C)
  (N : ℝ × ℝ) (hN : N = (3, 1))
  (hMidpoint : N = ((A.1 + B.1)/2, (A.2 + B.2)/2))
  (hTangent : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
    radius = Real.sqrt (((A.1 - B.1)^2 + (A.2 - B.2)^2)/4) ∧
    |Real.sqrt 2 * center.1 + center.2 + 1| = radius * Real.sqrt 3) :
  C = {p : ℝ × ℝ | p.1^2/24 + p.2^2/8 = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l300_30069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_problem_l300_30088

/-- Represents a vessel containing an alcohol mixture -/
structure Vessel where
  capacity : ℝ
  alcohol_percentage : ℝ

/-- Calculates the amount of pure alcohol in a vessel -/
noncomputable def alcohol_amount (v : Vessel) : ℝ :=
  v.capacity * (v.alcohol_percentage / 100)

/-- The problem statement -/
theorem alcohol_mixture_problem (vessel1 vessel2 : Vessel) 
  (h1 : vessel1.capacity = 2)
  (h2 : vessel1.alcohol_percentage = 30)
  (h3 : vessel2.capacity = 6)
  (h4 : vessel2.alcohol_percentage = 40)
  (h5 : vessel1.capacity + vessel2.capacity = 8) :
  let total_alcohol := alcohol_amount vessel1 + alcohol_amount vessel2
  let total_volume := vessel1.capacity + vessel2.capacity
  let new_concentration := (total_alcohol / total_volume) * 100
  new_concentration = 30.000000000000004 ∧ total_volume = 8 := by
  sorry

#eval 1 -- Add this line to ensure the file is parsed correctly

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_mixture_problem_l300_30088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l300_30047

/-- Represents the tax structure of Country X -/
structure TaxStructure where
  lowRate : ℚ  -- Tax rate for income up to the threshold
  highRate : ℚ  -- Tax rate for income above the threshold
  threshold : ℚ  -- Income threshold for tax rate change

/-- Calculates the total tax for a given income and tax structure -/
def calculateTax (income : ℚ) (tax : TaxStructure) : ℚ :=
  if income ≤ tax.threshold then
    income * tax.lowRate
  else
    tax.threshold * tax.lowRate + (income - tax.threshold) * tax.highRate

/-- Theorem: Given the tax structure and total tax, the income is $50,000 -/
theorem income_from_tax (tax : TaxStructure) (totalTax : ℚ) :
  tax.lowRate = 15/100 →
  tax.highRate = 20/100 →
  tax.threshold = 40000 →
  totalTax = 8000 →
  ∃ income, income = 50000 ∧ calculateTax income tax = totalTax := by
  sorry

#eval calculateTax 50000 { lowRate := 15/100, highRate := 20/100, threshold := 40000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_l300_30047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_unity_sum_l300_30073

-- Define w as a complex number
noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

-- State the theorem
theorem root_of_unity_sum :
  w / (1 + w^3) + w^2 / (1 + w^6) + w^3 / (1 + w^9) = -(w^3 + w^2 + w) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_of_unity_sum_l300_30073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_formula_l300_30060

/-- The speed of a particle with position function (t^2 + 2t + 3, 3t^2 - t + 2) at time t. -/
noncomputable def particleSpeed (t : ℝ) : ℝ :=
  let x := t^2 + 2*t + 3
  let y := 3*t^2 - t + 2
  let vx := 2*t + 2
  let vy := 6*t - 1
  Real.sqrt (vx^2 + vy^2)

/-- Theorem stating that the speed of the particle is equal to √(40t^2 - 4t + 5) -/
theorem particle_speed_formula (t : ℝ) : 
  particleSpeed t = Real.sqrt (40*t^2 - 4*t + 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_speed_formula_l300_30060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l300_30086

noncomputable def f (x y z w : ℝ) : ℝ := x / (x + y) + y / (y + z) + z / (z + x) + w / (w + x)

theorem f_bounds (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) :
  1 < f x y z w ∧ f x y z w < 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l300_30086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_function_form_l300_30008

open Complex

-- Define the complex function f
def f (z : ℂ) (ξ η ω τ : ℝ → ℝ) : ℂ :=
  let x := z.re
  let y := z.im
  (ξ x + ω y - 3 * x * y^2) + I * (η x + τ y + 3 * x^2 * y + 4 * x * y + 5 * x)

-- State the theorem
theorem analytic_function_form (ξ η ω τ : ℝ → ℝ) :
  Differentiable ℂ (λ z ↦ f z ξ η ω τ) →
  ∃ z₀ : ℂ, ∀ z, f z ξ η ω τ = z^3 + 2*z^2 + 5*I*z + z₀ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_analytic_function_form_l300_30008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l300_30077

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sqrt 3 * tan (x / 2 - π / 4)

-- State the theorem
theorem min_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_period_of_f_l300_30077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l300_30044

/-- The probability of person A hitting the target -/
def p : ℝ := sorry

/-- The probability of person B hitting the target -/
def q : ℝ := sorry

/-- Assumption that p is a valid probability -/
axiom hp : 0 ≤ p ∧ p ≤ 1

/-- Assumption that q is a valid probability -/
axiom hq : 0 ≤ q ∧ q ≤ 1

/-- The probability that person A shoots 4 times and misses the target at least once -/
def prob_A_misses_at_least_once : ℝ := 1 - p^4

/-- The probability that person A hits the target exactly 2 times out of 4 shots -/
def prob_A_hits_2_out_of_4 : ℝ := 6 * p^2 * (1-p)^2

/-- The probability that person B hits the target exactly 3 times out of 4 shots -/
def prob_B_hits_3_out_of_4 : ℝ := 4 * q^3 * (1-q)

/-- The probability that both persons shoot 4 times, with person A hitting the target exactly 2 times
    and person B hitting the target exactly 3 times -/
def prob_A_2_and_B_3 : ℝ := 24 * p^2 * q^3 * (1-p)^2 * (1-q)

theorem shooting_probabilities :
  (prob_A_misses_at_least_once = 1 - p^4) ∧
  (prob_A_hits_2_out_of_4 = 6 * p^2 * (1-p)^2) ∧
  (prob_B_hits_3_out_of_4 = 4 * q^3 * (1-q)) ∧
  (prob_A_2_and_B_3 = 24 * p^2 * q^3 * (1-p)^2 * (1-q)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l300_30044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_time_formula_l300_30005

/-- Calculates the total time taken by a train traveling three segments -/
noncomputable def totalTrainTime (a b c : ℝ) : ℝ :=
  (2 * a / 30) + (3 * b / 50) + (4 * c / 70)

/-- Theorem stating that the total time is equal to the given formula -/
theorem train_time_formula (a b c : ℝ) :
  totalTrainTime a b c = (140 * a + 126 * b + 120 * c) / 2100 := by
  unfold totalTrainTime
  -- Perform algebraic manipulations
  simp [add_div, mul_div_assoc]
  -- Simplify fractions
  simp [←div_div, div_eq_mul_inv]
  -- Combine terms
  ring
  -- QED


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_time_formula_l300_30005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_2017th_term_l300_30002

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_2017th_term :
  ∀ (d : ℝ),
  let a₁ : ℝ := -2017
  let S := arithmetic_sum a₁ d
  (S 2007 / 2007 - S 2005 / 2005 = 2) →
  arithmetic_sequence a₁ d 2017 = 2015 :=
by
  sorry

#check arithmetic_sequence_2017th_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_2017th_term_l300_30002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_negative_one_ninth_l300_30011

theorem sum_of_roots_equals_negative_one_ninth
  (f : ℝ → ℝ)
  (h : ∀ x, f (x / 3) = x^2 + x + 1)
  (c : ℝ) :
  let g := λ x ↦ 81 * x^2 + 9 * x + (1 - c)
  (∃ x₁ x₂, g x₁ = 0 ∧ g x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ x₁ x₂, g x₁ = 0 ∧ g x₂ = 0 ∧ x₁ + x₂ = -1/9) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_equals_negative_one_ninth_l300_30011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_denominators_l300_30081

/-- A digit is a natural number from 0 to 9 -/
def Digit : Type := { n : ℕ // n ≤ 9 }

/-- Convert three digits to a natural number -/
def digits_to_nat (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- The repeating decimal 0.abc̅ as a fraction -/
def repeating_decimal (a b c : Digit) : ℚ := (digits_to_nat a b c : ℚ) / 999

theorem count_denominators :
  ∃ (S : Finset ℕ),
    (∀ (a b c : Digit), ¬(a.val = 9 ∧ b.val = 9 ∧ c.val = 9) →
      (repeating_decimal a b c).den ∈ S) ∧
    S.card = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_denominators_l300_30081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_size_calculation_l300_30065

/-- The maximum affordable apartment size given a rental rate and budget -/
noncomputable def max_apartment_size (rate : ℝ) (budget : ℝ) : ℝ :=
  budget / rate

/-- Theorem: Given the rental rate and budget, the maximum affordable apartment size is 600 sq ft -/
theorem apartment_size_calculation (rate budget : ℝ) 
  (h_rate : rate = 1.25)
  (h_budget : budget = 750) :
  max_apartment_size rate budget = 600 := by
  -- Unfold the definition of max_apartment_size
  unfold max_apartment_size
  -- Substitute the given values
  rw [h_rate, h_budget]
  -- Perform the division
  norm_num

-- This line is removed as it's not computable
-- #eval max_apartment_size 1.25 750

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apartment_size_calculation_l300_30065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_tea_overlap_l300_30013

theorem coffee_tea_overlap (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 0.8)
  (h2 : tea_drinkers = 0.7) :
  max 0 (coffee_drinkers + tea_drinkers - 1) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coffee_tea_overlap_l300_30013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l300_30052

theorem honor_students_count
  (total_students : ℕ)
  (girls : ℕ)
  (boys : ℕ)
  (honor_girls : ℕ)
  (honor_boys : ℕ)
  (h1 : total_students < 30)
  (h2 : total_students = girls + boys)
  (h3 : (honor_girls : ℚ) / girls = 3 / 13)
  (h4 : (honor_boys : ℚ) / boys = 4 / 11)
  (h5 : girls > 0)
  (h6 : boys > 0) :
  honor_girls + honor_boys = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_honor_students_count_l300_30052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_APF_l300_30051

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- Calculate the sine of the angle APF -/
noncomputable def sinAPF (e : Ellipse) (p : Point) : ℝ :=
  sorry -- Actual calculation would go here

/-- Theorem stating the maximum value of sin APF -/
theorem max_sin_APF (e : Ellipse) :
  eccentricity e = 1/5 →
  ∃ (p : Point), p.y = 0 ∧ sinAPF e p ≤ 1/2 ∧
  ∀ (q : Point), q.y = 0 → sinAPF e q ≤ sinAPF e p :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sin_APF_l300_30051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_props_true_l300_30067

-- Define vectors in R²
def Vector2 := Fin 2 → ℝ

-- Define vector equality
def vector_eq (a b : Vector2) : Prop := ∀ i, a i = b i

-- Define vector magnitude
noncomputable def magnitude (a : Vector2) : ℝ := Real.sqrt ((a 0)^2 + (a 1)^2)

-- Define vector parallelism
def parallel (a b : Vector2) : Prop := ∃ k : ℝ, ∀ i, a i = k * b i

-- Define the propositions
def prop1 : Prop := ∀ a b : Vector2, magnitude a = magnitude b → vector_eq a b

def prop2 : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ Real.cos x = -2/3 → 
  x = Real.pi - Real.arccos (2/3)

def prop3 : Prop := ∀ a b c : Vector2, vector_eq a b ∧ vector_eq b c → vector_eq a c

def prop4 : Prop := ∀ a b : Vector2, vector_eq a b → 
  magnitude a = magnitude b ∧ parallel a b

-- Theorem stating that exactly 3 out of 4 propositions are true
theorem three_props_true : 
  (¬ prop1 ∧ prop2 ∧ prop3 ∧ prop4) ∧ 
  (¬ prop1 ∨ ¬ prop2 ∨ ¬ prop3 ∨ ¬ prop4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_props_true_l300_30067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_L_l300_30098

-- Define the curve C in polar coordinates
noncomputable def C (θ : ℝ) : ℝ := 2 / Real.sqrt (1 + 3 * Real.sin θ ^ 2)

-- Define the line L
noncomputable def L (x y : ℝ) : ℝ := x - 2 * y - 4 * Real.sqrt 2

-- Define the distance function from a point (x, y) to the line L
noncomputable def distance_to_L (x y : ℝ) : ℝ :=
  abs (L x y) / Real.sqrt 5

-- Theorem statement
theorem min_distance_to_L :
  ∃ (d : ℝ), d = 2 * Real.sqrt 10 / 5 ∧
  ∀ (θ : ℝ), distance_to_L (C θ * Real.cos θ) (C θ * Real.sin θ) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_L_l300_30098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quadrant_l300_30089

open Real

-- Define a function to check if an angle is in the third quadrant
def isInThirdQuadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

-- Define a function to check if an angle is in the second or fourth quadrant
def isInSecondOrFourthQuadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + Real.pi) ∨
           (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < k * Real.pi + 2 * Real.pi)

-- Theorem statement
theorem angle_bisector_quadrant (α : ℝ) :
  isInThirdQuadrant α → isInSecondOrFourthQuadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_quadrant_l300_30089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_value_l300_30035

/-- Represents the state of the program at each iteration -/
structure State where
  x : ℕ
  s : ℕ

/-- Updates the state according to the program rules -/
def updateState (st : State) : State :=
  { x := st.x + 3,
    s := st.s + st.x + 3 }

/-- Checks if the termination condition is met -/
def isTerminated (st : State) : Bool :=
  st.s ≥ 12000

/-- Computes the final state of the program -/
def finalState : State :=
  let rec loop (st : State) (fuel : ℕ) : State :=
    if fuel = 0 then st
    else if isTerminated st then st
    else loop (updateState st) (fuel - 1)
  loop { x := 5, s := 10 } 1000  -- Use a sufficiently large fuel value

/-- The theorem to be proved -/
theorem final_x_value :
  finalState.x = 275 := by
  sorry

#eval finalState.x  -- This will print the actual result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_x_value_l300_30035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_domain_range_l300_30091

noncomputable def f (a b x : ℝ) : ℝ := a^x + b

theorem exponential_function_domain_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (ha_neq : a ≠ 1) 
  (hf_domain : ∀ x, x ∈ Set.Icc (-1) 0 ↔ f a b x ∈ Set.Icc (-1) 0) 
  (hf_range : Set.range (f a b) = Set.Icc (-1) 0) : 
  a + b = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_domain_range_l300_30091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bailey_final_count_l300_30042

/-- The number of rubber bands each person has --/
structure RubberBands where
  bailey : ℕ
  justine : ℕ
  ylona : ℕ

/-- The initial state of rubber bands --/
def initial : RubberBands :=
  { bailey := 12,  -- We know Bailey had 12 initially
    justine := 22, -- We know Justine had 22 initially
    ylona := 24 }  -- Given in the problem

/-- The final state of rubber bands after Bailey gives away 2 to each --/
def final : RubberBands :=
  { bailey := initial.bailey - 4,
    justine := initial.justine + 2,
    ylona := initial.ylona + 2 }

/-- Theorem stating the conditions and the result to be proved --/
theorem bailey_final_count :
  initial.justine = initial.bailey + 10 ∧
  initial.justine = initial.ylona - 2 ∧
  final.bailey = 8 :=
by
  -- Split the conjunction into separate goals
  constructor
  · -- Prove initial.justine = initial.bailey + 10
    rfl
  constructor
  · -- Prove initial.justine = initial.ylona - 2
    rfl
  · -- Prove final.bailey = 8
    rfl

#eval final.bailey -- Should output 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bailey_final_count_l300_30042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_for_half_time_l300_30099

-- Define the initial distance and time
noncomputable def initial_distance : ℝ := 440
noncomputable def initial_time : ℝ := 3

-- Define the new time as half of the initial time
noncomputable def new_time : ℝ := initial_time / 2

-- Define the initial speed
noncomputable def initial_speed : ℝ := initial_distance / initial_time

-- Define the new speed
noncomputable def new_speed : ℝ := initial_distance / new_time

-- Theorem to prove
theorem speed_for_half_time :
  new_speed = 2 * initial_speed :=
by
  -- Unfold definitions
  unfold new_speed initial_speed new_time
  -- Simplify the expression
  simp [initial_distance, initial_time]
  -- Prove equality using field operations
  field_simp
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_for_half_time_l300_30099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_order_l300_30043

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an inverse proportion function y = k/x where k > 0 -/
noncomputable def InverseProportion (k : ℝ) : ℝ → ℝ :=
  fun x => k / x

theorem inverse_proportion_point_order
  (k : ℝ)
  (h_k : k > 0)
  (A B C : Point)
  (h_A : A.y = -3)
  (h_B : B.y = 2)
  (h_C : C.y = 6)
  (h_A_on_graph : A.y = InverseProportion k A.x)
  (h_B_on_graph : B.y = InverseProportion k B.x)
  (h_C_on_graph : C.y = InverseProportion k C.x) :
  A.x < C.x ∧ C.x < B.x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_point_order_l300_30043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_relationships_l300_30046

/-- Represents the correlation between two variables -/
inductive Correlation
| Positive
| Negative

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Given conditions and relationships between variables x, y, and z -/
theorem correlation_relationships 
  (regression_eq : LinearRegression)
  (y_x_correlation : Correlation) :
  regression_eq.slope = -0.1 →
  regression_eq.intercept = 1 →
  y_x_correlation = Correlation.Negative →
  (∃ (x_y_correlation x_z_correlation : Correlation),
    x_y_correlation = Correlation.Negative ∧
    x_z_correlation = Correlation.Positive) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_relationships_l300_30046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l300_30059

def A : Set ℝ := {x | x^2 ≥ 4}
def B : Set ℝ := {x | (6 - x) / (1 + x) ≥ 0}
def C : Set ℝ := {x | |x - 3| < 3}

theorem set_operations :
  ((Set.univ \ B) ∪ (Set.univ \ C) = {x : ℝ | x ≤ 0 ∨ x ≥ 6}) ∧
  (A ∩ (Set.univ \ (B ∩ C)) = {x : ℝ | x ≤ -2 ∨ x ≥ 6}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_l300_30059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_injective_finite_primes_growth_l300_30038

/-- A sequence of natural numbers is almost injective if there exists a constant C
    such that each value appears at most C times in the sequence. -/
def AlmostInjective (a : ℕ → ℕ) : Prop :=
  ∃ C : ℕ, ∀ k : ℕ, (Finset.filter (fun n => a n = k) (Finset.range (Nat.succ k))).card ≤ C

/-- A sequence of natural numbers has a finite number of prime divisors if
    there exists a finite set of primes that divide all terms of the sequence. -/
def FinitePrimeDivisors (a : ℕ → ℕ) : Prop :=
  ∃ S : Finset ℕ, (∀ p ∈ S, Nat.Prime p) ∧ 
    ∀ n : ℕ, ∀ p : ℕ, Nat.Prime p → p ∣ a n → p ∈ S

theorem almost_injective_finite_primes_growth 
  (a : ℕ → ℕ) (h_incr : Monotone a) (h_ainj : AlmostInjective a) (h_fpd : FinitePrimeDivisors a) :
  ∃ c : ℝ, c > 1 ∧ ∃ M : ℕ → ℝ, (∀ n, M n > 0) ∧ ∀ n, c^n ≤ M n * (a n : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_almost_injective_finite_primes_growth_l300_30038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_l300_30020

theorem cos_sin_sum (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 1) 
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2 * x) + Real.cos (2 * y) + Real.cos (2 * z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_sum_l300_30020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_equals_two_l300_30003

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions of the problem
noncomputable def special_triangle (A : ℝ) (c : ℝ) : Triangle where
  A := A
  B := 2 * A
  C := Real.pi - 3 * A  -- Since the sum of angles in a triangle is π
  a := 1
  b := Real.sqrt 3
  c := c

-- Theorem statement
theorem side_c_equals_two (A c : ℝ) (t : Triangle) 
    (h : t = special_triangle A c) : t.c = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_c_equals_two_l300_30003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_novels_readers_fraction_l300_30032

theorem novels_readers_fraction (total_students : ℕ) 
  (two_novels_percent : ℚ) (one_novel_fraction : ℚ) (no_novels : ℕ) 
  (h1 : total_students = 240)
  (h2 : two_novels_percent = 35 / 100)
  (h3 : one_novel_fraction = 5 / 12)
  (h4 : no_novels = 16) :
  (total_students - (total_students * two_novels_percent).floor - 
   (total_students * one_novel_fraction).floor - no_novels : ℚ) / 
  total_students = 1 / 6 := by
  sorry

#check novels_readers_fraction

end NUMINAMATH_CALUDE_ERRORFEEDBACK_novels_readers_fraction_l300_30032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_myoglobin_not_homeostasis_l300_30004

/-- Represents the options for the homeostasis question -/
inductive HomeostasisExample
  | pH_regulation
  | phagocyte_elimination
  | myoglobin_content
  | osmotic_pressure

/-- The correct answer to the homeostasis question -/
def correct_answer : HomeostasisExample := HomeostasisExample.myoglobin_content

/-- A theorem stating that the myoglobin content option is the correct answer -/
theorem myoglobin_not_homeostasis : 
  correct_answer = HomeostasisExample.myoglobin_content :=
by
  -- The proof is omitted as it's based on biological knowledge rather than mathematical deduction
  sorry

#check myoglobin_not_homeostasis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_myoglobin_not_homeostasis_l300_30004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2013_value_l300_30026

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 4 * x / (x + 4)

/-- The sequence x_n as defined by the recurrence relation -/
noncomputable def x : ℕ → ℝ
  | 0 => 1  -- Define for 0 to cover all natural numbers
  | n + 1 => f (x n)

/-- The main theorem stating that x₂₀₁₃ = 1/504 -/
theorem x_2013_value : x 2013 = 1 / 504 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_2013_value_l300_30026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gregorian_year_length_l300_30045

/-- Determines if a year is a leap year in the Gregorian calendar -/
def is_leap_year (n : ℕ) : Bool :=
  (n % 4 = 0 && n % 100 ≠ 0) || (n % 100 = 0 && n % 400 = 0)

/-- The number of days in a year -/
def days_in_year (n : ℕ) : ℕ :=
  if is_leap_year n then 366 else 365

/-- The total number of days in a 400-year cycle -/
def total_days_in_400_years : ℕ :=
  (Finset.range 400).sum days_in_year

/-- The average length of a year in the Gregorian calendar -/
noncomputable def average_year_length : ℚ :=
  (total_days_in_400_years : ℚ) / 400

theorem gregorian_year_length :
  average_year_length = 365.2425 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gregorian_year_length_l300_30045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_for_probability_l300_30096

/-- Represents a non-regular hexagon with given side lengths -/
structure NonRegularHexagon where
  sides : Fin 6 → ℝ
  side_positive : ∀ i, sides i > 0

/-- Represents a circle concentric with a hexagon -/
structure ConcentricCircle (h : NonRegularHexagon) where
  radius : ℝ
  radius_positive : radius > 0

/-- The probability of seeing four entire sides from a random point on the circle -/
noncomputable def probability_four_sides (h : NonRegularHexagon) (c : ConcentricCircle h) : ℝ := sorry

/-- The theorem stating the relationship between the hexagon, circle, and probability -/
theorem circle_radius_for_probability (h : NonRegularHexagon) 
  (h_sides : h.sides = ![3, 2, 4, 3, 2, 4]) 
  (c : ConcentricCircle h) 
  (prob : probability_four_sides h c = 1/3) : 
  ∃ ε > 0, |c.radius - 17| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_for_probability_l300_30096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_maximized_at_two_radians_l300_30009

/-- The length of the rope forming the sector -/
noncomputable def rope_length : ℝ := 20

/-- The area of a sector given its radius and central angle -/
noncomputable def sector_area (r : ℝ) (θ : ℝ) : ℝ := (1/2) * r^2 * θ

/-- The theorem stating that the area of the sector is maximized when the central angle is 2 radians -/
theorem sector_area_maximized_at_two_radians :
  ∃ (r : ℝ), r > 0 ∧ r < rope_length/2 ∧
  ∀ (r' : ℝ) (θ' : ℝ), 
    r' > 0 → r' < rope_length/2 → θ' > 0 → 
    2 * r' + r' * θ' = rope_length →
    sector_area r' θ' ≤ sector_area r 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_maximized_at_two_radians_l300_30009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_result_l300_30079

noncomputable def original_function (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (2 * x + Real.pi / 4)

noncomputable def translation : ℝ := Real.pi / 12

noncomputable def translated_function (x : ℝ) : ℝ := original_function (x - translation)

theorem translation_result :
  ∀ x : ℝ, translated_function x = Real.sqrt 2 * Real.sin (2 * x + Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_result_l300_30079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l300_30084

def f (c : ℝ) (x : ℝ) : ℝ := x^2 - x + c

theorem f_properties (c : ℝ) :
  (∀ x, x ∈ Set.Icc 0 1 → f c x ≤ c ∧ f c x ≥ c - 1/4) ∧
  (∀ x₁ x₂, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → |f c x₁ - f c x₂| ≤ 1/4) ∧
  ((∃ x₁ x₂, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0) → 0 ≤ c ∧ c < 1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l300_30084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l300_30023

/-- Geometric sequence sum -/
noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

/-- Proof of S₈ value for a specific geometric sequence -/
theorem geometric_sequence_sum_8 (a₁ q : ℝ) (h_q : q ≠ 1) :
  geometric_sum a₁ q 4 = -5 →
  geometric_sum a₁ q 6 = 21 * geometric_sum a₁ q 2 →
  geometric_sum a₁ q 8 = -85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l300_30023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subsequence_l300_30022

def original_number : ℕ := 2946835107

def is_subseq (s t : List ℕ) : Prop :=
  ∃ (l : List ℕ), l ++ s = t

def digits_to_num (l : List ℕ) : ℕ :=
  l.foldl (λ acc d ↦ 10 * acc + d) 0

def num_to_digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 10) ((m % 10) :: acc)
    aux n []

theorem largest_subsequence :
  ∀ (l : List ℕ),
    is_subseq l (num_to_digits original_number) →
    l.length = 5 →
    digits_to_num l ≤ 98517 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subsequence_l300_30022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l300_30017

/-- Represents a polynomial with integer coefficients -/
def MyPolynomial := List Int

/-- Horner's method for evaluating a polynomial at a given point -/
def horner_eval (p : MyPolynomial) (x : Int) : Int × Nat × Nat :=
  p.foldl
    (fun (acc, mults, adds) a =>
      let new_acc := acc * x + a
      (new_acc, mults + 1, adds + 1))
    (0, 0, 0)

/-- The specific polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f : MyPolynomial := [5, 4, 3, 2, 1, 1]

theorem horner_method_operations :
  let (result, mults, adds) := horner_eval f 2
  mults = 5 ∧ adds = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l300_30017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l300_30070

theorem trigonometric_simplification :
  let θ : Real := 15 * π / 180  -- 15 degrees in radians
  (Real.tan θ ^ 3 + (Real.tan θ)⁻¹ ^ 3) / (Real.tan θ + (Real.tan θ)⁻¹) = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l300_30070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_l₁_l₂_l300_30053

noncomputable section

/-- The angle between two lines in the plane -/
def angle_between_lines (m₁ m₂ : ℝ) : ℝ :=
  Real.arctan ((m₂ - m₁) / (1 + m₁ * m₂))

/-- Line l₁: √3x - y + 2 = 0 -/
def l₁ : ℝ → ℝ → Prop :=
  λ x y ↦ Real.sqrt 3 * x - y + 2 = 0

/-- Line l₂: 3x + √3y - 5 = 0 -/
def l₂ : ℝ → ℝ → Prop :=
  λ x y ↦ 3 * x + Real.sqrt 3 * y - 5 = 0

/-- The slope of line l₁ -/
def m₁ : ℝ := Real.sqrt 3

/-- The slope of line l₂ -/
def m₂ : ℝ := -(Real.sqrt 3)

theorem angle_between_l₁_l₂ : angle_between_lines m₁ m₂ = π / 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_l₁_l₂_l300_30053
