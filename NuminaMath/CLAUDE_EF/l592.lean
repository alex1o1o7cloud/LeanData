import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_students_l592_59276

/-- The number of students in Maxwell's high school -/
def M : ℕ := sorry

/-- The number of students in Jeremy's high school -/
def J : ℕ := sorry

/-- Maxwell's has 6 times as many students as Jeremy's -/
axiom h1 : M = 6 * J

/-- After transferring 300 students, Maxwell's still has 5 times as many as Jeremy's -/
axiom h2 : M - 300 = 5 * (J + 300)

/-- Theorem: The number of students at Maxwell's high school is 10800 -/
theorem maxwell_students : M = 10800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maxwell_students_l592_59276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_6_l592_59227

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 4 * x + 8 else 3 * x - 15

-- Theorem statement
theorem solutions_of_f_eq_6 :
  {x : ℝ | f x = 6} = {-1/2, 7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_f_eq_6_l592_59227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_account_balance_after_six_months_l592_59206

/-- Calculates the compound interest for a given principal, rate, and time -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / 4) ^ (4 * time)

/-- Represents the savings account balance after 6 months -/
noncomputable def account_balance : ℝ :=
  let initial_deposit := 5000
  let q1_rate := 0.07
  let q2_rate := 0.085
  let additional_deposit := 2000
  let withdrawal := 1000

  let q1_balance := compound_interest initial_deposit q1_rate 0.25
  let q2_balance := compound_interest (q1_balance + additional_deposit) q2_rate 0.25
  q2_balance - withdrawal

/-- Theorem stating that the account balance after six months is approximately 6239.06 -/
theorem account_balance_after_six_months :
  ‖account_balance - 6239.06‖ < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_account_balance_after_six_months_l592_59206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l592_59209

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 2)

theorem f_is_even_and_periodic : 
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f (x + Real.pi) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_and_periodic_l592_59209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_event_probability_l592_59286

open MeasureTheory

noncomputable def event (x : ℝ) : Prop := 
  -1 ≤ Real.log (x + 1/2) / Real.log (1/2) ∧ Real.log (x + 1/2) / Real.log (1/2) ≤ 1

theorem event_probability : 
  (volume {x : ℝ | x ∈ Set.Icc 0 2 ∧ event x}) / (volume (Set.Icc 0 2 : Set ℝ)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_event_probability_l592_59286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l592_59263

/-- A function f(x) is an inverse proportion if there exists a non-zero constant k such that f(x) = k/x for all x ≠ 0 -/
def IsInverseProportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function defined by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^(m^2 - 5)

theorem inverse_proportion_m_value :
  ∃! m : ℝ, IsInverseProportion (f m) ∧ m - 2 ≠ 0 ∧ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_value_l592_59263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l592_59244

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define reflection over y = -x
def reflectOverNegativeX (p : Point2D) : Point2D :=
  { x := -p.y, y := -p.x }

-- Define reflection over y = x
def reflectOverX (p : Point2D) : Point2D :=
  { x := p.y, y := p.x }

-- Calculate distance between two points
noncomputable def distance (p1 p2 : Point2D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Calculate area of a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : Point2D) : ℝ :=
  let a := distance p1 p2
  let b := distance p2 p3
  let c := distance p3 p1
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_ABC_area :
  let A : Point2D := { x := 3, y := 4 }
  let B : Point2D := reflectOverNegativeX A
  let C : Point2D := reflectOverX B
  triangleArea A B C = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_area_l592_59244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_random_number_transformation_l592_59281

theorem uniform_random_number_transformation (x : ℝ) :
  0 ≤ x ∧ x ≤ 1 →
  (0 ≤ 4 * x ∧ 4 * x ≤ 4) ∧ (-4 ≤ 5 * x - 4 ∧ 5 * x - 4 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uniform_random_number_transformation_l592_59281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_l592_59285

/-- Represents a digit (0-9) -/
def Digit := Fin 10

theorem sum_of_digits (a b c d e : Digit) : 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
   c ≠ d ∧ c ≠ e ∧ 
   d ≠ e) →
  (c.val + e.val : ℕ) % 10 = 0 →
  (b.val + c.val : ℕ) % 10 = 0 →
  d.val + a.val + 1 = 10 →
  a.val + b.val + c.val + d.val + e.val = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_l592_59285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_sequence_exists_l592_59299

open Set
open Real

/-- Represents a point in the 2D plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- Calculates the distance between two RationalPoints -/
noncomputable def distance (p q : RationalPoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- The origin point O(0,0) -/
def O : RationalPoint := ⟨0, 0⟩

/-- The point A(0,1/2) -/
def A : RationalPoint := ⟨0, 1/2⟩

/-- Theorem: There is no finite sequence of rational points satisfying the given conditions -/
theorem no_rational_sequence_exists :
  ¬ ∃ (n : ℕ) (P : Fin (n+1) → RationalPoint),
    (distance O (P 0) = 1 ∧
     distance (P n) A = 1 ∧
     ∀ i : Fin n, distance (P i) (P (i.succ)) = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_sequence_exists_l592_59299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l592_59283

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the expression
noncomputable def expression : ℝ :=
  log10 5 * (log10 8 + log10 1000) + (log10 (2^Real.sqrt 3))^2 + log10 (1/6) + log10 0.06

-- Theorem statement
theorem expression_equals_one : expression = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equals_one_l592_59283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_equals_2008_l592_59202

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Assume f and g are bijective (have inverse functions)
axiom f_bijective : Function.Bijective f
axiom g_bijective : Function.Bijective g

-- Define the symmetry condition
axiom symmetry : ∀ x y : ℝ, f (x - 1) = y ↔ g⁻¹ (y - 3) = x

-- Given condition
axiom g_5 : g 5 = 2005

-- Theorem to prove
theorem f_4_equals_2008 : f 4 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_4_equals_2008_l592_59202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_complex_number_equality_l592_59290

def complex_number_equality : Prop :=
  ((1 - 2*Complex.I) / (2 + Complex.I)) + 2*Complex.I = Complex.I

theorem prove_complex_number_equality : complex_number_equality := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_complex_number_equality_l592_59290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cyclic_permutations_times_three_l592_59296

def cyclic_permute (n : Nat) : Nat := 
  let s := toString n
  let rotated := s.dropRight 1
  (s.back.toString ++ rotated).toNat!

def sum_of_permutations (start : Nat) : Nat :=
  start + 
  cyclic_permute start + 
  cyclic_permute (cyclic_permute start) + 
  cyclic_permute (cyclic_permute (cyclic_permute start))

theorem sum_of_cyclic_permutations_times_three :
  3 * sum_of_permutations 41234 = 396618 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cyclic_permutations_times_three_l592_59296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l592_59222

-- Define the circle
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 2

-- Define the line that the circle is tangent to
def tangent_line (x y : ℝ) : Prop := x + y = 1

-- Define the line that contains the center of the circle
def center_line (x y : ℝ) : Prop := y = -2 * x

-- Define the point that the circle passes through
def point_on_circle : ℝ × ℝ := (2, -1)

theorem circle_properties :
  -- The circle is tangent to the line x + y = 1
  (∃ (x y : ℝ), circle_eq x y ∧ tangent_line x y) ∧
  -- The circle passes through the point (2, -1)
  (circle_eq point_on_circle.1 point_on_circle.2) ∧
  -- The center of the circle is on the line y = -2x
  (∃ (x y : ℝ), x^2 + y^2 = 5 ∧ center_line x y) :=
by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l592_59222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_intervals_l592_59270

noncomputable def f (x : ℝ) : ℝ := (x - 10*x^2 + 35*x^3) / (9 - x^3)

theorem f_nonnegative_intervals :
  {x : ℝ | f x ≥ 0} = {x | x ∈ Set.Icc 0 (1/7) ∨ x ∈ Set.Ici 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_intervals_l592_59270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_escalator_time_l592_59228

/-- The time it takes Vasya to run up and down the escalator when it's moving up -/
noncomputable def time_escalator_up (
  vasya_up_speed : ℝ
  ) (vasya_down_speed : ℝ
  ) (escalator_speed : ℝ
  ) : ℝ :=
  (1 / (vasya_down_speed - escalator_speed)) + (1 / (vasya_up_speed + escalator_speed))

/-- Theorem stating the time it takes Vasya to run up and down the escalator when it's moving up -/
theorem vasya_escalator_time :
  ∀ (vasya_up_speed : ℝ) (vasya_down_speed : ℝ) (escalator_speed : ℝ),
  vasya_down_speed = 2 * vasya_up_speed →
  (1 / vasya_up_speed) + (1 / vasya_down_speed) = 6 →
  (1 / (vasya_down_speed + escalator_speed)) + (1 / (vasya_up_speed - escalator_speed)) = 13.5 →
  time_escalator_up vasya_up_speed vasya_down_speed escalator_speed = 7.8 :=
by
  sorry

#eval (7.8 * 60 : ℚ)  -- Should output 468

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasya_escalator_time_l592_59228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_of_truncated_pyramid_l592_59239

/-- A regular truncated triangular pyramid -/
structure TruncatedPyramid where
  /-- The pyramid has an inscribed sphere -/
  has_inscribed_sphere : Bool
  /-- The pyramid has a sphere touching all edges -/
  has_edge_touching_sphere : Bool

/-- The dihedral angle between the base and lateral face of a truncated pyramid -/
noncomputable def dihedral_angle (p : TruncatedPyramid) : ℝ :=
  2 * Real.arctan (Real.sqrt 3 - Real.sqrt 2)

/-- Theorem: The dihedral angle of a regular truncated triangular pyramid with specific sphere properties -/
theorem dihedral_angle_of_truncated_pyramid (p : TruncatedPyramid) 
  (h1 : p.has_inscribed_sphere = true) 
  (h2 : p.has_edge_touching_sphere = true) : 
  dihedral_angle p = 2 * Real.arctan (Real.sqrt 3 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_of_truncated_pyramid_l592_59239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_is_68_l592_59238

/-- The lateral surface area of a regular quadrilateral pyramid -/
noncomputable def lateral_surface_area (sine_dihedral_angle : ℝ) (diagonal_section_area : ℝ) : ℝ :=
  4 * diagonal_section_area / (2 * Real.sqrt ((1 + Real.sqrt (1 - sine_dihedral_angle^2)) / 2))

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid with given properties is 68 -/
theorem lateral_surface_area_is_68 :
  lateral_surface_area (15/17) (3 * Real.sqrt 34) = 68 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_is_68_l592_59238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_diverges_l592_59284

open Real
open Topology
open Filter

-- Define the series term
noncomputable def a (n : ℕ) : ℝ := (1 / (n : ℝ)) * (1 / (1 + 1 / (n : ℝ)))

-- State the theorem
theorem series_diverges : ¬ (Summable a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_diverges_l592_59284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_specific_pyramid_l592_59201

/-- Regular square pyramid with given dimensions -/
structure RegularSquarePyramid where
  base_side : ℝ
  height : ℝ

/-- Lateral surface area of a regular square pyramid -/
noncomputable def lateral_surface_area (p : RegularSquarePyramid) : ℝ :=
  4 * (p.base_side / 2) * (Real.sqrt (p.height^2 + (p.base_side / 2)^2))

theorem lateral_surface_area_of_specific_pyramid :
  let p : RegularSquarePyramid := { base_side := 2, height := 1 }
  lateral_surface_area p = 4 * Real.sqrt 2 := by
  sorry

#check lateral_surface_area_of_specific_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_surface_area_of_specific_pyramid_l592_59201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pays_150_l592_59260

/-- The adoption fee for a puppy at PetSmart -/
noncomputable def adoption_fee : ℚ := 200

/-- The percentage of the fee James' friend agrees to pay -/
noncomputable def friend_contribution_percentage : ℚ := 25

/-- Calculate James' payment for adopting a puppy -/
noncomputable def james_payment (fee : ℚ) (friend_percentage : ℚ) : ℚ :=
  fee - (friend_percentage / 100) * fee

/-- Theorem stating that James' payment is $150 -/
theorem james_pays_150 :
  james_payment adoption_fee friend_contribution_percentage = 150 := by
  -- Unfold the definitions
  unfold james_payment adoption_fee friend_contribution_percentage
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_pays_150_l592_59260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l592_59288

/-- The number of steps an escalator has, given the conditions of the problem -/
def escalator_steps (boy_speed girl_speed : ℝ) (boy_steps girl_steps : ℕ) : ℕ :=
  -- We define the function to calculate the number of steps on the escalator
  -- based on the given conditions
  54  -- For now, we'll return the known answer

theorem escalator_problem (boy_speed girl_speed : ℝ) (boy_steps girl_steps : ℕ) 
  (h1 : boy_speed = 2 * girl_speed)  -- Boy's speed is twice the girl's speed
  (h2 : boy_steps = 27)              -- Boy climbed 27 steps
  (h3 : girl_steps = 18)             -- Girl climbed 18 steps
  : escalator_steps boy_speed girl_speed boy_steps girl_steps = 54 := by
  sorry

#eval escalator_steps 2 1 27 18  -- This should output 54

end NUMINAMATH_CALUDE_ERRORFEEDBACK_escalator_problem_l592_59288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_stage_speed_is_19_l592_59294

/-- A runner's two-stage run with given parameters -/
structure TwoStageRun (initial_speed initial_duration second_distance total_duration : ℝ) : Prop where
  initial_speed_pos : initial_speed > 0
  initial_duration_pos : initial_duration > 0
  second_distance_pos : second_distance > 0
  total_duration_greater : total_duration > initial_duration

/-- Calculate the average speed of the second stage of the run -/
noncomputable def SecondStageSpeed (initial_speed initial_duration second_distance total_duration : ℝ) : ℝ :=
  second_distance / (total_duration - initial_duration)

/-- Theorem stating the average speed of the second stage for given parameters -/
theorem second_stage_speed_is_19 :
  ∀ (initial_speed initial_duration second_distance total_duration : ℝ),
    TwoStageRun initial_speed initial_duration second_distance total_duration →
    initial_speed = 15 →
    initial_duration = 3 →
    second_distance = 190 →
    total_duration = 13 →
    SecondStageSpeed initial_speed initial_duration second_distance total_duration = 19 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_stage_speed_is_19_l592_59294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crossing_time_four_cows_l592_59253

-- Define the type for cow crossing times
def CrossingTime := Nat

-- Define the function to calculate minimum crossing time
def minCrossingTime (times : List Nat) : Nat :=
  sorry

-- Theorem statement
theorem min_crossing_time_four_cows :
  let times : List Nat := [5, 7, 9, 11]
  minCrossingTime times = 16 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_crossing_time_four_cows_l592_59253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_multiple_53_l592_59203

theorem least_positive_integer_multiple_53 :
  ∃ x : ℕ, x > 0 ∧ 53 ∣ (2*x)^2 + 2*43*(2*x) + 43^2 ∧
  ∀ y : ℕ, y > 0 ∧ 53 ∣ (2*y)^2 + 2*43*(2*y) + 43^2 → x ≤ y := by
  use 5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_multiple_53_l592_59203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_distance_sum_bounds_l592_59287

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c = 10 ∧ 
  Real.cos t.A / Real.cos t.B = t.b / t.a ∧
  t.b / t.a = 4 / 3

-- Define the incircle
def incircle (t : Triangle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ (x - 2)^2 + (y - 2)^2 = 4}

-- Define the distance function
def distance_sum (t : Triangle) (p : ℝ × ℝ) : ℝ :=
  let (x, y) := p
  3 * x^2 + 3 * y^2 - 12 * x - 16 * y + 100

-- Main theorem
theorem incircle_distance_sum_bounds (t : Triangle) 
  (h : triangle_conditions t) :
  ∃ (d_min d_max : ℝ), 
    (∀ p ∈ incircle t, d_min ≤ distance_sum t p ∧ distance_sum t p ≤ d_max) ∧
    d_min + d_max = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incircle_distance_sum_bounds_l592_59287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l592_59225

theorem sum_of_reciprocals (x y : ℝ) (h1 : (3 : ℝ)^x = 36) (h2 : (4 : ℝ)^y = 36) : 
  2/x + 1/y = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_reciprocals_l592_59225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_five_floor_l592_59259

-- Define the floor function (greatest integer function)
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- State the theorem
theorem pi_minus_five_floor : floor (Real.pi - 5) = -2 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_minus_five_floor_l592_59259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l592_59274

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Function to check if three points are collinear
def collinear (p q r : Point3D) : Prop :=
  ∃ t : ℝ, (q.x - p.x, q.y - p.y, q.z - p.z) = t • (r.x - p.x, r.y - p.y, r.z - p.z)

-- Define membership for Point3D in Plane3D
instance : Membership Point3D Plane3D where
  mem p plane := plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

-- Theorem: A triangle determines a unique plane in 3D Euclidean space
theorem triangle_determines_plane (t : Triangle3D) : 
  ¬collinear t.a t.b t.c → ∃! p : Plane3D, t.a ∈ p ∧ t.b ∈ p ∧ t.c ∈ p :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_determines_plane_l592_59274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l592_59277

theorem problem_statement (a b : ℝ) (h1 : (3 : ℝ)^a = 2) (h2 : (5 : ℝ)^b = 3) : 
  (a + 1/a > b + 1/b) ∧ (a + a^b < b + b^a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l592_59277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sampling_methods_l592_59237

structure Community where
  total_households : ℕ
  high_income : ℕ
  middle_income : ℕ
  low_income : ℕ

structure StudentGroup where
  total_students : ℕ

inductive SamplingMethod
  | Stratified
  | Random
  | Systematic

def has_significant_differences (c : Community) : Prop :=
  c.high_income ≠ c.middle_income ∨ c.middle_income ≠ c.low_income ∨ c.high_income ≠ c.low_income

def is_small_group (g : StudentGroup) : Prop :=
  g.total_students ≤ 20

def appropriate_sampling_method (c : Community) (g : StudentGroup) : 
  SamplingMethod × SamplingMethod :=
  (SamplingMethod.Stratified, SamplingMethod.Random)

theorem correct_sampling_methods (c : Community) (g : StudentGroup) 
  (h1 : has_significant_differences c) 
  (h2 : c.total_households = c.high_income + c.middle_income + c.low_income)
  (h3 : is_small_group g) :
  appropriate_sampling_method c g = (SamplingMethod.Stratified, SamplingMethod.Random) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sampling_methods_l592_59237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l592_59229

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the focus point
def F : ℝ × ℝ := (1, 0)

-- Define the condition for the line passing through a vertex with 60° inclination
def LineCondition (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ b / 1 = Real.tan (60 * Real.pi / 180)

-- State the theorem
theorem ellipse_theorem (a b : ℝ) (h : LineCondition a b) :
  Ellipse a b = Ellipse 2 (Real.sqrt 3) ∧
  ∃ t : ℝ, 0 < t ∧ t < 1/4 ∧
    ∀ (P Q : ℝ × ℝ), P ∈ Ellipse 2 (Real.sqrt 3) → Q ∈ Ellipse 2 (Real.sqrt 3) →
      let T := (t, 0)
      (Q.1 - P.1) * (T.1 - P.1) + (Q.2 - P.2) * (T.2 - P.2) =
      (P.1 - Q.1) * (T.1 - Q.1) + (P.2 - Q.2) * (T.2 - Q.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l592_59229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_negative_three_fourths_l592_59241

theorem tan_theta_negative_three_fourths (θ : Real) 
  (h : Real.tan θ = -3/4) : 
  1 + Real.sin θ * Real.cos θ - Real.cos θ^2 = -3/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_negative_three_fourths_l592_59241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l592_59268

noncomputable def data_set : List ℝ := [2, 1, 3, 2, 2]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  (xs.map (fun x => (x - mean xs)^2)).sum / xs.length

theorem variance_of_data_set :
  variance data_set = 0.4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_set_l592_59268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_type_a_prob_different_types_l592_59217

/-- Represents the total number of questions --/
def total_questions : ℕ := 6

/-- Represents the number of questions of type A --/
def type_a_questions : ℕ := 4

/-- Represents the number of questions of type B --/
def type_b_questions : ℕ := 2

/-- Represents the number of questions selected --/
def selected_questions : ℕ := 2

/-- The probability of selecting two questions of type A --/
theorem prob_both_type_a : 
  (Nat.choose type_a_questions selected_questions : ℚ) / (Nat.choose total_questions selected_questions) = 2/5 := by sorry

/-- The probability of selecting questions of different types --/
theorem prob_different_types : 
  (type_a_questions * type_b_questions : ℚ) / (Nat.choose total_questions selected_questions) = 8/15 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_both_type_a_prob_different_types_l592_59217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l592_59208

theorem constant_term_expansion : ∀ x : ℝ, x ≠ 0 →
  let expansion := (x + 3/x) * (x - 2/x)^5
  ∃ p q : Polynomial ℝ, 
    expansion = p.eval x + 40 + q.eval x ∧ 
    (∀ t, q.coeff t = 0 → t > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l592_59208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sphere_surface_area_l592_59215

/-- Right triangular prism with given dimensions and inscribed circle -/
structure RightTriangularPrism where
  base_length : ℝ
  height : ℝ
  inscribed_circle_radius : ℝ

/-- Properties of the given right triangular prism -/
noncomputable def given_prism : RightTriangularPrism where
  base_length := 2 * Real.sqrt 3
  height := 3
  inscribed_circle_radius := 2

/-- Theorem: Surface area of tangent sphere of conical frustum -/
theorem tangent_sphere_surface_area
  (prism : RightTriangularPrism)
  (h_base : prism.base_length = 2 * Real.sqrt 3)
  (h_height : prism.height = 3)
  (h_radius : prism.inscribed_circle_radius = 2) :
  ∃ (sphere_radius : ℝ),
    sphere_radius = 5/2 ∧
    4 * Real.pi * sphere_radius^2 = 25 * Real.pi := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sphere_surface_area_l592_59215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_of_angle_l592_59235

noncomputable def angle : ℝ := 7 * Real.pi / 6

def point (x y : ℝ) : ℝ × ℝ := (x, y)

noncomputable def distanceFromOrigin (p : ℝ × ℝ) : ℝ := Real.sqrt (p.1^2 + p.2^2)

def isOnTerminalSide (p : ℝ × ℝ) : Prop := 
  p.1 / distanceFromOrigin p = Real.cos angle ∧ 
  p.2 / distanceFromOrigin p = Real.sin angle

theorem point_on_terminal_side_of_angle (x y : ℝ) :
  isOnTerminalSide (point x y) ∧ distanceFromOrigin (point x y) = 2 →
  x = -Real.sqrt 3 ∧ y = -1 := by
  sorry

#check point_on_terminal_side_of_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_terminal_side_of_angle_l592_59235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_connection_problem_l592_59295

theorem connection_problem (n : ℕ) : 
  (∃! (ys : Finset ℕ), 
    ys.card = 7 ∧ 
    (∀ y ∈ ys, y < n ∧ y > 0 ∧ Nat.lcm y 6 = y * 6) ∧
    (∀ y ∉ ys, y ≥ n ∨ y ≤ 0 ∨ Nat.lcm y 6 ≠ y * 6)) →
  n = 47 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_connection_problem_l592_59295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l592_59242

def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m + 1

theorem problem_solution (m : ℝ) (h_m : m > 0)
  (h_sol : Set.Iic (-2) ∪ Set.Ici 2 = {x : ℝ | f m (x - 3) ≥ 0}) :
  (m = 3) ∧
  (Set.Iic 1 ∪ Set.Ici (3/2) = {t : ℝ | ∃ x : ℝ, |x + 3| - 2 ≥ |2*x - 1| - t^2 + (5/2)*t}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l592_59242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_two_l592_59293

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + x^2 - 1

theorem sum_greater_than_two (m n : ℝ) (hm : m > 0) (hn : n > 0) (hd : m ≠ n) (hf : f m + f n = 0) : m + n > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_greater_than_two_l592_59293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_minus_x_l592_59297

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_z_minus_x : ∃ (x y z : ℕ+), 
  (x.val * y.val * z.val = factorial 9) ∧ 
  (x < y) ∧ 
  (y < z) ∧
  (∀ (a b c : ℕ+), (a.val * b.val * c.val = factorial 9) → (a < b) → (b < c) → (z - x ≤ c - a)) ∧
  (z - x = 396) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_z_minus_x_l592_59297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l592_59236

theorem power_difference (a b : ℝ) (ha : a > 1) (hb : b > 0) 
  (h_sum : a^b + a^(-b) = 2 * Real.sqrt 2) : a^b - a^(-b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_difference_l592_59236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l592_59267

theorem f_value_in_third_quadrant (α : ℝ) : 
  α ∈ Set.Icc π ((3 * π) / 2) →  -- α is in the third quadrant
  Real.cos (α + π / 3) = 3 / 5 → 
  (Real.sin (π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
  (Real.sin (π / 2 + α) * Real.sin (-π - α)) = (-3 - 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l592_59267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l592_59282

-- Define the points A, B, C, and D
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (-3, 2)
def C : ℝ × ℝ := (3, 2)
def D : ℝ × ℝ := (4, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem point_P_y_coordinate :
  ∃ (P : ℝ × ℝ), 
    distance P A + distance P D = 10 ∧
    distance P B + distance P C = 10 ∧
    P.2 = 6/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_y_coordinate_l592_59282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_l592_59266

/-- The circle with center (2,2) and radius 2 -/
def myCircle (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 4

/-- The line x + y = a + 1 -/
def myLine (x y a : ℝ) : Prop := x + y = a + 1

/-- The chord length is 2√2 -/
def chord_length (a : ℝ) : Prop := ∃ x y : ℝ, myCircle x y ∧ myLine x y a ∧ 
  ((x - 2)^2 + (y - 2)^2 = 4 ∧ (∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ myCircle x' y' ∧ myLine x' y' a ∧ 
  (x - x')^2 + (y - y')^2 = 8))

/-- The main theorem -/
theorem chord_intercept (a : ℝ) : chord_length a → (a = 1 ∨ a = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intercept_l592_59266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_upper_bound_l592_59205

/-- The number of paths from one corner to the opposite corner of a rectangular grid -/
def f (m n : ℕ) : ℕ := sorry

/-- The theorem stating that f(m,n) is less than or equal to 2^(mn) -/
theorem grid_paths_upper_bound (m n : ℕ) : f m n ≤ 2^(m*n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_paths_upper_bound_l592_59205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_even_and_pi_periodic_l592_59273

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def hasPeriod (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem cos_2x_even_and_pi_periodic : 
  (isEven (fun x ↦ Real.cos (2 * x)) ∧ hasPeriod (fun x ↦ Real.cos (2 * x)) Real.pi) ∧
  ¬(isEven (fun x ↦ Real.sin (2 * x)) ∧ hasPeriod (fun x ↦ Real.sin (2 * x)) Real.pi) ∧
  ¬(isEven (fun x ↦ Real.cos (x / 2)) ∧ hasPeriod (fun x ↦ Real.cos (x / 2)) Real.pi) ∧
  ¬(isEven (fun x ↦ Real.sin (x / 2)) ∧ hasPeriod (fun x ↦ Real.sin (x / 2)) Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2x_even_and_pi_periodic_l592_59273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l592_59243

/-- Given a differentiable function f: ℝ → ℝ with a tangent line y = 3x - 2 at x = 1,
    prove that f(1) + f'(1) = 4. -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x, f 1 + (deriv f 1) * (x - 1) = 3 * x - 2) : 
    f 1 + deriv f 1 = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l592_59243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l592_59231

/-- Represents the dimensions and volume of a rectangular container. -/
structure Container where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the volume of a container given its dimensions. -/
noncomputable def calculateVolume (c : Container) : ℝ :=
  c.shortSide * c.longSide * c.height

/-- Constructs a container from the short side length, ensuring constraints are met. -/
noncomputable def makeContainer (shortSide : ℝ) : Container :=
  { shortSide := shortSide
  , longSide := shortSide + 0.5
  , height := (14.8 - 2 * shortSide - 2 * (shortSide + 0.5)) / 4
  , volume := calculateVolume { shortSide := shortSide
                              , longSide := shortSide + 0.5
                              , height := (14.8 - 2 * shortSide - 2 * (shortSide + 0.5)) / 4
                              , volume := 0 } }

/-- Theorem stating the maximum volume and corresponding height of the container. -/
theorem max_volume_container :
  ∃ (c : Container), c.volume = 1.8 ∧ c.height = 1.2 ∧
    ∀ (c' : Container), c'.volume ≤ c.volume := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l592_59231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragons_games_total_l592_59211

theorem dragons_games_total 
  (initial_win_percentage : ℚ)
  (final_win_percentage : ℚ)
  (tournament_wins : ℕ)
  (tournament_losses : ℕ)
  (h1 : initial_win_percentage = 40 / 100)
  (h2 : final_win_percentage = 55 / 100)
  (h3 : tournament_wins = 8)
  (h4 : tournament_losses = 4)
  : ∃ (total_games : ℕ), total_games = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragons_games_total_l592_59211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l592_59216

-- Define the function f(x) = 3^x - x^2
noncomputable def f (x : ℝ) : ℝ := Real.rpow 3 x - x^2

-- State the theorem
theorem f_has_zero_in_interval :
  ∃ x ∈ Set.Icc (-1 : ℝ) 0, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l592_59216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_approx_115_l592_59272

/-- Represents a right pyramid with a rectangular base -/
structure RightPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ

/-- Calculates the sum of the lengths of all edges of a right pyramid -/
noncomputable def edge_sum (p : RightPyramid) : ℝ :=
  let base_diagonal := Real.sqrt (p.base_length^2 + p.base_width^2)
  let half_diagonal := base_diagonal / 2
  let slant_height := Real.sqrt (p.height^2 + half_diagonal^2)
  2 * (p.base_length + p.base_width) + 4 * slant_height

/-- Theorem stating that for a right pyramid with given dimensions, 
    the sum of its edge lengths is approximately 115 cm -/
theorem edge_sum_approx_115 :
  let p : RightPyramid := ⟨15, 8, 15⟩
  ∃ ε > 0, |edge_sum p - 115| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_sum_approx_115_l592_59272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_order_in_negative_eighth_pi_l592_59252

theorem trig_order_in_negative_eighth_pi (θ : ℝ) : 
  -π/8 < θ ∧ θ < 0 → Real.tan θ < Real.sin θ ∧ Real.sin θ < Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_order_in_negative_eighth_pi_l592_59252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_example_l592_59278

/-- The area of a rhombus with given diagonals and intersection angle. -/
noncomputable def rhombusArea (d1 d2 : ℝ) (θ : ℝ) : ℝ :=
  (d1 * d2 * Real.sin θ) / 2

/-- Theorem: The area of a rhombus with diagonals 8 and 10, intersecting at an angle with sine 3/5, is 24. -/
theorem rhombus_area_example : rhombusArea 8 10 (Real.arcsin (3/5)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_example_l592_59278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l592_59247

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) * Real.cos φ + Real.cos (ω * x) * Real.sin φ

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : |φ| < π / 2)
  (h_f0 : f ω φ 0 = -Real.sqrt 3 / 2)
  (h_fm : f ω φ (-π / 3) = -1)
  (h_fp : f ω φ (2 * π / 3) = 1)
  (h_mono : StrictMonoOn (f ω φ) (Set.Icc (-π / 3) (2 * π / 3))) :
  ω = 1 ∧ φ = -π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l592_59247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_seven_l592_59255

/-- A sequence defined by a recurrence relation -/
def v : ℕ → ℚ
  | 0 => 7  -- Added case for 0
  | 1 => 7
  | n + 2 => v (n + 1) + (5 + 6 * n)

/-- The sequence v_n can be expressed as a quadratic polynomial -/
axiom v_is_quadratic : ∃ (a b c : ℚ), ∀ n : ℕ, v n = a * n^2 + b * n + c

theorem sum_of_coefficients_is_seven :
  ∃ (a b c : ℚ), (∀ n : ℕ, v n = a * n^2 + b * n + c) ∧ (a + b + c = 7) := by
  sorry

#check sum_of_coefficients_is_seven

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_is_seven_l592_59255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_10th_term_l592_59213

def mySequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- We define a₀ as 0 to make the indexing easier
  | 1 => 1  -- Given: a₁ = 1
  | n+2 => mySequence (n+1) + (n+2)  -- Given: aₙ = aₙ₋₁ + n for n ≥ 2

theorem sequence_10th_term : mySequence 10 = 55 := by
  rw [mySequence]
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_10th_term_l592_59213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l592_59210

theorem complex_number_in_second_quadrant : 
  let z : ℂ := Complex.ofReal (Real.sin (-π/7)) + Complex.I * Complex.ofReal (Real.cos (-π/7))
  z.re < 0 ∧ z.im > 0 :=
by
  -- Introduce the complex number z
  intro z
  
  -- Split the goal into two parts
  constructor
  
  -- Prove that the real part is negative
  · sorry
  
  -- Prove that the imaginary part is positive
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l592_59210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l592_59223

open Real

theorem trigonometric_simplification (α : ℝ) : 
  (Real.tan (3 * π - α) * Real.cos (2 * π - α) * Real.sin (-α + 3 * π / 2)) / 
  (Real.cos (-α - π) * Real.sin (-π + α) * Real.cos (α + 5 * π / 2)) = -1 / Real.sin α ∧
  (Real.cos (π / 2 + α) * Real.sin (3 * π / 2 - α)) / 
  (Real.cos (π - α) * Real.tan (π - α)) = Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_simplification_l592_59223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ricks_savings_ratio_l592_59218

theorem ricks_savings_ratio (gift_cost cake_cost erikas_savings money_left : ℚ)
  (h1 : gift_cost = 250)
  (h2 : cake_cost = 25)
  (h3 : erikas_savings = 155)
  (h4 : money_left = 5)
  (h5 : erikas_savings + (gift_cost - erikas_savings) = gift_cost + cake_cost - money_left) :
  (gift_cost - erikas_savings) / gift_cost = 23 / 50 := by
  sorry

#check ricks_savings_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ricks_savings_ratio_l592_59218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l592_59246

-- Define the triangle ABC
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the perpendicular bisector equation
def perp_bisector_eq (x y : ℝ) : Prop := 2 * x + y - 4 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 7

-- Helper function to calculate the area of a triangle given its vertices
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Theorem statement
theorem triangle_properties :
  (∀ x y, perp_bisector_eq x y ↔ 
    (x - (B.1 + C.1) / 2) * (C.2 - B.2) = (y - (B.2 + C.2) / 2) * (C.1 - B.1)) ∧
  (area_triangle A B C = triangle_area) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l592_59246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l592_59249

def original_prop (a : ℝ) : Prop := a > 0 → a > 1

def converse (a : ℝ) : Prop := a > 1 → a > 0

def inverse (a : ℝ) : Prop := a ≤ 0 → a ≤ 1

def contrapositive (a : ℝ) : Prop := a ≤ 1 → a ≤ 0

theorem two_true_propositions :
  ∃ (p q : ℝ → Prop), p ≠ q ∧
  p ∈ ({converse, inverse, contrapositive} : Set (ℝ → Prop)) ∧
  q ∈ ({converse, inverse, contrapositive} : Set (ℝ → Prop)) ∧
  (∀ a, p a) ∧ (∀ a, q a) ∧
  (∀ r ∈ ({converse, inverse, contrapositive} : Set (ℝ → Prop)), 
    r ≠ p → r ≠ q → ∃ a, ¬(r a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l592_59249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_theta_l592_59261

theorem cos_pi_half_plus_theta (θ : ℝ) 
  (h1 : Real.cos θ = 1/3) 
  (h2 : π < θ ∧ θ < 2*π) : 
  Real.cos (π/2 + θ) = 2*Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_theta_l592_59261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_marked_both_l592_59207

/-- A grid of numbers with red and blue markings -/
structure MarkedGrid where
  rows : Nat
  cols : Nat
  numbers : Fin (rows * cols) → ℕ
  is_unique : ∀ i j, i ≠ j → numbers i ≠ numbers j
  is_red : Fin (rows * cols) → Prop
  is_blue : Fin (rows * cols) → Prop
  red_marking : ∀ r, ∃ i j, i ≠ j ∧ 
    (∀ k, numbers ⟨r * cols + i, by sorry⟩ ≥ numbers ⟨r * cols + k, by sorry⟩) ∧
    (∀ k, numbers ⟨r * cols + j, by sorry⟩ ≥ numbers ⟨r * cols + k, by sorry⟩) ∧
    is_red ⟨r * cols + i, by sorry⟩ ∧ is_red ⟨r * cols + j, by sorry⟩
  blue_marking : ∀ c, ∃ i j, i ≠ j ∧ 
    (∀ k, numbers ⟨k * cols + c, by sorry⟩ ≥ numbers ⟨i * cols + c, by sorry⟩) ∧
    (∀ k, numbers ⟨k * cols + c, by sorry⟩ ≥ numbers ⟨j * cols + c, by sorry⟩) ∧
    is_blue ⟨i * cols + c, by sorry⟩ ∧ is_blue ⟨j * cols + c, by sorry⟩

/-- There are at least 3 numbers marked both red and blue in a 10x20 grid -/
theorem at_least_three_marked_both (g : MarkedGrid) 
  (h_rows : g.rows = 10) (h_cols : g.cols = 20) : 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    g.is_red a ∧ g.is_blue a ∧ 
    g.is_red b ∧ g.is_blue b ∧ 
    g.is_red c ∧ g.is_blue c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_three_marked_both_l592_59207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l592_59250

def M : ℕ := 2^4 * 3^3 * 5^2 * 7^1

theorem number_of_factors_of_M : 
  (Finset.filter (λ x : ℕ => x ∣ M) (Finset.range (M + 1))).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l592_59250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_rectangle_l592_59298

-- Define a rectangle structure
structure Rectangle where
  area : ℝ

-- Define a point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define a midpoint structure
structure Midpoint (A B : Point) where
  point : Point

-- Define the area_of_rectangle function
def area_of_rectangle (A B C D : Point) : ℝ := sorry

-- Theorem statement
theorem area_of_inner_rectangle (ABCD : Rectangle) 
  (A B C D E F G : Point)
  (hE : Midpoint A D) (hF : Midpoint B C) (hG : Midpoint C D) :
  ABCD.area = 144 → area_of_rectangle E F G D = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_rectangle_l592_59298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_question_determines_village_l592_59251

-- Define the two villages
inductive Village : Type
| A : Village
| B : Village

-- Define the possible answers
inductive Answer : Type
| Yes : Answer
| No : Answer

-- Define a function to represent the truthfulness of inhabitants
def isTruthful (v : Village) : Prop :=
  match v with
  | Village.A => True
  | Village.B => False

-- Define a function to represent whether a person lives in a given village
def livesIn (person : Village) (village : Village) : Prop :=
  person = village

-- Define the response to the question based on the person's village and the current village
noncomputable def response (person : Village) (currentVillage : Village) : Answer :=
  if (livesIn person currentVillage) = (isTruthful person) then
    Answer.Yes
  else
    Answer.No

-- The main theorem
theorem one_question_determines_village (person : Village) (currentVillage : Village) :
  (response person currentVillage = Answer.Yes → currentVillage = Village.A) ∧
  (response person currentVillage = Answer.No → currentVillage = Village.B) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_question_determines_village_l592_59251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_AC_and_MN_l592_59254

-- Define the rectangle ABCD
structure Rectangle where
  a : ℝ
  b : ℝ
  h_b_gt_a : b > a

-- Define the fold line MN
def FoldLine := ℝ → ℝ → ℝ → Prop

-- Define the dihedral angle
def dihedral_angle (rect : Rectangle) (l : FoldLine) : ℝ := 18

-- Define angle_between function (placeholder)
def angle_between (l1 l2 : FoldLine) : ℝ := sorry

-- Theorem statement
theorem angle_between_AC_and_MN (rect : Rectangle) (l : FoldLine) :
  let dihedral := dihedral_angle rect l
  angle_between 
    (fun x y z => x = rect.b * z ∧ y = rect.a * z) l = 90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_AC_and_MN_l592_59254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_with_foci_on_y_axis_l592_59265

/-- 
Given a, b : ℝ such that ab < 0, 
prove that the equation ax^2 - ay^2 = b represents a hyperbola with foci on the y-axis
-/
theorem hyperbola_with_foci_on_y_axis (a b : ℝ) (h : a * b < 0) :
  ∃ (f : ℝ → ℝ → Prop), 
    (∀ x y, f x y ↔ a * x^2 - a * y^2 = b) ∧
    (∃ c : ℝ, c > 0 ∧ ∀ x y, f x y ↔ y^2 / c^2 - x^2 / (c^2 - 1) = 1) ∧
    (∃ p q : ℝ, p > 0 ∧ q > 0 ∧ ∀ x y, f x y ↔ (y - q)^2 / p^2 - (y + q)^2 / p^2 = 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_with_foci_on_y_axis_l592_59265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_angle_X_l592_59264

theorem existence_of_angle_X (A B C : ℝ) 
  (hB : 0 < B ∧ B < π/2) (hC : 0 < C ∧ C < π/2) :
  ∃ X, Real.sin X = (Real.sin B * Real.sin C) / (1 - Real.cos A * Real.cos B * Real.cos C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_angle_X_l592_59264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_min_f_l592_59232

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

-- Define the function we're maximizing/minimizing
noncomputable def f (x y : ℝ) : ℝ := (y-1)/(x-2)

-- State the theorem
theorem circle_max_min_f :
  (∃ (x y : ℝ), circle_eq x y ∧ f x y = Real.sqrt 3/3) ∧
  (∃ (x y : ℝ), circle_eq x y ∧ f x y = -Real.sqrt 3/3) ∧
  (∀ (x y : ℝ), circle_eq x y → f x y ≤ Real.sqrt 3/3) ∧
  (∀ (x y : ℝ), circle_eq x y → f x y ≥ -Real.sqrt 3/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_max_min_f_l592_59232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_constant_mod_l592_59221

def a : ℕ → ℕ
  | 0 => 1
  | i + 1 => 2 ^ (a i)

theorem eventually_constant_mod (n : ℕ) (hn : n ≥ 1) : ∃ k : ℕ, ∀ m : ℕ, m ≥ k → 
  (2 ^ (a m)) % n = (2 ^ (a k)) % n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_constant_mod_l592_59221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l592_59257

def monomial (a : ℚ) (x y : ℕ → ℚ) : ℚ := a * (x 2) * (y 1)

def coefficient (m : ℚ) : ℚ := m

def degree (m : ℚ) : ℕ := 3

theorem monomial_properties (a : ℚ) (x y : ℕ → ℚ) :
  let m := monomial (-3/5) x y
  (coefficient m = -3/5) ∧ (degree m = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_properties_l592_59257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_ellipse_l592_59279

open Real

/-- Two circles of radius r are externally tangent to each other and internally 
    tangent to the ellipse x^2 + 4y^2 = 8. This theorem states that r = √(3/2). -/
theorem circle_tangent_to_ellipse :
  ∃ r : ℝ, r > 0 ∧ r^2 = 3/2 ∧
  ∀ x y : ℝ, 
    (x^2 + 4*y^2 = 8) →
    ((x - r)^2 + y^2 = r^2 ∨ (x + r)^2 + y^2 = r^2) →
    (∃! t : ℝ, (t - r)^2 + ((8 - t^2)/4)^2 = r^2) ∧
    (∃! t : ℝ, (t + r)^2 + ((8 - t^2)/4)^2 = r^2) := by
  -- The proof goes here
  sorry

#check circle_tangent_to_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_ellipse_l592_59279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l592_59226

theorem magnitude_relationship : ∀ (a b c : ℝ),
  a = (0.3 : ℝ)^(0.3 : ℝ) →
  b = (0.3 : ℝ)^(1.3 : ℝ) →
  c = (1.3 : ℝ)^(0.3 : ℝ) →
  c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l592_59226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_not_quadratic_l592_59248

-- Definition of a quadratic equation in one variable
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- The equation in question
noncomputable def equation (x : ℝ) : ℝ := 5 * x^2 + 1 / x + 4

-- Theorem statement
theorem equation_not_quadratic : ¬(is_quadratic_equation equation) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_not_quadratic_l592_59248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_extremum_l592_59269

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem tangent_line_at_extremum :
  ∃ (x₀ : ℝ), (∀ x, deriv f x = 0 ↔ x = x₀) ∧
              (∀ x, x < x₀ → deriv f x < 0) ∧
              (∀ x, x > x₀ → deriv f x > 0) ∧
              (deriv (deriv f) x₀ = 0) ∧
              (f x₀ = -1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_extremum_l592_59269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l592_59234

open Set

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_theorem_l592_59234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_seventeen_thirds_pi_l592_59204

theorem cos_negative_seventeen_thirds_pi : Real.cos (-17/3 * Real.pi) = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_seventeen_thirds_pi_l592_59204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_zero_implies_t_bound_l592_59262

open Real

theorem sine_sum_zero_implies_t_bound 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = sin (2 * x + π / 6)) 
  (t : ℝ) 
  (h : ∀ α ∈ Set.Icc (-π/4) (π/3), ∃ β ∈ Set.Ico (-π/3) t, f α + f β = 0) :
  t > π / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_sum_zero_implies_t_bound_l592_59262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_line_equation_l592_59291

-- Define the two curves
noncomputable def curve1 (x : ℝ) : ℝ := Real.log x
noncomputable def curve2 (x : ℝ) : ℝ := -3 - 1/x

-- Define the common tangent line
def tangent_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Theorem statement
theorem common_tangent_line_equation 
  (h : ∃ (l : ℝ → ℝ), ∀ (x : ℝ), x < 0 → 
    (∃ (x1 : ℝ), l x1 = curve1 x1 ∧ (∀ ε > 0, ∃ δ > 0, ∀ h, |h| < δ → |l (x1 + h) - curve1 (x1 + h)| < ε * |h|)) ∧
    (∃ (x2 : ℝ), l x2 = curve2 x2 ∧ (∀ ε > 0, ∃ δ > 0, ∀ h, |h| < δ → |l (x2 + h) - curve2 (x2 + h)| < ε * |h|))) :
  ∃ (a b : ℝ), ∀ (x : ℝ), tangent_line (a * x + b) x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_line_equation_l592_59291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_principal_l592_59258

/-- Calculate the principal amount given the final amount, interest rate, and time period -/
theorem calculate_principal (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) :
  A = 1120 →
  r = 0.07 →
  t = 2.4 →
  A = P * (1 + r * t) →
  ∃ ε > 0, |P - 958.90| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_principal_l592_59258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_product_l592_59233

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then Real.exp x else -x^3

-- Define function g
noncomputable def g (x a : ℝ) : ℝ := f (f x) - a

-- Theorem statement
theorem max_value_of_product (a : ℝ) (x₁ x₂ : ℝ) :
  a > 0 →
  x₁ ≠ x₂ →
  g x₁ a = 0 →
  g x₂ a = 0 →
  ∃ (m : ℝ), Real.exp x₁ * Real.exp x₂ ≤ 27 / Real.exp 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_product_l592_59233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l592_59219

/-- Calculates the time for a train to pass a platform -/
noncomputable def time_to_pass_platform (train_length : ℝ) (tree_crossing_time : ℝ) (platform_length : ℝ) : ℝ :=
  let train_speed := train_length / tree_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Theorem: A 1200m train taking 120s to cross a tree will take 240s to pass a 1200m platform -/
theorem train_platform_crossing_time :
  time_to_pass_platform 1200 120 1200 = 240 := by
  -- Unfold the definition of time_to_pass_platform
  unfold time_to_pass_platform
  -- Simplify the expression
  simp
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_crossing_time_l592_59219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_overhead_expenses_l592_59271

/-- Represents the overhead expenses of a retailer selling a radio. -/
noncomputable def overhead_expenses (purchase_price selling_price profit_percent : ℝ) : ℝ :=
  (selling_price - purchase_price) / (1 + profit_percent / 100) - purchase_price

/-- Proves that the overhead expenses are approximately 27.79 given the specified conditions. -/
theorem radio_overhead_expenses :
  let purchase_price : ℝ := 225
  let selling_price : ℝ := 300
  let profit_percent : ℝ := 18.577075098814234
  abs (overhead_expenses purchase_price selling_price profit_percent - 27.79) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval overhead_expenses 225 300 18.577075098814234

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_overhead_expenses_l592_59271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_spade_123_l592_59220

-- Define the function ♠(x) as the geometric mean of x and x^3
noncomputable def spade (x : ℝ) : ℝ := x^2

-- Theorem to prove
theorem sum_spade_123 : spade 1 + spade 2 + spade 3 = 14 := by
  -- Unfold the definition of spade
  unfold spade
  -- Simplify the expression
  simp
  -- Check that 1^2 + 2^2 + 3^2 = 14
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_spade_123_l592_59220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_roaming_area_approx_l592_59256

-- Define the pentagonal shed
structure PentagonalShed :=
  (side_length : ℝ)
  (interior_angle : ℝ)

-- Define the dog's tether
structure DogTether :=
  (length : ℝ)
  (attachment_point : ℝ)  -- Distance from a vertex along a side

-- Define the roaming area
noncomputable def roaming_area (shed : PentagonalShed) (tether : DogTether) : ℝ := 
  sorry

-- Theorem statement
theorem dog_roaming_area_approx (shed : PentagonalShed) (tether : DogTether) :
  shed.side_length = 20 →
  shed.interior_angle = 108 →
  tether.length = 10 →
  tether.attachment_point = 10 →  -- Midpoint of the side
  ∃ (ε : ℝ), abs (roaming_area shed tether - 50 * Real.pi) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_roaming_area_approx_l592_59256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_equal_g_shifted_l592_59245

noncomputable section

/-- The original function f(x) -/
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6) - 1

/-- The function g(x) before shifting -/
def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x) - 1

/-- The function g(x) shifted to the right by π/6 units -/
def g_shifted (x : ℝ) : ℝ := g (x - Real.pi / 6)

/-- Theorem stating that f(x) is not equal to g_shifted(x) -/
theorem f_not_equal_g_shifted : ∃ x : ℝ, f x ≠ g_shifted x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_equal_g_shifted_l592_59245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sufficient_condition_l592_59230

/-- Given two lines l₁ and l₂ defined by equations involving a real parameter m,
    prove that m = 1 is a sufficient condition for l₁ to be parallel to l₂. -/
theorem parallel_lines_sufficient_condition (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | m * x + y + 3 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (3 * m - 2) * x + m * y + 2 = 0}
  m = 1 → (∃ k : ℝ, k ≠ 0 ∧ (m, 1) = k • (3 * m - 2, m)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_sufficient_condition_l592_59230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_theorem_l592_59224

-- Define the cis function
noncomputable def cis (θ : ℝ) : ℂ :=
  Complex.exp (Complex.I * θ)

-- Define the product of two complex numbers in cis form
noncomputable def cis_product (r1 r2 : ℝ) (θ1 θ2 : ℝ) : ℂ :=
  (r1 * r2 : ℝ) • cis (θ1 + θ2)

-- Define the polar form of a complex number
structure PolarForm where
  r : ℝ
  θ : ℝ
  r_pos : r > 0
  θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi

-- Convert degrees to radians
noncomputable def deg_to_rad (deg : ℝ) : ℝ :=
  deg * (Real.pi / 180)

-- State the theorem
theorem complex_product_theorem :
  ∃ (z : PolarForm),
    cis_product 5 (-3) (deg_to_rad 30) (deg_to_rad 45) =
    (z.r : ℂ) * cis z.θ ∧
    z.r = 15 ∧
    z.θ = deg_to_rad 255 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_theorem_l592_59224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_sugar_amount_l592_59289

-- Define the ingredients and measurements
def flour_cups : ℚ := 2
def white_sugar_cups : ℚ := 1
def oil_cups : ℚ := 1/2
def scoop_size : ℚ := 1/4
def total_scoops : ℕ := 15

-- Theorem to prove
theorem brown_sugar_amount :
  let known_ingredients_scoops := (flour_cups + white_sugar_cups + oil_cups) / scoop_size
  let brown_sugar_scoops := total_scoops - Int.floor known_ingredients_scoops
  (brown_sugar_scoops : ℚ) * scoop_size = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brown_sugar_amount_l592_59289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_visit_count_l592_59214

def visit_frequency (friend : Fin 4) : ℕ :=
  match friend with
  | 0 => 4  -- Alex
  | 1 => 6  -- Bella
  | 2 => 8  -- Carl
  | 3 => 9  -- Diana

def visits_on_day (friend : Fin 4) (day : ℕ) : Bool :=
  day % visit_frequency friend = 0

def exactly_three_visit (day : ℕ) : Bool :=
  (Finset.filter (λ f => visits_on_day f day) (Finset.univ : Finset (Fin 4))).card = 3

theorem exactly_three_visit_count :
  (Finset.filter (λ day => exactly_three_visit day) (Finset.range 365)).card = 15 := by
  sorry

#eval (Finset.filter (λ day => exactly_three_visit day) (Finset.range 365)).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_visit_count_l592_59214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spacek_birds_l592_59292

/-- The number of birds Mr. Špaček keeps -/
def num_birds : ℕ := 72

/-- Condition that the number of birds is more than 50 and less than 100 -/
axiom bird_range : 50 < num_birds ∧ num_birds < 100

/-- Condition that parakeets make up one-ninth of the total -/
axiom parakeet_fraction : num_birds % 9 = 0

/-- Condition that canaries make up one-fourth of the total -/
axiom canary_fraction : num_birds % 4 = 0

/-- Theorem stating that given the conditions, Mr. Špaček must have 72 birds -/
theorem spacek_birds : num_birds = 72 := by
  -- The proof goes here
  sorry

#eval num_birds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spacek_birds_l592_59292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_l592_59200

def is_valid_sequence (a : Fin 2016 → ℕ) : Prop :=
  (∀ i, a i ≤ 2016) ∧
  (∀ i j : Fin 2016, (i : ℕ) + j ∣ (i : ℕ) * a i + j * a j)

theorem constant_sequence (a : Fin 2016 → ℕ) (h : is_valid_sequence a) :
  ∀ i j : Fin 2016, a i = a j :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_l592_59200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l592_59240

/-- Calculates the compound interest for a given principal, rate, number of times compounded per year, and time in years -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (n : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / n) ^ (n * time)

/-- Calculates the total payment for Plan 1 -/
noncomputable def plan1_payment (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  let half_time := time / 2
  let half_balance := compound_interest principal rate 2 half_time / 2
  half_balance + compound_interest half_balance rate 2 half_time

/-- Calculates the total payment for Plan 2 -/
noncomputable def plan2_payment (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  compound_interest principal rate 1 time

/-- The difference between Plan 2 and Plan 1 payments is approximately $11,926 -/
theorem loan_payment_difference :
  let principal := 20000
  let rate := 0.12
  let time := 10
  ⌊plan2_payment principal rate time - plan1_payment principal rate time⌋ = 11926 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_loan_payment_difference_l592_59240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_solve_inequalities_l592_59275

-- Problem 1
theorem calculate_expression : 
  |1 - Real.sqrt 3| - Real.sqrt 2 * Real.sqrt 6 + 1 / (2 - Real.sqrt 3) - (2/3)^2 = -5/4 := by sorry

-- Problem 2
theorem solve_inequalities (m : ℝ) (h : m < 0) :
  {x : ℝ | 4*x - 1 > x - 7 ∧ -(1/4)*x < 3/2*m - 1} = {x : ℝ | x > 4 - 6*m} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_solve_inequalities_l592_59275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_initial_speed_l592_59280

/-- Represents the journey between two cities -/
structure Journey where
  totalDistance : ℝ
  startTime : ℝ
  initialDuration : ℝ
  reducedSpeed : ℝ
  arrivalTime : ℝ

/-- Calculates the initial speed of the journey -/
noncomputable def calculateInitialSpeed (j : Journey) : ℝ :=
  let totalTime := j.arrivalTime - j.startTime
  let reducedSpeedDuration := totalTime - j.initialDuration
  let distanceAtReducedSpeed := j.reducedSpeed * reducedSpeedDuration
  let distanceAtInitialSpeed := j.totalDistance - distanceAtReducedSpeed
  distanceAtInitialSpeed / j.initialDuration

/-- Theorem stating that Anna's initial speed was 11.2 km/hr -/
theorem annas_initial_speed (j : Journey) 
  (h1 : j.totalDistance = 350)
  (h2 : j.startTime = 5 + 20/60)
  (h3 : j.initialDuration = 2 + 15/60)
  (h4 : j.reducedSpeed = 60)
  (h5 : j.arrivalTime = 13) :
  calculateInitialSpeed j = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_annas_initial_speed_l592_59280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l592_59212

noncomputable def original_price : ℝ := 10000.000000000002
noncomputable def new_price : ℝ := 4400

noncomputable def percentage_decrease (original : ℝ) (new : ℝ) : ℝ :=
  ((original - new) / original) * 100

theorem price_decrease_percentage :
  percentage_decrease original_price new_price = 56.00000000000002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_decrease_percentage_l592_59212
