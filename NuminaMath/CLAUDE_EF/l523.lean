import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_cos_3x_plus_sin_6x_l523_52336

/-- The period of y = cos 3x + sin 6x is 2π/3 -/
theorem period_cos_3x_plus_sin_6x :
  ∃ (p : ℝ), p > 0 ∧ p = 2 * Real.pi / 3 ∧
  ∀ (x : ℝ), (Real.cos (3 * x) + Real.sin (6 * x) = Real.cos (3 * (x + p)) + Real.sin (6 * (x + p))) ∧
  ∀ (q : ℝ), 0 < q ∧ q < p →
    ∃ (x : ℝ), Real.cos (3 * x) + Real.sin (6 * x) ≠ Real.cos (3 * (x + q)) + Real.sin (6 * (x + q)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_cos_3x_plus_sin_6x_l523_52336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l523_52378

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = π
  sine_law : a / (Real.sin A) = b / (Real.sin B)
  cosine_law : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

-- Define the theorem
theorem triangle_theorem (abc : Triangle) 
  (h1 : (2 * abc.a - abc.c) * Real.cos abc.B = abc.b * Real.cos abc.C)
  (h2 : 0 < abc.A ∧ abc.A < π)
  (h3 : 0 < abc.B ∧ abc.B < π) :
  abc.B = π/3 ∧ 
  (abc.a = Real.sqrt 3 ∧ abc.c = Real.sqrt 3 → abc.b = Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l523_52378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_l523_52346

theorem mixture_weight (initial_water_percentage final_water_percentage added_water initial_weight : ℝ) : 
  initial_water_percentage = 0.10 →
  final_water_percentage = 0.25 →
  added_water = 4 →
  (final_water_percentage * (initial_weight + added_water) = 
   initial_water_percentage * initial_weight + added_water) →
  initial_weight = 20 := by
  sorry

#check mixture_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_weight_l523_52346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_visitors_average_l523_52342

theorem library_visitors_average (sunday_visitors other_day_visitors : ℕ) 
  (h1 : sunday_visitors = 510) (h2 : other_day_visitors = 240) : 
  (5 * sunday_visitors + 25 * other_day_visitors) / 30 = 285 := by
  have total_sundays : ℕ := 5
  have total_other_days : ℕ := 25
  have total_days : ℕ := 30
  have total_visitors : ℕ := sunday_visitors * total_sundays + other_day_visitors * total_other_days
  
  -- Replace the actual proof with sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_visitors_average_l523_52342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l523_52388

/-- Given a triangle with two sides a and b, and an angle C between them, 
    this function calculates the length of the third side using the law of cosines. -/
noncomputable def third_side (a b : ℝ) (cos_C : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 - 2*a*b*cos_C)

/-- This theorem states that for a triangle with sides 6 and 10, 
    an acute angle between them, and an area of 18, 
    the length of the third side is 2√22. -/
theorem triangle_third_side : 
  ∀ (C : ℝ), 
  0 < C ∧ C < Real.pi/2 →  -- C is acute
  (1/2) * 6 * 10 * Real.sin C = 18 →  -- Area formula
  third_side 6 10 (Real.cos C) = 2 * Real.sqrt 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_third_side_l523_52388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l523_52395

/-- The function f(x) = ln x - x + 1 -/
noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

theorem f_inequality (a b : ℝ) (ha : 0 < a) (hb : a < b) :
  (f b - f a) / (b - a) < 1 / (a * (a + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l523_52395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_streetlight_theorem_l523_52399

/-- Represents a road with streetlights -/
structure Road where
  num_lights : ℕ
  deriving Repr

/-- Defines the probability of a light burning out -/
noncomputable def burnout_prob : ℝ := sorry

/-- Defines the condition for light replacement -/
def needs_replacement (road : Road) (burned_out : Set ℕ) : Prop :=
  ∃ i, i ∈ burned_out ∧ i + 1 ∈ burned_out

/-- Calculates the probability of exactly k lights needing replacement -/
noncomputable def prob_k_lights_replaced (road : Road) (k : ℕ) : ℝ := sorry

/-- Calculates the expected number of lights needing replacement -/
noncomputable def expected_lights_replaced (road : Road) : ℝ := sorry

/-- Main theorem stating the probability and expectation for a 9-light road -/
theorem streetlight_theorem (road : Road) (h : road.num_lights = 9) :
  prob_k_lights_replaced road 4 = 25 / 84 ∧
  expected_lights_replaced road = 837 / 252 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_streetlight_theorem_l523_52399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l523_52363

/-- Represents the total amount of work to be done -/
noncomputable def W : ℝ := 1

/-- Work rate of person A -/
noncomputable def rate_A : ℝ := W / 3

/-- Work rate of person B -/
noncomputable def rate_B : ℝ := W / 6

/-- Work rate of person C -/
noncomputable def rate_C : ℝ := W / 3

/-- Time taken by A and B together to complete the work -/
noncomputable def time_AB : ℝ := W / (rate_A + rate_B)

/-- Theorem stating that A and B together can complete the work in 2 hours -/
theorem work_completion_time : time_AB = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l523_52363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_number_l523_52385

-- Define the type for button operations
inductive Button
| A
| B

-- Define the function for button operations
def applyButton (x : ℕ) (b : Button) : ℕ :=
  match b with
  | Button.A => 2 * x + 1
  | Button.B => 3 * x - 1

-- Define a function to check if a number is two-digit
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

-- Define the theorem
theorem largest_two_digit_number (start : ℕ) (h : start = 5) :
  ∃ (sequence : List Button),
    let result := sequence.foldl applyButton start
    isTwoDigit result ∧
    (∀ (otherSequence : List Button),
      let otherResult := otherSequence.foldl applyButton start
      isTwoDigit otherResult → otherResult ≤ result) ∧
    result = 95 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_two_digit_number_l523_52385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_vectors_l523_52374

/-- A vector in the plane -/
structure PlaneVector where
  x : ℝ
  y : ℝ

/-- The distance between two plane vectors -/
noncomputable def distance (v w : PlaneVector) : ℝ :=
  Real.sqrt ((v.x - w.x)^2 + (v.y - w.y)^2)

/-- A set of vectors satisfying the problem conditions -/
structure VectorSet where
  a₁ : PlaneVector
  a₂ : PlaneVector
  b : Finset PlaneVector
  nonparallel : ∀ v w, v ∈ b → w ∈ b → v ≠ w → ¬ ∃ (t : ℝ), v = PlaneVector.mk (t * w.x) (t * w.y)
  a_distance : distance a₁ a₂ = 1
  b_distances : ∀ bᵢ, bᵢ ∈ b → distance a₁ bᵢ ∈ ({1, 2, 3} : Set ℝ) ∧ distance a₂ bᵢ ∈ ({1, 2, 3} : Set ℝ)

/-- The theorem stating the maximum number of b vectors -/
theorem max_b_vectors (vs : VectorSet) : vs.b.card ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_vectors_l523_52374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l523_52365

-- Define the functions for the curves
noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.exp (-x)

-- Define the enclosed area
noncomputable def enclosed_area : ℝ := ∫ x in (Set.Icc 0 1), (f x - g x)

-- Theorem statement
theorem area_enclosed_by_curves : enclosed_area = Real.exp 1 + Real.exp (-1) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l523_52365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_when_a_is_one_range_of_a_when_intersection_nonempty_l523_52380

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- Theorem 1
theorem complement_A_union_B_when_a_is_one :
  (Set.univ : Set ℝ) \ (A ∪ B 1) = Set.Iic 0 := by sorry

-- Theorem 2
theorem range_of_a_when_intersection_nonempty :
  ∀ a : ℝ, (A ∩ B a).Nonempty → a < 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_union_B_when_a_is_one_range_of_a_when_intersection_nonempty_l523_52380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l523_52318

theorem inequality_solution (x : ℝ) : 
  1 / (x + 2) + 4 / (x + 8) ≤ 3 / 4 ↔ x ∈ Set.Ioc (-8) (-4) ∪ Set.Icc (-4) (4/3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l523_52318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_theorem_l523_52316

/-- The line equation forming a triangle with coordinate axes -/
def line_equation (x y : ℝ) : Prop := 8 * x + 10 * y = 80

/-- The triangle formed by the line and coordinate axes -/
structure Triangle where
  x_intercept : ℝ
  y_intercept : ℝ
  hypotenuse : ℝ

/-- The sum of altitudes of the triangle -/
noncomputable def sum_of_altitudes (t : Triangle) : ℝ := 
  t.y_intercept + t.x_intercept + 40 / Real.sqrt 41

/-- Theorem stating the sum of altitudes of the triangle formed by the line 8x + 10y = 80 and coordinate axes -/
theorem sum_of_altitudes_theorem (t : Triangle) 
  (h1 : line_equation t.x_intercept 0)
  (h2 : line_equation 0 t.y_intercept)
  (h3 : t.hypotenuse^2 = t.x_intercept^2 + t.y_intercept^2) :
  sum_of_altitudes t = 18 + 40 / Real.sqrt 41 := by
  sorry

#check sum_of_altitudes_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_altitudes_theorem_l523_52316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_encounter_average_time_l523_52319

/-- Car's eastward velocity in miles per minute -/
noncomputable def car_velocity : ℝ := 4/5

/-- Storm's radius in miles -/
def storm_radius : ℝ := 75

/-- Storm's velocity in miles per minute -/
noncomputable def storm_velocity : ℝ := (3/5) * Real.sqrt 2

/-- Initial north distance between car and storm center in miles -/
def initial_distance : ℝ := 150

/-- Time when car enters the storm -/
noncomputable def t1 : ℝ := sorry

/-- Time when car leaves the storm -/
noncomputable def t2 : ℝ := sorry

/-- Theorem stating that the average of entry and exit times is 225 minutes -/
theorem storm_encounter_average_time : (t1 + t2) / 2 = 225 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_encounter_average_time_l523_52319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l523_52324

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Scalar multiplication for points -/
def smul (t : ℝ) (p : Point) : Point :=
  { x := t * p.x, y := t * p.y }

instance : SMul ℝ Point where
  smul := smul

/-- Addition for points -/
def add (p1 p2 : Point) : Point :=
  { x := p1.x + p2.x, y := p1.y + p2.y }

instance : Add Point where
  add := add

/-- Theorem: The eccentricity of a specific hyperbola is 2 -/
theorem hyperbola_eccentricity_is_two (h : Hyperbola) (O F1 F2 Q : Point) :
  -- The hyperbola equation
  (∀ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 → 
    ∃ (p : Point), p.x = x ∧ p.y = y) →
  -- F1 and F2 are the foci
  distance O F1 = distance O F2 →
  -- The circle is centered at F2 with radius OF2
  distance O F2 = distance F2 Q →
  -- The tangent line passes through F1 and Q
  (∃ (t : ℝ), (1 - t) • F1 + t • Q = F2) →
  -- F1Q is bisected by the asymptote
  (∃ (M : Point), distance F1 M = distance M Q ∧
    M.y / M.x = h.b / h.a) →
  -- The eccentricity is 2
  eccentricity h = 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l523_52324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sort_with_six_rearrangements_l523_52372

/-- Represents a sequence of elements that can be partially rearranged -/
structure PartiallyRearrangeableSequence (α : Type*) [LinearOrder α] where
  elements : Vector α 100
  rearrange_window : Nat
  rearrange_count : Nat

/-- Defines a rearrangement operation on a sequence -/
def rearrange {α : Type*} [LinearOrder α] (s : PartiallyRearrangeableSequence α) (start : Nat) : PartiallyRearrangeableSequence α :=
  sorry

/-- Checks if a sequence is sorted in descending order -/
def is_sorted_desc {α : Type*} [LinearOrder α] (s : PartiallyRearrangeableSequence α) : Prop :=
  sorry

/-- Main theorem: Any sequence of 100 elements can be sorted in descending order with 6 rearrangements -/
theorem sort_with_six_rearrangements {α : Type*} [LinearOrder α] (s : PartiallyRearrangeableSequence α) :
  s.rearrange_window = 50 →
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : Nat),
    is_sorted_desc (rearrange (rearrange (rearrange (rearrange (rearrange (rearrange s r₁) r₂) r₃) r₄) r₅) r₆) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sort_with_six_rearrangements_l523_52372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_application_theorem_l523_52302

/-- Represents the number of permutations resulting in applicant i getting the job -/
def n (i : Fin 10) : ℕ :=
  sorry

/-- The total number of possible permutations -/
def total_permutations : ℕ := Nat.factorial 10

theorem job_application_theorem :
  (∀ i j : Fin 10, i < j → n i > n j) ∧
  (n 8 = n 9) ∧ (n 9 = n 10) ∧
  (↑(n 1 + n 2 + n 3) / ↑total_permutations : ℚ) > 7/10 ∧
  (↑(n 8 + n 9 + n 10) / ↑total_permutations : ℚ) ≤ 1/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_application_theorem_l523_52302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l523_52398

theorem quadratic_factorization (b : ℤ) : 
  (∃ (d e f g : ℤ), (15 : ℤ) * x^2 + b * x + 45 = (d * x + e) * (f * x + g) ∧ 
   Nat.Prime d.natAbs ∧ Nat.Prime f.natAbs ∧ 
   (∀ (x : ℚ), (15 : ℚ) * x^2 + (b : ℚ) * x + 45 = ((d : ℚ) * x + (e : ℚ)) * ((f : ℚ) * x + (g : ℚ)))) →
  (∃ (k : ℤ), b = 2 * k) ∧ 
  ¬(∀ (k : ℤ), ∃ (d e f g : ℤ), (15 : ℤ) * x^2 + (2 * k) * x + 45 = (d * x + e) * (f * x + g) ∧ 
    Nat.Prime d.natAbs ∧ Nat.Prime f.natAbs ∧ 
    (∀ (x : ℚ), (15 : ℚ) * x^2 + (2 * k : ℚ) * x + 45 = ((d : ℚ) * x + (e : ℚ)) * ((f : ℚ) * x + (g : ℚ)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_factorization_l523_52398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_plus_pi_fourth_l523_52325

theorem tan_angle_plus_pi_fourth (θ : Real) : 
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ Real.tan θ = y / x) →
  Real.tan (θ + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_angle_plus_pi_fourth_l523_52325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_cylinder_from_14cm_square_l523_52389

/-- The volume of a cylinder formed by rotating a square about its vertical line of symmetry -/
noncomputable def cylinderVolumeFromSquare (sideLength : ℝ) : ℝ :=
  Real.pi * (sideLength / 2)^2 * sideLength

/-- Theorem: The volume of the cylinder formed by rotating a square with side length 14 cm
    about its vertical line of symmetry is 686π cubic centimeters -/
theorem volume_of_cylinder_from_14cm_square :
  cylinderVolumeFromSquare 14 = 686 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_cylinder_from_14cm_square_l523_52389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_digit_arrangement_l523_52356

theorem impossible_digit_arrangement : ¬ ∃ (arrangement : Fin 9 → Fin 9),
  (∀ i : Fin 9, arrangement i ≠ arrangement (i.succ)) ∧
  (∀ i : Fin 8, ∃ k : ℕ, 
    (arrangement i : ℕ) < (arrangement i.succ : ℕ) ∧
    (2 * k + 1 = (arrangement i.succ : ℕ) - (arrangement i : ℕ) - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_digit_arrangement_l523_52356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_75_l523_52351

def y : ℕ → ℚ
  | 0 => 75  -- Add this case to cover Nat.zero
  | 1 => 75
  | (k + 2) => y (k + 1) ^ 2 - y (k + 1)

noncomputable def series_sum : ℚ := ∑' n, 1 / (y n - 1)

theorem series_sum_equals_one_over_75 : series_sum = 1 / 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_over_75_l523_52351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beijing_lighting_scientific_notation_l523_52379

noncomputable def scientific_notation (a : ℝ) (n : ℤ) : ℝ := a * (10 : ℝ) ^ n

theorem beijing_lighting_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    scientific_notation a n = 13700000 ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 1.37 ∧
    n = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beijing_lighting_scientific_notation_l523_52379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersecting_line_theorem_l523_52386

/-- The ellipse C with given properties --/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/2
  h_rhombus_area : 4 * a * b = 2 * Real.sqrt 2

/-- The line l intersecting the ellipse --/
structure IntersectingLine (C : Ellipse) where
  k : ℝ
  t : ℝ
  h_t : t ≠ 1 ∧ t ≠ -1
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P_on_C : P.1^2 / C.a^2 + P.2^2 / C.b^2 = 1
  h_Q_on_C : Q.1^2 / C.a^2 + Q.2^2 / C.b^2 = 1
  h_P_on_l : P.2 = k * P.1 + t
  h_Q_on_l : Q.2 = k * Q.1 + t
  h_distinct : P ≠ Q

/-- The theorem to be proved --/
theorem ellipse_intersecting_line_theorem (C : Ellipse) (l : IntersectingLine C) :
  (C.a = Real.sqrt 2 ∧ C.b = 1) ∧ 
  (∀ (M N : ℝ), 
    (0 - C.b) / M = (l.P.2 - C.b) / l.P.1 →
    (0 - C.b) / N = (l.Q.2 - C.b) / l.Q.1 →
    |M| * |N| = 2 →
    l.t = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersecting_line_theorem_l523_52386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_sqrt_two_l523_52376

open Real

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := exp x * (log x + (x - m)^2)

-- Define the derivative of f
noncomputable def f_prime (m : ℝ) (x : ℝ) : ℝ := exp x * (1/x + 2*(x - m))

-- State the theorem
theorem m_less_than_sqrt_two (m : ℝ) :
  (∀ x > 0, f_prime m x - f m x > 0) → m < sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_less_than_sqrt_two_l523_52376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_all_acute_triangles_l523_52337

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- An angle is acute if it's less than 90 degrees -/
def isAcute (angle : ℝ) : Prop := angle > 0 ∧ angle < Real.pi / 2

/-- A triangle is acute if all its angles are acute -/
def isAcuteTriangle (A B C : Point2D) : Prop :=
  isAcute (angle A B C) ∧ isAcute (angle B C A) ∧ isAcute (angle C A B)
where
  angle (A B C : Point2D) : ℝ := sorry  -- Placeholder for angle calculation

/-- There do not exist four points on a plane forming only acute triangles -/
theorem no_four_points_all_acute_triangles :
  ¬ ∃ (A B C D : Point2D),
    isAcuteTriangle A B C ∧
    isAcuteTriangle B C D ∧
    isAcuteTriangle C D A ∧
    isAcuteTriangle D A B := by
  sorry  -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_four_points_all_acute_triangles_l523_52337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_height_l523_52300

def heights : List ℝ := [1.50, 1.60, 1.65, 1.70, 1.75, 1.80]
def counts : List ℕ := [2, 3, 3, 2, 4, 1]

def total_jumpers : ℕ := counts.sum

theorem median_height (h : total_jumpers = 15) :
  let sorted_heights := List.join (List.zipWith (λ h c => List.replicate c h) heights counts)
  sorted_heights.get? ((sorted_heights.length - 1) / 2) = some 1.65 := by
  sorry

#eval total_jumpers

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_height_l523_52300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_of_difference_l523_52312

-- Define the properties of x, y, z, p, q, and r
def is_valid_configuration (x y z p q r : ℤ) : Prop :=
  Nat.Prime x.natAbs ∧ Nat.Prime y.natAbs ∧ Nat.Prime z.natAbs ∧
  Nat.Prime p.natAbs ∧ Nat.Prime q.natAbs ∧ Nat.Prime r.natAbs ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  x = p^2 ∧ y = q^3 ∧ z = r^5

-- State the theorem
theorem least_value_of_difference (x y z p q r : ℤ) 
  (h : is_valid_configuration x y z p q r) : 
  x - y - z ≥ -3148 ∧ ∃ x' y' z' p' q' r' : ℤ, 
    is_valid_configuration x' y' z' p' q' r' ∧ 
    x' - y' - z' = -3148 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_of_difference_l523_52312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_fill_tank_l523_52317

/-- Represents the tank and pipe system -/
structure WaterSystem where
  tankCapacity : ℕ
  pipeARate : ℕ
  pipeBRate : ℕ
  pipeCRate : ℕ

/-- Calculates the net water added in one cycle -/
def netWaterPerCycle (sys : WaterSystem) : ℕ :=
  sys.pipeARate + 2 * sys.pipeBRate - 2 * sys.pipeCRate

/-- Calculates the time for one cycle -/
def timePerCycle : ℕ := 5

/-- Calculates the number of cycles needed to fill the tank -/
def numCycles (sys : WaterSystem) : ℕ :=
  sys.tankCapacity / netWaterPerCycle sys

/-- Theorem stating the time to fill the tank -/
theorem time_to_fill_tank (sys : WaterSystem) 
  (h1 : sys.tankCapacity = 2000)
  (h2 : sys.pipeARate = 200)
  (h3 : sys.pipeBRate = 50)
  (h4 : sys.pipeCRate = 25) :
  numCycles sys * timePerCycle = 40 := by
  sorry

-- Remove the #eval statement as it's causing issues
-- #eval time_to_fill_tank ⟨2000, 200, 50, 25⟩ rfl rfl rfl rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_fill_tank_l523_52317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_seats_l523_52321

/-- The number of people who want to ride the ferris wheel -/
noncomputable def num_people : ℝ := 14.0

/-- The number of times the ferris wheel has to run for everyone to get a turn -/
noncomputable def num_runs : ℝ := 2.333333333

/-- The number of seats on the ferris wheel -/
noncomputable def num_seats : ℝ := num_people / num_runs

theorem ferris_wheel_seats :
  Int.floor num_seats = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferris_wheel_seats_l523_52321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_angle_is_40_l523_52357

/-- Regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  base_edge : ℝ
  side_edge : ℝ

/-- The angle between the side edge and the base of a regular quadrilateral pyramid -/
noncomputable def side_base_angle (p : RegularQuadPyramid) : ℝ :=
  Real.arccos (p.base_edge / (2 * p.side_edge))

/-- The angle option closest to the side-base angle -/
noncomputable def closest_angle (p : RegularQuadPyramid) : ℝ :=
  let angle := side_base_angle p
  let options := [30, 40, 50, 60]
  match options.argmin (λ x => |x - angle * 180 / Real.pi|) with
  | some x => x
  | none => 0  -- This case should never occur as the list is non-empty

theorem closest_angle_is_40 (p : RegularQuadPyramid) 
  (h1 : p.base_edge = 2017) 
  (h2 : p.side_edge = 2000) : 
  closest_angle p = 40 := by
  sorry

-- We can't use #eval with noncomputable functions, so we'll use #check instead
#check closest_angle { base_edge := 2017, side_edge := 2000 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_angle_is_40_l523_52357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_approx_l523_52394

/-- The time (in seconds) it takes for a train to pass a bridge -/
noncomputable def train_pass_bridge_time (train_length bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train of length 1200 meters, traveling at 120 km/hour,
    will take approximately 51.01 seconds to pass a bridge of length 500 meters -/
theorem train_pass_bridge_time_approx :
  ∃ ε > 0, |train_pass_bridge_time 1200 500 120 - 51.01| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_pass_bridge_time_approx_l523_52394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l523_52326

theorem problem_1 (x : ℝ) : (2 : ℝ)^(x+3) * (3 : ℝ)^(x+3) = (36 : ℝ)^(x-2) ↔ x = 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_l523_52326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equality_implies_zero_sum_l523_52392

/-- Represents an arithmetic progression. -/
structure ArithmeticProgression where
  /-- The first term of the progression. -/
  a₁ : ℚ
  /-- The common difference of the progression. -/
  d : ℚ

/-- The sum of the first k terms of an arithmetic progression. -/
def sum (ap : ArithmeticProgression) (k : ℕ) : ℚ :=
  (k : ℚ) / 2 * (2 * ap.a₁ + ap.d * ((k : ℚ) - 1))

/-- 
Theorem: In an arithmetic progression, if the sum of the first m terms
equals the sum of the first n terms (where m ≠ n), then the sum of the
first (m + n) terms is zero.
-/
theorem sum_equality_implies_zero_sum
  (ap : ArithmeticProgression) (m n : ℕ) 
  (h1 : m ≠ n) (h2 : sum ap m = sum ap n) : 
  sum ap (m + n) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equality_implies_zero_sum_l523_52392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l523_52382

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The minor axis length of an ellipse -/
def Ellipse.minorAxisLength (e : Ellipse) : ℝ := 2 * e.b

theorem ellipse_equation (e : Ellipse) 
    (h_minor : e.minorAxisLength = 2)
    (h_ecc : e.eccentricity = Real.sqrt 2 / 2) :
    ∀ x y : ℝ, e.equation x y ↔ x^2 / 2 + y^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l523_52382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l523_52360

/-- The solution set of the inequality √(3x-5a) - √(x-a) > 1 -/
def SolutionSet (a : ℝ) : Set ℝ :=
  if a < -3/4 then
    {x | x ≥ a ∧ x ≥ 5*a/3}
  else if a < -1/2 then
    {x | (a ≤ x ∧ x < 2*a + 1 - Real.sqrt (a + 3/4) ∧ x ≥ 5*a/3) ∨
         (x > 2*a + 1 + Real.sqrt (a + 3/4) ∧ x ≥ 5*a/3)}
  else
    {x | x > 2*a + 1 + Real.sqrt (a + 3/4) ∧ x ≥ 5*a/3}

/-- The theorem stating that the SolutionSet is correct -/
theorem solution_set_correct (a : ℝ) :
  ∀ x, x ∈ SolutionSet a ↔ (Real.sqrt (3*x - 5*a) - Real.sqrt (x - a) > 1 ∧ x ≥ a ∧ x ≥ 5*a/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_correct_l523_52360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l523_52308

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x * (2016 + Real.log x)

-- State the theorem
theorem x_zero_value (x₀ : ℝ) (h : x₀ > 0) :
  (deriv f x₀ = 2017) → x₀ = 1 := by
  intro h_deriv
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_zero_value_l523_52308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_tangent_line_l523_52359

noncomputable section

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Companion circle of an ellipse -/
def companion_circle (e : Ellipse) : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = e.a^2 + e.b^2}

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : ℝ × ℝ) : Prop :=
  p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1

/-- A line passing through a point on the y-axis -/
structure Line where
  m : ℝ
  h_m_pos : m > 0

/-- A point is on the line -/
def on_line (l : Line) (p : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, p.2 = k * p.1 + l.m

/-- The line is tangent to the ellipse -/
def tangent_to_ellipse (e : Ellipse) (l : Line) : Prop :=
  ∃! p : ℝ × ℝ, on_ellipse e p ∧ on_line l p

/-- Chord length intercepted by a line on a circle -/
noncomputable def chord_length (c : Set (ℝ × ℝ)) (l : Line) : ℝ :=
  2 * Real.sqrt (5 - l.m^2 / (1 + l.m^2))

theorem ellipse_equation_and_tangent_line (e : Ellipse)
    (h_ecc : eccentricity e = Real.sqrt 3 / 2)
    (h_point : on_ellipse e (0, 1))
    (l : Line)
    (h_tangent : tangent_to_ellipse e l)
    (h_chord : chord_length (companion_circle e) l = 2 * Real.sqrt 2) :
    (e.a = 2 ∧ e.b = 1) ∧ l.m = 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_and_tangent_line_l523_52359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_when_a_neg_one_f_min_value_when_a_neg_f_min_value_when_a_pos_l523_52332

noncomputable section

-- Define the function f(x) = |x + a/x|
def f (a : ℝ) (x : ℝ) : ℝ := |x + a/x|

-- Theorem 1: Monotonicity when a = -1
theorem f_monotone_increasing_when_a_neg_one :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f (-1) x₁ < f (-1) x₂ := by sorry

-- Theorem 2: Minimum value when a < 0
theorem f_min_value_when_a_neg :
  ∀ a : ℝ, a < 0 → (∀ x : ℝ, x > 0 → f a x ≥ 0) ∧ ∃ x : ℝ, x > 0 ∧ f a x = 0 := by sorry

-- Theorem 3: Minimum value when a > 0
theorem f_min_value_when_a_pos :
  ∀ a : ℝ, a > 0 → (∀ x : ℝ, x > 0 → f a x ≥ 2 * Real.sqrt a) ∧ 
  ∃ x : ℝ, x > 0 ∧ f a x = 2 * Real.sqrt a := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_when_a_neg_one_f_min_value_when_a_neg_f_min_value_when_a_pos_l523_52332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l523_52352

theorem inequality_proof (a b c : ℝ) : 
  a = 31/32 → b = Real.cos (1/4) → c = 4 * Real.sin (1/4) → c > b ∧ b > a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l523_52352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_parallelogram_division_implies_center_of_symmetry_l523_52353

-- Define a convex polygon
structure ConvexPolygon where
  vertices : List (Real × Real)
  isConvex : Bool

-- Define a parallelogram
structure Parallelogram where
  vertices : List (Real × Real)

-- Define a function to check if a polygon can be divided into parallelograms
def canBeDividedIntoParallelograms (p : ConvexPolygon) : Prop :=
  ∃ (parallelograms : List Parallelogram), sorry -- Condition for division into parallelograms

-- Define center of symmetry
def hasCenterOfSymmetry (p : ConvexPolygon) : Prop :=
  ∃ (center : Real × Real), sorry -- Condition for having a center of symmetry

-- The theorem
theorem convex_polygon_parallelogram_division_implies_center_of_symmetry 
  (p : ConvexPolygon) 
  (h : canBeDividedIntoParallelograms p) : 
  hasCenterOfSymmetry p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_polygon_parallelogram_division_implies_center_of_symmetry_l523_52353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulsar_standing_time_l523_52366

/-- The time Pulsar stands on his back legs, in minutes -/
noncomputable def pulsar_time : ℝ := sorry

/-- The time Polly stands on her back legs, in minutes -/
noncomputable def polly_time : ℝ := 3 * pulsar_time

/-- The time Petra stands on his back legs, in minutes -/
noncomputable def petra_time : ℝ := (1/6) * polly_time

/-- The total time all three entertainers stand on their back legs, in minutes -/
def total_time : ℝ := 45

theorem pulsar_standing_time :
  pulsar_time + polly_time + petra_time = total_time →
  pulsar_time = 10 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulsar_standing_time_l523_52366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fewest_cookies_largest_area_l523_52306

-- Define the cookie shapes
inductive CookieShape
  | Trapezoid
  | Rectangle
  | Parallelogram
  | Triangle

-- Define a structure for a baker
structure Baker where
  name : String
  cookieShape : CookieShape
  cookieArea : ℝ

-- Define the problem setup
def bakeSale (totalDough : ℝ) (bakers : List Baker) : Prop :=
  ∀ b₁ b₂, b₁ ∈ bakers → b₂ ∈ bakers →
    b₁.cookieArea > b₂.cookieArea →
    (totalDough / b₁.cookieArea) < (totalDough / b₂.cookieArea)

-- Theorem statement
theorem fewest_cookies_largest_area (totalDough : ℝ) (bakers : List Baker) 
    (hd : totalDough > 0) (hl : bakers.length > 0) :
  bakeSale totalDough bakers →
  ∃ b, b ∈ bakers ∧ ∀ other, other ∈ bakers → 
    (totalDough / b.cookieArea) ≤ (totalDough / other.cookieArea) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fewest_cookies_largest_area_l523_52306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_a_h_l523_52348

/-- The sum of the vertical semi-axis length and the x-coordinate of the center of a hyperbola -/
def hyperbola_a_plus_h (asymptote1 asymptote2 : ℝ → ℝ) (point : ℝ × ℝ) : ℝ :=
  sorry

/-- Given asymptotes and a point on the hyperbola, prove that a + h = 4√2 - 1 -/
theorem hyperbola_sum_a_h :
  let asymptote1 := λ (x : ℝ) => 3 * x + 2
  let asymptote2 := λ (x : ℝ) => -3 * x - 4
  let point : ℝ × ℝ := (1, 5)
  hyperbola_a_plus_h asymptote1 asymptote2 point = 4 * Real.sqrt 2 - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_sum_a_h_l523_52348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l523_52387

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (3^(x-1)) - 3

-- Theorem statement
theorem f_neither_odd_nor_even :
  (∀ x : ℝ, f (-x) ≠ -f x) ∧ (∀ x : ℝ, f (-x) ≠ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l523_52387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_fraction_is_correct_hexagon_percentage_is_65_l523_52322

/-- Represents a tiling of the plane with squares and hexagons -/
structure SquareHexagonTiling where
  -- Side length of the square
  a : ℝ
  -- Assumption that a is positive
  a_pos : 0 < a

/-- The fraction of the plane enclosed by hexagons in the tiling -/
noncomputable def hexagon_fraction (t : SquareHexagonTiling) : ℝ :=
  3 * Real.sqrt 3 / 8

/-- Theorem stating that the fraction of the plane enclosed by hexagons is 3√3/8 -/
theorem hexagon_fraction_is_correct (t : SquareHexagonTiling) :
  hexagon_fraction t = 3 * Real.sqrt 3 / 8 :=
by
  -- Unfold the definition of hexagon_fraction
  unfold hexagon_fraction
  -- The equality follows directly from the definition
  rfl

/-- Theorem stating that the percentage of the plane enclosed by hexagons is approximately 65% -/
theorem hexagon_percentage_is_65 (t : SquareHexagonTiling) :
  65 / 100 < hexagon_fraction t ∧ hexagon_fraction t < 66 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_fraction_is_correct_hexagon_percentage_is_65_l523_52322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l523_52397

/-- An ellipse with foci at (-2,0) and (2,0) passing through (5/2, -3/2) has the standard equation x^2/10 + y^2/6 = 1 -/
theorem ellipse_standard_equation :
  ∀ (x y : ℝ),
  let foci₁ : ℝ × ℝ := (-2, 0)
  let foci₂ : ℝ × ℝ := (2, 0)
  let point : ℝ × ℝ := (5/2, -3/2)
  let dist₁ := Real.sqrt ((x - foci₁.1)^2 + (y - foci₁.2)^2)
  let dist₂ := Real.sqrt ((x - foci₂.1)^2 + (y - foci₂.2)^2)
  let sum_of_distances := dist₁ + dist₂
  (sum_of_distances = Real.sqrt ((point.1 - foci₁.1)^2 + (point.2 - foci₁.2)^2) +
                      Real.sqrt ((point.1 - foci₂.1)^2 + (point.2 - foci₂.2)^2)) →
  x^2 / 10 + y^2 / 6 = 1
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_standard_equation_l523_52397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l523_52311

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem monotonically_decreasing_interval
  (ω φ : ℝ)
  (h_ω_pos : ω > 0)
  (h_φ_bound : |φ| < π / 2)
  (h_period : ∀ x, f ω φ (x + π / 2) = f ω φ (x - π / 2))
  (h_symmetry : ∀ x, f ω φ (π / 6 + x) = f ω φ (π / 6 - x)) :
  ∀ x y, -5 * π / 6 ≤ x ∧ x < y ∧ y ≤ -π / 3 → f ω φ x > f ω φ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonically_decreasing_interval_l523_52311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_present_worth_approx_l523_52305

noncomputable section

-- Define the present value function
def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

-- Define the cash flows and their parameters
def cashFlow1 : ℝ := 242
def cashFlow2 : ℝ := 350
def cashFlow3 : ℝ := 500
def cashFlow4 : ℝ := 750

def interestRate1 : ℝ := 0.10
def interestRate2 : ℝ := 0.12
def interestRate3 : ℝ := 0.08
def interestRate4 : ℝ := 0.07

def years1 : ℕ := 2
def years2 : ℕ := 3
def years3 : ℕ := 4
def years4 : ℕ := 5

-- State the theorem
theorem total_present_worth_approx (ε : ℝ) (hε : ε > 0) :
  ∃ (totalPresentWorth : ℝ),
    totalPresentWorth = presentValue cashFlow1 interestRate1 years1 +
                        presentValue cashFlow2 interestRate2 years2 +
                        presentValue cashFlow3 interestRate3 years3 +
                        presentValue cashFlow4 interestRate4 years4 ∧
    abs (totalPresentWorth - 1351.59) < ε :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_present_worth_approx_l523_52305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_negative_numbers_l523_52335

noncomputable def number_list : List ℝ := [2, -0.4, 0, -3, 13/9, -1.2, 2023, 0.6]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_negative_numbers_l523_52335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_maximum_value_l523_52390

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), (f (x + π/2))^2 = (f (x + π/2 + T))^2) ∧
  (∀ (S : ℝ), S > 0 → (∀ (x : ℝ), (f (x + π/2))^2 = (f (x + π/2 + S))^2) → T ≤ S) ∧
  T = π :=
by sorry

-- Theorem for the maximum value
theorem maximum_value :
  ∃ (M : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ π/2 → f x * f (x - π/4) ≤ M) ∧
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ π/2 ∧ f x * f (x - π/4) = M) ∧
  M = 1 + Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_maximum_value_l523_52390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_approx_160_l523_52396

/-- Represents the sales data for a single location --/
structure LocationSales where
  robyn : ℚ
  lucy : ℚ

/-- Represents the sales data for all locations --/
structure TotalSales where
  neighborhood1 : LocationSales
  neighborhood2 : LocationSales
  neighborhood3 : LocationSales
  park1 : ℚ
  park2 : ℚ

/-- Calculates the total sales for Robyn and Lucy --/
noncomputable def calculate_total_sales (sales : TotalSales) : ℚ :=
  let park1_robyn := sales.park1 * (4/7)
  let park1_lucy := sales.park1 * (3/7)
  let park2_robyn := sales.park2 * (4/9)
  let park2_lucy := sales.park2 * (5/9)
  sales.neighborhood1.robyn + sales.neighborhood1.lucy +
  sales.neighborhood2.robyn + sales.neighborhood2.lucy +
  sales.neighborhood3.robyn + sales.neighborhood3.lucy +
  park1_robyn + park1_lucy + park2_robyn + park2_lucy

/-- Theorem stating that the total sales is approximately 160 --/
theorem total_sales_approx_160 (sales : TotalSales)
  (h1 : sales.neighborhood1 = ⟨15, 25/2⟩)
  (h2 : sales.neighborhood2 = ⟨23, 61/4⟩)
  (h3 : sales.neighborhood3 = ⟨71/4, 33/2⟩)
  (h4 : sales.park1 = 25)
  (h5 : sales.park2 = 35) :
  abs (calculate_total_sales sales - 160) < 1/100 := by
  sorry

#eval (160 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sales_approx_160_l523_52396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_610_in_third_quadrant_l523_52373

-- Define a function to determine the quadrant of an angle
noncomputable def quadrant (angle : ℝ) : ℕ :=
  let normalizedAngle := angle % 360
  if 0 ≤ normalizedAngle ∧ normalizedAngle < 90 then 1
  else if 90 ≤ normalizedAngle ∧ normalizedAngle < 180 then 2
  else if 180 ≤ normalizedAngle ∧ normalizedAngle < 270 then 3
  else 4

-- Theorem statement
theorem angle_610_in_third_quadrant :
  quadrant 610 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_610_in_third_quadrant_l523_52373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l523_52358

/-- Piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

/-- The theorem stating that if the range of f is ℝ, then a is in (1, 4] -/
theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → a ∈ Set.Ioo 1 4 ∪ {4} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l523_52358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_is_52_l523_52361

/-- The speed of the goods train given the conditions of the problem -/
noncomputable def goods_train_speed (woman_train_speed : ℝ) (passing_time : ℝ) (goods_train_length : ℝ) : ℝ :=
  let relative_speed := goods_train_length / passing_time * 3600 / 1000
  relative_speed - woman_train_speed

/-- Theorem stating that the speed of the goods train is 52 kmph -/
theorem goods_train_speed_is_52 :
  goods_train_speed 20 15 300 = 52 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval goods_train_speed 20 15 300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_speed_is_52_l523_52361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_triangle_area_l523_52341

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 4 ∧ 
  t.B = 2 * Real.pi / 3 ∧ 
  t.b * Real.sin t.C = 2 * Real.sin t.B

-- Theorem for the value of b
theorem b_value (t : Triangle) (h : triangle_conditions t) : 
  t.b = 2 * Real.sqrt 7 := by
  sorry

-- Theorem for the area of the triangle
theorem triangle_area (t : Triangle) (h : triangle_conditions t) : 
  (1/2) * t.a * t.c * Real.sin t.B = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_value_triangle_area_l523_52341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_route_choice_related_to_gender_route_A_higher_expected_value_l523_52349

-- Define the types for routes and genders
inductive Route : Type
| A : Route
| B : Route

inductive Gender : Type
| Male : Gender
| Female : Gender

-- Define the evaluation types
inductive Evaluation : Type
| Good : Evaluation
| Average : Evaluation

-- Define the survey data
def survey_data : Nat := 300

-- Define the chi-square test function
noncomputable def chi_square (a b c d : Nat) : ℝ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for α=0.001
def critical_value : ℝ := 10.828

-- Define the evaluation scores
def score (e : Evaluation) : Nat :=
  match e with
  | Evaluation.Good => 5
  | Evaluation.Average => 2

-- Define the expected value function
noncomputable def expected_value (good_prob average_prob : ℝ) : ℝ :=
  good_prob * (score Evaluation.Good : ℝ) + average_prob * (score Evaluation.Average : ℝ)

-- Theorem 1: Route choice is related to gender
theorem route_choice_related_to_gender :
  ∃ (a b c d : Nat), chi_square a b c d > critical_value := by
  sorry

-- Theorem 2: Route A has higher expected value
theorem route_A_higher_expected_value :
  ∃ (a_good a_avg b_good b_avg : ℝ),
    a_good + a_avg = 1 ∧ b_good + b_avg = 1 ∧
    expected_value a_good a_avg > expected_value b_good b_avg := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_route_choice_related_to_gender_route_A_higher_expected_value_l523_52349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_l523_52375

-- Define the Point type if it's not already defined in Mathlib
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to check if a point lies on an angle bisector
def lies_on_angle_bisector (D A B C : Point) : Prop := sorry

-- Define a distance function between two points
def dist (P Q : Point) : ℝ := sorry

theorem angle_bisector_ratio (A B C D : Point) 
  (h : lies_on_angle_bisector D A B C) :
  (dist A D) / (dist D C) = (dist A B) / (dist B C) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_ratio_l523_52375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_alcohol_percentage_correct_l523_52301

/-- The initial percentage of alcohol in a jar of whisky -/
noncomputable def initial_alcohol_percentage : ℚ := 40/100

/-- The percentage of alcohol in the replacement whisky -/
noncomputable def replacement_alcohol_percentage : ℚ := 19/100

/-- The final percentage of alcohol after replacement -/
noncomputable def final_alcohol_percentage : ℚ := 26/100

/-- The fraction of whisky replaced -/
noncomputable def replaced_fraction : ℚ := 2/3

/-- Theorem stating that the initial alcohol percentage is correct given the conditions -/
theorem initial_alcohol_percentage_correct :
  (1 - replaced_fraction) * initial_alcohol_percentage + 
  replaced_fraction * replacement_alcohol_percentage = 
  final_alcohol_percentage :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_alcohol_percentage_correct_l523_52301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_variance_is_21_l523_52383

/-- Represents the class composition and test scores -/
structure ClassData where
  num_girls : ℕ
  boys_avg : ℝ
  boys_var : ℝ
  girls_avg : ℝ
  girls_var : ℝ

/-- Calculates the variance of the whole class given the class data -/
noncomputable def class_variance (data : ClassData) : ℝ :=
  let num_boys := 2 * data.num_girls
  let total_students := 3 * data.num_girls
  let class_avg := (2 * data.num_girls * data.boys_avg + data.num_girls * data.girls_avg) / total_students
  (2 * data.num_girls / total_students) * (data.boys_var + (data.boys_avg - class_avg)^2) +
  (data.num_girls / total_students) * (data.girls_var + (data.girls_avg - class_avg)^2)

/-- Theorem stating that the class variance is 21 given the specific conditions -/
theorem class_variance_is_21 (data : ClassData) 
  (h1 : data.boys_avg = 120)
  (h2 : data.boys_var = 20)
  (h3 : data.girls_avg = 123)
  (h4 : data.girls_var = 17) :
  class_variance data = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_variance_is_21_l523_52383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l523_52371

/-- Parabola type representing y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point type representing a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line with slope and passing through a point -/
structure Line where
  slope : ℝ
  point : Point

/-- Function to get the focus of a parabola -/
noncomputable def focus (para : Parabola) : Point :=
  { x := para.p / 2, y := 0 }

/-- Function to check if a point is on a parabola -/
def on_parabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Function to check if a point is on a line -/
def on_line (l : Line) (pt : Point) : Prop :=
  pt.y = l.slope * (pt.x - l.point.x) + l.point.y

/-- Function to represent vector between two points -/
def vector (p1 p2 : Point) : Point :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

/-- Main theorem -/
theorem parabola_line_intersection 
  (para : Parabola) 
  (l : Line) 
  (A B M : Point) 
  (t : ℝ) : 
  l.slope = Real.tan (π/3) →
  l.point = focus para →
  on_parabola para A ∧ on_parabola para B →
  on_line l A ∧ on_line l B →
  M.x = -para.p/2 ∧ on_line l M →
  vector B M = { x := t * (A.x - M.x), y := t * (A.y - M.y) } →
  t = -1/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_l523_52371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l523_52384

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m₁ m₂ : ℝ) : Prop := m₁ = m₂

/-- The slope of line l₁: ax + 2y = 0 -/
noncomputable def slope_l₁ (a : ℝ) : ℝ := -a / 2

/-- The slope of line l₂: x + (a+1)y + 4 = 0 -/
noncomputable def slope_l₂ (a : ℝ) : ℝ := -1 / (a + 1)

theorem a_eq_one_sufficient_not_necessary (a : ℝ) :
  (a = 1 → parallel_lines (slope_l₁ a) (slope_l₂ a)) ∧
  ¬(parallel_lines (slope_l₁ a) (slope_l₂ a) → a = 1) := by
  sorry

#check a_eq_one_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_one_sufficient_not_necessary_l523_52384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l523_52333

/-- The polynomial x^3 - 6x^2 + 11x - 6 = 0 -/
def polynomial (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

/-- The roots of the polynomial -/
def roots : Set ℝ := {x | polynomial x = 0}

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem triangle_area_zero :
  ∀ a b c, a ∈ roots → b ∈ roots → c ∈ roots → triangleArea a b c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_zero_l523_52333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_f_l523_52367

/-- Represents a polynomial of degree 5 with integer coefficients -/
structure MyPolynomial where
  a₅ : ℤ
  a₄ : ℤ
  a₃ : ℤ
  a₂ : ℤ
  a₁ : ℤ
  a₀ : ℤ

/-- Counts the number of operations in Horner's method for a polynomial of degree 5 -/
def horner_operations (p : MyPolynomial) : ℕ := 8

/-- The specific polynomial f(x) = x^5 + 4x^4 + 3x^3 + 2x^2 + 1 -/
def f : MyPolynomial := ⟨1, 4, 3, 2, 0, 1⟩

theorem horner_operations_for_f :
  horner_operations f = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_operations_for_f_l523_52367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gadget_price_difference_is_913_l523_52369

/-- The price difference in cents between two stores selling Gadget Y -/
def gadget_price_difference : ℕ :=
  let list_price : ℚ := 78.5
  let mega_deals_price : ℚ := list_price * (1 - 0.12) - 5
  let quick_save_price : ℚ := list_price * (1 - 0.30)
  let price_difference : ℚ := mega_deals_price - quick_save_price
  (price_difference * 100).floor.toNat

/-- Proof that the price difference is 913 cents -/
theorem gadget_price_difference_is_913 : gadget_price_difference = 913 := by
  -- Unfold the definition of gadget_price_difference
  unfold gadget_price_difference
  -- Perform the calculation
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gadget_price_difference_is_913_l523_52369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_root_poly_count_l523_52315

/-- A polynomial of degree 7 with coefficients in {0,1} -/
def BinaryPoly7 := { p : Polynomial ℤ // p.degree ≤ 7 ∧ ∀ i, i ≤ 7 → p.coeff i ∈ ({0, 1} : Set ℤ) }

/-- The set of polynomials with exactly two different integer roots, 0 and 1 -/
def TwoRootPolys := { p : BinaryPoly7 // p.val.eval 0 = 0 ∧ p.val.eval 1 = 0 ∧
  ∀ x : ℤ, x ≠ 0 → x ≠ 1 → p.val.eval x ≠ 0 }

-- Assuming TwoRootPolys is finite
instance : Fintype TwoRootPolys := sorry

theorem two_root_poly_count : Fintype.card TwoRootPolys = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_root_poly_count_l523_52315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l523_52330

-- Define the line equation
def line_equation (x y : ℝ) : Prop := Real.sqrt 3 * x - 3 * y + 1 = 0

-- Define the slope of the line
noncomputable def line_slope : ℝ := Real.sqrt 3 / 3

-- Define the inclination angle
noncomputable def inclination_angle : ℝ := 30 * Real.pi / 180

-- Theorem statement
theorem line_inclination_angle :
  (∀ x y, line_equation x y → 
    0 ≤ inclination_angle ∧ inclination_angle < Real.pi ∧
    Real.tan inclination_angle = line_slope) →
  inclination_angle = 30 * Real.pi / 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l523_52330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_calculation_l523_52393

/-- Represents a rectangular cistern with water --/
structure WaterCistern where
  length : ℝ
  width : ℝ
  totalWetSurfaceArea : ℝ

/-- Calculates the depth of water in a cistern --/
noncomputable def waterDepth (c : WaterCistern) : ℝ :=
  (c.totalWetSurfaceArea - c.length * c.width) / (2 * (c.length + c.width))

/-- Theorem stating that for a cistern with given dimensions, the water depth is 1.5 m --/
theorem water_depth_calculation (c : WaterCistern) 
    (h1 : c.length = 10)
    (h2 : c.width = 8)
    (h3 : c.totalWetSurfaceArea = 134) :
  waterDepth c = 1.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval waterDepth { length := 10, width := 8, totalWetSurfaceArea := 134 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_depth_calculation_l523_52393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elberta_amount_l523_52314

-- Define the amounts for each person
noncomputable def granny_smith : ℝ := 54
noncomputable def anjou : ℝ := granny_smith / 4
noncomputable def elberta : ℝ := anjou + 3

-- Theorem to prove
theorem elberta_amount : elberta = 16.5 := by
  -- Unfold the definitions
  unfold elberta
  unfold anjou
  unfold granny_smith
  -- Simplify the expression
  simp
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_elberta_amount_l523_52314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l523_52339

/-- The custom operation ⊙ on real numbers -/
noncomputable def odot : ℝ → ℝ → ℝ := sorry

/-- Property 1: Commutativity -/
axiom odot_comm (a b : ℝ) : odot a b = odot b a

/-- Property 2: Identity element -/
axiom odot_zero (a : ℝ) : odot a 0 = a

/-- Property 3: Distributive property -/
axiom odot_distrib (a b c : ℝ) : odot (odot a b) c = odot (a * b) c + odot a c + odot b c - 2 * c

/-- The function to be minimized -/
noncomputable def f (x : ℝ) : ℝ := odot x (1 / x)

/-- The theorem stating the minimum value of f(x) for x > 0 -/
theorem min_value_of_f : 
  ∀ x > 0, f x ≥ 3 ∧ ∃ y > 0, f y = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l523_52339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_length_approx_l523_52362

/-- The length of a cuboid given its breadth, height, and total surface area. -/
noncomputable def cuboid_length (breadth height area : ℝ) : ℝ :=
  (area - 2 * breadth * height) / (2 * (breadth + height))

/-- Theorem: The length of a cuboid with breadth 8 cm, height 6 cm, and total surface area 480 cm² is approximately 13.71 cm. -/
theorem cuboid_length_approx :
  let b : ℝ := 8
  let h : ℝ := 6
  let a : ℝ := 480
  abs (cuboid_length b h a - 13.71) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_length_approx_l523_52362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_n_gon_l523_52350

noncomputable def sum_interior_angles (n : ℕ) : ℝ := sorry

theorem sum_interior_angles_n_gon (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (n - 2) * 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_interior_angles_n_gon_l523_52350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PT_l523_52391

-- Define the points
def P : ℝ × ℝ := (0, 4)
def Q : ℝ × ℝ := (4, 0)
def R : ℝ × ℝ := (1, 0)
def S : ℝ × ℝ := (3, 3)

-- Define the function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define T as the intersection point of PQ and RS
noncomputable def T : ℝ × ℝ := sorry

-- Theorem statement
theorem length_of_PT :
  distance P T = (4 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_PT_l523_52391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l523_52304

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  (∀ x y : ℝ, x < y → x < 0 → y ≤ 0 → f x < f y) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_interval_of_f_l523_52304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_implicit_function_derivative_l523_52377

/-- Given an implicit function xy^2 = 4, prove that its derivative at (1,2) is -1 -/
theorem implicit_function_derivative (x y : ℝ) (h : x * y^2 = 4) (hx : x = 1) (hy : y = 2) :
  let f : ℝ → ℝ → ℝ := fun x y => x * y^2 - 4
  (deriv (fun y => f x y) y) / (deriv (fun x => f x y) x) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_implicit_function_derivative_l523_52377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_g_upper_bound_l523_52327

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x) - (2 * x) / (x + 2)

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := f x - 4 / (x + 2)

-- Theorem 1: f(x) > 0 for all x > 0
theorem f_positive (x : ℝ) (h : x > 0) : f x > 0 := by
  sorry

-- Theorem 2: The smallest value of a for which g(x) < x + a holds for all x > 0 is -2
theorem g_upper_bound : 
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → g x < x + a) ↔ a > -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_g_upper_bound_l523_52327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l523_52307

-- Part 1
def f (a b c x : ℝ) := a * x^2 + b * x + c

noncomputable def F (a b c : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b c x else -f a b c x

theorem part1 (a b : ℝ) (h1 : a > 0) (h2 : f a b 1 (-1) = 0) 
  (h3 : ∀ x, f a b 1 x ≥ f a b 1 (-1)) :
  F a b 1 2 + F a b 1 (-2) = 8 := by sorry

-- Part 2
def g (b x : ℝ) := x^2 + b * x

theorem part2 (b : ℝ) (h : ∀ x ∈ Set.Ioo 0 1, |g b x| ≤ 1) :
  -2 ≤ b ∧ b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l523_52307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l523_52313

/-- Given a function f where the tangent line at some point is y = -x + 5, prove that f(3) + f'(3) = 1 -/
theorem tangent_line_property (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∃ x₀, (fun x => -x + 5) = fun x => f x₀ + (deriv f x₀) * (x - x₀)) : 
  f 3 + deriv f 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_property_l523_52313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l523_52368

-- Define the triangle ABC
structure Triangle where
  a : ℝ  -- side opposite to angle A
  b : ℝ  -- side opposite to angle B
  c : ℝ  -- side opposite to angle C
  angleC : ℝ  -- angle C

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c^2 = (t.a - t.b)^2 + 6 ∧ t.angleC = Real.pi/3

-- Define the area of the triangle
noncomputable def triangle_area (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.angleC

-- Theorem statement
theorem triangle_area_theorem (t : Triangle) 
  (h : triangle_conditions t) : triangle_area t = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l523_52368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_value_of_a_l523_52309

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel_lines (m1 m2 : ℝ) : Prop := m1 = m2

/-- The first line: y = ax - 2 -/
def line1 (a x : ℝ) : ℝ := a * x - 2

/-- The second line: 3x - (a+2)y + 1 = 1 -/
def line2 (a x y : ℝ) : Prop := 3 * x - (a + 2) * y + 1 = 1

/-- The slope of the second line -/
noncomputable def slope2 (a : ℝ) : ℝ := 1 / (a + 2)

theorem parallel_lines_value_of_a :
  ∃ a : ℝ, parallel_lines a (slope2 a) ∧ a = -1 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_value_of_a_l523_52309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l523_52320

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x)
def g (x : ℝ) : ℝ := 2 * x + 1

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := Set.range g

-- State the theorem
theorem domain_range_intersection :
  M ∩ N = Set.Ioi 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_range_intersection_l523_52320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l523_52328

-- Define the function f(x) = x^3 + 2x - 1
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

-- State the theorem
theorem zero_in_interval :
  ∃! x : ℝ, x > (1/4 : ℝ) ∧ x < (1/2 : ℝ) ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l523_52328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_k_fall_correct_expected_fallen_correct_l523_52338

-- Define the problem parameters
variable (n : ℕ) -- number of gnomes
variable (p : ℝ) -- probability of falling
variable (k : ℕ) -- number of fallen gnomes

-- Define the conditions
variable (h1 : 0 < p)
variable (h2 : p < 1)
variable (h3 : k ≤ n)

-- Define the probability function for exactly k gnomes falling
noncomputable def prob_k_fall (n k : ℕ) (p : ℝ) : ℝ := p * (1 - p) ^ (n - k)

-- Define the expected number of fallen gnomes
noncomputable def expected_fallen (n : ℕ) (p : ℝ) : ℝ := n + 1 - 1/p + (1 - p)^(n + 1)/p

-- State the theorems to be proved
theorem prob_k_fall_correct (n k : ℕ) (p : ℝ) : 
  prob_k_fall n k p = p * (1 - p) ^ (n - k) := by
  sorry

theorem expected_fallen_correct (n : ℕ) (p : ℝ) : 
  expected_fallen n p = n + 1 - 1/p + (1 - p)^(n + 1)/p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_k_fall_correct_expected_fallen_correct_l523_52338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l523_52331

theorem division_remainder_problem (x y : ℝ) 
  (hx : x > 0)
  (hy : y > 0)
  (h1 : x / y = 96.16)
  (h2 : y = 25.000000000000533) :
  Int.floor (x - y * Int.floor (x / y)) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_remainder_problem_l523_52331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l523_52310

/-- A function that returns true if a number is a 3-digit positive integer -/
def isThreeDigitPositive (n : ℕ) : Bool :=
  100 ≤ n ∧ n ≤ 999

/-- A function that returns the product of digits of a natural number -/
def digitProduct (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) * digitProduct (n / 10)

/-- The set of all 3-digit positive integers whose digits have a product of 30 -/
def validIntegers : Finset ℕ :=
  Finset.filter (fun n => isThreeDigitPositive n ∧ digitProduct n = 30) (Finset.range 1000)

theorem count_valid_integers : Finset.card validIntegers = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_integers_l523_52310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_is_36_l523_52340

/-- A function that returns true if a number is a valid three-digit number
    according to the problem conditions --/
def isValidNumber (n : ℕ) : Bool :=
  100 ≤ n ∧ n < 1000 ∧  -- Three-digit number
  n % 2 = 0 ∧  -- Even number
  (n / 10 % 10 + n % 10 = 10)  -- Sum of tens and units digits is 10

/-- The count of valid numbers --/
def validNumberCount : ℕ := (List.range 1000).filter isValidNumber |>.length

/-- Theorem stating that the count of valid numbers is 36 --/
theorem valid_number_count_is_36 : validNumberCount = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_number_count_is_36_l523_52340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l523_52344

noncomputable section

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the distance from a point to a line
noncomputable def distPointToLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (|a * p.1 + b * p.2 + c|) / Real.sqrt (a^2 + b^2)

theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- The center is on the positive x-axis
    center.1 > 0 ∧ center.2 = 0 ∧
    -- Point M(0,√3) is on the circle
    (0, Real.sqrt 3) ∈ Circle center radius ∧
    -- The distance from the center to the line 2x - y + 1 = 0 is 3√5/5
    distPointToLine center 2 (-1) 1 = (3 * Real.sqrt 5) / 5 ∧
    -- The equation of the circle is (x - 1)^2 + y^2 = 4
    center = (1, 0) ∧ radius = 2 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l523_52344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_skew_lines_l523_52303

/-- Represents a 3D line -/
structure Line3D where
  -- You might want to define this structure properly, 
  -- but for now we'll leave it as a placeholder

/-- Represents a 3D point -/
structure Point3D where
  -- You might want to define this structure properly, 
  -- but for now we'll leave it as a placeholder

/-- Checks if two lines are skew -/
def SkewLines (l m : Line3D) : Prop := sorry

/-- Checks if points are on a line with equal segments -/
def PointsOnLine (A B C : Point3D) (l : Line3D) : Prop := sorry

/-- Checks if lines are perpendicular and points are on the lines -/
def Perpendiculars (A B C D E F : Point3D) (l m : Line3D) : Prop := sorry

/-- Distance between two points -/
noncomputable def dist (p q : Point3D) : ℝ := sorry

/-- Distance between two lines -/
noncomputable def dist_line_to_line (l m : Line3D) : ℝ := sorry

/-- Main theorem -/
theorem distance_between_skew_lines
  (l m : Line3D)
  (A B C D E F : Point3D)
  (h1 : SkewLines l m)
  (h2 : PointsOnLine A B C l)
  (h3 : Perpendiculars A B C D E F l m)
  (h4 : dist A D = Real.sqrt 15)
  (h5 : dist B E = 7/2)
  (h6 : dist C F = Real.sqrt 10) :
  dist_line_to_line l m = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_skew_lines_l523_52303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_at_50_l523_52381

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry

axiom recurrence_relation (n : ℕ) :
  (sequence_a (n + 1), sequence_b (n + 1)) = (Real.sqrt 3 * sequence_a n - sequence_b n, Real.sqrt 3 * sequence_b n + sequence_a n)

axiom initial_values :
  (sequence_a 1, sequence_b 1) = (1, Real.sqrt 3)

theorem sum_at_50 :
  sequence_a 50 + sequence_b 50 = 2^49 * (-Real.sqrt 3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_at_50_l523_52381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l523_52370

-- Define the circle C
def circle_center (a : ℝ) : ℝ × ℝ := (-3*a - 3, a)

-- Define points A and B
def point_A : ℝ × ℝ := (1, 1)
def point_B : ℝ × ℝ := (2, -2)

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 3*y + 3 = 0

-- Define the line on which P moves
def line_P (x y : ℝ) : Prop := 3*x + 4*y - 21 = 0

-- State the theorem
theorem circle_and_tangents :
  ∃ (a : ℝ), 
    a = -1 ∧
    (∀ (x y : ℝ), (x - (-3*a - 3))^2 + (y - a)^2 = (1 - (-3*a - 3))^2 + (1 - a)^2) ∧
    (∀ (x y : ℝ), (x - (-3*a - 3))^2 + (y - a)^2 = (2 - (-3*a - 3))^2 + (-2 - a)^2) ∧
    (∀ (x y : ℝ), (x - (-3*a - 3))^2 + (y - a)^2 = 5) ∧
    (∀ (t : ℝ), t > Real.sqrt 5 → ∃ (S : ℝ), S = Real.sqrt 5 * Real.sqrt (t^2 - 5) ∧
      (∀ (S' : ℝ), S' = Real.sqrt 5 * Real.sqrt (t^2 - 5) → S' ≥ 10)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_tangents_l523_52370


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_two_nonpositive_l523_52355

theorem cos_power_two_nonpositive (α : ℝ) : 
  (∀ k : ℕ, Real.cos (2^k * α) ≤ 0) ↔ 
  (∃ n : ℤ, α = 2*Real.pi/3 + 2*n*Real.pi ∨ α = 4*Real.pi/3 + 2*n*Real.pi) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_power_two_nonpositive_l523_52355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_minimum_min_distance_is_achieved_l523_52343

open Real

/-- The first curve --/
noncomputable def f (x : ℝ) : ℝ := Real.exp (5 * x + 7)

/-- The second curve --/
noncomputable def g (x : ℝ) : ℝ := (log x - 7) / 5

/-- The distance function between a point on f and its corresponding point on g --/
noncomputable def distance (x : ℝ) : ℝ := Real.sqrt 2 * (f x - x)

/-- The minimum distance between the curves f and g --/
noncomputable def min_distance : ℝ := Real.sqrt 2 * ((8 + log 5) / 5)

/-- Theorem stating that min_distance is indeed the minimum distance between f and g --/
theorem min_distance_is_minimum :
  ∀ x : ℝ, distance x ≥ min_distance := by
  sorry

/-- Theorem stating that there exists a point where the distance equals min_distance --/
theorem min_distance_is_achieved :
  ∃ x : ℝ, distance x = min_distance := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_is_minimum_min_distance_is_achieved_l523_52343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_loop_condition_l523_52323

def loop_result (initial_i : ℕ) (initial_s : ℕ) (condition : ℕ → Bool) : ℕ :=
  let rec loop (i : ℕ) (s : ℕ) (fuel : ℕ) : ℕ :=
    if fuel = 0 then s
    else if condition i then
      loop (i + 1) (s * i) (fuel - 1)
    else
      s
  loop initial_i initial_s 100  -- Use a reasonable fuel value

theorem correct_loop_condition :
  loop_result 12 1 (λ i => i < 9) = 11880 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_loop_condition_l523_52323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l523_52345

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (A B c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (A^2 + B^2)

/-- Two lines are parallel if their normal vectors are proportional -/
def parallel_lines (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂

theorem distance_between_given_lines :
  let line1 := λ (x y : ℝ) => 2 * x + y - 3 = 0
  let line2 := λ (x y : ℝ) => 4 * x + 2 * y - 1 = 0
  parallel_lines 2 1 4 2 →
  distance_parallel_lines 4 2 (-6) (-1) = Real.sqrt 5 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l523_52345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l523_52354

-- Define the constants
noncomputable def a : ℝ := 1/2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 6 - 2

-- State the theorem
theorem order_of_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l523_52354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_l523_52364

/-- Given two vectors a and b in ℝ², prove that 3a - 2b equals (-7, -1) --/
theorem vector_calculation (a b : ℝ × ℝ) 
  (ha : a = (-3, 1)) (hb : b = (-1, 2)) : 
  (3 : ℝ) • a - (2 : ℝ) • b = (-7, -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_calculation_l523_52364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_equals_g_l523_52334

noncomputable def f (x : ℝ) := Real.sin (x + Real.pi/2)
noncomputable def g (x : ℝ) := Real.cos (x + 3*Real.pi/2)

theorem f_shifted_equals_g : ∀ x : ℝ, f (x - Real.pi/2) = g x := by
  intro x
  simp [f, g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_shifted_equals_g_l523_52334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_equals_200_l523_52329

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 5 * (3 * 5 + 1)
  | k + 1 => let n := sum_of_digits (a k); n * (3 * n + 1)

theorem a_2021_equals_200 : a 2021 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_equals_200_l523_52329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_m_is_1_solutions_of_f_m_eq_0_range_of_m_for_non_negative_f_l523_52347

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 2*x + 1

-- Part 1
theorem range_of_f_when_m_is_1 :
  Set.range (fun x => f 1 x) ∩ Set.Icc 0 9 = Set.Icc 0 9 := by sorry

-- Part 2
theorem solutions_of_f_m_eq_0 :
  {m : ℝ | f m m = 0} = {1, (-1 + Real.sqrt 5) / 2, (-1 - Real.sqrt 5) / 2} := by sorry

-- Part 3
theorem range_of_m_for_non_negative_f :
  {m : ℝ | ∀ x ∈ Set.Icc 1 2, f m x ≥ 0} = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_when_m_is_1_solutions_of_f_m_eq_0_range_of_m_for_non_negative_f_l523_52347
