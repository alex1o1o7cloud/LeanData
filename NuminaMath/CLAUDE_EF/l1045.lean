import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1045_104587

theorem cube_root_simplification :
  (((8 + 27 : ℝ) ^ (1/3 : ℝ)) * ((8 + 27 ^ (1/3 : ℝ)) ^ (1/3 : ℝ))) = 385 ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_simplification_l1045_104587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_seating_arrangements_l1045_104501

theorem athlete_seating_arrangements :
  let total_athletes : ℕ := 5
  let team_A_count : ℕ := 2
  let team_B_count : ℕ := 2
  let team_C_count : ℕ := 1
  let team_arrangements : ℕ := Nat.factorial 3
  let team_A_internal_arrangements : ℕ := Nat.factorial 2
  let team_B_internal_arrangements : ℕ := Nat.factorial 2
  let team_C_internal_arrangements : ℕ := Nat.factorial 1
  team_arrangements * team_A_internal_arrangements * team_B_internal_arrangements * team_C_internal_arrangements = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_athlete_seating_arrangements_l1045_104501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_implies_constant_value_l1045_104529

theorem identity_implies_constant_value :
  ∀ C : ℝ, (∀ x : ℝ, (1/2) * Real.sin x ^ 2 + C = -(1/4) * Real.cos (2*x)) →
  C = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_implies_constant_value_l1045_104529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_box_height_l1045_104552

/-- The height of a rectangular box containing spheres with specific properties -/
noncomputable def box_height (box_width : ℝ) (large_sphere_radius : ℝ) (small_sphere_radius : ℝ) : ℝ :=
  box_width + 4 * Real.sqrt 2

theorem sphere_box_height :
  let box_width := (6 : ℝ)
  let large_sphere_radius := (3 : ℝ)
  let small_sphere_radius := (1 : ℝ)
  let num_small_spheres := 8
  let k := box_height box_width large_sphere_radius small_sphere_radius
  (∀ (small_sphere : Fin num_small_spheres),
    ∃ (x y z : ℝ), x + y + z = 3 * small_sphere_radius ∧
    (x - box_width/2)^2 + (y - box_width/2)^2 + (z - k/2)^2 = (large_sphere_radius + small_sphere_radius)^2) →
  k = 6 + 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_box_height_l1045_104552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_equals_negative_six_l1045_104540

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add case for 0
  | 1 => 2
  | n + 1 => (1 + sequence_a n) / (1 - sequence_a n)

def product_to_2018 : ℚ := (Finset.range 2018).prod (λ i => sequence_a (i + 1))

theorem sequence_product_equals_negative_six : product_to_2018 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_product_equals_negative_six_l1045_104540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1045_104567

theorem sin_double_angle_special_case (α : Real) 
  (h1 : 0 < α) 
  (h2 : α < π) 
  (h3 : Real.cos α = -12/13) : 
  Real.sin (2 * α) = -120/169 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1045_104567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_after_25_days_l1045_104504

/-- Represents the number of days when two borrowers owe the same amount -/
noncomputable def days_equal_debt (
  liam_initial : ℝ
  ) (liam_rate : ℝ
  ) (olivia_initial : ℝ
  ) (olivia_rate : ℝ
  ) : ℝ :=
  (olivia_initial - liam_initial) / (liam_rate * liam_initial - olivia_rate * olivia_initial)

/-- Theorem stating that Liam and Olivia owe the same amount after 25 days -/
theorem equal_debt_after_25_days :
  days_equal_debt 200 0.08 300 0.04 = 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_debt_after_25_days_l1045_104504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_sequence_l1045_104546

def f : ℕ → ℚ
  | 0 => 1
  | 1 => 1/2
  | (n + 2) => f n

theorem sum_of_f_sequence : 
  (Finset.range 2017).sum (fun i => f (i + 1)) = 3025 / 2 := by
  sorry

#eval (Finset.range 2017).sum (fun i => f (i + 1))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_sequence_l1045_104546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sum_chord_length_sum_proof_l1045_104563

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Represents the chord length as a fraction -/
structure ChordLength where
  m : ℕ
  n : ℕ
  p : ℕ

/-- Main theorem -/
theorem chord_length_sum (c1 c2 c3 : Circle) (cl : ChordLength) : Prop :=
  are_externally_tangent c1 c2 ∧
  is_internally_tangent c1 c3 ∧
  is_internally_tangent c2 c3 ∧
  c1.radius = 5 ∧
  c2.radius = 11 ∧
  are_collinear c1.center c2.center c3.center ∧
  cl.m > 0 ∧ cl.n > 0 ∧ cl.p > 0 ∧
  Nat.Coprime cl.m cl.p ∧
  (∀ (prime : ℕ), Nat.Prime prime → ¬(prime^2 ∣ cl.n)) →
  cl.m + cl.n + cl.p = 725

/-- Proof of the theorem -/
theorem chord_length_sum_proof (c1 c2 c3 : Circle) (cl : ChordLength) : 
  chord_length_sum c1 c2 c3 cl := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_sum_chord_length_sum_proof_l1045_104563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_squared_l1045_104564

/-- A circle inscribed in a quadrilateral ABCD, tangent to AB at P and to CD at Q -/
structure InscribedCircle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: The square of the radius of the inscribed circle is 1040 -/
theorem inscribed_circle_radius_squared (circle : InscribedCircle)
  (h_AP : distance circle.A circle.P = 17)
  (h_PB : distance circle.P circle.B = 19)
  (h_CQ : distance circle.C circle.Q = 41)
  (h_QD : distance circle.Q circle.D = 31)
  (h_tangent_AB : distance circle.O circle.P = circle.r)
  (h_tangent_CD : distance circle.O circle.Q = circle.r) :
  circle.r^2 = 1040 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_squared_l1045_104564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_f_l1045_104590

def f (x : ℝ) : ℝ := x^2 - 8*x + 10

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 1

theorem minimize_sum_of_f (a : ℕ → ℝ) (h : isArithmeticSequence a) :
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, f x + f (x + 1) + f (x + 2) ≤ f y + f (y + 1) + f (y + 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_f_l1045_104590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_roots_sum_l1045_104568

theorem reciprocal_roots_sum :
  let p (x : ℝ) := 6 * x^2 - 5 * x + 3
  let α := (5 + Real.sqrt (25 - 72)) / 12
  let β := (5 - Real.sqrt (25 - 72)) / 12
  1/α + 1/β = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_roots_sum_l1045_104568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l1045_104524

theorem test_maximum_marks (passing_percentage : Real) 
                            (student_marks : ℕ) 
                            (failing_margin : ℕ) 
                            (h1 : passing_percentage = 0.30)
                            (h2 : student_marks = 80)
                            (h3 : failing_margin = 100) : 
  (student_marks + failing_margin) / passing_percentage = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l1045_104524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_ellipse_with_distance_l1045_104528

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the ellipse C: x²/4 + y²/3 = 1 -/
def on_ellipse (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- Definition of point A -/
def A : Point :=
  ⟨1, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: Points on the ellipse C that are at distance 3/2 from A have coordinates (1, ±3/2) -/
theorem points_on_ellipse_with_distance (P : Point) :
  on_ellipse P ∧ distance P A = 3/2 → P = ⟨1, 3/2⟩ ∨ P = ⟨1, -3/2⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_ellipse_with_distance_l1045_104528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_119_l1045_104532

def f : ℕ → ℕ
  | 0 => 2  -- Add this case to handle 0
  | 1 => 2
  | 2 => 3
  | n+3 => f (n+2) - f (n+1) + 2*(n+3)

theorem f_10_equals_119 : f 10 = 119 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_10_equals_119_l1045_104532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_divisions_l1045_104599

theorem bisection_method_divisions (D : Set ℝ) (accuracy : ℝ) : 
  D = Set.Icc 2 4 → accuracy = 0.1 → 
  (∃ (n : ℕ), (4 - 2) / (2 ^ n) ≤ accuracy ∧ 
    ∀ (m : ℕ), m < n → (4 - 2) / (2 ^ m) > accuracy) → 
  (∃ (n : ℕ), n = 5 ∧ (4 - 2) / (2 ^ n) ≤ accuracy ∧ 
    ∀ (m : ℕ), m < n → (4 - 2) / (2 ^ m) > accuracy) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_divisions_l1045_104599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nondivisible_sum_l1045_104542

theorem existence_of_nondivisible_sum (n : ℕ) (a : Fin n → ℕ+) 
  (h_n : n ≥ 3) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) :
  ∃ i j, i ≠ j ∧ ∀ k, ¬((a i + a j : ℕ) ∣ (3 * a k)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_nondivisible_sum_l1045_104542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sides_harmonic_mean_l1045_104539

noncomputable def harmonic_mean (a b c : ℝ) : ℝ := 3 / (1/a + 1/b + 1/c)

theorem rectangle_sides_harmonic_mean :
  let side1 : ℝ := 3
  let side2 : ℝ := 6
  let side3 : ℝ := 9
  harmonic_mean side1 side2 side3 = 54 / 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sides_harmonic_mean_l1045_104539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_2300_l1045_104558

noncomputable def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 501 * x + 10000 / x - 4500
  else 0

noncomputable def L (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 400 * x - 2000
  else if x ≥ 40 then -x - 10000 / x + 2500
  else 0

theorem max_profit_2300 :
  ∃ (x : ℝ), L x = 2300 ∧ ∀ (y : ℝ), L y ≤ L x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_2300_l1045_104558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equidistant_points_l1045_104555

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ
  h_r_pos : r > 0

/-- A line parallel to the x-axis -/
structure ParallelLine where
  y : ℝ

/-- Configuration of a circle and two parallel tangents -/
structure CircleTangentConfig where
  C : Circle
  t1 : ParallelLine
  t2 : ParallelLine
  s : ℝ
  h_s_pos : s > 0
  h_t1_dist : t1.y = C.O.2 + C.r + s
  h_t2_dist : t2.y = C.O.2 - (C.r + 2*s)

/-- Distance from a point to a circle -/
noncomputable def distToCircle (p : ℝ × ℝ) (C : Circle) : ℝ :=
  Real.sqrt ((p.1 - C.O.1)^2 + (p.2 - C.O.2)^2) - C.r

/-- Distance from a point to a parallel line -/
def distToParallelLine (p : ℝ × ℝ) (l : ParallelLine) : ℝ :=
  |p.2 - l.y|

/-- Main theorem: No points equidistant from circle and both tangents -/
theorem no_equidistant_points (config : CircleTangentConfig) :
  ∀ p : ℝ × ℝ, distToCircle p config.C ≠ distToParallelLine p config.t1 ∨
               distToCircle p config.C ≠ distToParallelLine p config.t2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equidistant_points_l1045_104555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_polynomial_l1045_104595

theorem min_degree_polynomial (P : Polynomial ℤ) (n : ℕ) : 
  (Polynomial.Monic P) →
  (∃ (S : Finset ℕ), S.card = 1000 ∧ (∀ r ∈ S, 1 ≤ r ∧ r ≤ 2013 ∧ (2013 ∣ P.eval (r : ℤ)))) →
  (∀ (T : Finset ℕ), T.card > 1000 → 
    ¬(∀ r ∈ T, 1 ≤ r ∧ r ≤ 2013 ∧ (2013 ∣ P.eval (r : ℤ)))) →
  (P.degree = n) →
  n ≥ 50 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_degree_polynomial_l1045_104595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_approximately_28_l1045_104550

/-- Represents the experiment of estimating the number of white balls in a box --/
structure BallEstimation where
  white_balls : ℕ
  black_balls : ℕ
  total_draws : ℕ
  black_draws : ℕ

/-- Calculates the estimated number of white balls based on the experiment results --/
noncomputable def estimate_white_balls (e : BallEstimation) : ℝ :=
  (e.black_balls * e.total_draws : ℝ) / e.black_draws - e.black_balls

/-- The main theorem stating that given the experimental conditions, 
    the estimated number of white balls is approximately 28 --/
theorem estimate_approximately_28 (e : BallEstimation) 
  (h1 : e.black_balls = 8)
  (h2 : e.total_draws = 400)
  (h3 : e.black_draws = 88) :
  ∃ ε > 0, |estimate_white_balls e - 28| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_approximately_28_l1045_104550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_equation_solution_l1045_104582

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the Fibonacci sequence
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem golden_ratio_equation_solution (n : ℕ) :
  ∃ (x y : ℤ), x * φ^(n + 1) + y * φ^n = 1 ∧
               x = (-1)^(n + 1) * (fib n) ∧
               y = (-1)^n * (fib (n + 1)) := by
  sorry

#check golden_ratio_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_ratio_equation_solution_l1045_104582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1045_104591

noncomputable def angle_alpha (m : ℝ) : ℝ := Real.arcsin ((Real.sqrt 3 / 4) * m)

theorem tan_alpha_value (m : ℝ) (h1 : m ≠ 0) :
  let α := angle_alpha m
  let P : ℝ × ℝ := (-Real.sqrt 3, m)
  ∃ (sign : ℝ), (sign = 1 ∨ sign = -1) ∧ Real.tan α = sign * Real.sqrt 7 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l1045_104591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_z_values_l1045_104527

/-- Definition of a three-digit number -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Definition of digit reversal for a three-digit number -/
def ReverseDigits (x y : ℕ) : Prop :=
  ∃ a b c : ℕ,
    x = 100 * a + 10 * b + c ∧
    y = 100 * c + 10 * b + a ∧
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

/-- Main theorem -/
theorem distinct_z_values
  (x y : ℕ)
  (hx : ThreeDigitNumber x)
  (hy : ThreeDigitNumber y)
  (hxy : ReverseDigits x y)
  (z : ℕ)
  (hz : z = Int.natAbs (x - y)) :
  ∃ S : Finset ℕ, (∀ w, w ∈ S ↔ ∃ a c : ℕ, z = 99 * Int.natAbs (a - c) ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9) ∧ Finset.card S = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_z_values_l1045_104527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_35_and_200_l1045_104570

theorem multiples_of_15_between_35_and_200 : 
  Finset.card (Finset.filter (fun n => 15 ∣ n ∧ 35 < n ∧ n < 200) (Finset.range 200)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_35_and_200_l1045_104570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_identity_l1045_104514

theorem tan_two_identity (θ : Real) (h : Real.tan θ = 2) :
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_two_identity_l1045_104514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_not_always_divisible_l1045_104516

/-- A sequence defined by x_{n+1} = x_n * x_{n-1} + 1 -/
def x (x₁ x₂ : ℕ) : ℕ → ℕ
  | 0 => 0  -- We define x₀ as 0 for convenience
  | 1 => x₁
  | 2 => x₂
  | (n + 3) => x x₁ x₂ (n + 2) * x x₁ x₂ (n + 1) + 1

theorem divisibility_property (x₁ x₂ : ℕ) (h_coprime : Nat.Coprime x₁ x₂) (h_pos : 0 < x₁ ∧ 0 < x₂) :
  ∀ i > 1, ∃ j > i, (x x₁ x₂ i)^i ∣ (x x₁ x₂ j)^i :=
by
  sorry

-- Part (b) counterexample
def counterexample : ℕ → ℕ := x 22 9

theorem not_always_divisible :
  ¬ ∃ j > 1, 22 ∣ (counterexample j)^j :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_not_always_divisible_l1045_104516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_one_greater_than_negative_two_l1045_104561

theorem only_negative_one_greater_than_negative_two :
  ∀ x : ℝ, x ∈ ({-2.5, -3, -1, -4} : Set ℝ) → (x > -2 ↔ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_negative_one_greater_than_negative_two_l1045_104561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_paths_l1045_104533

theorem spider_paths : Nat.choose 11 5 = 462 := by
  -- n := 5  -- number of upward steps
  -- m := 6  -- number of rightward steps
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_spider_paths_l1045_104533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_size_l1045_104556

/-- Proves that a cricket team has 11 members given specific conditions about the team's ages. -/
theorem cricket_team_size : ∃ (n : ℕ), n > 0 ∧ 
  (let captain_age : ℕ := 24;
   let keeper_age : ℕ := captain_age + 7;
   let team_average : ℕ := 23;
   let remaining_average : ℕ := team_average - 1;
   n * team_average = (n - 2) * remaining_average + captain_age + keeper_age ∧
   n = 11) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_team_size_l1045_104556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_product_35_l1045_104574

/-- A two-digit number divisible by 35 -/
structure Number35 where
  value : ℕ
  is_two_digit : 10 ≤ value ∧ value ≤ 99
  divisible_by_35 : value % 35 = 0

/-- The tens digit of a number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- The units digit of a number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- Theorem: The maximum product of tens and units digits for a two-digit number divisible by 35 is 15 -/
theorem max_digit_product_35 :
  ∀ n : Number35, ∃ m : Number35, tens_digit n.value * units_digit n.value ≤ tens_digit m.value * units_digit m.value ∧
  tens_digit m.value * units_digit m.value = 15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_product_35_l1045_104574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_taken_C_and_D_approx_2_86_l1045_104566

-- Define work rates for individuals A, B, and D
noncomputable def work_rate_A : ℝ := 1 / 4
noncomputable def work_rate_B : ℝ := 1 / 10
noncomputable def work_rate_D : ℝ := 1 / 5

-- Define the combined work rate of A, B, and C
noncomputable def work_rate_ABC : ℝ := 1 / 2

-- Define the function to calculate the time taken by two individuals to complete the work
noncomputable def time_taken (rate1 rate2 : ℝ) : ℝ := 1 / (rate1 + rate2)

-- Define the work rate of C
noncomputable def work_rate_C : ℝ := work_rate_ABC - work_rate_A - work_rate_B

-- Theorem statement
theorem time_taken_C_and_D_approx_2_86 : 
  ∃ ε > 0, abs (time_taken work_rate_C work_rate_D - 2.86) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_taken_C_and_D_approx_2_86_l1045_104566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1045_104535

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, (2 : ℝ)^x > 0) ↔ (∃ x : ℝ, (2 : ℝ)^x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1045_104535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_axes_perpendicular_l1045_104548

-- Define a plane figure
structure PlaneFigure where
  -- Add any necessary properties for a plane figure
  mk :: -- Empty for now, can be extended later

-- Define an axis of symmetry
structure AxisOfSymmetry where
  -- Add any necessary properties for an axis of symmetry
  mk :: -- Empty for now, can be extended later

-- Define a property for a figure having exactly two axes of symmetry
def hasTwoAxesOfSymmetry (f : PlaneFigure) (a1 a2 : AxisOfSymmetry) : Prop :=
  ∃! (x y : AxisOfSymmetry), (x = a1 ∧ y = a2) ∨ (x = a2 ∧ y = a1)

-- Define perpendicularity for axes of symmetry
def arePerpendicular (a1 a2 : AxisOfSymmetry) : Prop :=
  -- Add the mathematical definition of perpendicularity for axes
  True -- Placeholder, replace with actual definition when available

-- State the theorem
theorem two_axes_perpendicular (f : PlaneFigure) (a1 a2 : AxisOfSymmetry) :
  hasTwoAxesOfSymmetry f a1 a2 → arePerpendicular a1 a2 := by
  intro h
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_axes_perpendicular_l1045_104548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l1045_104549

/-- The point on a line that is closest to a given point -/
noncomputable def closest_point_on_line (a b : ℝ) (x₀ y₀ : ℝ) : ℝ × ℝ :=
  let m := -4  -- slope of the line
  let c := 10  -- y-intercept of the line
  let x := (m * (y₀ - c) + x₀) / (m^2 + 1)
  let y := m * x + c
  (x, y)

/-- The distance between two points -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem mouse_cheese_problem :
  let cheese := (14, 6)
  let mouse_path (x : ℝ) := -4 * x + 10
  let closest := closest_point_on_line 14 6 0 10
  closest.1 = 42/17 ∧ closest.2 = 2/17 ∧
  ∀ x, x ≠ closest.1 →
    distance 14 6 closest.1 closest.2 < distance 14 6 x (mouse_path x) :=
by sorry

#check mouse_cheese_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_problem_l1045_104549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_ratio_l1045_104593

theorem triangle_trig_ratio (P Q R : ℝ) : 
  8 = 10 * Real.sin Q / Real.sin R →
  6 = 10 * Real.sin P / Real.sin R →
  6 = 8 * Real.sin P / Real.sin Q →
  (Real.cos ((P - Q)/2) / Real.sin (R/2)) - (Real.sin ((P - Q)/2) / Real.cos (R/2)) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_ratio_l1045_104593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l1045_104583

theorem probability_at_most_three_heads_ten_coins :
  let n : ℕ := 10  -- number of coins
  let k : ℕ := 3   -- maximum number of heads
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := (Finset.range (k+1)).sum (λ i ↦ Nat.choose n i)
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_most_three_heads_ten_coins_l1045_104583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_four_solutions_l1045_104565

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the equation
def equation (x : ℝ) : Prop :=
  4 * x^2 - 40 * (floor x) + 51 = 0

-- Theorem statement
theorem equation_has_four_solutions :
  ∃ (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, equation x ∧
    ∀ y : ℝ, equation y → y ∈ s :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_has_four_solutions_l1045_104565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_is_random_l1045_104534

-- Define the events as axioms (assumptions)
axiom sun_rise : Prop
axiom traffic_light : Prop
axiom circle_points : Prop
axiom triangle_angles : Prop

-- Define what it means for an event to be random
def is_random (event : Prop) : Prop := ∃ (outcome : Bool), ¬(event ↔ outcome = true)

-- Theorem statement
theorem traffic_light_is_random :
  is_random traffic_light ∧
  ¬is_random sun_rise ∧
  ¬is_random circle_points ∧
  ¬is_random triangle_angles :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_light_is_random_l1045_104534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_slope_product_l1045_104571

noncomputable section

-- Define the parabola Γ
def Γ (x y : ℝ) : Prop := y^2 = 4 * Real.sqrt 3 * x

-- Define the ellipse C
def C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the focus F₁
def F₁ : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the directrix of Γ
def directrix_Γ (x : ℝ) : Prop := x = Real.sqrt 3

-- Define the distance function
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Define the slope of a line
def line_slope (x₁ y₁ x₂ y₂ : ℝ) : ℝ := (y₂ - y₁) / (x₂ - x₁)

-- Main theorem
theorem ellipse_constant_slope_product 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (A : ℝ × ℝ) 
  (h_A_on_Γ : Γ A.1 A.2) 
  (h_A_on_C : C A.1 A.2 a b) 
  (h_F₁_on_C : C F₁.1 F₁.2 a b) 
  (F₂ : ℝ × ℝ) 
  (h_F₂_on_C : C F₂.1 F₂.2 a b) 
  (h_distance_sum : distance A.1 A.2 F₁.1 F₁.2 + distance A.1 A.2 F₂.1 F₂.2 = 4) :
  ∃ (M N : ℝ × ℝ), 
    M.1 = -Real.sqrt 2 ∧ M.2 = 0 ∧ 
    N.1 = Real.sqrt 2 ∧ N.2 = 0 ∧
    ∀ (P : ℝ × ℝ), C P.1 P.2 2 1 → 
      line_slope P.1 P.2 M.1 M.2 * line_slope P.1 P.2 N.1 N.2 = -1/4 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_constant_slope_product_l1045_104571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_carbon_benzene_approx_l1045_104547

/-- The molar mass of carbon in g/mol -/
noncomputable def molar_mass_carbon : ℝ := 12.01

/-- The molar mass of hydrogen in g/mol -/
noncomputable def molar_mass_hydrogen : ℝ := 1.008

/-- The number of carbon atoms in a benzene molecule -/
def carbon_atoms_benzene : ℕ := 6

/-- The number of hydrogen atoms in a benzene molecule -/
def hydrogen_atoms_benzene : ℕ := 6

/-- The mass percentage of carbon in benzene -/
noncomputable def mass_percentage_carbon_benzene : ℝ :=
  let total_mass_carbon := (carbon_atoms_benzene : ℝ) * molar_mass_carbon
  let total_mass_hydrogen := (hydrogen_atoms_benzene : ℝ) * molar_mass_hydrogen
  let total_mass_benzene := total_mass_carbon + total_mass_hydrogen
  (total_mass_carbon / total_mass_benzene) * 100

theorem mass_percentage_carbon_benzene_approx :
  abs (mass_percentage_carbon_benzene - 92.26) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_carbon_benzene_approx_l1045_104547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l1045_104502

noncomputable def g (n : ℤ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((2 + Real.sqrt 7) / 3) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((2 - Real.sqrt 7) / 3) ^ n

theorem g_relation (n : ℤ) : g (n + 1) - g (n - 1) = g n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_relation_l1045_104502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_implies_m_equals_two_l1045_104544

-- Define the power function
noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := 
  (m^2 - m - 1) * (x^(m^2 - 2*m - 3))

-- State the theorem
theorem decreasing_power_function_implies_m_equals_two :
  ∀ m : ℝ, 
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → power_function m x₁ > power_function m x₂) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_power_function_implies_m_equals_two_l1045_104544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_odd_numbers_count_l1045_104588

def odd_digits : Finset Nat := {1, 3, 5, 7, 9}

theorem three_digit_odd_numbers_count : 
  (Finset.filter (fun n => 
    100 ≤ n ∧ n < 1000 ∧ 
    (n / 100 % 10) ∈ odd_digits ∧ 
    (n / 10 % 10) ∈ odd_digits ∧ 
    (n % 10) ∈ odd_digits
  ) (Finset.range 1000)).card = 125 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_odd_numbers_count_l1045_104588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l1045_104554

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : Option ℝ  -- None represents vertical lines
  y_intercept : ℝ

/-- The hyperbola x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 = 2

/-- A point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The point (√2, 0) -/
noncomputable def point_sqrt2_0 : Point :=
  ⟨Real.sqrt 2, 0⟩

/-- Predicate to check if a point is on a line -/
def on_line (p : Point) (l : Line) : Prop :=
  match l.slope with
  | none => p.x = l.y_intercept
  | some m => p.y = m * (p.x - l.y_intercept)

/-- Predicate to check if a line intersects the hyperbola at only one point -/
def intersects_hyperbola_once (l : Line) : Prop :=
  ∃! p : Point, on_line p l ∧ hyperbola p.x p.y

/-- Predicate to check if a line passes through the point (√2, 0) -/
def passes_through_sqrt2_0 (l : Line) : Prop :=
  on_line point_sqrt2_0 l

/-- The main theorem stating that there are exactly 3 lines satisfying the conditions -/
theorem exactly_three_lines :
  ∃! (lines : Finset Line),
    lines.card = 3 ∧
    ∀ l ∈ lines, passes_through_sqrt2_0 l ∧ intersects_hyperbola_once l :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_lines_l1045_104554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1045_104577

-- Define the point M
def M : ℝ × ℝ := (2, 4)

-- Define the property of perpendicular lines
def perpendicular (A B : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Define the property of A and B being on positive x-axis and y-axis respectively
def on_axes (A B : ℝ × ℝ) : Prop :=
  A.2 = 0 ∧ A.1 > 0 ∧ B.1 = 0 ∧ B.2 > 0

-- Define the property of AB bisecting OAMB
def bisects_quadrilateral (A B : ℝ × ℝ) : Prop :=
  (A.1 * B.2) / 2 = (B.2 * M.1 + A.1 * M.2 - A.1 * B.2) / 2

-- Theorem statement
theorem line_AB_equation :
  ∀ A B : ℝ × ℝ,
  perpendicular A B →
  on_axes A B →
  bisects_quadrilateral A B →
  (∃ k : ℝ, k * A.1 * B.2 = 5 ∧ k * (A.1 + 2 * B.2) = 5) ∨
  (∃ k : ℝ, k * A.1 * B.2 = 4 ∧ k * (2 * A.1 + B.2) = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_AB_equation_l1045_104577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_self_inverse_iff_b_eq_2_l1045_104530

-- Define a function f that is symmetric about y = x - 2
noncomputable def f : ℝ → ℝ := sorry

-- Define the property of f being symmetric about y = x - 2
def symmetric_about_x_minus_2 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y ↔ f (y + 2) = x + 2

-- Define h in terms of f and b
def h (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := λ x ↦ f x + b

-- Define the property of a function being its own inverse
def is_self_inverse (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (g x) = x

-- Theorem statement
theorem h_is_self_inverse_iff_b_eq_2 (f : ℝ → ℝ) (b : ℝ) 
  (h_sym : symmetric_about_x_minus_2 f) :
  is_self_inverse (h f b) ↔ b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_is_self_inverse_iff_b_eq_2_l1045_104530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_MQ_max_min_k_l1045_104522

-- Define the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 14*y + 45 = 0

-- Define point Q
def Q : ℝ × ℝ := (-2, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Part I
theorem max_min_distance_MQ :
  (∃ (M : ℝ × ℝ), circleC M.1 M.2 ∧ distance M Q = 6 * Real.sqrt 2) ∧
  (∃ (M : ℝ × ℝ), circleC M.1 M.2 ∧ distance M Q = 2 * Real.sqrt 2) ∧
  (∀ (M : ℝ × ℝ), circleC M.1 M.2 → 2 * Real.sqrt 2 ≤ distance M Q ∧ distance M Q ≤ 6 * Real.sqrt 2) :=
by sorry

-- Part II
theorem max_min_k :
  (∃ (m n : ℝ), m^2 + n^2 - 4*m - 14*n + 45 = 0 ∧ (n - 3) / (m + 2) = 2 + Real.sqrt 3) ∧
  (∃ (m n : ℝ), m^2 + n^2 - 4*m - 14*n + 45 = 0 ∧ (n - 3) / (m + 2) = 2 - Real.sqrt 3) ∧
  (∀ (m n : ℝ), m^2 + n^2 - 4*m - 14*n + 45 = 0 → 
    2 - Real.sqrt 3 ≤ (n - 3) / (m + 2) ∧ (n - 3) / (m + 2) ≤ 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_distance_MQ_max_min_k_l1045_104522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_eq_l1045_104526

/-- The range of real numbers m for which the solution set of the inequality 
    x^2 - (m+3)x + 3m < 0 with respect to x contains exactly 3 integers. -/
def solution_range : Set ℝ :=
  {m : ℝ | ∃ (a b c : ℤ), 
    (∀ x : ℝ, x^2 - (m+3)*x + 3*m < 0 ↔ (x > a ∧ x < b) ∨ (x > b ∧ x < c)) ∧
    (∀ x : ℤ, x^2 - (m+3)*x + 3*m < 0 ↔ x = a + 1 ∨ x = a + 2 ∨ x = a + 3)}

theorem solution_range_eq : solution_range = Set.Ioc (-1) 0 ∪ Set.Ioc 6 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_range_eq_l1045_104526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l1045_104523

/-- RegularHexagonalPyramid is a predicate that determines if a set of points in ℝ³ forms a regular hexagonal pyramid -/
def RegularHexagonalPyramid (pyramid : Set ℝ) : Prop := sorry

/-- BaseEdgeLength calculates the length of a base edge of a regular hexagonal pyramid -/
def BaseEdgeLength (pyramid : Set ℝ) : ℝ := sorry

/-- LateralEdgeLength calculates the length of a lateral edge of a regular hexagonal pyramid -/
def LateralEdgeLength (pyramid : Set ℝ) : ℝ := sorry

/-- Volume calculates the volume of a regular hexagonal pyramid -/
def Volume (pyramid : Set ℝ) : ℝ := sorry

/-- A regular hexagonal pyramid with base edge length 1 and lateral edge length √5 has volume √3 -/
theorem hexagonal_pyramid_volume :
  ∀ (pyramid : Set ℝ),
    RegularHexagonalPyramid pyramid →
    BaseEdgeLength pyramid = 1 →
    LateralEdgeLength pyramid = Real.sqrt 5 →
    Volume pyramid = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_l1045_104523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1045_104520

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the distance from a point to the focus
noncomputable def distToFocus (p x y : ℝ) : ℝ := Real.sqrt ((x - p)^2 + y^2)

-- Define the distance from the focus to the directrix
def focusToDirectrix (p : ℝ) : ℝ := 2*p

-- State the theorem
theorem parabola_focus_directrix_distance (p : ℝ) :
  parabola p 6 (Real.sqrt (12*p)) → distToFocus p 6 (Real.sqrt (12*p)) = 10 → focusToDirectrix p = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l1045_104520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_roots_of_unity_polynomial_l1045_104512

/-- A complex number z is a root of unity if z^n = 1 for some positive integer n -/
def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z ^ n = 1

/-- The polynomial z^3 + az^2 + bz + 1 = 0 -/
def polynomial (a b : ℤ) (z : ℂ) : ℂ :=
  z^3 + a*z^2 + b*z + 1

theorem eight_roots_of_unity_polynomial :
  ∃! (s : Finset ℂ), s.card = 8 ∧
    ∀ z ∈ s, is_root_of_unity z ∧
      ∃ a b : ℤ, polynomial a b z = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_roots_of_unity_polynomial_l1045_104512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_slope_product_l1045_104545

/-- An ellipse with semi-major axis a and semi-minor axis b. -/
structure Ellipse (a b : ℝ) where
  h1 : a > 0
  h2 : b > 0
  h3 : a > b

/-- The eccentricity of an ellipse. -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- A point on the ellipse. -/
structure PointOnEllipse (e : Ellipse a b) where
  x : ℝ
  y : ℝ
  on_ellipse : x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_eccentricity_from_slope_product 
  (e : Ellipse a b) 
  (P Q : PointOnEllipse e) 
  (h_symmetric : P.x = -Q.x ∧ P.y = Q.y) 
  (h_slope_product : (P.y / (P.x + a)) * (Q.y / (Q.x + a)) = 1/4) :
  eccentricity e = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_slope_product_l1045_104545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dorothy_museum_trip_l1045_104569

/-- Calculates the amount of money Dorothy has left after a museum trip with her family. -/
def dorothy_money_left (dorothy_age : ℕ) (family_size : ℕ) (regular_ticket_cost : ℚ) 
  (youth_discount : ℚ) (dorothy_initial_money : ℚ) : ℚ :=
  let discounted_ticket_cost := regular_ticket_cost * (1 - youth_discount)
  let youth_tickets := 2  -- Dorothy and her younger brother
  let adult_tickets := family_size - youth_tickets
  let total_cost := discounted_ticket_cost * youth_tickets + regular_ticket_cost * adult_tickets
  dorothy_initial_money - total_cost

/-- Main theorem proving Dorothy will have $26 left after the museum trip. -/
theorem dorothy_museum_trip : 
  dorothy_money_left 15 5 10 (30/100) 70 = 26 := by
  -- Unfold the definition of dorothy_money_left
  unfold dorothy_money_left
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dorothy_museum_trip_l1045_104569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l1045_104589

/-- Given vectors a, b, and c in ℝ², if k*a + b is collinear with c, then k = -2 -/
theorem collinear_vectors (a b c : ℝ × ℝ) (k : ℝ) : 
  a = (-1, 2) →
  b = (2, -3) →
  c = (-4, 7) →
  (∃ (t : ℝ), t ≠ 0 ∧ t • (k • a + b) = c) →
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_l1045_104589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem1_theorem2_theorem3_l1045_104505

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the tangent line l
def l (x y a b : ℝ) : Prop := b*x + a*y - a*b = 0

-- Define the conditions
def conditions (a b : ℝ) : Prop :=
  a > 2 ∧ b > 2 ∧ ∃ (x y : ℝ), C x y ∧ l x y a b

-- Theorem 1: (a - 2)(b - 2) = 2
theorem theorem1 (a b : ℝ) (h : conditions a b) : (a - 2) * (b - 2) = 2 := by sorry

-- Define the midpoint M
def M (x y : ℝ) : Prop := x > 1 ∧ y > 1 ∧ (x - 1) * (y - 1) = 1/2

-- Theorem 2: The locus of the midpoint M satisfies the given equation
theorem theorem2 (a b : ℝ) (h : conditions a b) : 
  ∃ (x y : ℝ), M x y ∧ x = a/2 ∧ y = b/2 := by sorry

-- Define the area of triangle AOB
noncomputable def area_AOB (a b : ℝ) : ℝ := (1/2) * a * b

-- Theorem 3: The minimum area of triangle AOB is 2√2 + 3
theorem theorem3 (a b : ℝ) (h : conditions a b) :
  (∀ (a' b' : ℝ), conditions a' b' → area_AOB a' b' ≥ area_AOB a b) →
  area_AOB a b = 2 * Real.sqrt 2 + 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_theorem1_theorem2_theorem3_l1045_104505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_in_20_moves_l1045_104559

/-- Represents the state of the circle with 6 sectors -/
def CircleState := Fin 6 → ℕ

/-- The initial state of the circle -/
def initial_state : CircleState := λ _ ↦ 1

/-- A valid move changes the state by moving one coin to a neighboring sector -/
def valid_move (s₁ s₂ : CircleState) : Prop :=
  ∃ i j : Fin 6, (j = i + 1 ∨ j = i - 1) ∧
    s₂ i = s₁ i - 1 ∧
    s₂ j = s₁ j + 1 ∧
    ∀ k : Fin 6, k ≠ i → k ≠ j → s₂ k = s₁ k

/-- A sequence of n valid moves -/
def valid_moves (n : ℕ) (s₁ s₂ : CircleState) : Prop :=
  match n with
  | 0 => s₁ = s₂
  | n + 1 => ∃ s' : CircleState, valid_move s₁ s' ∧ valid_moves n s' s₂

/-- The goal state where all coins are in one sector -/
def goal_state (s : CircleState) : Prop :=
  ∃ i : Fin 6, s i = 6 ∧ ∀ j : Fin 6, j ≠ i → s j = 0

/-- The main theorem stating that it's impossible to reach the goal state in exactly 20 moves -/
theorem impossible_in_20_moves :
  ¬∃ s : CircleState, valid_moves 20 initial_state s ∧ goal_state s :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_in_20_moves_l1045_104559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_omega_is_10th_root_omega_5_is_neg_one_l1045_104579

/-- Define ω as a complex number -/
noncomputable def ω : ℂ := Complex.exp (Complex.I * Real.pi / 5)

/-- The polynomial we want to prove has the specified roots -/
def f (x : ℂ) : ℂ := x^4 - x^3 + x^2 - x + 1

/-- Theorem stating that f has roots ω, ω³, ω⁷, ω⁹ -/
theorem roots_of_f :
  f ω = 0 ∧ f (ω^3) = 0 ∧ f (ω^7) = 0 ∧ f (ω^9) = 0 := by
  sorry

/-- Auxiliary theorem: ω is a 10th root of unity -/
theorem omega_is_10th_root : ω^10 = 1 := by
  sorry

/-- Auxiliary theorem: ω⁵ = -1 -/
theorem omega_5_is_neg_one : ω^5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_f_omega_is_10th_root_omega_5_is_neg_one_l1045_104579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l1045_104538

/-- The speed of a train in km/h, given its length in meters and the time it takes to pass a stationary point. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem stating that a 160-meter long train passing a point in 12 seconds has a speed of approximately 48 km/h. -/
theorem train_speed_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |train_speed 160 12 - 48| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_approximation_l1045_104538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_has_advantage_l1045_104525

/-- Represents a four-digit number split into two two-digit numbers -/
structure SplitNumber :=
  (left : Nat) (right : Nat)
  (h_left : left < 100) (h_right : right < 100)

/-- Determines if András pays Béla for a given split number -/
def andrasPays (split : SplitNumber) : Bool :=
  let k := min split.left split.right
  let n := max split.left split.right
  k = 0 || (n % k = 0)

/-- Calculates the payment for a given split number -/
def payment (split : SplitNumber) : Rat :=
  if andrasPays split then 1 else -1/10

/-- The set of all possible four-digit numbers -/
def allFourDigitNumbers : Finset Nat :=
  Finset.filter (fun n => n ≥ 1000 && n < 10000) (Finset.range 10000)

/-- Splits a four-digit number into a SplitNumber -/
def splitNumber (n : Nat) : SplitNumber :=
  { left := n / 100,
    right := n % 100,
    h_left := by
      sorry
    h_right := by
      sorry }

/-- The expected value of the game for Béla -/
noncomputable def expectedValue : Rat :=
  (allFourDigitNumbers.sum (fun n => payment (splitNumber n))) / allFourDigitNumbers.card

/-- Theorem stating that Béla has the advantage -/
theorem bela_has_advantage : expectedValue > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bela_has_advantage_l1045_104525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_abs_cos_range_l1045_104585

theorem sin_greater_abs_cos_range (x : ℝ) : 
  x ∈ Set.Ioo (0 : ℝ) (2 * Real.pi) →
  (Real.sin x > |Real.cos x|) ↔ x ∈ Set.Ioo (Real.pi / 4) (3 * Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_greater_abs_cos_range_l1045_104585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l1045_104517

/-- A triangle with given perimeter, side length, and inscribed circle radius -/
structure Triangle where
  k : ℝ  -- perimeter
  a : ℝ  -- one side length
  r : ℝ  -- radius of inscribed circle
  k_pos : k > 0
  a_pos : a > 0
  r_pos : r > 0
  a_lt_k : a < k

/-- A circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a common tangent exists between two circles -/
def common_tangent_exists (c1 c2 : Circle) : Prop :=
  sorry -- Definition of common tangent existence

/-- The existence of a triangle with given parameters -/
def triangle_exists (t : Triangle) : Prop :=
  ∃ (m : ℝ), m = t.k * t.r / t.a ∧ 
  ∃ (c1 c2 : Circle), 
    c1.radius = t.r ∧ 
    c2.radius = m ∧ 
    common_tangent_exists c1 c2

theorem triangle_construction_theorem (t : Triangle) : 
  triangle_exists t ↔ 
  ∃ (m : ℝ), m = t.k * t.r / t.a ∧ 
  ∃ (c1 c2 : Circle), 
    c1.radius = t.r ∧ 
    c2.radius = m ∧ 
    common_tangent_exists c1 c2 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l1045_104517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l1045_104508

noncomputable def option_a : ℝ := 12 - 3 * Real.sqrt 14
noncomputable def option_b : ℝ := 3 * Real.sqrt 14 - 12
noncomputable def option_c : ℝ := 20 - 5 * Real.sqrt 15
noncomputable def option_d : ℝ := 54 - 10 * Real.sqrt 27
noncomputable def option_e : ℝ := 10 * Real.sqrt 27 - 54

theorem smallest_positive_number :
  option_c > 0 ∧
  (option_a ≤ 0 ∨ option_c < option_a) ∧
  (option_b ≤ 0 ∨ option_c < option_b) ∧
  (option_d ≤ 0 ∨ option_c < option_d) ∧
  (option_e ≤ 0 ∨ option_c < option_e) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_number_l1045_104508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l1045_104518

-- Define the polynomial P(X) = X^3 - 3X^2 - 1
def P (X : ℂ) : ℂ := X^3 - 3*X^2 - 1

-- Define the roots of P(X)
def roots_of_P : Set ℂ := {r | P r = 0}

-- Theorem statement
theorem sum_of_cubes_of_roots :
  ∃ (r₁ r₂ r₃ : ℂ), r₁ ∈ roots_of_P ∧ r₂ ∈ roots_of_P ∧ r₃ ∈ roots_of_P ∧
  r₁^3 + r₂^3 + r₃^3 = 24 := by
  sorry

#check sum_of_cubes_of_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_roots_l1045_104518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_costs_l1045_104560

-- Define the structure for a family's trip details
structure TripDetails where
  highway_miles : Nat
  city_miles : Nat
  highway_mpg : Nat
  city_mpg : Nat
  highway_fuel_price : Rat
  city_fuel_price : Rat

-- Define the maintenance costs
def highway_maintenance_cost : Rat := 5/100
def city_maintenance_cost : Rat := 7/100

-- Function to calculate total cost for a family
noncomputable def calculate_total_cost (trip : TripDetails) : Rat :=
  let highway_fuel_cost := (trip.highway_miles / trip.highway_mpg : Rat) * trip.highway_fuel_price
  let city_fuel_cost := (trip.city_miles / trip.city_mpg : Rat) * trip.city_fuel_price
  let highway_maintenance := (trip.highway_miles : Rat) * highway_maintenance_cost
  let city_maintenance := (trip.city_miles : Rat) * city_maintenance_cost
  highway_fuel_cost + city_fuel_cost + highway_maintenance + city_maintenance

-- Define the trip details for each family
def jensen_trip : TripDetails := {
  highway_miles := 210,
  city_miles := 54,
  highway_mpg := 35,
  city_mpg := 18,
  highway_fuel_price := 37/10,
  city_fuel_price := 42/10
}

def smith_trip : TripDetails := {
  highway_miles := 240,
  city_miles := 60,
  highway_mpg := 30,
  city_mpg := 15,
  highway_fuel_price := 77/20,
  city_fuel_price := 4
}

def greens_trip : TripDetails := {
  highway_miles := 260,
  city_miles := 48,
  highway_mpg := 32,
  city_mpg := 20,
  highway_fuel_price := 15/4,
  city_fuel_price := 41/10
}

-- Theorem to prove
theorem road_trip_costs :
  let jensen_cost := calculate_total_cost jensen_trip
  let smith_cost := calculate_total_cost smith_trip
  let greens_cost := calculate_total_cost greens_trip
  let total_cost := jensen_cost + smith_cost + greens_cost
  (total_cost = 3375/20) ∧
  (smith_cost > jensen_cost) ∧
  (smith_cost > greens_cost) ∧
  (jensen_cost < greens_cost) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_road_trip_costs_l1045_104560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_student_percentage_l1045_104578

theorem absent_student_percentage
  (total_students : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h1 : total_students = 120)
  (h2 : boys = 72)
  (h3 : girls = 48)
  (h4 : boys_absent_fraction = 1 / 8)
  (h5 : girls_absent_fraction = 1 / 4)
  (h6 : total_students = boys + girls) :
  (((boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students) * 100 : ℚ) = 35 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absent_student_percentage_l1045_104578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_result_l1045_104573

/-- Represents the three people in the quiz -/
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

/-- Represents the score ordering between two people -/
def score_higher (p1 p2 : Person) : Prop := sorry

/-- All scores are different -/
axiom scores_different : ∀ (p1 p2 : Person), p1 ≠ p2 → score_higher p1 p2 ∨ score_higher p2 p1

/-- Only one prediction is correct -/
axiom one_correct_prediction : 
  (score_higher Person.A Person.B) ∨ 
  (score_higher Person.C Person.B ∧ score_higher Person.C Person.A) ∨ 
  (score_higher Person.C Person.B)

/-- The score relation is transitive -/
axiom score_transitive : ∀ (p1 p2 p3 : Person), 
  score_higher p1 p2 → score_higher p2 p3 → score_higher p1 p3

theorem quiz_result : 
  score_higher Person.A Person.B ∧ 
  score_higher Person.B Person.C ∧ 
  score_higher Person.A Person.C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quiz_result_l1045_104573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_is_60_l1045_104500

/-- The number of days it takes A and B to complete the project together -/
noncomputable def project_completion_time : ℝ := 60

/-- The time it takes A to complete half the project alone -/
noncomputable def A_half_project_time (x : ℝ) : ℝ := x - 10

/-- The time it takes B to complete half the project alone -/
noncomputable def B_half_project_time (x : ℝ) : ℝ := x + 15

/-- The combined work rate of A and B -/
noncomputable def combined_work_rate (x : ℝ) : ℝ := 1 / x

/-- The work rate of A alone -/
noncomputable def A_work_rate (x : ℝ) : ℝ := 1 / (2 * (A_half_project_time x))

/-- The work rate of B alone -/
noncomputable def B_work_rate (x : ℝ) : ℝ := 1 / (2 * (B_half_project_time x))

theorem project_completion_time_is_60 :
  A_work_rate project_completion_time + B_work_rate project_completion_time = combined_work_rate project_completion_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_time_is_60_l1045_104500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1045_104503

noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
def b : ℝ × ℝ := (0, -1)
noncomputable def c (k : ℝ) : ℝ × ℝ := (k, Real.sqrt 3)

theorem perpendicular_vectors (k : ℝ) : 
  let v := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  v.1 * (c k).1 + v.2 * (c k).2 = 0 → k = -3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1045_104503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1045_104507

/-- Geometric sequence with given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 * a 2 * a 3 = 8 ∧
  ∀ n : ℕ+, (Finset.sum (Finset.range (2 * n)) (fun i ↦ a (i + 1))) = 
    3 * (Finset.sum (Finset.range n) (fun i ↦ a (2 * i + 1)))

/-- Sum of first n terms of the geometric sequence -/
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := Finset.sum (Finset.range n) (fun i ↦ a (i + 1))

/-- Sequence b_n defined as n * S_n -/
def b (a : ℕ → ℝ) (n : ℕ) : ℝ := n * S a n

/-- Sum of first n terms of sequence b_n -/
def T (a : ℕ → ℝ) (n : ℕ) : ℝ := Finset.sum (Finset.range n) (fun i ↦ b a (i + 1))

theorem geometric_sequence_properties (a : ℕ → ℝ) (h : geometric_sequence a) :
  (∀ n : ℕ, a n = 2^(n - 1)) ∧
  (∀ n : ℕ, T a n = -n * (n + 1) / 2 + 2 + (n - 1) * 2^(n + 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1045_104507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_ratio_l1045_104513

theorem sine_cosine_ratio (x : ℝ) (h1 : 0 < x) (h2 : x < π/2) 
  (h3 : Real.sin x / (1 + Real.cos x) = 1/3) : 
  Real.sin (2*x) / (1 + Real.cos (2*x)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_ratio_l1045_104513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1045_104575

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  (1/2) * t.a * t.b * Real.sin t.C

/-- The condition given in the problem -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a / t.c = Real.cos t.A / (2 - Real.cos t.C) ∧ t.c = 2

/-- The theorem to be proved -/
theorem max_triangle_area (t : Triangle) (h : satisfiesCondition t) :
  ∃ (max_area : ℝ), max_area = 4/3 ∧ 
  ∀ (s : Triangle), satisfiesCondition s → triangleArea s ≤ max_area := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1045_104575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_ahead_l1045_104541

/-- Calculates the distance a participant is ahead in a race -/
noncomputable def distanceAhead (raceDistance : ℝ) (fasterTime slowerTime : ℝ) : ℝ :=
  let fasterSpeed := raceDistance / fasterTime
  fasterSpeed * (slowerTime - fasterTime)

theorem race_distance_ahead (raceDistance : ℝ) (bTime cTime : ℝ) 
  (h_raceDistance : raceDistance = 100)
  (h_bTime : bTime = 45)
  (h_cTime : cTime = 40) :
  distanceAhead raceDistance cTime bTime = 12.5 := by
  -- Unfold the definition of distanceAhead
  unfold distanceAhead
  -- Substitute the known values
  rw [h_raceDistance, h_bTime, h_cTime]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_ahead_l1045_104541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1045_104584

/-- Ellipse D -/
def ellipse_D (x y : ℝ) : Prop := x^2 / 50 + y^2 / 25 = 1

/-- Circle M -/
def circle_M (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 9

/-- Hyperbola G -/
def hyperbola_G (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- Foci of ellipse D -/
def foci_D : ℝ × ℝ × ℝ × ℝ := (-5, 0, 5, 0)

/-- Asymptotes of hyperbola G -/
noncomputable def asymptotes_G (a b : ℝ) : (ℝ → ℝ) × (ℝ → ℝ) :=
  (λ x ↦ (b / a) * x, λ x ↦ -(b / a) * x)

/-- Distance from a point to a line -/
noncomputable def distance_point_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

theorem hyperbola_equation :
  ∀ x y : ℝ,
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    hyperbola_G x y a b ∧
    (∀ x y : ℝ, ellipse_D x y → x^2 + y^2 = 25) ∧
    (let (f₁x, f₁y, f₂x, f₂y) := foci_D; (x - f₁x)^2 + (y - f₁y)^2 - ((x - f₂x)^2 + (y - f₂y)^2) = 4 * a * b) ∧
    (let (l₁, l₂) := asymptotes_G a b;
      distance_point_line 0 5 1 (-a/b) 0 = 3 ∧
      distance_point_line 0 5 1 (a/b) 0 = 3)) →
  x^2 / 9 - y^2 / 16 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l1045_104584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_shape_l1045_104543

/-- The area of the closed shape formed by y = (1/3)x and y = x - x^2 --/
noncomputable def closedShapeArea : ℝ :=
  ∫ x in (0)..(2/3), (x - x^2 - (1/3)*x)

/-- Theorem stating that the area of the closed shape is 4/81 --/
theorem area_of_closed_shape :
  closedShapeArea = 4/81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_closed_shape_l1045_104543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_sum_l1045_104580

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

-- Define the point M
def M : ℝ × ℝ := (-2, 1)

-- Define the right focus F
noncomputable def F : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define point N
def N : ℝ × ℝ := (1, 0)

-- Define the vector product
def vectorProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define t as a function of A and B
def t (A B : ℝ × ℝ) : ℝ :=
  vectorProduct (A.1 - M.1, A.2 - M.2) (B.1 - M.1, B.2 - M.2)

-- State the theorem
theorem ellipse_intersection_sum :
  ∃ (t₁ t₂ : ℝ),
    (∀ (A B : ℝ × ℝ),
      ellipse A.1 A.2 → ellipse B.1 B.2 →
      (∃ (k : ℝ), A.2 - N.2 = k * (A.1 - N.1) ∧ B.2 - N.2 = k * (B.1 - N.1)) →
      t₁ ≤ t A B ∧ t A B ≤ t₂) ∧
    t₁ + t₂ = 13/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_sum_l1045_104580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commodity_price_change_l1045_104551

theorem commodity_price_change (initial_price : ℝ) (x : ℝ) : 
  initial_price > 0 →
  let price1 := initial_price * (1 - 0.15)
  let price2 := price1 * (1 + 0.30)
  let price3 := price2 * (1 - 0.25)
  let price4 := price3 * (1 + 0.10)
  let final_price := price4 * (1 - x / 100)
  final_price = initial_price →
  x = 10 := by
  intro h_positive
  intro h_equal
  -- The proof steps would go here
  sorry

#check commodity_price_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_commodity_price_change_l1045_104551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nasobek_children_ages_l1045_104562

def is_valid_age_set (ages : List ℕ) : Prop :=
  ages.all (λ x => x > 0) ∧
  ages.prod = 1408 ∧
  ages.minimum? = some ((ages.maximum?.getD 0) / 2)

theorem nasobek_children_ages :
  ∀ (ages : List ℕ), is_valid_age_set ages → ages = [8, 11, 16] :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nasobek_children_ages_l1045_104562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l1045_104519

noncomputable def f (a b x : ℝ) : ℝ := x + a * x^2 + b * Real.log x

theorem f_upper_bound (a b : ℝ) :
  (f a b 1 = 0) →
  ((1 + 2 * a + b) = 2) →
  ∀ x > 0, f a b x ≤ 2 * x - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l1045_104519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1045_104594

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - Real.sin x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ π / 2 + 2 * π * ↑k} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1045_104594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1045_104531

noncomputable def f (x : ℝ) := Real.sin x ^ 2 - Real.cos x ^ 2 - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (f (2 * Real.pi / 3) = 2) ∧
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, (0 < S ∧ S < T) → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, ∀ x y : ℝ, k * Real.pi + Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ k * Real.pi + 2 * Real.pi / 3 → f x < f y) ∧
  (∀ x : ℝ, ∃ k : ℤ, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1045_104531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_product_l1045_104557

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (a + b)² - c² = 4 and C = 60°, then ab = 4/3 -/
theorem triangle_side_product (a b c : ℝ) (A B C : ℝ) :
  (a + b)^2 - c^2 = 4 →
  C = Real.pi / 3 →
  a * b = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_product_l1045_104557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_value_equation_l1045_104537

/-- Aggregate sales value of the output of the looms -/
def S : ℝ := 500000

/-- Number of looms -/
def num_looms : ℕ := 80

/-- Monthly manufacturing expenses -/
def manufacturing_expenses : ℝ := 150000

/-- Monthly establishment charges -/
def establishment_charges : ℝ := 75000

/-- Decrease in profit when one loom breaks down -/
def profit_decrease : ℝ := 4375

theorem sales_value_equation :
  profit_decrease = S / num_looms - manufacturing_expenses / num_looms := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_value_equation_l1045_104537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_weight_calculation_l1045_104506

noncomputable def initial_water : ℝ := 20
noncomputable def initial_food : ℝ := 10
noncomputable def initial_gear : ℝ := 20

noncomputable def mountainous_water_rate : ℝ := 3
noncomputable def mountainous_food_rate : ℝ := 1/2
noncomputable def mountainous_duration : ℝ := 2

noncomputable def hilly_water_rate : ℝ := 2
noncomputable def hilly_food_rate : ℝ := 1/3
noncomputable def hilly_duration : ℝ := 2

noncomputable def woodland_water_rate : ℝ := 1.5
noncomputable def woodland_food_rate : ℝ := 1/4
noncomputable def woodland_duration : ℝ := 2

noncomputable def additional_gear : ℝ := 7

theorem hiking_weight_calculation :
  let total_water_consumption := mountainous_water_rate * mountainous_duration +
                                 hilly_water_rate * hilly_duration +
                                 woodland_water_rate * woodland_duration
  let total_food_consumption := mountainous_food_rate * mountainous_duration +
                                hilly_food_rate * hilly_duration +
                                woodland_food_rate * woodland_duration
  let remaining_water := initial_water - total_water_consumption
  let remaining_food := initial_food - total_food_consumption
  let total_weight := remaining_water + remaining_food + initial_gear + additional_gear
  abs (total_weight - 41.83) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_weight_calculation_l1045_104506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1045_104597

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x * (x + 4) else x * (x - 4)

-- State the theorem
theorem f_has_three_zeros :
  ∃ (a b c : ℝ), (f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
  (∀ x : ℝ, f x = 0 → x = a ∨ x = b ∨ x = c) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1045_104597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_implies_k_equals_four_l1045_104510

-- Define the function f(x) as noncomputable
noncomputable def f (k : ℤ) (x : ℝ) : ℝ := Real.log x / Real.log 2 + x - k

-- State the theorem
theorem root_in_interval_implies_k_equals_four :
  ∀ k : ℤ, k ≠ 0 →
  (∃ x : ℝ, 2 < x ∧ x < 3 ∧ f k x = 0) →
  k = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_implies_k_equals_four_l1045_104510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1045_104511

noncomputable def f (x : ℝ) : ℝ := Real.sin (3 * Real.pi / 4 - x) - Real.sqrt 3 * Real.cos (x + Real.pi / 4)

theorem f_properties :
  (∀ x : ℝ, f (x + Real.pi) = f x) ∧
  (∀ x : ℝ, f (Real.pi / 6 - x) = f (Real.pi / 6 + x)) ∧
  (∀ x : ℝ, f x ≤ 2) ∧
  (∃ x : ℝ, f x = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1045_104511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_calculation_l1045_104598

theorem power_calculation (x y : ℕ) (h1 : x - y = 9) (h2 : x = 9) : 
  3^x * 4^y = 19683 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_calculation_l1045_104598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_invariant_l1045_104581

def transform (t : Fin 3 → ℤ) : Fin 3 → ℤ := 
  fun i => t ((i + 1) % 3) + t ((i + 2) % 3)

def initial_triplet : Fin 3 → ℤ := 
  fun i => match i with
  | 0 => 70
  | 1 => 61
  | 2 => 20

def triplet_max (t : Fin 3 → ℤ) : ℤ := 
  max (t 0) (max (t 1) (t 2))

def triplet_min (t : Fin 3 → ℤ) : ℤ := 
  min (t 0) (min (t 1) (t 2))

theorem difference_invariant (n : ℕ) :
  let final_triplet := (Nat.iterate transform n) initial_triplet
  (triplet_max final_triplet - triplet_min final_triplet : ℤ) = 
    (triplet_max initial_triplet - triplet_min initial_triplet : ℤ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_invariant_l1045_104581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1045_104572

theorem rectangle_perimeter (width : ℝ) (length : ℝ) (area : ℝ) (perimeter : ℝ) : 
  area = 400 →
  length = 2 * width →
  area = length * width →
  perimeter = 2 * (length + width) →
  perimeter = 60 * Real.sqrt 2 := by
  intros h1 h2 h3 h4
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l1045_104572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_area_l1045_104521

/-- The area of the L-shaped region in a square of side length 6, 
    with two non-overlapping inner squares of side lengths 2 and 4 --/
theorem l_shaped_area : 
  let outer_square_side : ℝ := 6
  let small_square_side : ℝ := 2
  let medium_square_side : ℝ := 4
  let outer_square_area := outer_square_side ^ 2
  let small_square_area := small_square_side ^ 2
  let medium_square_area := medium_square_side ^ 2
  let l_shaped_area := outer_square_area - (small_square_area + medium_square_area)
  l_shaped_area = 16 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_l_shaped_area_l1045_104521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l1045_104553

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 2) + x * (x + 3) / (x^2 + 3) + 3 * (x + 1) / (x * (x^2 + 3))

-- Theorem statement
theorem g_minimum_value (x : ℝ) (h : x > 0) : g x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_minimum_value_l1045_104553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1045_104515

/-- The eccentricity of an ellipse with semi-major axis a and semi-minor axis b -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - (b^2 / a^2))

/-- The proposition to be proved -/
theorem ellipse_eccentricity_range (a b : ℝ) (M : ℝ × ℝ) (P Q : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (∃ F : ℝ × ℝ, F.2 = 0 ∧ (M.1 - F.1)^2 + M.2^2 = (M.2)^2) ∧
  (P.1 = 0 ∧ Q.1 = 0) ∧
  ((M.1 - P.1)^2 + (M.2 - P.2)^2 = (M.1 - Q.1)^2 + (M.2 - Q.2)^2) ∧
  ((M.1 - P.1)^2 + (M.2 - P.2)^2 > (P.1 - Q.1)^2 + (P.2 - Q.2)^2 / 4) →
  0 < eccentricity a b ∧ eccentricity a b < (Real.sqrt 6 - Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l1045_104515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_adjacent_probability_l1045_104596

theorem not_adjacent_probability (n : ℕ) (h : n = 10) : 
  (Finset.card (Finset.univ : Finset (Fin n × Fin n)) - (n - 1)) / Finset.card (Finset.univ : Finset (Fin n × Fin n)) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_adjacent_probability_l1045_104596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_p_in_S_l1045_104576

def S : Set ℕ+ := {x | ∃ (a b : ℤ), x.val = a^2 + 5*b^2 ∧ Int.gcd a.natAbs b.natAbs = 1}

theorem two_p_in_S (p : ℕ+) (k : ℕ+) (h_prime : Nat.Prime p.val)
  (h_p_form : ∃ (n : ℕ), p.val = 4*n + 3) (h_kp_in_S : k*p ∈ S) : 2*p ∈ S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_p_in_S_l1045_104576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1045_104509

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line: 2ax+(a-1)y+2=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := -2 * a / (a - 1)

/-- The slope of the second line: (a+1)x+3ay+3=0 -/
noncomputable def slope2 (a : ℝ) : ℝ := -(a + 1) / (3 * a)

/-- The statement that a = 1/5 is sufficient but not necessary for perpendicularity -/
theorem sufficient_not_necessary : 
  (∃ (a : ℝ), a ≠ 1/5 ∧ perpendicular (slope1 a) (slope2 a)) ∧ 
  (perpendicular (slope1 (1/5)) (slope2 (1/5))) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_l1045_104509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1045_104536

def S : Set ℚ := {1/3, -22/7, 0, -1, 314/100, 2, -3, -6, 3/10, 23/100}

theorem number_categorization (S : Set ℚ) :
  S = {1/3, -22/7, 0, -1, 314/100, 2, -3, -6, 3/10, 23/100} →
  (S ∩ (Set.range (Int.cast : ℤ → ℚ)) = {0, -1, 2, -3, -6}) ∧
  (S ∩ {x : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b} = {1/3, -22/7, 314/100, 3/10, 23/100}) ∧
  (S ∩ {x : ℚ | x < 0} = {-22/7, -1, -3, -6}) ∧
  (S ∩ {x : ℚ | x ≥ 0 ∧ x ∈ Set.range (Int.cast : ℤ → ℚ)} = {0, 2}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l1045_104536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_problem_l1045_104592

/-- Given a square ABCD with side length s, point G on AD, and H on extended AB such that CH ⊥ CG -/
structure SquareConfiguration (s : ℝ) :=
  (A B C D G H : ℝ × ℝ)
  (is_square : (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧ 
               (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧ 
               (C.1 - D.1)^2 + (C.2 - D.2)^2 = s^2 ∧ 
               (D.1 - A.1)^2 + (D.2 - A.2)^2 = s^2)
  (G_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ G = (1 - t) • A + t • D)
  (H_on_extended_AB : ∃ t : ℝ, t ≥ 1 ∧ H = (1 - t) • A + t • B)
  (CH_perp_CG : (C.1 - H.1) * (C.1 - G.1) + (C.2 - H.2) * (C.2 - G.2) = 0)

/-- The theorem to be proved -/
theorem square_triangle_problem (config : SquareConfiguration 20) 
  (area_CGH : abs ((config.C.1 - config.G.1) * (config.H.2 - config.G.2) - 
                   (config.H.1 - config.G.1) * (config.C.2 - config.G.2)) / 2 = 240) :
  (config.B.1 - config.H.1)^2 + (config.B.2 - config.H.2)^2 = 15^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_triangle_problem_l1045_104592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l1045_104586

-- Define the ellipse parameters
noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3
noncomputable def c : ℝ := 1

-- Define the conditions
axiom a_positive : a > 0
axiom b_positive : b > 0
axiom a_greater_b : a > b
axiom focal_length : c = 1
axiom eccentricity : c / a = 1 / 2

-- Define the line parameter
noncomputable def m : ℝ := 2 * Real.sqrt 42 / 7

-- Define the theorem
theorem ellipse_intersection :
  -- Part 1: Equation of the ellipse
  (a = 2 ∧ b^2 = 3) ∧
  -- Part 2: Value of m for perpendicular chords
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    -- Points (x₁, y₁) and (x₂, y₂) lie on the ellipse
    x₁^2 / 4 + y₁^2 / 3 = 1 ∧
    x₂^2 / 4 + y₂^2 / 3 = 1 ∧
    -- Points lie on the line y = x + m
    y₁ = x₁ + m ∧
    y₂ = x₂ + m ∧
    -- OM ⊥ ON condition
    x₁ * x₂ + y₁ * y₂ = 0 ∧
    -- m takes the specified value
    (m = 2 * Real.sqrt 42 / 7 ∨ m = -2 * Real.sqrt 42 / 7)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_l1045_104586
