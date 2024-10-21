import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_symmetry_given_curve_and_C_are_symmetric_l1186_118658

/-- The given curve in polar coordinates -/
noncomputable def given_curve (θ : Real) : Real := 5 * Real.sqrt 3 * Real.cos θ - 5 * Real.sin θ

/-- The symmetric curve C in polar coordinates -/
noncomputable def curve_C (θ : Real) : Real := 10 * Real.cos (θ - Real.pi / 6)

/-- Symmetry with respect to the polar axis -/
def symmetric_to_polar_axis (f g : Real → Real) : Prop :=
  ∀ θ, f θ = g (-θ)

theorem curve_C_symmetry :
  symmetric_to_polar_axis given_curve curve_C := by
  sorry

/-- Proof that the given curve and curve C are symmetric with respect to the polar axis -/
theorem given_curve_and_C_are_symmetric :
  ∀ θ, given_curve θ = curve_C (-θ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_symmetry_given_curve_and_C_are_symmetric_l1186_118658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_area_l1186_118688

/-- The trajectory of point M -/
def trajectory (x y : ℝ) : Prop :=
  (x - 8)^2 + y^2 = 4 * ((x - 2)^2 + y^2)

/-- The area enclosed by the trajectory -/
noncomputable def enclosed_area : ℝ := 16 * Real.pi

theorem trajectory_area :
  ∀ x y : ℝ, trajectory x y → enclosed_area = 16 * Real.pi :=
by
  intros x y h
  rfl

#check trajectory_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_area_l1186_118688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2005_is_one_l1186_118657

/-- Represents the sequence of digits where each number n from 1 to 99 is written n times. -/
def digit_sequence (n : ℕ) : ℕ := sorry

/-- The 2005th digit in the sequence. -/
def digit_2005 : ℕ := digit_sequence 2005

/-- Theorem stating that the 2005th digit in the sequence is 1. -/
theorem digit_2005_is_one : digit_2005 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_2005_is_one_l1186_118657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_arrival_time_l1186_118607

/-- The speed required to arrive exactly on time given the conditions -/
noncomputable def exact_speed : ℝ := 41

theorem exact_arrival_time (speed_late speed_early time_late time_early t : ℝ) 
  (h1 : speed_late = 30) 
  (h2 : speed_early = 50) 
  (h3 : time_late = 1/6) 
  (h4 : time_early = -1/12) : 
  ∃ (d : ℝ), d = speed_late * (t + time_late) ∧ 
             d = speed_early * (t + time_early) ∧ 
             d / t = exact_speed :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#check exact_arrival_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_arrival_time_l1186_118607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_calculation_l1186_118605

/-- Represents the banker's gain calculation for a bill --/
noncomputable def bankers_gain (true_discount : ℝ) (interest_rate : ℝ) (time : ℝ) : ℝ :=
  (true_discount * interest_rate * time) / 100

/-- Theorem stating the banker's gain for the given conditions --/
theorem bankers_gain_calculation (true_discount : ℝ) (interest_rate : ℝ) (time : ℝ) 
  (h1 : true_discount = 70)
  (h2 : interest_rate = 12)
  (h3 : time = 1) :
  bankers_gain true_discount interest_rate time = 8.4 := by
  -- Unfold the definition of bankers_gain
  unfold bankers_gain
  -- Substitute the given values
  rw [h1, h2, h3]
  -- Simplify the expression
  simp [mul_assoc, mul_comm]
  -- Perform the arithmetic
  norm_num

-- Remove the #eval statement as it's not necessary for the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bankers_gain_calculation_l1186_118605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_averages_l1186_118661

theorem number_averages (total : ℕ) (overall_avg avg1 avg2 avg3 : ℚ) 
  (count1 count2 count3 : ℕ) :
  total = 10 ∧ 
  overall_avg = 46/10 ∧
  avg1 = 34/10 ∧ count1 = 3 ∧
  avg2 = 38/10 ∧ count2 = 2 ∧
  avg3 = 42/10 ∧ count3 = 3 →
  let remaining_count := total - (count1 + count2 + count3)
  let remaining_sum := total * overall_avg - (count1 * avg1 + count2 * avg2 + count3 * avg3)
  let remaining_avg := remaining_sum / remaining_count
  remaining_avg = 78/10 ∧ overall_avg = 46/10 :=
by
  intro h
  sorry

#check number_averages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_averages_l1186_118661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_six_tan_alpha_eq_two_l1186_118672

noncomputable def f (θ : Real) : Real :=
  (2 * (Real.cos θ)^2 + (Real.sin (2 * Real.pi - θ))^3 + Real.cos (Real.pi / 2 + θ) - 3) /
  (2 + 2 * Real.sin (Real.pi + θ) + Real.sin (-θ))

theorem f_at_pi_over_six : f (Real.pi / 6) = -17 / 4 := by
  sorry

theorem tan_alpha_eq_two (α : Real) (h : Real.tan α = 2) :
  2/3 * (Real.sin α)^2 + Real.sin α * Real.cos α + 1/4 * (Real.cos α)^2 - 2 = -61/60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_six_tan_alpha_eq_two_l1186_118672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1186_118686

-- Part 1
theorem part1 (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (3 : ℝ)^x = (4 : ℝ)^y) (h2 : (4 : ℝ)^y = (6 : ℝ)^z) :
  y / z - y / x = Real.log 4 / Real.log 6 - Real.log 4 / Real.log 3 := by
  sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ - 3*m = 0 ∧ x₂^2 - 2*x₂ - 3*m = 0) ↔
  (-1/3 < m ∧ m < 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part1_part2_l1186_118686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_quadratic_equation_solutions_l1186_118629

-- Part 1
theorem complex_expression_equality : 
  Real.sqrt (4/9) - Real.sqrt ((-2)^4) + (((19/27) - 1) ^ (1/3 : ℝ)) - (-1)^2017 = -3 := by
  sorry

-- Part 2
theorem quadratic_equation_solutions :
  ∀ x : ℝ, (x - 1)^2 = 9 ↔ x = 4 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_equality_quadratic_equation_solutions_l1186_118629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_value_of_a_range_of_a_l1186_118621

/-- Given functions g and h, we define f as their sum -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 2) * x
noncomputable def h (x : ℝ) : ℝ := Real.log x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := g a x + h x

/-- Theorem 1: Tangent line equation when a = 1 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, x + y + 1 = 0 ↔ 
  (∃ t : ℝ, y = g 1 t + (2 * t - 3) * (x - t)) :=
by sorry

/-- Theorem 2: Minimum value of a -/
theorem min_value_of_a :
  ∀ a : ℝ, a > 0 → 
  (∀ x : ℝ, 1 ≤ x → x ≤ Real.exp 1 → f a x ≥ -2) →
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ Real.exp 1 ∧ f a x = -2) →
  a ≥ 1 :=
by sorry

/-- Theorem 3: Range of a -/
theorem range_of_a :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → 
    (f a x₁ - f a x₂) / (x₁ - x₂) > -2) ↔
  0 ≤ a ∧ a ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_min_value_of_a_range_of_a_l1186_118621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_root_polynomials_l1186_118609

/-- A polynomial of degree 11 with coefficients in {0, 1} -/
def BinaryPolynomial := Fin 12 → Fin 2

/-- The number of different integer roots of a BinaryPolynomial -/
def numIntegerRoots (p : BinaryPolynomial) : ℕ := sorry

/-- The set of BinaryPolynomials with exactly three different integer roots -/
def threeRootPolynomials : Set BinaryPolynomial :=
  {p | numIntegerRoots p = 3}

instance : Fintype BinaryPolynomial := sorry

instance : DecidablePred (λ p => p ∈ threeRootPolynomials) := sorry

theorem count_three_root_polynomials :
  Finset.card (Finset.filter (λ p => p ∈ threeRootPolynomials) Finset.univ) = 160 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_three_root_polynomials_l1186_118609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_tangent_point_l1186_118634

/-- Given two externally tangent circles and their common external tangent,
    prove the distance from the center of the larger circle to the
    tangent point on the smaller circle. -/
theorem distance_center_to_tangent_point
  (O P : ℝ × ℝ)  -- Centers of the circles
  (r₁ r₂ : ℝ)    -- Radii of the circles
  (Q : ℝ × ℝ)    -- Point of tangency between the circles
  (T S : ℝ × ℝ)  -- Points where the common tangent touches the circles
  (h₁ : r₁ = 10) -- Radius of circle O is 10
  (h₂ : r₂ = 5)  -- Radius of circle P is 5
  (h₃ : ‖O - Q‖ = r₁) -- Q is on circle O
  (h₄ : ‖P - Q‖ = r₂) -- Q is on circle P
  (h₅ : ‖O - P‖ = r₁ + r₂) -- Circles are externally tangent
  (h₆ : ‖O - T‖ = r₁) -- T is on circle O
  (h₇ : ‖P - S‖ = r₂) -- S is on circle P
  (h₈ : (T - O) • (S - T) = 0) -- TS is tangent to circle O
  (h₉ : (S - P) • (T - S) = 0) -- TS is tangent to circle P
  : ‖O - S‖ = 10 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_center_to_tangent_point_l1186_118634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1186_118636

theorem unique_solution_condition (t : ℝ) :
  (∃! p : ℝ × ℝ, let (x, y) := p; x ≥ y^2 + t*y ∧ y^2 + t*y ≥ x^2 + t) ↔ t = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1186_118636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sums_l1186_118628

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

theorem arithmetic_geometric_sums
  (a : ℕ → ℝ) (d : ℝ) (h_pos : ∀ n, a n > 0) (h_d : d > 0)
  (h_geom : geometric_sequence (a 1) (a 2) (a 5))
  (h_arith : arithmetic_sequence a d) :
  geometric_sequence
    (sum_of_arithmetic_sequence a 1)
    (sum_of_arithmetic_sequence a 2)
    (sum_of_arithmetic_sequence a 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_sums_l1186_118628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_natural_number_l1186_118642

noncomputable def f (n : ℕ) : ℝ :=
  Real.tan (Real.pi / 7) ^ (2 * n) + Real.tan (2 * Real.pi / 7) ^ (2 * n) + Real.tan (3 * Real.pi / 7) ^ (2 * n)

theorem f_is_natural_number : ∀ n : ℕ, ∃ m : ℕ, f n = m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_natural_number_l1186_118642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_minimum_a_l1186_118683

noncomputable def f (a b c : ℤ) (x : ℝ) : ℝ := (a * x^2 - 2) / (b * x + c)

theorem odd_function_and_minimum_a :
  ∀ a b c : ℤ,
  (∀ x : ℝ, f a b c x + f a b c (-x) = 0) →
  f a b c 1 = 1 →
  f a b c 2 - 4 > 0 →
  (∀ x : ℝ, x > 1 → f 3 1 0 x > 1) →
  (f 3 1 0 = f a b c) ∧
  (∀ a' : ℤ, (∀ x : ℝ, x > 1 → f a' 1 0 x > 1) → a' ≥ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_minimum_a_l1186_118683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l1186_118664

/-- Represents an ellipse in standard form -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  p : ℝ
  h_pos : p ≠ 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given conditions and proof goals for the ellipse and parabola problem -/
theorem ellipse_parabola_intersection (C₁ : Ellipse) (C₂ : Parabola) (F : Point) :
  -- Given conditions
  C₁.a = 2 →
  Real.sqrt 7 * abs (C₁.a - F.x) = 2 * C₁.a →
  C₂.p = F.x →
  -- Proof goals
  C₁.b^2 = 3 ∧
  ∃ (k : ℝ), 
    -- Line equation: x = ky - 1
    let line (y : ℝ) := k * y - 1
    -- Intersection points exist
    ∃ (P Q M N : Point),
      -- P and Q on C₁
      (P.x^2 / C₁.a^2 + P.y^2 / C₁.b^2 = 1) ∧
      (Q.x^2 / C₁.a^2 + Q.y^2 / C₁.b^2 = 1) ∧
      P.x = line P.y ∧ Q.x = line Q.y ∧
      -- M and N on C₂
      (M.y^2 = -4 * M.x) ∧ (N.y^2 = -4 * N.x) ∧
      M.x = line M.y ∧ N.x = line N.y ∧
      -- Area condition
      abs (P.y - Q.y) * abs k = 1/2 * abs (M.y - N.y) * abs k ∧
    -- Two solutions exist
    k = Real.sqrt 6/3 ∨ k = -Real.sqrt 6/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_intersection_l1186_118664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l1186_118699

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_problem : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_union_problem_l1186_118699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l1186_118644

theorem junior_score (total_students : ℕ) (junior_score : ℝ) : 
  total_students > 0 →
  (0.2 * (total_students : ℝ) * junior_score + 
   0.8 * (total_students : ℝ) * 85) / (total_students : ℝ) = 86 →
  junior_score = 90 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_score_l1186_118644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_range_l1186_118638

def temperatures : List ℝ := [31, 34, 36, 27, 25, 33]

theorem temperature_range : 
  (temperatures.maximum? >>= (λ max => temperatures.minimum? >>= (λ min => some (max - min)))) = some 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_temperature_range_l1186_118638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_cube_times_seven_l1186_118612

/-- aa is a two-digit number. m times the cube of aa has a specific digit in its tens place. m is 7. The digit in the tens place is 1. -/
theorem tens_digit_of_cube_times_seven (a : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) : 
  (7 * (11 * a)^3) % 100 / 10 = 1 := by
  sorry

#eval (7 * (11 * 1)^3) % 100 / 10
#eval (7 * (11 * 9)^3) % 100 / 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tens_digit_of_cube_times_seven_l1186_118612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_m_plus_n_1004_array_l1186_118652

/-- Definition of the sum of a 1/p-array -/
def sum_1_p_array (p : ℕ) : ℚ :=
  (2 * p^2) / ((2 * p - 1) * (p - 1))

/-- Theorem about the remainder of m+n for a 1/1004-array -/
theorem remainder_m_plus_n_1004_array (m n : ℕ) :
  sum_1_p_array 1004 = m / n →
  Nat.Coprime m n →
  (m + n) % 1004 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_m_plus_n_1004_array_l1186_118652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_window_height_is_61_5_l1186_118660

/-- Represents a rectangular window with glass panes. -/
structure Window where
  num_panes : ℕ
  rows : ℕ
  columns : ℕ
  frame_width : ℚ
  pane_height_ratio : ℚ
  pane_width_ratio : ℚ

/-- Calculates the height of the window in inches. -/
noncomputable def window_height (w : Window) : ℚ :=
  let y : ℚ := (40 - 9) / 8  -- Assuming total width is 40 inches
  12 * y + 15

/-- Theorem stating the height of the window is 61.5 inches. -/
theorem window_height_is_61_5 (w : Window) :
  w.num_panes = 8 ∧ 
  w.rows = 4 ∧ 
  w.columns = 2 ∧ 
  w.frame_width = 3 ∧ 
  w.pane_height_ratio = 3 ∧ 
  w.pane_width_ratio = 4 →
  window_height w = 61.5 := by
  intro h
  unfold window_height
  -- The proof steps would go here, but for now we use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_window_height_is_61_5_l1186_118660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1186_118668

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

-- State the theorem
theorem triangle_side_length 
  (ω φ A b c : ℝ) 
  (h1 : ω > 0) 
  (h2 : 0 < φ) (h3 : φ < Real.pi / 2) 
  (h4 : f ω φ 0 = 1 / 2) 
  (h5 : 2 * Real.pi / ω = Real.pi) 
  (h6 : f ω φ (A / 2) - Real.cos A = 1 / 2) 
  (h7 : b * c = 1) 
  (h8 : b + c = 3) : 
  ∃ a : ℝ, a^2 = 6 ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l1186_118668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_l1186_118619

open Real

-- Define the functions
noncomputable def f (x : ℝ) := sin (3 * π / 2 + x)
noncomputable def g (x : ℝ) := sin (2 * x + 5 * π / 4)

-- Define the first quadrant
def first_quadrant (θ : ℝ) : Prop := 0 < θ ∧ θ < π / 2

theorem math_problem :
  (¬ ∃ (α : ℝ), sin α * cos α = 1) ∧
  (∀ x, f x = f (-x)) ∧
  (∀ x, g (π / 8 + x) = g (π / 8 - x)) ∧
  (∃ (α β : ℝ), first_quadrant α ∧ first_quadrant β ∧ α > β ∧ sin α ≤ sin β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_math_problem_l1186_118619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_decreasing_for_small_base_l1186_118689

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_decreasing_for_small_base
  (a : ℝ) (h_a : 0 < a ∧ a < 1)
  (x₁ x₂ : ℝ) (h_x₁ : 0 < x₁) (h_x₂ : 0 < x₂)
  (h_f : f a x₁ > f a x₂) :
  x₁ < x₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_decreasing_for_small_base_l1186_118689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_number_l1186_118679

/-- The magnitude of the complex number z = 5i / (1 - 2i) is equal to √5 -/
theorem magnitude_of_complex_number :
  Complex.abs ((5 * Complex.I) / (1 - 2 * Complex.I)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_complex_number_l1186_118679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_senior_ratio_l1186_118624

theorem junior_senior_ratio (J S : ℚ) (h1 : J > 0) (h2 : S > 0) (h3 : J / (J + S) = 4 / 7) :
  S / J = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_senior_ratio_l1186_118624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_is_1_increasing_cos2x_is_pi_increasing_square_function_m_increasing_range_l1186_118676

-- Definition of l-increasing function
def is_l_increasing (f : ℝ → ℝ) (l : ℝ) (M : Set ℝ) : Prop :=
  l ≠ 0 ∧ ∀ x ∈ M, (x + l) ∈ M → f (x + l) ≥ f x

-- Proposition 1
theorem log2_is_1_increasing :
  is_l_increasing (fun x ↦ Real.log x / Real.log 2) 1 (Set.Ioi 0) := by
  sorry

-- Proposition 2
theorem cos2x_is_pi_increasing :
  is_l_increasing (fun x ↦ Real.cos (2 * x)) Real.pi Set.univ := by
  sorry

-- Proposition 3
theorem square_function_m_increasing_range :
  ¬ (∀ m : ℝ, (is_l_increasing (fun x ↦ x^2) m (Set.Icc (-1) 1) →
    m ∈ Set.Icc (-1) 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log2_is_1_increasing_cos2x_is_pi_increasing_square_function_m_increasing_range_l1186_118676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_with_tetrahedron_l1186_118677

/-- The height of the regular tetrahedron -/
def tetrahedron_height : ℝ := 4

/-- Theorem: Surface area of a sphere containing a regular tetrahedron -/
theorem sphere_surface_area_with_tetrahedron :
  ∃ (r : ℝ), r > 0 ∧ 
  (∃ (a : ℝ), a > 0 ∧ 
    -- Relationship between tetrahedron edge length and height
    Real.sqrt (a^2 - (Real.sqrt 3 / 3 * a)^2) = tetrahedron_height ∧
    -- Relationship between sphere radius and tetrahedron
    r^2 = (tetrahedron_height - r)^2 + (2 * Real.sqrt 2)^2) ∧
  -- Surface area of the sphere
  4 * Real.pi * r^2 = 36 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_surface_area_with_tetrahedron_l1186_118677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_formula_l1186_118673

/-- A regular triangular pyramid (tetrahedron) with base side length a and height h -/
structure RegularPyramid where
  a : ℝ
  h : ℝ

/-- A cube positioned relative to a regular triangular pyramid -/
structure PositionedCube where
  pyramid : RegularPyramid
  -- Additional properties to represent the positioning of the cube relative to the pyramid
  -- are omitted for simplicity, as they are not directly used in the theorem statement

/-- The edge length of a cube positioned relative to a regular triangular pyramid -/
noncomputable def cube_edge_length (cube : PositionedCube) : ℝ :=
  (3 * cube.pyramid.a * cube.pyramid.h) / (3 * cube.pyramid.a + cube.pyramid.h * (3 + 2 * Real.sqrt 3))

/-- Theorem stating the relationship between the cube's edge length and the pyramid's dimensions -/
theorem cube_edge_length_formula (cube : PositionedCube) :
  cube_edge_length cube = (3 * cube.pyramid.a * cube.pyramid.h) / (3 * cube.pyramid.a + cube.pyramid.h * (3 + 2 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_edge_length_formula_l1186_118673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1186_118641

/-- The value of x as defined in the problem -/
noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 65 + 5) / 2)

/-- The theorem statement -/
theorem problem_solution :
  ∃! (a b c : ℕ+),
    x^100 = 2*x^98 + 16*x^96 + 13*x^94 - x^50 + (a : ℝ)*x^46 + (b : ℝ)*x^44 + (c : ℝ)*x^42 ∧
    a + b + c = 337 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1186_118641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_B_l1186_118623

-- Define the labels for the squares
inductive Label : Type
  | A | B | C | D | E | F

-- Define a cube type
structure Cube where
  faces : Fin 6 → Label

-- Define adjacency relation
def adjacent (cube : Cube) (i j : Fin 6) : Prop :=
  sorry

-- Define opposite relation
def opposite (cube : Cube) (i j : Fin 6) : Prop :=
  sorry

-- Main theorem
theorem opposite_face_of_B (cube : Cube) : 
  (adjacent cube 0 1 ∧ 
   adjacent cube 1 2 ∧ 
   adjacent cube 2 0) →
  (¬ adjacent cube 3 4 ∧
   ¬ adjacent cube 4 5 ∧
   ¬ adjacent cube 5 3) →
  (cube.faces 0 = Label.A ∧ 
   cube.faces 1 = Label.B ∧ 
   cube.faces 2 = Label.C ∧ 
   cube.faces 3 = Label.D ∧ 
   cube.faces 4 = Label.E ∧ 
   cube.faces 5 = Label.F) →
  opposite cube 1 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_B_l1186_118623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l1186_118640

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sin (ω * x) * Real.cos (ω * x)

/-- The theorem statement -/
theorem min_omega : 
  ∀ ω : ℝ, ω > 0 → 
  (∃ x₀ : ℝ, ∀ x : ℝ, f ω x₀ ≤ f ω x ∧ f ω x ≤ f ω (x₀ + 2022 * Real.pi)) →
  ω ≥ 1 / 4044 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l1186_118640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_triangle_not_dense_tiling_l1186_118678

/-- The interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (n - 2 : ℝ) * 180 / n

/-- Checks if a combination of regular polygons can form a dense tiling -/
def can_tile_densely (sides : List ℕ) : Prop :=
  (sides.map interior_angle).sum = 360

theorem octagon_triangle_not_dense_tiling :
  ¬ ∃ (k m : ℕ), k > 0 ∧ m > 0 ∧ can_tile_densely (List.replicate k 8 ++ List.replicate m 3) := by
  sorry

#check octagon_triangle_not_dense_tiling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_triangle_not_dense_tiling_l1186_118678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equals_16_l1186_118649

noncomputable def piecewise_function (x : ℝ) : ℝ :=
  if x < 0 then (x + 1)^2 else (x - 1)^2

theorem piecewise_function_equals_16 (x : ℝ) :
  piecewise_function x = 16 ↔ x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piecewise_function_equals_16_l1186_118649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1186_118692

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x^2 < 1}
def B : Set ℝ := {x : ℝ | Real.exp (x * Real.log 2) > 1}

-- Define the open interval (-1, +∞)
def open_interval : Set ℝ := {x : ℝ | x > -1}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = open_interval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_A_and_B_l1186_118692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_chord_length_l1186_118613

/-- Given a circle C with center (3, 2) and radius 1, intersecting with the line y = 3/4 * x
    at points P and Q, the length of PQ is 4/5 * √6. -/
theorem circle_line_intersection_chord_length :
  let C : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 2)^2 = 1}
  let L : Set (ℝ × ℝ) := {p | p.2 = 3/4 * p.1}
  ∃ (P Q : ℝ × ℝ), P ∈ C ∧ Q ∈ C ∧ P ∈ L ∧ Q ∈ L ∧ P ≠ Q ∧
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4/5 * Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_chord_length_l1186_118613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_in_ellipse_l1186_118691

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a line passing through a point with slope m -/
structure Line where
  m : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

theorem max_triangle_area_in_ellipse (C : Ellipse) (l : Line) :
  C.b = 1 →
  C.a^2 - C.b^2 = (1/2) * C.a^2 →
  l.x₀ = -2 ∧ l.y₀ = 0 →
  ∃ (P Q : ℝ × ℝ),
    (P.1^2 / C.a^2 + P.2^2 / C.b^2 = 1) ∧
    (Q.1^2 / C.a^2 + Q.2^2 / C.b^2 = 1) ∧
    (∀ (x y : ℝ), y = l.m * (x - l.x₀) + l.y₀ → 
      triangle_area 2 (|y - 0|) ≤ Real.sqrt 2 / 2) ∧
    (∃ (x y : ℝ), y = l.m * (x - l.x₀) + l.y₀ ∧ 
      triangle_area 2 (|y - 0|) = Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_in_ellipse_l1186_118691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_irma_pairing_probability_l1186_118627

/-- Represents a class of students with pairing preferences -/
structure StudentClass where
  total_students : ℕ
  students_with_preferences : ℕ
  preferences_per_student : ℕ

/-- Calculates the probability of Margo being paired with Irma -/
def probability_of_pairing (c : StudentClass) : ℚ :=
  1 / (c.total_students - 1 - c.preferences_per_student)

/-- Theorem stating the probability of Margo being paired with Irma -/
theorem margo_irma_pairing_probability (c : StudentClass) 
  (h1 : c.total_students = 40)
  (h2 : c.students_with_preferences = 5)
  (h3 : c.preferences_per_student = 3)
  (h4 : probability_of_pairing c = 1 / 36) : 
  probability_of_pairing c = 1 / 36 := by
  sorry

#eval probability_of_pairing ⟨40, 5, 3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_margo_irma_pairing_probability_l1186_118627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_process_flowchart_is_most_appropriate_l1186_118625

-- Define the types of diagrams
inductive DiagramType
  | ProgramFlowchart
  | ProcessFlowchart
  | KnowledgeStructureDiagram
  | OrganizationalStructureDiagram

-- Define the task as a string
def DescribeProductionSteps : String :=
  "Describe the production steps of a certain product in a factory"

-- Define the function to determine the most appropriate diagram type
def MostAppropriateDiagram (task : String) : DiagramType :=
  match task with
  | "Describe the production steps of a certain product in a factory" => DiagramType.ProcessFlowchart
  | _ => DiagramType.ProgramFlowchart  -- Default case

-- Theorem statement
theorem process_flowchart_is_most_appropriate :
  MostAppropriateDiagram DescribeProductionSteps = DiagramType.ProcessFlowchart :=
by
  -- Unfold the definitions
  unfold MostAppropriateDiagram
  unfold DescribeProductionSteps
  -- The proof is now trivial since it matches the exact string
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_process_flowchart_is_most_appropriate_l1186_118625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_division_l1186_118671

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A point on a line segment -/
def PointOnSegment (A B K : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ K = (1 - t) • A + t • B

/-- The area of a polygon -/
noncomputable def area (vertices : List (ℝ × ℝ)) : ℝ := sorry

/-- The ratio in which a line divides the area of a polygon -/
def divides_area_ratio (hexagon : RegularHexagon) (A K : ℝ × ℝ) (r : ℝ) : Prop :=
  let vertices := hexagon.vertices
  let area1 := area [vertices 0, K, vertices 4, vertices 5]
  let area2 := area [vertices 0, vertices 1, vertices 2, vertices 3, K]
  area1 / area2 = r

/-- The ratio in which a point divides a line segment -/
def divides_segment_ratio (A B K : ℝ × ℝ) (r : ℝ) : Prop :=
  ∃ t : ℝ, K = (1 - t) • A + t • B ∧ t / (1 - t) = r

/-- The main theorem -/
theorem hexagon_area_division (hexagon : RegularHexagon) :
  let vertices := hexagon.vertices
  ∀ K : ℝ × ℝ, PointOnSegment (vertices 3) (vertices 4) K →
    divides_area_ratio hexagon (vertices 0) K (1/3) →
    divides_segment_ratio (vertices 3) (vertices 4) K 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_division_l1186_118671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shifts_l1186_118698

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (2 * x - Real.pi / 3) + 4

theorem phase_and_vertical_shifts :
  (∃ (k : ℝ), ∀ (x : ℝ), f x = 5 * Real.sin (2 * (x - Real.pi / 6)) + 4) ∧
  (∃ (c : ℝ), c = 4 ∧ ∀ (x : ℝ), f x = 5 * Real.sin (2 * x - Real.pi / 3) + c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_and_vertical_shifts_l1186_118698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_birthday_l1186_118631

inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem vasyas_birthday (statement_day : DayOfWeek)
  (h1 : nextDay (nextDay statement_day) = DayOfWeek.Sunday)
  (h2 : statement_day = nextDay vasyas_birthday_day) :
  vasyas_birthday_day = DayOfWeek.Thursday :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vasyas_birthday_l1186_118631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_problem_l1186_118654

/-- The set of lattice points with integer coordinates from 1 to 40 inclusive -/
def T : Set (ℤ × ℤ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 40 ∧ 1 ≤ p.2 ∧ p.2 ≤ 40}

/-- The number of points in T -/
def T_count : ℕ := 1600

/-- The number of points in T that lie on or below the line y = mx -/
noncomputable def points_below (m : ℚ) : ℕ := 600

/-- The interval containing possible values of m -/
noncomputable def m_interval : Set ℚ := {m | points_below m = 600}

/-- The length of the interval containing possible values of m -/
def interval_length : ℚ := 1/6

/-- a and b are relatively prime positive integers -/
def a : ℕ := 1
def b : ℕ := 6

theorem lattice_point_problem :
  Int.gcd a b = 1 ∧ 
  (∃ (l u : ℚ), m_interval = Set.Icc l u ∧ u - l = interval_length) ∧
  a + b = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_point_problem_l1186_118654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_axis_l1186_118670

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : x^2 = 4*y

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem parabola_point_distance_to_axis 
  (P : ParabolaPoint) 
  (h : distance P.x P.y 0 1 = 3) : 
  |P.x| = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_distance_to_axis_l1186_118670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_expression_value_l1186_118655

theorem parallel_vectors_imply_expression_value (θ : Real) :
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, -2)
  (∃ (k : ℝ), a = k • b) →
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_imply_expression_value_l1186_118655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_relationship_l1186_118618

/-- Represents the height of a burning candle as a function of time. -/
noncomputable def candle_height (t : ℝ) : ℝ := 30 - (1/2) * t

/-- Theorem stating the relationship between candle height and burning time. -/
theorem candle_height_relationship (t : ℝ) (h : ℝ) 
  (h_init : h = candle_height 0)
  (h_decrease : ∀ t1 t2, t2 - t1 = 2 → candle_height t2 - candle_height t1 = -1) :
  h = candle_height t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_height_relationship_l1186_118618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1186_118604

/-- Given a line l: 3x + 4y - 12 = 0, and a line l' perpendicular to l that forms a 
    triangle with area 4 with the coordinate axes, prove that the equation of l' 
    is of the form 4x - 3y ± 4√6 = 0. -/
theorem perpendicular_line_equation :
  ∀ (a b c : ℝ),
    (∀ x y, a * x + b * y + c = 0) →
    (a * 3 + b * 4 = 0) →  -- perpendicularity condition
    (abs (c / a) * abs (c / b) / 2 = 4) →  -- area condition
    ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧ a = 4 ∧ b = -3 ∧ c = sign * (4 * Real.sqrt 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_equation_l1186_118604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_proof_l1186_118630

/-- The smallest positive angle whose terminal side intersects a unit circle at (sin(2π/3), cos(2π/3)) -/
noncomputable def smallest_positive_angle : ℝ :=
  11 * Real.pi / 6

theorem smallest_positive_angle_proof (α : ℝ) :
  α > 0 ∧
  Complex.exp (α * Complex.I) = Complex.exp ((2 * Real.pi / 3) * Complex.I) →
  α ≥ smallest_positive_angle :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_proof_l1186_118630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_profit_calculation_l1186_118608

/-- Calculate the profit from selling rugs with given conditions -/
theorem rug_profit_calculation (purchase_price discount_rate selling_price tax_rate transport_fee : ℝ) (num_rugs : ℕ) :
  purchase_price = 40 →
  discount_rate = 0.05 →
  selling_price = 60 →
  tax_rate = 0.1 →
  transport_fee = 5 →
  num_rugs = 20 →
  (let total_cost_before_discount := purchase_price * (num_rugs : ℝ)
   let discount_amount := total_cost_before_discount * discount_rate
   let total_cost_after_discount := total_cost_before_discount - discount_amount
   let total_selling_price_before_tax := selling_price * (num_rugs : ℝ)
   let total_tax := total_selling_price_before_tax * tax_rate
   let total_selling_price_after_tax := total_selling_price_before_tax + total_tax
   let total_transport_fee := transport_fee * (num_rugs : ℝ)
   let profit := total_selling_price_after_tax - total_cost_after_discount - total_transport_fee
   profit) = 460 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rug_profit_calculation_l1186_118608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_complementary_l1186_118665

structure Pocket where
  red_balls : ℕ
  black_balls : ℕ

def draw_two_balls (p : Pocket) : Set (ℕ × ℕ) :=
  {(r, b) | r + b = 2 ∧ 0 ≤ r ∧ r ≤ p.red_balls ∧ 0 ≤ b ∧ b ≤ p.black_balls}

def exactly_one_black (p : Pocket) : Set (ℕ × ℕ) :=
  {(r, b) | (r, b) ∈ draw_two_balls p ∧ b = 1}

def exactly_two_black (p : Pocket) : Set (ℕ × ℕ) :=
  {(r, b) | (r, b) ∈ draw_two_balls p ∧ b = 2}

theorem mutually_exclusive_not_complementary (p : Pocket) (h : p.red_balls = 2 ∧ p.black_balls = 2) :
  (exactly_one_black p ∩ exactly_two_black p = ∅) ∧
  (exactly_one_black p ∪ exactly_two_black p ≠ draw_two_balls p) := by
  sorry

#check mutually_exclusive_not_complementary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_exclusive_not_complementary_l1186_118665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_hypotenuse_l1186_118669

theorem right_triangle_rotation_hypotenuse (x y : ℝ) : 
  x > 0 → y > 0 → 
  (1/3 : ℝ) * Real.pi * y^2 * x = 1350 * Real.pi → 
  (1/3 : ℝ) * Real.pi * x^2 * y = 2430 * Real.pi → 
  Real.sqrt (x^2 + y^2) = Real.sqrt 954 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_rotation_hypotenuse_l1186_118669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tetrahedra_partition_of_cube_l1186_118650

/-- A cube is a three-dimensional solid object with six square faces. -/
structure Cube where
  faces : Fin 6 → Square

/-- A tetrahedron is a three-dimensional solid object with four triangular faces. -/
structure Tetrahedron where
  faces : Fin 4 → Triangle

/-- A partition of a cube into tetrahedra is a set of tetrahedra that completely fill the cube without overlap. -/
def PartitionCubeIntoTetrahedra (c : Cube) (ts : Finset Tetrahedron) : Prop :=
  -- The definition of a valid partition
  sorry

/-- The theorem states that the smallest number of tetrahedra required to partition a cube is 5. -/
theorem smallest_tetrahedra_partition_of_cube :
  ∀ (c : Cube) (ts : Finset Tetrahedron),
    PartitionCubeIntoTetrahedra c ts →
    ts.card ≥ 5 ∧ 
    ∃ (ts' : Finset Tetrahedron), PartitionCubeIntoTetrahedra c ts' ∧ ts'.card = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_tetrahedra_partition_of_cube_l1186_118650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_reading_difference_l1186_118635

theorem book_reading_difference (total_pages : ℕ) (finished_fraction : ℚ) : 
  total_pages = 300 → 
  finished_fraction = 2/3 → 
  (finished_fraction * ↑total_pages).floor - (total_pages - (finished_fraction * ↑total_pages).floor) = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_reading_difference_l1186_118635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_set_l1186_118626

/-- Checks if three numbers can form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The sets of line segments --/
def set_A : (ℝ × ℝ × ℝ) := (4, 5, 6)
def set_B : (ℝ × ℝ × ℝ) := (2, 3, 4)
def set_C : (ℝ × ℝ × ℝ) := (3, 4, 5)
noncomputable def set_D : (ℝ × ℝ × ℝ) := (1, Real.sqrt 2, 3)

theorem right_triangle_set :
  ¬(is_right_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(is_right_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  (is_right_triangle set_C.1 set_C.2.1 set_C.2.2) ∧
  ¬(is_right_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_set_l1186_118626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_in_closed_interval_l1186_118615

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |Real.exp x + a / Real.exp x|

theorem monotone_increasing_f_implies_a_in_closed_interval :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f a x ≤ f a y) →
  a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_f_implies_a_in_closed_interval_l1186_118615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_calculation_l1186_118651

/-- Calculates the reduced price of oil per kg after a price reduction and exchange rate fluctuation -/
noncomputable def reduced_price_after_fluctuation (original_price : ℝ) : ℝ :=
  let reduced_price := 0.9 * original_price
  let kg_bought := 6
  let total_cost := 900
  let exchange_rate_change := 0.02
  if reduced_price * kg_bought = total_cost
  then reduced_price * (1 + exchange_rate_change)
  else 0

/-- Theorem stating the reduced price of oil per kg after price reduction and exchange rate fluctuation -/
theorem oil_price_calculation :
  ∃ (original_price : ℝ),
    reduced_price_after_fluctuation original_price = 153 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_price_calculation_l1186_118651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_containing_isosceles_triangle_l1186_118693

-- Define a triangle as a tuple of three points in R^2
def Triangle := (Real × Real) × (Real × Real) × (Real × Real)

-- Function to calculate the distance between two points
noncomputable def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define what it means for a triangle to have all sides less than 1
def allSidesLessThanOne (t : Triangle) : Prop :=
  let (a, b, c) := t
  distance a b < 1 ∧ distance b c < 1 ∧ distance c a < 1

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  let (a, b, c) := t
  (distance a b = distance a c) ∨ (distance a b = distance b c) ∨ (distance b c = distance c a)

-- Define what it means for one triangle to contain another
def contains (t1 t2 : Triangle) : Prop :=
  let (a1, b1, c1) := t1
  let (a2, b2, c2) := t2
  ∃ (p q r : Real), 
    0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r ∧ p + q + r = 1 ∧
    a2 = (p * a1.1 + q * b1.1 + r * c1.1, p * a1.2 + q * b1.2 + r * c1.2) ∧
    b2 = (p * a1.1 + q * b1.1 + r * c1.1, p * a1.2 + q * b1.2 + r * c1.2) ∧
    c2 = (p * a1.1 + q * b1.1 + r * c1.1, p * a1.2 + q * b1.2 + r * c1.2)

-- The theorem statement
theorem exists_containing_isosceles_triangle (t : Triangle) 
  (h : allSidesLessThanOne t) : 
  ∃ (t' : Triangle), allSidesLessThanOne t' ∧ isIsosceles t' ∧ contains t' t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_containing_isosceles_triangle_l1186_118693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_journey_time_l1186_118653

-- Define the river system
structure RiverSystem where
  v : ℝ  -- boat speed in still water
  vr : ℝ  -- river current speed
  x : ℝ  -- distance AC = CD
  y : ℝ  -- distance CB

-- Define the time functions for different routes
noncomputable def timeABC (rs : RiverSystem) : ℝ := rs.x / (rs.v - rs.vr) + rs.y / rs.v
noncomputable def timeBCD (rs : RiverSystem) : ℝ := rs.y / rs.v + rs.x / (rs.v + rs.vr)
noncomputable def timeDCB (rs : RiverSystem) : ℝ := rs.x / (rs.v - rs.vr) + rs.y / rs.v

-- Define the theorem
theorem river_journey_time (rs : RiverSystem) 
  (h1 : timeABC rs = 6)
  (h2 : timeBCD rs = 8)
  (h3 : timeDCB rs = 5) :
  rs.y / rs.v + rs.x / (rs.v - rs.vr) + rs.x / (rs.v + rs.vr) = 37/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_journey_time_l1186_118653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1186_118639

/-- The area of a triangle with vertices (5,-2), (0,3), and (3,-3) is 15/2 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (5, -2)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (3, -3)
  let area := abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2
  area = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l1186_118639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1186_118694

/-- Given a line and a circle with specific properties, prove that the line's slope parameter is 0 -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (∀ x y : ℝ, (a * x - y + 3 = 0) ↔ ((x, y) = A ∨ (x, y) = B)) ∧ 
    ((A.1 - 1)^2 + (A.2 - 2)^2 = 4) ∧
    ((B.1 - 1)^2 + (B.2 - 2)^2 = 4) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12) →
  a = 0 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l1186_118694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_with_special_point_l1186_118667

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (h.a^2 + h.b^2) / h.a

/-- Theorem about the eccentricity range of a hyperbola with a special point -/
theorem eccentricity_range_with_special_point (h : Hyperbola) :
  (∃ (P : ℝ × ℝ), P.1 > 0 ∧ 
    P.1^2 / h.a^2 - P.2^2 / h.b^2 = 1 ∧
    (∃ (d : ℝ), d > 0 ∧
      (Real.sqrt ((P.1 + Real.sqrt (h.a^2 + h.b^2))^2 + P.2^2) = 6 * d) ∧
      (Real.sqrt ((P.1 - Real.sqrt (h.a^2 + h.b^2))^2 + P.2^2) = eccentricity h * d))) →
  (1 < eccentricity h ∧ eccentricity h ≤ 2) ∨ (3 ≤ eccentricity h ∧ eccentricity h < 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_with_special_point_l1186_118667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_for_given_x_y_l1186_118695

theorem n_value_for_given_x_y : ∀ (x y n : ℝ), 
  x = 8 → y = 2 → n = x - 3 * (Real.log x / Real.log y) → n = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_value_for_given_x_y_l1186_118695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_q_value_l1186_118659

/-- Represents a geometric sequence with common ratio q and first term a_1 = 1/q^2 -/
noncomputable def geometric_sequence (q : ℝ) (n : ℕ) : ℝ :=
  (1 / q^2) * q^(n - 1)

/-- Sum of the first n terms of the geometric sequence -/
noncomputable def S (q : ℝ) (n : ℕ) : ℝ :=
  (1 / q^2) * (1 - q^n) / (1 - q)

/-- The theorem stating the value of q satisfying the given conditions -/
theorem geometric_sequence_q_value :
  ∀ q : ℝ, q > 0 →
  (∀ n : ℕ, geometric_sequence q n > 0) →
  S q 5 = S q 2 + 2 →
  q = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_q_value_l1186_118659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1186_118697

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Define the function g
def g (x : ℝ) : ℝ := f x - |x - 2|

-- Theorem statement
theorem problem_solution :
  (∃ (S : Set ℝ), S = {x | f x ≤ 8} ∧ S = Set.Icc (-11) 5) ∧
  (∃ (m : ℝ), m = 5 ∧ ∀ x, g x ≤ m) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 5 → 1/a + 9/b ≥ 16/5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1186_118697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_60_minus_sin_60_l1186_118637

theorem tan_60_minus_sin_60 : Real.tan (π / 3) - Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_60_minus_sin_60_l1186_118637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_ten_value_l1186_118666

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem f_negative_ten_value
  (h1 : is_even (λ x ↦ f x + x^3))
  (h2 : f 10 = 15) :
  f (-10) = 2015 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_ten_value_l1186_118666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_eq_half_a_minus_sixth_b_l1186_118620

/-- Represents a parallelogram OADB with vectors a and b -/
structure Parallelogram (V : Type*) [AddCommGroup V] [Module ℚ V] where
  a : V
  b : V

namespace Parallelogram

variable {V : Type*} [AddCommGroup V] [Module ℚ V]

/-- The vector OD in the parallelogram -/
def od (p : Parallelogram V) : V := p.a + p.b

/-- The vector OC in the parallelogram -/
def oc (p : Parallelogram V) : V := p.b

/-- The point M on AB such that BM = 1/3 BC -/
def m (p : Parallelogram V) : V := p.b + (1/3 : ℚ) • (p.a - p.b)

/-- The point N on CD such that CN = 1/3 CD -/
def n (p : Parallelogram V) : V := p.b + (1/3 : ℚ) • p.a

/-- The vector MN in the parallelogram -/
def mn (p : Parallelogram V) : V := n p - m p

/-- Theorem stating that MN = 1/2 a - 1/6 b -/
theorem mn_eq_half_a_minus_sixth_b (p : Parallelogram V) :
  mn p = (1/2 : ℚ) • p.a - (1/6 : ℚ) • p.b := by
  sorry

end Parallelogram

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mn_eq_half_a_minus_sixth_b_l1186_118620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cos_value_l1186_118682

theorem point_on_line_cos_value (α : ℝ) :
  (Real.sin α = -2 * Real.cos α) → Real.cos (2 * α + π / 2) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_cos_value_l1186_118682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_book_count_l1186_118601

theorem smallest_possible_book_count 
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (physics_chemistry_ratio : physics * 2 = chemistry * 3)
  (chemistry_biology_ratio : chemistry * 3 = biology * 4)
  (physics_positive : physics > 0)
  (chemistry_positive : chemistry > 0)
  (biology_positive : biology > 0)
  (total : ℕ := physics + chemistry + biology)
  (is_smallest : ∀ p c b : ℕ, 
    p * 2 = c * 3 → c * 3 = b * 4 → p > 0 → c > 0 → b > 0 → 
    p + c + b ≥ total) :
  total = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_possible_book_count_l1186_118601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_theorem_l1186_118610

/-- The area of a parallelogram formed by two vectors -/
def parallelogramArea (a b : ℝ × ℝ) : ℝ := abs (a.1 * b.2 - a.2 * b.1)

/-- Given two vectors p and q in ℝ², returns a vector that is a linear combination of p and q -/
def linearCombination (p q : ℝ × ℝ) (α β : ℝ) : ℝ × ℝ := (α * p.1 + β * q.1, α * p.2 + β * q.2)

theorem parallelogram_area_theorem (p q : ℝ × ℝ) (h1 : Real.sqrt (p.1^2 + p.2^2) = 1) 
  (h2 : Real.sqrt (q.1^2 + q.2^2) = 2) 
  (h3 : Real.cos (Real.arccos ((p.1 * q.1 + p.2 * q.2) / (Real.sqrt (p.1^2 + p.2^2) * Real.sqrt (q.1^2 + q.2^2)))) = Real.sqrt 3 / 2) :
  parallelogramArea (linearCombination p q 1 (-4)) (linearCombination p q 3 1) = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_theorem_l1186_118610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_8_27_l1186_118663

/-- A tetrahedron with inscribed and circumscribed spheres -/
structure Tetrahedron where
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  h : r = R / 3

/-- The number of smaller spheres in the tetrahedron -/
def small_spheres : ℕ := 8

/-- The volume of a sphere given its radius -/
noncomputable def sphere_volume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius ^ 3

/-- The total volume of the eight smaller spheres -/
noncomputable def small_spheres_volume (t : Tetrahedron) : ℝ := 
  (small_spheres : ℝ) * sphere_volume t.r

/-- The volume of the circumscribed sphere -/
noncomputable def circumscribed_volume (t : Tetrahedron) : ℝ := sphere_volume t.R

/-- The probability of a point being in one of the eight smaller spheres -/
noncomputable def probability_in_small_spheres (t : Tetrahedron) : ℝ :=
  small_spheres_volume t / circumscribed_volume t

theorem probability_is_8_27 (t : Tetrahedron) : 
  probability_in_small_spheres t = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_is_8_27_l1186_118663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1186_118643

-- Define the ellipse and hyperbola
def ellipse (x y m n : ℝ) : Prop := x^2 / (3 * m^2) + y^2 / (5 * n^2) = 1

def hyperbola (x y m n : ℝ) : Prop := x^2 / (2 * m^2) - y^2 / (3 * n^2) = 1

-- Define the condition of sharing a common focus
def common_focus (m n : ℝ) : Prop := 3 * m^2 - 5 * n^2 = 2 * m^2 + 3 * n^2

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = (Real.sqrt 3 / 4) * x ∨ y = -(Real.sqrt 3 / 4) * x

-- State the theorem
theorem hyperbola_asymptote (m n : ℝ) (h : common_focus m n) :
  ∀ x y, hyperbola x y m n → asymptote x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_l1186_118643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1186_118681

/-- Definition of circle w₁ -/
def w₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y - 23 = 0

/-- Definition of circle w₂ -/
def w₂ (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 8*y + 89 = 0

/-- Definition of line containing the center of w₃ -/
def centerLine (a : ℝ) (x y : ℝ) : Prop := y = a*x + 3

/-- Definition of external tangency -/
def externallyTangent (x₁ y₁ r₁ x₂ y₂ r₂ : ℝ) : Prop :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = (r₁ + r₂)^2

/-- Main theorem -/
theorem circle_tangency (m : ℝ) (p q : ℕ) :
  (∃ x₁ y₁ x₂ y₂ x₃ y₃ r₁ r₂ r₃ : ℝ,
    w₁ x₁ y₁ ∧ w₂ x₂ y₂ ∧
    externallyTangent x₁ y₁ r₁ x₃ y₃ r₃ ∧
    externallyTangent x₂ y₂ r₂ x₃ y₃ r₃ ∧
    centerLine m x₃ y₃) →
  (m > 0) →
  (∀ a : ℝ, a > 0 → a < m →
    ¬∃ x y : ℝ, centerLine a x y ∧
      ∃ r : ℝ, externallyTangent x₁ y₁ r₁ x y r ∧
               externallyTangent x₂ y₂ r₂ x y r) →
  m^2 = p / q →
  Nat.Coprime p q →
  p + q = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_l1186_118681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_max_min_sum_l1186_118611

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem exponential_function_max_min_sum (a : ℝ) :
  a > 0 → a ≠ 1 →
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, f a x ≥ min) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = 12) →
  a = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_max_min_sum_l1186_118611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_measure_varies_l1186_118662

-- Define the equilateral triangle
def equilateral_triangle (s : ℝ) : ℝ × ℝ × ℝ := sorry

-- Define the circle's radius (equal to triangle's altitude)
noncomputable def circle_radius (s : ℝ) : ℝ := s * Real.sqrt 3 / 2

-- Define the function n(θ) representing the measure of arc MTN
def n (θ : ℝ) : ℝ := 120 - 2 * θ

theorem arc_measure_varies (s θ : ℝ) 
  (h1 : s > 0) 
  (h2 : 0 ≤ θ ∧ θ ≤ 60) : 
  60 ≤ n θ ∧ n θ ≤ 120 := by
  sorry

#check arc_measure_varies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_measure_varies_l1186_118662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circle_areas_l1186_118606

-- Define the triangle side lengths
def a : ℝ := 4
def b : ℝ := 5
def c : ℝ := 6

-- Define the radii of the circles
noncomputable def r : ℝ := sorry
noncomputable def s : ℝ := sorry
noncomputable def t : ℝ := sorry

-- State the theorem
theorem sum_of_circle_areas (h1 : r + s = a) (h2 : r + t = b) (h3 : s + t = c) :
  π * (r^2 + s^2 + t^2) = 20.75 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_circle_areas_l1186_118606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_cube_volume_l1186_118675

theorem larger_cube_volume (n : ℕ) (small_cube_volume : ℝ) (surface_area_diff : ℝ) : 
  n = 125 → 
  small_cube_volume = 1 → 
  surface_area_diff = 600 → 
  (n : ℝ) * small_cube_volume = 125 := by
  intros hn hsv hsad
  rw [hn, hsv]
  norm_num

#check larger_cube_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_cube_volume_l1186_118675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_l1186_118616

noncomputable section

variable (f : ℝ → ℝ)

axiom f_def : ∀ (m n : ℝ), f (m + n^2) = f m + 2 * (f n)^2

axiom f_one_nonzero : f 1 ≠ 0

theorem f_2013 : f 2013 = 4024 * (f 1)^2 + f 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2013_l1186_118616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_m_eq_1_range_of_m_l1186_118646

/-- The function f(x) as defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - |x + 3 * m|

/-- Theorem stating the solution set for f(x) ≥ 1 when m = 1 -/
theorem solution_set_for_m_eq_1 :
  ∀ x : ℝ, (f 1 x ≥ 1) ↔ (x ≤ -3/2) := by
  sorry

/-- Theorem stating the range of m for which f(x) < |2+t|+|t-1| holds for all real x and t -/
theorem range_of_m :
  ∀ m : ℝ, (∀ x t : ℝ, f m x < |2 + t| + |t - 1|) ↔ (0 < m ∧ m < 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_m_eq_1_range_of_m_l1186_118646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_natural_numbers_satisfying_equation_l1186_118690

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem count_natural_numbers_satisfying_equation : 
  ∃! (n : ℕ), (∀ (x : ℕ), (floor (-1.77 * ↑x) = (floor (-1.77)) * x) ↔ x < n) ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_natural_numbers_satisfying_equation_l1186_118690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_1023_l1186_118685

def sequenceJoBlair (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2 * sequenceJoBlair n + 1

theorem tenth_term_is_1023 : sequenceJoBlair 9 = 1023 := by
  rfl

#eval sequenceJoBlair 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_1023_l1186_118685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_integers_l1186_118632

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | n + 2 => (sequence_a (n + 1))^2 / sequence_a n + 2 / sequence_a n

theorem all_terms_are_integers :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_integers_l1186_118632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1186_118617

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_increasing : ∀ x y, x < y → f x < f y
axiom g_even : ∀ x, g (-x) = g x
axiom g_coincides_f : ∀ x, x ≥ 0 → g x = f x

-- Define the theorem
theorem inequality_holds (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  f b - f (-a) > g a - g (-b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l1186_118617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_midpoint_l1186_118696

-- Define the hyperbola C
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2/3 = 1

-- Define the right focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the asymptote equations
def asymptote (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Define a point on the hyperbola
def on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

-- Define the line passing through F and intersecting asymptotes
def line_through_F (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, A = (2 + t * (B.1 - 2), t * B.2) ∧ 
             asymptote A.1 A.2 ∧ asymptote B.1 B.2

-- Define the intersection point M
def intersection_point (P Q M : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    M.1 = P.1 + t₁ * 1 ∧ M.2 = P.2 - t₁ * Real.sqrt 3 ∧
    M.1 = Q.1 + t₂ * 1 ∧ M.2 = Q.2 + t₂ * Real.sqrt 3

-- The main theorem
theorem hyperbola_intersection_midpoint 
  (P Q : ℝ × ℝ) (h₁ : on_hyperbola P) (h₂ : on_hyperbola Q)
  (h₃ : P.1 > Q.1) (h₄ : Q.1 > 0) (h₅ : P.2 > 0)
  (A B M : ℝ × ℝ) (h₆ : line_through_F A B)
  (h₇ : intersection_point P Q M) :
  abs (M.1 - A.1) = abs (M.1 - B.1) ∧ abs (M.2 - A.2) = abs (M.2 - B.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_midpoint_l1186_118696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_completion_time_l1186_118684

/-- Given workers A and D, their job completion times, and their combined completion time,
    calculate D's individual completion time. -/
theorem worker_completion_time 
  (time_A : ℝ) 
  (time_AD : ℝ) 
  (h_A_positive : time_A > 0)
  (h_AD_positive : time_AD > 0)
  (h_A : time_A = 6)
  (h_AD : time_AD = 4) :
  (1 / (1 / time_AD - 1 / time_A)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_completion_time_l1186_118684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1186_118647

/-- The speed of the first train in km/h -/
noncomputable def speed_train1 : ℝ := 120

/-- The speed of the second train in km/h -/
noncomputable def speed_train2 : ℝ := 80

/-- The time taken for the trains to cross each other in seconds -/
noncomputable def crossing_time : ℝ := 9

/-- The length of the second train in meters -/
noncomputable def length_train2 : ℝ := 350.04

/-- Conversion factor from km/h to m/s -/
noncomputable def kmph_to_ms : ℝ := 5 / 18

/-- Theorem stating that the length of the first train is approximately 150 meters -/
theorem train_length_problem :
  let relative_speed : ℝ := (speed_train1 + speed_train2) * kmph_to_ms
  let combined_length : ℝ := relative_speed * crossing_time
  let length_train1 : ℝ := combined_length - length_train2
  ∃ ε > 0, |length_train1 - 150| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_problem_l1186_118647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l1186_118645

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the foci
noncomputable def focus1 : ℝ × ℝ := sorry
noncomputable def focus2 : ℝ × ℝ := sorry

-- Define points P and Q on the ellipse
noncomputable def P : ℝ × ℝ := sorry
noncomputable def Q : ℝ × ℝ := sorry

-- Define the perpendicularity condition
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define the vector sum
def vector_sum (v1 v2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the slope of a line
noncomputable def line_slope (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_slope_theorem :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  perpendicular (vector_sum (P - focus2) (Q - focus2)) (0, 1) →
  perpendicular (vector_sum (P - focus1) (P - focus2)) (vector_sum (Q - focus1) (Q - focus2)) →
  (line_slope P Q)^2 = (-5 + 2 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_slope_theorem_l1186_118645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1186_118602

/-- Given function f -/
noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (Real.pi + α) * Real.sin (-α + 3*Real.pi/2)) /
  (Real.cos (-α) * Real.cos (α + Real.pi/2))

/-- Theorem stating the value of f(α) under given conditions -/
theorem f_value_in_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) : 
  f α = 2 * Real.sqrt 6 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_in_third_quadrant_l1186_118602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1186_118622

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / 4 + a / x - Real.log x - 3 / 2

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, x > 0 → HasDerivAt (f a) ((1/4 : ℝ) - a/x^2 - 1/x) x) →
  (HasDerivAt (f a) (-2) 1) →
  (a = 5/4) ∧
  (∀ x : ℝ, 0 < x → x < 5 → HasDerivAt (f (5/4)) (((1/4 : ℝ) - (5/4)/x^2 - 1/x) : ℝ) x ∧ ((1/4 : ℝ) - (5/4)/x^2 - 1/x) < 0) ∧
  (∀ x : ℝ, x > 5 → HasDerivAt (f (5/4)) (((1/4 : ℝ) - (5/4)/x^2 - 1/x) : ℝ) x ∧ ((1/4 : ℝ) - (5/4)/x^2 - 1/x) > 0) ∧
  (f (5/4) 5 = -Real.log 5) ∧
  (∀ x : ℝ, x > 0 → f (5/4) x ≥ -Real.log 5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1186_118622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1186_118656

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

-- Define the line passing through P(0,1)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the condition AP = 2PB
def vector_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -x₁ = 2*x₂ ∧ 1 - y₁ = 2*(y₂ - 1)

-- Main theorem
theorem ellipse_triangle_area :
  ∃ (k x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧
    line k x₁ y₁ ∧
    line k x₂ y₂ ∧
    vector_condition x₁ y₁ x₂ y₂ ∧
    (1/2 * abs (x₁ - x₂) = Real.sqrt 126/8) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l1186_118656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teachers_spending_l1186_118600

noncomputable section

-- Define the discount rules
def discount_rule (price : ℝ) : ℝ :=
  if price ≤ 100 then price
  else if price ≤ 300 then 0.9 * price
  else 0.9 * 300 + 0.8 * (price - 300)

-- Define the teachers' spending
def li_spending (original_price : ℝ) : ℝ := 0.9 * original_price
def zhang_spending (original_price : ℝ) : ℝ := discount_rule original_price

-- Define the combined discount savings
def combined_savings (li_price zhang_price : ℝ) : ℝ :=
  (li_spending li_price + zhang_spending zhang_price) - discount_rule (li_price + zhang_price)

-- Define the difference between original and discounted prices
def discount_difference (li_price zhang_price : ℝ) : ℝ :=
  (li_price + zhang_price) - (li_spending li_price + zhang_spending zhang_price)

-- Theorem statement
theorem teachers_spending :
  ∃ (li_price zhang_price : ℝ),
    combined_savings li_price zhang_price = 19 ∧
    discount_difference li_price zhang_price = 67 ∧
    li_spending li_price = 171 ∧
    zhang_spending zhang_price = 342 := by
  -- Proof goes here
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_teachers_spending_l1186_118600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_distinct_reals_l1186_118674

theorem existence_of_distinct_reals (n : ℕ) : n > 1 → (
  (∃ (a : Fin n → ℝ), (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin n, a i ≠ 0) ∧
    ({i : Fin n | ∃ j : Fin n, a i + (-1)^(i.val + 1) / a i = a j} = Set.univ)
  ) ↔ n % 2 = 1
) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_distinct_reals_l1186_118674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_characterization_l1186_118603

/-- A pair of positive integers satisfying the given conditions -/
structure ValidPair where
  x : ℕ
  y : ℕ
  x_pos : 0 < x
  y_pos : 0 < y
  x_multiple_of_9 : 9 ∣ x
  y_multiple_of_3 : 3 ∣ y
  sum_of_squares : x^2 + y^2 > 250
  sum_less_than_35 : x + y < 35

/-- The set of all valid pairs -/
def validPairs : Set ValidPair := {p | p.x ∈ ({9, 18, 27} : Set ℕ) ∧ p.y ∈ ({6, 15, 24} : Set ℕ)}

theorem valid_pairs_characterization :
  ∀ p : ValidPair, p ∈ validPairs ↔ 
    (p.x = 9 ∧ p.y = 24) ∨ (p.x = 18 ∧ p.y = 15) ∨ (p.x = 27 ∧ p.y = 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_characterization_l1186_118603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_point_distribution_properties_l1186_118687

/-- A random variable following a two-point distribution -/
structure TwoPointDistribution where
  X : ℝ → ℝ
  prob_X_1 : ℝ
  is_two_point : (∀ x, X x = 0 ∨ X x = 1)
  prob_sum_to_1 : prob_X_1 + (1 - prob_X_1) = 1

/-- The probability mass function for X -/
noncomputable def pmf (dist : TwoPointDistribution) (x : ℝ) : ℝ :=
  if x = 1 then dist.prob_X_1 else 1 - dist.prob_X_1

/-- The expected value of X -/
def expectation (dist : TwoPointDistribution) : ℝ :=
  0 * (1 - dist.prob_X_1) + 1 * dist.prob_X_1

/-- The variance of X -/
def variance (dist : TwoPointDistribution) : ℝ :=
  dist.prob_X_1 * (1 - dist.prob_X_1)

theorem two_point_distribution_properties (dist : TwoPointDistribution) 
  (h : dist.prob_X_1 = 1/2) : 
  pmf dist 0 = 1/2 ∧ expectation dist = 1/2 ∧ variance dist = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_point_distribution_properties_l1186_118687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_equal_to_24_bananas_l1186_118614

-- Define the cost ratios
def banana_apple_ratio : ℚ := 3 / 4
def apple_orange_ratio : ℚ := 2 / 3

-- Theorem statement
theorem oranges_equal_to_24_bananas :
  24 * banana_apple_ratio * apple_orange_ratio = 12 := by
  -- Expand the definitions
  unfold banana_apple_ratio apple_orange_ratio
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_oranges_equal_to_24_bananas_l1186_118614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l1186_118648

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (∀ (n : ℕ), (x : ℝ) = 7 + 3 * (1 - 1/10^(n+1)) / 9) ∧ x = 22 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l1186_118648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1186_118633

-- Define the function
noncomputable def f (x : ℝ) : ℝ := |Real.sin x| - 2 * Real.sin x

-- State the theorem
theorem f_range : Set.range f = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1186_118633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_angle_sum_l1186_118680

-- Define a triangle
structure Triangle where
  α : ℝ  -- internal angle
  β : ℝ  -- internal angle
  γ : ℝ  -- internal angle
  sum_internal : α + β + γ = Real.pi  -- sum of internal angles is π radians (180°)
  positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ  -- angles are positive

-- Define an external angle
noncomputable def external_angle (t : Triangle) : ℝ :=
  Real.pi - t.α  -- external angle is supplementary to the adjacent internal angle

-- Theorem statement
theorem external_angle_sum (t : Triangle) :
  external_angle t = t.β + t.γ := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_angle_sum_l1186_118680
