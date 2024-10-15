import Mathlib

namespace NUMINAMATH_CALUDE_corrected_mean_l1272_127257

theorem corrected_mean (n : ℕ) (original_mean : ℝ) (wrong_value correct_value : ℝ) :
  n = 50 ∧ original_mean = 36 ∧ wrong_value = 23 ∧ correct_value = 45 →
  (n * original_mean + (correct_value - wrong_value)) / n = 36.44 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l1272_127257


namespace NUMINAMATH_CALUDE_class_average_mark_l1272_127248

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_avg : ℝ) (remaining_avg : ℝ) : 
  total_students = 30 →
  excluded_students = 5 →
  excluded_avg = 30 →
  remaining_avg = 90 →
  (total_students : ℝ) * (total_students * remaining_avg - excluded_students * excluded_avg) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l1272_127248


namespace NUMINAMATH_CALUDE_extreme_point_property_g_maximum_bound_l1272_127284

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - a*x - b

-- Define the function g
def g (a b x : ℝ) : ℝ := |f a b x|

theorem extreme_point_property (a b x₀ x₁ : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₀ - ε) (x₀ + ε), f a b x₀ ≤ f a b x) →
  f a b x₁ = f a b x₀ →
  x₁ ≠ x₀ →
  x₁ + 2*x₀ = 0 := by sorry

theorem g_maximum_bound (a b : ℝ) (ha : a > 0) :
  ∃ x ∈ Set.Icc (-1 : ℝ) 1, ∀ y ∈ Set.Icc (-1 : ℝ) 1, g a b x ≥ (1/4 : ℝ) ∧ g a b x ≥ g a b y := by sorry

end NUMINAMATH_CALUDE_extreme_point_property_g_maximum_bound_l1272_127284


namespace NUMINAMATH_CALUDE_twelve_non_congruent_triangles_l1272_127200

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℤ)
  (y : ℤ)

/-- The set of points on the grid -/
def grid_points : List Point := [
  ⟨0, 0⟩, ⟨1, 0⟩, ⟨2, 0⟩, ⟨3, 0⟩,
  ⟨0, 1⟩, ⟨1, 1⟩, ⟨2, 1⟩, ⟨3, 1⟩
]

/-- Determines if two triangles are congruent -/
def are_congruent (t1 t2 : Point × Point × Point) : Prop := sorry

/-- Counts the number of non-congruent triangles -/
def count_non_congruent_triangles (points : List Point) : ℕ := sorry

/-- The main theorem stating that there are 12 non-congruent triangles -/
theorem twelve_non_congruent_triangles :
  count_non_congruent_triangles grid_points = 12 := by sorry

end NUMINAMATH_CALUDE_twelve_non_congruent_triangles_l1272_127200


namespace NUMINAMATH_CALUDE_toms_age_problem_l1272_127211

/-- Tom's age problem -/
theorem toms_age_problem (T N : ℝ) : 
  T > 0 ∧ N > 0 ∧ 
  (∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = T) ∧
  T - N = 3 * (T - 4 * N) →
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_problem_l1272_127211


namespace NUMINAMATH_CALUDE_integer_roots_of_cubic_l1272_127216

def f (x : ℤ) : ℤ := x^3 - 6*x^2 - 4*x + 24

theorem integer_roots_of_cubic :
  ∀ x : ℤ, f x = 0 ↔ x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_cubic_l1272_127216


namespace NUMINAMATH_CALUDE_cone_volume_l1272_127208

/-- Given a cone circumscribed by a sphere, proves that the volume of the cone is 3π -/
theorem cone_volume (r l h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  (π * r * l = 2 * π * r^2) →  -- lateral area is twice base area
  (4 * π * (2^2) = 16 * π) →   -- surface area of circumscribing sphere is 16π
  (1/3) * π * r^2 * h = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l1272_127208


namespace NUMINAMATH_CALUDE_janes_calculation_l1272_127281

theorem janes_calculation (x y z : ℝ) 
  (h1 : x - (y - z) = 17) 
  (h2 : x - y - z = 5) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_janes_calculation_l1272_127281


namespace NUMINAMATH_CALUDE_starting_number_for_prime_factors_of_210_l1272_127262

def isPrime (n : Nat) : Prop := sorry

def isFactor (a b : Nat) : Prop := sorry

theorem starting_number_for_prime_factors_of_210 :
  ∃ (start : Nat),
    start ≤ 100 ∧
    (∀ p, isPrime p → p > start → p ≤ 100 → isFactor p 210 →
      ∃ (primes : Finset Nat),
        primes.card = 4 ∧
        (∀ q ∈ primes, isPrime q ∧ q > start ∧ q ≤ 100 ∧ isFactor q 210)) ∧
    (∀ start' > start,
      ¬(∃ (primes : Finset Nat),
        primes.card = 4 ∧
        (∀ q ∈ primes, isPrime q ∧ q > start' ∧ q ≤ 100 ∧ isFactor q 210))) ∧
    start = 1 :=
by sorry

end NUMINAMATH_CALUDE_starting_number_for_prime_factors_of_210_l1272_127262


namespace NUMINAMATH_CALUDE_average_multiplication_invariance_l1272_127250

theorem average_multiplication_invariance (S : Finset ℝ) (n : ℕ) (h : n > 0) :
  let avg := (S.sum id) / n
  let new_avg := (S.sum (fun x => 10 * x)) / n
  avg = 7 ∧ new_avg = 70 →
  ∃ (m : ℕ), m > 0 ∧ (S.sum id) / m = 7 ∧ (S.sum (fun x => 10 * x)) / m = 70 :=
by sorry

end NUMINAMATH_CALUDE_average_multiplication_invariance_l1272_127250


namespace NUMINAMATH_CALUDE_probability_tropical_temperate_l1272_127296

/-- The number of tropical fruits -/
def tropical_fruits : ℕ := 3

/-- The number of temperate fruits -/
def temperate_fruits : ℕ := 2

/-- The total number of fruits -/
def total_fruits : ℕ := tropical_fruits + temperate_fruits

/-- The number of fruits to be selected -/
def selection_size : ℕ := 2

/-- The probability of selecting one tropical and one temperate fruit -/
theorem probability_tropical_temperate :
  (Nat.choose tropical_fruits 1 * Nat.choose temperate_fruits 1 : ℚ) / 
  Nat.choose total_fruits selection_size = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_tropical_temperate_l1272_127296


namespace NUMINAMATH_CALUDE_cube_of_three_fifths_l1272_127294

theorem cube_of_three_fifths : (3 / 5 : ℚ) ^ 3 = 27 / 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_three_fifths_l1272_127294


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l1272_127273

/-- A line in the xy-plane can be represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Given that line b is parallel to y = 3x - 2 and passes through (3, 4), its y-intercept is -5. -/
theorem parallel_line_y_intercept :
  let reference_line : Line := { slope := 3, point := (0, -2) }
  let b : Line := { slope := reference_line.slope, point := (3, 4) }
  y_intercept b = -5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l1272_127273


namespace NUMINAMATH_CALUDE_equal_distribution_contribution_l1272_127221

def earnings : List ℕ := [18, 22, 30, 35, 45]

theorem equal_distribution_contribution :
  let total := earnings.sum
  let equal_share := total / earnings.length
  let max_earner := earnings.maximum?
  match max_earner with
  | some max => max - equal_share = 15
  | none => False
  := by sorry

end NUMINAMATH_CALUDE_equal_distribution_contribution_l1272_127221


namespace NUMINAMATH_CALUDE_circle_symmetry_l1272_127243

/-- The equation of a circle -/
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 + 2*a*x - 2*a*y = 0

/-- The line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := x + y = 0

/-- Theorem stating that the circle is symmetric with respect to the line x + y = 0 -/
theorem circle_symmetry (a : ℝ) (h : a ≠ 0) :
  ∃ (r : ℝ), r > 0 ∧
  ∀ (x y : ℝ), circle_equation x y a ↔
    ∃ (x₀ y₀ : ℝ), symmetry_line x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1272_127243


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_length_is_8_l1272_127289

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ → Prop :=
  λ aₙ ↦ aₙ = a₁ + (n - 1 : ℤ) * d

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 20 (-3) n (-2) ∧ 
  ∀ m : ℕ, m > n → ¬arithmetic_sequence 20 (-3) m (-2) := by
  sorry

theorem arithmetic_sequence_length_is_8 :
  ∃! n : ℕ, n > 0 ∧ arithmetic_sequence 20 (-3) n (-2) ∧ 
  ∀ m : ℕ, m > n → ¬arithmetic_sequence 20 (-3) m (-2) ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_length_is_8_l1272_127289


namespace NUMINAMATH_CALUDE_no_such_function_exists_l1272_127227

theorem no_such_function_exists : 
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l1272_127227


namespace NUMINAMATH_CALUDE_eleventh_term_value_l1272_127235

/-- An arithmetic progression is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

/-- The theorem states that for an arithmetic progression satisfying
    certain conditions, the 11th term is 109. -/
theorem eleventh_term_value
    (a : ℕ → ℝ)
    (h_ap : ArithmeticProgression a)
    (h_sum1 : a 4 + a 7 + a 10 = 207)
    (h_sum2 : a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 553) :
    a 11 = 109 := by
  sorry


end NUMINAMATH_CALUDE_eleventh_term_value_l1272_127235


namespace NUMINAMATH_CALUDE_wrapping_paper_rolls_l1272_127236

/-- The number of rolls of wrapping paper Savannah bought -/
def rolls_bought : ℕ := 3

/-- The total number of gifts Savannah has to wrap -/
def total_gifts : ℕ := 12

/-- The number of gifts wrapped with the first roll -/
def gifts_first_roll : ℕ := 3

/-- The number of gifts wrapped with the second roll -/
def gifts_second_roll : ℕ := 5

/-- The number of gifts wrapped with the third roll -/
def gifts_third_roll : ℕ := 4

theorem wrapping_paper_rolls :
  rolls_bought = 3 ∧
  total_gifts = 12 ∧
  gifts_first_roll = 3 ∧
  gifts_second_roll = 5 ∧
  gifts_third_roll = 4 ∧
  total_gifts = gifts_first_roll + gifts_second_roll + gifts_third_roll :=
by sorry

end NUMINAMATH_CALUDE_wrapping_paper_rolls_l1272_127236


namespace NUMINAMATH_CALUDE_golden_ratio_in_line_segment_l1272_127205

theorem golden_ratio_in_line_segment (A B C : ℝ) (k : ℝ) 
  (h1 : B > A ∧ B < C)  -- B is between A and C
  (h2 : (C - B) / (B - A) = k)  -- BC/AB = k
  (h3 : (B - A) / (C - A) = k)  -- AB/AC = k
  : k = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_in_line_segment_l1272_127205


namespace NUMINAMATH_CALUDE_range_of_x_l1272_127261

theorem range_of_x (x : ℝ) : 
  ((x + 2) * (x - 3) ≤ 0) ∧ (abs (x + 1) ≥ 2) → 
  x ∈ Set.Icc 1 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_x_l1272_127261


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1272_127229

theorem inequality_and_equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a ∧
  ((1/2) * (a + b)^2 + (1/4) * (a + b) = a * Real.sqrt b + b * Real.sqrt a ↔ 
    (a = 0 ∧ b = 0) ∨ (a = 1/4 ∧ b = 1/4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1272_127229


namespace NUMINAMATH_CALUDE_boxes_with_neither_l1272_127226

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : markers = 9)
  (h3 : crayons = 5)
  (h4 : both = 4) :
  total - (markers + crayons - both) = 5 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l1272_127226


namespace NUMINAMATH_CALUDE_max_non_managers_l1272_127283

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  (managers : ℚ) / non_managers > 7 / 32 →
  non_managers ≤ 36 →
  ∃ (max_non_managers : ℕ), max_non_managers = 36 ∧ 
    ∀ n : ℕ, n > max_non_managers → (managers : ℚ) / n ≤ 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_max_non_managers_l1272_127283


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1272_127219

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_sixth : a 6 = 8) :
  a 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1272_127219


namespace NUMINAMATH_CALUDE_bug_total_distance_l1272_127259

def bug_crawl (start : ℝ) (p1 p2 p3 : ℝ) : ℝ :=
  |p1 - start| + |p2 - p1| + |p3 - p2|

theorem bug_total_distance : bug_crawl 3 (-4) 6 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_bug_total_distance_l1272_127259


namespace NUMINAMATH_CALUDE_initial_cookies_count_l1272_127287

def cookies_eaten : ℕ := 15
def cookies_left : ℕ := 78

theorem initial_cookies_count : 
  cookies_eaten + cookies_left = 93 := by sorry

end NUMINAMATH_CALUDE_initial_cookies_count_l1272_127287


namespace NUMINAMATH_CALUDE_perfect_square_prime_sum_l1272_127241

theorem perfect_square_prime_sum (x y : ℤ) : 
  (∃ k : ℤ, 2 * x * y = k^2) ∧ 
  (∃ p : ℕ, Nat.Prime p ∧ x^2 + y^2 = p) →
  ((x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_prime_sum_l1272_127241


namespace NUMINAMATH_CALUDE_sum_squares_bounds_l1272_127252

theorem sum_squares_bounds (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + y = 10) :
  50 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 100 ∧
  (∃ a b : ℝ, a ≥ 0 ∧ b ≥ 0 ∧ a + b = 10 ∧ a^2 + b^2 = 50) ∧
  (∃ c d : ℝ, c ≥ 0 ∧ d ≥ 0 ∧ c + d = 10 ∧ c^2 + d^2 = 100) := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_bounds_l1272_127252


namespace NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l1272_127232

/-- Given complex numbers u, v, and w that are zeros of a cubic polynomial
    and form a right triangle in the complex plane, if the sum of their
    squared magnitudes is 400, then the square of the hypotenuse of the
    triangle is 720. -/
theorem hypotenuse_squared_of_complex_zeros (u v w : ℂ) (s t : ℂ) :
  (u^3 + s*u + t = 0) →
  (v^3 + s*v + t = 0) →
  (w^3 + s*w + t = 0) →
  (Complex.abs u)^2 + (Complex.abs v)^2 + (Complex.abs w)^2 = 400 →
  ∃ (a b : ℝ), a^2 + b^2 = (Complex.abs (u - v))^2 ∧
                a^2 + b^2 = (Complex.abs (v - w))^2 ∧
                a * b = (Complex.abs (u - w))^2 →
  (Complex.abs (u - w))^2 = 720 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_squared_of_complex_zeros_l1272_127232


namespace NUMINAMATH_CALUDE_inequality_always_holds_l1272_127279

theorem inequality_always_holds (a b c : ℝ) (h : a < b ∧ b < c) : a - c < b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l1272_127279


namespace NUMINAMATH_CALUDE_gcd_7920_14553_l1272_127256

theorem gcd_7920_14553 : Nat.gcd 7920 14553 = 11 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7920_14553_l1272_127256


namespace NUMINAMATH_CALUDE_alarm_system_probability_l1272_127233

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) : 
  1 - (1 - p)^2 = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_alarm_system_probability_l1272_127233


namespace NUMINAMATH_CALUDE_max_angle_C_l1272_127264

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = Real.pi

-- Define the condition sin²A + sin²B = 2sin²C
def specialCondition (t : Triangle) : Prop :=
  Real.sin t.A ^ 2 + Real.sin t.B ^ 2 = 2 * Real.sin t.C ^ 2

-- Theorem statement
theorem max_angle_C (t : Triangle) (h : specialCondition t) : 
  t.C ≤ Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_max_angle_C_l1272_127264


namespace NUMINAMATH_CALUDE_test_questions_count_l1272_127225

theorem test_questions_count (score : ℤ) (correct : ℤ) (incorrect : ℤ) :
  score = correct - 2 * incorrect →
  score = 61 →
  correct = 87 →
  correct + incorrect = 100 := by
sorry

end NUMINAMATH_CALUDE_test_questions_count_l1272_127225


namespace NUMINAMATH_CALUDE_problem_solution_l1272_127263

theorem problem_solution (a b : ℝ) (h1 : a * b = 4) (h2 : 2 / a + 1 / b = 1.5) :
  a + 2 * b = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1272_127263


namespace NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1272_127230

theorem largest_prime_divisor_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 10 + Nat.factorial 11) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 10 + Nat.factorial 11) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_factorial_sum_l1272_127230


namespace NUMINAMATH_CALUDE_rest_time_calculation_l1272_127268

theorem rest_time_calculation (walking_rate : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  walking_rate = 10 →
  total_distance = 50 →
  total_time = 328 →
  (∃ (rest_time : ℝ),
    rest_time * 4 = total_time - (total_distance / walking_rate * 60) ∧
    rest_time = 7) := by
  sorry

end NUMINAMATH_CALUDE_rest_time_calculation_l1272_127268


namespace NUMINAMATH_CALUDE_calculate_expression_l1272_127244

theorem calculate_expression : 15 * 35 + 45 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1272_127244


namespace NUMINAMATH_CALUDE_basketball_team_size_l1272_127220

theorem basketball_team_size (total_points : ℕ) (points_per_person : ℕ) (h1 : total_points = 18) (h2 : points_per_person = 2) :
  total_points / points_per_person = 9 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_size_l1272_127220


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1272_127251

theorem inequality_equivalence (y : ℝ) : 
  (3/10 : ℝ) + |2*y - 1/5| < 7/10 ↔ -1/10 < y ∧ y < 3/10 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1272_127251


namespace NUMINAMATH_CALUDE_fraction_difference_l1272_127237

theorem fraction_difference (a b c d : ℚ) : 
  a = 3/4 ∧ b = 7/8 ∧ c = 13/16 ∧ d = 1/2 →
  max a (max b (max c d)) - min a (min b (min c d)) = 3/8 := by
sorry

end NUMINAMATH_CALUDE_fraction_difference_l1272_127237


namespace NUMINAMATH_CALUDE_halfway_fraction_l1272_127272

theorem halfway_fraction (a b : ℚ) (ha : a = 1/4) (hb : b = 1/2) :
  (a + b) / 2 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1272_127272


namespace NUMINAMATH_CALUDE_group_b_more_stable_l1272_127245

/-- Represents a group of data with its variance -/
structure DataGroup where
  variance : ℝ

/-- Defines stability comparison between two data groups -/
def more_stable (a b : DataGroup) : Prop := a.variance < b.variance

/-- Theorem stating that Group B is more stable than Group A given their variances -/
theorem group_b_more_stable (group_a group_b : DataGroup)
  (h1 : group_a.variance = 0.2)
  (h2 : group_b.variance = 0.03) :
  more_stable group_b group_a := by
  sorry

#check group_b_more_stable

end NUMINAMATH_CALUDE_group_b_more_stable_l1272_127245


namespace NUMINAMATH_CALUDE_adam_action_figures_l1272_127274

/-- The total number of action figures Adam can fit on his shelves -/
def total_action_figures (initial_shelves : List Nat) (new_shelves : Nat) (new_shelf_capacity : Nat) : Nat :=
  (initial_shelves.sum) + (new_shelves * new_shelf_capacity)

/-- Theorem: Adam can fit 52 action figures on his shelves -/
theorem adam_action_figures :
  total_action_figures [9, 14, 7] 2 11 = 52 := by
  sorry

end NUMINAMATH_CALUDE_adam_action_figures_l1272_127274


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l1272_127299

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The original rectangle --/
def original : Rectangle := { length := 5, width := 7 }

/-- The rectangle after shortening one side --/
def shortened : Rectangle := { length := 3, width := 7 }

theorem rectangle_area_proof :
  area original = 35 ∧
  area shortened = 21 →
  area { length := 5, width := 5 } = 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l1272_127299


namespace NUMINAMATH_CALUDE_num_distinct_colorings_bound_l1272_127275

/-- Represents a coloring of a 5x5 two-sided paper --/
def Coloring (n : ℕ) := Fin 5 → Fin 5 → Fin n

/-- The group of symmetries for a square --/
inductive SquareSymmetry
| identity
| rotate90
| rotate180
| rotate270
| reflectHorizontal
| reflectVertical
| reflectDiagonal1
| reflectDiagonal2

/-- Applies a symmetry to a coloring --/
def applySymmetry (sym : SquareSymmetry) (c : Coloring n) : Coloring n :=
  sorry

/-- Checks if a coloring is fixed under a symmetry --/
def isFixed (sym : SquareSymmetry) (c : Coloring n) : Prop :=
  c = applySymmetry sym c

/-- The number of colorings fixed by a given symmetry --/
def numFixedColorings (sym : SquareSymmetry) (n : ℕ) : ℕ :=
  sorry

/-- The total number of distinct colorings --/
def numDistinctColorings (n : ℕ) : ℕ :=
  sorry

theorem num_distinct_colorings_bound (n : ℕ) :
  numDistinctColorings n ≤ (n^25 + 4*n^15 + n^13 + 2*n^7) / 8 :=
sorry

end NUMINAMATH_CALUDE_num_distinct_colorings_bound_l1272_127275


namespace NUMINAMATH_CALUDE_fraction_problem_l1272_127260

theorem fraction_problem (f : ℚ) : f * 10 + 6 = 11 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1272_127260


namespace NUMINAMATH_CALUDE_product_of_r_values_l1272_127292

theorem product_of_r_values : ∃ (r₁ r₂ : ℝ),
  (∀ r : ℝ, (∃! x : ℝ, (1 : ℝ) / (3 * x) = (r - x) / 6) ↔ (r = r₁ ∨ r = r₂)) ∧
  r₁ * r₂ = -8 :=
sorry

end NUMINAMATH_CALUDE_product_of_r_values_l1272_127292


namespace NUMINAMATH_CALUDE_flagstaff_height_l1272_127254

/-- Given a flagstaff and a building casting shadows under similar conditions, 
    this theorem proves the height of the flagstaff. -/
theorem flagstaff_height 
  (building_height : ℝ) 
  (building_shadow : ℝ) 
  (flagstaff_shadow : ℝ) 
  (h_building : building_height = 12.5)
  (h_building_shadow : building_shadow = 28.75)
  (h_flagstaff_shadow : flagstaff_shadow = 40.25) :
  (building_height * flagstaff_shadow) / building_shadow = 17.5 := by
  sorry

#check flagstaff_height

end NUMINAMATH_CALUDE_flagstaff_height_l1272_127254


namespace NUMINAMATH_CALUDE_certain_number_is_80_l1272_127224

theorem certain_number_is_80 : 
  ∃ x : ℝ, (70 : ℝ) = 0.6 * x + 22 ∧ x = 80 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_80_l1272_127224


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1272_127297

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 1995 * x - 1996
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (∃ s : ℝ, s = x + y ∧ f x = 0 ∧ f y = 0 → s = -1995) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l1272_127297


namespace NUMINAMATH_CALUDE_triangle_problem_l1272_127201

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.b * Real.cos t.A + t.a * Real.cos t.B = -2 * t.c * Real.cos t.C)
  (h2 : t.b = 2 * t.a)
  (h3 : 1/2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3) :
  t.C = 2 * Real.pi / 3 ∧ t.c = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1272_127201


namespace NUMINAMATH_CALUDE_no_snow_probability_l1272_127214

theorem no_snow_probability (p : ℚ) (h : p = 4/5) :
  (1 - p)^5 = 1/3125 := by sorry

end NUMINAMATH_CALUDE_no_snow_probability_l1272_127214


namespace NUMINAMATH_CALUDE_sum_of_squared_fractions_l1272_127285

theorem sum_of_squared_fractions (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_fractions_l1272_127285


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l1272_127291

theorem correct_equation_transformation (y : ℝ) :
  (5 * y = -4 * y + 2) ↔ (5 * y + 4 * y = 2) :=
by sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l1272_127291


namespace NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1272_127217

theorem sqrt_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_reciprocal_l1272_127217


namespace NUMINAMATH_CALUDE_complex_product_real_condition_l1272_127242

theorem complex_product_real_condition (a b c d : ℝ) :
  (Complex.mk a b * Complex.mk c d).im = 0 ↔ a * d + b * c = 0 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_condition_l1272_127242


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1272_127258

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x + 5| = 3 * x - 1 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1272_127258


namespace NUMINAMATH_CALUDE_max_value_inequality_l1272_127265

theorem max_value_inequality (x y : ℝ) (h1 : |x - 1| ≤ 1) (h2 : |y - 2| ≤ 1) :
  |x - 2*y + 1| ≤ 5 ∧ ∀ (z : ℝ), (∀ (a b : ℝ), |a - 1| ≤ 1 → |b - 2| ≤ 1 → |a - 2*b + 1| ≤ z) → 5 ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1272_127265


namespace NUMINAMATH_CALUDE_same_color_probability_5_8_l1272_127270

/-- The probability of drawing two balls of the same color from a bag containing
    5 green balls and 8 white balls. -/
def same_color_probability (green : ℕ) (white : ℕ) : ℚ :=
  let total := green + white
  let prob_both_green := (green / total) * ((green - 1) / (total - 1))
  let prob_both_white := (white / total) * ((white - 1) / (total - 1))
  prob_both_green + prob_both_white

/-- Theorem stating that the probability of drawing two balls of the same color
    from a bag with 5 green balls and 8 white balls is 19/39. -/
theorem same_color_probability_5_8 :
  same_color_probability 5 8 = 19 / 39 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_5_8_l1272_127270


namespace NUMINAMATH_CALUDE_expression_simplification_l1272_127218

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1272_127218


namespace NUMINAMATH_CALUDE_polynomial_has_real_root_l1272_127267

/-- The polynomial function for which we want to prove the existence of a real root -/
def f (a x : ℝ) : ℝ := x^5 + a*x^4 - x^3 + a*x^2 - x + a

/-- Theorem stating that for all real 'a', the polynomial has at least one real root -/
theorem polynomial_has_real_root :
  ∀ a : ℝ, ∃ x : ℝ, f a x = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_has_real_root_l1272_127267


namespace NUMINAMATH_CALUDE_output_for_five_l1272_127271

def program_output (x : ℤ) : ℤ :=
  if x < 3 then 2 * x
  else if x > 3 then x * x - 1
  else 2

theorem output_for_five :
  program_output 5 = 24 :=
by sorry

end NUMINAMATH_CALUDE_output_for_five_l1272_127271


namespace NUMINAMATH_CALUDE_exists_tetrahedron_no_triangle_l1272_127295

/-- A tetrahedron with an inscribed sphere -/
structure TangentialTetrahedron where
  /-- Lengths of tangents from vertices to points of contact with the inscribed sphere -/
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  /-- All lengths are positive -/
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  d_pos : d > 0

/-- Predicate to check if three lengths can form a triangle -/
def canFormTriangle (x y z : ℝ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

/-- Theorem stating that there exists a tangential tetrahedron where no combination
    of tangent lengths can form a triangle -/
theorem exists_tetrahedron_no_triangle :
  ∃ (t : TangentialTetrahedron),
    ¬(canFormTriangle t.a t.b t.c ∨
      canFormTriangle t.a t.b t.d ∨
      canFormTriangle t.a t.c t.d ∨
      canFormTriangle t.b t.c t.d) :=
sorry

end NUMINAMATH_CALUDE_exists_tetrahedron_no_triangle_l1272_127295


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1272_127228

-- Problem 1
theorem problem_1 (x : ℝ) : (-2 * x^2)^3 + 4 * x^3 * x^3 = -4 * x^6 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) : (3 * x^2 - x + 1) * (-4 * x) = -12 * x^3 + 4 * x^2 - 4 * x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1272_127228


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l1272_127222

theorem first_nonzero_digit_of_one_over_137 :
  ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ d < 10 ∧ 
  (∀ (k : ℕ), k < n → (10^(k+1) / 137 % 10 = 0)) ∧
  (10^(n+1) / 137 % 10 = d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_137_l1272_127222


namespace NUMINAMATH_CALUDE_expand_expression_l1272_127203

theorem expand_expression (x : ℝ) : (13 * x + 15) * (2 * x) = 26 * x^2 + 30 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1272_127203


namespace NUMINAMATH_CALUDE_soda_crate_weight_l1272_127209

/-- Given the following conditions:
  - Bridge weight limit is 20,000 pounds
  - Empty truck weight is 12,000 pounds
  - There are 20 soda crates
  - There are 3 dryers
  - Each dryer weighs 3,000 pounds
  - Weight of produce is twice the weight of soda
  - Fully loaded truck weighs 24,000 pounds

  Prove that each soda crate weighs 50 pounds -/
theorem soda_crate_weight :
  ∀ (bridge_limit : ℕ) 
    (empty_truck_weight : ℕ) 
    (num_soda_crates : ℕ) 
    (num_dryers : ℕ) 
    (dryer_weight : ℕ) 
    (loaded_truck_weight : ℕ),
  bridge_limit = 20000 →
  empty_truck_weight = 12000 →
  num_soda_crates = 20 →
  num_dryers = 3 →
  dryer_weight = 3000 →
  loaded_truck_weight = 24000 →
  ∃ (soda_weight produce_weight : ℕ),
    produce_weight = 2 * soda_weight ∧
    loaded_truck_weight = empty_truck_weight + num_dryers * dryer_weight + soda_weight + produce_weight →
    soda_weight / num_soda_crates = 50 :=
by sorry

end NUMINAMATH_CALUDE_soda_crate_weight_l1272_127209


namespace NUMINAMATH_CALUDE_part_one_part_two_l1272_127282

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1) + Real.sqrt (2 - x)}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≥ a, y = 2^x}

-- Part I
theorem part_one : 
  (Set.univ \ A) ∩ B 2 = Set.Ici 4 := by sorry

-- Part II
theorem part_two : 
  ∀ a : ℝ, (Set.univ \ A) ∪ B a = Set.univ ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1272_127282


namespace NUMINAMATH_CALUDE_spade_calculation_l1272_127210

-- Define the ♠ operation
def spade (a b : ℝ) : ℝ := |a - b|

-- Theorem statement
theorem spade_calculation : (spade 5 (spade 3 10)) * (spade 2 4) = 4 := by sorry

end NUMINAMATH_CALUDE_spade_calculation_l1272_127210


namespace NUMINAMATH_CALUDE_hunting_duration_is_three_weeks_l1272_127253

/-- Represents the hunting scenario in the forest -/
structure ForestHunt where
  initialWeasels : ℕ
  initialRabbits : ℕ
  foxes : ℕ
  weaselsPerFoxPerWeek : ℕ
  rabbitsPerFoxPerWeek : ℕ
  remainingRodents : ℕ

/-- Calculates the hunting duration in weeks -/
def huntingDuration (hunt : ForestHunt) : ℚ :=
  let initialRodents := hunt.initialWeasels + hunt.initialRabbits
  let rodentsCaughtPerWeek := hunt.foxes * (hunt.weaselsPerFoxPerWeek + hunt.rabbitsPerFoxPerWeek)
  let totalRodentsCaught := initialRodents - hunt.remainingRodents
  totalRodentsCaught / rodentsCaughtPerWeek

/-- Theorem stating that the hunting duration is 3 weeks for the given scenario -/
theorem hunting_duration_is_three_weeks (hunt : ForestHunt) 
    (h1 : hunt.initialWeasels = 100)
    (h2 : hunt.initialRabbits = 50)
    (h3 : hunt.foxes = 3)
    (h4 : hunt.weaselsPerFoxPerWeek = 4)
    (h5 : hunt.rabbitsPerFoxPerWeek = 2)
    (h6 : hunt.remainingRodents = 96) :
    huntingDuration hunt = 3 := by
  sorry


end NUMINAMATH_CALUDE_hunting_duration_is_three_weeks_l1272_127253


namespace NUMINAMATH_CALUDE_retired_faculty_surveys_l1272_127213

/-- Given a total number of surveys and a ratio of surveys from different groups,
    calculate the number of surveys from the retired faculty. -/
theorem retired_faculty_surveys
  (total_surveys : ℕ)
  (retired_ratio : ℕ)
  (current_ratio : ℕ)
  (student_ratio : ℕ)
  (h1 : total_surveys = 300)
  (h2 : retired_ratio = 2)
  (h3 : current_ratio = 8)
  (h4 : student_ratio = 40) :
  (total_surveys * retired_ratio) / (retired_ratio + current_ratio + student_ratio) = 12 := by
  sorry

#check retired_faculty_surveys

end NUMINAMATH_CALUDE_retired_faculty_surveys_l1272_127213


namespace NUMINAMATH_CALUDE_barbecue_packages_l1272_127255

/-- Represents the number of items in each package type -/
structure PackageSizes where
  hotDogs : Nat
  buns : Nat
  soda : Nat

/-- Represents the number of packages for each item type -/
structure PackageCounts where
  hotDogs : Nat
  buns : Nat
  soda : Nat

/-- Given package sizes, check if the package counts result in equal number of items -/
def hasEqualItems (sizes : PackageSizes) (counts : PackageCounts) : Prop :=
  sizes.hotDogs * counts.hotDogs = sizes.buns * counts.buns ∧
  sizes.hotDogs * counts.hotDogs = sizes.soda * counts.soda

/-- Check if a given package count is the smallest possible -/
def isSmallestCount (sizes : PackageSizes) (counts : PackageCounts) : Prop :=
  ∀ (other : PackageCounts),
    hasEqualItems sizes other →
    counts.hotDogs ≤ other.hotDogs ∧
    counts.buns ≤ other.buns ∧
    counts.soda ≤ other.soda

theorem barbecue_packages :
  let sizes : PackageSizes := ⟨9, 12, 15⟩
  let counts : PackageCounts := ⟨20, 15, 12⟩
  hasEqualItems sizes counts ∧ isSmallestCount sizes counts :=
by sorry

end NUMINAMATH_CALUDE_barbecue_packages_l1272_127255


namespace NUMINAMATH_CALUDE_sara_bought_fifteen_cards_l1272_127204

/-- Calculates the number of baseball cards Sara bought from Sally -/
def cards_bought_by_sara (initial_cards torn_cards remaining_cards : ℕ) : ℕ :=
  initial_cards - torn_cards - remaining_cards

/-- Theorem stating that Sara bought 15 cards from Sally -/
theorem sara_bought_fifteen_cards :
  let initial_cards := 39
  let torn_cards := 9
  let remaining_cards := 15
  cards_bought_by_sara initial_cards torn_cards remaining_cards = 15 := by
  sorry

end NUMINAMATH_CALUDE_sara_bought_fifteen_cards_l1272_127204


namespace NUMINAMATH_CALUDE_vasims_share_l1272_127288

/-- Proves that Vasim's share is 1500 given the specified conditions -/
theorem vasims_share (total : ℕ) (faruk vasim ranjith : ℕ) : 
  faruk + vasim + ranjith = total →
  3 * faruk = 3 * vasim →
  3 * faruk = 3 * vasim ∧ 3 * vasim = 7 * ranjith →
  ranjith - faruk = 2000 →
  vasim = 1500 := by
sorry

end NUMINAMATH_CALUDE_vasims_share_l1272_127288


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1272_127212

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
    x > y ∧ y > z →
    (1 : ℚ) / x + 2 / y + 3 / z = 1 →
    ((x = 36 ∧ y = 9 ∧ z = 4) ∨
     (x = 20 ∧ y = 10 ∧ z = 4) ∨
     (x = 15 ∧ y = 6 ∧ z = 5)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1272_127212


namespace NUMINAMATH_CALUDE_simple_interest_rate_problem_solution_l1272_127293

/-- Simple interest calculation -/
theorem simple_interest_rate (principal time interest : ℝ) (h : principal > 0) (h2 : time > 0) :
  interest = principal * (25 : ℝ) / 100 * time →
  25 = (interest * 100) / (principal * time) :=
by
  sorry

/-- Specific problem instance -/
theorem problem_solution :
  let principal : ℝ := 800
  let time : ℝ := 2
  let interest : ℝ := 400
  (interest * 100) / (principal * time) = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_problem_solution_l1272_127293


namespace NUMINAMATH_CALUDE_gumballs_per_package_l1272_127266

/-- Given that Nathan ate 20 gumballs, finished 4 whole boxes, and no gumballs were left,
    prove that there are 5 gumballs in each package. -/
theorem gumballs_per_package :
  ∀ (total_gumballs : ℕ) (boxes_finished : ℕ) (gumballs_per_package : ℕ),
    total_gumballs = 20 →
    boxes_finished = 4 →
    total_gumballs = boxes_finished * gumballs_per_package →
    gumballs_per_package = 5 := by
  sorry

end NUMINAMATH_CALUDE_gumballs_per_package_l1272_127266


namespace NUMINAMATH_CALUDE_nineteen_team_tournament_games_l1272_127277

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  is_single_elimination : Bool
  no_ties : Bool

/-- Calculates the number of games needed to determine a winner in a tournament. -/
def games_to_determine_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem stating that a single-elimination tournament with 19 teams and no ties requires 18 games to determine a winner. -/
theorem nineteen_team_tournament_games (t : Tournament) 
  (h1 : t.num_teams = 19) 
  (h2 : t.is_single_elimination = true) 
  (h3 : t.no_ties = true) : 
  games_to_determine_winner t = 18 := by
  sorry


end NUMINAMATH_CALUDE_nineteen_team_tournament_games_l1272_127277


namespace NUMINAMATH_CALUDE_total_amount_in_euros_l1272_127238

/-- Proves that the total amount in euros is 172.55 given the specified conditions --/
theorem total_amount_in_euros : 
  ∀ (x y z w : ℝ),
  y = 0.8 * x →
  z = 0.7 * x →
  w = 0.6 * x →
  y = 42 →
  z = 49 →
  x + w = 120 →
  (x + y + z + w) * 0.85 = 172.55 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_in_euros_l1272_127238


namespace NUMINAMATH_CALUDE_ant_movement_l1272_127215

-- Define the grid
structure Grid :=
  (has_black_pairs : Bool)
  (has_white_pairs : Bool)

-- Define the ant's position
inductive Position
  | Black
  | White

-- Define a single move
def move (p : Position) : Position :=
  match p with
  | Position.Black => Position.White
  | Position.White => Position.Black

-- Define the number of moves
def num_moves : Nat := 4

-- Define the function to count black finishing squares
def count_black_finish (g : Grid) : Nat :=
  sorry

-- Theorem statement
theorem ant_movement (g : Grid) :
  g.has_black_pairs = true →
  g.has_white_pairs = true →
  count_black_finish g = 6 :=
sorry

end NUMINAMATH_CALUDE_ant_movement_l1272_127215


namespace NUMINAMATH_CALUDE_negation_of_exponential_inequality_l1272_127247

theorem negation_of_exponential_inequality (P : Prop) :
  (P ↔ ∀ x : ℝ, x > 0 → Real.exp x > x + 1) →
  (¬P ↔ ∃ x : ℝ, x > 0 ∧ Real.exp x ≤ x + 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exponential_inequality_l1272_127247


namespace NUMINAMATH_CALUDE_stock_market_investment_l1272_127202

theorem stock_market_investment
  (initial_investment : ℝ)
  (first_year_increase : ℝ)
  (net_increase : ℝ)
  (h1 : first_year_increase = 0.8)
  (h2 : net_increase = 0.26)
  : ∃ (second_year_decrease : ℝ),
    second_year_decrease = 0.3 ∧
    (1 + first_year_increase) * (1 - second_year_decrease) = 1 + net_increase :=
by sorry

end NUMINAMATH_CALUDE_stock_market_investment_l1272_127202


namespace NUMINAMATH_CALUDE_probability_abs_diff_greater_quarter_l1272_127206

-- Define the coin-flipping method
def coin_flip_method : Real → Prop := sorry

-- Define the probability measure for the coin-flipping method
def P : (Real → Prop) → ℝ := sorry

-- Define the event where |x-y| > 1/4
def event : Real → Real → Prop :=
  fun x y => |x - y| > 1/4

-- Theorem statement
theorem probability_abs_diff_greater_quarter :
  P (fun x => P (fun y => event x y) = 423/1024) = 1 := by sorry

end NUMINAMATH_CALUDE_probability_abs_diff_greater_quarter_l1272_127206


namespace NUMINAMATH_CALUDE_expression_evaluation_l1272_127298

theorem expression_evaluation :
  let x : ℝ := 1
  let y : ℝ := -2
  3 * y^2 - x^2 + (2 * x - y) - (x^2 + 3 * y^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1272_127298


namespace NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l1272_127280

-- Define the ellipse Γ
def Γ (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (0, 1)

-- Define a line l that intersects Γ at two points
def l (k : ℝ) (x y : ℝ) : Prop := y = (k^2 - 1) / (4*k) * x - 1/2

-- Define the property that PQ is a diameter of the circumcircle of MPQ
def isPQDiameterOfCircumcircle (P Q : ℝ × ℝ) : Prop :=
  (P.1 - M.1) * (Q.1 - M.1) + (P.2 - M.2) * (Q.2 - M.2) = 0

theorem ellipse_intersection_fixed_point :
  ∀ (k : ℝ) (P Q : ℝ × ℝ),
    k ≠ 0 →
    Γ P.1 P.2 →
    Γ Q.1 Q.2 →
    l k P.1 P.2 →
    l k Q.1 Q.2 →
    isPQDiameterOfCircumcircle P Q →
    P ≠ M ∧ Q ≠ M →
    ∃ (x y : ℝ), l k x y ∧ x = 0 ∧ y = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_fixed_point_l1272_127280


namespace NUMINAMATH_CALUDE_total_amount_theorem_l1272_127207

/-- Represents the types of books in the collection -/
inductive BookType
  | Novel
  | Biography
  | ScienceBook

/-- Calculates the total amount received from book sales -/
def calculateTotalAmount (totalBooks : ℕ) 
                         (soldPercentages : BookType → ℚ)
                         (prices : BookType → ℕ)
                         (remainingBooks : BookType → ℕ) : ℕ :=
  sorry

/-- The main theorem stating the total amount received from book sales -/
theorem total_amount_theorem (totalBooks : ℕ)
                             (soldPercentages : BookType → ℚ)
                             (prices : BookType → ℕ)
                             (remainingBooks : BookType → ℕ) : 
  totalBooks = 300 ∧
  soldPercentages BookType.Novel = 3/5 ∧
  soldPercentages BookType.Biography = 2/3 ∧
  soldPercentages BookType.ScienceBook = 7/10 ∧
  prices BookType.Novel = 4 ∧
  prices BookType.Biography = 7 ∧
  prices BookType.ScienceBook = 6 ∧
  remainingBooks BookType.Novel = 30 ∧
  remainingBooks BookType.Biography = 35 ∧
  remainingBooks BookType.ScienceBook = 25 →
  calculateTotalAmount totalBooks soldPercentages prices remainingBooks = 1018 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_theorem_l1272_127207


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1272_127223

def U : Set Nat := {2, 3, 6, 8}
def A : Set Nat := {2, 3}
def B : Set Nat := {2, 6, 8}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {6, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1272_127223


namespace NUMINAMATH_CALUDE_absolute_value_sum_l1272_127269

theorem absolute_value_sum (x p : ℝ) : 
  (|x - 2| = p) → (x > 2) → (x + p = 2*p + 2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l1272_127269


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l1272_127286

theorem smallest_integer_solution (x : ℤ) : x^2 - x = 24 → x ≥ -4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l1272_127286


namespace NUMINAMATH_CALUDE_money_difference_proof_l1272_127276

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- Charles' quarters -/
def charles_quarters (q : ℕ) : ℕ := 7 * q + 3

/-- Richard's quarters -/
def richard_quarters (q : ℕ) : ℕ := 3 * q + 7

/-- The difference in money between Charles and Richard, expressed in nickels -/
def money_difference_in_nickels (q : ℕ) : ℕ := 
  nickels_per_quarter * (charles_quarters q - richard_quarters q)

theorem money_difference_proof (q : ℕ) : 
  money_difference_in_nickels q = 20 * (q - 1) := by
  sorry

end NUMINAMATH_CALUDE_money_difference_proof_l1272_127276


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1272_127249

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1272_127249


namespace NUMINAMATH_CALUDE_ladder_length_l1272_127278

/-- Proves that the length of a ladder is 9.2 meters, given specific conditions. -/
theorem ladder_length (angle : Real) (foot_distance : Real) (length : Real) : 
  angle = 60 * π / 180 →
  foot_distance = 4.6 →
  Real.cos angle = foot_distance / length →
  length = 9.2 := by
sorry

end NUMINAMATH_CALUDE_ladder_length_l1272_127278


namespace NUMINAMATH_CALUDE_calculation_proofs_l1272_127239

theorem calculation_proofs :
  (4800 / 125 = 38.4) ∧ (13 * 74 + 27 * 13 - 13 = 1300) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l1272_127239


namespace NUMINAMATH_CALUDE_rational_division_l1272_127246

theorem rational_division (x : ℚ) : (-2 : ℚ) / x = 8 → x = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_rational_division_l1272_127246


namespace NUMINAMATH_CALUDE_number_puzzle_l1272_127234

theorem number_puzzle : ∃ x : ℚ, 45 + 3 * x = 60 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l1272_127234


namespace NUMINAMATH_CALUDE_paperback_ratio_l1272_127231

/-- Represents the number of books Thabo owns in each category. -/
structure BookCollection where
  paperbackFiction : ℕ
  paperbackNonfiction : ℕ
  hardcoverNonfiction : ℕ

/-- Thabo's book collection satisfies the given conditions. -/
def validCollection (books : BookCollection) : Prop :=
  books.paperbackFiction + books.paperbackNonfiction + books.hardcoverNonfiction = 200 ∧
  books.paperbackNonfiction = books.hardcoverNonfiction + 20 ∧
  books.hardcoverNonfiction = 35

/-- The theorem stating that for any valid book collection, 
    the ratio of paperback fiction to paperback nonfiction is 2:1. -/
theorem paperback_ratio (books : BookCollection) 
  (h : validCollection books) : 
  books.paperbackFiction * 1 = books.paperbackNonfiction * 2 := by
  sorry

end NUMINAMATH_CALUDE_paperback_ratio_l1272_127231


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1272_127240

theorem roots_of_quadratic (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x : ℝ, x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1272_127240


namespace NUMINAMATH_CALUDE_hyperbola_center_l1272_127290

/-- The center of a hyperbola given by the equation ((3y+3)^2)/(7^2) - ((4x-8)^2)/(6^2) = 1 -/
theorem hyperbola_center (x y : ℝ) : 
  (((3 * y + 3)^2) / 7^2) - (((4 * x - 8)^2) / 6^2) = 1 → 
  (∃ (h k : ℝ), h = 2 ∧ k = -1 ∧ 
    ((y - k)^2) / ((7/3)^2) - ((x - h)^2) / ((3/2)^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1272_127290
