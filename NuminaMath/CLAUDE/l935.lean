import Mathlib

namespace unique_solution_condition_l935_93573

theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, x ≠ 2 ∧ x ≠ -1 ∧ |x + 2| = |x| + a) ↔ a ∈ Set.Ioo (-2) 0 ∪ Set.Ioo 0 2 := by
  sorry

end unique_solution_condition_l935_93573


namespace ellipse_equation_l935_93554

/-- Given a hyperbola and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (h : ℝ → ℝ → Prop) (e : ℝ → ℝ → Prop) :
  (∀ x y, h x y ↔ y^2/12 - x^2/4 = 1) →  -- Definition of the hyperbola
  (∃ a b, ∀ x y, e x y ↔ x^2/a + y^2/b = 1) →  -- General form of the ellipse
  (∃ v₁ v₂, v₁ ≠ v₂ ∧ h 0 v₁ ∧ h 0 (-v₁) ∧ 
    ∀ x y, e x y → (x - 0)^2 + (y - v₁)^2 + (x - 0)^2 + (y + v₁)^2 = 16) →  -- Vertices of hyperbola as foci of ellipse
  (∀ x y, e x y ↔ x^2/4 + y^2/16 = 1) :=
by sorry

end ellipse_equation_l935_93554


namespace man_son_age_ratio_l935_93539

/-- The ratio of a man's age to his son's age in two years -/
def age_ratio (man_age son_age : ℕ) : ℚ :=
  (man_age + 2) / (son_age + 2)

/-- Theorem stating the age ratio of a man to his son in two years -/
theorem man_son_age_ratio (son_age : ℕ) (h1 : son_age = 22) :
  age_ratio (son_age + 24) son_age = 2 := by
  sorry

#check man_son_age_ratio

end man_son_age_ratio_l935_93539


namespace circle_tangent_to_line_l935_93549

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Define the tangent line
def tangent_line (y : ℝ) : Prop := y + 1 = 0

-- Define the distance from a point to the tangent line
def distance_to_line (x y : ℝ) : ℝ := |y + 1|

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Theorem statement
theorem circle_tangent_to_line :
  ∀ x y : ℝ, circle_equation x y →
  distance_to_line x y = (4 : ℝ).sqrt ∧
  ∃ p : ℝ × ℝ, p.1 = x ∧ p.2 = y ∧ tangent_line p.2 :=
by sorry

end circle_tangent_to_line_l935_93549


namespace school_sample_size_l935_93565

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  total_population : ℕ
  stratum_size : ℕ
  stratum_sample : ℕ
  total_sample : ℕ

/-- Checks if the given stratified sample is proportional -/
def is_proportional_sample (s : StratifiedSample) : Prop :=
  s.stratum_sample * s.total_population = s.total_sample * s.stratum_size

/-- Theorem stating that for the given population and sample sizes, 
    the total sample size is 45 -/
theorem school_sample_size :
  ∀ (s : StratifiedSample), 
    s.total_population = 1500 →
    s.stratum_size = 400 →
    s.stratum_sample = 12 →
    is_proportional_sample s →
    s.total_sample = 45 := by
  sorry

end school_sample_size_l935_93565


namespace max_value_expression_l935_93512

theorem max_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hbc : b > c) (hca : c > a) (ha_neq_zero : a ≠ 0) : 
  ∃ (x : ℝ), x = ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / a^2 ∧ 
  x ≤ 44 ∧ 
  ∀ (y : ℝ), y = ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2) / a^2 → y ≤ x :=
by sorry

end max_value_expression_l935_93512


namespace parabola_max_value_l935_93546

theorem parabola_max_value :
  ∃ (max : ℝ), max = 4 ∧ ∀ (x : ℝ), -x^2 + 2*x + 3 ≤ max :=
by sorry

end parabola_max_value_l935_93546


namespace cos_450_degrees_l935_93503

theorem cos_450_degrees : Real.cos (450 * π / 180) = 0 := by
  sorry

end cos_450_degrees_l935_93503


namespace reciprocal_sum_of_quadratic_roots_l935_93529

theorem reciprocal_sum_of_quadratic_roots (γ δ : ℝ) : 
  (∃ r s : ℝ, 7 * r^2 + 5 * r + 3 = 0 ∧ 
              7 * s^2 + 5 * s + 3 = 0 ∧ 
              γ = 1 / r ∧ 
              δ = 1 / s) → 
  γ + δ = -5/3 := by
sorry

end reciprocal_sum_of_quadratic_roots_l935_93529


namespace simplified_expression_l935_93575

theorem simplified_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = 3*(a + b)) :
  a/b + b/a - 3/(a*b) = 1 := by
sorry

end simplified_expression_l935_93575


namespace janet_stickers_l935_93547

theorem janet_stickers (initial_stickers received_stickers : ℕ) : 
  initial_stickers = 3 → received_stickers = 53 → initial_stickers + received_stickers = 56 := by
  sorry

end janet_stickers_l935_93547


namespace savings_of_eight_hundred_bills_l935_93576

/-- The total savings amount when exchanged into a given number of $100 bills -/
def savings_amount (num_bills : ℕ) : ℕ := 100 * num_bills

/-- Theorem: If a person has 8 $100 bills after exchanging all their savings, 
    their total savings amount to $800 -/
theorem savings_of_eight_hundred_bills : 
  savings_amount 8 = 800 := by sorry

end savings_of_eight_hundred_bills_l935_93576


namespace linear_function_not_in_second_quadrant_l935_93501

/-- A linear function f(x) = mx + b does not pass through the second quadrant
    if its slope m is positive and its y-intercept b is negative. -/
theorem linear_function_not_in_second_quadrant 
  (f : ℝ → ℝ) (m b : ℝ) (h1 : ∀ x, f x = m * x + b) (h2 : m > 0) (h3 : b < 0) :
  ∃ x y, f x = y ∧ (x ≤ 0 ∨ y ≤ 0) :=
sorry

end linear_function_not_in_second_quadrant_l935_93501


namespace not_perfect_squares_l935_93513

theorem not_perfect_squares : ∃ (n : ℕ → ℕ), 
  (n 1 = 2048) ∧ 
  (n 2 = 2049) ∧ 
  (n 3 = 2050) ∧ 
  (n 4 = 2051) ∧ 
  (n 5 = 2052) ∧ 
  (∃ (a : ℕ), 1^(n 1) = a^2) ∧ 
  (¬∃ (b : ℕ), 2^(n 2) = b^2) ∧ 
  (∃ (c : ℕ), 3^(n 3) = c^2) ∧ 
  (¬∃ (d : ℕ), 4^(n 4) = d^2) ∧ 
  (∃ (e : ℕ), 5^(n 5) = e^2) :=
by sorry


end not_perfect_squares_l935_93513


namespace max_green_lily_students_l935_93506

-- Define variables
variable (x : ℝ) -- Cost of green lily
variable (y : ℝ) -- Cost of spider plant
variable (m : ℝ) -- Number of students taking care of green lilies

-- Define conditions
axiom condition1 : 2 * x + 3 * y = 36
axiom condition2 : x + 2 * y = 21
axiom total_students : m + (48 - m) = 48
axiom cost_constraint : m * x + (48 - m) * y ≤ 378

-- Theorem to prove
theorem max_green_lily_students : 
  ∃ m : ℝ, m ≤ 30 ∧ 
  ∀ n : ℝ, (n * x + (48 - n) * y ≤ 378 → n ≤ m) :=
sorry

end max_green_lily_students_l935_93506


namespace angle_D_value_l935_93500

-- Define the angles as real numbers
variable (A B C D F : ℝ)

-- State the theorem
theorem angle_D_value (h1 : A + B = 180)
                      (h2 : C = D)
                      (h3 : B = 90)
                      (h4 : F = 50)
                      (h5 : A + C + F = 180) : D = 40 := by
  sorry

end angle_D_value_l935_93500


namespace initial_subscribers_count_l935_93589

/-- Represents the monthly income of a streamer based on their number of subscribers -/
def streamer_income (initial_subscribers : ℕ) (gift_subscribers : ℕ) (income_per_subscriber : ℕ) : ℕ :=
  (initial_subscribers + gift_subscribers) * income_per_subscriber

/-- Proves that the initial number of subscribers is 150 given the problem conditions -/
theorem initial_subscribers_count :
  ∃ (x : ℕ), streamer_income x 50 9 = 1800 ∧ x = 150 := by
  sorry

end initial_subscribers_count_l935_93589


namespace daily_allowance_calculation_l935_93597

/-- Proves that if a person saves half of their daily allowance for 6 days
    and a quarter of their daily allowance for 1 day, and the total saved is $39,
    then their daily allowance is $12. -/
theorem daily_allowance_calculation (allowance : ℚ) : 
  (6 * (allowance / 2) + 1 * (allowance / 4) = 39) → allowance = 12 := by
  sorry

end daily_allowance_calculation_l935_93597


namespace linear_function_properties_l935_93567

def f (x : ℝ) := -2 * x + 1

theorem linear_function_properties :
  (∀ x y, x < y → f x > f y) ∧  -- decreasing
  (∀ x, f x - (-2 * x) = 1) ∧  -- parallel to y = -2x
  (f 0 = 1) ∧  -- intersection with y-axis
  (∃ x y z, x > 0 ∧ y < 0 ∧ z > 0 ∧ f x > 0 ∧ f y < 0 ∧ f z < 0) :=  -- passes through 1st, 2nd, and 4th quadrants
by sorry

end linear_function_properties_l935_93567


namespace not_always_zero_l935_93591

def heartsuit (x y : ℝ) : ℝ := |x - 2*y|

theorem not_always_zero : ¬ ∀ x : ℝ, heartsuit x x = 0 := by
  sorry

end not_always_zero_l935_93591


namespace problem_solution_l935_93563

theorem problem_solution (a z : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * z) : z = 2205 := by
  sorry

end problem_solution_l935_93563


namespace leahs_coins_value_l935_93574

theorem leahs_coins_value (p n : ℕ) : 
  p + n = 15 ∧ 
  p = 2 * (n + 3) → 
  5 * n + p = 27 :=
by sorry

end leahs_coins_value_l935_93574


namespace x_coordinate_range_l935_93514

-- Define the line L
def L (x y : ℝ) : Prop := x + y - 9 = 0

-- Define the circle M
def M (x y : ℝ) : Prop := 2*x^2 + 2*y^2 - 8*x - 8*y - 1 = 0

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  A_on_L : L A.1 A.2
  B_on_M : M B.1 B.2
  C_on_M : M C.1 C.2
  angle_BAC : Real.cos (45 * π / 180) = (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) /
    (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) * Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2))

-- Define that AB passes through the center of M
def AB_through_center (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 2 = A.1 + t * (B.1 - A.1) ∧ 2 = A.2 + t * (B.2 - A.2)

-- Main theorem
theorem x_coordinate_range (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C) (h_AB : AB_through_center A B) : 
  3 ≤ A.1 ∧ A.1 ≤ 6 := by
  sorry

end x_coordinate_range_l935_93514


namespace student_subject_assignment_l935_93587

/-- The number of ways to assign students to subjects. -/
def num_assignments (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of students. -/
def num_students : ℕ := 4

/-- The number of subjects. -/
def num_subjects : ℕ := 3

theorem student_subject_assignment :
  num_assignments num_students num_subjects = 81 := by
  sorry

end student_subject_assignment_l935_93587


namespace compute_expression_l935_93559

theorem compute_expression (x : ℝ) (h : x = 9) : 
  (x^9 - 27*x^6 + 729) / (x^6 - 27) = 730 + 1/26 := by
  sorry

end compute_expression_l935_93559


namespace fraction_simplification_l935_93570

theorem fraction_simplification :
  (12 : ℚ) / 11 * 15 / 28 * 44 / 45 = 4 / 7 := by
  sorry

end fraction_simplification_l935_93570


namespace five_thirteenths_repeating_decimal_sum_l935_93562

theorem five_thirteenths_repeating_decimal_sum (c d : ℕ) : 
  (5 : ℚ) / 13 = 0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d + 
    (0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + 0.00001 * c + 0.000001 * d) / 999999 →
  c + d = 11 :=
by sorry

end five_thirteenths_repeating_decimal_sum_l935_93562


namespace triangle_side_length_l935_93548

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if ∠B = 60°, ∠C = 75°, and a = 4, then b = 2√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  B = 60 * π / 180 →
  C = 75 * π / 180 →
  a = 4 →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) →
  b = 2 * Real.sqrt 6 := by
  sorry

end triangle_side_length_l935_93548


namespace arithmetic_sequence_sum_l935_93531

theorem arithmetic_sequence_sum (a₁ aₙ d : ℕ) (n : ℕ) (h1 : a₁ = 1) (h2 : aₙ = 31) (h3 : d = 3) (h4 : n = (aₙ - a₁) / d + 1) :
  (n : ℝ) / 2 * (a₁ + aₙ) = 176 := by
  sorry

end arithmetic_sequence_sum_l935_93531


namespace target_distribution_l935_93540

def target_parts : Nat := 10

def arrange_decreasing (n : Nat) (center : Nat) (middle : Nat) (outer : Nat) : Nat :=
  sorry

def arrange_equal_sum (n : Nat) (center : Nat) (middle : Nat) (outer : Nat) : Nat :=
  sorry

theorem target_distribution :
  (Nat.factorial target_parts = 3628800) ∧
  (arrange_decreasing target_parts 1 3 6 = 4320) ∧
  (arrange_equal_sum target_parts 1 3 6 = 34560) := by
  sorry

end target_distribution_l935_93540


namespace jacket_price_reduction_l935_93592

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 → 
  0 ≤ x → 
  x ≤ 100 → 
  P * (1 - x / 100) * (1 - 20 / 100) * (1 + 2 / 3) = P → 
  x = 25 := by
sorry

end jacket_price_reduction_l935_93592


namespace smallest_integer_satisfying_inequality_l935_93538

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * y - 4 > 2 * y + 5) → y ≥ 10 ∧ (3 * 10 - 4 > 2 * 10 + 5) := by
  sorry

end smallest_integer_satisfying_inequality_l935_93538


namespace tangent_point_and_perpendicular_line_l935_93511

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P₀
def P₀ : ℝ × ℝ := (-1, -4)

-- Define the slope of the line parallel to the tangent
def parallel_slope : ℝ := 4

theorem tangent_point_and_perpendicular_line :
  -- The tangent line at P₀ is parallel to 4x - y - 1 = 0
  curve_derivative (P₀.1) = parallel_slope →
  -- P₀ is in the third quadrant
  P₀.1 < 0 ∧ P₀.2 < 0 →
  -- P₀ lies on the curve
  curve P₀.1 = P₀.2 →
  -- The equation of the perpendicular line passing through P₀
  ∃ (a b c : ℝ), a * P₀.1 + b * P₀.2 + c = 0 ∧
                 a = 1 ∧ b = 4 ∧ c = 17 ∧
                 -- The perpendicular line is indeed perpendicular to the tangent
                 a * parallel_slope + b = 0 :=
by sorry

end tangent_point_and_perpendicular_line_l935_93511


namespace job_age_is_five_l935_93523

def freddy_age : ℕ := 18
def stephanie_age : ℕ := freddy_age + 2

theorem job_age_is_five :
  ∃ (job_age : ℕ), stephanie_age = 4 * job_age ∧ job_age = 5 := by
  sorry

end job_age_is_five_l935_93523


namespace problem_solution_l935_93595

theorem problem_solution : Real.sqrt 9 + 2⁻¹ + (-1)^2023 = 5/2 := by
  sorry

end problem_solution_l935_93595


namespace arithmetic_sequence_common_difference_l935_93557

/-- An arithmetic sequence with given terms -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a2 : a 2 = 12)
  (h_a6 : a 6 = 4) :
  ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l935_93557


namespace max_value_abc_expression_l935_93505

theorem max_value_abc_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≥ b * c^2) (hbc : b ≥ c * a^2) (hca : c ≥ a * b^2) :
  a * b * c * (a - b * c^2) * (b - c * a^2) * (c - a * b^2) ≤ 0 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    a₀ ≥ b₀ * c₀^2 ∧ b₀ ≥ c₀ * a₀^2 ∧ c₀ ≥ a₀ * b₀^2 ∧
    a₀ * b₀ * c₀ * (a₀ - b₀ * c₀^2) * (b₀ - c₀ * a₀^2) * (c₀ - a₀ * b₀^2) = 0 :=
by
  sorry

end max_value_abc_expression_l935_93505


namespace expression_simplification_l935_93502

theorem expression_simplification (b x : ℝ) 
  (hb : b ≠ 0) (hx : x ≠ 0) (hxb : x ≠ b/2) (hxb2 : x ≠ -2/b) :
  ((b*x + 4 + 4/(b*x)) / (2*b + (b^2 - 4)*x - 2*b*x^2) + 
   ((4*x^2 - b^2) / b) / ((b + 2*x)^2 - 8*b*x)) * (b*x/2) = 
  (x^2 - 1) / (2*x - b) := by
sorry

end expression_simplification_l935_93502


namespace number_divisible_by_nine_missing_digit_correct_l935_93535

/-- The missing digit in the five-digit number 385_7 that makes it divisible by 9 -/
def missing_digit : ℕ := 4

/-- The five-digit number with the missing digit filled in -/
def number : ℕ := 38547

theorem number_divisible_by_nine :
  number % 9 = 0 :=
sorry

theorem missing_digit_correct :
  ∃ (d : ℕ), d < 10 ∧ 38500 + d * 10 + 7 = number ∧ (38500 + d * 10 + 7) % 9 = 0 → d = missing_digit :=
sorry

end number_divisible_by_nine_missing_digit_correct_l935_93535


namespace parallel_vectors_magnitude_l935_93558

/-- Given two planar vectors m and n, where m is parallel to n, 
    prove that the magnitude of n is 2√5. -/
theorem parallel_vectors_magnitude (m n : ℝ × ℝ) : 
  m = (-1, 2) → 
  n.1 = 2 → 
  (∃ k : ℝ, n = k • m) → 
  ‖n‖ = 2 * Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l935_93558


namespace integral_one_plus_sin_l935_93542

theorem integral_one_plus_sin : ∫ x in -π..π, (1 + Real.sin x) = 2 * π := by sorry

end integral_one_plus_sin_l935_93542


namespace product_of_digits_5432_base8_l935_93543

/-- Convert a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculate the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_digits_5432_base8 :
  productOfList (toBase8 5432) = 0 := by
  sorry

end product_of_digits_5432_base8_l935_93543


namespace simplify_expression1_simplify_expression2_simplify_expression3_l935_93569

-- Expression 1
theorem simplify_expression1 (a b x : ℝ) (h : b ≠ 0) :
  (12 * a^3 * x^4 + 2 * a^2 * x^5) / (18 * a * b^2 * x + 3 * b^2 * x^2) = 
  (2 * a^2 * x^3) / (3 * b^2) :=
sorry

-- Expression 2
theorem simplify_expression2 (x : ℝ) (h : x ≠ -2) :
  (4 - 2*x + x^2) / (x + 2) - x - 2 = -6*x / (x + 2) :=
sorry

-- Expression 3
theorem simplify_expression3 (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  1 / ((a-b)*(a-c)) + 1 / ((b-a)*(b-c)) + 1 / ((c-a)*(c-b)) = 0 :=
sorry

end simplify_expression1_simplify_expression2_simplify_expression3_l935_93569


namespace subtraction_rule_rational_l935_93596

theorem subtraction_rule_rational (x : ℚ) : ∀ y : ℚ, y - x = y + (-x) := by
  sorry

end subtraction_rule_rational_l935_93596


namespace equilateral_triangle_third_vertex_l935_93537

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- Check if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Check if three points form an equilateral triangle -/
def isEquilateral (t : EquilateralTriangle) : Prop :=
  let d12 := ((t.v1.x - t.v2.x)^2 + (t.v1.y - t.v2.y)^2)
  let d23 := ((t.v2.x - t.v3.x)^2 + (t.v2.y - t.v3.y)^2)
  let d31 := ((t.v3.x - t.v1.x)^2 + (t.v3.y - t.v1.y)^2)
  d12 = d23 ∧ d23 = d31

theorem equilateral_triangle_third_vertex 
  (t : EquilateralTriangle)
  (h1 : t.v1 = ⟨0, 3⟩)
  (h2 : t.v2 = ⟨6, 3⟩)
  (h3 : isInFirstQuadrant t.v3)
  (h4 : isEquilateral t) :
  t.v3 = ⟨6, 3 + 3 * Real.sqrt 3⟩ := by
  sorry

end equilateral_triangle_third_vertex_l935_93537


namespace cube_surface_area_l935_93526

/-- Given a cube made up of 6 squares, each with a perimeter of 24 cm,
    prove that its surface area is 216 cm². -/
theorem cube_surface_area (cube_side_length : ℝ) (square_perimeter : ℝ) : 
  square_perimeter = 24 →
  cube_side_length = square_perimeter / 4 →
  6 * cube_side_length ^ 2 = 216 := by
  sorry

end cube_surface_area_l935_93526


namespace triangle_angle_bound_triangle_side_ratio_l935_93560

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a + t.c = 2 * t.b

theorem triangle_angle_bound (t : Triangle) (h : TriangleConditions t) : t.B ≤ Real.pi / 3 := by
  sorry

theorem triangle_side_ratio (t : Triangle) (h : TriangleConditions t) (h2 : t.C = 2 * t.A) :
  ∃ (k : ℝ), t.a = 4 * k ∧ t.b = 5 * k ∧ t.c = 6 * k := by
  sorry

end triangle_angle_bound_triangle_side_ratio_l935_93560


namespace tangent_line_intersection_l935_93582

theorem tangent_line_intersection (a : ℝ) : 
  (∃ x : ℝ, x + Real.log x = a * x^2 + (a + 2) * x + 1 ∧ 
   2 * x - 1 = a * x^2 + (a + 2) * x + 1) ↔ 
  a = 8 := by sorry

end tangent_line_intersection_l935_93582


namespace cos_alpha_minus_beta_l935_93504

theorem cos_alpha_minus_beta (α β : Real) 
  (h1 : α > -π/4 ∧ α < π/4) 
  (h2 : β > -π/4 ∧ β < π/4) 
  (h3 : Real.cos (2*α + 2*β) = -7/9) 
  (h4 : Real.sin α * Real.sin β = 1/4) : 
  Real.cos (α - β) = 5/6 := by
sorry

end cos_alpha_minus_beta_l935_93504


namespace snow_at_brecknock_l935_93516

/-- The amount of snow at Mrs. Hilt's house in inches -/
def mrs_hilt_snow : ℕ := 29

/-- The difference in snow between Mrs. Hilt's house and Brecknock Elementary School in inches -/
def snow_difference : ℕ := 12

/-- The amount of snow at Brecknock Elementary School in inches -/
def brecknock_snow : ℕ := mrs_hilt_snow - snow_difference

theorem snow_at_brecknock : brecknock_snow = 17 := by
  sorry

end snow_at_brecknock_l935_93516


namespace extreme_point_iff_a_eq_zero_l935_93536

/-- The function f(x) as defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

/-- Definition of an extreme point -/
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≥ f x ∨ f y ≤ f x

/-- The main theorem stating that x=1 is an extreme point of f(x) iff a=0 -/
theorem extreme_point_iff_a_eq_zero (a : ℝ) :
  is_extreme_point (f a) 1 ↔ a = 0 := by sorry

end extreme_point_iff_a_eq_zero_l935_93536


namespace translation_transforms_function_l935_93533

/-- The translation vector -/
def translation_vector : ℝ × ℝ := (2, -3)

/-- The original function -/
def original_function (x : ℝ) : ℝ := x^2 + 4*x + 7

/-- The translated function -/
def translated_function (x : ℝ) : ℝ := x^2

theorem translation_transforms_function :
  ∀ x y : ℝ,
  original_function (x - translation_vector.1) + translation_vector.2 = translated_function x :=
by sorry

end translation_transforms_function_l935_93533


namespace hydroflow_pump_calculation_l935_93507

/-- The rate at which the Hydroflow system pumps water, in gallons per hour -/
def pump_rate : ℝ := 360

/-- The time in minutes for which we want to calculate the amount of water pumped -/
def pump_time : ℝ := 30

/-- Theorem stating that the Hydroflow system pumps 180 gallons in 30 minutes -/
theorem hydroflow_pump_calculation : 
  pump_rate * (pump_time / 60) = 180 := by sorry

end hydroflow_pump_calculation_l935_93507


namespace best_of_three_prob_l935_93534

/-- The probability of winning a single set -/
def p : ℝ := 0.6

/-- The probability of winning a best of 3 sets match -/
def match_win_prob : ℝ := p^2 + 3 * p^2 * (1 - p)

theorem best_of_three_prob : match_win_prob = 0.648 := by sorry

end best_of_three_prob_l935_93534


namespace fabian_marbles_comparison_l935_93522

theorem fabian_marbles_comparison (fabian_marbles kyle_marbles miles_marbles : ℕ) : 
  fabian_marbles = 15 →
  fabian_marbles = 3 * kyle_marbles →
  kyle_marbles + miles_marbles = 8 →
  fabian_marbles = 5 * miles_marbles :=
by
  sorry

end fabian_marbles_comparison_l935_93522


namespace triangle_segment_calculation_l935_93571

/-- Given a triangle ABC with point D on AB and point E on AD, prove that FC has a specific value. -/
theorem triangle_segment_calculation (DC CB : ℝ) (h1 : DC = 10) (h2 : CB = 12)
  (AB AD ED : ℝ) (h3 : AB = (1/5) * AD) (h4 : ED = (2/3) * AD) : 
  ∃ (FC : ℝ), FC = 35/3 := by
  sorry

end triangle_segment_calculation_l935_93571


namespace equation_solution_l935_93515

theorem equation_solution :
  ∃! x : ℝ, x ≠ 3 ∧ x + 60 / (x - 3) = -13 :=
by
  -- The unique solution is x = -7
  use -7
  constructor
  · -- Prove that x = -7 satisfies the equation
    constructor
    · -- Prove -7 ≠ 3
      linarith
    · -- Prove -7 + 60 / (-7 - 3) = -13
      ring
  · -- Prove uniqueness
    intro y hy
    -- Assume y satisfies the equation
    have h1 : y ≠ 3 := hy.1
    have h2 : y + 60 / (y - 3) = -13 := hy.2
    -- Derive that y must equal -7
    sorry


end equation_solution_l935_93515


namespace min_sum_product_l935_93552

theorem min_sum_product (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 1/m + 9/n = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → m + n ≤ a + b) →
  m * n = 48 := by
sorry

end min_sum_product_l935_93552


namespace max_sides_cube_plane_intersection_l935_93579

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- A plane is a flat, two-dimensional surface -/
structure Plane where
  -- We don't need to define the specifics of a plane for this problem

/-- A polygon is a plane figure with straight sides -/
structure Polygon where
  sides : ℕ

/-- The cross-section formed when a plane intersects a cube -/
def crossSection (c : Cube) (p : Plane) : Polygon :=
  sorry -- Implementation details not needed for the statement

/-- The maximum number of sides a polygon can have when it's formed by a plane intersecting a cube is 6 -/
theorem max_sides_cube_plane_intersection (c : Cube) (p : Plane) :
  (crossSection c p).sides ≤ 6 ∧ ∃ (c : Cube) (p : Plane), (crossSection c p).sides = 6 :=
sorry

end max_sides_cube_plane_intersection_l935_93579


namespace algebraic_simplification_l935_93508

theorem algebraic_simplification (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n := by
  sorry

end algebraic_simplification_l935_93508


namespace x_range_lower_bound_l935_93527

theorem x_range_lower_bound (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) :
  x ≥ 12 := by
  sorry

end x_range_lower_bound_l935_93527


namespace equation_solutions_l935_93564

theorem equation_solutions : 
  ∀ x : ℝ, x^4 + (4 - x)^4 = 272 ↔ x = 2 + Real.sqrt 6 ∨ x = 2 - Real.sqrt 6 :=
by sorry

end equation_solutions_l935_93564


namespace complement_implies_sum_l935_93519

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (m : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - m) > 0}

-- Define the complement of A in U
def C_UA (m n : ℝ) : Set ℝ := Set.Icc (-1) (-n)

-- Theorem statement
theorem complement_implies_sum (m n : ℝ) : 
  C_UA m n = Set.compl (A m) → m + n = -2 := by
  sorry

end complement_implies_sum_l935_93519


namespace scalene_triangle_ratio_bounds_l935_93541

theorem scalene_triangle_ratio_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_scalene : a > b ∧ b > c) (h_avg : a + c = 2 * b) : 1/3 < c/a ∧ c/a < 1 := by
  sorry

end scalene_triangle_ratio_bounds_l935_93541


namespace sum_of_squares_geq_sum_of_products_inequality_of_square_roots_l935_93580

-- Statement 1
theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) : 
  a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := by sorry

-- Statement 2
theorem inequality_of_square_roots : 
  Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by sorry

end sum_of_squares_geq_sum_of_products_inequality_of_square_roots_l935_93580


namespace johns_apartment_rental_l935_93585

/-- John's apartment rental problem -/
theorem johns_apartment_rental 
  (num_subletters : ℕ) 
  (subletter_payment : ℕ) 
  (annual_profit : ℕ) 
  (monthly_rent : ℕ) : 
  num_subletters = 3 → 
  subletter_payment = 400 → 
  annual_profit = 3600 → 
  monthly_rent = 900 → 
  (num_subletters * subletter_payment - monthly_rent) * 12 = annual_profit :=
by sorry

end johns_apartment_rental_l935_93585


namespace circular_field_area_l935_93578

-- Define the constants
def fencing_cost_per_metre : ℝ := 4
def total_fencing_cost : ℝ := 5941.9251828093165

-- Define the theorem
theorem circular_field_area :
  ∃ (area : ℝ),
    (area ≥ 17.55 ∧ area ≤ 17.57) ∧
    (∃ (circumference radius : ℝ),
      circumference = total_fencing_cost / fencing_cost_per_metre ∧
      radius = circumference / (2 * Real.pi) ∧
      area = (Real.pi * radius ^ 2) / 10000) :=
by sorry

end circular_field_area_l935_93578


namespace cruise_ship_cabins_l935_93572

/-- Represents the total number of cabins on a cruise ship -/
def total_cabins : ℕ := 600

/-- Represents the number of Deluxe cabins -/
def deluxe_cabins : ℕ := 30

/-- Theorem stating that the total number of cabins on the cruise ship is 600 -/
theorem cruise_ship_cabins :
  (deluxe_cabins : ℝ) + 0.2 * total_cabins + 3/4 * total_cabins = total_cabins :=
by sorry

end cruise_ship_cabins_l935_93572


namespace trigonometric_simplification_l935_93566

theorem trigonometric_simplification (α : Real) :
  (1 - 2 * Real.sin α ^ 2) / (2 * Real.tan (5 * Real.pi / 4 + α) * Real.cos (Real.pi / 4 + α) ^ 2) -
  Real.tan α + Real.sin (Real.pi / 2 + α) - Real.cos (α - Real.pi / 2) =
  (2 * Real.sqrt 2 * Real.cos (Real.pi / 4 + α) * Real.cos (α / 2) ^ 2) / Real.cos α :=
by sorry

end trigonometric_simplification_l935_93566


namespace contrapositive_equality_l935_93550

theorem contrapositive_equality (a b : ℝ) : 
  (¬(a = 0 → a * b = 0)) ↔ (a * b ≠ 0 → a ≠ 0) := by sorry

end contrapositive_equality_l935_93550


namespace csc_135_deg_l935_93509

-- Define the cosecant function
noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

-- State the theorem
theorem csc_135_deg : csc (135 * π / 180) = Real.sqrt 2 := by
  -- Define the given conditions
  have sin_135 : Real.sin (135 * π / 180) = 1 / Real.sqrt 2 := by sorry
  have cos_135 : Real.cos (135 * π / 180) = -(1 / Real.sqrt 2) := by sorry

  -- Prove the theorem
  sorry

end csc_135_deg_l935_93509


namespace shaded_area_is_30_l935_93525

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_leg_length_10 : leg_length = 10

/-- A partition of the triangle into 25 congruent smaller triangles -/
structure Partition (t : IsoscelesRightTriangle) where
  num_small_triangles : ℕ
  is_25_triangles : num_small_triangles = 25

/-- The shaded region covering 15 of the smaller triangles -/
structure ShadedRegion (p : Partition t) where
  num_shaded_triangles : ℕ
  is_15_triangles : num_shaded_triangles = 15

/-- The theorem stating that the area of the shaded region is 30 -/
theorem shaded_area_is_30 (t : IsoscelesRightTriangle) (p : Partition t) (s : ShadedRegion p) :
  (t.leg_length ^ 2 / 2) * (s.num_shaded_triangles / p.num_small_triangles) = 30 :=
sorry

end shaded_area_is_30_l935_93525


namespace age_sum_proof_l935_93568

theorem age_sum_proof (a b c : ℕ+) : 
  a * b * c = 72 → 
  a ≤ b ∧ a ≤ c → 
  a + b + c = 15 := by
sorry

end age_sum_proof_l935_93568


namespace dinosaur_model_price_reduction_l935_93593

/-- The percentage reduction in dinosaur model prices for a school purchase --/
theorem dinosaur_model_price_reduction :
  -- Original price per model
  ∀ (original_price : ℕ),
  -- Number of models for kindergarten
  ∀ (k : ℕ),
  -- Number of models for elementary
  ∀ (e : ℕ),
  -- Total number of models
  ∀ (total : ℕ),
  -- Total amount paid
  ∀ (total_paid : ℕ),
  -- Conditions
  original_price = 100 →
  k = 2 →
  e = 2 * k →
  total = k + e →
  total > 5 →
  total_paid = 570 →
  -- Conclusion
  (1 - total_paid / (total * original_price : ℚ)) * 100 = 5 := by
sorry

end dinosaur_model_price_reduction_l935_93593


namespace reading_time_difference_l935_93520

-- Define the reading speeds and book length
def xanthia_speed : ℝ := 150  -- pages per hour
def molly_speed : ℝ := 75     -- pages per hour
def book_length : ℝ := 300    -- pages

-- Define the time difference in minutes
def time_difference : ℝ := 120 -- minutes

-- Theorem statement
theorem reading_time_difference :
  (book_length / molly_speed - book_length / xanthia_speed) * 60 = time_difference := by
  sorry

end reading_time_difference_l935_93520


namespace shift_proof_l935_93599

def original_function (x : ℝ) : ℝ := -3 * x + 2

def vertical_shift : ℝ := 3

def shifted_function (x : ℝ) : ℝ := original_function x + vertical_shift

theorem shift_proof : 
  ∀ x : ℝ, shifted_function x = -3 * x + 5 := by
sorry

end shift_proof_l935_93599


namespace savings_ratio_l935_93584

def january_amount : ℕ := 19
def march_amount : ℕ := 8
def total_amount : ℕ := 46

def february_amount : ℕ := total_amount - january_amount - march_amount

theorem savings_ratio : 
  (january_amount : ℚ) / (february_amount : ℚ) = 1 := by sorry

end savings_ratio_l935_93584


namespace inequality_equivalence_l935_93581

theorem inequality_equivalence (x : ℝ) : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1) ↔ x > 1 := by
  sorry

end inequality_equivalence_l935_93581


namespace square_root_problem_l935_93545

theorem square_root_problem (x a : ℝ) : 
  ((2 * a + 1) ^ 2 = x ∧ (4 - a) ^ 2 = x) → x = 81 := by
  sorry

end square_root_problem_l935_93545


namespace age_difference_l935_93555

/-- Given three people x, y, and z, where z is 10 decades younger than x,
    prove that the combined age of x and y is 100 years greater than
    the combined age of y and z. -/
theorem age_difference (x y z : ℕ) (h : z = x - 100) :
  (x + y) - (y + z) = 100 := by
  sorry

end age_difference_l935_93555


namespace binary_38_correct_l935_93594

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 38 -/
def binary_38 : List Bool := [false, true, true, false, false, true]

/-- Theorem stating that the binary representation of 38 is correct -/
theorem binary_38_correct : binary_to_decimal binary_38 = 38 := by
  sorry

#eval binary_to_decimal binary_38

end binary_38_correct_l935_93594


namespace initial_investment_rate_l935_93583

-- Define the initial investment
def initial_investment : ℝ := 1400

-- Define the additional investment
def additional_investment : ℝ := 700

-- Define the interest rate of the additional investment
def additional_rate : ℝ := 0.08

-- Define the total investment
def total_investment : ℝ := initial_investment + additional_investment

-- Define the desired total annual income rate
def total_income_rate : ℝ := 0.06

-- Define the function that calculates the total annual income
def total_annual_income (r : ℝ) : ℝ := 
  initial_investment * r + additional_investment * additional_rate

-- Theorem statement
theorem initial_investment_rate : 
  ∃ r : ℝ, total_annual_income r = total_income_rate * total_investment ∧ r = 0.05 := by
  sorry

end initial_investment_rate_l935_93583


namespace min_workers_for_painting_job_l935_93598

/-- Represents the painting job scenario -/
structure PaintingJob where
  totalDays : ℕ
  workedDays : ℕ
  initialWorkers : ℕ
  completedFraction : ℚ
  
/-- Calculates the minimum number of workers needed to complete the job on time -/
def minWorkersNeeded (job : PaintingJob) : ℕ :=
  sorry

/-- The theorem stating the minimum number of workers needed for the specific scenario -/
theorem min_workers_for_painting_job :
  let job := PaintingJob.mk 40 8 10 (2/5)
  minWorkersNeeded job = 4 := by
  sorry

end min_workers_for_painting_job_l935_93598


namespace geometric_sequence_properties_l935_93561

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 ∧ (a 1 + a 2 + a 3 + a 4 + a 5 = 31/2) :=
by sorry

end geometric_sequence_properties_l935_93561


namespace rectangular_prism_dimensions_sum_l935_93518

theorem rectangular_prism_dimensions_sum (A B C : ℝ) 
  (h1 : A * B = 30)
  (h2 : A * C = 40)
  (h3 : B * C = 60) :
  A + B + C = 9 * Real.sqrt 5 := by
sorry

end rectangular_prism_dimensions_sum_l935_93518


namespace volume_of_specific_tetrahedron_l935_93551

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- The angle between faces ABC and BCD in radians -/
  angle : ℝ
  /-- The area of face ABC -/
  area_ABC : ℝ
  /-- The area of face BCD -/
  area_BCD : ℝ
  /-- The length of edge BC -/
  length_BC : ℝ

/-- Calculates the volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 320 -/
theorem volume_of_specific_tetrahedron :
  let t : Tetrahedron := {
    angle := 30 * π / 180,  -- 30 degrees in radians
    area_ABC := 120,
    area_BCD := 80,
    length_BC := 10
  }
  volume t = 320 := by sorry

end volume_of_specific_tetrahedron_l935_93551


namespace day_crew_load_fraction_l935_93521

/-- Fraction of boxes loaded by day crew given night crew conditions -/
theorem day_crew_load_fraction (D W : ℚ) : 
  D > 0 → W > 0 →
  (D * W) / ((D * W) + ((3/4 * D) * (4/7 * W))) = 7/10 := by
  sorry

end day_crew_load_fraction_l935_93521


namespace f_g_f_3_equals_108_l935_93586

def f (x : ℝ) : ℝ := 2 * x + 4

def g (x : ℝ) : ℝ := 5 * x + 2

theorem f_g_f_3_equals_108 : f (g (f 3)) = 108 := by
  sorry

end f_g_f_3_equals_108_l935_93586


namespace negation_existence_inequality_l935_93530

theorem negation_existence_inequality :
  (¬ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end negation_existence_inequality_l935_93530


namespace framed_painting_ratio_approaches_one_l935_93556

/-- Represents the dimensions of a framed painting -/
structure FramedPainting where
  painting_width : ℝ
  painting_height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting -/
def framedDimensions (fp : FramedPainting) : ℝ × ℝ :=
  (fp.painting_width + 2 * fp.side_frame_width,
   fp.painting_height + 6 * fp.side_frame_width)

/-- Calculates the area of the framed painting -/
def framedArea (fp : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions fp
  w * h

/-- Theorem: The ratio of dimensions of the framed painting approaches 1:1 -/
theorem framed_painting_ratio_approaches_one (ε : ℝ) (hε : ε > 0) :
  ∃ (fp : FramedPainting),
    fp.painting_width = 30 ∧
    fp.painting_height = 20 ∧
    framedArea fp = fp.painting_width * fp.painting_height ∧
    let (w, h) := framedDimensions fp
    |w / h - 1| < ε :=
by sorry

end framed_painting_ratio_approaches_one_l935_93556


namespace bedroom_curtain_width_l935_93577

theorem bedroom_curtain_width :
  let initial_width : ℝ := 16
  let initial_height : ℝ := 12
  let living_room_width : ℝ := 4
  let living_room_height : ℝ := 6
  let bedroom_height : ℝ := 4
  let remaining_area : ℝ := 160
  let total_area := initial_width * initial_height
  let living_room_area := living_room_width * living_room_height
  let bedroom_width := (total_area - living_room_area - remaining_area) / bedroom_height
  bedroom_width = 2 := by sorry

end bedroom_curtain_width_l935_93577


namespace jr_high_selection_theorem_l935_93532

/-- Represents the structure of a school with different grade levels and classes --/
structure School where
  elem_grades : Nat
  elem_classes_per_grade : Nat
  jr_high_grades : Nat
  jr_high_classes_per_grade : Nat
  high_grades : Nat
  high_classes_per_grade : Nat

/-- Calculates the total number of classes in the school --/
def total_classes (s : School) : Nat :=
  s.elem_grades * s.elem_classes_per_grade +
  s.jr_high_grades * s.jr_high_classes_per_grade +
  s.high_grades * s.high_classes_per_grade

/-- Calculates the number of classes to be selected from each grade in junior high --/
def jr_high_classes_selected (s : School) (total_selected : Nat) : Nat :=
  (total_selected * s.jr_high_classes_per_grade) / (total_classes s)

theorem jr_high_selection_theorem (s : School) (total_selected : Nat) :
  s.elem_grades = 6 →
  s.elem_classes_per_grade = 6 →
  s.jr_high_grades = 3 →
  s.jr_high_classes_per_grade = 8 →
  s.high_grades = 3 →
  s.high_classes_per_grade = 12 →
  total_selected = 36 →
  jr_high_classes_selected s total_selected = 2 := by
  sorry

end jr_high_selection_theorem_l935_93532


namespace tan_alpha_two_implies_expression_equals_one_l935_93528

theorem tan_alpha_two_implies_expression_equals_one (α : Real) 
  (h : Real.tan α = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end tan_alpha_two_implies_expression_equals_one_l935_93528


namespace expression_evaluation_l935_93510

theorem expression_evaluation : 
  let x : ℚ := -1/2
  let expr := (x - 2) / ((x^2 + 4*x + 4) * ((x^2 + x - 6) / (x + 2) - x + 2))
  expr = 2/3 := by sorry

end expression_evaluation_l935_93510


namespace min_correct_answers_for_target_score_l935_93517

/-- A math competition with specific scoring rules -/
structure MathCompetition where
  total_questions : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int

/-- A participant in the math competition -/
structure Participant where
  answered_questions : Nat
  unanswered_questions : Nat

/-- Calculate the score based on the number of correct answers -/
def calculate_score (comp : MathCompetition) (part : Participant) (correct_answers : Nat) : Int :=
  correct_answers * comp.correct_points +
  (part.answered_questions - correct_answers) * comp.incorrect_points +
  part.unanswered_questions * comp.unanswered_points

/-- The main theorem to prove -/
theorem min_correct_answers_for_target_score 
  (comp : MathCompetition)
  (part : Participant)
  (target_score : Int) : Nat :=
  have h1 : comp.total_questions = 30 := by sorry
  have h2 : comp.correct_points = 8 := by sorry
  have h3 : comp.incorrect_points = -2 := by sorry
  have h4 : comp.unanswered_points = 2 := by sorry
  have h5 : part.answered_questions = 25 := by sorry
  have h6 : part.unanswered_questions = 5 := by sorry
  have h7 : target_score = 160 := by sorry

  20

#check min_correct_answers_for_target_score


end min_correct_answers_for_target_score_l935_93517


namespace milk_mixture_theorem_l935_93590

/-- Given two types of milk with different butterfat percentages, prove that mixing them in specific quantities results in a desired butterfat percentage. -/
theorem milk_mixture_theorem (x : ℝ) :
  -- Define the butterfat percentages
  let high_fat_percent : ℝ := 0.45
  let low_fat_percent : ℝ := 0.10
  let target_percent : ℝ := 0.20

  -- Define the quantities
  let low_fat_quantity : ℝ := 20
  
  -- Condition: The mixture's butterfat content equals the target percentage
  high_fat_percent * x + low_fat_percent * low_fat_quantity = target_percent * (x + low_fat_quantity) →
  -- Conclusion: The quantity of high-fat milk needed is 8 gallons
  x = 8 := by
  sorry

#check milk_mixture_theorem

end milk_mixture_theorem_l935_93590


namespace sum_of_digits_of_product_80_sevens_80_threes_l935_93524

/-- A number consisting of n repeated digits d -/
def repeated_digit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_product_80_sevens_80_threes : 
  sum_of_digits (repeated_digit 80 7 * repeated_digit 80 3) = 240 := by sorry

end sum_of_digits_of_product_80_sevens_80_threes_l935_93524


namespace second_person_average_pages_per_day_l935_93588

theorem second_person_average_pages_per_day 
  (summer_days : ℕ) 
  (books_read : ℕ) 
  (avg_pages_per_book : ℕ) 
  (second_person_percentage : ℚ) 
  (h1 : summer_days = 80)
  (h2 : books_read = 60)
  (h3 : avg_pages_per_book = 320)
  (h4 : second_person_percentage = 3/4) : 
  (books_read * avg_pages_per_book * second_person_percentage) / summer_days = 180 := by
  sorry

end second_person_average_pages_per_day_l935_93588


namespace inequality_proof_l935_93553

theorem inequality_proof (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > a*x ∧ a*x > a^2 := by
  sorry

end inequality_proof_l935_93553


namespace circle_properties_l935_93544

/-- A circle with diameter endpoints (2, -3) and (8, 9) has center (5, 3) and radius 3√5 -/
theorem circle_properties :
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (8, 9)
  let C : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let r : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  C = (5, 3) ∧ r = 3 * Real.sqrt 5 := by sorry

end circle_properties_l935_93544
