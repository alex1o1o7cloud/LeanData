import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_solution_existence_l1968_196850

theorem quadratic_equation_solution_existence 
  (a b c : ℝ) 
  (h_a : a ≠ 0)
  (h_1 : a * (3.24 : ℝ)^2 + b * (3.24 : ℝ) + c = -0.02)
  (h_2 : a * (3.25 : ℝ)^2 + b * (3.25 : ℝ) + c = 0.03) :
  ∃ x : ℝ, x > 3.24 ∧ x < 3.25 ∧ a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_existence_l1968_196850


namespace NUMINAMATH_CALUDE_root_sum_l1968_196867

-- Define the complex number 2i-3
def z : ℂ := -3 + 2*Complex.I

-- Define the quadratic equation
def quadratic (p q : ℝ) (x : ℂ) : ℂ := 2*x^2 + p*x + q

-- State the theorem
theorem root_sum (p q : ℝ) : 
  quadratic p q z = 0 → p + q = 38 := by sorry

end NUMINAMATH_CALUDE_root_sum_l1968_196867


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1968_196837

/-- The angle between asymptotes of a hyperbola -/
theorem hyperbola_asymptote_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := (2 * Real.sqrt 3) / 3
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  hyperbola x y ∧ e = Real.sqrt (1 + b^2 / a^2) →
  ∃ θ : ℝ, θ = π / 3 ∧ θ = 2 * Real.arctan (b / a) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1968_196837


namespace NUMINAMATH_CALUDE_number_difference_l1968_196810

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : |x - y| = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1968_196810


namespace NUMINAMATH_CALUDE_area_ratio_is_three_fourths_l1968_196818

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- Points on the sides of the octagon -/
structure OctagonPoints (oct : RegularOctagon) where
  I : ℝ × ℝ
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  on_sides : sorry
  equally_spaced : sorry

/-- The ratio of areas of the inner octagon to the outer octagon -/
def area_ratio (oct : RegularOctagon) (pts : OctagonPoints oct) : ℝ := sorry

/-- The main theorem -/
theorem area_ratio_is_three_fourths (oct : RegularOctagon) (pts : OctagonPoints oct) :
  area_ratio oct pts = 3/4 := by sorry

end NUMINAMATH_CALUDE_area_ratio_is_three_fourths_l1968_196818


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1968_196861

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1968_196861


namespace NUMINAMATH_CALUDE_hexagon_area_in_circle_l1968_196866

/-- The area of a regular hexagon inscribed in a circle -/
theorem hexagon_area_in_circle (circle_area : ℝ) (hexagon_area : ℝ) : 
  circle_area = 400 * Real.pi →
  hexagon_area = 600 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_in_circle_l1968_196866


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1968_196816

theorem ellipse_major_axis_length :
  let ellipse := {(x, y) : ℝ × ℝ | x^2 / 9 + y^2 / 4 = 1}
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ b > 0 ∧
    ellipse = {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1} ∧
    2 * a = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1968_196816


namespace NUMINAMATH_CALUDE_sum_three_consecutive_integers_divisible_by_three_l1968_196852

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : a > 1) :
  ∃ k : ℤ, (a - 1 : ℤ) + a + (a + 1) = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_sum_three_consecutive_integers_divisible_by_three_l1968_196852


namespace NUMINAMATH_CALUDE_equation_and_inequality_solution_l1968_196873

theorem equation_and_inequality_solution :
  (∃ x : ℝ, 3 * (x - 2) - (1 - 2 * x) = 3 ∧ x = 2) ∧
  (∀ x : ℝ, 2 * x - 1 < 4 * x + 3 ↔ x > -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_and_inequality_solution_l1968_196873


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1968_196820

theorem triangle_sine_sum_inequality (α β γ : Real) 
  (h : α + β + γ = Real.pi) : 
  Real.sin α + Real.sin β + Real.sin γ ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1968_196820


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l1968_196875

/-- The line passing through (-1, 0) and perpendicular to x+y=0 has equation x-y+1=0 -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  (x + y = 0 → (x + 1 = 0 ∧ y = 0) → x - y + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l1968_196875


namespace NUMINAMATH_CALUDE_simplest_common_denominator_example_l1968_196834

/-- The simplest common denominator of two fractions -/
def simplestCommonDenominator (f1 f2 : ℚ) : ℤ :=
  sorry

/-- Theorem: The simplest common denominator of 1/(m^2-9) and 1/(2m+6) is 2(m+3)(m-3) -/
theorem simplest_common_denominator_example (m : ℚ) :
  simplestCommonDenominator (1 / (m^2 - 9)) (1 / (2*m + 6)) = 2 * (m + 3) * (m - 3) :=
by sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_example_l1968_196834


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1968_196893

theorem simplify_fraction_product : (144 : ℚ) / 18 * 9 / 108 * 6 / 4 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1968_196893


namespace NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l1968_196894

/-- Given two curves C₁ and C₂, prove that if their tangent lines are perpendicular at x = 1, 
    then the parameter a of C₁ must equal -1 / (3e) -/
theorem perpendicular_tangents_imply_a_value (a : ℝ) :
  let C₁ : ℝ → ℝ := λ x => a * x^3 - x^2 + 2 * x
  let C₂ : ℝ → ℝ := λ x => Real.exp x
  let C₁' : ℝ → ℝ := λ x => 3 * a * x^2 - 2 * x + 2
  let C₂' : ℝ → ℝ := λ x => Real.exp x
  (C₁' 1 * C₂' 1 = -1) → a = -1 / (3 * Real.exp 1) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l1968_196894


namespace NUMINAMATH_CALUDE_unique_player_count_l1968_196824

/-- Given a total number of socks and the fact that each player contributes two socks,
    proves that there is only one possible number of players. -/
theorem unique_player_count (total_socks : ℕ) (h : total_socks = 22) :
  ∃! n : ℕ, n * 2 = total_socks := by sorry

end NUMINAMATH_CALUDE_unique_player_count_l1968_196824


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l1968_196848

/-- Represents the number of ways to make exactly n substitutions -/
def b (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => 12 * (12 - n) * b n

/-- The total number of possible substitution ways -/
def total_ways : Nat :=
  b 0 + b 1 + b 2 + b 3 + b 4 + b 5

theorem soccer_substitutions_remainder :
  total_ways % 100 = 93 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l1968_196848


namespace NUMINAMATH_CALUDE_last_group_count_l1968_196858

theorem last_group_count (total : Nat) (total_avg : ℚ) (first_group : Nat) (first_avg : ℚ) (middle : ℚ) (last_avg : ℚ) 
  (h_total : total = 13)
  (h_total_avg : total_avg = 60)
  (h_first_group : first_group = 6)
  (h_first_avg : first_avg = 57)
  (h_middle : middle = 50)
  (h_last_avg : last_avg = 61) :
  ∃ (last_group : Nat), last_group = total - first_group - 1 ∧ last_group = 6 := by
  sorry

#check last_group_count

end NUMINAMATH_CALUDE_last_group_count_l1968_196858


namespace NUMINAMATH_CALUDE_cubic_root_h_value_l1968_196868

theorem cubic_root_h_value (h : ℝ) : (3 : ℝ)^3 + h * 3 + 14 = 0 → h = -41/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_h_value_l1968_196868


namespace NUMINAMATH_CALUDE_quadratic_sum_l1968_196827

/-- The quadratic expression 20x^2 + 240x + 3200 can be written as a(x+b)^2+c -/
def quadratic (x : ℝ) : ℝ := 20*x^2 + 240*x + 3200

/-- The completed square form of the quadratic -/
def completed_square (x a b c : ℝ) : ℝ := a*(x+b)^2 + c

theorem quadratic_sum : 
  ∃ (a b c : ℝ), (∀ x, quadratic x = completed_square x a b c) ∧ (a + b + c = 2506) := by
sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1968_196827


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1968_196865

theorem exponent_multiplication (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1968_196865


namespace NUMINAMATH_CALUDE_tangent_slope_point_coordinates_l1968_196887

theorem tangent_slope_point_coordinates :
  ∀ (x y : ℝ), 
    y = 1 / x →  -- The curve equation
    (-1 / x^2) = -4 →  -- The slope of the tangent line
    ((x = 1/2 ∧ y = 2) ∨ (x = -1/2 ∧ y = -2)) := by sorry

end NUMINAMATH_CALUDE_tangent_slope_point_coordinates_l1968_196887


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l1968_196811

universe u

def U : Set Nat := {1, 2, 3, 4}
def P : Set Nat := {2, 3, 4}
def Q : Set Nat := {1, 2}

theorem complement_P_intersect_Q :
  (Set.univ \ P) ∩ Q = {1} := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l1968_196811


namespace NUMINAMATH_CALUDE_square_plus_fourth_power_equality_l1968_196862

theorem square_plus_fourth_power_equality (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : n > 3) 
  (h4 : m^2 + n^4 = 2*(m - 6)^2 + 2*(n + 1)^2) : 
  m^2 + n^4 = 1994 := by
sorry

end NUMINAMATH_CALUDE_square_plus_fourth_power_equality_l1968_196862


namespace NUMINAMATH_CALUDE_student_rank_theorem_l1968_196845

/-- Given a line of students, this function calculates a student's rank from the right
    based on their rank from the left and the total number of students. -/
def rankFromRight (totalStudents : ℕ) (rankFromLeft : ℕ) : ℕ :=
  totalStudents - rankFromLeft + 1

/-- Theorem stating that for a line of 10 students, 
    a student ranked 5th from the left is ranked 6th from the right. -/
theorem student_rank_theorem :
  rankFromRight 10 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_theorem_l1968_196845


namespace NUMINAMATH_CALUDE_cubic_equation_value_l1968_196801

theorem cubic_equation_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2*a^2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l1968_196801


namespace NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1968_196859

-- Define the repeating decimal 0.4555...
def repeating_decimal : ℚ := 0.4555555555555555

-- Theorem statement
theorem repeating_decimal_as_fraction : repeating_decimal = 41 / 90 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_as_fraction_l1968_196859


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1968_196870

theorem max_value_of_expression (x y : ℝ) (h : x^2 + y^2 ≤ 1) :
  |x^2 + 2*x*y - y^2| ≤ Real.sqrt 2 ∧ ∃ x y, x^2 + y^2 ≤ 1 ∧ |x^2 + 2*x*y - y^2| = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1968_196870


namespace NUMINAMATH_CALUDE_amy_homework_time_l1968_196807

/-- Calculates the total time needed to complete homework with breaks -/
def total_homework_time (math_problems : ℕ) (spelling_problems : ℕ) 
  (math_rate : ℕ) (spelling_rate : ℕ) (break_duration : ℚ) : ℚ :=
  let work_hours : ℚ := (math_problems / math_rate + spelling_problems / spelling_rate : ℚ)
  let break_hours : ℚ := (work_hours.floor - 1) * break_duration
  work_hours + break_hours

/-- Theorem: Amy will take 11 hours to finish her homework -/
theorem amy_homework_time : 
  total_homework_time 18 6 3 2 (1/4) = 11 := by sorry

end NUMINAMATH_CALUDE_amy_homework_time_l1968_196807


namespace NUMINAMATH_CALUDE_gumball_ratio_l1968_196864

/-- Gumball machine problem -/
theorem gumball_ratio : 
  ∀ (red green blue : ℕ),
  red = 16 →
  green = 4 * blue →
  red + green + blue = 56 →
  blue * 2 = red :=
by
  sorry

end NUMINAMATH_CALUDE_gumball_ratio_l1968_196864


namespace NUMINAMATH_CALUDE_max_k_value_l1968_196817

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l1968_196817


namespace NUMINAMATH_CALUDE_tim_gave_six_kittens_to_sara_l1968_196828

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara (initial_kittens : ℕ) (kittens_to_jessica : ℕ) (remaining_kittens : ℕ) : ℕ :=
  initial_kittens - kittens_to_jessica - remaining_kittens

/-- Proof that Tim gave 6 kittens to Sara -/
theorem tim_gave_six_kittens_to_sara :
  kittens_to_sara 18 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_tim_gave_six_kittens_to_sara_l1968_196828


namespace NUMINAMATH_CALUDE_charity_amount_l1968_196879

def small_price : ℚ := 2
def medium_price : ℚ := 3
def large_price : ℚ := 5

def small_count : ℕ := 150
def medium_count : ℕ := 221
def large_count : ℕ := 185

def total_raised : ℚ := small_price * small_count + medium_price * medium_count + large_price * large_count

theorem charity_amount : total_raised = 1888 := by
  sorry

end NUMINAMATH_CALUDE_charity_amount_l1968_196879


namespace NUMINAMATH_CALUDE_expression_factorization_l1968_196874

/-- 
Given a, b, and c, prove that the expression 
a^4 (b^2 - c^2) + b^4 (c^2 - a^2) + c^4 (a^2 - b^2) 
can be factorized into the form (a - b)(b - c)(c - a) q(a, b, c),
where q(a, b, c) = a^3 b^2 + a^2 b^3 + b^3 c^2 + b^2 c^3 + c^3 a^2 + c^2 a^3
-/
theorem expression_factorization (a b c : ℝ) : 
  a^4 * (b^2 - c^2) + b^4 * (c^2 - a^2) + c^4 * (a^2 - b^2) = 
  (a - b) * (b - c) * (c - a) * (a^3 * b^2 + a^2 * b^3 + b^3 * c^2 + b^2 * c^3 + c^3 * a^2 + c^2 * a^3) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1968_196874


namespace NUMINAMATH_CALUDE_simplify_expressions_l1968_196882

theorem simplify_expressions :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (Real.sqrt (1 / 3) + Real.sqrt 27 * Real.sqrt 9 = 28 * Real.sqrt 3 / 3) ∧
    (Real.sqrt 32 - 3 * Real.sqrt (1 / 2) + Real.sqrt (1 / 8) = 11 * Real.sqrt 2 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1968_196882


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1968_196823

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + 2*y + 3*z = 1) :
  1/(x+2*y) + 4/(2*y+3*z) + 9/(3*z+x) ≥ 18 := by
sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ 
  1/(x+2*y) + 4/(2*y+3*z) + 9/(3*z+x) < 18 + ε := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1968_196823


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1968_196831

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1968_196831


namespace NUMINAMATH_CALUDE_min_distance_squared_l1968_196878

theorem min_distance_squared (a b c d : ℝ) 
  (h1 : Real.log (b + 1) + a - 3 * b = 0)
  (h2 : 2 * d - c + Real.sqrt 5 = 0) :
  ∃ (x y : ℝ), ∀ (a' b' c' d' : ℝ),
    Real.log (b' + 1) + a' - 3 * b' = 0 →
    2 * d' - c' + Real.sqrt 5 = 0 →
    (a' - c')^2 + (b' - d')^2 ≥ 1 ∧
    (x - y)^2 + (a - c)^2 + (b - d)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l1968_196878


namespace NUMINAMATH_CALUDE_no_real_solution_condition_l1968_196802

theorem no_real_solution_condition (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a^x ≠ x) ↔ a > Real.exp (1 / Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_condition_l1968_196802


namespace NUMINAMATH_CALUDE_x_value_l1968_196844

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 15 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1968_196844


namespace NUMINAMATH_CALUDE_consumption_decrease_l1968_196872

/-- Represents a country with its production capabilities -/
structure Country where
  zucchini : ℕ
  cauliflower : ℕ

/-- Calculates the total consumption of each crop under free trade -/
def freeTradeTotalConsumption (a b : Country) : ℕ := by
  sorry

/-- Calculates the total consumption of each crop under autarky -/
def autarkyTotalConsumption (a b : Country) : ℕ := by
  sorry

/-- Theorem stating that consumption decreases by 4 tons when countries merge and trade is banned -/
theorem consumption_decrease (a b : Country) 
  (h1 : a.zucchini = 20 ∧ a.cauliflower = 16)
  (h2 : b.zucchini = 36 ∧ b.cauliflower = 24) :
  freeTradeTotalConsumption a b - autarkyTotalConsumption a b = 4 := by
  sorry

end NUMINAMATH_CALUDE_consumption_decrease_l1968_196872


namespace NUMINAMATH_CALUDE_expression_evaluation_l1968_196805

/-- Evaluates the expression (3x^3 - 7x^2 + 4x - 9) / (2x - 0.5) for x = 100 -/
theorem expression_evaluation :
  let x : ℝ := 100
  let numerator := 3 * x^3 - 7 * x^2 + 4 * x - 9
  let denominator := 2 * x - 0.5
  abs ((numerator / denominator) - 14684.73534) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1968_196805


namespace NUMINAMATH_CALUDE_airplane_luggage_problem_l1968_196846

/-- Calculates the number of bags per person given the problem conditions -/
def bagsPerPerson (numPeople : ℕ) (bagWeight : ℕ) (totalCapacity : ℕ) (additionalBags : ℕ) : ℕ :=
  let totalBags := totalCapacity / bagWeight
  let currentBags := totalBags - additionalBags
  currentBags / numPeople

/-- Theorem stating that under the given conditions, each person has 5 bags -/
theorem airplane_luggage_problem :
  bagsPerPerson 6 50 6000 90 = 5 := by
  sorry

end NUMINAMATH_CALUDE_airplane_luggage_problem_l1968_196846


namespace NUMINAMATH_CALUDE_median_invariant_after_remove_min_max_l1968_196833

/-- A function that returns the median of a list of real numbers -/
def median (l : List ℝ) : ℝ := sorry

/-- A function that removes the minimum and maximum elements from a list -/
def removeMinMax (l : List ℝ) : List ℝ := sorry

theorem median_invariant_after_remove_min_max (data : List ℝ) :
  data.length > 2 →
  data.Nodup →
  median data = median (removeMinMax data) :=
sorry

end NUMINAMATH_CALUDE_median_invariant_after_remove_min_max_l1968_196833


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1968_196826

theorem sum_of_x_and_y (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y + 10) :
  x + y = 14 ∨ x + y = -2 := by sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1968_196826


namespace NUMINAMATH_CALUDE_jake_bitcoin_theorem_l1968_196832

def jake_bitcoin_problem (initial_fortune : ℕ) (first_donation : ℕ) (final_amount : ℕ) : Prop :=
  let after_first_donation := initial_fortune - first_donation
  let after_giving_to_brother := after_first_donation / 2
  let after_tripling := after_giving_to_brother * 3
  let final_donation := after_tripling - final_amount
  final_donation = 10

theorem jake_bitcoin_theorem :
  jake_bitcoin_problem 80 20 80 := by sorry

end NUMINAMATH_CALUDE_jake_bitcoin_theorem_l1968_196832


namespace NUMINAMATH_CALUDE_max_value_abc_l1968_196857

theorem max_value_abc (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  10 * a + 3 * b + 15 * c ≤ Real.sqrt (337 / 36) :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l1968_196857


namespace NUMINAMATH_CALUDE_gilbert_herb_plants_l1968_196853

/-- The number of herb plants Gilbert had at the end of spring -/
def herb_plants_at_end_of_spring : ℕ :=
  let initial_basil : ℕ := 3
  let initial_parsley : ℕ := 1
  let initial_mint : ℕ := 2
  let new_basil : ℕ := 1
  let eaten_mint : ℕ := 2
  (initial_basil + initial_parsley + initial_mint + new_basil) - eaten_mint

theorem gilbert_herb_plants : herb_plants_at_end_of_spring = 5 := by
  sorry

end NUMINAMATH_CALUDE_gilbert_herb_plants_l1968_196853


namespace NUMINAMATH_CALUDE_total_travel_time_travel_time_calculation_l1968_196804

/-- Calculates the total travel time between two towns given specific conditions -/
theorem total_travel_time (total_distance : ℝ) (initial_fraction : ℝ) (lunch_time : ℝ) 
  (second_fraction : ℝ) (pit_stop_time : ℝ) (speed_increase : ℝ) : ℝ :=
  let initial_distance := initial_fraction * total_distance
  let initial_speed := initial_distance
  let remaining_distance := total_distance - initial_distance
  let second_distance := second_fraction * remaining_distance
  let final_distance := remaining_distance - second_distance
  let final_speed := initial_speed + speed_increase
  initial_fraction + lunch_time + (second_distance / initial_speed) + 
  pit_stop_time + (final_distance / final_speed)

/-- The total travel time between the two towns is 5.25 hours -/
theorem travel_time_calculation : 
  total_travel_time 200 (1/4) 1 (1/2) (1/2) 10 = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_total_travel_time_travel_time_calculation_l1968_196804


namespace NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l1968_196895

theorem positive_integer_sum_greater_than_product (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  m + n > m * n ↔ m = 1 ∨ n = 1 := by
sorry

end NUMINAMATH_CALUDE_positive_integer_sum_greater_than_product_l1968_196895


namespace NUMINAMATH_CALUDE_max_value_fraction_l1968_196842

theorem max_value_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hbc : b + c ≤ a) :
  ∃ (max : ℝ), max = 1/8 ∧ ∀ x, x = b * c / (a^2 + 2*a*b + b^2) → x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1968_196842


namespace NUMINAMATH_CALUDE_adjacent_pair_properties_l1968_196836

/-- Definition of "adjacent number pairs" -/
def adjacent_pair (m n : ℚ) : Prop :=
  m / 2 + n / 5 = (m + n) / 7

theorem adjacent_pair_properties :
  ∃ (m n : ℚ),
    /- Part 1 -/
    (adjacent_pair 2 n → n = -25 / 2) ∧
    /- Part 2① -/
    (adjacent_pair m n → m = -4 * n / 25) ∧
    /- Part 2② -/
    (adjacent_pair m n ∧ 25 * m + n = 6 → m = 8 / 25 ∧ n = -2) := by
  sorry

end NUMINAMATH_CALUDE_adjacent_pair_properties_l1968_196836


namespace NUMINAMATH_CALUDE_expression_value_l1968_196854

theorem expression_value (a b : ℤ) (ha : a = 3) (hb : b = -2) :
  -a^2 - b^3 + a*b = -7 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1968_196854


namespace NUMINAMATH_CALUDE_original_men_count_prove_original_men_count_l1968_196885

/-- Represents the amount of work to be done -/
def work : ℝ := 1

/-- The number of days taken by the original group to complete the work -/
def original_days : ℕ := 60

/-- The number of days taken by the augmented group to complete the work -/
def augmented_days : ℕ := 50

/-- The number of additional men in the augmented group -/
def additional_men : ℕ := 8

/-- Theorem stating that the original number of men is 48 -/
theorem original_men_count : ℕ :=
  48

/-- Proof that the original number of men is 48 -/
theorem prove_original_men_count : 
  ∃ (m : ℕ), 
    (m * (work / original_days) = (m + additional_men) * (work / augmented_days)) ∧ 
    (m = original_men_count) := by
  sorry

end NUMINAMATH_CALUDE_original_men_count_prove_original_men_count_l1968_196885


namespace NUMINAMATH_CALUDE_marys_max_earnings_l1968_196821

/-- Calculates the maximum weekly earnings for a worker with the given parameters. -/
def maxWeeklyEarnings (maxHours regularHours : ℕ) (regularRate : ℚ) (overtimeRateIncrease : ℚ) : ℚ :=
  let regularEarnings := regularRate * regularHours
  let overtimeRate := regularRate * (1 + overtimeRateIncrease)
  let overtimeHours := maxHours - regularHours
  let overtimeEarnings := overtimeRate * overtimeHours
  regularEarnings + overtimeEarnings

/-- Theorem stating that Mary's maximum weekly earnings are $460 -/
theorem marys_max_earnings :
  maxWeeklyEarnings 50 20 8 (1/4) = 460 := by
  sorry

#eval maxWeeklyEarnings 50 20 8 (1/4)

end NUMINAMATH_CALUDE_marys_max_earnings_l1968_196821


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l1968_196860

/-- Given a point P with coordinates (x, -4), if the distance from the x-axis to P
    is half the distance from the y-axis to P, then the distance from the y-axis to P is 8. -/
theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -4)
  let dist_to_x_axis : ℝ := |P.2|
  let dist_to_y_axis : ℝ := |P.1|
  dist_to_x_axis = (1/2 : ℝ) * dist_to_y_axis →
  dist_to_y_axis = 8 := by
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l1968_196860


namespace NUMINAMATH_CALUDE_cone_volume_l1968_196806

/-- The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters. -/
theorem cone_volume (π : ℝ) (h : π > 0) : 
  let slant_height : ℝ := 15
  let height : ℝ := 9
  let radius : ℝ := Real.sqrt (slant_height^2 - height^2)
  let volume : ℝ := (1/3) * π * radius^2 * height
  volume = 432 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1968_196806


namespace NUMINAMATH_CALUDE_royalties_sales_ratio_decrease_l1968_196851

/-- Calculate the percentage decrease in the ratio of royalties to sales --/
theorem royalties_sales_ratio_decrease (first_royalties second_royalties : ℝ)
  (first_sales second_sales : ℝ) :
  first_royalties = 6 →
  first_sales = 20 →
  second_royalties = 9 →
  second_sales = 108 →
  let first_ratio := first_royalties / first_sales
  let second_ratio := second_royalties / second_sales
  let percentage_decrease := (first_ratio - second_ratio) / first_ratio * 100
  abs (percentage_decrease - 72.23) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_royalties_sales_ratio_decrease_l1968_196851


namespace NUMINAMATH_CALUDE_certain_number_is_three_l1968_196863

theorem certain_number_is_three :
  ∀ certain_number : ℕ,
  (2^14 : ℕ) - (2^12 : ℕ) = certain_number * (2^12 : ℕ) →
  certain_number = 3 := by
sorry

end NUMINAMATH_CALUDE_certain_number_is_three_l1968_196863


namespace NUMINAMATH_CALUDE_zuzkas_number_l1968_196843

theorem zuzkas_number : ∃! n : ℕ, 
  10000 ≤ n ∧ n < 100000 ∧ 
  10 * n + 1 = 3 * (100000 + n) := by
sorry

end NUMINAMATH_CALUDE_zuzkas_number_l1968_196843


namespace NUMINAMATH_CALUDE_welders_problem_l1968_196877

/-- The number of days needed to complete the order with all welders -/
def total_days : ℝ := 3

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 12

/-- The number of additional days needed by remaining welders to complete the order -/
def remaining_days : ℝ := 3.0000000000000004

/-- The initial number of welders -/
def initial_welders : ℕ := 36

theorem welders_problem :
  (initial_welders - leaving_welders : ℝ) / initial_welders * remaining_days = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_welders_problem_l1968_196877


namespace NUMINAMATH_CALUDE_average_trees_planted_l1968_196889

def tree_data : List ℕ := [10, 8, 9, 9]

theorem average_trees_planted : 
  (List.sum tree_data) / (List.length tree_data : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_planted_l1968_196889


namespace NUMINAMATH_CALUDE_candle_burn_time_l1968_196822

/-- Given that a candle lasts 8 nights and burning it for 2 hours a night uses 6 candles over 24 nights,
    prove that Carmen burns the candle for 1 hour every night in the first scenario. -/
theorem candle_burn_time (candle_duration : ℕ) (nights_per_candle : ℕ) (burn_time_second_scenario : ℕ) 
  (candles_used : ℕ) (total_nights : ℕ) :
  candle_duration = 8 ∧ 
  nights_per_candle = 8 ∧
  burn_time_second_scenario = 2 ∧
  candles_used = 6 ∧
  total_nights = 24 →
  ∃ (burn_time_first_scenario : ℕ), burn_time_first_scenario = 1 :=
by sorry

end NUMINAMATH_CALUDE_candle_burn_time_l1968_196822


namespace NUMINAMATH_CALUDE_workshop_workers_count_l1968_196849

theorem workshop_workers_count :
  let total_average : ℝ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℝ := 14000
  let non_technician_average : ℝ := 6000
  ∃ (total_workers : ℕ) (non_technician_workers : ℕ),
    total_workers = technician_count + non_technician_workers ∧
    total_average * (technician_count + non_technician_workers : ℝ) =
      technician_average * technician_count + non_technician_average * non_technician_workers ∧
    total_workers = 28 :=
by
  sorry

#check workshop_workers_count

end NUMINAMATH_CALUDE_workshop_workers_count_l1968_196849


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1968_196839

theorem fixed_point_on_line (m : ℝ) : (m - 1) * (7/2) - (m + 3) * (5/2) - (m - 11) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1968_196839


namespace NUMINAMATH_CALUDE_parabola_translation_l1968_196808

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 2 0 0  -- y = 2x²
  let translated := translate original 3 4
  y = translated.a * x^2 + translated.b * x + translated.c ↔
  y = 2 * (x + 3)^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1968_196808


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l1968_196890

theorem mistaken_multiplication (x y : ℕ) : 
  x ≥ 1000000 ∧ x ≤ 9999999 ∧
  y ≥ 1000000 ∧ y ≤ 9999999 ∧
  (10^7 : ℕ) * x + y = 3 * x * y →
  x = 3333333 ∧ y = 3333334 := by
sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l1968_196890


namespace NUMINAMATH_CALUDE_age_problem_l1968_196896

/-- Proves that given the age conditions, Mária is 36 2/3 years old and Anna is 7 1/3 years old -/
theorem age_problem (x y : ℚ) : 
  x + y = 44 → 
  x = 2 * (y - (-1/2 * x + 3/2 * (2/3 * y))) → 
  x = 110/3 ∧ y = 22/3 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1968_196896


namespace NUMINAMATH_CALUDE_binomial_coefficient_8_3_l1968_196891

theorem binomial_coefficient_8_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_8_3_l1968_196891


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l1968_196888

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 - x + 1)) →
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l1968_196888


namespace NUMINAMATH_CALUDE_total_holes_dug_l1968_196838

-- Define Pearl's digging rate
def pearl_rate : ℚ := 4 / 7

-- Define Miguel's digging rate
def miguel_rate : ℚ := 2 / 3

-- Define the duration of work
def work_duration : ℕ := 21

-- Theorem to prove
theorem total_holes_dug : 
  ⌊(pearl_rate * work_duration) + (miguel_rate * work_duration)⌋ = 26 := by
  sorry


end NUMINAMATH_CALUDE_total_holes_dug_l1968_196838


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_9_l1968_196835

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem five_digit_divisible_by_9 (B : ℕ) : 
  B < 10 → 
  is_divisible_by_9 (40000 + 1000 * B + 100 * B + 10 + 3) → 
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_9_l1968_196835


namespace NUMINAMATH_CALUDE_monotonicity_of_g_no_solutions_for_equation_l1968_196803

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a
def g (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a) / (x + 1)

theorem monotonicity_of_g :
  a = 1 →
  (∀ x₁ x₂, -1 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp 1 - 1 → g a x₁ < g a x₂) ∧
  (∀ x₁ x₂, Real.exp 1 - 1 < x₁ ∧ x₁ < x₂ → g a x₁ > g a x₂) :=
sorry

theorem no_solutions_for_equation :
  0 < a → a < 2/3 → ∀ x, f a x ≠ (x + 1) * g a x :=
sorry

end NUMINAMATH_CALUDE_monotonicity_of_g_no_solutions_for_equation_l1968_196803


namespace NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1968_196813

theorem no_primes_in_factorial_range (n : ℕ) (h : n > 2) :
  ∀ k : ℕ, n! + 2 < k ∧ k < n! + n + 1 → ¬ Nat.Prime k :=
by sorry

end NUMINAMATH_CALUDE_no_primes_in_factorial_range_l1968_196813


namespace NUMINAMATH_CALUDE_min_trig_expression_l1968_196847

open Real

theorem min_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  (3 * cos θ + 1 / sin θ + 4 * tan θ) ≥ 3 * (6 ^ (1 / 3)) ∧
  ∃ θ₀, 0 < θ₀ ∧ θ₀ < π / 2 ∧ 3 * cos θ₀ + 1 / sin θ₀ + 4 * tan θ₀ = 3 * (6 ^ (1 / 3)) :=
sorry

end NUMINAMATH_CALUDE_min_trig_expression_l1968_196847


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l1968_196800

theorem sunshine_orchard_pumpkins (moonglow_pumpkins : ℕ) (sunshine_pumpkins : ℕ) : 
  moonglow_pumpkins = 14 →
  sunshine_pumpkins = 3 * moonglow_pumpkins + 12 →
  sunshine_pumpkins = 54 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l1968_196800


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1968_196815

/-- An ellipse with foci at (3, 5) and (23, 40) that is tangent to the y-axis has a major axis of length 43.835 -/
theorem ellipse_major_axis_length : 
  ∀ (E : Set (ℝ × ℝ)) (F₁ F₂ Y : ℝ × ℝ),
  F₁ = (3, 5) →
  F₂ = (23, 40) →
  (∀ P ∈ E, ∃ k, dist P F₁ + dist P F₂ = k) →
  (∃ t, Y = (0, t) ∧ Y ∈ E) →
  (∀ P : ℝ × ℝ, P.1 = 0 → dist P F₁ + dist P F₂ ≥ dist Y F₁ + dist Y F₂) →
  dist F₁ F₂ = 43.835 := by
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1968_196815


namespace NUMINAMATH_CALUDE_joel_laps_l1968_196819

/-- Given that Yvonne swims 10 laps in 5 minutes, her younger sister swims half as many laps,
    and Joel swims three times as many laps as the younger sister,
    prove that Joel swims 15 laps in 5 minutes. -/
theorem joel_laps (yvonne_laps : ℕ) (younger_sister_ratio : ℚ) (joel_ratio : ℕ) :
  yvonne_laps = 10 →
  younger_sister_ratio = 1 / 2 →
  joel_ratio = 3 →
  (yvonne_laps : ℚ) * younger_sister_ratio * joel_ratio = 15 := by
  sorry

end NUMINAMATH_CALUDE_joel_laps_l1968_196819


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l1968_196881

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l1968_196881


namespace NUMINAMATH_CALUDE_production_rate_equation_l1968_196809

/-- Represents the production rates of a master and apprentice -/
theorem production_rate_equation (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 40) 
  (h3 : x + (40 - x) = 40) 
  (h4 : ∃ t : ℝ, t > 0 ∧ x * t = 300 ∧ (40 - x) * t = 100) : 
  300 / x = 100 / (40 - x) := by
  sorry

end NUMINAMATH_CALUDE_production_rate_equation_l1968_196809


namespace NUMINAMATH_CALUDE_population_growth_rate_l1968_196829

/-- Proves that given an initial population of 1200, a 25% increase in the first year,
    and a final population of 1950 after two years, the percentage increase in the second year is 30%. -/
theorem population_growth_rate (initial_population : ℕ) (first_year_increase : ℚ) 
  (final_population : ℕ) (second_year_increase : ℚ) : 
  initial_population = 1200 →
  first_year_increase = 25 / 100 →
  final_population = 1950 →
  (initial_population * (1 + first_year_increase) * (1 + second_year_increase) = final_population) →
  second_year_increase = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_rate_l1968_196829


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1968_196899

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1968_196899


namespace NUMINAMATH_CALUDE_debby_ate_nine_candies_l1968_196825

/-- Represents the number of candy pieces Debby ate -/
def candy_eaten (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proves that Debby ate 9 pieces of candy -/
theorem debby_ate_nine_candies : candy_eaten 12 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_debby_ate_nine_candies_l1968_196825


namespace NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l1968_196892

/-- The set A for a given real number a -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - a * x + 1 ≤ 0}

/-- Theorem stating the equivalence between A being empty and the range of a -/
theorem A_empty_iff_a_in_range : 
  ∀ a : ℝ, A a = ∅ ↔ 0 ≤ a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l1968_196892


namespace NUMINAMATH_CALUDE_number_of_operations_is_important_indicator_l1968_196884

-- Define the concept of an algorithm
structure Algorithm where
  operations : ℕ → ℕ  -- Number of operations as a function of input size

-- Define the concept of computer characteristics
structure ComputerCharacteristics where
  speed_importance : Prop  -- Speed is an important characteristic

-- Define the concept of algorithm quality indicators
structure QualityIndicator where
  is_important : Prop  -- Whether the indicator is important for algorithm quality

-- Define the specific indicator for number of operations
def number_of_operations : QualityIndicator where
  is_important := sorry  -- We'll prove this

-- State the theorem
theorem number_of_operations_is_important_indicator 
  (computer : ComputerCharacteristics) 
  (algo_quality_multifactor : Prop) : 
  computer.speed_importance → 
  algo_quality_multifactor → 
  number_of_operations.is_important :=
by sorry


end NUMINAMATH_CALUDE_number_of_operations_is_important_indicator_l1968_196884


namespace NUMINAMATH_CALUDE_books_in_year_l1968_196869

/-- The number of books Jack can read in a day -/
def books_per_day : ℕ := 9

/-- The number of days in a year -/
def days_in_year : ℕ := 365

/-- Theorem: Jack can read 3285 books in a year -/
theorem books_in_year : books_per_day * days_in_year = 3285 := by
  sorry

end NUMINAMATH_CALUDE_books_in_year_l1968_196869


namespace NUMINAMATH_CALUDE_train_length_l1968_196841

/-- Calculates the length of a train given its speed and the time and distance it takes to cross a bridge -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 57.6 * (1000 / 3600) →
  bridge_length = 150 →
  crossing_time = 25 →
  (train_speed * crossing_time) - bridge_length = 250 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1968_196841


namespace NUMINAMATH_CALUDE_distance_to_circle_center_l1968_196855

/-- The distance from a point in polar coordinates to the center of a circle defined by a polar equation --/
theorem distance_to_circle_center (ρ₀ : ℝ) (θ₀ : ℝ) :
  let circle := fun θ => 2 * Real.cos θ
  let center_x := 1
  let center_y := 0
  let point_x := ρ₀ * Real.cos θ₀
  let point_y := ρ₀ * Real.sin θ₀
  (ρ₀ = 2 ∧ θ₀ = Real.pi / 3) →
  Real.sqrt ((point_x - center_x)^2 + (point_y - center_y)^2) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_circle_center_l1968_196855


namespace NUMINAMATH_CALUDE_remainder_sum_l1968_196856

theorem remainder_sum (a b : ℤ) : 
  a % 45 = 37 → b % 30 = 9 → (a + b) % 15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l1968_196856


namespace NUMINAMATH_CALUDE_probability_not_raining_l1968_196883

theorem probability_not_raining (p : ℚ) (h : p = 4/9) : 1 - p = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_raining_l1968_196883


namespace NUMINAMATH_CALUDE_triangle_area_extension_l1968_196876

/-- Given a triangle ABC with area 36 and base BC of length 7, and an extended triangle BCD
    with CD of length 30, prove that the area of BCD is 1080/7. -/
theorem triangle_area_extension (h : ℝ) : 
  36 = (1/2) * 7 * h →  -- Area of ABC
  (1/2) * 30 * h = 1080/7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_extension_l1968_196876


namespace NUMINAMATH_CALUDE_locus_is_two_ellipses_l1968_196898

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the locus of points
def LocusOfPoints (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | abs (dist p c1.center - c1.radius) = abs (dist p c2.center - c2.radius)}

-- Define the ellipse
def Ellipse (f1 f2 : ℝ × ℝ) (major_axis : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p f1 + dist p f2 = major_axis}

-- Theorem statement
theorem locus_is_two_ellipses (c1 c2 : Circle) 
  (h1 : c1.radius > c2.radius) 
  (h2 : dist c1.center c2.center < c1.radius - c2.radius) :
  LocusOfPoints c1 c2 = 
    Ellipse c1.center c2.center (c1.radius + c2.radius) ∪
    Ellipse c1.center c2.center (c1.radius - c2.radius) := by
  sorry


end NUMINAMATH_CALUDE_locus_is_two_ellipses_l1968_196898


namespace NUMINAMATH_CALUDE_lens_discount_l1968_196830

def old_camera_price : ℝ := 4000
def lens_original_price : ℝ := 400
def total_paid : ℝ := 5400
def price_increase_percentage : ℝ := 0.30

theorem lens_discount (new_camera_price : ℝ) (lens_paid : ℝ) 
  (h1 : new_camera_price = old_camera_price * (1 + price_increase_percentage))
  (h2 : total_paid = new_camera_price + lens_paid) :
  lens_original_price - lens_paid = 200 := by
sorry

end NUMINAMATH_CALUDE_lens_discount_l1968_196830


namespace NUMINAMATH_CALUDE_multiplication_subtraction_equality_l1968_196814

theorem multiplication_subtraction_equality : 210 * 6 - 52 * 5 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_subtraction_equality_l1968_196814


namespace NUMINAMATH_CALUDE_min_fruits_problem_l1968_196812

theorem min_fruits_problem : ∃ n : ℕ, n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 4 ∧ 
  n % 6 = 5 ∧ 
  (∀ m : ℕ, m > 0 → m % 3 = 2 → m % 4 = 3 → m % 5 = 4 → m % 6 = 5 → m ≥ n) ∧
  n = 59 := by
sorry

end NUMINAMATH_CALUDE_min_fruits_problem_l1968_196812


namespace NUMINAMATH_CALUDE_sin_240_degrees_l1968_196871

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l1968_196871


namespace NUMINAMATH_CALUDE_rearrangeable_shapes_exist_l1968_196840

/-- Represents a shape that can be divided and rearranged -/
structure Divisible2DShape where
  area : ℝ
  can_form_square : Bool
  can_form_triangle : Bool

/-- Represents a set of shapes that can be rearranged -/
def ShapeSet := List Divisible2DShape

/-- Function to check if a shape set can form a square -/
def can_form_square (shapes : ShapeSet) : Bool :=
  shapes.any (·.can_form_square)

/-- Function to check if a shape set can form a triangle -/
def can_form_triangle (shapes : ShapeSet) : Bool :=
  shapes.any (·.can_form_triangle)

/-- The main theorem statement -/
theorem rearrangeable_shapes_exist (a : ℝ) (h : a > 0) :
  ∃ (shapes : ShapeSet),
    -- The total area of shapes is greater than the initial square
    (shapes.map (·.area)).sum > a^2 ∧
    -- The shape set can form two different squares
    can_form_square shapes ∧
    -- The shape set can form two different triangles
    can_form_triangle shapes :=
  sorry


end NUMINAMATH_CALUDE_rearrangeable_shapes_exist_l1968_196840


namespace NUMINAMATH_CALUDE_pigeon_count_l1968_196886

/-- The number of pigeons in the pigeon house -/
def num_pigeons : ℕ := 600

/-- The number of days the feed lasts if 75 pigeons are sold -/
def days_after_selling : ℕ := 20

/-- The number of days the feed lasts if 100 pigeons are bought -/
def days_after_buying : ℕ := 15

/-- The number of pigeons sold -/
def pigeons_sold : ℕ := 75

/-- The number of pigeons bought -/
def pigeons_bought : ℕ := 100

/-- Theorem stating that the number of pigeons in the pigeon house is 600 -/
theorem pigeon_count : 
  (num_pigeons - pigeons_sold) * days_after_selling = (num_pigeons + pigeons_bought) * days_after_buying :=
by sorry

end NUMINAMATH_CALUDE_pigeon_count_l1968_196886


namespace NUMINAMATH_CALUDE_fibonacci_gcd_l1968_196880

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_gcd :
  Nat.gcd (fib 2017) (fib 99 * fib 101 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_gcd_l1968_196880


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l1968_196897

theorem positive_real_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2) ∧
  (a^3 + b^3 + c^3 + 1/a + 1/b + 1/c ≥ 2 * (a + b + c)) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l1968_196897
