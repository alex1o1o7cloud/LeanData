import Mathlib

namespace NUMINAMATH_CALUDE_hall_width_is_25_l940_94088

/-- Represents the dimensions and cost parameters of a rectangular hall --/
structure HallParameters where
  length : ℝ
  height : ℝ
  cost_per_sqm : ℝ
  total_cost : ℝ

/-- Calculates the total area to be covered in the hall --/
def total_area (params : HallParameters) (width : ℝ) : ℝ :=
  params.length * width + 2 * (params.length * params.height) + 2 * (width * params.height)

/-- Theorem stating that the width of the hall is 25 meters given the specified parameters --/
theorem hall_width_is_25 (params : HallParameters) 
    (h1 : params.length = 20)
    (h2 : params.height = 5)
    (h3 : params.cost_per_sqm = 40)
    (h4 : params.total_cost = 38000) :
    ∃ w : ℝ, w = 25 ∧ total_area params w * params.cost_per_sqm = params.total_cost :=
  sorry

end NUMINAMATH_CALUDE_hall_width_is_25_l940_94088


namespace NUMINAMATH_CALUDE_investment_theorem_l940_94095

/-- Calculates the total investment with interest after one year -/
def total_investment_with_interest (initial_investment : ℝ) (amount_at_5_percent : ℝ) (rate_5_percent : ℝ) (rate_6_percent : ℝ) : ℝ :=
  let amount_at_6_percent := initial_investment - amount_at_5_percent
  let interest_5_percent := amount_at_5_percent * rate_5_percent
  let interest_6_percent := amount_at_6_percent * rate_6_percent
  initial_investment + interest_5_percent + interest_6_percent

/-- Theorem stating that the total investment with interest is $1,054 -/
theorem investment_theorem :
  total_investment_with_interest 1000 600 0.05 0.06 = 1054 := by
  sorry

end NUMINAMATH_CALUDE_investment_theorem_l940_94095


namespace NUMINAMATH_CALUDE_pool_depth_is_10_feet_l940_94015

-- Define the pool parameters
def drainRate : ℝ := 60
def poolWidth : ℝ := 80
def poolLength : ℝ := 150
def drainTime : ℝ := 2000

-- Theorem statement
theorem pool_depth_is_10_feet :
  let totalVolume := drainRate * drainTime
  let poolArea := poolWidth * poolLength
  totalVolume / poolArea = 10 := by
  sorry

end NUMINAMATH_CALUDE_pool_depth_is_10_feet_l940_94015


namespace NUMINAMATH_CALUDE_expression_simplification_l940_94032

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) :
  (a^(7/3) - 2*a^(5/3)*b^(2/3) + a*b^(4/3)) / (a^(5/3) - a^(4/3)*b^(1/3) - a*b^(2/3) + a^(2/3)*b) / a^(1/3) = a^(1/3) + b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l940_94032


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l940_94089

-- Define the points and line
def M : ℝ × ℝ := (-3, 4)
def N : ℝ × ℝ := (2, 6)
def l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the reflection property
def is_reflection (M N M' : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), M' = (a, b) ∧
  (b - M.2) / (a - M.1) = -1 ∧
  (M.1 + a) / 2 - (b + M.2) / 2 + 3 = 0

-- Define the property of a line passing through two points
def line_through_points (P Q : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - P.2) * (Q.1 - P.1) = (x - P.1) * (Q.2 - P.2)

-- Theorem statement
theorem reflected_ray_equation 
  (h_reflection : ∃ M' : ℝ × ℝ, is_reflection M N M' l) :
  ∀ x y : ℝ, line_through_points M' N x y ↔ 6 * x - y - 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l940_94089


namespace NUMINAMATH_CALUDE_tetrahedron_max_lateral_area_l940_94096

/-- Given a tetrahedron A-BCD where AB, AC, AD are mutually perpendicular
    and the radius of the circumscribed sphere is 2,
    prove that the maximum lateral surface area S of the tetrahedron is 8. -/
theorem tetrahedron_max_lateral_area :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 + c^2 = 16 →
  (∀ (S : ℝ), S = (a * b + b * c + a * c) / 2 → S ≤ 8) ∧
  (∃ (S : ℝ), S = (a * b + b * c + a * c) / 2 ∧ S = 8) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_max_lateral_area_l940_94096


namespace NUMINAMATH_CALUDE_point_in_region_l940_94037

theorem point_in_region (m : ℝ) :
  (2 * m + 3 < 4) → m < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_l940_94037


namespace NUMINAMATH_CALUDE_cubic_expression_factorization_l940_94012

theorem cubic_expression_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) =
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_expression_factorization_l940_94012


namespace NUMINAMATH_CALUDE_normal_distribution_mean_half_l940_94069

-- Define a random variable following normal distribution
def normal_distribution (μ : ℝ) (σ : ℝ) (hσ : σ > 0) : Type := ℝ

-- Define the probability function
noncomputable def P (ξ : normal_distribution μ σ hσ) (pred : ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_mean_half 
  (μ σ : ℝ) (hσ : σ > 0) (ξ : normal_distribution μ σ hσ) 
  (h : P ξ (λ x => x < 0) + P ξ (λ x => x < 1) = 1) : 
  μ = 1/2 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_mean_half_l940_94069


namespace NUMINAMATH_CALUDE_max_quadratic_solution_l940_94042

theorem max_quadratic_solution (k a b c : ℕ+) (r : ℝ) :
  (∃ m n l : ℕ, a = k ^ m ∧ b = k ^ n ∧ c = k ^ l) →
  (a * r ^ 2 - b * r + c = 0) →
  (∀ x : ℝ, x ≠ r → a * x ^ 2 - b * x + c ≠ 0) →
  r < 100 →
  r ≤ 64 := by
sorry

end NUMINAMATH_CALUDE_max_quadratic_solution_l940_94042


namespace NUMINAMATH_CALUDE_seokjin_drank_least_l940_94084

def seokjin_milk : ℚ := 11/10
def jungkook_milk : ℚ := 13/10
def yoongi_milk : ℚ := 7/6

theorem seokjin_drank_least :
  seokjin_milk < jungkook_milk ∧ seokjin_milk < yoongi_milk :=
by sorry

end NUMINAMATH_CALUDE_seokjin_drank_least_l940_94084


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l940_94003

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_rate_increase : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 16)
  (h2 : regular_hours = 40)
  (h3 : overtime_rate_increase = 0.75)
  (h4 : total_hours = 52) :
  let overtime_hours := total_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * overtime_hours
  regular_pay + overtime_pay = 976 := by
sorry


end NUMINAMATH_CALUDE_bus_driver_compensation_l940_94003


namespace NUMINAMATH_CALUDE_weight_differences_l940_94053

/-- Given the weights of four individuals, prove the weight differences between one individual and the other three. -/
theorem weight_differences (H E1 E2 E3 : ℕ) 
  (h_H : H = 87)
  (h_E1 : E1 = 58)
  (h_E2 : E2 = 56)
  (h_E3 : E3 = 64) :
  (H - E1 = 29) ∧ (H - E2 = 31) ∧ (H - E3 = 23) := by
  sorry

end NUMINAMATH_CALUDE_weight_differences_l940_94053


namespace NUMINAMATH_CALUDE_largest_inscribed_circle_circumference_l940_94038

theorem largest_inscribed_circle_circumference (square_side : ℝ) (h : square_side = 12) :
  let circle_radius := square_side / 2
  2 * Real.pi * circle_radius = 12 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_circle_circumference_l940_94038


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l940_94000

theorem quadratic_root_difference (x : ℝ) : 
  let a : ℝ := 1
  let b : ℝ := -9
  let c : ℝ := 4
  let r₁ : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x^2 - 9*x + 4 = 0 → abs (r₁ - r₂) = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l940_94000


namespace NUMINAMATH_CALUDE_original_fraction_l940_94027

theorem original_fraction (x y : ℚ) : 
  (x > 0) → (y > 0) → 
  ((6/5 * x) / (9/10 * y) = 20/21) → 
  (x / y = 10/21) := by
sorry

end NUMINAMATH_CALUDE_original_fraction_l940_94027


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l940_94098

/-- 
Given a quadratic equation ax^2 - 4x - 1 = 0, this theorem states the conditions
on 'a' for the equation to have two distinct real roots.
-/
theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    a * x^2 - 4 * x - 1 = 0 ∧ 
    a * y^2 - 4 * y - 1 = 0) ↔ 
  (a > -4 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l940_94098


namespace NUMINAMATH_CALUDE_second_year_sample_size_l940_94090

/-- Represents the number of students to be sampled from each year group -/
structure SampleSize where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  fourth_year : ℕ

/-- Calculates the sample size for stratified sampling -/
def stratified_sample (total_students : ℕ) (sample_size : ℕ) (ratio : List ℕ) : SampleSize :=
  sorry

/-- Theorem stating the correct number of second-year students to be sampled -/
theorem second_year_sample_size :
  let total_students : ℕ := 5000
  let sample_size : ℕ := 260
  let ratio : List ℕ := [5, 4, 3, 1]
  let result := stratified_sample total_students sample_size ratio
  result.second_year = 80 := by sorry

end NUMINAMATH_CALUDE_second_year_sample_size_l940_94090


namespace NUMINAMATH_CALUDE_cookies_in_fridge_l940_94080

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 256

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 15

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 23

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * tim_cookies

/-- The number of cookies put in the fridge -/
def fridge_cookies : ℕ := total_cookies - (tim_cookies + mike_cookies + anna_cookies)

theorem cookies_in_fridge : fridge_cookies = 188 := by
  sorry

end NUMINAMATH_CALUDE_cookies_in_fridge_l940_94080


namespace NUMINAMATH_CALUDE_six_power_plus_one_same_digits_l940_94060

def has_same_digits (m : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ ∀ k : ℕ, (m / 10^k) % 10 = d

theorem six_power_plus_one_same_digits :
  {n : ℕ | n > 0 ∧ has_same_digits (6^n + 1)} = {1, 5} := by sorry

end NUMINAMATH_CALUDE_six_power_plus_one_same_digits_l940_94060


namespace NUMINAMATH_CALUDE_negation_of_implication_l940_94009

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → a^2 > b^2) ↔ (a ≤ b → a^2 ≤ b^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l940_94009


namespace NUMINAMATH_CALUDE_product_of_numbers_l940_94033

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 106) :
  x * y = 32 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l940_94033


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l940_94066

def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

theorem intersection_of_A_and_B : A ∩ B = {8, 10} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l940_94066


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l940_94016

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 2*a + 3*b ≤ x → x ≥ 3) ∧ 
  (∀ y, 2*a + 3*b ≥ y → y ≤ -3) :=
by sorry

#check range_of_2a_plus_3b

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l940_94016


namespace NUMINAMATH_CALUDE_steves_remaining_oranges_l940_94052

/-- Given that Steve has 46 oranges initially, shares 4 with Patrick and 7 with Samantha,
    prove that he will have 35 oranges left. -/
theorem steves_remaining_oranges :
  ∀ (initial shared_patrick shared_samantha : ℕ),
    initial = 46 →
    shared_patrick = 4 →
    shared_samantha = 7 →
    initial - (shared_patrick + shared_samantha) = 35 := by
  sorry

end NUMINAMATH_CALUDE_steves_remaining_oranges_l940_94052


namespace NUMINAMATH_CALUDE_cars_sold_first_three_days_l940_94019

/-- Proves that the number of cars sold each day for the first three days is 5 --/
theorem cars_sold_first_three_days :
  let total_quota : ℕ := 50
  let cars_sold_next_four_days : ℕ := 3 * 4
  let remaining_cars_to_sell : ℕ := 23
  let cars_per_day_first_three_days : ℕ := (total_quota - cars_sold_next_four_days - remaining_cars_to_sell) / 3
  cars_per_day_first_three_days = 5 := by
  sorry

#eval (50 - 3 * 4 - 23) / 3

end NUMINAMATH_CALUDE_cars_sold_first_three_days_l940_94019


namespace NUMINAMATH_CALUDE_sum_squares_equality_l940_94091

theorem sum_squares_equality (N : ℕ) : 
  (1^2 + 2^2 + 3^2 + 4^2) / 4 = (2000^2 + 2001^2 + 2002^2 + 2003^2) / N → N = 2134 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_equality_l940_94091


namespace NUMINAMATH_CALUDE_gcd_of_256_450_720_l940_94073

theorem gcd_of_256_450_720 : Nat.gcd 256 (Nat.gcd 450 720) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_450_720_l940_94073


namespace NUMINAMATH_CALUDE_math_teacher_initial_amount_l940_94085

theorem math_teacher_initial_amount :
  let basic_calculator_cost : ℕ := 8
  let scientific_calculator_cost : ℕ := 2 * basic_calculator_cost
  let graphing_calculator_cost : ℕ := 3 * scientific_calculator_cost
  let total_cost : ℕ := basic_calculator_cost + scientific_calculator_cost + graphing_calculator_cost
  let change : ℕ := 28
  let initial_amount : ℕ := total_cost + change
  initial_amount = 100
  := by sorry

end NUMINAMATH_CALUDE_math_teacher_initial_amount_l940_94085


namespace NUMINAMATH_CALUDE_square_equation_solution_l940_94024

theorem square_equation_solution : ∃! (M : ℕ), M > 0 ∧ 14^2 * 35^2 = 70^2 * M^2 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l940_94024


namespace NUMINAMATH_CALUDE_max_value_of_function_max_value_achievable_l940_94028

theorem max_value_of_function (x : ℝ) : 
  x^6 / (x^8 + 2*x^7 - 4*x^6 + 8*x^5 + 16*x^4) ≤ 1/12 :=
sorry

theorem max_value_achievable : 
  ∃ x : ℝ, x^6 / (x^8 + 2*x^7 - 4*x^6 + 8*x^5 + 16*x^4) = 1/12 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_max_value_achievable_l940_94028


namespace NUMINAMATH_CALUDE_rectangular_prism_painted_faces_l940_94045

theorem rectangular_prism_painted_faces (a : ℕ) : 
  2 < a → a < 5 → (a - 2) * 3 * 4 = 4 * 3 + 4 * 4 → a = 4 := by sorry

end NUMINAMATH_CALUDE_rectangular_prism_painted_faces_l940_94045


namespace NUMINAMATH_CALUDE_ball_distribution_and_fairness_l940_94064

-- Define the total number of balls
def total_balls : ℕ := 4

-- Define the probabilities
def prob_red_or_yellow : ℚ := 3/4
def prob_yellow_or_blue : ℚ := 1/2

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 1
def blue_balls : ℕ := 1

-- Define the probabilities of drawing same color and different colors
def prob_same_color : ℚ := 3/8
def prob_diff_color : ℚ := 5/8

theorem ball_distribution_and_fairness :
  (red_balls + yellow_balls + blue_balls = total_balls) ∧
  (red_balls : ℚ) / total_balls + (yellow_balls : ℚ) / total_balls = prob_red_or_yellow ∧
  (yellow_balls : ℚ) / total_balls + (blue_balls : ℚ) / total_balls = prob_yellow_or_blue ∧
  prob_diff_color > prob_same_color :=
sorry

end NUMINAMATH_CALUDE_ball_distribution_and_fairness_l940_94064


namespace NUMINAMATH_CALUDE_sin_minus_cos_tan_one_third_l940_94087

theorem sin_minus_cos_tan_one_third (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan θ = 1/3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_tan_one_third_l940_94087


namespace NUMINAMATH_CALUDE_darnell_initial_fabric_l940_94005

/-- Calculates the initial amount of fabric Darnell had --/
def initial_fabric (square_side : ℕ) (wide_length wide_width : ℕ) (tall_length tall_width : ℕ)
  (num_square num_wide num_tall : ℕ) (fabric_left : ℕ) : ℕ :=
  let square_area := square_side * square_side
  let wide_area := wide_length * wide_width
  let tall_area := tall_length * tall_width
  let total_used := square_area * num_square + wide_area * num_wide + tall_area * num_tall
  total_used + fabric_left

/-- Theorem stating that Darnell initially had 1000 square feet of fabric --/
theorem darnell_initial_fabric :
  initial_fabric 4 5 3 3 5 16 20 10 294 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_darnell_initial_fabric_l940_94005


namespace NUMINAMATH_CALUDE_factorization_proof_l940_94047

theorem factorization_proof (x y : ℝ) : 
  y^2 * (x - 2) + 16 * (2 - x) = (x - 2) * (y + 4) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l940_94047


namespace NUMINAMATH_CALUDE_probability_divisible_by_4_l940_94061

/-- Represents the possible outcomes of a single spin -/
inductive SpinOutcome
| one
| two
| three

/-- Represents a three-digit number formed by three spins -/
structure ThreeDigitNumber where
  hundreds : SpinOutcome
  tens : SpinOutcome
  units : SpinOutcome

/-- Checks if a ThreeDigitNumber is divisible by 4 -/
def isDivisibleBy4 (n : ThreeDigitNumber) : Prop := sorry

/-- The total number of possible three-digit numbers -/
def totalOutcomes : ℕ := sorry

/-- The number of three-digit numbers divisible by 4 -/
def divisibleBy4Outcomes : ℕ := sorry

/-- The main theorem stating the probability of getting a number divisible by 4 -/
theorem probability_divisible_by_4 :
  (divisibleBy4Outcomes : ℚ) / totalOutcomes = 2 / 9 := sorry

end NUMINAMATH_CALUDE_probability_divisible_by_4_l940_94061


namespace NUMINAMATH_CALUDE_right_angled_triangle_set_l940_94021

theorem right_angled_triangle_set : 
  ∃! (a b c : ℝ), (a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3) ∧ 
  a^2 + b^2 = c^2 ∧ 
  ((a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 1 ∧ b = 1 ∧ c = 2) ∨ 
   (a = 5 ∧ b = 12 ∧ c = 15) ∨ 
   (a = 1 ∧ b = Real.sqrt 2 ∧ c = Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_set_l940_94021


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l940_94010

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * x + 1 / (x^2) ≥ 4 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 0) : 
  (3 * x + 1 / (x^2) = 4) ↔ (x = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l940_94010


namespace NUMINAMATH_CALUDE_empty_set_condition_l940_94007

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | Real.sqrt (x - 3) = a * x + 1}

-- State the theorem
theorem empty_set_condition (a : ℝ) :
  IsEmpty (A a) ↔ a < -1/2 ∨ a > 1/6 := by sorry

end NUMINAMATH_CALUDE_empty_set_condition_l940_94007


namespace NUMINAMATH_CALUDE_twelfth_term_equals_three_over_512_l940_94046

/-- The nth term of a geometric sequence -/
def geometricSequenceTerm (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

/-- The 12th term of the specific geometric sequence -/
def twelfthTerm : ℚ :=
  geometricSequenceTerm 12 (1/2) 12

theorem twelfth_term_equals_three_over_512 :
  twelfthTerm = 3/512 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_equals_three_over_512_l940_94046


namespace NUMINAMATH_CALUDE_base2_to_base4_example_l940_94063

/-- Converts a natural number from base 2 to base 4 -/
def base2ToBase4 (n : ℕ) : ℕ := sorry

theorem base2_to_base4_example : base2ToBase4 0b10111010000 = 0x11310 := by sorry

end NUMINAMATH_CALUDE_base2_to_base4_example_l940_94063


namespace NUMINAMATH_CALUDE_parabola_increasing_condition_l940_94093

/-- A parabola defined by y = (a - 1)x^2 + 1 -/
def parabola (a : ℝ) (x : ℝ) : ℝ := (a - 1) * x^2 + 1

/-- The parabola increases as x increases when x ≥ 0 -/
def increases_for_nonneg_x (a : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → parabola a x₁ < parabola a x₂

theorem parabola_increasing_condition (a : ℝ) :
  increases_for_nonneg_x a → a > 1 := by sorry

end NUMINAMATH_CALUDE_parabola_increasing_condition_l940_94093


namespace NUMINAMATH_CALUDE_new_average_height_l940_94026

/-- Calculates the new average height of a class after some students leave and others join. -/
theorem new_average_height
  (initial_size : ℕ)
  (initial_avg : ℝ)
  (left_size : ℕ)
  (left_avg : ℝ)
  (joined_size : ℕ)
  (joined_avg : ℝ)
  (h_initial_size : initial_size = 35)
  (h_initial_avg : initial_avg = 180)
  (h_left_size : left_size = 7)
  (h_left_avg : left_avg = 120)
  (h_joined_size : joined_size = 7)
  (h_joined_avg : joined_avg = 140)
  : (initial_size * initial_avg - left_size * left_avg + joined_size * joined_avg) / initial_size = 184 := by
  sorry

end NUMINAMATH_CALUDE_new_average_height_l940_94026


namespace NUMINAMATH_CALUDE_triangle_problem_l940_94051

theorem triangle_problem (A B C : ℝ) (a b c S : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  S = (1/2) * a * b * Real.sin C ∧
  (Real.cos B) / (Real.cos C) = -b / (2*a + c) →
  (B = 2*π/3 ∧
   (a = 4 ∧ S = 5 * Real.sqrt 3 → b = Real.sqrt 61)) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l940_94051


namespace NUMINAMATH_CALUDE_constant_term_expansion_l940_94036

/-- The constant term in the expansion of (x - 1/(2x))^6 is -5/2 -/
theorem constant_term_expansion : 
  let f : ℝ → ℝ := λ x => (x - 1/(2*x))^6
  ∃ (c : ℝ), (∀ x ≠ 0, f x = c + x * (f x - c) / x) ∧ c = -5/2 :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l940_94036


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l940_94068

theorem complex_fraction_equality : (1 + I : ℂ) / (2 - I) = 1/5 + 3/5 * I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l940_94068


namespace NUMINAMATH_CALUDE_find_z_l940_94002

theorem find_z (M N : Set ℂ) (i : ℂ) (z : ℂ) : 
  M = {1, 2, z * i} →
  N = {3, 4} →
  M ∩ N = {4} →
  i * i = -1 →
  z = 4 * i :=
by sorry

end NUMINAMATH_CALUDE_find_z_l940_94002


namespace NUMINAMATH_CALUDE_light_bulb_conditional_probability_l940_94079

theorem light_bulb_conditional_probability 
  (p_3000 : ℝ) 
  (p_4500 : ℝ) 
  (h1 : p_3000 = 0.8) 
  (h2 : p_4500 = 0.2) 
  (h3 : p_3000 ≠ 0) : 
  p_4500 / p_3000 = 0.25 := by
sorry

end NUMINAMATH_CALUDE_light_bulb_conditional_probability_l940_94079


namespace NUMINAMATH_CALUDE_inverse_f_at_142_l940_94017

def f (x : ℝ) : ℝ := 5 * x^3 + 7

theorem inverse_f_at_142 : f⁻¹ 142 = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_142_l940_94017


namespace NUMINAMATH_CALUDE_integer_condition_l940_94029

theorem integer_condition (x : ℝ) : 
  (∀ x : ℤ, ∃ y : ℤ, 2 * (x : ℝ) + 1 = y) ∧ 
  (∃ x : ℝ, ∃ y : ℤ, 2 * x + 1 = y ∧ ¬∃ z : ℤ, x = z) :=
sorry

end NUMINAMATH_CALUDE_integer_condition_l940_94029


namespace NUMINAMATH_CALUDE_remainder_theorem_l940_94039

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_at_20 : Q 20 = 100
axiom Q_at_100 : Q 100 = 20

-- Define the remainder function
def remainder (f : ℝ → ℝ) (x : ℝ) : ℝ := -x + 120

-- State the theorem
theorem remainder_theorem :
  ∃ (R : ℝ → ℝ), ∀ x, Q x = (x - 20) * (x - 100) * R x + remainder Q x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l940_94039


namespace NUMINAMATH_CALUDE_sin_2theta_value_l940_94070

theorem sin_2theta_value (θ : Real) (h : Real.sin θ + Real.cos θ = Real.sqrt 7 / 2) :
  Real.sin (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l940_94070


namespace NUMINAMATH_CALUDE_pie_eating_contest_l940_94022

theorem pie_eating_contest (first_student second_student : ℚ) 
  (h1 : first_student = 6/7)
  (h2 : second_student = 3/4) :
  first_student - second_student = 3/28 := by
  sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l940_94022


namespace NUMINAMATH_CALUDE_x_plus_x_squared_equals_twelve_l940_94030

theorem x_plus_x_squared_equals_twelve (x : ℝ) (h : x = 3) : x + x^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_x_squared_equals_twelve_l940_94030


namespace NUMINAMATH_CALUDE_sine_translation_stretch_l940_94020

/-- The transformation of the sine function -/
theorem sine_translation_stretch (x : ℝ) :
  let f := λ x : ℝ => Real.sin x
  let g := λ x : ℝ => Real.sin (x / 2 - π / 8)
  g x = (f ∘ (λ y => y - π / 8) ∘ (λ y => y / 2)) x :=
by sorry

end NUMINAMATH_CALUDE_sine_translation_stretch_l940_94020


namespace NUMINAMATH_CALUDE_sum_RS_ST_l940_94049

/-- Represents a polygon PQRSTU -/
structure Polygon :=
  (area : ℝ)
  (PQ : ℝ)
  (QR : ℝ)
  (TU : ℝ)

/-- Theorem stating the sum of RS and ST in the polygon PQRSTU -/
theorem sum_RS_ST (poly : Polygon) (h1 : poly.area = 70) (h2 : poly.PQ = 10) 
  (h3 : poly.QR = 7) (h4 : poly.TU = 6) : ∃ (RS ST : ℝ), RS + ST = 80 := by
  sorry

#check sum_RS_ST

end NUMINAMATH_CALUDE_sum_RS_ST_l940_94049


namespace NUMINAMATH_CALUDE_beef_cost_calculation_l940_94056

/-- Proves that the cost of a pound of beef is $5 given the initial amount,
    cheese cost, quantities purchased, and remaining amount. -/
theorem beef_cost_calculation (initial_amount : ℕ) (cheese_cost : ℕ) 
  (cheese_quantity : ℕ) (beef_quantity : ℕ) (remaining_amount : ℕ) :
  initial_amount = 87 →
  cheese_cost = 7 →
  cheese_quantity = 3 →
  beef_quantity = 1 →
  remaining_amount = 61 →
  initial_amount - remaining_amount - (cheese_cost * cheese_quantity) = 5 :=
by sorry

end NUMINAMATH_CALUDE_beef_cost_calculation_l940_94056


namespace NUMINAMATH_CALUDE_set_operations_l940_94023

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,2,5}

theorem set_operations :
  (A ∩ B = {2,5}) ∧ (A ∪ (U \ B) = {2,3,4,5,6}) := by sorry

end NUMINAMATH_CALUDE_set_operations_l940_94023


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l940_94083

theorem four_digit_number_problem (n : ℕ) : 
  (1000 ≤ n) ∧ (n < 10000) ∧  -- n is a four-digit number
  (n % 10 = 9) ∧              -- the ones digit of n is 9
  ((n - 3) + 57 = 1823)       -- the sum of the mistaken number and 57 is 1823
  → n = 1769 := by
sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l940_94083


namespace NUMINAMATH_CALUDE_pen_cost_l940_94074

/-- The cost of a pen given Elizabeth's budget and purchasing constraints -/
theorem pen_cost (total_budget : ℝ) (pencil_cost : ℝ) (pencil_count : ℕ) (pen_count : ℕ) :
  total_budget = 20 →
  pencil_cost = 1.6 →
  pencil_count = 5 →
  pen_count = 6 →
  (pencil_count * pencil_cost + pen_count * ((total_budget - pencil_count * pencil_cost) / pen_count) = total_budget) →
  (total_budget - pencil_count * pencil_cost) / pen_count = 2 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l940_94074


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l940_94040

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  (x - 2)^2 + 2^2 = 10^2 → 
  x = 2 + 4 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l940_94040


namespace NUMINAMATH_CALUDE_percentage_increase_l940_94082

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 800 → final = 1680 → 
  ((final - initial) / initial) * 100 = 110 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l940_94082


namespace NUMINAMATH_CALUDE_smallest_four_digit_number_with_remainders_l940_94031

theorem smallest_four_digit_number_with_remainders : ∃ (n : ℕ),
  (n ≥ 1000 ∧ n < 10000) ∧
  (n % 5 = 1) ∧
  (n % 7 = 4) ∧
  (n % 11 = 9) ∧
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 5 = 1 ∧ m % 7 = 4 ∧ m % 11 = 9 → n ≤ m) ∧
  n = 1131 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_number_with_remainders_l940_94031


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l940_94004

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_properties (a b : ℝ) :
  f a b 0 = 6 ∧ f a b 1 = 5 →
  (∀ x, f a b x = x^2 - 2*x + 6) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a b x ≥ 5) ∧
  (∀ x ∈ Set.Icc (-2) 2, f a b x ≤ 14) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a b x = 5) ∧
  (∃ x ∈ Set.Icc (-2) 2, f a b x = 14) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l940_94004


namespace NUMINAMATH_CALUDE_school_rewards_problem_l940_94035

/-- The price of a practical backpack -/
def backpack_price : ℝ := 60

/-- The price of a multi-functional pencil case -/
def pencil_case_price : ℝ := 40

/-- The total budget for purchases -/
def total_budget : ℝ := 1140

/-- The total number of items to be purchased -/
def total_items : ℕ := 25

/-- The maximum number of backpacks that can be purchased -/
def max_backpacks : ℕ := 7

theorem school_rewards_problem :
  (3 * backpack_price + 2 * pencil_case_price = 260) ∧
  (5 * backpack_price + 4 * pencil_case_price = 460) ∧
  (∀ m : ℕ, m ≤ total_items → 
    backpack_price * m + pencil_case_price * (total_items - m) ≤ total_budget) →
  max_backpacks = 7 := by sorry

end NUMINAMATH_CALUDE_school_rewards_problem_l940_94035


namespace NUMINAMATH_CALUDE_triangle_side_length_l940_94018

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  b = 7 →
  c = 6 →
  Real.cos (B - C) = 15/16 →
  a = 5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l940_94018


namespace NUMINAMATH_CALUDE_triangle_shape_not_unique_l940_94001

/-- A triangle with sides a, b, c and angles A, B, C --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The shape of a triangle is not uniquely determined by the product of two sides and the angle between them --/
theorem triangle_shape_not_unique (p : ℝ) (γ : ℝ) :
  ∃ (t1 t2 : Triangle), t1 ≠ t2 ∧ t1.a * t1.b = p ∧ t1.C = γ ∧ t2.a * t2.b = p ∧ t2.C = γ :=
sorry

end NUMINAMATH_CALUDE_triangle_shape_not_unique_l940_94001


namespace NUMINAMATH_CALUDE_swap_correct_specific_swap_l940_94076

def swap_values (a b : ℕ) : ℕ × ℕ := 
  let c := a
  let a' := b
  let b' := c
  (a', b')

theorem swap_correct (a b : ℕ) : 
  let (a', b') := swap_values a b
  a' = b ∧ b' = a := by
sorry

theorem specific_swap : 
  let (a', b') := swap_values 10 20
  a' = 20 ∧ b' = 10 := by
sorry

end NUMINAMATH_CALUDE_swap_correct_specific_swap_l940_94076


namespace NUMINAMATH_CALUDE_inequality_proof_l940_94071

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l940_94071


namespace NUMINAMATH_CALUDE_surjective_injective_ge_equal_l940_94094

theorem surjective_injective_ge_equal (f g : ℕ → ℕ) 
  (hf : Function.Surjective f)
  (hg : Function.Injective g)
  (h : ∀ n : ℕ, f n ≥ g n) :
  f = g := by
  sorry

end NUMINAMATH_CALUDE_surjective_injective_ge_equal_l940_94094


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l940_94077

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 3 ∧ x ≠ 1 ∧ (x / (x - 3) = (x + 1) / (x - 1)) ↔ x = -3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l940_94077


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l940_94078

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point P -/
def x_coord (m : ℝ) : ℝ := 3 - m

/-- The y-coordinate of point P -/
def y_coord (m : ℝ) : ℝ := m - 1

/-- If point P(3-m, m-1) is in the second quadrant, then m > 3 -/
theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (x_coord m) (y_coord m) → m > 3 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l940_94078


namespace NUMINAMATH_CALUDE_root_relationship_l940_94034

theorem root_relationship (a b c x y : ℝ) (ha : a ≠ 0) :
  a * x^2 + b * x + c = 0 ∧ y^2 + b * y + a * c = 0 → x = y / a := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l940_94034


namespace NUMINAMATH_CALUDE_doctor_team_formation_l940_94044

theorem doctor_team_formation (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 5)
  (h2 : female_doctors = 4)
  (h3 : team_size = 3) : 
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1 + 
   Nat.choose male_doctors 1 * Nat.choose female_doctors 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_doctor_team_formation_l940_94044


namespace NUMINAMATH_CALUDE_counterclockwise_notation_l940_94057

/-- Represents the direction of rotation -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Represents a rotation with a direction and an angle -/
structure Rotation :=
  (direction : RotationDirection)
  (angle : ℝ)

/-- Notation for a rotation -/
def rotationNotation (r : Rotation) : ℝ :=
  match r.direction with
  | RotationDirection.Clockwise => r.angle
  | RotationDirection.Counterclockwise => -r.angle

theorem counterclockwise_notation 
  (h : rotationNotation { direction := RotationDirection.Clockwise, angle := 60 } = 60) :
  rotationNotation { direction := RotationDirection.Counterclockwise, angle := 15 } = -15 :=
by
  sorry

end NUMINAMATH_CALUDE_counterclockwise_notation_l940_94057


namespace NUMINAMATH_CALUDE_complex_calculation_l940_94041

theorem complex_calculation : 
  (((3.242^2 * (16 + 8)) / (100 - (3 * 25))) + (32 - 10)^2) = 494.09014144 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_l940_94041


namespace NUMINAMATH_CALUDE_sprint_medal_theorem_l940_94075

/-- Represents the number of ways to award medals in a specific sprinting competition scenario. -/
def medalAwardingWays (totalSprinters : ℕ) (americanSprinters : ℕ) (canadianSprinters : ℕ) : ℕ :=
  -- The actual computation is not provided here
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario. -/
theorem sprint_medal_theorem :
  medalAwardingWays 10 4 3 = 552 := by
  sorry

end NUMINAMATH_CALUDE_sprint_medal_theorem_l940_94075


namespace NUMINAMATH_CALUDE_square_fencing_cost_theorem_l940_94097

/-- Represents the cost of fencing a square. -/
structure SquareFencingCost where
  totalCost : ℝ
  sideCost : ℝ

/-- The cost of fencing a square with equal side costs. -/
def fencingCost (s : SquareFencingCost) : Prop :=
  s.totalCost = 4 * s.sideCost

theorem square_fencing_cost_theorem (s : SquareFencingCost) :
  s.totalCost = 316 → fencingCost s → s.sideCost = 79 := by
  sorry

end NUMINAMATH_CALUDE_square_fencing_cost_theorem_l940_94097


namespace NUMINAMATH_CALUDE_fraction_denominator_l940_94050

theorem fraction_denominator (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (2 * y) / x + (3 * y) / 10 = 0.7 * y) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_l940_94050


namespace NUMINAMATH_CALUDE_kelly_games_given_away_l940_94014

/-- Given that Kelly initially had 50 Nintendo games and now has 35 games left,
    prove that she gave away 15 games. -/
theorem kelly_games_given_away :
  let initial_games : ℕ := 50
  let remaining_games : ℕ := 35
  let games_given_away := initial_games - remaining_games
  games_given_away = 15 :=
by sorry

end NUMINAMATH_CALUDE_kelly_games_given_away_l940_94014


namespace NUMINAMATH_CALUDE_x_plus_y_values_l940_94067

theorem x_plus_y_values (x y : ℝ) (h1 : |x| = 5) (h2 : y = Real.sqrt 9) :
  x + y = -2 ∨ x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l940_94067


namespace NUMINAMATH_CALUDE_f_composition_value_l940_94062

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else Real.sin x

theorem f_composition_value : f (f ((7 * Real.pi) / 6)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_value_l940_94062


namespace NUMINAMATH_CALUDE_convex_polygon_diagonal_inequality_l940_94058

theorem convex_polygon_diagonal_inequality (n : ℕ) (d p : ℝ) (h1 : n ≥ 3) (h2 : d > 0) (h3 : p > 0) : 
  (n : ℝ) - 3 < 2 * d / p ∧ 2 * d / p < ↑(n / 2) * ↑((n + 1) / 2) - 2 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonal_inequality_l940_94058


namespace NUMINAMATH_CALUDE_work_completion_time_l940_94065

/-- The time taken for Ganesh, Ram, and Sohan to complete a work together, given their individual work rates. -/
theorem work_completion_time 
  (ganesh_ram_rate : ℚ) -- Combined work rate of Ganesh and Ram
  (sohan_rate : ℚ)       -- Work rate of Sohan
  (h1 : ganesh_ram_rate = 1 / 24) -- Ganesh and Ram can complete the work in 24 days
  (h2 : sohan_rate = 1 / 48)      -- Sohan can complete the work in 48 days
  : (1 : ℚ) / (ganesh_ram_rate + sohan_rate) = 16 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l940_94065


namespace NUMINAMATH_CALUDE_inequality_system_solution_l940_94013

def solution_set : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3}

def satisfies_inequalities (x : ℤ) : Prop :=
  2 * x ≥ 3 * (x - 1) ∧ 2 - x / 2 < 5

theorem inequality_system_solution :
  ∀ x : ℤ, x ∈ solution_set ↔ satisfies_inequalities x :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l940_94013


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l940_94081

/-- 
Given an arithmetic progression with three consecutive terms a, a+d, a+2d,
this theorem states the conditions for d such that the squares of these terms
form a geometric progression.
-/
theorem arithmetic_to_geometric_progression (a d : ℝ) :
  (∃ r : ℝ, (a + d)^2 = a^2 * r ∧ (a + 2*d)^2 = (a + d)^2 * r) ↔ 
  (d = 0 ∨ d = a*(-2 + Real.sqrt 2) ∨ d = a*(-2 - Real.sqrt 2)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_progression_l940_94081


namespace NUMINAMATH_CALUDE_line_equation_proof_l940_94025

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using the general form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a function to check if two lines are perpendicular
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem statement
theorem line_equation_proof :
  let p : Point2D := { x := -1, y := 2 }
  let l1 : Line2D := { a := 2, b := -3, c := 4 }
  let l2 : Line2D := { a := 1, b := 4, c := -6 }
  (pointOnLine p l2) ∧ (perpendicularLines l1 l2) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l940_94025


namespace NUMINAMATH_CALUDE_fruits_problem_solution_l940_94055

def fruits_problem (x : ℕ) : Prop :=
  let last_night_apples : ℕ := 3
  let last_night_bananas : ℕ := 1
  let last_night_oranges : ℕ := 4
  let today_apples : ℕ := last_night_apples + 4
  let today_bananas : ℕ := x * last_night_bananas
  let today_oranges : ℕ := 2 * today_apples
  let total_fruits : ℕ := 39
  (last_night_apples + last_night_bananas + last_night_oranges + 
   today_apples + today_bananas + today_oranges) = total_fruits

theorem fruits_problem_solution : fruits_problem 10 := by
  sorry

end NUMINAMATH_CALUDE_fruits_problem_solution_l940_94055


namespace NUMINAMATH_CALUDE_abs_diff_roots_sum_of_cubes_l940_94086

-- Define the quadratic equation
def quadratic (x : ℝ) : ℝ := 2 * x^2 + 7 * x - 4

-- Define the roots
def x₁ : ℝ := sorry
def x₂ : ℝ := sorry

-- Axioms for the roots
axiom root₁ : quadratic x₁ = 0
axiom root₂ : quadratic x₂ = 0

-- Theorems to prove
theorem abs_diff_roots : |x₁ - x₂| = 9/2 := sorry

theorem sum_of_cubes : x₁^3 + x₂^3 = -511/8 := sorry

end NUMINAMATH_CALUDE_abs_diff_roots_sum_of_cubes_l940_94086


namespace NUMINAMATH_CALUDE_school_teachers_count_l940_94043

theorem school_teachers_count (total_people : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total_people = 3200)
  (h2 : sample_size = 160)
  (h3 : students_in_sample = 150) :
  total_people - (total_people * students_in_sample / sample_size) = 200 := by
  sorry

end NUMINAMATH_CALUDE_school_teachers_count_l940_94043


namespace NUMINAMATH_CALUDE_parallelogram_point_B_trajectory_l940_94006

-- Define the parallelogram ABCD
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define the coordinates of points A and C
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)

-- Define the line on which D moves
def D_line (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the trajectory of point B
def B_trajectory (x y : ℝ) : Prop := 3 * x - y - 20 = 0 ∧ x ≠ 3

-- Theorem statement
theorem parallelogram_point_B_trajectory 
  (ABCD : Parallelogram) 
  (h1 : ABCD.A = A) 
  (h2 : ABCD.C = C) 
  (h3 : ∀ x y, ABCD.D = (x, y) → D_line x y) :
  ∀ x y, ABCD.B = (x, y) → B_trajectory x y :=
sorry

end NUMINAMATH_CALUDE_parallelogram_point_B_trajectory_l940_94006


namespace NUMINAMATH_CALUDE_discriminant_nonnegativity_l940_94072

theorem discriminant_nonnegativity (x : ℤ) :
  x^2 * (81 - 56 * x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegativity_l940_94072


namespace NUMINAMATH_CALUDE_line_through_points_l940_94008

/-- Given a line y = ax + b passing through points (3,6) and (7,26), prove that a - b = 14 -/
theorem line_through_points (a b : ℝ) : 
  (6 : ℝ) = a * 3 + b ∧ (26 : ℝ) = a * 7 + b → a - b = 14 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l940_94008


namespace NUMINAMATH_CALUDE_fraction_simplification_l940_94059

theorem fraction_simplification :
  1 / (1 / ((1/2)^2) + 1 / ((1/2)^3) + 1 / ((1/2)^4) + 1 / ((1/2)^5)) = 1 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l940_94059


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l940_94011

theorem quadratic_inequality_solution (m n : ℝ) : 
  (∀ x : ℝ, 2*x^2 + m*x + n > 0 ↔ x > 3 ∨ x < -2) → 
  m + n = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l940_94011


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l940_94092

/-- The axis of symmetry of a parabola y = ax² + bx + c is x = -b/(2a) -/
theorem parabola_axis_of_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  ∃! x₀, ∀ x, f (x₀ + x) = f (x₀ - x) :=
by sorry

/-- The axis of symmetry of the parabola y = -1/2 x² + x - 5/2 is x = 1 -/
theorem specific_parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => -1/2 * x^2 + x - 5/2
  ∃! x₀, ∀ x, f (x₀ + x) = f (x₀ - x) ∧ x₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_specific_parabola_axis_of_symmetry_l940_94092


namespace NUMINAMATH_CALUDE_unique_digits_for_divisibility_l940_94048

-- Define the number 13xy45z as a function of x, y, z
def number (x y z : ℕ) : ℕ := 13000000 + x * 100000 + y * 10000 + 4500 + z

-- Define the divisibility condition
def is_divisible_by_792 (n : ℕ) : Prop := n % 792 = 0

-- Theorem statement
theorem unique_digits_for_divisibility :
  ∃! (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10 ∧ is_divisible_by_792 (number x y z) ∧ x = 2 ∧ y = 3 ∧ z = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_digits_for_divisibility_l940_94048


namespace NUMINAMATH_CALUDE_expression_simplification_l940_94054

theorem expression_simplification :
  (4 * 7) / (12 * 14) * ((9 * 12 * 14) / (4 * 7 * 9))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l940_94054


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l940_94099

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : ℕ
  corner_squares : ℕ
  center_half_triangles : ℕ

/-- Calculates the shaded fraction of a quilt block -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  let corner_area := q.corner_squares
  let center_area := q.center_half_triangles / 2
  (corner_area + center_area) / q.total_squares

/-- Theorem stating that the shaded fraction of the described quilt block is 3/8 -/
theorem quilt_shaded_fraction :
  let q := QuiltBlock.mk 16 4 4
  shaded_fraction q = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l940_94099
