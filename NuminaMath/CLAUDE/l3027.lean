import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3027_302775

theorem quadratic_equation_real_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x - 1 = 0 ∧ m * y^2 + 2 * y - 1 = 0) ↔ 
  (m ≥ -1 ∧ m ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3027_302775


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l3027_302717

theorem relationship_between_x_and_y (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l3027_302717


namespace NUMINAMATH_CALUDE_cube_sum_odd_numbers_l3027_302706

theorem cube_sum_odd_numbers (m : ℕ) : 
  (∃ k : ℕ, k ≥ m^2 - m + 1 ∧ k ≤ m^2 + m - 1 ∧ k = 2015) → m = 45 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_odd_numbers_l3027_302706


namespace NUMINAMATH_CALUDE_cos_sin_180_degrees_l3027_302765

theorem cos_sin_180_degrees :
  Real.cos (180 * π / 180) = -1 ∧ Real.sin (180 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_180_degrees_l3027_302765


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3027_302751

theorem bowling_ball_weight (canoe_weight : ℝ) (h1 : canoe_weight = 35) :
  let total_canoe_weight := 2 * canoe_weight
  let bowling_ball_weight := total_canoe_weight / 9
  bowling_ball_weight = 70 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3027_302751


namespace NUMINAMATH_CALUDE_hamburger_sales_average_l3027_302737

theorem hamburger_sales_average (total_hamburgers : ℕ) (days_in_week : ℕ) 
  (h1 : total_hamburgers = 63)
  (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
sorry

end NUMINAMATH_CALUDE_hamburger_sales_average_l3027_302737


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3027_302743

/-- Represents the different types of sampling methods. -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a school with different student groups. -/
structure School where
  elementaryStudents : ℕ
  juniorHighStudents : ℕ
  highSchoolStudents : ℕ

/-- Determines if there are significant differences between student groups. -/
def hasDifferences (s : School) : Prop :=
  sorry -- Definition of significant differences

/-- Determines the most appropriate sampling method given a school and sample size. -/
def mostAppropriateSamplingMethod (s : School) (sampleSize : ℕ) : SamplingMethod :=
  sorry -- Definition of most appropriate sampling method

/-- Theorem stating that stratified sampling is the most appropriate method for the given conditions. -/
theorem stratified_sampling_most_appropriate (s : School) (sampleSize : ℕ) :
  s.elementaryStudents = 125 →
  s.juniorHighStudents = 280 →
  s.highSchoolStudents = 95 →
  sampleSize = 100 →
  hasDifferences s →
  mostAppropriateSamplingMethod s sampleSize = SamplingMethod.Stratified :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l3027_302743


namespace NUMINAMATH_CALUDE_b_equals_two_l3027_302755

theorem b_equals_two (x y z a b : ℝ) 
  (eq1 : x + y = 2)
  (eq2 : x * y - z^2 = a)
  (eq3 : b = x + y + z) :
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_b_equals_two_l3027_302755


namespace NUMINAMATH_CALUDE_min_value_of_f_l3027_302760

/-- The function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem: x = 3 minimizes the function f(x) = 3x^2 - 18x + 7 -/
theorem min_value_of_f :
  ∀ x : ℝ, f 3 ≤ f x :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3027_302760


namespace NUMINAMATH_CALUDE_smallest_number_with_distinct_sums_ending_in_two_l3027_302734

def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def digitSumSequence (x : Nat) : List Nat :=
  [x, sumOfDigits x, sumOfDigits (sumOfDigits x), sumOfDigits (sumOfDigits (sumOfDigits x))]

theorem smallest_number_with_distinct_sums_ending_in_two :
  ∀ y : Nat, y < 2999 →
    ¬(List.Pairwise (· ≠ ·) (digitSumSequence y) ∧
      (digitSumSequence y).getLast? = some 2) ∧
    (List.Pairwise (· ≠ ·) (digitSumSequence 2999) ∧
     (digitSumSequence 2999).getLast? = some 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_distinct_sums_ending_in_two_l3027_302734


namespace NUMINAMATH_CALUDE_number_of_hens_l3027_302799

def number_of_goats : ℕ := 5
def total_cost : ℕ := 2500
def price_of_hen : ℕ := 50
def price_of_goat : ℕ := 400

theorem number_of_hens : 
  ∃ (h : ℕ), h * price_of_hen + number_of_goats * price_of_goat = total_cost ∧ h = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_hens_l3027_302799


namespace NUMINAMATH_CALUDE_coffee_order_total_cost_l3027_302725

/-- The total cost of a coffee order -/
def coffee_order_cost (drip_coffee_price : ℝ) (drip_coffee_quantity : ℕ)
                      (espresso_price : ℝ) (espresso_quantity : ℕ)
                      (latte_price : ℝ) (latte_quantity : ℕ)
                      (vanilla_syrup_price : ℝ) (vanilla_syrup_quantity : ℕ)
                      (cold_brew_price : ℝ) (cold_brew_quantity : ℕ)
                      (cappuccino_price : ℝ) (cappuccino_quantity : ℕ) : ℝ :=
  drip_coffee_price * drip_coffee_quantity +
  espresso_price * espresso_quantity +
  latte_price * latte_quantity +
  vanilla_syrup_price * vanilla_syrup_quantity +
  cold_brew_price * cold_brew_quantity +
  cappuccino_price * cappuccino_quantity

/-- The theorem stating that the given coffee order costs $25.00 -/
theorem coffee_order_total_cost :
  coffee_order_cost 2.25 2 3.50 1 4.00 2 0.50 1 2.50 2 3.50 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_coffee_order_total_cost_l3027_302725


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3027_302750

theorem election_votes_theorem (V : ℕ) (W L : ℕ) : 
  W + L = V →  -- Total votes
  W - L = V / 10 →  -- Initial margin
  (L + 1500) - (W - 1500) = V / 10 →  -- New margin after vote change
  V = 30000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3027_302750


namespace NUMINAMATH_CALUDE_function_range_and_triangle_area_l3027_302752

noncomputable def f (x : ℝ) := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem function_range_and_triangle_area 
  (A B C : ℝ) (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 3), f x ∈ Set.Icc 0 (Real.sqrt 3)) ∧
  (f (A / 2) = Real.sqrt 3 / 2) ∧
  (a = 4) ∧
  (b + c = 5) →
  (Set.range (fun x => f x) = Set.Icc 0 (Real.sqrt 3)) ∧
  (1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_function_range_and_triangle_area_l3027_302752


namespace NUMINAMATH_CALUDE_digit_count_proof_l3027_302769

/-- The number of valid digits for each position after the first -/
def valid_digits : ℕ := 4

/-- The number of valid digits for the first position -/
def valid_first_digits : ℕ := 3

/-- The total count of numbers with the given properties -/
def total_count : ℕ := 192

/-- The number of digits in the numbers -/
def n : ℕ := 4

theorem digit_count_proof :
  valid_first_digits * valid_digits^(n - 1) = total_count :=
sorry

end NUMINAMATH_CALUDE_digit_count_proof_l3027_302769


namespace NUMINAMATH_CALUDE_f_convex_when_a_negative_a_range_when_f_bounded_l3027_302733

-- Define the function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Define convexity condition
def is_convex (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≥ (f x₁ + f x₂) / 2

-- Theorem 1: f is convex when a < 0
theorem f_convex_when_a_negative (a : ℝ) (h : a < 0) : is_convex (f a) := by
  sorry

-- Theorem 2: Range of a when |f(x)| ≤ 1 for x ∈ [0, 1]
theorem a_range_when_f_bounded (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1) → -2 ≤ a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_convex_when_a_negative_a_range_when_f_bounded_l3027_302733


namespace NUMINAMATH_CALUDE_range_of_f_is_real_l3027_302783

noncomputable def f (x : ℝ) := x^3 - 3*x

theorem range_of_f_is_real : 
  (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
  (f 2 = 2) ∧ 
  (deriv f 2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_is_real_l3027_302783


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3027_302761

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x| ≤ 1} = Set.Icc (-1) 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3027_302761


namespace NUMINAMATH_CALUDE_art_club_teams_l3027_302735

theorem art_club_teams (n : ℕ) (h : n.choose 2 = 15) :
  n.choose 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_art_club_teams_l3027_302735


namespace NUMINAMATH_CALUDE_problem_solution_l3027_302702

theorem problem_solution (x : ℚ) : 5 * x - 8 = 12 * x + 15 → 5 * (x + 4) = 25 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3027_302702


namespace NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3027_302711

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), x_max = 7 ∧
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → x ≤ x_max) ∧
  (x_max^2 - 12*x_max + 35 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_value_quadratic_inequality_l3027_302711


namespace NUMINAMATH_CALUDE_circle_tangent_to_y_axis_equation_l3027_302726

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle is tangent to the y-axis --/
def is_tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius

/-- The equation of a circle --/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_tangent_to_y_axis_equation 
  (c : Circle) 
  (h1 : c.center = (1, 2)) 
  (h2 : is_tangent_to_y_axis c) : 
  ∀ x y : ℝ, circle_equation c x y ↔ (x - 1)^2 + (y - 2)^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_y_axis_equation_l3027_302726


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l3027_302777

theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) 
  (hπ : h = 6 * Real.pi) (wπ : w = 4 * Real.pi) :
  ∃ (r : ℝ), (r = 3 ∨ r = 2) ∧ 
    (2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
     2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l3027_302777


namespace NUMINAMATH_CALUDE_real_part_of_reciprocal_l3027_302722

theorem real_part_of_reciprocal (z : ℂ) : 
  z ≠ (1 : ℂ) →
  Complex.abs z = 1 → 
  z = Complex.exp (Complex.I * Real.pi / 3) →
  Complex.re (1 / (1 - z)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_reciprocal_l3027_302722


namespace NUMINAMATH_CALUDE_jeans_price_markup_l3027_302715

theorem jeans_price_markup (cost : ℝ) (h : cost > 0) :
  let retailer_price := cost * 1.4
  let customer_price := retailer_price * 1.1
  (customer_price - cost) / cost = 0.54 := by
sorry

end NUMINAMATH_CALUDE_jeans_price_markup_l3027_302715


namespace NUMINAMATH_CALUDE_brians_breath_holding_factor_l3027_302727

/-- Given Brian's breath-holding practice over three weeks, prove the factor of increase after the first week. -/
theorem brians_breath_holding_factor
  (initial_time : ℝ)
  (final_time : ℝ)
  (h_initial : initial_time = 10)
  (h_final : final_time = 60)
  (F : ℝ)
  (h_week2 : F * initial_time * 2 = F * initial_time * 2)
  (h_week3 : F * initial_time * 2 * 1.5 = final_time) :
  F = 2 := by
  sorry

end NUMINAMATH_CALUDE_brians_breath_holding_factor_l3027_302727


namespace NUMINAMATH_CALUDE_andrew_total_hours_l3027_302786

/-- Andrew's work on his Science report -/
def andrew_work : ℝ → ℝ → ℝ := fun days hours_per_day => days * hours_per_day

/-- The theorem stating the total hours Andrew worked -/
theorem andrew_total_hours : andrew_work 3 2.5 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_andrew_total_hours_l3027_302786


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3027_302789

theorem arithmetic_sequence_common_difference
  (a : ℕ+ → ℝ)
  (h : ∀ n : ℕ+, a n + a (n + 1) = 4 * n)
  : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ+, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3027_302789


namespace NUMINAMATH_CALUDE_class_ratio_theorem_l3027_302704

theorem class_ratio_theorem (boys girls : ℕ) (h : boys * 7 = girls * 8) :
  -- 1. The number of girls is 7/8 of the number of boys
  (girls : ℚ) / boys = 7 / 8 ∧
  -- 2. The number of boys accounts for 8/15 of the total number of students
  (boys : ℚ) / (boys + girls) = 8 / 15 ∧
  -- 3. The number of girls accounts for 7/15 of the total number of students
  (girls : ℚ) / (boys + girls) = 7 / 15 ∧
  -- 4. If there are 45 students in total, there are 24 boys
  (boys + girls = 45 → boys = 24) :=
by sorry

end NUMINAMATH_CALUDE_class_ratio_theorem_l3027_302704


namespace NUMINAMATH_CALUDE_divisors_of_27n_cubed_l3027_302784

/-- The number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_27n_cubed (n : ℕ) (h_odd : Odd n) (h_divisors : num_divisors n = 12) :
  num_divisors (27 * n^3) = 256 := by sorry

end NUMINAMATH_CALUDE_divisors_of_27n_cubed_l3027_302784


namespace NUMINAMATH_CALUDE_cos_2alpha_value_l3027_302767

theorem cos_2alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos (2 * α) = -Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_value_l3027_302767


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3027_302700

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  let y := x * Real.sqrt 2
  (((x^2 + y^2) * (4 * x^2 + y^2)).sqrt) / (x * y) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l3027_302700


namespace NUMINAMATH_CALUDE_unique_prime_cube_plus_two_l3027_302716

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The main theorem stating that there is exactly one positive integer n ≥ 2 
    such that n^3 + 2 is prime -/
theorem unique_prime_cube_plus_two :
  ∃! (n : ℕ), n ≥ 2 ∧ isPrime (n^3 + 2) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_cube_plus_two_l3027_302716


namespace NUMINAMATH_CALUDE_abs_m_eq_abs_half_implies_m_eq_plus_minus_half_l3027_302792

theorem abs_m_eq_abs_half_implies_m_eq_plus_minus_half (m : ℝ) : 
  |(-m)| = |(-1/2)| → m = -1/2 ∨ m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_abs_m_eq_abs_half_implies_m_eq_plus_minus_half_l3027_302792


namespace NUMINAMATH_CALUDE_anna_quiz_goal_impossible_l3027_302763

theorem anna_quiz_goal_impossible (total_quizzes : Nat) (goal_percentage : Rat) 
  (completed_quizzes : Nat) (completed_as : Nat) : 
  total_quizzes = 60 →
  goal_percentage = 85 / 100 →
  completed_quizzes = 40 →
  completed_as = 30 →
  ¬∃ (remaining_as : Nat), 
    (completed_as + remaining_as : Rat) / total_quizzes ≥ goal_percentage ∧ 
    remaining_as ≤ total_quizzes - completed_quizzes :=
by sorry

end NUMINAMATH_CALUDE_anna_quiz_goal_impossible_l3027_302763


namespace NUMINAMATH_CALUDE_expression_calculation_l3027_302740

theorem expression_calculation : 
  (0.86 : ℝ)^3 - (0.1 : ℝ)^3 / (0.86 : ℝ)^2 + 0.086 + (0.1 : ℝ)^2 = 0.730704 := by
  sorry

end NUMINAMATH_CALUDE_expression_calculation_l3027_302740


namespace NUMINAMATH_CALUDE_office_age_problem_l3027_302796

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℝ) (group1_persons : ℕ) (avg_age_group1 : ℝ) (group2_persons : ℕ) (person15_age : ℝ) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_persons = 5 →
  avg_age_group1 = 14 →
  group2_persons = 9 →
  person15_age = 56 →
  (total_persons * avg_age_all - group1_persons * avg_age_group1 - person15_age) / group2_persons = 16 := by
  sorry

end NUMINAMATH_CALUDE_office_age_problem_l3027_302796


namespace NUMINAMATH_CALUDE_garden_max_area_exists_max_area_garden_l3027_302723

/-- Represents a rectangular garden with fencing on three sides --/
structure Garden where
  width : ℝ
  length : ℝ
  fencing : ℝ
  fence_constraint : fencing = 2 * width + length

/-- The area of a rectangular garden --/
def Garden.area (g : Garden) : ℝ := g.width * g.length

/-- The maximum possible area of a garden with 400 feet of fencing --/
def max_garden_area : ℝ := 20000

/-- Theorem stating that the maximum area of a garden with 400 feet of fencing is 20000 square feet --/
theorem garden_max_area :
  ∀ g : Garden, g.fencing = 400 → g.area ≤ max_garden_area :=
by
  sorry

/-- Theorem stating that there exists a garden configuration achieving the maximum area --/
theorem exists_max_area_garden :
  ∃ g : Garden, g.fencing = 400 ∧ g.area = max_garden_area :=
by
  sorry

end NUMINAMATH_CALUDE_garden_max_area_exists_max_area_garden_l3027_302723


namespace NUMINAMATH_CALUDE_choir_meeting_interval_l3027_302793

/-- The number of days between drama club meetings -/
def drama_interval : ℕ := 3

/-- The number of days until the next joint meeting -/
def next_joint_meeting : ℕ := 15

/-- The number of days between choir meetings -/
def choir_interval : ℕ := 5

theorem choir_meeting_interval :
  (next_joint_meeting % drama_interval = 0) ∧
  (next_joint_meeting % choir_interval = 0) ∧
  (∀ x : ℕ, x > 1 ∧ x < choir_interval →
    ¬(next_joint_meeting % x = 0 ∧ next_joint_meeting % drama_interval = 0)) :=
by sorry

end NUMINAMATH_CALUDE_choir_meeting_interval_l3027_302793


namespace NUMINAMATH_CALUDE_square_triangle_equal_area_l3027_302798

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (triangle_base : ℝ) :
  square_perimeter = 64 →
  triangle_height = 64 →
  (square_perimeter / 4)^2 = (1/2) * triangle_height * triangle_base →
  triangle_base = 8 := by
sorry

end NUMINAMATH_CALUDE_square_triangle_equal_area_l3027_302798


namespace NUMINAMATH_CALUDE_rainfall_rate_l3027_302718

/-- Rainfall problem statement -/
theorem rainfall_rate (monday_hours monday_rate tuesday_hours wednesday_hours total_rainfall : ℝ) 
  (h1 : monday_hours = 7)
  (h2 : monday_rate = 1)
  (h3 : tuesday_hours = 4)
  (h4 : wednesday_hours = 2)
  (h5 : total_rainfall = 23)
  : ∃ tuesday_rate : ℝ, 
    monday_hours * monday_rate + tuesday_hours * tuesday_rate + wednesday_hours * (2 * tuesday_rate) = total_rainfall ∧ 
    tuesday_rate = 2 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_rate_l3027_302718


namespace NUMINAMATH_CALUDE_erasers_remaining_l3027_302742

/-- The number of erasers left in a box after some are removed -/
def erasers_left (initial : ℕ) (removed : ℕ) : ℕ := initial - removed

/-- Theorem: Given 69 initial erasers and 54 removed, 15 erasers are left -/
theorem erasers_remaining : erasers_left 69 54 = 15 := by
  sorry

end NUMINAMATH_CALUDE_erasers_remaining_l3027_302742


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3027_302730

theorem arithmetic_geometric_sequence (a b : ℝ) : 
  (2 * a = 1 + b) →  -- arithmetic sequence condition
  (b^2 = a) →        -- geometric sequence condition
  (a ≠ b) →          -- given condition
  (a = 1/4) :=       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3027_302730


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l3027_302744

theorem cubic_sum_problem (a b c : ℝ) 
  (sum_condition : a + b + c = 7)
  (product_sum_condition : a * b + a * c + b * c = 9)
  (product_condition : a * b * c = -18) :
  a^3 + b^3 + c^3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l3027_302744


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l3027_302781

/-- Given a ratio of blue to white paint and an amount of white paint, 
    calculate the amount of blue paint required. -/
def blue_paint_amount (blue_ratio : ℚ) (white_ratio : ℚ) (white_amount : ℚ) : ℚ :=
  (blue_ratio / white_ratio) * white_amount

/-- Theorem stating that given the specific ratio and white paint amount, 
    the blue paint amount is 12 quarts. -/
theorem blue_paint_calculation :
  let blue_ratio : ℚ := 4
  let white_ratio : ℚ := 5
  let white_amount : ℚ := 15
  blue_paint_amount blue_ratio white_ratio white_amount = 12 := by
sorry

#eval blue_paint_amount 4 5 15

end NUMINAMATH_CALUDE_blue_paint_calculation_l3027_302781


namespace NUMINAMATH_CALUDE_store_fruit_cost_l3027_302705

/-- The cost of fruit in a store -/
structure FruitCost where
  banana_to_apple : ℚ  -- Ratio of banana cost to apple cost
  apple_to_orange : ℚ  -- Ratio of apple cost to orange cost

/-- Given the cost ratios, calculate how many oranges cost the same as a given number of bananas -/
def bananas_to_oranges (cost : FruitCost) (num_bananas : ℕ) : ℚ :=
  (num_bananas : ℚ) * cost.apple_to_orange * cost.banana_to_apple

theorem store_fruit_cost (cost : FruitCost) 
  (h1 : cost.banana_to_apple = 3 / 4)
  (h2 : cost.apple_to_orange = 5 / 7) :
  bananas_to_oranges cost 28 = 15 := by
  sorry

end NUMINAMATH_CALUDE_store_fruit_cost_l3027_302705


namespace NUMINAMATH_CALUDE_one_statement_implies_negation_l3027_302719

theorem one_statement_implies_negation (p q r : Prop) : 
  let statement1 := p ∧ q ∧ ¬r
  let statement2 := p ∧ ¬q ∧ r
  let statement3 := ¬p ∧ q ∧ ¬r
  let statement4 := ¬p ∧ ¬q ∧ r
  let negation := ¬((p ∧ q) ∨ r)
  ∃! x : Fin 4, match x with
    | 0 => statement1 → negation
    | 1 => statement2 → negation
    | 2 => statement3 → negation
    | 3 => statement4 → negation
  := by sorry

end NUMINAMATH_CALUDE_one_statement_implies_negation_l3027_302719


namespace NUMINAMATH_CALUDE_missing_number_proof_l3027_302773

theorem missing_number_proof (x : ℤ) : (4 + 3) + (8 - x - 1) = 11 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3027_302773


namespace NUMINAMATH_CALUDE_square_perimeter_l3027_302729

theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s ^ 2 = 625) : 4 * s = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3027_302729


namespace NUMINAMATH_CALUDE_harriet_driving_speed_l3027_302753

/-- Harriet's driving problem -/
theorem harriet_driving_speed 
  (total_time : ℝ) 
  (time_to_b : ℝ) 
  (speed_back : ℝ) : 
  total_time = 5 → 
  time_to_b = 192 / 60 → 
  speed_back = 160 → 
  (total_time - time_to_b) * speed_back / time_to_b = 90 := by
  sorry

end NUMINAMATH_CALUDE_harriet_driving_speed_l3027_302753


namespace NUMINAMATH_CALUDE_expression_values_l3027_302731

theorem expression_values (a b : ℝ) : 
  (∀ x : ℝ, |a| ≤ |x|) → (b * b = 1) → 
  (|a - 2| - b^2023 = 1 ∨ |a - 2| - b^2023 = 3) := by
  sorry

end NUMINAMATH_CALUDE_expression_values_l3027_302731


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l3027_302748

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A set of line segments can form a triangle if they satisfy the triangle inequality. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem set_b_forms_triangle :
  can_form_triangle 8 6 3 := by
  sorry

end NUMINAMATH_CALUDE_set_b_forms_triangle_l3027_302748


namespace NUMINAMATH_CALUDE_batting_cage_pitches_per_token_l3027_302703

/-- The number of pitches per token at a batting cage -/
def pitches_per_token : ℕ := 15

/-- Macy's number of tokens -/
def macy_tokens : ℕ := 11

/-- Piper's number of tokens -/
def piper_tokens : ℕ := 17

/-- Macy's number of hits -/
def macy_hits : ℕ := 50

/-- Piper's number of hits -/
def piper_hits : ℕ := 55

/-- Total number of missed pitches -/
def total_misses : ℕ := 315

theorem batting_cage_pitches_per_token :
  (macy_tokens + piper_tokens) * pitches_per_token =
  macy_hits + piper_hits + total_misses :=
by sorry

end NUMINAMATH_CALUDE_batting_cage_pitches_per_token_l3027_302703


namespace NUMINAMATH_CALUDE_both_samples_stratified_l3027_302738

/-- Represents a sample of students -/
structure Sample :=
  (numbers : List Nat)

/-- Represents the school population -/
structure School :=
  (total_students : Nat)
  (first_grade : Nat)
  (second_grade : Nat)
  (third_grade : Nat)

/-- Checks if a sample is valid for stratified sampling -/
def is_valid_stratified_sample (school : School) (sample : Sample) : Prop :=
  sample.numbers.length = 10 ∧
  sample.numbers.all (λ n => n > 0 ∧ n ≤ school.total_students) ∧
  sample.numbers.Nodup

/-- The given school configuration -/
def junior_high : School :=
  { total_students := 300
  , first_grade := 120
  , second_grade := 90
  , third_grade := 90 }

/-- Sample ① -/
def sample1 : Sample :=
  { numbers := [7, 37, 67, 97, 127, 157, 187, 217, 247, 277] }

/-- Sample ③ -/
def sample3 : Sample :=
  { numbers := [11, 41, 71, 101, 131, 161, 191, 221, 251, 281] }

theorem both_samples_stratified :
  is_valid_stratified_sample junior_high sample1 ∧
  is_valid_stratified_sample junior_high sample3 := by
  sorry

end NUMINAMATH_CALUDE_both_samples_stratified_l3027_302738


namespace NUMINAMATH_CALUDE_smallest_circle_area_l3027_302741

theorem smallest_circle_area (p1 p2 : ℝ × ℝ) (h : p1 = (-3, -2) ∧ p2 = (2, 4)) :
  let d := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let r := d / 2
  let A := π * r^2
  A = (61 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_circle_area_l3027_302741


namespace NUMINAMATH_CALUDE_combinations_to_arrangements_l3027_302780

theorem combinations_to_arrangements (n : ℕ) (h1 : n ≥ 2) (h2 : Nat.choose n 2 = 15) :
  (n.factorial / (n - 2).factorial) = 30 := by
  sorry

end NUMINAMATH_CALUDE_combinations_to_arrangements_l3027_302780


namespace NUMINAMATH_CALUDE_x_range_for_inequality_l3027_302764

theorem x_range_for_inequality (x t : ℝ) :
  (t ∈ Set.Icc 1 3) →
  (((1/8) * (2*x - x^2) ≤ t^2 - 3*t + 2) ∧ (t^2 - 3*t + 2 ≤ 3 - x^2)) →
  (x ∈ Set.Icc (-1) (1 - Real.sqrt 3)) := by
sorry

end NUMINAMATH_CALUDE_x_range_for_inequality_l3027_302764


namespace NUMINAMATH_CALUDE_room_length_with_veranda_l3027_302757

/-- Represents a rectangular room with a surrounding veranda -/
structure RoomWithVeranda where
  roomLength : ℝ
  roomWidth : ℝ
  verandaWidth : ℝ

/-- Calculates the area of the veranda -/
def verandaArea (r : RoomWithVeranda) : ℝ :=
  (r.roomLength + 2 * r.verandaWidth) * (r.roomWidth + 2 * r.verandaWidth) - r.roomLength * r.roomWidth

theorem room_length_with_veranda (r : RoomWithVeranda) :
  r.roomWidth = 12 ∧ r.verandaWidth = 2 ∧ verandaArea r = 144 → r.roomLength = 20 := by
  sorry

end NUMINAMATH_CALUDE_room_length_with_veranda_l3027_302757


namespace NUMINAMATH_CALUDE_weekly_calorie_allowance_is_11700_l3027_302701

/-- Represents the weekly calorie allowance calculation for a person in their 60's --/
def weekly_calorie_allowance : ℕ :=
  let average_daily_allowance : ℕ := 2000
  let daily_reduction : ℕ := 500
  let reduced_daily_allowance : ℕ := average_daily_allowance - daily_reduction
  let intense_workout_days : ℕ := 2
  let moderate_exercise_days : ℕ := 3
  let rest_days : ℕ := 2
  let intense_workout_extra_calories : ℕ := 300
  let moderate_exercise_extra_calories : ℕ := 200
  
  (reduced_daily_allowance + intense_workout_extra_calories) * intense_workout_days +
  (reduced_daily_allowance + moderate_exercise_extra_calories) * moderate_exercise_days +
  reduced_daily_allowance * rest_days

/-- Theorem stating that the weekly calorie allowance is 11700 calories --/
theorem weekly_calorie_allowance_is_11700 : 
  weekly_calorie_allowance = 11700 := by
  sorry

end NUMINAMATH_CALUDE_weekly_calorie_allowance_is_11700_l3027_302701


namespace NUMINAMATH_CALUDE_acid_dilution_l3027_302782

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% acid solution yields a 25% acid solution -/
theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 →
  initial_concentration = 0.4 →
  added_water = 30 →
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration := by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l3027_302782


namespace NUMINAMATH_CALUDE_bulls_win_in_seven_l3027_302758

/-- The probability of the Knicks winning a single game -/
def p_knicks_win : ℚ := 3/4

/-- The probability of the Bulls winning a single game -/
def p_bulls_win : ℚ := 1 - p_knicks_win

/-- The number of games needed to win the series -/
def games_to_win : ℕ := 4

/-- The total number of games played when the series goes to 7 games -/
def total_games : ℕ := 7

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose_3_of_6 : ℕ := 20

theorem bulls_win_in_seven (
  p_knicks_win : ℚ) 
  (p_bulls_win : ℚ) 
  (games_to_win : ℕ) 
  (total_games : ℕ) 
  (ways_to_choose_3_of_6 : ℕ) :
  p_knicks_win = 3/4 →
  p_bulls_win = 1 - p_knicks_win →
  games_to_win = 4 →
  total_games = 7 →
  ways_to_choose_3_of_6 = 20 →
  (ways_to_choose_3_of_6 : ℚ) * p_bulls_win^3 * p_knicks_win^3 * p_bulls_win = 540/16384 :=
by sorry

end NUMINAMATH_CALUDE_bulls_win_in_seven_l3027_302758


namespace NUMINAMATH_CALUDE_square_perimeter_relation_l3027_302795

/-- Given two squares A and B, where A has a perimeter of 40 cm and B has an area
    equal to one-third the area of A, the perimeter of B is (40√3)/3 cm. -/
theorem square_perimeter_relation (A B : Real → Real → Prop) :
  (∃ s, A s s ∧ 4 * s = 40) →  -- Square A has perimeter 40 cm
  (∀ x y, B x y ↔ x = y ∧ x^2 = (1/3) * s^2) →  -- B's area is 1/3 of A's area
  (∃ p, ∀ x y, B x y → 4 * x = p ∧ p = (40 * Real.sqrt 3) / 3) :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_relation_l3027_302795


namespace NUMINAMATH_CALUDE_min_sum_of_primes_for_99_consecutive_sum_l3027_302714

/-- The sum of 99 consecutive natural numbers -/
def sum_99_consecutive (x : ℕ) : ℕ := 99 * x

/-- Predicate to check if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem min_sum_of_primes_for_99_consecutive_sum :
  ∃ (a b c d : ℕ), 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧
    (∃ x : ℕ, sum_99_consecutive x = a * b * c * d) ∧
    (∀ a' b' c' d' : ℕ, 
      is_prime a' ∧ is_prime b' ∧ is_prime c' ∧ is_prime d' ∧
      (∃ x : ℕ, sum_99_consecutive x = a' * b' * c' * d') →
      a + b + c + d ≤ a' + b' + c' + d') ∧
    a + b + c + d = 70 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_for_99_consecutive_sum_l3027_302714


namespace NUMINAMATH_CALUDE_amaro_roses_l3027_302759

theorem amaro_roses :
  ∀ (total_roses : ℕ),
  (3 * total_roses / 4 : ℚ) + (3 * total_roses / 16 : ℚ) = 75 →
  total_roses = 80 := by
sorry

end NUMINAMATH_CALUDE_amaro_roses_l3027_302759


namespace NUMINAMATH_CALUDE_nancy_work_hours_nancy_specific_case_l3027_302724

/-- Given Nancy's earnings and work hours, calculate the number of hours needed to earn a target amount -/
theorem nancy_work_hours (earnings : ℝ) (work_hours : ℝ) (target_amount : ℝ) :
  earnings > 0 ∧ work_hours > 0 ∧ target_amount > 0 →
  let hourly_rate := earnings / work_hours
  (target_amount / hourly_rate) = (target_amount * work_hours) / earnings :=
by sorry

/-- Nancy's specific work scenario -/
theorem nancy_specific_case :
  let earnings := 28
  let work_hours := 4
  let target_amount := 70
  let hours_needed := (target_amount * work_hours) / earnings
  hours_needed = 10 :=
by sorry

end NUMINAMATH_CALUDE_nancy_work_hours_nancy_specific_case_l3027_302724


namespace NUMINAMATH_CALUDE_largest_valid_n_l3027_302794

def is_valid_n (n : ℕ) : Prop :=
  ∃ (P : ℤ → ℤ), ∀ (k : ℕ), (2020 ∣ P^[k] 0) ↔ (n ∣ k)

theorem largest_valid_n : 
  (∃ (N : ℕ), N ∈ Finset.range 2020 ∧ is_valid_n N ∧ 
    ∀ (M : ℕ), M ∈ Finset.range 2020 → is_valid_n M → M ≤ N) ∧
  (∀ (N : ℕ), N ∈ Finset.range 2020 ∧ is_valid_n N ∧ 
    (∀ (M : ℕ), M ∈ Finset.range 2020 → is_valid_n M → M ≤ N) → N = 1980) :=
by sorry


end NUMINAMATH_CALUDE_largest_valid_n_l3027_302794


namespace NUMINAMATH_CALUDE_max_difference_consecutive_means_l3027_302713

theorem max_difference_consecutive_means (a b : ℕ) : 
  0 < a ∧ 0 < b ∧ a < 1000 ∧ b < 1000 →
  ∃ (k : ℕ), (a + b) / 2 = 2 * k + 1 ∧ Real.sqrt (a * b) = 2 * k - 1 →
  a - b ≤ 62 := by
sorry

end NUMINAMATH_CALUDE_max_difference_consecutive_means_l3027_302713


namespace NUMINAMATH_CALUDE_custom_op_solution_l3027_302774

/-- The custom operation ※ -/
def custom_op (a b : ℕ) : ℕ := (b * (2 * a + b - 1)) / 2

theorem custom_op_solution :
  ∀ a : ℕ, custom_op a 15 = 165 → a = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l3027_302774


namespace NUMINAMATH_CALUDE_four_door_room_ways_l3027_302779

/-- The number of ways to enter or exit a room with a given number of doors. -/
def waysToEnterOrExit (numDoors : ℕ) : ℕ := numDoors

/-- The number of different ways to enter and exit a room with a given number of doors. -/
def totalWays (numDoors : ℕ) : ℕ :=
  (waysToEnterOrExit numDoors) * (waysToEnterOrExit numDoors)

/-- Theorem: In a room with four doors, there are 16 different ways to enter and exit. -/
theorem four_door_room_ways :
  totalWays 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_four_door_room_ways_l3027_302779


namespace NUMINAMATH_CALUDE_divisibility_n_plus_seven_l3027_302745

theorem divisibility_n_plus_seven (n : ℕ+) : n ∣ n + 7 ↔ n = 1 ∨ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_n_plus_seven_l3027_302745


namespace NUMINAMATH_CALUDE_rectangle_longer_side_l3027_302791

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    prove that the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ)
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : circle_area = Real.pi * circle_radius ^ 2)
  (h4 : rectangle_area = 2 * circle_radius * longer_side) :
  longer_side = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_l3027_302791


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3027_302746

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → (abs m < 1) ∧
  ¬(∀ m : ℝ, (∀ x : ℝ, x^2 - m*x + 1 ≠ 0) → abs m < 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3027_302746


namespace NUMINAMATH_CALUDE_cyclists_meeting_time_l3027_302720

/-- Two cyclists meeting problem -/
theorem cyclists_meeting_time
  (b : ℝ) -- distance between towns A and B in km
  (peter_speed : ℝ) -- Peter's speed in km/h
  (john_speed : ℝ) -- John's speed in km/h
  (h1 : peter_speed = 7) -- Peter's speed is 7 km/h
  (h2 : john_speed = 5) -- John's speed is 5 km/h
  : ∃ p : ℝ, p = b / (peter_speed + john_speed) ∧ p = b / 12 :=
by sorry

end NUMINAMATH_CALUDE_cyclists_meeting_time_l3027_302720


namespace NUMINAMATH_CALUDE_painted_cells_theorem_l3027_302710

/-- Represents a rectangular grid with painted and unpainted cells. -/
structure PaintedRectangle where
  rows : Nat
  cols : Nat
  painted_cells : Nat

/-- Checks if the given PaintedRectangle satisfies the problem conditions. -/
def is_valid_painting (rect : PaintedRectangle) : Prop :=
  ∃ k l : Nat,
    rect.rows = 2 * k + 1 ∧
    rect.cols = 2 * l + 1 ∧
    k * l = 74 ∧
    rect.painted_cells = (2 * k + 1) * (2 * l + 1) - 74

/-- The main theorem stating the only possible numbers of painted cells. -/
theorem painted_cells_theorem :
  ∀ rect : PaintedRectangle,
    is_valid_painting rect →
    (rect.painted_cells = 373 ∨ rect.painted_cells = 301) :=
by sorry

end NUMINAMATH_CALUDE_painted_cells_theorem_l3027_302710


namespace NUMINAMATH_CALUDE_four_at_three_equals_thirty_l3027_302707

-- Define the operation @
def at_op (a b : ℤ) : ℤ := 3 * a^2 - 2 * b^2

-- Theorem statement
theorem four_at_three_equals_thirty : at_op 4 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_four_at_three_equals_thirty_l3027_302707


namespace NUMINAMATH_CALUDE_paint_theorem_l3027_302797

def paint_problem (initial_amount : ℚ) : Prop :=
  let first_day_remaining := initial_amount - (1/2 * initial_amount)
  let second_day_remaining := first_day_remaining - (1/4 * first_day_remaining)
  let third_day_remaining := second_day_remaining - (1/3 * second_day_remaining)
  third_day_remaining = 1/4 * initial_amount

theorem paint_theorem : paint_problem 1 := by
  sorry

end NUMINAMATH_CALUDE_paint_theorem_l3027_302797


namespace NUMINAMATH_CALUDE_bicycle_cost_calculation_l3027_302776

/-- Given two bicycles sold at a certain price, with specified profit and loss percentages,
    calculate the total cost of both bicycles. -/
theorem bicycle_cost_calculation 
  (selling_price : ℚ) 
  (profit_percent : ℚ) 
  (loss_percent : ℚ) : 
  selling_price = 990 →
  profit_percent = 10 / 100 →
  loss_percent = 10 / 100 →
  ∃ (cost1 cost2 : ℚ),
    cost1 * (1 + profit_percent) = selling_price ∧
    cost2 * (1 - loss_percent) = selling_price ∧
    cost1 + cost2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_cost_calculation_l3027_302776


namespace NUMINAMATH_CALUDE_complex_z_and_magnitude_l3027_302785

def complex_number (a b : ℝ) : ℂ := Complex.mk a b

theorem complex_z_and_magnitude : 
  let i : ℂ := complex_number 0 1
  let z : ℂ := (1 - i) / (1 + i) + 2*i
  (z = i) ∧ (Complex.abs z = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_z_and_magnitude_l3027_302785


namespace NUMINAMATH_CALUDE_reflection_squared_is_identity_l3027_302754

-- Define a reflection matrix over a non-zero vector
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

-- Theorem: The square of a reflection matrix is the identity matrix
theorem reflection_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = !![1, 0; 0, 1] :=
sorry

end NUMINAMATH_CALUDE_reflection_squared_is_identity_l3027_302754


namespace NUMINAMATH_CALUDE_fraction_equality_l3027_302770

theorem fraction_equality : (900 ^ 2 : ℝ) / (306 ^ 2 - 294 ^ 2) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3027_302770


namespace NUMINAMATH_CALUDE_calories_per_chip_l3027_302756

/-- Represents the number of chips in a bag -/
def chips_per_bag : ℕ := 24

/-- Represents the cost of a bag in dollars -/
def cost_per_bag : ℚ := 2

/-- Represents the total calories Peter wants to consume -/
def total_calories : ℕ := 480

/-- Represents the total amount Peter needs to spend in dollars -/
def total_spent : ℚ := 4

/-- Theorem stating that each chip contains 10 calories -/
theorem calories_per_chip : 
  (total_calories : ℚ) / (total_spent / cost_per_bag * chips_per_bag) = 10 := by
  sorry

end NUMINAMATH_CALUDE_calories_per_chip_l3027_302756


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3027_302747

-- Define the concept of opposite
def opposite (x : ℤ) : ℤ := -x

-- Theorem stating that the opposite of -2 is 2
theorem opposite_of_negative_two : opposite (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3027_302747


namespace NUMINAMATH_CALUDE_interesting_numbers_l3027_302728

def is_interesting (n : ℕ) : Prop :=
  20 ≤ n ∧ n ≤ 90 ∧
  ∃ (p : ℕ) (k : ℕ), 
    Nat.Prime p ∧ 
    k ≥ 2 ∧ 
    n = p^k

theorem interesting_numbers : 
  {n : ℕ | is_interesting n} = {25, 27, 32, 49, 64, 81} :=
by sorry

end NUMINAMATH_CALUDE_interesting_numbers_l3027_302728


namespace NUMINAMATH_CALUDE_vector_simplification_l3027_302772

-- Define the vector space
variable {V : Type*} [AddCommGroup V]

-- Define the vectors
variable (O P Q M : V)

-- State the theorem
theorem vector_simplification :
  (P - O) + (Q - P) - (Q - M) = M - O :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3027_302772


namespace NUMINAMATH_CALUDE_base7_divisibility_by_19_l3027_302709

def base7ToDecimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem base7_divisibility_by_19 :
  ∃ (x : ℕ), x < 7 ∧ 19 ∣ (base7ToDecimal 2 5 x 3) ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_base7_divisibility_by_19_l3027_302709


namespace NUMINAMATH_CALUDE_johnny_attend_probability_l3027_302736

-- Define the probabilities
def p_rain : ℝ := 0.3
def p_sunny : ℝ := 0.5
def p_cloudy : ℝ := 1 - p_rain - p_sunny

def p_attend_given_rain : ℝ := 0.5
def p_attend_given_sunny : ℝ := 0.9
def p_attend_given_cloudy : ℝ := 0.7

-- Define the theorem
theorem johnny_attend_probability :
  p_attend_given_rain * p_rain + p_attend_given_sunny * p_sunny + p_attend_given_cloudy * p_cloudy = 0.74 := by
  sorry


end NUMINAMATH_CALUDE_johnny_attend_probability_l3027_302736


namespace NUMINAMATH_CALUDE_stair_climbing_and_descending_l3027_302749

def climbStairs (n : ℕ) : ℕ :=
  if n ≤ 2 then n else climbStairs (n - 1) + climbStairs (n - 2)

def descendStairs (n : ℕ) : ℕ := 2^(n - 1)

theorem stair_climbing_and_descending :
  (climbStairs 10 = 89) ∧ (descendStairs 10 = 512) := by
  sorry

end NUMINAMATH_CALUDE_stair_climbing_and_descending_l3027_302749


namespace NUMINAMATH_CALUDE_widgets_per_carton_is_three_l3027_302766

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculate the number of widgets per carton -/
def widgetsPerCarton (cartonDim : BoxDimensions) (shippingBoxDim : BoxDimensions) (totalWidgets : ℕ) : ℕ :=
  let cartonsPerLayer := (shippingBoxDim.width / cartonDim.width) * (shippingBoxDim.length / cartonDim.length)
  let layers := shippingBoxDim.height / cartonDim.height
  let totalCartons := cartonsPerLayer * layers
  totalWidgets / totalCartons

theorem widgets_per_carton_is_three :
  let cartonDim : BoxDimensions := ⟨4, 4, 5⟩
  let shippingBoxDim : BoxDimensions := ⟨20, 20, 20⟩
  let totalWidgets : ℕ := 300
  widgetsPerCarton cartonDim shippingBoxDim totalWidgets = 3 := by
  sorry

end NUMINAMATH_CALUDE_widgets_per_carton_is_three_l3027_302766


namespace NUMINAMATH_CALUDE_circumscribed_circle_diameter_l3027_302721

/-- The diameter of a triangle's circumscribed circle, given one side and its opposite angle. -/
theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h_side : side = 18) (h_angle : angle = π/4) :
  let diameter := side / Real.sin angle
  diameter = 18 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_diameter_l3027_302721


namespace NUMINAMATH_CALUDE_hotel_room_encoding_l3027_302708

theorem hotel_room_encoding (x : ℕ) : 
  1 ≤ x ∧ x ≤ 30 → x % 5 = 3 → x % 7 = 6 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_encoding_l3027_302708


namespace NUMINAMATH_CALUDE_coffee_packages_solution_l3027_302768

/-- Represents the number of 10-ounce packages -/
def num_10oz : ℕ := 4

/-- Represents the number of 5-ounce packages -/
def num_5oz : ℕ := num_10oz + 2

/-- Total ounces of coffee -/
def total_ounces : ℕ := 115

/-- Cost of a 5-ounce package in cents -/
def cost_5oz : ℕ := 150

/-- Cost of a 10-ounce package in cents -/
def cost_10oz : ℕ := 250

/-- Maximum total cost in cents -/
def max_cost : ℕ := 2000

theorem coffee_packages_solution :
  (num_10oz * 10 + num_5oz * 5 = total_ounces) ∧
  (num_10oz * cost_10oz + num_5oz * cost_5oz ≤ max_cost) :=
by sorry

end NUMINAMATH_CALUDE_coffee_packages_solution_l3027_302768


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3027_302788

/-- The coefficient of x^4 in the expansion of (1 + √x)^10 -/
def coefficient_x4 : ℕ :=
  Nat.choose 10 8

theorem expansion_coefficient (n : ℕ) :
  coefficient_x4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3027_302788


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3027_302712

/-- Given vectors a and b in ℝ², prove that if a is perpendicular to b, then m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a = (-1, 2) ∧ b = (m, 1)) :
  a.1 * b.1 + a.2 * b.2 = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3027_302712


namespace NUMINAMATH_CALUDE_g_1989_of_5_eq_5_l3027_302790

def g (x : ℚ) : ℚ := (2 - x) / (1 + 2 * x)

def g_n : ℕ → (ℚ → ℚ)
| 0 => λ x => x
| n + 1 => λ x => g (g_n n x)

theorem g_1989_of_5_eq_5 : g_n 1989 5 = 5 := by sorry

end NUMINAMATH_CALUDE_g_1989_of_5_eq_5_l3027_302790


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3027_302778

theorem trigonometric_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (170 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3027_302778


namespace NUMINAMATH_CALUDE_f_properties_l3027_302739

def f (x y : ℝ) : ℝ × ℝ := (x - y, x + y)

theorem f_properties :
  (f 3 5 = (-2, 8)) ∧ (f 4 1 = (3, 5)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3027_302739


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_l3027_302771

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

/-- The ratio of brand 'a' to brand 'b' in the mixture by volume -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3440

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 370

theorem vegetable_ghee_mixture :
  weight_a * (ratio_a * total_volume / (ratio_a + ratio_b)) +
  weight_b * (ratio_b * total_volume / (ratio_a + ratio_b)) = total_weight :=
sorry

end NUMINAMATH_CALUDE_vegetable_ghee_mixture_l3027_302771


namespace NUMINAMATH_CALUDE_sweater_price_theorem_l3027_302787

def total_price_shirts : ℕ := 400
def num_shirts : ℕ := 25
def num_sweaters : ℕ := 75
def price_diff : ℕ := 4

theorem sweater_price_theorem :
  let avg_shirt_price := total_price_shirts / num_shirts
  let avg_sweater_price := avg_shirt_price + price_diff
  avg_sweater_price * num_sweaters = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_theorem_l3027_302787


namespace NUMINAMATH_CALUDE_unique_solution_sum_l3027_302732

theorem unique_solution_sum (x y : ℝ) : 
  (|x - 5| = |y - 11|) →
  (|x - 11| = 2*|y - 5|) →
  (x + y = 16) →
  (x + y = 16) :=
by
  sorry

#check unique_solution_sum

end NUMINAMATH_CALUDE_unique_solution_sum_l3027_302732


namespace NUMINAMATH_CALUDE_jerry_collection_cost_l3027_302762

/-- The amount of money Jerry needs to complete his action figure collection -/
def jerry_needs_money (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Jerry needs $72 to complete his collection -/
theorem jerry_collection_cost : jerry_needs_money 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerry_collection_cost_l3027_302762
