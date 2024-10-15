import Mathlib

namespace NUMINAMATH_CALUDE_notebook_pen_cost_ratio_l1203_120335

theorem notebook_pen_cost_ratio : 
  let pen_cost : ℚ := 3/2  -- $1.50 as a rational number
  let notebooks_cost : ℚ := 18  -- Total cost of 4 notebooks
  let notebooks_count : ℕ := 4  -- Number of notebooks
  let notebook_cost : ℚ := notebooks_cost / notebooks_count  -- Cost of one notebook
  (notebook_cost / pen_cost) = 3 := by sorry

end NUMINAMATH_CALUDE_notebook_pen_cost_ratio_l1203_120335


namespace NUMINAMATH_CALUDE_students_remaining_after_three_stops_l1203_120352

def initial_students : ℕ := 60

def remaining_after_first_stop (initial : ℕ) : ℕ :=
  initial - initial / 3

def remaining_after_second_stop (after_first : ℕ) : ℕ :=
  after_first - after_first / 4

def remaining_after_third_stop (after_second : ℕ) : ℕ :=
  after_second - after_second / 5

theorem students_remaining_after_three_stops :
  remaining_after_third_stop (remaining_after_second_stop (remaining_after_first_stop initial_students)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_students_remaining_after_three_stops_l1203_120352


namespace NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1203_120316

theorem lcm_of_ratio_and_hcf (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 2 / 3 → 
  Nat.gcd a b = 6 → 
  Nat.lcm a b = 36 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_and_hcf_l1203_120316


namespace NUMINAMATH_CALUDE_tshirt_cost_calculation_l1203_120324

def sweatshirt_cost : ℕ := 15
def num_sweatshirts : ℕ := 3
def num_tshirts : ℕ := 2
def total_spent : ℕ := 65

theorem tshirt_cost_calculation :
  ∃ (tshirt_cost : ℕ), 
    num_sweatshirts * sweatshirt_cost + num_tshirts * tshirt_cost = total_spent ∧
    tshirt_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_calculation_l1203_120324


namespace NUMINAMATH_CALUDE_negation_of_existence_l1203_120379

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1203_120379


namespace NUMINAMATH_CALUDE_trig_problem_l1203_120385

theorem trig_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (π - α) + Real.cos (2 * π + α) = Real.sqrt 2 / 3) : 
  (Real.sin α - Real.cos α = 4 / 3) ∧ 
  (Real.tan α = -(9 + 4 * Real.sqrt 2) / 7) := by
  sorry

end NUMINAMATH_CALUDE_trig_problem_l1203_120385


namespace NUMINAMATH_CALUDE_arun_weight_upper_limit_l1203_120356

/-- The upper limit of Arun's weight according to his own opinion -/
def arun_upper_limit : ℝ := 69

/-- Arun's lower weight limit -/
def arun_lower_limit : ℝ := 66

/-- The average of Arun's probable weights -/
def arun_average_weight : ℝ := 68

/-- Brother's upper limit for Arun's weight -/
def brother_upper_limit : ℝ := 70

/-- Mother's upper limit for Arun's weight -/
def mother_upper_limit : ℝ := 69

theorem arun_weight_upper_limit :
  arun_upper_limit = 69 ∧
  arun_lower_limit < arun_upper_limit ∧
  arun_lower_limit < brother_upper_limit ∧
  arun_upper_limit ≤ mother_upper_limit ∧
  arun_upper_limit ≤ brother_upper_limit ∧
  (arun_lower_limit + arun_upper_limit) / 2 = arun_average_weight :=
by sorry

end NUMINAMATH_CALUDE_arun_weight_upper_limit_l1203_120356


namespace NUMINAMATH_CALUDE_second_odd_integer_l1203_120369

theorem second_odd_integer (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n - 1 ∧ b = 2*n + 1 ∧ c = 2*n + 3) →  -- consecutive odd integers
  (a + c = 128) →                                      -- sum of first and third is 128
  b = 64                                               -- second integer is 64
:= by sorry

end NUMINAMATH_CALUDE_second_odd_integer_l1203_120369


namespace NUMINAMATH_CALUDE_at_least_one_composite_l1203_120388

theorem at_least_one_composite (a b c : ℕ) (h1 : c ≥ 2) (h2 : 1 / a + 1 / b = 1 / c) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (a + c = x * y ∨ b + c = x * y)) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_composite_l1203_120388


namespace NUMINAMATH_CALUDE_max_value_quadratic_max_value_quadratic_achievable_l1203_120387

theorem max_value_quadratic (p : ℝ) : -3 * p^2 + 54 * p - 30 ≤ 213 := by sorry

theorem max_value_quadratic_achievable : ∃ p : ℝ, -3 * p^2 + 54 * p - 30 = 213 := by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_max_value_quadratic_achievable_l1203_120387


namespace NUMINAMATH_CALUDE_horner_method_for_f_l1203_120376

def f (x : ℝ) : ℝ := 5*x^5 + 2*x^4 + 3.5*x^3 - 2.6*x^2 + 1.7*x - 0.8

theorem horner_method_for_f :
  f 3 = 1452.4 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l1203_120376


namespace NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l1203_120357

theorem greatest_c_for_quadratic_range (c : ℤ) : 
  (∀ x : ℝ, x^2 + c*x + 18 ≠ -6) ↔ c ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_c_for_quadratic_range_l1203_120357


namespace NUMINAMATH_CALUDE_ten_player_modified_round_robin_l1203_120390

/-- The number of matches in a modified round-robin tournament --/
def modifiedRoundRobinMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2 - 1

/-- Theorem: In a round-robin tournament with 10 players, where each player
    plays every other player once, but the match between the first and
    second players is not held, the total number of matches is 44. --/
theorem ten_player_modified_round_robin :
  modifiedRoundRobinMatches 10 = 44 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_modified_round_robin_l1203_120390


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l1203_120399

theorem least_positive_integer_to_multiple_of_five : 
  ∀ n : ℕ, n > 0 → (725 + n) % 5 = 0 → n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l1203_120399


namespace NUMINAMATH_CALUDE_inequality_range_l1203_120304

theorem inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0) ↔ k ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1203_120304


namespace NUMINAMATH_CALUDE_fraction_simplification_l1203_120342

theorem fraction_simplification :
  (2 * Real.sqrt 7) / (Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 6) =
  (2 * Real.sqrt 14 + 8 * Real.sqrt 21 + 2 * Real.sqrt 42 + 8 * Real.sqrt 63) / 23 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1203_120342


namespace NUMINAMATH_CALUDE_lanas_roses_l1203_120394

theorem lanas_roses (tulips : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) :
  tulips = 36 → used_flowers = 70 → extra_flowers = 3 →
  used_flowers + extra_flowers - tulips = 37 := by
  sorry

end NUMINAMATH_CALUDE_lanas_roses_l1203_120394


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1203_120314

/-- Represents a plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- Perpendicularity relation between planes -/
def perpendicular (p q : Plane) : Prop :=
  sorry

/-- Parallelism relation between planes -/
def parallel (p q : Plane) : Prop :=
  sorry

theorem sufficient_not_necessary_condition 
  (α β γ : Plane) 
  (h_different : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h_perp : perpendicular α γ) :
  (∀ (α β γ : Plane), parallel α β → perpendicular β γ) ∧
  (∃ (α β γ : Plane), perpendicular β γ ∧ ¬parallel α β) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1203_120314


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l1203_120362

theorem max_sum_with_constraint (a b : ℝ) (h : a^2 - a*b + b^2 = 1) :
  a + b ≤ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀^2 - a₀*b₀ + b₀^2 = 1 ∧ a₀ + b₀ = 2 := by
sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l1203_120362


namespace NUMINAMATH_CALUDE_lottery_probabilities_l1203_120341

-- Define the lottery setup
def total_balls : ℕ := 10
def balls_with_2 : ℕ := 8
def balls_with_5 : ℕ := 2
def drawn_balls : ℕ := 3

-- Define the possible prize amounts
def prize_amounts : List ℕ := [6, 9, 12]

-- Define the corresponding probabilities
def probabilities : List ℚ := [7/15, 7/15, 1/15]

-- Theorem statement
theorem lottery_probabilities :
  let possible_outcomes := List.zip prize_amounts probabilities
  ∀ (outcome : ℕ × ℚ), outcome ∈ possible_outcomes →
    (∃ (n2 n5 : ℕ), n2 + n5 = drawn_balls ∧
      n2 * 2 + n5 * 5 = outcome.1 ∧
      (n2.choose balls_with_2 * n5.choose balls_with_5) / drawn_balls.choose total_balls = outcome.2) :=
by sorry

end NUMINAMATH_CALUDE_lottery_probabilities_l1203_120341


namespace NUMINAMATH_CALUDE_vacation_cost_l1203_120337

theorem vacation_cost (hotel_cost_per_person_per_day : ℕ) 
                      (total_vacation_cost : ℕ) 
                      (num_days : ℕ) 
                      (num_people : ℕ) : 
  hotel_cost_per_person_per_day = 12 →
  total_vacation_cost = 120 →
  num_days = 3 →
  num_people = 2 →
  (total_vacation_cost - (hotel_cost_per_person_per_day * num_days * num_people)) / num_people = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_l1203_120337


namespace NUMINAMATH_CALUDE_nth_prime_upper_bound_and_prime_counting_lower_bound_l1203_120321

-- Define the n-th prime number
def nth_prime (n : ℕ) : ℕ := sorry

-- Define the prime counting function
def prime_counting_function (x : ℝ) : ℝ := sorry

theorem nth_prime_upper_bound_and_prime_counting_lower_bound :
  (∀ n : ℕ, nth_prime n ≤ 2^(2^n)) ∧
  (∃ c : ℝ, c > 0 ∧ ∀ x : ℝ, x > Real.exp 1 → prime_counting_function x ≥ c * Real.log (Real.log x)) :=
sorry

end NUMINAMATH_CALUDE_nth_prime_upper_bound_and_prime_counting_lower_bound_l1203_120321


namespace NUMINAMATH_CALUDE_equal_expressions_imply_abs_difference_l1203_120350

theorem equal_expressions_imply_abs_difference (x y : ℝ) :
  ((x + y = x - y ∧ x + y = x / y) ∨
   (x + y = x - y ∧ x + y = x * y) ∨
   (x + y = x / y ∧ x + y = x * y) ∨
   (x - y = x / y ∧ x - y = x * y) ∨
   (x - y = x / y ∧ x * y = x / y) ∨
   (x + y = x / y ∧ x - y = x / y)) →
  |y| - |x| = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equal_expressions_imply_abs_difference_l1203_120350


namespace NUMINAMATH_CALUDE_max_value_on_curve_l1203_120347

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 + y^2/3 = 1

-- Define the function to be maximized
def f (x y : ℝ) : ℝ := 3*x + y

-- Theorem statement
theorem max_value_on_curve :
  ∃ (M : ℝ), M = 2 * Real.sqrt 3 ∧
  (∀ (x y : ℝ), C x y → f x y ≤ M) ∧
  (∃ (x y : ℝ), C x y ∧ f x y = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_on_curve_l1203_120347


namespace NUMINAMATH_CALUDE_original_orange_price_l1203_120323

theorem original_orange_price 
  (price_increase : ℝ) 
  (original_mango_price : ℝ) 
  (new_total_cost : ℝ) 
  (h1 : price_increase = 0.15)
  (h2 : original_mango_price = 50)
  (h3 : new_total_cost = 1035) :
  ∃ (original_orange_price : ℝ),
    original_orange_price = 40 ∧
    new_total_cost = 10 * (original_orange_price * (1 + price_increase)) + 
                     10 * (original_mango_price * (1 + price_increase)) :=
by sorry

end NUMINAMATH_CALUDE_original_orange_price_l1203_120323


namespace NUMINAMATH_CALUDE_probability_of_scoring_five_l1203_120358

def num_balls : ℕ := 2
def num_draws : ℕ := 3
def red_ball_score : ℕ := 2
def black_ball_score : ℕ := 1
def target_score : ℕ := 5

def probability_of_drawing_red : ℚ := 1 / 2

theorem probability_of_scoring_five (n : ℕ) (k : ℕ) (p : ℚ) :
  n = num_draws →
  k = 2 →
  p = probability_of_drawing_red →
  (Nat.choose n k * p^k * (1 - p)^(n - k) : ℚ) = 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_scoring_five_l1203_120358


namespace NUMINAMATH_CALUDE_quadratic_solution_l1203_120366

theorem quadratic_solution (a : ℝ) : (1 : ℝ)^2 + 1 + 2*a = 0 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1203_120366


namespace NUMINAMATH_CALUDE_dozen_chocolate_cost_l1203_120359

/-- The cost of a dozen chocolate bars given the relative prices of magazines and chocolates -/
theorem dozen_chocolate_cost (magazine_price : ℝ) (chocolate_bar_price : ℝ) : 
  magazine_price = 1 →
  4 * chocolate_bar_price = 8 * magazine_price →
  12 * chocolate_bar_price = 24 := by
  sorry

end NUMINAMATH_CALUDE_dozen_chocolate_cost_l1203_120359


namespace NUMINAMATH_CALUDE_proposition_correctness_l1203_120307

-- Define the propositions
def prop1 (a b c : ℝ) : Prop := a > b → a * c^2 > b * c^2
def prop2 (a b : ℝ) : Prop := a > |b| → a^2 > b^2
def prop3 (a b : ℝ) : Prop := |a| > b → a^2 > b^2
def prop4 (a b : ℝ) : Prop := a > b → a^3 > b^3

-- Theorem stating the correctness of propositions
theorem proposition_correctness :
  (∃ a b c : ℝ, ¬(prop1 a b c)) ∧
  (∀ a b : ℝ, prop2 a b) ∧
  (∃ a b : ℝ, ¬(prop3 a b)) ∧
  (∀ a b : ℝ, prop4 a b) :=
sorry

end NUMINAMATH_CALUDE_proposition_correctness_l1203_120307


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1203_120349

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 2 → x^2 + 2*x - 8 > 0) ∧ 
  (∃ x : ℝ, x ≤ 2 ∧ x^2 + 2*x - 8 > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1203_120349


namespace NUMINAMATH_CALUDE_product_of_zero_functions_is_zero_function_l1203_120383

-- Define the concept of a zero function on a domain D
def is_zero_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, f x = 0

-- State the theorem
theorem product_of_zero_functions_is_zero_function 
  (f g : ℝ → ℝ) (D : Set ℝ) 
  (hf : is_zero_function f D) (hg : is_zero_function g D) : 
  is_zero_function (fun x ↦ f x * g x) D :=
sorry

end NUMINAMATH_CALUDE_product_of_zero_functions_is_zero_function_l1203_120383


namespace NUMINAMATH_CALUDE_only_B_on_x_axis_l1203_120353

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (3, 0)
def point_C : ℝ × ℝ := (0, -1)
def point_D : ℝ × ℝ := (-5, 6)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis :
  ¬(is_on_x_axis point_A) ∧
  is_on_x_axis point_B ∧
  ¬(is_on_x_axis point_C) ∧
  ¬(is_on_x_axis point_D) :=
by sorry

end NUMINAMATH_CALUDE_only_B_on_x_axis_l1203_120353


namespace NUMINAMATH_CALUDE_square_field_area_l1203_120391

theorem square_field_area (side_length : ℝ) (h : side_length = 25) : 
  side_length * side_length = 625 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l1203_120391


namespace NUMINAMATH_CALUDE_unique_solution_iff_p_zero_l1203_120340

/-- The system of equations has exactly one solution if and only if p = 0 -/
theorem unique_solution_iff_p_zero (p : ℝ) :
  (∃! x y : ℝ, x^2 - y^2 = 0 ∧ x*y + p*x - p*y = p^2) ↔ p = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_p_zero_l1203_120340


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1203_120374

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem intersection_complement_equals_set : N ∩ (U \ M) = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1203_120374


namespace NUMINAMATH_CALUDE_min_product_value_l1203_120329

def is_monic_nonneg_int_coeff (p : ℕ → ℕ) : Prop :=
  p 0 = 1 ∧ ∀ n, p n ≥ 0

def satisfies_inequality (p q : ℕ → ℕ) : Prop :=
  ∀ x : ℕ, x ≥ 2 → (1 : ℚ) / (5 * x) ≥ 1 / (q x) - 1 / (p x) ∧ 1 / (q x) - 1 / (p x) ≥ 1 / (3 * x^2)

theorem min_product_value (p q : ℕ → ℕ) :
  is_monic_nonneg_int_coeff p →
  is_monic_nonneg_int_coeff q →
  satisfies_inequality p q →
  (∀ p' q' : ℕ → ℕ, is_monic_nonneg_int_coeff p' → is_monic_nonneg_int_coeff q' → 
    satisfies_inequality p' q' → p' 1 * q' 1 ≥ p 1 * q 1) →
  p 1 * q 1 = 3 :=
sorry

end NUMINAMATH_CALUDE_min_product_value_l1203_120329


namespace NUMINAMATH_CALUDE_range_of_negative_values_l1203_120361

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonically decreasing on (-∞, 0] if
    for all x, y ∈ (-∞, 0], x ≤ y implies f(x) ≥ f(y) -/
def MonoDecreasingNonPositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 0 → f x ≥ f y

/-- The main theorem -/
theorem range_of_negative_values
  (f : ℝ → ℝ)
  (h_even : EvenFunction f)
  (h_mono : MonoDecreasingNonPositive f)
  (h_f1 : f 1 = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_negative_values_l1203_120361


namespace NUMINAMATH_CALUDE_sequence_property_l1203_120370

theorem sequence_property (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ n : ℕ, a (n + 3) = a n)
  (h2 : ∀ n : ℕ, a n * a (n + 3) - a (n + 1) * a (n + 2) = c) :
  (∀ n : ℕ, a (n + 1) = a n ∧ c = 0) ∨ 
  (∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 0 ∧ 4 * c - 3 * (a n)^2 > 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l1203_120370


namespace NUMINAMATH_CALUDE_pen_difference_l1203_120330

/-- A collection of pens and pencils -/
structure PenCollection where
  blue_pens : ℕ
  black_pens : ℕ
  red_pens : ℕ
  pencils : ℕ

/-- Properties of the pen collection -/
def valid_collection (c : PenCollection) : Prop :=
  c.black_pens = c.blue_pens + 10 ∧
  c.blue_pens = 2 * c.pencils ∧
  c.pencils = 8 ∧
  c.blue_pens + c.black_pens + c.red_pens = 48 ∧
  c.red_pens < c.pencils

theorem pen_difference (c : PenCollection) 
  (h : valid_collection c) : c.pencils - c.red_pens = 2 := by
  sorry

end NUMINAMATH_CALUDE_pen_difference_l1203_120330


namespace NUMINAMATH_CALUDE_min_value_of_f_l1203_120306

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 2023

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = 1996) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1203_120306


namespace NUMINAMATH_CALUDE_sum_binomial_coefficients_l1203_120339

theorem sum_binomial_coefficients (n : ℕ) : 
  (Finset.range (n + 1)).sum (fun k => Nat.choose n k) = 2^n := by
  sorry

end NUMINAMATH_CALUDE_sum_binomial_coefficients_l1203_120339


namespace NUMINAMATH_CALUDE_shirt_cost_l1203_120382

theorem shirt_cost (total_cost coat_price shirt_price : ℚ) : 
  total_cost = 600 →
  shirt_price = (1 / 3) * coat_price →
  total_cost = shirt_price + coat_price →
  shirt_price = 150 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_l1203_120382


namespace NUMINAMATH_CALUDE_salary_change_l1203_120345

theorem salary_change (original_salary : ℝ) (h : original_salary > 0) :
  let increased_salary := original_salary * 1.3
  let final_salary := increased_salary * 0.7
  (final_salary - original_salary) / original_salary = -0.09 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l1203_120345


namespace NUMINAMATH_CALUDE_henry_age_is_20_l1203_120367

/-- Henry's present age -/
def henry_age : ℕ := sorry

/-- Jill's present age -/
def jill_age : ℕ := sorry

/-- The sum of Henry and Jill's present ages is 33 -/
axiom sum_of_ages : henry_age + jill_age = 33

/-- Six years ago, Henry was twice the age of Jill -/
axiom ages_relation : henry_age - 6 = 2 * (jill_age - 6)

/-- Henry's present age is 20 years -/
theorem henry_age_is_20 : henry_age = 20 := by sorry

end NUMINAMATH_CALUDE_henry_age_is_20_l1203_120367


namespace NUMINAMATH_CALUDE_smallest_a_is_390_l1203_120368

/-- A polynomial with three positive integer roots -/
structure PolynomialWithThreeIntegerRoots where
  a : ℕ
  b : ℕ
  root1 : ℕ+
  root2 : ℕ+
  root3 : ℕ+
  root_product : root1 * root2 * root3 = 2310
  root_sum : root1 + root2 + root3 = a

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a : ℕ := 390

/-- Theorem stating that 390 is the smallest possible value of a -/
theorem smallest_a_is_390 :
  ∀ p : PolynomialWithThreeIntegerRoots, p.a ≥ smallest_a :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_is_390_l1203_120368


namespace NUMINAMATH_CALUDE_inequality_relationship_l1203_120389

theorem inequality_relationship (a : ℝ) (h : a^2 + a < 0) :
  -a > a^2 ∧ a^2 > -a^2 ∧ -a^2 > a := by sorry

end NUMINAMATH_CALUDE_inequality_relationship_l1203_120389


namespace NUMINAMATH_CALUDE_opposite_of_two_l1203_120360

theorem opposite_of_two : 
  ∃ x : ℝ, x + 2 = 0 ∧ x = -2 :=
sorry

end NUMINAMATH_CALUDE_opposite_of_two_l1203_120360


namespace NUMINAMATH_CALUDE_tea_sales_revenue_l1203_120371

/-- Represents the sales data for tea leaves over two years -/
structure TeaSalesData where
  price_ratio : ℝ  -- Ratio of this year's price to last year's
  yield_this_year : ℝ  -- Yield in kg this year
  yield_difference : ℝ  -- Difference in yield compared to last year
  revenue_increase : ℝ  -- Increase in revenue compared to last year

/-- Calculates the sales revenue for this year given the tea sales data -/
def calculate_revenue (data : TeaSalesData) : ℝ :=
  let yield_last_year := data.yield_this_year + data.yield_difference
  let revenue_last_year := yield_last_year
  revenue_last_year + data.revenue_increase

/-- Theorem stating that given the specific conditions, the sales revenue this year is 9930 yuan -/
theorem tea_sales_revenue 
  (data : TeaSalesData)
  (h1 : data.price_ratio = 10)
  (h2 : data.yield_this_year = 198.6)
  (h3 : data.yield_difference = 87.4)
  (h4 : data.revenue_increase = 8500) :
  calculate_revenue data = 9930 := by
  sorry

#eval calculate_revenue ⟨10, 198.6, 87.4, 8500⟩

end NUMINAMATH_CALUDE_tea_sales_revenue_l1203_120371


namespace NUMINAMATH_CALUDE_problem_statement_l1203_120300

theorem problem_statement : 
  (Real.sqrt (5 + Real.sqrt 6) + Real.sqrt (5 - Real.sqrt 6)) / Real.sqrt (Real.sqrt 6 - 1) - Real.sqrt (4 - 2 * Real.sqrt 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1203_120300


namespace NUMINAMATH_CALUDE_pascal_triangle_formula_l1203_120313

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define binomial coefficient
def binomial_coeff (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

-- Theorem statement
theorem pascal_triangle_formula (n k : ℕ) (h : k ≤ n) :
  binomial_coeff n k = factorial n / (factorial k * factorial (n - k)) :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_formula_l1203_120313


namespace NUMINAMATH_CALUDE_inequality_solution_set_F_zero_points_range_l1203_120380

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := |2*x - 1|

-- Define the inequality
def inequality (x : ℝ) : Prop := f (x + 5) ≤ x * g x

-- Define the function F
def F (x a : ℝ) : ℝ := f (x + 2) + f x + a

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | inequality x} = {x : ℝ | x ≥ 2} :=
sorry

-- Theorem for the range of a when F has zero points
theorem F_zero_points_range (a : ℝ) :
  (∃ x, F x a = 0) ↔ a ∈ Set.Iic (-2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_F_zero_points_range_l1203_120380


namespace NUMINAMATH_CALUDE_demographic_prediction_basis_l1203_120384

/-- Represents the possible bases for demographic predictions -/
inductive DemographicBasis
  | PopulationQuantityAndDensity
  | AgeComposition
  | GenderRatio
  | BirthAndDeathRates

/-- Represents different countries -/
inductive Country
  | Mexico
  | UnitedStates
  | Sweden
  | Germany

/-- Represents the prediction for population growth -/
inductive PopulationPrediction
  | Increase
  | Stable
  | Decrease

/-- Function that assigns a population prediction to each country -/
def countryPrediction : Country → PopulationPrediction
  | Country.Mexico => PopulationPrediction.Increase
  | Country.UnitedStates => PopulationPrediction.Increase
  | Country.Sweden => PopulationPrediction.Stable
  | Country.Germany => PopulationPrediction.Decrease

/-- The main basis used by demographers for their predictions -/
def mainBasis : DemographicBasis := DemographicBasis.AgeComposition

theorem demographic_prediction_basis :
  (∀ c : Country, ∃ p : PopulationPrediction, countryPrediction c = p) →
  mainBasis = DemographicBasis.AgeComposition :=
by sorry

end NUMINAMATH_CALUDE_demographic_prediction_basis_l1203_120384


namespace NUMINAMATH_CALUDE_tangent_circles_max_product_l1203_120393

/-- Two externally tangent circles with given equations have a maximum product of their x-offsets --/
theorem tangent_circles_max_product (a b : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4) →
  (∃ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1) →
  (∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4 → (x + b)^2 + (y + 2)^2 = 1 → 
    ∃ t : ℝ, (x - a)^2 + (y + 2)^2 = (x + b)^2 + (y + 2)^2 + ((2 - 1) * t)^2) →
  (∀ c : ℝ, a * b ≤ c → c ≤ 9/4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circles_max_product_l1203_120393


namespace NUMINAMATH_CALUDE_golf_ball_goal_l1203_120320

theorem golf_ball_goal (goal : ℕ) (saturday : ℕ) (sunday : ℕ) : 
  goal = 48 → saturday = 16 → sunday = 18 → 
  goal - (saturday + sunday) = 14 :=
by sorry

end NUMINAMATH_CALUDE_golf_ball_goal_l1203_120320


namespace NUMINAMATH_CALUDE_cubic_polynomial_properties_l1203_120318

/-- The cubic polynomial f(x) = x³ + px + q -/
noncomputable def f (p q x : ℝ) : ℝ := x^3 + p*x + q

theorem cubic_polynomial_properties (p q : ℝ) :
  (p ≥ 0 → ∀ x y : ℝ, x < y → f p q x < f p q y) ∧ 
  (p < 0 → ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f p q x = 0 ∧ f p q y = 0 ∧ f p q z = 0) ∧
  (p < 0 → ∃! x y : ℝ, x ≠ y ∧ (∀ z : ℝ, f p q x ≤ f p q z) ∧ (∀ z : ℝ, f p q y ≥ f p q z)) ∧
  (p < 0 → ∃ x y : ℝ, x ≠ y ∧ (∀ z : ℝ, f p q x ≤ f p q z) ∧ (∀ z : ℝ, f p q y ≥ f p q z) ∧ x = -y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_properties_l1203_120318


namespace NUMINAMATH_CALUDE_table_tennis_expected_scores_l1203_120312

/-- Win probability for a match-up -/
structure MatchProbability where
  team_a_win : ℚ
  team_b_win : ℚ
  sum_to_one : team_a_win + team_b_win = 1

/-- Team scores -/
structure TeamScores where
  team_a : ℕ
  team_b : ℕ
  sum_to_three : team_a + team_b = 3

/-- Expected value of a discrete random variable -/
def expectedValue (probs : List ℚ) (values : List ℚ) : ℚ :=
  (probs.zip values).map (fun (p, v) => p * v) |>.sum

/-- Main theorem -/
theorem table_tennis_expected_scores 
  (match1 : MatchProbability) 
  (match2 : MatchProbability) 
  (match3 : MatchProbability) 
  (h1 : match1.team_a_win = 2/3)
  (h2 : match2.team_a_win = 2/5)
  (h3 : match3.team_a_win = 2/5) :
  let scores := TeamScores
  let ξ_probs := [8/75, 28/75, 2/5, 3/25]
  let ξ_values := [3, 2, 1, 0]
  let η_probs := [3/25, 2/5, 28/75, 8/75]
  let η_values := [3, 2, 1, 0]
  expectedValue ξ_probs ξ_values = 22/15 ∧ 
  expectedValue η_probs η_values = 23/15 := by
sorry


end NUMINAMATH_CALUDE_table_tennis_expected_scores_l1203_120312


namespace NUMINAMATH_CALUDE_congruence_solution_l1203_120325

theorem congruence_solution (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 → x % 6 = 1 % 6 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l1203_120325


namespace NUMINAMATH_CALUDE_base10_115_eq_base11_A5_l1203_120378

/-- Converts a digit to its character representation in base 11 --/
def toBase11Char (d : ℕ) : Char :=
  if d < 10 then Char.ofNat (d + 48) else 'A'

/-- Converts a natural number to its base 11 representation --/
def toBase11 (n : ℕ) : String :=
  if n < 11 then String.mk [toBase11Char n]
  else toBase11 (n / 11) ++ String.mk [toBase11Char (n % 11)]

/-- Theorem stating that 115 in base 10 is equivalent to A5 in base 11 --/
theorem base10_115_eq_base11_A5 : toBase11 115 = "A5" := by
  sorry

end NUMINAMATH_CALUDE_base10_115_eq_base11_A5_l1203_120378


namespace NUMINAMATH_CALUDE_bernard_red_notebooks_l1203_120377

def bernard_notebooks (red blue white given_away left : ℕ) : Prop :=
  red + blue + white = given_away + left

theorem bernard_red_notebooks : 
  ∃ (red : ℕ), bernard_notebooks red 17 19 46 5 ∧ red = 15 := by sorry

end NUMINAMATH_CALUDE_bernard_red_notebooks_l1203_120377


namespace NUMINAMATH_CALUDE_calculate_expression_l1203_120348

theorem calculate_expression : (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1203_120348


namespace NUMINAMATH_CALUDE_m_range_equivalence_l1203_120397

theorem m_range_equivalence (m : ℝ) : 
  Real.sqrt (2 * m + 1) > Real.sqrt (m^2 + m - 1) ↔ 
  m ≥ (Real.sqrt 5 - 1) / 2 ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_equivalence_l1203_120397


namespace NUMINAMATH_CALUDE_square_plus_double_is_perfect_square_l1203_120319

theorem square_plus_double_is_perfect_square (a : ℕ) : 
  ∃ (k : ℕ), a^2 + 2*a = k^2 ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_double_is_perfect_square_l1203_120319


namespace NUMINAMATH_CALUDE_coloring_books_bought_l1203_120396

theorem coloring_books_bought (initial books_given_away final : ℕ) : 
  initial = 45 → books_given_away = 6 → final = 59 → 
  final - (initial - books_given_away) = 20 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_bought_l1203_120396


namespace NUMINAMATH_CALUDE_equation_equivalence_l1203_120310

theorem equation_equivalence (x : ℝ) : 
  (x + 3) / 3 - (x - 1) / 6 = (5 - x) / 2 ↔ 2*x + 6 - x + 1 = 15 - 3*x := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1203_120310


namespace NUMINAMATH_CALUDE_factorization_proof_l1203_120328

theorem factorization_proof (x a b : ℝ) : 
  (4 * x^2 - 64 = 4 * (x + 4) * (x - 4)) ∧ 
  (4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1203_120328


namespace NUMINAMATH_CALUDE_tree_growth_rate_l1203_120395

/-- Proves that the annual increase in tree height is 1 foot -/
theorem tree_growth_rate (h : ℝ) : 
  (4 : ℝ) + 6 * h = ((4 : ℝ) + 4 * h) * (5/4) → h = 1 := by
  sorry

end NUMINAMATH_CALUDE_tree_growth_rate_l1203_120395


namespace NUMINAMATH_CALUDE_theater_ticket_pricing_l1203_120364

theorem theater_ticket_pricing (total_tickets : ℕ) (total_revenue : ℕ) 
  (balcony_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 370 →
  total_revenue = 3320 →
  balcony_price = 8 →
  balcony_orchestra_diff = 190 →
  ∃ (orchestra_price : ℕ),
    orchestra_price = 12 ∧
    (total_tickets - balcony_orchestra_diff) / 2 * orchestra_price + 
    (total_tickets + balcony_orchestra_diff) / 2 * balcony_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_theater_ticket_pricing_l1203_120364


namespace NUMINAMATH_CALUDE_emily_buys_12_cucumbers_l1203_120309

/-- The cost of one apple -/
def apple_cost : ℝ := sorry

/-- The cost of one banana -/
def banana_cost : ℝ := sorry

/-- The cost of one cucumber -/
def cucumber_cost : ℝ := sorry

/-- Six apples cost the same as three bananas -/
axiom six_apples_eq_three_bananas : 6 * apple_cost = 3 * banana_cost

/-- Three bananas cost the same as four cucumbers -/
axiom three_bananas_eq_four_cucumbers : 3 * banana_cost = 4 * cucumber_cost

/-- The number of cucumbers Emily can buy for the price of 18 apples -/
def cucumbers_for_18_apples : ℕ := sorry

/-- Proof that Emily can buy 12 cucumbers for the price of 18 apples -/
theorem emily_buys_12_cucumbers : cucumbers_for_18_apples = 12 := by
  sorry

end NUMINAMATH_CALUDE_emily_buys_12_cucumbers_l1203_120309


namespace NUMINAMATH_CALUDE_f_composition_comparison_f_inverse_solutions_l1203_120381

noncomputable section

def f (x : ℝ) : ℝ :=
  if x < 1 then -2 * x + 1 else x^2 - 2 * x

theorem f_composition_comparison : f (f (-3)) > f (f 3) := by sorry

theorem f_inverse_solutions (x : ℝ) :
  f x = 1 ↔ x = 0 ∨ x = 1 + Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_f_composition_comparison_f_inverse_solutions_l1203_120381


namespace NUMINAMATH_CALUDE_tiling_8x1_board_remainder_l1203_120317

/-- Represents a tiling of an 8x1 board -/
structure Tiling :=
  (num_1x1 : ℕ)
  (num_2x1 : ℕ)
  (h_sum : num_1x1 + 2 * num_2x1 = 8)

/-- Calculates the number of valid colorings for a given tiling -/
def validColorings (t : Tiling) : ℕ :=
  3^(t.num_1x1 + t.num_2x1) - 3 * 2^(t.num_1x1 + t.num_2x1) + 3

/-- The set of all possible tilings -/
def allTilings : Finset Tiling :=
  sorry

theorem tiling_8x1_board_remainder (M : ℕ) (h_M : M = (allTilings.sum validColorings)) :
  M % 1000 = 328 :=
sorry

end NUMINAMATH_CALUDE_tiling_8x1_board_remainder_l1203_120317


namespace NUMINAMATH_CALUDE_angle_properties_l1203_120351

theorem angle_properties (α β : Real) : 
  α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi) →  -- α is in the fourth quadrant
  Real.sin (Real.pi + α) = 2 * Real.sqrt 5 / 5 →
  Real.tan (α + β) = 1 / 7 →
  Real.cos (Real.pi / 3 + α) = (Real.sqrt 5 + 2 * Real.sqrt 15) / 10 ∧ 
  Real.tan β = 3 := by
sorry

end NUMINAMATH_CALUDE_angle_properties_l1203_120351


namespace NUMINAMATH_CALUDE_fraction_simplification_l1203_120343

theorem fraction_simplification (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + b) / (a - b) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1203_120343


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1203_120398

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → 1007 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1203_120398


namespace NUMINAMATH_CALUDE_convex_polyhedron_symmetry_l1203_120332

-- Define a structure for a polyhedron
structure Polyhedron where
  -- Add necessary fields (omitted for simplicity)

-- Define a property for convexity
def is_convex (p : Polyhedron) : Prop :=
  sorry

-- Define a property for central symmetry of faces
def has_centrally_symmetric_faces (p : Polyhedron) : Prop :=
  sorry

-- Define a property for subdivision into smaller polyhedra
def can_be_subdivided (p : Polyhedron) (subdivisions : List Polyhedron) : Prop :=
  sorry

-- Main theorem
theorem convex_polyhedron_symmetry 
  (p : Polyhedron) 
  (subdivisions : List Polyhedron) :
  is_convex p → 
  can_be_subdivided p subdivisions → 
  (∀ sub ∈ subdivisions, has_centrally_symmetric_faces sub) → 
  has_centrally_symmetric_faces p :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_symmetry_l1203_120332


namespace NUMINAMATH_CALUDE_circular_cross_section_solids_l1203_120373

-- Define the geometric solids
inductive GeometricSolid
  | Cube
  | Cylinder
  | Cone
  | TriangularPrism

-- Define a predicate for having a circular cross-section
def has_circular_cross_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => true
  | GeometricSolid.Cone => true
  | _ => false

-- Theorem statement
theorem circular_cross_section_solids :
  ∀ (solid : GeometricSolid),
    has_circular_cross_section solid ↔
      (solid = GeometricSolid.Cylinder ∨ solid = GeometricSolid.Cone) :=
by sorry

end NUMINAMATH_CALUDE_circular_cross_section_solids_l1203_120373


namespace NUMINAMATH_CALUDE_total_paths_A_to_C_l1203_120333

/-- The number of paths between two points -/
def num_paths (start finish : ℕ) : ℕ := sorry

theorem total_paths_A_to_C : 
  let paths_A_to_B := num_paths 1 2
  let paths_B_to_D := num_paths 2 3
  let paths_D_to_C := num_paths 3 4
  let direct_paths_A_to_C := num_paths 1 4
  
  paths_A_to_B = 2 →
  paths_B_to_D = 2 →
  paths_D_to_C = 2 →
  direct_paths_A_to_C = 2 →
  
  paths_A_to_B * paths_B_to_D * paths_D_to_C + direct_paths_A_to_C = 10 :=
by sorry

end NUMINAMATH_CALUDE_total_paths_A_to_C_l1203_120333


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l1203_120354

/-- Calculates the systematic sampling interval for a given population and sample size -/
def systematicSamplingInterval (population : ℕ) (sampleSize : ℕ) : ℕ :=
  (population - (population % sampleSize)) / sampleSize

theorem systematic_sampling_interval_for_given_problem :
  systematicSamplingInterval 1203 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_for_given_problem_l1203_120354


namespace NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l1203_120315

theorem smallest_square_enclosing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by sorry

end NUMINAMATH_CALUDE_smallest_square_enclosing_circle_l1203_120315


namespace NUMINAMATH_CALUDE_factor_polynomial_l1203_120305

theorem factor_polynomial (x : ℝ) : 60 * x^5 - 135 * x^9 = 15 * x^5 * (4 - 9 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1203_120305


namespace NUMINAMATH_CALUDE_line_hyperbola_tangency_l1203_120392

/-- The line y = k(x - √2) and the hyperbola x^2 - y^2 = 1 have only one point in common if and only if k = 1 or k = -1 -/
theorem line_hyperbola_tangency (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * (p.1 - Real.sqrt 2) ∧ p.1^2 - p.2^2 = 1) ↔ (k = 1 ∨ k = -1) := by
  sorry


end NUMINAMATH_CALUDE_line_hyperbola_tangency_l1203_120392


namespace NUMINAMATH_CALUDE_f_increasing_scaled_l1203_120338

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_increasing_scaled (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : x₁ * f x₁ < x₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_scaled_l1203_120338


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_when_not_zero_l1203_120302

theorem exp_gt_one_plus_x_when_not_zero (x : ℝ) (h : x ≠ 0) : Real.exp x > 1 + x := by
  sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_when_not_zero_l1203_120302


namespace NUMINAMATH_CALUDE_delta_triple_72_l1203_120311

/-- Definition of Δ function -/
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

/-- Theorem stating that Δ(Δ(Δ72)) = 7.728 -/
theorem delta_triple_72 : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end NUMINAMATH_CALUDE_delta_triple_72_l1203_120311


namespace NUMINAMATH_CALUDE_russian_alphabet_sum_sequence_exists_l1203_120327

theorem russian_alphabet_sum_sequence_exists : ∃ (π : Fin 33 → Fin 33), Function.Bijective π ∧
  ∀ (i j : Fin 33), i ≠ j → (π i + i : Fin 33) ≠ (π j + j : Fin 33) := by
  sorry

end NUMINAMATH_CALUDE_russian_alphabet_sum_sequence_exists_l1203_120327


namespace NUMINAMATH_CALUDE_tree_planting_event_girls_count_l1203_120336

theorem tree_planting_event_girls_count (boys : ℕ) (difference : ℕ) (total_percentage : ℚ) (partial_count : ℕ) 
  (h1 : boys = 600)
  (h2 : difference = 400)
  (h3 : total_percentage = 60 / 100)
  (h4 : partial_count = 960) : 
  ∃ (girls : ℕ), girls = 1000 ∧ girls > boys := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_event_girls_count_l1203_120336


namespace NUMINAMATH_CALUDE_expand_expression_l1203_120375

theorem expand_expression (x : ℝ) : (1 - x^2) * (1 + x^4) = 1 - x^2 + x^4 - x^6 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1203_120375


namespace NUMINAMATH_CALUDE_arrange_balls_theorem_l1203_120344

/-- The number of ways to arrange balls of different colors in a row -/
def arrangeMulticolorBalls (red : ℕ) (yellow : ℕ) (white : ℕ) : ℕ :=
  Nat.factorial (red + yellow + white) / (Nat.factorial red * Nat.factorial yellow * Nat.factorial white)

/-- Theorem: There are 1260 ways to arrange 2 red, 3 yellow, and 4 white indistinguishable balls in a row -/
theorem arrange_balls_theorem : arrangeMulticolorBalls 2 3 4 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_theorem_l1203_120344


namespace NUMINAMATH_CALUDE_difference_ones_zeros_157_l1203_120365

def binary_representation (n : ℕ) : List ℕ :=
  sorry

theorem difference_ones_zeros_157 :
  let binary := binary_representation 157
  let x := (binary.filter (· = 0)).length
  let y := (binary.filter (· = 1)).length
  y - x = 2 := by sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_157_l1203_120365


namespace NUMINAMATH_CALUDE_range_of_a_l1203_120334

theorem range_of_a (a : ℝ) : (∃ x : ℝ, Real.sqrt (3 * x + 6) + Real.sqrt (14 - x) > a) → a < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1203_120334


namespace NUMINAMATH_CALUDE_max_table_height_for_specific_triangle_l1203_120346

/-- Triangle ABC with sides a, b, c -/
structure Triangle :=
  (a b c : ℝ)

/-- The maximum height of the table constructed from the triangle -/
def maxTableHeight (t : Triangle) : ℝ := sorry

/-- The theorem to be proved -/
theorem max_table_height_for_specific_triangle :
  let t := Triangle.mk 23 27 30
  maxTableHeight t = (40 * Real.sqrt 221) / 57 := by sorry

end NUMINAMATH_CALUDE_max_table_height_for_specific_triangle_l1203_120346


namespace NUMINAMATH_CALUDE_problem_solution_l1203_120301

theorem problem_solution (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b * c = 1) (h5 : a + 1 / c = 7) (h6 : b + 1 / a = 34) :
  c + 1 / b = 43 / 237 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1203_120301


namespace NUMINAMATH_CALUDE_student_number_problem_l1203_120322

theorem student_number_problem (x : ℝ) : 7 * x - 150 = 130 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1203_120322


namespace NUMINAMATH_CALUDE_complex_exponent_l1203_120386

theorem complex_exponent (x : ℂ) (h : x - 1/x = 2*I) : x^729 - 1/x^729 = -4*I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponent_l1203_120386


namespace NUMINAMATH_CALUDE_circle_theorem_l1203_120372

-- Define a circle
def Circle : Type := {p : ℝ × ℝ // ∃ (center : ℝ × ℝ) (radius : ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define points on the circle
variable (ω₁ : Circle)
variable (A B C D : Circle)

-- Define the order of points on the circle
def InOrder (A C B D : Circle) : Prop := sorry

-- Define the distance between two points
def Distance (p q : Circle) : ℝ := sorry

-- Define the midpoint of an arc
def IsMidpointOfArc (M A B : Circle) : Prop := sorry

-- The main theorem
theorem circle_theorem (h_order : InOrder A C B D) :
  (Distance C D)^2 = (Distance A C) * (Distance B C) + (Distance A D) * (Distance B D) ↔
  (IsMidpointOfArc C A B ∨ IsMidpointOfArc D A B) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_l1203_120372


namespace NUMINAMATH_CALUDE_rectangle_division_integer_dimension_l1203_120331

/-- A rectangle with dimensions a and b can be divided into unit-width strips -/
structure RectangleDivision (a b : ℝ) : Prop where
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (can_divide : ∃ (strips : Set (ℝ × ℝ)), 
    (∀ s ∈ strips, (s.1 = 1 ∨ s.2 = 1)) ∧ 
    (∀ x y, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b → 
      ∃ s ∈ strips, (0 ≤ x - s.1 ∧ x < s.1) ∧ (0 ≤ y - s.2 ∧ y < s.2)))

/-- If a rectangle can be divided into unit-width strips, then one of its dimensions is an integer -/
theorem rectangle_division_integer_dimension (a b : ℝ) 
  (h : RectangleDivision a b) : 
  ∃ n : ℕ, (a = n) ∨ (b = n) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_integer_dimension_l1203_120331


namespace NUMINAMATH_CALUDE_function_form_proof_l1203_120303

theorem function_form_proof (f : ℝ → ℝ) (k : ℝ) 
  (h_continuous : Continuous f)
  (h_zero : f 0 = 0)
  (h_inequality : ∀ x y, f (x + y) ≥ f x + f y + k * x * y) :
  ∃ b : ℝ, ∀ x, f x = k / 2 * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_function_form_proof_l1203_120303


namespace NUMINAMATH_CALUDE_farmer_max_animals_l1203_120363

/-- Represents the farmer's animal purchasing problem --/
def FarmerProblem (budget goatCost sheepCost : ℕ) : Prop :=
  ∃ (goats sheep : ℕ),
    goats > 0 ∧
    sheep > 0 ∧
    goats = 2 * sheep ∧
    goatCost * goats + sheepCost * sheep ≤ budget ∧
    ∀ (g s : ℕ),
      g > 0 →
      s > 0 →
      g = 2 * s →
      goatCost * g + sheepCost * s ≤ budget →
      g + s ≤ goats + sheep

theorem farmer_max_animals :
  FarmerProblem 2000 35 40 →
  ∃ (goats sheep : ℕ),
    goats = 36 ∧
    sheep = 18 ∧
    goats + sheep = 54 ∧
    FarmerProblem 2000 35 40 :=
by sorry

end NUMINAMATH_CALUDE_farmer_max_animals_l1203_120363


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1203_120308

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 4 < 0) → (a < -4 ∨ a > 4) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1203_120308


namespace NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l1203_120355

theorem sum_x_y_equals_twenty (x y : ℝ) (h : (x + 1 + (y - 1)) / 2 = 10) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_twenty_l1203_120355


namespace NUMINAMATH_CALUDE_apple_distribution_l1203_120326

theorem apple_distribution (total_apples : ℕ) (new_people : ℕ) (apple_reduction : ℕ) 
  (h1 : total_apples = 2750)
  (h2 : new_people = 60)
  (h3 : apple_reduction = 12) :
  ∃ (original_people : ℕ),
    (total_apples / original_people : ℚ) - 
    (total_apples / (original_people + new_people) : ℚ) = apple_reduction ∧
    total_apples / original_people = 30 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l1203_120326
