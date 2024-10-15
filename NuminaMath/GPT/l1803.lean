import Mathlib

namespace NUMINAMATH_GPT_ff1_is_1_l1803_180332

noncomputable def f (x : ℝ) := Real.log x - 2 * x + 3

theorem ff1_is_1 : f (f 1) = 1 := by
  sorry

end NUMINAMATH_GPT_ff1_is_1_l1803_180332


namespace NUMINAMATH_GPT_inequality_for_positive_real_numbers_l1803_180398

theorem inequality_for_positive_real_numbers 
  (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  (a / (b + 2 * c + 3 * d) + 
   b / (c + 2 * d + 3 * a) + 
   c / (d + 2 * a + 3 * b) + 
   d / (a + 2 * b + 3 * c)) ≥ (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_positive_real_numbers_l1803_180398


namespace NUMINAMATH_GPT_distance_to_lateral_face_l1803_180399

theorem distance_to_lateral_face 
  (height : ℝ) 
  (angle : ℝ) 
  (h_height : height = 6 * Real.sqrt 6)
  (h_angle : angle = Real.pi / 4) : 
  ∃ (distance : ℝ), distance = 6 * Real.sqrt 30 / 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_lateral_face_l1803_180399


namespace NUMINAMATH_GPT_ralph_socks_l1803_180377

theorem ralph_socks (x y z : ℕ) (h1 : x + y + z = 12) (h2 : x + 3 * y + 4 * z = 24) (h3 : 1 ≤ x) (h4 : 1 ≤ y) (h5 : 1 ≤ z) : x = 7 :=
sorry

end NUMINAMATH_GPT_ralph_socks_l1803_180377


namespace NUMINAMATH_GPT_solution_set_inequality_l1803_180389

theorem solution_set_inequality (x : ℝ) : 
  x * (x - 1) ≥ x ↔ x ≤ 0 ∨ x ≥ 2 := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1803_180389


namespace NUMINAMATH_GPT_spending_on_games_l1803_180311

-- Definitions converted from conditions
def totalAllowance := 48
def fractionClothes := 1 / 4
def fractionBooks := 1 / 3
def fractionSnacks := 1 / 6
def spentClothes := fractionClothes * totalAllowance
def spentBooks := fractionBooks * totalAllowance
def spentSnacks := fractionSnacks * totalAllowance
def spentGames := totalAllowance - (spentClothes + spentBooks + spentSnacks)

-- The theorem that needs to be proven
theorem spending_on_games : spentGames = 12 :=
by sorry

end NUMINAMATH_GPT_spending_on_games_l1803_180311


namespace NUMINAMATH_GPT_longer_string_length_l1803_180321

theorem longer_string_length 
  (total_length : ℕ) 
  (length_diff : ℕ)
  (h_total_length : total_length = 348)
  (h_length_diff : length_diff = 72) :
  ∃ (L S : ℕ), 
  L - S = length_diff ∧
  L + S = total_length ∧ 
  L = 210 :=
by
  sorry

end NUMINAMATH_GPT_longer_string_length_l1803_180321


namespace NUMINAMATH_GPT_math_problem_l1803_180347

theorem math_problem (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y + x + y = 83) (h4 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 := by 
  sorry

end NUMINAMATH_GPT_math_problem_l1803_180347


namespace NUMINAMATH_GPT_tangent_line_parallel_coordinates_l1803_180381

theorem tangent_line_parallel_coordinates :
  ∃ (x y : ℝ), y = x^3 + x - 2 ∧ (3 * x^2 + 1 = 4) ∧ (x, y) = (-1, -4) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parallel_coordinates_l1803_180381


namespace NUMINAMATH_GPT_probability_of_earning_1900_equals_6_over_125_l1803_180330

-- Representation of a slot on the spinner.
inductive Slot
| Bankrupt 
| Dollar1000
| Dollar500
| Dollar4000
| Dollar400 
deriving DecidableEq

-- Condition: There are 5 slots and each has the same probability.
noncomputable def slots := [Slot.Bankrupt, Slot.Dollar1000, Slot.Dollar500, Slot.Dollar4000, Slot.Dollar400]

-- Probability of earning exactly $1900 in three spins.
def probability_of_1900 : ℚ :=
  let target_combination := [Slot.Dollar500, Slot.Dollar400, Slot.Dollar1000]
  let total_ways := 125
  let successful_ways := 6
  (successful_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_earning_1900_equals_6_over_125 :
  probability_of_1900 = 6 / 125 :=
sorry

end NUMINAMATH_GPT_probability_of_earning_1900_equals_6_over_125_l1803_180330


namespace NUMINAMATH_GPT_ex1_l1803_180392

theorem ex1 (a b : ℕ) (h₀ : a = 3) (h₁ : b = 4) : ∃ n : ℕ, 3^(7*a + b) = n^7 :=
by
  use 27
  sorry

end NUMINAMATH_GPT_ex1_l1803_180392


namespace NUMINAMATH_GPT_total_bird_count_correct_l1803_180334

-- Define initial counts
def initial_sparrows : ℕ := 89
def initial_pigeons : ℕ := 68
def initial_finches : ℕ := 74

-- Define additional birds
def additional_sparrows : ℕ := 42
def additional_pigeons : ℕ := 51
def additional_finches : ℕ := 27

-- Define total counts
def initial_total : ℕ := 231
def final_total : ℕ := 312

theorem total_bird_count_correct :
  initial_sparrows + initial_pigeons + initial_finches = initial_total ∧
  (initial_sparrows + additional_sparrows) + 
  (initial_pigeons + additional_pigeons) + 
  (initial_finches + additional_finches) = final_total := by
    sorry

end NUMINAMATH_GPT_total_bird_count_correct_l1803_180334


namespace NUMINAMATH_GPT_sum_of_numbers_l1803_180317

theorem sum_of_numbers (a b c : ℝ) 
  (h₁ : a^2 + b^2 + c^2 = 62) 
  (h₂ : ab + bc + ca = 131) : 
  a + b + c = 18 :=
sorry

end NUMINAMATH_GPT_sum_of_numbers_l1803_180317


namespace NUMINAMATH_GPT_problem_statement_l1803_180378

noncomputable def tangent_sum_formula (x y : ℝ) : ℝ :=
  (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)

theorem problem_statement
  (α β : ℝ)
  (hαβ1 : 0 < α ∧ α < π)
  (hαβ2 : 0 < β ∧ β < π)
  (h1 : Real.tan (α - β) = 1 / 2)
  (h2 : Real.tan β = - 1 / 7)
  : 2 * α - β = - (3 * π / 4) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1803_180378


namespace NUMINAMATH_GPT_area_excircle_gteq_four_times_area_l1803_180338

-- Define the area function
def area (A B C : Point) : ℝ := sorry -- Area of triangle ABC (this will be implemented later)

-- Define the centers of the excircles (this needs precise definitions and setup)
def excircle_center (A B C : Point) : Point := sorry -- Centers of the excircles of triangle ABC (implementation would follow)

-- Define the area of the triangle formed by the excircle centers
def excircle_area (A B C : Point) : ℝ :=
  let O1 := excircle_center A B C
  let O2 := excircle_center B C A
  let O3 := excircle_center C A B
  area O1 O2 O3

-- Prove the main statement
theorem area_excircle_gteq_four_times_area (A B C : Point) :
  excircle_area A B C ≥ 4 * area A B C :=
by sorry

end NUMINAMATH_GPT_area_excircle_gteq_four_times_area_l1803_180338


namespace NUMINAMATH_GPT_investment_duration_p_l1803_180372

-- Given the investments ratio, profits ratio, and time period for q,
-- proving the time period of p's investment is 7 months.
theorem investment_duration_p (T_p T_q : ℕ) 
  (investment_ratio : 7 * T_p = 5 * T_q) 
  (profit_ratio : 7 * T_p / T_q = 7 / 10)
  (T_q_eq : T_q = 14) : T_p = 7 :=
by
  sorry

end NUMINAMATH_GPT_investment_duration_p_l1803_180372


namespace NUMINAMATH_GPT_number_of_color_copies_l1803_180342

def charge_shop_X (n : ℕ) : ℝ := 1.20 * n
def charge_shop_Y (n : ℕ) : ℝ := 1.70 * n
def difference := 20

theorem number_of_color_copies (n : ℕ) (h : charge_shop_Y n = charge_shop_X n + difference) : n = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_color_copies_l1803_180342


namespace NUMINAMATH_GPT_prove_inequality_l1803_180375

variable (f : ℝ → ℝ)

-- Conditions
axiom condition : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

-- Proof of the desired statement
theorem prove_inequality : ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
by
  sorry

end NUMINAMATH_GPT_prove_inequality_l1803_180375


namespace NUMINAMATH_GPT_eggs_per_meal_l1803_180325

noncomputable def initial_eggs_from_store : ℕ := 12
noncomputable def additional_eggs_from_neighbor : ℕ := 12
noncomputable def eggs_used_for_cooking : ℕ := 2 + 4
noncomputable def remaining_eggs_after_cooking : ℕ := initial_eggs_from_store + additional_eggs_from_neighbor - eggs_used_for_cooking
noncomputable def eggs_given_to_aunt : ℕ := remaining_eggs_after_cooking / 2
noncomputable def remaining_eggs_after_giving_to_aunt : ℕ := remaining_eggs_after_cooking - eggs_given_to_aunt
noncomputable def planned_meals : ℕ := 3

theorem eggs_per_meal : remaining_eggs_after_giving_to_aunt / planned_meals = 3 := 
by 
  sorry

end NUMINAMATH_GPT_eggs_per_meal_l1803_180325


namespace NUMINAMATH_GPT_isosceles_triangle_area_l1803_180352

theorem isosceles_triangle_area (s b : ℝ) (h₁ : s + b = 20) (h₂ : b^2 + 10^2 = s^2) : 
  1/2 * 2 * b * 10 = 75 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l1803_180352


namespace NUMINAMATH_GPT_ram_marks_l1803_180382

theorem ram_marks (total_marks : ℕ) (percentage : ℕ) (h_total : total_marks = 500) (h_percentage : percentage = 90) : 
  (percentage * total_marks / 100) = 450 := by
  sorry

end NUMINAMATH_GPT_ram_marks_l1803_180382


namespace NUMINAMATH_GPT_exists_line_intersecting_circle_and_passing_origin_l1803_180313

theorem exists_line_intersecting_circle_and_passing_origin :
  ∃ m : ℝ, (m = 1 ∨ m = -4) ∧ 
  ∃ (x y : ℝ), 
    ((x - 1) ^ 2 + (y + 2) ^ 2 = 9) ∧ 
    ((x - y + m = 0) ∧ 
     ∃ (x' y' : ℝ),
      ((x' - 1) ^ 2 + (y' + 2) ^ 2 = 9) ∧ 
      ((x' - y' + m = 0) ∧ ((x + x') / 2 = 0 ∧ (y + y') / 2 = 0))) :=
by 
  sorry

end NUMINAMATH_GPT_exists_line_intersecting_circle_and_passing_origin_l1803_180313


namespace NUMINAMATH_GPT_minimum_area_of_quadrilateral_l1803_180379

theorem minimum_area_of_quadrilateral
  (ABCD : Type)
  (O : Type)
  (S_ABO : ℝ)
  (S_CDO : ℝ)
  (BC : ℝ)
  (cos_angle_ADC : ℝ)
  (h1 : S_ABO = 3 / 2)
  (h2 : S_CDO = 3 / 2)
  (h3 : BC = 3 * Real.sqrt 2)
  (h4 : cos_angle_ADC = 3 / Real.sqrt 10) :
  ∃ S_ABCD : ℝ, S_ABCD = 6 :=
sorry

end NUMINAMATH_GPT_minimum_area_of_quadrilateral_l1803_180379


namespace NUMINAMATH_GPT_ms_smith_books_divided_l1803_180380

theorem ms_smith_books_divided (books_for_girls : ℕ) (girls boys : ℕ) (books_per_girl : ℕ)
  (h1 : books_for_girls = 225)
  (h2 : girls = 15)
  (h3 : boys = 10)
  (h4 : books_for_girls / girls = books_per_girl)
  (h5 : books_per_girl * boys + books_for_girls = 375) : 
  books_for_girls / girls * (girls + boys) = 375 := 
by
  sorry

end NUMINAMATH_GPT_ms_smith_books_divided_l1803_180380


namespace NUMINAMATH_GPT_election_winning_candidate_votes_l1803_180319

theorem election_winning_candidate_votes (V : ℕ) 
  (h1 : V = (4 / 7) * V + 2000 + 4000) : 
  (4 / 7) * V = 8000 :=
by
  sorry

end NUMINAMATH_GPT_election_winning_candidate_votes_l1803_180319


namespace NUMINAMATH_GPT_minimum_k_l1803_180386

theorem minimum_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℝ) :
  a 1 = (1/2) ∧ (∀ n, 2 * a (n + 1) + S n = 0) ∧ (∀ n, S n ≤ k) → k = (1/2) :=
sorry

end NUMINAMATH_GPT_minimum_k_l1803_180386


namespace NUMINAMATH_GPT_domain_of_sqrt_one_minus_ln_l1803_180301

def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ Real.exp 1

theorem domain_of_sqrt_one_minus_ln (x : ℝ) : (1 - Real.log x ≥ 0) ∧ (x > 0) ↔ domain x := by
sorry

end NUMINAMATH_GPT_domain_of_sqrt_one_minus_ln_l1803_180301


namespace NUMINAMATH_GPT_equation_of_hyperbola_l1803_180359

-- Definitions for conditions

def center_at_origin (center : ℝ × ℝ) : Prop :=
  center = (0, 0)

def focus_point (focus : ℝ × ℝ) : Prop :=
  focus = (Real.sqrt 2, 0)

def distance_to_asymptote (focus : ℝ × ℝ) (distance : ℝ) : Prop :=
  -- Placeholder for the actual distance calculation
  distance = 1 -- The given distance condition in the problem

-- The mathematical proof problem statement

theorem equation_of_hyperbola :
  center_at_origin (0,0) ∧
  focus_point (Real.sqrt 2, 0) ∧
  distance_to_asymptote (Real.sqrt 2, 0) 1 → 
    ∃ a b : ℝ, (a > 0) ∧ (b > 0) ∧
    (a^2 + b^2 = 2) ∧ (a^2 = 1) ∧ (b^2 = 1) ∧ 
    (∀ x y : ℝ, b^2*y^2 = x^2 - a^2*y^2 → (y = 0 ∧ x^2 = 1)) :=
sorry

end NUMINAMATH_GPT_equation_of_hyperbola_l1803_180359


namespace NUMINAMATH_GPT_a_in_A_l1803_180369

def A := {x : ℝ | x ≥ 2 * Real.sqrt 2}
def a : ℝ := 3

theorem a_in_A : a ∈ A :=
by 
  sorry

end NUMINAMATH_GPT_a_in_A_l1803_180369


namespace NUMINAMATH_GPT_total_tickets_sold_l1803_180357

-- Definitions of the conditions as given in the problem
def price_adult : ℕ := 7
def price_child : ℕ := 4
def total_revenue : ℕ := 5100
def child_tickets_sold : ℕ := 400

-- The main statement (theorem) to prove
theorem total_tickets_sold:
  ∃ (A C : ℕ), C = child_tickets_sold ∧ price_adult * A + price_child * C = total_revenue ∧ (A + C = 900) :=
by
  sorry

end NUMINAMATH_GPT_total_tickets_sold_l1803_180357


namespace NUMINAMATH_GPT_smallest_number_l1803_180335

theorem smallest_number (n : ℕ) : 
  (∀ k ∈ [12, 16, 18, 21, 28, 35, 39], ∃ m : ℕ, (n - 3) = k * m) → 
  n = 65517 := by
  sorry

end NUMINAMATH_GPT_smallest_number_l1803_180335


namespace NUMINAMATH_GPT_greatest_measure_length_l1803_180363

theorem greatest_measure_length :
  let l1 := 18000
  let l2 := 50000
  let l3 := 1520
  ∃ d, d = Int.gcd (Int.gcd l1 l2) l3 ∧ d = 40 :=
by
  sorry

end NUMINAMATH_GPT_greatest_measure_length_l1803_180363


namespace NUMINAMATH_GPT_find_x_l1803_180383

noncomputable def arctan := Real.arctan

theorem find_x :
  (∃ x : ℝ, 3 * arctan (1 / 4) + arctan (1 / 5) + arctan (1 / x) = π / 4 ∧ x = -250 / 37) :=
  sorry

end NUMINAMATH_GPT_find_x_l1803_180383


namespace NUMINAMATH_GPT_number_of_four_digit_numbers_l1803_180323

/-- The number of four-digit numbers greater than 3999 where the product of the middle two digits exceeds 8 is 2400. -/
def four_digit_numbers_with_product_exceeding_8 : Nat :=
  let first_digit_options := 6  -- Choices are 4,5,6,7,8,9
  let last_digit_options := 10  -- Choices are 0,1,...,9
  let middle_digit_options := 40  -- Valid pairs counted from the solution
  first_digit_options * middle_digit_options * last_digit_options

theorem number_of_four_digit_numbers : four_digit_numbers_with_product_exceeding_8 = 2400 :=
by
  sorry

end NUMINAMATH_GPT_number_of_four_digit_numbers_l1803_180323


namespace NUMINAMATH_GPT_equalize_costs_l1803_180393

variable (L B C : ℝ)
variable (h1 : L < B)
variable (h2 : B < C)

theorem equalize_costs : (B + C - 2 * L) / 3 = ((L + B + C) / 3 - L) :=
by sorry

end NUMINAMATH_GPT_equalize_costs_l1803_180393


namespace NUMINAMATH_GPT_max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l1803_180306

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x

theorem max_value_of_f : ∃ x, (f x) = 1/2 :=
sorry

theorem period_of_f : ∀ x, f (x + π) = f x :=
sorry

theorem not_monotonically_increasing : ¬ ∀ x y, 0 < x ∧ x < y ∧ y < π/2 → f x < f y :=
sorry

theorem incorrect_zeros : ∃ x y z, (0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ π) ∧ (f x = 0 ∧ f y = 0 ∧ f z = 0) :=
sorry

end NUMINAMATH_GPT_max_value_of_f_period_of_f_not_monotonically_increasing_incorrect_zeros_l1803_180306


namespace NUMINAMATH_GPT_calculate_expression_l1803_180339

/-- Calculate the expression 2197 + 180 ÷ 60 × 3 - 197. -/
theorem calculate_expression : 2197 + (180 / 60) * 3 - 197 = 2009 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1803_180339


namespace NUMINAMATH_GPT_page_problem_insufficient_information_l1803_180385

theorem page_problem_insufficient_information
  (total_problems : ℕ)
  (finished_problems : ℕ)
  (remaining_pages : ℕ)
  (x y : ℕ)
  (O E : ℕ)
  (h1 : total_problems = 450)
  (h2 : finished_problems = 185)
  (h3 : remaining_pages = 15)
  (h4 : O + E = remaining_pages)
  (h5 : O * x + E * y = total_problems - finished_problems) :
  ∀ (x y : ℕ), O * x + E * y = 265 → x = x ∧ y = y :=
by
  sorry

end NUMINAMATH_GPT_page_problem_insufficient_information_l1803_180385


namespace NUMINAMATH_GPT_two_digit_sum_l1803_180314

theorem two_digit_sum (x y n : ℕ) (hx : 10 ≤ x ∧ x < 100)
  (hy : 10 ≤ y ∧ y < 100) (h_rev : y = (x % 10) * 10 + x / 10)
  (h_diff_square : x^2 - y^2 = n^2) : x + y + n = 154 :=
sorry

end NUMINAMATH_GPT_two_digit_sum_l1803_180314


namespace NUMINAMATH_GPT_integral_curves_l1803_180300

theorem integral_curves (y x : ℝ) : 
  (∃ k : ℝ, (y - x) / (y + x) = k) → 
  (∃ c : ℝ, y = x * (c + 1) / (c - 1)) ∨ (y = 0) ∨ (y = x) ∨ (x = 0) :=
by
  sorry

end NUMINAMATH_GPT_integral_curves_l1803_180300


namespace NUMINAMATH_GPT_pascal_sum_of_squares_of_interior_l1803_180362

theorem pascal_sum_of_squares_of_interior (eighth_row_interior : List ℕ) 
    (h : eighth_row_interior = [7, 21, 35, 35, 21, 7]) : 
    (eighth_row_interior.map (λ x => x * x)).sum = 3430 := 
by
  sorry

end NUMINAMATH_GPT_pascal_sum_of_squares_of_interior_l1803_180362


namespace NUMINAMATH_GPT_smallest_number_mod_l1803_180391

theorem smallest_number_mod (x : ℕ) :
  (x % 2 = 1) → (x % 3 = 2) → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_mod_l1803_180391


namespace NUMINAMATH_GPT_product_of_two_numbers_l1803_180384

theorem product_of_two_numbers (x y : ℝ) (h₁ : x + y = 23) (h₂ : x^2 + y^2 = 289) : x * y = 120 := by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1803_180384


namespace NUMINAMATH_GPT_probability_condition_l1803_180367

namespace SharedPowerBank

def P (event : String) : ℚ :=
  match event with
  | "A" => 3 / 4
  | "B" => 1 / 2
  | _   => 0 -- Default case for any other event

def probability_greater_than_1000_given_greater_than_500 : ℚ :=
  P "B" / P "A"

theorem probability_condition :
  probability_greater_than_1000_given_greater_than_500 = 2 / 3 :=
by 
  sorry

end SharedPowerBank

end NUMINAMATH_GPT_probability_condition_l1803_180367


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1803_180358

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : 
  ∃ p, Nat.Prime p ∧ digits_sum p = 23 ∧ ∀ q, Nat.Prime q ∧ digits_sum q = 23 → p ≤ q :=
sorry

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1803_180358


namespace NUMINAMATH_GPT_least_pos_integer_with_8_factors_l1803_180356

theorem least_pos_integer_with_8_factors : 
  ∃ k : ℕ, (k > 0 ∧ ((∃ m n p q : ℕ, k = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, k = p^7 ∧ Prime p)) ∧ 
            ∀ l : ℕ, (l > 0 ∧ ((∃ m n p q : ℕ, l = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, l = p^7 ∧ Prime p)) → k ≤ l)) ∧ k = 24 :=
sorry

end NUMINAMATH_GPT_least_pos_integer_with_8_factors_l1803_180356


namespace NUMINAMATH_GPT_driving_time_ratio_l1803_180368

theorem driving_time_ratio 
  (t : ℝ)
  (h : 30 * t + 60 * (2 * t) = 75) : 
  t / (2 * t) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_driving_time_ratio_l1803_180368


namespace NUMINAMATH_GPT_range_of_T_l1803_180366

open Real

theorem range_of_T (x y z : ℝ) (h : x^2 + 2 * y^2 + 3 * z^2 = 4) : 
    - (2 * sqrt 6) / 3 ≤ x * y + y * z ∧ x * y + y * z ≤ (2 * sqrt 6) / 3 := 
by 
    sorry

end NUMINAMATH_GPT_range_of_T_l1803_180366


namespace NUMINAMATH_GPT_Juanita_spends_more_l1803_180318

def Grant_annual_spend : ℝ := 200
def weekday_spend : ℝ := 0.50
def sunday_spend : ℝ := 2.00
def weeks_in_year : ℝ := 52

def Juanita_weekly_spend : ℝ := (6 * weekday_spend) + sunday_spend
def Juanita_annual_spend : ℝ := weeks_in_year * Juanita_weekly_spend
def spending_difference : ℝ := Juanita_annual_spend - Grant_annual_spend

theorem Juanita_spends_more : spending_difference = 60 := by
  sorry

end NUMINAMATH_GPT_Juanita_spends_more_l1803_180318


namespace NUMINAMATH_GPT_no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l1803_180304

-- Part (1): Prove that there do not exist positive integers m and n such that m(m+2) = n(n+1)
theorem no_solutions_m_m_plus_2_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 2) = n * (n + 1) :=
sorry

-- Part (2): Given k ≥ 3,
-- Case (a): Prove that for k=3, there do not exist positive integers m and n such that m(m+3) = n(n+1)
theorem no_solutions_m_m_plus_3_eq_n_n_plus_1 : ¬ ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 3) = n * (n + 1) :=
sorry

-- Case (b): Prove that for k ≥ 4, there exist positive integers m and n such that m(m+k) = n(n+1)
theorem solutions_exist_m_m_plus_k_eq_n_n_plus_1 (k : ℕ) (h : k ≥ 4) : 
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1) :=
sorry

end NUMINAMATH_GPT_no_solutions_m_m_plus_2_eq_n_n_plus_1_no_solutions_m_m_plus_3_eq_n_n_plus_1_solutions_exist_m_m_plus_k_eq_n_n_plus_1_l1803_180304


namespace NUMINAMATH_GPT_subtract_vectors_l1803_180395

def vec_a : ℤ × ℤ × ℤ := (5, -3, 2)
def vec_b : ℤ × ℤ × ℤ := (-2, 4, 1)
def vec_result : ℤ × ℤ × ℤ := (9, -11, 0)

theorem subtract_vectors :
  vec_a - 2 • vec_b = vec_result :=
by sorry

end NUMINAMATH_GPT_subtract_vectors_l1803_180395


namespace NUMINAMATH_GPT_parallelogram_base_length_l1803_180374

theorem parallelogram_base_length (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_eq_2b : h = 2 * b) (area_eq_98 : area = 98) 
  (area_def : area = b * h) : b = 7 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_base_length_l1803_180374


namespace NUMINAMATH_GPT_isosceles_trapezoid_diagonal_length_l1803_180329

theorem isosceles_trapezoid_diagonal_length
  (AB CD : ℝ) (AD BC : ℝ) :
  AB = 15 →
  CD = 9 →
  AD = 12 →
  BC = 12 →
  (AC : ℝ) = Real.sqrt 279 :=
by
  intros hAB hCD hAD hBC
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_diagonal_length_l1803_180329


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_a_range_l1803_180388

section
variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) := |2 * x - a| + a

-- Problem 1
theorem problem1_solution_set (h : a = 3) : {x | f x a ≤ 6} = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

def g (x : ℝ) := |2 * x - 3|

-- Problem 2
theorem problem2_a_range : ∀ a : ℝ, ∀ x : ℝ, f x a + g x ≥ 5 ↔ 4 ≤ a :=
by
  sorry
end

end NUMINAMATH_GPT_problem1_solution_set_problem2_a_range_l1803_180388


namespace NUMINAMATH_GPT_number_of_girls_l1803_180365

theorem number_of_girls (d c : ℕ) (h1 : c = 2 * (d - 15)) (h2 : d - 15 = 5 * (c - 45)) : d = 40 := 
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l1803_180365


namespace NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1803_180316

variable {a_n : ℕ → ℝ}
variable {b_n : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n + d

def is_geometric_sequence (b_n : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, b_n (n + 1) = b_n n * r

def arithmetic_to_geometric (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) : Prop :=
  b_n 0 = a_n 2 ∧ b_n 1 = a_n 3 ∧ b_n 2 = a_n 7

-- Mathematical Proof Problem
theorem common_ratio_of_geometric_sequence :
  ∀ (a_n : ℕ → ℝ) (d : ℝ), d ≠ 0 →
  is_arithmetic_sequence a_n d →
  (∃ (b_n : ℕ → ℝ) (r : ℝ), arithmetic_to_geometric a_n b_n ∧ is_geometric_sequence b_n r) →
  ∃ r, r = 4 :=
sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_sequence_l1803_180316


namespace NUMINAMATH_GPT_exists_element_x_l1803_180351

open Set

theorem exists_element_x (n : ℕ) (S : Finset (Fin n)) (A : Fin n → Finset (Fin n)) 
  (h_distinct : ∀ i j : Fin n, i ≠ j → A i ≠ A j) : 
  ∃ x ∈ S, ∀ i j : Fin n, i ≠ j → (A i \ {x}) ≠ (A j \ {x}) :=
sorry

end NUMINAMATH_GPT_exists_element_x_l1803_180351


namespace NUMINAMATH_GPT_principal_amount_l1803_180341

theorem principal_amount (P R T SI : ℝ) (hR : R = 4) (hT : T = 5) (hSI : SI = P - 2240) 
    (h_formula : SI = (P * R * T) / 100) : P = 2800 :=
by 
  sorry

end NUMINAMATH_GPT_principal_amount_l1803_180341


namespace NUMINAMATH_GPT_probability_none_solve_l1803_180331

theorem probability_none_solve (a b c : ℕ) (ha : 0 < a ∧ a < 10)
                               (hb : 0 < b ∧ b < 10)
                               (hc : 0 < c ∧ c < 10)
                               (P_A : ℚ := 1 / a)
                               (P_B : ℚ := 1 / b)
                               (P_C : ℚ := 1 / c)
                               (H : (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15) :
                               -- Conclusion: The probability that none of them solve the problem is 8/15
                               (1 - (1 / a)) * (1 - (1 / b)) * (1 - (1 / c)) = 8 / 15 :=
sorry

end NUMINAMATH_GPT_probability_none_solve_l1803_180331


namespace NUMINAMATH_GPT_part1_l1803_180350

theorem part1 (a b c t m n : ℝ) (h1 : a > 0) (h2 : m = n) (h3 : t = (3 + (t + 1)) / 2) : t = 4 :=
sorry

end NUMINAMATH_GPT_part1_l1803_180350


namespace NUMINAMATH_GPT_relationship_p_q_l1803_180333

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem relationship_p_q (x a p q : ℝ) (h1 : 0 ≤ x) (h2 : x < 1) (h3 : a > 0) (h4 : a ≠ 1)
  (hp : p = |log_a a (1 + x)|) (hq : q = |log_a a (1 - x)|) : p ≤ q :=
sorry

end NUMINAMATH_GPT_relationship_p_q_l1803_180333


namespace NUMINAMATH_GPT_range_of_m_l1803_180340
noncomputable def f (x : ℝ) : ℝ := ((x - 1) / (x + 1))^2

noncomputable def f_inv (x : ℝ) : ℝ := (1 + Real.sqrt x) / (1 - Real.sqrt x)

theorem range_of_m {x : ℝ} (m : ℝ) (h1 : 1 / 16 ≤ x) (h2 : x ≤ 1 / 4) 
  (h3 : ∀ (x : ℝ), (1 - Real.sqrt x) * f_inv x > m * (m - Real.sqrt x)): 
  -1 < m ∧ m < 5 / 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1803_180340


namespace NUMINAMATH_GPT_number_of_ordered_quadruples_l1803_180364

theorem number_of_ordered_quadruples (x1 x2 x3 x4 : ℕ) (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0) (h4 : x4 > 0) (h_sum : x1 + x2 + x3 + x4 = 100) : 
  ∃ n : ℕ, n = 156849 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_ordered_quadruples_l1803_180364


namespace NUMINAMATH_GPT_baba_yaga_departure_and_speed_l1803_180315

variables (T : ℕ) (d : ℕ)

theorem baba_yaga_departure_and_speed :
  (50 * (T + 2) = 150 * (T - 2)) →
  (12 - T = 8) ∧ (d = 50 * (T + 2)) →
  (d = 300) ∧ ((d / T) = 75) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_baba_yaga_departure_and_speed_l1803_180315


namespace NUMINAMATH_GPT_polynomial_simplification_l1803_180397

theorem polynomial_simplification :
  ∃ A B C D : ℤ,
  (∀ x : ℤ, x ≠ D → (x^3 + 5 * x^2 + 8 * x + 4) / (x + 1) = A * x^2 + B * x + C)
  ∧ (A + B + C + D = 8) :=
sorry

end NUMINAMATH_GPT_polynomial_simplification_l1803_180397


namespace NUMINAMATH_GPT_wire_cut_problem_l1803_180327

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ :=
  let x := total_length / (1 + ratio)
  x

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 35 → ratio = 5/2 → shorter_length = 10 → shorter_piece_length total_length ratio = shorter_length := by
  intros h1 h2 h3
  unfold shorter_piece_length
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_wire_cut_problem_l1803_180327


namespace NUMINAMATH_GPT_equality_conditions_l1803_180371

theorem equality_conditions (a b c d : ℝ) :
  a + bcd = (a + b) * (a + c) * (a + d) ↔ a = 0 ∨ a^2 + a * (b + c + d) + bc + bd + cd = 1 :=
by
  sorry

end NUMINAMATH_GPT_equality_conditions_l1803_180371


namespace NUMINAMATH_GPT_selling_price_is_correct_l1803_180376

noncomputable def cost_price : ℝ := 192
def profit_percentage : ℝ := 0.25
def profit (cp : ℝ) (pp : ℝ) : ℝ := pp * cp
def selling_price (cp : ℝ) (pft : ℝ) : ℝ := cp + pft

theorem selling_price_is_correct : selling_price cost_price (profit cost_price profit_percentage) = 240 :=
sorry

end NUMINAMATH_GPT_selling_price_is_correct_l1803_180376


namespace NUMINAMATH_GPT_each_mouse_not_visit_with_every_other_once_l1803_180310

theorem each_mouse_not_visit_with_every_other_once : 
    (∃ mice: Finset ℕ, mice.card = 24 ∧ (∀ f : ℕ → Finset ℕ, 
    (∀ n, (f n).card = 4) ∧ 
    (∀ i j, i ≠ j → (f i ∩ f j ≠ ∅) → (f i ∩ f j).card ≠ (mice.card - 1)))
    ) → false := 
by
  sorry

end NUMINAMATH_GPT_each_mouse_not_visit_with_every_other_once_l1803_180310


namespace NUMINAMATH_GPT_milk_production_days_l1803_180387

theorem milk_production_days (y : ℕ) :
  (y + 4) * (y + 2) * (y + 6) / (y * (y + 3) * (y + 4)) = y * (y + 3) * (y + 6) / ((y + 2) * (y + 4)) :=
sorry

end NUMINAMATH_GPT_milk_production_days_l1803_180387


namespace NUMINAMATH_GPT_dan_helmet_craters_l1803_180360

variable (D S : ℕ)
variable (h1 : D = S + 10)
variable (h2 : D + S + 15 = 75)

theorem dan_helmet_craters : D = 35 := by
  sorry

end NUMINAMATH_GPT_dan_helmet_craters_l1803_180360


namespace NUMINAMATH_GPT_find_x_in_equation_l1803_180396

theorem find_x_in_equation :
  ∃ x : ℝ, 2.5 * ( (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) ) = 2000.0000000000002 ∧ x = 0.3 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_in_equation_l1803_180396


namespace NUMINAMATH_GPT_g_of_minus_3_l1803_180353

noncomputable def f (x : ℝ) : ℝ := 4 * x - 7
noncomputable def g (y : ℝ) : ℝ := 3 * ((y + 7) / 4) ^ 2 + 4 * ((y + 7) / 4) + 1

theorem g_of_minus_3 : g (-3) = 8 :=
by
  sorry

end NUMINAMATH_GPT_g_of_minus_3_l1803_180353


namespace NUMINAMATH_GPT_ammonium_iodide_requirement_l1803_180303

theorem ammonium_iodide_requirement :
  ∀ (NH4I KOH NH3 KI H2O : ℕ),
  (NH4I + KOH = NH3 + KI + H2O) → 
  (NH4I = 3) →
  (KOH = 3) →
  (NH3 = 3) →
  (KI = 3) →
  (H2O = 3) →
  NH4I = 3 :=
by
  intros NH4I KOH NH3 KI H2O reaction_balanced NH4I_req KOH_req NH3_prod KI_prod H2O_prod
  exact NH4I_req

end NUMINAMATH_GPT_ammonium_iodide_requirement_l1803_180303


namespace NUMINAMATH_GPT_determine_n_l1803_180349

theorem determine_n (n : ℕ) (h : 3^n = 3^2 * 9^4 * 81^3) : n = 22 := 
by
  sorry

end NUMINAMATH_GPT_determine_n_l1803_180349


namespace NUMINAMATH_GPT_center_of_circle_l1803_180307

theorem center_of_circle :
  ∀ (x y : ℝ), (x + 2)^2 + (y - 1)^2 = 1 → (x = -2 ∧ y = 1) :=
by
  intros x y hyp
  -- Here, we would perform the steps of comparing to the standard form and proving the center.
  sorry

end NUMINAMATH_GPT_center_of_circle_l1803_180307


namespace NUMINAMATH_GPT_rope_costs_purchasing_plans_minimum_cost_l1803_180320

theorem rope_costs (x y m : ℕ) :
  (10 * x + 5 * y = 175) →
  (15 * x + 10 * y = 300) →
  x = 10 ∧ y = 15 :=
sorry

theorem purchasing_plans (m : ℕ) :
  (10 * 10 + 15 * 15 = 300) →
  23 ≤ m ∧ m ≤ 25 :=
sorry

theorem minimum_cost (m : ℕ) :
  (23 ≤ m ∧ m ≤ 25) →
  m = 25 →
  10 * m + 15 * (45 - m) = 550 :=
sorry

end NUMINAMATH_GPT_rope_costs_purchasing_plans_minimum_cost_l1803_180320


namespace NUMINAMATH_GPT_circle_diameter_from_area_l1803_180354

theorem circle_diameter_from_area (A r d : ℝ) (hA : A = 64 * Real.pi) (h_area : A = Real.pi * r^2) : d = 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_from_area_l1803_180354


namespace NUMINAMATH_GPT_sum_reciprocals_transformed_roots_l1803_180346

theorem sum_reciprocals_transformed_roots (a b c : ℝ) (h : ∀ x, (x^3 - 2 * x - 5 = 0) → (x = a) ∨ (x = b) ∨ (x = c)) : 
  (1 / (a - 2)) + (1 / (b - 2)) + (1 / (c - 2)) = 10 := 
by sorry

end NUMINAMATH_GPT_sum_reciprocals_transformed_roots_l1803_180346


namespace NUMINAMATH_GPT_total_dinners_l1803_180370

def monday_dinners := 40
def tuesday_dinners := monday_dinners + 40
def wednesday_dinners := tuesday_dinners / 2
def thursday_dinners := wednesday_dinners + 3

theorem total_dinners : monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners = 203 := by
  sorry

end NUMINAMATH_GPT_total_dinners_l1803_180370


namespace NUMINAMATH_GPT_percentage_entree_cost_l1803_180361

-- Conditions
def total_spent : ℝ := 50.0
def num_appetizers : ℝ := 2
def cost_per_appetizer : ℝ := 5.0
def total_appetizer_cost : ℝ := num_appetizers * cost_per_appetizer
def total_entree_cost : ℝ := total_spent - total_appetizer_cost

-- Proof Problem
theorem percentage_entree_cost :
  (total_entree_cost / total_spent) * 100 = 80 :=
sorry

end NUMINAMATH_GPT_percentage_entree_cost_l1803_180361


namespace NUMINAMATH_GPT_number_is_100_l1803_180324

theorem number_is_100 (x : ℝ) (h : 0.60 * (3 / 5) * x = 36) : x = 100 :=
by sorry

end NUMINAMATH_GPT_number_is_100_l1803_180324


namespace NUMINAMATH_GPT_expression_equals_one_l1803_180302

theorem expression_equals_one : 
  (Real.sqrt 6 / Real.sqrt 2) + abs (1 - Real.sqrt 3) - Real.sqrt 12 + (1 / 2)⁻¹ = 1 := 
by sorry

end NUMINAMATH_GPT_expression_equals_one_l1803_180302


namespace NUMINAMATH_GPT_problem_statement_l1803_180337

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1/a) + Real.sqrt (b + 1/b) + Real.sqrt (c + 1/c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1803_180337


namespace NUMINAMATH_GPT_max_value_of_reciprocal_sums_of_zeros_l1803_180355

noncomputable def quadratic_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + 2 * x - 1

noncomputable def linear_part (k : ℝ) (x : ℝ) : ℝ :=
  k * x + 1

theorem max_value_of_reciprocal_sums_of_zeros (k : ℝ) (x1 x2 : ℝ)
  (h0 : -1 < k ∧ k < 0)
  (hx1 : x1 ∈ Set.Ioc 0 1 → quadratic_part k x1 = 0)
  (hx2 : x2 ∈ Set.Ioi 1 → linear_part k x2 = 0)
  (hx_distinct : x1 ≠ x2) :
  (1 / x1) + (1 / x2) = 9 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_of_reciprocal_sums_of_zeros_l1803_180355


namespace NUMINAMATH_GPT_cos_double_angle_identity_l1803_180326

theorem cos_double_angle_identity (x : ℝ) (h : Real.sin (Real.pi / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_identity_l1803_180326


namespace NUMINAMATH_GPT_evaluate_f_at_3_l1803_180343

noncomputable def f : ℝ → ℝ := sorry

axiom h_odd : ∀ x : ℝ, f (-x) = -f x 
axiom h_periodic : ∀ x : ℝ, f x = f (x + 4)
axiom h_def : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem evaluate_f_at_3 : f 3 = -2 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_3_l1803_180343


namespace NUMINAMATH_GPT_fish_served_l1803_180390

theorem fish_served (H E P : ℕ) 
  (h1 : H = E) (h2 : E = P) 
  (fat_herring fat_eel fat_pike total_fat : ℕ) 
  (herring_fat : fat_herring = 40) 
  (eel_fat : fat_eel = 20)
  (pike_fat : fat_pike = 30)
  (total_fat_served : total_fat = 3600) 
  (fat_eq : 40 * H + 20 * E + 30 * P = 3600) : 
  H = 40 ∧ E = 40 ∧ P = 40 := by
  sorry

end NUMINAMATH_GPT_fish_served_l1803_180390


namespace NUMINAMATH_GPT_chessboard_property_exists_l1803_180305

theorem chessboard_property_exists (n : ℕ) (x : Fin n → Fin n → ℝ) 
  (h : ∀ i j k : Fin n, x i j + x j k + x k i = 0) :
  ∃ (t : Fin n → ℝ), ∀ i j, x i j = t i - t j := 
sorry

end NUMINAMATH_GPT_chessboard_property_exists_l1803_180305


namespace NUMINAMATH_GPT_increasing_geometric_progression_l1803_180309

noncomputable def golden_ratio : ℝ := (Real.sqrt 5 + 1) / 2

theorem increasing_geometric_progression (a : ℝ) (ha : 0 < a)
  (h1 : ∃ b c q : ℝ, b = Int.floor a ∧ c = a - b ∧ a = b + c ∧ c = b * q ∧ a = c * q ∧ 1 < q) : 
  a = golden_ratio :=
sorry

end NUMINAMATH_GPT_increasing_geometric_progression_l1803_180309


namespace NUMINAMATH_GPT_problem_statement_l1803_180344

noncomputable def necessary_but_not_sufficient_condition (x y : ℝ) (hx : x > 0) : Prop :=
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|)

theorem problem_statement
  (x y : ℝ)
  (hx : x > 0)
  : necessary_but_not_sufficient_condition x y hx :=
sorry

end NUMINAMATH_GPT_problem_statement_l1803_180344


namespace NUMINAMATH_GPT_abe_family_total_yen_l1803_180345

theorem abe_family_total_yen (yen_checking : ℕ) (yen_savings : ℕ) (h₁ : yen_checking = 6359) (h₂ : yen_savings = 3485) : yen_checking + yen_savings = 9844 :=
by
  sorry

end NUMINAMATH_GPT_abe_family_total_yen_l1803_180345


namespace NUMINAMATH_GPT_solutions_eq1_solutions_eq2_l1803_180322

noncomputable def equation_sol1 : Set ℝ :=
{ x | x^2 - 8 * x + 1 = 0 }

noncomputable def equation_sol2 : Set ℝ :=
{ x | x * (x - 2) - x + 2 = 0 }

theorem solutions_eq1 : ∀ x ∈ equation_sol1, x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by
  intro x hx
  sorry

theorem solutions_eq2 : ∀ x ∈ equation_sol2, x = 2 ∨ x = 1 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_solutions_eq1_solutions_eq2_l1803_180322


namespace NUMINAMATH_GPT_multiplication_correct_l1803_180348

theorem multiplication_correct : 121 * 54 = 6534 := by
  sorry

end NUMINAMATH_GPT_multiplication_correct_l1803_180348


namespace NUMINAMATH_GPT_shaded_fraction_l1803_180312

noncomputable def fraction_shaded (l w : ℝ) : ℝ :=
  1 - (1 / 8)

theorem shaded_fraction (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  fraction_shaded l w = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_shaded_fraction_l1803_180312


namespace NUMINAMATH_GPT_find_jordana_and_james_age_l1803_180328

variable (current_age_of_Jennifer : ℕ) (current_age_of_Jordana : ℕ) (current_age_of_James : ℕ)

-- Conditions
axiom jennifer_40_in_twenty_years : current_age_of_Jennifer + 20 = 40
axiom jordana_twice_jennifer_in_twenty_years : current_age_of_Jordana + 20 = 2 * (current_age_of_Jennifer + 20)
axiom james_ten_years_younger_in_twenty_years : current_age_of_James + 20 = 
  (current_age_of_Jennifer + 20) + (current_age_of_Jordana + 20) - 10

-- Prove that Jordana is currently 60 years old and James is currently 90 years old
theorem find_jordana_and_james_age : current_age_of_Jordana = 60 ∧ current_age_of_James = 90 :=
  sorry

end NUMINAMATH_GPT_find_jordana_and_james_age_l1803_180328


namespace NUMINAMATH_GPT_AK_eq_CK_l1803_180373

variable {α : Type*} [LinearOrder α] [LinearOrder ℝ]

variable (A B C L K : ℝ)
variable (triangle : ℝ)
variable (h₁ : AL = LB)
variable (h₂ : AK = CL)

--  Given that in triangle ABC,
--     AL is a bisector such that AL = LB,
--     and AK is on ray AL with AK = CL,
--     prove that AK = CK.
theorem AK_eq_CK (h₁ : AL = LB) (h₂ : AK = CL) : AK = CK := by
  sorry

end NUMINAMATH_GPT_AK_eq_CK_l1803_180373


namespace NUMINAMATH_GPT_no_common_factor_l1803_180394

open Polynomial

theorem no_common_factor (f g : ℤ[X]) : f = X^2 + X - 1 → g = X^2 + 2 * X → ∀ d : ℤ[X], d ∣ f ∧ d ∣ g → d = 1 :=
by
  intros h1 h2 d h_dv
  rw [h1, h2] at h_dv
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_no_common_factor_l1803_180394


namespace NUMINAMATH_GPT_find_m_l1803_180308

open Set

def A (m: ℝ) := {x : ℝ | x^2 - m * x + m^2 - 19 = 0}

def B := {x : ℝ | x^2 - 5 * x + 6 = 0}

def C := ({2, -4} : Set ℝ)

theorem find_m (m : ℝ) (ha : A m ∩ B ≠ ∅) (hb : A m ∩ C = ∅) : m = -2 :=
  sorry

end NUMINAMATH_GPT_find_m_l1803_180308


namespace NUMINAMATH_GPT_largest_possible_number_of_neither_l1803_180336

theorem largest_possible_number_of_neither
  (writers : ℕ)
  (editors : ℕ)
  (attendees : ℕ)
  (x : ℕ)
  (N : ℕ)
  (h_writers : writers = 45)
  (h_editors_gt : editors > 38)
  (h_attendees : attendees = 90)
  (h_both : N = 2 * x)
  (h_equation : writers + editors - x + N = attendees) :
  N = 12 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_number_of_neither_l1803_180336
