import Mathlib

namespace NUMINAMATH_GPT_giyoon_above_average_subjects_l250_25028

def points_korean : ℕ := 80
def points_mathematics : ℕ := 94
def points_social_studies : ℕ := 82
def points_english : ℕ := 76
def points_science : ℕ := 100
def number_of_subjects : ℕ := 5

def total_points : ℕ := points_korean + points_mathematics + points_social_studies + points_english + points_science
def average_points : ℚ := total_points / number_of_subjects

def count_above_average_points : ℕ := 
  (if points_korean > average_points then 1 else 0) + 
  (if points_mathematics > average_points then 1 else 0) +
  (if points_social_studies > average_points then 1 else 0) +
  (if points_english > average_points then 1 else 0) +
  (if points_science > average_points then 1 else 0)

theorem giyoon_above_average_subjects : count_above_average_points = 2 := by
  sorry

end NUMINAMATH_GPT_giyoon_above_average_subjects_l250_25028


namespace NUMINAMATH_GPT_wage_of_one_man_l250_25075

/-- Proof that the wage of one man is Rs. 24 given the conditions. -/
theorem wage_of_one_man (M W_w B_w : ℕ) (H1 : 120 = 5 * M + W_w * 5 + B_w * 8) 
  (H2 : 5 * M = W_w * 5) (H3 : W_w * 5 = B_w * 8) : M = 24 :=
by
  sorry

end NUMINAMATH_GPT_wage_of_one_man_l250_25075


namespace NUMINAMATH_GPT_number_of_pumps_l250_25041

theorem number_of_pumps (P : ℕ) : 
  (P * 8 * 2 = 8 * 6) → P = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_number_of_pumps_l250_25041


namespace NUMINAMATH_GPT_third_term_is_18_l250_25082

-- Define the first term and the common ratio
def a_1 : ℕ := 2
def q : ℕ := 3

-- Define the function for the nth term of an arithmetic-geometric sequence
def a_n (n : ℕ) : ℕ :=
  a_1 * q^(n-1)

-- Prove that the third term is 18
theorem third_term_is_18 : a_n 3 = 18 := by
  sorry

end NUMINAMATH_GPT_third_term_is_18_l250_25082


namespace NUMINAMATH_GPT_prices_and_subsidy_l250_25049

theorem prices_and_subsidy (total_cost : ℕ) (price_leather_jacket : ℕ) (price_sweater : ℕ) (subsidy_percentage : ℕ) 
  (leather_jacket_condition : price_leather_jacket = 5 * price_sweater + 600)
  (cost_condition : price_leather_jacket + price_sweater = total_cost)
  (total_sold : ℕ) (max_subsidy : ℕ) :
  (total_cost = 3000 ∧
   price_leather_jacket = 2600 ∧
   price_sweater = 400 ∧
   subsidy_percentage = 10) ∧ 
  ∃ a : ℕ, (2200 * a ≤ 50000 ∧ total_sold - a ≥ 128) :=
by
  sorry

end NUMINAMATH_GPT_prices_and_subsidy_l250_25049


namespace NUMINAMATH_GPT_boards_tested_l250_25040

-- Define the initial conditions and problem
def total_thumbtacks : ℕ := 450
def thumbtacks_remaining_each_can : ℕ := 30
def initial_thumbtacks_each_can := total_thumbtacks / 3
def thumbtacks_used_each_can := initial_thumbtacks_each_can - thumbtacks_remaining_each_can
def total_thumbtacks_used := thumbtacks_used_each_can * 3
def thumbtacks_per_board := 3

-- Define the proposition to prove 
theorem boards_tested (B : ℕ) : 
  (B = total_thumbtacks_used / thumbtacks_per_board) → B = 120 :=
by
  -- Proof skipped with sorry
  sorry

end NUMINAMATH_GPT_boards_tested_l250_25040


namespace NUMINAMATH_GPT_f_2019_value_l250_25070

noncomputable def B : Set ℚ := {q : ℚ | q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1}

noncomputable def g (x : ℚ) (h : x ∈ B) : ℚ :=
  1 - (2 / x)

noncomputable def f (x : ℚ) (h : x ∈ B) : ℝ :=
  sorry

theorem f_2019_value (h2019 : 2019 ∈ B) :
  f 2019 h2019 = Real.log ((2019 - 0.5) ^ 2 / 2018.5) :=
sorry

end NUMINAMATH_GPT_f_2019_value_l250_25070


namespace NUMINAMATH_GPT_quadratic_to_vertex_form_l250_25057

theorem quadratic_to_vertex_form:
  ∀ (x : ℝ), (x^2 - 4 * x + 3 = (x - 2)^2 - 1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_to_vertex_form_l250_25057


namespace NUMINAMATH_GPT_largest_number_in_set_l250_25025

theorem largest_number_in_set (b : ℕ) (h₀ : 2 + 6 + b = 18) (h₁ : 2 ≤ 6 ∧ 6 ≤ b):
  b = 10 :=
sorry

end NUMINAMATH_GPT_largest_number_in_set_l250_25025


namespace NUMINAMATH_GPT_required_amount_of_water_l250_25045

/-- 
Given:
- A solution of 12 ounces with 60% alcohol,
- A desired final concentration of 40% alcohol,

Prove:
- The required amount of water to add is 6 ounces.
-/
theorem required_amount_of_water 
    (original_volume : ℚ)
    (initial_concentration : ℚ)
    (desired_concentration : ℚ)
    (final_volume : ℚ)
    (amount_of_water : ℚ)
    (h1 : original_volume = 12)
    (h2 : initial_concentration = 0.6)
    (h3 : desired_concentration = 0.4)
    (h4 : final_volume = original_volume + amount_of_water)
    (h5 : amount_of_alcohol = original_volume * initial_concentration)
    (h6 : desired_amount_of_alcohol = final_volume * desired_concentration)
    (h7 : amount_of_alcohol = desired_amount_of_alcohol) : 
  amount_of_water = 6 := 
sorry

end NUMINAMATH_GPT_required_amount_of_water_l250_25045


namespace NUMINAMATH_GPT_triangle_side_length_l250_25073

theorem triangle_side_length (y z : ℝ) (cos_Y_minus_Z : ℝ) (h_y : y = 7) (h_z : z = 6) (h_cos : cos_Y_minus_Z = 17 / 18) : 
  ∃ x : ℝ, x = Real.sqrt 65 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l250_25073


namespace NUMINAMATH_GPT_product_of_geometric_sequence_l250_25071

theorem product_of_geometric_sequence (x y z : ℝ) 
  (h_seq : ∃ r, x = r * 1 ∧ y = r * x ∧ z = r * y ∧ 4 = r * z) : 
  1 * x * y * z * 4 = 32 :=
by
  sorry

end NUMINAMATH_GPT_product_of_geometric_sequence_l250_25071


namespace NUMINAMATH_GPT_darryl_parts_cost_l250_25055

-- Define the conditions
def patent_cost : ℕ := 4500
def machine_price : ℕ := 180
def break_even_units : ℕ := 45
def total_revenue := break_even_units * machine_price

-- Define the theorem using the conditions
theorem darryl_parts_cost :
  ∃ (parts_cost : ℕ), parts_cost = total_revenue - patent_cost ∧ parts_cost = 3600 := by
  sorry

end NUMINAMATH_GPT_darryl_parts_cost_l250_25055


namespace NUMINAMATH_GPT_evaluate_power_l250_25006

theorem evaluate_power (a : ℝ) (b : ℝ) (hb : b = 16) (hc : b = a ^ 4) : (b ^ (1 / 4)) ^ 12 = 4096 := by
  sorry

end NUMINAMATH_GPT_evaluate_power_l250_25006


namespace NUMINAMATH_GPT_base_10_uniqueness_l250_25064

theorem base_10_uniqueness : 
  (∀ a : ℕ, 12 = 3 * 4 ∧ 56 = 7 * 8 ↔ (a * b + (a + 1) = (a + 2) * (a + 3))) → b = 10 :=
sorry

end NUMINAMATH_GPT_base_10_uniqueness_l250_25064


namespace NUMINAMATH_GPT_find_const_s_l250_25037

noncomputable def g (x : ℝ) (a b c d : ℝ) := (x + 2 * a) * (x + 2 * b) * (x + 2 * c) * (x + 2 * d)

theorem find_const_s (a b c d : ℝ) (p q r s : ℝ) (h1 : 1 + p + q + r + s = 4041)
  (h2 : g 1 a b c d = 1 + p + q + r + s) :
  s = 3584 := 
sorry

end NUMINAMATH_GPT_find_const_s_l250_25037


namespace NUMINAMATH_GPT_rectangle_width_l250_25074

theorem rectangle_width (L W : ℕ)
  (h1 : W = L + 3)
  (h2 : 2 * L + 2 * W = 54) :
  W = 15 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_width_l250_25074


namespace NUMINAMATH_GPT_combined_average_score_l250_25038

theorem combined_average_score (M A : ℝ) (m a : ℝ)
  (hM : M = 78) (hA : A = 85) (h_ratio : m = 2 * a / 3) :
  (78 * (2 * a / 3) + 85 * a) / ((2 * a / 3) + a) = 82 := by
  sorry

end NUMINAMATH_GPT_combined_average_score_l250_25038


namespace NUMINAMATH_GPT_amount_spent_on_shorts_l250_25042

def amount_spent_on_shirt := 12.14
def amount_spent_on_jacket := 7.43
def total_amount_spent_on_clothes := 33.56

theorem amount_spent_on_shorts : total_amount_spent_on_clothes - amount_spent_on_shirt - amount_spent_on_jacket = 13.99 :=
by
  sorry

end NUMINAMATH_GPT_amount_spent_on_shorts_l250_25042


namespace NUMINAMATH_GPT_incorrect_value_in_polynomial_progression_l250_25095

noncomputable def polynomial_values (x : ℕ) : ℕ :=
  match x with
  | 0 => 1
  | 1 => 9
  | 2 => 35
  | 3 => 99
  | 4 => 225
  | 5 => 441
  | 6 => 784
  | 7 => 1296
  | _ => 0  -- This is a dummy value just to complete the function

theorem incorrect_value_in_polynomial_progression :
  ¬ (∃ (a b c d : ℝ), ∀ x : ℕ,
    polynomial_values x = (a * x ^ 3 + b * x ^ 2 + c * x + d + if x ≤ 7 then 0 else 1)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_incorrect_value_in_polynomial_progression_l250_25095


namespace NUMINAMATH_GPT_exponential_inequality_l250_25030

-- Define the conditions
variables {n : ℤ} {x : ℝ}

theorem exponential_inequality 
  (h1 : n ≥ 2) 
  (h2 : |x| < 1) 
  : 2^n > (1 - x)^n + (1 + x)^n :=
sorry

end NUMINAMATH_GPT_exponential_inequality_l250_25030


namespace NUMINAMATH_GPT_perimeter_of_region_l250_25096

noncomputable def side_length : ℝ := 2 / Real.pi

noncomputable def semicircle_perimeter : ℝ := 2

theorem perimeter_of_region (s : ℝ) (p : ℝ) (h1 : s = 2 / Real.pi) (h2 : p = 2) :
  4 * (p / 2) = 4 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_region_l250_25096


namespace NUMINAMATH_GPT_find_ab_unique_l250_25010

theorem find_ab_unique (a b : ℕ) (h1 : a > 1) (h2 : b > a) (h3 : a ≤ 20) (h4 : b ≤ 20) (h5 : a * b = 52) (h6 : a + b = 17) : a = 4 ∧ b = 13 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_find_ab_unique_l250_25010


namespace NUMINAMATH_GPT_false_proposition_C_l250_25000

variable (a b c : ℝ)
noncomputable def f (x : ℝ) := a * x^2 + b * x + c

theorem false_proposition_C 
  (ha : a > 0)
  (x0 : ℝ)
  (hx0 : x0 = -b / (2 * a)) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 :=
by
  sorry

end NUMINAMATH_GPT_false_proposition_C_l250_25000


namespace NUMINAMATH_GPT_log_expression_in_terms_of_a_l250_25060

noncomputable def log3 (x : ℝ) : ℝ := Real.log x / Real.log 3

variable (a : ℝ) (h : a = log3 2)

theorem log_expression_in_terms_of_a : log3 8 - 2 * log3 6 = a - 2 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_in_terms_of_a_l250_25060


namespace NUMINAMATH_GPT_trebled_resultant_is_correct_l250_25059

-- Let's define the initial number and the transformations
def initial_number := 17
def doubled (n : ℕ) := n * 2
def added_five (n : ℕ) := n + 5
def trebled (n : ℕ) := n * 3

-- Finally, we state the problem to prove
theorem trebled_resultant_is_correct : 
  trebled (added_five (doubled initial_number)) = 117 :=
by
  -- Here we just print sorry which means the proof is expected but not provided yet.
  sorry

end NUMINAMATH_GPT_trebled_resultant_is_correct_l250_25059


namespace NUMINAMATH_GPT_min_employees_wednesday_l250_25015

noncomputable def minWednesdayBirthdays (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) :
  ℕ :=
  40

theorem min_employees_wednesday (total_employees : ℕ) 
  (diff_birthdays : (List ℕ → Prop)) 
  (max_birthdays : (ℕ → List ℕ → Prop)) 
  (h1 : total_employees = 61) 
  (h2 : ∃ lst, diff_birthdays lst ∧ max_birthdays 40 lst) :
  minWednesdayBirthdays total_employees diff_birthdays max_birthdays = 40 := 
sorry

end NUMINAMATH_GPT_min_employees_wednesday_l250_25015


namespace NUMINAMATH_GPT_crayons_eaten_correct_l250_25058

variable (initial_crayons final_crayons : ℕ)

def crayonsEaten (initial_crayons final_crayons : ℕ) : ℕ :=
  initial_crayons - final_crayons

theorem crayons_eaten_correct : crayonsEaten 87 80 = 7 :=
  by
  sorry

end NUMINAMATH_GPT_crayons_eaten_correct_l250_25058


namespace NUMINAMATH_GPT_total_pens_bought_l250_25035

theorem total_pens_bought (r : ℕ) (h1 : r > 10) (h2 : 357 % r = 0) (h3 : 441 % r = 0) : 
  357 / r + 441 / r = 38 := 
sorry

end NUMINAMATH_GPT_total_pens_bought_l250_25035


namespace NUMINAMATH_GPT_find_p_plus_q_l250_25090

noncomputable def f (k p : ℚ) : ℚ := 5 * k^2 - 2 * k + p
noncomputable def g (k q : ℚ) : ℚ := 4 * k^2 + q * k - 6

theorem find_p_plus_q (p q : ℚ) (h : ∀ k : ℚ, f k p * g k q = 20 * k^4 - 18 * k^3 - 31 * k^2 + 12 * k + 18) :
  p + q = -3 :=
sorry

end NUMINAMATH_GPT_find_p_plus_q_l250_25090


namespace NUMINAMATH_GPT_sam_bought_9_cans_l250_25079

-- Definitions based on conditions
def spent_amount_dollars := 20 - 5.50
def spent_amount_cents := 1450 -- to avoid floating point precision issues we equate to given value in cents
def coupon_discount_cents := 5 * 25
def total_cost_no_discount := spent_amount_cents + coupon_discount_cents
def cost_per_can := 175

-- Main statement to prove
theorem sam_bought_9_cans : total_cost_no_discount / cost_per_can = 9 :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_sam_bought_9_cans_l250_25079


namespace NUMINAMATH_GPT_apples_in_baskets_l250_25063

theorem apples_in_baskets (total_apples : ℕ) (first_basket : ℕ) (increase : ℕ) (baskets : ℕ) :
  total_apples = 495 ∧ first_basket = 25 ∧ increase = 2 ∧
  (total_apples = (baskets / 2) * (2 * first_basket + (baskets - 1) * increase)) -> baskets = 13 :=
by sorry

end NUMINAMATH_GPT_apples_in_baskets_l250_25063


namespace NUMINAMATH_GPT_increasing_or_decreasing_subseq_l250_25067

theorem increasing_or_decreasing_subseq (a : Fin (m * n + 1) → ℝ) :
  ∃ (s : Fin (m + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (s i) ≤ a (s j)) ∨
  ∃ (t : Fin (n + 1) → Fin (m * n + 1)), (∀ i j, i < j → a (t i) ≥ a (t j)) :=
sorry

end NUMINAMATH_GPT_increasing_or_decreasing_subseq_l250_25067


namespace NUMINAMATH_GPT_count_multiples_5_or_7_not_35_l250_25081

def count_multiples_5 (n : ℕ) : ℕ := n / 5
def count_multiples_7 (n : ℕ) : ℕ := n / 7
def count_multiples_35 (n : ℕ) : ℕ := n / 35
def inclusion_exclusion (a b c : ℕ) : ℕ := a + b - c

theorem count_multiples_5_or_7_not_35 : 
  inclusion_exclusion (count_multiples_5 3000) (count_multiples_7 3000) (count_multiples_35 3000) = 943 :=
by
  sorry

end NUMINAMATH_GPT_count_multiples_5_or_7_not_35_l250_25081


namespace NUMINAMATH_GPT_product_increases_exactly_13_times_by_subtracting_3_l250_25076

theorem product_increases_exactly_13_times_by_subtracting_3 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    13 * (n1 * n2 * n3 * n4 * n5 * n6 * n7) =
      ((n1 - 3) * (n2 - 3) * (n3 - 3) * (n4 - 3) * (n5 - 3) * (n6 - 3) * (n7 - 3)) :=
sorry

end NUMINAMATH_GPT_product_increases_exactly_13_times_by_subtracting_3_l250_25076


namespace NUMINAMATH_GPT_popsicle_sticks_ratio_l250_25046

/-- Sam, Sid, and Steve brought popsicle sticks for their group activity in their Art class. Sid has twice as many popsicle sticks as Steve. If Steve has 12 popsicle sticks and they can use 108 popsicle sticks for their Art class activity, prove that the ratio of the number of popsicle sticks Sam has to the number Sid has is 3:1. -/
theorem popsicle_sticks_ratio (Sid Sam Steve : ℕ) 
    (h1 : Sid = 2 * Steve) 
    (h2 : Steve = 12) 
    (h3 : Sam + Sid + Steve = 108) : 
    Sam / Sid = 3 :=
by 
    -- Proof steps go here
    sorry

end NUMINAMATH_GPT_popsicle_sticks_ratio_l250_25046


namespace NUMINAMATH_GPT_three_digit_cubes_divisible_by_eight_l250_25086

theorem three_digit_cubes_divisible_by_eight :
  (∃ n1 n2 : ℕ, 100 ≤ n1 ∧ n1 < 1000 ∧ n2 < n1 ∧ 100 ≤ n2 ∧ n2 < 1000 ∧
  (∃ m1 m2 : ℕ, 2 ≤ m1 ∧ 2 ≤ m2 ∧ n1 = 8 * m1^3  ∧ n2 = 8 * m2^3)) :=
sorry

end NUMINAMATH_GPT_three_digit_cubes_divisible_by_eight_l250_25086


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l250_25072

theorem intersection_of_M_and_N (x : ℝ) :
  {x | x > 1} ∩ {x | x^2 - 2 * x < 0} = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l250_25072


namespace NUMINAMATH_GPT_problem1_problem2_l250_25044

noncomputable def arcSin (x : ℝ) : ℝ := Real.arcsin x

theorem problem1 :
  (S : ℝ) = 3 * Real.pi + 2 * Real.sqrt 2 - 6 * arcSin (Real.sqrt (2 / 3)) :=
by
  sorry

theorem problem2 :
  (S : ℝ) = 3 * arcSin (Real.sqrt (2 / 3)) - Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l250_25044


namespace NUMINAMATH_GPT_average_of_first_12_results_l250_25019

theorem average_of_first_12_results
  (average_25_results : ℝ)
  (average_last_12_results : ℝ)
  (result_13th : ℝ)
  (total_results : ℕ)
  (num_first_12 : ℕ)
  (num_last_12 : ℕ)
  (total_sum : ℝ)
  (sum_first_12 : ℝ)
  (sum_last_12 : ℝ)
  (A : ℝ)
  (h1 : average_25_results = 24)
  (h2 : average_last_12_results = 17)
  (h3 : result_13th = 228)
  (h4 : total_results = 25)
  (h5 : num_first_12 = 12)
  (h6 : num_last_12 = 12)
  (h7 : total_sum = average_25_results * total_results)
  (h8 : sum_last_12 = average_last_12_results * num_last_12)
  (h9 : total_sum = sum_first_12 + result_13th + sum_last_12)
  (h10 : sum_first_12 = A * num_first_12) :
  A = 14 :=
by
  sorry

end NUMINAMATH_GPT_average_of_first_12_results_l250_25019


namespace NUMINAMATH_GPT_find_range_a_l250_25080

def setA (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def setB (x a : ℝ) : Prop := |x - a| < 5
def real_line (x : ℝ) : Prop := True

theorem find_range_a (a : ℝ) :
  (∀ x, setA x ∨ setB x a) ↔ (-3:ℝ) ≤ a ∧ a ≤ 1 := by
sorry

end NUMINAMATH_GPT_find_range_a_l250_25080


namespace NUMINAMATH_GPT_possible_values_of_a_l250_25039

-- Define the sets P and Q under the conditions given
def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

-- Prove that if Q ⊆ P, then a ∈ {0, 1/3, -1/2}
theorem possible_values_of_a (a : ℝ) (h : Q a ⊆ P) : a = 0 ∨ a = 1/3 ∨ a = -1/2 :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_l250_25039


namespace NUMINAMATH_GPT_numbers_with_digit_one_are_more_numerous_l250_25029

theorem numbers_with_digit_one_are_more_numerous :
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  total_numbers - numbers_without_one > numbers_without_one :=
by
  let n := 9999998
  let total_numbers := 10^7
  let numbers_without_one := 9^7 
  show total_numbers - numbers_without_one > numbers_without_one
  sorry

end NUMINAMATH_GPT_numbers_with_digit_one_are_more_numerous_l250_25029


namespace NUMINAMATH_GPT_age_difference_l250_25018

theorem age_difference (a b c : ℕ) (h₁ : b = 8) (h₂ : c = b / 2) (h₃ : a + b + c = 22) : a - b = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l250_25018


namespace NUMINAMATH_GPT_solve_cryptarithm_l250_25066

-- Declare non-computable constants for the letters
variables {A B C : ℕ}

-- Conditions from the problem
-- Different letters represent different digits
axiom diff_digits : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- A ≠ 0
axiom A_nonzero : A ≠ 0

-- Given cryptarithm equation
axiom cryptarithm_eq : 100 * C + 10 * B + A + 100 * A + 10 * A + A = 100 * B + A

-- The proof to show the correct values
theorem solve_cryptarithm : A = 5 ∧ B = 9 ∧ C = 3 :=
sorry

end NUMINAMATH_GPT_solve_cryptarithm_l250_25066


namespace NUMINAMATH_GPT_weaving_output_first_day_l250_25013

theorem weaving_output_first_day (x : ℝ) :
  (x + 2*x + 4*x + 8*x + 16*x = 5) → x = 5 / 31 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_weaving_output_first_day_l250_25013


namespace NUMINAMATH_GPT_conditional_two_exits_one_effective_l250_25097

def conditional_structure (decide : Bool) : Prop :=
  if decide then True else False

theorem conditional_two_exits_one_effective (decide : Bool) :
  conditional_structure decide ↔ True :=
by
  sorry

end NUMINAMATH_GPT_conditional_two_exits_one_effective_l250_25097


namespace NUMINAMATH_GPT_sin_double_angle_half_l250_25021

theorem sin_double_angle_half (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_half_l250_25021


namespace NUMINAMATH_GPT_div_relation_l250_25054

variable {a b c : ℚ}

theorem div_relation (h1 : a / b = 3) (h2 : b / c = 2/5) : c / a = 5/6 := by
  sorry

end NUMINAMATH_GPT_div_relation_l250_25054


namespace NUMINAMATH_GPT_maximize_greenhouse_planting_area_l250_25004

theorem maximize_greenhouse_planting_area
    (a b : ℝ)
    (h : a * b = 800)
    (planting_area : ℝ := (a - 4) * (b - 2)) :
  (a = 40 ∧ b = 20) ↔ planting_area = 648 :=
by
  sorry

end NUMINAMATH_GPT_maximize_greenhouse_planting_area_l250_25004


namespace NUMINAMATH_GPT_pencil_count_l250_25022

def total_pencils (drawer : Nat) (desk_0 : Nat) (add_dan : Nat) (remove_sarah : Nat) : Nat :=
  let desk_1 := desk_0 + add_dan
  let desk_2 := desk_1 - remove_sarah
  drawer + desk_2

theorem pencil_count :
  total_pencils 43 19 16 7 = 71 :=
by
  sorry

end NUMINAMATH_GPT_pencil_count_l250_25022


namespace NUMINAMATH_GPT_parabola_equation_and_orthogonality_l250_25091

theorem parabola_equation_and_orthogonality 
  (p : ℝ) (h_p_pos : p > 0) 
  (F : ℝ × ℝ) (h_focus : F = (p / 2, 0)) 
  (A B : ℝ × ℝ) (y : ℝ → ℝ) (C : ℝ × ℝ) 
  (h_parabola : ∀ (x y : ℝ), y^2 = 2 * p * x) 
  (h_line : ∀ (x : ℝ), y x = x - 8) 
  (h_intersect : ∃ x, y x = 0)
  (h_intersection_points : ∃ (x1 x2 : ℝ), y x1 = 0 ∧ y x2 = 0)
  (O : ℝ × ℝ) (h_origin : O = (0, 0)) 
  (h_vector_relation : 3 * F.fst = C.fst - F.fst)
  (h_C_x_axis : C = (8, 0)) :
  (p = 4 → y^2 = 8 * x) ∧ 
  (∀ (A B : ℝ × ℝ), (A.snd * B.snd = -64) ∧ 
  ((A.fst = (A.snd)^2 / 8) ∧ (B.fst = (B.snd)^2 / 8)) → 
  (A.fst * B.fst + A.snd * B.snd = 0)) := 
sorry

end NUMINAMATH_GPT_parabola_equation_and_orthogonality_l250_25091


namespace NUMINAMATH_GPT_sum_of_central_squares_is_34_l250_25099

-- Defining the parameters and conditions
def is_adjacent (i j : ℕ) : Prop := 
  (i = j + 1 ∨ i = j - 1 ∨ i = j + 4 ∨ i = j - 4)

def valid_matrix (M : Fin 4 → Fin 4 → ℕ) : Prop := 
  ∀ (i j : Fin 4), 
  i < 3 ∧ j < 3 → is_adjacent (M i j) (M (i + 1) j) ∧ is_adjacent (M i j) (M i (j + 1))

def corners_sum_to_34 (M : Fin 4 → Fin 4 → ℕ) : Prop :=
  M 0 0 + M 0 3 + M 3 0 + M 3 3 = 34

-- Stating the proof problem
theorem sum_of_central_squares_is_34 :
  ∃ (M : Fin 4 → Fin 4 → ℕ), valid_matrix M ∧ corners_sum_to_34 M → 
  (M 1 1 + M 1 2 + M 2 1 + M 2 2 = 34) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_central_squares_is_34_l250_25099


namespace NUMINAMATH_GPT_calculate_area_of_shaded_region_l250_25012

namespace Proof

noncomputable def AreaOfShadedRegion (width height : ℝ) (divisions : ℕ) : ℝ :=
  let small_width := width
  let small_height := height / divisions
  let area_of_small := small_width * small_height
  let shaded_in_small := area_of_small / 2
  let total_shaded := divisions * shaded_in_small
  total_shaded

theorem calculate_area_of_shaded_region :
  AreaOfShadedRegion 3 14 4 = 21 := by
  sorry

end Proof

end NUMINAMATH_GPT_calculate_area_of_shaded_region_l250_25012


namespace NUMINAMATH_GPT_probability_odd_divisor_15_factorial_l250_25001

theorem probability_odd_divisor_15_factorial :
  let number_of_divisors_15_fact : ℕ := (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  let number_of_odd_divisors_15_fact : ℕ := (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)
  (number_of_odd_divisors_15_fact : ℝ) / (number_of_divisors_15_fact : ℝ) = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_odd_divisor_15_factorial_l250_25001


namespace NUMINAMATH_GPT_gravitational_field_height_depth_equality_l250_25078

theorem gravitational_field_height_depth_equality
  (R G ρ : ℝ) (hR : R > 0) :
  ∃ x : ℝ, x = R * ((-1 + Real.sqrt 5) / 2) ∧
  (G * ρ * ((4 / 3) * Real.pi * R^3) / (R + x)^2 = G * ρ * ((4 / 3) * Real.pi * (R - x)^3) / (R - x)^2) :=
by
  sorry

end NUMINAMATH_GPT_gravitational_field_height_depth_equality_l250_25078


namespace NUMINAMATH_GPT_pigeon_percentage_l250_25050

-- Define the conditions
variables (total_birds : ℕ)
variables (geese swans herons ducks pigeons : ℕ)
variables (h1 : geese = total_birds * 20 / 100)
variables (h2 : swans = total_birds * 30 / 100)
variables (h3 : herons = total_birds * 15 / 100)
variables (h4 : ducks = total_birds * 25 / 100)
variables (h5 : pigeons = total_birds * 10 / 100)

-- Define the target problem
theorem pigeon_percentage (h_total : total_birds = 100) :
  (pigeons * 100 / (total_birds - swans)) = 14 :=
by sorry

end NUMINAMATH_GPT_pigeon_percentage_l250_25050


namespace NUMINAMATH_GPT_is_exact_time_now_321_l250_25017

noncomputable def current_time_is_321 : Prop :=
  exists t : ℝ, 0 < t ∧ t < 60 ∧ |(6 * t + 48) - (90 + 0.5 * (t - 4))| = 180

theorem is_exact_time_now_321 : current_time_is_321 := 
  sorry

end NUMINAMATH_GPT_is_exact_time_now_321_l250_25017


namespace NUMINAMATH_GPT_latoya_initial_payment_l250_25003

variable (cost_per_minute : ℝ) (call_duration : ℝ) (remaining_credit : ℝ) 
variable (initial_credit : ℝ)

theorem latoya_initial_payment : 
  ∀ (cost_per_minute call_duration remaining_credit initial_credit : ℝ),
  cost_per_minute = 0.16 →
  call_duration = 22 →
  remaining_credit = 26.48 →
  initial_credit = (cost_per_minute * call_duration) + remaining_credit →
  initial_credit = 30 :=
by
  intros cost_per_minute call_duration remaining_credit initial_credit
  sorry

end NUMINAMATH_GPT_latoya_initial_payment_l250_25003


namespace NUMINAMATH_GPT_apple_cost_price_l250_25048

theorem apple_cost_price (SP : ℝ) (loss_ratio : ℝ) (CP : ℝ) (h1 : SP = 18) (h2 : loss_ratio = 1/6) (h3 : SP = CP - loss_ratio * CP) : CP = 21.6 :=
by
  sorry

end NUMINAMATH_GPT_apple_cost_price_l250_25048


namespace NUMINAMATH_GPT_intersection_complement_equivalence_l250_25053

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equivalence :
  ((U \ M) ∩ N) = {3} := by
  sorry

end NUMINAMATH_GPT_intersection_complement_equivalence_l250_25053


namespace NUMINAMATH_GPT_royal_children_l250_25036

variable (d n : ℕ)

def valid_children_number (num_children : ℕ) : Prop :=
  num_children <= 20

theorem royal_children :
  (∃ d n, 35 = n * (d + 1) ∧ valid_children_number (d + 3)) →
  (d + 3 = 7 ∨ d + 3 = 9) :=
by intro h; sorry

end NUMINAMATH_GPT_royal_children_l250_25036


namespace NUMINAMATH_GPT_employed_males_population_percentage_l250_25069

-- Define the conditions of the problem
variables (P : Type) (population : ℝ) (employed_population : ℝ) (employed_females : ℝ)

-- Assume total population is 100
def total_population : ℝ := 100

-- 70 percent of the population are employed
def employed_population_percentage : ℝ := total_population * 0.70

-- 70 percent of the employed people are females
def employed_females_percentage : ℝ := employed_population_percentage * 0.70

-- 21 percent of the population are employed males
def employed_males_percentage : ℝ := 21

-- Main statement to be proven
theorem employed_males_population_percentage :
  employed_males_percentage = ((employed_population_percentage - employed_females_percentage) / total_population) * 100 :=
sorry

end NUMINAMATH_GPT_employed_males_population_percentage_l250_25069


namespace NUMINAMATH_GPT_income_distribution_after_tax_l250_25023

theorem income_distribution_after_tax (x : ℝ) (hx : 10 * x = 100) :
  let poor_income_initial := x
  let middle_income_initial := 4 * x
  let rich_income_initial := 5 * x
  let tax_rate := (x^2 / 4) + x
  let tax_collected := tax_rate * rich_income_initial / 100
  let poor_income_after := poor_income_initial + 3 / 4 * tax_collected
  let middle_income_after := middle_income_initial + 1 / 4 * tax_collected
  let rich_income_after := rich_income_initial - tax_collected
  poor_income_after = 0.23125 * 100 ∧
  middle_income_after = 0.44375 * 100 ∧
  rich_income_after = 0.325 * 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_income_distribution_after_tax_l250_25023


namespace NUMINAMATH_GPT_Tyler_has_200_puppies_l250_25020

-- Define the number of dogs
def numDogs : ℕ := 25

-- Define the number of puppies per dog
def puppiesPerDog : ℕ := 8

-- Define the total number of puppies
def totalPuppies : ℕ := numDogs * puppiesPerDog

-- State the theorem we want to prove
theorem Tyler_has_200_puppies : totalPuppies = 200 := by
  exact (by norm_num : 25 * 8 = 200)

end NUMINAMATH_GPT_Tyler_has_200_puppies_l250_25020


namespace NUMINAMATH_GPT_problem1_problem2_l250_25026

-- define problem 1 as a theorem
theorem problem1: 
  ((-0.4) * (-0.8) * (-1.25) * 2.5 = -1) :=
  sorry

-- define problem 2 as a theorem
theorem problem2: 
  ((- (5:ℚ) / 8) * (3 / 14) * ((-16) / 5) * ((-7) / 6) = -1 / 2) :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l250_25026


namespace NUMINAMATH_GPT_abs_eq_abs_of_unique_solution_l250_25014

variables {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0)
theorem abs_eq_abs_of_unique_solution
  (h : ∃ x : ℝ, ∀ y : ℝ, a * (y - a)^2 + b * (y - b)^2 = 0 ↔ y = x) :
  |a| = |b| :=
sorry

end NUMINAMATH_GPT_abs_eq_abs_of_unique_solution_l250_25014


namespace NUMINAMATH_GPT_exchanges_count_l250_25032

theorem exchanges_count (n : ℕ) :
  ∀ (initial_pencils_XZ initial_pens_XL : ℕ) 
    (pencils_per_exchange pens_per_exchange : ℕ)
    (final_pencils_multiplier : ℕ)
    (pz : initial_pencils_XZ = 200) 
    (pl : initial_pens_XL = 20)
    (pe : pencils_per_exchange = 6)
    (se : pens_per_exchange = 1)
    (fm : final_pencils_multiplier = 11),
    (initial_pencils_XZ - pencils_per_exchange * n = final_pencils_multiplier * (initial_pens_XL - pens_per_exchange * n)) ↔ n = 4 :=
by
  intros initial_pencils_XZ initial_pens_XL pencils_per_exchange pens_per_exchange final_pencils_multiplier pz pl pe se fm
  sorry

end NUMINAMATH_GPT_exchanges_count_l250_25032


namespace NUMINAMATH_GPT_parallelepiped_face_areas_l250_25089

theorem parallelepiped_face_areas
    (h₁ : ℝ := 2)  -- height corresponding to face area x
    (h₂ : ℝ := 3)  -- height corresponding to face area y
    (h₃ : ℝ := 4)  -- height corresponding to face area z
    (total_surface_area : ℝ := 36) : 
    ∃ (x y z : ℝ), 
    2 * x + 2 * y + 2 * z = total_surface_area ∧
    (∃ V : ℝ, V = h₁ * x ∧ V = h₂ * y ∧ V = h₃ * z) ∧
    x = 108 / 13 ∧ y = 72 / 13 ∧ z = 54 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_parallelepiped_face_areas_l250_25089


namespace NUMINAMATH_GPT_find_a20_l250_25077

variable {a : ℕ → ℤ}
variable {d : ℤ}
variable {a_1 : ℤ}

def isArithmeticSeq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def formsGeomSeq (a1 a3 a4 : ℤ) : Prop :=
  (a3 - a1)^2 = a1 * (a4 - a1)

theorem find_a20 (h1 : isArithmeticSeq a (-2))
                 (h2 : formsGeomSeq a_1 (a_1 + 2*(-2)) (a_1 + 3*(-2)))
                 (ha1 : a_1 = 8) :
  a 20 = -30 :=
by
  sorry

end NUMINAMATH_GPT_find_a20_l250_25077


namespace NUMINAMATH_GPT_train_length_l250_25083

theorem train_length (V L : ℝ) (h₁ : V = L / 18) (h₂ : V = (L + 200) / 30) : L = 300 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l250_25083


namespace NUMINAMATH_GPT_cooking_time_per_side_l250_25087

-- Defining the problem conditions
def total_guests : ℕ := 30
def guests_wanting_2_burgers : ℕ := total_guests / 2
def guests_wanting_1_burger : ℕ := total_guests / 2
def burgers_per_guest_2 : ℕ := 2
def burgers_per_guest_1 : ℕ := 1
def total_burgers : ℕ := guests_wanting_2_burgers * burgers_per_guest_2 + guests_wanting_1_burger * burgers_per_guest_1
def burgers_per_batch : ℕ := 5
def total_batches : ℕ := total_burgers / burgers_per_batch
def total_cooking_time : ℕ := 72
def time_per_batch : ℕ := total_cooking_time / total_batches
def sides_per_burger : ℕ := 2

-- the theorem to prove the desired cooking time per side
theorem cooking_time_per_side : (time_per_batch / sides_per_burger) = 4 := by {
    -- Here we would enter the proof steps, but this is omitted as per the instructions.
    sorry
}

end NUMINAMATH_GPT_cooking_time_per_side_l250_25087


namespace NUMINAMATH_GPT_average_percentage_increase_is_correct_l250_25043

def initial_prices : List ℝ := [300, 450, 600]
def price_increases : List ℝ := [0.10, 0.15, 0.20]

noncomputable def total_original_price : ℝ :=
  initial_prices.sum

noncomputable def total_new_price : ℝ :=
  (List.zipWith (λ p i => p * (1 + i)) initial_prices price_increases).sum

noncomputable def total_price_increase : ℝ :=
  total_new_price - total_original_price

noncomputable def average_percentage_increase : ℝ :=
  (total_price_increase / total_original_price) * 100

theorem average_percentage_increase_is_correct :
  average_percentage_increase = 16.11 := by
  sorry

end NUMINAMATH_GPT_average_percentage_increase_is_correct_l250_25043


namespace NUMINAMATH_GPT_original_number_is_9_l250_25062

theorem original_number_is_9 (x : ℤ) (h : 10 * x = x + 81) : x = 9 :=
sorry

end NUMINAMATH_GPT_original_number_is_9_l250_25062


namespace NUMINAMATH_GPT_smallest_integer_in_set_l250_25056

def avg_seven_consecutive_integers (n : ℤ) : ℤ :=
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7

theorem smallest_integer_in_set : ∃ (n : ℤ), n = 0 ∧ (n + 6 < 3 * avg_seven_consecutive_integers n) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_in_set_l250_25056


namespace NUMINAMATH_GPT_number_of_solutions_l250_25085

open Real

-- Define the main equation in terms of absolute values 
def equation (x : ℝ) : Prop := abs (x - abs (2 * x + 1)) = 3

-- Prove that there are exactly 2 distinct solutions to the equation
theorem number_of_solutions : 
  ∃! x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂ :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l250_25085


namespace NUMINAMATH_GPT_tracy_sold_paintings_l250_25031

-- Definitions based on conditions
def total_customers := 20
def customers_2_paintings := 4
def customers_1_painting := 12
def customers_4_paintings := 4

def paintings_per_customer_2 := 2
def paintings_per_customer_1 := 1
def paintings_per_customer_4 := 4

-- Theorem statement
theorem tracy_sold_paintings : 
  (customers_2_paintings * paintings_per_customer_2) + 
  (customers_1_painting * paintings_per_customer_1) + 
  (customers_4_paintings * paintings_per_customer_4) = 36 := 
by
  sorry

end NUMINAMATH_GPT_tracy_sold_paintings_l250_25031


namespace NUMINAMATH_GPT_three_gt_sqrt_seven_l250_25011

theorem three_gt_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_three_gt_sqrt_seven_l250_25011


namespace NUMINAMATH_GPT_equation_of_curve_C_range_of_m_l250_25094

theorem equation_of_curve_C (x y m : ℝ) (hx : x ≠ 0) (hm : m > 1) (k1 k2 : ℝ) 
  (h_k1 : k1 = (y - 1) / x) (h_k2 : k2 = (y + 1) / (2 * x))
  (h_prod : k1 * k2 = -1 / m^2) :
  (x^2) / (m^2) + (y^2) = 1 := 
sorry

theorem range_of_m (m : ℝ) :
  (1 < m ∧ m ≤ Real.sqrt 3)
  ∨ (m < 1 ∨ m > Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_equation_of_curve_C_range_of_m_l250_25094


namespace NUMINAMATH_GPT_negate_one_even_l250_25009

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_one_even (a b c : ℕ) :
  (∃! x, x = a ∨ x = b ∨ x = c ∧ is_even x) ↔
  (∃ x y, x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧
    x ≠ y ∧ is_even x ∧ is_even y) ∨
  (is_odd a ∧ is_odd b ∧ is_odd c) :=
by {
  sorry
}

end NUMINAMATH_GPT_negate_one_even_l250_25009


namespace NUMINAMATH_GPT_three_numbers_less_or_equal_than_3_l250_25065

theorem three_numbers_less_or_equal_than_3 : 
  let a := 0.8
  let b := 0.5
  let c := 0.9
  (a ≤ 3) ∧ (b ≤ 3) ∧ (c ≤ 3) → 
  3 = 3 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_three_numbers_less_or_equal_than_3_l250_25065


namespace NUMINAMATH_GPT_minimum_n_is_835_l250_25084

def problem_statement : Prop :=
  ∀ (S : Finset ℕ), S.card = 835 → (∀ (T : Finset ℕ), T ⊆ S → T.card = 4 →
    ∃ (a b c d : ℕ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ d ∈ T ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + 2 * b + 3 * c = d)

theorem minimum_n_is_835 : problem_statement :=
sorry

end NUMINAMATH_GPT_minimum_n_is_835_l250_25084


namespace NUMINAMATH_GPT_expand_binomials_l250_25051

variable (x y : ℝ)

theorem expand_binomials : 
  (3 * x - 2) * (2 * x + 4 * y + 1) = 6 * x^2 + 12 * x * y - x - 8 * y - 2 :=
by
  sorry

end NUMINAMATH_GPT_expand_binomials_l250_25051


namespace NUMINAMATH_GPT_return_kittens_due_to_rehoming_problems_l250_25002

def num_breeding_rabbits : Nat := 10
def kittens_first_spring : Nat := num_breeding_rabbits * num_breeding_rabbits
def kittens_adopted_first_spring : Nat := kittens_first_spring / 2
def kittens_second_spring : Nat := 60
def kittens_adopted_second_spring : Nat := 4
def total_rabbits : Nat := 121

def non_breeding_rabbits_from_first_spring : Nat :=
  total_rabbits - num_breeding_rabbits - kittens_second_spring

def kittens_returned_to_lola : Prop :=
  non_breeding_rabbits_from_first_spring - kittens_adopted_first_spring = 1

theorem return_kittens_due_to_rehoming_problems : kittens_returned_to_lola :=
sorry

end NUMINAMATH_GPT_return_kittens_due_to_rehoming_problems_l250_25002


namespace NUMINAMATH_GPT_locus_square_l250_25027

open Real

variables {x y c1 c2 d1 d2 : ℝ}

/-- The locus of points in a square -/
theorem locus_square (h_square: d1 < d2 ∧ c1 < c2) (h_x: d1 ≤ x ∧ x ≤ d2) (h_y: c1 ≤ y ∧ y ≤ c2) :
  |y - c1| + |y - c2| = |x - d1| + |x - d2| :=
by sorry

end NUMINAMATH_GPT_locus_square_l250_25027


namespace NUMINAMATH_GPT_sugar_merchant_profit_l250_25052

theorem sugar_merchant_profit 
    (total_sugar : ℕ)
    (sold_at_18 : ℕ)
    (remain_sugar : ℕ)
    (whole_profit : ℕ)
    (profit_18 : ℕ)
    (rem_profit_percent : ℕ) :
    total_sugar = 1000 →
    sold_at_18 = 600 →
    remain_sugar = total_sugar - sold_at_18 →
    whole_profit = 14 →
    profit_18 = 18 →
    (600 * profit_18 / 100) + (remain_sugar * rem_profit_percent / 100) = 
    (total_sugar * whole_profit / 100) →
    rem_profit_percent = 80 :=
by
    sorry

end NUMINAMATH_GPT_sugar_merchant_profit_l250_25052


namespace NUMINAMATH_GPT_find_x_in_isosceles_triangle_l250_25005

def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (b = c) ∨ (a = c)

def triangle_inequality (a b c : ℝ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem find_x_in_isosceles_triangle (x : ℝ) :
  is_isosceles (x + 3) (2 * x + 1) 11 ∧ triangle_inequality (x + 3) (2 * x + 1) 11 →
  (x = 8) ∨ (x = 5) :=
sorry

end NUMINAMATH_GPT_find_x_in_isosceles_triangle_l250_25005


namespace NUMINAMATH_GPT_porter_previous_painting_price_l250_25024

-- definitions from the conditions
def most_recent_sale : ℕ := 44000

-- definitions for the problem statement
def sale_equation (P : ℕ) : Prop :=
  most_recent_sale = 5 * P - 1000

theorem porter_previous_painting_price (P : ℕ) (h : sale_equation P) : P = 9000 :=
by {
  sorry
}

end NUMINAMATH_GPT_porter_previous_painting_price_l250_25024


namespace NUMINAMATH_GPT_increase_in_output_with_assistant_l250_25007

theorem increase_in_output_with_assistant (B H : ℝ) (hB : 0 < B) (hH : 0 < H) :
  ((1.80 * B) / (0.90 * H)) / (B / H) - 1 = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_increase_in_output_with_assistant_l250_25007


namespace NUMINAMATH_GPT_sqrt_problem_l250_25034

theorem sqrt_problem (a m : ℝ) (ha : 0 < a) 
  (h1 : a = (3 * m - 1) ^ 2) 
  (h2 : a = (-2 * m - 2) ^ 2) : 
  a = 64 ∨ a = 64 / 25 := 
sorry

end NUMINAMATH_GPT_sqrt_problem_l250_25034


namespace NUMINAMATH_GPT_find_k_l250_25098

variable (x y z k : ℝ)

def fractions_are_equal : Prop := (9 / (x + y) = k / (x + z) ∧ k / (x + z) = 15 / (z - y))

theorem find_k (h : fractions_are_equal x y z k) : k = 24 := by
  sorry

end NUMINAMATH_GPT_find_k_l250_25098


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l250_25088

variable {a : ℕ → ℤ}

noncomputable def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
∀ (m n k : ℕ), a m * a k = a n * a (m + k - n)

noncomputable def is_root_of_quadratic (x y : ℤ) : Prop :=
x^2 + 3*x + 1 = 0 ∧ y^2 + 3*y + 1 = 0

theorem necessary_but_not_sufficient_condition 
  (a : ℕ → ℤ)
  (hgeo : is_geometric_sequence a)
  (hroots : is_root_of_quadratic (a 4) (a 12)) :
  a 8 = -1 ↔ (∃ x y : ℤ, is_root_of_quadratic x y ∧ x + y = -3 ∧ x * y = 1) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l250_25088


namespace NUMINAMATH_GPT_sector_properties_l250_25016

-- Definitions for the conditions
def central_angle (α : ℝ) : Prop := α = 2 * Real.pi / 3

def radius (r : ℝ) : Prop := r = 6

def sector_perimeter (l r : ℝ) : Prop := l + 2 * r = 20

-- The statement encapsulating the proof problem
theorem sector_properties :
  (central_angle (2 * Real.pi / 3) ∧ radius 6 →
    ∃ l S, l = 4 * Real.pi ∧ S = 12 * Real.pi) ∧
  (∃ l r, sector_perimeter l r ∧ 
    ∃ α S, α = 2 ∧ S = 25) := by
  sorry

end NUMINAMATH_GPT_sector_properties_l250_25016


namespace NUMINAMATH_GPT_solution_interval_l250_25008

theorem solution_interval (x : ℝ) : 
  (3/8 + |x - 1/4| < 7/8) ↔ (-1/4 < x ∧ x < 3/4) := 
sorry

end NUMINAMATH_GPT_solution_interval_l250_25008


namespace NUMINAMATH_GPT_division_problem_l250_25068

theorem division_problem (n : ℕ) (h : n / 6 = 209) : n = 1254 := 
sorry

end NUMINAMATH_GPT_division_problem_l250_25068


namespace NUMINAMATH_GPT_mark_last_shots_l250_25033

theorem mark_last_shots (h1 : 0.60 * 15 = 9) (h2 : 0.65 * 25 = 16.25) : 
  ∀ (successful_shots_first_15 successful_shots_total: ℤ),
  successful_shots_first_15 = 9 ∧ 
  successful_shots_total = 16 → 
  successful_shots_total - successful_shots_first_15 = 7 := by
  sorry

end NUMINAMATH_GPT_mark_last_shots_l250_25033


namespace NUMINAMATH_GPT_pages_remaining_total_l250_25092

-- Define the conditions
def total_pages_book1 : ℕ := 563
def read_pages_book1 : ℕ := 147

def total_pages_book2 : ℕ := 849
def read_pages_book2 : ℕ := 389

def total_pages_book3 : ℕ := 700
def read_pages_book3 : ℕ := 134

-- The theorem to be proved
theorem pages_remaining_total :
  (total_pages_book1 - read_pages_book1) + 
  (total_pages_book2 - read_pages_book2) + 
  (total_pages_book3 - read_pages_book3) = 1442 := 
by
  sorry

end NUMINAMATH_GPT_pages_remaining_total_l250_25092


namespace NUMINAMATH_GPT_total_emeralds_l250_25093

theorem total_emeralds (D R E : ℕ) 
  (h1 : 2 * D + 2 * E + 2 * R = 6)
  (h2 : R = D + 15) : 
  E = 12 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_emeralds_l250_25093


namespace NUMINAMATH_GPT_find_a_l250_25061

def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

theorem find_a (a : ℝ) (h : f (f a) = f 9 + 1) : a = -1/4 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l250_25061


namespace NUMINAMATH_GPT_difference_of_squares_example_l250_25047

theorem difference_of_squares_example (a b : ℕ) (h1 : a = 123) (h2 : b = 23) : a^2 - b^2 = 14600 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_difference_of_squares_example_l250_25047
