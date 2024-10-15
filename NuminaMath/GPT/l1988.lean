import Mathlib

namespace NUMINAMATH_GPT_smallest_lcm_l1988_198821

theorem smallest_lcm (m n : ℕ) (hm : 10000 ≤ m ∧ m < 100000) (hn : 10000 ≤ n ∧ n < 100000) (h : Nat.gcd m n = 5) : Nat.lcm m n = 20030010 :=
sorry

end NUMINAMATH_GPT_smallest_lcm_l1988_198821


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1988_198811

def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 < x ∧ x < 5}

theorem intersection_of_A_and_B :
  A ∩ B = {3, 4} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1988_198811


namespace NUMINAMATH_GPT_triangle_altitude_angle_l1988_198854

noncomputable def angle_between_altitudes (α : ℝ) : ℝ :=
if α ≤ 90 then α else 180 - α

theorem triangle_altitude_angle (α : ℝ) (hα : 0 < α ∧ α < 180) : 
  (angle_between_altitudes α = α ↔ α ≤ 90) ∧ (angle_between_altitudes α = 180 - α ↔ α > 90) := 
by
  sorry

end NUMINAMATH_GPT_triangle_altitude_angle_l1988_198854


namespace NUMINAMATH_GPT_num_adult_tickets_l1988_198877

theorem num_adult_tickets (adult_ticket_cost child_ticket_cost total_tickets_sold total_receipts : ℕ) 
  (h1 : adult_ticket_cost = 12) 
  (h2 : child_ticket_cost = 4) 
  (h3 : total_tickets_sold = 130) 
  (h4 : total_receipts = 840) :
  ∃ A C : ℕ, A + C = total_tickets_sold ∧ adult_ticket_cost * A + child_ticket_cost * C = total_receipts ∧ A = 40 :=
by {
  sorry
}

end NUMINAMATH_GPT_num_adult_tickets_l1988_198877


namespace NUMINAMATH_GPT_geometric_sum_first_8_terms_eq_17_l1988_198883

theorem geometric_sum_first_8_terms_eq_17 (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = 2 * a n)
  (h2 : a 0 + a 1 + a 2 + a 3 = 1) : 
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 17 :=
sorry

end NUMINAMATH_GPT_geometric_sum_first_8_terms_eq_17_l1988_198883


namespace NUMINAMATH_GPT_find_b2_l1988_198838

theorem find_b2 (b : ℕ → ℝ) (h1 : b 1 = 23) (h10 : b 10 = 123) 
  (h : ∀ n ≥ 3, b n = (b 1 + b 2 + (n - 3) * b 3) / (n - 1)) : b 2 = 223 :=
sorry

end NUMINAMATH_GPT_find_b2_l1988_198838


namespace NUMINAMATH_GPT_louie_pie_share_l1988_198841

theorem louie_pie_share :
  let leftover := (6 : ℝ) / 7
  let people := 3
  leftover / people = (2 : ℝ) / 7 := 
by
  sorry

end NUMINAMATH_GPT_louie_pie_share_l1988_198841


namespace NUMINAMATH_GPT_max_value_of_expression_l1988_198858

theorem max_value_of_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 1) ≤ Real.sqrt 29 := 
sorry

end NUMINAMATH_GPT_max_value_of_expression_l1988_198858


namespace NUMINAMATH_GPT_sale_price_for_50_percent_profit_l1988_198892

theorem sale_price_for_50_percent_profit
  (C L: ℝ)
  (h1: 892 - C = C - L)
  (h2: 1005 = 1.5 * C) :
  1.5 * C = 1005 :=
by
  sorry

end NUMINAMATH_GPT_sale_price_for_50_percent_profit_l1988_198892


namespace NUMINAMATH_GPT_max_number_of_rectangles_in_square_l1988_198843

-- Definitions and conditions
def area_square (n : ℕ) : ℕ := 4 * n^2
def area_rectangle (n : ℕ) : ℕ := n + 1
def max_rectangles (n : ℕ) : ℕ := area_square n / area_rectangle n

-- Lean theorem statement for the proof problem
theorem max_number_of_rectangles_in_square (n : ℕ) (h : n ≥ 4) :
  max_rectangles n = 4 * (n - 1) :=
sorry

end NUMINAMATH_GPT_max_number_of_rectangles_in_square_l1988_198843


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1988_198874

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h₁ : a 1 = 39) (h₂ : a 1 + a 3 = 74) : 
  ∀ n, a n = 41 - 2 * n :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l1988_198874


namespace NUMINAMATH_GPT_probability_no_shaded_rectangle_l1988_198869

-- Definitions
def total_rectangles_per_row : ℕ := (2005 * 2004) / 2
def shaded_rectangles_per_row : ℕ := 1002 * 1002

-- Proposition to prove
theorem probability_no_shaded_rectangle : 
  (1 - (shaded_rectangles_per_row : ℝ) / (total_rectangles_per_row : ℝ)) = (0.25 / 1002.25) := 
sorry

end NUMINAMATH_GPT_probability_no_shaded_rectangle_l1988_198869


namespace NUMINAMATH_GPT_elizabeth_stickers_l1988_198814

def total_stickers (initial_bottles lost_bottles stolen_bottles stickers_per_bottle : ℕ) : ℕ :=
  let remaining_bottles := initial_bottles - lost_bottles - stolen_bottles
  remaining_bottles * stickers_per_bottle

theorem elizabeth_stickers :
  total_stickers 10 2 1 3 = 21 :=
by
  sorry

end NUMINAMATH_GPT_elizabeth_stickers_l1988_198814


namespace NUMINAMATH_GPT_largest_integral_solution_l1988_198889

noncomputable def largest_integral_value : ℤ :=
  let a : ℚ := 1 / 4
  let b : ℚ := 7 / 11 
  let lower_bound : ℚ := 7 * a
  let upper_bound : ℚ := 7 * b
  let x := 3  -- The largest integral value within the bounds
  x

-- A theorem to prove that x = 3 satisfies the inequality conditions and is the largest integer.
theorem largest_integral_solution (x : ℤ) (h₁ : 1 / 4 < x / 7) (h₂ : x / 7 < 7 / 11) : x = 3 := by
  sorry

end NUMINAMATH_GPT_largest_integral_solution_l1988_198889


namespace NUMINAMATH_GPT_shopkeeper_profit_percentage_goal_l1988_198806

-- Definitions for CP, MP and discount percentage
variable (CP : ℝ)
noncomputable def MP : ℝ := CP * 1.32
noncomputable def discount_percentage : ℝ := 0.18939393939393938
noncomputable def SP : ℝ := MP CP - (discount_percentage * MP CP)
noncomputable def profit : ℝ := SP CP - CP
noncomputable def profit_percentage : ℝ := (profit CP / CP) * 100

-- Theorem stating that the profit percentage is approximately 7%
theorem shopkeeper_profit_percentage_goal :
  abs (profit_percentage CP - 7) < 0.01 := sorry

end NUMINAMATH_GPT_shopkeeper_profit_percentage_goal_l1988_198806


namespace NUMINAMATH_GPT_molecular_weight_of_acetic_acid_l1988_198870

-- Define the molecular weight of 7 moles of acetic acid
def molecular_weight_7_moles_acetic_acid := 420 

-- Define the number of moles of acetic acid
def moles_acetic_acid := 7

-- Define the molecular weight of 1 mole of acetic acid
def molecular_weight_1_mole_acetic_acid := molecular_weight_7_moles_acetic_acid / moles_acetic_acid

-- The theorem stating that given the molecular weight of 7 moles of acetic acid, we have the molecular weight of the acetic acid
theorem molecular_weight_of_acetic_acid : molecular_weight_1_mole_acetic_acid = 60 := by
  -- proof to be solved
  sorry

end NUMINAMATH_GPT_molecular_weight_of_acetic_acid_l1988_198870


namespace NUMINAMATH_GPT_exponentiation_identity_l1988_198803

variable {a : ℝ}

theorem exponentiation_identity : (-a) ^ 2 * a ^ 3 = a ^ 5 := sorry

end NUMINAMATH_GPT_exponentiation_identity_l1988_198803


namespace NUMINAMATH_GPT_solve_fruit_juice_problem_l1988_198864

open Real

noncomputable def fruit_juice_problem : Prop :=
  ∃ x, ((0.12 * 3 + x) / (3 + x) = 0.185) ∧ (x = 0.239)

theorem solve_fruit_juice_problem : fruit_juice_problem :=
sorry

end NUMINAMATH_GPT_solve_fruit_juice_problem_l1988_198864


namespace NUMINAMATH_GPT_quadrilateral_area_l1988_198835

/-
Proof Statement: For a square with a side length of 8 cm, each of whose sides is divided by a point into two equal segments, 
prove that the area of the quadrilateral formed by connecting these points is 32 cm².
-/

theorem quadrilateral_area (side_len : ℝ) (h : side_len = 8) :
  let quadrilateral_area := (side_len * side_len) / 2
  quadrilateral_area = 32 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1988_198835


namespace NUMINAMATH_GPT_math_proof_equivalent_l1988_198862

theorem math_proof_equivalent :
  (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end NUMINAMATH_GPT_math_proof_equivalent_l1988_198862


namespace NUMINAMATH_GPT_extra_discount_percentage_l1988_198815

theorem extra_discount_percentage 
  (initial_price : ℝ)
  (first_discount : ℝ)
  (new_price : ℝ)
  (final_price : ℝ)
  (extra_discount_amount : ℝ)
  (x : ℝ)
  (discount_formula : x = (extra_discount_amount * 100) / new_price) :
  initial_price = 50 ∧ 
  first_discount = 2.08 ∧ 
  new_price = 47.92 ∧ 
  final_price = 46 ∧ 
  extra_discount_amount = new_price - final_price → 
  x = 4 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_extra_discount_percentage_l1988_198815


namespace NUMINAMATH_GPT_calculate_expression_l1988_198818

variables (x y : ℝ)

theorem calculate_expression (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x - y) / (Real.sqrt x + Real.sqrt y) - (x - 2 * Real.sqrt (x * y) + y) / (Real.sqrt x - Real.sqrt y) = 0 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1988_198818


namespace NUMINAMATH_GPT_value_of_y_l1988_198845

theorem value_of_y (x y : ℝ) (h1 : 3 * x = 0.75 * y) (h2 : x = 24) : y = 96 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l1988_198845


namespace NUMINAMATH_GPT_total_money_at_least_108_l1988_198879

-- Definitions for the problem
def tram_ticket_cost : ℕ := 1
def passenger_coins (n : ℕ) : Prop := n = 2 ∨ n = 5

-- Condition that conductor had no change initially
def initial_conductor_money : ℕ := 0

-- Condition that each passenger can pay exactly 1 Ft and receive change
def can_pay_ticket_with_change (coins : List ℕ) : Prop := 
  ∀ c ∈ coins, passenger_coins c → 
    ∃ change : List ℕ, (change.sum = c - tram_ticket_cost) ∧ 
      (∀ x ∈ change, passenger_coins x)

-- Assume we have 20 passengers with only 2 Ft and 5 Ft coins
def passengers_coins : List (List ℕ) :=
  -- Simplified representation
  List.replicate 20 [2, 5]

noncomputable def total_passenger_money : ℕ :=
  (passengers_coins.map List.sum).sum

-- Lean statement for the proof problem
theorem total_money_at_least_108 : total_passenger_money ≥ 108 :=
sorry

end NUMINAMATH_GPT_total_money_at_least_108_l1988_198879


namespace NUMINAMATH_GPT_loaned_out_books_l1988_198898

def initial_books : ℕ := 75
def added_books : ℕ := 10 + 15 + 6
def removed_books : ℕ := 3 + 2 + 4
def end_books : ℕ := 90
def return_percentage : ℝ := 0.80

theorem loaned_out_books (L : ℕ) :
  (end_books - initial_books = added_books - removed_books - ⌊(1 - return_percentage) * L⌋) →
  (L = 35) :=
sorry

end NUMINAMATH_GPT_loaned_out_books_l1988_198898


namespace NUMINAMATH_GPT_solve_investment_problem_l1988_198802

def remaining_rate_proof (A I A1 R1 A2 R2 x : ℚ) : Prop :=
  let income1 := A1 * (R1 / 100)
  let income2 := A2 * (R2 / 100)
  let remaining := A - A1 - A2
  let required_income := I - (income1 + income2)
  let expected_rate_in_float := (required_income / remaining) * 100
  expected_rate_in_float = x

theorem solve_investment_problem :
  remaining_rate_proof 15000 800 5000 3 6000 4.5 9.5 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_solve_investment_problem_l1988_198802


namespace NUMINAMATH_GPT_factor_expression_l1988_198878

-- Define variables s and m
variables (s m : ℤ)

-- State the theorem to be proven: If s = 5, then m^2 - sm - 24 can be factored as (m - 8)(m + 3)
theorem factor_expression (hs : s = 5) : m^2 - s * m - 24 = (m - 8) * (m + 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_factor_expression_l1988_198878


namespace NUMINAMATH_GPT_amy_money_left_l1988_198824

-- Definitions for item prices
def stuffed_toy_price : ℝ := 2
def hot_dog_price : ℝ := 3.5
def candy_apple_price : ℝ := 1.5
def soda_price : ℝ := 1.75
def ferris_wheel_ticket_price : ℝ := 2.5

-- Tax rate
def tax_rate : ℝ := 0.1 

-- Initial amount Amy had
def initial_amount : ℝ := 15

-- Function to calculate price including tax
def price_with_tax (price : ℝ) (tax_rate : ℝ) : ℝ := price * (1 + tax_rate)

-- Prices including tax
def stuffed_toy_price_with_tax := price_with_tax stuffed_toy_price tax_rate
def hot_dog_price_with_tax := price_with_tax hot_dog_price tax_rate
def candy_apple_price_with_tax := price_with_tax candy_apple_price tax_rate
def soda_price_with_tax := price_with_tax soda_price tax_rate
def ferris_wheel_ticket_price_with_tax := price_with_tax ferris_wheel_ticket_price tax_rate

-- Discount rates
def discount_most_expensive : ℝ := 0.5
def discount_second_most_expensive : ℝ := 0.25

-- Applying discounts
def discounted_hot_dog_price := hot_dog_price_with_tax * (1 - discount_most_expensive)
def discounted_ferris_wheel_ticket_price := ferris_wheel_ticket_price_with_tax * (1 - discount_second_most_expensive)

-- Total cost with discounts
def total_cost_with_discounts : ℝ := 
  stuffed_toy_price_with_tax + discounted_hot_dog_price + candy_apple_price_with_tax +
  soda_price_with_tax + discounted_ferris_wheel_ticket_price

-- Amount left after purchases
def amount_left : ℝ := initial_amount - total_cost_with_discounts

theorem amy_money_left : amount_left = 5.23 := by
  -- Here the proof will be provided.
  sorry

end NUMINAMATH_GPT_amy_money_left_l1988_198824


namespace NUMINAMATH_GPT_pie_distribution_l1988_198849

theorem pie_distribution (x y : ℕ) (h1 : x + y + 2 * x = 13) (h2 : x < y) (h3 : y < 2 * x) :
  x = 3 ∧ y = 4 ∧ 2 * x = 6 := by
  sorry

end NUMINAMATH_GPT_pie_distribution_l1988_198849


namespace NUMINAMATH_GPT_domain_of_tan_l1988_198861

open Real

noncomputable def function_domain : Set ℝ :=
  {x | ∀ k : ℤ, x ≠ k * π + 3 * π / 4}

theorem domain_of_tan : ∀ x : ℝ,
  (∃ k : ℤ, x = k * π + 3 * π / 4) → ¬ (∃ y : ℝ, y = tan (π / 4 - x)) :=
by
  intros x hx
  obtain ⟨k, hk⟩ := hx
  sorry

end NUMINAMATH_GPT_domain_of_tan_l1988_198861


namespace NUMINAMATH_GPT_remainder_of_product_modulo_12_l1988_198807

theorem remainder_of_product_modulo_12 : (1625 * 1627 * 1629) % 12 = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_of_product_modulo_12_l1988_198807


namespace NUMINAMATH_GPT_avg_temp_in_october_l1988_198844

theorem avg_temp_in_october (a A : ℝ)
  (h1 : 28 = a + A)
  (h2 : 18 = a - A)
  (x := 10)
  (temperature : ℝ := a + A * Real.cos (π / 6 * (x - 6))) :
  temperature = 20.5 :=
by
  sorry

end NUMINAMATH_GPT_avg_temp_in_october_l1988_198844


namespace NUMINAMATH_GPT_distinct_integer_roots_iff_l1988_198894

theorem distinct_integer_roots_iff (a : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ a = -2 ∨ a = 18 :=
by
  sorry

end NUMINAMATH_GPT_distinct_integer_roots_iff_l1988_198894


namespace NUMINAMATH_GPT_embankment_construction_l1988_198832

theorem embankment_construction :
  (∃ r : ℚ, 0 < r ∧ (1 / 2 = 60 * r * 3)) →
  (∃ t : ℕ, 1 = 45 * 1 / 360 * t) :=
by
  sorry

end NUMINAMATH_GPT_embankment_construction_l1988_198832


namespace NUMINAMATH_GPT_supermarket_flour_import_l1988_198871

theorem supermarket_flour_import :
  let long_grain_rice := (9 : ℚ) / 20
  let glutinous_rice := (7 : ℚ) / 20
  let combined_rice := long_grain_rice + glutinous_rice
  let less_amount := (3 : ℚ) / 20
  let flour : ℚ := combined_rice - less_amount
  flour = (13 : ℚ) / 20 :=
by
  sorry

end NUMINAMATH_GPT_supermarket_flour_import_l1988_198871


namespace NUMINAMATH_GPT_smallest_positive_angle_l1988_198896

theorem smallest_positive_angle :
  ∃ y : ℝ, 0 < y ∧ y < 90 ∧ (6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 / 2) ∧ y = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_l1988_198896


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1988_198890

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (4 - x)}
def B : Set ℝ := {x | x > 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 1 < x ∧ x ≤ 4} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1988_198890


namespace NUMINAMATH_GPT_expected_value_winnings_l1988_198801

def probability_heads : ℚ := 2 / 5
def probability_tails : ℚ := 3 / 5
def win_amount_heads : ℚ := 5
def lose_amount_tails : ℚ := -4

theorem expected_value_winnings : 
  probability_heads * win_amount_heads + probability_tails * lose_amount_tails = -2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_expected_value_winnings_l1988_198801


namespace NUMINAMATH_GPT_total_selling_price_l1988_198865

theorem total_selling_price (cost1 cost2 cost3 : ℕ) (profit1 profit2 profit3 : ℚ) 
  (h1 : cost1 = 280) (h2 : cost2 = 350) (h3 : cost3 = 500) 
  (h4 : profit1 = 30) (h5 : profit2 = 45) (h6 : profit3 = 25) : 
  (cost1 + (profit1 / 100) * cost1) + (cost2 + (profit2 / 100) * cost2) + (cost3 + (profit3 / 100) * cost3) = 1496.5 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_l1988_198865


namespace NUMINAMATH_GPT_countMultiplesOf30Between900And27000_l1988_198827

noncomputable def smallestPerfectSquareDivisibleBy30 : ℕ :=
  900

noncomputable def smallestPerfectCubeDivisibleBy30 : ℕ :=
  27000

theorem countMultiplesOf30Between900And27000 :
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  upper_bound - lower_bound + 1 = 871 :=
  by
  let lower_bound := smallestPerfectSquareDivisibleBy30 / 30;
  let upper_bound := smallestPerfectCubeDivisibleBy30 / 30;
  show upper_bound - lower_bound + 1 = 871;
  sorry

end NUMINAMATH_GPT_countMultiplesOf30Between900And27000_l1988_198827


namespace NUMINAMATH_GPT_difference_of_squares_l1988_198831

theorem difference_of_squares (a b : ℝ) : -4 * a^2 + b^2 = (b + 2 * a) * (b - 2 * a) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_l1988_198831


namespace NUMINAMATH_GPT_prize_distribution_l1988_198887

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem prize_distribution :
  let total_ways := 
    (binomial_coefficient 7 3) * 5 * (Nat.factorial 4) + 
    (binomial_coefficient 7 2 * binomial_coefficient 5 2 / 2) * 
    (binomial_coefficient 5 2) * (Nat.factorial 3)
  total_ways = 10500 :=
by 
  sorry

end NUMINAMATH_GPT_prize_distribution_l1988_198887


namespace NUMINAMATH_GPT_exists_divisor_for_all_f_values_l1988_198826

theorem exists_divisor_for_all_f_values (f : ℕ → ℕ) (h_f_range : ∀ n, 1 < f n) (h_f_div : ∀ m n, f (m + n) ∣ f m + f n) :
  ∃ c : ℕ, c > 1 ∧ ∀ n, c ∣ f n := 
sorry

end NUMINAMATH_GPT_exists_divisor_for_all_f_values_l1988_198826


namespace NUMINAMATH_GPT_tan_double_angle_l1988_198897

open Real

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) : tan (2 * α) = 3 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1988_198897


namespace NUMINAMATH_GPT_evaluate_expression_l1988_198895

theorem evaluate_expression : (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1988_198895


namespace NUMINAMATH_GPT_max_zeros_consecutive_two_digit_product_l1988_198839

theorem max_zeros_consecutive_two_digit_product :
  ∃ a b : ℕ, 10 ≤ a ∧ a < 100 ∧ b = a + 1 ∧ 10 ≤ b ∧ b < 100 ∧
  (∀ c, (c * 10) ∣ a * b → c ≤ 2) := 
  by
    sorry

end NUMINAMATH_GPT_max_zeros_consecutive_two_digit_product_l1988_198839


namespace NUMINAMATH_GPT_trapezoid_geometry_proof_l1988_198873

theorem trapezoid_geometry_proof
  (midline_length : ℝ)
  (segment_midpoints : ℝ)
  (angle1 angle2 : ℝ)
  (h_midline : midline_length = 5)
  (h_segment_midpoints : segment_midpoints = 3)
  (h_angle1 : angle1 = 30)
  (h_angle2 : angle2 = 60) :
  ∃ (AD BC AB : ℝ), AD = 8 ∧ BC = 2 ∧ AB = 3 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_geometry_proof_l1988_198873


namespace NUMINAMATH_GPT_inequality_solution_set_l1988_198875

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 else -1

theorem inequality_solution_set :
  { x : ℝ | (x+1) * f x > 2 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 1 } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1988_198875


namespace NUMINAMATH_GPT_percentage_less_than_l1988_198876

variable (x y z n : ℝ)
variable (hx : x = 8 * y)
variable (hy : y = 2 * |z - n|)
variable (hz : z = 1.1 * n)

theorem percentage_less_than (hx : x = 8 * y) (hy : y = 2 * |z - n|) (hz : z = 1.1 * n) :
  ((x - y) / x) * 100 = 87.5 := sorry

end NUMINAMATH_GPT_percentage_less_than_l1988_198876


namespace NUMINAMATH_GPT_initial_storks_count_l1988_198819

-- Definitions based on the conditions provided
def initialBirds : ℕ := 3
def additionalStorks : ℕ := 6
def totalBirdsAndStorks : ℕ := 13

-- The mathematical statement to be proved
theorem initial_storks_count (S : ℕ) (h : initialBirds + S + additionalStorks = totalBirdsAndStorks) : S = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_storks_count_l1988_198819


namespace NUMINAMATH_GPT_portia_high_school_students_l1988_198833

theorem portia_high_school_students
  (L P M : ℕ)
  (h1 : P = 4 * L)
  (h2 : M = 2 * L)
  (h3 : P + L + M = 4200) :
  P = 2400 :=
sorry

end NUMINAMATH_GPT_portia_high_school_students_l1988_198833


namespace NUMINAMATH_GPT_find_tangent_perpendicular_t_l1988_198852

noncomputable def y (x : ℝ) : ℝ := x * Real.log x

theorem find_tangent_perpendicular_t (t : ℝ) (ht : 0 < t) (h_perpendicular : (1 : ℝ) * (1 + Real.log t) = -1) :
  t = Real.exp (-2) :=
by
  sorry

end NUMINAMATH_GPT_find_tangent_perpendicular_t_l1988_198852


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1988_198857

theorem relationship_between_a_and_b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b)
    (h₃ : ∀ x : ℝ, |(3 * x + 1) - 4| < a → |x - 1| < b) : a ≥ 3 * b :=
by
  -- Applying the given conditions, we want to demonstrate that a ≥ 3b.
  sorry

end NUMINAMATH_GPT_relationship_between_a_and_b_l1988_198857


namespace NUMINAMATH_GPT_geom_seq_identity_l1988_198866

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, ∃ r, a (n+1) = r * a n

theorem geom_seq_identity (a : ℕ → ℝ) (r : ℝ) (h1 : geometric_sequence a) (h2 : a 2 + a 4 = 2) :
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 4 := 
  sorry

end NUMINAMATH_GPT_geom_seq_identity_l1988_198866


namespace NUMINAMATH_GPT_max_value_of_m_l1988_198884

theorem max_value_of_m (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (2 / a) + (1 / b) = 1 / 4) : 2 * a + b ≥ 36 :=
by 
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_max_value_of_m_l1988_198884


namespace NUMINAMATH_GPT_find_a2_and_sum_l1988_198809

theorem find_a2_and_sum (a a1 a2 a3 a4 : ℝ) (x : ℝ) (h1 : (1 + 2 * x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a2 = 24 ∧ a + a1 + a2 + a3 + a4 = 81 :=
by
  sorry

end NUMINAMATH_GPT_find_a2_and_sum_l1988_198809


namespace NUMINAMATH_GPT_distance_traveled_l1988_198846

-- Define constants for speed and time
def speed : ℝ := 60
def time : ℝ := 5

-- Define the expected distance
def expected_distance : ℝ := 300

-- Theorem statement
theorem distance_traveled : speed * time = expected_distance :=
by
  sorry

end NUMINAMATH_GPT_distance_traveled_l1988_198846


namespace NUMINAMATH_GPT_find_C_l1988_198868

-- Variables and conditions
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + B + C = 1000
def condition2 : Prop := A + C = 700
def condition3 : Prop := B + C = 600

-- The statement to be proved
theorem find_C (h1 : condition1 A B C) (h2 : condition2 A C) (h3 : condition3 B C) : C = 300 :=
sorry

end NUMINAMATH_GPT_find_C_l1988_198868


namespace NUMINAMATH_GPT_value_of_k_l1988_198853

theorem value_of_k (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = (k - 1) * x + k^2 - 1)
  (h2 : ∃ m : ℝ, y = m * x)
  (h3 : k ≠ 1) :
  k = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1988_198853


namespace NUMINAMATH_GPT_point_inside_circle_l1988_198880

theorem point_inside_circle (O A : Type) (r OA : ℝ) (h1 : r = 6) (h2 : OA = 5) :
  OA < r :=
by
  sorry

end NUMINAMATH_GPT_point_inside_circle_l1988_198880


namespace NUMINAMATH_GPT_smallest_divisor_28_l1988_198893

theorem smallest_divisor_28 : ∃ (d : ℕ), d > 0 ∧ d ∣ 28 ∧ ∀ (d' : ℕ), d' > 0 ∧ d' ∣ 28 → d ≤ d' := by
  sorry

end NUMINAMATH_GPT_smallest_divisor_28_l1988_198893


namespace NUMINAMATH_GPT_find_x0_l1988_198804

/-- Given that the tangent line to the curve y = x^2 - 1 at the point x = x0 is parallel 
to the tangent line to the curve y = 1 - x^3 at the point x = x0, prove that x0 = 0 
or x0 = -2/3. -/
theorem find_x0 (x0 : ℝ) (h : (∃ x0, (2 * x0) = (-3 * x0 ^ 2))) : x0 = 0 ∨ x0 = -2/3 := 
sorry

end NUMINAMATH_GPT_find_x0_l1988_198804


namespace NUMINAMATH_GPT_paint_quantity_l1988_198820

variable (totalPaint : ℕ) (blueRatio greenRatio whiteRatio : ℕ)

theorem paint_quantity 
  (h_total_paint : totalPaint = 45)
  (h_ratio_blue : blueRatio = 5)
  (h_ratio_green : greenRatio = 3)
  (h_ratio_white : whiteRatio = 7) :
  let totalRatio := blueRatio + greenRatio + whiteRatio
  let partQuantity := totalPaint / totalRatio
  let bluePaint := blueRatio * partQuantity
  let greenPaint := greenRatio * partQuantity
  let whitePaint := whiteRatio * partQuantity
  bluePaint = 15 ∧ greenPaint = 9 ∧ whitePaint = 21 :=
by
  sorry

end NUMINAMATH_GPT_paint_quantity_l1988_198820


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_problem_e_l1988_198808

section problem_a
  -- Conditions
  def rainbow_russian_first_letters_sequence := ["к", "о", "ж", "з", "г", "с", "ф"]
  
  -- Theorem (question == answer)
  theorem problem_a : rainbow_russian_first_letters_sequence[4] = "г" ∧
                      rainbow_russian_first_letters_sequence[5] = "с" ∧
                      rainbow_russian_first_letters_sequence[6] = "ф" :=
  by
    -- Skip proof: sorry
    sorry
end problem_a

section problem_b
  -- Conditions
  def russian_alphabet_alternating_sequence := ["а", "в", "г", "ё", "ж", "з", "л", "м", "н", "о", "п", "т", "у"]
 
  -- Theorem (question == answer)
  theorem problem_b : russian_alphabet_alternating_sequence[10] = "п" ∧
                      russian_alphabet_alternating_sequence[11] = "т" ∧
                      russian_alphabet_alternating_sequence[12] = "у" :=
  by
    -- Skip proof: sorry
    sorry
end problem_b

section problem_c
  -- Conditions
  def russian_number_of_letters_sequence := ["один", "четыре", "шесть", "пять", "семь", "восемь"]
  
  -- Theorem (question == answer)
  theorem problem_c : russian_number_of_letters_sequence[4] = "семь" ∧
                      russian_number_of_letters_sequence[5] = "восемь" :=
  by
    -- Skip proof: sorry
    sorry
end problem_c

section problem_d
  -- Conditions
  def approximate_symmetry_letters_sequence := ["Ф", "Х", "Ш", "В"]

  -- Theorem (question == answer)
  theorem problem_d : approximate_symmetry_letters_sequence[3] = "В" :=
  by
    -- Skip proof: sorry
    sorry
end problem_d

section problem_e
  -- Conditions
  def russian_loops_in_digit_sequence := ["0", "д", "т", "ч", "п", "ш", "с", "в", "д"]

  -- Theorem (question == answer)
  theorem problem_e : russian_loops_in_digit_sequence[7] = "в" ∧
                      russian_loops_in_digit_sequence[8] = "д" :=
  by
    -- Skip proof: sorry
    sorry
end problem_e

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_problem_e_l1988_198808


namespace NUMINAMATH_GPT_range_of_k_value_of_k_l1988_198837

-- Defining the quadratic equation having two real roots condition
def has_real_roots (k : ℝ) : Prop :=
  let Δ := 9 - 4 * (k - 2)
  Δ ≥ 0

-- First part: range of k
theorem range_of_k (k : ℝ) : has_real_roots k ↔ k ≤ 17 / 4 :=
  sorry

-- Second part: specific value of k given additional condition
theorem value_of_k (x1 x2 k : ℝ) (h1 : (x1 + x2) = 3) (h2 : (x1 * x2) = k - 2) (h3 : (x1 + x2 - x1 * x2) = 1) : k = 4 :=
  sorry

end NUMINAMATH_GPT_range_of_k_value_of_k_l1988_198837


namespace NUMINAMATH_GPT_expenditures_ratio_l1988_198848

open Real

variables (I1 I2 E1 E2 : ℝ)
variables (x : ℝ)

theorem expenditures_ratio 
  (h1 : I1 = 4500)
  (h2 : I1 / I2 = 5 / 4)
  (h3 : I1 - E1 = 1800)
  (h4 : I2 - E2 = 1800) : 
  E1 / E2 = 3 / 2 :=
by
  have h5 : I1 / 5 = x := by sorry
  have h6 : I2 = 4 * x := by sorry
  have h7 : I2 = 3600 := by sorry
  have h8 : E1 = 2700 := by sorry
  have h9 : E2 = 1800 := by sorry
  exact sorry 

end NUMINAMATH_GPT_expenditures_ratio_l1988_198848


namespace NUMINAMATH_GPT_average_salary_of_technicians_l1988_198850

theorem average_salary_of_technicians
  (total_workers : ℕ)
  (avg_salary_all_workers : ℕ)
  (total_technicians : ℕ)
  (avg_salary_non_technicians : ℕ)
  (h1 : total_workers = 18)
  (h2 : avg_salary_all_workers = 8000)
  (h3 : total_technicians = 6)
  (h4 : avg_salary_non_technicians = 6000) :
  (72000 / total_technicians) = 12000 := 
  sorry

end NUMINAMATH_GPT_average_salary_of_technicians_l1988_198850


namespace NUMINAMATH_GPT_tomatoes_picked_yesterday_l1988_198825

-- Definitions corresponding to the conditions in the problem.
def initial_tomatoes : Nat := 160
def tomatoes_left_after_yesterday : Nat := 104

-- Statement of the problem proving the number of tomatoes picked yesterday.
theorem tomatoes_picked_yesterday : initial_tomatoes - tomatoes_left_after_yesterday = 56 :=
by
  sorry

end NUMINAMATH_GPT_tomatoes_picked_yesterday_l1988_198825


namespace NUMINAMATH_GPT_smallest_number_is_10_l1988_198867

/-- Define the set of numbers. -/
def numbers : List Int := [10, 11, 12, 13, 14]

theorem smallest_number_is_10 :
  ∃ n ∈ numbers, (∀ m ∈ numbers, n ≤ m) ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_is_10_l1988_198867


namespace NUMINAMATH_GPT_average_price_of_cow_l1988_198836

variable (price_cow price_goat : ℝ)

theorem average_price_of_cow (h1 : 2 * price_cow + 8 * price_goat = 1400)
                             (h2 : price_goat = 60) :
                             price_cow = 460 := 
by
  -- The following line allows the Lean code to compile successfully without providing a proof.
  sorry

end NUMINAMATH_GPT_average_price_of_cow_l1988_198836


namespace NUMINAMATH_GPT_circle_equation_m_l1988_198842
open Real

theorem circle_equation_m (m : ℝ) : (x^2 + y^2 + 4 * x + 2 * y + m = 0 → m < 5) := sorry

end NUMINAMATH_GPT_circle_equation_m_l1988_198842


namespace NUMINAMATH_GPT_weight_of_b_l1988_198810

theorem weight_of_b (a b c : ℝ) (h1 : (a + b + c) / 3 = 45) (h2 : (a + b) / 2 = 40) (h3 : (b + c) / 2 = 43) : b = 31 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_b_l1988_198810


namespace NUMINAMATH_GPT_greatest_integer_x_l1988_198881

theorem greatest_integer_x
    (x : ℤ) : 
    (7 / 9 : ℚ) > (x : ℚ) / 13 → x ≤ 10 :=
by
    sorry

end NUMINAMATH_GPT_greatest_integer_x_l1988_198881


namespace NUMINAMATH_GPT_balloon_volume_safety_l1988_198882

theorem balloon_volume_safety (p V : ℝ) (h_prop : p = 90 / V) (h_burst : p ≤ 150) : 0.6 ≤ V :=
by {
  sorry
}

end NUMINAMATH_GPT_balloon_volume_safety_l1988_198882


namespace NUMINAMATH_GPT_symmetric_circle_eq_l1988_198822

/-- The definition of the original circle equation. -/
def original_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The definition of the line of symmetry equation. -/
def line_eq (x y : ℝ) : Prop := x - y - 2 = 0

/-- The statement that the equation of the circle that is symmetric to the original circle 
    about the given line is (x - 4)^2 + (y + 1)^2 = 1. -/
theorem symmetric_circle_eq : 
  (∃ x y : ℝ, original_circle_eq x y ∧ line_eq x y) →
  (∀ x y : ℝ, (x - 4)^2 + (y + 1)^2 = 1) :=
by sorry

end NUMINAMATH_GPT_symmetric_circle_eq_l1988_198822


namespace NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1988_198891

theorem tan_alpha_minus_pi_over_4 
  (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (β + π/4) = 3) 
  : Real.tan (α - π/4) = -1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_minus_pi_over_4_l1988_198891


namespace NUMINAMATH_GPT_Arrow_velocity_at_impact_l1988_198800

def Edward_initial_distance := 1875 -- \(\text{ft}\)
def Edward_initial_velocity := 0 -- \(\text{ft/s}\)
def Edward_acceleration := 1 -- \(\text{ft/s}^2\)
def Arrow_initial_distance := 0 -- \(\text{ft}\)
def Arrow_initial_velocity := 100 -- \(\text{ft/s}\)
def Arrow_deceleration := -1 -- \(\text{ft/s}^2\)
def time_impact := 25 -- \(\text{s}\)

theorem Arrow_velocity_at_impact : 
  (Arrow_initial_velocity + Arrow_deceleration * time_impact) = 75 := 
by
  sorry

end NUMINAMATH_GPT_Arrow_velocity_at_impact_l1988_198800


namespace NUMINAMATH_GPT_area_difference_equal_28_5_l1988_198899

noncomputable def square_side_length (d: ℝ) : ℝ := d / Real.sqrt 2
noncomputable def square_area (d: ℝ) : ℝ := (square_side_length d) ^ 2
noncomputable def circle_radius (D: ℝ) : ℝ := D / 2
noncomputable def circle_area (D: ℝ) : ℝ := Real.pi * (circle_radius D) ^ 2
noncomputable def area_difference (d D : ℝ) : ℝ := |circle_area D - square_area d|

theorem area_difference_equal_28_5 :
  ∀ (d D : ℝ), d = 10 → D = 10 → area_difference d D = 28.5 :=
by
  intros d D hd hD
  rw [hd, hD]
  -- Remaining steps involve computing the known areas and their differences
  sorry

end NUMINAMATH_GPT_area_difference_equal_28_5_l1988_198899


namespace NUMINAMATH_GPT_existence_of_ab_l1988_198817

theorem existence_of_ab (n : ℕ) (hn : 0 < n) : ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ n ∣ (4 * a^2 + 9 * b^2 - 1) :=
by 
  sorry

end NUMINAMATH_GPT_existence_of_ab_l1988_198817


namespace NUMINAMATH_GPT_sqrt_mul_l1988_198805

theorem sqrt_mul (h₁ : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3) : Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_mul_l1988_198805


namespace NUMINAMATH_GPT_find_a_l1988_198828

-- Given function and its condition
def f (a x : ℝ) := a * x ^ 3 + 3 * x ^ 2 + 2
def f' (a x : ℝ) := 3 * a * x ^ 2 + 6 * x

-- Condition and proof that a = -2 given the condition f'(-1) = -12
theorem find_a 
  (a : ℝ)
  (h : f' a (-1) = -12) : 
  a = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l1988_198828


namespace NUMINAMATH_GPT_proof_problem_l1988_198812

theorem proof_problem (x : ℝ) 
    (h1 : (x - 1) * (x + 1) = x^2 - 1)
    (h2 : (x - 1) * (x^2 + x + 1) = x^3 - 1)
    (h3 : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1)
    (h4 : (x - 1) * (x^4 + x^3 + x^2 + x + 1) = -2) :
    x^2023 = -1 := 
by 
  sorry -- Proof is omitted

end NUMINAMATH_GPT_proof_problem_l1988_198812


namespace NUMINAMATH_GPT_aria_spent_on_cookies_l1988_198856

def aria_spent : ℕ := 2356

theorem aria_spent_on_cookies :
  (let cookies_per_day := 4
  let cost_per_cookie := 19
  let days_in_march := 31
  let total_cookies := days_in_march * cookies_per_day
  let total_cost := total_cookies * cost_per_cookie
  total_cost = aria_spent) :=
  sorry

end NUMINAMATH_GPT_aria_spent_on_cookies_l1988_198856


namespace NUMINAMATH_GPT_return_trip_amount_l1988_198863

noncomputable def gasoline_expense : ℝ := 8
noncomputable def lunch_expense : ℝ := 15.65
noncomputable def gift_expense_per_person : ℝ := 5
noncomputable def grandma_gift_per_person : ℝ := 10
noncomputable def initial_amount : ℝ := 50

theorem return_trip_amount : 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  initial_amount - total_expense + total_money_gifted = 36.35 :=
by 
  let total_expense := gasoline_expense + lunch_expense + (gift_expense_per_person * 2)
  let total_money_gifted := grandma_gift_per_person * 2
  sorry

end NUMINAMATH_GPT_return_trip_amount_l1988_198863


namespace NUMINAMATH_GPT_minimum_bamboo_fencing_length_l1988_198851

theorem minimum_bamboo_fencing_length 
  (a b z : ℝ) 
  (h1 : a * b = 50)
  (h2 : a + 2 * b = z) : 
  z ≥ 20 := 
  sorry

end NUMINAMATH_GPT_minimum_bamboo_fencing_length_l1988_198851


namespace NUMINAMATH_GPT_percentage_girls_l1988_198834

theorem percentage_girls (initial_boys : ℕ) (initial_girls : ℕ) (added_boys : ℕ) :
  initial_boys = 11 → initial_girls = 13 → added_boys = 1 → 
  100 * initial_girls / (initial_boys + added_boys + initial_girls) = 52 :=
by
  intros h_boys h_girls h_added
  sorry

end NUMINAMATH_GPT_percentage_girls_l1988_198834


namespace NUMINAMATH_GPT_original_price_l1988_198888

theorem original_price (P : ℕ) (h : (1 / 8) * P = 8) : P = 64 :=
sorry

end NUMINAMATH_GPT_original_price_l1988_198888


namespace NUMINAMATH_GPT_age_of_15th_person_l1988_198840

theorem age_of_15th_person (avg_16 : ℝ) (avg_5 : ℝ) (avg_9 : ℝ) (total_16 : ℝ) (total_5 : ℝ) (total_9 : ℝ) :
  avg_16 = 15 ∧ avg_5 = 14 ∧ avg_9 = 16 ∧
  total_16 = 16 * avg_16 ∧ total_5 = 5 * avg_5 ∧ total_9 = 9 * avg_9 →
  (total_16 - total_5 - total_9) = 26 :=
by
  sorry

end NUMINAMATH_GPT_age_of_15th_person_l1988_198840


namespace NUMINAMATH_GPT_range_of_a_l1988_198823

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 < x → (a / x - 4 / x^2 < 1)) → a < 4 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1988_198823


namespace NUMINAMATH_GPT_prove_necessary_but_not_sufficient_l1988_198872

noncomputable def necessary_but_not_sufficient_condition (m : ℝ) :=
  (∀ x : ℝ, x^2 + 2*x + m > 0) → (m > 0) ∧ ¬ (∀ x : ℝ, x^2 + 2*x + m > 0 → m <= 1)

theorem prove_necessary_but_not_sufficient
    (m : ℝ) :
    necessary_but_not_sufficient_condition m :=
by
  sorry

end NUMINAMATH_GPT_prove_necessary_but_not_sufficient_l1988_198872


namespace NUMINAMATH_GPT_hcf_36_84_l1988_198886

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_GPT_hcf_36_84_l1988_198886


namespace NUMINAMATH_GPT_biggest_number_l1988_198830

theorem biggest_number (A B C D : ℕ) (h1 : A / B = 2 / 3) (h2 : B / C = 3 / 4) (h3 : C / D = 4 / 5) (h4 : A + B + C + D = 1344) : D = 480 := 
sorry

end NUMINAMATH_GPT_biggest_number_l1988_198830


namespace NUMINAMATH_GPT_running_race_total_students_l1988_198885

theorem running_race_total_students 
  (number_of_first_grade_students number_of_second_grade_students : ℕ)
  (h1 : number_of_first_grade_students = 8)
  (h2 : number_of_second_grade_students = 5 * number_of_first_grade_students) :
  number_of_first_grade_students + number_of_second_grade_students = 48 := 
by
  -- we will leave the proof empty
  sorry

end NUMINAMATH_GPT_running_race_total_students_l1988_198885


namespace NUMINAMATH_GPT_find_f_expression_l1988_198860

theorem find_f_expression (f : ℝ → ℝ) (x : ℝ) (h : f (Real.log x) = 3 * x + 4) : 
  f x = 3 * Real.exp x + 4 := 
by
  sorry

end NUMINAMATH_GPT_find_f_expression_l1988_198860


namespace NUMINAMATH_GPT_net_price_change_l1988_198855

theorem net_price_change (P : ℝ) : 
  let decreased_price := P * (1 - 0.30)
  let increased_price := decreased_price * (1 + 0.20)
  increased_price - P = -0.16 * P :=
by
  -- The proof would go here. We just need the statement as per the prompt.
  sorry

end NUMINAMATH_GPT_net_price_change_l1988_198855


namespace NUMINAMATH_GPT_find_original_class_strength_l1988_198859

-- Definitions based on given conditions
def original_average_age : ℝ := 40
def additional_students : ℕ := 12
def new_students_average_age : ℝ := 32
def decrease_in_average : ℝ := 4
def new_average_age : ℝ := original_average_age - decrease_in_average

-- The equation setup
theorem find_original_class_strength (N : ℕ) (T : ℝ) 
  (h1 : T = original_average_age * N) 
  (h2 : T + additional_students * new_students_average_age = new_average_age * (N + additional_students)) : 
  N = 12 := 
sorry

end NUMINAMATH_GPT_find_original_class_strength_l1988_198859


namespace NUMINAMATH_GPT_remaining_books_l1988_198816

def initial_books : Nat := 500
def num_people_donating : Nat := 10
def books_per_person : Nat := 8
def borrowed_books : Nat := 220

theorem remaining_books :
  (initial_books + num_people_donating * books_per_person - borrowed_books) = 360 := 
by 
  -- This will contain the mathematical proof
  sorry

end NUMINAMATH_GPT_remaining_books_l1988_198816


namespace NUMINAMATH_GPT_money_bounds_l1988_198847

variables (c d : ℝ)

theorem money_bounds :
  (7 * c + d > 84) ∧ (5 * c - d = 35) → (c > 9.92 ∧ d > 14.58) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_money_bounds_l1988_198847


namespace NUMINAMATH_GPT_polygon_six_sides_l1988_198813

theorem polygon_six_sides (n : ℕ) (h1 : (n - 2) * 180 = 2 * 360) : n = 6 := by
  sorry

end NUMINAMATH_GPT_polygon_six_sides_l1988_198813


namespace NUMINAMATH_GPT_eval_expression_l1988_198829

-- We define the expression that needs to be evaluated
def expression := (0.76)^3 - (0.1)^3 / (0.76)^2 + 0.076 + (0.1)^2

-- The statement to prove
theorem eval_expression : expression = 0.5232443982683983 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1988_198829
