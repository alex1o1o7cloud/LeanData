import Mathlib

namespace NUMINAMATH_GPT_distance_between_trees_l528_52830

theorem distance_between_trees
  (num_trees : ℕ)
  (length_of_yard : ℝ)
  (one_tree_at_each_end : True)
  (h1 : num_trees = 26)
  (h2 : length_of_yard = 400) :
  length_of_yard / (num_trees - 1) = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l528_52830


namespace NUMINAMATH_GPT_field_trip_count_l528_52865

theorem field_trip_count (vans: ℕ) (buses: ℕ) (people_per_van: ℕ) (people_per_bus: ℕ)
  (hv: vans = 9) (hb: buses = 10) (hpv: people_per_van = 8) (hpb: people_per_bus = 27):
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end NUMINAMATH_GPT_field_trip_count_l528_52865


namespace NUMINAMATH_GPT_find_positive_real_solutions_l528_52854

open Real

theorem find_positive_real_solutions 
  (x : ℝ) 
  (h : (1/3 * (4 * x^2 - 2)) = ((x^2 - 60 * x - 15) * (x^2 + 30 * x + 3))) :
  x = 30 + sqrt 917 ∨ x = -15 + (sqrt 8016) / 6 :=
by sorry

end NUMINAMATH_GPT_find_positive_real_solutions_l528_52854


namespace NUMINAMATH_GPT_unique_k_solves_eq_l528_52867

theorem unique_k_solves_eq (k : ℕ) (hpos_k : k > 0) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = k * a * b) ↔ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_k_solves_eq_l528_52867


namespace NUMINAMATH_GPT_gallons_bought_l528_52837

variable (total_needed : ℕ) (existing_paint : ℕ) (needed_more : ℕ)

theorem gallons_bought (H : total_needed = 70) (H1 : existing_paint = 36) (H2 : needed_more = 11) : 
  total_needed - existing_paint - needed_more = 23 := 
sorry

end NUMINAMATH_GPT_gallons_bought_l528_52837


namespace NUMINAMATH_GPT_row_length_in_feet_l528_52878

theorem row_length_in_feet (seeds_per_row : ℕ) (space_per_seed : ℕ) (inches_per_foot : ℕ) (H1 : seeds_per_row = 80) (H2 : space_per_seed = 18) (H3 : inches_per_foot = 12) : 
  seeds_per_row * space_per_seed / inches_per_foot = 120 :=
by
  sorry

end NUMINAMATH_GPT_row_length_in_feet_l528_52878


namespace NUMINAMATH_GPT_maximum_value_expression_l528_52885

-- Definitions
def f (x : ℝ) := -3 * x^2 + 18 * x - 1

-- Lean statement to prove that the maximum value of the function f is 26.
theorem maximum_value_expression : ∃ x : ℝ, f x = 26 :=
sorry

end NUMINAMATH_GPT_maximum_value_expression_l528_52885


namespace NUMINAMATH_GPT_probability_of_digit_six_l528_52890

theorem probability_of_digit_six :
  let total_numbers := 90
  let favorable_numbers := 18
  0 < total_numbers ∧ 0 < favorable_numbers →
  (favorable_numbers / total_numbers : ℚ) = 1 / 5 :=
by
  intros total_numbers favorable_numbers h
  sorry

end NUMINAMATH_GPT_probability_of_digit_six_l528_52890


namespace NUMINAMATH_GPT_total_winning_team_points_l528_52874

/-!
# Lean 4 Math Proof Problem

Prove that the total points scored by the winning team at the end of the game is 50 points given the conditions provided.
-/

-- Definitions
def losing_team_points_first_quarter : ℕ := 10
def winning_team_points_first_quarter : ℕ := 2 * losing_team_points_first_quarter
def winning_team_points_second_quarter : ℕ := winning_team_points_first_quarter + 10
def winning_team_points_third_quarter : ℕ := winning_team_points_second_quarter + 20

-- Theorem statement
theorem total_winning_team_points : winning_team_points_third_quarter = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_winning_team_points_l528_52874


namespace NUMINAMATH_GPT_mary_total_payment_l528_52831

def fixed_fee : ℕ := 17
def hourly_charge : ℕ := 7
def rental_duration : ℕ := 9
def total_payment (f : ℕ) (h : ℕ) (r : ℕ) : ℕ := f + (h * r)

theorem mary_total_payment:
  total_payment fixed_fee hourly_charge rental_duration = 80 :=
by
  sorry

end NUMINAMATH_GPT_mary_total_payment_l528_52831


namespace NUMINAMATH_GPT_cyclist_waits_15_minutes_l528_52808

-- Definitions
def hiker_rate := 7 -- miles per hour
def cyclist_rate := 28 -- miles per hour
def wait_time := 15 / 60 -- hours, as the cyclist waits 15 minutes, converted to hours

-- The statement to be proven
theorem cyclist_waits_15_minutes :
  ∃ t : ℝ, t = 15 / 60 ∧
  (∀ d : ℝ, d = (hiker_rate * wait_time) →
            d = (cyclist_rate * t - hiker_rate * t)) :=
by
  sorry

end NUMINAMATH_GPT_cyclist_waits_15_minutes_l528_52808


namespace NUMINAMATH_GPT_oranges_apples_bananas_equiv_l528_52886

-- Define weights
variable (w_orange w_apple w_banana : ℝ)

-- Conditions
def condition1 : Prop := 9 * w_orange = 6 * w_apple
def condition2 : Prop := 4 * w_banana = 3 * w_apple

-- Main problem
theorem oranges_apples_bananas_equiv :
  ∀ (w_orange w_apple w_banana : ℝ),
  (9 * w_orange = 6 * w_apple) →
  (4 * w_banana = 3 * w_apple) →
  ∃ (a b : ℕ), a = 17 ∧ b = 13 ∧ (a + 3/4 * b = (45/9) * 6) :=
by
  intros w_orange w_apple w_banana h1 h2
  -- note: actual proof would go here
  sorry

end NUMINAMATH_GPT_oranges_apples_bananas_equiv_l528_52886


namespace NUMINAMATH_GPT_jessica_deposit_fraction_l528_52807

theorem jessica_deposit_fraction (init_balance withdraw_amount final_balance : ℝ)
  (withdraw_fraction remaining_fraction deposit_fraction : ℝ) :
  remaining_fraction = withdraw_fraction - (2/5) → 
  init_balance * withdraw_fraction = init_balance - withdraw_amount →
  init_balance * remaining_fraction + deposit_fraction * (init_balance * remaining_fraction) = final_balance →
  init_balance = 500 →
  final_balance = 450 →
  withdraw_amount = 200 →
  remaining_fraction = (3/5) →
  deposit_fraction = 1/2 :=
by
  intros hr hw hrb hb hf hwamount hr_remain
  sorry

end NUMINAMATH_GPT_jessica_deposit_fraction_l528_52807


namespace NUMINAMATH_GPT_total_spent_l528_52824

theorem total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ)
  (total : ℕ) :
  bracelet_price = 4 →
  keychain_price = 5 →
  coloring_book_price = 3 →
  paula_bracelets = 2 →
  paula_keychains = 1 →
  olive_coloring_books = 1 →
  olive_bracelets = 1 →
  total = paula_bracelets * bracelet_price + paula_keychains * keychain_price +
          olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price →
  total = 20 :=
by sorry

end NUMINAMATH_GPT_total_spent_l528_52824


namespace NUMINAMATH_GPT_no_linear_term_in_product_l528_52877

theorem no_linear_term_in_product (m : ℝ) :
  (∀ (x : ℝ), (x - 3) * (3 * x + m) - (3 * x^2 - 3 * m) = 0) → m = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_no_linear_term_in_product_l528_52877


namespace NUMINAMATH_GPT_hawks_score_l528_52851

theorem hawks_score (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 18) : y = 16 := by
  sorry

end NUMINAMATH_GPT_hawks_score_l528_52851


namespace NUMINAMATH_GPT_power_function_at_point_l528_52818

theorem power_function_at_point (f : ℝ → ℝ) (h : ∃ α, ∀ x, f x = x^α) (hf : f 2 = 4) : f 3 = 9 :=
sorry

end NUMINAMATH_GPT_power_function_at_point_l528_52818


namespace NUMINAMATH_GPT_smallest_multiple_of_3_l528_52881

theorem smallest_multiple_of_3 (a : ℕ) (h : ∀ i j : ℕ, i < 6 → j < 6 → 3 * (a + i) = 3 * (a + 10 + j) → a = 50) : 3 * a = 150 :=
by
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_3_l528_52881


namespace NUMINAMATH_GPT_baby_guppies_l528_52866

theorem baby_guppies (x : ℕ) (h1 : 7 + x + 9 = 52) : x = 36 :=
by
  sorry

end NUMINAMATH_GPT_baby_guppies_l528_52866


namespace NUMINAMATH_GPT_quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l528_52844

-- Proof Problem (1)
theorem quadratic_inequality_roots_a_eq_neg1
  (a : ℝ)
  (h : ∀ x, (-1 < x ∧ x < 3) → ax^2 - 2 * a * x + 3 > 0) :
  a = -1 :=
sorry

-- Proof Problem (2)
theorem quadratic_inequality_for_all_real_a_range
  (a : ℝ)
  (h : ∀ x, ax^2 - 2 * a * x + 3 > 0) :
  0 ≤ a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_roots_a_eq_neg1_quadratic_inequality_for_all_real_a_range_l528_52844


namespace NUMINAMATH_GPT_find_f2_plus_g2_l528_52892

-- Functions f and g are defined
variable (f g : ℝ → ℝ)

-- Conditions based on the problem
def even_function : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function : Prop := ∀ x : ℝ, g (-x) = g x
def function_equation : Prop := ∀ x : ℝ, f x - g x = x^3 + 2^(-x)

-- Lean Theorem Statement
theorem find_f2_plus_g2 (h1 : even_function f) (h2 : odd_function g) (h3 : function_equation f g) :
  f 2 + g 2 = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_plus_g2_l528_52892


namespace NUMINAMATH_GPT_solve_system_of_equations_l528_52847

-- Define the given system of equations and conditions
theorem solve_system_of_equations (a b c x y z : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : yz / (y + z) = a) 
  (h2 : xz / (x + z) = b) 
  (h3 : xy / (x + y) = c) :
  x = 2 * a * b * c / (a * c + a * b - b * c) ∧ 
  y = 2 * a * b * c / (a * b + b * c - a * c) ∧ 
  z = 2 * a * b * c / (a * c + b * c - a * b) := sorry

end NUMINAMATH_GPT_solve_system_of_equations_l528_52847


namespace NUMINAMATH_GPT_arithmetic_contains_geometric_l528_52863

theorem arithmetic_contains_geometric (a d : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) : 
  ∃ b q : ℕ, (b = a) ∧ (q = 1 + d) ∧ (∀ n : ℕ, ∃ k : ℕ, a * (1 + d)^n = a + k * d) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_contains_geometric_l528_52863


namespace NUMINAMATH_GPT_yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l528_52813

-- Defining "Yazhong point"
def yazhong (A B M : ℝ) : Prop := abs (M - A) = abs (M - B)

-- Problem 1
theorem yazhong_point_1 {A B M : ℝ} (hA : A = -5) (hB : B = 1) (hM : yazhong A B M) : M = -2 :=
sorry

-- Problem 2
theorem yazhong_point_2 {A B M : ℝ} (hM : M = 2) (hAB : B - A = 9) (h_order : A < B) (hY : yazhong A B M) :
  (A = -5/2) ∧ (B = 13/2) :=
sorry

-- Problem 3 Part ①
theorem yazhong_point_3_part1 (A : ℝ) (B : ℝ) (m : ℤ) 
  (hA : A = -6) (hB_range : -4 ≤ B ∧ B ≤ -2) (hM : yazhong A B m) : 
  m = -5 ∨ m = -4 :=
sorry

-- Problem 3 Part ②
theorem yazhong_point_3_part2 (C D : ℝ) (n : ℤ)
  (hC : C = -4) (hD : D = -2) (hM : yazhong (-6) (C + D + 2 * n) 0) : 
  8 ≤ n ∧ n ≤ 10 :=
sorry

end NUMINAMATH_GPT_yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l528_52813


namespace NUMINAMATH_GPT_at_least_two_equal_l528_52880

theorem at_least_two_equal (x y z : ℝ) (h1 : x * y + z = y * z + x) (h2 : y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end NUMINAMATH_GPT_at_least_two_equal_l528_52880


namespace NUMINAMATH_GPT_gcd_204_85_l528_52861

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_204_85_l528_52861


namespace NUMINAMATH_GPT_smallest_z_value_l528_52802

-- Definitions: w, x, y, and z as consecutive even positive integers
def consecutive_even_cubes (w x y z : ℤ) : Prop :=
  w % 2 = 0 ∧ x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  w < x ∧ x < y ∧ y < z ∧
  x = w + 2 ∧ y = x + 2 ∧ z = y + 2

-- Problem statement: Smallest possible value of z
theorem smallest_z_value :
  ∃ w x y z : ℤ, consecutive_even_cubes w x y z ∧ w^3 + x^3 + y^3 = z^3 ∧ z = 12 :=
by
  sorry

end NUMINAMATH_GPT_smallest_z_value_l528_52802


namespace NUMINAMATH_GPT_interest_rate_per_annum_l528_52853
noncomputable def interest_rate_is_10 : ℝ := 10
theorem interest_rate_per_annum (P R : ℝ) : 
  (1200 * ((1 + R / 100)^2 - 1) - 1200 * R * 2 / 100 = 12) → P = 1200 → R = 10 := 
by sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l528_52853


namespace NUMINAMATH_GPT_calc_sum_of_digits_l528_52869

theorem calc_sum_of_digits (x y : ℕ) (hx : x < 10) (hy : y < 10) 
(hm : 10 * 3 + x = 34) (hmy : 34 * (10 * y + 4) = 136) : x + y = 7 :=
sorry

end NUMINAMATH_GPT_calc_sum_of_digits_l528_52869


namespace NUMINAMATH_GPT_symmetric_points_y_axis_l528_52838

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : (a, 3) = (-2, 3)) (h₂ : (2, b) = (2, 3)) : (a + b) ^ 2015 = 1 := by
  sorry

end NUMINAMATH_GPT_symmetric_points_y_axis_l528_52838


namespace NUMINAMATH_GPT_ben_final_salary_is_2705_l528_52849

def initial_salary : ℕ := 3000

def salary_after_raise (salary : ℕ) : ℕ :=
  salary * 110 / 100

def salary_after_pay_cut (salary : ℕ) : ℕ :=
  salary * 85 / 100

def final_salary (initial : ℕ) : ℕ :=
  (salary_after_pay_cut (salary_after_raise initial)) - 100

theorem ben_final_salary_is_2705 : final_salary initial_salary = 2705 := 
by 
  sorry

end NUMINAMATH_GPT_ben_final_salary_is_2705_l528_52849


namespace NUMINAMATH_GPT_smallest_num_conditions_l528_52817

theorem smallest_num_conditions :
  ∃ n : ℕ, (n % 2 = 1) ∧ (n % 3 = 2) ∧ (n % 4 = 3) ∧ n = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_num_conditions_l528_52817


namespace NUMINAMATH_GPT_find_ratio_l528_52841

-- Definition of the function
def f (x : ℝ) (a b: ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

-- Statement to be proved
theorem find_ratio (a b : ℝ) (h1: f 1 a b = 10) (h2 : (3 * 1^2 + 2 * a * 1 + b = 0)) : b = -a / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_ratio_l528_52841


namespace NUMINAMATH_GPT_cubic_difference_pos_l528_52828

theorem cubic_difference_pos {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 :=
sorry

end NUMINAMATH_GPT_cubic_difference_pos_l528_52828


namespace NUMINAMATH_GPT_bernardo_wins_at_5_l528_52832

theorem bernardo_wins_at_5 :
  (∀ N : ℕ, (16 * N + 900 < 1000) → (920 ≤ 16 * N + 840) → N ≥ 5)
    ∧ (5 < 10 ∧ 16 * 5 + 900 < 1000 ∧ 920 ≤ 16 * 5 + 840) := by
{
  sorry
}

end NUMINAMATH_GPT_bernardo_wins_at_5_l528_52832


namespace NUMINAMATH_GPT_percentage_deposited_to_wife_is_33_l528_52834

-- Definitions based on the conditions
def total_income : ℝ := 800000
def children_distribution_rate : ℝ := 0.20
def number_of_children : ℕ := 3
def donation_rate : ℝ := 0.05
def final_amount : ℝ := 40000

-- We can compute the intermediate values to use them in the final proof
def amount_distributed_to_children : ℝ := total_income * children_distribution_rate * number_of_children
def remaining_after_distribution : ℝ := total_income - amount_distributed_to_children
def donation_amount : ℝ := remaining_after_distribution * donation_rate
def remaining_after_donation : ℝ := remaining_after_distribution - donation_amount
def deposited_to_wife : ℝ := remaining_after_donation - final_amount

-- The statement to prove
theorem percentage_deposited_to_wife_is_33 :
  (deposited_to_wife / total_income) * 100 = 33 := by
  sorry

end NUMINAMATH_GPT_percentage_deposited_to_wife_is_33_l528_52834


namespace NUMINAMATH_GPT_isosceles_right_triangle_third_angle_l528_52884

/-- In an isosceles right triangle where one of the angles opposite the equal sides measures 45 degrees, 
    the measure of the third angle is 90 degrees. -/
theorem isosceles_right_triangle_third_angle (θ : ℝ) 
  (h1 : θ = 45)
  (h2 : ∀ (a b c : ℝ), a + b + c = 180) : θ + θ + 90 = 180 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_third_angle_l528_52884


namespace NUMINAMATH_GPT_consecutive_sum_to_20_has_one_set_l528_52846

theorem consecutive_sum_to_20_has_one_set :
  ∃ n a : ℕ, (n ≥ 2) ∧ (a ≥ 1) ∧ (n * (2 * a + n - 1) = 40) ∧
  (n = 5 ∧ a = 2) ∧ 
  (∀ n' a', (n' ≥ 2) → (a' ≥ 1) → (n' * (2 * a' + n' - 1) = 40) → (n' = 5 ∧ a' = 2)) := sorry

end NUMINAMATH_GPT_consecutive_sum_to_20_has_one_set_l528_52846


namespace NUMINAMATH_GPT_exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l528_52848

theorem exists_n_such_that_an_is_cube_and_bn_is_fifth_power
  (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (n : ℕ), n ≥ 1 ∧ (∃ k : ℤ, a * n = k^3) ∧ (∃ l : ℤ, b * n = l^5) := 
by
  sorry

end NUMINAMATH_GPT_exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l528_52848


namespace NUMINAMATH_GPT_infinite_chain_resistance_l528_52819

noncomputable def resistance_of_infinite_chain (R₀ : ℝ) : ℝ :=
  (R₀ * (1 + Real.sqrt 5)) / 2

theorem infinite_chain_resistance : resistance_of_infinite_chain 10 = 5 + 5 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_infinite_chain_resistance_l528_52819


namespace NUMINAMATH_GPT_interest_rate_and_years_l528_52815

theorem interest_rate_and_years
    (P : ℝ)
    (n : ℕ)
    (e : ℝ)
    (h1 : P * (e ^ n) * e = P * (e ^ (n + 1)) + 4156.02)
    (h2 : P * (e ^ (n - 1)) = P * (e ^ n) - 3996.12) :
    (e = 1.04) ∧ (P = 60000) ∧ (E = 4/100) ∧ (n = 14) := by
  sorry

end NUMINAMATH_GPT_interest_rate_and_years_l528_52815


namespace NUMINAMATH_GPT_jason_gave_seashells_to_tim_l528_52888

-- Defining the conditions
def original_seashells : ℕ := 49
def current_seashells : ℕ := 36

-- The proof statement
theorem jason_gave_seashells_to_tim :
  original_seashells - current_seashells = 13 :=
by
  sorry

end NUMINAMATH_GPT_jason_gave_seashells_to_tim_l528_52888


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l528_52898

-- Lean 4 statement for the given problem.
theorem hyperbola_eccentricity_range {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h : ∀ (x y : ℝ), y = x * Real.sqrt 3 → y^2 / b^2 - x^2 / a^2 = 1 ∨ ∃ (z : ℝ), y = x * Real.sqrt 3 ∧ z^2 / b^2 - x^2 / a^2 = 1) :
  1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a < 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l528_52898


namespace NUMINAMATH_GPT_find_sum_squares_l528_52852

variables (x y : ℝ)

theorem find_sum_squares (h1 : y + 4 = (x - 2)^2) (h2 : x + 4 = (y - 2)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 15 :=
sorry

end NUMINAMATH_GPT_find_sum_squares_l528_52852


namespace NUMINAMATH_GPT_total_cans_given_away_l528_52829

-- Define constants
def initial_stock : ℕ := 2000

-- Define conditions day 1
def people_day1 : ℕ := 500
def cans_per_person_day1 : ℕ := 1
def restock_day1 : ℕ := 1500

-- Define conditions day 2
def people_day2 : ℕ := 1000
def cans_per_person_day2 : ℕ := 2
def restock_day2 : ℕ := 3000

-- Define the question as a theorem
theorem total_cans_given_away : (people_day1 * cans_per_person_day1 + people_day2 * cans_per_person_day2) = 2500 := by
  sorry

end NUMINAMATH_GPT_total_cans_given_away_l528_52829


namespace NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l528_52894

-- Proof Problem 1
theorem factorize_expr1 (a : ℝ) : 
  (a^2 - 4 * a + 4 - 4 * (a - 2) + 4) = (a - 4)^2 :=
sorry

-- Proof Problem 2
theorem factorize_expr2 (x y : ℝ) : 
  16 * x^4 - 81 * y^4 = (4 * x^2 + 9 * y^2) * (2 * x + 3 * y) * (2 * x - 3 * y) :=
sorry

end NUMINAMATH_GPT_factorize_expr1_factorize_expr2_l528_52894


namespace NUMINAMATH_GPT_complementary_angles_decrease_percent_l528_52862

theorem complementary_angles_decrease_percent
    (a b : ℝ) 
    (h1 : a + b = 90) 
    (h2 : a / b = 3 / 7) 
    (h3 : new_a = a * 1.15) 
    (h4 : new_a + new_b = 90) : 
    (new_b / b * 100) = 93.57 := 
sorry

end NUMINAMATH_GPT_complementary_angles_decrease_percent_l528_52862


namespace NUMINAMATH_GPT_find_a2023_l528_52895

theorem find_a2023 (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n > 0 → a n + a (n + 1) = n) : a 2023 = 1012 :=
sorry

end NUMINAMATH_GPT_find_a2023_l528_52895


namespace NUMINAMATH_GPT_sufficient_condition_l528_52870

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem sufficient_condition (a : ℝ) : (∀ x y : ℝ, N x y a → M x y) ↔ (a ≥ 5 / 4) := 
sorry

end NUMINAMATH_GPT_sufficient_condition_l528_52870


namespace NUMINAMATH_GPT_ratio_of_kids_l528_52810

theorem ratio_of_kids (k2004 k2005 k2006 : ℕ) 
  (h2004: k2004 = 60) 
  (h2005: k2005 = k2004 / 2)
  (h2006: k2006 = 20) :
  (k2006 : ℚ) / k2005 = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_kids_l528_52810


namespace NUMINAMATH_GPT_solve_inequality_l528_52896

theorem solve_inequality (x : ℝ) : (x - 3) * (x + 2) < 0 ↔ x ∈ Set.Ioo (-2) (3) :=
sorry

end NUMINAMATH_GPT_solve_inequality_l528_52896


namespace NUMINAMATH_GPT_degree_of_p_is_unbounded_l528_52842

theorem degree_of_p_is_unbounded (p : Polynomial ℝ) (h : ∀ x : ℝ, p.eval (x^2 - 1) = (p.eval x) * (p.eval (-x))) : False :=
sorry

end NUMINAMATH_GPT_degree_of_p_is_unbounded_l528_52842


namespace NUMINAMATH_GPT_Anna_s_wear_size_l528_52891

theorem Anna_s_wear_size
  (A : ℕ)
  (Becky_size : ℕ)
  (Ginger_size : ℕ)
  (h1 : Becky_size = 3 * A)
  (h2 : Ginger_size = 2 * Becky_size - 4)
  (h3 : Ginger_size = 8) :
  A = 2 :=
by
  sorry

end NUMINAMATH_GPT_Anna_s_wear_size_l528_52891


namespace NUMINAMATH_GPT_right_handed_players_count_l528_52887

theorem right_handed_players_count (total_players throwers left_handed_non_throwers right_handed_non_throwers : ℕ) 
  (h1 : total_players = 70)
  (h2 : throwers = 46)
  (h3 : left_handed_non_throwers = (total_players - throwers) / 3)
  (h4 : right_handed_non_throwers = total_players - throwers - left_handed_non_throwers)
  (h5 : ∀ n, n = throwers + right_handed_non_throwers) : 
  (throwers + right_handed_non_throwers) = 62 := 
by 
  sorry

end NUMINAMATH_GPT_right_handed_players_count_l528_52887


namespace NUMINAMATH_GPT_count_two_digit_decimals_between_0_40_and_0_50_l528_52868

theorem count_two_digit_decimals_between_0_40_and_0_50 : 
  ∃ (n : ℕ), n = 9 ∧ ∀ x : ℝ, 0.40 < x ∧ x < 0.50 → (exists d : ℕ, (1 ≤ d ∧ d ≤ 9 ∧ x = 0.4 + d * 0.01)) :=
by
  sorry

end NUMINAMATH_GPT_count_two_digit_decimals_between_0_40_and_0_50_l528_52868


namespace NUMINAMATH_GPT_min_height_of_cuboid_l528_52801

theorem min_height_of_cuboid (h : ℝ) (side_len : ℝ) (small_spheres_r : ℝ) (large_sphere_r : ℝ) :
  side_len = 4 → 
  small_spheres_r = 1 → 
  large_sphere_r = 2 → 
  ∃ h_min : ℝ, h_min = 2 + 2 * Real.sqrt 7 ∧ h ≥ h_min := 
by
  sorry

end NUMINAMATH_GPT_min_height_of_cuboid_l528_52801


namespace NUMINAMATH_GPT_animal_eyes_count_l528_52856

noncomputable def total_animal_eyes (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ) : ℕ :=
frogs * eyes_per_frog + crocodiles * eyes_per_crocodile

theorem animal_eyes_count (frogs : ℕ) (crocodiles : ℕ) (eyes_per_frog : ℕ) (eyes_per_crocodile : ℕ):
  frogs = 20 → crocodiles = 10 → eyes_per_frog = 2 → eyes_per_crocodile = 2 → total_animal_eyes frogs crocodiles eyes_per_frog eyes_per_crocodile = 60 :=
by
  sorry

end NUMINAMATH_GPT_animal_eyes_count_l528_52856


namespace NUMINAMATH_GPT_time_for_Q_to_finish_job_alone_l528_52839

theorem time_for_Q_to_finish_job_alone (T_Q : ℝ) 
  (h1 : 0 < T_Q)
  (rate_P : ℝ := 1 / 4) 
  (rate_Q : ℝ := 1 / T_Q)
  (combined_work_rate : ℝ := 3 * (rate_P + rate_Q))
  (remaining_work : ℝ := 0.1) -- 0.4 * rate_P
  (total_work_done : ℝ := 0.9) -- 1 - remaining_work
  (h2 : combined_work_rate = total_work_done) : T_Q = 20 :=
by sorry

end NUMINAMATH_GPT_time_for_Q_to_finish_job_alone_l528_52839


namespace NUMINAMATH_GPT_kolya_start_time_l528_52822

-- Definitions of conditions as per the initial problem statement
def angle_moved_by_minute_hand (x : ℝ) : ℝ := 6 * x
def angle_moved_by_hour_hand (x : ℝ) : ℝ := 30 + 0.5 * x

theorem kolya_start_time (x : ℝ) :
  (angle_moved_by_minute_hand x = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) ∨
  (angle_moved_by_minute_hand x - 180 = (angle_moved_by_hour_hand x + angle_moved_by_hour_hand x + 60) / 2) :=
sorry

end NUMINAMATH_GPT_kolya_start_time_l528_52822


namespace NUMINAMATH_GPT_min_value_expr_l528_52806

theorem min_value_expr : ∃ (x : ℝ), (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 ∧ 
  ∀ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l528_52806


namespace NUMINAMATH_GPT_max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l528_52875

open Real

theorem max_min_2sinx_minus_3 : 
  ∀ x : ℝ, 
    -5 ≤ 2 * sin x - 3 ∧ 
    2 * sin x - 3 ≤ -1 :=
by sorry

theorem max_min_7_fourth_sinx_minus_sinx_squared : 
  ∀ x : ℝ, 
    -1/4 ≤ (7/4 + sin x - sin x ^ 2) ∧ 
    (7/4 + sin x - sin x ^ 2) ≤ 2 :=
by sorry

end NUMINAMATH_GPT_max_min_2sinx_minus_3_max_min_7_fourth_sinx_minus_sinx_squared_l528_52875


namespace NUMINAMATH_GPT_remainder_7_pow_63_mod_8_l528_52864

theorem remainder_7_pow_63_mod_8 : 7^63 % 8 = 7 :=
by sorry

end NUMINAMATH_GPT_remainder_7_pow_63_mod_8_l528_52864


namespace NUMINAMATH_GPT_geometric_sequence_a5_l528_52827

theorem geometric_sequence_a5
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_ratio : ∀ n, a (n + 1) = 2 * a n)
  (h_product : a 3 * a 11 = 16) :
  a 5 = 1 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l528_52827


namespace NUMINAMATH_GPT_greatest_value_2q_sub_r_l528_52821

theorem greatest_value_2q_sub_r : 
  ∃ (q r : ℕ), 965 = 22 * q + r ∧ 2 * q - r = 67 := 
by 
  sorry

end NUMINAMATH_GPT_greatest_value_2q_sub_r_l528_52821


namespace NUMINAMATH_GPT_problem_solved_by_half_participants_l528_52882

variables (n m : ℕ)
variable (solve : ℕ → ℕ → Prop)  -- solve i j means participant i solved problem j

axiom half_n_problems_solved : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)

theorem problem_solved_by_half_participants (h : ∀ i, i < m → (∃ s, s ≥ n / 2 ∧ ∀ j, j < n → solve i j → j < s)) : 
  ∃ j, j < n ∧ (∃ count, count ≥ m / 2 ∧ (∃ i, i < m → solve i j)) :=
  sorry

end NUMINAMATH_GPT_problem_solved_by_half_participants_l528_52882


namespace NUMINAMATH_GPT_square_area_l528_52836

theorem square_area (x y : ℝ) 
  (h1 : x = 20 ∧ y = 20)
  (h2 : x = 20 ∧ y = 5)
  (h3 : x = x ∧ y = 5)
  (h4 : x = x ∧ y = 20)
  : (∃ a : ℝ, a = 225) :=
sorry

end NUMINAMATH_GPT_square_area_l528_52836


namespace NUMINAMATH_GPT_find_f_neg_19_div_3_l528_52871

noncomputable def f (x : ℝ) : ℝ := 
  if 0 < x ∧ x < 1 then 
    8^x 
  else 
    sorry -- The full definition is complex and not needed for the statement

-- Define the properties of f
lemma f_periodic (x : ℝ) : f (x + 2) = f x := 
  sorry

lemma f_odd (x : ℝ) : f (-x) = -f x := 
  sorry

theorem find_f_neg_19_div_3 : f (-19/3) = -2 :=
  sorry

end NUMINAMATH_GPT_find_f_neg_19_div_3_l528_52871


namespace NUMINAMATH_GPT_transformed_center_coordinates_l528_52857

theorem transformed_center_coordinates (S : (ℝ × ℝ)) (hS : S = (3, -4)) : 
  let reflected_S := (S.1, -S.2)
  let translated_S := (reflected_S.1, reflected_S.2 + 5)
  translated_S = (3, 9) :=
by
  sorry

end NUMINAMATH_GPT_transformed_center_coordinates_l528_52857


namespace NUMINAMATH_GPT_area_of_rectangle_at_stage_4_l528_52812

def area_at_stage (n : ℕ) : ℕ :=
  let square_area := 16
  let initial_squares := 2
  let common_difference := 2
  let total_squares := initial_squares + common_difference * (n - 1)
  total_squares * square_area

theorem area_of_rectangle_at_stage_4 :
  area_at_stage 4 = 128 :=
by
  -- computation and transformations are omitted
  sorry

end NUMINAMATH_GPT_area_of_rectangle_at_stage_4_l528_52812


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l528_52835

theorem necessary_and_sufficient_condition {a : ℝ} :
    (∃ x : ℝ, a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l528_52835


namespace NUMINAMATH_GPT_regular_polygon_sides_l528_52826

theorem regular_polygon_sides (h : ∀ n, (180 * (n - 2) = 150 * n)) : n = 12 := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l528_52826


namespace NUMINAMATH_GPT_find_third_number_l528_52823

theorem find_third_number (x y z : ℝ) 
  (h1 : y = 3 * x - 7)
  (h2 : z = 2 * x + 2)
  (h3 : x + y + z = 168) : z = 60 :=
sorry

end NUMINAMATH_GPT_find_third_number_l528_52823


namespace NUMINAMATH_GPT_unique_pair_a_b_l528_52825

open Complex

theorem unique_pair_a_b :
  ∃! (a b : ℂ), a^4 * b^3 = 1 ∧ a^6 * b^7 = 1 := by
  sorry

end NUMINAMATH_GPT_unique_pair_a_b_l528_52825


namespace NUMINAMATH_GPT_impossible_list_10_numbers_with_given_conditions_l528_52876

theorem impossible_list_10_numbers_with_given_conditions :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 0 ≤ i ∧ i ≤ 7 → (a i * a (i + 1) * a (i + 2)) % 6 = 0) ∧
    (∀ i, 0 ≤ i ∧ i ≤ 8 → (a i * a (i + 1)) % 6 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_impossible_list_10_numbers_with_given_conditions_l528_52876


namespace NUMINAMATH_GPT_find_k_l528_52859

theorem find_k (x y k : ℝ) (hx1 : x - 4 * y + 3 ≤ 0) (hx2 : 3 * x + 5 * y - 25 ≤ 0) (hx3 : x ≥ 1)
  (hmax : ∃ (z : ℝ), z = 12 ∧ z = k * x + y) (hmin : ∃ (z : ℝ), z = 3 ∧ z = k * x + y) :
  k = 2 :=
sorry

end NUMINAMATH_GPT_find_k_l528_52859


namespace NUMINAMATH_GPT_negation_of_proposition_l528_52889

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 0 ≤ x → x^3 + x ≥ 0)) ↔ (∃ x : ℝ, 0 ≤ x ∧ x^3 + x < 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l528_52889


namespace NUMINAMATH_GPT_product_of_N_l528_52814

theorem product_of_N (M L : ℝ) (N : ℝ) 
  (h1 : M = L + N) 
  (h2 : ∀ M4 L4 : ℝ, M4 = M - 7 → L4 = L + 5 → |M4 - L4| = 4) :
  N = 16 ∨ N = 8 ∧ (16 * 8 = 128) := 
by 
  sorry

end NUMINAMATH_GPT_product_of_N_l528_52814


namespace NUMINAMATH_GPT_complex_quadrant_l528_52879

open Complex

theorem complex_quadrant (z : ℂ) (h : (1 + I) * z = 2 * I) : 
  z.re > 0 ∧ z.im < 0 :=
  sorry

end NUMINAMATH_GPT_complex_quadrant_l528_52879


namespace NUMINAMATH_GPT_complex_multiplication_l528_52803

variable (i : ℂ)
axiom imaginary_unit : i^2 = -1

theorem complex_multiplication :
  i * (2 * i - 1) = -2 - i :=
  sorry

end NUMINAMATH_GPT_complex_multiplication_l528_52803


namespace NUMINAMATH_GPT_chris_initial_donuts_l528_52820

theorem chris_initial_donuts (D : ℝ) (H1 : D * 0.90 - 4 = 23) : D = 30 := 
by
sorry

end NUMINAMATH_GPT_chris_initial_donuts_l528_52820


namespace NUMINAMATH_GPT_range_of_m_l528_52899

noncomputable def proof_problem (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) : Prop :=
  ∃ x y : ℝ, (0 < x) ∧ (0 < y) ∧ (1/x + 2/y = 1) ∧ (x + y / 2 < m^2 + 3 * m) ↔ (m < -4 ∨ m > 1)

theorem range_of_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 1/x + 2/y = 1) :
  proof_problem x y m hx hy hxy :=
sorry

end NUMINAMATH_GPT_range_of_m_l528_52899


namespace NUMINAMATH_GPT_proof_l528_52809

noncomputable def proof_problem (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem proof (
  a b c : ℝ
) (h1 : (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3 = 3 * (a^3 - b^3) * (b^3 - c^3) * (c^3 - a^3))
  (h2 : (a - b)^3 + (b - c)^3 + (c - a)^3 = 3 * (a - b) * (b - c) * (c - a)) :
  proof_problem a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end NUMINAMATH_GPT_proof_l528_52809


namespace NUMINAMATH_GPT_imaginary_part_of_z_l528_52833

open Complex

-- Define the context
variables (z : ℂ) (a b : ℂ)

-- Define the condition
def condition := (1 - 2*I) * z = 5 * I

-- Lean 4 statement to prove the imaginary part of z 
theorem imaginary_part_of_z (h : condition z) : z.im = 1 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l528_52833


namespace NUMINAMATH_GPT_base_three_to_decimal_l528_52845

theorem base_three_to_decimal :
  let n := 20121 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0) = 178 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_three_to_decimal_l528_52845


namespace NUMINAMATH_GPT_range_of_m_l528_52893

theorem range_of_m (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → (f x₁ - f x₂) / (x₁ - x₂) < 0)
  (h_inequality : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f (2 * m * x - Real.log x - 3) ≥ 2 * f 3 - f (-2 * m * x + Real.log x + 3)) :
  ∃ m, m ∈ Set.Icc (1 / (2 * Real.exp 1)) (1 + Real.log 3 / 6) :=
sorry

end NUMINAMATH_GPT_range_of_m_l528_52893


namespace NUMINAMATH_GPT_teamX_total_games_l528_52897

variables (x : ℕ)

-- Conditions
def teamX_wins := (3/4) * x
def teamX_loses := (1/4) * x

def teamY_wins := (2/3) * (x + 10)
def teamY_loses := (1/3) * (x + 10)

-- Question: Prove team X played 20 games
theorem teamX_total_games :
  teamY_wins - teamX_wins = 5 ∧ teamY_loses - teamX_loses = 5 → x = 20 := by
sorry

end NUMINAMATH_GPT_teamX_total_games_l528_52897


namespace NUMINAMATH_GPT_train_length_proof_l528_52811

noncomputable def speed_km_per_hr : ℝ := 108
noncomputable def time_seconds : ℝ := 9
noncomputable def length_of_train : ℝ := 270
noncomputable def km_to_m : ℝ := 1000
noncomputable def hr_to_s : ℝ := 3600

theorem train_length_proof : 
  (speed_km_per_hr * (km_to_m / hr_to_s) * time_seconds) = length_of_train :=
  by
  sorry

end NUMINAMATH_GPT_train_length_proof_l528_52811


namespace NUMINAMATH_GPT_ratio_of_administrators_to_teachers_l528_52800

-- Define the conditions
def graduates : ℕ := 50
def parents_per_graduate : ℕ := 2
def teachers : ℕ := 20
def total_chairs : ℕ := 180

-- Calculate intermediate values
def parents : ℕ := graduates * parents_per_graduate
def graduates_and_parents_chairs : ℕ := graduates + parents
def total_graduates_parents_teachers_chairs : ℕ := graduates_and_parents_chairs + teachers
def administrators : ℕ := total_chairs - total_graduates_parents_teachers_chairs

-- Specify the theorem to prove the ratio of administrators to teachers
theorem ratio_of_administrators_to_teachers : administrators / teachers = 1 / 2 :=
by
  -- Proof is omitted; placeholder 'sorry'
  sorry

end NUMINAMATH_GPT_ratio_of_administrators_to_teachers_l528_52800


namespace NUMINAMATH_GPT_total_shaded_cubes_l528_52873

/-
The large cube consists of 27 smaller cubes, each face is a 3x3 grid.
Opposite faces are shaded in an identical manner, with each face having 5 shaded smaller cubes.
-/

theorem total_shaded_cubes (number_of_smaller_cubes : ℕ)
  (face_shade_pattern : ∀ (face : ℕ), ℕ)
  (opposite_face_same_shade : ∀ (face1 face2 : ℕ), face1 = face2 → face_shade_pattern face1 = face_shade_pattern face2)
  (faces_possible : ∀ (face : ℕ), face < 6)
  (each_face_shaded_squares : ∀ (face : ℕ), face_shade_pattern face = 5)
  : ∃ (n : ℕ), n = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_shaded_cubes_l528_52873


namespace NUMINAMATH_GPT_approx_ineq_l528_52883

noncomputable def approx (x : ℝ) : ℝ := 1 + 6 * (-0.002 : ℝ)

theorem approx_ineq (x : ℝ) (h : x = 0.998) : 
  abs ((x^6) - approx x) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_approx_ineq_l528_52883


namespace NUMINAMATH_GPT_find_a_l528_52840

open Real

theorem find_a :
  ∃ a : ℝ, (1/5) * (0.5 + a + 1 + 1.4 + 1.5) = 0.28 * 3 + 0.16 := by
  use 0.6
  sorry

end NUMINAMATH_GPT_find_a_l528_52840


namespace NUMINAMATH_GPT_cos_alpha_third_quadrant_l528_52804

theorem cos_alpha_third_quadrant (α : ℝ) (hα1 : π < α ∧ α < 3 * π / 2) (hα2 : Real.tan α = 4 / 3) :
  Real.cos α = -3 / 5 :=
sorry

end NUMINAMATH_GPT_cos_alpha_third_quadrant_l528_52804


namespace NUMINAMATH_GPT_rightmost_four_digits_of_5_pow_2023_l528_52872

theorem rightmost_four_digits_of_5_pow_2023 :
  5 ^ 2023 % 5000 = 3125 :=
  sorry

end NUMINAMATH_GPT_rightmost_four_digits_of_5_pow_2023_l528_52872


namespace NUMINAMATH_GPT_reciprocal_of_2022_l528_52843

noncomputable def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_of_2022 : reciprocal 2022 = 1 / 2022 :=
by
  -- Define reciprocal
  sorry

end NUMINAMATH_GPT_reciprocal_of_2022_l528_52843


namespace NUMINAMATH_GPT_factor_poly_eq_factored_form_l528_52855

-- Defining the polynomial expressions
def poly1 (x : ℝ) := x^2 + 4 * x + 3
def poly2 (x : ℝ) := x^2 + 8 * x + 15
def poly3 (x : ℝ) := x^2 + 6 * x - 8

-- The main expression which needs to be factored
def main_expr (x : ℝ) := (poly1 x) * (poly2 x) + (poly3 x)

-- Stating the goal factored form
def factored_form (x : ℝ) := (x^2 + 6 * x + 19) * (x^2 + 6 * x - 2)

-- The theorem statement
theorem factor_poly_eq_factored_form (x : ℝ) : 
  main_expr x = factored_form x := 
by
  sorry

end NUMINAMATH_GPT_factor_poly_eq_factored_form_l528_52855


namespace NUMINAMATH_GPT_min_value_of_expression_l528_52805

noncomputable def target_expression (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x)

theorem min_value_of_expression : (∀ x : ℝ, target_expression x ≥ -784) ∧ (∃ x : ℝ, target_expression x = -784) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l528_52805


namespace NUMINAMATH_GPT_ratio_of_areas_l528_52860

noncomputable def area_ratio (a : ℝ) : ℝ :=
  let side_triangle : ℝ := a
  let area_triangle : ℝ := (1 / 2) * side_triangle * side_triangle
  let height_rhombus : ℝ := side_triangle * Real.sin (Real.pi / 3)
  let area_rhombus : ℝ := height_rhombus * side_triangle
  area_rhombus / area_triangle

theorem ratio_of_areas (a : ℝ) (h : a > 0) : area_ratio a = 3 := by
  -- The proof would be here
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l528_52860


namespace NUMINAMATH_GPT_stripe_width_l528_52850

theorem stripe_width (x : ℝ) (h : 60 * x - x^2 = 400) : x = 30 - 5 * Real.sqrt 5 := 
  sorry

end NUMINAMATH_GPT_stripe_width_l528_52850


namespace NUMINAMATH_GPT_find_triangle_value_l528_52816

variables (triangle q r : ℝ)
variables (h1 : triangle + q = 75) (h2 : triangle + q + r = 138) (h3 : r = q / 3)

theorem find_triangle_value : triangle = -114 :=
by
  sorry

end NUMINAMATH_GPT_find_triangle_value_l528_52816


namespace NUMINAMATH_GPT_price_of_brand_y_pen_l528_52858

-- Definitions based on the conditions
def num_brand_x_pens : ℕ := 8
def price_per_brand_x_pen : ℝ := 4.0
def total_spent : ℝ := 40.0
def total_pens : ℕ := 12

-- price of brand Y that needs to be proven
def price_per_brand_y_pen : ℝ := 2.0

-- Proof statement
theorem price_of_brand_y_pen :
  let num_brand_y_pens := total_pens - num_brand_x_pens
  let spent_on_brand_x_pens := num_brand_x_pens * price_per_brand_x_pen
  let spent_on_brand_y_pens := total_spent - spent_on_brand_x_pens
  spent_on_brand_y_pens / num_brand_y_pens = price_per_brand_y_pen :=
by
  sorry

end NUMINAMATH_GPT_price_of_brand_y_pen_l528_52858
