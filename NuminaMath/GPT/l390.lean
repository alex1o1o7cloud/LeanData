import Mathlib

namespace NUMINAMATH_GPT_solve_fraction_eq_l390_39027

theorem solve_fraction_eq (x : ℝ) (h : x ≠ -2) : (x = -1) ↔ ((x^2 + 2 * x + 3) / (x + 2) = x + 3) := 
by 
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_l390_39027


namespace NUMINAMATH_GPT_value_of_x_l390_39035

theorem value_of_x (x : ℚ) (h : (x + 10 + 17 + 3 * x + 15 + 3 * x + 6) / 5 = 26) : x = 82 / 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l390_39035


namespace NUMINAMATH_GPT_bryden_receives_amount_l390_39009

variable (q : ℝ) (p : ℝ) (num_quarters : ℝ)

-- Define the conditions
def face_value_of_quarter : Prop := q = 0.25
def percentage_offer : Prop := p = 25 * q
def number_of_quarters : Prop := num_quarters = 5

-- Define the theorem to be proved
theorem bryden_receives_amount (h1 : face_value_of_quarter q) (h2 : percentage_offer q p) (h3 : number_of_quarters num_quarters) :
  (p * num_quarters * q) = 31.25 :=
by
  sorry

end NUMINAMATH_GPT_bryden_receives_amount_l390_39009


namespace NUMINAMATH_GPT_selection_probabilities_l390_39070

-- Define the probabilities of selection for Ram, Ravi, and Rani
def prob_ram : ℚ := 5 / 7
def prob_ravi : ℚ := 1 / 5
def prob_rani : ℚ := 3 / 4

-- State the theorem that combines these probabilities
theorem selection_probabilities : prob_ram * prob_ravi * prob_rani = 3 / 28 :=
by
  sorry


end NUMINAMATH_GPT_selection_probabilities_l390_39070


namespace NUMINAMATH_GPT_wilfred_carrots_on_tuesday_l390_39062

theorem wilfred_carrots_on_tuesday :
  ∀ (carrots_eaten_Wednesday carrots_eaten_Thursday total_carrots desired_total: ℕ),
    carrots_eaten_Wednesday = 6 →
    carrots_eaten_Thursday = 5 →
    desired_total = 15 →
    desired_total - (carrots_eaten_Wednesday + carrots_eaten_Thursday) = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_wilfred_carrots_on_tuesday_l390_39062


namespace NUMINAMATH_GPT_probability_point_between_C_and_E_l390_39022

noncomputable def length_between_points (total_length : ℝ) (ratio : ℝ) : ℝ :=
ratio * total_length

theorem probability_point_between_C_and_E
  (A B C D E : ℝ)
  (h1 : A < B)
  (h2 : C < E)
  (h3 : B - A = 4 * (D - A))
  (h4 : B - A = 8 * (B - C))
  (h5 : B - E = 2 * (E - C)) :
  (E - C) / (B - A) = 1 / 24 :=
by 
  sorry

end NUMINAMATH_GPT_probability_point_between_C_and_E_l390_39022


namespace NUMINAMATH_GPT_max_value_expr_l390_39000

open Real

noncomputable def expr (x : ℝ) : ℝ :=
  (x^4 + 3 * x^2 - sqrt (x^8 + 9)) / x^2

theorem max_value_expr : ∀ (x y : ℝ), (0 < x) → (y = x + 1 / x) → expr x = 15 / 7 :=
by
  intros x y hx hy
  sorry

end NUMINAMATH_GPT_max_value_expr_l390_39000


namespace NUMINAMATH_GPT_Mia_studied_fraction_l390_39071

-- Define the conditions
def total_minutes_per_day := 1440
def time_spent_watching_TV := total_minutes_per_day * 1 / 5
def time_spent_studying := 288
def remaining_time := total_minutes_per_day - time_spent_watching_TV
def fraction_studying := time_spent_studying / remaining_time

-- State the proof goal
theorem Mia_studied_fraction : fraction_studying = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_Mia_studied_fraction_l390_39071


namespace NUMINAMATH_GPT_find_g_of_3_l390_39005

theorem find_g_of_3 (f g : ℝ → ℝ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) :
  g 3 = 5 :=
sorry

end NUMINAMATH_GPT_find_g_of_3_l390_39005


namespace NUMINAMATH_GPT_prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l390_39042

-- Define events and their probabilities.
def prob_A : ℝ := 0.8
def prob_B : ℝ := 0.8

-- Given P(A and B) = P(A) * P(B)
def prob_AB : ℝ := prob_A * prob_B

-- Statements to prove
theorem prob_both_hit : prob_AB = 0.64 :=
by
  -- P(A and B) = 0.8 * 0.8 = 0.64
  exact sorry

theorem prob_exactly_one_hit : (prob_A * (1 - prob_B) + (1 - prob_A) * prob_B) = 0.32 :=
by
  -- P(A and not B) + P(not A and B) = 0.8 * 0.2 + 0.2 * 0.8 = 0.32
  exact sorry

theorem prob_at_least_one_hit : (1 - (1 - prob_A) * (1 - prob_B)) = 0.96 :=
by
  -- 1 - P(not A and not B) = 1 - 0.04 = 0.96
  exact sorry

end NUMINAMATH_GPT_prob_both_hit_prob_exactly_one_hit_prob_at_least_one_hit_l390_39042


namespace NUMINAMATH_GPT_liu_xiang_hurdles_l390_39083

theorem liu_xiang_hurdles :
  let total_distance := 110
  let first_hurdle_distance := 13.72
  let last_hurdle_distance := 14.02
  let best_time_first_segment := 2.5
  let best_time_last_segment := 1.4
  let hurdle_cycle_time := 0.96
  let num_hurdles := 10
  (total_distance - first_hurdle_distance - last_hurdle_distance) / num_hurdles = 8.28 ∧
  best_time_first_segment + num_hurdles * hurdle_cycle_time + best_time_last_segment  = 12.1 :=
by
  sorry

end NUMINAMATH_GPT_liu_xiang_hurdles_l390_39083


namespace NUMINAMATH_GPT_minimum_value_sum_l390_39047

theorem minimum_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / (3 * b) + b / (5 * c) + c / (6 * a)) >= (3 / (90^(1/3))) :=
by 
  sorry

end NUMINAMATH_GPT_minimum_value_sum_l390_39047


namespace NUMINAMATH_GPT_sum_xyz_l390_39068

variables (x y z : ℤ)

theorem sum_xyz (h1 : y = 3 * x) (h2 : z = 3 * y - x) : x + y + z = 12 * x :=
by 
  -- skip the proof
  sorry

end NUMINAMATH_GPT_sum_xyz_l390_39068


namespace NUMINAMATH_GPT_find_x_if_vectors_parallel_l390_39057

theorem find_x_if_vectors_parallel (x : ℝ)
  (a : ℝ × ℝ := (x - 1, 2))
  (b : ℝ × ℝ := (2, 1)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → x = 5 :=
by sorry

end NUMINAMATH_GPT_find_x_if_vectors_parallel_l390_39057


namespace NUMINAMATH_GPT_spilled_bag_candies_l390_39085

theorem spilled_bag_candies (c1 c2 c3 c4 c5 c6 c7 : ℕ) (avg_candies_per_bag : ℕ) (x : ℕ) 
  (h_counts : c1 = 12 ∧ c2 = 14 ∧ c3 = 18 ∧ c4 = 22 ∧ c5 = 24 ∧ c6 = 26 ∧ c7 = 29)
  (h_avg : avg_candies_per_bag = 22)
  (h_total : c1 + c2 + c3 + c4 + c5 + c6 + c7 + x = 8 * avg_candies_per_bag) : x = 31 := 
by
  sorry

end NUMINAMATH_GPT_spilled_bag_candies_l390_39085


namespace NUMINAMATH_GPT_angles_terminal_yaxis_l390_39093

theorem angles_terminal_yaxis :
  {θ : ℝ | ∃ k : ℤ, θ = 2 * k * Real.pi + Real.pi / 2 ∨ θ = 2 * k * Real.pi + 3 * Real.pi / 2} =
  {θ : ℝ | ∃ n : ℤ, θ = n * Real.pi + Real.pi / 2} :=
by sorry

end NUMINAMATH_GPT_angles_terminal_yaxis_l390_39093


namespace NUMINAMATH_GPT_max_cookies_andy_can_eat_l390_39018

theorem max_cookies_andy_can_eat (A B C : ℕ) (hB_pos : B > 0) (hC_pos : C > 0) (hB : B ∣ A) (hC : C ∣ A) (h_sum : A + B + C = 36) :
  A ≤ 30 := by
  sorry

end NUMINAMATH_GPT_max_cookies_andy_can_eat_l390_39018


namespace NUMINAMATH_GPT_difference_of_sums_1500_l390_39056

def sum_of_first_n_odd_numbers (n : ℕ) : ℕ :=
  n * n

def sum_of_first_n_even_numbers (n : ℕ) : ℕ :=
  n * (n + 1)

theorem difference_of_sums_1500 :
  sum_of_first_n_even_numbers 1500 - sum_of_first_n_odd_numbers 1500 = 1500 :=
by
  sorry

end NUMINAMATH_GPT_difference_of_sums_1500_l390_39056


namespace NUMINAMATH_GPT_translation_correct_l390_39069

def vector_a : ℝ × ℝ := (1, 1)

def translate_right (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1 + d, v.2)
def translate_down (v : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (v.1, v.2 - d)

def vector_b := translate_down (translate_right vector_a 2) 1

theorem translation_correct :
  vector_b = (3, 0) :=
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_translation_correct_l390_39069


namespace NUMINAMATH_GPT_arithmetic_sequence_number_of_terms_l390_39065

def arithmetic_sequence_terms_count (a d l : ℕ) : ℕ :=
  sorry

theorem arithmetic_sequence_number_of_terms :
  arithmetic_sequence_terms_count 13 3 73 = 21 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_number_of_terms_l390_39065


namespace NUMINAMATH_GPT_power_comparison_l390_39082

theorem power_comparison : (9^20 : ℝ) < (9999^10 : ℝ) :=
sorry

end NUMINAMATH_GPT_power_comparison_l390_39082


namespace NUMINAMATH_GPT_chef_earns_less_than_manager_l390_39029

noncomputable def manager_wage : ℝ := 6.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage + 0.2 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 2.60 :=
by
  sorry

end NUMINAMATH_GPT_chef_earns_less_than_manager_l390_39029


namespace NUMINAMATH_GPT_one_python_can_eat_per_week_l390_39036

-- Definitions based on the given conditions
def burmese_pythons := 5
def alligators_eaten := 15
def weeks := 3

-- Theorem statement to prove the number of alligators one python can eat per week
theorem one_python_can_eat_per_week : (alligators_eaten / burmese_pythons) / weeks = 1 := 
by 
-- sorry is used to skip the actual proof
sorry

end NUMINAMATH_GPT_one_python_can_eat_per_week_l390_39036


namespace NUMINAMATH_GPT_john_roommates_multiple_of_bob_l390_39004

theorem john_roommates_multiple_of_bob (bob_roommates john_roommates : ℕ) (multiple : ℕ) 
  (h1 : bob_roommates = 10) 
  (h2 : john_roommates = 25) 
  (h3 : john_roommates = multiple * bob_roommates + 5) : 
  multiple = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_roommates_multiple_of_bob_l390_39004


namespace NUMINAMATH_GPT_find_n_values_l390_39053

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def A_n_k (n k : ℕ) : ℕ := (10^n + 54 * 10^k - 1) / 9

def every_A_n_k_prime (n : ℕ) : Prop :=
  ∀ k, k < n → is_prime (A_n_k n k)

theorem find_n_values :
  ∀ n : ℕ, every_A_n_k_prime n → n = 1 ∨ n = 2 := sorry

end NUMINAMATH_GPT_find_n_values_l390_39053


namespace NUMINAMATH_GPT_find_quadruples_l390_39099

def valid_quadruple (x1 x2 x3 x4 : ℝ) : Prop :=
  x1 + x2 * x3 * x4 = 2 ∧ 
  x2 + x3 * x4 * x1 = 2 ∧ 
  x3 + x4 * x1 * x2 = 2 ∧ 
  x4 + x1 * x2 * x3 = 2

theorem find_quadruples (x1 x2 x3 x4 : ℝ) :
  valid_quadruple x1 x2 x3 x4 ↔ (x1, x2, x3, x4) = (1, 1, 1, 1) ∨ 
                                   (x1, x2, x3, x4) = (3, -1, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, 3, -1, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, 3, -1) ∨ 
                                   (x1, x2, x3, x4) = (-1, -1, -1, 3) := by
  sorry

end NUMINAMATH_GPT_find_quadruples_l390_39099


namespace NUMINAMATH_GPT_least_students_with_brown_eyes_and_lunch_box_l390_39034

variable (U : Finset ℕ) (B L : Finset ℕ)
variables (hU : U.card = 25) (hB : B.card = 15) (hL : L.card = 18)

theorem least_students_with_brown_eyes_and_lunch_box : 
  (B ∩ L).card ≥ 8 := by
  sorry

end NUMINAMATH_GPT_least_students_with_brown_eyes_and_lunch_box_l390_39034


namespace NUMINAMATH_GPT_jack_black_balloons_l390_39074

def nancy_balloons := 7
def mary_balloons := 4 * nancy_balloons
def total_mary_nancy_balloons := nancy_balloons + mary_balloons
def jack_balloons := total_mary_nancy_balloons + 3

theorem jack_black_balloons : jack_balloons = 38 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_jack_black_balloons_l390_39074


namespace NUMINAMATH_GPT_rays_total_grocery_bill_l390_39012

-- Conditions
def hamburger_meat_cost : ℝ := 5.0
def crackers_cost : ℝ := 3.50
def frozen_veg_cost_per_bag : ℝ := 2.0
def frozen_veg_bags : ℕ := 4
def cheese_cost : ℝ := 3.50
def discount_rate : ℝ := 0.10

-- Total cost before discount
def total_cost_before_discount : ℝ :=
  hamburger_meat_cost + crackers_cost + (frozen_veg_cost_per_bag * frozen_veg_bags) + cheese_cost

-- Discount amount
def discount_amount : ℝ := discount_rate * total_cost_before_discount

-- Total cost after discount
def total_cost_after_discount : ℝ := total_cost_before_discount - discount_amount

-- Theorem: Ray's total grocery bill
theorem rays_total_grocery_bill : total_cost_after_discount = 18.0 :=
  by
    sorry

end NUMINAMATH_GPT_rays_total_grocery_bill_l390_39012


namespace NUMINAMATH_GPT_total_time_for_5_smoothies_l390_39023

-- Definitions for the conditions
def freeze_time : ℕ := 40
def blend_time_per_smoothie : ℕ := 3
def chop_time_apples_per_smoothie : ℕ := 2
def chop_time_bananas_per_smoothie : ℕ := 3
def chop_time_strawberries_per_smoothie : ℕ := 4
def chop_time_mangoes_per_smoothie : ℕ := 5
def chop_time_pineapples_per_smoothie : ℕ := 6
def number_of_smoothies : ℕ := 5

-- Total chopping time per smoothie
def chop_time_per_smoothie : ℕ := chop_time_apples_per_smoothie + 
                                  chop_time_bananas_per_smoothie + 
                                  chop_time_strawberries_per_smoothie + 
                                  chop_time_mangoes_per_smoothie + 
                                  chop_time_pineapples_per_smoothie

-- Total chopping time for 5 smoothies
def total_chop_time : ℕ := chop_time_per_smoothie * number_of_smoothies

-- Total blending time for 5 smoothies
def total_blend_time : ℕ := blend_time_per_smoothie * number_of_smoothies

-- Total time to make 5 smoothies
def total_time : ℕ := total_chop_time + total_blend_time

-- Theorem statement
theorem total_time_for_5_smoothies : total_time = 115 := by
  sorry

end NUMINAMATH_GPT_total_time_for_5_smoothies_l390_39023


namespace NUMINAMATH_GPT_worker_total_amount_l390_39015

-- Definitions of the conditions
def pay_per_day := 20
def deduction_per_idle_day := 3
def total_days := 60
def idle_days := 40
def worked_days := total_days - idle_days
def earnings := worked_days * pay_per_day
def deductions := idle_days * deduction_per_idle_day

-- Statement of the problem
theorem worker_total_amount : earnings - deductions = 280 := by
  sorry

end NUMINAMATH_GPT_worker_total_amount_l390_39015


namespace NUMINAMATH_GPT_earnings_bc_l390_39055

variable (A B C : ℕ)

theorem earnings_bc :
  A + B + C = 600 →
  A + C = 400 →
  C = 100 →
  B + C = 300 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_earnings_bc_l390_39055


namespace NUMINAMATH_GPT_area_difference_is_correct_l390_39094

noncomputable def area_rectangle (length width : ℝ) : ℝ := length * width

noncomputable def area_equilateral_triangle (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side ^ 2

noncomputable def area_circle (diameter : ℝ) : ℝ := (Real.pi * (diameter / 2) ^ 2)

noncomputable def combined_area_difference : ℝ :=
  (area_rectangle 11 11 + area_rectangle 5.5 11) - 
  (area_equilateral_triangle 6 + area_circle 4)
 
theorem area_difference_is_correct :
  |combined_area_difference - 153.35| < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_area_difference_is_correct_l390_39094


namespace NUMINAMATH_GPT_shaded_area_l390_39024

def radius (R : ℝ) : Prop := R > 0
def angle (α : ℝ) : Prop := α = 20 * (Real.pi / 180)

theorem shaded_area (R : ℝ) (hR : radius R) (hα : angle (20 * (Real.pi / 180))) :
  let S0 := Real.pi * R^2 / 2
  let sector_radius := 2 * R
  let sector_angle := 20 * (Real.pi / 180)
  (2 * sector_radius * sector_radius * sector_angle / 2) / sector_angle = 2 * Real.pi * R^2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l390_39024


namespace NUMINAMATH_GPT_two_pow_p_plus_three_pow_p_not_nth_power_l390_39043

theorem two_pow_p_plus_three_pow_p_not_nth_power (p n : ℕ) (prime_p : Nat.Prime p) (one_lt_n : 1 < n) :
  ¬ ∃ k : ℕ, 2 ^ p + 3 ^ p = k ^ n :=
sorry

end NUMINAMATH_GPT_two_pow_p_plus_three_pow_p_not_nth_power_l390_39043


namespace NUMINAMATH_GPT_tan_2x_eq_sin_x_has_three_solutions_l390_39076

theorem tan_2x_eq_sin_x_has_three_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧ S.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_2x_eq_sin_x_has_three_solutions_l390_39076


namespace NUMINAMATH_GPT_centers_collinear_l390_39086

theorem centers_collinear (k : ℝ) (hk : k ≠ -1) :
    ∀ p : ℝ × ℝ, p = (-k, -2*k-5) → (2*p.1 - p.2 - 5 = 0) :=
by
  sorry

end NUMINAMATH_GPT_centers_collinear_l390_39086


namespace NUMINAMATH_GPT_james_final_weight_l390_39026

noncomputable def initial_weight : ℝ := 120
noncomputable def muscle_gain : ℝ := 0.20 * initial_weight
noncomputable def fat_gain : ℝ := muscle_gain / 4
noncomputable def final_weight (initial_weight muscle_gain fat_gain : ℝ) : ℝ :=
  initial_weight + muscle_gain + fat_gain

theorem james_final_weight :
  final_weight initial_weight muscle_gain fat_gain = 150 :=
by
  sorry

end NUMINAMATH_GPT_james_final_weight_l390_39026


namespace NUMINAMATH_GPT_regions_first_two_sets_regions_all_sets_l390_39090

-- Definitions for the problem
def triangle_regions_first_two_sets (n : ℕ) : ℕ :=
  (n + 1) * (n + 1)

def triangle_regions_all_sets (n : ℕ) : ℕ :=
  3 * n * n + 3 * n + 1

-- Proof Problem 1: Given n points on AB and AC, prove the regions are (n + 1)^2
theorem regions_first_two_sets (n : ℕ) :
  (n * (n + 1) + (n + 1)) = (n + 1) * (n + 1) :=
by sorry

-- Proof Problem 2: Given n points on AB, AC, and BC, prove the regions are 3n^2 + 3n + 1
theorem regions_all_sets (n : ℕ) :
  ((n + 1) * (n + 1) + n * (2 * n + 1)) = 3 * n * n + 3 * n + 1 :=
by sorry

end NUMINAMATH_GPT_regions_first_two_sets_regions_all_sets_l390_39090


namespace NUMINAMATH_GPT_stratified_sampling_third_year_l390_39080

theorem stratified_sampling_third_year :
  ∀ (total students_first_year students_second_year sample_size students_third_year sampled_students : ℕ),
  (total = 900) →
  (students_first_year = 240) →
  (students_second_year = 260) →
  (sample_size = 45) →
  (students_third_year = total - students_first_year - students_second_year) →
  (sampled_students = sample_size * students_third_year / total) →
  sampled_students = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_stratified_sampling_third_year_l390_39080


namespace NUMINAMATH_GPT_canvas_bag_lower_carbon_solution_l390_39002

theorem canvas_bag_lower_carbon_solution :
  let canvas_release_oz := 9600
  let plastic_per_trip_oz := 32
  canvas_release_oz / plastic_per_trip_oz = 300 :=
by
  sorry

end NUMINAMATH_GPT_canvas_bag_lower_carbon_solution_l390_39002


namespace NUMINAMATH_GPT_find_second_number_l390_39006

theorem find_second_number (a b c : ℝ) (h1 : a + b + c = 3.622) (h2 : a = 3.15) (h3 : c = 0.458) : b = 0.014 :=
sorry

end NUMINAMATH_GPT_find_second_number_l390_39006


namespace NUMINAMATH_GPT_new_quadratic_eq_l390_39061

def quadratic_roots_eq (a b c : ℝ) (x1 x2 : ℝ) : Prop :=
  a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0

theorem new_quadratic_eq
  (a b c : ℝ) (x1 x2 : ℝ)
  (h1 : quadratic_roots_eq a b c x1 x2)
  (h_sum : x1 + x2 = -b / a)
  (h_prod : x1 * x2 = c / a) :
  a^3 * x^2 - a * b^2 * x + 2 * c * (b^2 - 2 * a * c) = 0 :=
sorry

end NUMINAMATH_GPT_new_quadratic_eq_l390_39061


namespace NUMINAMATH_GPT_profit_percent_300_l390_39016

theorem profit_percent_300 (SP : ℝ) (h : SP ≠ 0) (CP : ℝ) (h1 : CP = 0.25 * SP) : 
  (SP - CP) / CP * 100 = 300 := 
  sorry

end NUMINAMATH_GPT_profit_percent_300_l390_39016


namespace NUMINAMATH_GPT_series_sum_equals_1_over_400_l390_39079

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_sum_equals_1_over_400 :
  ∑' n, series_term (n + 1) = 1 / 400 := by
  sorry

end NUMINAMATH_GPT_series_sum_equals_1_over_400_l390_39079


namespace NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l390_39084

theorem tan_105_eq_neg2_sub_sqrt3 : Real.tan (105 * Real.pi / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_105_eq_neg2_sub_sqrt3_l390_39084


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l390_39033

theorem tenth_term_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 23)
  (h2 : a + 7 * d = 55) :
  a + 9 * d = 71 :=
sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l390_39033


namespace NUMINAMATH_GPT_evaluate_expression_l390_39007

noncomputable def a : ℕ := 3^2 + 5^2 + 7^2
noncomputable def b : ℕ := 2^2 + 4^2 + 6^2

theorem evaluate_expression : (a / b : ℚ) - (b / a : ℚ) = 3753 / 4656 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l390_39007


namespace NUMINAMATH_GPT_range_is_80_l390_39081

def dataSet : List ℕ := [60, 100, 80, 40, 20]

def minValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl min x xs

def maxValue (l : List ℕ) : ℕ :=
  match l with
  | [] => 0
  | (x :: xs) => List.foldl max x xs

def range (l : List ℕ) : ℕ :=
  maxValue l - minValue l

theorem range_is_80 : range dataSet = 80 :=
by
  sorry

end NUMINAMATH_GPT_range_is_80_l390_39081


namespace NUMINAMATH_GPT_sum_of_cubes_divisible_by_nine_l390_39044

theorem sum_of_cubes_divisible_by_nine (n : ℕ) (h : 0 < n) : 9 ∣ (n^3 + (n + 1)^3 + (n + 2)^3) :=
by sorry

end NUMINAMATH_GPT_sum_of_cubes_divisible_by_nine_l390_39044


namespace NUMINAMATH_GPT_arithmetic_seq_common_difference_l390_39054

theorem arithmetic_seq_common_difference (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 7 * a 11 = 6) (h2 : a 4 + a (14) = 5) : 
  d = 1 / 4 ∨ d = -1 / 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_common_difference_l390_39054


namespace NUMINAMATH_GPT_probability_of_green_ball_is_2_over_5_l390_39049

noncomputable def container_probabilities : ℚ :=
  let prob_A_selected : ℚ := 1/2
  let prob_B_selected : ℚ := 1/2
  let prob_green_in_A : ℚ := 5/10
  let prob_green_in_B : ℚ := 3/10

  prob_A_selected * prob_green_in_A + prob_B_selected * prob_green_in_B

theorem probability_of_green_ball_is_2_over_5 :
  container_probabilities = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_of_green_ball_is_2_over_5_l390_39049


namespace NUMINAMATH_GPT_find_four_digit_number_l390_39089

/-- 
  If there exists a positive integer M and M² both end in the same sequence of 
  five digits abcde in base 10 where a ≠ 0, 
  then the four-digit number abcd derived from M = 96876 is 9687.
-/
theorem find_four_digit_number
  (M : ℕ)
  (h_end_digits : (M % 100000) = (M * M % 100000))
  (h_first_digit_nonzero : 10000 ≤ M % 100000  ∧ M % 100000 < 100000)
  : (M = 96876 → (M / 10 % 10000 = 9687)) :=
by { sorry }

end NUMINAMATH_GPT_find_four_digit_number_l390_39089


namespace NUMINAMATH_GPT_sum_digits_n_plus_one_l390_39014

/-- 
Let S(n) be the sum of the digits of a positive integer n.
Given S(n) = 29, prove that the possible values of S(n + 1) are 3, 12, or 30.
-/
theorem sum_digits_n_plus_one (S : ℕ → ℕ) (n : ℕ) (h : S n = 29) :
  S (n + 1) = 3 ∨ S (n + 1) = 12 ∨ S (n + 1) = 30 := 
sorry

end NUMINAMATH_GPT_sum_digits_n_plus_one_l390_39014


namespace NUMINAMATH_GPT_solution_to_equation_l390_39098

theorem solution_to_equation (x : ℝ) : x * (x - 2) = 2 * x ↔ (x = 0 ∨ x = 4) := by
  sorry

end NUMINAMATH_GPT_solution_to_equation_l390_39098


namespace NUMINAMATH_GPT_product_fraction_l390_39064

theorem product_fraction :
  (1 + 1/2) * (1 + 1/4) * (1 + 1/6) * (1 + 1/8) * (1 + 1/10) = 693 / 256 := by
  sorry

end NUMINAMATH_GPT_product_fraction_l390_39064


namespace NUMINAMATH_GPT_count_integer_values_l390_39032

theorem count_integer_values (π : Real) (hπ : Real.pi = π):
  ∃ n : ℕ, n = 27 ∧ ∀ x : ℤ, |(x:Real)| < 4 * π + 1 ↔ -13 ≤ x ∧ x ≤ 13 :=
by sorry

end NUMINAMATH_GPT_count_integer_values_l390_39032


namespace NUMINAMATH_GPT_quadratic_points_order_l390_39010

theorem quadratic_points_order (c y1 y2 : ℝ) 
  (hA : y1 = 0^2 - 6 * 0 + c)
  (hB : y2 = 4^2 - 6 * 4 + c) : 
  y1 > y2 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_points_order_l390_39010


namespace NUMINAMATH_GPT_machine_A_production_l390_39001

-- Definitions based on the conditions
def machine_production (A B: ℝ) (TA TB: ℝ) : Prop :=
  B = 1.10 * A ∧
  TA = TB + 10 ∧
  A * TA = 660 ∧
  B * TB = 660

-- The main statement to be proved: Machine A produces 6 sprockets per hour.
theorem machine_A_production (A B: ℝ) (TA TB: ℝ) 
  (h : machine_production A B TA TB) : 
  A = 6 := 
by sorry

end NUMINAMATH_GPT_machine_A_production_l390_39001


namespace NUMINAMATH_GPT_compare_negative_fractions_l390_39020

theorem compare_negative_fractions :
  (-3 : ℚ) / 4 > (-4 : ℚ) / 5 :=
sorry

end NUMINAMATH_GPT_compare_negative_fractions_l390_39020


namespace NUMINAMATH_GPT_work_days_together_l390_39025

-- Conditions
variable {W : ℝ} (h_a_alone : ∀ (W : ℝ), W / a_work_time = W / 16)
variable {a_work_time : ℝ} (h_work_time_a : a_work_time = 16)

-- Question translated to proof problem
theorem work_days_together (D : ℝ) :
  (10 * (W / D) + 12 * (W / 16) = W) → D = 40 :=
by
  intros h
  have eq1 : 10 * (W / D) + 12 * (W / 16) = W := h
  sorry

end NUMINAMATH_GPT_work_days_together_l390_39025


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l390_39013

noncomputable def arithmetic_sequence (a : ℕ → ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + 2

theorem problem_part1 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) :
  a 2 = 4 := 
sorry

theorem problem_part2 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 2) (h2 : S 2 = a 3) (h3 : arithmetic_sequence a) 
  (h4 : ∀ n : ℕ, S n = n * (a 1 + a n) / 2) :
  S 10 = 110 :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l390_39013


namespace NUMINAMATH_GPT_trains_meet_at_9am_l390_39067

-- Definitions of conditions
def distance_AB : ℝ := 65
def speed_train_A : ℝ := 20
def speed_train_B : ℝ := 25
def start_time_train_A : ℝ := 7
def start_time_train_B : ℝ := 8

-- This function calculates the meeting time of the two trains
noncomputable def meeting_time (distance_AB : ℝ) (speed_train_A : ℝ) (speed_train_B : ℝ) 
    (start_time_train_A : ℝ) (start_time_train_B : ℝ) : ℝ :=
  let distance_train_A := speed_train_A * (start_time_train_B - start_time_train_A)
  let remaining_distance := distance_AB - distance_train_A
  let relative_speed := speed_train_A + speed_train_B
  start_time_train_B + remaining_distance / relative_speed

-- Theorem stating the time when the two trains meet
theorem trains_meet_at_9am :
    meeting_time distance_AB speed_train_A speed_train_B start_time_train_A start_time_train_B = 9 := sorry

end NUMINAMATH_GPT_trains_meet_at_9am_l390_39067


namespace NUMINAMATH_GPT_find_f_4_l390_39088

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2

theorem find_f_4 : f 4 = 2 := 
by {
    -- The proof is omitted as per the task.
    sorry
}

end NUMINAMATH_GPT_find_f_4_l390_39088


namespace NUMINAMATH_GPT_mrs_sheridan_fish_distribution_l390_39039

theorem mrs_sheridan_fish_distribution :
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium
  fish_in_large_aquarium = 225 :=
by {
  let initial_fish := 125
  let additional_fish := 250
  let total_fish := initial_fish + additional_fish
  let small_aquarium_capacity := 150
  let fish_in_small_aquarium := small_aquarium_capacity
  let fish_in_large_aquarium := total_fish - fish_in_small_aquarium

  have : fish_in_large_aquarium = 225 := by sorry
  exact this
}

end NUMINAMATH_GPT_mrs_sheridan_fish_distribution_l390_39039


namespace NUMINAMATH_GPT_solve_x_l390_39037

theorem solve_x (x : ℝ) (h : (4 * x + 3) / (3 * x ^ 2 + 4 * x - 4) = 3 * x / (3 * x - 2)) :
  x = (-1 + Real.sqrt 10) / 3 ∨ x = (-1 - Real.sqrt 10) / 3 :=
by sorry

end NUMINAMATH_GPT_solve_x_l390_39037


namespace NUMINAMATH_GPT_derivative_at_one_l390_39073

open Real

noncomputable def f (x : ℝ) : ℝ := exp x / x

theorem derivative_at_one : deriv f 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_one_l390_39073


namespace NUMINAMATH_GPT_square_field_diagonal_l390_39028

theorem square_field_diagonal (a : ℝ) (d : ℝ) (h : a^2 = 800) : d = 40 :=
by
  sorry

end NUMINAMATH_GPT_square_field_diagonal_l390_39028


namespace NUMINAMATH_GPT_gwen_total_books_l390_39021

theorem gwen_total_books
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (mystery_shelves_count : mystery_shelves = 3)
  (picture_shelves_count : picture_shelves = 5)
  (each_shelf_books : books_per_shelf = 9) :
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf) = 72 := by
  sorry

end NUMINAMATH_GPT_gwen_total_books_l390_39021


namespace NUMINAMATH_GPT_values_of_fractions_l390_39059

theorem values_of_fractions (A B : ℝ) :
  (∀ x : ℝ, 3 * x ^ 2 + 2 * x - 8 ≠ 0) →
  (∀ x : ℝ, (6 * x - 7) / (3 * x ^ 2 + 2 * x - 8) = A / (x - 2) + B / (3 * x + 4)) →
  A = 1 / 2 ∧ B = 4.5 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_values_of_fractions_l390_39059


namespace NUMINAMATH_GPT_anya_hairs_wanted_more_l390_39095

def anya_initial_number_of_hairs : ℕ := 0 -- for simplicity, assume she starts with 0 hairs
def hairs_lost_washing : ℕ := 32
def hairs_lost_brushing : ℕ := hairs_lost_washing / 2
def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing
def hairs_to_grow_back : ℕ := 49

theorem anya_hairs_wanted_more : total_hairs_lost + hairs_to_grow_back = 97 :=
by
  sorry

end NUMINAMATH_GPT_anya_hairs_wanted_more_l390_39095


namespace NUMINAMATH_GPT_min_value_frac_eq_nine_halves_l390_39038

theorem min_value_frac_eq_nine_halves {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 2) :
  ∃ (x y : ℝ), 2 / x + 1 / y = 9 / 2 := by
  sorry

end NUMINAMATH_GPT_min_value_frac_eq_nine_halves_l390_39038


namespace NUMINAMATH_GPT_correct_statements_proof_l390_39017

theorem correct_statements_proof :
  (∀ (a b : ℤ), a - 3 = b - 3 → a = b) ∧
  ¬ (∀ (a b c : ℤ), a = b → a + c = b - c) ∧
  (∀ (a b m : ℤ), m ≠ 0 → (a / m) = (b / m) → a = b) ∧
  ¬ (∀ (a : ℤ), a^2 = 2 * a → a = 2) :=
by
  -- Here we would prove the statements individually:
  -- sorry is a placeholder suggesting that the proofs need to be filled in.
  sorry

end NUMINAMATH_GPT_correct_statements_proof_l390_39017


namespace NUMINAMATH_GPT_perfect_cubes_in_range_l390_39008

theorem perfect_cubes_in_range (K : ℤ) (hK_pos : K > 1) (Z : ℤ) 
  (hZ_eq : Z = K ^ 3) (hZ_range: 600 < Z ∧ Z < 2000) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 :=
by
  sorry

end NUMINAMATH_GPT_perfect_cubes_in_range_l390_39008


namespace NUMINAMATH_GPT_isosceles_trapezoid_side_length_l390_39066

theorem isosceles_trapezoid_side_length (A b1 b2 : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 48) (hb1 : b1 = 9) (hb2 : b2 = 15) 
  (h_area : A = 1 / 2 * (b1 + b2) * h) 
  (h_h : h = 4)
  (h_s : s^2 = h^2 + ((b2 - b1) / 2)^2) :
  s = 5 :=
by sorry

end NUMINAMATH_GPT_isosceles_trapezoid_side_length_l390_39066


namespace NUMINAMATH_GPT_smallest_multiple_of_6_8_12_l390_39011

theorem smallest_multiple_of_6_8_12 : ∃ n : ℕ, n > 0 ∧ n % 6 = 0 ∧ n % 8 = 0 ∧ n % 12 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 6 = 0 ∧ m % 8 = 0 ∧ m % 12 = 0) → n ≤ m := 
sorry

end NUMINAMATH_GPT_smallest_multiple_of_6_8_12_l390_39011


namespace NUMINAMATH_GPT_sum_of_roots_eq_neg_five_l390_39030

theorem sum_of_roots_eq_neg_five (x₁ x₂ : ℝ) (h₁ : x₁^2 + 5 * x₁ - 2 = 0) (h₂ : x₂^2 + 5 * x₂ - 2 = 0) (h_distinct : x₁ ≠ x₂) :
  x₁ + x₂ = -5 := sorry

end NUMINAMATH_GPT_sum_of_roots_eq_neg_five_l390_39030


namespace NUMINAMATH_GPT_four_people_pairing_l390_39048

theorem four_people_pairing
    (persons : Fin 4 → Type)
    (common_language : ∀ (i j : Fin 4), Prop)
    (communicable : ∀ (i j k : Fin 4), common_language i j ∨ common_language j k ∨ common_language k i)
    : ∃ (i j : Fin 4) (k l : Fin 4), i ≠ j ∧ k ≠ l ∧ common_language i j ∧ common_language k l := 
sorry

end NUMINAMATH_GPT_four_people_pairing_l390_39048


namespace NUMINAMATH_GPT_tank_capacity_l390_39050

noncomputable def inflow_A (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_B (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def inflow_C (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_X (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

noncomputable def outflow_Y (rate : ℕ) (time_hr : ℕ) : ℕ :=
rate * 60 * time_hr

theorem tank_capacity
  (fA : ℕ := inflow_A 8 7)
  (fB : ℕ := inflow_B 12 3)
  (fC : ℕ := inflow_C 6 4)
  (oX : ℕ := outflow_X 20 7)
  (oY : ℕ := outflow_Y 15 5) :
  fA + fB + fC = 6960 ∧ oX + oY = 12900 ∧ 12900 - 6960 = 5940 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l390_39050


namespace NUMINAMATH_GPT_incorrect_mark_l390_39058

theorem incorrect_mark (n : ℕ) (correct_mark incorrect_entry : ℕ) (average_increase : ℕ) :
  n = 40 → correct_mark = 63 → average_increase = 1/2 →
  incorrect_entry - correct_mark = average_increase * n →
  incorrect_entry = 83 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end NUMINAMATH_GPT_incorrect_mark_l390_39058


namespace NUMINAMATH_GPT_hausdorff_dimension_union_sup_l390_39019

open Set

noncomputable def Hausdorff_dimension (A : Set ℝ) : ℝ :=
sorry -- Definition for Hausdorff dimension is nontrivial and can be added here

theorem hausdorff_dimension_union_sup {A : ℕ → Set ℝ} :
  Hausdorff_dimension (⋃ i, A i) = ⨆ i, Hausdorff_dimension (A i) :=
sorry

end NUMINAMATH_GPT_hausdorff_dimension_union_sup_l390_39019


namespace NUMINAMATH_GPT_star_intersections_l390_39096

theorem star_intersections (n k : ℕ) (h_coprime : Nat.gcd n k = 1) (h_n_ge_5 : 5 ≤ n) (h_k_lt_n_div_2 : k < n / 2) :
    k = 25 → n = 2018 → n * (k - 1) = 48432 := by
  intros
  sorry

end NUMINAMATH_GPT_star_intersections_l390_39096


namespace NUMINAMATH_GPT_greatest_number_of_unit_segments_l390_39060

-- Define the conditions
def is_equilateral (n : ℕ) : Prop := n > 0

-- Define the theorem
theorem greatest_number_of_unit_segments (n : ℕ) (h : is_equilateral n) : 
  -- Prove the greatest number of unit segments such that no three of them form a single triangle
  ∃(m : ℕ), m = n * (n + 1) := 
sorry

end NUMINAMATH_GPT_greatest_number_of_unit_segments_l390_39060


namespace NUMINAMATH_GPT_value_2_stddev_less_than_mean_l390_39072

theorem value_2_stddev_less_than_mean :
  let mean := 17.5
  let stddev := 2.5
  mean - 2 * stddev = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_value_2_stddev_less_than_mean_l390_39072


namespace NUMINAMATH_GPT_find_k_l390_39051

theorem find_k : ∀ (x y k : ℤ), (x = -y) → (2 * x + 5 * y = k) → (x - 3 * y = 16) → (k = -12) :=
by
  intros x y k h1 h2 h3
  sorry

end NUMINAMATH_GPT_find_k_l390_39051


namespace NUMINAMATH_GPT_RupertCandles_l390_39077

-- Definitions corresponding to the conditions
def PeterAge : ℕ := 10
def RupertRelativeAge : ℝ := 3.5

-- Define Rupert's age based on Peter's age and the given relative age factor
def RupertAge : ℝ := RupertRelativeAge * PeterAge

-- Statement of the theorem
theorem RupertCandles : RupertAge = 35 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_RupertCandles_l390_39077


namespace NUMINAMATH_GPT_oak_trees_remaining_is_7_l390_39091

-- Define the number of oak trees initially in the park
def initial_oak_trees : ℕ := 9

-- Define the number of oak trees cut down by workers
def oak_trees_cut_down : ℕ := 2

-- Define the remaining oak trees calculation
def remaining_oak_trees : ℕ := initial_oak_trees - oak_trees_cut_down

-- Prove that the remaining oak trees is equal to 7
theorem oak_trees_remaining_is_7 : remaining_oak_trees = 7 := by
  sorry

end NUMINAMATH_GPT_oak_trees_remaining_is_7_l390_39091


namespace NUMINAMATH_GPT_largest_of_five_consecutive_sum_l390_39003

theorem largest_of_five_consecutive_sum (n : ℕ) 
  (h : n + (n+1) + (n+2) + (n+3) + (n+4) = 90) : 
  n + 4 = 20 :=
sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_sum_l390_39003


namespace NUMINAMATH_GPT_sum_of_ages_l390_39078

theorem sum_of_ages (J M R : ℕ) (hJ : J = 10) (hM : M = J - 3) (hR : R = J + 2) : M + R = 19 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l390_39078


namespace NUMINAMATH_GPT_non_right_triangle_option_l390_39046

-- Definitions based on conditions
def optionA (A B C : ℝ) : Prop := A + B = C
def optionB (A B C : ℝ) : Prop := A - B = C
def optionC (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def optionD (A B C : ℝ) : Prop := A = B ∧ A = 3 * C

-- Given conditions for a right triangle
def is_right_triangle (A B C : ℝ) : Prop := ∃(θ : ℝ), θ = 90 ∧ (A = θ ∨ B = θ ∨ C = θ)

-- The proof problem
theorem non_right_triangle_option (A B C : ℝ) :
  optionD A B C ∧ ¬(is_right_triangle A B C) := sorry

end NUMINAMATH_GPT_non_right_triangle_option_l390_39046


namespace NUMINAMATH_GPT_speed_of_stream_l390_39040

/-- Given Athul's rowing conditions, prove the speed of the stream is 1 km/h. -/
theorem speed_of_stream 
  (A S : ℝ)
  (h1 : 16 = (A - S) * 4)
  (h2 : 24 = (A + S) * 4) : 
  S = 1 := 
sorry

end NUMINAMATH_GPT_speed_of_stream_l390_39040


namespace NUMINAMATH_GPT_bees_flew_in_l390_39041

theorem bees_flew_in (initial_bees : ℕ) (total_bees : ℕ) (new_bees : ℕ) (h1 : initial_bees = 16) (h2 : total_bees = 23) (h3 : total_bees = initial_bees + new_bees) : new_bees = 7 :=
by
  sorry

end NUMINAMATH_GPT_bees_flew_in_l390_39041


namespace NUMINAMATH_GPT_range_of_fraction_l390_39075

-- Definition of the quadratic equation with roots within specified intervals
variables (a b : ℝ)
variables (x1 x2 : ℝ)
variables (h_distinct_roots : x1 ≠ x2)
variables (h_interval_x1 : 0 < x1 ∧ x1 < 1)
variables (h_interval_x2 : 1 < x2 ∧ x2 < 2)
variables (h_quadratic : ∀ x : ℝ, x^2 + a * x + 2 * b - 2 = 0)

-- Prove range of expression
theorem range_of_fraction (a b : ℝ)
  (x1 x2 h_distinct_roots : ℝ) (h_interval_x1 : 0 < x1 ∧ x1 < 1)
  (h_interval_x2 : 1 < x2 ∧ x2 < 2)
  (h_quadratic : ∀ x, x^2 + a * x + 2 * b - 2 = 0) :
  (1/2 < (b - 4) / (a - 1)) ∧ ((b - 4) / (a - 1) < 3/2) :=
by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_range_of_fraction_l390_39075


namespace NUMINAMATH_GPT_david_number_sum_l390_39097

theorem david_number_sum :
  ∃ (x y : ℕ), (10 ≤ x ∧ x < 100) ∧ (100 ≤ y ∧ y < 1000) ∧ (1000 * x + y = 4 * x * y) ∧ (x + y = 266) :=
sorry

end NUMINAMATH_GPT_david_number_sum_l390_39097


namespace NUMINAMATH_GPT_hyperbola_equation_correct_l390_39087

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :=
  (x y : ℝ) -> (x^2 / 5) - (y^2 / 20) = 1

theorem hyperbola_equation_correct {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :
  hyperbola_equation a b a_pos b_pos focal_len asymptote_slope :=
by {
  sorry
}

end NUMINAMATH_GPT_hyperbola_equation_correct_l390_39087


namespace NUMINAMATH_GPT_last_digit_square_of_second_l390_39052

def digit1 := 1
def digit2 := 3
def digit3 := 4
def digit4 := 9

theorem last_digit_square_of_second :
  digit4 = digit2 ^ 2 :=
by
  -- Conditions
  have h1 : digit1 = digit2 / 3 := by sorry
  have h2 : digit3 = digit1 + digit2 := by sorry
  sorry

end NUMINAMATH_GPT_last_digit_square_of_second_l390_39052


namespace NUMINAMATH_GPT_sum_of_roots_l390_39063

theorem sum_of_roots (x : ℝ) :
  (x + 2) * (x - 3) = 16 →
  ∃ a b : ℝ, (a ≠ x ∧ b ≠ x ∧ (x - a) * (x - b) = 0) ∧
             (a + b = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_roots_l390_39063


namespace NUMINAMATH_GPT_total_expenditure_is_3500_l390_39092

def expenditure_mon : ℕ := 450
def expenditure_tue : ℕ := 600
def expenditure_wed : ℕ := 400
def expenditure_thurs : ℕ := 500
def expenditure_sat : ℕ := 550
def expenditure_sun : ℕ := 300
def cost_earphone : ℕ := 620
def cost_pen : ℕ := 30
def cost_notebook : ℕ := 50

def expenditure_fri : ℕ := cost_earphone + cost_pen + cost_notebook
def total_expenditure : ℕ := expenditure_mon + expenditure_tue + expenditure_wed + expenditure_thurs + expenditure_fri + expenditure_sat + expenditure_sun

theorem total_expenditure_is_3500 : total_expenditure = 3500 := by
  sorry

end NUMINAMATH_GPT_total_expenditure_is_3500_l390_39092


namespace NUMINAMATH_GPT_zoo_initial_animals_l390_39045

theorem zoo_initial_animals (X : ℕ) :
  X - 6 + 1 + 3 + 8 + 16 = 90 → X = 68 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_zoo_initial_animals_l390_39045


namespace NUMINAMATH_GPT_find_m_n_l390_39031

def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := x^3 + m * x^2 + n * x + 1

theorem find_m_n (m n : ℝ) (x : ℝ) (hx : x ≠ 0 ∧ f x m n = 1 ∧ (3 * x^2 + 2 * m * x + n = 0) ∧ (∀ y, f y m n ≥ -31 ∧ f (-2) m n = -31)) :
  m = 12 ∧ n = 36 :=
sorry

end NUMINAMATH_GPT_find_m_n_l390_39031
