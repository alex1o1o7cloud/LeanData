import Mathlib

namespace NUMINAMATH_GPT_cost_of_each_item_l1456_145650

theorem cost_of_each_item (initial_order items : ℕ) (price per_item_reduction additional_orders : ℕ) (reduced_order total_order reduced_price profit_per_item : ℕ) 
  (h1 : initial_order = 60)
  (h2 : price = 100)
  (h3 : per_item_reduction = 1)
  (h4 : additional_orders = 3)
  (h5 : reduced_price = price - price * 4 / 100)
  (h6 : total_order = initial_order + additional_orders * (price * 4 / 100))
  (h7 : reduced_order = total_order)
  (h8 : profit_per_item = price - per_item_reduction )
  (h9 : profit_per_item = 24)
  (h10 : items * profit_per_item = reduced_order * (profit_per_item - per_item_reduction)) :
  (price - profit_per_item = 76) :=
by sorry

end NUMINAMATH_GPT_cost_of_each_item_l1456_145650


namespace NUMINAMATH_GPT_mode_of_shoe_sizes_is_25_5_l1456_145679

def sales_data := [(24, 2), (24.5, 5), (25, 3), (25.5, 6), (26, 4)]

theorem mode_of_shoe_sizes_is_25_5 
  (h : ∀ x ∈ sales_data, 2 ≤ x.1 ∧ 
        (∀ y ∈ sales_data, x.2 ≤ y.2 → x.1 = 25.5 ∨ x.2 < 6)) : 
  (∃ s, s ∈ sales_data ∧ s.1 = 25.5 ∧ s.2 = 6) :=
sorry

end NUMINAMATH_GPT_mode_of_shoe_sizes_is_25_5_l1456_145679


namespace NUMINAMATH_GPT_percentage_non_defective_l1456_145655

theorem percentage_non_defective :
  let total_units : ℝ := 100
  let M1_percentage : ℝ := 0.20
  let M2_percentage : ℝ := 0.25
  let M3_percentage : ℝ := 0.30
  let M4_percentage : ℝ := 0.15
  let M5_percentage : ℝ := 0.10
  let M1_defective_percentage : ℝ := 0.02
  let M2_defective_percentage : ℝ := 0.04
  let M3_defective_percentage : ℝ := 0.05
  let M4_defective_percentage : ℝ := 0.07
  let M5_defective_percentage : ℝ := 0.08

  let M1_total := total_units * M1_percentage
  let M2_total := total_units * M2_percentage
  let M3_total := total_units * M3_percentage
  let M4_total := total_units * M4_percentage
  let M5_total := total_units * M5_percentage

  let M1_defective := M1_total * M1_defective_percentage
  let M2_defective := M2_total * M2_defective_percentage
  let M3_defective := M3_total * M3_defective_percentage
  let M4_defective := M4_total * M4_defective_percentage
  let M5_defective := M5_total * M5_defective_percentage

  let total_defective := M1_defective + M2_defective + M3_defective + M4_defective + M5_defective
  let total_non_defective := total_units - total_defective
  let percentage_non_defective := (total_non_defective / total_units) * 100

  percentage_non_defective = 95.25 := by
  sorry

end NUMINAMATH_GPT_percentage_non_defective_l1456_145655


namespace NUMINAMATH_GPT_find_side_length_l1456_145625

theorem find_side_length (a : ℝ) (b : ℝ) (A B : ℝ) (ha : a = 4) (hA : A = 45) (hB : B = 60) :
    b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_find_side_length_l1456_145625


namespace NUMINAMATH_GPT_vaccination_target_failure_l1456_145604

noncomputable def percentage_vaccination_target_failed (original_target : ℕ) (first_year : ℕ) (second_year_increase_rate : ℚ) (third_year : ℕ) : ℚ :=
  let second_year := first_year + second_year_increase_rate * first_year
  let total_vaccinated := first_year + second_year + third_year
  let shortfall := original_target - total_vaccinated
  (shortfall / original_target) * 100

theorem vaccination_target_failure :
  percentage_vaccination_target_failed 720 60 (65/100 : ℚ) 150 = 57.11 := 
  by sorry

end NUMINAMATH_GPT_vaccination_target_failure_l1456_145604


namespace NUMINAMATH_GPT_suma_work_rate_l1456_145693

theorem suma_work_rate (r s : ℝ) (hr : r = 1 / 5) (hrs : r + s = 1 / 4) : 1 / s = 20 := by
  sorry

end NUMINAMATH_GPT_suma_work_rate_l1456_145693


namespace NUMINAMATH_GPT_correct_average_weight_is_58_6_l1456_145627

noncomputable def initial_avg_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def incorrect_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 60
noncomputable def correct_avg_weight := (initial_avg_weight * num_boys + (correct_weight - incorrect_weight)) / num_boys

theorem correct_average_weight_is_58_6 :
  correct_avg_weight = 58.6 :=
sorry

end NUMINAMATH_GPT_correct_average_weight_is_58_6_l1456_145627


namespace NUMINAMATH_GPT_possible_birches_l1456_145685

theorem possible_birches (N B L : ℕ) (hN : N = 130) (h_sum : B + L = 130)
  (h_linden_false : ∀ l, l < L → (∀ b, b < B → b + l < N → b < B → False))
  (h_birch_false : ∃ b, b < B ∧ (∀ l, l < L → l + b < N → l + b = 2 * B))
  : B = 87 :=
sorry

end NUMINAMATH_GPT_possible_birches_l1456_145685


namespace NUMINAMATH_GPT_zero_in_interval_l1456_145659

theorem zero_in_interval (a b : ℝ) (ha : 1 < a) (hb : 0 < b ∧ b < 1) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ (a^x + x - b = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_zero_in_interval_l1456_145659


namespace NUMINAMATH_GPT_discriminant_of_quad_eq_l1456_145621

def a : ℕ := 5
def b : ℕ := 8
def c : ℤ := -6

def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

theorem discriminant_of_quad_eq : discriminant 5 8 (-6) = 184 :=
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_discriminant_of_quad_eq_l1456_145621


namespace NUMINAMATH_GPT_flower_bouquet_violets_percentage_l1456_145692

theorem flower_bouquet_violets_percentage
  (total_flowers yellow_flowers purple_flowers : ℕ)
  (yellow_daisies yellow_tulips purple_violets : ℕ)
  (h_yellow_flowers : yellow_flowers = (total_flowers / 2))
  (h_purple_flowers : purple_flowers = (total_flowers / 2))
  (h_yellow_daisies : yellow_daisies = (yellow_flowers / 5))
  (h_yellow_tulips : yellow_tulips = yellow_flowers - yellow_daisies)
  (h_purple_violets : purple_violets = (purple_flowers / 2)) :
  ((purple_violets : ℚ) / total_flowers) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_flower_bouquet_violets_percentage_l1456_145692


namespace NUMINAMATH_GPT_other_number_remainder_l1456_145605

theorem other_number_remainder (x : ℕ) (k n : ℤ) (hx : x > 0) (hk : 200 = k * x + 2) (hnk : n ≠ k) : ∃ m : ℤ, (n * ↑x + 2) = m * ↑x + 2 ∧ (n * ↑x + 2) % x = 2 := 
by
  sorry

end NUMINAMATH_GPT_other_number_remainder_l1456_145605


namespace NUMINAMATH_GPT_fundraiser_full_price_revenue_l1456_145619

theorem fundraiser_full_price_revenue :
  ∃ (f h p : ℕ), f + h = 200 ∧ 
                f * p + h * (p / 2) = 2700 ∧ 
                f * p = 600 :=
by 
  sorry

end NUMINAMATH_GPT_fundraiser_full_price_revenue_l1456_145619


namespace NUMINAMATH_GPT_find_Finley_age_l1456_145640

variable (Roger Jill Finley : ℕ)
variable (Jill_age : Jill = 20)
variable (Roger_age : Roger = 2 * Jill + 5)
variable (Finley_condition : 15 + (Roger - Jill) = Finley - 30)

theorem find_Finley_age : Finley = 55 :=
by
  sorry

end NUMINAMATH_GPT_find_Finley_age_l1456_145640


namespace NUMINAMATH_GPT_mans_speed_against_current_l1456_145656

theorem mans_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (h1 : speed_with_current = 25)
  (h2 : speed_of_current = 2.5) :
  speed_with_current - 2 * speed_of_current = 20 := 
by
  sorry

end NUMINAMATH_GPT_mans_speed_against_current_l1456_145656


namespace NUMINAMATH_GPT_calculate_expression_l1456_145669

theorem calculate_expression : (3072 - 2993) ^ 2 / 121 = 49 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1456_145669


namespace NUMINAMATH_GPT_units_digit_n_l1456_145677

theorem units_digit_n (m n : ℕ) (h1 : m * n = 14^5) (h2 : m % 10 = 8) : n % 10 = 3 :=
sorry

end NUMINAMATH_GPT_units_digit_n_l1456_145677


namespace NUMINAMATH_GPT_determine_teeth_l1456_145616

theorem determine_teeth (x V : ℝ) (h1 : V = 63 * x / (x + 10)) (h2 : V = 28 * (x + 10)) :
  x = 20 ∧ (x + 10) = 30 :=
by
  sorry

end NUMINAMATH_GPT_determine_teeth_l1456_145616


namespace NUMINAMATH_GPT_permutation_six_two_l1456_145601

-- Definition for permutation
def permutation (n k : ℕ) : ℕ := n * (n - 1)

-- Theorem stating that the permutation of 6 taken 2 at a time is 30
theorem permutation_six_two : permutation 6 2 = 30 :=
by
  -- proof will be filled here
  sorry

end NUMINAMATH_GPT_permutation_six_two_l1456_145601


namespace NUMINAMATH_GPT_initial_value_divisible_by_456_l1456_145660

def initial_value := 374
def to_add := 82
def divisor := 456

theorem initial_value_divisible_by_456 : (initial_value + to_add) % divisor = 0 := by
  sorry

end NUMINAMATH_GPT_initial_value_divisible_by_456_l1456_145660


namespace NUMINAMATH_GPT_solve_quadratic_l1456_145608

theorem solve_quadratic : ∀ (x : ℝ), x^2 - 5 * x + 1 = 0 →
  (x = (5 + Real.sqrt 21) / 2) ∨ (x = (5 - Real.sqrt 21) / 2) :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1456_145608


namespace NUMINAMATH_GPT_find_angle_C_find_side_c_l1456_145649

variable {A B C a b c : ℝ}
variable {AD CD area_ABD : ℝ}

-- Conditions for question 1
variable (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))

-- Conditions for question 2
variable (h2 : AD = 4)
variable (h3 : CD = 4)
variable (h4 : area_ABD = 8 * Real.sqrt 3)
variable (h5 : C = Real.pi / 3)

-- Lean 4 statement for both parts of the problem
theorem find_angle_C (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A)) : 
  C = Real.pi / 3 :=
sorry

theorem find_side_c (h2 : AD = 4) (h3 : CD = 4) (h4 : area_ABD = 8 * Real.sqrt 3) (h5 : C = Real.pi / 3) : 
  c = 4 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_find_angle_C_find_side_c_l1456_145649


namespace NUMINAMATH_GPT_binary_digit_one_l1456_145652
-- We import the necessary libraries

-- Define the problem and prove the statement as follows
def fractional_part_in_binary (x : ℝ) : ℕ → ℕ := sorry

def sqrt_fractional_binary (k : ℕ) (i : ℕ) : ℕ :=
  fractional_part_in_binary (Real.sqrt ((k : ℝ) * (k + 1))) i

theorem binary_digit_one {k n : ℕ} (hk : 0 < k) (hn : 0 < n) :
  ∃ i, n + 1 ≤ i ∧ i ≤ 2 * n + 1 ∧ sqrt_fractional_binary k i = 1 :=
sorry

end NUMINAMATH_GPT_binary_digit_one_l1456_145652


namespace NUMINAMATH_GPT_quadratic_decreasing_conditions_l1456_145681

theorem quadratic_decreasing_conditions (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → ∃ y : ℝ, y = ax^2 + 4*(a+1)*x - 3 ∧ (∀ z : ℝ, z ≥ x → y ≥ (ax^2 + 4*(a+1)*z - 3))) ↔ a ∈ Set.Iic (-1 / 2) :=
sorry

end NUMINAMATH_GPT_quadratic_decreasing_conditions_l1456_145681


namespace NUMINAMATH_GPT_sales_tax_percentage_l1456_145602

theorem sales_tax_percentage 
  (total_spent : ℝ)
  (tip_percent : ℝ)
  (food_price : ℝ) 
  (total_with_tip : total_spent = food_price * (1 + tip_percent / 100))
  (sales_tax_percent : ℝ) 
  (total_paid : total_spent = food_price * (1 + sales_tax_percent / 100) * (1 + tip_percent / 100)) :
  sales_tax_percent = 10 :=
by sorry

end NUMINAMATH_GPT_sales_tax_percentage_l1456_145602


namespace NUMINAMATH_GPT_lcm_210_297_l1456_145696

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := 
by sorry

end NUMINAMATH_GPT_lcm_210_297_l1456_145696


namespace NUMINAMATH_GPT_crews_complete_job_l1456_145653

-- Define the productivity rates for each crew
variables (x y z : ℝ)

-- Define the conditions derived from the problem
def condition1 : Prop := 1/(x + y) = 1/z - 3/5
def condition2 : Prop := 1/(x + z) = 1/y
def condition3 : Prop := 1/(y + z) = 2/(7 * x)

-- Target proof: the combined time for all three crews
def target_proof : Prop := 1/(x + y + z) = 4/3

-- Final Lean 4 statement combining all conditions and proof requirement
theorem crews_complete_job (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) (h3 : condition3 x y z) : target_proof x y z :=
sorry

end NUMINAMATH_GPT_crews_complete_job_l1456_145653


namespace NUMINAMATH_GPT_Derek_more_than_Zoe_l1456_145688

-- Define the variables for the number of books Emily, Derek, and Zoe have
variables (E : ℝ)

-- Condition: Derek has 75% more books than Emily
def Derek_books : ℝ := 1.75 * E

-- Condition: Zoe has 50% more books than Emily
def Zoe_books : ℝ := 1.5 * E

-- Statement asserting that Derek has 16.67% more books than Zoe
theorem Derek_more_than_Zoe (hD: Derek_books E = 1.75 * E) (hZ: Zoe_books E = 1.5 * E) :
  (Derek_books E - Zoe_books E) / Zoe_books E = 0.1667 :=
by
  sorry

end NUMINAMATH_GPT_Derek_more_than_Zoe_l1456_145688


namespace NUMINAMATH_GPT_find_solutions_l1456_145626

theorem find_solutions (k : ℤ) (x y : ℤ) (h : x^2 - 2*y^2 = k) :
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ (t = x + 2*y ∨ t = x - 2*y) ∧ (u = x + y ∨ u = x - y) :=
sorry

end NUMINAMATH_GPT_find_solutions_l1456_145626


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l1456_145666

def p (x : ℝ) : Prop := abs x = -x
def q (x : ℝ) : Prop := x^2 ≥ -x

theorem p_necessary_not_sufficient_for_q : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l1456_145666


namespace NUMINAMATH_GPT_min_value_frac_l1456_145622

theorem min_value_frac (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  ∃ (x : ℝ), x = 1/m + 4/n ∧ x ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_l1456_145622


namespace NUMINAMATH_GPT_cube_identity_simplification_l1456_145690

theorem cube_identity_simplification (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3 * x * y * z) / (x * y * z) = 6 :=
by
  sorry

end NUMINAMATH_GPT_cube_identity_simplification_l1456_145690


namespace NUMINAMATH_GPT_propositions_p_q_l1456_145636

theorem propositions_p_q
  (p q : Prop)
  (h : ¬(p ∧ q) = False) : p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_propositions_p_q_l1456_145636


namespace NUMINAMATH_GPT_find_blue_shirts_l1456_145647

-- Statements of the problem conditions
def total_shirts : ℕ := 23
def green_shirts : ℕ := 17

-- Definition that we want to prove
def blue_shirts : ℕ := total_shirts - green_shirts

-- Proof statement (no need to include the proof itself)
theorem find_blue_shirts : blue_shirts = 6 := by
  sorry

end NUMINAMATH_GPT_find_blue_shirts_l1456_145647


namespace NUMINAMATH_GPT_paul_final_balance_l1456_145689

def initial_balance : ℝ := 400
def transfer1 : ℝ := 90
def transfer2 : ℝ := 60
def service_charge_rate : ℝ := 0.02

def service_charge (x : ℝ) : ℝ := service_charge_rate * x

def total_deduction : ℝ := transfer1 + service_charge transfer1 + service_charge transfer2

def final_balance (init_balance : ℝ) (deduction : ℝ) : ℝ := init_balance - deduction

theorem paul_final_balance :
  final_balance initial_balance total_deduction = 307 :=
by
  sorry

end NUMINAMATH_GPT_paul_final_balance_l1456_145689


namespace NUMINAMATH_GPT_rectangle_area_l1456_145641

theorem rectangle_area (L W : ℕ) (h1 : 2 * L + 2 * W = 280) (h2 : L = 5 * (W / 2)) : L * W = 4000 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l1456_145641


namespace NUMINAMATH_GPT_cost_of_each_card_is_2_l1456_145695

-- Define the conditions
def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def total_spent : ℝ := 70

-- Define the total number of cards
def total_cards : ℕ := christmas_cards + birthday_cards

-- Define the cost per card
noncomputable def cost_per_card : ℝ := total_spent / total_cards

-- The theorem
theorem cost_of_each_card_is_2 : cost_per_card = 2 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_card_is_2_l1456_145695


namespace NUMINAMATH_GPT_a_n_strictly_monotonic_increasing_l1456_145614

noncomputable def a_n (n : ℕ) : ℝ := 
  2 * ((1 + 1 / (n : ℝ)) ^ (2 * n + 1)) / (((1 + 1 / (n : ℝ)) ^ n) + ((1 + 1 / (n : ℝ)) ^ (n + 1)))

theorem a_n_strictly_monotonic_increasing : ∀ n : ℕ, a_n (n + 1) > a_n n :=
sorry

end NUMINAMATH_GPT_a_n_strictly_monotonic_increasing_l1456_145614


namespace NUMINAMATH_GPT_cumulus_to_cumulonimbus_ratio_l1456_145683

theorem cumulus_to_cumulonimbus_ratio (cirrus cumulonimbus cumulus : ℕ) (x : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = x * cumulonimbus)
  (h3 : cumulonimbus = 3)
  (h4 : cirrus = 144) :
  x = 12 := by
  sorry

end NUMINAMATH_GPT_cumulus_to_cumulonimbus_ratio_l1456_145683


namespace NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1456_145603

theorem bus_speed_excluding_stoppages 
  (v_s : ℕ) -- Speed including stoppages in kmph
  (stop_duration_minutes : ℕ) -- Duration of stoppages in minutes per hour
  (stop_duration_fraction : ℚ := stop_duration_minutes / 60) -- Fraction of hour stopped
  (moving_fraction : ℚ := 1 - stop_duration_fraction) -- Fraction of hour moving
  (distance_per_hour : ℚ := v_s) -- Distance traveled per hour including stoppages
  (v : ℚ) -- Speed excluding stoppages
  
  (h1 : v_s = 50)
  (h2 : stop_duration_minutes = 10)
  
  -- Equation representing the total distance equals the distance traveled moving
  (h3 : v * moving_fraction = distance_per_hour)
: v = 60 := sorry

end NUMINAMATH_GPT_bus_speed_excluding_stoppages_l1456_145603


namespace NUMINAMATH_GPT_student_failed_by_40_marks_l1456_145651

theorem student_failed_by_40_marks (total_marks : ℕ) (passing_percentage : ℝ) (marks_obtained : ℕ) (h1 : total_marks = 500) (h2 : passing_percentage = 33) (h3 : marks_obtained = 125) :
  ((passing_percentage / 100) * total_marks - marks_obtained : ℝ) = 40 :=
sorry

end NUMINAMATH_GPT_student_failed_by_40_marks_l1456_145651


namespace NUMINAMATH_GPT_num_fish_when_discovered_l1456_145662

open Nat

/-- Definition of the conditions given in the problem --/
def initial_fish := 60
def fish_per_day_eaten := 2
def additional_fish := 8
def weeks_before_addition := 2
def extra_week := 1

/-- The proof problem statement --/
theorem num_fish_when_discovered : 
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  final_fish = 26 := 
by
  let days := (weeks_before_addition + extra_week) * 7
  let total_fish_eaten := days * fish_per_day_eaten
  let fish_after_addition := initial_fish + additional_fish
  let final_fish := fish_after_addition - total_fish_eaten
  have h : final_fish = 26 := sorry
  exact h

end NUMINAMATH_GPT_num_fish_when_discovered_l1456_145662


namespace NUMINAMATH_GPT_exponential_inequality_l1456_145631

theorem exponential_inequality (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) (h3 : x₂ < 1) : 
  x₂ * Real.exp x₁ > x₁ * Real.exp x₂ :=
sorry

end NUMINAMATH_GPT_exponential_inequality_l1456_145631


namespace NUMINAMATH_GPT_stephanie_total_remaining_bills_l1456_145674

-- Conditions
def electricity_bill : ℕ := 60
def electricity_paid : ℕ := electricity_bill
def gas_bill : ℕ := 40
def gas_paid : ℕ := (3 * gas_bill) / 4 + 5
def water_bill : ℕ := 40
def water_paid : ℕ := water_bill / 2
def internet_bill : ℕ := 25
def internet_payment : ℕ := 5
def internet_paid : ℕ := 4 * internet_payment

-- Define
def remaining_electricity : ℕ := electricity_bill - electricity_paid
def remaining_gas : ℕ := gas_bill - gas_paid
def remaining_water : ℕ := water_bill - water_paid
def remaining_internet : ℕ := internet_bill - internet_paid

def total_remaining : ℕ := remaining_electricity + remaining_gas + remaining_water + remaining_internet

-- Problem Statement
theorem stephanie_total_remaining_bills :
  total_remaining = 30 :=
by
  -- proof goes here (not required as per the instructions)
  sorry

end NUMINAMATH_GPT_stephanie_total_remaining_bills_l1456_145674


namespace NUMINAMATH_GPT_question1_1_question1_2_question2_l1456_145661

open Set

noncomputable def universal_set : Set ℝ := univ

def setA : Set ℝ := { x | x^2 - 9 * x + 18 ≥ 0 }

def setB : Set ℝ := { x | -2 < x ∧ x < 9 }

def setC (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem question1_1 : ∀ x, x ∈ setA ∨ x ∈ setB :=
by sorry

theorem question1_2 : ∀ x, x ∈ (universal_set \ setA) ∩ setB ↔ (3 < x ∧ x < 6) :=
by sorry

theorem question2 (a : ℝ) (h : setC a ⊆ setB) : -2 ≤ a ∧ a ≤ 8 :=
by sorry

end NUMINAMATH_GPT_question1_1_question1_2_question2_l1456_145661


namespace NUMINAMATH_GPT_smallest_number_of_students_l1456_145615

theorem smallest_number_of_students (n : ℕ) :
  (n % 3 = 2) ∧
  (n % 5 = 3) ∧
  (n % 8 = 5) →
  n = 53 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_smallest_number_of_students_l1456_145615


namespace NUMINAMATH_GPT_bookshop_shipment_correct_l1456_145684

noncomputable def bookshop_shipment : ℕ :=
  let Initial_books := 743
  let Saturday_instore_sales := 37
  let Saturday_online_sales := 128
  let Sunday_instore_sales := 2 * Saturday_instore_sales
  let Sunday_online_sales := Saturday_online_sales + 34
  let books_sold := Saturday_instore_sales + Saturday_online_sales + Sunday_instore_sales + Sunday_online_sales
  let Final_books := 502
  Final_books - (Initial_books - books_sold)

theorem bookshop_shipment_correct : bookshop_shipment = 160 := by
  sorry

end NUMINAMATH_GPT_bookshop_shipment_correct_l1456_145684


namespace NUMINAMATH_GPT_probability_square_area_l1456_145686

theorem probability_square_area (AB : ℝ) (M : ℝ) (h1 : AB = 12) (h2 : 0 ≤ M) (h3 : M ≤ AB) :
  (∃ (AM : ℝ), (AM = M) ∧ (36 ≤ AM^2 ∧ AM^2 ≤ 81)) → 
  (∃ (p : ℝ), p = 1/4) :=
by
  sorry

end NUMINAMATH_GPT_probability_square_area_l1456_145686


namespace NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1456_145680

def p (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 3
def q (x : ℝ) : Prop := x ≠ 0

theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_p_sufficient_not_necessary_for_q_l1456_145680


namespace NUMINAMATH_GPT_probability_pair_tile_l1456_145606

def letters_in_word : List Char := ['P', 'R', 'O', 'B', 'A', 'B', 'I', 'L', 'I', 'T', 'Y']
def target_letters : List Char := ['P', 'A', 'I', 'R']

def num_favorable_outcomes : Nat :=
  -- Count occurrences of letters in target_letters within letters_in_word
  List.count 'P' letters_in_word +
  List.count 'A' letters_in_word +
  List.count 'I' letters_in_word + 
  List.count 'R' letters_in_word

def total_outcomes : Nat := letters_in_word.length

theorem probability_pair_tile :
  (num_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_probability_pair_tile_l1456_145606


namespace NUMINAMATH_GPT_calculate_percentage_l1456_145618

theorem calculate_percentage :
  let total_students := 40
  let A_on_both := 4
  let B_on_both := 6
  let C_on_both := 3
  let D_on_Test1_C_on_Test2 := 2
  let valid_students := A_on_both + B_on_both + C_on_both + D_on_Test1_C_on_Test2
  (valid_students / total_students) * 100 = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_percentage_l1456_145618


namespace NUMINAMATH_GPT_min_value_when_a_is_negative_one_max_value_bounds_l1456_145697

-- Conditions
def f (a x : ℝ) : ℝ := a * x^2 + x
def a1 : ℝ := -1
def a : ℝ := -2
def a_lower_bound : ℝ := -2
def a_upper_bound : ℝ := 0
def interval : Set ℝ := Set.Icc 0 2

-- Part I: Minimum value when a = -1
theorem min_value_when_a_is_negative_one : 
  ∃ x ∈ interval, f a1 x = -2 := 
by
  sorry

-- Part II: Maximum value criterions
theorem max_value_bounds (a : ℝ) (H : a ∈ Set.Icc a_lower_bound a_upper_bound) :
  (∀ x ∈ interval, 
    (a ≥ -1/4 → f a ( -1 / (2 * a) ) = -1 / (4 * a)) 
    ∧ (a < -1/4 → f a 2 = 4 * a + 2 )) :=
by
  sorry

end NUMINAMATH_GPT_min_value_when_a_is_negative_one_max_value_bounds_l1456_145697


namespace NUMINAMATH_GPT_min_green_beads_l1456_145676

theorem min_green_beads (B R G : ℕ) (h : B + R + G = 80)
  (hB : ∀ i j : ℕ, (i < j ∧ j ≤ B → ∃ k, i < k ∧ k < j ∧ k ≤ R)) 
  (hR : ∀ i j : ℕ, (i < j ∧ j ≤ R → ∃ k, i < k ∧ k < j ∧ k ≤ G)) :
  G >= 27 := 
sorry

end NUMINAMATH_GPT_min_green_beads_l1456_145676


namespace NUMINAMATH_GPT_speed_ratio_l1456_145637

theorem speed_ratio :
  ∀ (v_A v_B : ℝ), (v_A / v_B = 3 / 2) ↔ (v_A = 3 * v_B / 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_speed_ratio_l1456_145637


namespace NUMINAMATH_GPT_cube_surface_area_difference_l1456_145624

theorem cube_surface_area_difference :
  let large_cube_volume := 8
  let small_cube_volume := 1
  let num_small_cubes := 8
  let large_cube_side := (large_cube_volume : ℝ) ^ (1 / 3)
  let small_cube_side := (small_cube_volume : ℝ) ^ (1 / 3)
  let large_cube_surface_area := 6 * (large_cube_side ^ 2)
  let small_cube_surface_area := 6 * (small_cube_side ^ 2)
  let total_small_cubes_surface_area := num_small_cubes * small_cube_surface_area
  total_small_cubes_surface_area - large_cube_surface_area = 24 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_difference_l1456_145624


namespace NUMINAMATH_GPT_amount_transferred_l1456_145657

def original_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : original_balance - remaining_balance = 69 :=
by
  sorry

end NUMINAMATH_GPT_amount_transferred_l1456_145657


namespace NUMINAMATH_GPT_initial_percentage_decrease_l1456_145664

theorem initial_percentage_decrease (x : ℝ) (P : ℝ) (h₀ : P > 0)
  (initial_decrease : ∀ (x : ℝ), P * (1 - x / 100) * 1.3 = P * 1.04) :
  x = 20 :=
by 
  sorry

end NUMINAMATH_GPT_initial_percentage_decrease_l1456_145664


namespace NUMINAMATH_GPT_condition_B_is_necessary_but_not_sufficient_l1456_145665

-- Definitions of conditions A and B
def condition_A (x : ℝ) : Prop := 0 < x ∧ x < 5
def condition_B (x : ℝ) : Prop := abs (x - 2) < 3

-- The proof problem statement
theorem condition_B_is_necessary_but_not_sufficient : 
∀ x, condition_A x → condition_B x ∧ ¬(∀ x, condition_B x → condition_A x) := 
sorry

end NUMINAMATH_GPT_condition_B_is_necessary_but_not_sufficient_l1456_145665


namespace NUMINAMATH_GPT_intersection_M_N_l1456_145675

def M : Set ℝ := {x | x < 2016}

def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1456_145675


namespace NUMINAMATH_GPT_eggs_remaining_l1456_145668

-- Assign the given constants
def hens : ℕ := 3
def eggs_per_hen_per_day : ℕ := 3
def days_gone : ℕ := 7
def eggs_taken_by_neighbor : ℕ := 12
def eggs_dropped_by_myrtle : ℕ := 5

-- Calculate the expected number of eggs Myrtle should have
noncomputable def total_eggs :=
  hens * eggs_per_hen_per_day * days_gone - eggs_taken_by_neighbor - eggs_dropped_by_myrtle

-- Prove that the total number of eggs equals the correct answer
theorem eggs_remaining : total_eggs = 46 :=
by
  sorry

end NUMINAMATH_GPT_eggs_remaining_l1456_145668


namespace NUMINAMATH_GPT_length_of_overlapping_part_l1456_145610

theorem length_of_overlapping_part
  (l_p : ℕ)
  (n : ℕ)
  (total_length : ℕ)
  (l_o : ℕ) :
  n = 3 →
  l_p = 217 →
  total_length = 627 →
  3 * l_p - 2 * l_o = total_length →
  l_o = 12 := by
  intros n_eq l_p_eq total_length_eq equation
  sorry

end NUMINAMATH_GPT_length_of_overlapping_part_l1456_145610


namespace NUMINAMATH_GPT_optimal_purchase_interval_discount_advantage_l1456_145667

/- The functions and assumptions used here. -/
def purchase_feed_days (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) : ℕ :=
-- Implementation omitted
sorry

def should_use_discount (feed_per_day : ℕ) (price_per_kg : ℝ) 
  (storage_cost_per_kg_per_day : ℝ) (transportation_fee : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) : Prop :=
-- Implementation omitted
sorry

/- Conditions -/
def conditions : Prop :=
  let feed_per_day := 200
  let price_per_kg := 1.8
  let storage_cost_per_kg_per_day := 0.03
  let transportation_fee := 300
  let discount_threshold := 5000 -- in kg, since 5 tons = 5000 kg
  let discount_rate := 0.85
  True -- We apply these values in the proofs below.

/- Main statements -/
theorem optimal_purchase_interval : conditions → 
  purchase_feed_days 200 1.8 0.03 300 = 10 :=
by
  intros
  -- Proof is omitted.
  sorry

theorem discount_advantage : conditions →
  should_use_discount 200 1.8 0.03 300 5000 0.85 :=
by
  intros
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_optimal_purchase_interval_discount_advantage_l1456_145667


namespace NUMINAMATH_GPT_geom_seq_min_m_l1456_145694

def initial_capital : ℝ := 50
def growth_rate : ℝ := 0.5
def annual_payment (t : ℝ) : Prop := t ≤ 2500
def capital_remaining (aₙ : ℕ → ℝ) (n : ℕ) (t : ℝ) : ℝ := aₙ n * (1 + growth_rate) - t

theorem geom_seq (aₙ : ℕ → ℝ) (t : ℝ) (h₁ : annual_payment t) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (t ≠ 2500) →
  ∃ r : ℝ, ∀ n, aₙ n - 2 * t = (aₙ 0 - 2 * t) * r ^ n :=
sorry

theorem min_m (t : ℝ) (h₁ : t = 1500) (aₙ : ℕ → ℝ) :
  (∀ n, aₙ (n + 1) = 3 / 2 * aₙ n - t) →
  (aₙ 0 = initial_capital * (1 + growth_rate) - t) →
  ∃ m : ℕ, aₙ m > 21000 ∧ ∀ k < m, aₙ k ≤ 21000 :=
sorry

end NUMINAMATH_GPT_geom_seq_min_m_l1456_145694


namespace NUMINAMATH_GPT_expected_digits_fair_icosahedral_die_l1456_145645

noncomputable def expected_number_of_digits : ℝ :=
  let one_digit_count := 9
  let two_digit_count := 11
  let total_faces := 20
  let prob_one_digit := one_digit_count / total_faces
  let prob_two_digit := two_digit_count / total_faces
  (prob_one_digit * 1) + (prob_two_digit * 2)

theorem expected_digits_fair_icosahedral_die :
  expected_number_of_digits = 1.55 :=
by
  sorry

end NUMINAMATH_GPT_expected_digits_fair_icosahedral_die_l1456_145645


namespace NUMINAMATH_GPT_keegan_total_school_time_l1456_145663

-- Definition of the conditions
def keegan_classes : Nat := 7
def history_and_chemistry_time : ℝ := 1.5
def other_class_time : ℝ := 1.2

-- The theorem stating that given these conditions, Keegan spends 7.5 hours a day in school.
theorem keegan_total_school_time : 
  (history_and_chemistry_time + 5 * other_class_time) = 7.5 := 
by
  sorry

end NUMINAMATH_GPT_keegan_total_school_time_l1456_145663


namespace NUMINAMATH_GPT_number_of_pairs_l1456_145670

theorem number_of_pairs (n : ℕ) (h : n = 2835) :
  ∃ (count : ℕ), count = 20 ∧
  (∀ (x y : ℕ), (0 < x ∧ 0 < y ∧ x < y ∧ (x^2 + y^2) % (x + y) = 0 ∧ (x^2 + y^2) / (x + y) ∣ n) → count = 20) := 
sorry

end NUMINAMATH_GPT_number_of_pairs_l1456_145670


namespace NUMINAMATH_GPT_polyhedra_impossible_l1456_145691

noncomputable def impossible_polyhedra_projections (p1_outer : List (ℝ × ℝ)) (p1_inner : List (ℝ × ℝ))
                                                  (p2_outer : List (ℝ × ℝ)) (p2_inner : List (ℝ × ℝ)) : Prop :=
  -- Add definitions for the vertices labeling here 
  let vertices_outer := ["A", "B", "C", "D"]
  let vertices_inner := ["A1", "B1", "C1", "D1"]
  -- Add the conditions for projection (a) and (b) 
  p1_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p1_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] ∧
  p2_outer = [(0,0), (1,0), (1,1), (0,1)] ∧
  p2_inner = [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] →
  -- Prove that the polyhedra corresponding to these projections are impossible.
  false

-- Now let's state the theorem
theorem polyhedra_impossible : impossible_polyhedra_projections [(0,0), (1,0), (1,1), (0,1)] 
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)]
                                                                [(0,0), (1,0), (1,1), (0,1)]
                                                                [(0.25,0.25), (0.75,0.25), (0.75,0.75), (0.25,0.75)] := 
by {
  sorry
}

end NUMINAMATH_GPT_polyhedra_impossible_l1456_145691


namespace NUMINAMATH_GPT_expression_evaluation_l1456_145682

theorem expression_evaluation : 
  (1 : ℝ)^(6 * z - 3) / (7⁻¹ + 4⁻¹) = 28 / 11 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1456_145682


namespace NUMINAMATH_GPT_tadpole_catch_l1456_145638

variable (T : ℝ) (H1 : T * 0.25 = 45)

theorem tadpole_catch (T : ℝ) (H1 : T * 0.25 = 45) : T = 180 :=
sorry

end NUMINAMATH_GPT_tadpole_catch_l1456_145638


namespace NUMINAMATH_GPT_calculation_A_correct_l1456_145623

theorem calculation_A_correct : (-1: ℝ)^4 * (-1: ℝ)^3 = 1 := by
  sorry

end NUMINAMATH_GPT_calculation_A_correct_l1456_145623


namespace NUMINAMATH_GPT_correct_operation_l1456_145673

theorem correct_operation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = - (a^2 * b) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1456_145673


namespace NUMINAMATH_GPT_total_campers_rowing_and_hiking_l1456_145699

def campers_morning_rowing : ℕ := 41
def campers_morning_hiking : ℕ := 4
def campers_afternoon_rowing : ℕ := 26

theorem total_campers_rowing_and_hiking :
  campers_morning_rowing + campers_morning_hiking + campers_afternoon_rowing = 71 :=
by
  -- We are skipping the proof since instructions specify only the statement is needed
  sorry

end NUMINAMATH_GPT_total_campers_rowing_and_hiking_l1456_145699


namespace NUMINAMATH_GPT_remainder_of_polynomial_l1456_145698

theorem remainder_of_polynomial :
  ∀ (x : ℂ), (x^4 + x^3 + x^2 + x + 1 = 0) → (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^4 + x^3 + x^2 + x + 1) = 2 :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_l1456_145698


namespace NUMINAMATH_GPT_find_constants_l1456_145687

noncomputable def f (x m n : ℝ) := (m * x + 1) / (x + n)

theorem find_constants (m n : ℝ) (h_symm : ∀ x y, f x m n = y → f (4 - x) m n = 8 - y) : 
  m = 4 ∧ n = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_constants_l1456_145687


namespace NUMINAMATH_GPT_exists_factorial_with_first_digits_2015_l1456_145648

theorem exists_factorial_with_first_digits_2015 : ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, 2015 * (10^k) ≤ n! ∧ n! < 2016 * (10^k)) :=
sorry

end NUMINAMATH_GPT_exists_factorial_with_first_digits_2015_l1456_145648


namespace NUMINAMATH_GPT_agatha_bike_budget_l1456_145672

def total_initial : ℕ := 60
def cost_frame : ℕ := 15
def cost_front_wheel : ℕ := 25
def total_spent : ℕ := cost_frame + cost_front_wheel
def total_left : ℕ := total_initial - total_spent

theorem agatha_bike_budget : total_left = 20 := by
  sorry

end NUMINAMATH_GPT_agatha_bike_budget_l1456_145672


namespace NUMINAMATH_GPT_minimum_guests_l1456_145613

theorem minimum_guests (total_food : ℤ) (max_food_per_guest : ℤ) (food_bound : total_food = 325) (guest_bound : max_food_per_guest = 2) : (⌈total_food / max_food_per_guest⌉ : ℤ) = 163 :=
by {
  sorry 
}

end NUMINAMATH_GPT_minimum_guests_l1456_145613


namespace NUMINAMATH_GPT_max_sum_m_n_l1456_145646

noncomputable def ellipse_and_hyperbola_max_sum : Prop :=
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ (∃ x y : ℝ, (x^2 / 25 + y^2 / m^2 = 1 ∧ x^2 / 7 - y^2 / n^2 = 1)) ∧
  (25 - m^2 = 7 + n^2) ∧ (m + n = 6)

theorem max_sum_m_n : ellipse_and_hyperbola_max_sum :=
  sorry

end NUMINAMATH_GPT_max_sum_m_n_l1456_145646


namespace NUMINAMATH_GPT_locus_of_P_l1456_145628

-- Definitions based on conditions
def F : ℝ × ℝ := (2, 0)
def Q (k : ℝ) : ℝ × ℝ := (0, -2 * k)
def T (k : ℝ) : ℝ × ℝ := (-2 * k^2, 0)
def P (k : ℝ) : ℝ × ℝ := (2 * k^2, -4 * k)

-- Theorem statement based on the proof problem
theorem locus_of_P (x y : ℝ) (k : ℝ) (hf : F = (2, 0)) (hq : Q k = (0, -2 * k))
  (ht : T k = (-2 * k^2, 0)) (hp : P k = (2 * k^2, -4 * k)) :
  y^2 = 8 * x :=
sorry

end NUMINAMATH_GPT_locus_of_P_l1456_145628


namespace NUMINAMATH_GPT_product_of_y_coordinates_l1456_145633

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem product_of_y_coordinates : 
  let P1 := (1, 2 + 4 * Real.sqrt 2)
  let P2 := (1, 2 - 4 * Real.sqrt 2)
  distance (5, 2) P1 = 12 ∧ distance (5, 2) P2 = 12 →
  (P1.2 * P2.2 = -28) :=
by
  intros
  sorry

end NUMINAMATH_GPT_product_of_y_coordinates_l1456_145633


namespace NUMINAMATH_GPT_sum_of_integers_l1456_145632

theorem sum_of_integers (a b : ℕ) (h1 : a * a + b * b = 585) (h2 : Nat.gcd a b + Nat.lcm a b = 87) : a + b = 33 := 
sorry

end NUMINAMATH_GPT_sum_of_integers_l1456_145632


namespace NUMINAMATH_GPT_solve_problem_l1456_145654

theorem solve_problem (a : ℝ) (x : ℝ) (h1 : 3 * x + |a - 2| = -3) (h2 : 3 * x + 4 = 0) :
  (a = 3 ∨ a = 1) → ((a - 2) ^ 2010 - 2 * a + 1 = -4 ∨ (a - 2) ^ 2010 - 2 * a + 1 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_problem_l1456_145654


namespace NUMINAMATH_GPT_mechanic_worked_hours_l1456_145620

theorem mechanic_worked_hours (total_spent : ℕ) (cost_per_part : ℕ) (labor_cost_per_minute : ℚ) (parts_needed : ℕ) :
  total_spent = 220 → cost_per_part = 20 → labor_cost_per_minute = 0.5 → parts_needed = 2 →
  (total_spent - cost_per_part * parts_needed) / labor_cost_per_minute / 60 = 6 := by
  -- Proof will be inserted here
  sorry

end NUMINAMATH_GPT_mechanic_worked_hours_l1456_145620


namespace NUMINAMATH_GPT_f_at_2023_l1456_145639

noncomputable def f (a x : ℝ) : ℝ := (a - x) / (a + 2 * x)

noncomputable def g (a x : ℝ) : ℝ := (f a (x - 2023)) + (1 / 2)

def is_odd (g : ℝ → ℝ) := ∀ x : ℝ, g (-x) = -g x

variable (a : ℝ)
variable (h_a : a ≠ 0)
variable (h_odd : is_odd (g a))

theorem f_at_2023 : f a 2023 = 1 / 4 :=
sorry

end NUMINAMATH_GPT_f_at_2023_l1456_145639


namespace NUMINAMATH_GPT_absolute_value_inequality_solution_l1456_145611

theorem absolute_value_inequality_solution (x : ℝ) :
  |x - 2| + |x - 4| ≤ 3 ↔ (3 / 2 ≤ x ∧ x < 4) :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_inequality_solution_l1456_145611


namespace NUMINAMATH_GPT_maximize_container_volume_l1456_145612

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ (∀ y : ℝ, 0 < y ∧ y < 24 → (90 - 2*y) * (48 - 2*y) * y ≤ (90 - 2*x) * (48 - 2*x) * x) ∧ x = 10 :=
sorry

end NUMINAMATH_GPT_maximize_container_volume_l1456_145612


namespace NUMINAMATH_GPT_calc_factorial_sum_l1456_145617

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end NUMINAMATH_GPT_calc_factorial_sum_l1456_145617


namespace NUMINAMATH_GPT_dot_product_vec_a_vec_b_l1456_145629

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem dot_product_vec_a_vec_b : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 3 := by
  sorry

end NUMINAMATH_GPT_dot_product_vec_a_vec_b_l1456_145629


namespace NUMINAMATH_GPT_cos_neg_300_l1456_145609

theorem cos_neg_300 : Real.cos (-(300 : ℝ) * Real.pi / 180) = 1 / 2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cos_neg_300_l1456_145609


namespace NUMINAMATH_GPT_intersection_M_N_l1456_145634

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x * x = x}

theorem intersection_M_N :
  M ∩ N = {0, 1} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l1456_145634


namespace NUMINAMATH_GPT_quadratic_always_positive_l1456_145642

theorem quadratic_always_positive (x : ℝ) : x^2 + x + 1 > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_always_positive_l1456_145642


namespace NUMINAMATH_GPT_problem_statement_l1456_145643

variable {x y : ℝ}

theorem problem_statement 
  (h1 : y > x)
  (h2 : x > 0)
  (h3 : x + y = 1) :
  x < 2 * x * y ∧ 2 * x * y < (x + y) / 2 ∧ (x + y) / 2 < y := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1456_145643


namespace NUMINAMATH_GPT_root_equation_solution_l1456_145630

-- Given conditions from the problem
def is_root_of_quadratic (m : ℝ) : Prop :=
  m^2 - m - 110 = 0

-- Statement of the proof problem
theorem root_equation_solution (m : ℝ) (h : is_root_of_quadratic m) : (m - 1)^2 + m = 111 := 
sorry

end NUMINAMATH_GPT_root_equation_solution_l1456_145630


namespace NUMINAMATH_GPT_polynomial_identity_l1456_145635

theorem polynomial_identity (a b c : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a^2 + b^2 + c^2 = 15) 
  (h3 : a^3 + b^3 + c^3 = 47) : 
  (a^2 + ab + b^2) * (b^2 + bc + c^2) * (c^2 + ca + a^2) = 625 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_identity_l1456_145635


namespace NUMINAMATH_GPT_total_boxes_l1456_145671

variable (N_initial : ℕ) (N_nonempty : ℕ) (N_new_boxes : ℕ)

theorem total_boxes (h_initial : N_initial = 7) 
                     (h_nonempty : N_nonempty = 10)
                     (h_new_boxes : N_new_boxes = N_nonempty * 7) :
  N_initial + N_new_boxes = 77 :=
by 
  have : N_initial = 7 := h_initial
  have : N_new_boxes = N_nonempty * 7 := h_new_boxes
  have : N_nonempty = 10 := h_nonempty
  sorry

end NUMINAMATH_GPT_total_boxes_l1456_145671


namespace NUMINAMATH_GPT_find_three_digit_number_l1456_145607

def is_three_digit_number (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000

def digits_sum (n : ℕ) : ℕ :=
  let a := n / 100
  let b := (n % 100) / 10
  let c := n % 10
  a + b + c

theorem find_three_digit_number : 
  ∃ n : ℕ, is_three_digit_number n ∧ n^2 = (digits_sum n)^5 ∧ n = 243 :=
sorry

end NUMINAMATH_GPT_find_three_digit_number_l1456_145607


namespace NUMINAMATH_GPT_value_of_six_inch_cube_l1456_145644

-- Defining the conditions
def original_cube_weight : ℝ := 5 -- in pounds
def original_cube_value : ℝ := 600 -- in dollars
def original_cube_side : ℝ := 4 -- in inches

def new_cube_side : ℝ := 6 -- in inches

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

-- Statement of the theorem
theorem value_of_six_inch_cube :
  cube_volume new_cube_side / cube_volume original_cube_side * original_cube_value = 2025 :=
by
  -- Here goes the proof
  sorry

end NUMINAMATH_GPT_value_of_six_inch_cube_l1456_145644


namespace NUMINAMATH_GPT_range_of_a_l1456_145600

def is_ellipse (a : ℝ) : Prop :=
  2 * a > 0 ∧ 3 * a - 6 > 0 ∧ 2 * a < 3 * a - 6

def discriminant_neg (a : ℝ) : Prop :=
  a^2 + 8 * a - 48 < 0

def p (a : ℝ) : Prop := is_ellipse a
def q (a : ℝ) : Prop := discriminant_neg a

theorem range_of_a (a : ℝ) : p a ∧ q a → 2 < a ∧ a < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1456_145600


namespace NUMINAMATH_GPT_range_of_a_l1456_145658

noncomputable def set_A (a : ℝ) : Set ℝ := {x | x < a}
noncomputable def set_B : Set ℝ := {x | 1 < x ∧ x < 2}
noncomputable def complement_B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 2 }

theorem range_of_a (a : ℝ) : (set_A a ∪ complement_B) = Set.univ ↔ 2 ≤ a := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1456_145658


namespace NUMINAMATH_GPT_lcm_60_30_40_eq_120_l1456_145678

theorem lcm_60_30_40_eq_120 : (Nat.lcm (Nat.lcm 60 30) 40) = 120 := 
sorry

end NUMINAMATH_GPT_lcm_60_30_40_eq_120_l1456_145678
