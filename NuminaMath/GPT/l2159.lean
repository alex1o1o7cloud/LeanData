import Mathlib

namespace NUMINAMATH_GPT_prob_mc_tf_correct_prob_at_least_one_mc_correct_l2159_215988

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2
def total_outcomes : ℕ := total_questions * (total_questions - 1)

-- Probability calculation for one drawing a multiple-choice and the other drawing a true/false question
def prob_mc_tf : ℚ := (multiple_choice_questions * true_false_questions + true_false_questions * multiple_choice_questions) / total_outcomes

-- Probability calculation for at least one drawing a multiple-choice question
def prob_at_least_one_mc : ℚ := 1 - (true_false_questions * (true_false_questions - 1)) / total_outcomes

theorem prob_mc_tf_correct : prob_mc_tf = 3/5 := by
  sorry

theorem prob_at_least_one_mc_correct : prob_at_least_one_mc = 9/10 := by
  sorry

end NUMINAMATH_GPT_prob_mc_tf_correct_prob_at_least_one_mc_correct_l2159_215988


namespace NUMINAMATH_GPT_average_velocity_mass_flow_rate_available_horsepower_l2159_215946

/-- Average velocity of water flowing out of the sluice gate. -/
theorem average_velocity (g h₁ h₂ : ℝ) (h1_5m : h₁ = 5) (h2_5_4m : h₂ = 5.4) (g_9_81 : g = 9.81) :
    (1 / 2) * (Real.sqrt (2 * g * h₁) + Real.sqrt (2 * g * h₂)) = 10.1 :=
by
  sorry

/-- Mass flow rate of water per second when given average velocity and opening dimensions. -/
theorem mass_flow_rate (v A : ℝ) (v_10_1 : v = 10.1) (A_0_6 : A = 0.4 * 1.5) (rho : ℝ) (rho_1000 : rho = 1000) :
    ρ * A * v = 6060 :=
by
  sorry

/-- Available horsepower through turbines given mass flow rate and average velocity. -/
theorem available_horsepower (m v : ℝ) (m_6060 : m = 6060) (v_10_1 : v = 10.1 ) (hp : ℝ)
    (hp_735_5 : hp = 735.5 ) :
    (1 / 2) * m * v^2 / hp = 420 :=
by
  sorry

end NUMINAMATH_GPT_average_velocity_mass_flow_rate_available_horsepower_l2159_215946


namespace NUMINAMATH_GPT_anne_cleans_in_12_hours_l2159_215963

theorem anne_cleans_in_12_hours (B A C : ℝ) (h1 : B + A + C = 1/4)
    (h2 : B + 2 * A + 3 * C = 1/3) (h3 : B + C = 1/6) : 1 / A = 12 :=
by
    sorry

end NUMINAMATH_GPT_anne_cleans_in_12_hours_l2159_215963


namespace NUMINAMATH_GPT_positive_iff_sum_and_product_positive_l2159_215992

theorem positive_iff_sum_and_product_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) :=
by
  sorry

end NUMINAMATH_GPT_positive_iff_sum_and_product_positive_l2159_215992


namespace NUMINAMATH_GPT_solution_l2159_215909

noncomputable def problem_statement (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem solution (f : ℝ → ℝ) (h : problem_statement f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end NUMINAMATH_GPT_solution_l2159_215909


namespace NUMINAMATH_GPT_find_x_l2159_215927

theorem find_x 
  (b : ℤ) (h_b : b = 0) 
  (a z y x w : ℤ)
  (h1 : z + a = 1)
  (h2 : y + z + a = 0)
  (h3 : x + y + z = a)
  (h4 : w + x + y = z)
  :
  x = 2 :=
by {
    sorry
}    

end NUMINAMATH_GPT_find_x_l2159_215927


namespace NUMINAMATH_GPT_joe_first_lift_weight_l2159_215951

theorem joe_first_lift_weight (x y : ℕ) 
  (h1 : x + y = 900)
  (h2 : 2 * x = y + 300) :
  x = 400 :=
by
  sorry

end NUMINAMATH_GPT_joe_first_lift_weight_l2159_215951


namespace NUMINAMATH_GPT_bill_pays_sales_tax_correct_l2159_215979

def take_home_salary : ℝ := 40000
def property_tax : ℝ := 2000
def gross_salary : ℝ := 50000
def income_tax (gs : ℝ) : ℝ := 0.10 * gs
def total_taxes_paid (gs th : ℝ) : ℝ := gs - th
def sales_tax (ttp it pt : ℝ) : ℝ := ttp - it - pt

theorem bill_pays_sales_tax_correct :
  sales_tax
    (total_taxes_paid gross_salary take_home_salary)
    (income_tax gross_salary)
    property_tax = 3000 :=
by sorry

end NUMINAMATH_GPT_bill_pays_sales_tax_correct_l2159_215979


namespace NUMINAMATH_GPT_value_of_k_l2159_215999

theorem value_of_k (a b k : ℝ) (h1 : 2 * a = k) (h2 : 3 * b = k) (h3 : 2 * a + b = a * b) (h4 : k ≠ 1) : k = 8 := 
sorry

end NUMINAMATH_GPT_value_of_k_l2159_215999


namespace NUMINAMATH_GPT_mean_score_is_82_l2159_215952

noncomputable def mean_score 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : ℝ := 
  (M * m + A * a) / (m + a)

theorem mean_score_is_82 
  (M A m a : ℝ) 
  (hM : M = 90) 
  (hA : A = 75) 
  (hm : m / a = 4 / 5) : 
  mean_score M A m a hM hA hm = 82 := 
    sorry

end NUMINAMATH_GPT_mean_score_is_82_l2159_215952


namespace NUMINAMATH_GPT_min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l2159_215944

-- Part (a): For n = 12:
theorem min_sticks_to_break_for_square_12 : ∀ (n : ℕ), n = 12 → 
  (∃ (sticks : Finset ℕ), sticks.card = 12 ∧ sticks.sum id = 78 ∧ (¬ (78 % 4 = 0) → 
  ∃ (b : ℕ), b = 2)) := 
by sorry

-- Part (b): For n = 15:
theorem can_form_square_without_breaking_15 : ∀ (n : ℕ), n = 15 → 
  (∃ (sticks : Finset ℕ), sticks.card = 15 ∧ sticks.sum id = 120 ∧ (120 % 4 = 0)) :=
by sorry

end NUMINAMATH_GPT_min_sticks_to_break_for_square_12_can_form_square_without_breaking_15_l2159_215944


namespace NUMINAMATH_GPT_completing_the_square_l2159_215971

theorem completing_the_square :
  ∃ d, (∀ x: ℝ, (x^2 - 6 * x + 5 = 0) → ((x - 3)^2 = d)) ∧ d = 4 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_completing_the_square_l2159_215971


namespace NUMINAMATH_GPT_down_payment_amount_l2159_215931

-- Define the monthly savings per person
def monthly_savings_per_person : ℤ := 1500

-- Define the number of people
def number_of_people : ℤ := 2

-- Define the total monthly savings
def total_monthly_savings : ℤ := monthly_savings_per_person * number_of_people

-- Define the number of years they will save
def years_saving : ℤ := 3

-- Define the number of months in a year
def months_in_year : ℤ := 12

-- Define the total number of months
def total_months : ℤ := years_saving * months_in_year

-- Define the total savings needed for the down payment
def total_savings_needed : ℤ := total_monthly_savings * total_months

-- Prove that the total amount needed for the down payment is $108,000
theorem down_payment_amount : total_savings_needed = 108000 := by
  -- This part requires a proof, which we skip with sorry
  sorry

end NUMINAMATH_GPT_down_payment_amount_l2159_215931


namespace NUMINAMATH_GPT_find_constant_a_l2159_215997

theorem find_constant_a (x y a : ℝ) (h1 : (ax + 4 * y) / (x - 2 * y) = 13) (h2 : x / (2 * y) = 5 / 2) : a = 7 :=
sorry

end NUMINAMATH_GPT_find_constant_a_l2159_215997


namespace NUMINAMATH_GPT_fedya_initial_deposit_l2159_215970

theorem fedya_initial_deposit (n k : ℕ) (h₁ : k < 30) (h₂ : n * (100 - k) = 84700) : 
  n = 1100 :=
by
  sorry

end NUMINAMATH_GPT_fedya_initial_deposit_l2159_215970


namespace NUMINAMATH_GPT_last_four_digits_of_power_of_5_2017_l2159_215936

theorem last_four_digits_of_power_of_5_2017 :
  (5 ^ 2017 % 10000) = 3125 :=
by
  sorry

end NUMINAMATH_GPT_last_four_digits_of_power_of_5_2017_l2159_215936


namespace NUMINAMATH_GPT_decrease_percent_in_revenue_l2159_215972

theorem decrease_percent_in_revenue
  (T C : ℝ)
  (h_pos_T : 0 < T)
  (h_pos_C : 0 < C)
  (h_new_tax : T_new = 0.80 * T)
  (h_new_consumption : C_new = 1.20 * C) :
  let original_revenue := T * C
  let new_revenue := 0.80 * T * 1.20 * C
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 4 := by
sorry

end NUMINAMATH_GPT_decrease_percent_in_revenue_l2159_215972


namespace NUMINAMATH_GPT_hypotenuse_length_l2159_215981

noncomputable def side_lengths_to_hypotenuse (a b : ℝ) : ℝ := 
  Real.sqrt (a^2 + b^2)

theorem hypotenuse_length 
  (AB BC : ℝ) 
  (h1 : Real.sqrt (AB * BC) = 8) 
  (h2 : (1 / 2) * AB * BC = 48) :
  side_lengths_to_hypotenuse AB BC = 4 * Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2159_215981


namespace NUMINAMATH_GPT_percentage_chain_l2159_215994

theorem percentage_chain (n : ℝ) (h : n = 6000) : 0.1 * (0.3 * (0.5 * n)) = 90 := by
  sorry

end NUMINAMATH_GPT_percentage_chain_l2159_215994


namespace NUMINAMATH_GPT_cos_pi_minus_alpha_l2159_215995

theorem cos_pi_minus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π - α) = - (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_cos_pi_minus_alpha_l2159_215995


namespace NUMINAMATH_GPT_proof_f_2008_l2159_215973

theorem proof_f_2008 {f : ℝ → ℝ} 
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (3 * x + 1) = f (3 * (x + 1) + 1))
  (h3 : f (-1) = -1) : 
  f 2008 = 1 := 
by
  sorry

end NUMINAMATH_GPT_proof_f_2008_l2159_215973


namespace NUMINAMATH_GPT_hyperbola_foci_coords_l2159_215980

theorem hyperbola_foci_coords :
  let a := 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  ∀ x y : ℝ, 4 * y^2 - 25 * x^2 = 100 →
  (x = 0 ∧ (y = c ∨ y = -c)) := by
  intros a b c x y h
  have h1 : 4 * y^2 = 100 + 25 * x^2 := by linarith
  have h2 : y^2 = 25 + 25/4 * x^2 := by linarith
  have h3 : x = 0 := by sorry
  have h4 : y = c ∨ y = -c := by sorry
  exact ⟨h3, h4⟩

end NUMINAMATH_GPT_hyperbola_foci_coords_l2159_215980


namespace NUMINAMATH_GPT_find_m_l2159_215974

theorem find_m (x₁ x₂ y₁ y₂ : ℝ) (m : ℝ) 
  (h_parabola_A : y₁ = 2 * x₁^2) 
  (h_parabola_B : y₂ = 2 * x₂^2) 
  (h_symmetry : y₂ - y₁ = 2 * (x₂^2 - x₁^2)) 
  (h_product : x₁ * x₂ = -1/2) 
  (h_midpoint : (y₂ + y₁) / 2 = (x₂ + x₁) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l2159_215974


namespace NUMINAMATH_GPT_distance_between_trees_l2159_215934

theorem distance_between_trees
  (yard_length : ℕ)
  (num_trees : ℕ)
  (h_yard_length : yard_length = 441)
  (h_num_trees : num_trees = 22) :
  (yard_length / (num_trees - 1)) = 21 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l2159_215934


namespace NUMINAMATH_GPT_option_C_sets_same_l2159_215901

-- Define the sets for each option
def option_A_set_M : Set (ℕ × ℕ) := {(3, 2)}
def option_A_set_N : Set (ℕ × ℕ) := {(2, 3)}

def option_B_set_M : Set (ℕ × ℕ) := {p | p.1 + p.2 = 1}
def option_B_set_N : Set ℕ := { y | ∃ x, x + y = 1 }

def option_C_set_M : Set ℕ := {4, 5}
def option_C_set_N : Set ℕ := {5, 4}

def option_D_set_M : Set ℕ := {1, 2}
def option_D_set_N : Set (ℕ × ℕ) := {(1, 2)}

-- Prove that option C sets represent the same set
theorem option_C_sets_same : option_C_set_M = option_C_set_N := by
  sorry

end NUMINAMATH_GPT_option_C_sets_same_l2159_215901


namespace NUMINAMATH_GPT_joe_paint_usage_l2159_215959

theorem joe_paint_usage :
  ∀ (total_paint initial_remaining_paint final_remaining_paint paint_first_week paint_second_week total_used : ℕ),
  total_paint = 360 →
  initial_remaining_paint = total_paint - paint_first_week →
  final_remaining_paint = initial_remaining_paint - paint_second_week →
  paint_first_week = (2 * total_paint) / 3 →
  paint_second_week = (1 * initial_remaining_paint) / 5 →
  total_used = paint_first_week + paint_second_week →
  total_used = 264 :=
by
  sorry

end NUMINAMATH_GPT_joe_paint_usage_l2159_215959


namespace NUMINAMATH_GPT_product_of_consecutive_integers_sqrt_73_l2159_215941

theorem product_of_consecutive_integers_sqrt_73 : 
  ∃ (m n : ℕ), (m < n) ∧ ∃ (j k : ℕ), (j = 8) ∧ (k = 9) ∧ (m = j) ∧ (n = k) ∧ (m * n = 72) := by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_sqrt_73_l2159_215941


namespace NUMINAMATH_GPT_relay_race_total_time_l2159_215975

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 10
  let t3 := t2 - 15
  let t4 := t1 - 25
  t1 + t2 + t3 + t4 = 200 := by
    sorry

end NUMINAMATH_GPT_relay_race_total_time_l2159_215975


namespace NUMINAMATH_GPT_remainder_of_f_100_div_100_l2159_215908

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem remainder_of_f_100_div_100 : 
  (pascal_triangle_row_sum 100) % 100 = 74 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_f_100_div_100_l2159_215908


namespace NUMINAMATH_GPT_average_of_remaining_numbers_l2159_215985

theorem average_of_remaining_numbers 
  (numbers : List ℝ)
  (h_len : numbers.length = 15)
  (h_avg : (numbers.sum / 15) = 100)
  (h_remove : [80, 90, 95] ⊆ numbers) :
  ((numbers.sum - 80 - 90 - 95) / 12) = (1235 / 12) :=
sorry

end NUMINAMATH_GPT_average_of_remaining_numbers_l2159_215985


namespace NUMINAMATH_GPT_cubic_inequality_l2159_215945

theorem cubic_inequality (a : ℝ) (h : a ≠ -1) : 
  (1 + a^3) / (1 + a)^3 ≥ 1 / 4 :=
by sorry

end NUMINAMATH_GPT_cubic_inequality_l2159_215945


namespace NUMINAMATH_GPT_length_of_shorter_angle_trisector_l2159_215943

theorem length_of_shorter_angle_trisector (BC AC : ℝ) (h1 : BC = 3) (h2 : AC = 4) :
  let AB := Real.sqrt (BC^2 + AC^2)
  let x := 2 * (12 / (4 * Real.sqrt 3 + 3))
  let PC := 2 * x
  AB = 5 ∧ PC = (32 * Real.sqrt 3 - 24) / 13 :=
by
  sorry

end NUMINAMATH_GPT_length_of_shorter_angle_trisector_l2159_215943


namespace NUMINAMATH_GPT_polygon_sides_equation_l2159_215922

theorem polygon_sides_equation (n : ℕ) 
  (h1 : (n-2) * 180 = 4 * 360) : n = 10 := 
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_equation_l2159_215922


namespace NUMINAMATH_GPT_certain_number_is_213_l2159_215947

theorem certain_number_is_213 (n : ℕ) (h : n * 16 = 3408) : n = 213 :=
sorry

end NUMINAMATH_GPT_certain_number_is_213_l2159_215947


namespace NUMINAMATH_GPT_fraction_problem_l2159_215940

theorem fraction_problem :
  (1 / 4 + 3 / 8) - 1 / 8 = 1 / 2 :=
by
  -- The proof steps are skipped
  sorry

end NUMINAMATH_GPT_fraction_problem_l2159_215940


namespace NUMINAMATH_GPT_dealer_can_determine_values_l2159_215993

def card_value_determined (a : Fin 100 → Fin 100) : Prop :=
  (∀ i j : Fin 100, i > j → a i > a j) ∧ (a 0 > a 99) ∧
  (∀ k : Fin 100, a k = k + 1)

theorem dealer_can_determine_values :
  ∃ (messages : Fin 100 → Fin 100), card_value_determined messages :=
sorry

end NUMINAMATH_GPT_dealer_can_determine_values_l2159_215993


namespace NUMINAMATH_GPT_candy_bar_cost_l2159_215916

-- Definitions of conditions
def soft_drink_cost : ℕ := 4
def num_soft_drinks : ℕ := 2
def num_candy_bars : ℕ := 5
def total_cost : ℕ := 28

-- Proof Statement
theorem candy_bar_cost : (total_cost - num_soft_drinks * soft_drink_cost) / num_candy_bars = 4 := by
  sorry

end NUMINAMATH_GPT_candy_bar_cost_l2159_215916


namespace NUMINAMATH_GPT_joe_speed_first_part_l2159_215902

theorem joe_speed_first_part (v : ℝ) :
  let d1 := 420 -- distance of the first part in miles
  let d2 := 120 -- distance of the second part in miles
  let v2 := 40  -- speed during the second part in miles per hour
  let d_total := d1 + d2 -- total distance
  let avg_speed := 54 -- average speed in miles per hour
  let t1 := d1 / v -- time for the first part
  let t2 := d2 / v2 -- time for the second part
  let t_total := t1 + t2 -- total time
  (d_total / t_total) = avg_speed -> v = 60 :=
by
  intros
  sorry

end NUMINAMATH_GPT_joe_speed_first_part_l2159_215902


namespace NUMINAMATH_GPT_division_remainder_l2159_215978

theorem division_remainder (dividend divisor quotient remainder : ℕ)
  (h₁ : dividend = 689)
  (h₂ : divisor = 36)
  (h₃ : quotient = 19)
  (h₄ : dividend = divisor * quotient + remainder) :
  remainder = 5 :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l2159_215978


namespace NUMINAMATH_GPT_probability_of_xiao_li_l2159_215905

def total_students : ℕ := 5
def xiao_li : ℕ := 1

noncomputable def probability_xiao_li_chosen : ℚ :=
  (xiao_li : ℚ) / (total_students : ℚ)

theorem probability_of_xiao_li : probability_xiao_li_chosen = 1 / 5 :=
sorry

end NUMINAMATH_GPT_probability_of_xiao_li_l2159_215905


namespace NUMINAMATH_GPT_sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l2159_215949

-- Conditions
variable (x : ℝ) (h₀ : 0 < x) (h₁ : x + 1 / x = 150)

-- Statement to prove
theorem sqrt_x_plus_inv_sqrt_x_eq_sqrt_152 : (Real.sqrt x + Real.sqrt (1 / x) = Real.sqrt 152) := 
sorry -- Proof not needed, skip with sorry

end NUMINAMATH_GPT_sqrt_x_plus_inv_sqrt_x_eq_sqrt_152_l2159_215949


namespace NUMINAMATH_GPT_prob_triangle_inequality_l2159_215991

theorem prob_triangle_inequality (x y z : ℕ) (h1 : 1 ≤ x ∧ x ≤ 6) (h2 : 1 ≤ y ∧ y ≤ 6) (h3 : 1 ≤ z ∧ z ≤ 6) : 
  (∃ (p : ℚ), p = 37 / 72) := 
sorry

end NUMINAMATH_GPT_prob_triangle_inequality_l2159_215991


namespace NUMINAMATH_GPT_sum_of_decimals_as_fraction_l2159_215911

theorem sum_of_decimals_as_fraction :
  let x := (0 : ℝ) + 1 / 3;
  let y := (0 : ℝ) + 2 / 3;
  let z := (0 : ℝ) + 2 / 5;
  x + y + z = 7 / 5 :=
by
  let x := (0 : ℝ) + 1 / 3
  let y := (0 : ℝ) + 2 / 3
  let z := (0 : ℝ) + 2 / 5
  show x + y + z = 7 / 5
  sorry

end NUMINAMATH_GPT_sum_of_decimals_as_fraction_l2159_215911


namespace NUMINAMATH_GPT_percentage_design_black_is_57_l2159_215990

noncomputable def circleRadius (n : ℕ) : ℝ :=
  3 * (n + 1)

noncomputable def circleArea (n : ℕ) : ℝ :=
  Real.pi * (circleRadius n) ^ 2

noncomputable def totalArea : ℝ :=
  circleArea 6

noncomputable def blackAreas : ℝ :=
  circleArea 0 + (circleArea 2 - circleArea 1) +
  (circleArea 4 - circleArea 3) +
  (circleArea 6 - circleArea 5)

noncomputable def percentageBlack : ℝ :=
  (blackAreas / totalArea) * 100

theorem percentage_design_black_is_57 :
  percentageBlack = 57 := 
by
  sorry

end NUMINAMATH_GPT_percentage_design_black_is_57_l2159_215990


namespace NUMINAMATH_GPT_geometric_seq_reciprocal_sum_l2159_215996

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop := ∀ n, a (n + 1) = a n * r

theorem geometric_seq_reciprocal_sum
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : geometric_sequence a r)
  (h1 : a 2 * a 5 = -3/4)
  (h2 : a 2 + a 3 + a 4 + a 5 = 5/4) :
  (1 / a 2) + (1 / a 3) + (1 / a 4) + (1 / a 5) = -5/3 := sorry

end NUMINAMATH_GPT_geometric_seq_reciprocal_sum_l2159_215996


namespace NUMINAMATH_GPT_required_run_rate_l2159_215982

theorem required_run_rate (run_rate_first_10_overs : ℝ) (target_runs total_overs first_overs : ℕ) :
  run_rate_first_10_overs = 4.2 ∧ target_runs = 282 ∧ total_overs = 50 ∧ first_overs = 10 →
  (target_runs - run_rate_first_10_overs * first_overs) / (total_overs - first_overs) = 6 :=
by
  sorry

end NUMINAMATH_GPT_required_run_rate_l2159_215982


namespace NUMINAMATH_GPT_train_length_proof_l2159_215913

-- Defining the conditions
def speed_kmph : ℕ := 72
def platform_length : ℕ := 250  -- in meters
def time_seconds : ℕ := 26

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℕ) : ℕ := (v * 1000) / 3600

-- The main goal: the length of the train
def train_length (speed_kmph : ℕ) (platform_length : ℕ) (time_seconds : ℕ) : ℕ :=
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_seconds
  total_distance - platform_length

theorem train_length_proof : train_length speed_kmph platform_length time_seconds = 270 := 
by 
  unfold train_length kmph_to_mps
  sorry

end NUMINAMATH_GPT_train_length_proof_l2159_215913


namespace NUMINAMATH_GPT_smallest_number_of_marbles_l2159_215900

theorem smallest_number_of_marbles (M : ℕ) (h1 : M ≡ 2 [MOD 5]) (h2 : M ≡ 2 [MOD 6]) (h3 : M ≡ 2 [MOD 7]) (h4 : 1 < M) : M = 212 :=
by sorry

end NUMINAMATH_GPT_smallest_number_of_marbles_l2159_215900


namespace NUMINAMATH_GPT_min_value_fraction_condition_l2159_215964

noncomputable def minValue (a b : ℝ) := 1 / (2 * a) + a / (b + 1)

theorem min_value_fraction_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  minValue a b = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_condition_l2159_215964


namespace NUMINAMATH_GPT_parallelogram_area_l2159_215955

theorem parallelogram_area (base height : ℝ) (h_base : base = 25) (h_height : height = 15) :
  base * height = 375 :=
by
  subst h_base
  subst h_height
  sorry

end NUMINAMATH_GPT_parallelogram_area_l2159_215955


namespace NUMINAMATH_GPT_anne_wandering_time_l2159_215984

theorem anne_wandering_time (distance speed : ℝ) (h_dist : distance = 3.0) (h_speed : speed = 2.0) : 
  distance / speed = 1.5 :=
by
  rw [h_dist, h_speed]
  norm_num

end NUMINAMATH_GPT_anne_wandering_time_l2159_215984


namespace NUMINAMATH_GPT_energy_loss_per_bounce_l2159_215917

theorem energy_loss_per_bounce
  (h : ℝ) (t : ℝ) (g : ℝ) (y : ℝ)
  (h_conds : h = 0.2)
  (t_conds : t = 18)
  (g_conds : g = 10)
  (model : t = Real.sqrt (2 * h / g) + 2 * (Real.sqrt (2 * h * y / g)) / (1 - Real.sqrt y)) :
  1 - y = 0.36 :=
by
  sorry

end NUMINAMATH_GPT_energy_loss_per_bounce_l2159_215917


namespace NUMINAMATH_GPT_point_not_in_region_l2159_215914

theorem point_not_in_region (A B C D : ℝ × ℝ) :
  (A = (0, 0) ∧ 3 * A.1 + 2 * A.2 < 6) ∧
  (B = (1, 1) ∧ 3 * B.1 + 2 * B.2 < 6) ∧
  (C = (0, 2) ∧ 3 * C.1 + 2 * C.2 < 6) ∧
  (D = (2, 0) ∧ ¬ ( 3 * D.1 + 2 * D.2 < 6 )) :=
by {
  sorry
}

end NUMINAMATH_GPT_point_not_in_region_l2159_215914


namespace NUMINAMATH_GPT_find_pqr_abs_l2159_215907

variables {p q r : ℝ}

-- Conditions as hypotheses
def conditions (p q r : ℝ) : Prop :=
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
  (p^2 + 2/q = q^2 + 2/r) ∧ (q^2 + 2/r = r^2 + 2/p)

-- Statement of the theorem
theorem find_pqr_abs (h : conditions p q r) : |p * q * r| = 2 :=
sorry

end NUMINAMATH_GPT_find_pqr_abs_l2159_215907


namespace NUMINAMATH_GPT_tetrahedron_inscribed_in_pyramid_edge_length_l2159_215954

noncomputable def edge_length_of_tetrahedron := (Real.sqrt 2) / 2

theorem tetrahedron_inscribed_in_pyramid_edge_length :
  let A := (0,0,0)
  let B := (1,0,0)
  let C := (1,1,0)
  let D := (0,1,0)
  let E := (0.5, 0.5, 1)
  let v₁ := (0.5, 0, 0)
  let v₂ := (1, 0.5, 0)
  let v₃ := (0, 0.5, 0)
  dist (v₁ : ℝ × ℝ × ℝ) v₂ = edge_length_of_tetrahedron ∧
  dist v₂ v₃ = edge_length_of_tetrahedron ∧
  dist v₃ v₁ = edge_length_of_tetrahedron ∧
  dist E v₁ = dist E v₂ ∧
  dist E v₂ = dist E v₃ :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_inscribed_in_pyramid_edge_length_l2159_215954


namespace NUMINAMATH_GPT_solve_for_x_l2159_215969

theorem solve_for_x (x : ℚ) : 
  5*x + 9*x = 450 - 10*(x - 5) -> x = 125/6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2159_215969


namespace NUMINAMATH_GPT_max_true_statements_l2159_215926

theorem max_true_statements (x : ℝ) :
  (∀ x, -- given the conditions
    (0 < x^2 ∧ x^2 < 1) →
    (x^2 > 1) →
    (-1 < x ∧ x < 0) →
    (0 < x ∧ x < 1) →
    (0 < x - x^2 ∧ x - x^2 < 1)) →
  -- Prove the maximum number of these statements that can be true is 3
  (∃ (count : ℕ), count = 3) :=
sorry

end NUMINAMATH_GPT_max_true_statements_l2159_215926


namespace NUMINAMATH_GPT_total_cost_of_square_park_l2159_215912

-- Define the cost per side and number of sides
def cost_per_side : ℕ := 56
def sides_of_square : ℕ := 4

-- The total cost of fencing the park
def total_cost_of_fencing (cost_per_side : ℕ) (sides_of_square : ℕ) : ℕ := cost_per_side * sides_of_square

-- The statement we need to prove
theorem total_cost_of_square_park : total_cost_of_fencing cost_per_side sides_of_square = 224 :=
by sorry

end NUMINAMATH_GPT_total_cost_of_square_park_l2159_215912


namespace NUMINAMATH_GPT_natural_pairs_prime_l2159_215928

theorem natural_pairs_prime (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) (h_eq : p = xy^2 / (x + y))
  : (x, y) = (2, 2) ∨ (x, y) = (6, 2) :=
sorry

end NUMINAMATH_GPT_natural_pairs_prime_l2159_215928


namespace NUMINAMATH_GPT_factorize_polynomial_l2159_215956

theorem factorize_polynomial (x : ℝ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := 
sorry

end NUMINAMATH_GPT_factorize_polynomial_l2159_215956


namespace NUMINAMATH_GPT_max_S_n_value_arithmetic_sequence_l2159_215968

-- Definitions and conditions
def S_n (n : ℕ) : ℤ := 3 * n - n^2

def a_n (n : ℕ) : ℤ := 
if n = 0 then 0 else S_n n - S_n (n - 1)

-- Statement of the first part of the proof problem
theorem max_S_n_value (n : ℕ) (h : n = 1 ∨ n = 2) : S_n n = 2 :=
sorry

-- Statement of the second part of the proof problem
theorem arithmetic_sequence :
  ∀ n : ℕ, n ≥ 1 → a_n (n + 1) - a_n n = -2 :=
sorry

end NUMINAMATH_GPT_max_S_n_value_arithmetic_sequence_l2159_215968


namespace NUMINAMATH_GPT_sharpening_cost_l2159_215958

theorem sharpening_cost
  (trees_chopped : ℕ)
  (trees_per_sharpening : ℕ)
  (total_cost : ℕ)
  (min_trees_chopped : trees_chopped ≥ 91)
  (trees_per_sharpening_eq : trees_per_sharpening = 13)
  (total_cost_eq : total_cost = 35) :
  total_cost / (trees_chopped / trees_per_sharpening) = 5 := by
  sorry

end NUMINAMATH_GPT_sharpening_cost_l2159_215958


namespace NUMINAMATH_GPT_problem1_problem2_l2159_215950

-- Problem 1 Lean statement
theorem problem1 (x y : ℝ) (hx : x ≠ 1) (hx' : x ≠ -1) (hy : y ≠ 0) :
    (x^2 - 1) / y / ((x + 1) / y^2) = y * (x - 1) :=
sorry

-- Problem 2 Lean statement
theorem problem2 (m n : ℝ) (hm1 : m ≠ n) (hm2 : m ≠ -n) :
    m / (m + n) + n / (m - n) - 2 * m^2 / (m^2 - n^2) = -1 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l2159_215950


namespace NUMINAMATH_GPT_longest_side_AB_l2159_215953

-- Definitions of angles in the quadrilateral
def angle_ABC := 65
def angle_BCD := 70
def angle_CDA := 60

/-- In a quadrilateral ABCD with angles as specified, prove that AB is the longest side. -/
theorem longest_side_AB (AB BC CD DA : ℝ) : 
  (angle_ABC = 65 ∧ angle_BCD = 70 ∧ angle_CDA = 60) → 
  AB > DA ∧ AB > BC ∧ AB > CD :=
by
  intros h
  sorry

end NUMINAMATH_GPT_longest_side_AB_l2159_215953


namespace NUMINAMATH_GPT_necessary_condition_l2159_215987

variable (P Q : Prop)

/-- If the presence of the dragon city's flying general implies that
    the horses of the Hu people will not cross the Yin Mountains,
    then "not letting the horses of the Hu people cross the Yin Mountains"
    is a necessary condition for the presence of the dragon city's flying general. -/
theorem necessary_condition (h : P → Q) : ¬Q → ¬P :=
by sorry

end NUMINAMATH_GPT_necessary_condition_l2159_215987


namespace NUMINAMATH_GPT_train_crossing_time_l2159_215998

def length_of_train : ℕ := 120
def speed_of_train_kmph : ℕ := 54
def length_of_bridge : ℕ := 660

def speed_of_train_mps : ℕ := speed_of_train_kmph * 1000 / 3600
def total_distance : ℕ := length_of_train + length_of_bridge
def time_to_cross_bridge : ℕ := total_distance / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross_bridge = 52 :=
sorry

end NUMINAMATH_GPT_train_crossing_time_l2159_215998


namespace NUMINAMATH_GPT_percentage_of_students_choose_harvard_l2159_215910

theorem percentage_of_students_choose_harvard
  (total_applicants : ℕ)
  (acceptance_rate : ℝ)
  (students_attend_harvard : ℕ)
  (students_attend_other : ℝ)
  (percentage_attended_harvard : ℝ) :
  total_applicants = 20000 →
  acceptance_rate = 0.05 →
  students_attend_harvard = 900 →
  students_attend_other = 0.10 →
  percentage_attended_harvard = ((students_attend_harvard / (total_applicants * acceptance_rate)) * 100) →
  percentage_attended_harvard = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_percentage_of_students_choose_harvard_l2159_215910


namespace NUMINAMATH_GPT_ounces_per_container_l2159_215918

def weight_pounds : ℝ := 3.75
def num_containers : ℕ := 4
def pound_to_ounces : ℕ := 16

theorem ounces_per_container :
  (weight_pounds * pound_to_ounces) / num_containers = 15 :=
by
  sorry

end NUMINAMATH_GPT_ounces_per_container_l2159_215918


namespace NUMINAMATH_GPT_complement_union_of_M_and_N_l2159_215948

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

theorem complement_union_of_M_and_N :
  (U \ (M ∪ N)) = {5} :=
by sorry

end NUMINAMATH_GPT_complement_union_of_M_and_N_l2159_215948


namespace NUMINAMATH_GPT_license_plate_count_is_correct_l2159_215961

/-- Define the number of consonants in the English alphabet --/
def num_consonants : Nat := 20

/-- Define the number of possibilities for 'A' --/
def num_A : Nat := 1

/-- Define the number of even digits --/
def num_even_digits : Nat := 5

/-- Define the total number of valid four-character license plates --/
def total_license_plate_count : Nat :=
  num_consonants * num_A * num_consonants * num_even_digits

/-- Theorem stating that the total number of license plates is 2000 --/
theorem license_plate_count_is_correct : 
  total_license_plate_count = 2000 :=
  by
    -- The proof is omitted
    sorry

end NUMINAMATH_GPT_license_plate_count_is_correct_l2159_215961


namespace NUMINAMATH_GPT_fuel_cost_per_liter_l2159_215967

def service_cost_per_vehicle : ℝ := 2.20
def num_minivans : ℕ := 3
def num_trucks : ℕ := 2
def total_cost : ℝ := 347.7
def mini_van_tank_capacity : ℝ := 65
def truck_tank_increase : ℝ := 1.2
def truck_tank_capacity : ℝ := mini_van_tank_capacity * (1 + truck_tank_increase)

theorem fuel_cost_per_liter : 
  let total_service_cost := (num_minivans + num_trucks) * service_cost_per_vehicle
  let total_capacity_minivans := num_minivans * mini_van_tank_capacity
  let total_capacity_trucks := num_trucks * truck_tank_capacity
  let total_fuel_capacity := total_capacity_minivans + total_capacity_trucks
  let fuel_cost := total_cost - total_service_cost
  let cost_per_liter := fuel_cost / total_fuel_capacity
  cost_per_liter = 0.70 := 
  sorry

end NUMINAMATH_GPT_fuel_cost_per_liter_l2159_215967


namespace NUMINAMATH_GPT_sum_of_squares_mul_l2159_215966

theorem sum_of_squares_mul (a b c d : ℝ) :
(a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_mul_l2159_215966


namespace NUMINAMATH_GPT_three_solutions_exists_l2159_215924

theorem three_solutions_exists (n : ℕ) (h_pos : 0 < n) (h_sol : ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = n) :
  ∃ x1 y1 x2 y2 x3 y3 : ℤ, (x1^3 - 3 * x1 * y1^2 + y1^3 = n) ∧ (x2^3 - 3 * x2 * y2^2 + y2^3 = n) ∧ (x3^3 - 3 * x3 * y3^2 + y3^3 = n) ∧ (x1, y1) ≠ (x2, y2) ∧ (x2, y2) ≠ (x3, y3) ∧ (x1, y1) ≠ (x3, y3) :=
by
  sorry

end NUMINAMATH_GPT_three_solutions_exists_l2159_215924


namespace NUMINAMATH_GPT_total_fencing_l2159_215904

def playground_side_length : ℕ := 27
def garden_length : ℕ := 12
def garden_width : ℕ := 9

def perimeter_square (side : ℕ) : ℕ := 4 * side
def perimeter_rectangle (length width : ℕ) : ℕ := 2 * length + 2 * width

theorem total_fencing (side playground_side_length : ℕ) (garden_length garden_width : ℕ) :
  perimeter_square playground_side_length + perimeter_rectangle garden_length garden_width = 150 :=
by
  sorry

end NUMINAMATH_GPT_total_fencing_l2159_215904


namespace NUMINAMATH_GPT_average_cookies_per_package_is_fifteen_l2159_215960

def average_cookies_count (cookies : List ℕ) (n : ℕ) : ℕ :=
  (cookies.sum / n : ℕ)

theorem average_cookies_per_package_is_fifteen :
  average_cookies_count [5, 12, 18, 20, 21] 5 = 15 :=
by
  sorry

end NUMINAMATH_GPT_average_cookies_per_package_is_fifteen_l2159_215960


namespace NUMINAMATH_GPT_reverse_digits_difference_l2159_215935

theorem reverse_digits_difference (q r : ℕ) (x y : ℕ) 
  (hq : q = 10 * x + y)
  (hr : r = 10 * y + x)
  (hq_r_pos : q > r)
  (h_diff_lt_20 : q - r < 20)
  (h_max_diff : q - r = 18) :
  x - y = 2 := 
by
  sorry

end NUMINAMATH_GPT_reverse_digits_difference_l2159_215935


namespace NUMINAMATH_GPT_solve_inequality_l2159_215930

theorem solve_inequality (a x : ℝ) (ha : a ≠ 0) :
  (a > 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 2 * a ∨ x > 3 * a))) ∧
  (a < 0 → (x^2 - 5 * a * x + 6 * a^2 > 0 ↔ (x < 3 * a ∨ x > 2 * a))) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2159_215930


namespace NUMINAMATH_GPT_contrapositive_false_of_implication_false_l2159_215939

variable (p q : Prop)

-- The statement we need to prove: If "if p then q" is false, 
-- then "if not q then not p" must be false.
theorem contrapositive_false_of_implication_false (h : ¬ (p → q)) : ¬ (¬ q → ¬ p) :=
by
sorry

end NUMINAMATH_GPT_contrapositive_false_of_implication_false_l2159_215939


namespace NUMINAMATH_GPT_sufficient_and_necessary_cond_l2159_215923

theorem sufficient_and_necessary_cond (x : ℝ) : |x| > 2 ↔ (x > 2) :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_cond_l2159_215923


namespace NUMINAMATH_GPT_determine_value_of_x_l2159_215938

theorem determine_value_of_x (x y : ℝ) (h1 : x ≠ 0) (h2 : x / 3 = y^2) (h3 : x / 2 = 6 * y) : x = 48 :=
by
  sorry

end NUMINAMATH_GPT_determine_value_of_x_l2159_215938


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2159_215915

-- Definitions of sets A and B
def A : Set ℤ := {1, 0, 3}
def B : Set ℤ := {-1, 1, 2, 3}

-- Statement of the theorem
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2159_215915


namespace NUMINAMATH_GPT_number_of_books_l2159_215932

-- Define the given conditions as variables
def movies_in_series : Nat := 62
def books_read : Nat := 4
def books_yet_to_read : Nat := 15

-- State the proposition we need to prove
theorem number_of_books : (books_read + books_yet_to_read) = 19 :=
by
  sorry

end NUMINAMATH_GPT_number_of_books_l2159_215932


namespace NUMINAMATH_GPT_speed_of_second_train_is_16_l2159_215920

def speed_second_train (v : ℝ) : Prop :=
  ∃ t : ℝ, 
    (20 * t = v * t + 70) ∧ -- Condition: the first train traveled 70 km more than the second train
    (20 * t + v * t = 630)  -- Condition: total distance between stations

theorem speed_of_second_train_is_16 : speed_second_train 16 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_is_16_l2159_215920


namespace NUMINAMATH_GPT_candy_eaten_l2159_215929

theorem candy_eaten 
  {initial_pieces remaining_pieces eaten_pieces : ℕ} 
  (h₁ : initial_pieces = 12) 
  (h₂ : remaining_pieces = 3) 
  (h₃ : eaten_pieces = initial_pieces - remaining_pieces) 
  : eaten_pieces = 9 := 
by 
  sorry

end NUMINAMATH_GPT_candy_eaten_l2159_215929


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2159_215983

theorem arithmetic_sequence_sum (c d : ℕ) 
  (h1 : ∀ (a1 a2 a3 a4 a5 a6 : ℕ), a1 = 3 → a2 = 10 → a3 = 17 → a6 = 38 → (a2 - a1 = a3 - a2) → (a3 - a2 = c - a3) → (c - a3 = d - c) → (d - c = a6 - d)) : 
  c + d = 55 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2159_215983


namespace NUMINAMATH_GPT_each_person_gets_9_wings_l2159_215925

noncomputable def chicken_wings_per_person (initial_wings : ℕ) (additional_wings : ℕ) (friends : ℕ) : ℕ :=
  (initial_wings + additional_wings) / friends

theorem each_person_gets_9_wings :
  chicken_wings_per_person 20 25 5 = 9 :=
by
  sorry

end NUMINAMATH_GPT_each_person_gets_9_wings_l2159_215925


namespace NUMINAMATH_GPT_least_number_to_add_l2159_215919

theorem least_number_to_add (a : ℕ) (p q r : ℕ) (h : a = 1076) (hp : p = 41) (hq : q = 59) (hr : r = 67) :
  ∃ k : ℕ, k = 171011 ∧ (a + k) % (lcm p (lcm q r)) = 0 :=
sorry

end NUMINAMATH_GPT_least_number_to_add_l2159_215919


namespace NUMINAMATH_GPT_positive_difference_solutions_abs_l2159_215933

theorem positive_difference_solutions_abs (x1 x2 : ℝ) 
  (h1 : 2 * x1 - 3 = 18 ∨ 2 * x1 - 3 = -18) 
  (h2 : 2 * x2 - 3 = 18 ∨ 2 * x2 - 3 = -18) : 
  |x1 - x2| = 18 :=
sorry

end NUMINAMATH_GPT_positive_difference_solutions_abs_l2159_215933


namespace NUMINAMATH_GPT_min_trips_is_157_l2159_215965

theorem min_trips_is_157 :
  ∃ x y : ℕ, 31 * x + 32 * y = 5000 ∧ x + y = 157 :=
sorry

end NUMINAMATH_GPT_min_trips_is_157_l2159_215965


namespace NUMINAMATH_GPT_distance_travelled_l2159_215937

theorem distance_travelled (speed time distance : ℕ) 
  (h1 : speed = 25)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 125 :=
by
  sorry

end NUMINAMATH_GPT_distance_travelled_l2159_215937


namespace NUMINAMATH_GPT_negation_of_all_have_trap_consumption_l2159_215986

-- Definitions for the conditions
def domestic_mobile_phone : Type := sorry

def has_trap_consumption (phone : domestic_mobile_phone) : Prop := sorry

def all_have_trap_consumption : Prop := ∀ phone : domestic_mobile_phone, has_trap_consumption phone

-- Statement of the problem
theorem negation_of_all_have_trap_consumption :
  ¬ all_have_trap_consumption ↔ ∃ phone : domestic_mobile_phone, ¬ has_trap_consumption phone :=
sorry

end NUMINAMATH_GPT_negation_of_all_have_trap_consumption_l2159_215986


namespace NUMINAMATH_GPT_triangle_dimensions_l2159_215989

theorem triangle_dimensions (a b c : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : a = 2 * c) (h4 : b - 2 = c) (h5 : 2 * a / 3 = b) :
  a = 12 ∧ b = 8 ∧ c = 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_dimensions_l2159_215989


namespace NUMINAMATH_GPT_most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l2159_215976

noncomputable def most_entries_with_80_yuan : Nat :=
let cost_a := 120
let cost_b := 60
let cost_c := 40
let entry_b := 2
let entry_c := 3
let budget := 80
let entries_b := (budget - cost_b) / entry_b
let entries_c := (budget - cost_c) / entry_c
let entries_no_pass := budget / 10
if cost_a <= budget then 
  0
else
  max entries_b (max entries_c entries_no_pass)

theorem most_entries_with_80_yuan_is_c_pass : most_entries_with_80_yuan = 13 :=
by
  sorry

noncomputable def is_pass_a_cost_effective (x : Nat) : Prop :=
let cost_a := 120
let cost_b_entries := 60 + 2 * x
let cost_c_entries := 40 + 3 * x
let cost_no_pass := 10 * x
x > 30 → cost_a < cost_b_entries ∧ cost_a < cost_c_entries ∧ cost_a < cost_no_pass

theorem pass_a_is_cost_effective_after_30_entries : ∀ x : Nat, is_pass_a_cost_effective x :=
by
  sorry

end NUMINAMATH_GPT_most_entries_with_80_yuan_is_c_pass_pass_a_is_cost_effective_after_30_entries_l2159_215976


namespace NUMINAMATH_GPT_ratio_perimeters_of_squares_l2159_215957

theorem ratio_perimeters_of_squares (a b : ℝ) (h_diag : (a * Real.sqrt 2) / (b * Real.sqrt 2) = 2.5) : (4 * a) / (4 * b) = 10 :=
by
  sorry

end NUMINAMATH_GPT_ratio_perimeters_of_squares_l2159_215957


namespace NUMINAMATH_GPT_square_of_any_real_number_not_always_greater_than_zero_l2159_215921

theorem square_of_any_real_number_not_always_greater_than_zero (a : ℝ) : 
    (∀ x : ℝ, x^2 ≥ 0) ∧ (exists x : ℝ, x = 0 ∧ x^2 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_square_of_any_real_number_not_always_greater_than_zero_l2159_215921


namespace NUMINAMATH_GPT_digit_205_of_14_div_360_l2159_215903

noncomputable def decimal_expansion_of_fraction (n d : ℕ) : ℕ → ℕ := sorry

theorem digit_205_of_14_div_360 : 
  decimal_expansion_of_fraction 14 360 205 = 8 :=
sorry

end NUMINAMATH_GPT_digit_205_of_14_div_360_l2159_215903


namespace NUMINAMATH_GPT_nail_pierces_one_cardboard_only_l2159_215906

/--
Seryozha cut out two identical figures from cardboard. He placed them overlapping
at the bottom of a rectangular box. The bottom turned out to be completely covered. 
A nail was driven into the center of the bottom. Prove that it is possible for the 
nail to pierce one cardboard piece without piercing the other.
-/
theorem nail_pierces_one_cardboard_only 
  (identical_cardboards : Prop)
  (overlapping : Prop)
  (fully_covered_bottom : Prop)
  (nail_center : Prop) 
  : ∃ (layout : Prop), layout ∧ nail_center → nail_pierces_one :=
sorry

end NUMINAMATH_GPT_nail_pierces_one_cardboard_only_l2159_215906


namespace NUMINAMATH_GPT_sequence_an_expression_l2159_215942

theorem sequence_an_expression (a : ℕ → ℕ) : 
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 1 → (a n / n - a (n - 1) / (n - 1)) = 2) → (∀ n : ℕ, a n = 2 * n * n - n) :=
by
  sorry

end NUMINAMATH_GPT_sequence_an_expression_l2159_215942


namespace NUMINAMATH_GPT_prob_club_then_diamond_then_heart_l2159_215977

noncomputable def prob_first_card_club := 13 / 52
noncomputable def prob_second_card_diamond_given_first_club := 13 / 51
noncomputable def prob_third_card_heart_given_first_club_second_diamond := 13 / 50

noncomputable def overall_probability := 
  prob_first_card_club * 
  prob_second_card_diamond_given_first_club * 
  prob_third_card_heart_given_first_club_second_diamond

theorem prob_club_then_diamond_then_heart :
  overall_probability = 2197 / 132600 :=
by
  sorry

end NUMINAMATH_GPT_prob_club_then_diamond_then_heart_l2159_215977


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2159_215962

theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 - 4 = 0 → x + 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l2159_215962
