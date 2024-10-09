import Mathlib

namespace total_spent_correct_l2386_238606

def shorts : ℝ := 13.99
def shirt : ℝ := 12.14
def jacket : ℝ := 7.43
def total_spent : ℝ := 33.56

theorem total_spent_correct : shorts + shirt + jacket = total_spent :=
by
  sorry

end total_spent_correct_l2386_238606


namespace Kim_total_hours_l2386_238660

-- Define the initial conditions
def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_class : ℕ := 1

-- The proof problem: Given the initial conditions, prove the total hours of classes per day is 6
theorem Kim_total_hours : (initial_classes - dropped_class) * hours_per_class = 6 := by
  sorry

end Kim_total_hours_l2386_238660


namespace eval_expression_l2386_238622

theorem eval_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / (a * b) - (a^2 + a * b) / (a^2 + b^2) = (a^4 + b^4 + a^2 * b^2 - a^2 * b - a * b^2) / (a * b * (a^2 + b^2)) :=
by
  sorry

end eval_expression_l2386_238622


namespace number_of_children_l2386_238673

theorem number_of_children (total_people : ℕ) (num_adults num_children : ℕ)
  (h1 : total_people = 42)
  (h2 : num_children = 2 * num_adults)
  (h3 : num_adults + num_children = total_people) :
  num_children = 28 :=
by
  sorry

end number_of_children_l2386_238673


namespace remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l2386_238695

theorem remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one :
  ((x - 1) ^ 2028) % (x^2 - x + 1) = 1 :=
by
  sorry

end remainder_of_x_sub_one_pow_2028_mod_x_sq_sub_x_add_one_l2386_238695


namespace ratio_of_areas_l2386_238683
-- Define the conditions and the ratio to be proven
theorem ratio_of_areas (t r : ℝ) (h : 3 * t = 2 * π * r) : 
  (π^2 / 18) = (π^2 * r^2 / 9) / (2 * r^2) :=
by 
  sorry

end ratio_of_areas_l2386_238683


namespace final_total_cost_is_12_70_l2386_238634

-- Definitions and conditions
def sandwich_count : ℕ := 2
def sandwich_cost_per_unit : ℝ := 2.45

def soda_count : ℕ := 4
def soda_cost_per_unit : ℝ := 0.87

def chips_count : ℕ := 3
def chips_cost_per_unit : ℝ := 1.29

def sandwich_discount : ℝ := 0.10
def sales_tax : ℝ := 0.08

-- Final price after discount and tax
noncomputable def total_cost : ℝ :=
  let sandwiches_total := sandwich_count * sandwich_cost_per_unit
  let discounted_sandwiches := sandwiches_total * (1 - sandwich_discount)
  let sodas_total := soda_count * soda_cost_per_unit
  let chips_total := chips_count * chips_cost_per_unit
  let subtotal := discounted_sandwiches + sodas_total + chips_total
  let final_total := subtotal * (1 + sales_tax)
  final_total

theorem final_total_cost_is_12_70 : total_cost = 12.70 :=
by 
  sorry

end final_total_cost_is_12_70_l2386_238634


namespace factory_days_worked_l2386_238608

-- Define the number of refrigerators produced per hour
def refrigerators_per_hour : ℕ := 90

-- Define the number of coolers produced per hour
def coolers_per_hour : ℕ := refrigerators_per_hour + 70

-- Define the number of working hours per day
def working_hours_per_day : ℕ := 9

-- Define the total products produced per hour
def products_per_hour : ℕ := refrigerators_per_hour + coolers_per_hour

-- Define the total products produced in a day
def products_per_day : ℕ := products_per_hour * working_hours_per_day

-- Define the total number of products produced in given days
def total_products : ℕ := 11250

-- Define the number of days worked
def days_worked : ℕ := total_products / products_per_day

-- Prove that the number of days worked equals 5
theorem factory_days_worked : days_worked = 5 :=
by
  sorry

end factory_days_worked_l2386_238608


namespace complex_problem_l2386_238665

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem complex_problem :
  (1 - z) * (1 - z^2) * (1 - z^3) * (1 - z^4) = 5 :=
by
  sorry

end complex_problem_l2386_238665


namespace geometric_mean_of_4_and_9_l2386_238627

theorem geometric_mean_of_4_and_9 : ∃ (G : ℝ), G = 6 ∨ G = -6 :=
by
  sorry

end geometric_mean_of_4_and_9_l2386_238627


namespace projectile_height_reaches_49_l2386_238646

theorem projectile_height_reaches_49 (t : ℝ) :
  (∃ t : ℝ, 49 = -20 * t^2 + 100 * t) → t = 0.7 :=
by
  sorry

end projectile_height_reaches_49_l2386_238646


namespace find_coefficients_l2386_238600

def polynomial (a b : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 3 - 3 * x ^ 2 + b * x - 7

theorem find_coefficients (a b : ℝ) :
  polynomial a b 2 = -17 ∧ polynomial a b (-1) = -11 → a = 0 ∧ b = -1 :=
by
  sorry

end find_coefficients_l2386_238600


namespace find_D_plus_E_plus_F_l2386_238659

noncomputable def g (x : ℝ) (D E F : ℝ) : ℝ := (x^2) / (D * x^2 + E * x + F)

theorem find_D_plus_E_plus_F (D E F : ℤ) 
  (h1 : ∀ x : ℝ, x > 3 → g x D E F > 0.3)
  (h2 : ∀ x : ℝ, ¬(D * x^2 + E * x + F = 0 ↔ (x = -3 ∨ x = 2))) :
  D + E + F = -8 :=
sorry

end find_D_plus_E_plus_F_l2386_238659


namespace negation_of_proposition_l2386_238685

theorem negation_of_proposition : 
    (¬ (∀ x : ℝ, x^2 - 2 * |x| ≥ 0)) ↔ (∃ x : ℝ, x^2 - 2 * |x| < 0) :=
by sorry

end negation_of_proposition_l2386_238685


namespace find_a_l2386_238636

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a^2 - 1

theorem find_a (a : ℝ) (h : ∀ x ∈ (Set.Icc 1 2), f x a ≤ 16 ∧ ∃ y ∈ (Set.Icc 1 2), f y a = 16) : a = 3 ∨ a = -3 :=
by
  sorry

end find_a_l2386_238636


namespace mod_3_pow_2040_eq_1_mod_5_l2386_238637

theorem mod_3_pow_2040_eq_1_mod_5 :
  (3 ^ 2040) % 5 = 1 := by
  -- Here the theorem states that the remainder of 3^2040 when divided by 5 is equal to 1
  sorry

end mod_3_pow_2040_eq_1_mod_5_l2386_238637


namespace sum_remainder_l2386_238610

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l2386_238610


namespace chairs_per_row_l2386_238643

/-- There are 10 rows of chairs, with the first row for awardees, the second and third rows for
    administrators and teachers, the last two rows for parents, and the remaining five rows for students.
    Given that 4/5 of the student seats are occupied, and there are 15 vacant seats among the students,
    proves that the number of chairs per row is 15. --/
theorem chairs_per_row (x : ℕ) (h1 : 10 = 1 + 1 + 1 + 5 + 2)
  (h2 : 4 / 5 * (5 * x) + 1 / 5 * (5 * x) = 5 * x)
  (h3 : 1 / 5 * (5 * x) = 15) : x = 15 :=
sorry

end chairs_per_row_l2386_238643


namespace standard_parts_bounds_l2386_238626

noncomputable def n : ℕ := 900
noncomputable def p : ℝ := 0.9
noncomputable def confidence_level : ℝ := 0.95
noncomputable def lower_bound : ℝ := 792
noncomputable def upper_bound : ℝ := 828

theorem standard_parts_bounds : 
  792 ≤ n * p - 1.96 * (n * p * (1 - p)).sqrt ∧ 
  n * p + 1.96 * (n * p * (1 - p)).sqrt ≤ 828 :=
sorry

end standard_parts_bounds_l2386_238626


namespace symmetrical_parabola_eq_l2386_238640

/-- 
  Given a parabola y = (x-1)^2 + 3, prove that its symmetrical parabola 
  about the x-axis is y = -(x-1)^2 - 3.
-/
theorem symmetrical_parabola_eq (x : ℝ) : 
  (x-1)^2 + 3 = -(x-1)^2 - 3 ↔ y = -(x-1)^2 - 3 := 
sorry

end symmetrical_parabola_eq_l2386_238640


namespace probability_of_one_pair_one_triplet_l2386_238625

-- Define the necessary conditions
def six_sided_die_rolls (n : ℕ) : ℕ := 6 ^ n

def successful_outcomes : ℕ :=
  6 * 20 * 5 * 3 * 4

def total_outcomes : ℕ :=
  six_sided_die_rolls 6

def probability_success : ℚ :=
  successful_outcomes / total_outcomes

-- The theorem we want to prove
theorem probability_of_one_pair_one_triplet :
  probability_success = 25/162 :=
sorry

end probability_of_one_pair_one_triplet_l2386_238625


namespace before_lunch_rush_customers_l2386_238678

def original_customers_before_lunch := 29
def added_customers_during_lunch := 20
def customers_no_tip := 34
def customers_tip := 15

theorem before_lunch_rush_customers : 
  original_customers_before_lunch + added_customers_during_lunch = customers_no_tip + customers_tip → 
  original_customers_before_lunch = 29 := 
by
  sorry

end before_lunch_rush_customers_l2386_238678


namespace job_candidates_excel_nights_l2386_238621

theorem job_candidates_excel_nights (hasExcel : ℝ) (dayShift : ℝ) 
    (h1 : hasExcel = 0.2) (h2 : dayShift = 0.7) : 
    (1 - dayShift) * hasExcel = 0.06 :=
by
  sorry

end job_candidates_excel_nights_l2386_238621


namespace price_of_second_tea_l2386_238691

theorem price_of_second_tea (P : ℝ) (h1 : 1 * 64 + 1 * P = 2 * 69) : P = 74 := 
by
  sorry

end price_of_second_tea_l2386_238691


namespace largest_among_numbers_l2386_238635

theorem largest_among_numbers :
  ∀ (a b c d e : ℝ), 
  a = 0.997 ∧ b = 0.9799 ∧ c = 0.999 ∧ d = 0.9979 ∧ e = 0.979 →
  c > a ∧ c > b ∧ c > d ∧ c > e :=
by intros a b c d e habcde
   rcases habcde with ⟨ha, hb, hc, hd, he⟩
   simp [ha, hb, hc, hd, he]
   sorry

end largest_among_numbers_l2386_238635


namespace n_squared_plus_n_is_even_l2386_238672

theorem n_squared_plus_n_is_even (n : ℤ) : Even (n^2 + n) :=
by
  sorry

end n_squared_plus_n_is_even_l2386_238672


namespace election_max_k_1002_l2386_238688

/-- There are 2002 candidates initially. 
In each round, one candidate with the least number of votes is eliminated unless a candidate receives more than half the votes.
Determine the highest possible value of k if Ostap Bender is elected in the 1002nd round. -/
theorem election_max_k_1002 
  (number_of_candidates : ℕ)
  (number_of_rounds : ℕ)
  (k : ℕ)
  (h1 : number_of_candidates = 2002)
  (h2 : number_of_rounds = 1002)
  (h3 : k ≤ number_of_candidates - 1)
  (h4 : ∀ n : ℕ, n < number_of_rounds → (k + n) % (number_of_candidates - n) ≠ 0) : 
  k = 2001 := sorry

end election_max_k_1002_l2386_238688


namespace probability_of_winning_l2386_238663

def probability_of_losing : ℚ := 3 / 7

theorem probability_of_winning (h : probability_of_losing + p = 1) : p = 4 / 7 :=
by 
  sorry

end probability_of_winning_l2386_238663


namespace track_meet_total_people_l2386_238607

theorem track_meet_total_people (B G : ℕ) (H1 : B = 30)
  (H2 : ∃ G, (3 * G) / 5 + (2 * G) / 5 = G)
  (H3 : ∀ G, 2 * G / 5 = 10) :
  B + G = 55 :=
by
  sorry

end track_meet_total_people_l2386_238607


namespace lcm_fractions_l2386_238613

theorem lcm_fractions (x : ℕ) (hx : x > 0) :
  lcm (1 / (2 * x)) (lcm (1 / (4 * x)) (lcm (1 / (6 * x)) (1 / (12 * x)))) = 1 / (12 * x) :=
sorry

end lcm_fractions_l2386_238613


namespace calculate_expression_l2386_238632

theorem calculate_expression : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end calculate_expression_l2386_238632


namespace lesser_number_l2386_238653

theorem lesser_number (x y : ℕ) (h1 : x + y = 58) (h2 : x - y = 6) : y = 26 :=
by
  sorry

end lesser_number_l2386_238653


namespace painting_price_decrease_l2386_238624

theorem painting_price_decrease (P : ℝ) (h1 : 1.10 * P - 0.935 * P = x * 1.10 * P) :
  x = 0.15 := by
  sorry

end painting_price_decrease_l2386_238624


namespace intersection_C_U_M_N_l2386_238651

open Set

-- Define U, M and N
def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

-- Define complement C_U M in U
def C_U_M : Set ℕ := U \ M

-- The theorem to prove
theorem intersection_C_U_M_N : (C_U_M ∩ N) = {3} := by
  sorry

end intersection_C_U_M_N_l2386_238651


namespace correct_multiplication_l2386_238684

theorem correct_multiplication :
  ∃ (n : ℕ), 98765 * n = 888885 ∧ (98765 * n = 867559827931 → n = 9) :=
by
  sorry

end correct_multiplication_l2386_238684


namespace largest_consecutive_multiple_l2386_238656

theorem largest_consecutive_multiple (n : ℕ) (h : 3 * n + 3 * (n + 1) + 3 * (n + 2) = 117) : 3 * (n + 2) = 42 :=
sorry

end largest_consecutive_multiple_l2386_238656


namespace x_varies_as_nth_power_of_z_l2386_238629

theorem x_varies_as_nth_power_of_z 
  (k j z : ℝ) 
  (h1 : ∃ y : ℝ, x = k * y^4 ∧ y = j * z^(1/3)) : 
  ∃ m : ℝ, x = m * z^(4/3) := 
 sorry

end x_varies_as_nth_power_of_z_l2386_238629


namespace expected_value_ball_draw_l2386_238616

noncomputable def E_xi : ℚ :=
  let prob_xi_2 := 3/5
  let prob_xi_3 := 3/10
  let prob_xi_4 := 1/10
  2 * prob_xi_2 + 3 * prob_xi_3 + 4 * prob_xi_4

theorem expected_value_ball_draw : E_xi = 5 / 2 := by
  sorry

end expected_value_ball_draw_l2386_238616


namespace A_loses_240_l2386_238617

def initial_house_value : ℝ := 12000
def house_value_after_A_sells : ℝ := initial_house_value * 0.85
def house_value_after_B_sells_back : ℝ := house_value_after_A_sells * 1.2

theorem A_loses_240 : house_value_after_B_sells_back - initial_house_value = 240 := by
  sorry

end A_loses_240_l2386_238617


namespace fraction_operation_correct_l2386_238699

theorem fraction_operation_correct (a b : ℝ) (h : 0.2 * a + 0.5 * b ≠ 0) : 
  (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) :=
sorry

end fraction_operation_correct_l2386_238699


namespace anna_phone_chargers_l2386_238641

theorem anna_phone_chargers (p l : ℕ) (h₁ : l = 5 * p) (h₂ : l + p = 24) : p = 4 :=
by
  sorry

end anna_phone_chargers_l2386_238641


namespace cast_cost_l2386_238677

theorem cast_cost (C : ℝ) 
  (visit_cost : ℝ := 300)
  (insurance_coverage : ℝ := 0.60)
  (out_of_pocket_cost : ℝ := 200) :
  0.40 * (visit_cost + C) = out_of_pocket_cost → 
  C = 200 := by
  sorry

end cast_cost_l2386_238677


namespace factorization_correct_l2386_238647

theorem factorization_correct : ∀ y : ℝ, y^2 - 4*y + 4 = (y - 2)^2 := by
  intro y
  sorry

end factorization_correct_l2386_238647


namespace admin_fee_percentage_l2386_238652

noncomputable def percentage_deducted_for_admin_fees 
  (amt_johnson : ℕ) (amt_sutton : ℕ) (amt_rollin : ℕ)
  (amt_school : ℕ) (amt_after_deduction : ℕ) : ℚ :=
  ((amt_school - amt_after_deduction) * 100) / amt_school

theorem admin_fee_percentage : 
  ∃ (amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction : ℕ),
  amt_johnson = 2300 ∧
  amt_johnson = 2 * amt_sutton ∧
  amt_sutton * 8 = amt_rollin ∧
  amt_rollin * 3 = amt_school ∧
  amt_after_deduction = 27048 ∧
  percentage_deducted_for_admin_fees amt_johnson amt_sutton amt_rollin amt_school amt_after_deduction = 2 :=
by
  sorry

end admin_fee_percentage_l2386_238652


namespace find_constants_l2386_238694

theorem find_constants :
  ∃ (A B C : ℚ), 
  (A = 1 ∧ B = 4 ∧ C = 1) ∧ 
  (∀ x, x ≠ -1 → x ≠ 3/2 → x ≠ 2 → 
    (6 * x^2 - 13 * x + 6) / (2 * x^3 + 3 * x^2 - 11 * x - 6) = 
    (A / (x + 1) + B / (2 * x - 3) + C / (x - 2))) :=
by
  sorry

end find_constants_l2386_238694


namespace parabola_expression_l2386_238649

theorem parabola_expression 
  (a b : ℝ) 
  (h : 9 = a * (-2)^2 + b * (-2) + 5) : 
  2 * a - b + 6 = 8 :=
by
  sorry

end parabola_expression_l2386_238649


namespace line1_line2_line3_l2386_238618

-- Line 1: Through (-1, 3), parallel to x - 2y + 3 = 0.
theorem line1 (x y : ℝ) : (x - 2 * y + 3 = 0) ∧ (x = -1) ∧ (y = 3) →
                              (x - 2 * y + 7 = 0) :=
by sorry

-- Line 2: Through (3, 4), perpendicular to 3x - y + 2 = 0.
theorem line2 (x y : ℝ) : (3 * x - y + 2 = 0) ∧ (x = 3) ∧ (y = 4) →
                              (x + 3 * y - 15 = 0) :=
by sorry

-- Line 3: Through (1, 2), with equal intercepts on both axes.
theorem line3 (x y : ℝ) : (x = y) ∧ (x = 1) ∧ (y = 2) →
                              (x + y - 3 = 0) :=
by sorry

end line1_line2_line3_l2386_238618


namespace vector_normalization_condition_l2386_238611

variables {a b : ℝ} -- Ensuring that Lean understands ℝ refers to real numbers and specifically vectors in ℝ before using it in the next parts.

-- Definitions of the vector variables
variables (a b : ℝ) (ab_non_zero : a ≠ 0 ∧ b ≠ 0)

-- Required statement
theorem vector_normalization_condition (a b : ℝ) 
(h₀ : a ≠ 0 ∧ b ≠ 0) :
  (a / abs a = b / abs b) ↔ (a = 2 * b) :=
sorry

end vector_normalization_condition_l2386_238611


namespace Moscow1964_27th_MMO_l2386_238671

theorem Moscow1964_27th_MMO {a : ℤ} (h : ∀ k : ℤ, k ≠ 27 → ∃ m : ℤ, a - k^1964 = m * (27 - k)) : 
  a = 27^1964 :=
sorry

end Moscow1964_27th_MMO_l2386_238671


namespace percentage_equivalence_l2386_238601

theorem percentage_equivalence (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.4 * 0.3 * x = 36 :=
by
  sorry

end percentage_equivalence_l2386_238601


namespace daniella_lap_time_l2386_238675

theorem daniella_lap_time
  (T_T : ℕ) (H_TT : T_T = 56)
  (meet_time : ℕ) (H_meet : meet_time = 24) :
  ∃ T_D : ℕ, T_D = 42 :=
by
  sorry

end daniella_lap_time_l2386_238675


namespace nesbitts_inequality_l2386_238620

theorem nesbitts_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (a + c) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end nesbitts_inequality_l2386_238620


namespace men_who_wore_glasses_l2386_238693

theorem men_who_wore_glasses (total_people : ℕ) (women_ratio men_with_glasses_ratio : ℚ)  
  (h_total : total_people = 1260) 
  (h_women_ratio : women_ratio = 7 / 18)
  (h_men_with_glasses_ratio : men_with_glasses_ratio = 6 / 11)
  : ∃ (men_with_glasses : ℕ), men_with_glasses = 420 := 
by
  sorry

end men_who_wore_glasses_l2386_238693


namespace length_of_MN_l2386_238638

noncomputable def curve_eq (α : ℝ) : ℝ × ℝ := (2 * Real.cos α + 1, 2 * Real.sin α)

noncomputable def line_eq (ρ θ : ℝ) : ℝ × ℝ := 
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem length_of_MN : ∀ (M N : ℝ × ℝ), 
  M ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  N ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2)^2 = 4} ∧
  M ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} ∧
  N ∈ {p : ℝ × ℝ | p.1 + p.2 = 2} →
  dist M N = Real.sqrt 14 :=
by
  sorry

end length_of_MN_l2386_238638


namespace residue_12_2040_mod_19_l2386_238639

theorem residue_12_2040_mod_19 :
  12^2040 % 19 = 7 := 
sorry

end residue_12_2040_mod_19_l2386_238639


namespace distance_between_parallel_lines_l2386_238619

theorem distance_between_parallel_lines (r d : ℝ) 
  (h1 : ∃ p1 p2 p3 : ℝ, p1 = 40 ∧ p2 = 40 ∧ p3 = 36) 
  (h2 : ∀ θ : ℝ, ∃ A B C D : ℝ → ℝ, 
    (A θ - B θ) = 40 ∧ (C θ - D θ) = 36) : d = 6 :=
sorry

end distance_between_parallel_lines_l2386_238619


namespace remainder_of_17_power_1801_mod_28_l2386_238668

theorem remainder_of_17_power_1801_mod_28 : (17 ^ 1801) % 28 = 17 := 
by
  sorry

end remainder_of_17_power_1801_mod_28_l2386_238668


namespace mean_score_is_74_l2386_238681

theorem mean_score_is_74 (M SD : ℝ) 
  (h1 : 58 = M - 2 * SD) 
  (h2 : 98 = M + 3 * SD) : 
  M = 74 := 
by 
  -- problem statement without solving steps
  sorry

end mean_score_is_74_l2386_238681


namespace value_of_kaftan_l2386_238631

theorem value_of_kaftan (K : ℝ) (h : (7 / 12) * (12 + K) = 5 + K) : K = 4.8 :=
by
  sorry

end value_of_kaftan_l2386_238631


namespace seed_mixture_ryegrass_percent_l2386_238602

theorem seed_mixture_ryegrass_percent (R : ℝ) :
  let X := 0.40
  let percentage_X_in_mixture := 1 / 3
  let percentage_Y_in_mixture := 2 / 3
  let final_ryegrass := 0.30
  (final_ryegrass = percentage_X_in_mixture * X + percentage_Y_in_mixture * R) → 
  R = 0.25 :=
by
  intros X percentage_X_in_mixture percentage_Y_in_mixture final_ryegrass H
  sorry

end seed_mixture_ryegrass_percent_l2386_238602


namespace correct_statement_c_l2386_238696

theorem correct_statement_c (five_boys_two_girls : Nat := 7) (select_three : Nat := 3) :
  (∃ boys girls : Nat, boys + girls = five_boys_two_girls ∧ boys = 5 ∧ girls = 2) →
  (∃ selected_boys selected_girls : Nat, selected_boys + selected_girls = select_three ∧ selected_boys > 0) :=
by
  sorry

end correct_statement_c_l2386_238696


namespace polar_to_rectangular_inequality_range_l2386_238605

-- Part A: Transforming a polar coordinate equation to a rectangular coordinate equation
theorem polar_to_rectangular (ρ θ : ℝ) : 
  (ρ^2 * Real.cos θ - ρ = 0) ↔ ((ρ = 0 ∧ 0 = 1) ∨ (ρ ≠ 0 ∧ Real.cos θ = 1 / ρ)) := 
sorry

-- Part B: Determining range for an inequality
theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2-x| + |x+1| ≤ a) ↔ (a ≥ 9) := 
sorry

end polar_to_rectangular_inequality_range_l2386_238605


namespace strictly_increasing_arithmetic_seq_l2386_238682

theorem strictly_increasing_arithmetic_seq 
  (s : ℕ → ℕ) 
  (hs_incr : ∀ n, s n < s (n + 1)) 
  (hs_seq1 : ∃ D1, ∀ n, s (s n) = s (s 0) + n * D1) 
  (hs_seq2 : ∃ D2, ∀ n, s (s n + 1) = s (s 0 + 1) + n * D2) : 
  ∃ d, ∀ n, s (n + 1) = s n + d :=
sorry

end strictly_increasing_arithmetic_seq_l2386_238682


namespace daliah_garbage_l2386_238697

theorem daliah_garbage (D : ℝ) (h1 : 4 * (D - 2) = 62) : D = 17.5 :=
by
  sorry

end daliah_garbage_l2386_238697


namespace arithmetic_sequence_a15_l2386_238633

variable {α : Type*} [LinearOrderedField α]

-- Conditions for the arithmetic sequence
variable (a : ℕ → α)
variable (d : α)
variable (a1 : α)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 5)
variable (h_a10 : a 10 = 15)

-- To prove that a15 = 25
theorem arithmetic_sequence_a15 : a 15 = 25 := by
  sorry

end arithmetic_sequence_a15_l2386_238633


namespace find_inverse_sum_l2386_238628

variable {R : Type*} [OrderedRing R]

-- Define the function f and its inverse
variable (f : R → R)
variable (f_inv : R → R)

-- Conditions
axiom f_inverse : ∀ y, f (f_inv y) = y
axiom f_prop : ∀ x, f x + f (1 - x) = 2

-- The theorem we need to prove
theorem find_inverse_sum (x : R) : f_inv (x - 2) + f_inv (4 - x) = 1 :=
by
  sorry

end find_inverse_sum_l2386_238628


namespace quadratic_func_max_value_l2386_238658

theorem quadratic_func_max_value (b c x y : ℝ) (h1 : y = -x^2 + b * x + c)
(h1_x1 : (y = 0) → x = -1 ∨ x = 3) :
    -x^2 + 2 * x + 3 ≤ 4 :=
sorry

end quadratic_func_max_value_l2386_238658


namespace hotel_friends_count_l2386_238655

theorem hotel_friends_count
  (n : ℕ)
  (friend_share extra friend_payment : ℕ)
  (h1 : 7 * 80 + friend_payment = 720)
  (h2 : friend_payment = friend_share + extra)
  (h3 : friend_payment = 160)
  (h4 : extra = 70)
  (h5 : friend_share = 90) :
  n = 8 :=
sorry

end hotel_friends_count_l2386_238655


namespace range_of_b_l2386_238614

theorem range_of_b (b : ℝ) : (∃ x : ℝ, |x - 2| + |x - 5| < b) → b > 3 :=
by 
-- This is where the proof would go.
sorry

end range_of_b_l2386_238614


namespace parabola_intersection_prob_l2386_238666

noncomputable def prob_intersect_parabolas : ℚ :=
  57 / 64

theorem parabola_intersection_prob :
  ∀ (a b c d : ℤ), (1 ≤ a ∧ a ≤ 8) → (1 ≤ b ∧ b ≤ 8) →
  (1 ≤ c∧ c ≤ 8) → (1 ≤ d ∧ d ≤ 8) →
  prob_intersect_parabolas = 57 / 64 :=
by
  intros a b c d ha hb hc hd
  sorry

end parabola_intersection_prob_l2386_238666


namespace calculate_PC_l2386_238689
noncomputable def ratio (a b : ℝ) : ℝ := a / b

theorem calculate_PC (AB BC CA PC PA : ℝ) (h1: AB = 6) (h2: BC = 10) (h3: CA = 8)
  (h4: ratio PC PA = ratio 8 6)
  (h5: ratio PA (PC + 10) = ratio 6 10) :
  PC = 40 :=
sorry

end calculate_PC_l2386_238689


namespace fish_caught_l2386_238644

noncomputable def total_fish_caught (chris_trips : ℕ) (chris_fish_per_trip : ℕ) (brian_trips : ℕ) (brian_fish_per_trip : ℕ) : ℕ :=
  chris_trips * chris_fish_per_trip + brian_trips * brian_fish_per_trip

theorem fish_caught (chris_trips : ℕ) (brian_factor : ℕ) (brian_fish_per_trip : ℕ) (ratio_numerator : ℕ) (ratio_denominator : ℕ) :
  chris_trips = 10 → brian_factor = 2 → brian_fish_per_trip = 400 → ratio_numerator = 3 → ratio_denominator = 5 →
  total_fish_caught chris_trips (brian_fish_per_trip * ratio_denominator / ratio_numerator) (chris_trips * brian_factor) brian_fish_per_trip = 14660 :=
by
  intros h_chris_trips h_brian_factor h_brian_fish_per_trip h_ratio_numer h_ratio_denom
  rw [h_chris_trips, h_brian_factor, h_brian_fish_per_trip, h_ratio_numer, h_ratio_denom]
  -- adding actual arithmetic would resolve the statement correctly
  sorry

end fish_caught_l2386_238644


namespace complex_z_power_l2386_238676

theorem complex_z_power:
  ∀ (z : ℂ), (z + 1/z = 2 * Real.cos (5 * Real.pi / 180)) →
  z^1000 + (1/z)^1000 = 2 * Real.cos (20 * Real.pi / 180) :=
by
  sorry

end complex_z_power_l2386_238676


namespace article_production_l2386_238690

-- Conditions
variables (x z : ℕ) (hx : 0 < x) (hz : 0 < z)
-- The given condition: x men working x hours a day for x days produce 2x^2 articles.
def articles_produced_x (x : ℕ) : ℕ := 2 * x^2

-- The question: the number of articles produced by z men working z hours a day for z days
def articles_produced_z (x z : ℕ) : ℕ := 2 * z^3 / x

-- Prove that the number of articles produced by z men working z hours a day for z days is 2 * (z^3) / x
theorem article_production (hx : 0 < x) (hz : 0 < z) :
  articles_produced_z x z = 2 * z^3 / x :=
sorry

end article_production_l2386_238690


namespace doug_fires_l2386_238692

theorem doug_fires (D : ℝ) (Kai_fires : ℝ) (Eli_fires : ℝ) 
    (hKai : Kai_fires = 3 * D)
    (hEli : Eli_fires = 1.5 * D)
    (hTotal : D + Kai_fires + Eli_fires = 110) : 
  D = 20 := 
by
  sorry

end doug_fires_l2386_238692


namespace negation_of_universal_l2386_238667

theorem negation_of_universal : (¬ ∀ x : ℝ, x^2 + 2 * x - 1 = 0) ↔ ∃ x : ℝ, x^2 + 2 * x - 1 ≠ 0 :=
by sorry

end negation_of_universal_l2386_238667


namespace yellow_tiles_count_l2386_238654

theorem yellow_tiles_count
  (total_tiles : ℕ)
  (yellow_tiles : ℕ)
  (blue_tiles : ℕ)
  (purple_tiles : ℕ)
  (white_tiles : ℕ)
  (h1 : total_tiles = 20)
  (h2 : blue_tiles = yellow_tiles + 1)
  (h3 : purple_tiles = 6)
  (h4 : white_tiles = 7)
  (h5 : total_tiles = yellow_tiles + blue_tiles + purple_tiles + white_tiles) :
  yellow_tiles = 3 :=
by sorry

end yellow_tiles_count_l2386_238654


namespace largest_divisor_even_triplet_l2386_238615

theorem largest_divisor_even_triplet :
  ∀ (n : ℕ), 24 ∣ (2 * n) * (2 * n + 2) * (2 * n + 4) :=
by intros; sorry

end largest_divisor_even_triplet_l2386_238615


namespace color_stamps_sold_l2386_238642

theorem color_stamps_sold :
    let total_stamps : ℕ := 1102609
    let black_and_white_stamps : ℕ := 523776
    total_stamps - black_and_white_stamps = 578833 := 
by
  sorry

end color_stamps_sold_l2386_238642


namespace painting_ways_correct_l2386_238662

noncomputable def num_ways_to_paint : ℕ :=
  let red := 1
  let green_or_blue := 2
  let total_ways_case1 := red
  let total_ways_case2 := (green_or_blue ^ 4)
  let total_ways_case3 := green_or_blue ^ 3
  let total_ways_case4 := green_or_blue ^ 2
  let total_ways_case5 := green_or_blue
  let total_ways_case6 := red
  total_ways_case1 + total_ways_case2 + total_ways_case3 + total_ways_case4 + total_ways_case5 + total_ways_case6

theorem painting_ways_correct : num_ways_to_paint = 32 :=
  by
  sorry

end painting_ways_correct_l2386_238662


namespace convex_polyhedron_in_inscribed_sphere_l2386_238674

-- Definitions based on conditions
variables (S c r : ℝ) (S' V R : ℝ)

-- The given relationship for a convex polygon.
def poly_relationship := S = (1 / 2) * c * r

-- The desired relationship for a convex polyhedron.
def polyhedron_relationship := V = (1 / 3) * S' * R

-- Proof statement
theorem convex_polyhedron_in_inscribed_sphere (S c r S' V R : ℝ) 
  (poly : S = (1 / 2) * c * r) : V = (1 / 3) * S' * R :=
sorry

end convex_polyhedron_in_inscribed_sphere_l2386_238674


namespace branches_count_eq_6_l2386_238650

theorem branches_count_eq_6 (x : ℕ) (h : 1 + x + x^2 = 43) : x = 6 :=
sorry

end branches_count_eq_6_l2386_238650


namespace number_of_candidates_is_three_l2386_238687

variable (votes : List ℕ) (totalVotes : ℕ)

def determineNumberOfCandidates (votes : List ℕ) (totalVotes : ℕ) : ℕ :=
  votes.length

theorem number_of_candidates_is_three (V : ℕ) 
  (h_votes : [2500, 5000, 20000].sum = V) 
  (h_percent : 20000 = 7273 / 10000 * V): 
  determineNumberOfCandidates [2500, 5000, 20000] V = 3 := 
by 
  sorry

end number_of_candidates_is_three_l2386_238687


namespace find_number_l2386_238603

noncomputable def number_divided_by_seven_is_five_fourteen (x : ℝ) : Prop :=
  x / 7 = 5 / 14

theorem find_number (x : ℝ) (h : number_divided_by_seven_is_five_fourteen x) : x = 2.5 :=
by
  sorry

end find_number_l2386_238603


namespace geometric_series_sum_l2386_238698

-- Conditions
def is_geometric_series (a r : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a * r ^ n

-- The problem statement translated into Lean: Proving the sum of the series
theorem geometric_series_sum : ∃ S : ℕ → ℝ, is_geometric_series 1 (1/4) S ∧ ∑' n, S n = 4/3 :=
by
  sorry

end geometric_series_sum_l2386_238698


namespace sin_trig_identity_l2386_238679

theorem sin_trig_identity (α : ℝ) (h : Real.sin (α - π/4) = 1/2) : Real.sin ((5 * π) / 4 - α) = 1/2 := 
by 
  sorry

end sin_trig_identity_l2386_238679


namespace isosceles_triangle_base_length_l2386_238612

theorem isosceles_triangle_base_length :
  ∃ (x y : ℝ), 
    ((x + x / 2 = 15 ∧ y + x / 2 = 6) ∨ (x + x / 2 = 6 ∧ y + x / 2 = 15)) ∧ y = 1 :=
by
  sorry

end isosceles_triangle_base_length_l2386_238612


namespace articles_produced_l2386_238664

theorem articles_produced (a b c p q r : Nat) (h : a * b * c = abc) : p * q * r = pqr := sorry

end articles_produced_l2386_238664


namespace triangle_ABC_properties_l2386_238630

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x

theorem triangle_ABC_properties
  (xA xB xC : ℝ)
  (h_seq : xA < xB ∧ xB < xC ∧ 2 * xB = xA + xC)
  : (f xB + (f xA + f xC) / 2 > f ((xA + xC) / 2)) ∧ (f xA ≠ f xB ∧ f xB ≠ f xC) := 
sorry

end triangle_ABC_properties_l2386_238630


namespace count_integers_in_range_l2386_238669

theorem count_integers_in_range : 
  let lower_bound := -2.8
  let upper_bound := Real.pi
  let in_range (x : ℤ) := (lower_bound : ℝ) < (x : ℝ) ∧ (x : ℝ) ≤ upper_bound
  (Finset.filter in_range (Finset.Icc (Int.floor lower_bound) (Int.floor upper_bound))).card = 6 :=
by
  sorry

end count_integers_in_range_l2386_238669


namespace gcd_of_1887_and_2091_is_51_l2386_238623

variable (a b : Nat)
variable (coefficient1 coefficient2 quotient1 quotient2 quotient3 remainder1 remainder2 : Nat)

def gcd_condition1 : Prop := (b = 1 * a + remainder1)
def gcd_condition2 : Prop := (a = quotient1 * remainder1 + remainder2)
def gcd_condition3 : Prop := (remainder1 = quotient2 * remainder2)

def numbers_1887_and_2091 : Prop := (a = 1887) ∧ (b = 2091)

theorem gcd_of_1887_and_2091_is_51 :
  numbers_1887_and_2091 a b ∧
  gcd_condition1 a b remainder1 ∧ 
  gcd_condition2 a remainder1 remainder2 quotient1 ∧ 
  gcd_condition3 remainder1 remainder2 quotient2 → 
  Nat.gcd 1887 2091 = 51 :=
by
  sorry

end gcd_of_1887_and_2091_is_51_l2386_238623


namespace find_positive_integer_pairs_l2386_238657

theorem find_positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a^2 = 3 * b^3) ↔ ∃ d : ℕ, 0 < d ∧ a = 18 * d^3 ∧ b = 6 * d^2 :=
by
  sorry

end find_positive_integer_pairs_l2386_238657


namespace no_such_cuboid_exists_l2386_238645

theorem no_such_cuboid_exists (a b c : ℝ) :
  a + b + c = 12 ∧ ab + bc + ca = 1 ∧ abc = 12 → false :=
by
  sorry

end no_such_cuboid_exists_l2386_238645


namespace product_of_roots_is_12_l2386_238604

theorem product_of_roots_is_12 :
  (81 ^ (1 / 4) * 8 ^ (1 / 3) * 4 ^ (1 / 2)) = 12 := by
  sorry

end product_of_roots_is_12_l2386_238604


namespace find_initial_marbles_l2386_238680

-- Definitions based on conditions
def loses_to_street (initial_marbles : ℕ) : ℕ := initial_marbles - (initial_marbles * 60 / 100)
def loses_to_sewer (marbles_after_street : ℕ) : ℕ := marbles_after_street / 2

-- The given number of marbles left
def remaining_marbles : ℕ := 20

-- Proof statement
theorem find_initial_marbles (initial_marbles : ℕ) : 
  loses_to_sewer (loses_to_street initial_marbles) = remaining_marbles -> 
  initial_marbles = 100 :=
by
  sorry

end find_initial_marbles_l2386_238680


namespace parabola_conditions_l2386_238648

theorem parabola_conditions 
  (a b c : ℝ) 
  (ha : a < 0) 
  (hb : b = 2 * a) 
  (hc : c = -3 * a) 
  (hA : a * (-3)^2 + b * (-3) + c = 0) 
  (hB : a * (1)^2 + b * (1) + c = 0) : 
  (b^2 - 4 * a * c > 0) ∧ (3 * b + 2 * c = 0) :=
sorry

end parabola_conditions_l2386_238648


namespace quadratic_has_real_solutions_iff_l2386_238686

theorem quadratic_has_real_solutions_iff (m : ℝ) :
  ∃ x y : ℝ, (y = m * x + 3) ∧ (y = (3 * m - 2) * x ^ 2 + 5) ↔ 
  (m ≤ 12 - 8 * Real.sqrt 2) ∨ (m ≥ 12 + 8 * Real.sqrt 2) :=
by
  sorry

end quadratic_has_real_solutions_iff_l2386_238686


namespace find_sum_l2386_238609

-- Define the prime conditions
variables (P : ℝ) (SI15 SI12 : ℝ)

-- Assume conditions for the problem
axiom h1 : SI15 = P * 15 / 100 * 2
axiom h2 : SI12 = P * 12 / 100 * 2
axiom h3 : SI15 - SI12 = 840

-- Prove that P = 14000
theorem find_sum : P = 14000 :=
sorry

end find_sum_l2386_238609


namespace perimeter_triangle_ABC_l2386_238661

-- Define the conditions and statement
theorem perimeter_triangle_ABC 
  (r : ℝ) (AP PB altitude : ℝ) 
  (h1 : r = 30) 
  (h2 : AP = 26) 
  (h3 : PB = 32) 
  (h4 : altitude = 96) :
  (2 * (58 + 34.8)) = 185.6 :=
by
  sorry

end perimeter_triangle_ABC_l2386_238661


namespace find_cost_per_batch_l2386_238670

noncomputable def cost_per_tire : ℝ := 8
noncomputable def selling_price_per_tire : ℝ := 20
noncomputable def profit_per_tire : ℝ := 10.5
noncomputable def number_of_tires : ℕ := 15000

noncomputable def total_cost (C : ℝ) : ℝ := C + cost_per_tire * number_of_tires
noncomputable def total_revenue : ℝ := selling_price_per_tire * number_of_tires
noncomputable def total_profit : ℝ := profit_per_tire * number_of_tires

theorem find_cost_per_batch (C : ℝ) :
  total_profit = total_revenue - total_cost C → C = 22500 := by
  sorry

end find_cost_per_batch_l2386_238670
