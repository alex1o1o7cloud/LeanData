import Mathlib

namespace hives_needed_for_candles_l1875_187564

theorem hives_needed_for_candles (h : (3 : ℕ) * c = 12) : (96 : ℕ) / c = 24 :=
by
  sorry

end hives_needed_for_candles_l1875_187564


namespace solve_for_x_l1875_187575

theorem solve_for_x : (∃ x : ℝ, (1/2 - 1/3 = 1/x)) ↔ (x = 6) := sorry

end solve_for_x_l1875_187575


namespace sugar_left_in_grams_l1875_187598

theorem sugar_left_in_grams 
  (initial_ounces : ℝ) (spilled_ounces : ℝ) (conversion_factor : ℝ)
  (h_initial : initial_ounces = 9.8) (h_spilled : spilled_ounces = 5.2)
  (h_conversion : conversion_factor = 28.35) :
  (initial_ounces - spilled_ounces) * conversion_factor = 130.41 := 
by
  sorry

end sugar_left_in_grams_l1875_187598


namespace yearly_profit_l1875_187500

variable (num_subletters : ℕ) (rent_per_subletter_per_month rent_per_month : ℕ)

theorem yearly_profit (h1 : num_subletters = 3)
                     (h2 : rent_per_subletter_per_month = 400)
                     (h3 : rent_per_month = 900) :
  12 * (num_subletters * rent_per_subletter_per_month - rent_per_month) = 3600 :=
by
  sorry

end yearly_profit_l1875_187500


namespace large_cube_side_length_l1875_187509

theorem large_cube_side_length (s1 s2 s3 : ℝ) (h1 : s1 = 1) (h2 : s2 = 6) (h3 : s3 = 8) : 
  ∃ s_large : ℝ, s_large^3 = s1^3 + s2^3 + s3^3 ∧ s_large = 9 := 
by 
  use 9
  rw [h1, h2, h3]
  norm_num

end large_cube_side_length_l1875_187509


namespace maximum_value_of_x_plus_2y_l1875_187587

theorem maximum_value_of_x_plus_2y (x y : ℝ) (h : x^2 - 2 * x + 4 * y = 5) : ∃ m, m = x + 2 * y ∧ m ≤ 9/2 := by
  sorry

end maximum_value_of_x_plus_2y_l1875_187587


namespace counterexample_to_prime_statement_l1875_187566

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ is_prime n

theorem counterexample_to_prime_statement 
  (n : ℕ) 
  (h_n_composite : is_composite n) 
  (h_n_minus_3_not_prime : ¬ is_prime (n - 3)) : 
  n = 18 ∨ n = 24 :=
by 
  sorry

end counterexample_to_prime_statement_l1875_187566


namespace one_half_percent_as_decimal_l1875_187553

def percent_to_decimal (x : ℚ) := x / 100

theorem one_half_percent_as_decimal : percent_to_decimal (1 / 2) = 0.005 := 
by
  sorry

end one_half_percent_as_decimal_l1875_187553


namespace find_number_l1875_187545

theorem find_number (x : ℕ) (h : x / 4 + 3 = 5) : x = 8 :=
by sorry

end find_number_l1875_187545


namespace tom_final_payment_l1875_187583

noncomputable def cost_of_fruit (kg: ℝ) (rate_per_kg: ℝ) := kg * rate_per_kg

noncomputable def total_bill := 
  cost_of_fruit 15.3 1.85 + cost_of_fruit 12.7 2.45 + cost_of_fruit 10.5 3.20 + cost_of_fruit 6.2 4.50

noncomputable def discount (bill: ℝ) := 0.10 * bill

noncomputable def discounted_total (bill: ℝ) := bill - discount bill

noncomputable def sales_tax (amount: ℝ) := 0.06 * amount

noncomputable def final_amount (bill: ℝ) := discounted_total bill + sales_tax (discounted_total bill)

theorem tom_final_payment : final_amount total_bill = 115.36 :=
  sorry

end tom_final_payment_l1875_187583


namespace largest_divisor_of_n_l1875_187569

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 72 ∣ n^2) : 12 ∣ n :=
by
  sorry

end largest_divisor_of_n_l1875_187569


namespace unique_solution_to_equation_l1875_187548

theorem unique_solution_to_equation (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^y - y = 2005) : x = 1003 ∧ y = 1 :=
by
  sorry

end unique_solution_to_equation_l1875_187548


namespace problem_condition_necessary_and_sufficient_l1875_187540

theorem problem_condition_necessary_and_sufficient (a b : ℝ) (h : a * b > 0) :
  (a > b) ↔ (1 / a < 1 / b) :=
sorry

end problem_condition_necessary_and_sufficient_l1875_187540


namespace fraction_addition_l1875_187522

/--
The value of 2/5 + 1/3 is 11/15.
-/
theorem fraction_addition :
  (2 / 5 : ℚ) + (1 / 3) = 11 / 15 := 
sorry

end fraction_addition_l1875_187522


namespace range_of_a_l1875_187533

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x ≤ f a (x + ε))) → a ≤ -1/4 :=
sorry

end range_of_a_l1875_187533


namespace sector_max_area_l1875_187543

theorem sector_max_area (r : ℝ) (α : ℝ) (S : ℝ) :
  (0 < r ∧ r < 10) ∧ (2 * r + r * α = 20) ∧ (S = (1 / 2) * r * (r * α)) →
  (α = 2 ∧ S = 25) :=
by
  sorry

end sector_max_area_l1875_187543


namespace tenth_term_arithmetic_sequence_l1875_187502

theorem tenth_term_arithmetic_sequence (a d : ℕ) 
  (h1 : a + 2 * d = 10) 
  (h2 : a + 5 * d = 16) : 
  a + 9 * d = 24 := 
by 
  sorry

end tenth_term_arithmetic_sequence_l1875_187502


namespace yolanda_walking_rate_correct_l1875_187526

-- Definitions and conditions
def distance_XY : ℕ := 65
def bobs_walking_rate : ℕ := 7
def bobs_distance_when_met : ℕ := 35
def yolanda_start_time (t: ℕ) : ℕ := t + 1 -- Yolanda starts walking 1 hour earlier

-- Yolanda's walking rate calculation
def yolandas_walking_rate : ℕ := 5

theorem yolanda_walking_rate_correct { time_bob_walked : ℕ } 
  (h1 : distance_XY = 65)
  (h2 : bobs_walking_rate = 7)
  (h3 : bobs_distance_when_met = 35) 
  (h4 : time_bob_walked = bobs_distance_when_met / bobs_walking_rate)
  (h5 : yolanda_start_time time_bob_walked = 6) -- since bob walked 5 hours, yolanda walked 6 hours
  (h6 : distance_XY - bobs_distance_when_met = 30) :
  yolandas_walking_rate = ((distance_XY - bobs_distance_when_met) / yolanda_start_time time_bob_walked) := 
sorry

end yolanda_walking_rate_correct_l1875_187526


namespace non_obtuse_triangle_range_l1875_187576

noncomputable def range_of_2a_over_c (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) : Set ℝ :=
  {x | ∃ (a b c A : ℝ), x = (2 * a) / c ∧ 1 < x ∧ x ≤ 4}

theorem non_obtuse_triangle_range (a b c A C : ℝ) (h1 : B = π / 3) (h2 : A + C = 2 * π / 3) (h3 : π / 6 < C ∧ C ≤ π / 2) :
  (2 * a) / c ∈ range_of_2a_over_c a b c A C h1 h2 h3 := 
sorry

end non_obtuse_triangle_range_l1875_187576


namespace exists_valid_numbers_l1875_187541

noncomputable def sum_of_numbers_is_2012_using_two_digits : Prop :=
  ∃ (a b c d : ℕ), (a < 1000) ∧ (b < 1000) ∧ (c < 1000) ∧ (d < 1000) ∧ 
                    (∀ n ∈ [a, b, c, d], ∃ x y, (x ≠ y) ∧ ((∀ d ∈ [n / 100 % 10, n / 10 % 10, n % 10], d = x ∨ d = y))) ∧
                    (a + b + c + d = 2012)

theorem exists_valid_numbers : sum_of_numbers_is_2012_using_two_digits :=
  sorry

end exists_valid_numbers_l1875_187541


namespace jon_percentage_increase_l1875_187578

def initial_speed : ℝ := 80
def trainings : ℕ := 4
def weeks_per_training : ℕ := 4
def speed_increase_per_week : ℝ := 1

theorem jon_percentage_increase :
  let total_weeks := trainings * weeks_per_training
  let total_increase := total_weeks * speed_increase_per_week
  let final_speed := initial_speed + total_increase
  let percentage_increase := (total_increase / initial_speed) * 100
  percentage_increase = 20 :=
by
  sorry

end jon_percentage_increase_l1875_187578


namespace squares_difference_l1875_187580

theorem squares_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 :=
by sorry

end squares_difference_l1875_187580


namespace syllogism_arrangement_l1875_187593

theorem syllogism_arrangement : 
  (∀ n : ℕ, Odd n → ¬ (n % 2 = 0)) → 
  Odd 2013 → 
  (¬ (2013 % 2 = 0)) :=
by
  intros h1 h2
  exact h1 2013 h2

end syllogism_arrangement_l1875_187593


namespace jerome_classmates_count_l1875_187532

theorem jerome_classmates_count (C F : ℕ) (h1 : F = C / 2) (h2 : 33 = C + F + 3) : C = 20 :=
by
  sorry

end jerome_classmates_count_l1875_187532


namespace midpoint_sum_of_coordinates_l1875_187501

theorem midpoint_sum_of_coordinates
  (M : ℝ × ℝ) (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hmx : (C.1 + D.1) / 2 = M.1)
  (hmy : (C.2 + D.2) / 2 = M.2)
  (hM : M = (3, 5))
  (hC : C = (5, 3)) :
  D.1 + D.2 = 8 :=
by
  sorry

end midpoint_sum_of_coordinates_l1875_187501


namespace seven_large_power_mod_seventeen_l1875_187581

theorem seven_large_power_mod_seventeen :
  (7 : ℤ)^1985 % 17 = 7 :=
by
  have h1 : (7 : ℤ)^2 % 17 = 15 := sorry
  have h2 : (7 : ℤ)^4 % 17 = 16 := sorry
  have h3 : (7 : ℤ)^8 % 17 = 1 := sorry
  have h4 : 1985 = 8 * 248 + 1 := sorry
  sorry

end seven_large_power_mod_seventeen_l1875_187581


namespace negation_of_proposition_l1875_187588

theorem negation_of_proposition (x : ℝ) : 
  ¬ (|x| < 2 → x < 2) ↔ (|x| ≥ 2 → x ≥ 2) :=
sorry

end negation_of_proposition_l1875_187588


namespace boat_travel_time_difference_l1875_187521

noncomputable def travel_time_difference (v : ℝ) : ℝ :=
  let d := 90
  let t_downstream := 2.5191640969412834
  let t_upstream := d / (v - 3)
  t_upstream - t_downstream

theorem boat_travel_time_difference :
  ∃ v : ℝ, travel_time_difference v = 0.5088359030587166 := 
by
  sorry

end boat_travel_time_difference_l1875_187521


namespace joan_change_received_l1875_187592

theorem joan_change_received :
  let cat_toy_cost := 8.77
  let cage_cost := 10.97
  let payment := 20.00
  let total_cost := cat_toy_cost + cage_cost
  let change_received := payment - total_cost
  change_received = 0.26 :=
by
  sorry

end joan_change_received_l1875_187592


namespace power_complex_l1875_187591

theorem power_complex (a b : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : -64 = (-4)^3) (h3 : (a^b)^((3:ℝ) / 2) = a^(b * ((3:ℝ) / 2))) (h4 : (-4:ℂ)^(1/2) = 2 * i) :
  (↑(-64):ℂ) ^ (3/2) = 512 * i :=
by
  sorry

end power_complex_l1875_187591


namespace remainder_when_98_mul_102_divided_by_11_l1875_187571

theorem remainder_when_98_mul_102_divided_by_11 :
  (98 * 102) % 11 = 1 :=
by
  sorry

end remainder_when_98_mul_102_divided_by_11_l1875_187571


namespace ellipse_sum_l1875_187586

noncomputable def h : ℝ := 3
noncomputable def k : ℝ := 0
noncomputable def a : ℝ := 5
noncomputable def b : ℝ := Real.sqrt 21
noncomputable def F_1 : (ℝ × ℝ) := (1, 0)
noncomputable def F_2 : (ℝ × ℝ) := (5, 0)

theorem ellipse_sum :
  (F_1 = (1, 0)) → 
  (F_2 = (5, 0)) →
  (∀ P : (ℝ × ℝ), (Real.sqrt ((P.1 - F_1.1)^2 + (P.2 - F_1.2)^2) + Real.sqrt ((P.1 - F_2.1)^2 + (P.2 - F_2.2)^2) = 10)) →
  (h + k + a + b = 8 + Real.sqrt 21) :=
by
  intros
  sorry

end ellipse_sum_l1875_187586


namespace mod_congruence_l1875_187596

theorem mod_congruence (N : ℕ) (hN : N > 1) (h1 : 69 % N = 90 % N) (h2 : 90 % N = 125 % N) : 81 % N = 4 := 
by {
    sorry
}

end mod_congruence_l1875_187596


namespace stadium_ticket_price_l1875_187513

theorem stadium_ticket_price
  (original_price : ℝ)
  (decrease_rate : ℝ)
  (increase_rate : ℝ)
  (new_price : ℝ) 
  (h1 : original_price = 400)
  (h2 : decrease_rate = 0.2)
  (h3 : increase_rate = 0.05) 
  (h4 : (original_price * (1 + increase_rate) / (1 - decrease_rate)) = new_price) :
  new_price = 525 := 
by
  -- Proof omitted for this task.
  sorry

end stadium_ticket_price_l1875_187513


namespace mabel_total_tomatoes_l1875_187508

def tomatoes_first_plant : ℕ := 12

def tomatoes_second_plant : ℕ := (2 * tomatoes_first_plant) - 6

def tomatoes_combined_first_two : ℕ := tomatoes_first_plant + tomatoes_second_plant

def tomatoes_third_plant : ℕ := tomatoes_combined_first_two / 2

def tomatoes_each_fourth_fifth_plant : ℕ := 3 * tomatoes_combined_first_two

def tomatoes_combined_fourth_fifth : ℕ := 2 * tomatoes_each_fourth_fifth_plant

def tomatoes_each_sixth_seventh_plant : ℕ := (3 * tomatoes_combined_first_two) / 2

def tomatoes_combined_sixth_seventh : ℕ := 2 * tomatoes_each_sixth_seventh_plant

def total_tomatoes : ℕ := tomatoes_first_plant + tomatoes_second_plant + tomatoes_third_plant + tomatoes_combined_fourth_fifth + tomatoes_combined_sixth_seventh

theorem mabel_total_tomatoes : total_tomatoes = 315 :=
by
  sorry

end mabel_total_tomatoes_l1875_187508


namespace relationship_among_abc_l1875_187518

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

theorem relationship_among_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_among_abc_l1875_187518


namespace parallelogram_area_ratio_l1875_187527

theorem parallelogram_area_ratio (
  AB CD BC AD AP CQ BP DQ: ℝ)
  (h1 : AB = 13)
  (h2 : CD = 13)
  (h3 : BC = 15)
  (h4 : AD = 15)
  (h5 : AP = 10 / 3)
  (h6 : CQ = 10 / 3)
  (h7 : BP = 29 / 3)
  (h8 : DQ = 29 / 3)
  : ((area_APDQ / area_BPCQ) = 19) :=
sorry

end parallelogram_area_ratio_l1875_187527


namespace find_m_same_foci_l1875_187550

theorem find_m_same_foci (m : ℝ) 
(hyperbola_eq : ∃ x y : ℝ, x^2 - y^2 = m) 
(ellipse_eq : ∃ x y : ℝ, 2 * x^2 + 3 * y^2 = m + 1) 
(same_foci : ∀ a b : ℝ, (x^2 - y^2 = m) ∧ (2 * x^2 + 3 * y^2 = m + 1) → 
               let c_ellipse := (m + 1) / 6
               let c_hyperbola := 2 * m
               c_ellipse = c_hyperbola ) : 
m = 1 / 11 := 
sorry

end find_m_same_foci_l1875_187550


namespace correct_system_of_equations_l1875_187565

noncomputable def system_of_equations (x y : ℝ) : Prop :=
x + y = 150 ∧ 3 * x + (1 / 3) * y = 210

theorem correct_system_of_equations : ∃ x y : ℝ, system_of_equations x y :=
sorry

end correct_system_of_equations_l1875_187565


namespace problem1_problem2_problem3_problem4_l1875_187590

theorem problem1 : (-20 + (-14) - (-18) - 13) = -29 := by
  sorry

theorem problem2 : (-6 * (-2) / (1 / 8)) = 96 := by
  sorry

theorem problem3 : (-24 * (-3 / 4 - 5 / 6 + 7 / 8)) = 17 := by
  sorry

theorem problem4 : (-1^4 - (1 - 0.5) * (1 / 3) * (-3)^2) = -5 / 2 := by
  sorry

end problem1_problem2_problem3_problem4_l1875_187590


namespace fraction_inequality_fraction_inequality_equality_case_l1875_187570

variables {α β a b : ℝ}

theorem fraction_inequality 
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b) ≤ (β / α + α / β) :=
sorry

-- Additional equality statement
theorem fraction_inequality_equality_case
  (h_alpha_beta_pos : 0 < α ∧ 0 < β)
  (h_bounds_a : α ≤ a ∧ a ≤ β)
  (h_bounds_b : α ≤ b ∧ b ≤ β) :
  (b / a + a / b = β / α + α / β) ↔ (a = α ∧ b = β ∨ a = β ∧ b = α) :=
sorry

end fraction_inequality_fraction_inequality_equality_case_l1875_187570


namespace technicians_count_l1875_187557

theorem technicians_count 
    (total_workers : ℕ) (avg_salary_all : ℕ) (avg_salary_technicians : ℕ) (avg_salary_rest : ℕ)
    (h_workers : total_workers = 28) (h_avg_all : avg_salary_all = 8000) 
    (h_avg_tech : avg_salary_technicians = 14000) (h_avg_rest : avg_salary_rest = 6000) : 
    ∃ T R : ℕ, T + R = total_workers ∧ (avg_salary_technicians * T + avg_salary_rest * R = avg_salary_all * total_workers) ∧ T = 7 :=
by
  sorry

end technicians_count_l1875_187557


namespace most_irregular_acute_triangle_l1875_187514

theorem most_irregular_acute_triangle :
  ∃ (α β γ : ℝ), α ≤ β ∧ β ≤ γ ∧ γ ≤ (90:ℝ) ∧ 
  ((β - α ≤ 15) ∧ (γ - β ≤ 15) ∧ (90 - γ ≤ 15)) ∧
  (α + β + γ = 180) ∧ 
  (α = 45 ∧ β = 60 ∧ γ = 75) := sorry

end most_irregular_acute_triangle_l1875_187514


namespace plan1_has_higher_expected_loss_l1875_187559

noncomputable def prob_minor_flooding : ℝ := 0.2
noncomputable def prob_major_flooding : ℝ := 0.05
noncomputable def cost_plan1 : ℝ := 4000
noncomputable def loss_major_plan1 : ℝ := 30000
noncomputable def loss_minor_plan2 : ℝ := 15000
noncomputable def loss_major_plan2 : ℝ := 30000

noncomputable def expected_loss_plan1 : ℝ :=
  (loss_major_plan1 * prob_major_flooding) + (cost_plan1 * prob_minor_flooding) + cost_plan1

noncomputable def expected_loss_plan2 : ℝ :=
  (loss_major_plan2 * prob_major_flooding) + (loss_minor_plan2 * prob_minor_flooding)

theorem plan1_has_higher_expected_loss : expected_loss_plan1 > expected_loss_plan2 :=
by
  sorry

end plan1_has_higher_expected_loss_l1875_187559


namespace inequality_of_power_sums_l1875_187558

variable (a b c : ℝ)

theorem inequality_of_power_sums (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a < b + c) (h5 : b < c + a) (h6 : c < a + b) :
  a^4 + b^4 + c^4 < 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) := sorry

end inequality_of_power_sums_l1875_187558


namespace coefficients_sum_correct_l1875_187567

noncomputable def poly_expr (x : ℝ) : ℝ := (x + 2)^4

def coefficients_sum (a a_1 a_2 a_3 a_4 : ℝ) : ℝ :=
  a_1 + a_2 + a_3 + a_4

theorem coefficients_sum_correct (a a_1 a_2 a_3 a_4 : ℝ) :
  poly_expr 1 = a_4 * 1 ^ 4 + a_3 * 1 ^ 3 + a_2 * 1 ^ 2 + a_1 * 1 + a →
  a = 16 → coefficients_sum a a_1 a_2 a_3 a_4 = 65 :=
by
  intro h₁ h₂
  sorry

end coefficients_sum_correct_l1875_187567


namespace find_negative_integer_l1875_187594

theorem find_negative_integer (N : ℤ) (h : N^2 + N = -12) : N = -4 := 
by sorry

end find_negative_integer_l1875_187594


namespace initial_population_l1875_187551

theorem initial_population (P : ℝ) (h1 : ∀ n : ℕ, n = 2 → P * (0.7 ^ n) = 3920) : P = 8000 := by
  sorry

end initial_population_l1875_187551


namespace find_integer_a_l1875_187577

theorem find_integer_a (x d e a : ℤ) :
  ((x - a)*(x - 8) - 3 = (x + d)*(x + e)) → (a = 6) :=
by
  sorry

end find_integer_a_l1875_187577


namespace shelby_rain_drive_time_eq_3_l1875_187519

-- Definitions as per the conditions
def distance (v : ℝ) (t : ℝ) : ℝ := v * t
def total_distance := 24 -- in miles
def total_time := 50 / 60 -- in hours (converted to minutes)
def non_rainy_speed := 30 / 60 -- in miles per minute
def rainy_speed := 20 / 60 -- in miles per minute

-- Lean statement of the proof problem
theorem shelby_rain_drive_time_eq_3 :
  ∃ x : ℝ,
  (distance non_rainy_speed (total_time - x / 60) + distance rainy_speed (x / 60) = total_distance)
  ∧ (0 ≤ x) ∧ (x ≤ total_time * 60) →
  x = 3 := 
sorry

end shelby_rain_drive_time_eq_3_l1875_187519


namespace solve_for_k_l1875_187511

-- Define the hypotheses as Lean statements
theorem solve_for_k (x k : ℝ) (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
by {
  sorry
}

end solve_for_k_l1875_187511


namespace moon_temp_difference_l1875_187539

def temp_difference (T_day T_night : ℤ) : ℤ := T_day - T_night

theorem moon_temp_difference :
  temp_difference 127 (-183) = 310 :=
by
  sorry

end moon_temp_difference_l1875_187539


namespace systematic_sampling_l1875_187549

-- Define the conditions
def total_products : ℕ := 100
def selected_products (n : ℕ) : ℕ := 3 + 10 * n
def is_systematic (f : ℕ → ℕ) : Prop :=
  ∃ k b, ∀ n, f n = b + k * n

-- Theorem to prove that the selection method is systematic sampling
theorem systematic_sampling : is_systematic selected_products :=
  sorry

end systematic_sampling_l1875_187549


namespace num_valid_permutations_l1875_187538

theorem num_valid_permutations : 
  let digits := [2, 0, 2, 3]
  let num_2 := 2
  let total_permutations := Nat.factorial 4 / (Nat.factorial num_2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_start_2 := Nat.factorial 3
  let valid_start_3 := Nat.factorial 3 / Nat.factorial 2
  total_permutations = 12 ∧ valid_start_2 = 6 ∧ valid_start_3 = 3 ∧ (valid_start_2 + valid_start_3 = 9) := 
by
  sorry

end num_valid_permutations_l1875_187538


namespace absolute_difference_avg_median_l1875_187546

theorem absolute_difference_avg_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((3 + 4 * a + 2 * b) / 4) - (a + b / 2 + 1)| = 1 / 4 :=
by
  sorry

end absolute_difference_avg_median_l1875_187546


namespace part_I_part_II_l1875_187589

noncomputable def f (x m : ℝ) : ℝ := |3 * x + m|
noncomputable def g (x m : ℝ) : ℝ := f x m - 2 * |x - 1|

theorem part_I (m : ℝ) : (∀ x : ℝ, (f x m - m ≤ 9) ↔ (-1 ≤ x ∧ x ≤ 3)) → m = -3 :=
by
  sorry

theorem part_II (m : ℝ) (h : m > 0) : (∃ A B C : ℝ × ℝ, 
  let A := (-m-2, 0)
  let B := ((2-m)/5, 0)
  let C := (-m/3, -2*m/3-2)
  let Area : ℝ := 1/2 * |(B.1 - A.1) * (C.2 - 0) - (B.2 - A.2) * (C.1 - A.1)|
  Area > 60 ) → m > 12 :=
by
  sorry

end part_I_part_II_l1875_187589


namespace manuscript_typing_total_cost_is_1400_l1875_187561

-- Defining the variables and constants based on given conditions
def cost_first_time_per_page := 10
def cost_revision_per_page := 5
def total_pages := 100
def pages_revised_once := 20
def pages_revised_twice := 30
def pages_no_revision := total_pages - pages_revised_once - pages_revised_twice

-- Calculations based on the given conditions
def cost_first_time :=
  total_pages * cost_first_time_per_page

def cost_revised_once :=
  pages_revised_once * cost_revision_per_page

def cost_revised_twice :=
  pages_revised_twice * cost_revision_per_page * 2

def total_cost :=
  cost_first_time + cost_revised_once + cost_revised_twice

-- Prove that the total cost equals the calculated value
theorem manuscript_typing_total_cost_is_1400 :
  total_cost = 1400 := by
  sorry

end manuscript_typing_total_cost_is_1400_l1875_187561


namespace identify_radioactive_balls_l1875_187552

theorem identify_radioactive_balls (balls : Fin 11 → Bool) (measure : (Finset (Fin 11)) → Bool) :
  (∃ (t1 t2 : Fin 11), ¬ t1 = t2 ∧ balls t1 = true ∧ balls t2 = true) →
  (∃ (pairs : List (Finset (Fin 11))), pairs.length ≤ 7 ∧
    ∀ t1 t2, t1 ≠ t2 ∧ balls t1 = true ∧ balls t2 = true →
      ∃ pair ∈ pairs, measure pair = true ∧ (t1 ∈ pair ∨ t2 ∈ pair)) :=
by
  sorry

end identify_radioactive_balls_l1875_187552


namespace average_marks_combined_l1875_187523

theorem average_marks_combined (avg1 : ℝ) (students1 : ℕ) (avg2 : ℝ) (students2 : ℕ) :
  avg1 = 30 → students1 = 30 → avg2 = 60 → students2 = 50 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2) = 48.75 := 
by
  intros h_avg1 h_students1 h_avg2 h_students2
  sorry

end average_marks_combined_l1875_187523


namespace lcm_and_sum_of_14_21_35_l1875_187556

def lcm_of_numbers_and_sum (a b c : ℕ) : ℕ × ℕ :=
  (Nat.lcm (Nat.lcm a b) c, a + b + c)

theorem lcm_and_sum_of_14_21_35 :
  lcm_of_numbers_and_sum 14 21 35 = (210, 70) :=
  sorry

end lcm_and_sum_of_14_21_35_l1875_187556


namespace abc_value_l1875_187563

theorem abc_value 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (hab : a * b = 24) 
  (hac : a * c = 40) 
  (hbc : b * c = 60) : 
  a * b * c = 240 := 
by sorry

end abc_value_l1875_187563


namespace initial_oranges_in_box_l1875_187544

theorem initial_oranges_in_box (o_taken_out o_left_in_box : ℕ) (h1 : o_taken_out = 35) (h2 : o_left_in_box = 20) :
  o_taken_out + o_left_in_box = 55 := 
by
  sorry

end initial_oranges_in_box_l1875_187544


namespace range_of_a_l1875_187530

-- Let us define the problem conditions and statement in Lean
theorem range_of_a
  (a : ℝ)
  (h : ∀ x y : ℝ, x < y → (3 - a)^x > (3 - a)^y) :
  2 < a ∧ a < 3 :=
sorry

end range_of_a_l1875_187530


namespace wheel_rpm_l1875_187529

noncomputable def radius : ℝ := 175
noncomputable def speed_kmh : ℝ := 66
noncomputable def speed_cmm := speed_kmh * 100000 / 60 -- convert from km/h to cm/min
noncomputable def circumference := 2 * Real.pi * radius -- circumference of the wheel
noncomputable def rpm := speed_cmm / circumference -- revolutions per minute

theorem wheel_rpm : rpm = 1000 := by
  sorry

end wheel_rpm_l1875_187529


namespace value_of_y_at_3_l1875_187542

-- Define the function
def f (x : ℕ) : ℕ := 2 * x^2 + 1

-- Prove that when x = 3, y = 19
theorem value_of_y_at_3 : f 3 = 19 :=
by
  -- Provide the definition and conditions
  let x := 3
  let y := f x
  have h : y = 2 * x^2 + 1 := rfl
  -- State the actual proof could go here
  sorry

end value_of_y_at_3_l1875_187542


namespace min_sum_a_b_l1875_187597

theorem min_sum_a_b (a b : ℝ) (h_cond: 1/a + 4/b = 1) (a_pos : 0 < a) (b_pos : 0 < b) : 
  a + b ≥ 9 :=
sorry

end min_sum_a_b_l1875_187597


namespace fourth_term_expansion_l1875_187520

def binomial_term (n r : ℕ) (a b : ℚ) : ℚ :=
  (Nat.descFactorial n r) / (Nat.factorial r) * a^(n - r) * b^r

theorem fourth_term_expansion (x : ℚ) (hx : x ≠ 0) : 
  binomial_term 6 3 2 (-(1 / (x^(1/3)))) = (-160 / x) :=
by
  sorry

end fourth_term_expansion_l1875_187520


namespace identity_equality_l1875_187573

theorem identity_equality (a b m n x y : ℝ) :
  ((a^2 + b^2) * (m^2 + n^2) * (x^2 + y^2)) =
  ((a * n * y - a * m * x - b * m * y + b * n * x)^2 + (a * m * y + a * n * x + b * m * x - b * n * y)^2) :=
by
  sorry

end identity_equality_l1875_187573


namespace maximum_n_l1875_187505

/-- Definition of condition (a): For any three people, there exist at least two who know each other. -/
def condition_a (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 3 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), G.Adj a b

/-- Definition of condition (b): For any four people, there exist at least two who do not know each other. -/
def condition_b (G : SimpleGraph V) : Prop :=
  ∀ (s : Finset V), s.card = 4 → ∃ (a b : V) (ha : a ∈ s) (hb : b ∈ s), ¬ G.Adj a b

theorem maximum_n (G : SimpleGraph V) [Fintype V] (h1 : condition_a G) (h2 : condition_b G) : 
  Fintype.card V ≤ 8 :=
by
  sorry

end maximum_n_l1875_187505


namespace total_invested_expression_l1875_187507

variables (x y T : ℝ)

axiom annual_income_exceed_65 : 0.10 * x - 0.08 * y = 65
axiom total_invested_is_T : x + y = T

theorem total_invested_expression :
  T = 1.8 * y + 650 :=
sorry

end total_invested_expression_l1875_187507


namespace distance_between_parallel_lines_l1875_187517

theorem distance_between_parallel_lines
  (l1 : ∀ (x y : ℝ), 2*x + y + 1 = 0)
  (l2 : ∀ (x y : ℝ), 4*x + 2*y - 1 = 0) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 5 / 10 := by
  sorry

end distance_between_parallel_lines_l1875_187517


namespace balance_blue_balls_l1875_187510

variables (G B Y W : ℝ)

-- Definitions based on conditions
def condition1 : Prop := 3 * G = 6 * B
def condition2 : Prop := 2 * Y = 5 * B
def condition3 : Prop := 6 * B = 4 * W

-- Statement of the problem
theorem balance_blue_balls (h1 : condition1 G B) (h2 : condition2 Y B) (h3 : condition3 B W) :
  4 * G + 2 * Y + 2 * W = 16 * B :=
sorry

end balance_blue_balls_l1875_187510


namespace telephone_number_A_value_l1875_187562

theorem telephone_number_A_value :
  ∃ A B C D E F G H I J : ℕ,
    A > B ∧ B > C ∧
    D > E ∧ E > F ∧
    G > H ∧ H > I ∧ I > J ∧
    (D = E + 1) ∧ (E = F + 1) ∧
    G + H + I + J = 20 ∧
    A + B + C = 15 ∧
    A = 8 := sorry

end telephone_number_A_value_l1875_187562


namespace factor_expression_equals_one_l1875_187554

theorem factor_expression_equals_one (a b c : ℝ) :
  ((a^2 - b^2)^2 + (b^2 - c^2)^2 + (c^2 - a^2)^2) / ((a - b)^2 + (b - c)^2 + (c - a)^2) = 1 :=
by
  sorry

end factor_expression_equals_one_l1875_187554


namespace functional_equation_solution_l1875_187516

noncomputable def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * x + f x * f y) = x * f (x + y)

theorem functional_equation_solution (f : ℝ → ℝ) :
  func_equation f →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end functional_equation_solution_l1875_187516


namespace product_of_two_numbers_l1875_187535

-- Definitions and conditions
def HCF (a b : ℕ) : ℕ := 9
def LCM (a b : ℕ) : ℕ := 200

-- Theorem statement
theorem product_of_two_numbers (a b : ℕ) (H₁ : HCF a b = 9) (H₂ : LCM a b = 200) : a * b = 1800 :=
by
  -- Injecting HCF and LCM conditions into the problem
  sorry

end product_of_two_numbers_l1875_187535


namespace shared_vertex_angle_of_triangle_and_square_l1875_187536

theorem shared_vertex_angle_of_triangle_and_square (α β γ δ ε ζ η θ : ℝ) :
  (α = 60 ∧ β = 60 ∧ γ = 60 ∧ δ = 90 ∧ ε = 90 ∧ ζ = 90 ∧ η = 90 ∧ θ = 90) →
  θ = 90 :=
by
  sorry

end shared_vertex_angle_of_triangle_and_square_l1875_187536


namespace meatballs_left_l1875_187599
open Nat

theorem meatballs_left (meatballs_per_plate sons : ℕ)
  (hp : meatballs_per_plate = 3) 
  (hs : sons = 3) 
  (fraction_eaten : ℚ)
  (hf : fraction_eaten = 2 / 3): 
  (meatballs_per_plate - meatballs_per_plate * fraction_eaten) * sons = 3 := by
  -- Placeholder proof; the details would be filled in by a full proof.
  sorry

end meatballs_left_l1875_187599


namespace order_scores_l1875_187584

theorem order_scores
  (J K M Q S : ℕ)
  (h1 : J ≥ Q) (h2 : J ≥ M) (h3 : J ≥ S) (h4 : J ≥ K)
  (h5 : M > Q ∨ M > S ∨ M > K)
  (h6 : K < S) (h7 : S < J) :
  K < S ∧ S < M ∧ M < Q :=
by
  sorry

end order_scores_l1875_187584


namespace prove_jimmy_is_2_determine_rachel_age_l1875_187531

-- Define the conditions of the problem
variables (a b c r1 r2 : ℤ)

-- Condition 1: Rachel's age and Jimmy's age are roots of the quadratic equation
def is_root (p : ℤ → ℤ) (x : ℤ) : Prop := p x = 0

def quadratic_eq (x : ℤ) : ℤ := a * x^2 + b * x + c

-- Condition 2: Sum of the coefficients is a prime number
def sum_of_coefficients_is_prime : Prop :=
  Nat.Prime (a + b + c).natAbs

-- Condition 3: Substituting Rachel’s age into the quadratic equation gives -55
def substitute_rachel_is_minus_55 (r : ℤ) : Prop :=
  quadratic_eq a b c r = -55

-- Question 1: Prove Jimmy is 2 years old
theorem prove_jimmy_is_2 (h1 : is_root (quadratic_eq a b c) r1)
                           (h2 : is_root (quadratic_eq a b c) r2)
                           (h3 : sum_of_coefficients_is_prime a b c)
                           (h4 : substitute_rachel_is_minus_55 a b c r1) :
  r2 = 2 :=
sorry

-- Question 2: Determine Rachel's age
theorem determine_rachel_age (h1 : is_root (quadratic_eq a b c) r1)
                             (h2 : is_root (quadratic_eq a b c) r2)
                             (h3 : sum_of_coefficients_is_prime a b c)
                             (h4 : substitute_rachel_is_minus_55 a b c r1)
                             (h5 : r2 = 2) :
  r1 = 7 :=
sorry

end prove_jimmy_is_2_determine_rachel_age_l1875_187531


namespace geom_seq_inv_sum_eq_l1875_187503

noncomputable def geom_seq (a_1 r : ℚ) (n : ℕ) : ℚ := a_1 * r^n

theorem geom_seq_inv_sum_eq
    (a_1 r : ℚ)
    (h_sum : geom_seq a_1 r 0 + geom_seq a_1 r 1 + geom_seq a_1 r 2 + geom_seq a_1 r 3 = 15/8)
    (h_prod : geom_seq a_1 r 1 * geom_seq a_1 r 2 = -9/8) :
  1 / geom_seq a_1 r 0 + 1 / geom_seq a_1 r 1 + 1 / geom_seq a_1 r 2 + 1 / geom_seq a_1 r 3 = -5/3 :=
sorry

end geom_seq_inv_sum_eq_l1875_187503


namespace largest_possible_value_of_p_l1875_187525

theorem largest_possible_value_of_p (m n p : ℕ) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * m * n * p = (m + 2) * (n + 2) * (p + 2)) : p ≤ 130 :=
by
  sorry

end largest_possible_value_of_p_l1875_187525


namespace percent_markdown_l1875_187579

theorem percent_markdown (P S : ℝ) (h : S * 1.25 = P) : (P - S) / P * 100 = 20 := by
  sorry

end percent_markdown_l1875_187579


namespace simplify_and_evaluate_expression_l1875_187572

theorem simplify_and_evaluate_expression : 
  ∀ (x y : ℤ), x = -1 → y = 2 → -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end simplify_and_evaluate_expression_l1875_187572


namespace no_divisor_30_to_40_of_2_pow_28_minus_1_l1875_187534

theorem no_divisor_30_to_40_of_2_pow_28_minus_1 :
  ¬ ∃ n : ℕ, (30 ≤ n ∧ n ≤ 40 ∧ n ∣ (2^28 - 1)) :=
by
  sorry

end no_divisor_30_to_40_of_2_pow_28_minus_1_l1875_187534


namespace find_angle_D_l1875_187560

noncomputable def calculate_angle (A B C D : ℝ) : ℝ :=
  if (A + B = 180) ∧ (C = D) ∧ (A = 2 * D - 10) then D else 0

theorem find_angle_D (A B C D : ℝ) (h1: A + B = 180) (h2: C = D) (h3: A = 2 * D - 10) : D = 70 :=
by
  sorry

end find_angle_D_l1875_187560


namespace possible_values_a1_l1875_187506

def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum a

theorem possible_values_a1 {a : ℕ → ℤ} (h1 : ∀ n : ℕ, a n + a (n + 1) = 2 * n - 1)
  (h2 : ∃ k : ℕ, sequence_sum a k = 190 ∧ sequence_sum a (k + 1) = 190) :
  (a 0 = -20 ∨ a 0 = 19) :=
sorry

end possible_values_a1_l1875_187506


namespace mean_weight_players_l1875_187547

/-- Definitions for the weights of the players and proving the mean weight. -/
def weights : List ℕ := [62, 65, 70, 73, 73, 76, 78, 79, 81, 81, 82, 84, 87, 89, 89, 89, 90, 93, 95]

def mean (lst : List ℕ) : ℚ := (lst.sum : ℚ) / lst.length

theorem mean_weight_players : mean weights = 80.84 := by
  sorry

end mean_weight_players_l1875_187547


namespace share_of_C_l1875_187504

theorem share_of_C (A B C : ℝ) (h1 : A = (2/3) * B) (h2 : B = (1/4) * C) (h3 : A + B + C = 578) : 
  C = 408 :=
by
  -- Proof goes here
  sorry

end share_of_C_l1875_187504


namespace books_printed_l1875_187585

-- Definitions of the conditions
def book_length := 600
def pages_per_sheet := 8
def total_sheets := 150

-- The theorem to prove
theorem books_printed : (total_sheets * pages_per_sheet / book_length) = 2 := by
  sorry

end books_printed_l1875_187585


namespace find_number_l1875_187515

theorem find_number (x : ℕ) (n : ℕ) (h1 : x = 4) (h2 : x + n = 5) : n = 1 :=
by
  sorry

end find_number_l1875_187515


namespace directrix_of_parabola_l1875_187582

theorem directrix_of_parabola :
  ∀ (x y : ℝ), (y = (x^2 - 4 * x + 4) / 8) → y = -2 :=
sorry

end directrix_of_parabola_l1875_187582


namespace max_incircle_circumcircle_ratio_l1875_187595

theorem max_incircle_circumcircle_ratio (c : ℝ) (α : ℝ) 
  (hα : 0 < α ∧ α < π / 2) :
  let a := c * Real.cos α
  let b := c * Real.sin α
  let R := c / 2
  let r := (a + b - c) / 2
  (r / R <= Real.sqrt 2 - 1) :=
by
  sorry

end max_incircle_circumcircle_ratio_l1875_187595


namespace min_value_of_c_l1875_187512

theorem min_value_of_c (a b c : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_ineq1 : a < b) 
  (h_ineq2 : b < 2 * b) 
  (h_ineq3 : 2 * b < c)
  (h_unique_sol : ∃ x : ℝ, 3 * x + (|x - a| + |x - b| + |x - (2 * b)| + |x - c|) = 3000) :
  c = 502 := sorry

end min_value_of_c_l1875_187512


namespace passing_marks_l1875_187574

-- Define the conditions and prove P = 160 given these conditions
theorem passing_marks (T P : ℝ) (h1 : 0.40 * T = P - 40) (h2 : 0.60 * T = P + 20) : P = 160 :=
by
  sorry

end passing_marks_l1875_187574


namespace readers_both_l1875_187537

-- Definitions of the number of readers
def total_readers : ℕ := 150
def readers_science_fiction : ℕ := 120
def readers_literary_works : ℕ := 90

-- Statement of the proof problem
theorem readers_both :
  (readers_science_fiction + readers_literary_works - total_readers) = 60 :=
by
  -- Proof omitted
  sorry

end readers_both_l1875_187537


namespace linda_savings_l1875_187524

theorem linda_savings :
  let original_savings := 880
  let cost_of_tv := 220
  let amount_spent_on_furniture := original_savings - cost_of_tv
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings
  fraction_spent_on_furniture = 3 / 4 :=
by
  -- original savings
  let original_savings := 880
  -- cost of the TV
  let cost_of_tv := 220
  -- amount spent on furniture
  let amount_spent_on_furniture := original_savings - cost_of_tv
  -- fraction spent on furniture
  let fraction_spent_on_furniture := amount_spent_on_furniture / original_savings

  -- need to show that this fraction is 3/4
  sorry

end linda_savings_l1875_187524


namespace find_ordered_pair_l1875_187555

theorem find_ordered_pair : 
  ∃ (x y : ℚ), 7 * x = -5 - 3 * y ∧ 4 * x = 5 * y - 34 ∧
  x = -127 / 47 ∧ y = 218 / 47 :=
by
  sorry

end find_ordered_pair_l1875_187555


namespace repeating_decimal_to_fraction_l1875_187568

/--
Express \(2.\overline{06}\) as a reduced fraction, given that \(0.\overline{01} = \frac{1}{99}\)
-/
theorem repeating_decimal_to_fraction : 
  (0.01:ℚ) = 1 / 99 → (2.06:ℚ) = 68 / 33 := 
by 
  sorry 

end repeating_decimal_to_fraction_l1875_187568


namespace smallest_b_for_N_fourth_power_l1875_187528

theorem smallest_b_for_N_fourth_power : 
  ∃ (b : ℤ), (∀ n : ℤ, 7 * b^2 + 7 * b + 7 = n^4) ∧ b = 18 :=
by
  sorry

end smallest_b_for_N_fourth_power_l1875_187528
