import Mathlib

namespace max_value_of_x_plus_3y_l1047_104774

theorem max_value_of_x_plus_3y (x y : ℝ) (h : x^2 / 9 + y^2 = 1) : 
    ∃ θ : ℝ, x = 3 * Real.cos θ ∧ y = Real.sin θ ∧ (x + 3 * y) ≤ 3 * Real.sqrt 2 :=
by
  sorry

end max_value_of_x_plus_3y_l1047_104774


namespace find_certain_number_l1047_104732

theorem find_certain_number (x y : ℕ) (h1 : x = 19) (h2 : x + y = 36) :
  8 * x + 3 * y = 203 := by
  sorry

end find_certain_number_l1047_104732


namespace boys_without_notebooks_l1047_104751

/-
Given that:
1. There are 16 boys in Ms. Green's history class.
2. 20 students overall brought their notebooks to class.
3. 11 of the students who brought notebooks are girls.

Prove that the number of boys who did not bring their notebooks is 7.
-/

theorem boys_without_notebooks (total_boys : ℕ) (total_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (hb : total_boys = 16) (hn : total_notebooks = 20) (hg : girls_with_notebooks = 11) : 
  (total_boys - (total_notebooks - girls_with_notebooks) = 7) :=
by
  sorry

end boys_without_notebooks_l1047_104751


namespace largest_possible_pencils_in_each_package_l1047_104747

def ming_pencils : ℕ := 48
def catherine_pencils : ℕ := 36
def lucas_pencils : ℕ := 60

theorem largest_possible_pencils_in_each_package (d : ℕ) (h_ming: ming_pencils % d = 0) (h_catherine: catherine_pencils % d = 0) (h_lucas: lucas_pencils % d = 0) : d ≤ ming_pencils ∧ d ≤ catherine_pencils ∧ d ≤ lucas_pencils ∧ (∀ e, (ming_pencils % e = 0 ∧ catherine_pencils % e = 0 ∧ lucas_pencils % e = 0) → e ≤ d) → d = 12 :=
by 
  sorry

end largest_possible_pencils_in_each_package_l1047_104747


namespace smallest_positive_period_max_min_values_l1047_104724

noncomputable def f (x a : ℝ) : ℝ :=
  (Real.cos x) * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * Real.sin x ^ 2

theorem smallest_positive_period (a : ℝ) (h : f (Real.pi / 12) a = 0) : 
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) a = f x a) ∧ (∀ ε > 0, ε < T → ∃ y, y < T ∧ f y a ≠ f 0 a) := 
sorry

theorem max_min_values (a : ℝ) (h : f (Real.pi / 12) a = 0) :
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x a ≤ Real.sqrt 3) ∧ 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), -2 ≤ f x a) := 
sorry

end smallest_positive_period_max_min_values_l1047_104724


namespace tangent_line_at_P_l1047_104726

def tangent_line_eq (x y : ℝ) : ℝ := x - 2 * y + 1

theorem tangent_line_at_P (x y : ℝ) (h : x ^ 2 + y ^ 2 - 4 * x + 2 * y = 0 ∧ (x, y) = (1, 1)) :
    tangent_line_eq x y = 0 := 
sorry

end tangent_line_at_P_l1047_104726


namespace f_3_2_plus_f_5_1_l1047_104718

def f (a b : ℤ) : ℚ :=
  if a - b ≤ 2 then (a * b - a - 1) / (3 * a)
  else (a * b + b - 1) / (-3 * b)

theorem f_3_2_plus_f_5_1 :
  f 3 2 + f 5 1 = -13 / 9 :=
by
  sorry

end f_3_2_plus_f_5_1_l1047_104718


namespace total_attendance_l1047_104720

theorem total_attendance (first_concert : ℕ) (second_concert : ℕ) (third_concert : ℕ) :
  first_concert = 65899 →
  second_concert = first_concert + 119 →
  third_concert = 2 * second_concert →
  first_concert + second_concert + third_concert = 263953 :=
by
  intros h_first h_second h_third
  rw [h_first, h_second, h_third]
  sorry

end total_attendance_l1047_104720


namespace larger_cylinder_candies_l1047_104742

theorem larger_cylinder_candies (v₁ v₂ : ℝ) (c₁ c₂ : ℕ) (h₁ : v₁ = 72) (h₂ : c₁ = 30) (h₃ : v₂ = 216) (h₄ : (c₁ : ℝ)/v₁ = (c₂ : ℝ)/v₂) : c₂ = 90 := by
  -- v1 h1 h2 v2 c2 h4 are directly appearing in the conditions
  -- ratio h4 states the condition for densities to be the same 
  sorry

end larger_cylinder_candies_l1047_104742


namespace quadratic_c_over_b_l1047_104794

theorem quadratic_c_over_b :
  ∃ (b c : ℤ), (x^2 + 500 * x + 1000 = (x + b)^2 + c) ∧ (c / b = -246) :=
by sorry

end quadratic_c_over_b_l1047_104794


namespace water_removal_l1047_104738

theorem water_removal (n : ℕ) : 
  (∀n, (2:ℚ) / (n + 2) = 1 / 8) ↔ (n = 14) := 
by 
  sorry

end water_removal_l1047_104738


namespace max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l1047_104770

variable (p q : ℝ) (x y : ℝ)
variable (A B C α β γ : ℝ)

-- Conditions
axiom hp : 0 ≤ p ∧ p ≤ 1 
axiom hq : 0 ≤ q ∧ q ≤ 1 
axiom h1 : (p * x + (1 - p) * y)^2 = A * x^2 + B * x * y + C * y^2
axiom h2 : (p * x + (1 - p) * y) * (q * x + (1 - q) * y) = α * x^2 + β * x * y + γ * y^2

-- Problem
theorem max_ABC_ge_4_9 : max A (max B C) ≥ 4 / 9 := 
sorry

theorem max_alpha_beta_gamma_ge_4_9 : max α (max β γ) ≥ 4 / 9 := 
sorry

end max_ABC_ge_4_9_max_alpha_beta_gamma_ge_4_9_l1047_104770


namespace inequality_proof_l1047_104788

theorem inequality_proof (x y : ℝ) (h : x^8 + y^8 ≤ 1) : 
  x^12 - y^12 + 2 * x^6 * y^6 ≤ π / 2 :=
sorry

end inequality_proof_l1047_104788


namespace convert_500_to_base5_l1047_104754

def base10_to_base5 (n : ℕ) : ℕ :=
  -- A function to convert base 10 to base 5 would be defined here
  sorry

theorem convert_500_to_base5 : base10_to_base5 500 = 4000 := 
by 
  -- The actual proof would go here
  sorry

end convert_500_to_base5_l1047_104754


namespace tv_purchase_price_correct_l1047_104784

theorem tv_purchase_price_correct (x : ℝ) (h : (1.4 * x * 0.8 - x) = 270) : x = 2250 :=
by
  sorry

end tv_purchase_price_correct_l1047_104784


namespace geometric_sequence_n_l1047_104793

-- Definition of the conditions

-- a_1 + a_n = 82
def condition1 (a₁ an : ℕ) : Prop := a₁ + an = 82
-- a_3 * a_{n-2} = 81
def condition2 (a₃ aₙm2 : ℕ) : Prop := a₃ * aₙm2 = 81
-- S_n = 121
def condition3 (Sₙ : ℕ) : Prop := Sₙ = 121

-- Prove n = 5 given the above conditions
theorem geometric_sequence_n (a₁ a₃ an aₙm2 Sₙ n : ℕ)
  (h1 : condition1 a₁ an)
  (h2 : condition2 a₃ aₙm2)
  (h3 : condition3 Sₙ) :
  n = 5 :=
sorry

end geometric_sequence_n_l1047_104793


namespace length_of_the_train_l1047_104775

noncomputable def length_of_train (s1 s2 : ℝ) (t1 t2 : ℕ) : ℝ :=
  (s1 * t1 + s2 * t2) / 2

theorem length_of_the_train :
  ∀ (s1 s2 : ℝ) (t1 t2 : ℕ), s1 = 25 → t1 = 8 → s2 = 100 / 3 → t2 = 6 → length_of_train s1 s2 t1 t2 = 200 :=
by
  intros s1 s2 t1 t2 hs1 ht1 hs2 ht2
  rw [hs1, ht1, hs2, ht2]
  simp [length_of_train]
  norm_num

end length_of_the_train_l1047_104775


namespace sin_cos_inequality_l1047_104772

theorem sin_cos_inequality (x : ℝ) (n : ℕ) : 
  (Real.sin (2 * x))^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
by
  sorry

end sin_cos_inequality_l1047_104772


namespace inv_matrix_A_l1047_104705

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ -2, 1 ],
     ![ (3/2 : ℚ), -1/2 ] ]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ 1, 2 ],
     ![ 3, 4 ] ]

theorem inv_matrix_A : A⁻¹ = A_inv := by
  sorry

end inv_matrix_A_l1047_104705


namespace hyperbola_condition_l1047_104733

theorem hyperbola_condition (m : ℝ) : 
  (∀ x y : ℝ, (m-2) * (m+3) < 0 → (x^2) / (m-2) + (y^2) / (m+3) = 1) ↔ -3 < m ∧ m < 2 :=
by
  sorry

end hyperbola_condition_l1047_104733


namespace sum_and_count_evens_20_30_l1047_104765

def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count_evens_20_30 :
  let x := sum_integers 20 30
  let y := count_even_integers 20 30
  x + y = 281 :=
by
  sorry

end sum_and_count_evens_20_30_l1047_104765


namespace jane_reading_days_l1047_104762

theorem jane_reading_days
  (pages : ℕ)
  (half_pages : ℕ)
  (speed_first_half : ℕ)
  (speed_second_half : ℕ)
  (days_first_half : ℕ)
  (days_second_half : ℕ)
  (total_days : ℕ)
  (h1 : pages = 500)
  (h2 : half_pages = pages / 2)
  (h3 : speed_first_half = 10)
  (h4 : speed_second_half = 5)
  (h5 : days_first_half = half_pages / speed_first_half)
  (h6 : days_second_half = half_pages / speed_second_half)
  (h7 : total_days = days_first_half + days_second_half) :
  total_days = 75 :=
by
  sorry

end jane_reading_days_l1047_104762


namespace geometric_seq_value_l1047_104702

variable (a : ℕ → ℝ)
variable (g : ∀ n m : ℕ, a n * a m = a ((n + m) / 2) ^ 2)

theorem geometric_seq_value (h1 : a 2 = 1 / 3) (h2 : a 8 = 27) : a 5 = 3 ∨ a 5 = -3 := by
  sorry

end geometric_seq_value_l1047_104702


namespace function_decreasing_interval_l1047_104769

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

def decreasing_interval (a b : ℝ) : Prop :=
  ∀ x : ℝ, a < x ∧ x < b → 0 > (deriv f x)

theorem function_decreasing_interval : decreasing_interval (-1) 3 :=
by 
  sorry

end function_decreasing_interval_l1047_104769


namespace tree_growth_rate_l1047_104760

-- Given conditions
def currentHeight : ℝ := 52
def futureHeightInches : ℝ := 1104
def oneFootInInches : ℝ := 12
def years : ℝ := 8

-- Prove the yearly growth rate in feet
theorem tree_growth_rate:
  (futureHeightInches / oneFootInInches - currentHeight) / years = 5 := 
by
  sorry

end tree_growth_rate_l1047_104760


namespace angle_ACD_measure_l1047_104787

theorem angle_ACD_measure {ABD BAE ABC ACD : ℕ} 
  (h1 : ABD = 125) 
  (h2 : BAE = 95) 
  (h3 : ABC = 180 - ABD) 
  (h4 : ABD + ABC = 180 ) : 
  ACD = 180 - (BAE + ABC) :=
by 
  sorry

end angle_ACD_measure_l1047_104787


namespace sum_of_possible_values_of_x_l1047_104716

theorem sum_of_possible_values_of_x :
  ∀ x : ℝ, (x + 2) * (x - 3) = 20 → ∃ s, s = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l1047_104716


namespace students_average_age_l1047_104746

theorem students_average_age (A : ℝ) (students_count teacher_age total_average new_count : ℝ) 
  (h1 : students_count = 30)
  (h2 : teacher_age = 45)
  (h3 : new_count = students_count + 1)
  (h4 : total_average = 15) 
  (h5 : total_average = (A * students_count + teacher_age) / new_count) : 
  A = 14 :=
by
  sorry

end students_average_age_l1047_104746


namespace intersection_of_lines_l1047_104796

theorem intersection_of_lines :
  ∃ x y : ℚ, 3 * y = -2 * x + 6 ∧ 2 * y = 6 * x - 4 ∧ x = 12 / 11 ∧ y = 14 / 11 := by
  sorry

end intersection_of_lines_l1047_104796


namespace solve_fraction_eq_l1047_104750

theorem solve_fraction_eq (x : ℚ) (h : (x^2 + 3 * x + 4) / (x + 3) = x + 6) : x = -7 / 3 :=
sorry

end solve_fraction_eq_l1047_104750


namespace inequality_proof_l1047_104791

open Real

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_cond : a^2 + b^2 + c^2 = 3) :
  (a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2) :=
by
  sorry

end inequality_proof_l1047_104791


namespace sara_spent_correct_amount_on_movies_l1047_104730

def cost_ticket : ℝ := 10.62
def num_tickets : ℕ := 2
def cost_rented_movie : ℝ := 1.59
def cost_purchased_movie : ℝ := 13.95

def total_amount_spent : ℝ :=
  num_tickets * cost_ticket + cost_rented_movie + cost_purchased_movie

theorem sara_spent_correct_amount_on_movies :
  total_amount_spent = 36.78 :=
sorry

end sara_spent_correct_amount_on_movies_l1047_104730


namespace remainder_of_n_when_divided_by_7_l1047_104752

theorem remainder_of_n_when_divided_by_7 (n : ℕ) :
  (n^2 ≡ 2 [MOD 7]) ∧ (n^3 ≡ 6 [MOD 7]) → (n ≡ 3 [MOD 7]) :=
by sorry

end remainder_of_n_when_divided_by_7_l1047_104752


namespace number_of_children_admitted_l1047_104759

variable (children adults : ℕ)

def admission_fee_children : ℝ := 1.5
def admission_fee_adults  : ℝ := 4

def total_people : ℕ := 315
def total_fees   : ℝ := 810

theorem number_of_children_admitted :
  ∃ (C A : ℕ), C + A = total_people ∧ admission_fee_children * C + admission_fee_adults * A = total_fees ∧ C = 180 :=
by
  sorry

end number_of_children_admitted_l1047_104759


namespace find_y_l1047_104748

theorem find_y (y : ℝ) (h : 9 * y^2 + 36 * y^2 + 9 * y^2 = 1300) : 
  y = Real.sqrt 1300 / Real.sqrt 54 :=
by 
  sorry

end find_y_l1047_104748


namespace parabola_intersects_x_axis_expression_l1047_104708

theorem parabola_intersects_x_axis_expression (m : ℝ) (h : m^2 - m - 1 = 0) : m^2 - m + 2017 = 2018 := 
by 
  sorry

end parabola_intersects_x_axis_expression_l1047_104708


namespace smallest_four_digit_number_divisible_by_40_l1047_104703

theorem smallest_four_digit_number_divisible_by_40 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 40 = 0 ∧ ∀ m : ℕ, m >= 1000 ∧ m < 10000 ∧ m % 40 = 0 → n <= m :=
by
  use 1000
  sorry

end smallest_four_digit_number_divisible_by_40_l1047_104703


namespace sonika_years_in_bank_l1047_104712

variable (P A1 A2 : ℚ)
variables (r t : ℚ)

def simple_interest (P r t : ℚ) : ℚ := P * r * t / 100
def amount_with_interest (P r t : ℚ) : ℚ := P + simple_interest P r t

theorem sonika_years_in_bank :
  P = 9000 → A1 = 10200 → A2 = 10740 →
  amount_with_interest P r t = A1 →
  amount_with_interest P (r + 2) t = A2 →
  t = 3 :=
by
  intros hP hA1 hA2 hA1_eq hA2_eq
  sorry

end sonika_years_in_bank_l1047_104712


namespace polygon_sides_l1047_104790

-- Definition of the problem conditions
def interiorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)
def givenAngleSum (n : ℕ) : ℕ := 140 + 145 * (n - 1)

-- Problem statement: proving the number of sides
theorem polygon_sides (n : ℕ) (h : interiorAngleSum n = givenAngleSum n) : n = 10 :=
sorry

end polygon_sides_l1047_104790


namespace jason_and_lisa_cards_l1047_104715

-- Define the number of cards Jason originally had
def jason_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- Define the number of cards Lisa originally had
def lisa_original_cards (remaining: ℕ) (given_away: ℕ) : ℕ :=
  remaining + given_away

-- State the main theorem to be proved
theorem jason_and_lisa_cards :
  jason_original_cards 4 9 + lisa_original_cards 7 15 = 35 :=
by
  sorry

end jason_and_lisa_cards_l1047_104715


namespace inequality_sine_cosine_l1047_104764

theorem inequality_sine_cosine (t : ℝ) (ht : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := 
sorry

end inequality_sine_cosine_l1047_104764


namespace relationship_of_y_values_l1047_104781

theorem relationship_of_y_values (m : ℝ) (y1 y2 y3 : ℝ) :
  (∀ x y, (x = -2 ∧ y = y1 ∨ x = -1 ∧ y = y2 ∨ x = 1 ∧ y = y3) → (y = (m^2 + 1) / x)) →
  y2 < y1 ∧ y1 < y3 :=
by
  sorry

end relationship_of_y_values_l1047_104781


namespace ticket_cost_correct_l1047_104768

theorem ticket_cost_correct : 
  ∀ (a : ℝ), 
  (3 * a + 5 * (a / 2) = 30) → 
  10 * a + 8 * (a / 2) ≥ 10 * a + 8 * (a / 2) * 0.9 →
  10 * a + 8 * (a / 2) * 0.9 = 68.733 :=
by
  intro a
  intro h1 h2
  sorry

end ticket_cost_correct_l1047_104768


namespace identify_ATM_mistakes_additional_security_measures_l1047_104706

-- Define the conditions as Boolean variables representing different mistakes and measures
variables (writing_PIN_on_card : Prop)
variables (using_ATM_despite_difficulty : Prop)
variables (believing_stranger : Prop)
variables (walking_away_without_card : Prop)
variables (use_trustworthy_locations : Prop)
variables (presence_during_transactions : Prop)
variables (enable_SMS_notifications : Prop)
variables (call_bank_for_suspicious_activities : Prop)
variables (be_cautious_of_fake_SMS_alerts : Prop)
variables (store_transaction_receipts : Prop)
variables (shield_PIN : Prop)
variables (use_chipped_cards : Prop)
variables (avoid_high_risk_ATMs : Prop)

-- Prove that the identified mistakes occur given the conditions
theorem identify_ATM_mistakes :
  writing_PIN_on_card ∧ using_ATM_despite_difficulty ∧ 
  believing_stranger ∧ walking_away_without_card := sorry

-- Prove that the additional security measures should be followed
theorem additional_security_measures :
  use_trustworthy_locations ∧ presence_during_transactions ∧ 
  enable_SMS_notifications ∧ call_bank_for_suspicious_activities ∧ 
  be_cautious_of_fake_SMS_alerts ∧ store_transaction_receipts ∧ 
  shield_PIN ∧ use_chipped_cards ∧ avoid_high_risk_ATMs := sorry

end identify_ATM_mistakes_additional_security_measures_l1047_104706


namespace number_of_baskets_l1047_104779

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : total_apples / apples_per_basket = 37 :=
  by sorry

end number_of_baskets_l1047_104779


namespace projection_vector_satisfies_conditions_l1047_104773

variable (v1 v2 : ℚ)

def line_l (t : ℚ) : ℚ × ℚ :=
(2 + 3 * t, 5 - 2 * t)

def line_m (s : ℚ) : ℚ × ℚ :=
(-2 + 3 * s, 7 - 2 * s)

theorem projection_vector_satisfies_conditions :
  3 * v1 + 2 * v2 = 6 ∧ 
  ∃ k : ℚ, v1 = k * 3 ∧ v2 = k * (-2) → 
  (v1, v2) = (18 / 5, -12 / 5) :=
by
  sorry

end projection_vector_satisfies_conditions_l1047_104773


namespace range_of_m_if_neg_proposition_false_l1047_104713

theorem range_of_m_if_neg_proposition_false :
  (¬ ∃ x_0 : ℝ, x_0^2 + m * x_0 + 2 * m - 3 < 0) ↔ (2 ≤ m ∧ m ≤ 6) :=
by
  sorry

end range_of_m_if_neg_proposition_false_l1047_104713


namespace larger_solution_of_quadratic_equation_l1047_104735

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l1047_104735


namespace problem_a_problem_b_l1047_104771

-- Define necessary elements for the problem
def is_divisible_by_seven (n : ℕ) : Prop := n % 7 = 0

-- Define the method to check divisibility by seven
noncomputable def check_divisibility_by_seven (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let remaining_digits := n / 10
  remaining_digits - 2 * last_digit

-- Problem a: Prove that 4578 is divisible by 7
theorem problem_a : is_divisible_by_seven 4578 :=
  sorry

-- Problem b: Prove that there are 13 three-digit numbers of the form AB5 divisible by 7
theorem problem_b : ∃ (count : ℕ), count = 13 ∧ (∀ a b : ℕ, a ≠ 0 ∧ 1 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 → is_divisible_by_seven (100 * a + 10 * b + 5) → count = count + 1) :=
  sorry

end problem_a_problem_b_l1047_104771


namespace max_surface_area_l1047_104755

theorem max_surface_area (l w h : ℕ) (h_conditions : l + w + h = 88) : 
  2 * (l * w + l * h + w * h) ≤ 224 :=
sorry

end max_surface_area_l1047_104755


namespace smallest_possible_value_of_N_l1047_104753

-- Define the dimensions of the block
variables (l m n : ℕ) 

-- Define the condition that the product of dimensions minus one is 143
def hidden_cubes_count (l m n : ℕ) : Prop := (l - 1) * (m - 1) * (n - 1) = 143

-- Define the total number of cubes in the outer block
def total_cubes (l m n : ℕ) : ℕ := l * m * n

-- The final proof statement
theorem smallest_possible_value_of_N : 
  ∃ (l m n : ℕ), hidden_cubes_count l m n → N = total_cubes l m n → N = 336 :=
sorry

end smallest_possible_value_of_N_l1047_104753


namespace larry_expression_correct_l1047_104763

theorem larry_expression_correct (a b c d : ℤ) (e : ℤ) :
  (a = 1) → (b = 2) → (c = 3) → (d = 4) →
  (a - b - c - d + e = -2 - e) → (e = 3) :=
by
  intros ha hb hc hd heq
  rw [ha, hb, hc, hd] at heq
  linarith

end larry_expression_correct_l1047_104763


namespace hyperbola_eccentricity_l1047_104782

theorem hyperbola_eccentricity : 
  (∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 2) ∧ ∀ e : ℝ, e = Real.sqrt (1 + b^2 / a^2) → e = Real.sqrt 3) :=
by 
  sorry

end hyperbola_eccentricity_l1047_104782


namespace baker_made_cakes_l1047_104786

theorem baker_made_cakes (sold_cakes left_cakes total_cakes : ℕ) (h1 : sold_cakes = 108) (h2 : left_cakes = 59) :
  total_cakes = sold_cakes + left_cakes → total_cakes = 167 := by
  intro h
  rw [h1, h2] at h
  exact h

-- The proof part is omitted since only the statement is required

end baker_made_cakes_l1047_104786


namespace solution_of_fraction_l1047_104744

theorem solution_of_fraction (x : ℝ) (h1 : x^2 - 9 = 0) (h2 : x + 3 ≠ 0) : x = 3 :=
by
  sorry

end solution_of_fraction_l1047_104744


namespace one_angle_greater_135_l1047_104749

noncomputable def angles_sum_not_form_triangle (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : Prop :=
  ∀ (A B C : ℝ), 
   (A < a + b ∧ A < a + c ∧ A < b + c) →
  (B < a + b ∧ B < a + c ∧ B < b + c) →
  (C < a + b ∧ C < a + c ∧ C < b + c) →
  ∃ α β γ, α > 135 ∧ β < 60 ∧ γ < 60 ∧ α + β + γ = 180

theorem one_angle_greater_135 {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : angles_sum_not_form_triangle a b c ha hb hc) :
  ∃ α β γ, α > 135 ∧ α + β + γ = 180 :=
sorry

end one_angle_greater_135_l1047_104749


namespace carpenters_time_l1047_104723

theorem carpenters_time (t1 t2 t3 t4 : ℝ) (ht1 : t1 = 1) (ht2 : t2 = 2)
  (ht3 : t3 = 3) (ht4 : t4 = 4) : (1 / (1 / t1 + 1 / t2 + 1 / t3 + 1 / t4)) = 12 / 25 := by
  sorry

end carpenters_time_l1047_104723


namespace find_a_plus_b_l1047_104797

theorem find_a_plus_b (a b : ℝ) (h1 : 2 * a = -6) (h2 : a^2 - b = 4) : a + b = 2 :=
by
  sorry

end find_a_plus_b_l1047_104797


namespace judge_guilty_cases_l1047_104761

theorem judge_guilty_cases :
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  remaining_cases - innocent_cases - delayed_rulings = 4 :=
by
  let total_cases := 27
  let dismissed_cases := 3
  let remaining_cases := total_cases - dismissed_cases
  let innocent_cases := 3 * remaining_cases / 4
  let delayed_rulings := 2
  show remaining_cases - innocent_cases - delayed_rulings = 4
  sorry

end judge_guilty_cases_l1047_104761


namespace number_of_packs_l1047_104737

-- Given conditions
def cost_per_pack : ℕ := 11
def total_money : ℕ := 110

-- Statement to prove
theorem number_of_packs :
  total_money / cost_per_pack = 10 := by
  sorry

end number_of_packs_l1047_104737


namespace production_steps_description_l1047_104728

-- Definition of the choices
inductive FlowchartType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

-- Conditions
def describeProductionSteps (flowchart : FlowchartType) : Prop :=
flowchart = FlowchartType.ProcessFlowchart

-- The statement to be proved
theorem production_steps_description:
  describeProductionSteps FlowchartType.ProcessFlowchart := 
sorry -- proof to be provided

end production_steps_description_l1047_104728


namespace linear_function_point_l1047_104743

theorem linear_function_point (a b : ℝ) (h : b = 2 * a - 1) : 2 * a - b + 1 = 2 :=
by
  sorry

end linear_function_point_l1047_104743


namespace parabola_vertex_f_l1047_104756

theorem parabola_vertex_f (d e f : ℝ) (h_vertex : ∀ y, (d * (y - 3)^2 + 5) = (d * y^2 + e * y + f))
  (h_point : d * (6 - 3)^2 + 5 = 2) : f = 2 :=
by
  sorry

end parabola_vertex_f_l1047_104756


namespace farm_problem_l1047_104758

theorem farm_problem (D C : ℕ) (h1 : D + C = 15) (h2 : 2 * D + 4 * C = 42) : C = 6 :=
sorry

end farm_problem_l1047_104758


namespace unit_digit_3_pow_2012_sub_1_l1047_104725

theorem unit_digit_3_pow_2012_sub_1 :
  (3 ^ 2012 - 1) % 10 = 0 :=
sorry

end unit_digit_3_pow_2012_sub_1_l1047_104725


namespace even_function_value_at_2_l1047_104736

theorem even_function_value_at_2 {a : ℝ} (h : ∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) : 
  ((2 + 1) * (2 - a)) = 3 := by
  sorry

end even_function_value_at_2_l1047_104736


namespace solve_for_x_l1047_104740

theorem solve_for_x (x : ℝ) (h : x - 5.90 = 9.28) : x = 15.18 :=
by
  sorry

end solve_for_x_l1047_104740


namespace count_negative_terms_in_sequence_l1047_104739

theorem count_negative_terms_in_sequence : 
  ∃ (s : List ℕ), (∀ n ∈ s, n^2 - 8*n + 12 < 0) ∧ s.length = 3 ∧ (∀ n ∈ s, 2 < n ∧ n < 6) :=
by
  sorry

end count_negative_terms_in_sequence_l1047_104739


namespace problem1_proof_problem2_proof_l1047_104783

noncomputable def problem1 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a| + |b| ≤ Real.sqrt 2

noncomputable def problem2 (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : Prop :=
  |a^3 / b| + |b^3 / a| ≥ 1

theorem problem1_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem1 a b h₁ h₂ h₃ :=
  sorry

theorem problem2_proof (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a^2 + b^2 = 1) : problem2 a b h₁ h₂ h₃ :=
  sorry

end problem1_proof_problem2_proof_l1047_104783


namespace books_borrowed_by_lunchtime_l1047_104707

theorem books_borrowed_by_lunchtime (x : ℕ) :
  (∀ x : ℕ, 100 - x + 40 - 30 = 60) → (x = 50) :=
by
  intro h
  have eqn := h x
  sorry

end books_borrowed_by_lunchtime_l1047_104707


namespace polynomial_coeff_sum_l1047_104798

/-- 
Given that the product of the polynomials (4x^2 - 6x + 5)(8 - 3x) can be written as
ax^3 + bx^2 + cx + d, prove that 9a + 3b + c + d = 19.
-/
theorem polynomial_coeff_sum :
  ∃ a b c d : ℝ, 
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) ∧
  9 * a + 3 * b + c + d = 19 :=
sorry

end polynomial_coeff_sum_l1047_104798


namespace center_of_circle_l1047_104717

noncomputable def center_is_correct (x y : ℚ) : Prop :=
  (5 * x - 2 * y = -10) ∧ (3 * x + y = 0)

theorem center_of_circle : center_is_correct (-10 / 11) (30 / 11) :=
by
  sorry

end center_of_circle_l1047_104717


namespace ellipse_triangle_is_isosceles_right_l1047_104729

theorem ellipse_triangle_is_isosceles_right (e : ℝ) (a b c k : ℝ)
  (H1 : e = (c / a))
  (H2 : e = (Real.sqrt 2) / 2)
  (H3 : b^2 = a^2 * (1 - e^2))
  (H4 : a = 2 * k)
  (H5 : b = k * Real.sqrt 2)
  (H6 : c = k * Real.sqrt 2) :
  (4 * k)^2 = (2 * (k * Real.sqrt 2))^2 + (2 * (k * Real.sqrt 2))^2 :=
by
  sorry

end ellipse_triangle_is_isosceles_right_l1047_104729


namespace volume_of_region_l1047_104714

-- Define the conditions
def condition1 (x y z : ℝ) := abs (x + y + 2 * z) + abs (x + y - 2 * z) ≤ 12
def condition2 (x : ℝ) := x ≥ 0
def condition3 (y : ℝ) := y ≥ 0
def condition4 (z : ℝ) := z ≥ 0

-- Define the volume function
def volume (x y z : ℝ) := 18 * 3

-- Proof statement
theorem volume_of_region : ∀ (x y z : ℝ),
  condition1 x y z →
  condition2 x →
  condition3 y →
  condition4 z →
  volume x y z = 54 := by
  sorry

end volume_of_region_l1047_104714


namespace problem_l1047_104789

theorem problem (a b c : ℤ) :
  (∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)) →
  (∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 :=
by
  intros h1 h2
  sorry

end problem_l1047_104789


namespace carla_smoothies_serving_l1047_104778

theorem carla_smoothies_serving :
  ∀ (watermelon_puree : ℕ) (cream : ℕ) (serving_size : ℕ),
  watermelon_puree = 500 → cream = 100 → serving_size = 150 →
  (watermelon_puree + cream) / serving_size = 4 :=
by
  intros watermelon_puree cream serving_size
  intro h1 -- watermelon_puree = 500
  intro h2 -- cream = 100
  intro h3 -- serving_size = 150
  sorry

end carla_smoothies_serving_l1047_104778


namespace min_shift_value_l1047_104785

theorem min_shift_value (φ : ℝ) (hφ : φ > 0) :
  (∃ k : ℤ, φ = -k * π / 3 + π / 6) →
  ∃ φ_min : ℝ, φ_min = π / 6 ∧ (∀ φ', φ' > 0 → ∃ k' : ℤ, φ' = -k' * π / 3 + π / 6 → φ_min ≤ φ') :=
by
  intro h
  use π / 6
  constructor
  . sorry
  . sorry

end min_shift_value_l1047_104785


namespace score_comparison_l1047_104727

theorem score_comparison :
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  combined_score - opponent_score = 143 :=
by
  let sammy_score := 20
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let alex_score := cher_score + cher_score / 10
  let combined_score := sammy_score + gab_score + cher_score + alex_score
  let opponent_score := 85
  sorry

end score_comparison_l1047_104727


namespace value_expression_at_5_l1047_104701

theorem value_expression_at_5 (x : ℕ) (hx : x = 5) : 2 * x^2 + 4 = 54 :=
by
  -- Adding sorry to skip the proof.
  sorry

end value_expression_at_5_l1047_104701


namespace factor_expression_l1047_104777

variable (x : ℝ)

theorem factor_expression : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) :=
by
  sorry

end factor_expression_l1047_104777


namespace sum_of_first_six_terms_geometric_sequence_l1047_104799

-- conditions
def a : ℚ := 1/4
def r : ℚ := 1/4

-- geometric series sum function
def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- target sum of first six terms
def S_6 : ℚ := geom_sum a r 6

-- proof statement
theorem sum_of_first_six_terms_geometric_sequence :
  S_6 = 1365 / 4096 :=
by 
  sorry

end sum_of_first_six_terms_geometric_sequence_l1047_104799


namespace dice_sum_not_11_l1047_104709

/-- Jeremy rolls three standard six-sided dice, with each showing a different number and the product of the numbers on the upper faces is 72.
    Prove that the sum 11 is not possible. --/
theorem dice_sum_not_11 : 
  ∃ (a b c : ℕ), 
    (1 ≤ a ∧ a ≤ 6) ∧ 
    (1 ≤ b ∧ b ≤ 6) ∧ 
    (1 ≤ c ∧ c ≤ 6) ∧ 
    (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
    (a * b * c = 72) ∧ 
    (a > 4 ∨ b > 4 ∨ c > 4) → 
    a + b + c ≠ 11 := 
by
  sorry

end dice_sum_not_11_l1047_104709


namespace domain_of_function_l1047_104719

def domain_condition_1 (x : ℝ) : Prop := 1 - x > 0
def domain_condition_2 (x : ℝ) : Prop := x + 3 ≥ 0

def in_domain (x : ℝ) : Prop := domain_condition_1 x ∧ domain_condition_2 x

theorem domain_of_function : ∀ x : ℝ, in_domain x ↔ (-3 : ℝ) ≤ x ∧ x < 1 := 
by sorry

end domain_of_function_l1047_104719


namespace total_people_in_class_l1047_104757

-- Define the number of people based on their interests
def likes_both: Nat := 5
def only_baseball: Nat := 2
def only_football: Nat := 3
def likes_neither: Nat := 6

-- Define the total number of people in the class
def total_people := likes_both + only_baseball + only_football + likes_neither

-- Theorem statement
theorem total_people_in_class : total_people = 16 :=
by
  -- Proof is skipped
  sorry

end total_people_in_class_l1047_104757


namespace sequence_sum_is_25_div_3_l1047_104792

noncomputable def sum_of_arithmetic_sequence (a n d : ℝ) : ℝ := (n / 2) * (2 * a + (n - 1) * d)

theorem sequence_sum_is_25_div_3 (a d : ℝ)
  (h1 : a + 4 * d = 1)
  (h2 : 3 * a + 15 * d = 2 * a + 8 * d) :
  sum_of_arithmetic_sequence a 10 d = 25 / 3 := by
  sorry

end sequence_sum_is_25_div_3_l1047_104792


namespace field_area_is_36_square_meters_l1047_104795

theorem field_area_is_36_square_meters (side_length : ℕ) (h : side_length = 6) : side_length * side_length = 36 :=
by
  sorry

end field_area_is_36_square_meters_l1047_104795


namespace fraction_identity_l1047_104711

theorem fraction_identity (N F : ℝ) (hN : N = 8) (h : 0.5 * N = F * N + 2) : F = 1 / 4 :=
by {
  -- proof will go here
  sorry
}

end fraction_identity_l1047_104711


namespace volleyballTeam_starters_l1047_104780

noncomputable def chooseStarters (totalPlayers : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  let remainingPlayers := totalPlayers - quadruplets
  let chooseQuadruplet := quadruplets
  let chooseRemaining := Nat.choose remainingPlayers (starters - 1)
  chooseQuadruplet * chooseRemaining

theorem volleyballTeam_starters :
  chooseStarters 16 4 6 = 3168 :=
by
  sorry

end volleyballTeam_starters_l1047_104780


namespace triangle_properties_l1047_104731

noncomputable def triangle_side_lengths (m1 m2 m3 : ℝ) : Prop :=
  ∃ a b c s,
    m1 = 20 ∧
    m2 = 24 ∧
    m3 = 30 ∧
    a = 36.28 ∧
    b = 30.24 ∧
    c = 24.19 ∧
    s = 362.84

theorem triangle_properties :
  triangle_side_lengths 20 24 30 :=
by
  sorry

end triangle_properties_l1047_104731


namespace scientific_notation_of_86000000_l1047_104710

theorem scientific_notation_of_86000000 :
  ∃ (x : ℝ) (y : ℤ), 86000000 = x * 10^y ∧ x = 8.6 ∧ y = 7 :=
by
  use 8.6
  use 7
  sorry

end scientific_notation_of_86000000_l1047_104710


namespace twenty_percent_greater_l1047_104722

theorem twenty_percent_greater (x : ℕ) : 
  x = 80 + (20 * 80 / 100) → x = 96 :=
by
  sorry

end twenty_percent_greater_l1047_104722


namespace sequence_formula_l1047_104734

theorem sequence_formula (a : ℕ → ℕ) (c : ℕ) (h₁ : a 1 = 2) (h₂ : ∀ n, a (n + 1) = a n + c * n) 
(h₃ : a 1 ≠ a 2) (h₄ : a 2 * a 2 = a 1 * a 3) : c = 2 ∧ ∀ n, a n = n^2 - n + 2 :=
by
  sorry

end sequence_formula_l1047_104734


namespace pb_distance_l1047_104767

theorem pb_distance (a b c d PA PD PC PB : ℝ)
  (hPA : PA = 5)
  (hPD : PD = 6)
  (hPC : PC = 7)
  (h1 : a^2 + b^2 = PA^2)
  (h2 : b^2 + c^2 = PC^2)
  (h3 : c^2 + d^2 = PD^2)
  (h4 : d^2 + a^2 = PB^2) :
  PB = Real.sqrt 38 := by
  sorry

end pb_distance_l1047_104767


namespace total_number_of_people_l1047_104704

variables (A B : ℕ)

def pencils_brought_by_assoc_profs (A : ℕ) : ℕ := 2 * A
def pencils_brought_by_asst_profs (B : ℕ) : ℕ := B
def charts_brought_by_assoc_profs (A : ℕ) : ℕ := A
def charts_brought_by_asst_profs (B : ℕ) : ℕ := 2 * B

axiom pencils_total : pencils_brought_by_assoc_profs A + pencils_brought_by_asst_profs B = 10
axiom charts_total : charts_brought_by_assoc_profs A + charts_brought_by_asst_profs B = 11

theorem total_number_of_people : A + B = 7 :=
sorry

end total_number_of_people_l1047_104704


namespace sum_of_tangents_l1047_104700

noncomputable def function_f (x : ℝ) : ℝ :=
  max (max (4 * x + 20) (-x + 2)) (5 * x - 3)

theorem sum_of_tangents (q : ℝ → ℝ) (a b c : ℝ) (h1 : ∀ x, q x - (4 * x + 20) = q x - function_f x)
  (h2 : ∀ x, q x - (-x + 2) = q x - function_f x)
  (h3 : ∀ x, q x - (5 * x - 3) = q x - function_f x) :
  a + b + c = -83 / 10 :=
sorry

end sum_of_tangents_l1047_104700


namespace max_sum_of_integer_pairs_on_circle_l1047_104766

theorem max_sum_of_integer_pairs_on_circle : 
  ∃ (x y : ℤ), x^2 + y^2 = 169 ∧ ∀ (a b : ℤ), a^2 + b^2 = 169 → x + y ≥ a + b :=
sorry

end max_sum_of_integer_pairs_on_circle_l1047_104766


namespace range_of_sin_cos_expression_l1047_104745

variable (a b c A B C : ℝ)

theorem range_of_sin_cos_expression
  (h1 : a = b)
  (h2 : c * Real.sin A = -a * Real.cos C) :
  1 < 2 * Real.sin (A + Real.pi / 6) :=
sorry

end range_of_sin_cos_expression_l1047_104745


namespace sum_proof_l1047_104776

-- Define the context and assumptions
variables (F S T : ℕ)
axiom sum_of_numbers : F + S + T = 264
axiom first_number_twice_second : F = 2 * S
axiom third_number_one_third_first : T = F / 3
axiom second_number_given : S = 72

-- The theorem to prove the sum is 264 given the conditions
theorem sum_proof : F + S + T = 264 :=
by
  -- Given conditions already imply the theorem, the actual proof follows from these
  sorry

end sum_proof_l1047_104776


namespace cube_path_length_l1047_104721

noncomputable def path_length_dot_cube : ℝ :=
  let edge_length := 2
  let radius1 := Real.sqrt 5
  let radius2 := 1
  (radius1 + radius2) * Real.pi

theorem cube_path_length :
  path_length_dot_cube = (Real.sqrt 5 + 1) * Real.pi :=
by
  sorry

end cube_path_length_l1047_104721


namespace complement_of_A_in_R_l1047_104741

open Set

variable (R : Set ℝ) (A : Set ℝ)

def real_numbers : Set ℝ := {x | true}

def set_A : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2}

theorem complement_of_A_in_R : (real_numbers \ set_A) = {y | y < 0} := by
  sorry

end complement_of_A_in_R_l1047_104741
