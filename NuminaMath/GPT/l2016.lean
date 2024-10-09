import Mathlib

namespace fraction_irreducible_l2016_201685

theorem fraction_irreducible (n : ℕ) (hn : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by sorry

end fraction_irreducible_l2016_201685


namespace ways_A_not_head_is_600_l2016_201694

-- Definitions for the problem conditions
def num_people : ℕ := 6
def valid_positions_for_A : ℕ := 5
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The total number of ways person A can be placed in any position except the first
def num_ways_A_not_head : ℕ := valid_positions_for_A * factorial (num_people - 1)

-- The theorem to prove
theorem ways_A_not_head_is_600 : num_ways_A_not_head = 600 := by
  sorry

end ways_A_not_head_is_600_l2016_201694


namespace seven_segments_impossible_l2016_201692

theorem seven_segments_impossible :
  ¬(∃(segments : Fin 7 → Set (Fin 7)), (∀i, ∃ (S : Finset (Fin 7)), S.card = 3 ∧ ∀ j ∈ S, i ≠ j ∧ segments i j) ∧ (∀ i j, i ≠ j → segments i j → segments j i)) :=
sorry

end seven_segments_impossible_l2016_201692


namespace bakery_combinations_l2016_201650

theorem bakery_combinations 
  (total_breads : ℕ) (bread_types : Finset ℕ) (purchases : Finset ℕ)
  (h_total : total_breads = 8)
  (h_bread_types : bread_types.card = 5)
  (h_purchases : purchases.card = 2) : 
  ∃ (combinations : ℕ), combinations = 70 := 
sorry

end bakery_combinations_l2016_201650


namespace fill_bathtub_time_l2016_201676

theorem fill_bathtub_time
  (r_cold : ℚ := 1/10)
  (r_hot : ℚ := 1/15)
  (r_empty : ℚ := -1/12)
  (net_rate : ℚ := r_cold + r_hot + r_empty) :
  net_rate = 1/12 → 
  t = 12 :=
by
  sorry

end fill_bathtub_time_l2016_201676


namespace supplement_of_double_complement_l2016_201686

def angle : ℝ := 30

def complement (θ : ℝ) : ℝ :=
  90 - θ

def double_complement (θ : ℝ) : ℝ :=
  2 * (complement θ)

def supplement (θ : ℝ) : ℝ :=
  180 - θ

theorem supplement_of_double_complement (θ : ℝ) (h : θ = angle) : supplement (double_complement θ) = 60 :=
by
  sorry

end supplement_of_double_complement_l2016_201686


namespace solve_equation_l2016_201637

theorem solve_equation (x : ℝ) (h : 16 * x^2 = 81) : x = 9 / 4 ∨ x = - (9 / 4) :=
by
  sorry

end solve_equation_l2016_201637


namespace total_carrots_l2016_201627

def sally_carrots : ℕ := 6
def fred_carrots : ℕ := 4
def mary_carrots : ℕ := 10

theorem total_carrots : sally_carrots + fred_carrots + mary_carrots = 20 := by
  sorry

end total_carrots_l2016_201627


namespace polynomial_product_l2016_201638

theorem polynomial_product (x : ℝ) : (x - 1) * (x + 3) * (x + 5) = x^3 + 7*x^2 + 7*x - 15 :=
by
  sorry

end polynomial_product_l2016_201638


namespace sum_of_decimals_l2016_201681

theorem sum_of_decimals :
  5.467 + 2.349 + 3.785 = 11.751 :=
sorry

end sum_of_decimals_l2016_201681


namespace new_batting_average_l2016_201680

def initial_runs (A : ℕ) := 16 * A
def additional_runs := 85
def increased_average := 3
def runs_in_5_innings := 100 + 120 + 45 + 75 + 65
def total_runs_17_innings (A : ℕ) := 17 * (A + increased_average)
def A : ℕ := 34
def total_runs_22_innings := total_runs_17_innings A + runs_in_5_innings
def number_of_innings := 22
def new_average := total_runs_22_innings / number_of_innings

theorem new_batting_average : new_average = 47 :=
by sorry

end new_batting_average_l2016_201680


namespace reflect_curve_maps_onto_itself_l2016_201684

theorem reflect_curve_maps_onto_itself (a b c : ℝ) :
    ∃ (x0 y0 : ℝ), 
    x0 = -a / 3 ∧ 
    y0 = 2 * a^3 / 27 - a * b / 3 + c ∧
    ∀ x y x' y', 
    y = x^3 + a * x^2 + b * x + c → 
    x' = 2 * x0 - x → 
    y' = 2 * y0 - y → 
    y' = x'^3 + a * x'^2 + b * x' + c := 
    by sorry

end reflect_curve_maps_onto_itself_l2016_201684


namespace candy_last_days_l2016_201602

def pieces_from_neighbors : ℝ := 11.0
def pieces_from_sister : ℝ := 5.0
def pieces_per_day : ℝ := 8.0
def total_pieces : ℝ := pieces_from_neighbors + pieces_from_sister

theorem candy_last_days : total_pieces / pieces_per_day = 2 := by
    sorry

end candy_last_days_l2016_201602


namespace smallest_d_l2016_201667

theorem smallest_d (c d : ℕ) (h1 : c - d = 8)
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : d = 4 := by
  sorry

end smallest_d_l2016_201667


namespace arithmetic_sequences_count_l2016_201645

noncomputable def countArithmeticSequences (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4

theorem arithmetic_sequences_count :
  ∀ n : ℕ, countArithmeticSequences n = if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4 :=
by sorry

end arithmetic_sequences_count_l2016_201645


namespace negation_of_universal_prop_correct_l2016_201618

def negation_of_universal_prop : Prop :=
  ¬ (∀ x : ℝ, x = |x|) ↔ ∃ x : ℝ, x ≠ |x|

theorem negation_of_universal_prop_correct : negation_of_universal_prop := 
by
  sorry

end negation_of_universal_prop_correct_l2016_201618


namespace net_income_calculation_l2016_201665

-- Definitions based on conditions
def rent_per_hour := 20
def monday_hours := 8
def wednesday_hours := 8
def friday_hours := 6
def sunday_hours := 5
def maintenance_cost := 35
def insurance_fee := 15
def rental_days := 4

-- Derived values based on conditions
def total_income_per_week :=
  (monday_hours + wednesday_hours) * rent_per_hour * 2 + 
  friday_hours * rent_per_hour + 
  sunday_hours * rent_per_hour

def total_expenses_per_week :=
  maintenance_cost + 
  insurance_fee * rental_days

def net_income_per_week := 
  total_income_per_week - total_expenses_per_week

-- The final proof statement
theorem net_income_calculation : net_income_per_week = 445 := by
  sorry

end net_income_calculation_l2016_201665


namespace fixed_fee_1430_l2016_201647

def fixed_monthly_fee (f p : ℝ) : Prop :=
  f + p = 20.60 ∧ f + 3 * p = 33.20

theorem fixed_fee_1430 (f p: ℝ) (h : fixed_monthly_fee f p) : 
  f = 14.30 :=
by
  sorry

end fixed_fee_1430_l2016_201647


namespace midterm_exam_2022_option_probabilities_l2016_201664

theorem midterm_exam_2022_option_probabilities :
  let no_option := 4
  let prob_distribution := (1 : ℚ) / 3
  let combs_with_4_correct := 1
  let combs_with_3_correct := 4
  let combs_with_2_correct := 6
  let prob_4_correct := prob_distribution
  let prob_3_correct := prob_distribution / combs_with_3_correct
  let prob_2_correct := prob_distribution / combs_with_2_correct
  
  let prob_B_correct := combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct
  let prob_C_given_event_A := combs_with_3_correct * prob_3_correct / (combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct)
  
  (prob_B_correct > 1 / 2) ∧ (prob_C_given_event_A = 1 / 3) :=
by 
  sorry

end midterm_exam_2022_option_probabilities_l2016_201664


namespace range_of_a_for_negative_root_l2016_201630

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) →
  - (1/2 : ℝ) < a ∧ a ≤ (1/16 : ℝ) :=
by
  sorry

end range_of_a_for_negative_root_l2016_201630


namespace find_abc_sum_l2016_201625

theorem find_abc_sum (A B C : ℤ) (h : ∀ x : ℝ, x^3 + A * x^2 + B * x + C = (x + 1) * (x - 3) * (x - 4)) : A + B + C = 11 :=
by {
  -- This statement asserts that, given the conditions, the sum A + B + C equals 11
  sorry
}

end find_abc_sum_l2016_201625


namespace bicycle_cost_correct_l2016_201617

def pay_rate : ℕ := 5
def hours_p_week : ℕ := 2 + 1 + 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

theorem bicycle_cost_correct :
  pay_rate * hours_p_week * weeks = bicycle_cost :=
by
  sorry

end bicycle_cost_correct_l2016_201617


namespace simplification_qrt_1_simplification_qrt_2_l2016_201652

-- Problem 1
theorem simplification_qrt_1 : (2 * Real.sqrt 12 + 3 * Real.sqrt 3 - Real.sqrt 27) = 4 * Real.sqrt 3 :=
by
  sorry

-- Problem 2
theorem simplification_qrt_2 : (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2 * 12) + Real.sqrt 24) = 4 + Real.sqrt 6 :=
by
  sorry

end simplification_qrt_1_simplification_qrt_2_l2016_201652


namespace tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l2016_201673

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x * Real.exp x - Real.exp 1

-- Part (Ⅰ)
theorem tangent_line_at_one (h_a : a = 0) : ∃ m b : ℝ, ∀ x : ℝ, 2 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0 := sorry

-- Part (Ⅱ)
theorem unique_zero_of_f (h_a : a > 0) : ∃! t : ℝ, f a t = 0 := sorry

-- Part (Ⅲ)
theorem exists_lower_bound_of_f (h_a : a < 0) : ∃ m : ℝ, ∀ x : ℝ, f a x ≥ m := sorry

end tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l2016_201673


namespace perpendicular_slope_of_line_l2016_201663

theorem perpendicular_slope_of_line (x y : ℤ) : 
    (5 * x - 4 * y = 20) → 
    ∃ m : ℚ, m = -4 / 5 := 
by 
    sorry

end perpendicular_slope_of_line_l2016_201663


namespace chess_champion_probability_l2016_201607

theorem chess_champion_probability :
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  1000 * P = 343 :=
by 
  let P_R := 0.6
  let P_S := 0.3
  let P_D := 0.1
  let P := 0.06 + 0.126 + 0.024 + 0.021 + 0.03 + 0.072 + 0.01
  show 1000 * P = 343
  sorry

end chess_champion_probability_l2016_201607


namespace calculate_A_plus_B_l2016_201605

theorem calculate_A_plus_B (A B : ℝ) (h1 : A ≠ B) 
  (h2 : ∀ x : ℝ, (A * (B * x^2 + A * x + 1)^2 + B * (B * x^2 + A * x + 1) + 1) 
                - (B * (A * x^2 + B * x + 1)^2 + A * (A * x^2 + B * x + 1) + 1) 
                = x^4 + 5 * x^3 + x^2 - 4 * x) : A + B = 0 :=
by
  sorry

end calculate_A_plus_B_l2016_201605


namespace parents_years_in_america_before_aziz_birth_l2016_201682

noncomputable def aziz_birth_year (current_year : ℕ) (aziz_age : ℕ) : ℕ :=
  current_year - aziz_age

noncomputable def years_parents_in_america_before_aziz_birth (arrival_year : ℕ) (aziz_birth_year : ℕ) : ℕ :=
  aziz_birth_year - arrival_year

theorem parents_years_in_america_before_aziz_birth 
  (current_year : ℕ := 2021) 
  (aziz_age : ℕ := 36) 
  (arrival_year : ℕ := 1982) 
  (expected_years : ℕ := 3) :
  years_parents_in_america_before_aziz_birth arrival_year (aziz_birth_year current_year aziz_age) = expected_years :=
by 
  sorry

end parents_years_in_america_before_aziz_birth_l2016_201682


namespace linear_function_difference_l2016_201608

noncomputable def linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

theorem linear_function_difference (f : ℝ → ℝ) 
  (h_linear : linear_function f)
  (h_cond1 : f 10 - f 5 = 20)
  (h_cond2 : f 0 = 3) :
  f 15 - f 5 = 40 :=
sorry

end linear_function_difference_l2016_201608


namespace Tony_fills_pool_in_90_minutes_l2016_201614

def minutes (r : ℚ) : ℚ := 1 / r

theorem Tony_fills_pool_in_90_minutes (J S T : ℚ) 
  (hJ : J = 1 / 30)       -- Jim's rate in pools per minute
  (hS : S = 1 / 45)       -- Sue's rate in pools per minute
  (h_combined : J + S + T = 1 / 15) -- Combined rate of all three

  : minutes T = 90 :=     -- Tony can fill the pool alone in 90 minutes
by sorry

end Tony_fills_pool_in_90_minutes_l2016_201614


namespace part_a_part_b_l2016_201657

noncomputable def volume_of_prism (V : ℝ) : ℝ :=
  (9 / 250) * V

noncomputable def max_volume_of_prism (V : ℝ) : ℝ :=
  (1 / 12) * V

theorem part_a (V : ℝ) :
  volume_of_prism V = (9 / 250) * V :=
  by sorry

theorem part_b (V : ℝ) :
  max_volume_of_prism V = (1 / 12) * V :=
  by sorry

end part_a_part_b_l2016_201657


namespace range_of_a_for_propositions_p_and_q_l2016_201633

theorem range_of_a_for_propositions_p_and_q :
  {a : ℝ | ∃ x, (x^2 + 2 * a * x + 4 = 0) ∧ (3 - 2 * a > 1)} = {a | a ≤ -2} := sorry

end range_of_a_for_propositions_p_and_q_l2016_201633


namespace largest_n_for_inequality_l2016_201674

theorem largest_n_for_inequality :
  ∃ n : ℕ, 3 * n^2007 < 3^4015 ∧ ∀ m : ℕ, 3 * m^2007 < 3^4015 → m ≤ 8 ∧ n = 8 :=
by
  sorry

end largest_n_for_inequality_l2016_201674


namespace income_recording_l2016_201636

theorem income_recording (exp_200 : Int := -200) (income_60 : Int := 60) : exp_200 = -200 → income_60 = 60 →
  (income_60 > 0) :=
by
  intro h_exp h_income
  sorry

end income_recording_l2016_201636


namespace SmartMart_science_kits_l2016_201659

theorem SmartMart_science_kits (sc pz : ℕ) (h1 : pz = sc - 9) (h2 : pz = 36) : sc = 45 := by
  sorry

end SmartMart_science_kits_l2016_201659


namespace tangent_line_eq_max_f_val_in_interval_a_le_2_l2016_201683

-- Definitions based on given conditions
def f (x : ℝ) (a : ℝ) : ℝ := x ^ 3 - a * x ^ 2

def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x ^ 2 - 2 * a * x

-- (I) (i) Proof that the tangent line equation is y = 3x - 2 at (1, f(1))
theorem tangent_line_eq (a : ℝ) (h : f_prime 1 a = 3) : y = 3 * x - 2 :=
by sorry

-- (I) (ii) Proof that the max value of f(x) in [0,2] is 8
theorem max_f_val_in_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x 0 ≤ f 2 0 :=
by sorry

-- (II) Proof that a ≤ 2 if f(x) + x ≥ 0 for all x ∈ [0,2]
theorem a_le_2 (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x a + x ≥ 0) : a ≤ 2 :=
by sorry

end tangent_line_eq_max_f_val_in_interval_a_le_2_l2016_201683


namespace f_iterated_result_l2016_201632

def f (x : ℕ) : ℕ :=
  if Even x then 3 * x / 2 else 2 * x + 1

theorem f_iterated_result : f (f (f (f 1))) = 31 := by
  sorry

end f_iterated_result_l2016_201632


namespace quadratic_inequality_solution_l2016_201691

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x - 8 ≤ 0 ↔ -4/3 ≤ x ∧ x ≤ 2 :=
sorry

end quadratic_inequality_solution_l2016_201691


namespace michael_age_multiple_l2016_201662

theorem michael_age_multiple (M Y O k : ℤ) (hY : Y = 5) (hO : O = 3 * Y) (h_combined : M + O + Y = 28) (h_relation : O = k * (M - 1) + 1) : k = 2 :=
by
  -- Definitions and given conditions are provided:
  have hY : Y = 5 := hY
  have hO : O = 3 * Y := hO
  have h_combined : M + O + Y = 28 := h_combined
  have h_relation : O = k * (M - 1) + 1 := h_relation
  
  -- Begin the proof by using the provided conditions
  sorry

end michael_age_multiple_l2016_201662


namespace minimum_value_inverse_sum_l2016_201693

variables {m n : ℝ}

theorem minimum_value_inverse_sum 
  (hm : m > 0) 
  (hn : n > 0) 
  (hline : ∀ x y : ℝ, m * x + n * y + 2 = 0 → (x + 3)^2 + (y + 1)^2 = 1)
  (hchord : ∀ x1 y1 x2 y2 : ℝ, m * x1 + n * y1 + 2 = 0 ∧ m * x2 + n * y2 + 2 = 0 → 
    (x1 - x2)^2 + (y1 - y2)^2 = 4) : 
  ∃ m n : ℝ, 3 * m + n = 2 ∧ m > 0 ∧ n > 0 ∧ 
    (∀ m' n' : ℝ, 3 * m' + n' = 2 → m' > 0 → n' > 0 → 
      (1 / m' + 3 / n' ≥ 6)) :=
sorry

end minimum_value_inverse_sum_l2016_201693


namespace fred_washing_cars_l2016_201624

theorem fred_washing_cars :
  ∀ (initial_amount final_amount money_made : ℕ),
  initial_amount = 23 →
  final_amount = 86 →
  money_made = final_amount - initial_amount →
  money_made = 63 := by
    intros initial_amount final_amount money_made h_initial h_final h_calc
    rw [h_initial, h_final] at h_calc
    exact h_calc

end fred_washing_cars_l2016_201624


namespace linear_inequalities_solution_range_l2016_201648

theorem linear_inequalities_solution_range (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) ↔ m > 2 / 3 :=
by
  sorry

end linear_inequalities_solution_range_l2016_201648


namespace value_of_3_over_x_l2016_201603

theorem value_of_3_over_x (x : ℝ) (hx : 1 - 6 / x + 9 / x^2 - 4 / x^3 = 0) : 
  (3 / x = 3 ∨ 3 / x = 3 / 4) :=
  sorry

end value_of_3_over_x_l2016_201603


namespace find_a_5_l2016_201672

theorem find_a_5 (a : ℕ → ℤ) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1)
  (h₂ : a 2 + a 4 + a 6 = 18) : a 5 = 5 := 
sorry

end find_a_5_l2016_201672


namespace point_in_third_quadrant_l2016_201623

theorem point_in_third_quadrant (x y : ℝ) (h1 : x + y < 0) (h2 : x * y > 0) : x < 0 ∧ y < 0 := 
sorry

end point_in_third_quadrant_l2016_201623


namespace find_a1_l2016_201620

noncomputable def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ, a (n+1) + a n = 4*n

theorem find_a1 (a : ℕ → ℕ) (h : is_arithmetic_sequence a) : a 1 = 1 := by
  sorry

end find_a1_l2016_201620


namespace max_unsuccessful_attempts_l2016_201654

theorem max_unsuccessful_attempts (n_rings letters_per_ring : ℕ) (h_rings : n_rings = 3) (h_letters : letters_per_ring = 6) : 
  (letters_per_ring ^ n_rings) - 1 = 215 := 
by 
  -- conditions
  rw [h_rings, h_letters]
  -- necessary imports and proof generation
  sorry

end max_unsuccessful_attempts_l2016_201654


namespace expression_equals_33_l2016_201656

noncomputable def calculate_expression : ℚ :=
  let part1 := 25 * 52
  let part2 := 46 * 15
  let diff := part1 - part2
  (2013 / diff) * 10

theorem expression_equals_33 : calculate_expression = 33 := sorry

end expression_equals_33_l2016_201656


namespace play_role_assignments_l2016_201643

def specific_role_assignments (men women remaining either_gender_roles : ℕ) : ℕ :=
  men * women * Nat.choose remaining either_gender_roles

theorem play_role_assignments :
  specific_role_assignments 6 7 11 4 = 13860 := by
  -- The given problem statement implies evaluating the specific role assignments
  sorry

end play_role_assignments_l2016_201643


namespace positive_roots_implies_nonnegative_m_l2016_201610

variables {x1 x2 m : ℝ}

theorem positive_roots_implies_nonnegative_m (h1 : x1 > 0) (h2 : x2 > 0)
  (h3 : x1 * x2 = 1) (h4 : x1 + x2 = m + 2) : m ≥ 0 :=
by
  sorry

end positive_roots_implies_nonnegative_m_l2016_201610


namespace measure_six_pints_l2016_201661
-- Importing the necessary library

-- Defining the problem conditions
def total_wine : ℕ := 12
def capacity_8_pint_vessel : ℕ := 8
def capacity_5_pint_vessel : ℕ := 5

-- The problem to prove: it is possible to measure 6 pints into the 8-pint container
theorem measure_six_pints :
  ∃ (n : ℕ), n = 6 ∧ n ≤ capacity_8_pint_vessel := 
sorry

end measure_six_pints_l2016_201661


namespace tom_age_ratio_l2016_201689

-- Define the constants T and N with the given conditions
variables (T N : ℕ)
-- Tom's age T years, sum of three children's ages is also T
-- N years ago, Tom's age was three times the sum of children's ages then

-- We need to prove that T / N = 4 under these conditions
theorem tom_age_ratio (h1 : T = 3 * T - 8 * N) : T / N = 4 :=
sorry

end tom_age_ratio_l2016_201689


namespace marc_trip_equation_l2016_201601

theorem marc_trip_equation (t : ℝ) 
  (before_stop_speed : ℝ := 90)
  (stop_time : ℝ := 0.5)
  (after_stop_speed : ℝ := 110)
  (total_distance : ℝ := 300)
  (total_trip_time : ℝ := 3.5) :
  before_stop_speed * t + after_stop_speed * (total_trip_time - stop_time - t) = total_distance :=
by 
  sorry

end marc_trip_equation_l2016_201601


namespace composite_proposition_l2016_201688

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 5 ≤ 4

noncomputable def q : Prop := ∀ x : ℝ, 0 < x ∧ x < Real.pi / 2 → ¬ (∀ v : ℝ, v = (Real.sin x + 4 / Real.sin x) → v = 4)

theorem composite_proposition : p ∧ ¬q := 
by 
  sorry

end composite_proposition_l2016_201688


namespace pos_int_solutions_3x_2y_841_l2016_201631

theorem pos_int_solutions_3x_2y_841 :
  {n : ℕ // ∃ (x y : ℕ), 3 * x + 2 * y = 841 ∧ x > 0 ∧ y > 0} =
  {n : ℕ // n = 140} := 
sorry

end pos_int_solutions_3x_2y_841_l2016_201631


namespace factor_expression_l2016_201646

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) :=
by
  sorry

end factor_expression_l2016_201646


namespace m_range_iff_four_distinct_real_roots_l2016_201671

noncomputable def four_distinct_real_roots (m : ℝ) : Prop :=
∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
(x1^2 - 4 * |x1| + 5 = m) ∧
(x2^2 - 4 * |x2| + 5 = m) ∧
(x3^2 - 4 * |x3| + 5 = m) ∧
(x4^2 - 4 * |x4| + 5 = m)

theorem m_range_iff_four_distinct_real_roots (m : ℝ) :
  four_distinct_real_roots m ↔ 1 < m ∧ m < 5 :=
sorry

end m_range_iff_four_distinct_real_roots_l2016_201671


namespace gcd_largest_value_l2016_201622

/-- Given two positive integers x and y such that x + y = 780,
    this definition states that the largest possible value of gcd(x, y) is 390. -/
theorem gcd_largest_value (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x + y = 780) : ∃ d, d = Nat.gcd x y ∧ d = 390 :=
sorry

end gcd_largest_value_l2016_201622


namespace scientific_notation_example_l2016_201666

theorem scientific_notation_example : (8485000 : ℝ) = 8.485 * 10 ^ 6 := 
by 
  sorry

end scientific_notation_example_l2016_201666


namespace find_radius_l2016_201635

-- Defining the conditions as given in the math problem
def sectorArea (r : ℝ) (L : ℝ) : ℝ := 0.5 * r * L

theorem find_radius (h1 : sectorArea r 5.5 = 13.75) : r = 5 :=
by sorry

end find_radius_l2016_201635


namespace distance_between_chords_l2016_201660

-- Definitions based on the conditions
structure CircleGeometry where
  radius: ℝ
  d1: ℝ -- distance from the center to the closest chord (34 units)
  d2: ℝ -- distance from the center to the second chord (38 units)
  d3: ℝ -- distance from the center to the outermost chord (38 units)

-- The problem itself
theorem distance_between_chords (circle: CircleGeometry) (h1: circle.d2 = 3) (h2: circle.d1 = 3 * circle.d2) (h3: circle.d3 = circle.d2) :
  2 * circle.d2 = 6 :=
by
  sorry

end distance_between_chords_l2016_201660


namespace min_inquiries_for_parity_l2016_201669

-- Define the variables and predicates
variables (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n)

-- Define the main theorem we need to prove
theorem min_inquiries_for_parity (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n) : 
  ∃ k, (k = m + n - 4) := 
sorry

end min_inquiries_for_parity_l2016_201669


namespace packs_needed_l2016_201629

-- Define the problem conditions
def bulbs_bedroom : ℕ := 2
def bulbs_bathroom : ℕ := 1
def bulbs_kitchen : ℕ := 1
def bulbs_basement : ℕ := 4
def bulbs_pack : ℕ := 2

def total_bulbs_main_areas : ℕ := bulbs_bedroom + bulbs_bathroom + bulbs_kitchen + bulbs_basement
def bulbs_garage : ℕ := total_bulbs_main_areas / 2

def total_bulbs : ℕ := total_bulbs_main_areas + bulbs_garage

def total_packs : ℕ := total_bulbs / bulbs_pack

-- The proof statement
theorem packs_needed : total_packs = 6 :=
by
  sorry

end packs_needed_l2016_201629


namespace P_iff_q_l2016_201615

variables (a b c: ℝ)

def P : Prop := a * c < 0
def q : Prop := ∃ α β : ℝ, α * β < 0 ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0

theorem P_iff_q : P a c ↔ q a b c := 
sorry

end P_iff_q_l2016_201615


namespace triangle_area_l2016_201634

theorem triangle_area (a b c : ℝ) (K : ℝ) (m n p : ℕ) (h1 : a = 10) (h2 : b = 12) (h3 : c = 15)
  (h4 : K = 240 * Real.sqrt 7 / 7)
  (h5 : Int.gcd m p = 1) -- m and p are relatively prime
  (h6 : n ≠ 1 ∧ ¬ (∃ x, x^2 ∣ n ∧ x > 1)) -- n is not divisible by the square of any prime
  : m + n + p = 254 := sorry

end triangle_area_l2016_201634


namespace rational_expression_simplification_l2016_201687

theorem rational_expression_simplification
  (a b c : ℚ) 
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( ((a^2 * b^2) / c^2 - (2 / c) + (1 / (a^2 * b^2)) + (2 * a * b) / c^2 - (2 / (a * b * c))) 
      / ((2 / (a * b)) - (2 * a * b) / c) ) 
      / (101 / c) = - (1 / 202) :=
by sorry

end rational_expression_simplification_l2016_201687


namespace find_speed_of_boat_l2016_201612

theorem find_speed_of_boat (r d t : ℝ) (x : ℝ) (h_rate : r = 4) (h_dist : d = 33.733333333333334) (h_time : t = 44 / 60) 
  (h_eq : d = (x + r) * t) : x = 42.09090909090909 :=
  sorry

end find_speed_of_boat_l2016_201612


namespace minimum_money_lost_l2016_201621

-- Define the conditions and setup the problem

def check_amount : ℕ := 1270
def T_used (F : ℕ) : Σ' T, (T = F + 1 ∨ T = F - 1) :=
sorry

def money_used (T F : ℕ) : ℕ := 10 * T + 50 * F

def total_bills_used (T F : ℕ) : Prop := T + F = 15

theorem minimum_money_lost : (∃ T F, (T = F + 1 ∨ T = F - 1) ∧ T + F = 15 ∧ (check_amount - (10 * T + 50 * F) = 800)) :=
sorry

end minimum_money_lost_l2016_201621


namespace length_of_fence_l2016_201613

theorem length_of_fence (side_length : ℕ) (h : side_length = 28) : 4 * side_length = 112 :=
by
  sorry

end length_of_fence_l2016_201613


namespace area_square_field_l2016_201616

-- Define the side length of the square
def side_length : ℕ := 12

-- Define the area of the square with the given side length
def area_of_square (side : ℕ) : ℕ := side * side

-- The theorem to state and prove
theorem area_square_field : area_of_square side_length = 144 :=
by
  sorry

end area_square_field_l2016_201616


namespace candies_eaten_l2016_201640

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l2016_201640


namespace inradius_of_equal_area_and_perimeter_l2016_201658

theorem inradius_of_equal_area_and_perimeter
  (a b c : ℝ)
  (A : ℝ)
  (h1 : A = a + b + c)
  (s : ℝ := (a + b + c) / 2)
  (h2 : A = s * (2 * A / (a + b + c))) :
  ∃ r : ℝ, r = 2 := by
  sorry

end inradius_of_equal_area_and_perimeter_l2016_201658


namespace cylinder_height_same_volume_as_cone_l2016_201639

theorem cylinder_height_same_volume_as_cone
    (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V : ℝ)
    (h_volume_cone_eq : V = (1 / 3) * Real.pi * r_cone ^ 2 * h_cone)
    (r_cone_val : r_cone = 2)
    (h_cone_val : h_cone = 6)
    (r_cylinder_val : r_cylinder = 1) :
    ∃ h_cylinder : ℝ, (V = Real.pi * r_cylinder ^ 2 * h_cylinder) ∧ h_cylinder = 8 :=
by
  -- Here you would provide the proof for the theorem.
  sorry

end cylinder_height_same_volume_as_cone_l2016_201639


namespace gcd_108_45_l2016_201600

theorem gcd_108_45 :
  ∃ g, g = Nat.gcd 108 45 ∧ g = 9 :=
by
  sorry

end gcd_108_45_l2016_201600


namespace total_surface_area_l2016_201626

theorem total_surface_area (a b c : ℝ)
    (h1 : a + b + c = 40)
    (h2 : a^2 + b^2 + c^2 = 625)
    (h3 : a * b * c = 600) : 
    2 * (a * b + b * c + c * a) = 975 :=
by
  sorry

end total_surface_area_l2016_201626


namespace geometric_sequence_divisible_by_ten_million_l2016_201609

theorem geometric_sequence_divisible_by_ten_million 
  (a1 a2 : ℝ)
  (h1 : a1 = 1 / 2)
  (h2 : a2 = 50) :
  ∀ n : ℕ, (n ≥ 5) → (∃ k : ℕ, (a1 * (a2 / a1)^(n - 1)) = k * 10^7) :=
by
  sorry

end geometric_sequence_divisible_by_ten_million_l2016_201609


namespace value_of_x_m_minus_n_l2016_201679

variables {x : ℝ} {m n : ℝ}

theorem value_of_x_m_minus_n (hx_m : x^m = 6) (hx_n : x^n = 3) : x^(m - n) = 2 := 
by 
  sorry

end value_of_x_m_minus_n_l2016_201679


namespace percentage_of_loss_l2016_201619

theorem percentage_of_loss
    (CP SP : ℝ)
    (h1 : CP = 1200)
    (h2 : SP = 1020)
    (Loss : ℝ)
    (h3 : Loss = CP - SP)
    (Percentage_of_Loss : ℝ)
    (h4 : Percentage_of_Loss = (Loss / CP) * 100) :
  Percentage_of_Loss = 15 := by
  sorry

end percentage_of_loss_l2016_201619


namespace inclination_angle_l2016_201644

theorem inclination_angle (θ : ℝ) (h : 0 ≤ θ ∧ θ < 180) :
  (∀ x y : ℝ, x - y + 3 = 0 → θ = 45) :=
sorry

end inclination_angle_l2016_201644


namespace anna_age_l2016_201675

-- Define the conditions as given in the problem
variable (x : ℕ)
variable (m n : ℕ)

-- Translate the problem statement into Lean
axiom perfect_square_condition : x - 4 = m^2
axiom perfect_cube_condition : x + 3 = n^3

-- The proof problem statement in Lean 4
theorem anna_age : x = 5 :=
by
  sorry

end anna_age_l2016_201675


namespace distance_relation_possible_l2016_201697

-- Define a structure representing points in 2D space
structure Point where
  x : ℤ
  y : ℤ

-- Define the artificial geometry distance function (Euclidean distance)
def varrho (p1 p2 : Point) : ℝ :=
  ((p1.x - p2.x)^2 + (p1.y - p2.y)^2).sqrt

-- Define the non-collinearity condition for points A, B, and C
def non_collinear (A B C : Point) : Prop :=
  ¬(A.x = B.x ∧ B.x = C.x) ∧ ¬(A.y = B.y ∧ B.y = C.y)

theorem distance_relation_possible :
  ∃ (A B C : Point), non_collinear A B C ∧ varrho A C ^ 2 + varrho B C ^ 2 = varrho A B ^ 2 :=
by
  sorry

end distance_relation_possible_l2016_201697


namespace minimize_expression_l2016_201649

theorem minimize_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (a, b) = (15 / 4, 15) ↔ (∀ x y : ℝ, 0 < x → 0 < y → (4 * x + y = 30) → (1 / x + 4 / y) ≥ (1 / (15 / 4) + 4 / 15)) := by
sorry

end minimize_expression_l2016_201649


namespace different_lists_count_l2016_201677

def numberOfLists : Nat := 5

theorem different_lists_count :
  let conditions := ∃ (d : Fin 6 → ℕ), d 0 + d 1 + d 2 + d 3 + d 4 + d 5 = 5 ∧
                                      ∀ i, d i ≤ 5 ∧
                                      ∀ i j, i < j → d i ≥ d j
  conditions →
  numberOfLists = 5 :=
sorry

end different_lists_count_l2016_201677


namespace student_second_subject_percentage_l2016_201695

theorem student_second_subject_percentage (x : ℝ) (h : (50 + x + 90) / 3 = 70) : x = 70 :=
by { sorry }

end student_second_subject_percentage_l2016_201695


namespace nth_odd_positive_integer_is_199_l2016_201696

def nth_odd_positive_integer (n : ℕ) : ℕ :=
  2 * n - 1

theorem nth_odd_positive_integer_is_199 :
  nth_odd_positive_integer 100 = 199 :=
by
  sorry

end nth_odd_positive_integer_is_199_l2016_201696


namespace chairs_to_remove_l2016_201641

-- Defining the conditions
def chairs_per_row : Nat := 15
def total_chairs : Nat := 180
def expected_attendees : Nat := 125

-- Main statement to prove
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ) : 
  chairs_per_row = 15 → 
  total_chairs = 180 → 
  expected_attendees = 125 → 
  ∃ n, total_chairs - (chairs_per_row * n) = 45 ∧ n * chairs_per_row ≥ expected_attendees := 
by
  intros h1 h2 h3
  sorry

end chairs_to_remove_l2016_201641


namespace sum_of_m_and_n_l2016_201668

noncomputable section

variable {a b m n : ℕ}

theorem sum_of_m_and_n 
  (h1 : a = n * b)
  (h2 : (a + b) = m * (a - b)) :
  m + n = 5 :=
sorry

end sum_of_m_and_n_l2016_201668


namespace part1_part2_l2016_201628

def f (x a : ℝ) : ℝ := |x - a| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x 2 > 5 ↔ x < - 1 / 3 ∨ x > 3 :=
by sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ |a - 2|) → a ≤ 3 / 2 :=
by sorry

end part1_part2_l2016_201628


namespace number_of_extremum_points_of_f_l2016_201690

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (x + 1)^3 * Real.exp (x + 1) else (-(x + 1))^3 * Real.exp (-(x + 1))

theorem number_of_extremum_points_of_f :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    ((f (x1 - epsilon) < f x1 ∧ f x1 > f (x1 + epsilon)) ∨ (f (x1 - epsilon) > f x1 ∧ f x1 < f (x1 + epsilon))) ∧
    ((f (x2 - epsilon) < f x2 ∧ f x2 > f (x2 + epsilon)) ∨ (f (x2 - epsilon) > f x2 ∧ f x2 < f (x2 + epsilon))) ∧
    ((f (x3 - epsilon) < f x3 ∧ f x3 > f (x3 + epsilon)) ∨ (f (x3 - epsilon) > f x3 ∧ f x3 < f (x3 + epsilon)))) :=
sorry

end number_of_extremum_points_of_f_l2016_201690


namespace construction_rates_construction_cost_l2016_201670

-- Defining the conditions as Lean hypotheses

def length := 1650
def diff_rate := 30
def time_ratio := 3/2

-- Daily construction rates (questions answered as hypotheses as well)
def daily_rate_A := 60
def daily_rate_B := 90

-- Additional conditions for cost calculations
def cost_A_per_day := 90000
def cost_B_per_day := 120000
def total_days := 14
def alone_days_A := 5

-- Problem stated as proofs to be completed
theorem construction_rates :
  (∀ (x : ℕ), x = daily_rate_A ∧ (x + diff_rate) = daily_rate_B ∧ 
  (1650 / (x + diff_rate)) * (3/2) = (1650 / x) → 
  60 = daily_rate_A ∧ (60 + 30) = daily_rate_B ) :=
by sorry

theorem construction_cost :
  (∀ (m : ℕ), m = alone_days_A ∧ 
  (cost_A_per_day * total_days + cost_B_per_day * (total_days - alone_days_A)) / 1000 = 2340) :=
by sorry

end construction_rates_construction_cost_l2016_201670


namespace total_age_l2016_201606

theorem total_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : b = 8) : a + b + c = 22 :=
by
  sorry

end total_age_l2016_201606


namespace external_angle_at_C_l2016_201642

-- Definitions based on conditions
def angleA : ℝ := 40
def B := 2 * angleA
def sum_of_angles_in_triangle (A B C : ℝ) : Prop := A + B + C = 180
def external_angle (C : ℝ) : ℝ := 180 - C

-- Theorem statement
theorem external_angle_at_C :
  ∃ C : ℝ, sum_of_angles_in_triangle angleA B C ∧ external_angle C = 120 :=
sorry

end external_angle_at_C_l2016_201642


namespace quadratic_roots_distinct_real_l2016_201651

theorem quadratic_roots_distinct_real (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = 0)
    (Δ : ℝ := b^2 - 4 * a * c) (hΔ : Δ > 0) :
    (∀ r1 r2 : ℝ, r1 ≠ r2) :=
by
  sorry

end quadratic_roots_distinct_real_l2016_201651


namespace cos_x_minus_pi_over_3_l2016_201653

theorem cos_x_minus_pi_over_3 (x : ℝ) (h : Real.sin (x + π / 6) = 4 / 5) :
  Real.cos (x - π / 3) = 4 / 5 :=
sorry

end cos_x_minus_pi_over_3_l2016_201653


namespace zero_point_in_interval_l2016_201611

noncomputable def f (x a : ℝ) := 2^x - 2/x - a

theorem zero_point_in_interval (a : ℝ) : 
  (∃ x, 1 < x ∧ x < 2 ∧ f x a = 0) → 0 < a ∧ a < 3 :=
by
  sorry

end zero_point_in_interval_l2016_201611


namespace square_area_l2016_201699

/- Given: 
    1. The area of the isosceles right triangle ΔAEF is 1 cm².
    2. The area of the rectangle EFGH is 10 cm².
- To prove: 
    The area of the square ABCD is 24.5 cm².
-/

theorem square_area
  (h1 : ∃ a : ℝ, (0 < a) ∧ (a * a / 2 = 1))  -- Area of isosceles right triangle ΔAEF is 1 cm²
  (h2 : ∃ w l : ℝ, (w = 2) ∧ (l * w = 10))  -- Area of rectangle EFGH is 10 cm²
  : ∃ s : ℝ, (s * s = 24.5) := -- Area of the square ABCD is 24.5 cm²
sorry

end square_area_l2016_201699


namespace expression_divisible_by_41_l2016_201678

theorem expression_divisible_by_41 (n : ℕ) : 41 ∣ (5 * 7^(2*(n+1)) + 2^(3*n)) :=
  sorry

end expression_divisible_by_41_l2016_201678


namespace number_of_solutions_sine_quadratic_l2016_201698

theorem number_of_solutions_sine_quadratic :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 * Real.pi → 3 * (Real.sin x) ^ 2 - 5 * (Real.sin x) + 2 = 0 →
  ∃ a b c, x = a ∨ x = b ∨ x = c ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c :=
sorry

end number_of_solutions_sine_quadratic_l2016_201698


namespace meena_cookies_left_l2016_201655

def dozen : ℕ := 12

def baked_cookies : ℕ := 5 * dozen
def mr_stone_buys : ℕ := 2 * dozen
def brock_buys : ℕ := 7
def katy_buys : ℕ := 2 * brock_buys
def total_sold : ℕ := mr_stone_buys + brock_buys + katy_buys
def cookies_left : ℕ := baked_cookies - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end meena_cookies_left_l2016_201655


namespace width_of_margin_l2016_201604

-- Given conditions as definitions
def total_area : ℝ := 20 * 30
def percentage_used : ℝ := 0.64
def used_area : ℝ := percentage_used * total_area

-- Definition of the width of the typing area
def width_after_margin (x : ℝ) : ℝ := 20 - 2 * x

-- Definition of the length after top and bottom margins
def length_after_margin : ℝ := 30 - 6

-- Calculate the area used considering the margins
def typing_area (x : ℝ) : ℝ := (width_after_margin x) * length_after_margin

-- Statement to prove
theorem width_of_margin : ∃ x : ℝ, typing_area x = used_area ∧ x = 2 := by
  -- We give the prompt to eventually prove the theorem with the correct value
  sorry

end width_of_margin_l2016_201604
