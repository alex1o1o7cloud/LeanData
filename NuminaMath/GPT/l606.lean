import Mathlib

namespace major_axis_length_of_ellipse_l606_60663

-- Definition of the conditions
def line (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def ellipse (x y : ℝ) (m : ℝ) : Prop := (x^2) / m + (y^2) / 2 = 1
def is_focus (x y m : ℝ) : Prop := line x y ∧ ellipse x y m

theorem major_axis_length_of_ellipse (m : ℝ) (h₀ : m > 0) :
  (∃ (x y : ℝ), is_focus x y m) → 2 * Real.sqrt 6 = 2 * Real.sqrt m :=
sorry

end major_axis_length_of_ellipse_l606_60663


namespace population_increase_l606_60636

theorem population_increase (P : ℕ)
  (birth_rate1_per_1000 : ℕ := 25)
  (death_rate1_per_1000 : ℕ := 12)
  (immigration_rate1 : ℕ := 15000)
  (birth_rate2_per_1000 : ℕ := 30)
  (death_rate2_per_1000 : ℕ := 8)
  (immigration_rate2 : ℕ := 30000)
  (pop_increase1_perc : ℤ := 200)
  (pop_increase2_perc : ℤ := 300) :
  (12 * P - P) / P * 100 = 1100 := by
  sorry

end population_increase_l606_60636


namespace system_solutions_l606_60659

noncomputable def f (t : ℝ) : ℝ := 4 * t^2 / (1 + 4 * t^2)

theorem system_solutions (x y z : ℝ) :
  (f x = y ∧ f y = z ∧ f z = x) ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_l606_60659


namespace half_angle_in_second_quadrant_l606_60615

def quadrant_of_half_alpha (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : Prop :=
  π / 2 < α / 2 ∧ α / 2 < 3 * π / 4

theorem half_angle_in_second_quadrant (α : ℝ) (hα1 : π < α) (hα2 : α < 3 * π / 2) (hcos : abs (Real.cos (α / 2)) = -Real.cos (α / 2)) : quadrant_of_half_alpha α hα1 hα2 hcos :=
sorry

end half_angle_in_second_quadrant_l606_60615


namespace example_problem_l606_60693

theorem example_problem : 2 + 3 * 4 - 5 + 6 / 3 = 11 := by
  sorry

end example_problem_l606_60693


namespace polynomial_inequality_l606_60629

-- Define P(x) as a polynomial with non-negative coefficients
def isNonNegativePolynomial (P : Polynomial ℝ) : Prop :=
  ∀ i, P.coeff i ≥ 0

-- The main theorem, which states that for any polynomial P with non-negative coefficients,
-- if P(1) * P(1) ≥ 1, then P(x) * P(1/x) ≥ 1 for all positive x.
theorem polynomial_inequality (P : Polynomial ℝ) (hP : isNonNegativePolynomial P) (hP1 : P.eval 1 * P.eval 1 ≥ 1) :
  ∀ x : ℝ, 0 < x → P.eval x * P.eval (1 / x) ≥ 1 :=
by {
  sorry
}

end polynomial_inequality_l606_60629


namespace expense_of_5_yuan_is_minus_5_yuan_l606_60639

def income (x : Int) : Int :=
  x

def expense (x : Int) : Int :=
  -x

theorem expense_of_5_yuan_is_minus_5_yuan : expense 5 = -5 :=
by
  unfold expense
  sorry

end expense_of_5_yuan_is_minus_5_yuan_l606_60639


namespace equal_elements_l606_60651

theorem equal_elements {n : ℕ} (a : ℕ → ℝ) (h₁ : n ≥ 2) (h₂ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≠ -1) 
  (h₃ : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 2) = (a i ^ 2 + a i) / (a (i + 1) + 1)) 
  (hn1 : a (n + 1) = a 1) (hn2 : a (n + 2) = a 2) :
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i = a 1 := by
  sorry

end equal_elements_l606_60651


namespace part1_part2_part3_l606_60676

-- Part (1): Proving \( p \implies m > \frac{3}{2} \)
theorem part1 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0) → (m > 3 / 2) :=
by
  sorry

-- Part (2): Proving \( q \implies (m < -1 \text{ or } m > 2) \)
theorem part2 (m : ℝ) : (∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → (m < -1 ∨ m > 2) :=
by
  sorry

-- Part (3): Proving \( (p ∨ q) \implies ((-\infty, -1) ∪ (\frac{3}{2}, +\infty)) \)
theorem part3 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0 ∨ ∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → ((m < -1) ∨ (3 / 2 < m)) :=
by
  sorry

end part1_part2_part3_l606_60676


namespace parabola_equation_l606_60670

theorem parabola_equation (x y : ℝ) (hx : x = -2) (hy : y = 3) :
  (y^2 = -(9 / 2) * x) ∨ (x^2 = (4 / 3) * y) :=
by
  sorry

end parabola_equation_l606_60670


namespace farmer_shipped_30_boxes_this_week_l606_60656

-- Defining the given conditions
def last_week_boxes : ℕ := 10
def last_week_pomelos : ℕ := 240
def this_week_dozen : ℕ := 60
def pomelos_per_dozen : ℕ := 12

-- Translating conditions into mathematical statements
def pomelos_per_box_last_week : ℕ := last_week_pomelos / last_week_boxes
def this_week_pomelos_total : ℕ := this_week_dozen * pomelos_per_dozen
def boxes_shipped_this_week : ℕ := this_week_pomelos_total / pomelos_per_box_last_week

-- The theorem we prove, that given the conditions, the number of boxes shipped this week is 30.
theorem farmer_shipped_30_boxes_this_week :
  boxes_shipped_this_week = 30 :=
sorry

end farmer_shipped_30_boxes_this_week_l606_60656


namespace length_of_AB_l606_60675

-- Defining the parabola and the condition on x1 and x2
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def condition (x1 x2 : ℝ) : Prop := x1 + x2 = 9

-- The main statement to prove |AB| = 11
theorem length_of_AB (x1 x2 y1 y2 : ℝ) (h1 : parabola x1 y1) (h2 : parabola x2 y2) (hx : condition x1 x2) :
  abs (x1 - x2) + abs (y1 - y2) = 11 :=
sorry

end length_of_AB_l606_60675


namespace ratio_of_areas_l606_60608

theorem ratio_of_areas
  (PQ QR RP : ℝ)
  (PQ_pos : 0 < PQ)
  (QR_pos : 0 < QR)
  (RP_pos : 0 < RP)
  (s t u : ℝ)
  (s_pos : 0 < s)
  (t_pos : 0 < t)
  (u_pos : 0 < u)
  (h1 : s + t + u = 3 / 4)
  (h2 : s^2 + t^2 + u^2 = 1 / 2)
  : (1 - (s * (1 - u) + t * (1 - s) + u * (1 - t))) = 7 / 32 := by
  sorry

end ratio_of_areas_l606_60608


namespace work_completed_together_in_4_days_l606_60683

/-- A can do the work in 6 days. -/
def A_work_rate : ℚ := 1 / 6

/-- B can do the work in 12 days. -/
def B_work_rate : ℚ := 1 / 12

/-- Combined work rate of A and B working together. -/
def combined_work_rate : ℚ := A_work_rate + B_work_rate

/-- Number of days for A and B to complete the work together. -/
def days_to_complete : ℚ := 1 / combined_work_rate

theorem work_completed_together_in_4_days : days_to_complete = 4 := by
  sorry

end work_completed_together_in_4_days_l606_60683


namespace find_coefficients_l606_60625

theorem find_coefficients (a b : ℚ) (h_a_nonzero : a ≠ 0)
  (h_prod : (3 * b - 2 * a = 0) ∧ (-2 * b + 3 = 0)) : 
  a = 9 / 4 ∧ b = 3 / 2 :=
by
  sorry

end find_coefficients_l606_60625


namespace arithmetic_sequence_a5_l606_60626

theorem arithmetic_sequence_a5 {a : ℕ → ℕ} 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 2 + a 8 = 12) : 
  a 5 = 6 :=
by
  sorry

end arithmetic_sequence_a5_l606_60626


namespace min_value_expression_l606_60606

theorem min_value_expression (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : 4 * a + b = 1) :
  (1 / a) + (4 / b) = 16 := sorry

end min_value_expression_l606_60606


namespace concert_revenue_l606_60680

-- Define the prices and attendees
def adult_price := 26
def teenager_price := 18
def children_price := adult_price / 2
def num_adults := 183
def num_teenagers := 75
def num_children := 28

-- Calculate total revenue
def total_revenue := num_adults * adult_price + num_teenagers * teenager_price + num_children * children_price

-- The goal is to prove that total_revenue equals 6472
theorem concert_revenue : total_revenue = 6472 :=
by
  sorry

end concert_revenue_l606_60680


namespace folding_cranes_together_l606_60684

theorem folding_cranes_together (rateA rateB combined_time : ℝ)
  (hA : rateA = 1 / 30)
  (hB : rateB = 1 / 45)
  (combined_rate : ℝ := rateA + rateB)
  (h_combined_rate : combined_rate = 1 / combined_time):
  combined_time = 18 :=
by
  sorry

end folding_cranes_together_l606_60684


namespace sequence_v_n_l606_60677

theorem sequence_v_n (v : ℕ → ℝ)
  (h_recurr : ∀ n, v (n+2) = 3 * v (n+1) - v n)
  (h_init1 : v 3 = 16)
  (h_init2 : v 6 = 211) : 
  v 5 = 81.125 :=
sorry

end sequence_v_n_l606_60677


namespace sum_f_neg_l606_60644

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_neg {x1 x2 x3 : ℝ}
  (h1 : x1 + x2 > 0)
  (h2 : x2 + x3 > 0)
  (h3 : x3 + x1 > 0) :
  f x1 + f x2 + f x3 < 0 :=
by
  sorry

end sum_f_neg_l606_60644


namespace solution_set_inequality_l606_60679

theorem solution_set_inequality (x : ℝ) : (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 :=
by
  -- Proof omitted
  sorry

end solution_set_inequality_l606_60679


namespace pump_fills_tank_without_leak_l606_60624

variable (T : ℝ)
-- Condition: The effective rate with the leak is equal to the rate it takes for both to fill the tank.
def effective_rate_with_leak (T : ℝ) : Prop :=
  1 / T - 1 / 21 = 1 / 3.5

-- Conclude: the time it takes the pump to fill the tank without the leak
theorem pump_fills_tank_without_leak : effective_rate_with_leak T → T = 3 :=
by
  intro h
  sorry

end pump_fills_tank_without_leak_l606_60624


namespace correct_equations_l606_60619

theorem correct_equations (m n : ℕ) (h1 : n = 4 * m - 2) (h2 : n = 2 * m + 58) :
  (4 * m - 2 = 2 * m + 58 ∨ (n + 2) / 4 = (n - 58) / 2) :=
by
  sorry

end correct_equations_l606_60619


namespace last_two_digits_of_7_pow_2016_l606_60694

theorem last_two_digits_of_7_pow_2016 : (7^2016 : ℕ) % 100 = 1 := 
by {
  sorry
}

end last_two_digits_of_7_pow_2016_l606_60694


namespace no_valid_angles_l606_60664

open Real

theorem no_valid_angles (θ : ℝ) (h1 : 0 < θ) (h2 : θ < 2 * π)
    (h3 : ∀ k : ℤ, θ ≠ k * (π / 2))
    (h4 : cos θ * tan θ = sin θ ^ 3) : false :=
by
  -- The proof goes here
  sorry

end no_valid_angles_l606_60664


namespace cost_of_camel_proof_l606_60668

noncomputable def cost_of_camel (C H O E : ℕ) : ℕ :=
  if 10 * C = 24 * H ∧ 16 * H = 4 * O ∧ 6 * O = 4 * E ∧ 10 * E = 120000 then 4800 else 0

theorem cost_of_camel_proof (C H O E : ℕ) 
  (h1 : 10 * C = 24 * H) (h2 : 16 * H = 4 * O) (h3 : 6 * O = 4 * E) (h4 : 10 * E = 120000) :
  cost_of_camel C H O E = 4800 :=
by
  sorry

end cost_of_camel_proof_l606_60668


namespace unit_vector_opposite_AB_is_l606_60689

open Real

noncomputable def unit_vector_opposite_dir (A B : ℝ × ℝ) : ℝ × ℝ :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BA := (-AB.1, -AB.2)
  let mag_BA := sqrt (BA.1^2 + BA.2^2)
  (BA.1 / mag_BA, BA.2 / mag_BA)

theorem unit_vector_opposite_AB_is (A B : ℝ × ℝ) (hA : A = (1, 2)) (hB : B = (-2, 6)) :
  unit_vector_opposite_dir A B = (3/5, -4/5) :=
by
  sorry

end unit_vector_opposite_AB_is_l606_60689


namespace gasoline_price_increase_l606_60643

theorem gasoline_price_increase (highest_price lowest_price : ℝ) (h1 : highest_price = 24) (h2 : lowest_price = 15) : 
  ((highest_price - lowest_price) / lowest_price) * 100 = 60 :=
by
  sorry

end gasoline_price_increase_l606_60643


namespace find_value_l606_60617

theorem find_value : 3 + 2 * (8 - 3) = 13 := by
  sorry

end find_value_l606_60617


namespace chipped_marbles_is_22_l606_60637

def bags : List ℕ := [20, 22, 25, 30, 32, 34, 36]

-- Jane and George take some bags and one bag with chipped marbles is left.
theorem chipped_marbles_is_22
  (h1 : ∃ (jane_bags george_bags : List ℕ) (remaining_bag : ℕ),
    (jane_bags ++ george_bags ++ [remaining_bag] = bags ∧
     jane_bags.length = 3 ∧
     (george_bags.length = 2 ∨ george_bags.length = 3) ∧
     3 * remaining_bag = List.sum jane_bags + List.sum george_bags)) :
  ∃ (c : ℕ), c = 22 := 
sorry

end chipped_marbles_is_22_l606_60637


namespace three_g_of_x_l606_60616

noncomputable def g (x : ℝ) : ℝ := 3 / (3 + x)

theorem three_g_of_x (x : ℝ) (h : x > 0) : 3 * g x = 27 / (9 + x) :=
by
  sorry

end three_g_of_x_l606_60616


namespace proof_problem_l606_60691

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1 / x

noncomputable def f'' (x : ℝ) : ℝ := Real.exp x + 2 / x^3

theorem proof_problem {x0 m n : ℝ} (hx0_pos : 0 < x0)
  (H : f'' x0 = 0) (hm : 0 < m) (hmx0 : m < x0) (hn : x0 < n) :
  f'' m < 0 ∧ f'' n > 0 := sorry

end proof_problem_l606_60691


namespace john_newspapers_l606_60611

theorem john_newspapers (N : ℕ) (selling_price buying_price total_cost total_revenue : ℝ) 
  (h1 : selling_price = 2)
  (h2 : buying_price = 0.25 * selling_price)
  (h3 : total_cost = N * buying_price)
  (h4 : total_revenue = 0.8 * N * selling_price)
  (h5 : total_revenue - total_cost = 550) :
  N = 500 := 
by 
  -- actual proof here
  sorry

end john_newspapers_l606_60611


namespace result_more_than_half_l606_60646

theorem result_more_than_half (x : ℕ) (h : x = 4) : (2 * x + 5) - (x / 2) = 11 := by
  sorry

end result_more_than_half_l606_60646


namespace counterexamples_count_l606_60647

def sum_of_digits (n : Nat) : Nat :=
  -- Function to calculate the sum of digits of n
  sorry

def no_zeros (n : Nat) : Prop :=
  -- Function to check that there are no zeros in the digits of n
  sorry

def is_prime (n : Nat) : Prop :=
  -- Function to check if a number is prime
  sorry

theorem counterexamples_count : 
  ∃ (M : List Nat), 
  (∀ m ∈ M, sum_of_digits m = 5 ∧ no_zeros m) ∧ 
  (∀ m ∈ M, ¬ is_prime m) ∧
  M.length = 9 := 
sorry

end counterexamples_count_l606_60647


namespace carsProducedInEurope_l606_60609

-- Definitions of the conditions
def carsProducedInNorthAmerica : ℕ := 3884
def totalCarsProduced : ℕ := 6755

-- Theorem statement
theorem carsProducedInEurope : ∃ (carsProducedInEurope : ℕ), totalCarsProduced = carsProducedInNorthAmerica + carsProducedInEurope ∧ carsProducedInEurope = 2871 := by
  sorry

end carsProducedInEurope_l606_60609


namespace integer_values_abs_lt_5pi_l606_60699

theorem integer_values_abs_lt_5pi : 
  ∃ n : ℕ, n = 31 ∧ ∀ x : ℤ, |(x : ℝ)| < 5 * Real.pi → x ∈ (Finset.Icc (-15) 15) := 
sorry

end integer_values_abs_lt_5pi_l606_60699


namespace constant_term_of_product_l606_60681

def P(x: ℝ) : ℝ := x^6 + 2 * x^2 + 3
def Q(x: ℝ) : ℝ := x^4 + x^3 + 4
def R(x: ℝ) : ℝ := 2 * x^2 + 3 * x + 7

theorem constant_term_of_product :
  let C := (P 0) * (Q 0) * (R 0)
  C = 84 :=
by
  let C := (P 0) * (Q 0) * (R 0)
  show C = 84
  sorry

end constant_term_of_product_l606_60681


namespace reciprocal_inequalities_l606_60648

theorem reciprocal_inequalities (a b c : ℝ)
  (h1 : -1 < a ∧ a < -2/3)
  (h2 : -1/3 < b ∧ b < 0)
  (h3 : 1 < c) :
  1/c < 1/(b - a) ∧ 1/(b - a) < 1/(a * b) :=
by
  sorry

end reciprocal_inequalities_l606_60648


namespace total_winnings_l606_60682

theorem total_winnings (x : ℝ)
  (h1 : x / 4 = first_person_share)
  (h2 : x / 7 = second_person_share)
  (h3 : third_person_share = 17)
  (h4 : first_person_share + second_person_share + third_person_share = x) :
  x = 28 := 
by sorry

end total_winnings_l606_60682


namespace problem_statement_l606_60601

-- Define the given condition
def cond_1 (x : ℝ) := x + 1/x = 5

-- State the theorem that needs to be proven
theorem problem_statement (x : ℝ) (h : cond_1 x) : x^3 + 1/x^3 = 110 :=
sorry

end problem_statement_l606_60601


namespace find_a_prove_inequality_l606_60618

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x + 2 * x + a * Real.log x

theorem find_a (a : ℝ) (h : (2 * Real.exp 1 + 2 + a) * (-1 / 2) = -1) : a = -2 * Real.exp 1 :=
by
  sorry

theorem prove_inequality (a : ℝ) (h1 : a = -2 * Real.exp 1) :
    ∀ x : ℝ, x > 0 → f x a > x^2 + 2 :=
by
  sorry

end find_a_prove_inequality_l606_60618


namespace total_money_of_james_and_ali_l606_60685

def jamesOwns : ℕ := 145
def jamesAliDifference : ℕ := 40
def aliOwns : ℕ := jamesOwns - jamesAliDifference

theorem total_money_of_james_and_ali :
  jamesOwns + aliOwns = 250 := by
  sorry

end total_money_of_james_and_ali_l606_60685


namespace wristband_distribution_l606_60610

open Nat 

theorem wristband_distribution (x y : ℕ) 
  (h1 : 2 * x + 2 * y = 460) 
  (h2 : 2 * x = 3 * y) : x = 138 :=
sorry

end wristband_distribution_l606_60610


namespace cos_sin_eq_l606_60657

theorem cos_sin_eq (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (Real.sin x + 3 * Real.cos x = (2 * Real.sqrt 6 - 3) / 5) ∨
  (Real.sin x + 3 * Real.cos x = -(2 * Real.sqrt 6 + 3) / 5) := 
by
  sorry

end cos_sin_eq_l606_60657


namespace direct_variation_y_value_l606_60698

theorem direct_variation_y_value (x y : ℝ) (hx1 : x ≤ 10 → y = 3 * x)
  (hx2 : x > 10 → y = 6 * x) : 
  x = 20 → y = 120 := by
  sorry

end direct_variation_y_value_l606_60698


namespace total_selection_methods_l606_60633

def num_courses_group_A := 3
def num_courses_group_B := 4
def total_courses_selected := 3

theorem total_selection_methods 
  (at_least_one_from_each : num_courses_group_A > 0 ∧ num_courses_group_B > 0)
  (total_courses : total_courses_selected = 3) :
  ∃ N, N = 30 :=
sorry

end total_selection_methods_l606_60633


namespace maximum_q_minus_r_l606_60602

theorem maximum_q_minus_r : 
  ∀ q r : ℕ, (1027 = 23 * q + r) ∧ (q > 0) ∧ (r > 0) → q - r ≤ 29 := 
by
  sorry

end maximum_q_minus_r_l606_60602


namespace min_value_x_squared_y_squared_z_squared_l606_60650

theorem min_value_x_squared_y_squared_z_squared
  (x y z : ℝ)
  (h : x + 2 * y + 3 * z = 6) :
  x^2 + y^2 + z^2 ≥ (18 / 7) :=
sorry

end min_value_x_squared_y_squared_z_squared_l606_60650


namespace not_possible_in_five_trips_possible_in_six_trips_l606_60614

def truck_capacity := 2000
def rice_sacks := 150
def corn_sacks := 100
def rice_weight_per_sack := 60
def corn_weight_per_sack := 25

def total_rice_weight := rice_sacks * rice_weight_per_sack
def total_corn_weight := corn_sacks * corn_weight_per_sack
def total_weight := total_rice_weight + total_corn_weight

theorem not_possible_in_five_trips : total_weight > 5 * truck_capacity :=
by
  sorry

theorem possible_in_six_trips : total_weight <= 6 * truck_capacity :=
by
  sorry

#print axioms not_possible_in_five_trips
#print axioms possible_in_six_trips

end not_possible_in_five_trips_possible_in_six_trips_l606_60614


namespace total_balloons_sam_and_dan_l606_60695

noncomputable def sam_initial_balloons : ℝ := 46.0
noncomputable def balloons_given_to_fred : ℝ := 10.0
noncomputable def dan_balloons : ℝ := 16.0

theorem total_balloons_sam_and_dan :
  (sam_initial_balloons - balloons_given_to_fred) + dan_balloons = 52.0 := 
by 
  sorry

end total_balloons_sam_and_dan_l606_60695


namespace compare_powers_l606_60640

theorem compare_powers:
  (2 ^ 2023) * (7 ^ 2023) < (3 ^ 2023) * (5 ^ 2023) :=
  sorry

end compare_powers_l606_60640


namespace fraction_addition_l606_60603

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end fraction_addition_l606_60603


namespace final_cost_is_35_l606_60612

-- Definitions based on conditions
def original_price : ℕ := 50
def discount_rate : ℚ := 0.30
def discount_amount : ℚ := original_price * discount_rate
def final_cost : ℚ := original_price - discount_amount

-- The theorem we need to prove
theorem final_cost_is_35 : final_cost = 35 := by
  sorry

end final_cost_is_35_l606_60612


namespace least_power_divisible_by_240_l606_60696

theorem least_power_divisible_by_240 (n : ℕ) (a : ℕ) (h_a : a = 60) (h : a^n % 240 = 0) : 
  n = 2 :=
by
  sorry

end least_power_divisible_by_240_l606_60696


namespace probability_leftmost_blue_off_rightmost_red_on_l606_60638

noncomputable def calculate_probability : ℚ :=
  let total_arrangements := Nat.choose 8 4
  let total_on_choices := Nat.choose 8 4
  let favorable_arrangements := Nat.choose 6 3 * Nat.choose 7 3
  favorable_arrangements / (total_arrangements * total_on_choices)

theorem probability_leftmost_blue_off_rightmost_red_on :
  calculate_probability = 1 / 7 := 
by
  sorry

end probability_leftmost_blue_off_rightmost_red_on_l606_60638


namespace certain_number_is_l606_60635

theorem certain_number_is (x : ℝ) : 
  x * (-4.5) = 2 * (-4.5) - 36 → x = 10 :=
by
  intro h
  -- proof goes here
  sorry

end certain_number_is_l606_60635


namespace blue_balls_needed_l606_60634

-- Conditions
variables (R Y B W : ℝ)
axiom h1 : 2 * R = 5 * B
axiom h2 : 3 * Y = 7 * B
axiom h3 : 9 * B = 6 * W

-- Proof Problem
theorem blue_balls_needed : (3 * R + 4 * Y + 3 * W) = (64 / 3) * B := by
  sorry

end blue_balls_needed_l606_60634


namespace sum_of_reciprocals_of_squares_l606_60673

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 41) :
  (1 / (a^2) + 1 / (b^2)) = 1682 / 1681 := sorry

end sum_of_reciprocals_of_squares_l606_60673


namespace no_real_solutions_eq_l606_60632

theorem no_real_solutions_eq (x y : ℝ) :
  x^2 + y^2 - 2 * x + 4 * y + 6 ≠ 0 :=
sorry

end no_real_solutions_eq_l606_60632


namespace evaluate_expr_l606_60688

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 5

theorem evaluate_expr : 3 * g 2 + 2 * g (-4) = 169 :=
by
  sorry

end evaluate_expr_l606_60688


namespace Sam_has_seven_watermelons_l606_60621

-- Declare the initial number of watermelons
def initial_watermelons : Nat := 4

-- Declare the additional number of watermelons Sam grew
def more_watermelons : Nat := 3

-- Prove that the total number of watermelons is 7
theorem Sam_has_seven_watermelons : initial_watermelons + more_watermelons = 7 :=
by
  sorry

end Sam_has_seven_watermelons_l606_60621


namespace original_team_members_l606_60641

theorem original_team_members (m p total_points : ℕ) (h_m : m = 3) (h_p : p = 2) (h_total : total_points = 12) :
  (total_points / p) + m = 9 := by
  sorry

end original_team_members_l606_60641


namespace total_numbers_l606_60630

-- Setting up constants and conditions
variables (n : ℕ)
variables (s1 s2 s3 : ℕ → ℝ)

-- Conditions
axiom avg_all : (s1 n + s2 n + s3 n) / n = 2.5
axiom avg_2_1 : s1 2 / 2 = 1.1
axiom avg_2_2 : s2 2 / 2 = 1.4
axiom avg_2_3 : s3 2 / 2 = 5.0

-- Proposed theorem to prove
theorem total_numbers : n = 6 :=
by
  sorry

end total_numbers_l606_60630


namespace min_needed_framing_l606_60666

-- Define the original dimensions of the picture
def original_width_inch : ℕ := 5
def original_height_inch : ℕ := 7

-- Define the factor by which the dimensions are doubled
def doubling_factor : ℕ := 2

-- Define the width of the border
def border_width_inch : ℕ := 3

-- Define the function to calculate the new dimensions after doubling
def new_width_inch : ℕ := original_width_inch * doubling_factor
def new_height_inch : ℕ := original_height_inch * doubling_factor

-- Define the function to calculate dimensions including the border
def total_width_inch : ℕ := new_width_inch + 2 * border_width_inch
def total_height_inch : ℕ := new_height_inch + 2 * border_width_inch

-- Define the function to calculate the perimeter of the picture with border
def perimeter_inch : ℕ := 2 * (total_width_inch + total_height_inch)

-- Conversision from inches to feet (1 foot = 12 inches)
def inch_to_foot_conversion_factor : ℕ := 12

-- Define the function to calculate the minimum linear feet of framing needed
noncomputable def min_linear_feet_of_framing : ℕ := (perimeter_inch + inch_to_foot_conversion_factor - 1) / inch_to_foot_conversion_factor

-- The main theorem statement
theorem min_needed_framing : min_linear_feet_of_framing = 6 := by
  -- Proof construction is omitted as per the instructions
  sorry

end min_needed_framing_l606_60666


namespace max_value_ln_x_minus_x_on_interval_l606_60642

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem max_value_ln_x_minus_x_on_interval : 
  ∃ x ∈ Set.Ioc 0 (Real.exp 1), ∀ y ∈ Set.Ioc 0 (Real.exp 1), f y ≤ f x ∧ f x = -1 :=
by
  sorry

end max_value_ln_x_minus_x_on_interval_l606_60642


namespace digits_property_l606_60654

theorem digits_property (n : ℕ) (h : 100 ≤ n ∧ n < 1000) :
  (∃ (f : ℕ → Prop), ∀ d ∈ [n / 100, (n / 10) % 10, n % 10], f d ∧ (¬ d = 0 ∧ ¬ Nat.Prime d)) ↔ 
  (∀ d ∈ [n / 100, (n / 10) % 10, n % 10], d ∈ [1, 4, 6, 8, 9]) :=
sorry

end digits_property_l606_60654


namespace problem_statement_l606_60690

theorem problem_statement (x y : ℝ) : 
  ((-3 * x * y^2)^3 * (-6 * x^2 * y) / (9 * x^4 * y^5) = 18 * x * y^2) :=
by sorry

end problem_statement_l606_60690


namespace area_of_rectangle_l606_60653

-- Definitions from problem conditions
variable (AB CD x : ℝ)
variable (h1 : AB = 24)
variable (h2 : CD = 60)
variable (h3 : BC = x)
variable (h4 : BF = 2 * x)
variable (h5 : similar (triangle AEB) (triangle FDC))

-- Goal: Prove the area of rectangle BCFE
theorem area_of_rectangle (h1 : AB = 24) (h2 : CD = 60) (x y : ℝ) 
  (h3 : BC = x) (h4 : BF = 2 * x) (h5 : BC * BF = y) : y = 1440 :=
sorry -- proof will be provided here

end area_of_rectangle_l606_60653


namespace trip_to_market_distance_l606_60631

theorem trip_to_market_distance 
  (school_trip_one_way : ℝ) (school_days_per_week : ℕ) 
  (weekly_total_mileage : ℝ) (round_trips_per_day : ℕ) (market_trip_count : ℕ) :
  (school_trip_one_way = 2.5) →
  (school_days_per_week = 4) →
  (round_trips_per_day = 2) →
  (weekly_total_mileage = 44) →
  (market_trip_count = 1) →
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  market_trip_distance = 2 :=
by
  intros h1 h2 h3 h4 h5
  let school_mileage := (school_trip_one_way * 2 * round_trips_per_day * school_days_per_week)
  let total_market_mileage := weekly_total_mileage - school_mileage
  let market_trip_distance := total_market_mileage / (2 * market_trip_count)
  sorry

end trip_to_market_distance_l606_60631


namespace smallest_number_is_61_point_4_l606_60687

theorem smallest_number_is_61_point_4 (x y z t : ℝ)
  (h1 : y = 2 * x)
  (h2 : z = 4 * y)
  (h3 : t = (y + z) / 3)
  (h4 : (x + y + z + t) / 4 = 220) :
  x = 2640 / 43 :=
by sorry

end smallest_number_is_61_point_4_l606_60687


namespace maximum_m_value_l606_60622

variable {a b c : ℝ}

noncomputable def maximum_m : ℝ := 9/8

theorem maximum_m_value 
  (h1 : (a - b)^2 + (b - c)^2 + (c - a)^2 ≥ maximum_m * a^2)
  (h2 : b^2 - 4 * a * c ≥ 0) : 
  maximum_m = 9 / 8 :=
sorry

end maximum_m_value_l606_60622


namespace inequality_proof_l606_60607

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) :=
by
  sorry

end inequality_proof_l606_60607


namespace find_y_l606_60672

theorem find_y 
  (x y : ℝ) 
  (h1 : 3 * x - 2 * y = 18) 
  (h2 : x + 2 * y = 10) : 
  y = 1.5 := 
by 
  sorry

end find_y_l606_60672


namespace molecular_weight_l606_60604

-- Definitions of the molar masses of the elements
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_I : ℝ := 126.90
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.07
def molar_mass_O : ℝ := 16.00

-- Definition of the molar masses of the compounds
def molar_mass_NH4I : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_I
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Number of moles
def moles_NH4I : ℝ := 3
def moles_CaSO4 : ℝ := 2

-- Total mass calculation
def total_mass : ℝ :=
  moles_NH4I * molar_mass_NH4I + 
  moles_CaSO4 * molar_mass_CaSO4

-- Problem statement
theorem molecular_weight : total_mass = 707.15 := by
  sorry

end molecular_weight_l606_60604


namespace number_of_welders_left_l606_60674

-- Definitions for the given problem
def total_welders : ℕ := 36
def initial_days : ℝ := 1
def remaining_days : ℝ := 3.0000000000000004
def total_days : ℝ := 3

-- Condition equations
variable (r : ℝ) -- rate at which each welder works
variable (W : ℝ) -- total work

-- Equation representing initial total work
def initial_work : W = total_welders * r * total_days := by sorry

-- Welders who left for another project
variable (X : ℕ) -- number of welders who left

-- Equation representing remaining work
def remaining_work : (total_welders - X) * r * remaining_days = W - (total_welders * r * initial_days) := by sorry

-- Theorem to prove
theorem number_of_welders_left :
  (total_welders * total_days : ℝ) = W →
  (total_welders - X) * remaining_days = W - (total_welders * r * initial_days) →
  X = 12 :=
sorry

end number_of_welders_left_l606_60674


namespace missy_yells_total_l606_60671

variable {O S M : ℕ}
variable (yells_at_obedient : ℕ)

-- Conditions:
def yells_stubborn (yells_at_obedient : ℕ) : ℕ := 4 * yells_at_obedient
def yells_mischievous (yells_at_obedient : ℕ) : ℕ := 2 * yells_at_obedient

-- Prove the total yells equal to 84 when yells_at_obedient = 12
theorem missy_yells_total (h : yells_at_obedient = 12) :
  yells_at_obedient + yells_stubborn yells_at_obedient + yells_mischievous yells_at_obedient = 84 :=
by
  sorry

end missy_yells_total_l606_60671


namespace total_population_l606_60658

-- Definitions based on given conditions
variables (b g t : ℕ)
variables (h1 : b = 4 * g) (h2 : g = 8 * t)

-- Theorem statement
theorem total_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * t :=
by
  sorry

end total_population_l606_60658


namespace customer_paid_amount_l606_60620

def cost_price : Real := 7239.13
def percentage_increase : Real := 0.15
def selling_price := (1 + percentage_increase) * cost_price

theorem customer_paid_amount :
  selling_price = 8325.00 :=
by
  sorry

end customer_paid_amount_l606_60620


namespace jellybean_probability_l606_60649

theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 6
  let blue_jellybeans := 3
  let white_jellybeans := 6
  let total_chosen := 4
  let total_combinations := Nat.choose total_jellybeans total_chosen
  let red_combinations := Nat.choose red_jellybeans 3
  let non_red_combinations := Nat.choose (blue_jellybeans + white_jellybeans) 1
  let successful_outcomes := red_combinations * non_red_combinations
  let probability := (successful_outcomes : ℚ) / total_combinations
  probability = 4 / 91 :=
by 
  sorry

end jellybean_probability_l606_60649


namespace youngest_brother_age_l606_60686

theorem youngest_brother_age 
  (Rick_age : ℕ)
  (oldest_brother_age : ℕ)
  (middle_brother_age : ℕ)
  (smallest_brother_age : ℕ)
  (youngest_brother_age : ℕ)
  (h1 : Rick_age = 15)
  (h2 : oldest_brother_age = 2 * Rick_age)
  (h3 : middle_brother_age = oldest_brother_age / 3)
  (h4 : smallest_brother_age = middle_brother_age / 2)
  (h5 : youngest_brother_age = smallest_brother_age - 2) :
  youngest_brother_age = 3 := 
sorry

end youngest_brother_age_l606_60686


namespace at_least_one_not_land_designated_area_l606_60600

variable (p q : Prop)

theorem at_least_one_not_land_designated_area : ¬p ∨ ¬q ↔ ¬ (p ∧ q) :=
by sorry

end at_least_one_not_land_designated_area_l606_60600


namespace find_a_l606_60667

theorem find_a (x y a : ℕ) (h₁ : x = 2) (h₂ : y = 3) (h₃ : a * x + 3 * y = 13) : a = 2 :=
by 
  sorry

end find_a_l606_60667


namespace least_non_lucky_multiple_of_12_l606_60692

/- Defines what it means for a number to be a lucky integer -/
def isLucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

/- Proves the least positive multiple of 12 that is not a lucky integer is 96 -/
theorem least_non_lucky_multiple_of_12 : ∃ n, n % 12 = 0 ∧ ¬isLucky n ∧ ∀ m, m % 12 = 0 ∧ ¬isLucky m → n ≤ m :=
  by
  sorry

end least_non_lucky_multiple_of_12_l606_60692


namespace james_total_chore_time_l606_60697

theorem james_total_chore_time
  (V C L : ℝ)
  (hV : V = 3)
  (hC : C = 3 * V)
  (hL : L = C / 2) :
  V + C + L = 16.5 := by
  sorry

end james_total_chore_time_l606_60697


namespace probability_neither_defective_l606_60669

noncomputable def n := 9
noncomputable def k := 2
noncomputable def total_pens := 9
noncomputable def defective_pens := 3
noncomputable def non_defective_pens := total_pens - defective_pens

noncomputable def total_combinations := Nat.choose total_pens k
noncomputable def non_defective_combinations := Nat.choose non_defective_pens k

theorem probability_neither_defective :
  (non_defective_combinations : ℚ) / total_combinations = 5 / 12 := by
sorry

end probability_neither_defective_l606_60669


namespace find_consecutive_integers_sum_eq_l606_60665

theorem find_consecutive_integers_sum_eq 
    (M : ℤ) : ∃ n k : ℤ, (0 ≤ k ∧ k ≤ 9) ∧ (M = (9 * n + 45 - k)) := 
sorry

end find_consecutive_integers_sum_eq_l606_60665


namespace average_expenditure_Feb_to_July_l606_60660

theorem average_expenditure_Feb_to_July (avg_Jan_to_Jun : ℝ) (spend_Jan : ℝ) (spend_July : ℝ) 
    (total_Jan_to_Jun : avg_Jan_to_Jun = 4200) (spend_Jan_eq : spend_Jan = 1200) (spend_July_eq : spend_July = 1500) :
    (4200 * 6 - 1200 + 1500) / 6 = 4250 :=
by
  sorry

end average_expenditure_Feb_to_July_l606_60660


namespace find_four_numbers_l606_60605

theorem find_four_numbers
  (a d : ℕ)
  (h_pos : 0 < a - d ∧ 0 < a ∧ 0 < a + d)
  (h_sum : (a - d) + a + (a + d) = 48)
  (b c : ℕ)
  (h_geo : b = a ∧ c = a + d)
  (last : ℕ)
  (h_last_val : last = 25)
  (h_geometric_seq : (a + d) * (a + d) = b * last)
  : (a - d, a, a + d, last) = (12, 16, 20, 25) := 
  sorry

end find_four_numbers_l606_60605


namespace find_sin_2alpha_l606_60645

theorem find_sin_2alpha (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 4) Real.pi) 
  (h2 : 3 * Real.cos (2 * α) = 4 * Real.sin (Real.pi / 4 - α)) : 
  Real.sin (2 * α) = -1 / 9 :=
sorry

end find_sin_2alpha_l606_60645


namespace mul_mental_math_l606_60678

theorem mul_mental_math :
  96 * 104 = 9984 := by
  sorry

end mul_mental_math_l606_60678


namespace volume_of_rectangular_prism_l606_60623

theorem volume_of_rectangular_prism :
  ∃ (a b c : ℝ), (a * b = 54) ∧ (b * c = 56) ∧ (a * c = 60) ∧ (a * b * c = 379) :=
by sorry

end volume_of_rectangular_prism_l606_60623


namespace angle_C_max_l606_60661

theorem angle_C_max (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_cond : Real.sin B / Real.sin A = 2 * Real.cos (A + B))
  (h_max_B : B = Real.pi / 3) :
  C = 2 * Real.pi / 3 :=
by
  sorry

end angle_C_max_l606_60661


namespace combined_degrees_l606_60655

theorem combined_degrees (S J W : ℕ) (h1 : S = 150) (h2 : J = S - 5) (h3 : W = S - 3) : S + J + W = 442 :=
by
  sorry

end combined_degrees_l606_60655


namespace determine_z_l606_60627

theorem determine_z (z : ℝ) (h1 : ∃ x : ℤ, 3 * (x : ℝ) ^ 2 + 19 * (x : ℝ) - 84 = 0 ∧ (x : ℝ) = ⌊z⌋) (h2 : 4 * (z - ⌊z⌋) ^ 2 - 14 * (z - ⌊z⌋) + 6 = 0) : 
  z = -11 :=
  sorry

end determine_z_l606_60627


namespace trains_clear_time_l606_60628

noncomputable def time_to_clear (length_train1 length_train2 speed_train1 speed_train2 : ℕ) : ℝ :=
  (length_train1 + length_train2) / ((speed_train1 + speed_train2) * 1000 / 3600)

theorem trains_clear_time :
  time_to_clear 121 153 80 65 = 6.803 :=
by
  -- This is a placeholder for the proof
  sorry

end trains_clear_time_l606_60628


namespace top_weight_l606_60662

theorem top_weight (T : ℝ) : 
    (9 * 0.8 + 7 * T = 10.98) → T = 0.54 :=
by 
  intro h
  have H_sum := h
  simp only [mul_add, add_assoc, mul_assoc, mul_comm, add_comm, mul_comm 7] at H_sum
  sorry

end top_weight_l606_60662


namespace sophia_estimate_larger_l606_60652

theorem sophia_estimate_larger (x y a b : ℝ) (hx : x > y) (hy : y > 0) (ha : a > 0) (hb : b > 0) :
  (x + a) - (y - b) > x - y := by
  sorry

end sophia_estimate_larger_l606_60652


namespace plane_equation_through_point_parallel_l606_60613

theorem plane_equation_through_point_parallel (A B C D : ℤ) (hx hy hz : ℤ) (x y z : ℤ)
  (h_point : (A, B, C, D) = (-2, 1, -3, 10))
  (h_coordinates : (hx, hy, hz) = (2, -3, 1))
  (h_plane_parallel : ∀ x y z, -2 * x + y - 3 * z = 7 ↔ A * x + B * y + C * z + D = 0)
  (h_form : A > 0):
  ∃ A' B' C' D', A' * (x : ℤ) + B' * (y : ℤ) + C' * (z : ℤ) + D' = 0 :=
by
  sorry

end plane_equation_through_point_parallel_l606_60613
