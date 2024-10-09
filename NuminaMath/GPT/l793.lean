import Mathlib

namespace olivia_earnings_this_week_l793_79383

variable (hourly_rate : ℕ) (hours_monday hours_wednesday hours_friday : ℕ)

theorem olivia_earnings_this_week : 
  hourly_rate = 9 → 
  hours_monday = 4 → 
  hours_wednesday = 3 → 
  hours_friday = 6 → 
  (hourly_rate * hours_monday + hourly_rate * hours_wednesday + hourly_rate * hours_friday) = 117 := 
by
  intros
  sorry

end olivia_earnings_this_week_l793_79383


namespace gcd_1248_585_l793_79343

theorem gcd_1248_585 : Nat.gcd 1248 585 = 39 := by
  sorry

end gcd_1248_585_l793_79343


namespace jack_leftover_money_l793_79390

theorem jack_leftover_money :
  let saved_money_base8 : ℕ := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 : ℕ := 1200
  saved_money_base8 - ticket_cost_base10 = 847 :=
by
  let saved_money_base8 := 3 * 8^3 + 7 * 8^2 + 7 * 8^1 + 7 * 8^0
  let ticket_cost_base10 := 1200
  show saved_money_base8 - ticket_cost_base10 = 847
  sorry

end jack_leftover_money_l793_79390


namespace perpendicular_lines_l793_79377

-- Definitions of conditions
def condition1 (α β γ δ : ℝ) : Prop := α = 90 ∧ α + β = 180 ∧ α + γ = 180 ∧ α + δ = 180
def condition2 (α β γ δ : ℝ) : Prop := α = β ∧ β = γ ∧ γ = δ
def condition3 (α β : ℝ) : Prop := α = β ∧ α + β = 180
def condition4 (α β : ℝ) : Prop := α = β ∧ α + β = 180

-- Main theorem statement
theorem perpendicular_lines (α β γ δ : ℝ) :
  (condition1 α β γ δ ∨ condition2 α β γ δ ∨
   condition3 α β ∨ condition4 α β) → α = 90 :=
by sorry

end perpendicular_lines_l793_79377


namespace chris_newspapers_l793_79338

theorem chris_newspapers (C L : ℕ) 
  (h1 : L = C + 23) 
  (h2 : C + L = 65) : 
  C = 21 := 
by 
  sorry

end chris_newspapers_l793_79338


namespace problem_A_value_l793_79349

theorem problem_A_value (x y A : ℝ) (h : (x + 2 * y) ^ 2 = (x - 2 * y) ^ 2 + A) : A = 8 * x * y :=
by {
    sorry
}

end problem_A_value_l793_79349


namespace cylinder_volume_eq_pi_over_4_l793_79325

theorem cylinder_volume_eq_pi_over_4
  (r : ℝ)
  (h₀ : r > 0)
  (h₁ : 2 * r = r * 2)
  (h₂ : 4 * π * r^2 = π) : 
  (π * r^2 * (2 * r) = π / 4) :=
by
  sorry

end cylinder_volume_eq_pi_over_4_l793_79325


namespace meaningful_fraction_condition_l793_79387

theorem meaningful_fraction_condition (x : ℝ) : (4 - 2 * x ≠ 0) ↔ (x ≠ 2) :=
by {
  sorry
}

end meaningful_fraction_condition_l793_79387


namespace areas_of_shared_parts_l793_79309

-- Define the areas of the non-overlapping parts
def area_non_overlap_1 : ℝ := 68
def area_non_overlap_2 : ℝ := 110
def area_non_overlap_3 : ℝ := 87

-- Define the total area of each circle
def total_area : ℝ := area_non_overlap_2 + area_non_overlap_3 - area_non_overlap_1

-- Define the areas of the shared parts A and B
def area_shared_A : ℝ := total_area - area_non_overlap_2
def area_shared_B : ℝ := total_area - area_non_overlap_3

-- Prove the areas of the shared parts
theorem areas_of_shared_parts :
  area_shared_A = 19 ∧ area_shared_B = 42 :=
by
  sorry

end areas_of_shared_parts_l793_79309


namespace inequality_negatives_l793_79370

theorem inequality_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (b / a) < 1 :=
by
  sorry

end inequality_negatives_l793_79370


namespace general_term_arithmetic_sequence_l793_79351

theorem general_term_arithmetic_sequence {a : ℕ → ℕ} (d : ℕ) (h_d : d ≠ 0)
  (h1 : a 3 + a 10 = 15)
  (h2 : (a 2 + d) * (a 2 + 10 * d) = (a 2 + 4 * d) * (a 2 + d))
  : ∀ n, a n = n + 1 :=
sorry

end general_term_arithmetic_sequence_l793_79351


namespace surveyed_households_count_l793_79379

theorem surveyed_households_count 
  (neither : ℕ) (only_R : ℕ) (both_B : ℕ) (both : ℕ) (h_main : Ξ)
  (H1 : neither = 80)
  (H2 : only_R = 60)
  (H3 : both = 40)
  (H4 : both_B = 3 * both) : 
  neither + only_R + both_B + both = 300 :=
by
  sorry

end surveyed_households_count_l793_79379


namespace incorrect_mode_l793_79368

theorem incorrect_mode (data : List ℕ) (hdata : data = [1, 2, 4, 3, 5]) : ¬ (∃ mode, mode = 5 ∧ (data.count mode > 1)) :=
by
  sorry

end incorrect_mode_l793_79368


namespace complement_of_A_in_U_l793_79334

def U : Set ℕ := {1,3,5,7,9}
def A : Set ℕ := {1,9}
def complement_U_A : Set ℕ := {3,5,7}

theorem complement_of_A_in_U : (U \ A) = complement_U_A := by
  sorry

end complement_of_A_in_U_l793_79334


namespace fraction_of_female_attendees_on_time_l793_79316

theorem fraction_of_female_attendees_on_time (A : ℝ)
  (h1 : 3 / 5 * A = M)
  (h2 : 7 / 8 * M = M_on_time)
  (h3 : 0.115 * A = n_A_not_on_time) :
  0.9 * F = (A - M_on_time - n_A_not_on_time)/((2 / 5) * A - n_A_not_on_time) :=
by
  sorry

end fraction_of_female_attendees_on_time_l793_79316


namespace inequality_proof_l793_79312

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_mul : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 :=
sorry

end inequality_proof_l793_79312


namespace necessary_but_not_sufficient_l793_79306

variable (p q : Prop)

theorem necessary_but_not_sufficient (hp : p) : p ∧ q ↔ p ∧ (p ∧ q → q) :=
  sorry

end necessary_but_not_sufficient_l793_79306


namespace speed_of_current_l793_79371
  
  theorem speed_of_current (v c : ℝ)
    (h1 : 64 = (v + c) * 8)
    (h2 : 24 = (v - c) * 8) :
    c = 2.5 :=
  by {
    sorry
  }
  
end speed_of_current_l793_79371


namespace sequence_a_2016_value_l793_79342

theorem sequence_a_2016_value (a : ℕ → ℕ) 
  (h1 : a 4 = 1)
  (h2 : a 11 = 9)
  (h3 : ∀ n : ℕ, a n + a (n+1) + a (n+2) = 15) :
  a 2016 = 5 :=
sorry

end sequence_a_2016_value_l793_79342


namespace find_a_l793_79350

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l793_79350


namespace find_number_l793_79323

def is_three_digit_number (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (0 ≤ y ∧ y ≤ 9) ∧ (0 ≤ z ∧ z ≤ 9) ∧
  n = 100 * x + 10 * y + z ∧ (100 * x + 10 * y + z) / 11 = x^2 + y^2 + z^2

theorem find_number : ∃ n : ℕ, is_three_digit_number n ∧ n = 550 :=
sorry

end find_number_l793_79323


namespace initial_deposit_l793_79378

/-- 
A person deposits some money in a bank at an interest rate of 7% per annum (of the original amount). 
After two years, the total amount in the bank is $6384. Prove that the initial amount deposited is $5600.
-/
theorem initial_deposit (P : ℝ) (h : (P + 0.07 * P) + 0.07 * P = 6384) : P = 5600 :=
by
  sorry

end initial_deposit_l793_79378


namespace find_x_l793_79382

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 80) : x = 26 :=
sorry

end find_x_l793_79382


namespace no_nat_solutions_no_int_solutions_l793_79365

theorem no_nat_solutions (x y : ℕ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

theorem no_int_solutions (x y : ℤ) : x^3 + 5 * y = y^3 + 5 * x → x = y :=
by sorry

end no_nat_solutions_no_int_solutions_l793_79365


namespace find_p_q_l793_79358

noncomputable def cubicFunction (p q : ℝ) (x : ℂ) : ℂ :=
  2 * x^3 + p * x^2 + q * x

theorem find_p_q (p q : ℝ) :
  cubicFunction p q (2 * Complex.I - 3) = 0 ∧ 
  cubicFunction p q (-2 * Complex.I - 3) = 0 → 
  p = 12 ∧ q = 26 :=
by
  sorry

end find_p_q_l793_79358


namespace simplest_fraction_is_one_l793_79391

theorem simplest_fraction_is_one :
  ∃ m : ℕ, 
  (∃ k : ℕ, 45 * m = k^2) ∧ 
  (∃ n : ℕ, 56 * m = n^3) → 
  45 * m / 56 * m = 1 := by
  sorry

end simplest_fraction_is_one_l793_79391


namespace tan_beta_value_l793_79310

theorem tan_beta_value (α β : ℝ) (h1 : Real.tan α = 2) (h2 : Real.tan (α + β) = -1) : Real.tan β = 3 :=
by
  sorry

end tan_beta_value_l793_79310


namespace find_last_number_l793_79317

theorem find_last_number
  (A B C D : ℝ)
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11) :
  D = 4 :=
by
  sorry

end find_last_number_l793_79317


namespace find_salary_of_Thomas_l793_79359

-- Declare the variables representing the salaries of Raj, Roshan, and Thomas
variables (R S T : ℝ)

-- Given conditions as definitions
def avg_salary_Raj_Roshan : Prop := (R + S) / 2 = 4000
def avg_salary_Raj_Roshan_Thomas : Prop := (R + S + T) / 3 = 5000

-- Stating the theorem
theorem find_salary_of_Thomas
  (h1 : avg_salary_Raj_Roshan R S)
  (h2 : avg_salary_Raj_Roshan_Thomas R S T) : T = 7000 :=
by
  sorry

end find_salary_of_Thomas_l793_79359


namespace sum_first_10_terms_abs_a_n_l793_79386

noncomputable def a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else 3 * n - 7

def abs_a_n (n : ℕ) : ℤ :=
  if n = 1 ∨ n = 2 then -3 * n + 7 else 3 * n - 7

def sum_abs_a_n (n : ℕ) : ℤ :=
  if n = 0 then 0 else List.sum (List.map abs_a_n (List.range n))

theorem sum_first_10_terms_abs_a_n : sum_abs_a_n 10 = 105 := 
  sorry

end sum_first_10_terms_abs_a_n_l793_79386


namespace geometric_sequence_common_ratio_l793_79369

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_sum_ratio : (a 0 + a 1 + a 2) / a 2 = 7) :
  q = 1 / 2 :=
sorry

end geometric_sequence_common_ratio_l793_79369


namespace find_other_number_l793_79376

theorem find_other_number (hcf lcm a b: ℕ) (hcf_value: hcf = 12) (lcm_value: lcm = 396) (a_value: a = 36) (gcd_ab: Nat.gcd a b = hcf) (lcm_ab: Nat.lcm a b = lcm) : b = 132 :=
by
  sorry

end find_other_number_l793_79376


namespace intersection_A_B_l793_79308

-- Conditions
def A : Set (ℕ × ℕ) := { (1, 2), (2, 1) }
def B : Set (ℕ × ℕ) := { p | p.fst - p.snd = 1 }

-- Problem statement
theorem intersection_A_B : A ∩ B = { (2, 1) } :=
by
  sorry

end intersection_A_B_l793_79308


namespace count_integers_congruent_to_7_mod_13_l793_79394

theorem count_integers_congruent_to_7_mod_13:
  (∃ S : Finset ℕ, S.card = 154 ∧ ∀ n ∈ S, n < 2000 ∧ n % 13 = 7) :=
by
  sorry

end count_integers_congruent_to_7_mod_13_l793_79394


namespace first_term_geometric_progression_l793_79347

theorem first_term_geometric_progression (a r : ℝ) 
  (h1 : a / (1 - r) = 6)
  (h2 : a + a * r = 9 / 2) :
  a = 3 ∨ a = 9 := 
sorry -- Proof omitted

end first_term_geometric_progression_l793_79347


namespace newspaper_pages_l793_79372

theorem newspaper_pages (p : ℕ) (h₁ : p >= 21) (h₂ : 8•2 - 1 ≤ p) (h₃ : p ≤ 8•3) : p = 28 :=
sorry

end newspaper_pages_l793_79372


namespace min_expr_value_l793_79305

theorem min_expr_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) :
  (∃ a, a = (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ∧ a ≥ 0) → 
  (∀ (u v : ℝ), u = x + 2 → v = 3 * y + 4 → u * v = 16) →
  (4 / (x + 2) + (3 * x - 7) / (3 * y + 4)) ≥ 11 / 16 :=
sorry

end min_expr_value_l793_79305


namespace required_average_for_tickets_l793_79395

theorem required_average_for_tickets 
  (june_score : ℝ) (patty_score : ℝ) (josh_score : ℝ) (henry_score : ℝ)
  (num_children : ℝ) (total_score : ℝ) (average_score : ℝ) (S : ℝ)
  (h1 : june_score = 97) (h2 : patty_score = 85) (h3 : josh_score = 100) 
  (h4 : henry_score = 94) (h5 : num_children = 4) 
  (h6 : total_score = june_score + patty_score + josh_score + henry_score)
  (h7 : average_score = total_score / num_children) 
  (h8 : average_score = 94)
  : S ≤ 94 :=
sorry

end required_average_for_tickets_l793_79395


namespace rate_of_current_l793_79313

def downstream_eq (b c : ℝ) : Prop := (b + c) * 4 = 24
def upstream_eq (b c : ℝ) : Prop := (b - c) * 6 = 24

theorem rate_of_current (b c : ℝ) (h1 : downstream_eq b c) (h2 : upstream_eq b c) : c = 1 :=
by sorry

end rate_of_current_l793_79313


namespace max_f_value_l793_79389

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x / (x^2 + m)

theorem max_f_value (m : ℝ) : 
  (m > 1) ↔ (∀ x : ℝ, f x m < 1) ∧ ¬((∀ x : ℝ, f x m < 1) → (m > 1)) :=
by
  sorry

end max_f_value_l793_79389


namespace calculate_expression_l793_79318

theorem calculate_expression :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5 / 4 :=
by
  sorry

end calculate_expression_l793_79318


namespace best_fit_of_regression_model_l793_79388

-- Define the context of regression analysis and the coefficient of determination
def regression_analysis : Type := sorry
def coefficient_of_determination (r : regression_analysis) : ℝ := sorry

-- Definitions of each option for clarity in our context
def A (r : regression_analysis) : Prop := sorry -- the linear relationship is stronger
def B (r : regression_analysis) : Prop := sorry -- the linear relationship is weaker
def C (r : regression_analysis) : Prop := sorry -- better fit of the model
def D (r : regression_analysis) : Prop := sorry -- worse fit of the model

-- The formal statement we need to prove
theorem best_fit_of_regression_model (r : regression_analysis) (R2 : ℝ) (h1 : coefficient_of_determination r = R2) (h2 : R2 = 1) : C r :=
by
  sorry

end best_fit_of_regression_model_l793_79388


namespace prove_nabla_squared_l793_79337

theorem prove_nabla_squared:
  ∃ (odot nabla : ℕ), odot < 20 ∧ nabla < 20 ∧ odot ≠ nabla ∧
  (nabla * nabla * odot = nabla) ∧ (nabla * nabla = 64) :=
by
  sorry

end prove_nabla_squared_l793_79337


namespace fraction_uncovered_l793_79314

def area_rug (length width : ℕ) : ℕ := length * width
def area_square (side : ℕ) : ℕ := side * side

theorem fraction_uncovered 
  (rug_length rug_width floor_area : ℕ)
  (h_rug_length : rug_length = 2)
  (h_rug_width : rug_width = 7)
  (h_floor_area : floor_area = 64)
  : (floor_area - area_rug rug_length rug_width) / floor_area = 25 / 32 := 
sorry

end fraction_uncovered_l793_79314


namespace minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l793_79399

-- Definition for the minimum number of uninteresting vertices
def minimum_uninteresting_vertices (n : ℕ) (h : n > 3) : ℕ := 2

-- Theorem for the minimum number of uninteresting vertices
theorem minimum_uninteresting_vertices_correct (n : ℕ) (h : n > 3) :
  minimum_uninteresting_vertices n h = 2 := 
sorry

-- Definition for the maximum number of unusual vertices
def maximum_unusual_vertices (n : ℕ) (h : n > 3) : ℕ := 3

-- Theorem for the maximum number of unusual vertices
theorem maximum_unusual_vertices_correct (n : ℕ) (h : n > 3) :
  maximum_unusual_vertices n h = 3 :=
sorry

end minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l793_79399


namespace abc_value_l793_79380

variables (a b c : ℂ)

theorem abc_value :
  (a * b + 4 * b = -16) →
  (b * c + 4 * c = -16) →
  (c * a + 4 * a = -16) →
  a * b * c = 64 :=
by
  intros h1 h2 h3
  sorry

end abc_value_l793_79380


namespace sufficient_condition_for_q_l793_79366

def p (a : ℝ) : Prop := a ≥ 0
def q (a : ℝ) : Prop := a^2 + a ≥ 0

theorem sufficient_condition_for_q (a : ℝ) : p a → q a := by 
  sorry

end sufficient_condition_for_q_l793_79366


namespace blue_line_length_l793_79330

theorem blue_line_length (w b : ℝ) (h1 : w = 7.666666666666667) (h2 : w = b + 4.333333333333333) :
  b = 3.333333333333334 :=
by sorry

end blue_line_length_l793_79330


namespace geometric_sequence_iff_q_neg_one_l793_79374

theorem geometric_sequence_iff_q_neg_one {p q : ℝ} (h1 : p ≠ 0) (h2 : p ≠ 1)
  (S : ℕ → ℝ) (hS : ∀ n, S n = p^n + q) :
  (∃ (a : ℕ → ℝ), (∀ n, a (n+1) = (p - 1) * p^n) ∧ (∀ n, a (n+1) = S (n+1) - S n) ∧
                    (∀ n, a (n+1) / a n = p)) ↔ q = -1 :=
sorry

end geometric_sequence_iff_q_neg_one_l793_79374


namespace interest_rate_second_type_l793_79315

variable (totalInvestment : ℝ) (interestFirstTypeRate : ℝ) (investmentSecondType : ℝ) (totalInterestRate : ℝ) 
variable [Nontrivial ℝ]

theorem interest_rate_second_type :
    totalInvestment = 100000 ∧
    interestFirstTypeRate = 0.09 ∧
    investmentSecondType = 29999.999999999993 ∧
    totalInterestRate = 9 + 3 / 5 →
    (9.6 * totalInvestment - (interestFirstTypeRate * (totalInvestment - investmentSecondType))) / investmentSecondType = 0.11 :=
by
  sorry

end interest_rate_second_type_l793_79315


namespace derivative_at_one_eq_neg_one_l793_79354

variable {α : Type*} [TopologicalSpace α] {f : ℝ → ℝ}
-- condition: f is differentiable
variable (hf_diff : Differentiable ℝ f)
-- condition: limit condition
variable (h_limit : Tendsto (fun Δx => (f (1 + 2 * Δx) - f 1) / Δx) (𝓝 0) (𝓝 (-2)))

-- proof goal: f'(1) = -1
theorem derivative_at_one_eq_neg_one : deriv f 1 = -1 := 
by
  sorry

end derivative_at_one_eq_neg_one_l793_79354


namespace smallest_k_power_l793_79345

theorem smallest_k_power (k : ℕ) (hk : ∀ m : ℕ, m < 14 → 7^m ≤ 4^19) : 7^14 > 4^19 :=
sorry

end smallest_k_power_l793_79345


namespace tangent_line_equation_l793_79328

theorem tangent_line_equation (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ∃ (m b : ℝ), y = m * x + b ∧ y = 4 * x - 2 :=
by
  sorry

end tangent_line_equation_l793_79328


namespace nickel_ate_2_chocolates_l793_79398

def nickels_chocolates (r n : Nat) : Prop :=
r = n + 7

theorem nickel_ate_2_chocolates (r : Nat) (h : r = 9) (h1 : nickels_chocolates r 2) : 2 = 2 :=
by
  sorry

end nickel_ate_2_chocolates_l793_79398


namespace S2014_value_l793_79344

variable (S : ℕ → ℤ) -- S_n represents sum of the first n terms of the arithmetic sequence
variable (a1 : ℤ) -- First term of the arithmetic sequence
variable (d : ℤ) -- Common difference of the arithmetic sequence

-- Given conditions
variable (h1 : a1 = -2016)
variable (h2 : (S 2016) / 2016 - (S 2010) / 2010 = 6)

-- The proof problem
theorem S2014_value :
  S 2014 = -6042 :=
sorry -- Proof omitted

end S2014_value_l793_79344


namespace find_b_neg_l793_79361

noncomputable def h (x : ℝ) : ℝ := if x ≤ 0 then -x else 3 * x - 50

theorem find_b_neg (b : ℝ) (h_neg_b : b < 0) : 
  h (h (h 15)) = h (h (h b)) → b = - (55 / 3) :=
by
  sorry

end find_b_neg_l793_79361


namespace three_pow_y_plus_two_l793_79304

theorem three_pow_y_plus_two (y : ℕ) (h : 3^y = 81) : 3^(y+2) = 729 := sorry

end three_pow_y_plus_two_l793_79304


namespace sum_of_integers_l793_79364

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 6) (h2 : x * y = 112) (h3 : x > y) : x + y = 22 :=
sorry

end sum_of_integers_l793_79364


namespace highest_score_is_174_l793_79346

theorem highest_score_is_174
  (avg_40_innings : ℝ)
  (highest_exceeds_lowest : ℝ)
  (avg_excl_two : ℝ)
  (total_runs_40 : ℝ)
  (total_runs_38 : ℝ)
  (sum_H_L : ℝ)
  (new_avg_38 : ℝ)
  (H : ℝ)
  (L : ℝ)
  (H_eq_L_plus_172 : H = L + 172)
  (total_runs_40_eq : total_runs_40 = 40 * avg_40_innings)
  (total_runs_38_eq : total_runs_38 = 38 * new_avg_38)
  (sum_H_L_eq : sum_H_L = total_runs_40 - total_runs_38)
  (new_avg_eq : new_avg_38 = avg_40_innings - 2)
  (sum_H_L_val : sum_H_L = 176)
  (avg_40_val : avg_40_innings = 50) :
  H = 174 :=
sorry

end highest_score_is_174_l793_79346


namespace candle_burning_time_l793_79335

theorem candle_burning_time :
  ∃ t : ℚ, (1 - t / 5) = 3 * (1 - t / 4) ∧ t = 40 / 11 :=
by {
  sorry
}

end candle_burning_time_l793_79335


namespace largest_whole_number_solution_for_inequality_l793_79385

theorem largest_whole_number_solution_for_inequality :
  ∀ (x : ℕ), ((1 : ℝ) / 4 + (x : ℝ) / 5 < 2) → x ≤ 23 :=
by sorry

end largest_whole_number_solution_for_inequality_l793_79385


namespace arrange_pencils_l793_79384

-- Definition to express the concept of pencil touching
def pencil_touches (a b : Type) : Prop := sorry

-- Assume we have six pencils represented as 6 distinct variables.
variables (A B C D E F : Type)

-- Main theorem statement
theorem arrange_pencils :
  ∃ (A B C D E F : Type), (pencil_touches A B) ∧ (pencil_touches A C) ∧ 
  (pencil_touches A D) ∧ (pencil_touches A E) ∧ (pencil_touches A F) ∧ 
  (pencil_touches B C) ∧ (pencil_touches B D) ∧ (pencil_touches B E) ∧ 
  (pencil_touches B F) ∧ (pencil_touches C D) ∧ (pencil_touches C E) ∧ 
  (pencil_touches C F) ∧ (pencil_touches D E) ∧ (pencil_touches D F) ∧ 
  (pencil_touches E F) :=
sorry

end arrange_pencils_l793_79384


namespace sufficient_condition_for_proposition_l793_79381

theorem sufficient_condition_for_proposition :
  ∀ (a : ℝ), (0 < a ∧ a < 4) → (∀ x : ℝ, a * x ^ 2 + a * x + 1 > 0) := 
sorry

end sufficient_condition_for_proposition_l793_79381


namespace common_difference_arithmetic_sequence_l793_79303

-- Define the arithmetic sequence properties
variable (S : ℕ → ℕ) -- S represents the sum of the first n terms
variable (a : ℕ → ℕ) -- a represents the terms in the arithmetic sequence
variable (d : ℤ) -- common difference

-- Define the conditions
axiom S2_eq_6 : S 2 = 6
axiom a1_eq_4 : a 1 = 4

-- The problem: show that d = -2
theorem common_difference_arithmetic_sequence :
  (a 2 - a 1 = d) → d = -2 :=
by
  sorry

end common_difference_arithmetic_sequence_l793_79303


namespace tiffany_lives_next_level_l793_79352

theorem tiffany_lives_next_level (L1 L2 L3 : ℝ)
    (h1 : L1 = 43.0)
    (h2 : L2 = 14.0)
    (h3 : L3 = 84.0) :
    L3 - (L1 + L2) = 27 :=
by
  rw [h1, h2, h3]
  -- The proof is skipped with "sorry"
  sorry

end tiffany_lives_next_level_l793_79352


namespace segment_outside_spheres_l793_79393

noncomputable def fraction_outside_spheres (α : ℝ) : ℝ :=
  (1 - (Real.cos (α / 2))^2) / (1 + (Real.cos (α / 2))^2)

theorem segment_outside_spheres (R α : ℝ) (hR : R > 0) (hα : 0 < α ∧ α < 2 * Real.pi) :
  fraction_outside_spheres α = (1 - Real.cos (α / 2)^2) / (1 + (Real.cos (α / 2))^2) :=
  by sorry

end segment_outside_spheres_l793_79393


namespace find_x_l793_79367

def f (x : ℝ) : ℝ := 2 * x - 3 -- Definition of the function f

def c : ℝ := 11 -- Definition of the constant c

theorem find_x : 
  ∃ x : ℝ, 2 * f x - c = f (x - 2) ↔ x = 5 :=
by 
  sorry

end find_x_l793_79367


namespace probability_of_perfect_square_sum_l793_79392

def two_dice_probability_of_perfect_square_sum : ℚ :=
  let totalOutcomes := 12 * 12
  let perfectSquareOutcomes := 3 + 8 + 9 -- ways to get sums 4, 9, and 16
  (perfectSquareOutcomes : ℚ) / (totalOutcomes : ℚ)

theorem probability_of_perfect_square_sum :
  two_dice_probability_of_perfect_square_sum = 5 / 36 :=
by
  sorry

end probability_of_perfect_square_sum_l793_79392


namespace tip_percentage_l793_79340

theorem tip_percentage
  (original_bill : ℝ)
  (shared_per_person : ℝ)
  (num_people : ℕ)
  (total_shared : ℝ)
  (tip_percent : ℝ)
  (h1 : original_bill = 139.0)
  (h2 : shared_per_person = 50.97)
  (h3 : num_people = 3)
  (h4 : total_shared = shared_per_person * num_people)
  (h5 : total_shared - original_bill = 13.91) :
  tip_percent = 13.91 / 139.0 * 100 := 
sorry

end tip_percentage_l793_79340


namespace range_of_a_l793_79397

theorem range_of_a 
    (a : ℝ) 
    (f : ℝ → ℝ) 
    (h : ∀ x : ℝ, f x = Real.exp (|x - a|)) 
    (increasing_on_interval : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f x ≤ f y) :
    a ≤ 1 :=
sorry

end range_of_a_l793_79397


namespace horse_saddle_ratio_l793_79356

theorem horse_saddle_ratio (total_cost : ℕ) (saddle_cost : ℕ) (horse_cost : ℕ) 
  (h_total : total_cost = 5000)
  (h_saddle : saddle_cost = 1000)
  (h_sum : horse_cost + saddle_cost = total_cost) : 
  horse_cost / saddle_cost = 4 :=
by sorry

end horse_saddle_ratio_l793_79356


namespace students_difference_l793_79319

theorem students_difference 
  (C : ℕ → ℕ) 
  (hC1 : C 1 = 24) 
  (hC2 : ∀ n, C n.succ = C n - d)
  (h_total : C 1 + C 2 + C 3 + C 4 + C 5 = 100) :
  d = 2 :=
by sorry

end students_difference_l793_79319


namespace goods_train_pass_time_l793_79360

theorem goods_train_pass_time 
  (speed_mans_train_kmph : ℝ) (speed_goods_train_kmph : ℝ) (length_goods_train_m : ℝ) :
  speed_mans_train_kmph = 20 → 
  speed_goods_train_kmph = 92 → 
  length_goods_train_m = 280 → 
  abs ((length_goods_train_m / ((speed_mans_train_kmph + speed_goods_train_kmph) * 1000 / 3600)) - 8.99) < 0.01 :=
by
  sorry

end goods_train_pass_time_l793_79360


namespace order_of_mnpq_l793_79339

theorem order_of_mnpq 
(m n p q : ℝ) 
(h1 : m < n)
(h2 : p < q)
(h3 : (p - m) * (p - n) < 0)
(h4 : (q - m) * (q - n) < 0) 
: m < p ∧ p < q ∧ q < n := 
by
  sorry

end order_of_mnpq_l793_79339


namespace crates_on_third_trip_l793_79333

variable (x : ℕ) -- Denote the number of crates carried on the third trip

-- Conditions
def crate_weight := 1250
def max_weight := 6250
def trip3_weight (x : ℕ) := x * crate_weight

-- The problem statement: Prove that x (the number of crates on the third trip) == 5
theorem crates_on_third_trip : trip3_weight x <= max_weight → x = 5 :=
by
  sorry -- No proof required, just statement

end crates_on_third_trip_l793_79333


namespace sum_of_coefficients_of_poly_l793_79357

-- Define the polynomial
def poly (x y : ℕ) := (2 * x + 3 * y) ^ 12

-- Define the sum of coefficients
def sum_of_coefficients := poly 1 1

-- The theorem stating the result
theorem sum_of_coefficients_of_poly : sum_of_coefficients = 244140625 :=
by
  -- Proof is skipped
  sorry

end sum_of_coefficients_of_poly_l793_79357


namespace solution_set_condition_l793_79307

theorem solution_set_condition {a : ℝ} : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ (x ≥ 3)) → a < 3 := 
by 
  intros h
  sorry

end solution_set_condition_l793_79307


namespace expression_range_l793_79301

theorem expression_range (a b c x : ℝ) (h : a^2 + b^2 + c^2 ≠ 0) :
  ∃ y : ℝ, y = (a * Real.cos x + b * Real.sin x + c) / (Real.sqrt (a^2 + b^2 + c^2)) 
           ∧ y ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end expression_range_l793_79301


namespace option_C_equals_a5_l793_79320

theorem option_C_equals_a5 (a : ℕ) : (a^4 * a = a^5) :=
by sorry

end option_C_equals_a5_l793_79320


namespace compare_08_and_one_eighth_l793_79322

theorem compare_08_and_one_eighth :
  0.8 - (1 / 8 : ℝ) = 0.675 := 
sorry

end compare_08_and_one_eighth_l793_79322


namespace max_students_before_new_year_l793_79329

theorem max_students_before_new_year (N M k l : ℕ) (h1 : 100 * M = k * N) (h2 : 100 * (M + 1) = l * (N + 3)) (h3 : 3 * l < 300) :
      N ≤ 197 := by
  sorry

end max_students_before_new_year_l793_79329


namespace new_rectangle_area_l793_79321

theorem new_rectangle_area (L W : ℝ) (h : L * W = 300) :
  let L_new := 2 * L
  let W_new := 3 * W
  L_new * W_new = 1800 :=
by
  let L_new := 2 * L
  let W_new := 3 * W
  sorry

end new_rectangle_area_l793_79321


namespace find_range_of_a_l793_79336

-- Definitions
def is_decreasing_function (a : ℝ) : Prop :=
  0 < a ∧ a < 1

def no_real_roots_of_poly (a : ℝ) : Prop :=
  4 * a < 1

def problem_statement (a : ℝ) : Prop :=
  (is_decreasing_function a ∨ no_real_roots_of_poly a) ∧ ¬ (is_decreasing_function a ∧ no_real_roots_of_poly a)

-- Main theorem
theorem find_range_of_a (a : ℝ) : problem_statement a ↔ (0 < a ∧ a ≤ 1 / 4) ∨ (a ≥ 1) :=
by
  -- Proof omitted
  sorry

end find_range_of_a_l793_79336


namespace additional_tanks_needed_l793_79362

theorem additional_tanks_needed 
    (initial_tanks : ℕ) 
    (initial_capacity_per_tank : ℕ) 
    (total_fish_needed : ℕ) 
    (new_capacity_per_tank : ℕ)
    (h_t1 : initial_tanks = 3)
    (h_t2 : initial_capacity_per_tank = 15)
    (h_t3 : total_fish_needed = 75)
    (h_t4 : new_capacity_per_tank = 10) : 
    (total_fish_needed - initial_tanks * initial_capacity_per_tank) / new_capacity_per_tank = 3 := 
by {
    sorry
}

end additional_tanks_needed_l793_79362


namespace students_in_class_l793_79373

theorem students_in_class {S : ℕ} 
  (h1 : 20 < S)
  (h2 : S < 30)
  (chess_club_condition : ∃ (n : ℕ), S = 3 * n) 
  (draughts_club_condition : ∃ (m : ℕ), S = 4 * m) : 
  S = 24 := 
sorry

end students_in_class_l793_79373


namespace original_price_of_racket_l793_79353

theorem original_price_of_racket (P : ℝ) (h : (3 / 2) * P = 90) : P = 60 :=
sorry

end original_price_of_racket_l793_79353


namespace tenth_term_arithmetic_sequence_l793_79363

def a : ℚ := 2 / 3
def d : ℚ := 2 / 3

theorem tenth_term_arithmetic_sequence : 
  let a := 2 / 3
  let d := 2 / 3
  let n := 10
  a + (n - 1) * d = 20 / 3 := by
  sorry

end tenth_term_arithmetic_sequence_l793_79363


namespace quotient_when_divided_by_8_l793_79311

theorem quotient_when_divided_by_8
  (n : ℕ)
  (h1 : n = 12 * 7 + 5)
  : (n / 8) = 11 :=
by
  -- the proof is omitted
  sorry

end quotient_when_divided_by_8_l793_79311


namespace friend_saves_per_week_l793_79396

theorem friend_saves_per_week (x : ℕ) : 
  160 + 7 * 25 = 210 + x * 25 → x = 5 := 
by 
  sorry

end friend_saves_per_week_l793_79396


namespace jessica_purchase_cost_l793_79326

noncomputable def c_toy : Real := 10.22
noncomputable def c_cage : Real := 11.73
noncomputable def c_total : Real := c_toy + c_cage

theorem jessica_purchase_cost : c_total = 21.95 :=
by
  sorry

end jessica_purchase_cost_l793_79326


namespace obtuse_triangle_side_range_l793_79348

theorem obtuse_triangle_side_range
  (a : ℝ)
  (h1 : a > 0)
  (h2 : (a + 4)^2 > a^2 + (a + 2)^2)
  (h3 : (a + 2)^2 + (a + 4)^2 < a^2) : 
  2 < a ∧ a < 6 := 
sorry

end obtuse_triangle_side_range_l793_79348


namespace sum_of_prime_h_l793_79332

def h (n : ℕ) := n^4 - 380 * n^2 + 600

theorem sum_of_prime_h (S : Finset ℕ) (hS : S = { n | Nat.Prime (h n) }) :
  S.sum h = 0 :=
by
  sorry

end sum_of_prime_h_l793_79332


namespace probability_of_three_5s_in_eight_rolls_l793_79324

-- Conditions
def total_outcomes : ℕ := 6 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3

-- The probability that the number 5 appears exactly three times in eight rolls of a fair die
theorem probability_of_three_5s_in_eight_rolls :
  (favorable_outcomes / total_outcomes : ℚ) = (56 / 1679616 : ℚ) :=
by
  sorry

end probability_of_three_5s_in_eight_rolls_l793_79324


namespace largest_n_divisibility_l793_79355

theorem largest_n_divisibility :
  ∃ n : ℕ, (n^3 + 100) % (n + 10) = 0 ∧
  (∀ m : ℕ, (m^3 + 100) % (m + 10) = 0 → m ≤ n) ∧ n = 890 :=
by
  sorry

end largest_n_divisibility_l793_79355


namespace unique_7_tuple_count_l793_79300

theorem unique_7_tuple_count :
  ∃! (x : ℕ → ℝ) (zero_le_x : (∀ i, 0 ≤ i → i ≤ 6 → true)),
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 8 :=
by
  sorry

end unique_7_tuple_count_l793_79300


namespace mod_x_squared_l793_79327

theorem mod_x_squared :
  (∃ x : ℤ, 5 * x ≡ 9 [ZMOD 26] ∧ 4 * x ≡ 15 [ZMOD 26]) →
  ∃ y : ℤ, y ≡ 10 [ZMOD 26] :=
by
  intro h
  rcases h with ⟨x, h₁, h₂⟩
  exists x^2
  sorry

end mod_x_squared_l793_79327


namespace chip_price_reduction_equation_l793_79341

-- Define initial price
def initial_price : ℝ := 400

-- Define final price after reductions
def final_price : ℝ := 144

-- Define the price reduction percentage
variable (x : ℝ)

-- The equation we need to prove
theorem chip_price_reduction_equation :
  initial_price * (1 - x) ^ 2 = final_price :=
sorry

end chip_price_reduction_equation_l793_79341


namespace sum_of_numbers_l793_79331

theorem sum_of_numbers (a b c : ℝ) (h_ratio : a / 1 = b / 2 ∧ b / 2 = c / 3) (h_sum_squares : a^2 + b^2 + c^2 = 2744) : 
  a + b + c = 84 := 
sorry

end sum_of_numbers_l793_79331


namespace miley_total_cost_l793_79302

-- Define the cost per cellphone
def cost_per_cellphone : ℝ := 800

-- Define the number of cellphones
def number_of_cellphones : ℝ := 2

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the total cost without discount
def total_cost_without_discount : ℝ := cost_per_cellphone * number_of_cellphones

-- Define the discount amount
def discount_amount : ℝ := total_cost_without_discount * discount_rate

-- Define the total cost with discount
def total_cost_with_discount : ℝ := total_cost_without_discount - discount_amount

-- Prove that the total amount Miley paid is $1520
theorem miley_total_cost : total_cost_with_discount = 1520 := by
  sorry

end miley_total_cost_l793_79302


namespace piecewise_function_continuity_l793_79375

theorem piecewise_function_continuity :
  (∀ x, if x > (3 : ℝ) 
        then 2 * (a : ℝ) * x + 4 = (x : ℝ) ^ 2 - 1
        else if x < -1 
        then 3 * (x : ℝ) - (c : ℝ) = (x : ℝ) ^ 2 - 1
        else (x : ℝ) ^ 2 - 1 = (x : ℝ) ^ 2 - 1) →
  a = 2 / 3 →
  c = -3 →
  a + c = -7 / 3 :=
by
  intros h ha hc
  simp [ha, hc]
  sorry

end piecewise_function_continuity_l793_79375
