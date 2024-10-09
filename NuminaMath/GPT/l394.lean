import Mathlib

namespace price_returns_to_initial_l394_39445

theorem price_returns_to_initial {P₀ P₁ P₂ P₃ P₄ : ℝ} (y : ℝ) (h₁ : P₀ = 100)
  (h₂ : P₁ = P₀ * 1.30) (h₃ : P₂ = P₁ * 0.70) (h₄ : P₃ = P₂ * 1.40) 
  (h₅ : P₄ = P₃ * (1 - y / 100)) : P₄ = P₀ → y = 22 :=
by
  sorry

end price_returns_to_initial_l394_39445


namespace each_person_eats_3_Smores_l394_39490

-- Definitions based on the conditions in (a)
def people := 8
def cost_per_4_Smores := 3
def total_cost := 18

-- The statement we need to prove
theorem each_person_eats_3_Smores (h1 : total_cost = people * (cost_per_4_Smores * 4 / 3)) :
  (total_cost / cost_per_4_Smores) * 4 / people = 3 :=
by
  sorry

end each_person_eats_3_Smores_l394_39490


namespace Darla_electricity_bill_l394_39464

theorem Darla_electricity_bill :
  let tier1_rate := 4
  let tier2_rate := 3.5
  let tier3_rate := 3
  let tier1_limit := 300
  let tier2_limit := 500
  let late_fee1 := 150
  let late_fee2 := 200
  let late_fee3 := 250
  let consumption := 1200
  let cost_tier1 := tier1_limit * tier1_rate
  let cost_ttier2 := tier2_limit * tier2_rate
  let cost_tier3 := (consumption - (tier1_limit + tier2_limit)) * tier3_rate
  let total_cost := cost_tier1 + cost_tier2 + cost_tier3
  let late_fee := late_fee3
  let final_cost := total_cost + late_fee
  final_cost = 4400 :=
by
  sorry

end Darla_electricity_bill_l394_39464


namespace already_installed_windows_l394_39478

-- Definitions based on given conditions
def total_windows : ℕ := 9
def hours_per_window : ℕ := 6
def remaining_hours : ℕ := 18

-- Main statement to prove
theorem already_installed_windows : (total_windows - remaining_hours / hours_per_window) = 6 :=
by
  -- To prove: total_windows - (remaining_hours / hours_per_window) = 6
  -- This step is intentionally left incomplete (proof to be filled in by the user)
  sorry

end already_installed_windows_l394_39478


namespace range_of_f_l394_39485

noncomputable def f (x : ℝ) : ℝ := 1 / x - 4 / Real.sqrt x + 3

theorem range_of_f : ∀ y, (∃ x, (1/16 : ℝ) ≤ x ∧ x ≤ 1 ∧ f x = y) ↔ -1 ≤ y ∧ y ≤ 3 := by
  sorry

end range_of_f_l394_39485


namespace area_between_curves_l394_39480

-- Function definitions:
def quartic (a b c d e x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e
def line (p q x : ℝ) : ℝ := p * x + q

-- Conditions:
variables (a b c d e p q α β : ℝ)
variable (a_ne_zero : a ≠ 0)
variable (α_lt_β : α < β)
variable (touch_at_α : quartic a b c d e α = line p q α ∧ deriv (quartic a b c d e) α = p)
variable (touch_at_β : quartic a b c d e β = line p q β ∧ deriv (quartic a b c d e) β = p)

-- Theorem:
theorem area_between_curves :
  ∫ x in α..β, |quartic a b c d e x - line p q x| = (a * (β - α)^5) / 30 :=
by sorry

end area_between_curves_l394_39480


namespace count_special_four_digit_integers_is_100_l394_39432

def count_special_four_digit_integers : Nat := sorry

theorem count_special_four_digit_integers_is_100 :
  count_special_four_digit_integers = 100 :=
sorry

end count_special_four_digit_integers_is_100_l394_39432


namespace ratio_of_areas_l394_39467

noncomputable def area (A B C D : ℝ) : ℝ := 0  -- Placeholder, exact area definition will require geometrical formalism.

variables (A B C D P Q R S : ℝ)

-- Define the conditions
variables (h1 : AB = BP) (h2 : BC = CQ) (h3 : CD = DR) (h4 : DA = AS)

-- Lean 4 statement for the proof problem
theorem ratio_of_areas : area A B C D / area P Q R S = 1/5 :=
sorry

end ratio_of_areas_l394_39467


namespace order_of_p_q_r_l394_39412

theorem order_of_p_q_r (p q r : ℝ) (h1 : p = Real.sqrt 2) (h2 : q = Real.sqrt 7 - Real.sqrt 3) (h3 : r = Real.sqrt 6 - Real.sqrt 2) :
  p > r ∧ r > q :=
by
  sorry

end order_of_p_q_r_l394_39412


namespace pen_and_notebook_cost_l394_39468

theorem pen_and_notebook_cost :
  ∃ (p n : ℕ), 15 * p + 5 * n = 130 ∧ p > n ∧ p + n = 10 := by
  sorry

end pen_and_notebook_cost_l394_39468


namespace percentage_of_copper_buttons_l394_39466

-- Definitions for conditions
def total_items : ℕ := 100
def pin_percentage : ℕ := 30
def button_percentage : ℕ := 100 - pin_percentage
def brass_button_percentage : ℕ := 60
def copper_button_percentage : ℕ := 100 - brass_button_percentage

-- Theorem statement proving the question
theorem percentage_of_copper_buttons (h1 : pin_percentage = 30)
  (h2 : button_percentage = total_items - pin_percentage)
  (h3 : brass_button_percentage = 60)
  (h4 : copper_button_percentage = total_items - brass_button_percentage) :
  (button_percentage * copper_button_percentage) / total_items = 28 := 
sorry

end percentage_of_copper_buttons_l394_39466


namespace people_receiving_roses_l394_39470

-- Defining the conditions.
def initial_roses : Nat := 40
def stolen_roses : Nat := 4
def roses_per_person : Nat := 4

-- Stating the theorem.
theorem people_receiving_roses : 
  (initial_roses - stolen_roses) / roses_per_person = 9 :=
by sorry

end people_receiving_roses_l394_39470


namespace relationship_between_T_and_S_l394_39496

variable (a b : ℝ)

def T : ℝ := a + 2 * b
def S : ℝ := a + b^2 + 1

theorem relationship_between_T_and_S : T a b ≤ S a b := by
  sorry

end relationship_between_T_and_S_l394_39496


namespace solve_x_l394_39463

theorem solve_x (x: ℝ) (h: -4 * x - 15 = 12 * x + 5) : x = -5 / 4 :=
sorry

end solve_x_l394_39463


namespace pyramid_sphere_area_l394_39429

theorem pyramid_sphere_area (a : ℝ) (PA PB PC : ℝ) 
  (h1 : PA = PB) (h2 : PA = 2 * PC) 
  (h3 : PA = 2 * a) (h4 : PB = 2 * a) 
  (h5 : 4 * π * (PA^2 + PB^2 + PC^2) / 9 = 9 * π) :
  a = 1 :=
by
  sorry

end pyramid_sphere_area_l394_39429


namespace imaginary_part_z1_mul_z2_l394_39424

def z1 : ℂ := ⟨1, -1⟩
def z2 : ℂ := ⟨2, 4⟩

theorem imaginary_part_z1_mul_z2 : (z1 * z2).im = 2 := by
  sorry

end imaginary_part_z1_mul_z2_l394_39424


namespace compare_logs_l394_39492

noncomputable def e := Real.exp 1
noncomputable def log_base_10 (x : Real) := Real.log x / Real.log 10

theorem compare_logs (x : Real) (hx : e < x ∧ x < 10) :
  let a := Real.log (Real.log x)
  let b := log_base_10 (log_base_10 x)
  let c := Real.log (log_base_10 x)
  let d := log_base_10 (Real.log x)
  c < b ∧ b < d ∧ d < a := 
sorry

end compare_logs_l394_39492


namespace solve_inequality_l394_39461

noncomputable def f (x : ℝ) : ℝ :=
  x^3 + x + 2^x - 2^(-x)

theorem solve_inequality (x : ℝ) : 
  f (Real.exp x - x) ≤ 7/2 ↔ x = 0 := 
sorry

end solve_inequality_l394_39461


namespace greatest_common_divisor_456_108_lt_60_l394_39416

theorem greatest_common_divisor_456_108_lt_60 : 
  let divisors_456 := {d : ℕ | d ∣ 456}
  let divisors_108 := {d : ℕ | d ∣ 108}
  let common_divisors := divisors_456 ∩ divisors_108
  let common_divisors_lt_60 := {d ∈ common_divisors | d < 60}
  ∃ d, d ∈ common_divisors_lt_60 ∧ ∀ e ∈ common_divisors_lt_60, e ≤ d ∧ d = 12 := by {
    sorry
  }

end greatest_common_divisor_456_108_lt_60_l394_39416


namespace frank_money_remaining_l394_39437

-- Define the conditions
def cost_cheapest_lamp : ℕ := 20
def factor_most_expensive : ℕ := 3
def initial_money : ℕ := 90

-- Define the cost of the most expensive lamp
def cost_most_expensive_lamp : ℕ := cost_cheapest_lamp * factor_most_expensive

-- Define the money remaining after purchase
def money_remaining : ℕ := initial_money - cost_most_expensive_lamp

-- The theorem we need to prove
theorem frank_money_remaining : money_remaining = 30 := by
  sorry

end frank_money_remaining_l394_39437


namespace unit_price_in_range_l394_39406

-- Given definitions and conditions
def Q (x : ℝ) : ℝ := 220 - 2 * x
def f (x : ℝ) : ℝ := x * Q x

-- The desired range for the unit price to maintain a production value of at least 60 million yuan
def valid_unit_price_range (x : ℝ) : Prop := 50 < x ∧ x < 60

-- The main theorem that needs to be proven
theorem unit_price_in_range (x : ℝ) (h₁ : 0 < x) (h₂ : x < 500) (h₃ : f x ≥ 60 * 10^6) : valid_unit_price_range x :=
sorry

end unit_price_in_range_l394_39406


namespace max_quotient_l394_39402

theorem max_quotient (x y : ℝ) (hx : 100 ≤ x ∧ x ≤ 300) (hy : 900 ≤ y ∧ y ≤ 1800) : 
  (∀ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) → y / x ≤ 18) ∧ 
  (∃ x y, (100 ≤ x ∧ x ≤ 300) ∧ (900 ≤ y ∧ y ≤ 1800) ∧ y / x = 18) :=
by
  sorry

end max_quotient_l394_39402


namespace find_m_and_f_max_l394_39448

noncomputable def f (x m : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin (2 * x) + 2 * (Real.cos x)^2 + m

theorem find_m_and_f_max (m a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x m ≥ 3) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), ∃ y, f y m = 3) →
  (∀ x ∈ Set.Icc a (a + Real.pi), ∃ y, f y m = 6) →
  m = 3 ∧ ∀ x ∈ Set.Icc a (a + Real.pi), f x 3 ≤ 6 :=
sorry

end find_m_and_f_max_l394_39448


namespace no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l394_39487

-- Part (a)
theorem no_integers_a_b_existence (a b : ℤ) :
  ¬(a^2 - 3 * (b^2) = 8) :=
sorry

-- Part (b)
theorem no_positive_integers_a_b_c_existence (a b c : ℕ) (ha: a > 0) (hb: b > 0) (hc: c > 0 ) :
  ¬(a^2 + b^2 = 3 * (c^2)) :=
sorry

end no_integers_a_b_existence_no_positive_integers_a_b_c_existence_l394_39487


namespace binom_1300_2_l394_39415

theorem binom_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_l394_39415


namespace snowman_volume_l394_39462

theorem snowman_volume (r1 r2 r3 : ℝ) (V1 V2 V3 : ℝ) (π : ℝ) 
  (h1 : r1 = 4) (h2 : r2 = 6) (h3 : r3 = 8) 
  (hV1 : V1 = (4/3) * π * (r1^3)) 
  (hV2 : V2 = (4/3) * π * (r2^3)) 
  (hV3 : V3 = (4/3) * π * (r3^3)) :
  V1 + V2 + V3 = (3168/3) * π :=
by 
  sorry

end snowman_volume_l394_39462


namespace value_of_x_l394_39494

theorem value_of_x (x : ℝ) :
  (4 / x) * 12 = 8 ↔ x = 6 :=
by
  sorry

end value_of_x_l394_39494


namespace john_age_multiple_of_james_age_l394_39486

-- Define variables for the problem conditions
def john_current_age : ℕ := 39
def john_age_3_years_ago : ℕ := john_current_age - 3

def james_brother_age : ℕ := 16
def james_brother_older : ℕ := 4

def james_current_age : ℕ := james_brother_age - james_brother_older
def james_age_in_6_years : ℕ := james_current_age + 6

-- The goal is to prove the multiple relationship
theorem john_age_multiple_of_james_age :
  john_age_3_years_ago = 2 * james_age_in_6_years :=
by {
  -- Skip the proof
  sorry
}

end john_age_multiple_of_james_age_l394_39486


namespace annual_interest_rate_last_year_l394_39427

-- Define the conditions
def increased_by_ten_percent (r : ℝ) : ℝ := 1.10 * r

-- Statement of the problem
theorem annual_interest_rate_last_year (r : ℝ) (h : increased_by_ten_percent r = 0.11) : r = 0.10 :=
sorry

end annual_interest_rate_last_year_l394_39427


namespace time_for_model_M_l394_39408

variable (T : ℝ) -- Time taken by model M computer to complete the task in minutes.
variable (n_m : ℝ := 12) -- Number of model M computers
variable (n_n : ℝ := 12) -- Number of model N computers
variable (time_n : ℝ := 18) -- Time taken by model N computer to complete the task in minutes

theorem time_for_model_M :
  n_m / T + n_n / time_n = 1 → T = 36 := by
sorry

end time_for_model_M_l394_39408


namespace trader_loses_l394_39489

theorem trader_loses 
  (l_1 l_2 q : ℝ) 
  (h1 : l_1 ≠ l_2) 
  (p_1 p_2 : ℝ) 
  (h2 : p_1 = q * (l_2 / l_1)) 
  (h3 : p_2 = q * (l_1 / l_2)) :
  p_1 + p_2 > 2 * q :=
by {
  sorry
}

end trader_loses_l394_39489


namespace hyperbola_asymptote_focal_length_l394_39444

theorem hyperbola_asymptote_focal_length (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : c = 2 * Real.sqrt 5) (h4 : b / a = 2) : a = 2 :=
by
  sorry

end hyperbola_asymptote_focal_length_l394_39444


namespace smallest_possible_N_l394_39495

theorem smallest_possible_N (l m n : ℕ) (h_visible : (l - 1) * (m - 1) * (n - 1) = 252) : l * m * n = 392 :=
sorry

end smallest_possible_N_l394_39495


namespace find_q_l394_39473

variable (p q : ℝ) (hp : p > 1) (hq : q > 1) (h_cond1 : 1 / p + 1 / q = 1) (h_cond2 : p * q = 9)

theorem find_q : q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l394_39473


namespace find_angle_F_l394_39438

-- Declaring the necessary angles
variables (E F G H : ℝ) -- Angles are real numbers

-- Declaring the conditions
axiom parallel_lines : E = 3 * H
axiom angle_relation1 : G = 2 * F
axiom supplementary_angles : F + G = 180

-- The theorem statement
theorem find_angle_F (h1 : E = 3 * H) (h2 : G = 2 * F) (h3 : F + G = 180) : F = 60 :=
  sorry

end find_angle_F_l394_39438


namespace arithmetic_sequence_100th_term_l394_39477

-- Define the first term and the common difference
def first_term : ℕ := 3
def common_difference : ℕ := 7

-- Define the formula for the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Theorem: The 100th term of the arithmetic sequence is 696.
theorem arithmetic_sequence_100th_term :
  nth_term first_term common_difference 100 = 696 :=
  sorry

end arithmetic_sequence_100th_term_l394_39477


namespace min_value_f_l394_39453

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + (1 / x)))

theorem min_value_f : ∃ x > 0, ∀ y > 0, f y ≥ f x ∧ f x = 5 / 2 :=
by
  sorry

end min_value_f_l394_39453


namespace simplify_expression_1_combine_terms_l394_39459

variable (a b : ℝ)

-- Problem 1: Simplification
theorem simplify_expression_1 : 2 * (2 * a^2 + 9 * b) + (-3 * a^2 - 4 * b) = a^2 + 14 * b := by 
  sorry

-- Problem 2: Combine like terms
theorem combine_terms : 3 * a^2 * b + 2 * a * b^2 - 5 - 3 * a^2 * b - 5 * a * b^2 + 2 = -3 * a * b^2 - 3 := by 
  sorry

end simplify_expression_1_combine_terms_l394_39459


namespace find_natural_numbers_l394_39484

theorem find_natural_numbers (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 3^5) : 
  (x = 6 ∧ y = 3) := 
sorry

end find_natural_numbers_l394_39484


namespace total_popsicle_sticks_l394_39471

def Gino_popsicle_sticks : ℕ := 63
def My_popsicle_sticks : ℕ := 50
def Nick_popsicle_sticks : ℕ := 82

theorem total_popsicle_sticks : Gino_popsicle_sticks + My_popsicle_sticks + Nick_popsicle_sticks = 195 := by
  sorry

end total_popsicle_sticks_l394_39471


namespace solve_system_of_equations_l394_39423

theorem solve_system_of_equations :
  ∃ x y : ℝ, (x - y = 2) ∧ (2 * x + y = 7) ∧ x = 3 ∧ y = 1 :=
by
  sorry

end solve_system_of_equations_l394_39423


namespace quadratic_range_l394_39436

open Real

def quadratic (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 5

theorem quadratic_range :
  ∀ x, -2 ≤ x ∧ x ≤ 2 → -8 ≤ quadratic x ∧ quadratic x ≤ 19 :=
by
  intro x h
  sorry

end quadratic_range_l394_39436


namespace weavers_in_first_group_l394_39447

theorem weavers_in_first_group :
  (∃ W : ℕ, (W * 4 = 4) ∧ (12 * 12 = 36) ∧ (4 / (W * 4) = 36 / (12 * 12))) -> (W = 4) :=
by
  sorry

end weavers_in_first_group_l394_39447


namespace find_n_value_l394_39442

theorem find_n_value : ∃ n : ℤ, 3^3 - 7 = 4^2 + n ∧ n = 4 :=
by
  use 4
  sorry

end find_n_value_l394_39442


namespace ellen_total_legos_l394_39431

-- Conditions
def ellen_original_legos : ℝ := 2080.0
def ellen_winning_legos : ℝ := 17.0

-- Theorem statement
theorem ellen_total_legos : ellen_original_legos + ellen_winning_legos = 2097.0 :=
by
  -- The proof would go here, but we will use sorry to indicate it is skipped.
  sorry

end ellen_total_legos_l394_39431


namespace minimum_buses_needed_l394_39457

theorem minimum_buses_needed (bus_capacity : ℕ) (students : ℕ) (h : bus_capacity = 38 ∧ students = 411) :
  ∃ n : ℕ, 38 * n ≥ students ∧ ∀ m : ℕ, 38 * m ≥ students → n ≤ m :=
by sorry

end minimum_buses_needed_l394_39457


namespace work_together_days_l394_39426

noncomputable def A_per_day := 1 / 78
noncomputable def B_per_day := 1 / 39

theorem work_together_days 
  (A : ℝ) (B : ℝ) 
  (hA : A = 1 / 78)
  (hB : B = 1 / 39) : 
  1 / (A + B) = 26 :=
by
  rw [hA, hB]
  sorry

end work_together_days_l394_39426


namespace exists_long_segment_between_parabolas_l394_39443

def parabola1 (x : ℝ) : ℝ :=
  x ^ 2

def parabola2 (x : ℝ) : ℝ :=
  x ^ 2 - 1

def in_between_parabolas (x y : ℝ) : Prop :=
  (parabola2 x) ≤ y ∧ y ≤ (parabola1 x)

theorem exists_long_segment_between_parabolas :
  ∃ (M1 M2: ℝ × ℝ), in_between_parabolas M1.1 M1.2 ∧ in_between_parabolas M2.1 M2.2 ∧ dist M1 M2 > 10^6 :=
sorry

end exists_long_segment_between_parabolas_l394_39443


namespace arithmetic_avg_salary_technicians_l394_39435

noncomputable def avg_salary_technicians_problem : Prop :=
  let average_salary_all := 8000
  let total_workers := 21
  let average_salary_rest := 6000
  let technician_count := 7
  let total_salary_all := average_salary_all * total_workers
  let total_salary_rest := average_salary_rest * (total_workers - technician_count)
  let total_salary_technicians := total_salary_all - total_salary_rest
  let average_salary_technicians := total_salary_technicians / technician_count
  average_salary_technicians = 12000

theorem arithmetic_avg_salary_technicians :
  avg_salary_technicians_problem :=
by {
  sorry -- Proof is omitted as per instructions.
}

end arithmetic_avg_salary_technicians_l394_39435


namespace min_value_of_expression_l394_39455

variable (a b c : ℝ)
variable (h1 : a + b + c = 1)
variable (h2 : 0 < a ∧ a < 1)
variable (h3 : 0 < b ∧ b < 1)
variable (h4 : 0 < c ∧ c < 1)
variable (h5 : 3 * a + 2 * b = 2)

theorem min_value_of_expression : (2 / a + 1 / (3 * b)) ≥ 16 / 3 := 
  sorry

end min_value_of_expression_l394_39455


namespace volume_ratio_l394_39428

theorem volume_ratio (A B C : ℚ) (h1 : (3/4) * A = (2/3) * B) (h2 : (2/3) * B = (1/2) * C) :
  A / C = 2 / 3 :=
sorry

end volume_ratio_l394_39428


namespace opposite_of_neg_two_l394_39417

-- Define what it means for 'b' to be the opposite of 'a'
def is_opposite (a b : Int) : Prop := a + b = 0

-- The theorem to be proved
theorem opposite_of_neg_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_l394_39417


namespace carlos_local_tax_deduction_l394_39460

theorem carlos_local_tax_deduction :
  let hourly_wage_dollars := 25
  let hourly_wage_cents := hourly_wage_dollars * 100
  let tax_rate := 2.5 / 100
  hourly_wage_cents * tax_rate = 62.5 :=
by
  sorry

end carlos_local_tax_deduction_l394_39460


namespace find_number_l394_39483

theorem find_number (x : ℝ) (h : ((x / 3) * 24) - 7 = 41) : x = 6 :=
by
  sorry

end find_number_l394_39483


namespace xy_value_l394_39452

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 :=
by
  sorry

end xy_value_l394_39452


namespace total_hunts_is_21_l394_39440

-- Define the initial conditions
def Sam_hunts : Nat := 6
def Rob_hunts : Nat := Sam_hunts / 2
def Rob_Sam_total_hunt : Nat := Sam_hunts + Rob_hunts
def Mark_hunts : Nat := Rob_Sam_total_hunt / 3
def Peter_hunts : Nat := Mark_hunts * 3

-- The main theorem to prove
theorem total_hunts_is_21 : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 :=
by
  sorry

end total_hunts_is_21_l394_39440


namespace find_numbers_l394_39401

theorem find_numbers (x y : ℤ) (h1 : x > y) (h2 : x^2 - y^2 = 100) : 
  x = 26 ∧ y = 24 := 
  sorry

end find_numbers_l394_39401


namespace monotonically_increasing_intervals_exists_a_decreasing_l394_39450

noncomputable def f (x : ℝ) (a : ℝ) := Real.exp x - a * x - 1

theorem monotonically_increasing_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, 0 ≤ Real.exp x - a) ∧
  (a > 0 → ∀ x : ℝ, x ≥ Real.log a → 0 ≤ Real.exp x - a) :=
by sorry

theorem exists_a_decreasing (a : ℝ) :
  (a ≥ Real.exp 3) ↔ ∀ x : ℝ, -2 < x ∧ x < 3 → Real.exp x - a ≤ 0 :=
by sorry

end monotonically_increasing_intervals_exists_a_decreasing_l394_39450


namespace rope_for_second_post_l394_39419

theorem rope_for_second_post 
(r1 r2 r3 r4 : ℕ) 
(h_total : r1 + r2 + r3 + r4 = 70)
(h_r1 : r1 = 24)
(h_r3 : r3 = 14)
(h_r4 : r4 = 12) 
: r2 = 20 := 
by 
  sorry

end rope_for_second_post_l394_39419


namespace max_blocks_fit_l394_39405

-- Define the dimensions of the block and the box
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

-- Define the volumes calculation
def volume (d : Dimensions) : ℕ :=
  d.length * d.width * d.height

-- Define the dimensions of the block and the box
def block : Dimensions := { length := 3, width := 1, height := 2 }
def box : Dimensions := { length := 4, width := 3, height := 6 }

-- Prove that the maximum number of blocks that can fit in the box is 12
theorem max_blocks_fit : (volume box) / (volume block) = 12 := by sorry

end max_blocks_fit_l394_39405


namespace smallest_n_satisfying_conditions_l394_39465

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ n ≡ 1 [MOD 7] ∧ n ≡ 1 [MOD 4] ∧ n = 113 :=
by
  sorry

end smallest_n_satisfying_conditions_l394_39465


namespace buses_in_parking_lot_l394_39498

def initial_buses : ℕ := 7
def additional_buses : ℕ := 6
def total_buses : ℕ := initial_buses + additional_buses

theorem buses_in_parking_lot : total_buses = 13 := by
  sorry

end buses_in_parking_lot_l394_39498


namespace center_of_hyperbola_l394_39497

theorem center_of_hyperbola :
  ∃ (h k : ℝ), (h = 2 ∧ k = 4) ∧ (9 * (x - h)^2 - 16 * (y - k)^2 = 180) :=
  sorry

end center_of_hyperbola_l394_39497


namespace white_roses_needed_l394_39499

theorem white_roses_needed (bouquets table_decorations white_roses_per_table_decoration white_roses_per_bouquet : ℕ)
  (h_bouquets : bouquets = 5)
  (h_table_decorations : table_decorations = 7)
  (h_white_roses_per_table_decoration : white_roses_per_table_decoration = 12)
  (h_white_roses_per_bouquet : white_roses_per_bouquet = 5):
  bouquets * white_roses_per_bouquet + table_decorations * white_roses_per_table_decoration = 109 := by
  sorry

end white_roses_needed_l394_39499


namespace solution_set_of_inequality_l394_39414

theorem solution_set_of_inequality :
  { x : ℝ | 2 / (x - 1) ≥ 1 } = { x : ℝ | 1 < x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l394_39414


namespace unattainable_y_l394_39439

theorem unattainable_y (x : ℚ) (y : ℚ) (h : y = (1 - 2 * x) / (3 * x + 4)) (hx : x ≠ -4 / 3) : y ≠ -2 / 3 :=
by {
  sorry
}

end unattainable_y_l394_39439


namespace polynomial_form_l394_39474

def is_even_poly (P : ℝ → ℝ) : Prop := 
  ∀ x, P x = P (-x)

theorem polynomial_form (P : ℝ → ℝ) (hP : ∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) : 
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x ^ 4 + b * x ^ 2 := 
  sorry

end polynomial_form_l394_39474


namespace problem1_problem2_l394_39441

-- Problem 1
theorem problem1 (m n : ℚ) (h : m ≠ n) : 
  (m / (m - n)) + (n / (n - m)) = 1 := 
by
  -- Proof steps would go here
  sorry

-- Problem 2
theorem problem2 (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) : 
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := 
by
  -- Proof steps would go here
  sorry

end problem1_problem2_l394_39441


namespace problem_inequality_l394_39493

theorem problem_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 :=
sorry

end problem_inequality_l394_39493


namespace circle_diameter_l394_39456

theorem circle_diameter (r d : ℝ) (h1 : π * r^2 = 4 * π) (h2 : d = 2 * r) : d = 4 :=
by {
  sorry
}

end circle_diameter_l394_39456


namespace problem_part1_and_part2_l394_39446

noncomputable def g (x a b : ℝ) : ℝ := a * Real.log x + 0.5 * x ^ 2 + (1 - b) * x

-- Given: the function definition and conditions
variables (a b : ℝ)
variables (x1 x2 : ℝ)
variables (hx1 : x1 ∈ Set.Ioi 0) (hx2 : x2 ∈ Set.Ioi 0)
variables (h_tangent : 8 * 1 - 2 * g 1 a b - 3 = 0)
variables (h_extremes : b = a + 1)

-- Prove the values of a and b as well as the inequality
theorem problem_part1_and_part2 :
  (a = 1 ∧ b = -1) ∧ (g x1 a b + g x2 a b < -4) :=
sorry

end problem_part1_and_part2_l394_39446


namespace difference_divisible_by_9_l394_39404

-- Define the integers a and b
variables (a b : ℤ)

-- Define the theorem statement
theorem difference_divisible_by_9 (a b : ℤ) : 9 ∣ ((3 * a + 2) ^ 2 - (3 * b + 2) ^ 2) :=
sorry

end difference_divisible_by_9_l394_39404


namespace final_points_l394_39475

-- Definitions of the points in each round
def first_round_points : Int := 16
def second_round_points : Int := 33
def last_round_points : Int := -48

-- The theorem to prove Emily's final points
theorem final_points :
  first_round_points + second_round_points + last_round_points = 1 :=
by
  sorry

end final_points_l394_39475


namespace num_rows_seat_9_people_l394_39454

-- Define the premises of the problem.
def seating_arrangement (x y : ℕ) : Prop := (9 * x + 7 * y = 58)

-- The theorem stating the number of rows seating exactly 9 people.
theorem num_rows_seat_9_people
  (x y : ℕ)
  (h : seating_arrangement x y) :
  x = 1 :=
by
  -- Proof is not required as per the instruction
  sorry

end num_rows_seat_9_people_l394_39454


namespace length_of_third_side_l394_39425

-- Definitions for sides and perimeter condition
variables (a b : ℕ) (h1 : a = 3) (h2 : b = 10) (p : ℕ) (h3 : p % 6 = 0)
variable (c : ℕ)

-- Definition for the triangle inequality
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove the length of the third side
theorem length_of_third_side (h4 : triangle_inequality a b c)
  (h5 : p = a + b + c) : c = 11 :=
sorry

end length_of_third_side_l394_39425


namespace charles_total_money_l394_39410

-- Definitions based on the conditions in step a)
def number_of_pennies : ℕ := 6
def number_of_nickels : ℕ := 3
def value_of_penny : ℕ := 1
def value_of_nickel : ℕ := 5

-- Calculations in Lean terms
def total_pennies_value : ℕ := number_of_pennies * value_of_penny
def total_nickels_value : ℕ := number_of_nickels * value_of_nickel
def total_money : ℕ := total_pennies_value + total_nickels_value

-- The final proof statement based on step c)
theorem charles_total_money : total_money = 21 := by
  sorry

end charles_total_money_l394_39410


namespace marbles_left_l394_39458

theorem marbles_left (initial_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 64 → given_marbles = 14 → remaining_marbles = (initial_marbles - given_marbles) → remaining_marbles = 50 :=
by
  intros h_initial h_given h_calculation
  rw [h_initial, h_given] at h_calculation
  exact h_calculation

end marbles_left_l394_39458


namespace find_k_from_inequality_l394_39400

variable (k x : ℝ)

theorem find_k_from_inequality (h : ∀ x ∈ Set.Ico (-2 : ℝ) 1, 1 + k / (x - 1) ≤ 0)
  (h₂: 1 + k / (-2 - 1) = 0) :
  k = 3 :=
by
  sorry

end find_k_from_inequality_l394_39400


namespace weekly_exercise_time_l394_39421

def milesWalked := 3
def walkingSpeed := 3 -- in miles per hour
def milesRan := 10
def runningSpeed := 5 -- in miles per hour
def daysInWeek := 7

theorem weekly_exercise_time : (milesWalked / walkingSpeed + milesRan / runningSpeed) * daysInWeek = 21 := 
by
  -- The actual proof part is intentionally omitted as per the instruction
  sorry

end weekly_exercise_time_l394_39421


namespace calculate_expression_l394_39403

theorem calculate_expression : 
  (2^10 + (3^6 / 3^2)) = 1105 := 
by 
  -- Steps involve intermediate calculations
  -- for producing (2^10 = 1024), (3^6 = 729), (3^2 = 9)
  -- and then finding (729 / 9 = 81), (1024 + 81 = 1105)
  sorry

end calculate_expression_l394_39403


namespace area_of_new_geometric_figure_correct_l394_39488

noncomputable def area_of_new_geometric_figure (a b : ℝ) : ℝ := 
  let d := Real.sqrt (a^2 + b^2)
  a * b + (b * d) / 4

theorem area_of_new_geometric_figure_correct (a b : ℝ) :
  area_of_new_geometric_figure a b = a * b + (b * Real.sqrt (a^2 + b^2)) / 4 :=
by 
  sorry

end area_of_new_geometric_figure_correct_l394_39488


namespace sin_y_eq_neg_one_l394_39482

noncomputable def α := Real.arccos (-1 / 5)

theorem sin_y_eq_neg_one (x y z : ℝ) (h1 : x = y - α) (h2 : z = y + α)
  (h3 : (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y) ^ 2) : Real.sin y = -1 :=
sorry

end sin_y_eq_neg_one_l394_39482


namespace square_side_length_l394_39418

noncomputable def diagonal_in_inches : ℝ := 2 * Real.sqrt 2
noncomputable def inches_to_feet : ℝ := 1 / 12
noncomputable def diagonal_in_feet := diagonal_in_inches * inches_to_feet
noncomputable def factor_sqrt_2 : ℝ := 1 / Real.sqrt 2

theorem square_side_length :
  let diagonal_feet := diagonal_in_feet 
  let side_length_feet := diagonal_feet * factor_sqrt_2
  side_length_feet = 1 / 6 :=
sorry

end square_side_length_l394_39418


namespace john_ate_12_ounces_of_steak_l394_39472

-- Conditions
def original_weight : ℝ := 30
def burned_fraction : ℝ := 0.5
def eaten_fraction : ℝ := 0.8

-- Theorem statement
theorem john_ate_12_ounces_of_steak :
  (original_weight * (1 - burned_fraction) * eaten_fraction) = 12 := by
  sorry

end john_ate_12_ounces_of_steak_l394_39472


namespace circle_center_polar_coords_l394_39449

noncomputable def polar_center (ρ θ : ℝ) : (ℝ × ℝ) :=
  (-1, 0)

theorem circle_center_polar_coords : 
  ∀ ρ θ : ℝ, ρ = -2 * Real.cos θ → polar_center ρ θ = (1, π) :=
by
  intro ρ θ h
  sorry

end circle_center_polar_coords_l394_39449


namespace positive_difference_of_two_numbers_l394_39434

theorem positive_difference_of_two_numbers 
  (x y : ℝ) 
  (h1 : x + y = 10) 
  (h2 : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := 
sorry

end positive_difference_of_two_numbers_l394_39434


namespace table_fill_impossible_l394_39407

/-- Proposition: Given a 7x3 table filled with 0s and 1s, it is impossible to prevent any 2x2 submatrix from having all identical numbers. -/
theorem table_fill_impossible : 
  ¬ ∃ (M : (Fin 7) → (Fin 3) → Fin 2), 
      ∀ i j, (i < 6) → (j < 2) → 
              (M i j = M i.succ j) ∨ 
              (M i j = M i j.succ) ∨ 
              (M i j = M i.succ j.succ) ∨ 
              (M i.succ j = M i j.succ → M i j = M i.succ j.succ) :=
sorry

end table_fill_impossible_l394_39407


namespace sqrt6_op_sqrt6_l394_39422

variable (x y : ℝ)

noncomputable def op (x y : ℝ) := (x + y)^2 - (x - y)^2

theorem sqrt6_op_sqrt6 : ∀ (x y : ℝ), op (Real.sqrt 6) (Real.sqrt 6) = 24 := by
  sorry

end sqrt6_op_sqrt6_l394_39422


namespace total_paintable_area_correct_l394_39433

namespace BarnPainting

-- Define the dimensions of the barn
def barn_width : ℕ := 12
def barn_length : ℕ := 15
def barn_height : ℕ := 6

-- Define the dimensions of the windows
def window_width : ℕ := 2
def window_height : ℕ := 3
def num_windows : ℕ := 2

-- Calculate the total number of square yards to be painted
def total_paintable_area : ℕ :=
  let wall1_area := barn_height * barn_width
  let wall2_area := barn_height * barn_length
  let wall_area := 2 * wall1_area + 2 * wall2_area
  let window_area := num_windows * (window_width * window_height)
  let painted_walls_area := wall_area - window_area
  let ceiling_area := barn_width * barn_length
  let total_area := 2 * painted_walls_area + ceiling_area
  total_area

theorem total_paintable_area_correct : total_paintable_area = 780 :=
  by sorry

end BarnPainting

end total_paintable_area_correct_l394_39433


namespace slope_of_line_l394_39411

-- Defining the conditions
def intersects_on_line (s x y : ℝ) : Prop :=
  (2 * x + 3 * y = 8 * s + 6) ∧ (x + 2 * y = 5 * s - 1)

-- Theorem stating that the slope of the line on which all intersections lie is 2
theorem slope_of_line {s x y : ℝ} :
  (∃ s x y, intersects_on_line s x y) → (∃ (m : ℝ), m = 2) :=
by sorry

end slope_of_line_l394_39411


namespace common_divisors_sum_diff_l394_39430

theorem common_divisors_sum_diff (A B : ℤ) (h : Int.gcd A B = 1) : 
  {d : ℤ | d ∣ A + B ∧ d ∣ A - B} = {1, 2} :=
sorry

end common_divisors_sum_diff_l394_39430


namespace total_length_of_fence_l394_39413

theorem total_length_of_fence (x : ℝ) (h1 : 2 * x * x = 1250) : 2 * x + 2 * x = 100 :=
by
  sorry

end total_length_of_fence_l394_39413


namespace thompson_class_average_l394_39479

theorem thompson_class_average
  (n : ℕ) (initial_avg : ℚ) (final_avg : ℚ) (bridget_index : ℕ) (first_n_score_sum : ℚ)
  (total_students : ℕ) (final_score_sum : ℚ)
  (h1 : n = 17) -- Number of students initially graded
  (h2 : initial_avg = 76) -- Average score of the first 17 students
  (h3 : final_avg = 78) -- Average score after adding Bridget's test
  (h4 : bridget_index = 18) -- Total number of students
  (h5 : total_students = 18) -- Total number of students
  (h6 : first_n_score_sum = n * initial_avg) -- Total score of the first 17 students
  (h7 : final_score_sum = total_students * final_avg) -- Total score of the 18 students):
  -- Bridget's score
  (bridgets_score : ℚ) :
  bridgets_score = final_score_sum - first_n_score_sum :=
sorry

end thompson_class_average_l394_39479


namespace find_number_l394_39481

theorem find_number (n x : ℤ)
  (h1 : (2 * x + 1) = (x - 7)) 
  (h2 : ∃ x : ℤ, n = (2 * x + 1) ^ 2) : 
  n = 25 := 
sorry

end find_number_l394_39481


namespace erased_number_l394_39476

theorem erased_number (n i : ℕ) (h : (n * (n + 1) / 2 - i) / (n - 1) = 602 / 17) : i = 7 :=
sorry

end erased_number_l394_39476


namespace factorial_simplification_l394_39420

theorem factorial_simplification :
  Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 728 := 
sorry

end factorial_simplification_l394_39420


namespace sufficient_condition_for_ellipse_l394_39451

theorem sufficient_condition_for_ellipse (m : ℝ) (h : m^2 > 5) : m^2 > 4 := by
  sorry

end sufficient_condition_for_ellipse_l394_39451


namespace inscribed_circle_radius_l394_39469

theorem inscribed_circle_radius :
  ∀ (r : ℝ), 
    (∀ (R : ℝ), R = 12 →
      (∀ (d : ℝ), d = 12 → r = 3)) :=
by sorry

end inscribed_circle_radius_l394_39469


namespace range_of_a_l394_39409

noncomputable def proof_problem (x : ℝ) (a : ℝ) : Prop :=
  (x^2 - 4*x + 3 < 0) ∧ (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + a < 0)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, proof_problem x a) ↔ a ≤ 9 :=
by
  sorry

end range_of_a_l394_39409


namespace percentage_of_sum_is_14_l394_39491

-- Define variables x, y as real numbers
variables (x y P : ℝ)

-- Define condition 1: y is 17.647058823529413% of x
def y_is_percentage_of_x : Prop := y = 0.17647058823529413 * x

-- Define condition 2: 20% of (x - y) is equal to P% of (x + y)
def percentage_equation : Prop := 0.20 * (x - y) = (P / 100) * (x + y)

-- Define the statement to be proved: P is 14
theorem percentage_of_sum_is_14 (h1 : y_is_percentage_of_x x y) (h2 : percentage_equation x y P) : 
  P = 14 :=
by
  sorry

end percentage_of_sum_is_14_l394_39491
