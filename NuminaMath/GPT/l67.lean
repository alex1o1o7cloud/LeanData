import Mathlib

namespace percentage_increase_in_freelance_l67_67077

open Real

def initial_part_time_earnings := 65
def new_part_time_earnings := 72
def initial_freelance_earnings := 45
def new_freelance_earnings := 72

theorem percentage_increase_in_freelance :
  (new_freelance_earnings - initial_freelance_earnings) / initial_freelance_earnings * 100 = 60 :=
by
  -- Proof will go here
  sorry

end percentage_increase_in_freelance_l67_67077


namespace polynomial_evaluation_l67_67873

theorem polynomial_evaluation (p : Polynomial ℚ) 
  (hdeg : p.degree = 7)
  (h : ∀ n : ℕ, n ≤ 7 → p.eval (2^n) = 1 / 2^(n + 1)) : 
  p.eval 0 = 255 / 2^28 := 
sorry

end polynomial_evaluation_l67_67873


namespace one_of_18_consecutive_is_divisible_l67_67130

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define what it means for one number to be divisible by another
def divisible (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- The main theorem
theorem one_of_18_consecutive_is_divisible : 
  ∀ (n : ℕ), 100 ≤ n ∧ n + 17 ≤ 999 → ∃ (k : ℕ), n ≤ k ∧ k ≤ (n + 17) ∧ divisible k (sum_of_digits k) :=
by
  intros n h
  sorry

end one_of_18_consecutive_is_divisible_l67_67130


namespace degree_diploma_salary_ratio_l67_67180

theorem degree_diploma_salary_ratio
  (jared_salary : ℕ)
  (diploma_monthly_salary : ℕ)
  (h_annual_salary : jared_salary = 144000)
  (h_diploma_annual_salary : 12 * diploma_monthly_salary = 48000) :
  (jared_salary / (12 * diploma_monthly_salary)) = 3 := 
by sorry

end degree_diploma_salary_ratio_l67_67180


namespace zoe_total_earnings_l67_67383

theorem zoe_total_earnings
  (weeks : ℕ → ℝ)
  (weekly_hours : ℕ → ℝ)
  (wage_per_hour : ℝ)
  (h1 : weekly_hours 3 = 28)
  (h2 : weekly_hours 2 = 18)
  (h3 : weeks 3 - weeks 2 = 64.40)
  (h_same_wage : ∀ n, weeks n = weekly_hours n * wage_per_hour) :
  weeks 3 + weeks 2 = 296.24 :=
sorry

end zoe_total_earnings_l67_67383


namespace solution_set_l67_67976

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l67_67976


namespace number_of_bottles_poured_l67_67157

/-- Definition of full cylinder capacity (fixed as 80 bottles) --/
def full_capacity : ℕ := 80

/-- Initial fraction of full capacity --/
def initial_fraction : ℚ := 3 / 4

/-- Final fraction of full capacity --/
def final_fraction : ℚ := 4 / 5

/-- Proof problem: Prove the number of bottles of oil poured into the cylinder --/
theorem number_of_bottles_poured :
  (final_fraction * full_capacity) - (initial_fraction * full_capacity) = 4 := by
  sorry

end number_of_bottles_poured_l67_67157


namespace units_digit_of_sum_64_8_75_8_is_1_l67_67178

def units_digit_in_base_8_sum (a b : ℕ) : ℕ :=
  (a + b) % 8

theorem units_digit_of_sum_64_8_75_8_is_1 :
  units_digit_in_base_8_sum 0o64 0o75 = 1 :=
sorry

end units_digit_of_sum_64_8_75_8_is_1_l67_67178


namespace brendan_remaining_money_l67_67649

-- Definitions based on conditions
def earned_amount : ℕ := 5000
def recharge_rate : ℕ := 1/2
def car_cost : ℕ := 1500

-- Proof Statement
theorem brendan_remaining_money : 
  (earned_amount * recharge_rate) - car_cost = 1000 :=
sorry

end brendan_remaining_money_l67_67649


namespace bruce_michael_total_goals_l67_67132

theorem bruce_michael_total_goals (bruce_goals : ℕ) (michael_goals : ℕ) 
  (h₁ : bruce_goals = 4) (h₂ : michael_goals = 3 * bruce_goals) : bruce_goals + michael_goals = 16 :=
by sorry

end bruce_michael_total_goals_l67_67132


namespace ratio_of_sums_l67_67569

open Nat

def sum_multiples_of_3 (n : Nat) : Nat :=
  let m := n / 3
  m * (3 + 3 * m) / 2

def sum_first_n_integers (n : Nat) : Nat :=
  n * (n + 1) / 2

theorem ratio_of_sums :
  (sum_multiples_of_3 600) / (sum_first_n_integers 300) = 4 / 3 :=
by
  sorry

end ratio_of_sums_l67_67569


namespace right_triangle_sides_l67_67322

theorem right_triangle_sides (a d : ℝ) (k : ℕ) (h_pos_a : 0 < a) (h_pos_d : 0 < d) (h_pos_k : 0 < k) :
  (a = 3) ∧ (d = 1) ∧ (k = 2) ↔ (a^2 + (a + d)^2 = (a + k * d)^2) :=
by 
  sorry

end right_triangle_sides_l67_67322


namespace expected_deviation_10_greater_than_100_l67_67295

-- Definitions for deviations for 10 and 100 tosses
def deviation_10_tosses (m₁₀: ℕ) : ℝ := (m₁₀ / 10) - 0.5
def deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  ∑ i, αs i / 10

-- Definitions for absolute deviations
def abs_deviation_10_tosses (m₁₀: ℕ) : ℝ := abs (deviation_10_tosses m₁₀)
def abs_deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  abs (deviation_100_tosses m₁₀₀ αs)

-- Expected values of absolute deviations
def expected_abs_deviation_10_tosses : ℝ := sorry
def expected_abs_deviation_100_tosses : ℝ := sorry

theorem expected_deviation_10_greater_than_100 :
  expected_abs_deviation_10_tosses > expected_abs_deviation_100_tosses :=
sorry

end expected_deviation_10_greater_than_100_l67_67295


namespace find_number_l67_67384

theorem find_number 
  (a b c d : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 6 * a + 9 * b + 3 * c + d = 88)
  (h6 : a - b + c - d = -6)
  (h7 : a - 9 * b + 3 * c - d = -46) : 
  1000 * a + 100 * b + 10 * c + d = 6507 := 
sorry

end find_number_l67_67384


namespace find_base_l67_67220

theorem find_base (b : ℝ) (h : 2.134 * b^3 < 21000) : b ≤ 21 :=
by
  have h1 : b < (21000 / 2.134) ^ (1 / 3) := sorry
  have h2 : (21000 / 2.134) ^ (1 / 3) < 21.5 := sorry
  have h3 : b ≤ 21 := sorry
  exact h3

end find_base_l67_67220


namespace product_end_digit_3_mod_5_l67_67503

theorem product_end_digit_3_mod_5 : 
  let lst := list.range' 0 10 in
  let lst := list.map (λ n, 10 * n + 3) lst in
  (list.prod lst) % 5 = 4 :=
by
  let lst := list.range' 0 10;
  let lst := list.map (λ n, 10 * n + 3) lst;
  show (list.prod lst) % 5 = 4;
  sorry

end product_end_digit_3_mod_5_l67_67503


namespace find_number_l67_67412

theorem find_number (x : ℝ) (h : x / 3 = 1.005 * 400) : x = 1206 := 
by 
sorry

end find_number_l67_67412


namespace expectation_absolute_deviation_l67_67282

noncomputable def absolute_deviation (n : ℕ) (m : ℕ) : ℝ :=
    |(m / n : ℝ) - 0.5|

theorem expectation_absolute_deviation (n m : ℕ) 
    (h₁ : n = 10) (h₂ : n = 100) (pn : ℕ → Real) :
    let E_α := ∫ x, absolute_deviation 10 (pn x): ℝ in
    let E_β := ∫ x, absolute_deviation 100 (pn x): ℝ in
    E_α > E_β :=
begin
  sorry
end

end expectation_absolute_deviation_l67_67282


namespace age_difference_is_58_l67_67410

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Age_difference : ℕ := Grandfather_age - Milena_age

theorem age_difference_is_58 : Age_difference = 58 := by
  sorry

end age_difference_is_58_l67_67410


namespace james_matches_l67_67859

theorem james_matches (dozen_boxes : ℕ) (matches_per_box : ℕ) (dozen_value : ℕ) 
  (h1 : dozen_boxes = 5) (h2 : matches_per_box = 20) (h3 : dozen_value = 12) :
  dozen_boxes * dozen_value * matches_per_box = 1200 :=
by
  rw [h1, h2, h3]
  calc
    5 * 12 * 20 = 60 * 20 := by norm_num
    ... = 1200 := by norm_num

end james_matches_l67_67859


namespace log_ratio_l67_67574

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : log_base 4 a = log_base 6 b)
  (h4 : log_base 6 b = log_base 9 (a + b)) :
  b / a = (1 + Real.sqrt 5) / 2 := sorry

end log_ratio_l67_67574


namespace problem1_problem2_l67_67168

-- Define a and b as real numbers
variables (a b : ℝ)

-- Problem 1: Prove (a-2b)^2 - (b-a)(a+b) = 2a^2 - 4ab + 3b^2
theorem problem1 : (a - 2 * b) ^ 2 - (b - a) * (a + b) = 2 * a ^ 2 - 4 * a * b + 3 * b ^ 2 :=
sorry

-- Problem 2: Prove (2a-b)^2 \cdot (2a+b)^2 = 16a^4 - 8a^2b^2 + b^4
theorem problem2 : (2 * a - b) ^ 2 * (2 * a + b) ^ 2 = 16 * a ^ 4 - 8 * a ^ 2 * b ^ 2 + b ^ 4 :=
sorry

end problem1_problem2_l67_67168


namespace numbers_masha_thought_l67_67403

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l67_67403


namespace average_speed_last_segment_l67_67703

theorem average_speed_last_segment (D : ℝ) (T_mins : ℝ) (S1 S2 : ℝ) (t : ℝ) (S_last : ℝ) :
  D = 150 ∧ T_mins = 135 ∧ S1 = 50 ∧ S2 = 60 ∧ t = 45 →
  S_last = 90 :=
by
    sorry

end average_speed_last_segment_l67_67703


namespace perpendicular_vectors_l67_67035

-- Definitions based on the conditions
def vector_a (x : ℝ) := (x, 3)
def vector_b := (3, 1)

-- Statement to prove
theorem perpendicular_vectors (x : ℝ) :
  (vector_a x).1 * (vector_b).1 + (vector_a x).2 * (vector_b).2 = 0 → x = -1 := by
  -- Proof goes here
  sorry

end perpendicular_vectors_l67_67035


namespace sin_cos_condition_necessary_but_not_sufficient_l67_67929

noncomputable def condition_sin_cos (k x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ k * sin x * cos x < x

noncomputable def necessary_but_not_sufficient (k : ℝ) : Prop :=
  (∀ x, condition_sin_cos k x) → k < 1

theorem sin_cos_condition_necessary_but_not_sufficient (k : ℝ) :
  necessary_but_not_sufficient k :=
sorry

end sin_cos_condition_necessary_but_not_sufficient_l67_67929


namespace complex_imaginary_unit_theorem_l67_67423

def complex_imaginary_unit_equality : Prop :=
  let i := Complex.I
  i * (i + 1) = -1 + i

theorem complex_imaginary_unit_theorem : complex_imaginary_unit_equality :=
by
  sorry

end complex_imaginary_unit_theorem_l67_67423


namespace perfume_price_l67_67477

variable (P : ℝ)

theorem perfume_price (h_increase : 1.10 * P = P + 0.10 * P)
    (h_decrease : 0.935 * P = 1.10 * P - 0.15 * 1.10 * P)
    (h_final_price : P - 0.935 * P = 78) : P = 1200 := 
by
  sorry

end perfume_price_l67_67477


namespace one_corresponds_to_36_l67_67851

-- Define the given conditions
def corresponds (n : Nat) (s : String) : Prop :=
match n with
| 2  => s = "36"
| 3  => s = "363"
| 4  => s = "364"
| 5  => s = "365"
| 36 => s = "2"
| _  => False

-- Statement for the proof problem: Prove that 1 corresponds to 36
theorem one_corresponds_to_36 : corresponds 1 "36" :=
by
  sorry

end one_corresponds_to_36_l67_67851


namespace total_crayons_lost_or_given_away_l67_67882

/-
Paul gave 52 crayons to his friends.
Paul lost 535 crayons.
Paul had 492 crayons left.
Prove that the total number of crayons lost or given away is 587.
-/
theorem total_crayons_lost_or_given_away
  (crayons_given : ℕ)
  (crayons_lost : ℕ)
  (crayons_left : ℕ)
  (h_crayons_given : crayons_given = 52)
  (h_crayons_lost : crayons_lost = 535)
  (h_crayons_left : crayons_left = 492) :
  crayons_given + crayons_lost = 587 := 
sorry

end total_crayons_lost_or_given_away_l67_67882


namespace partition_number_10_elements_l67_67797

open Finset

noncomputable def num_partitions (n : ℕ) : ℕ :=
  (2^n - 2) / 2

theorem partition_number_10_elements :
  num_partitions 10 = 511 := 
by
  sorry

end partition_number_10_elements_l67_67797


namespace sample_size_of_village_A_l67_67848

theorem sample_size_of_village_A
  (ratio_A : ℚ) (ratio_B : ℚ) (ratio_C : ℚ)
  (sample_A : ℕ) (total_sample_ratio : ℚ)
  (ratios_sum : ratio_A + ratio_B + ratio_C = total_sample_ratio)
  (village_A_ratio : ratio_A = 3 / total_sample_ratio)
  (sample_A_val : sample_A = 15) :
  ∃ n : ℕ, n = 70 :=
begin
  use 70,
  sorry
end

end sample_size_of_village_A_l67_67848


namespace set_equivalence_l67_67822

variable (M : Set ℕ)

theorem set_equivalence (h : M ∪ {1} = {1, 2, 3}) : M = {1, 2, 3} :=
sorry

end set_equivalence_l67_67822


namespace polynomial_roots_sum_l67_67725

theorem polynomial_roots_sum (a b c d e : ℝ) (h₀ : a ≠ 0)
  (h1 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
  (h2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h3 : a * 2^4 + b * 2^3 + c * 2^2 + d * 2 + e = 0) :
  (b + d) / a = -2677 := 
sorry

end polynomial_roots_sum_l67_67725


namespace opposite_of_neg_2_l67_67259

noncomputable def opposite (a : ℤ) : ℤ := 
  a * (-1)

theorem opposite_of_neg_2 : opposite (-2) = 2 := by
  -- definition of opposite
  unfold opposite
  -- calculation using the definition
  rfl

end opposite_of_neg_2_l67_67259


namespace factor_expression_l67_67027

variables (a : ℝ)

theorem factor_expression : (45 * a^2 + 135 * a + 90 * a^3) = 45 * a * (90 * a^2 + a + 3) :=
by sorry

end factor_expression_l67_67027


namespace find_f2023_l67_67256

-- Define the function and conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def satisfies_condition (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (1 - x)

-- Define the main statement to prove that f(2023) = 2 given conditions
theorem find_f2023 (f : ℝ → ℝ)
  (h1 : is_even f)
  (h2 : satisfies_condition f)
  (h3 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x) :
  f 2023 = 2 :=
sorry

end find_f2023_l67_67256


namespace solve_for_x_l67_67634

theorem solve_for_x (x : ℝ) (h : 0 < x) (h_property : (x / 100) * x^2 = 9) : x = 10 := by
  sorry

end solve_for_x_l67_67634


namespace part_a_l67_67911

theorem part_a (students : Fin 64 → Fin 8 × Fin 8 × Fin 8) :
  ∃ (A B : Fin 64), (students A).1 ≥ (students B).1 ∧ (students A).2.1 ≥ (students B).2.1 ∧ (students A).2.2 ≥ (students B).2.2 :=
sorry

end part_a_l67_67911


namespace larger_pie_flour_amount_l67_67076

variable (p1 : ℕ) (f1 : ℚ) (p2 : ℕ) (f2 : ℚ)

def prepared_pie_crusts (p1 p2 : ℕ) (f1 : ℚ) (f2 : ℚ) : Prop :=
  p1 * f1 = p2 * f2

theorem larger_pie_flour_amount (h : prepared_pie_crusts 40 25 (1/8) f2) : f2 = 1/5 :=
by
  sorry

end larger_pie_flour_amount_l67_67076


namespace exp_abs_dev_10_gt_100_l67_67293

open ProbabilityTheory
open MeasureTheory

def deviation (n m : ℕ) : ℝ := (m / n : ℝ) - 0.5
def abs_deviation (n m : ℕ) : ℝ := abs ((m / n : ℝ) - 0.5)
def exp_abs_deviation (n : ℕ) : ℝ := sorry -- The expected value of the absolute deviation

theorem exp_abs_dev_10_gt_100 :
  (exp_abs_deviation 10) > (exp_abs_deviation 100) :=
sorry

end exp_abs_dev_10_gt_100_l67_67293


namespace purchase_in_april_l67_67896

namespace FamilySavings

def monthly_income : ℕ := 150000
def monthly_expenses : ℕ := 115000
def initial_savings : ℕ := 45000
def furniture_cost : ℕ := 127000

noncomputable def monthly_savings : ℕ := monthly_income - monthly_expenses
noncomputable def additional_amount_needed : ℕ := furniture_cost - initial_savings
noncomputable def months_required : ℕ := (additional_amount_needed + monthly_savings - 1) / monthly_savings  -- ceiling division

theorem purchase_in_april : months_required = 3 :=
by
  -- Proof goes here
  sorry

end FamilySavings

end purchase_in_april_l67_67896


namespace problem1_problem2_l67_67326

-- Problem 1
theorem problem1 (x : ℝ) : (1 : ℝ) * (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := 
sorry

-- Problem 2
theorem problem2 (a b : ℝ) : (a - b)^2 * (b - a)^4 + (b - a)^3 * (a - b)^3 = 0 := 
sorry

end problem1_problem2_l67_67326


namespace problem_statement_l67_67521

open Real

noncomputable def f (ω x : ℝ) : ℝ := (sin (ω * x) * cos (ω * x) - cos (ω * x) ^ 2)

theorem problem_statement (ω : ℝ) (ω_pos : ω > 0) :
  (∃ T > 0, ∀ x, f ω (x + T) = f ω x) → ω = 2 ∧ 
  ∀ (a b c : ℝ) (h : b^2 = a * c) (x : angle B),
    sin (4 * x - π / 2) - 1 ∈ set.Icc (-2 : ℝ) 0  :=
by
  sorry

end problem_statement_l67_67521


namespace percentage_of_x_is_40_l67_67064

theorem percentage_of_x_is_40 
  (x p : ℝ)
  (h1 : (1 / 2) * x = 200)
  (h2 : p * x = 160) : 
  p * 100 = 40 := 
by 
  sorry

end percentage_of_x_is_40_l67_67064


namespace find_value_l67_67927

theorem find_value : 3 + 2 * (8 - 3) = 13 := by
  sorry

end find_value_l67_67927


namespace retirement_hiring_year_l67_67146

theorem retirement_hiring_year (A W Y : ℕ)
  (hired_on_32nd_birthday : A = 32)
  (eligible_to_retire_in_2007 : 32 + (2007 - Y) = 70) : 
  Y = 1969 := by
  sorry

end retirement_hiring_year_l67_67146


namespace John_lost_socks_l67_67554

theorem John_lost_socks (initial_socks remaining_socks : ℕ) (H1 : initial_socks = 20) (H2 : remaining_socks = 14) : initial_socks - remaining_socks = 6 :=
by
-- Proof steps can be skipped
sorry

end John_lost_socks_l67_67554


namespace goods_train_speed_l67_67314

theorem goods_train_speed
  (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
  (h_train_length : length_train = 250.0416)
  (h_platform_length : length_platform = 270)
  (h_time : time_seconds = 26) :
  (length_train + length_platform) / time_seconds * 3.6 = 72 := by
    sorry

end goods_train_speed_l67_67314


namespace remainder_of_product_mod_5_l67_67498

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l67_67498


namespace roots_of_polynomial_l67_67492

noncomputable def p (x : ℝ) : ℝ := x^3 - 3 * x^2 - x + 3

theorem roots_of_polynomial : {x : ℝ | p x = 0} = {1, -1, 3} :=
by
  sorry

end roots_of_polynomial_l67_67492


namespace estimate_ratio_l67_67788

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end estimate_ratio_l67_67788


namespace volume_ratio_surface_area_ratio_l67_67118

theorem volume_ratio_surface_area_ratio (V1 V2 S1 S2 : ℝ) (h : V1 / V2 = 8 / 27) :
  S1 / S2 = 4 / 9 :=
by
  sorry

end volume_ratio_surface_area_ratio_l67_67118


namespace product_of_distinct_integers_l67_67207

def is2008thPower (n : ℕ) : Prop := ∃ k : ℕ, n = k ^ 2008

theorem product_of_distinct_integers {x y z : ℕ} (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x)
  (h4 : y = (x + z) / 2) (h5 : x > 0) (h6 : y > 0) (h7 : z > 0) 
  : is2008thPower (x * y * z) :=
  sorry

end product_of_distinct_integers_l67_67207


namespace simplify_expression_l67_67572

variable (x y : ℝ)

theorem simplify_expression :
  3 * x + 4 * x^2 + 2 - (5 - 3 * x - 5 * x^2 + 2 * y) = 9 * x^2 + 6 * x - 2 * y - 3 :=
by
  sorry

end simplify_expression_l67_67572


namespace original_price_of_bag_l67_67782

theorem original_price_of_bag (P : ℝ) 
  (h1 : ∀ x, 0 < x → x < 1 → x * 100 = 75)
  (h2 : 2 * (0.25 * P) = 3)
  : P = 6 :=
sorry

end original_price_of_bag_l67_67782


namespace min_value_cos2_sin_l67_67581

theorem min_value_cos2_sin (x : ℝ) : 
  cos x ^ 2 - 2 * sin x ≥ -2 :=
by
  sorry

end min_value_cos2_sin_l67_67581


namespace rainbow_preschool_full_day_students_l67_67954

theorem rainbow_preschool_full_day_students (total_students : ℕ) (half_day_percent : ℝ)
  (h1 : total_students = 80) (h2 : half_day_percent = 0.25) :
  (total_students * (1 - half_day_percent)).to_nat = 60 :=
by
  -- Transform percentage to a fraction
  let fraction_full_day := 1 - half_day_percent
  -- Calculate full-day students
  have h_full_day_students : ℝ := total_students * fraction_full_day
  -- Convert to natural number
  exact (floor h_full_day_students).to_nat = 60

end rainbow_preschool_full_day_students_l67_67954


namespace gymnastics_average_people_per_team_l67_67028

def average_people_per_team (boys girls teams : ℕ) : ℕ :=
  (boys + girls) / teams

theorem gymnastics_average_people_per_team:
  average_people_per_team 83 77 4 = 40 :=
by
  sorry

end gymnastics_average_people_per_team_l67_67028


namespace intersection_M_N_l67_67527

def M : Set ℝ := { x | x^2 - x - 6 ≤ 0 }
def N : Set ℝ := { x | -2 < x ∧ x ≤ 4 }

theorem intersection_M_N : (M ∩ N) = { x | -2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_M_N_l67_67527


namespace blue_ball_higher_numbered_bin_l67_67001

noncomputable def probability_higher_numbered_bin :
  ℝ := sorry

theorem blue_ball_higher_numbered_bin :
  probability_higher_numbered_bin = 7 / 16 :=
sorry

end blue_ball_higher_numbered_bin_l67_67001


namespace jeremy_can_win_in_4_turns_l67_67552

noncomputable def game_winnable_in_4_turns (left right : ℕ) : Prop :=
∃ n1 n2 n3 n4 : ℕ,
  n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧
  (left + n1 + n2 + n3 + n4 = right * n1 * n2 * n3 * n4)

theorem jeremy_can_win_in_4_turns (left right : ℕ) (hleft : left = 17) (hright : right = 5) : game_winnable_in_4_turns left right :=
by
  rw [hleft, hright]
  sorry

end jeremy_can_win_in_4_turns_l67_67552


namespace ball_selection_count_l67_67991

theorem ball_selection_count :
  let even := ∑ (k : ℕ), (k % 2 = 0)
  let odd := ∑ (k : ℕ), (k % 2 = 1)
in
∑ (r g y : ℕ) in { (2005, even, _) ∪ (2005, _, odd) }, 1
= Nat.binomial 2007 2 - Nat.binomial 1004 2 :=
sorry

end ball_selection_count_l67_67991


namespace masha_numbers_unique_l67_67401

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l67_67401


namespace value_of_ak_l67_67999

noncomputable def Sn (n : ℕ) : ℤ := n^2 - 9 * n
noncomputable def a (n : ℕ) : ℤ := Sn n - Sn (n - 1)

theorem value_of_ak (k : ℕ) (hk : 5 < a k ∧ a k < 8) : a k = 6 := by
  sorry

end value_of_ak_l67_67999


namespace geometric_progression_common_point_l67_67775

theorem geometric_progression_common_point (a r : ℝ) :
  ∀ x y : ℝ, (a ≠ 0 ∧ x = 1 ∧ y = 0) ↔ (a * x + (a * r) * y = a * r^2) := by
  sorry

end geometric_progression_common_point_l67_67775


namespace distinct_real_roots_form_geometric_progression_eq_170_l67_67336

theorem distinct_real_roots_form_geometric_progression_eq_170 
  (a : ℝ) :
  (∃ (u : ℝ) (v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hv1 : |v| ≠ 1), 
  (16 * u^12 + (2 * a + 17) * u^6 * v^3 - a * u^9 * v - a * u^3 * v^9 + 16 = 0)) 
  → a = 170 :=
by sorry

end distinct_real_roots_form_geometric_progression_eq_170_l67_67336


namespace georgie_window_ways_l67_67150

theorem georgie_window_ways (n : Nat) (h : n = 8) :
  let ways := n * (n - 1)
  ways = 56 := by
  sorry

end georgie_window_ways_l67_67150


namespace numbers_masha_thought_l67_67402

noncomputable def distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions : ℕ → ℕ → Prop :=
λ a b, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (¬ (∃ x y, x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b ∧ (x ≠ a ∧ y ≠ b)))

theorem numbers_masha_thought (a b : ℕ) (h : distinct_natural_numbers_greater_than_eleven_and_sum_to_satisfy_conditions a b) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end numbers_masha_thought_l67_67402


namespace fraction_of_original_water_after_four_replacements_l67_67143

-- Define the initial condition and process
def initial_water_volume : ℚ := 10
def initial_alcohol_volume : ℚ := 10
def initial_total_volume : ℚ := initial_water_volume + initial_alcohol_volume

def fraction_remaining_after_removal (fraction_remaining : ℚ) : ℚ :=
  fraction_remaining * (initial_total_volume - 5) / initial_total_volume

-- Define the function counting the iterations process
def fraction_after_replacements (n : ℕ) (fraction_remaining : ℚ) : ℚ :=
  Nat.iterate fraction_remaining_after_removal n fraction_remaining

-- We have 4 replacements, start with 1 (because initially half of tank is water, 
-- fraction is 1 means we start with all original water)
def fraction_of_original_water_remaining : ℚ := (fraction_after_replacements 4 1)

-- Our goal in proof form
theorem fraction_of_original_water_after_four_replacements :
  fraction_of_original_water_remaining = (81 / 256) := by
  sorry

end fraction_of_original_water_after_four_replacements_l67_67143


namespace angle_sum_proof_l67_67382

theorem angle_sum_proof (A B C x y : ℝ) 
  (hA : A = 35) 
  (hB : B = 65) 
  (hC : C = 40) 
  (hx : x = 130 - C)
  (hy : y = 90 - A) :
  x + y = 140 := by
  sorry

end angle_sum_proof_l67_67382


namespace jonah_calories_burned_l67_67099

theorem jonah_calories_burned (rate hours1 hours2 : ℕ) (h_rate : rate = 30) (h_hours1 : hours1 = 2) (h_hours2 : hours2 = 5) :
  rate * hours2 - rate * hours1 = 90 :=
by {
  have h1 : rate * hours1 = 60, { rw [h_rate, h_hours1], norm_num },
  have h2 : rate * hours2 = 150, { rw [h_rate, h_hours2], norm_num },
  rw [h1, h2],
  norm_num,
  sorry
}

end jonah_calories_burned_l67_67099


namespace min_value_f_l67_67348

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x) + 200 * x^2

-- State the theorem to be proved
theorem min_value_f : ∃ (x : ℝ), (∀ y : ℝ, f y ≥ 33) ∧ f x = 33 := by
  sorry

end min_value_f_l67_67348


namespace not_approximately_equal_exp_l67_67343

noncomputable def multinomial_approximation (n k₁ k₂ k₃ k₄ k₅ : ℕ) : ℝ :=
  (n.factorial : ℝ) / ((k₁.factorial : ℝ) * (k₂.factorial : ℝ) * (k₃.factorial : ℝ) * (k₄.factorial : ℝ) * (k₅.factorial : ℝ))

theorem not_approximately_equal_exp (e : ℝ) (h1 : e > 0) :
  e ^ 2737 ≠ multinomial_approximation 1000 70 270 300 220 140 :=
by 
  sorry  

end not_approximately_equal_exp_l67_67343


namespace length_of_field_l67_67458

variable (l w : ℝ)

theorem length_of_field : 
  (l = 2 * w) ∧ (8 * 8 = 64) ∧ ((8 * 8) = (1 / 50) * l * w) → l = 80 :=
by
  sorry

end length_of_field_l67_67458


namespace find_b_minus_c_l67_67198

theorem find_b_minus_c (a b c : ℤ) (h : (x^2 + a * x - 3) * (x + 1) = x^3 + b * x^2 + c * x - 3) : b - c = 4 := by
  -- We would normally construct the proof here.
  sorry

end find_b_minus_c_l67_67198


namespace Jamie_owns_2_Maine_Coons_l67_67864

-- Definitions based on conditions
variables (Jamie_MaineCoons Gordon_MaineCoons Hawkeye_MaineCoons Jamie_Persians Gordon_Persians Hawkeye_Persians : ℕ)

-- Conditions
axiom Jamie_owns_4_Persians : Jamie_Persians = 4
axiom Gordon_owns_half_as_many_Persians_as_Jamie : Gordon_Persians = Jamie_Persians / 2
axiom Gordon_owns_one_more_Maine_Coon_than_Jamie : Gordon_MaineCoons = Jamie_MaineCoons + 1
axiom Hawkeye_owns_one_less_Maine_Coon_than_Gordon : Hawkeye_MaineCoons = Gordon_MaineCoons - 1
axiom Hawkeye_owns_no_Persian_cats : Hawkeye_Persians = 0
axiom total_number_of_cats_is_13 : Jamie_Persians + Jamie_MaineCoons + Gordon_Persians + Gordon_MaineCoons + Hawkeye_Persians + Hawkeye_MaineCoons = 13

-- Theorem statement
theorem Jamie_owns_2_Maine_Coons : Jamie_MaineCoons = 2 :=
by {
  -- Ideally, you would provide the proof here, stepping through algebraically as shown in the solution,
  -- but we are skipping the proof as specified in the instructions.
  sorry
}

end Jamie_owns_2_Maine_Coons_l67_67864


namespace total_goals_l67_67134

theorem total_goals (B M : ℕ) (hB : B = 4) (hM : M = 3 * B) : B + M = 16 := by
  sorry

end total_goals_l67_67134


namespace find_numbers_l67_67404

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l67_67404


namespace min_value_f_min_value_achieved_l67_67491

noncomputable def f (x y : ℝ) : ℝ :=
  (x^4 / y^4) + (y^4 / x^4) - (x^2 / y^2) - (y^2 / x^2) + (x / y) + (y / x)

theorem min_value_f :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → f x y ≥ 2 :=
sorry

theorem min_value_achieved :
  ∀ (x y : ℝ), (0 < x ∧ 0 < y) → (f x y = 2) ↔ (x = y) :=
sorry

end min_value_f_min_value_achieved_l67_67491


namespace exponent_sum_l67_67029

theorem exponent_sum : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end exponent_sum_l67_67029


namespace expected_deviation_10_greater_than_100_l67_67297

-- Definitions for deviations for 10 and 100 tosses
def deviation_10_tosses (m₁₀: ℕ) : ℝ := (m₁₀ / 10) - 0.5
def deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  ∑ i, αs i / 10

-- Definitions for absolute deviations
def abs_deviation_10_tosses (m₁₀: ℕ) : ℝ := abs (deviation_10_tosses m₁₀)
def abs_deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  abs (deviation_100_tosses m₁₀₀ αs)

-- Expected values of absolute deviations
def expected_abs_deviation_10_tosses : ℝ := sorry
def expected_abs_deviation_100_tosses : ℝ := sorry

theorem expected_deviation_10_greater_than_100 :
  expected_abs_deviation_10_tosses > expected_abs_deviation_100_tosses :=
sorry

end expected_deviation_10_greater_than_100_l67_67297


namespace exists_xyz_t_l67_67429

theorem exists_xyz_t (x y z t : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : t > 0) (h5 : x + y + z + t = 15) : ∃ y, y = 12 :=
by
  sorry

end exists_xyz_t_l67_67429


namespace science_club_election_l67_67644

theorem science_club_election :
  let total_candidates := 20
  let past_officers := 10
  let non_past_officers := total_candidates - past_officers
  let positions := 6
  let total_ways := Nat.choose total_candidates positions
  let no_past_officer_ways := Nat.choose non_past_officers positions
  let exactly_one_past_officer_ways := past_officers * Nat.choose non_past_officers (positions - 1)
  total_ways - no_past_officer_ways - exactly_one_past_officer_ways = 36030 := by
    sorry

end science_club_election_l67_67644


namespace distinct_square_roots_l67_67538

theorem distinct_square_roots (m : ℝ) (h : 2 * m - 4 ≠ 3 * m - 1) : ∃ n : ℝ, (2 * m - 4) * (2 * m - 4) = n ∧ (3 * m - 1) * (3 * m - 1) = n ∧ n = 4 :=
by
  sorry

end distinct_square_roots_l67_67538


namespace area_of_triangle_LMN_l67_67303

-- Define the vertices
def point := ℝ × ℝ
def L: point := (2, 3)
def M: point := (5, 1)
def N: point := (3, 5)

-- Shoelace formula for the area of a triangle
noncomputable def triangle_area (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2))

-- Statement to prove the area
theorem area_of_triangle_LMN : triangle_area L M N = 4 := by
  -- Proof would go here
  sorry

end area_of_triangle_LMN_l67_67303


namespace Kath_takes_3_friends_l67_67694

theorem Kath_takes_3_friends
  (total_paid: Int)
  (price_before_6: Int)
  (price_reduction: Int)
  (num_family_members: Int)
  (start_time: Int)
  (start_time_condition: start_time < 18)
  (total_payment_condition: total_paid = 30)
  (admission_cost_before_6: price_before_6 = 8 - price_reduction)
  (num_family_members_condition: num_family_members = 3):
  (total_paid / price_before_6 - num_family_members = 3) := 
by
  -- Since no proof is required, simply add sorry to skip the proof
  sorry

end Kath_takes_3_friends_l67_67694


namespace neg_ten_plus_three_l67_67114

theorem neg_ten_plus_three :
  -10 + 3 = -7 := by
  sorry

end neg_ten_plus_three_l67_67114


namespace sides_of_triangle_inequality_l67_67905

theorem sides_of_triangle_inequality (a b c : ℝ) (h : a + b > c) : a + b > c := 
by 
  exact h

end sides_of_triangle_inequality_l67_67905


namespace circle_equation_l67_67473

theorem circle_equation 
    (a : ℝ)
    (x y : ℝ)
    (tangent_lines : x + y = 0 ∧ x + y = 4)
    (center_line : x - y = a)
    (center_point : ∃ (a : ℝ), x = a ∧ y = a) :
    ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 :=
by
  sorry

end circle_equation_l67_67473


namespace work_done_in_11_days_l67_67137

-- Given conditions as definitions
def a_days := 24
def b_days := 30
def c_days := 40
def combined_work_rate := (1 / a_days) + (1 / b_days) + (1 / c_days)
def days_c_leaves_before_completion := 4

-- Statement of the problem to be proved
theorem work_done_in_11_days :
  ∃ (D : ℕ), D = 11 ∧ ((D - days_c_leaves_before_completion) * combined_work_rate) + 
  (days_c_leaves_before_completion * ((1 / a_days) + (1 / b_days))) = 1 :=
sorry

end work_done_in_11_days_l67_67137


namespace muffin_banana_costs_l67_67638

variable (m b : ℕ) -- Using natural numbers for non-negativity

theorem muffin_banana_costs (h : 3 * (3 * m + 5 * b) = 4 * m + 10 * b) : m = b :=
by
  sorry

end muffin_banana_costs_l67_67638


namespace fraction_meaningful_iff_l67_67838

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l67_67838


namespace center_of_circle_l67_67253

-- Definition of the main condition: the given circle equation
def circle_equation (x y : ℝ) := x^2 + y^2 = 10 * x - 4 * y + 14

-- Statement to prove: that x + y = 3 when (x, y) is the center of the circle described by circle_equation
theorem center_of_circle {x y : ℝ} (h : circle_equation x y) : x + y = 3 := 
by 
  sorry

end center_of_circle_l67_67253


namespace probability_of_less_than_5_is_one_half_l67_67750

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l67_67750


namespace unique_B_for_A47B_divisible_by_7_l67_67339

-- Define the conditions
def A : ℕ := 4

-- Define the main proof problem statement
theorem unique_B_for_A47B_divisible_by_7 : 
  ∃! B : ℕ, B ≤ 9 ∧ (100 * A + 70 + B) % 7 = 0 :=
        sorry

end unique_B_for_A47B_divisible_by_7_l67_67339


namespace cos_triple_angle_l67_67823

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 :=
by
  sorry

end cos_triple_angle_l67_67823


namespace parity_equivalence_l67_67390

def p_q_parity_condition (p q : ℕ) : Prop :=
  (p^3 - q^3) % 2 = 0 ↔ (p + q) % 2 = 0

theorem parity_equivalence (p q : ℕ) : p_q_parity_condition p q :=
by sorry

end parity_equivalence_l67_67390


namespace perpendicular_lines_l67_67375

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y - a = 0) → (a * x - (2 * a - 3) * y - 1 = 0) → 
    (∀ x y : ℝ, ( -1 / a ) * ( -a / (2 * a - 3)) = 1 )) → a = 3 := 
by
  sorry

end perpendicular_lines_l67_67375


namespace car_C_has_highest_average_speed_l67_67273

-- Define the distances traveled by each car
def distance_car_A_1st_hour := 140
def distance_car_A_2nd_hour := 130
def distance_car_A_3rd_hour := 120

def distance_car_B_1st_hour := 170
def distance_car_B_2nd_hour := 90
def distance_car_B_3rd_hour := 130

def distance_car_C_1st_hour := 120
def distance_car_C_2nd_hour := 140
def distance_car_C_3rd_hour := 150

-- Define the total distance and average speed calculations
def total_distance_car_A := distance_car_A_1st_hour + distance_car_A_2nd_hour + distance_car_A_3rd_hour
def total_distance_car_B := distance_car_B_1st_hour + distance_car_B_2nd_hour + distance_car_B_3rd_hour
def total_distance_car_C := distance_car_C_1st_hour + distance_car_C_2nd_hour + distance_car_C_3rd_hour

def total_time := 3

def average_speed_car_A := total_distance_car_A / total_time
def average_speed_car_B := total_distance_car_B / total_time
def average_speed_car_C := total_distance_car_C / total_time

-- Lean proof statement
theorem car_C_has_highest_average_speed :
  average_speed_car_C > average_speed_car_A ∧ average_speed_car_C > average_speed_car_B :=
by
  sorry

end car_C_has_highest_average_speed_l67_67273


namespace ordinary_eq_from_param_eq_l67_67582

theorem ordinary_eq_from_param_eq (α : ℝ) :
  (∃ (x y : ℝ), x = 3 * Real.cos α + 1 ∧ y = - Real.cos α → x + 3 * y - 1 = 0 ∧ (-2 ≤ x ∧ x ≤ 4)) := 
sorry

end ordinary_eq_from_param_eq_l67_67582


namespace bruce_michael_total_goals_l67_67131

theorem bruce_michael_total_goals (bruce_goals : ℕ) (michael_goals : ℕ) 
  (h₁ : bruce_goals = 4) (h₂ : michael_goals = 3 * bruce_goals) : bruce_goals + michael_goals = 16 :=
by sorry

end bruce_michael_total_goals_l67_67131


namespace sixth_grade_students_total_l67_67266

noncomputable def total_students (x y : ℕ) : ℕ := x + y

theorem sixth_grade_students_total (x y : ℕ) 
(h1 : x + (1 / 3) * y = 105) 
(h2 : y + (1 / 2) * x = 105) 
: total_students x y = 147 := 
by
  sorry

end sixth_grade_students_total_l67_67266


namespace domain_of_sqrt_tan_x_minus_sqrt_3_l67_67988

noncomputable def domain_of_function : Set Real :=
  {x | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2}

theorem domain_of_sqrt_tan_x_minus_sqrt_3 :
  { x : Real | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2 } = domain_of_function :=
by
  sorry

end domain_of_sqrt_tan_x_minus_sqrt_3_l67_67988


namespace isosceles_trapezoid_area_l67_67517

theorem isosceles_trapezoid_area (m h : ℝ) (hg : h = 3) (mg : m = 15) : 
  (m * h = 45) :=
by
  simp [hg, mg]
  sorry

end isosceles_trapezoid_area_l67_67517


namespace exists_prime_not_in_list_l67_67413

open Nat

theorem exists_prime_not_in_list (l : List ℕ) (h : ∀ p ∈ l, Prime p) : 
  ∃ q, Prime q ∧ q ∉ l := 
sorry

end exists_prime_not_in_list_l67_67413


namespace polygon_angles_change_l67_67840

theorem polygon_angles_change (n : ℕ) :
  let initial_sum_interior := (n - 2) * 180
  let initial_sum_exterior := 360
  let new_sum_interior := (n + 2 - 2) * 180
  let new_sum_exterior := 360
  new_sum_exterior = initial_sum_exterior ∧ new_sum_interior - initial_sum_interior = 360 :=
by
  sorry

end polygon_angles_change_l67_67840


namespace kitchen_upgrade_cost_l67_67593

def total_kitchen_upgrade_cost (num_knobs : ℕ) (cost_per_knob : ℝ) (num_pulls : ℕ) (cost_per_pull : ℝ) : ℝ :=
  (num_knobs * cost_per_knob) + (num_pulls * cost_per_pull)

theorem kitchen_upgrade_cost : total_kitchen_upgrade_cost 18 2.50 8 4.00 = 77.00 :=
  by
    sorry

end kitchen_upgrade_cost_l67_67593


namespace probability_of_rolling_less_than_5_l67_67745

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l67_67745


namespace total_population_correct_l67_67072

-- Given conditions
def number_of_cities : ℕ := 25
def average_population : ℕ := 3800

-- Statement to prove
theorem total_population_correct : number_of_cities * average_population = 95000 :=
by
  sorry

end total_population_correct_l67_67072


namespace fewer_cubes_needed_l67_67105

variable (cubeVolume : ℕ) (length : ℕ) (width : ℕ) (depth : ℕ) (TVolume : ℕ)

theorem fewer_cubes_needed : 
  cubeVolume = 5 → 
  length = 7 → 
  width = 7 → 
  depth = 6 → 
  TVolume = 3 → 
  (length * width * depth - TVolume = 291) :=
by
  intros hc hl hw hd ht
  sorry

end fewer_cubes_needed_l67_67105


namespace boat_upstream_time_is_1_5_hours_l67_67469

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l67_67469


namespace sum_of_two_numbers_l67_67112

variable {x y : ℝ}

theorem sum_of_two_numbers (h1 : x * y = 120) (h2 : x^2 + y^2 = 289) : x + y = 22 :=
sorry

end sum_of_two_numbers_l67_67112


namespace sequence_m_l67_67902

noncomputable def a (n : ℕ) : ℕ :=
  if n = 0 then 0  -- We usually start sequences from n = 1; hence, a_0 is irrelevant
  else (n * n) - n + 1

theorem sequence_m (m : ℕ) (h_positive : m > 0) (h_bound : 43 < a m ∧ a m < 73) : m = 8 :=
by {
  sorry
}

end sequence_m_l67_67902


namespace find_price_per_craft_l67_67530

-- Definitions based on conditions
def price_per_craft (x : ℝ) : Prop :=
  let crafts_sold := 3
  let extra_money := 7
  let deposit := 18
  let remaining_money := 25
  let total_before_deposit := 43
  3 * x + extra_money = total_before_deposit

-- Statement of the problem to prove x = 12 given conditions
theorem find_price_per_craft : ∃ x : ℝ, price_per_craft x ∧ x = 12 :=
by
  sorry

end find_price_per_craft_l67_67530


namespace integer_sequence_unique_l67_67784

theorem integer_sequence_unique (a : ℕ → ℤ) :
  (∀ n : ℕ, ∃ p q : ℕ, p ≠ q ∧ a p > 0 ∧ a q < 0) ∧
  (∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → a i % (n : ℤ) ≠ a j % (n : ℤ))
  → ∀ x : ℤ, ∃! i : ℕ, a i = x :=
by
  sorry

end integer_sequence_unique_l67_67784


namespace probability_of_rolling_less_than_five_l67_67740

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l67_67740


namespace triangle_perimeter_l67_67261

theorem triangle_perimeter (r A : ℝ) (p : ℝ)
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) : 
  p = 20 :=
by 
  sorry

end triangle_perimeter_l67_67261


namespace problem_f_neg2_equals_2_l67_67812

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem_f_neg2_equals_2 (f : ℝ → ℝ) (b : ℝ) 
  (h_odd : is_odd_function f)
  (h_def : ∀ x : ℝ, x ≥ 0 → f x = x^2 - 3 * x + b) 
  (h_b : b = 0) :
  f (-2) = 2 :=
by
  sorry

end problem_f_neg2_equals_2_l67_67812


namespace parabola_symmetry_l67_67051

-- Define the function f as explained in the problem
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Lean theorem to prove the inequality based on given conditions
theorem parabola_symmetry (b c : ℝ) (h : ∀ t : ℝ, f (3 + t) b c = f (3 - t) b c) :
  f 3 b c < f 1 b c ∧ f 1 b c < f 6 b c :=
by
  sorry

end parabola_symmetry_l67_67051


namespace certain_event_l67_67921

theorem certain_event (a : ℝ) : a^2 ≥ 0 := 
sorry

end certain_event_l67_67921


namespace probability_red_blue_yellow_l67_67723

-- Define the probabilities for white, green, and black marbles
def p_white : ℚ := 1/4
def p_green : ℚ := 1/6
def p_black : ℚ := 1/8

-- Define the problem: calculating the probability of drawing a red, blue, or yellow marble
theorem probability_red_blue_yellow : 
  p_white = 1/4 → p_green = 1/6 → p_black = 1/8 →
  (1 - (p_white + p_green + p_black)) = 11/24 := 
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end probability_red_blue_yellow_l67_67723


namespace problem_solution_l67_67655

def diamond (x y k : ℝ) : ℝ := x^2 - k * y

theorem problem_solution (h : ℝ) (k : ℝ) (hk : k = 3) : 
  diamond h (diamond h h k) k = -2 * h^2 + 9 * h :=
by
  rw [hk, diamond, diamond]
  sorry

end problem_solution_l67_67655


namespace function_pair_solution_l67_67030

-- Define the conditions for f and g
variables (f g : ℝ → ℝ)

-- Define the main hypothesis
def main_hypothesis : Prop := 
∀ (x y : ℝ), 
  x ≠ 0 → y ≠ 0 → 
  f (x + y) = g (1/x + 1/y) * (x * y) ^ 2008

-- The theorem that proves f and g are of the given form
theorem function_pair_solution (c : ℝ) (h : main_hypothesis f g) : 
  (∀ x, f x = c * x ^ 2008) ∧ 
  (∀ x, g x = c * x ^ 2008) :=
sorry

end function_pair_solution_l67_67030


namespace pam_total_apples_l67_67566

theorem pam_total_apples (pam_bags : ℕ) (gerald_apples_per_bag : ℕ) (pam_apples_per_bag : ℕ) (gerald_bags_for_pam_bag : ℕ) :
  pam_bags = 10 →
  gerald_apples_per_bag = 40 →
  gerald_bags_for_pam_bag = 3 →
  pam_apples_per_bag = gerald_bags_for_pam_bag * gerald_apples_per_bag →
  pam_bags * pam_apples_per_bag = 1200 := 
by
  intros h_pam_bags h_gerald_apples h_gerald_bags_for_pam h_pam_apples
  rw [h_pam_bags, h_gerald_apples, h_gerald_bags_for_pam, h_pam_apples]
  calc
    10 * (3 * 40) = 10 * 120 : by rfl
               ... = 1200 : by rfl
  done

end pam_total_apples_l67_67566


namespace full_day_students_l67_67956

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_l67_67956


namespace expected_absolute_deviation_greater_in_10_tosses_l67_67286

variable {n m : ℕ}

def frequency_of_heads (m n : ℕ) : ℚ := m / n

def deviation_from_probability (m n : ℕ) : ℚ :=
  m / n - 0.5

def absolute_deviation (m n : ℕ) : ℚ :=
  |m / n - 0.5|

noncomputable def expectation_absolute_deviation (n : ℕ) : ℚ := 
  if n = 10 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 10 tosses
    sorry 
  else if n = 100 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 100 tosses
    sorry 
  else 0

theorem expected_absolute_deviation_greater_in_10_tosses (a b : ℚ) :
  expectation_absolute_deviation 10 > expectation_absolute_deviation 100 :=
by sorry

end expected_absolute_deviation_greater_in_10_tosses_l67_67286


namespace total_cost_898_8_l67_67457

theorem total_cost_898_8 :
  ∀ (M R F : ℕ → ℝ), 
    (10 * M 1 = 24 * R 1) →
    (6 * F 1 = 2 * R 1) →
    (F 1 = 21) →
    (4 * M 1 + 3 * R 1 + 5 * F 1 = 898.8) :=
by
  intros M R F h1 h2 h3
  sorry

end total_cost_898_8_l67_67457


namespace initial_population_l67_67264

theorem initial_population (P : ℝ) (h : (0.9 : ℝ)^2 * P = 4860) : P = 6000 :=
by
  sorry

end initial_population_l67_67264


namespace factorize_expression_triangle_is_isosceles_l67_67783

-- Define the first problem: Factorize the expression.
theorem factorize_expression (a b : ℝ) : a^2 - 4 * a - b^2 + 4 = (a + b - 2) * (a - b - 2) := 
by
  sorry

-- Define the second problem: Determine the shape of the triangle.
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : a = b ∨ a = c :=
by
  sorry

end factorize_expression_triangle_is_isosceles_l67_67783


namespace cylinder_cut_is_cylinder_l67_67488

-- Define what it means to be a cylinder
structure Cylinder (r h : ℝ) : Prop :=
(r_pos : r > 0)
(h_pos : h > 0)

-- Define the condition of cutting a cylinder with two parallel planes
def cut_by_parallel_planes (c : Cylinder r h) (d : ℝ) : Prop :=
d > 0 ∧ d < h

-- Prove that the part between the parallel planes is still a cylinder
theorem cylinder_cut_is_cylinder (r h d : ℝ) (c : Cylinder r h) (H : cut_by_parallel_planes c d) :
  ∃ r' h', Cylinder r' h' :=
sorry

end cylinder_cut_is_cylinder_l67_67488


namespace find_code_l67_67951

theorem find_code (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 11 * (A + B + C) = 242) :
  A = 5 ∧ B = 8 ∧ C = 9 ∨ A = 5 ∧ B = 9 ∧ C = 8 :=
by
  sorry

end find_code_l67_67951


namespace purchase_in_april_l67_67895

namespace FamilySavings

def monthly_income : ℕ := 150000
def monthly_expenses : ℕ := 115000
def initial_savings : ℕ := 45000
def furniture_cost : ℕ := 127000

noncomputable def monthly_savings : ℕ := monthly_income - monthly_expenses
noncomputable def additional_amount_needed : ℕ := furniture_cost - initial_savings
noncomputable def months_required : ℕ := (additional_amount_needed + monthly_savings - 1) / monthly_savings  -- ceiling division

theorem purchase_in_april : months_required = 3 :=
by
  -- Proof goes here
  sorry

end FamilySavings

end purchase_in_april_l67_67895


namespace james_matches_l67_67860

theorem james_matches (dozen_boxes : ℕ) (matches_per_box : ℕ) (dozen_value : ℕ) 
  (h1 : dozen_boxes = 5) (h2 : matches_per_box = 20) (h3 : dozen_value = 12) :
  dozen_boxes * dozen_value * matches_per_box = 1200 :=
by
  rw [h1, h2, h3]
  calc
    5 * 12 * 20 = 60 * 20 := by norm_num
    ... = 1200 := by norm_num

end james_matches_l67_67860


namespace product_mod5_is_zero_l67_67328

theorem product_mod5_is_zero :
  (2023 * 2024 * 2025 * 2026) % 5 = 0 :=
by
  sorry

end product_mod5_is_zero_l67_67328


namespace cos_3theta_l67_67826

theorem cos_3theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_3theta_l67_67826


namespace chess_tournament_l67_67438

def number_of_players := 30

def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

theorem chess_tournament : total_games number_of_players = 435 := by
  sorry

end chess_tournament_l67_67438


namespace proposition_B_correct_l67_67127

theorem proposition_B_correct (a b c : ℝ) (hc : c ≠ 0) : ac^2 > b * c^2 → a > b := sorry

end proposition_B_correct_l67_67127


namespace arithmetic_sequence_a8_l67_67684

theorem arithmetic_sequence_a8 (a : ℕ → ℤ) (d : ℤ) :
  a 2 = 4 → a 4 = 2 → a 8 = -2 :=
by intros ha2 ha4
   sorry

end arithmetic_sequence_a8_l67_67684


namespace contrapositive_of_happy_people_possess_it_l67_67106

variable (P Q : Prop)

theorem contrapositive_of_happy_people_possess_it
  (h : P → Q) : ¬ Q → ¬ P := by
  intro hq
  intro p
  apply hq
  apply h
  exact p

#check contrapositive_of_happy_people_possess_it

end contrapositive_of_happy_people_possess_it_l67_67106


namespace cos_triple_angle_l67_67824

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 :=
by
  sorry

end cos_triple_angle_l67_67824


namespace range_of_r_l67_67377

theorem range_of_r (r : ℝ) (h_r : r > 0) :
  let M := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 4}
  let N := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ r^2}
  (∀ p, p ∈ N → p ∈ M) → 0 < r ∧ r ≤ 2 - Real.sqrt 2 :=
by
  sorry

end range_of_r_l67_67377


namespace rectangle_side_lengths_l67_67111

theorem rectangle_side_lengths:
  ∃ x : ℝ, ∃ y : ℝ, (2 * (x + y) * 2 = x * y) ∧ (y = x + 3) ∧ (x > 0) ∧ (y > 0) ∧ x = 8 ∧ y = 11 :=
by
  sorry

end rectangle_side_lengths_l67_67111


namespace Jim_runs_total_distance_l67_67553

-- Definitions based on the conditions
def miles_day_1 := 5
def miles_day_31 := 10
def miles_day_61 := 20

def days_period := 30

-- Mathematical statement to prove
theorem Jim_runs_total_distance :
  let total_distance := 
    (miles_day_1 * days_period) + 
    (miles_day_31 * days_period) + 
    (miles_day_61 * days_period)
  total_distance = 1050 := by
  sorry

end Jim_runs_total_distance_l67_67553


namespace g_iterated_six_times_is_2_l67_67696

def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem g_iterated_six_times_is_2 : g (g (g (g (g (g 2))))) = 2 := 
by 
  sorry

end g_iterated_six_times_is_2_l67_67696


namespace boat_upstream_time_is_1_5_hours_l67_67468

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l67_67468


namespace linear_function_quadrants_passing_through_l67_67578

theorem linear_function_quadrants_passing_through :
  ∀ (x : ℝ) (y : ℝ), (y = 2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end linear_function_quadrants_passing_through_l67_67578


namespace area_of_original_square_l67_67185

theorem area_of_original_square 
  (x : ℝ) 
  (h0 : x * (x - 3) = 40) 
  (h1 : 0 < x) : 
  x ^ 2 = 64 := 
sorry

end area_of_original_square_l67_67185


namespace find_QS_l67_67573

theorem find_QS (cosR : ℝ) (RS QR QS : ℝ) (h1 : cosR = 3 / 5) (h2 : RS = 10) (h3 : cosR = QR / RS) (h4: QR ^ 2 + QS ^ 2 = RS ^ 2) : QS = 8 :=
by 
  sorry

end find_QS_l67_67573


namespace floor_equation_solution_l67_67969

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l67_67969


namespace line_equation_cartesian_circle_equation_cartesian_l67_67687

theorem line_equation_cartesian (t : ℝ) (x y : ℝ) : 
  (x = 3 - (Real.sqrt 2 / 2) * t ∧ y = Real.sqrt 5 + (Real.sqrt 2 / 2) * t) -> 
  y = -2 * x + 6 + Real.sqrt 5 :=
sorry

theorem circle_equation_cartesian (ρ θ x y : ℝ) : 
  (ρ = 2 * Real.sqrt 5 * Real.sin θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) -> 
  x^2 = 0 :=
sorry

end line_equation_cartesian_circle_equation_cartesian_l67_67687


namespace flight_duration_problem_l67_67233

def problem_conditions : Prop :=
  let la_departure_pst := (7, 15) -- 7:15 AM PST
  let ny_arrival_est := (17, 40) -- 5:40 PM EST (17:40 in 24-hour format)
  let time_difference := 3 -- Hours difference (EST is 3 hours ahead of PST)
  let dst_adjustment := 1 -- Daylight saving time adjustment in hours
  ∃ (h m : ℕ), (0 < m ∧ m < 60) ∧ ((h = 7 ∧ m = 25) ∧ (h + m = 32))

theorem flight_duration_problem :
  problem_conditions :=
by
  -- Placeholder for the proof that shows the conditions established above imply h + m = 32
  sorry

end flight_duration_problem_l67_67233


namespace cos_difference_l67_67199

theorem cos_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1 / 2) 
  (h2 : Real.cos A + Real.cos B = 3 / 2) : 
  Real.cos (A - B) = 1 / 4 :=
by
  sorry

end cos_difference_l67_67199


namespace relay_race_total_time_l67_67182

-- Definitions based on the problem conditions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25

-- Problem statement
theorem relay_race_total_time : 
  athlete1_time + athlete2_time + athlete3_time + athlete4_time = 200 := 
by 
  sorry

end relay_race_total_time_l67_67182


namespace simplify_expression_l67_67571

variable (x : Int)

theorem simplify_expression : 3 * x + 5 * x + 7 * x = 15 * x :=
  by
  sorry

end simplify_expression_l67_67571


namespace full_day_students_count_l67_67953

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end full_day_students_count_l67_67953


namespace segment_length_l67_67092

theorem segment_length (A B C : ℝ) (hAB : abs (A - B) = 3) (hBC : abs (B - C) = 5) :
  abs (A - C) = 2 ∨ abs (A - C) = 8 := by
  sorry

end segment_length_l67_67092


namespace power_sum_l67_67340

theorem power_sum :
  (-3)^3 + (-3)^2 + (-3) + 3 + 3^2 + 3^3 = 18 :=
by
  sorry

end power_sum_l67_67340


namespace hyperbola_lattice_points_count_l67_67769

def is_lattice_point (x y : ℤ) : Prop :=
  x^2 - 2 * y^2 = 2000^2

def count_lattice_points (points : List (ℤ × ℤ)) : ℕ :=
  points.length

theorem hyperbola_lattice_points_count : count_lattice_points [(2000, 0), (-2000, 0)] = 2 :=
by
  sorry

end hyperbola_lattice_points_count_l67_67769


namespace reduced_price_per_kg_of_oil_l67_67924

theorem reduced_price_per_kg_of_oil
  (P : ℝ)
  (h : (1000 / (0.75 * P) - 1000 / P = 5)) :
  0.75 * (1000 / 15) = 50 := 
sorry

end reduced_price_per_kg_of_oil_l67_67924


namespace largest_possible_green_cards_l67_67145

-- Definitions of conditions
variables (g y t : ℕ)

-- Defining the total number of cards t
def total_cards := g + y

-- Condition on maximum number of cards
def max_total_cards := total_cards g y ≤ 2209

-- Probability condition for drawing 3 same-color cards
def probability_condition := 
  g * (g - 1) * (g - 2) + y * (y - 1) * (y - 2) 
  = (1 : ℚ) / 3 * t * (t - 1) * (t - 2)

-- Proving the largest possible number of green cards
theorem largest_possible_green_cards
  (h1 : total_cards g y = t)
  (h2 : max_total_cards g y)
  (h3 : probability_condition g y t) :
  g ≤ 1092 :=
sorry

end largest_possible_green_cards_l67_67145


namespace calories_difference_l67_67100

def calories_burnt (hours : ℕ) : ℕ := 30 * hours

theorem calories_difference :
  calories_burnt 5 - calories_burnt 2 = 90 :=
by
  sorry

end calories_difference_l67_67100


namespace perimeter_of_triangle_l67_67262

theorem perimeter_of_triangle
  {r A P : ℝ} (hr : r = 2.5) (hA : A = 25) :
  P = 20 :=
by
  sorry

end perimeter_of_triangle_l67_67262


namespace inequality_solution_l67_67903

open Set

theorem inequality_solution :
  {x : ℝ | |x + 1| - 2 > 0} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1} :=
by
  sorry

end inequality_solution_l67_67903


namespace min_air_routes_l67_67067

theorem min_air_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) : 
  a + b + c ≥ 21 :=
sorry

end min_air_routes_l67_67067


namespace rectangle_longer_side_length_l67_67939

theorem rectangle_longer_side_length (r : ℝ) (h1 : r = 4) 
  (h2 : ∃ w l, w * l = 2 * (π * r^2) ∧ w = 2 * r) : 
  ∃ l, l = 4 * π :=
by 
  obtain ⟨w, l, h_area, h_shorter_side⟩ := h2
  sorry

end rectangle_longer_side_length_l67_67939


namespace ratio_amy_jeremy_l67_67641

variable (Amy Chris Jeremy : ℕ)

theorem ratio_amy_jeremy (h1 : Amy + Jeremy + Chris = 132) (h2 : Jeremy = 66) (h3 : Chris = 2 * Amy) : 
  Amy / Jeremy = 1 / 3 :=
by
  sorry

end ratio_amy_jeremy_l67_67641


namespace common_chord_condition_l67_67580

theorem common_chord_condition 
    (h d1 d2 : ℝ) (C1 C2 D1 D2 : ℝ) 
    (hyp_len : (C1 * D1 = C2 * D2)) : 
    (C1 * D1 = C2 * D2) ↔ (1 / h^2 = 1 / d1^2 + 1 / d2^2) :=
by
  sorry

end common_chord_condition_l67_67580


namespace find_sum_of_abc_l67_67123

noncomputable def m (a b c : ℕ) : ℝ := a - b * Real.sqrt c

theorem find_sum_of_abc (a b c : ℕ) (ha : ¬ (c % 2 = 0) ∧ ∀ p : ℕ, Prime p → ¬ p * p ∣ c) 
  (hprob : ((30 - m a b c) ^ 2 / 30 ^ 2 = 0.75)) : a + b + c = 48 := 
by
  sorry

end find_sum_of_abc_l67_67123


namespace sum_of_solutions_l67_67686

theorem sum_of_solutions (x y : ℝ) (h1 : y = 9) (h2 : x^2 + y^2 = 225) : 2 * x = 0 :=
by
  sorry

end sum_of_solutions_l67_67686


namespace initial_pepper_amount_l67_67959
-- Import the necessary libraries.

-- Declare the problem as a theorem.
theorem initial_pepper_amount (used left : ℝ) (h₁ : used = 0.16) (h₂ : left = 0.09) :
  used + left = 0.25 :=
by
  -- The proof is not required here.
  sorry

end initial_pepper_amount_l67_67959


namespace floor_eq_solution_l67_67979

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l67_67979


namespace solution_set_l67_67978

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l67_67978


namespace continuity_at_x0_l67_67141

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 4
def x0 := 3

theorem continuity_at_x0 : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - x0| < δ → |f x - f x0| < ε :=
by
  sorry

end continuity_at_x0_l67_67141


namespace eq_of_nonzero_real_x_l67_67338

theorem eq_of_nonzero_real_x (x : ℝ) (hx : x ≠ 0) (a b : ℝ) (ha : a = 9) (hb : b = 18) :
  ((a * x) ^ 10 = (b * x) ^ 5) → x = 2 / 9 :=
by
  sorry

end eq_of_nonzero_real_x_l67_67338


namespace mike_total_hours_worked_l67_67856

-- Define the conditions
def time_to_wash_car := 10
def time_to_change_oil := 15
def time_to_change_tires := 30

def number_of_cars_washed := 9
def number_of_oil_changes := 6
def number_of_tire_changes := 2

-- Define the conversion factor
def minutes_per_hour := 60

-- Prove that the total time worked equals 4 hours
theorem mike_total_hours_worked : 
  (number_of_cars_washed * time_to_wash_car + 
   number_of_oil_changes * time_to_change_oil + 
   number_of_tire_changes * time_to_change_tires) / minutes_per_hour = 4 := by
  sorry

end mike_total_hours_worked_l67_67856


namespace parallelogram_base_length_l67_67987

theorem parallelogram_base_length (Area Height : ℝ) (h1 : Area = 216) (h2 : Height = 18) : 
  Area / Height = 12 := 
by 
  sorry

end parallelogram_base_length_l67_67987


namespace find_value_of_expression_l67_67801

theorem find_value_of_expression (a : ℝ) (h : a^2 - a - 1 = 0) : a^3 - a^2 - a + 2023 = 2023 :=
by
  sorry

end find_value_of_expression_l67_67801


namespace remainder_when_3_pow_305_div_13_l67_67279

theorem remainder_when_3_pow_305_div_13 :
  (3 ^ 305) % 13 = 9 := 
by {
  sorry
}

end remainder_when_3_pow_305_div_13_l67_67279


namespace age_ratio_in_years_l67_67596

theorem age_ratio_in_years (p c x : ℕ) 
  (H1 : p - 2 = 3 * (c - 2)) 
  (H2 : p - 4 = 4 * (c - 4)) 
  (H3 : (p + x) / (c + x) = 2) : 
  x = 4 :=
sorry

end age_ratio_in_years_l67_67596


namespace solve_equation_l67_67755

theorem solve_equation (x : ℝ) : (3 * x - 2 * (10 - x) = 5) → x = 5 :=
by {
  sorry
}

end solve_equation_l67_67755


namespace full_day_students_l67_67957

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_l67_67957


namespace mary_flour_requirement_l67_67561

theorem mary_flour_requirement (total_flour : ℕ) (added_flour : ℕ) (remaining_flour : ℕ) 
  (h1 : total_flour = 7) 
  (h2 : added_flour = 2) 
  (h3 : remaining_flour = total_flour - added_flour) : 
  remaining_flour = 5 :=
sorry

end mary_flour_requirement_l67_67561


namespace average_first_21_multiples_of_6_l67_67917

-- Define the arithmetic sequence and its conditions.
def arithmetic_sequence (a1 d : ℕ) (n : ℕ) := a1 + (n - 1) * d

-- Define the problem statement.
theorem average_first_21_multiples_of_6 :
  let a1 := 6
  let d := 6
  let n := 21
  let an := arithmetic_sequence a1 d n
  (a1 + an) / 2 = 66 := by
  sorry

end average_first_21_multiples_of_6_l67_67917


namespace expected_deviation_10_greater_than_100_l67_67296

-- Definitions for deviations for 10 and 100 tosses
def deviation_10_tosses (m₁₀: ℕ) : ℝ := (m₁₀ / 10) - 0.5
def deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  ∑ i, αs i / 10

-- Definitions for absolute deviations
def abs_deviation_10_tosses (m₁₀: ℕ) : ℝ := abs (deviation_10_tosses m₁₀)
def abs_deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  abs (deviation_100_tosses m₁₀₀ αs)

-- Expected values of absolute deviations
def expected_abs_deviation_10_tosses : ℝ := sorry
def expected_abs_deviation_100_tosses : ℝ := sorry

theorem expected_deviation_10_greater_than_100 :
  expected_abs_deviation_10_tosses > expected_abs_deviation_100_tosses :=
sorry

end expected_deviation_10_greater_than_100_l67_67296


namespace average_marks_l67_67117

theorem average_marks
  (M P C : ℕ)
  (h1 : M + P = 70)
  (h2 : C = P + 20) :
  (M + C) / 2 = 45 :=
sorry

end average_marks_l67_67117


namespace rainbow_preschool_full_day_students_l67_67955

theorem rainbow_preschool_full_day_students (total_students : ℕ) (half_day_percent : ℝ)
  (h1 : total_students = 80) (h2 : half_day_percent = 0.25) :
  (total_students * (1 - half_day_percent)).to_nat = 60 :=
by
  -- Transform percentage to a fraction
  let fraction_full_day := 1 - half_day_percent
  -- Calculate full-day students
  have h_full_day_students : ℝ := total_students * fraction_full_day
  -- Convert to natural number
  exact (floor h_full_day_students).to_nat = 60

end rainbow_preschool_full_day_students_l67_67955


namespace rate_of_stream_equation_l67_67617

theorem rate_of_stream_equation 
  (v : ℝ) 
  (boat_speed : ℝ) 
  (travel_time : ℝ) 
  (distance : ℝ)
  (h_boat_speed : boat_speed = 16)
  (h_travel_time : travel_time = 5)
  (h_distance : distance = 105)
  (h_equation : distance = (boat_speed + v) * travel_time) : v = 5 :=
by 
  sorry

end rate_of_stream_equation_l67_67617


namespace find_x_l67_67916

theorem find_x (x : ℝ) (h : (0.4 + x) / 2 = 0.2025) : x = 0.005 :=
by
  sorry

end find_x_l67_67916


namespace power_of_power_l67_67602

theorem power_of_power {a : ℝ} : (a^2)^3 = a^6 := 
by
  sorry

end power_of_power_l67_67602


namespace sum_congruent_mod_9_l67_67022

theorem sum_congruent_mod_9 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by 
  -- Proof steps here
  sorry

end sum_congruent_mod_9_l67_67022


namespace combined_tax_rate_33_33_l67_67165

-- Define the necessary conditions
def mork_tax_rate : ℝ := 0.40
def mindy_tax_rate : ℝ := 0.30
def mindy_income_ratio : ℝ := 2.0

-- Main theorem statement
theorem combined_tax_rate_33_33 :
  ∀ (X : ℝ), ((mork_tax_rate * X + mindy_income_ratio * mindy_tax_rate * X) / (X + mindy_income_ratio * X) * 100) = 100 / 3 :=
by
  intro X
  sorry

end combined_tax_rate_33_33_l67_67165


namespace traveled_distance_is_9_l67_67630

-- Let x be the usual speed in mph
variable (x : ℝ)
-- Let t be the usual time in hours
variable (t : ℝ)

-- Conditions
axiom condition1 : x * t = (x + 0.5) * (3 / 4 * t)
axiom condition2 : x * t = (x - 0.5) * (t + 3)

-- The journey distance d in miles
def distance_in_miles : ℝ := x * t

-- We can now state the theorem to prove that the distance he traveled is 9 miles
theorem traveled_distance_is_9 : distance_in_miles x t = 9 := by
  sorry

end traveled_distance_is_9_l67_67630


namespace problem_I_number_of_zeros_problem_II_inequality_l67_67370

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x * Real.exp 1 - 1

theorem problem_I_number_of_zeros : 
  ∃! (x1 x2 : ℝ), f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 := 
sorry

theorem problem_II_inequality (a : ℝ) (h_a : a ≤ 0) (x : ℝ) (h_x : x ≥ 1) : 
  f x ≥ a * Real.log x - 1 := 
sorry

end problem_I_number_of_zeros_problem_II_inequality_l67_67370


namespace rotameter_gas_phase_measurement_l67_67010

theorem rotameter_gas_phase_measurement
  (liquid_inch_per_lpm : ℝ) (liquid_liter_per_minute : ℝ) (gas_inch_movement_ratio : ℝ) (gas_liter_passed : ℝ) :
  liquid_inch_per_lpm = 2.5 → liquid_liter_per_minute = 60 → gas_inch_movement_ratio = 0.5 → gas_liter_passed = 192 →
  (gas_inch_movement_ratio * liquid_inch_per_lpm * gas_liter_passed / liquid_liter_per_minute) = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end rotameter_gas_phase_measurement_l67_67010


namespace find_number_l67_67613

theorem find_number (x : ℕ) (h : 112 * x = 70000) : x = 625 :=
by
  sorry

end find_number_l67_67613


namespace find_range_of_a_l67_67933

variable (x a : ℝ)

/-- Given p: 2 * x^2 - 9 * x + a < 0 and q: the negation of p is sufficient 
condition for the negation of q,
prove to find the range of the real number a. -/
theorem find_range_of_a (hp: 2 * x^2 - 9 * x + a < 0) (hq: ¬ (2 * x^2 - 9 * x + a < 0) → ¬ q) :
  ∃ a : ℝ, sorry := sorry

end find_range_of_a_l67_67933


namespace index_card_area_reduction_index_card_area_when_other_side_shortened_l67_67244

-- Conditions
def original_length := 4
def original_width := 6
def shortened_length := 2
def target_area := 12
def shortened_other_width := 5

-- Theorems to prove
theorem index_card_area_reduction :
  (original_length - 2) * original_width = target_area := by
  sorry

theorem index_card_area_when_other_side_shortened :
  (original_length) * (original_width - 1) = 20 := by
  sorry

end index_card_area_reduction_index_card_area_when_other_side_shortened_l67_67244


namespace at_least_one_solves_l67_67757

variable (A B : Ω → Prop)
variable (P : MeasureTheory.ProbabilityMeasure Ω)

-- Given conditions
def probA : ℝ := 0.4
def probB : ℝ := 0.5

-- Main statement to prove
theorem at_least_one_solves :
  P (A ∨ B) = 0.7 :=
by
  have prob_not_A := 1 - probA
  have prob_not_B := 1 - probB
  have prob_neither := prob_not_A * prob_not_B
  have prob_at_least_one := 1 - prob_neither
  sorry

end at_least_one_solves_l67_67757


namespace calc_result_l67_67209

theorem calc_result : 
  let a := 82 + 3/5
  let b := 1/15
  let c := 3
  let d := 42 + 7/10
  (a / b) * c - d = 3674.3 :=
by
  sorry

end calc_result_l67_67209


namespace polygon_triangle_division_l67_67883

theorem polygon_triangle_division (n k : ℕ) (h : k * 3 = n * 3 - 6) : k ≥ n - 2 := sorry

end polygon_triangle_division_l67_67883


namespace expectation_absolute_deviation_l67_67284

noncomputable def absolute_deviation (n : ℕ) (m : ℕ) : ℝ :=
    |(m / n : ℝ) - 0.5|

theorem expectation_absolute_deviation (n m : ℕ) 
    (h₁ : n = 10) (h₂ : n = 100) (pn : ℕ → Real) :
    let E_α := ∫ x, absolute_deviation 10 (pn x): ℝ in
    let E_β := ∫ x, absolute_deviation 100 (pn x): ℝ in
    E_α > E_β :=
begin
  sorry
end

end expectation_absolute_deviation_l67_67284


namespace real_solutions_of_equation_l67_67350

theorem real_solutions_of_equation :
  (∃! x : ℝ, (5 * x) / (x^2 + 2 * x + 4) + (6 * x) / (x^2 - 6 * x + 4) = -4 / 3) :=
sorry

end real_solutions_of_equation_l67_67350


namespace relay_race_total_time_l67_67183

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 10
  let t3 := t2 - 15
  let t4 := t1 - 25
  t1 + t2 + t3 + t4 = 200 := by
    sorry

end relay_race_total_time_l67_67183


namespace donation_amount_per_person_l67_67546

theorem donation_amount_per_person (m n : ℕ) 
  (h1 : m + 11 = n + 9) 
  (h2 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (m + 11)) 
  (h3 : ∃ d : ℕ, (m * n + 9 * m + 11 * n + 145) = d * (n + 9))
  : ∃ k : ℕ, k = 25 ∨ k = 47 :=
by
  sorry

end donation_amount_per_person_l67_67546


namespace correct_finance_specialization_l67_67380

-- Variables representing percentages of students specializing in different subjects
variables (students : Type) -- Type of students
           (is_specializing_finance : students → Prop) -- Predicate for finance specialization
           (is_specializing_marketing : students → Prop) -- Predicate for marketing specialization

-- Given conditions
def finance_specialization_percentage : ℝ := 0.88 -- 88% of students are taking finance specialization
def marketing_specialization_percentage : ℝ := 0.76 -- 76% of students are taking marketing specialization

-- The proof statement
theorem correct_finance_specialization (h_finance : finance_specialization_percentage = 0.88) :
  finance_specialization_percentage = 0.88 :=
by
  sorry

end correct_finance_specialization_l67_67380


namespace expression_value_l67_67341

theorem expression_value :
  (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 :=
by sorry

end expression_value_l67_67341


namespace probability_of_rolling_less_than_five_l67_67742

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l67_67742


namespace range_a_ineq_value_of_a_plus_b_l67_67044

open Real

def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)
def g (a x : ℝ) : ℝ := a - abs (x - 2)

noncomputable def range_a (a : ℝ) : Prop :=
  ∃ x : ℝ, f x < g a x

theorem range_a_ineq (a : ℝ) : range_a a ↔ 4 < a := sorry

def solution_set (b : ℝ) : Prop :=
  ∀ x : ℝ, f x < g ((13/2) : ℝ) x ↔ (b < x ∧ x < 7/2)

theorem value_of_a_plus_b (b : ℝ) (h : solution_set b) : (13/2) + b = 6 := sorry

end range_a_ineq_value_of_a_plus_b_l67_67044


namespace sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l67_67307

-- Problem 1: Given that tan(α) = 3, prove that sin(π - α) * cos(2π - α) = 3 / 10.
theorem sin_pi_minus_alpha_cos_2pi_minus_alpha (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin (Real.pi - α) * Real.cos (2 * Real.pi - α) = 3 / 10 :=
by
  sorry

-- Problem 2: Given that sin(α) * cos(α) = 1/4 and 0 < α < π/4, prove that sin(α) - cos(α) = - sqrt(2) / 2.
theorem sin_minus_cos (α : ℝ) (h₁ : Real.sin α * Real.cos α = 1 / 4) (h₂ : 0 < α) (h₃ : α < Real.pi / 4) :
  Real.sin α - Real.cos α = - (Real.sqrt 2) / 2 :=
by
  sorry

end sin_pi_minus_alpha_cos_2pi_minus_alpha_sin_minus_cos_l67_67307


namespace find_y1_l67_67034

variable {y1 y2 y3 : ℝ}

theorem find_y1:
  (0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1) →
  (1 - y1)^2 + 2 * (y1 - y2)^2 + 3 * (y2 - y3)^2 + 4 * y3^2 = 1 / 2 →
  y1 = (3 * Real.sqrt 6 - 6) / 6 := sorry

end find_y1_l67_67034


namespace marcy_votes_correct_l67_67391

-- Definition of variables based on the conditions
def joey_votes : ℕ := 8
def barry_votes : ℕ := 2 * (joey_votes + 3)
def marcy_votes : ℕ := 3 * barry_votes

-- The main statement to prove
theorem marcy_votes_correct : marcy_votes = 66 := 
by 
  sorry

end marcy_votes_correct_l67_67391


namespace tangent_line_at_minus2_l67_67361

open Real

-- Define that f is an even function and differentiable
variables {f : ℝ → ℝ}
hypothesis h_even : ∀ x, f (-x) = f x
hypothesis h_diff : Differentiable ℝ f

-- Given limit condition
noncomputable 
def limit_condition : Prop := 
  ∀ L : ℝ, (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < abs x ∧ abs x < δ → abs ((f (2 + x) - f 2) / (2 * x) - L) < ε) ∧ -1 = L

-- Required proof that the tangent line at (-2,1) is y = 2x + 5
theorem tangent_line_at_minus2 : 
  limit_condition f → ∀ x : ℝ, f' (-2) = 2 → ∃ m b, f (-2) = 1 ∧ m = 2 ∧ b = 5 := by
  sorry

end tangent_line_at_minus2_l67_67361


namespace algebraic_expression_value_l67_67061

theorem algebraic_expression_value (x : ℝ) (h : x ^ 2 - 3 * x = 4) : 2 * x ^ 2 - 6 * x - 3 = 5 :=
by
  sorry

end algebraic_expression_value_l67_67061


namespace triangular_weight_60_grams_l67_67267

-- Define the weights as variables
variables {R T : ℝ} -- round weights and triangular weights are real numbers

-- Define the conditions as hypotheses
theorem triangular_weight_60_grams
  (h1 : R + T = 3 * R)
  (h2 : 4 * R + T = T + R + 90) :
  T = 60 :=
by
  -- indicate that the actual proof is omitted
  sorry

end triangular_weight_60_grams_l67_67267


namespace williams_probability_at_least_one_correct_l67_67450

theorem williams_probability_at_least_one_correct :
  let p_wrong := (1 / 2 : ℝ)
  let p_all_wrong := p_wrong ^ 3
  let p_at_least_one_right := 1 - p_all_wrong
  p_at_least_one_right = 7 / 8 :=
by
  sorry

end williams_probability_at_least_one_correct_l67_67450


namespace find_number_l67_67306

theorem find_number (n : ℤ) 
  (h : (69842 * 69842 - n * n) / (69842 - n) = 100000) : 
  n = 30158 :=
sorry

end find_number_l67_67306


namespace gcd_problem_l67_67871

def gcd3 (x y z : ℕ) : ℕ := Int.gcd x (Int.gcd y z)

theorem gcd_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : gcd3 (a^2 - 1) (b^2 - 1) (c^2 - 1) = 1) :
  gcd3 (a * b + c) (b * c + a) (c * a + b) = gcd3 a b c :=
by
  sorry

end gcd_problem_l67_67871


namespace keith_total_spent_l67_67078

def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tire_cost : ℝ := 112.46
def num_tires : ℕ := 4
def printer_cable_cost : ℝ := 14.85
def num_printer_cables : ℕ := 2
def blank_cd_pack_cost : ℝ := 0.98
def num_blank_cds : ℕ := 10
def sales_tax_rate : ℝ := 0.0825

theorem keith_total_spent : 
  speakers_cost +
  cd_player_cost +
  (num_tires * tire_cost) +
  (num_printer_cables * printer_cable_cost) +
  (num_blank_cds * blank_cd_pack_cost) *
  (1 + sales_tax_rate) = 827.87 := 
sorry

end keith_total_spent_l67_67078


namespace light_flashes_in_three_quarters_hour_l67_67316

theorem light_flashes_in_three_quarters_hour (flash_interval seconds_in_three_quarters_hour : ℕ) 
  (h1 : flash_interval = 15) (h2 : seconds_in_three_quarters_hour = 2700) : 
  (seconds_in_three_quarters_hour / flash_interval = 180) :=
by
  sorry

end light_flashes_in_three_quarters_hour_l67_67316


namespace find_b_l67_67673

-- Definitions
def quadratic (x b c : ℝ) : ℝ := x^2 + b * x + c

theorem find_b (b c : ℝ) 
  (h_diff : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → (∀ y : ℝ, 1 ≤ y ∧ y ≤ 7 → quadratic x b c - quadratic y b c = 25)) :
  b = -4 ∨ b = -12 :=
by sorry

end find_b_l67_67673


namespace polynomial_remainder_l67_67601

theorem polynomial_remainder (z : ℂ) :
  let dividend := 4*z^3 - 5*z^2 - 17*z + 4
  let divisor := 4*z + 6
  let quotient := z^2 - 4*z + (1/4 : ℝ)
  let remainder := 5*z^2 + 6*z + (5/2 : ℝ)
  dividend = divisor * quotient + remainder := sorry

end polynomial_remainder_l67_67601


namespace union_is_correct_l67_67461

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}
def union_set : Set ℤ := {-1, 0, 1, 2}

theorem union_is_correct : M ∪ N = union_set :=
  by sorry

end union_is_correct_l67_67461


namespace area_of_circle_B_l67_67019

theorem area_of_circle_B (rA rB : ℝ) (h : π * rA^2 = 16 * π) (h1 : rB = 2 * rA) : π * rB^2 = 64 * π :=
by
  sorry

end area_of_circle_B_l67_67019


namespace percentage_greater_than_88_l67_67544

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h : x = 88 + percentage * 88) (hx : x = 132) : 
  percentage = 0.5 :=
by
  sorry

end percentage_greater_than_88_l67_67544


namespace starting_number_is_100_l67_67589

theorem starting_number_is_100 (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 1000) (h2 : ∃ k : ℕ, k = 10 ∧ n = 1000 - (k - 1) * 100) :
  n = 100 := by
  sorry

end starting_number_is_100_l67_67589


namespace triangle_perimeter_l67_67480

def is_triangle (a b c : ℕ) := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_perimeter {a b c : ℕ} (h : is_triangle 15 11 19) : 15 + 11 + 19 = 45 := by
  sorry

end triangle_perimeter_l67_67480


namespace track_circumference_is_720_l67_67931

variable (P Q : Type) -- Define the types of P and Q, e.g., as points or runners.

noncomputable def circumference_of_the_track (C : ℝ) : Prop :=
  ∃ y : ℝ, 
  (∃ first_meeting_condition : Prop, first_meeting_condition = (150 = y - 150) ∧
  ∃ second_meeting_condition : Prop, second_meeting_condition = (2*y - 90 = y + 90) ∧
  C = 2 * y)

theorem track_circumference_is_720 :
  circumference_of_the_track 720 :=
by
  sorry

end track_circumference_is_720_l67_67931


namespace determine_a_l67_67367

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (a : ℝ) (x : ℝ) : ℝ :=
  x^3 * (a * 2^x - 2^(-x))

theorem determine_a : ∃ a : ℝ, is_even_function (f a) ∧ a = 1 :=
by
  use 1
  sorry

end determine_a_l67_67367


namespace ratio_A_B_l67_67790

/-- Definition of A and B from the problem. -/
def A : ℕ :=
  1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28

def B : ℕ :=
  1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20

/-- Proof of the inequality 0 < A / B < 1 given the definitions of A and B. -/
theorem ratio_A_B (hA : A = 1400) (hB : B = 1500) : 0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  rw [hA, hB]
  norm_num
  sorry

end ratio_A_B_l67_67790


namespace tan_240_eq_sqrt3_l67_67169

theorem tan_240_eq_sqrt3 :
  ∀ (θ : ℝ), θ = 120 → tan (240 * (π / 180)) = sqrt 3 :=
by
  assume θ
  assume h : θ = 120
  rw [h]
  have h1 : tan ((360 - θ) * (π / 180)) = -tan (θ * (π / 180)), by sorry
  have h2 : tan (120 * (π / 180)) = -sqrt 3, by sorry
  rw [←sub_eq_iff_eq_add, mul_sub, sub_mul, one_mul, sub_eq_add_neg, 
    mul_assoc, ←neg_mul_eq_neg_mul] at h1 
  sorry

end tan_240_eq_sqrt3_l67_67169


namespace four_digit_number_with_divisors_l67_67434

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_minimal_divisor (n p : Nat) : Prop :=
  p > 1 ∧ n % p = 0
  
def is_maximal_divisor (n q : Nat) : Prop :=
  q < n ∧ n % q = 0
  
theorem four_digit_number_with_divisors :
  ∃ (n p : Nat), is_four_digit n ∧ is_minimal_divisor n p ∧ n = 49 * p * p :=
by
  sorry

end four_digit_number_with_divisors_l67_67434


namespace jason_manager_years_l67_67866

-- Definitions based on the conditions
def jason_bartender_years : ℕ := 9
def jason_total_months : ℕ := 150
def additional_months_excluded : ℕ := 6

-- Conversion from months to years
def total_years := jason_total_months / 12
def excluded_years := additional_months_excluded / 12

-- Lean statement for the proof problem
theorem jason_manager_years :
  total_years - jason_bartender_years - excluded_years = 3 := by
  sorry

end jason_manager_years_l67_67866


namespace perimeter_of_triangle_l67_67263

theorem perimeter_of_triangle
  {r A P : ℝ} (hr : r = 2.5) (hA : A = 25) :
  P = 20 :=
by
  sorry

end perimeter_of_triangle_l67_67263


namespace incorrect_multiplicative_inverse_product_l67_67449

theorem incorrect_multiplicative_inverse_product:
  ∃ (a b : ℝ), a + b = 0 ∧ a * b ≠ 1 :=
by
  sorry

end incorrect_multiplicative_inverse_product_l67_67449


namespace squares_to_nine_l67_67900

theorem squares_to_nine (x : ℤ) : x^2 = 9 ↔ x = 3 ∨ x = -3 :=
sorry

end squares_to_nine_l67_67900


namespace ratio_of_speeds_l67_67008

variable (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r)))
variable (f1 f2 : ℝ) (h2 : b * (1/4) + b * (3/4) = b)

theorem ratio_of_speeds (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r))) : b = 3 * r :=
by sorry

end ratio_of_speeds_l67_67008


namespace combined_annual_income_after_expenses_l67_67778

noncomputable def brady_monthly_incomes : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
noncomputable def dwayne_monthly_incomes : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_annual_expense : ℕ := 450
def dwayne_annual_expense : ℕ := 300

def annual_income (monthly_incomes : List ℕ) : ℕ :=
  monthly_incomes.foldr (· + ·) 0

theorem combined_annual_income_after_expenses :
  (annual_income brady_monthly_incomes - brady_annual_expense) +
  (annual_income dwayne_monthly_incomes - dwayne_annual_expense) = 3930 :=
by
  sorry

end combined_annual_income_after_expenses_l67_67778


namespace determine_numbers_l67_67409

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l67_67409


namespace exp_abs_dev_10_gt_100_l67_67291

open ProbabilityTheory
open MeasureTheory

def deviation (n m : ℕ) : ℝ := (m / n : ℝ) - 0.5
def abs_deviation (n m : ℕ) : ℝ := abs ((m / n : ℝ) - 0.5)
def exp_abs_deviation (n : ℕ) : ℝ := sorry -- The expected value of the absolute deviation

theorem exp_abs_dev_10_gt_100 :
  (exp_abs_deviation 10) > (exp_abs_deviation 100) :=
sorry

end exp_abs_dev_10_gt_100_l67_67291


namespace tangent_line_min_slope_l67_67819

noncomputable def curve (x : ℝ) : ℝ := x^3 + 3*x - 1

noncomputable def curve_derivative (x : ℝ) : ℝ := 3*x^2 + 3

theorem tangent_line_min_slope :
  ∃ k b : ℝ, (∀ x : ℝ, curve_derivative x ≥ 3) ∧ 
             k = 3 ∧ b = 1 ∧
             (∀ x y : ℝ, y = k * x + b ↔ 3 * x - y + 1 = 0) := 
by {
  sorry
}

end tangent_line_min_slope_l67_67819


namespace probability_both_genders_among_selected_l67_67310

open Finset

theorem probability_both_genders_among_selected :
  (choose 7 3).toRat ≠ 0 →
  (5.choose 3).toRat / (7.choose 3).toRat + (2.choose 3).toRat / (7.choose 3).toRat ≠ 1 →
  ∀ (n : ℕ), n = (7.choose 3).toRat →
  ((n - (5.choose 3).toRat - (2.choose 3).toRat) / n = (3 / 5 : ℝ)) :=
by
  sorry

end probability_both_genders_among_selected_l67_67310


namespace sixty_percent_of_total_is_960_l67_67892

-- Definitions from the conditions
def boys : ℕ := 600
def difference : ℕ := 400
def girls : ℕ := boys + difference
def total : ℕ := boys + girls
def sixty_percent_of_total : ℕ := total * 60 / 100

-- The theorem to prove
theorem sixty_percent_of_total_is_960 :
  sixty_percent_of_total = 960 := 
  sorry

end sixty_percent_of_total_is_960_l67_67892


namespace circumference_of_base_of_cone_l67_67009

theorem circumference_of_base_of_cone (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) 
  (h1 : V = 24 * Real.pi) (h2 : h = 6) (h3 : V = (1/3) * Real.pi * r^2 * h) 
  (h4 : r = Real.sqrt 12) : C = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end circumference_of_base_of_cone_l67_67009


namespace exists_Q_l67_67238

-- Defining the problem conditions
variables {P : Polynomial ℝ} (hP : ∀ x : ℝ, P (Real.cos x) = P (Real.sin x))

-- Stating the theorem to be proved
theorem exists_Q (P : Polynomial ℝ) (hP : ∀ x : ℝ, P (Real.cos x) = P (Real.sin x)) : 
  ∃ Q : Polynomial ℝ, P = Polynomial.comp Q (Polynomial.X^4 - Polynomial.X^2) :=
sorry

end exists_Q_l67_67238


namespace complex_z_pow_condition_l67_67542

theorem complex_z_pow_condition (z : ℂ) (h : z + z⁻¹ = 2 * real.sqrt 2) : z^100 + z^(-100) = -2 := by
  sorry

end complex_z_pow_condition_l67_67542


namespace pam_total_apples_l67_67567

theorem pam_total_apples (pam_bags : ℕ) (gerald_bags_apples : ℕ) (gerald_bags_factor : ℕ) 
  (pam_bags_count : pam_bags = 10)
  (gerald_apples_count : gerald_bags_apples = 40)
  (gerald_bags_ratio : gerald_bags_factor = 3) : 
  pam_bags * gerald_bags_factor * gerald_bags_apples = 1200 := by
  sorry

end pam_total_apples_l67_67567


namespace abs_val_eq_two_l67_67833

theorem abs_val_eq_two (x : ℝ) (h : |x| = 2) : x = 2 ∨ x = -2 := 
sorry

end abs_val_eq_two_l67_67833


namespace tiling_tetromino_divisibility_l67_67656

theorem tiling_tetromino_divisibility (n : ℕ) : 
  (∃ (t : ℕ), n = 4 * t) ↔ (∃ (k : ℕ), n * n = 4 * k) :=
by
  sorry

end tiling_tetromino_divisibility_l67_67656


namespace complement_intersection_l67_67358

open Set

theorem complement_intersection (M N : Set ℝ) (hM : M = {x | 0 < x ∧ x < 1}) (hN : N = {y | 2 ≤ y}) :
  (@compl ℝ _ M) ∩ N = {y | 2 ≤ y} :=
by
  -- assume definitions of M and N
  rw [hM, hN]
  -- simplifications
  rw [compl_set_of]
  -- reformulate the intersection in terms of set properties
  congr
  -- endpoints and intervals
  rw [← Set.union_comm, uIcc_compl (le_refl (0:ℝ))]

  sorry -- proof steps to be filled in.

end complement_intersection_l67_67358


namespace sum_in_range_l67_67961

noncomputable def mixed_number_sum : ℚ :=
  3 + 1/8 + 4 + 3/7 + 6 + 2/21

theorem sum_in_range : 13.5 ≤ mixed_number_sum ∧ mixed_number_sum < 14 := by
  sorry

end sum_in_range_l67_67961


namespace total_distance_crawled_l67_67479

theorem total_distance_crawled :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let pos4 := 0
  abs (pos2 - pos1) + abs (pos3 - pos2) + abs (pos4 - pos3) = 29 :=
by
  sorry

end total_distance_crawled_l67_67479


namespace probability_of_event_E_l67_67737

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l67_67737


namespace find_x_l67_67548

-- Definitions based on the problem conditions
def angle_CDE : ℝ := 90 -- angle CDE in degrees
def angle_ECB : ℝ := 68 -- angle ECB in degrees

-- Theorem statement
theorem find_x (x : ℝ) 
  (h1 : angle_CDE = 90) 
  (h2 : angle_ECB = 68) 
  (h3 : angle_CDE + x + angle_ECB = 180) : 
  x = 22 := 
by
  sorry

end find_x_l67_67548


namespace factorize_expr_l67_67663

theorem factorize_expr (x : ℝ) : 75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := 
by
  sorry

end factorize_expr_l67_67663


namespace floor_equation_solution_l67_67968

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l67_67968


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l67_67970

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l67_67970


namespace john_payment_and_hourly_rate_l67_67232

variable (court_hours : ℕ) (prep_hours : ℕ) (upfront_fee : ℕ) 
variable (total_payment : ℕ) (brother_contribution_factor : ℕ)
variable (hourly_rate : ℚ) (john_payment : ℚ)

axiom condition1 : upfront_fee = 1000
axiom condition2 : court_hours = 50
axiom condition3 : prep_hours = 2 * court_hours
axiom condition4 : total_payment = 8000
axiom condition5 : brother_contribution_factor = 2

theorem john_payment_and_hourly_rate :
  (john_payment = total_payment / brother_contribution_factor + upfront_fee) ∧
  (hourly_rate = (total_payment - upfront_fee) / (court_hours + prep_hours)) :=
by
  sorry

end john_payment_and_hourly_rate_l67_67232


namespace max_square_plots_l67_67626

theorem max_square_plots (width height internal_fence_length : ℕ) 
(h_w : width = 60) (h_h : height = 30) (h_fence: internal_fence_length = 2400) : 
  ∃ n : ℕ, (60 * 30 / (n * n) = 400 ∧ 
  (30 * (60 / n - 1) + 60 * (30 / n - 1) + 60 + 30) ≤ internal_fence_length) :=
sorry

end max_square_plots_l67_67626


namespace bridge_length_l67_67317

   noncomputable def walking_speed_km_per_hr : ℝ := 6
   noncomputable def walking_time_minutes : ℝ := 15

   noncomputable def length_of_bridge (speed_km_per_hr : ℝ) (time_min : ℝ) : ℝ :=
     (speed_km_per_hr * 1000 / 60) * time_min

   theorem bridge_length :
     length_of_bridge walking_speed_km_per_hr walking_time_minutes = 1500 := 
   by
     sorry
   
end bridge_length_l67_67317


namespace number_of_fish_l67_67550

theorem number_of_fish (initial_fish : ℕ) (double_day : ℕ → ℕ → ℕ) (remove_fish : ℕ → ℕ → ℕ) (add_fish : ℕ → ℕ → ℕ) :
  (initial_fish = 6) →
  (∀ n m, double_day n m = n * 2) →
  (∀ n d m, d = 3 ∨ d = 5 → remove_fish n d = n - n / m) →
  (∀ n d, d = 7 → add_fish n d = n + 15) →
  (double_day 6 1 = 12) →
  (double_day 12 2 = 24) →
  (remove_fish 24 3 = 16) →
  (double_day 16 4 = 32) →
  (double_day 32 5 = 64) →
  (remove_fish 64 5 = 48) →
  (double_day 48 6 = 96) →
  (double_day 96 7 = 192) →
  (add_fish 192 7 = 207) →
  207 = 207 :=
begin
  intros,
  -- Proof omitted since it's not required
  sorry,
end

end number_of_fish_l67_67550


namespace jo_bob_pulled_chain_first_time_l67_67386

/-- Given the conditions of the balloon ride, prove that Jo-Bob pulled the chain
    for the first time for 15 minutes. --/
theorem jo_bob_pulled_chain_first_time (x : ℕ) : 
  (50 * x - 100 + 750 = 1400) → (x = 15) :=
by
  intro h
  sorry

end jo_bob_pulled_chain_first_time_l67_67386


namespace problem1_problem2_l67_67387

variable {a b : ℝ}

theorem problem1 (h : a > b) : a - 3 > b - 3 :=
by sorry

theorem problem2 (h : a > b) : -4 * a < -4 * b :=
by sorry

end problem1_problem2_l67_67387


namespace solid_id_views_not_cylinder_l67_67945

theorem solid_id_views_not_cylinder :
  ∀ (solid : Type),
  (∃ (shape1 shape2 shape3 : solid),
    shape1 = shape2 ∧ shape2 = shape3) →
  solid ≠ cylinder :=
by 
  sorry

end solid_id_views_not_cylinder_l67_67945


namespace family_can_purchase_furniture_in_april_l67_67898

noncomputable def monthly_income : ℤ := 150000
noncomputable def monthly_expenses : ℤ := 115000
noncomputable def initial_savings : ℤ := 45000
noncomputable def furniture_cost : ℤ := 127000

theorem family_can_purchase_furniture_in_april : 
  ∃ (months : ℕ), months = 3 ∧ 
  (initial_savings + months * (monthly_income - monthly_expenses) >= furniture_cost) :=
by
  -- proof will be written here
  sorry

end family_can_purchase_furniture_in_april_l67_67898


namespace domain_of_function_range_of_function_l67_67756

-- Problem 1:
theorem domain_of_function :
  { x : ℝ | 3 - x ≥ 0 ∧ x - 1 ≠ 0 } = { x : ℝ | x ≤ 3 ∧ x ≠ 1 } :=
by sorry

-- Problem 2:
theorem range_of_function :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 4 → (-x^2 + 4 * x - 2) ∈ set.Icc (-2 : ℝ) 2 :=
by sorry

end domain_of_function_range_of_function_l67_67756


namespace product_end_digit_3_mod_5_l67_67504

theorem product_end_digit_3_mod_5 : 
  let lst := list.range' 0 10 in
  let lst := list.map (λ n, 10 * n + 3) lst in
  (list.prod lst) % 5 = 4 :=
by
  let lst := list.range' 0 10;
  let lst := list.map (λ n, 10 * n + 3) lst;
  show (list.prod lst) % 5 = 4;
  sorry

end product_end_digit_3_mod_5_l67_67504


namespace max_children_tickets_l67_67151

theorem max_children_tickets 
  (total_budget : ℕ) (adult_ticket_cost : ℕ) 
  (child_ticket_cost_individual : ℕ) (child_ticket_cost_group : ℕ) (min_group_tickets : ℕ) 
  (remaining_budget : ℕ) :
  total_budget = 75 →
  adult_ticket_cost = 12 →
  child_ticket_cost_individual = 6 →
  child_ticket_cost_group = 4 →
  min_group_tickets = 5 →
  (remaining_budget = total_budget - adult_ticket_cost) →
  ∃ (n : ℕ), n = 15 ∧ n * child_ticket_cost_group ≤ remaining_budget :=
by
  intros h_total_budget h_adult_ticket_cost h_child_ticket_cost_individual h_child_ticket_cost_group h_min_group_tickets h_remaining_budget
  sorry

end max_children_tickets_l67_67151


namespace trig_expression_value_l67_67037

theorem trig_expression_value (α : ℝ) (h : Real.tan α = 3) : 
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = 2 :=
sorry

end trig_expression_value_l67_67037


namespace triangle_perimeter_l67_67260

theorem triangle_perimeter (r A : ℝ) (p : ℝ)
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) : 
  p = 20 :=
by 
  sorry

end triangle_perimeter_l67_67260


namespace units_digit_k_squared_plus_2_k_l67_67389

noncomputable def k : ℕ := 2017^2 + 2^2017

theorem units_digit_k_squared_plus_2_k : (k^2 + 2^k) % 10 = 3 := 
  sorry

end units_digit_k_squared_plus_2_k_l67_67389


namespace probability_of_less_than_5_is_one_half_l67_67751

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l67_67751


namespace find_principal_amount_l67_67094

theorem find_principal_amount (P r : ℝ) 
    (h1 : 815 - P = P * r * 3) 
    (h2 : 850 - P = P * r * 4) : 
    P = 710 :=
by
  -- proof steps will go here
  sorry

end find_principal_amount_l67_67094


namespace at_least_one_ge_one_l67_67237

theorem at_least_one_ge_one (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  let a := x1 / x2
  let b := x2 / x3
  let c := x3 / x1
  a + b + c ≥ 3 → (a ≥ 1 ∨ b ≥ 1 ∨ c ≥ 1) :=
by
  intros
  sorry

end at_least_one_ge_one_l67_67237


namespace largest_number_of_square_plots_l67_67625

theorem largest_number_of_square_plots (n : ℕ) 
  (field_length : ℕ := 30) 
  (field_width : ℕ := 60) 
  (total_fence : ℕ := 2400) 
  (square_length : ℕ := field_length / n) 
  (fencing_required : ℕ := 60 * n) :
  field_length % n = 0 → 
  field_width % square_length = 0 → 
  fencing_required = total_fence → 
  2 * n^2 = 3200 :=
by
  intros h1 h2 h3
  sorry

end largest_number_of_square_plots_l67_67625


namespace parabola_directrix_l67_67475

theorem parabola_directrix (p : ℝ) (A B : ℝ × ℝ) (O D : ℝ × ℝ) :
  A ≠ B →
  O = (0, 0) →
  D = (1, 2) →
  (∃ k, k = ((2:ℝ) - 0) / ((1:ℝ) - 0) ∧ k = 2) →
  (∃ k, k = - 1 / 2) →
  (∀ x y, y^2 = 2 * p * x) →
  p = 5 / 2 →
  O.1 * A.1 + O.2 * A.2 = 0 →
  O.1 * B.1 + O.2 * B.2 = 0 →
  A.1 * B.1 + A.2 * B.2 = 0 →
  (∃ k, (y - 2) = k * (x - 1) ∧ (A.1 * B.1) = 25 ∧ (A.1 + B.1) = 10 + 8 * p) →
  ∃ dir_eq, dir_eq = -5 / 4 :=
by
  sorry

end parabola_directrix_l67_67475


namespace relay_race_total_time_l67_67181

-- Definitions based on the problem conditions
def athlete1_time : ℕ := 55
def athlete2_time : ℕ := athlete1_time + 10
def athlete3_time : ℕ := athlete2_time - 15
def athlete4_time : ℕ := athlete1_time - 25

-- Problem statement
theorem relay_race_total_time : 
  athlete1_time + athlete2_time + athlete3_time + athlete4_time = 200 := 
by 
  sorry

end relay_race_total_time_l67_67181


namespace math_majors_consecutive_probability_l67_67731

def twelve_people := 12
def math_majors := 5
def physics_majors := 4
def biology_majors := 3

def total_ways := Nat.choose twelve_people math_majors

-- Computes the probability that all five math majors sit in consecutive seats
theorem math_majors_consecutive_probability :
  (12 : ℕ) / (Nat.choose twelve_people math_majors) = 1 / 66 := by
  sorry

end math_majors_consecutive_probability_l67_67731


namespace john_billed_minutes_l67_67508

theorem john_billed_minutes
  (monthly_fee : ℝ := 5)
  (cost_per_minute : ℝ := 0.25)
  (total_bill : ℝ := 12.02) :
  ∃ (minutes : ℕ), minutes = 28 :=
by
  have amount_for_minutes := total_bill - monthly_fee
  have minutes_float := amount_for_minutes / cost_per_minute
  have minutes := floor minutes_float
  use minutes
  have : 28 = (floor minutes_float : ℕ) := sorry
  exact this.symm

end john_billed_minutes_l67_67508


namespace rectangular_prism_cut_l67_67483

theorem rectangular_prism_cut
  (x y : ℕ)
  (original_volume : ℕ := 15 * 5 * 4) 
  (remaining_volume : ℕ := 120) 
  (cut_out_volume_eq : original_volume - remaining_volume = 5 * x * y) 
  (x_condition : 1 < x) 
  (x_condition_2 : x < 4) 
  (y_condition : 1 < y) 
  (y_condition_2 : y < 15) : 
  x + y = 15 := 
sorry

end rectangular_prism_cut_l67_67483


namespace abc_value_l67_67212

variables (a b c : ℝ)

theorem abc_value (h1 : a * (b + c) = 156) (h2 : b * (c + a) = 168) (h3 : c * (a + b) = 180) :
  a * b * c = 288 * Real.sqrt 7 :=
sorry

end abc_value_l67_67212


namespace class_sizes_l67_67245

theorem class_sizes
  (finley_students : ℕ)
  (johnson_students : ℕ)
  (garcia_students : ℕ)
  (smith_students : ℕ)
  (h1 : finley_students = 24)
  (h2 : johnson_students = 10 + finley_students / 2)
  (h3 : garcia_students = 2 * johnson_students)
  (h4 : smith_students = finley_students / 3) :
  finley_students = 24 ∧ johnson_students = 22 ∧ garcia_students = 44 ∧ smith_students = 8 :=
by
  sorry

end class_sizes_l67_67245


namespace probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l67_67724

/-- 
There are 30 tiles in box C numbered from 1 to 30 and 30 tiles in box D numbered from 21 to 50. 
We want to prove that the probability of drawing a tile less than 20 from box C and a tile that 
is either odd or greater than 40 from box D is 19/45. 
-/
theorem probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40 :
  (19 / 30) * (2 / 3) = (19 / 45) :=
by sorry

end probability_tile_C_less_than_20_and_tile_D_odd_or_greater_than_40_l67_67724


namespace floor_eq_l67_67983

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l67_67983


namespace Masha_thought_of_numbers_l67_67398

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l67_67398


namespace vampire_pints_per_person_l67_67637

-- Definitions based on conditions
def gallons_per_week : ℕ := 7
def pints_per_gallon : ℕ := 8
def days_per_week : ℕ := 7
def people_per_day : ℕ := 4

-- The statement to be proven
theorem vampire_pints_per_person :
  (gallons_per_week * pints_per_gallon) / (days_per_week * people_per_day) = 2 :=
by
  sorry

end vampire_pints_per_person_l67_67637


namespace root_of_quadratic_property_l67_67215

theorem root_of_quadratic_property (m : ℝ) (h : m^2 - 2 * m - 1 = 0) :
  m^2 + (1 / m^2) = 6 :=
sorry

end root_of_quadratic_property_l67_67215


namespace train_speed_is_36_kph_l67_67773

noncomputable def speed_of_train (length_train length_bridge time_to_pass : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_mps := total_distance / time_to_pass
  let speed_kph := speed_mps * 3600 / 1000
  speed_kph

theorem train_speed_is_36_kph :
  speed_of_train 360 140 50 = 36 :=
by
  sorry

end train_speed_is_36_kph_l67_67773


namespace john_billed_for_28_minutes_l67_67507

variable (monthlyFee : ℝ) (costPerMinute : ℝ) (totalBill : ℝ)
variable (minutesBilled : ℝ)

def is_billed_correctly (monthlyFee totalBill costPerMinute minutesBilled : ℝ) : Prop :=
  totalBill - monthlyFee = minutesBilled * costPerMinute ∧ minutesBilled = 28

theorem john_billed_for_28_minutes : 
  is_billed_correctly 5 12.02 0.25 28 := 
by
  sorry

end john_billed_for_28_minutes_l67_67507


namespace no_solution_for_k_l67_67668

theorem no_solution_for_k 
  (a1 a2 a3 a4 : ℝ) 
  (h_pos1 : 0 < a1) (h_pos2 : a1 < a2) 
  (h_pos3 : a2 < a3) (h_pos4 : a3 < a4) 
  (x1 x2 x3 x4 k : ℝ) 
  (h1 : x1 + x2 + x3 + x4 = 1) 
  (h2 : a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 = k) 
  (h3 : a1^2 * x1 + a2^2 * x2 + a3^2 * x3 + a4^2 * x4 = k^2) 
  (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (hx3 : 0 ≤ x3) (hx4 : 0 ≤ x4) :
  false := 
sorry

end no_solution_for_k_l67_67668


namespace value_of_x_l67_67059

theorem value_of_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end value_of_x_l67_67059


namespace fg_sqrt2_eq_neg5_l67_67541

noncomputable def f (x : ℝ) : ℝ := 4 - 3 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + 1

theorem fg_sqrt2_eq_neg5 : f (g (Real.sqrt 2)) = -5 := by
  sorry

end fg_sqrt2_eq_neg5_l67_67541


namespace rectangle_perimeter_given_square_l67_67226

-- Defining the problem conditions
def square_side_length (p : ℕ) : ℕ := p / 4

def rectangle_perimeter (s : ℕ) : ℕ := 2 * (s + (s / 2))

-- Stating the theorem: Given the perimeter of the square is 80, prove the perimeter of one of the rectangles is 60
theorem rectangle_perimeter_given_square (p : ℕ) (h : p = 80) : rectangle_perimeter (square_side_length p) = 60 :=
by
  sorry

end rectangle_perimeter_given_square_l67_67226


namespace unique_triplet_exists_l67_67665

theorem unique_triplet_exists (a b p : ℕ) (hp : Nat.Prime p) : 
  (a + b)^p = p^a + p^b → (a = 1 ∧ b = 1 ∧ p = 2) :=
by sorry

end unique_triplet_exists_l67_67665


namespace expected_absolute_deviation_greater_in_10_tosses_l67_67288

variable {n m : ℕ}

def frequency_of_heads (m n : ℕ) : ℚ := m / n

def deviation_from_probability (m n : ℕ) : ℚ :=
  m / n - 0.5

def absolute_deviation (m n : ℕ) : ℚ :=
  |m / n - 0.5|

noncomputable def expectation_absolute_deviation (n : ℕ) : ℚ := 
  if n = 10 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 10 tosses
    sorry 
  else if n = 100 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 100 tosses
    sorry 
  else 0

theorem expected_absolute_deviation_greater_in_10_tosses (a b : ℚ) :
  expectation_absolute_deviation 10 > expectation_absolute_deviation 100 :=
by sorry

end expected_absolute_deviation_greater_in_10_tosses_l67_67288


namespace paula_bracelets_count_l67_67629

-- Defining the given conditions
def cost_bracelet := 4
def cost_keychain := 5
def cost_coloring_book := 3
def total_spent := 20

-- Defining the cost for Paula's items
def cost_paula (B : ℕ) := B * cost_bracelet + cost_keychain

-- Defining the cost for Olive's items
def cost_olive := cost_coloring_book + cost_bracelet

-- Defining the main problem
theorem paula_bracelets_count (B : ℕ) (h : cost_paula B + cost_olive = total_spent) : B = 2 := by
  sorry

end paula_bracelets_count_l67_67629


namespace full_day_students_count_l67_67952

-- Define the conditions
def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

-- Define the statement to prove
theorem full_day_students_count :
  total_students - (total_students * percentage_half_day_students / 100) = 60 :=
by
  sorry

end full_day_students_count_l67_67952


namespace checkerboard_probability_not_on_perimeter_l67_67091

def total_squares : ℕ := 81

def perimeter_squares : ℕ := 32

def non_perimeter_squares : ℕ := total_squares - perimeter_squares

noncomputable def probability_not_on_perimeter : ℚ := non_perimeter_squares / total_squares

theorem checkerboard_probability_not_on_perimeter :
  probability_not_on_perimeter = 49 / 81 :=
by
  sorry

end checkerboard_probability_not_on_perimeter_l67_67091


namespace elevator_max_weight_l67_67705

theorem elevator_max_weight :
  let avg_weight_adult := 150
  let num_adults := 7
  let avg_weight_child := 70
  let num_children := 5
  let orig_max_weight := 1500
  let weight_adults := num_adults * avg_weight_adult
  let weight_children := num_children * avg_weight_child
  let current_weight := weight_adults + weight_children
  let upgrade_percentage := 0.10
  let new_max_weight := orig_max_weight * (1 + upgrade_percentage)
  new_max_weight - current_weight = 250 := 
  by
    sorry

end elevator_max_weight_l67_67705


namespace airplane_fraction_l67_67012

noncomputable def driving_time : ℕ := 195

noncomputable def airport_drive_time : ℕ := 10

noncomputable def waiting_time : ℕ := 20

noncomputable def get_off_time : ℕ := 10

noncomputable def faster_by : ℕ := 90

theorem airplane_fraction :
  ∃ x : ℕ, 195 = 40 + x + 90 ∧ x = 65 ∧ x = driving_time / 3 := sorry

end airplane_fraction_l67_67012


namespace heather_oranges_l67_67529

theorem heather_oranges (initial_oranges additional_oranges : ℝ) (h1 : initial_oranges = 60.5) (h2 : additional_oranges = 35.8) :
  initial_oranges + additional_oranges = 96.3 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end heather_oranges_l67_67529


namespace obtuse_angle_between_line_and_plane_l67_67949

-- Define the problem conditions
def is_obtuse_angle (θ : ℝ) : Prop := θ > 90 ∧ θ < 180

-- Define what we are proving
theorem obtuse_angle_between_line_and_plane (θ : ℝ) (h1 : θ = angle_between_line_and_plane) :
  is_obtuse_angle θ :=
sorry

end obtuse_angle_between_line_and_plane_l67_67949


namespace power_sum_l67_67650

theorem power_sum (h : (9 : ℕ) = 3^2) : (2^567 + (9^5 / 3^2) : ℕ) = 2^567 + 6561 := by
  sorry

end power_sum_l67_67650


namespace probability_of_disturbance_l67_67250

theorem probability_of_disturbance :
    let n := 6 in
    let prob_first_no_disturb := 2 / n in
    let prob_second_no_disturb := 2 / (n - 1) in
    let prob_third_no_disturb := 2 / (n - 2) in
    let prob_fourth_no_disturb := 2 / (n - 3) in
    let total_uninterrupted_prob := prob_first_no_disturb * prob_second_no_disturb * prob_third_no_disturb * prob_fourth_no_disturb in
    let prob_disturbance := 1 - total_uninterrupted_prob in
    n = 6 ∧ prob_disturbance = 43 / 45 := by
  sorry

end probability_of_disturbance_l67_67250


namespace required_ratio_l67_67847

theorem required_ratio (M N : ℕ) (hM : M = 8) (hN : N = 27) : 8 / 27 < 10 / 27 :=
by { sorry }

end required_ratio_l67_67847


namespace expected_absolute_deviation_greater_in_10_tosses_l67_67289

variable {n m : ℕ}

def frequency_of_heads (m n : ℕ) : ℚ := m / n

def deviation_from_probability (m n : ℕ) : ℚ :=
  m / n - 0.5

def absolute_deviation (m n : ℕ) : ℚ :=
  |m / n - 0.5|

noncomputable def expectation_absolute_deviation (n : ℕ) : ℚ := 
  if n = 10 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 10 tosses
    sorry 
  else if n = 100 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 100 tosses
    sorry 
  else 0

theorem expected_absolute_deviation_greater_in_10_tosses (a b : ℚ) :
  expectation_absolute_deviation 10 > expectation_absolute_deviation 100 :=
by sorry

end expected_absolute_deviation_greater_in_10_tosses_l67_67289


namespace solve_for_x_l67_67540

theorem solve_for_x (x : ℝ) (h :  9 / x^2 = x / 25) : x = 5 :=
by 
  sorry

end solve_for_x_l67_67540


namespace ellipse_focal_length_l67_67519

theorem ellipse_focal_length (k : ℝ) :
  (∀ x y : ℝ, x^2 / k + y^2 / 2 = 1) →
  (∃ c : ℝ, 2 * c = 2 ∧ (k = 1 ∨ k = 3)) :=
by
  -- Given condition: equation of ellipse and focal length  
  intro h  
  sorry

end ellipse_focal_length_l67_67519


namespace best_fit_slope_eq_l67_67299

theorem best_fit_slope_eq :
  let x1 := 150
  let y1 := 2
  let x2 := 160
  let y2 := 3
  let x3 := 170
  let y3 := 4
  (x2 - x1 = 10 ∧ x3 - x2 = 10) →
  let slope := (x1 - x2) * (y1 - y2) + (x3 - x2) * (y3 - y2) / (x1 - x2)^2 + (x3 - x2)^2
  slope = 1 / 10 :=
sorry

end best_fit_slope_eq_l67_67299


namespace sum_binomial_coefficients_l67_67816

theorem sum_binomial_coefficients (a b : ℕ) (h1 : a = 2^3) (h2 : b = (2 + 1)^3) : a + b = 35 :=
by
  sorry

end sum_binomial_coefficients_l67_67816


namespace football_team_progress_l67_67765

theorem football_team_progress (lost_yards gained_yards : Int) : lost_yards = -5 → gained_yards = 13 → lost_yards + gained_yards = 8 := 
by
  intros h_lost h_gained
  rw [h_lost, h_gained]
  sorry

end football_team_progress_l67_67765


namespace find_rth_term_l67_67032

theorem find_rth_term (n r : ℕ) (S : ℕ → ℕ) (hS : ∀ n, S n = 4 * n + 5 * n^2) :
  r > 0 → (S r) - (S (r - 1)) = 10 * r - 1 :=
by
  intro h
  have hr_pos := h
  sorry

end find_rth_term_l67_67032


namespace positive_integer_solutions_count_l67_67254

theorem positive_integer_solutions_count :
  (∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ x + y + z = 2010) → (336847 = 336847) :=
by {
  sorry
}

end positive_integer_solutions_count_l67_67254


namespace area_of_right_triangle_l67_67908

theorem area_of_right_triangle (A B C : ℝ) (hA : A = 64) (hB : B = 36) (hC : C = 100) : 
  (1 / 2) * (Real.sqrt A) * (Real.sqrt B) = 24 :=
by
  sorry

end area_of_right_triangle_l67_67908


namespace factorize_expr_l67_67664

theorem factorize_expr (x : ℝ) : 75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := 
by
  sorry

end factorize_expr_l67_67664


namespace investment_plans_count_l67_67628

noncomputable def investment_plans (projects : Finset ℕ) (cities : Finset ℕ) : ℕ :=
  let scenario1 := (projects.card.choose 2) * (cities.card.choose 2) * (cities.card - 2)
  let scenario2 := (cities.card - 1) * (projects.card.factorial / (projects.card - (cities.card - 1)).factorial)
  scenario1 + scenario2

theorem investment_plans_count : investment_plans {1, 2, 3} {1, 2, 3, 4} = 60 :=
  by sorry

end investment_plans_count_l67_67628


namespace sum_of_squares_of_roots_l67_67430

theorem sum_of_squares_of_roots (α β : ℝ)
  (h_root1 : 10 * α^2 - 14 * α - 24 = 0)
  (h_root2 : 10 * β^2 - 14 * β - 24 = 0)
  (h_distinct : α ≠ β) :
  α^2 + β^2 = 169 / 25 :=
sorry

end sum_of_squares_of_roots_l67_67430


namespace probability_at_least_one_multiple_of_4_l67_67325

theorem probability_at_least_one_multiple_of_4 :
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  (probability_at_least_one_multiple_of_4 = 528 / 1250) := 
by
  -- Define the conditions
  let bound := 50
  let multiples_of_4 := 12
  let probability_no_multiple_of_4 := (38 / 50) * (38 / 50)
  let probability_at_least_one_multiple_of_4 := 1 - probability_no_multiple_of_4
  sorry

end probability_at_least_one_multiple_of_4_l67_67325


namespace a_plus_b_l67_67520

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x)

theorem a_plus_b (a b : ℝ) (h : f (a - 1) + f b = 0) : a + b = 1 :=
by
  sorry

end a_plus_b_l67_67520


namespace oil_leak_l67_67481

theorem oil_leak (a b c : ℕ) (h₁ : a = 6522) (h₂ : b = 11687) (h₃ : c = b - a) : c = 5165 :=
by
  rw [h₁, h₂] at h₃
  exact h₃

end oil_leak_l67_67481


namespace scientific_notation_of_distance_l67_67844

theorem scientific_notation_of_distance :
  ∃ a n, (1 ≤ a ∧ a < 10) ∧ 384000 = a * 10^n ∧ a = 3.84 ∧ n = 5 :=
by
  sorry

end scientific_notation_of_distance_l67_67844


namespace james_total_matches_l67_67862

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end james_total_matches_l67_67862


namespace solve_fractional_equation_l67_67985

theorem solve_fractional_equation :
  {x : ℝ | 1 / (x^2 + 8 * x - 6) + 1 / (x^2 + 5 * x - 6) + 1 / (x^2 - 14 * x - 6) = 0}
  = {3, -2, -6, 1} :=
by
  sorry

end solve_fractional_equation_l67_67985


namespace distance_to_place_l67_67904

-- Define the conditions
def speed_boat_standing_water : ℝ := 16
def speed_stream : ℝ := 2
def total_time_taken : ℝ := 891.4285714285714

-- Define the calculated speeds
def downstream_speed : ℝ := speed_boat_standing_water + speed_stream
def upstream_speed : ℝ := speed_boat_standing_water - speed_stream

-- Define the variable for the distance
variable (D : ℝ)

-- State the theorem to prove
theorem distance_to_place :
  D / downstream_speed + D / upstream_speed = total_time_taken →
  D = 7020 :=
by
  intro h
  sorry

end distance_to_place_l67_67904


namespace find_f_neg2007_l67_67085

variable (f : ℝ → ℝ)

-- Conditions
axiom cond1 (x y w : ℝ) (hx : x > y) (hw : f x + x ≥ w ∧ w ≥ f y + y) : 
  ∃ z ∈ Set.Icc y x, f z = w - z

axiom cond2 : ∃ u, f u = 0 ∧ ∀ v, f v = 0 → u ≤ v

axiom cond3 : f 0 = 1

axiom cond4 : f (-2007) ≤ 2008

axiom cond5 (x y : ℝ) : f x * f y = f (x * f y + y * f x + x * y)

theorem find_f_neg2007 : f (-2007) = 2008 := 
sorry

end find_f_neg2007_l67_67085


namespace incorrect_statement_A_l67_67090

-- We need to prove that statement (A) is incorrect given the provided conditions.

theorem incorrect_statement_A :
  ¬(∀ (a b : ℝ), a > b → ∀ (c : ℝ), c < 0 → a * c > b * c ∧ a / c > b / c) := 
sorry

end incorrect_statement_A_l67_67090


namespace min_value_expression_l67_67698

theorem min_value_expression (a b c : ℝ) 
  (h1 : a > b) 
  (h2 : b > c) 
  (h3 : (a - b) * (b - c) * (c - a) = -16) : 
  ∃ x : ℝ, x = (1 / (a - b)) + (1 / (b - c)) - (1 / (c - a)) ∧ x = 5 / 4 :=
by
  sorry

end min_value_expression_l67_67698


namespace red_balloon_is_one_l67_67439

open Nat

theorem red_balloon_is_one (R B : Nat) (h1 : R + B = 85) (h2 : R ≥ 1) (h3 : ∀ i j, i < R → j < R → i ≠ j → (i < B ∨ j < B)) : R = 1 :=
by
  sorry

end red_balloon_is_one_l67_67439


namespace find_S_11_l67_67688

variables (a : ℕ → ℤ)
variables (d : ℤ) (n : ℕ)

def is_arithmetic_sequence : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (n : ℕ) : ℤ :=
  (n * (a 1 + a n)) / 2

noncomputable def a_3 := a 3
noncomputable def a_6 := a 6
noncomputable def a_9 := a 9

theorem find_S_11
  (h1 : is_arithmetic_sequence a d)
  (h2 : a_3 + a_9 = 18 - a_6) :
  sum_first_n_terms a 11 = 66 :=
sorry

end find_S_11_l67_67688


namespace frozenFruitSold_l67_67251

variable (totalFruit : ℕ) (freshFruit : ℕ)

-- Define the condition that the total fruit sold is 9792 pounds
def totalFruitSold := totalFruit = 9792

-- Define the condition that the fresh fruit sold is 6279 pounds
def freshFruitSold := freshFruit = 6279

-- Define the question as a Lean statement
theorem frozenFruitSold
  (h1 : totalFruitSold totalFruit)
  (h2 : freshFruitSold freshFruit) :
  totalFruit - freshFruit = 3513 := by
  sorry

end frozenFruitSold_l67_67251


namespace number_of_boys_l67_67852

def initial_girls : ℕ := 706
def new_girls : ℕ := 418
def total_pupils : ℕ := 1346
def total_girls := initial_girls + new_girls

theorem number_of_boys : 
  total_pupils = total_girls + 222 := 
by
  sorry

end number_of_boys_l67_67852


namespace fraction_div_addition_l67_67124

noncomputable def fraction_5_6 : ℚ := 5 / 6
noncomputable def fraction_9_10 : ℚ := 9 / 10
noncomputable def fraction_1_15 : ℚ := 1 / 15
noncomputable def fraction_402_405 : ℚ := 402 / 405

theorem fraction_div_addition :
  (fraction_5_6 / fraction_9_10) + fraction_1_15 = fraction_402_405 :=
by
  sorry

end fraction_div_addition_l67_67124


namespace largest_number_of_square_plots_l67_67624

theorem largest_number_of_square_plots (n : ℕ) 
  (field_length : ℕ := 30) 
  (field_width : ℕ := 60) 
  (total_fence : ℕ := 2400) 
  (square_length : ℕ := field_length / n) 
  (fencing_required : ℕ := 60 * n) :
  field_length % n = 0 → 
  field_width % square_length = 0 → 
  fencing_required = total_fence → 
  2 * n^2 = 3200 :=
by
  intros h1 h2 h3
  sorry

end largest_number_of_square_plots_l67_67624


namespace maia_daily_client_requests_l67_67242

theorem maia_daily_client_requests (daily_requests : ℕ) (remaining_requests : ℕ) (days : ℕ) 
  (received_requests : ℕ) (total_requests : ℕ) (worked_requests : ℕ) :
  (daily_requests = 6) →
  (remaining_requests = 10) →
  (days = 5) →
  (received_requests = daily_requests * days) →
  (total_requests = received_requests - remaining_requests) →
  (worked_requests = total_requests / days) →
  worked_requests = 4 :=
by
  sorry

end maia_daily_client_requests_l67_67242


namespace cube_face_coloring_l67_67210

-- Define the type of a cube's face coloring
inductive FaceColor
| black
| white

open FaceColor

def countDistinctColorings : Nat :=
  -- Function to count the number of distinct colorings considering rotational symmetry
  10

theorem cube_face_coloring :
  countDistinctColorings = 10 :=
by
  -- Skip the proof, indicating it should be proved.
  sorry

end cube_face_coloring_l67_67210


namespace distance_from_dormitory_to_city_l67_67455

theorem distance_from_dormitory_to_city (D : ℝ)
  (h1 : (1 / 5) * D + (2 / 3) * D + 4 = D) : D = 30 := by
  sorry

end distance_from_dormitory_to_city_l67_67455


namespace annual_interest_rate_approx_l67_67588

-- Definitions of the variables
def FV : ℝ := 1764    -- Face value of the bill
def TD : ℝ := 189     -- True discount
def PV : ℝ := FV - TD -- Present value, calculated as per the problem statement

-- Simple interest formula components
def P : ℝ := PV       -- Principal
def T : ℝ := 9 / 12   -- Time period in years

-- Given conditions as definitions:
def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

-- Statement to prove that the annual interest rate R equals 16%
theorem annual_interest_rate_approx : ∃ R : ℝ, simple_interest P R T = TD ∧ R ≈ 16 := by
  use 16
  sorry

end annual_interest_rate_approx_l67_67588


namespace room_width_l67_67713

theorem room_width (W : ℝ) (L : ℝ := 17) (veranda_width : ℝ := 2) (veranda_area : ℝ := 132) :
  (21 * (W + veranda_width) - L * W = veranda_area) → W = 12 :=
by
  -- setup of the problem
  have total_length := L + 2 * veranda_width
  have total_width := W + 2 * veranda_width
  have area_room_incl_veranda := total_length * total_width - (L * W)
  -- the statement is already provided in the form of the theorem to be proven
  sorry

end room_width_l67_67713


namespace area_triangle_PQR_eq_2sqrt2_l67_67103

noncomputable def areaOfTrianglePQR : ℝ :=
  let sideAB := 3
  let altitudeAE := 6
  let EB := Real.sqrt (sideAB^2 + altitudeAE^2)
  let ED := EB
  let EC := Real.sqrt ((sideAB * Real.sqrt 2)^2 + altitudeAE^2)
  let EP := (2 / 3) * EB
  let EQ := EP
  let ER := (1 / 3) * EC
  let PR := Real.sqrt (ER^2 + EP^2 - 2 * ER * EP * (EB^2 + EC^2 - sideAB^2) / (2 * EB * EC))
  let PQ := 2
  let RS := Real.sqrt (PR^2 - (PQ / 2)^2)
  (1 / 2) * PQ * RS

theorem area_triangle_PQR_eq_2sqrt2 : areaOfTrianglePQR = 2 * Real.sqrt 2 :=
  sorry

end area_triangle_PQR_eq_2sqrt2_l67_67103


namespace probability_number_greater_than_3_from_0_5_l67_67378

noncomputable def probability_number_greater_than_3_in_0_5 : ℝ :=
  let total_interval_length := 5 - 0
  let event_interval_length := 5 - 3
  event_interval_length / total_interval_length

theorem probability_number_greater_than_3_from_0_5 :
  probability_number_greater_than_3_in_0_5 = 2 / 5 :=
by
  sorry

end probability_number_greater_than_3_from_0_5_l67_67378


namespace family_can_purchase_furniture_in_april_l67_67897

noncomputable def monthly_income : ℤ := 150000
noncomputable def monthly_expenses : ℤ := 115000
noncomputable def initial_savings : ℤ := 45000
noncomputable def furniture_cost : ℤ := 127000

theorem family_can_purchase_furniture_in_april : 
  ∃ (months : ℕ), months = 3 ∧ 
  (initial_savings + months * (monthly_income - monthly_expenses) >= furniture_cost) :=
by
  -- proof will be written here
  sorry

end family_can_purchase_furniture_in_april_l67_67897


namespace angle_between_tangents_l67_67912

theorem angle_between_tangents (R1 R2 : ℝ) (k : ℝ) (h_ratio : R1 = 2 * k ∧ R2 = 3 * k)
  (h_touching : (∃ O1 O2 : ℝ, (R2 - R1 = k))) : 
  ∃ θ : ℝ, θ = 90 := sorry

end angle_between_tangents_l67_67912


namespace index_card_area_reduction_l67_67014

theorem index_card_area_reduction :
  ∀ (length width : ℕ),
  (length = 5 ∧ width = 7) →
  ((length - 2) * width = 21) →
  (length * (width - 2) = 25) :=
by
  intros length width h1 h2
  rcases h1 with ⟨h_length, h_width⟩
  sorry

end index_card_area_reduction_l67_67014


namespace star_five_three_l67_67334

def star (a b : ℤ) : ℤ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end star_five_three_l67_67334


namespace tetrahedron_volume_le_one_eight_l67_67427

theorem tetrahedron_volume_le_one_eight {A B C D : Type} 
  (e₁_AB e₂_AC e₃_AD e₄_BC e₅_BD : ℝ) (h₁ : e₁_AB ≤ 1) (h₂ : e₂_AC ≤ 1) (h₃ : e₃_AD ≤ 1)
  (h₄ : e₄_BC ≤ 1) (h₅ : e₅_BD ≤ 1) : 
  ∃ (vol : ℝ), vol ≤ 1 / 8 :=
sorry

end tetrahedron_volume_le_one_eight_l67_67427


namespace turtle_hare_race_headstart_l67_67088

noncomputable def hare_time_muddy (distance speed_reduction hare_speed : ℝ) : ℝ :=
  distance / (hare_speed * speed_reduction)

noncomputable def hare_time_sandy (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def hare_time_regular (distance hare_speed : ℝ) : ℝ :=
  distance / hare_speed

noncomputable def turtle_time_muddy (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def turtle_time_sandy (distance speed_increase turtle_speed : ℝ) : ℝ :=
  distance / (turtle_speed * speed_increase)

noncomputable def turtle_time_regular (distance turtle_speed : ℝ) : ℝ :=
  distance / turtle_speed

noncomputable def hare_total_time (hare_speed : ℝ) : ℝ :=
  hare_time_muddy 20 0.5 hare_speed + hare_time_sandy 10 hare_speed + hare_time_regular 20 hare_speed

noncomputable def turtle_total_time (turtle_speed : ℝ) : ℝ :=
  turtle_time_muddy 20 turtle_speed + turtle_time_sandy 10 1.5 turtle_speed + turtle_time_regular 20 turtle_speed

theorem turtle_hare_race_headstart (hare_speed turtle_speed : ℝ) (t_hs : ℝ) :
  hare_speed = 10 →
  turtle_speed = 1 →
  t_hs = 39.67 →
  hare_total_time hare_speed + t_hs = turtle_total_time turtle_speed :=
by
  intros 
  sorry

end turtle_hare_race_headstart_l67_67088


namespace distance_walked_by_friend_P_l67_67139

def trail_length : ℝ := 33
def speed_ratio : ℝ := 1.20

theorem distance_walked_by_friend_P (v t d_P : ℝ) 
  (h1 : t = 33 / (2.20 * v)) 
  (h2 : d_P = 1.20 * v * t) 
  : d_P = 18 := by
  sorry

end distance_walked_by_friend_P_l67_67139


namespace least_positive_integer_condition_l67_67918

theorem least_positive_integer_condition :
  ∃ (n : ℕ), n > 0 ∧ (n % 2 = 1) ∧ (n % 5 = 4) ∧ (n % 7 = 6) ∧ n = 69 :=
by
  sorry

end least_positive_integer_condition_l67_67918


namespace probability_of_rolling_less_than_5_l67_67743

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l67_67743


namespace hardware_contract_probability_l67_67762

noncomputable def P_S' : ℚ := 3 / 5
noncomputable def P_at_least_one : ℚ := 5 / 6
noncomputable def P_H_and_S : ℚ := 0.31666666666666654 -- 19 / 60 in fraction form
noncomputable def P_S : ℚ := 1 - P_S'

theorem hardware_contract_probability :
  (P_at_least_one = P_H + P_S - P_H_and_S) →
  P_H = 0.75 :=
by
  sorry

end hardware_contract_probability_l67_67762


namespace perimeter_of_original_square_l67_67320

-- Definitions
variables {x : ℝ}
def rect_width := x
def rect_length := 4 * x
def rect_perimeter := 56
def original_square_perimeter := 32

-- Statement
theorem perimeter_of_original_square (x : ℝ) (h : 28 * x = 56) : 4 * (4 * x) = 32 :=
by
  -- Since the proof is not required, we apply sorry to end the theorem.
  sorry

end perimeter_of_original_square_l67_67320


namespace inequality_of_reals_l67_67798

theorem inequality_of_reals (a b c d : ℝ) : 
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := 
  sorry

end inequality_of_reals_l67_67798


namespace number_of_triples_l67_67770

theorem number_of_triples : 
  ∃ n : ℕ, 
  n = 2 ∧
  ∀ (a b c : ℕ), 
    (2 ≤ a ∧ a ≤ b ∧ b ≤ c) →
    (a * b * c = 4 * (a * b + b * c + c * a)) →
    n = 2 :=
sorry

end number_of_triples_l67_67770


namespace pizza_volume_l67_67318

theorem pizza_volume (h : ℝ) (d : ℝ) (n : ℕ) 
  (h_cond : h = 1/2) 
  (d_cond : d = 16) 
  (n_cond : n = 8) 
  : (π * (d / 2) ^ 2 * h / n = 4 * π) :=
by
  sorry

end pizza_volume_l67_67318


namespace negation_of_proposition_is_false_l67_67428

theorem negation_of_proposition_is_false :
  (¬ ∀ (x : ℝ), x < 0 → x^2 > 0) = true :=
by
  sorry

end negation_of_proposition_is_false_l67_67428


namespace income_percentage_l67_67879

theorem income_percentage (J T M : ℝ) 
  (h1 : T = 0.5 * J) 
  (h2 : M = 1.6 * T) : 
  M = 0.8 * J :=
by 
  sorry

end income_percentage_l67_67879


namespace abc_equality_l67_67104

theorem abc_equality (a b c : ℕ) (h1 : b = a^2 - a) (h2 : c = b^2 - b) (h3 : a = c^2 - c) : 
  a = 2 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end abc_equality_l67_67104


namespace C_total_days_l67_67760

-- Conditions
def A_days : ℕ := 30
def B_days : ℕ := 30
def A_worked_days : ℕ := 10
def B_worked_days : ℕ := 10
def C_finished_days : ℕ := 10

-- Work rates
def A_work_rate : ℚ := 1 / A_days
def B_work_rate : ℚ := 1 / B_days

-- Work done
def A_work_done : ℚ := A_work_rate * A_worked_days
def B_work_done : ℚ := B_work_rate * B_worked_days
def total_work_done : ℚ := A_work_done + B_work_done
def work_left_for_C : ℚ := 1 - total_work_done
def C_work_rate : ℚ := work_left_for_C / C_finished_days

-- Equivalent proof problem
theorem C_total_days (d : ℕ) (h : C_work_rate = (1 : ℚ) / d) : d = 30 :=
by sorry

end C_total_days_l67_67760


namespace compare_negatives_l67_67652

theorem compare_negatives : -2 < -3 / 2 :=
by sorry

end compare_negatives_l67_67652


namespace cos_3theta_l67_67825

theorem cos_3theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (3 * θ) = -11 / 16 := by
  sorry

end cos_3theta_l67_67825


namespace probability_one_blue_one_white_one_red_l67_67758

theorem probability_one_blue_one_white_one_red :
  let total_marbles := 16
  let marbles_to_draw := 13
  let combinations := Nat.choose
  let total_ways := combinations total_marbles marbles_to_draw
  let favorable_ways :=
    (combinations 5 4) * (combinations 7 6) * (combinations 4 3)
  (total_ways > 0) → 
  (favorable_ways / total_ways) = (1 : ℚ) / (8 : ℚ) := 
by
  sorry

end probability_one_blue_one_white_one_red_l67_67758


namespace line_equation_l67_67415

-- Given conditions
variables (k x x0 y y0 : ℝ)
variable (line_passes_through : ∀ x0 y0, y0 = k * x0 + l)
variable (M0 : (ℝ × ℝ))

-- Main statement we need to prove
theorem line_equation (k x x0 y y0 : ℝ) (M0 : (ℝ × ℝ)) (line_passes_through : ∀ x0 y0, y0 = k * x0 + l) :
  y - y0 = k * (x - x0) :=
sorry

end line_equation_l67_67415


namespace total_pennies_thrown_l67_67093

theorem total_pennies_thrown (R G X M T : ℝ) (hR : R = 1500)
  (hG : G = (2 / 3) * R) (hX : X = (3 / 4) * G) 
  (hM : M = 3.5 * X) (hT : T = (4 / 5) * M) : 
  R + G + X + M + T = 7975 :=
by
  sorry

end total_pennies_thrown_l67_67093


namespace multiply_scaled_values_l67_67754

theorem multiply_scaled_values (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by 
  sorry

end multiply_scaled_values_l67_67754


namespace geometric_sequence_ratio_l67_67512

theorem geometric_sequence_ratio (a1 : ℕ) (S : ℕ → ℕ) (r : ℤ) (h1 : r = -2) (h2 : ∀ n, S n = a1 * (1 - r ^ n) / (1 - r)) :
  S 4 / S 2 = 5 :=
by
  -- Placeholder for proof steps
  sorry

end geometric_sequence_ratio_l67_67512


namespace Masha_thought_of_numbers_l67_67399

theorem Masha_thought_of_numbers : ∃ a b : ℕ, a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a + b = 28) ∧ (a % 2 = 0 ∨ b % 2 = 0) ∧ (a = 12 ∧ b = 16 ∨ a = 16 ∧ b = 12) :=
by
  sorry

end Masha_thought_of_numbers_l67_67399


namespace jade_savings_per_month_l67_67831

def jade_monthly_income : ℝ := 1600
def jade_living_expense_rate : ℝ := 0.75
def jade_insurance_rate : ℝ := 0.2

theorem jade_savings_per_month : 
  jade_monthly_income * (1 - jade_living_expense_rate - jade_insurance_rate) = 80 := by
  sorry

end jade_savings_per_month_l67_67831


namespace marble_catch_up_time_l67_67121

theorem marble_catch_up_time 
    (a b c : ℝ) 
    (L : ℝ)
    (h1 : a - b = L / 50)
    (h2 : a - c = L / 40) 
    : (110 * (c - b)) / (c - b) = 110 := 
by 
    sorry

end marble_catch_up_time_l67_67121


namespace average_age_union_l67_67604

theorem average_age_union (students_A students_B students_C : ℕ)
  (sumA sumB sumC : ℕ) (avgA avgB avgC avgAB avgAC avgBC : ℚ)
  (hA : avgA = (sumA : ℚ) / students_A)
  (hB : avgB = (sumB : ℚ) / students_B)
  (hC : avgC = (sumC : ℚ) / students_C)
  (hAB : avgAB = (sumA + sumB) / (students_A + students_B))
  (hAC : avgAC = (sumA + sumC) / (students_A + students_C))
  (hBC : avgBC = (sumB + sumC) / (students_B + students_C))
  (h_avgA: avgA = 34)
  (h_avgB: avgB = 25)
  (h_avgC: avgC = 45)
  (h_avgAB: avgAB = 30)
  (h_avgAC: avgAC = 42)
  (h_avgBC: avgBC = 36) :
  (sumA + sumB + sumC : ℚ) / (students_A + students_B + students_C) = 33 := 
  sorry

end average_age_union_l67_67604


namespace exp_abs_dev_10_gt_100_l67_67292

open ProbabilityTheory
open MeasureTheory

def deviation (n m : ℕ) : ℝ := (m / n : ℝ) - 0.5
def abs_deviation (n m : ℕ) : ℝ := abs ((m / n : ℝ) - 0.5)
def exp_abs_deviation (n : ℕ) : ℝ := sorry -- The expected value of the absolute deviation

theorem exp_abs_dev_10_gt_100 :
  (exp_abs_deviation 10) > (exp_abs_deviation 100) :=
sorry

end exp_abs_dev_10_gt_100_l67_67292


namespace not_perfect_square_l67_67419

theorem not_perfect_square (a : ℤ) : ¬ (∃ x : ℤ, a^2 + 4 = x^2) := 
sorry

end not_perfect_square_l67_67419


namespace gcd_sequence_inequality_l67_67560

-- Add your Lean 4 statement here
theorem gcd_sequence_inequality {n : ℕ} 
  (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 35 → Nat.gcd n k < Nat.gcd n (k+1)) : 
  Nat.gcd n 35 < Nat.gcd n 36 := 
sorry

end gcd_sequence_inequality_l67_67560


namespace value_of_x_l67_67056

theorem value_of_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end value_of_x_l67_67056


namespace tetrahedron_cut_off_vertices_l67_67659

theorem tetrahedron_cut_off_vertices :
  ∀ (V E : ℕ) (cut_effect : ℕ → ℕ),
    -- Initial conditions
    V = 4 → E = 6 →
    -- Effect of each cut (cutting one vertex introduces 3 new edges)
    (∀ v, v ≤ V → cut_effect v = 3 * v) →
    -- Prove the number of edges in the new figure
    (E + cut_effect V) = 18 :=
by
  intros V E cut_effect hV hE hcut
  sorry

end tetrahedron_cut_off_vertices_l67_67659


namespace probability_of_event_E_l67_67738

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l67_67738


namespace matrix_vec_addition_l67_67653

def matrix := (Fin 2 → Fin 2 → ℤ)
def vector := Fin 2 → ℤ

def m : matrix := ![![4, -2], ![6, 5]]
def v1 : vector := ![-2, 3]
def v2 : vector := ![1, -1]

def matrix_vec_mul (m : matrix) (v : vector) : vector :=
  ![m 0 0 * v 0 + m 0 1 * v 1,
    m 1 0 * v 0 + m 1 1 * v 1]

def vec_add (v1 v2 : vector) : vector :=
  ![v1 0 + v2 0, v1 1 + v2 1]

theorem matrix_vec_addition :
  vec_add (matrix_vec_mul m v1) v2 = ![-13, 2] :=
by
  sorry

end matrix_vec_addition_l67_67653


namespace number_of_maple_trees_planted_l67_67907

def before := 53
def after := 64
def planted := after - before

theorem number_of_maple_trees_planted : planted = 11 := by
  sorry

end number_of_maple_trees_planted_l67_67907


namespace birds_count_is_30_l67_67478

def total_animals : ℕ := 77
def number_of_kittens : ℕ := 32
def number_of_hamsters : ℕ := 15

def number_of_birds : ℕ := total_animals - number_of_kittens - number_of_hamsters

theorem birds_count_is_30 : number_of_birds = 30 := by
  sorry

end birds_count_is_30_l67_67478


namespace largest_polygon_area_l67_67195

variable (area : ℕ → ℝ)

def polygon_A_area : ℝ := 6
def polygon_B_area : ℝ := 3 + 4 * 0.5
def polygon_C_area : ℝ := 4 + 5 * 0.5
def polygon_D_area : ℝ := 7
def polygon_E_area : ℝ := 2 + 6 * 0.5

theorem largest_polygon_area : polygon_D_area = max (max (max polygon_A_area polygon_B_area) polygon_C_area) polygon_E_area :=
by
  sorry

end largest_polygon_area_l67_67195


namespace current_walnut_trees_l67_67271

theorem current_walnut_trees (x : ℕ) (h : x + 55 = 77) : x = 22 :=
by
  sorry

end current_walnut_trees_l67_67271


namespace rotameter_percentage_l67_67011

theorem rotameter_percentage (l_inch_flow : ℝ) (l_liters_flow : ℝ) (g_inch_flow : ℝ) (g_liters_flow : ℝ) :
  l_inch_flow = 2.5 → l_liters_flow = 60 → g_inch_flow = 4 → g_liters_flow = 192 → 
  (g_liters_flow / g_inch_flow) / (l_liters_flow / l_inch_flow) * 100 = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end rotameter_percentage_l67_67011


namespace dice_probability_event_l67_67623

/-- A fair 8-sided die is rolled until a number 5 or greater appears.
    Calculate the probability that both numbers 2 and 4 appear at least once
    before any number from 5 to 8 is rolled.--/
theorem dice_probability_event (fair_die : Type) [fintype fair_die] [decidable_eq fair_die] 
  (P_number_2_4 : fair_die → Prop) (P_number_5_8 : fair_die → Prop) 
  [decidable_pred P_number_2_4] [decidable_pred P_number_5_8]
  (H_die : (∀ x : fair_die, P_number_2_4 x ∨ P_number_5_8 x) ∧
            (∀ x : fair_die, ¬P_number_2_4 x ∨ ¬P_number_5_8 x)) :
  (∑' (n : ℕ) (h : n ≥ 3), (1 / 2 ^ n) * ((2 ^ (n - 1)) / 2 ^ (n - 1))) = 1 / 4 :=
by
  sorry

end dice_probability_event_l67_67623


namespace simplify_expression_simplify_and_evaluate_evaluate_expression_l67_67887

theorem simplify_expression (a b : ℝ) : 8 * (a + b) + 6 * (a + b) - 2 * (a + b) = 12 * (a + b) := 
by sorry

theorem simplify_and_evaluate (x y : ℝ) (h : x + y = 1/2) : 
  9 * (x + y)^2 + 3 * (x + y) + 7 * (x + y)^2 - 7 * (x + y) = 2 := 
by sorry

theorem evaluate_expression (x y : ℝ) (h : x^2 - 2 * y = 4) : -3 * x^2 + 6 * y + 2 = -10 := 
by sorry

end simplify_expression_simplify_and_evaluate_evaluate_expression_l67_67887


namespace all_radii_equal_l67_67432
-- Lean 4 statement

theorem all_radii_equal (r : ℝ) (h : r = 2) : r = 2 :=
by
  sorry

end all_radii_equal_l67_67432


namespace floor_eq_solution_l67_67981

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l67_67981


namespace value_of_x_l67_67058

theorem value_of_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end value_of_x_l67_67058


namespace min_value_a_over_b_l67_67800

theorem min_value_a_over_b (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 2 * Real.sqrt a + b = 1) : ∃ c, c = 0 := 
by
  -- We need to show that the minimum value of a / b is 0 
  sorry

end min_value_a_over_b_l67_67800


namespace calculate_result_l67_67167

theorem calculate_result : (-3 : ℝ)^(2022) * (1 / 3 : ℝ)^(2023) = 1 / 3 := 
by sorry

end calculate_result_l67_67167


namespace line_passes_through_fixed_point_l67_67524

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y = k * x + 2 * k + 1 :=
by
  sorry

end line_passes_through_fixed_point_l67_67524


namespace temperature_below_zero_l67_67219

theorem temperature_below_zero (t₁ t₂ : ℤ) (h₁ : t₁ = 4) (h₂ : t₂ = -6) :
  (h₁ = 4 → true) → (h₂ = -6 → true) :=
sorry

end temperature_below_zero_l67_67219


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l67_67972

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l67_67972


namespace masha_numbers_l67_67394

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l67_67394


namespace james_total_matches_l67_67857

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end james_total_matches_l67_67857


namespace proof_problem_l67_67834

theorem proof_problem (p q r : ℝ) 
  (h1 : p + q = 20)
  (h2 : p * q = 144) 
  (h3 : q + r = 52) 
  (h4 : 4 * (r + p) = r * p) : 
  r - p = 32 := 
sorry

end proof_problem_l67_67834


namespace range_of_a_l67_67672

theorem range_of_a (a : ℝ) :
  let f (x : ℝ) := if x ≥ 0 then x^2 - 2*x + 2 else x + a/x + 3*a in
  (∀ y ∈ set.range f, ∃ x, f x = y) ↔ (a ∈ set.Iic 0 ∪ set.Ici 1) :=
by sorry

end range_of_a_l67_67672


namespace gingerbreads_per_tray_l67_67657

-- Given conditions
def total_baked_gb (x : ℕ) : Prop := 4 * 25 + 3 * x = 160

-- The problem statement
theorem gingerbreads_per_tray (x : ℕ) (h : total_baked_gb x) : x = 20 := 
by sorry

end gingerbreads_per_tray_l67_67657


namespace m_above_x_axis_m_on_line_l67_67994

namespace ComplexNumberProblem

def above_x_axis (m : ℝ) : Prop :=
  m^2 - 2 * m - 15 > 0

def on_line (m : ℝ) : Prop :=
  2 * m^2 + 3 * m - 4 = 0

theorem m_above_x_axis (m : ℝ) : above_x_axis m → (m < -3 ∨ m > 5) :=
  sorry

theorem m_on_line (m : ℝ) : on_line m → 
  (m = (-3 + Real.sqrt 41) / 4) ∨ (m = (-3 - Real.sqrt 41) / 4) :=
  sorry

end ComplexNumberProblem

end m_above_x_axis_m_on_line_l67_67994


namespace problem1_problem2_l67_67459

-- Proof for Problem 1
theorem problem1 : (99^2 + 202*99 + 101^2) = 40000 := 
by {
  -- proof
  sorry
}

-- Proof for Problem 2
theorem problem2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : ((1 / (x - 1) - 2) / ((2 * x - 3) / (x^2 - 1))) = -x - 1 :=
by {
  -- proof
  sorry
}

end problem1_problem2_l67_67459


namespace intersection_M_complement_N_l67_67046

open Set Real

def M : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}
def N : Set ℝ := {x | (Real.log 2) ^ (1 - x) < 1}
def complement_N := {x : ℝ | x ≥ 1}

theorem intersection_M_complement_N :
  M ∩ complement_N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_complement_N_l67_67046


namespace parallelepiped_diagonal_relationship_l67_67735

theorem parallelepiped_diagonal_relationship {a b c d e f g : ℝ} 
  (h1 : c = d) 
  (h2 : e = e) 
  (h3 : f = f) 
  (h4 : g = g) 
  : a^2 + b^2 + c^2 + g^2 = d^2 + e^2 + f^2 :=
by
  sorry

end parallelepiped_diagonal_relationship_l67_67735


namespace math_problem_l67_67584

open Real

variable (x : ℝ)
variable (h : x + 1 / x = sqrt 3)

theorem math_problem : x^7 - 3 * x^5 + x^2 = -5 * x + 4 * sqrt 3 :=
by sorry

end math_problem_l67_67584


namespace symmetric_points_power_l67_67547

theorem symmetric_points_power 
  (a b : ℝ) 
  (h1 : 2 * a = 8) 
  (h2 : 2 = a + b) :
  a^b = 1/16 := 
by sorry

end symmetric_points_power_l67_67547


namespace car_speed_after_modifications_l67_67863

theorem car_speed_after_modifications (s : ℕ) (p : ℝ) (w : ℕ) :
  s = 150 →
  p = 0.30 →
  w = 10 →
  s + (p * s) + w = 205 := 
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  done

end car_speed_after_modifications_l67_67863


namespace ax_by_power5_l67_67049

-- Define the real numbers a, b, x, and y
variables (a b x y : ℝ)

-- Define the conditions as assumptions
axiom axiom1 : a * x + b * y = 3
axiom axiom2 : a * x^2 + b * y^2 = 7
axiom axiom3 : a * x^3 + b * y^3 = 16
axiom axiom4 : a * x^4 + b * y^4 = 42

-- State the theorem to prove ax^5 + by^5 = 20
theorem ax_by_power5 : a * x^5 + b * y^5 = 20 :=
  sorry

end ax_by_power5_l67_67049


namespace arithmetic_sequence_solution_l67_67070

variable {a_n : ℕ → ℚ}
variable {a_1 : ℚ}
variable {a_2 a_5 : ℚ}
variable {n : ℕ}
variable {S_n : ℕ → ℚ}
variable {d : ℚ}

-- Conditions
def condition_1 : a_1 = (1 / 3) := by sorry
def condition_2 : a_2 + a_5 = 4 := by sorry
def condition_3 : a_n 50 = 33 := by sorry

-- Arithmetical definitions
def common_difference : d := by sorry
def term_n (n : ℕ) : ℚ := a_1 + (n - 1) * d
def sum_n (n : ℕ) : ℚ := n * (a_1 + term_n n) / 2

-- Question
theorem arithmetic_sequence_solution : 
  condition_1 → 
  condition_2 → 
  condition_3 → 
  (n = 50) ∧ (S_n n = 850) := 
begin
  intros,
  -- Necessary steps can be added here
  sorry
end

end arithmetic_sequence_solution_l67_67070


namespace bowler_overs_l67_67937

theorem bowler_overs (x : ℕ) (h1 : ∀ y, y ≤ 3 * x) 
                     (h2 : y = 10) : x = 4 := by
  sorry

end bowler_overs_l67_67937


namespace sum_first_10_terms_eq_l67_67806

def sequence_a : ℕ → ℚ
| 1     := 1
| (n+1) := sequence_a n + (n + 1)

noncomputable def sum_first_10_terms : ℚ :=
  (Finset.range 10).sum (λ i, 1 / sequence_a (i + 1))

theorem sum_first_10_terms_eq : sum_first_10_terms = 20 / 11 :=
by
  sorry

end sum_first_10_terms_eq_l67_67806


namespace people_count_l67_67563

theorem people_count (wheels_per_person total_wheels : ℕ) (h1 : wheels_per_person = 4) (h2 : total_wheels = 320) :
  total_wheels / wheels_per_person = 80 :=
sorry

end people_count_l67_67563


namespace original_square_area_l67_67190

noncomputable def area_of_original_square (x : ℝ) (w : ℝ) (remaining_area : ℝ) : Prop :=
  x * x = remaining_area + x * w

theorem original_square_area : ∃ (x : ℝ), area_of_original_square x 3 40 → x * x = 64 :=
sorry

end original_square_area_l67_67190


namespace find_value_of_m_l67_67610

variable (i m : ℂ)

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_value_of_m 
  (i_imag : i.im = 1 ∧ i.re = 0)
  (pure_imag : is_pure_imaginary ((3 - i) * (m + i))) :
  m = - (1 / 3) :=
sorry

end find_value_of_m_l67_67610


namespace range_of_f_l67_67388

open Real

noncomputable def f (x y z w : ℝ) : ℝ :=
  x / (x + y) + y / (y + z) + z / (z + x) + w / (w + x)

theorem range_of_f (x y z w : ℝ) (h1x : 0 < x) (h1y : 0 < y) (h1z : 0 < z) (h1w : 0 < w) :
  1 < f x y z w ∧ f x y z w < 2 :=
  sorry

end range_of_f_l67_67388


namespace geometric_sequence_y_value_l67_67196

theorem geometric_sequence_y_value (q : ℝ) 
  (h1 : 2 * q^4 = 18) 
  : 2 * (q^2) = 6 := by
  sorry

end geometric_sequence_y_value_l67_67196


namespace min_value_reciprocal_l67_67353

variable {a b : ℝ}

theorem min_value_reciprocal (h1 : a * b > 0) (h2 : a + 4 * b = 1) : 
  ∃ (a b : ℝ), (a > 0) ∧ (b > 0) ∧ ((1/a) + (1/b) = 9) := 
by
  sorry

end min_value_reciprocal_l67_67353


namespace time_to_cross_first_platform_l67_67635

-- Define the given conditions
def length_first_platform : ℕ := 140
def length_second_platform : ℕ := 250
def length_train : ℕ := 190
def time_cross_second_platform : Nat := 20
def speed := (length_train + length_second_platform) / time_cross_second_platform

-- The theorem to be proved
theorem time_to_cross_first_platform : 
  (length_train + length_first_platform) / speed = 15 :=
sorry

end time_to_cross_first_platform_l67_67635


namespace ratio_A_B_l67_67789

theorem ratio_A_B :
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by
  let A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
  let B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)
  have hA : A = 1400 := rfl
  have hB : B = 1500 := rfl
  have h_ratio : A / B = 14 / 15 := by
    rw [hA, hB]
    norm_num
  exact And.intro (by norm_num) (by norm_num)

end ratio_A_B_l67_67789


namespace intersection_with_x_axis_l67_67891

theorem intersection_with_x_axis :
  (∃ x, ∃ y, y = 0 ∧ y = -3 * x + 3 ∧ (x = 1 ∧ y = 0)) :=
by
  -- proof will go here
  sorry

end intersection_with_x_axis_l67_67891


namespace fish_count_seventh_day_l67_67551

-- Define the initial state and transformations
def fish_count (n: ℕ) :=
  if n = 0 then 6
  else
    if n = 3 then fish_count (n-1) / 3 * 2 * 2 * 2 - fish_count (n-1) / 3
    else if n = 5 then (fish_count (n-1) * 2) / 4 * 3
    else if n = 6 then fish_count (n-1) * 2 + 15
    else fish_count (n-1) * 2

theorem fish_count_seventh_day : fish_count 7 = 207 :=
by
  sorry

end fish_count_seventh_day_l67_67551


namespace mother_stickers_given_l67_67089

-- Definitions based on the conditions
def initial_stickers : ℝ := 20.0
def bought_stickers : ℝ := 26.0
def birthday_stickers : ℝ := 20.0
def sister_stickers : ℝ := 6.0
def total_stickers : ℝ := 130.0

-- Statement of the problem to be proved in Lean 4.
theorem mother_stickers_given :
  initial_stickers + bought_stickers + birthday_stickers + sister_stickers + 58.0 = total_stickers :=
by
  sorry

end mother_stickers_given_l67_67089


namespace red_paint_cans_l67_67881

theorem red_paint_cans (total_cans : ℕ) (ratio_red_blue : ℕ) (ratio_blue : ℕ) (h_ratio : ratio_red_blue = 4) (h_blue : ratio_blue = 1) (h_total_cans : total_cans = 50) : 
  (total_cans * ratio_red_blue) / (ratio_red_blue + ratio_blue) = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end red_paint_cans_l67_67881


namespace tan_240_eq_sqrt_3_l67_67170

open Real

noncomputable def Q : ℝ × ℝ := (-1/2, -sqrt(3)/2)

theorem tan_240_eq_sqrt_3 (h1 : Q = (-1/2, -sqrt(3)/2)) : 
  tan 240 = sqrt 3 :=
by
  sorry

end tan_240_eq_sqrt_3_l67_67170


namespace min_value_of_x_prime_factors_l67_67236

theorem min_value_of_x_prime_factors (x y a b c d : ℕ) (hx : x > 0) (hy : y > 0)
    (h : 5 * x^7 = 13 * y^11)
    (hx_factorization : x = a^c * b^d) : a + b + c + d = 32 := sorry

end min_value_of_x_prime_factors_l67_67236


namespace find_f_11_5_l67_67084

-- Definitions based on the conditions.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def periodic_with_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -1 / f x

def f_defined_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ -2 → f x = 2 * x

-- The main theorem to prove.
theorem find_f_11_5 (f : ℝ → ℝ) :
  is_even_function f →
  functional_eqn f →
  f_defined_on_interval f →
  periodic_with_period f 6 →
  f 11.5 = 1 / 5 :=
  by
    intros h_even h_fun_eqn h_interval h_periodic
    sorry  -- proof goes here

end find_f_11_5_l67_67084


namespace find_different_weighted_coins_l67_67926

-- Define the conditions and the theorem
def num_coins : Nat := 128
def weight_types : Nat := 2
def coins_of_each_weight : Nat := 64

theorem find_different_weighted_coins (weighings_at_most : Nat := 7) :
  ∃ (w1 w2 : Nat) (coins : Fin num_coins → Nat), w1 ≠ w2 ∧ 
  (∃ (pair : Fin num_coins × Fin num_coins), pair.fst ≠ pair.snd ∧ coins pair.fst ≠ coins pair.snd) :=
sorry

end find_different_weighted_coins_l67_67926


namespace sum_of_xyz_l67_67697

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : 1/x + y + z = 3) 
  (h2 : x + 1/y + z = 3) 
  (h3 : x + y + 1/z = 3) : 
  ∃ m n : ℕ, m = 9 ∧ n = 2 ∧ Nat.gcd m n = 1 ∧ 100 * m + n = 902 := 
sorry

end sum_of_xyz_l67_67697


namespace tan_pi_minus_alpha_l67_67677

/-- Given the conditions: tan(α + π / 4) = sin(2α) + cos^2(α) and α ∈ (π / 2, π),
    then prove tan(π - α) = 3 -/
theorem tan_pi_minus_alpha (α : ℝ) 
  (h1 : tan (α + π / 4) = sin (2 * α) + cos α ^ 2)
  (h2 : α ∈ (π / 2, π)) :
  tan (π - α) = 3 :=
sorry

end tan_pi_minus_alpha_l67_67677


namespace unfolded_paper_has_four_symmetrical_holes_l67_67482

structure Paper :=
  (width : ℤ) (height : ℤ) (hole_x : ℤ) (hole_y : ℤ)

structure Fold :=
  (direction : String) (fold_line : ℤ)

structure UnfoldedPaper :=
  (holes : List (ℤ × ℤ))

-- Define the initial paper, folds, and punching
def initial_paper : Paper := {width := 4, height := 6, hole_x := 2, hole_y := 1}
def folds : List Fold := 
  [{direction := "bottom_to_top", fold_line := initial_paper.height / 2}, 
   {direction := "left_to_right", fold_line := initial_paper.width / 2}]
def punch : (ℤ × ℤ) := (initial_paper.hole_x, initial_paper.hole_y)

-- The theorem to prove the resulting unfolded paper
theorem unfolded_paper_has_four_symmetrical_holes (p : Paper) (fs : List Fold) (punch : ℤ × ℤ) :
  UnfoldedPaper :=
  { holes := [(1, 1), (1, 5), (3, 1), (3, 5)] } -- Four symmetrically placed holes.

end unfolded_paper_has_four_symmetrical_holes_l67_67482


namespace sector_area_l67_67841

theorem sector_area (r l : ℝ) (h_r : r = 2) (h_l : l = 3) : (1/2) * l * r = 3 :=
by
  rw [h_r, h_l]
  norm_num

end sector_area_l67_67841


namespace root_of_quadratic_property_l67_67216

theorem root_of_quadratic_property (m : ℝ) (h : m^2 - 2 * m - 1 = 0) :
  m^2 + (1 / m^2) = 6 :=
sorry

end root_of_quadratic_property_l67_67216


namespace ian_saves_per_day_l67_67535

-- Let us define the given conditions
def total_saved : ℝ := 0.40 -- Ian saved a total of $0.40
def days : ℕ := 40 -- Ian saved for 40 days

-- Now, we need to prove that Ian saved 0.01 dollars/day
theorem ian_saves_per_day (h : total_saved = 0.40 ∧ days = 40) : total_saved / days = 0.01 :=
by
  sorry

end ian_saves_per_day_l67_67535


namespace triangle_ABC_area_l67_67965

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1, 2)
def C : point := (2, 0)

def triangle_area (A B C : point) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

theorem triangle_ABC_area :
  triangle_area A B C = 2 :=
by
  sorry

end triangle_ABC_area_l67_67965


namespace time_to_complete_together_l67_67505

-- Definitions for the given conditions
variables (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Theorem statement for the mathematically equivalent proof problem
theorem time_to_complete_together (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
   (1 : ℝ) / ((1 / x) + (1 / y)) = x * y / (x + y) :=
sorry

end time_to_complete_together_l67_67505


namespace three_digit_integer_conditions_l67_67531

theorem three_digit_integer_conditions:
  ∃ n : ℕ, 
    n % 5 = 3 ∧ 
    n % 7 = 4 ∧ 
    n % 4 = 2 ∧
    100 ≤ n ∧ n < 1000 ∧ 
    n = 548 :=
sorry

end three_digit_integer_conditions_l67_67531


namespace intersection_of_complement_l67_67270

open Set

theorem intersection_of_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6})
  (hA : A = {1, 3, 4}) (hB : B = {2, 3, 4, 5}) : A ∩ (U \ B) = {1} :=
by
  rw [hU, hA, hB]
  -- Proof steps go here
  sorry

end intersection_of_complement_l67_67270


namespace interval_monotonic_increase_max_min_values_range_of_m_l67_67208

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

-- The interval of monotonic increase for f(x)
theorem interval_monotonic_increase :
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} = 
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} := 
by sorry

-- Maximum and minimum values of f(x) when x ∈ [π/4, π/2]
theorem max_min_values (x : ℝ) (h : x ∈ Set.Icc (π / 4) (π / 2)) :
  (f x ≤ 0 ∧ (f x = 0 ↔ x = π / 3)) ∧ (f x ≥ -1/2 ∧ (f x = -1/2 ↔ x = π / 2)) :=
by sorry

-- Range of m for the inequality |f(x) - m| < 1 when x ∈ [π/4, π/2]
theorem range_of_m (m : ℝ) (h : ∀ x ∈ Set.Icc (π / 4) (π / 2), |f x - m| < 1) :
  m ∈ Set.Ioo (-1) (1/2) :=
by sorry

end interval_monotonic_increase_max_min_values_range_of_m_l67_67208


namespace v_not_closed_under_operations_l67_67240

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def v : Set ℕ := {n | ∃ m : ℕ, n = m * m}

def addition_followed_by_multiplication (a b : ℕ) : ℕ :=
  (a + b) * a

def multiplication_followed_by_addition (a b : ℕ) : ℕ :=
  (a * b) + a

def division_followed_by_subtraction (a b : ℕ) : ℕ :=
  if b ≠ 0 then (a / b) - b else 0

def extraction_root_followed_by_multiplication (a b : ℕ) : ℕ :=
  (Nat.sqrt a) * (Nat.sqrt b)

theorem v_not_closed_under_operations : 
  ¬ (∀ a ∈ v, ∀ b ∈ v, addition_followed_by_multiplication a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, multiplication_followed_by_addition a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, division_followed_by_subtraction a b ∈ v) ∧
  ¬ (∀ a ∈ v, ∀ b ∈ v, extraction_root_followed_by_multiplication a b ∈ v) :=
sorry

end v_not_closed_under_operations_l67_67240


namespace congruent_triangle_sides_l67_67042

variable {x y : ℕ}

theorem congruent_triangle_sides (h_congruent : ∃ (a b c d e f : ℕ), (a = x) ∧ (b = 2) ∧ (c = 6) ∧ (d = 5) ∧ (e = 6) ∧ (f = y) ∧ (a = d) ∧ (b = f) ∧ (c = e)) : 
  x + y = 7 :=
sorry

end congruent_triangle_sides_l67_67042


namespace range_of_a_l67_67670

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + 2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f' x ≥ a) → (a ≤ 2) :=
by
  sorry

end range_of_a_l67_67670


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l67_67971

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l67_67971


namespace tunnel_length_correct_l67_67433

noncomputable def tunnel_length (truck_length : ℝ) (time_to_exit : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
let speed_fps := (speed_mph * mile_to_feet) / 3600
let total_distance := speed_fps * time_to_exit
total_distance - truck_length

theorem tunnel_length_correct :
  tunnel_length 66 6 45 5280 = 330 :=
by
  sorry

end tunnel_length_correct_l67_67433


namespace determine_x_l67_67874

theorem determine_x (x : Nat) (h1 : x % 9 = 0) (h2 : x^2 > 225) (h3 : x < 30) : x = 18 ∨ x = 27 :=
sorry

end determine_x_l67_67874


namespace original_square_area_l67_67188

noncomputable def area_of_original_square (x : ℝ) (w : ℝ) (remaining_area : ℝ) : Prop :=
  x * x = remaining_area + x * w

theorem original_square_area : ∃ (x : ℝ), area_of_original_square x 3 40 → x * x = 64 :=
sorry

end original_square_area_l67_67188


namespace f_at_7_l67_67200

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_property : ∀ x, f (x + 4) = f x
axiom specific_interval_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

theorem f_at_7 : f 7 = -2 := 
  by sorry

end f_at_7_l67_67200


namespace solve_for_x_l67_67050

theorem solve_for_x (x : ℝ) (h : (x / 4) / 2 = 4 / (x / 2)) : x = 8 ∨ x = -8 :=
by
  sorry

end solve_for_x_l67_67050


namespace triangle_area_correct_l67_67915

open Real

def triangle_area (v1 v2 v3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (v1.1 * v2.2 + v2.1 * v3.2 + v3.1 * v1.2 - v1.2 * v2.1 - v2.2 * v3.1 - v3.2 * v1.1)

theorem triangle_area_correct :
  triangle_area (4, 6) (-4, 6) (0, 2) = 16 :=
by
  sorry

end triangle_area_correct_l67_67915


namespace defective_units_shipped_for_sale_l67_67606

theorem defective_units_shipped_for_sale (d p : ℝ) (h1 : d = 0.09) (h2 : p = 0.04) : (d * p * 100 = 0.36) :=
by 
  -- Assuming some calculation steps 
  sorry

end defective_units_shipped_for_sale_l67_67606


namespace birds_in_sky_l67_67075

theorem birds_in_sky (wings total_wings : ℕ) (h1 : total_wings = 26) (h2 : wings = 2) : total_wings / wings = 13 := 
by
  sorry

end birds_in_sky_l67_67075


namespace geometric_sequence_y_value_l67_67197

theorem geometric_sequence_y_value (q : ℝ) 
  (h1 : 2 * q^4 = 18) 
  : 2 * (q^2) = 6 := by
  sorry

end geometric_sequence_y_value_l67_67197


namespace union_P_complement_Q_l67_67853

open Set

def P : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def Q : Set ℝ := { x | x^2 ≥ 4 }
def R : Set ℝ := { x | -2 < x ∧ x < 2 }
def PQ_union : Set ℝ := P ∪ R

theorem union_P_complement_Q : PQ_union = { x | -2 < x ∧ x ≤ 3 } :=
by sorry

end union_P_complement_Q_l67_67853


namespace candy_distribution_l67_67174

theorem candy_distribution (n : Nat) : ∃ k : Nat, n = 2 ^ k :=
sorry

end candy_distribution_l67_67174


namespace payment_required_l67_67311

-- Definitions of the conditions
def price_suit : ℕ := 200
def price_tie : ℕ := 40
def num_suits : ℕ := 20
def discount_option_1 (x : ℕ) (hx : x > 20) : ℕ := price_suit * num_suits + (x - num_suits) * price_tie
def discount_option_2 (x : ℕ) (hx : x > 20) : ℕ := (price_suit * num_suits + x * price_tie) * 9 / 10

-- Theorem that needs to be proved
theorem payment_required (x : ℕ) (hx : x > 20) :
  discount_option_1 x hx = 40 * x + 3200 ∧ discount_option_2 x hx = 3600 + 36 * x :=
by sorry

end payment_required_l67_67311


namespace slope_of_dividing_line_l67_67071

/--
Given a rectangle with vertices at (0,0), (0,4), (5,4), (5,2),
and a right triangle with vertices at (5,2), (7,2), (5,0),
prove that the slope of the line through the origin that divides the area
of this L-shaped region exactly in half is 16/11.
-/
theorem slope_of_dividing_line :
  let rectangle_area := 5 * 4
  let triangle_area := (1 / 2) * 2 * 2
  let total_area := rectangle_area + triangle_area
  let half_area := total_area / 2
  let x_division := half_area / 4
  let slope := 4 / x_division
  slope = 16 / 11 :=
by
  sorry

end slope_of_dividing_line_l67_67071


namespace area_A₀B₀C₀_eq_2_area_AC₁BA₁CB₁_area_A₀B₀C₀_ge_4_area_ABC_l67_67682

open EuclideanGeometry

-- Definitions and conditions
variables {A B C A₁ B₁ C₁ A₀ B₀ C₀ : Point}
variables {α β γ : ℝ}

-- Given conditions about the triangle
axioms
  (hA₁ : IsAngleBisector A B C A₁)
  (hB₁ : IsAngleBisector B C A B₁)
  (hC₁ : IsAngleBisector C A B C₁)
  (hAA₁ : IntersectsCircumcircle A A₁)
  (hB₀ : IsExternalAngleBisector A₁ A C B₀)
  (hC₀ : IsExternalAngleBisector A₁ A B C₀)

-- First proof goal: 
theorem area_A₀B₀C₀_eq_2_area_AC₁BA₁CB₁ :
  area (triangle A₀ B₀ C₀) = 2 * area (hexagon A C₁ B A₁ C B₁) :=
sorry

-- Second proof goal:
theorem area_A₀B₀C₀_ge_4_area_ABC :
  area (triangle A₀ B₀ C₀) ≥ 4 * area (triangle A B C) :=
sorry

end area_A₀B₀C₀_eq_2_area_AC₁BA₁CB₁_area_A₀B₀C₀_ge_4_area_ABC_l67_67682


namespace figure_100_squares_l67_67794

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 2 * n + 1

theorem figure_100_squares : f 100 = 1020201 :=
by
  -- The proof will go here
  sorry

end figure_100_squares_l67_67794


namespace boa_constrictors_in_park_l67_67719

theorem boa_constrictors_in_park :
  ∃ (B : ℕ), (∃ (p : ℕ), p = 3 * B) ∧ (B + 3 * B + 40 = 200) ∧ B = 40 :=
by
  sorry

end boa_constrictors_in_park_l67_67719


namespace drums_of_grapes_per_day_l67_67820

-- Definitions derived from conditions
def pickers := 235
def raspberry_drums_per_day := 100
def total_days := 77
def total_drums := 17017

-- Prove the main theorem
theorem drums_of_grapes_per_day : (total_drums - total_days * raspberry_drums_per_day) / total_days = 121 := by
  sorry

end drums_of_grapes_per_day_l67_67820


namespace original_square_area_l67_67189

noncomputable def area_of_original_square (x : ℝ) (w : ℝ) (remaining_area : ℝ) : Prop :=
  x * x = remaining_area + x * w

theorem original_square_area : ∃ (x : ℝ), area_of_original_square x 3 40 → x * x = 64 :=
sorry

end original_square_area_l67_67189


namespace planks_ratio_l67_67020

theorem planks_ratio (P S : ℕ) (H : S + 100 + 20 + 30 = 200) (T : P = 200) (R : S = 200 / 2) : 
(S : ℚ) / P = 1 / 2 :=
by
  sorry

end planks_ratio_l67_67020


namespace park_length_l67_67631

theorem park_length (width : ℕ) (trees_per_sqft : ℕ) (num_trees : ℕ) (total_area : ℕ) (length : ℕ)
  (hw : width = 2000)
  (ht : trees_per_sqft = 20)
  (hn : num_trees = 100000)
  (ha : total_area = num_trees * trees_per_sqft)
  (hl : length = total_area / width) :
  length = 1000 :=
by
  sorry

end park_length_l67_67631


namespace three_digit_integers_with_at_least_one_two_but_no_four_l67_67534

-- Define the properties
def is_three_digit (n: ℕ) : Prop := 100 ≤ n ∧ n < 1000
def contains_digit (n: ℕ) (d: ℕ) : Prop := ∃ i, i < 3 ∧ d = n / 10^i % 10
def no_four (n: ℕ) : Prop := ¬ contains_digit n 4

-- Define the sets A and B
def setA (n: ℕ) : Prop := is_three_digit n ∧ no_four n
def setB (n: ℕ) : Prop := setA n ∧ ¬ contains_digit n 2

-- The final theorem statement
theorem three_digit_integers_with_at_least_one_two_but_no_four : 
  {n : ℕ | contains_digit n 2 ∧ setA n}.card = 200 :=
sorry

end three_digit_integers_with_at_least_one_two_but_no_four_l67_67534


namespace second_section_area_l67_67158

theorem second_section_area 
  (sod_area_per_square : ℕ := 4)
  (total_squares : ℕ := 1500)
  (first_section_length : ℕ := 30)
  (first_section_width : ℕ := 40)
  (total_area_needed : ℕ := total_squares * sod_area_per_square)
  (first_section_area : ℕ := first_section_length * first_section_width) :
  total_area_needed = first_section_area + 4800 := 
by 
  sorry

end second_section_area_l67_67158


namespace total_oranges_in_box_l67_67119

def initial_oranges_in_box : ℝ := 55.0
def oranges_added_by_susan : ℝ := 35.0

theorem total_oranges_in_box :
  initial_oranges_in_box + oranges_added_by_susan = 90.0 := by
  sorry

end total_oranges_in_box_l67_67119


namespace largest_fraction_l67_67810

theorem largest_fraction (p q r s : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) :
  (∃ (x : ℝ), x = (r + s) / (p + q) ∧ 
  (x > (p + s) / (q + r)) ∧ 
  (x > (p + q) / (r + s)) ∧ 
  (x > (q + r) / (p + s)) ∧ 
  (x > (q + s) / (p + r))) :=
sorry

end largest_fraction_l67_67810


namespace transport_cost_l67_67436

theorem transport_cost (weight_g : ℕ) (cost_per_kg : ℕ) (weight_kg : ℕ) (total_cost : ℕ)
  (h1 : weight_g = 2000)
  (h2 : cost_per_kg = 15000)
  (h3 : weight_kg = weight_g / 1000)
  (h4 : total_cost = weight_kg * cost_per_kg) :
  total_cost = 30000 :=
by
  sorry

end transport_cost_l67_67436


namespace average_age_of_dance_group_l67_67889

theorem average_age_of_dance_group (S_f S_m : ℕ) (avg_females avg_males : ℕ) 
(hf : avg_females = S_f / 12) (hm : avg_males = S_m / 18) 
(h1 : avg_females = 25) (h2 : avg_males = 40) : 
  (S_f + S_m) / 30 = 34 :=
by
  sorry

end average_age_of_dance_group_l67_67889


namespace Callum_points_l67_67869

theorem Callum_points
  (points_per_win : ℕ := 10)
  (total_matches : ℕ := 8)
  (krishna_win_fraction : ℚ := 3/4) :
  let callum_win_fraction := 1 - krishna_win_fraction in
  let callum_wins := callum_win_fraction * total_matches in
  let callum_points := callum_wins * points_per_win in
  callum_points = 20 := 
by
  sorry

end Callum_points_l67_67869


namespace g_neg_3_eq_neg_9_l67_67039

-- Define even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Given functions and values
variables (f g : ℝ → ℝ) (h_even : is_even_function f) (h_f_g : ∀ x, f x = g x - 2 * x)
variables (h_g3 : g 3 = 3)

-- Goal: Prove that g (-3) = -9
theorem g_neg_3_eq_neg_9 : g (-3) = -9 :=
sorry

end g_neg_3_eq_neg_9_l67_67039


namespace probability_all_same_color_l67_67618

open scoped Classical

noncomputable def num_black : ℕ := 5
noncomputable def num_red : ℕ := 4
noncomputable def num_green : ℕ := 6
noncomputable def num_blue : ℕ := 3
noncomputable def num_yellow : ℕ := 2

noncomputable def total_marbles : ℕ :=
  num_black + num_red + num_green + num_blue + num_yellow

noncomputable def prob_all_same_color : ℚ :=
  let p_black := if num_black >= 4 then 
      (num_black / total_marbles) * ((num_black - 1) / (total_marbles - 1)) *
      ((num_black - 2) / (total_marbles - 2)) * ((num_black - 3) / (total_marbles - 3)) else 0
  let p_green := if num_green >= 4 then 
      (num_green / total_marbles) * ((num_green - 1) / (total_marbles - 1)) *
      ((num_green - 2) / (total_marbles - 2)) * ((num_green - 3) / (total_marbles - 3)) else 0
  p_black + p_green

theorem probability_all_same_color :
  prob_all_same_color = 0.004128 :=
sorry

end probability_all_same_color_l67_67618


namespace find_width_of_room_l67_67894

section RoomWidth

variable (l C P A W : ℝ)
variable (h1 : l = 5.5)
variable (h2 : C = 16500)
variable (h3 : P = 750)
variable (h4 : A = C / P)
variable (h5 : A = l * W)

theorem find_width_of_room : W = 4 := by
  sorry

end RoomWidth

end find_width_of_room_l67_67894


namespace cylinder_lateral_surface_area_l67_67154

theorem cylinder_lateral_surface_area
    (r h : ℝ) (hr : r = 3) (hh : h = 10) :
    2 * Real.pi * r * h = 60 * Real.pi :=
by
  sorry

end cylinder_lateral_surface_area_l67_67154


namespace solve_system_l67_67252

theorem solve_system (x y z : ℝ) :
  x^2 = y^2 + z^2 ∧
  x^2024 = y^2024 + z^2024 ∧
  x^2025 = y^2025 + z^2025 ↔
  (y = x ∧ z = 0) ∨
  (y = -x ∧ z = 0) ∨
  (y = 0 ∧ z = x) ∨
  (y = 0 ∧ z = -x) :=
by {
  sorry -- The detailed proof will be filled here.
}

end solve_system_l67_67252


namespace problem_solution_l67_67416

theorem problem_solution (x y z : ℝ) (h1 : x * y + y * z + z * x = 4) (h2 : x * y * z = 6) :
  (x * y - (3 / 2) * (x + y)) * (y * z - (3 / 2) * (y + z)) * (z * x - (3 / 2) * (z + x)) = 81 / 4 :=
by
  sorry

end problem_solution_l67_67416


namespace non_neg_ints_less_than_pi_l67_67899

-- Define the condition: non-negative integers with absolute value less than π
def condition (x : ℕ) : Prop := |(x : ℝ)| < Real.pi

-- Prove that the set satisfying the condition is {0, 1, 2, 3}
theorem non_neg_ints_less_than_pi :
  {x : ℕ | condition x} = {0, 1, 2, 3} := by
  sorry

end non_neg_ints_less_than_pi_l67_67899


namespace product_divisible_by_3_product_residue_1_mod_3_product_residue_2_mod_3_l67_67445

noncomputable def residue_probability_zero (a b : ℕ) : ℚ :=
  if (a % 3 = 0 ∧ b % 3 = 0) ∨ (a % 3 = 0 ∧ b % 3 = 1) ∨ (a % 3 = 0 ∧ b % 3 = 2) ∨ (a % 3 = 1 ∧ b % 3 = 0) ∨ (a % 3 = 2 ∧ b % 3 = 0) then
    5 / 9 else 0

noncomputable def residue_probability_one (a b : ℕ) : ℚ :=
  if (a % 3 = 1 ∧ b % 3 = 1) ∨ (a % 3 = 2 ∧ b % 3 = 2) then
    2 / 9 else 0

noncomputable def residue_probability_two (a b : ℕ) : ℚ :=
  if (a % 3 = 1 ∧ b % 3 = 2) ∨ (a % 3 = 2 ∧ b % 3 = 1) then
    2 / 9 else 0

-- Proof statements for each probability
theorem product_divisible_by_3 (a b : ℕ) :
  (a % 3) * (b % 3) % 3 = 0 → residue_probability_zero a b = 5 / 9 := by
  sorry

theorem product_residue_1_mod_3 (a b : ℕ) :
  (a % 3) * (b % 3) % 3 = 1 → residue_probability_one a b = 2 / 9 := by
  sorry

theorem product_residue_2_mod_3 (a b : ℕ) :
  (a % 3) * (b % 3) % 3 = 2 → residue_probability_two a b = 2 / 9 := by
  sorry

end product_divisible_by_3_product_residue_1_mod_3_product_residue_2_mod_3_l67_67445


namespace soap_last_duration_l67_67392

-- Definitions of the given conditions
def cost_per_bar := 8 -- cost in dollars
def total_spent := 48 -- total spent in dollars
def months_in_year := 12

-- Definition of the query statement/proof goal
theorem soap_last_duration (h₁ : total_spent = 48) (h₂ : cost_per_bar = 8) (h₃ : months_in_year = 12) : months_in_year / (total_spent / cost_per_bar) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end soap_last_duration_l67_67392


namespace terminal_side_in_second_quadrant_l67_67811

theorem terminal_side_in_second_quadrant (α : ℝ) 
  (hcos : Real.cos α = -1/5) 
  (hsin : Real.sin α = 2 * Real.sqrt 6 / 5) : 
  (π / 2 < α ∧ α < π) :=
by
  sorry

end terminal_side_in_second_quadrant_l67_67811


namespace question_inequality_l67_67993

theorem question_inequality (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3/4 * (x - y)^2) (max (3/4 * (y - z)^2) (3/4 * (z - x)^2)) := 
sorry

end question_inequality_l67_67993


namespace simplify_sqrt_mul_l67_67960

theorem simplify_sqrt_mul : (Real.sqrt 5 * Real.sqrt (4 / 5) = 2) :=
by
  sorry

end simplify_sqrt_mul_l67_67960


namespace distance_ran_by_Juan_l67_67693

-- Definitions based on the condition
def speed : ℝ := 10 -- in miles per hour
def time : ℝ := 8 -- in hours

-- Theorem statement
theorem distance_ran_by_Juan : speed * time = 80 := by
  sorry

end distance_ran_by_Juan_l67_67693


namespace remainder_eq_six_l67_67919

theorem remainder_eq_six
  (Dividend : ℕ) (Divisor : ℕ) (Quotient : ℕ) (Remainder : ℕ)
  (h1 : Dividend = 139)
  (h2 : Divisor = 19)
  (h3 : Quotient = 7)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Remainder = 6 :=
by
  sorry

end remainder_eq_six_l67_67919


namespace milton_sold_total_pies_l67_67163

-- Definitions for the given conditions.
def apple_pie_slices : ℕ := 8
def peach_pie_slices : ℕ := 6
def cherry_pie_slices : ℕ := 10

def apple_slices_ordered : ℕ := 88
def peach_slices_ordered : ℕ := 78
def cherry_slices_ordered : ℕ := 45

-- Function to compute the number of pies, rounding up as necessary
noncomputable def pies_sold (ordered : ℕ) (slices : ℕ) : ℕ :=
  (ordered + slices - 1) / slices  -- Using integer division to round up

-- The theorem asserting the total number of pies sold 
theorem milton_sold_total_pies : 
  pies_sold apple_slices_ordered apple_pie_slices +
  pies_sold peach_slices_ordered peach_pie_slices +
  pies_sold cherry_slices_ordered cherry_pie_slices = 29 :=
by sorry

end milton_sold_total_pies_l67_67163


namespace find_x_l67_67054

theorem find_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end find_x_l67_67054


namespace coat_price_reduction_l67_67583

theorem coat_price_reduction:
  ∀ (original_price reduction_amount : ℕ),
  original_price = 500 →
  reduction_amount = 350 →
  (reduction_amount : ℝ) / original_price * 100 = 70 :=
by
  intros original_price reduction_amount h1 h2
  sorry

end coat_price_reduction_l67_67583


namespace train_speed_l67_67636

theorem train_speed
  (length_train : ℝ)
  (length_bridge : ℝ)
  (time_seconds : ℝ) :
  length_train = 140 →
  length_bridge = 235.03 →
  time_seconds = 30 →
  (length_train + length_bridge) / time_seconds * 3.6 = 45.0036 :=
by
  intros h1 h2 h3
  sorry

end train_speed_l67_67636


namespace find_x_l67_67053

theorem find_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end find_x_l67_67053


namespace zumish_12_words_remainder_l67_67222

def zumishWords n :=
  if n < 2 then (0, 0, 0)
  else if n == 2 then (4, 4, 4)
  else let (a, b, c) := zumishWords (n - 1)
       (2 * (a + c) % 1000, 2 * a % 1000, 2 * b % 1000)

def countZumishWords (n : Nat) :=
  let (a, b, c) := zumishWords n
  (a + b + c) % 1000

theorem zumish_12_words_remainder :
  countZumishWords 12 = 322 :=
by
  intros
  sorry

end zumish_12_words_remainder_l67_67222


namespace greatest_integer_x_l67_67278

theorem greatest_integer_x
    (x : ℤ) : 
    (7 / 9 : ℚ) > (x : ℚ) / 13 → x ≤ 10 :=
by
    sorry

end greatest_integer_x_l67_67278


namespace boat_upstream_time_is_1_5_hours_l67_67467

noncomputable def time_to_cover_distance_upstream
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ) : ℝ :=
  distance_downstream / (speed_boat_still_water - speed_stream)

theorem boat_upstream_time_is_1_5_hours
  (speed_stream : ℝ)
  (speed_boat_still_water : ℝ)
  (time_downstream : ℝ)
  (downstream_distance : ℝ)
  (h1 : speed_stream = 3)
  (h2 : speed_boat_still_water = 15)
  (h3 : time_downstream = 1)
  (h4 : downstream_distance = speed_boat_still_water + speed_stream) :
  time_to_cover_distance_upstream speed_stream speed_boat_still_water time_downstream downstream_distance = 1.5 :=
by
  sorry

end boat_upstream_time_is_1_5_hours_l67_67467


namespace find_number_of_students_l67_67315

-- Parameters
variable (n : ℕ) (C : ℕ)
def first_and_last_picked_by_sam (n : ℕ) (C : ℕ) : Prop := 
  C + 1 = 2 * n

-- Conditions: number of candies is 120, the bag completes 2 full rounds at the table.
theorem find_number_of_students
  (C : ℕ) (h_C: C = 120) (h_rounds: 2 * n = C):
  n = 60 :=
by
  sorry

end find_number_of_students_l67_67315


namespace baseball_team_earnings_l67_67935

theorem baseball_team_earnings (S : ℝ) (W : ℝ) (Total : ℝ) 
    (h1 : S = 2662.50) 
    (h2 : W = S - 142.50) 
    (h3 : Total = W + S) : 
  Total = 5182.50 :=
sorry

end baseball_team_earnings_l67_67935


namespace equalize_costs_l67_67080

variable (L B C : ℝ)
variable (h1 : L < B)
variable (h2 : B < C)

theorem equalize_costs : (B + C - 2 * L) / 3 = ((L + B + C) / 3 - L) :=
by sorry

end equalize_costs_l67_67080


namespace sufficient_not_necessary_condition_l67_67511

variable (x a : ℝ)

def p := x ≤ -1
def q := a ≤ x ∧ x < a + 2

-- If q is sufficient but not necessary for p, then the range of a is (-∞, -3]
theorem sufficient_not_necessary_condition : 
  (∀ x, q x a → p x) ∧ ∃ x, p x ∧ ¬ q x a → a ≤ -3 :=
by
  sorry

end sufficient_not_necessary_condition_l67_67511


namespace find_x_value_l67_67462

theorem find_x_value (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
sorry

end find_x_value_l67_67462


namespace evaluate_g_at_6_l67_67086

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 30 * x^2 - 35 * x - 75

theorem evaluate_g_at_6 : g 6 = 363 :=
by
  -- Proof skipped
  sorry

end evaluate_g_at_6_l67_67086


namespace simplify_and_evaluate_div_expr_l67_67707

variable (m : ℤ)

theorem simplify_and_evaluate_div_expr (h : m = 2) :
  ( (m^2 - 9) / (m^2 - 6 * m + 9) / (1 - 2 / (m - 3)) = -5 / 3) :=
by
  sorry

end simplify_and_evaluate_div_expr_l67_67707


namespace total_emeralds_l67_67766

theorem total_emeralds (D R E : ℕ) 
  (h1 : 2 * D + 2 * E + 2 * R = 6)
  (h2 : R = D + 15) : 
  E = 12 :=
by
  -- Proof omitted
  sorry

end total_emeralds_l67_67766


namespace no_such_integers_l67_67786

theorem no_such_integers (x y z : ℤ) : ¬ ((x - y)^3 + (y - z)^3 + (z - x)^3 = 2011) :=
sorry

end no_such_integers_l67_67786


namespace consecutive_odd_integer_sum_l67_67116

theorem consecutive_odd_integer_sum {n : ℤ} (h1 : n = 17 ∨ n + 2 = 17) (h2 : n + n + 2 ≥ 36) : (n = 17 → n + 2 = 19) ∧ (n + 2 = 17 → n = 15) :=
by
  sorry

end consecutive_odd_integer_sum_l67_67116


namespace quadratic_no_real_roots_l67_67217

theorem quadratic_no_real_roots (m : ℝ) : (4 + 4 * m < 0) → (m < -1) :=
by
  intro h
  linarith

end quadratic_no_real_roots_l67_67217


namespace percentage_students_went_on_trip_l67_67484

theorem percentage_students_went_on_trip
  (total_students : ℕ)
  (students_march : ℕ)
  (students_march_more_than_100 : ℕ)
  (students_june : ℕ)
  (students_june_more_than_100 : ℕ)
  (total_more_than_100_either_trip : ℕ) :
  total_students = 100 → students_march = 20 → students_march_more_than_100 = 7 →
  students_june = 15 → students_june_more_than_100 = 6 →
  70 * total_more_than_100_either_trip = 7 * 100 →
  (students_march + students_june) * 100 / total_students = 35 :=
by
  intros h_total h_march h_march_100 h_june h_june_100 h_total_100
  sorry

end percentage_students_went_on_trip_l67_67484


namespace min_value_expression_l67_67496

theorem min_value_expression (x y : ℝ) : 
  (∃ (x_min y_min : ℝ), 
  (x_min = 1/2 ∧ y_min = 0) ∧ 
  3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39/4) :=
by
  sorry

end min_value_expression_l67_67496


namespace find_cos_F1PF2_l67_67514

noncomputable def cos_angle_P_F1_F2 : ℝ :=
  let F1 := (-(4:ℝ), 0)
  let F2 := ((4:ℝ), 0)
  let a := (5:ℝ)
  let b := (3:ℝ)
  let P : ℝ × ℝ := sorry -- P is a point on the ellipse
  let area_triangle : ℝ := 3 * Real.sqrt 3
  let cos_angle : ℝ := 1 / 2
  cos_angle

def cos_angle_F1PF2_lemma (F1 F2 : ℝ × ℝ) (ellipse_Area : ℝ) (cos_angle : ℝ) : Prop :=
  cos_angle = 1/2

theorem find_cos_F1PF2 (a b : ℝ) (F1 F2 : ℝ × ℝ) (P : ℝ × ℝ) (Area_PF1F2 : ℝ) :
  (F1 = (-(4:ℝ), 0) ∧ F2 = ((4:ℝ), 0)) ∧ (Area_PF1F2 = 3 * Real.sqrt 3) ∧
  (P.1^2 / (a^2) + P.2^2 / (b^2) = 1) → cos_angle_F1PF2_lemma F1 F2 Area_PF1F2 (cos_angle_P_F1_F2)
:=
  sorry

end find_cos_F1PF2_l67_67514


namespace radius_of_inscribed_semicircle_in_isosceles_triangle_l67_67330

theorem radius_of_inscribed_semicircle_in_isosceles_triangle
    (BC : ℝ) (h : ℝ) (r : ℝ)
    (H_eq : BC = 24)
    (H_height : h = 18)
    (H_area : 0.5 * BC * h = 0.5 * 24 * 18) :
    r = 18 / π := by
    sorry

end radius_of_inscribed_semicircle_in_isosceles_triangle_l67_67330


namespace stock_price_end_of_second_year_l67_67024

noncomputable def initial_price : ℝ := 120
noncomputable def price_after_first_year (initial_price : ℝ) : ℝ := initial_price * 2
noncomputable def price_after_second_year (price_after_first_year : ℝ) : ℝ := price_after_first_year * 0.7

theorem stock_price_end_of_second_year : 
  price_after_second_year (price_after_first_year initial_price) = 168 := 
by 
  sorry

end stock_price_end_of_second_year_l67_67024


namespace squirrel_acorns_l67_67683

theorem squirrel_acorns :
  ∃ (c s r : ℕ), (4 * c = 5 * s) ∧ (3 * r = 4 * c) ∧ (r = s + 3) ∧ (5 * s = 40) :=
by
  sorry

end squirrel_acorns_l67_67683


namespace inequality_inequality_l67_67536

theorem inequality_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b) ^ 2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b) ^ 2 / (8 * b) :=
sorry

end inequality_inequality_l67_67536


namespace grape_ratio_new_new_cans_from_grape_l67_67149

-- Definitions derived from the problem conditions
def apple_ratio_initial : ℚ := 1 / 6
def grape_ratio_initial : ℚ := 1 / 10
def apple_ratio_new : ℚ := 1 / 5

-- Prove the new grape_ratio
theorem grape_ratio_new : ℚ :=
  let total_volume_per_can := apple_ratio_initial + grape_ratio_initial
  let grape_ratio_new_reciprocal := (total_volume_per_can - apple_ratio_new)
  1 / grape_ratio_new_reciprocal

-- Required final quantity of cans
theorem new_cans_from_grape : 
  (1 / grape_ratio_new) = 15 :=
sorry

end grape_ratio_new_new_cans_from_grape_l67_67149


namespace benny_number_of_days_worked_l67_67645

-- Define the conditions
def total_hours_worked : ℕ := 18
def hours_per_day : ℕ := 3

-- Define the problem statement in Lean
theorem benny_number_of_days_worked : (total_hours_worked / hours_per_day) = 6 := 
by
  sorry

end benny_number_of_days_worked_l67_67645


namespace Z_real_axis_Z_first_quadrant_Z_on_line_l67_67995

-- Definitions based on the problem conditions
def Z_real (m : ℝ) : ℝ := m^2 + 5*m + 6
def Z_imag (m : ℝ) : ℝ := m^2 - 2*m - 15

-- Lean statement for the equivalent proof problem

theorem Z_real_axis (m : ℝ) :
  Z_imag m = 0 ↔ (m = -3 ∨ m = 5) := sorry

theorem Z_first_quadrant (m : ℝ) :
  (Z_real m > 0 ∧ Z_imag m > 0) ↔ (m > 5) := sorry

theorem Z_on_line (m : ℝ) :
  (Z_real m + Z_imag m + 5 = 0) ↔ (m = (-5 + Real.sqrt 41) / 2) := sorry

end Z_real_axis_Z_first_quadrant_Z_on_line_l67_67995


namespace evaluate_expression_l67_67662

theorem evaluate_expression : (831 * 831) - (830 * 832) = 1 :=
by
  sorry

end evaluate_expression_l67_67662


namespace quadratic_inequality_solution_l67_67376

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end quadratic_inequality_solution_l67_67376


namespace solution_correctness_l67_67717

noncomputable def solution_set : Set ℝ := { x : ℝ | (x + 1) * (x - 2) > 0 }

theorem solution_correctness (x : ℝ) :
  (x ∈ solution_set) ↔ (x < -1 ∨ x > 2) :=
by sorry

end solution_correctness_l67_67717


namespace calc_expr_eq_l67_67962

theorem calc_expr_eq : 2 + 3 / (4 + 5 / 6) = 76 / 29 := 
by 
  sorry

end calc_expr_eq_l67_67962


namespace expected_deviation_10_greater_than_100_l67_67294

-- Definitions for deviations for 10 and 100 tosses
def deviation_10_tosses (m₁₀: ℕ) : ℝ := (m₁₀ / 10) - 0.5
def deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  ∑ i, αs i / 10

-- Definitions for absolute deviations
def abs_deviation_10_tosses (m₁₀: ℕ) : ℝ := abs (deviation_10_tosses m₁₀)
def abs_deviation_100_tosses (m₁₀₀: ℕ) (αs: Fin 10 → ℝ) : ℝ :=
  abs (deviation_100_tosses m₁₀₀ αs)

-- Expected values of absolute deviations
def expected_abs_deviation_10_tosses : ℝ := sorry
def expected_abs_deviation_100_tosses : ℝ := sorry

theorem expected_deviation_10_greater_than_100 :
  expected_abs_deviation_10_tosses > expected_abs_deviation_100_tosses :=
sorry

end expected_deviation_10_greater_than_100_l67_67294


namespace six_degree_below_zero_is_minus_six_degrees_l67_67218

def temp_above_zero (temp: Int) : String := "+" ++ toString temp ++ "°C"

def temp_below_zero (temp: Int) : String := "-" ++ toString temp ++ "°C"

-- Statement of the theorem
theorem six_degree_below_zero_is_minus_six_degrees:
  temp_below_zero 6 = "-6°C" :=
by
  sorry

end six_degree_below_zero_is_minus_six_degrees_l67_67218


namespace always_30_blue_white_rectangles_l67_67107

noncomputable def board : Type :=
  { c : ℕ // c < 100 }

structure cell :=
  (row : ℕ) (col : ℕ) (valid : row < 10 ∧ col < 10)

structure board_coloring :=
  (color : cell → char)
  (valid_colors : ∀ c, color c ∈ ['R', 'B', 'W'])

-- Conditions:
def coloring_condition (bc : board_coloring) : Prop :=
  ∀ c1 c2 : cell, (c1.row = c2.row ∧ (c1.col = c2.col + 1 ∨ c1.col + 1 = c2.col) ∨ c1.col = c2.col ∧ (c1.row = c2.row + 1 ∨ c1.row + 1 = c2.row)) → bc.color c1 ≠ bc.color c2

def red_cells_condition (bc : board_coloring) : Prop :=
  ∃ reds : list cell, reds.length = 20 ∧ ∀ r ∈ reds, bc.color r = 'R'

-- Question:
theorem always_30_blue_white_rectangles (bc : board_coloring) (h_coloring : coloring_condition bc) (h_reds : red_cells_condition bc) :
  ∃ blue_white_rectangles : list (cell × cell), blue_white_rectangles.length = 30 ∧
  ∀ (r1 r2 : cell), (r1, r2) ∈ blue_white_rectangles → (bc.color r1 = 'B' ∧ bc.color r2 = 'W') ∨ (bc.color r1 = 'W' ∧ bc.color r2 = 'B') :=
  sorry

end always_30_blue_white_rectangles_l67_67107


namespace sqrt_expression_evaluation_l67_67779

theorem sqrt_expression_evaluation :
  (Real.sqrt 48 - 6 * Real.sqrt (1 / 3) - Real.sqrt 18 / Real.sqrt 6) = Real.sqrt 3 :=
by
  sorry

end sqrt_expression_evaluation_l67_67779


namespace remaining_number_is_divisible_by_divisor_l67_67599

def initial_number : ℕ := 427398
def subtracted_number : ℕ := 8
def remaining_number : ℕ := initial_number - subtracted_number
def divisor : ℕ := 10

theorem remaining_number_is_divisible_by_divisor :
  remaining_number % divisor = 0 :=
by {
  sorry
}

end remaining_number_is_divisible_by_divisor_l67_67599


namespace abs_inequality_holds_l67_67506

theorem abs_inequality_holds (m x : ℝ) (h : -1 ≤ m ∧ m ≤ 6) : 
  |x - 2| + |x + 4| ≥ m^2 - 5 * m :=
sorry

end abs_inequality_holds_l67_67506


namespace upstream_travel_time_l67_67465

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l67_67465


namespace max_square_plots_l67_67627

theorem max_square_plots (width height internal_fence_length : ℕ) 
(h_w : width = 60) (h_h : height = 30) (h_fence: internal_fence_length = 2400) : 
  ∃ n : ℕ, (60 * 30 / (n * n) = 400 ∧ 
  (30 * (60 / n - 1) + 60 * (30 / n - 1) + 60 + 30) ≤ internal_fence_length) :=
sorry

end max_square_plots_l67_67627


namespace Karen_packs_piece_of_cake_days_l67_67868

theorem Karen_packs_piece_of_cake_days 
(Total Ham_Days : ℕ) (Ham_probability Cake_probability : ℝ) 
  (H_Total : Total = 5) 
  (H_Ham_Days : Ham_Days = 3) 
  (H_Ham_probability : Ham_probability = (3 / 5)) 
  (H_Cake_probability : Ham_probability * (Cake_probability / 5) = 0.12) : 
  Cake_probability = 1 := 
by
  sorry

end Karen_packs_piece_of_cake_days_l67_67868


namespace inequality_solution_l67_67493

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 4) * (x + 1) / (x - 2)

theorem inequality_solution :
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | 2 < x ∧ x ≤ 8/3 } ∪ { x : ℝ | 4 ≤ x } :=
by sorry

end inequality_solution_l67_67493


namespace compare_mixed_decimal_l67_67486

def mixed_number_value : ℚ := -2 - 1 / 3  -- Representation of -2 1/3 as a rational number
def decimal_value : ℚ := -2.3             -- Representation of -2.3 as a rational number

theorem compare_mixed_decimal : mixed_number_value < decimal_value :=
sorry

end compare_mixed_decimal_l67_67486


namespace range_of_a_l67_67513

theorem range_of_a {a : ℝ} (h : (a^2) / 4 + 1 / 2 < 1) : -Real.sqrt 2 < a ∧ a < Real.sqrt 2 :=
sorry

end range_of_a_l67_67513


namespace initial_number_of_red_balls_l67_67721

theorem initial_number_of_red_balls 
  (num_white_balls num_red_balls : ℕ)
  (h1 : num_red_balls = 4 * num_white_balls + 3)
  (num_actions : ℕ)
  (h2 : 4 + 5 * num_actions = num_white_balls)
  (h3 : 34 + 17 * num_actions = num_red_balls) : 
  num_red_balls = 119 := 
by
  sorry

end initial_number_of_red_balls_l67_67721


namespace find_Minchos_chocolate_l67_67420

variable (M : ℕ)  -- Define M as a natural number

-- Define the conditions as Lean hypotheses
def TaeminChocolate := 5 * M
def KibumChocolate := 3 * M
def TotalChocolate := TaeminChocolate M + KibumChocolate M

theorem find_Minchos_chocolate (h : TotalChocolate M = 160) : M = 20 :=
by
  sorry

end find_Minchos_chocolate_l67_67420


namespace find_c2013_l67_67528

theorem find_c2013 :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ),
    (a 1 = 3) →
    (b 1 = 3) →
    (∀ n : ℕ, 1 ≤ n → a (n+1) - a n = 3) →
    (∀ n : ℕ, 1 ≤ n → b (n+1) = 3 * b n) →
    (∀ n : ℕ, c n = b (a n)) →
    c 2013 = 27^2013 := by
  sorry

end find_c2013_l67_67528


namespace num_rooms_l67_67639

theorem num_rooms (r1 r2 w1 w2 p w_paint : ℕ) (h_r1 : r1 = 5) (h_r2 : r2 = 4) (h_w1 : w1 = 4) (h_w2 : w2 = 5)
    (h_p : p = 5) (h_w_paint : w_paint = 8) (h_total_walls_family : p * w_paint = (r1 * w1 + r2 * w2)) :
    (r1 + r2 = 9) :=
by
  sorry

end num_rooms_l67_67639


namespace transformed_graph_equation_l67_67066

theorem transformed_graph_equation (x y x' y' : ℝ)
  (h1 : x' = 5 * x)
  (h2 : y' = 3 * y)
  (h3 : x^2 + y^2 = 1) :
  x'^2 / 25 + y'^2 / 9 = 1 :=
by
  sorry

end transformed_graph_equation_l67_67066


namespace solve_abs_inequality_l67_67101

theorem solve_abs_inequality (x : ℝ) :
  (|x - 2| + |x - 4| > 6) ↔ (x < 0 ∨ 12 < x) :=
by
  sorry

end solve_abs_inequality_l67_67101


namespace sum_mean_median_mode_l67_67125

theorem sum_mean_median_mode (l : List ℚ) (h : l = [1, 2, 2, 3, 3, 3, 3, 4, 5]) :
    let mean := (1 + 2 + 2 + 3 + 3 + 3 + 3 + 4 + 5) / 9
    let median := 3
    let mode := 3
    mean + median + mode = 8.888 :=
by
  sorry

end sum_mean_median_mode_l67_67125


namespace compute_H_five_times_l67_67152

def H (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem compute_H_five_times : H (H (H (H (H 2)))) = -1 := by
  sorry

end compute_H_five_times_l67_67152


namespace age_ratio_in_4_years_l67_67595

-- Definitions based on the conditions
def pete_age (years_ago : ℕ) (p : ℕ) (c : ℕ) : Prop :=
match years_ago with
  | 2 => p - 2 = 3 * (c - 2)
  | 4 => p - 4 = 4 * (c - 4)
  | _ => true
end

-- Question: In how many years will the ratio of their ages be 2:1?
def age_ratio (years : ℕ) (p : ℕ) (c : ℕ) : Prop :=
(p + years) / (c + years) = 2

-- Proof problem
theorem age_ratio_in_4_years {p c : ℕ} (h1 : pete_age 2 p c) (h2 : pete_age 4 p c) : 
  age_ratio 4 p c :=
sorry

end age_ratio_in_4_years_l67_67595


namespace min_workers_for_profit_l67_67621

theorem min_workers_for_profit
    (maintenance_fees : ℝ)
    (worker_hourly_wage : ℝ)
    (widgets_per_hour : ℝ)
    (widget_price : ℝ)
    (work_hours : ℝ)
    (n : ℕ)
    (h_maintenance : maintenance_fees = 470)
    (h_wage : worker_hourly_wage = 10)
    (h_production : widgets_per_hour = 6)
    (h_price : widget_price = 3.5)
    (h_hours : work_hours = 8) :
  470 + 80 * n < 168 * n → n ≥ 6 := 
by
  sorry

end min_workers_for_profit_l67_67621


namespace parallel_line_through_point_l67_67495

theorem parallel_line_through_point :
  ∃ c : ℝ, ∀ x y : ℝ, (x = -1) → (y = 3) → (x - 2*y + 3 = 0) → (x - 2*y + c = 0) :=
sorry

end parallel_line_through_point_l67_67495


namespace masha_numbers_l67_67397

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l67_67397


namespace jackson_earnings_l67_67690

def hourly_rate_usd : ℝ := 5
def hourly_rate_gbp : ℝ := 3
def hourly_rate_jpy : ℝ := 400

def hours_vacuuming : ℝ := 2
def sessions_vacuuming : ℝ := 2

def hours_washing_dishes : ℝ := 0.5
def hours_cleaning_bathroom := hours_washing_dishes * 3

def exchange_rate_gbp_to_usd : ℝ := 1.35
def exchange_rate_jpy_to_usd : ℝ := 0.009

def earnings_in_usd : ℝ := (hours_vacuuming * sessions_vacuuming * hourly_rate_usd)
def earnings_in_gbp : ℝ := (hours_washing_dishes * hourly_rate_gbp)
def earnings_in_jpy : ℝ := (hours_cleaning_bathroom * hourly_rate_jpy)

def converted_gbp_to_usd : ℝ := earnings_in_gbp * exchange_rate_gbp_to_usd
def converted_jpy_to_usd : ℝ := earnings_in_jpy * exchange_rate_jpy_to_usd

def total_earnings_usd : ℝ := earnings_in_usd + converted_gbp_to_usd + converted_jpy_to_usd

theorem jackson_earnings : total_earnings_usd = 27.425 := by
  sorry

end jackson_earnings_l67_67690


namespace solution_set_of_x_l67_67973

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l67_67973


namespace sum_of_net_gains_is_correct_l67_67148

namespace DepartmentRevenue

def revenueIncreaseA : ℝ := 0.1326
def revenueIncreaseB : ℝ := 0.0943
def revenueIncreaseC : ℝ := 0.7731
def taxRate : ℝ := 0.235
def initialRevenue : ℝ := 4.7 -- in millions

def netGain (revenueIncrease : ℝ) (taxRate : ℝ) (initialRevenue : ℝ) : ℝ :=
  (initialRevenue * (1 + revenueIncrease)) * (1 - taxRate)

def netGainA : ℝ := netGain revenueIncreaseA taxRate initialRevenue
def netGainB : ℝ := netGain revenueIncreaseB taxRate initialRevenue
def netGainC : ℝ := netGain revenueIncreaseC taxRate initialRevenue

def netGainSum : ℝ := netGainA + netGainB + netGainC

theorem sum_of_net_gains_is_correct :
  netGainSum = 14.38214 := by
    sorry

end DepartmentRevenue

end sum_of_net_gains_is_correct_l67_67148


namespace sally_rum_l67_67248

theorem sally_rum (x : ℕ) (h₁ : 3 * x = x + 12 + 8) : x = 10 := by
  sorry

end sally_rum_l67_67248


namespace quadratic_function_value_at_neg_one_l67_67522

theorem quadratic_function_value_at_neg_one (b c : ℝ) 
  (h1 : (1:ℝ) ^ 2 + b * 1 + c = 0) 
  (h2 : (3:ℝ) ^ 2 + b * 3 + c = 0) : 
  ((-1:ℝ) ^ 2 + b * (-1) + c = 8) :=
by
  sorry

end quadratic_function_value_at_neg_one_l67_67522


namespace num_squares_in_6x6_grid_l67_67966

/-- Define the number of kxk squares in an nxn grid -/
def num_squares (n k : ℕ) : ℕ := (n + 1 - k) * (n + 1 - k)

/-- Prove the total number of different squares in a 6x6 grid is 86 -/
theorem num_squares_in_6x6_grid : 
  (num_squares 6 1) + (num_squares 6 2) + (num_squares 6 3) + (num_squares 6 4) = 86 :=
by sorry

end num_squares_in_6x6_grid_l67_67966


namespace fraction_meaningful_domain_l67_67837

theorem fraction_meaningful_domain (x : ℝ) : (∃ f : ℝ, f = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_domain_l67_67837


namespace big_cows_fewer_than_small_cows_l67_67113

theorem big_cows_fewer_than_small_cows (b s : ℕ) (h1 : b = 6) (h2 : s = 7) : 
  (s - b) / s = 1 / 7 :=
by
  sorry

end big_cows_fewer_than_small_cows_l67_67113


namespace find_c_minus_d_l67_67166

variable (g : ℝ → ℝ)
variable (c d : ℝ)
variable (invertible_g : Function.Injective g)
variable (g_at_c : g c = d)
variable (g_at_d : g d = 5)

theorem find_c_minus_d : c - d = -3 := by
  sorry

end find_c_minus_d_l67_67166


namespace factorization_6x2_minus_24x_plus_18_l67_67996

theorem factorization_6x2_minus_24x_plus_18 :
    ∀ x : ℝ, 6 * x^2 - 24 * x + 18 = 6 * (x - 1) * (x - 3) :=
by
  intro x
  sorry

end factorization_6x2_minus_24x_plus_18_l67_67996


namespace sheila_weekly_earnings_l67_67418

-- Definitions for conditions
def hours_per_day_on_MWF : ℕ := 8
def days_worked_on_MWF : ℕ := 3
def hours_per_day_on_TT : ℕ := 6
def days_worked_on_TT : ℕ := 2
def hourly_rate : ℕ := 10

-- Total weekly hours worked
def total_weekly_hours : ℕ :=
  (hours_per_day_on_MWF * days_worked_on_MWF) + (hours_per_day_on_TT * days_worked_on_TT)

-- Total weekly earnings
def weekly_earnings : ℕ :=
  total_weekly_hours * hourly_rate

-- Lean statement for the proof
theorem sheila_weekly_earnings : weekly_earnings = 360 :=
  sorry

end sheila_weekly_earnings_l67_67418


namespace problem_number_of_ways_to_choose_2005_balls_l67_67990

def number_of_ways_to_choose_balls (n : ℕ) : ℕ :=
  binomial (n + 2) 2 - binomial ((n + 1) / 2 + 1) 2

theorem problem_number_of_ways_to_choose_2005_balls :
  number_of_ways_to_choose_balls 2005 = binomial 2007 2 - binomial 1004 2 :=
by
  -- Proof will be provided here.
  sorry

end problem_number_of_ways_to_choose_2005_balls_l67_67990


namespace triangle_is_equilateral_l67_67229

   def sides_in_geometric_progression (a b c : ℝ) : Prop :=
     b^2 = a * c

   def angles_in_arithmetic_progression (A B C : ℝ) : Prop :=
     ∃ α δ : ℝ, A = α - δ ∧ B = α ∧ C = α + δ

   theorem triangle_is_equilateral {a b c A B C : ℝ} 
     (ha : a > 0) (hb : b > 0) (hc : c > 0)
     (hA : A > 0) (hB : B > 0) (hC : C > 0)
     (sum_angles : A + B + C = 180)
     (h1 : sides_in_geometric_progression a b c)
     (h2 : angles_in_arithmetic_progression A B C) : 
     a = b ∧ b = c ∧ A = 60 ∧ B = 60 ∧ C = 60 :=
   sorry
   
end triangle_is_equilateral_l67_67229


namespace find_f_2017_div_2_l67_67998

noncomputable def is_odd_function {X Y : Type*} [AddGroup X] [AddGroup Y] (f : X → Y) :=
  ∀ x : X, f (-x) = -f x

noncomputable def is_periodic_function {X Y : Type*} [AddGroup X] [AddGroup Y] (p : X) (f : X → Y) :=
  ∀ x : X, f (x + p) = f x

noncomputable def f : ℝ → ℝ 
| x => if -1 ≤ x ∧ x ≤ 0 then x * x + x else sorry

theorem find_f_2017_div_2 : f (2017 / 2) = 1 / 4 :=
by
  have h_odd : is_odd_function f := sorry
  have h_period : is_periodic_function 2 f := sorry
  unfold f
  sorry

end find_f_2017_div_2_l67_67998


namespace total_amount_shared_l67_67640

theorem total_amount_shared (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) 
  (portion_a : ℕ) (portion_b : ℕ) (portion_c : ℕ)
  (h_ratio : ratio_a = 3 ∧ ratio_b = 4 ∧ ratio_c = 9)
  (h_portion_a : portion_a = 30)
  (h_portion_b : portion_b = 2 * portion_a + 10)
  (h_portion_c : portion_c = (ratio_c / ratio_a) * portion_a) :
  portion_a + portion_b + portion_c = 190 :=
by sorry

end total_amount_shared_l67_67640


namespace perpendicular_OP_CD_l67_67704

variables {Point : Type}

-- Definitions of all the points involved
variables (A B C D P O : Point)
-- Definitions for distances / lengths
variables (dist : Point → Point → ℝ)
-- Definitions for relationships
variables (circumcenter : Point → Point → Point → Point)
variables (perpendicular : Point → Point → Point → Point → Prop)

-- Segment meet condition
variables (meet_at : Point → Point → Point → Prop)

-- Assuming the given conditions
theorem perpendicular_OP_CD 
  (meet : meet_at A C P)
  (meet' : meet_at B D P)
  (h1 : dist P A = dist P D)
  (h2 : dist P B = dist P C)
  (hO : circumcenter P A B = O) :
  perpendicular O P C D :=
sorry

end perpendicular_OP_CD_l67_67704


namespace minimize_total_cost_l67_67324

noncomputable def event_probability_without_measures : ℚ := 0.3
noncomputable def loss_if_event_occurs : ℚ := 4000000
noncomputable def cost_measure_A : ℚ := 450000
noncomputable def prob_event_not_occurs_measure_A : ℚ := 0.9
noncomputable def cost_measure_B : ℚ := 300000
noncomputable def prob_event_not_occurs_measure_B : ℚ := 0.85

noncomputable def total_cost_no_measures : ℚ :=
  event_probability_without_measures * loss_if_event_occurs

noncomputable def total_cost_measure_A : ℚ :=
  cost_measure_A + (1 - prob_event_not_occurs_measure_A) * loss_if_event_occurs

noncomputable def total_cost_measure_B : ℚ :=
  cost_measure_B + (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

noncomputable def total_cost_measures_A_and_B : ℚ :=
  cost_measure_A + cost_measure_B + (1 - prob_event_not_occurs_measure_A) * (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

theorem minimize_total_cost :
  min (min total_cost_no_measures total_cost_measure_A) (min total_cost_measure_B total_cost_measures_A_and_B) = total_cost_measures_A_and_B :=
by sorry

end minimize_total_cost_l67_67324


namespace find_a_for_even_function_l67_67365

theorem find_a_for_even_function (a : ℝ) (h : ∀ x : ℝ, x^3 * (a * 2^x - 2^(-x)) = (-x)^3 * (a * 2^(-x) - 2^x)) : a = 1 :=
by
  -- Placeholder for proof
  sorry

end find_a_for_even_function_l67_67365


namespace find_partition_l67_67155

open Nat

def isBad (S : Finset ℕ) : Prop :=
  ∃ T : Finset ℕ, T ⊆ S ∧ T.sum id = 2012

def partition_not_bad (S : Finset ℕ) (n : ℕ) : Prop :=
  ∃ (P : Finset (Finset ℕ)), P.card = n ∧ (∀ p ∈ P, isBad p = false) ∧ (S = P.sup id)

theorem find_partition :
  ∃ n : ℕ, n = 2 ∧ partition_not_bad (Finset.range (2012 - 503) \ Finset.range 503) n :=
by
  sorry

end find_partition_l67_67155


namespace cos_pi_minus_alpha_cos_double_alpha_l67_67036

open Real

theorem cos_pi_minus_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (π - α) = - sqrt 7 / 3 :=
by
  sorry

theorem cos_double_alpha (α : ℝ) (h1 : sin α = sqrt 2 / 3) (h2 : 0 < α ∧ α < π / 2) :
  cos (2 * α) = 5 / 9 :=
by
  sorry

end cos_pi_minus_alpha_cos_double_alpha_l67_67036


namespace problem1_problem2_l67_67817

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 2)

theorem problem1 (m : ℝ) (h₀ : m > 3) (h₁ : ∃ m, (1/2) * (((m - 1) / 2) - (-(m + 1) / 2) + 3) * (m - 3) = 7 / 2) : m = 4 := by
  sorry

theorem problem2 (a : ℝ) (h₂ : ∃ x, (0 ≤ x ∧ x ≤ 2) ∧ f x ≥ abs (a - 3)) : -2 ≤ a ∧ a ≤ 8 := by
  sorry

end problem1_problem2_l67_67817


namespace exam_combinations_l67_67846

/-- In the "$3+1+2$" examination plan in Hubei Province, 2021,
there are three compulsory subjects: Chinese, Mathematics, and English.
Candidates must choose one subject from Physics and History.
Candidates must choose two subjects from Chemistry, Biology, Ideological and Political Education, and Geography.
Prove that the total number of different combinations of examination subjects is 12.
-/
theorem exam_combinations : exists n : ℕ, n = 12 :=
by
  have compulsory_choice := 1
  have physics_history_choice := 2
  have remaining_subjects_choice := Nat.choose 4 2
  exact Exists.intro (compulsory_choice * physics_history_choice * remaining_subjects_choice) sorry

end exam_combinations_l67_67846


namespace min_x_plus_y_l67_67829

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 := by
  sorry

end min_x_plus_y_l67_67829


namespace inverse_proportion_shift_l67_67126

theorem inverse_proportion_shift (x : ℝ) : 
  (∀ x, y = 6 / x) -> (y = 6 / (x - 3)) :=
by
  intro h
  sorry

end inverse_proportion_shift_l67_67126


namespace average_gas_mileage_round_trip_l67_67312

theorem average_gas_mileage_round_trip :
  (300 / ((150 / 28) + (150 / 18))) = 22 := by
sorry

end average_gas_mileage_round_trip_l67_67312


namespace expression_evaluation_l67_67082

variables {a b c : ℝ}

theorem expression_evaluation 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (habc : a + b + c = 0)
  (h_abacbc : ab + ac + bc ≠ 0) :
  (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = 7 :=
begin
  sorry
end

end expression_evaluation_l67_67082


namespace hours_spent_gaming_l67_67129

def total_hours_in_day : ℕ := 24

def sleeping_fraction : ℚ := 1/3

def studying_fraction : ℚ := 3/4

def gaming_fraction : ℚ := 1/4

theorem hours_spent_gaming :
  let sleeping_hours := total_hours_in_day * sleeping_fraction
  let remaining_hours_after_sleeping := total_hours_in_day - sleeping_hours
  let studying_hours := remaining_hours_after_sleeping * studying_fraction
  let remaining_hours_after_studying := remaining_hours_after_sleeping - studying_hours
  remaining_hours_after_studying * gaming_fraction = 1 :=
by
  sorry

end hours_spent_gaming_l67_67129


namespace medical_team_count_l67_67142

noncomputable def calculateWays : Nat :=
  let C := Nat.choose
  C 3 3 * C 4 1 * C 5 1 +  -- case 1: 3 orthopedic, 1 neurosurgeon, 1 internist
  C 3 1 * C 4 3 * C 5 1 +  -- case 2: 1 orthopedic, 3 neurosurgeons, 1 internist
  C 3 1 * C 4 1 * C 5 3 +  -- case 3: 1 orthopedic, 1 neurosurgeon, 3 internists
  C 3 2 * C 4 2 * C 5 1 +  -- case 4: 2 orthopedic, 2 neurosurgeons, 1 internist
  C 3 1 * C 4 2 * C 5 2 +  -- case 5: 1 orthopedic, 2 neurosurgeons, 2 internists
  C 3 2 * C 4 1 * C 5 2    -- case 6: 2 orthopeics, 1 neurosurgeon, 2 internists

theorem medical_team_count : calculateWays = 630 := by
  sorry

end medical_team_count_l67_67142


namespace tangent_sum_l67_67031

theorem tangent_sum (tan : ℝ → ℝ)
  (h1 : ∀ A B, tan (A + B) = (tan A + tan B) / (1 - tan A * tan B))
  (h2 : tan 60 = Real.sqrt 3) :
  tan 20 + tan 40 + Real.sqrt 3 * tan 20 * tan 40 = Real.sqrt 3 := 
by
  sorry

end tangent_sum_l67_67031


namespace arithmetic_sequence_general_term_geometric_sequence_sum_l67_67355

section ArithmeticSequence

variable {a_n : ℕ → ℤ} {d : ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) (d : ℤ) :=
  ∀ n, a_n (n + 1) - a_n n = d

theorem arithmetic_sequence_general_term (h : is_arithmetic_sequence a_n 2) :
  ∃ a1 : ℤ, ∀ n, a_n n = 2 * n + a1 :=
sorry

end ArithmeticSequence

section GeometricSequence

variable {b_n : ℕ → ℤ} {a_n : ℕ → ℤ}

def is_geometric_sequence_with_reference (b_n : ℕ → ℤ) (a_n : ℕ → ℤ) :=
  b_n 1 = a_n 1 ∧ b_n 2 = a_n 4 ∧ b_n 3 = a_n 13

theorem geometric_sequence_sum (h : is_geometric_sequence_with_reference b_n a_n)
  (h_arith : is_arithmetic_sequence a_n 2) :
  ∃ b1 : ℤ, ∀ n, b_n n = b1 * 3^(n - 1) ∧
                (∃ Sn : ℕ → ℤ, Sn n = (3 * (3^n - 1)) / 2) :=
sorry

end GeometricSequence

end arithmetic_sequence_general_term_geometric_sequence_sum_l67_67355


namespace reciprocals_of_each_other_l67_67374

theorem reciprocals_of_each_other (a b : ℝ) (h : (a + b)^2 - (a - b)^2 = 4) : a * b = 1 :=
by 
  sorry

end reciprocals_of_each_other_l67_67374


namespace angela_final_figures_l67_67159

/-
Problem: Prove that Angela has 7 action figures left given the conditions.
-/

-- Define the initial conditions
def angela_initial_figures := 24
def percentage_increase := 8.3 / 100

-- Compute the new total after percentage increase
def increase := Real.to_rat (Float.round (24 * percentage_increase))
def new_total := angela_initial_figures + increase

-- Compute the number of figures sold
def fraction_sold := 3 / 10
def sold_figures := Real.to_rat (Float.round (new_total * fraction_sold))

-- Remaining figures after selling
def remaining_after_selling := new_total - sold_figures

-- Compute the number of figures given to her daughter
def fraction_given_daughter := 7 / 15
def given_daughter := Real.to_rat (Float.round (remaining_after_selling * fraction_given_daughter))

-- Remaining figures after giving to daughter
def remaining_after_daughter := remaining_after_selling - given_daughter

-- Compute the number of figures given to her nephew
def fraction_given_nephew := 1 / 4
def given_nephew := Real.to_rat (Float.round (remaining_after_daughter * fraction_given_nephew))

-- Remaining figures after giving to nephew 
def remaining_after_nephew := remaining_after_daughter - given_nephew

-- The final number of action figures Angela has left
theorem angela_final_figures : remaining_after_nephew = 7 := by
  sorry

end angela_final_figures_l67_67159


namespace value_of_f_2017_l67_67997

def f (x : ℕ) : ℕ := x^2 - x * (0 : ℕ) - 1

theorem value_of_f_2017 : f 2017 = 2016 * 2018 := by
  sorry

end value_of_f_2017_l67_67997


namespace factorization_correct_l67_67716

theorem factorization_correct (a b : ℝ) : 
  a^2 + 2 * b - b^2 - 1 = (a - b + 1) * (a + b - 1) :=
by
  sorry

end factorization_correct_l67_67716


namespace max_value_of_x3_div_y4_l67_67808

theorem max_value_of_x3_div_y4 (x y : ℝ) (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) :
  ∃ (k : ℝ), k = 27 ∧ ∀ (z : ℝ), z = x^3 / y^4 → z ≤ k :=
by
  sorry

end max_value_of_x3_div_y4_l67_67808


namespace range_of_a_l67_67667

-- Define the function f
def f (a x : ℝ) : ℝ := a^2 * x^2 + a * x - 2

-- Proposition P: f(x) has a root in the interval [-1, 1]
def P (a : ℝ) : Prop := ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f a x = 0

-- Proposition Q: There is only one real number x satisfying the inequality
def Q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0

-- The theorem stating the range of a if either P or Q is false
theorem range_of_a (a : ℝ) : ¬(P a) ∨ ¬(Q a) → (a > -1 ∧ a < 0) ∨ (a > 0 ∧ a < 1) :=
sorry

end range_of_a_l67_67667


namespace vector_dot_product_problem_l67_67813

variables {a b : ℝ}

theorem vector_dot_product_problem (h1 : a + 2 * b = 0) (h2 : (a + b) * a = 2) : a * b = -2 :=
sorry

end vector_dot_product_problem_l67_67813


namespace rolls_combinations_l67_67144

theorem rolls_combinations {n k : ℕ} (h_n : n = 4) (h_k : k = 5) :
  (Nat.choose (n + k - 1) k) = 56 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end rolls_combinations_l67_67144


namespace floor_eq_solution_l67_67980

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l67_67980


namespace complex_expression_solve_combination_l67_67934

-- First Problem
theorem complex_expression (i : ℂ) (h : i = complex.I) :
  ((1 + i) / (1 - i))^2 + complex.abs (3 + 4 * i) - i^2017 = 4 - i :=
  sorry

-- Second Problem
theorem solve_combination (x : ℕ) (hx : x > 6) :
  2 * nat.choose (x - 3) (x - 6) = 5 * nat.choose (x - 4) 2 → x = 18 :=
  sorry

end complex_expression_solve_combination_l67_67934


namespace tan_double_angle_l67_67359

theorem tan_double_angle (α : Real) (h1 : α > π ∧ α < 3 * π / 2) (h2 : Real.sin (π - α) = -3/5) :
  Real.tan (2 * α) = 24/7 := 
by
  sorry

end tan_double_angle_l67_67359


namespace lives_per_each_player_l67_67272

def num_initial_players := 8
def num_quit_players := 3
def total_remaining_lives := 15
def num_remaining_players := num_initial_players - num_quit_players
def lives_per_remaining_player := total_remaining_lives / num_remaining_players

theorem lives_per_each_player :
  lives_per_remaining_player = 3 := by
  sorry

end lives_per_each_player_l67_67272


namespace simplify_and_evaluate_div_expr_l67_67706

variable (m : ℤ)

theorem simplify_and_evaluate_div_expr (h : m = 2) :
  ( (m^2 - 9) / (m^2 - 6 * m + 9) / (1 - 2 / (m - 3)) = -5 / 3) :=
by
  sorry

end simplify_and_evaluate_div_expr_l67_67706


namespace time_upstream_is_correct_l67_67470

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l67_67470


namespace trapezoid_perimeter_calc_l67_67228

theorem trapezoid_perimeter_calc 
  (EF GH : ℝ) (d : ℝ)
  (h_parallel : EF = 10) 
  (h_eq : GH = 22) 
  (h_distance : d = 5) 
  (h_parallel_cond : EF = 10 ∧ GH = 22 ∧ d = 5) 
: 32 + 2 * Real.sqrt 61 = (10 : ℝ) + 2 * (Real.sqrt ((12 / 2)^2 + 5^2)) + 22 := 
by {
  -- The proof goes here, but for now it's omitted
  sorry
}

end trapezoid_perimeter_calc_l67_67228


namespace john_took_11_more_chickens_than_ray_l67_67421

noncomputable def chickens_taken_by_john (mary_chickens : ℕ) : ℕ := mary_chickens + 5
noncomputable def chickens_taken_by_ray (mary_chickens : ℕ) : ℕ := mary_chickens - 6
def ray_chickens : ℕ := 10

-- The theorem to prove:
theorem john_took_11_more_chickens_than_ray :
  ∃ (mary_chickens : ℕ), chickens_taken_by_john mary_chickens - ray_chickens = 11 :=
by
  -- Initial assumptions and derivation steps should be provided here.
  sorry

end john_took_11_more_chickens_than_ray_l67_67421


namespace cost_per_toy_initially_l67_67385

-- defining conditions
def num_toys : ℕ := 200
def percent_sold : ℝ := 0.8
def price_per_toy : ℝ := 30
def profit : ℝ := 800

-- defining the problem
theorem cost_per_toy_initially :
  ((num_toys * percent_sold) * price_per_toy - profit) / (num_toys * percent_sold) = 25 :=
by
  sorry

end cost_per_toy_initially_l67_67385


namespace point_outside_circle_l67_67872

theorem point_outside_circle (a b : ℝ) (h_intersect : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a*x + b*y = 1) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l67_67872


namespace percent_of_x_is_z_l67_67679

def condition1 (z y : ℝ) : Prop := 0.45 * z = 0.72 * y
def condition2 (y x : ℝ) : Prop := y = 0.75 * x
def condition3 (w z : ℝ) : Prop := w = 0.60 * z^2
def condition4 (z w : ℝ) : Prop := z = 0.30 * w^(1/3)

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : condition1 z y) 
  (h2 : condition2 y x)
  (h3 : condition3 w z)
  (h4 : condition4 z w) : 
  z / x = 1.2 :=
sorry

end percent_of_x_is_z_l67_67679


namespace floor_equation_solution_l67_67967

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l67_67967


namespace triangles_needed_for_hexagon_with_perimeter_19_l67_67000

def num_triangles_to_construct_hexagon (perimeter : ℕ) : ℕ :=
  match perimeter with
  | 19 => 59
  | _ => 0  -- We handle only the case where perimeter is 19

theorem triangles_needed_for_hexagon_with_perimeter_19 :
  num_triangles_to_construct_hexagon 19 = 59 :=
by
  -- Here we assert that the number of triangles to construct the hexagon with perimeter 19 is 59
  sorry

end triangles_needed_for_hexagon_with_perimeter_19_l67_67000


namespace original_triangle_area_quadrupled_l67_67712

theorem original_triangle_area_quadrupled {A : ℝ} (h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64)) : A = 4 :=
by
  have h1 : ∀ (a : ℝ), a > 0 → (a * 16 = 64) := by
    intro a ha
    sorry
  sorry

end original_triangle_area_quadrupled_l67_67712


namespace simplify_expression_l67_67191

theorem simplify_expression (a : ℝ) (h : 3 < a ∧ a < 5) : 
  Real.sqrt ((a - 2) ^ 2) + Real.sqrt ((a - 8) ^ 2) = 6 :=
by
  sorry

end simplify_expression_l67_67191


namespace problem_a_problem_b_problem_c_l67_67695

noncomputable def inequality_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (0 * y + 1)) + 1 / (y * (0 * z + 1)) + 1 / (z * (0 * x + 1))) ≥ 3

noncomputable def inequality_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : Prop :=
  (1 / (x * (1 * y + 0)) + 1 / (y * (1 * z + 0)) + 1 / (z * (1 * x + 0))) ≥ 3

noncomputable def inequality_c (x y z : ℝ) (a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : Prop :=
  (1 / (x * (a * y + b)) + 1 / (y * (a * z + b)) + 1 / (z * (a * x + b))) ≥ 3

theorem problem_a (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_a x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_b (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : inequality_b x y z h1 h2 h3 h4 := 
  by sorry

theorem problem_c (x y z a b : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) (h5 : a + b = 1) (h6 : a > 0) (h7 : b > 0) : inequality_c x y z a b h1 h2 h3 h4 h5 h6 h7 :=
  by sorry

end problem_a_problem_b_problem_c_l67_67695


namespace line_repr_exists_same_line_iff_scalar_multiple_l67_67304

-- Given that D is a line in 3D space, there exist a, b, c not all zero
theorem line_repr_exists
  (D : Set (ℝ × ℝ × ℝ)) :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧ 
  (D = {p | ∃ (u v w : ℝ), p = (u, v, w) ∧ a * u + b * v + c * w = 0}) :=
sorry

-- Given two lines represented by different coefficients being the same
-- Prove that the coefficients are scalar multiples of each other
theorem same_line_iff_scalar_multiple
  (α1 β1 γ1 α2 β2 γ2 : ℝ) :
  (∀ (u v w : ℝ), α1 * u + β1 * v + γ1 * w = 0 ↔ α2 * u + β2 * v + γ2 * w = 0) ↔
  (∃ k : ℝ, k ≠ 0 ∧ α2 = k * α1 ∧ β2 = k * β1 ∧ γ2 = k * γ1) :=
sorry

end line_repr_exists_same_line_iff_scalar_multiple_l67_67304


namespace find_annual_interest_rate_l67_67587

theorem find_annual_interest_rate 
  (TD : ℝ) (FV : ℝ) (T : ℝ) (expected_R: ℝ)
  (hTD : TD = 189)
  (hFV : FV = 1764)
  (hT : T = 9 / 12)
  (hExpected : expected_R = 16) : 
  ∃ R : ℝ, 
  (TD = (FV - (FV - TD)) * R * T / 100) ∧ 
  R = expected_R := 
by 
  sorry

end find_annual_interest_rate_l67_67587


namespace total_students_l67_67120

theorem total_students (absent_percent : ℝ) (present_students : ℕ) (total_students : ℝ) :
  absent_percent = 0.14 → present_students = 43 → total_students * (1 - absent_percent) = present_students → total_students = 50 := 
by
  intros
  sorry

end total_students_l67_67120


namespace elevator_passengers_probability_l67_67950

noncomputable def binomial_pdf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem elevator_passengers_probability :
  binomial_pdf 5 (1/3) 4 = 10 / 243 :=
by
  sorry

end elevator_passengers_probability_l67_67950


namespace exponent_evaluation_problem_l67_67298

theorem exponent_evaluation_problem (m : ℕ) : 
  (m^2 * m^3 ≠ m^6) → 
  (m^2 + m^4 ≠ m^6) → 
  ((m^3)^3 ≠ m^6) → 
  (m^7 / m = m^6) :=
by
  intros hA hB hC
  -- Provide the proof here
  sorry

end exponent_evaluation_problem_l67_67298


namespace count_triple_solutions_eq_336847_l67_67255

theorem count_triple_solutions_eq_336847 :
  {n : ℕ // (n = 336847)} :=
begin
  let x y z : ℕ,
  let solutions := { (x, y, z) | x + y + z = 2010 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≤ y ∧ y ≤ z},
  have pos_int_triple_solution_count : solutions.card = 336847,
  {
    -- proof goes here
    sorry
  },
  use 336847,
  exact pos_int_triple_solution_count,
end

end count_triple_solutions_eq_336847_l67_67255


namespace distance_covered_at_40_kmph_l67_67753

theorem distance_covered_at_40_kmph
   (total_distance : ℝ)
   (speed1 : ℝ)
   (speed2 : ℝ)
   (total_time : ℝ)
   (part_distance1 : ℝ) :
   total_distance = 250 ∧
   speed1 = 40 ∧
   speed2 = 60 ∧
   total_time = 6 ∧
   (part_distance1 / speed1 + (total_distance - part_distance1) / speed2 = total_time) →
   part_distance1 = 220 :=
by sorry

end distance_covered_at_40_kmph_l67_67753


namespace range_of_m_length_of_chord_l67_67814

-- Definition of Circle C
def CircleC (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0

-- Definition of Circle D
def CircleD (x y : ℝ) := (x + 3)^2 + (y + 1)^2 = 16

-- Definition of Line l
def LineL (x y : ℝ) := x + 2*y - 4 = 0

-- Problem 1: Prove range of values for m
theorem range_of_m (m : ℝ) : (∀ x y, CircleC x y m) → m < 5 := by
  sorry

-- Problem 2: Prove length of chord MN
theorem length_of_chord (x y : ℝ) :
  CircleC x y 4 ∧ CircleD x y ∧ LineL x y →
  (∃ MN, MN = (4*Real.sqrt 5) / 5) := by
    sorry

end range_of_m_length_of_chord_l67_67814


namespace james_total_matches_l67_67861

-- Define the conditions
def dozen : Nat := 12
def boxes_per_dozen : Nat := 5
def matches_per_box : Nat := 20

-- Calculate the expected number of matches
def expected_matches : Nat := boxes_per_dozen * dozen * matches_per_box

-- State the theorem to be proved
theorem james_total_matches : expected_matches = 1200 := by
  sorry

end james_total_matches_l67_67861


namespace triangle_height_and_segments_l67_67269

-- Define the sides of the triangle
noncomputable def a : ℝ := 13
noncomputable def b : ℝ := 14
noncomputable def c : ℝ := 15

-- Define the height h and the segments m and 15 - m
noncomputable def m : ℝ := 6.6
noncomputable def h : ℝ := 11.2
noncomputable def base_segment_left : ℝ := m
noncomputable def base_segment_right : ℝ := c - m

-- The height and segments calculation theorem
theorem triangle_height_and_segments :
  h = 11.2 ∧ m = 6.6 ∧ (c - m) = 8.4 :=
by {
  sorry
}

end triangle_height_and_segments_l67_67269


namespace three_digit_permuted_mean_l67_67986

theorem three_digit_permuted_mean (N : ℕ) :
  (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
    (N = 111 ∨ N = 222 ∨ N = 333 ∨ N = 444 ∨ N = 555 ∨ N = 666 ∨ N = 777 ∨ N = 888 ∨ N = 999 ∨
     N = 407 ∨ N = 518 ∨ N = 629 ∨ N = 370 ∨ N = 481 ∨ N = 592)) ↔
    (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧ 7 * x = 3 * y + 4 * z) := by
sorry

end three_digit_permuted_mean_l67_67986


namespace tyler_saltwater_aquariums_l67_67733

def num_animals_per_aquarium : ℕ := 39
def total_saltwater_animals : ℕ := 2184

theorem tyler_saltwater_aquariums : 
  total_saltwater_animals / num_animals_per_aquarium = 56 := 
by
  sorry

end tyler_saltwater_aquariums_l67_67733


namespace min_h4_for_ahai_avg_ge_along_avg_plus_4_l67_67013

-- Definitions from conditions
variables (a1 a2 a3 a4 : ℝ)
variables (h1 h2 h3 h4 : ℝ)

-- Conditions from the problem
axiom a1_gt_80 : a1 > 80
axiom a2_gt_80 : a2 > 80
axiom a3_gt_80 : a3 > 80
axiom a4_gt_80 : a4 > 80

axiom h1_eq_a1_plus_1 : h1 = a1 + 1
axiom h2_eq_a2_plus_2 : h2 = a2 + 2
axiom h3_eq_a3_plus_3 : h3 = a3 + 3

-- Lean 4 statement for the problem
theorem min_h4_for_ahai_avg_ge_along_avg_plus_4 : h4 ≥ 99 :=
by
  sorry

end min_h4_for_ahai_avg_ge_along_avg_plus_4_l67_67013


namespace average_price_initial_l67_67243

noncomputable def total_cost_initial (P : ℕ) := 5 * P
noncomputable def total_cost_remaining := 3 * 12
noncomputable def total_cost_returned := 2 * 32

theorem average_price_initial (P : ℕ) : total_cost_initial P = total_cost_remaining + total_cost_returned → P = 20 := 
by
  sorry

end average_price_initial_l67_67243


namespace class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l67_67761

noncomputable def average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + pitch + innovation) / 3

noncomputable def weighted_average_score (costume pitch innovation : ℕ) : ℚ :=
  (costume + 7 * pitch + 2 * innovation) / 10

theorem class_7th_grade_1_has_higher_average_score :
  average_score 90 77 85 > average_score 74 95 80 :=
by sorry

theorem class_7th_grade_2_has_higher_weighted_score :
  weighted_average_score 74 95 80 > weighted_average_score 90 77 85 :=
by sorry

end class_7th_grade_1_has_higher_average_score_class_7th_grade_2_has_higher_weighted_score_l67_67761


namespace brendan_remaining_money_l67_67646

-- Definitions given in the conditions
def weekly_pay (total_monthly_earnings : ℕ) (weeks_in_month : ℕ) : ℕ := total_monthly_earnings / weeks_in_month
def weekly_recharge_amount (weekly_pay : ℕ) : ℕ := weekly_pay / 2
def total_recharge_amount (weekly_recharge_amount : ℕ) (weeks_in_month : ℕ) : ℕ := weekly_recharge_amount * weeks_in_month
def remaining_money_after_car_purchase (total_monthly_earnings : ℕ) (car_cost : ℕ) : ℕ := total_monthly_earnings - car_cost
def total_remaining_money (remaining_money_after_car_purchase : ℕ) (total_recharge_amount : ℕ) : ℕ := remaining_money_after_car_purchase - total_recharge_amount

-- The actual statement to prove
theorem brendan_remaining_money
  (total_monthly_earnings : ℕ := 5000)
  (weeks_in_month : ℕ := 4)
  (car_cost : ℕ := 1500)
  (weekly_pay := weekly_pay total_monthly_earnings weeks_in_month)
  (weekly_recharge_amount := weekly_recharge_amount weekly_pay)
  (total_recharge_amount := total_recharge_amount weekly_recharge_amount weeks_in_month)
  (remaining_money_after_car_purchase := remaining_money_after_car_purchase total_monthly_earnings car_cost)
  (total_remaining_money := total_remaining_money remaining_money_after_car_purchase total_recharge_amount) :
  total_remaining_money = 1000 :=
sorry

end brendan_remaining_money_l67_67646


namespace remainder_is_v_l67_67633

theorem remainder_is_v (x y u v : ℤ) (hx : x > 0) (hy : y > 0)
  (hdiv : x = u * y + v) (hv_range : 0 ≤ v ∧ v < y) :
  (x + (2 * u + 1) * y) % y = v :=
by
  sorry

end remainder_is_v_l67_67633


namespace tables_needed_for_luncheon_l67_67941

theorem tables_needed_for_luncheon (invited attending remaining tables_needed : ℕ) (H1 : invited = 24) (H2 : remaining = 10) (H3 : attending = invited - remaining) (H4 : tables_needed = attending / 7) : tables_needed = 2 :=
by
  sorry

end tables_needed_for_luncheon_l67_67941


namespace expected_allergies_correct_expected_both_correct_l67_67913

noncomputable def p_allergies : ℚ := 2 / 7
noncomputable def sample_size : ℕ := 350
noncomputable def expected_allergies : ℚ := (2 / 7) * 350

noncomputable def p_left_handed : ℚ := 3 / 10
noncomputable def expected_both : ℚ := (3 / 10) * (2 / 7) * 350

theorem expected_allergies_correct : expected_allergies = 100 := by
  sorry

theorem expected_both_correct : expected_both = 30 := by
  sorry

end expected_allergies_correct_expected_both_correct_l67_67913


namespace solve_proof_problem_l67_67692

noncomputable def proof_problem : Prop :=
  let short_videos_per_day := 2
  let short_video_time := 2
  let longer_videos_per_day := 1
  let week_days := 7
  let total_weekly_video_time := 112
  let total_short_video_time_per_week := short_videos_per_day * short_video_time * week_days
  let total_longer_video_time_per_week := total_weekly_video_time - total_short_video_time_per_week
  let longer_video_multiple := total_longer_video_time_per_week / short_video_time
  longer_video_multiple = 42

theorem solve_proof_problem : proof_problem :=
by
  /- Proof goes here -/
  sorry

end solve_proof_problem_l67_67692


namespace range_of_a_l67_67235

open Real

def has_two_distinct_real_roots (a b : ℝ) : Prop :=
  let f (x : ℝ) := a * x^2 + b * (x + 1) - 2
  let equation := λ x => f x - x
  let discriminant := (x : ℝ) => ((b - 1)^2 - 4 * a * (b - 2)) > 0
  ∀ b : ℝ, (discriminant b) > 0

theorem range_of_a (a : ℝ) :
  (∀ b : ℝ, has_two_distinct_real_roots a b) → (0 < a ∧ a < 1) :=
sorry

end range_of_a_l67_67235


namespace problem_statement_l67_67239

noncomputable def myFunction (f : ℝ → ℝ) := 
  (∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x ^ 2 - f x * f y + f y ^ 2)) 

theorem problem_statement (f : ℝ → ℝ) 
  (h : myFunction f) : 
  ∀ x : ℝ, f (2005 * x) = 2005 * f x :=
sorry

end problem_statement_l67_67239


namespace shapes_values_correct_l67_67073

-- Define variable types and conditions
variables (x y z w : ℕ)
variables (sum1 sum2 sum3 sum4 T : ℕ)

-- Define the conditions for the problem as given in (c)
axiom row_sum1 : x + y + z = sum1
axiom row_sum2 : y + z + w = sum2
axiom row_sum3 : z + w + x = sum3
axiom row_sum4 : w + x + y = sum4
axiom col_sum  : x + y + z + w = T

-- Define the variables with specific values as determined in the solution
def triangle := 2
def square := 0
def a_tilde := 6
def O_value := 1

-- Prove that the assigned values satisfy the conditions
theorem shapes_values_correct :
  x = triangle ∧ y = square ∧ z = a_tilde ∧ w = O_value :=
by { sorry }

end shapes_values_correct_l67_67073


namespace monica_milk_l67_67787

theorem monica_milk (don_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) (h_don : don_milk = 3 / 4)
  (h_rachel : rachel_fraction = 1 / 2) (h_monica : monica_fraction = 1 / 3) :
  monica_fraction * (rachel_fraction * don_milk) = 1 / 8 :=
by
  sorry

end monica_milk_l67_67787


namespace isabel_earned_l67_67689

theorem isabel_earned :
  let bead_necklace_price := 4
  let gemstone_necklace_price := 8
  let bead_necklace_count := 3
  let gemstone_necklace_count := 3
  let sales_tax_rate := 0.05
  let discount_rate := 0.10

  let total_cost_before_tax := bead_necklace_count * bead_necklace_price + gemstone_necklace_count * gemstone_necklace_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  let discount := total_cost_after_tax * discount_rate
  let final_amount_earned := total_cost_after_tax - discount

  final_amount_earned = 34.02 :=
by {
  sorry
}

end isabel_earned_l67_67689


namespace min_value_of_quadratic_l67_67337

theorem min_value_of_quadratic (x : ℝ) : 
  ∃ m : ℝ, (∀ z : ℝ, z = 5 * x ^ 2 + 20 * x + 25 → z ≥ m) ∧ m = 5 :=
by
  sorry

end min_value_of_quadratic_l67_67337


namespace cookies_with_five_cups_l67_67774

theorem cookies_with_five_cups (cookies_per_four_cups : ℕ) (flour_for_four_cups : ℕ) (flour_for_five_cups : ℕ) (h : 24 / 4 = cookies_per_four_cups / 5) :
  cookies_per_four_cups = 30 :=
by
  sorry

end cookies_with_five_cups_l67_67774


namespace triangle_area_heron_l67_67017

open Real

theorem triangle_area_heron : 
  ∀ (a b c : ℝ), a = 12 → b = 13 → c = 5 → 
  (let s := (a + b + c) / 2 in sqrt (s * (s - a) * (s - b) * (s - c)) = 30) :=
by
  intros a b c h1 h2 h3
  sorry

end triangle_area_heron_l67_67017


namespace seq_eighth_term_l67_67065

theorem seq_eighth_term : (8^2 + 2 * 8 - 1 = 79) :=
by
  sorry

end seq_eighth_term_l67_67065


namespace x_squared_y_minus_xy_squared_l67_67666

theorem x_squared_y_minus_xy_squared (x y : ℝ) (h1 : x - y = -2) (h2 : x * y = 3) : x^2 * y - x * y^2 = -6 := 
by 
  sorry

end x_squared_y_minus_xy_squared_l67_67666


namespace sum_abs_values_of_factors_l67_67680

theorem sum_abs_values_of_factors (a w c d : ℤ)
  (h1 : 6 * (x : ℤ)^2 + x - 12 = (a * x + w) * (c * x + d)) :
  abs a + abs w + abs c + abs d = 22 :=
sorry

end sum_abs_values_of_factors_l67_67680


namespace product_of_two_numbers_l67_67594

theorem product_of_two_numbers (x y : ℚ) 
  (h1 : x + y = 8 * (x - y)) 
  (h2 : x * y = 15 * (x - y)) : 
  x * y = 100 / 7 := 
by 
  sorry

end product_of_two_numbers_l67_67594


namespace same_terminal_side_l67_67710

theorem same_terminal_side (k : ℤ): ∃ k : ℤ, 1303 = k * 360 - 137 := by
  -- Proof left as an exercise.
  sorry

end same_terminal_side_l67_67710


namespace ay_bz_cx_lt_S_squared_l67_67557

theorem ay_bz_cx_lt_S_squared 
  (S : ℝ) (a b c x y z : ℝ) 
  (hS : 0 < S) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h1 : a + x = S) 
  (h2 : b + y = S) 
  (h3 : c + z = S) : 
  a * y + b * z + c * x < S^2 := 
sorry

end ay_bz_cx_lt_S_squared_l67_67557


namespace part_one_solution_set_part_two_range_of_m_l67_67671

-- Part I
theorem part_one_solution_set (x : ℝ) : (|x + 1| + |x - 2| - 5 > 0) ↔ (x > 3 ∨ x < -2) :=
sorry

-- Part II
theorem part_two_range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) ↔ (m ≤ 1) :=
sorry

end part_one_solution_set_part_two_range_of_m_l67_67671


namespace vacation_cost_eq_l67_67586

theorem vacation_cost_eq (C : ℕ) (h : C / 3 - C / 5 = 50) : C = 375 :=
sorry

end vacation_cost_eq_l67_67586


namespace athlete_groups_l67_67002

/-- A school has athletes divided into groups.
   - If there are 7 people per group, there will be 3 people left over.
   - If there are 8 people per group, there will be a shortage of 5 people.
The goal is to prove that the system of equations is valid --/
theorem athlete_groups (x y : ℕ) :
  7 * y = x - 3 ∧ 8 * y = x + 5 := 
by 
  sorry

end athlete_groups_l67_67002


namespace largest_integer_l67_67460

theorem largest_integer (n : ℕ) : n ^ 200 < 5 ^ 300 → n <= 11 :=
by
  sorry

end largest_integer_l67_67460


namespace james_total_matches_l67_67858

def boxes_count : ℕ := 5 * 12
def matches_per_box : ℕ := 20
def total_matches (boxes : ℕ) (matches_per_box : ℕ) : ℕ := boxes * matches_per_box

theorem james_total_matches : total_matches boxes_count matches_per_box = 1200 :=
by {
  sorry
}

end james_total_matches_l67_67858


namespace incorrect_inequality_l67_67799

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬ (-4 * a < -4 * b) :=
by sorry

end incorrect_inequality_l67_67799


namespace remainder_when_x_squared_divided_by_20_l67_67539

theorem remainder_when_x_squared_divided_by_20
  (x : ℤ)
  (h1 : 5 * x ≡ 10 [ZMOD 20])
  (h2 : 2 * x ≡ 8 [ZMOD 20]) :
  x^2 ≡ 16 [ZMOD 20] :=
sorry

end remainder_when_x_squared_divided_by_20_l67_67539


namespace expression_simplifies_to_neg_seven_l67_67083

theorem expression_simplifies_to_neg_seven (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
(h₃ : a + b + c = 0) (h₄ : ab + ac + bc ≠ 0) : 
    (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
by
  sorry

end expression_simplifies_to_neg_seven_l67_67083


namespace compound_oxygen_atoms_l67_67622

theorem compound_oxygen_atoms 
  (C_atoms : ℕ)
  (H_atoms : ℕ)
  (total_molecular_weight : ℝ)
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (atomic_weight_O : ℝ) :
  C_atoms = 4 →
  H_atoms = 8 →
  total_molecular_weight = 88 →
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  atomic_weight_O = 16.00 →
  (total_molecular_weight - (C_atoms * atomic_weight_C + H_atoms * atomic_weight_H)) / atomic_weight_O = 2 := 
by 
  intros;
  sorry

end compound_oxygen_atoms_l67_67622


namespace nm_odd_if_squares_sum_odd_l67_67832

theorem nm_odd_if_squares_sum_odd
  (n m : ℤ)
  (h : (n^2 + m^2) % 2 = 1) :
  (n * m) % 2 = 1 :=
sorry

end nm_odd_if_squares_sum_odd_l67_67832


namespace expected_absolute_deviation_greater_in_10_tosses_l67_67287

variable {n m : ℕ}

def frequency_of_heads (m n : ℕ) : ℚ := m / n

def deviation_from_probability (m n : ℕ) : ℚ :=
  m / n - 0.5

def absolute_deviation (m n : ℕ) : ℚ :=
  |m / n - 0.5|

noncomputable def expectation_absolute_deviation (n : ℕ) : ℚ := 
  if n = 10 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 10 tosses
    sorry 
  else if n = 100 then 
    -- Here we would ideally have a calculation of the expectation based on given conditions for 100 tosses
    sorry 
  else 0

theorem expected_absolute_deviation_greater_in_10_tosses (a b : ℚ) :
  expectation_absolute_deviation 10 > expectation_absolute_deviation 100 :=
by sorry

end expected_absolute_deviation_greater_in_10_tosses_l67_67287


namespace upstream_travel_time_l67_67466

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l67_67466


namespace arithmetic_seq_a12_l67_67356

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 = 1)
  (h2 : a 7 + a 9 = 16)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 12 = 15 :=
by sorry

end arithmetic_seq_a12_l67_67356


namespace op_proof_l67_67333

-- Definition of the operation \(\oplus\)
def op (x y : ℝ) : ℝ := x^2 + y

-- Theorem statement for the given proof problem
theorem op_proof (h : ℝ) : op h (op h h) = 2 * h^2 + h :=
by 
  sorry

end op_proof_l67_67333


namespace guessing_game_l67_67643

-- Define the conditions
def number : ℕ := 33
def result : ℕ := 2 * 51 - 3

-- Define the factor (to be proven)
def factor (n r : ℕ) : ℕ := r / n

-- The theorem to be proven
theorem guessing_game (n r : ℕ) (h1 : n = 33) (h2 : r = 2 * 51 - 3) : 
  factor n r = 3 := by
  -- Placeholder for the actual proof
  sorry

end guessing_game_l67_67643


namespace greatest_four_digit_number_divisible_by_6_and_12_l67_67598

theorem greatest_four_digit_number_divisible_by_6_and_12 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 6 = 0) ∧ (n % 12 = 0) ∧ 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ (m % 6 = 0) ∧ (m % 12 = 0) → m ≤ n) ∧
  n = 9996 := 
by
  sorry

end greatest_four_digit_number_divisible_by_6_and_12_l67_67598


namespace circle_diameter_line_eq_l67_67815

theorem circle_diameter_line_eq (x y : ℝ) :
  x^2 + y^2 - 2*x + 6*y + 8 = 0 → (2 * 1 + (-3) + 1 = 0) :=
by
  sorry

end circle_diameter_line_eq_l67_67815


namespace jian_wins_cases_l67_67736

inductive Move
| rock : Move
| paper : Move
| scissors : Move

def wins (jian shin : Move) : Prop :=
  (jian = Move.rock ∧ shin = Move.scissors) ∨
  (jian = Move.paper ∧ shin = Move.rock) ∨
  (jian = Move.scissors ∧ shin = Move.paper)

theorem jian_wins_cases : ∃ n : Nat, n = 3 ∧ (∀ jian shin, wins jian shin → n = 3) :=
by
  sorry

end jian_wins_cases_l67_67736


namespace root_quadratic_l67_67213

theorem root_quadratic (m : ℝ) (h : m^2 - 2*m - 1 = 0) : m^2 + 1/m^2 = 6 :=
sorry

end root_quadratic_l67_67213


namespace parabola1_right_of_parabola2_l67_67173

def parabola1 (x : ℝ) : ℝ := x^2 - x + 3
def parabola2 (x : ℝ) : ℝ := x^2 + x + 5

theorem parabola1_right_of_parabola2 :
  ∃ x1 x2 : ℝ, x1 > x2 ∧ parabola1 x1 < parabola2 x2 :=
by
  sorry

end parabola1_right_of_parabola2_l67_67173


namespace fraction_neg_range_l67_67843

theorem fraction_neg_range (x : ℝ) : (x ≠ 0 ∧ x < 1) ↔ (x - 1 < 0 ∧ x^2 > 0) := by
  sorry

end fraction_neg_range_l67_67843


namespace solution_set_of_x_l67_67975

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l67_67975


namespace ratio_A_B_l67_67791

noncomputable def A := (1 * 2 * 7) + (2 * 4 * 14) + (3 * 6 * 21) + (4 * 8 * 28)
noncomputable def B := (1 * 3 * 5) + (2 * 6 * 10) + (3 * 9 * 15) + (4 * 12 * 20)

theorem ratio_A_B :
  let r := (A : ℚ) / (B : ℚ) in 0 < r ∧ r < 1 :=
by {
  -- Proof steps are omitted with sorry
  sorry
}

end ratio_A_B_l67_67791


namespace tangent_line_eqn_l67_67177

noncomputable def f (x : ℝ) : ℝ := 5 * x + Real.log x

theorem tangent_line_eqn : ∀ x y : ℝ, (x, y) = (1, f 1) → 6 * x - y - 1 = 0 := 
by
  intro x y h
  sorry

end tangent_line_eqn_l67_67177


namespace problem_part_1_problem_part_2_l67_67372

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b : ℝ × ℝ := (3, -Real.sqrt 3)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * vector_b.1 + (vector_a x).2 * vector_b.2

theorem problem_part_1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) : 
  (vector_a x).1 * vector_b.2 = (vector_a x).2 * vector_b.1 → 
  x = 5 * Real.pi / 6 :=
by
  sorry

theorem problem_part_2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi) :
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≤ f t) → x = 0 ∧ f 0 = 3 ∧ 
  (∀ t, 0 ≤ t ∧ t ≤ Real.pi → f x ≥ f t) → x = 5 * Real.pi / 6 ∧ f (5 * Real.pi / 6) = -2 * Real.sqrt 3 :=
by
  sorry

end problem_part_1_problem_part_2_l67_67372


namespace ratio_of_areas_inequality_l67_67928

theorem ratio_of_areas_inequality (a x m : ℝ) (h1 : a > 0) (h2 : x > 0) (h3 : x < a) :
  m = (3 * x^2 - 3 * a * x + a^2) / a^2 →
  (1 / 4 ≤ m ∧ m < 1) :=
sorry

end ratio_of_areas_inequality_l67_67928


namespace intersection_M_N_l67_67204

open Set

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_M_N :
  M ∩ N = { x | -2 ≤ x ∧ x ≤ -1 } := by
  sorry

end intersection_M_N_l67_67204


namespace line_tangent_through_A_l67_67603

theorem line_tangent_through_A {A : ℝ × ℝ} (hA : A = (1, 2)) : 
  ∃ m b : ℝ, (b = 2) ∧ (∀ x : ℝ, y = m * x + b) ∧ (∀ y x : ℝ, y^2 = 4*x → y = 2) :=
by
  sorry

end line_tangent_through_A_l67_67603


namespace solution_set_l67_67977

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l67_67977


namespace simon_sand_dollars_l67_67570

theorem simon_sand_dollars (S G P : ℕ) (h1 : G = 3 * S) (h2 : P = 5 * G) (h3 : S + G + P = 190) : S = 10 := by
  sorry

end simon_sand_dollars_l67_67570


namespace min_moves_to_equalize_boxes_l67_67440

def initialCoins : List ℕ := [5, 8, 11, 17, 20, 15, 10]

def targetCoins (boxes : List ℕ) : ℕ := boxes.sum / boxes.length

def movesRequiredToBalance : List ℕ → ℕ
| [5, 8, 11, 17, 20, 15, 10] => 22
| _ => sorry

theorem min_moves_to_equalize_boxes :
  movesRequiredToBalance initialCoins = 22 :=
by
  sorry

end min_moves_to_equalize_boxes_l67_67440


namespace B_join_time_l67_67752

theorem B_join_time (x : ℕ) (hx : (45000 * 12) / (27000 * (12 - x)) = 2) : x = 2 :=
sorry

end B_join_time_l67_67752


namespace selection_methods_count_l67_67729

/-- Consider a school with 16 teachers, divided into four departments (First grade, Second grade, Third grade, and Administrative department), with 4 teachers each. 
We need to select 3 leaders such that not all leaders are from the same department and at least one leader is from the Administrative department. 
Prove that the number of different selection methods that satisfy these conditions is 336. -/
theorem selection_methods_count :
  let num_teachers := 16
  let teachers_per_department := 4
  ∃ (choose : ℕ → ℕ → ℕ), 
  choose num_teachers 3 = 336 :=
  sorry

end selection_methods_count_l67_67729


namespace fraction_sum_possible_l67_67780

theorem fraction_sum_possible :
  ∃ a b c d e f g h i : ℕ,
    {a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i ∧
    (a.toRat / b.toRat) + (c.toRat / d.toRat) + (e.toRat / f.toRat) + (g.toRat / h.toRat) = i :=
sorry

end fraction_sum_possible_l67_67780


namespace find_numbers_l67_67405

theorem find_numbers (a b : ℕ) (h1 : a > 11) (h2 : b > 11) (h3 : a ≠ b)
  (h4 : (∃ S, S = a + b) ∧ (∀ (x y : ℕ), x ≠ y → x + y = a + b → (x > 11) → (y > 11) → ¬(x = a ∨ y = a) → ¬(x = b ∨ y = b)))
  (h5 : even a ∨ even b) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end find_numbers_l67_67405


namespace max_n_is_11_l67_67890

noncomputable def max_n (a1 d : ℝ) : ℕ :=
if h : d < 0 then
  11
else
  sorry

theorem max_n_is_11 (d : ℝ) (a1 : ℝ) (c : ℝ) :
  (d / 2) * (22 ^ 2) + (a1 - (d / 2)) * 22 + c ≥ 0 →
  22 = (a1 - (d / 2)) / (- (d / 2)) →
  max_n a1 d = 11 :=
by
  intros h1 h2
  rw [max_n]
  split_ifs
  · exact rfl
  · exact sorry

end max_n_is_11_l67_67890


namespace simplify_fractional_exponents_l67_67485

theorem simplify_fractional_exponents :
  (5 ^ (1/6) * 5 ^ (1/2)) / 5 ^ (1/3) = 5 ^ (1/6) :=
by
  sorry

end simplify_fractional_exponents_l67_67485


namespace cooper_pies_days_l67_67332

theorem cooper_pies_days :
  ∃ d : ℕ, 7 * d - 50 = 34 ∧ d = 12 :=
by
  sorry

end cooper_pies_days_l67_67332


namespace noemi_lost_on_roulette_l67_67700

theorem noemi_lost_on_roulette (initial_purse := 1700) (final_purse := 800) (loss_on_blackjack := 500) :
  (initial_purse - final_purse) - loss_on_blackjack = 400 := by
  sorry

end noemi_lost_on_roulette_l67_67700


namespace power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l67_67510

theorem power_function_condition (m : ℝ) : m^2 + 2 * m = 1 ↔ m = -1 + Real.sqrt 2 ∨ m = -1 - Real.sqrt 2 :=
by sorry

theorem direct_proportionality_condition (m : ℝ) : (m^2 + m - 1 = 1 ∧ m^2 + 3 * m ≠ 0) ↔ m = 1 :=
by sorry

theorem inverse_proportionality_condition (m : ℝ) : (m^2 + m - 1 = -1 ∧ m^2 + 3 * m ≠ 0) ↔ m = -1 :=
by sorry

end power_function_condition_direct_proportionality_condition_inverse_proportionality_condition_l67_67510


namespace sum_first_five_even_numbers_l67_67441

theorem sum_first_five_even_numbers : (2 + 4 + 6 + 8 + 10) = 30 :=
by
  sorry

end sum_first_five_even_numbers_l67_67441


namespace ratio_of_spinsters_to_cats_l67_67265

-- Defining the problem in Lean 4
theorem ratio_of_spinsters_to_cats (S C : ℕ) (h₁ : S = 22) (h₂ : C = S + 55) : S / gcd S C = 2 ∧ C / gcd S C = 7 :=
by
  sorry

end ratio_of_spinsters_to_cats_l67_67265


namespace find_f_of_3_l67_67559

theorem find_f_of_3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x * f y - y) = x * y - f y) 
  (h2 : f 0 = 0) (h3 : ∀ x : ℝ, f (-x) = -f x) : f 3 = 3 :=
sorry

end find_f_of_3_l67_67559


namespace time_upstream_is_correct_l67_67472

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l67_67472


namespace number_of_people_in_group_l67_67575

theorem number_of_people_in_group :
  ∀ (N : ℕ), (75 - 35) = 5 * N → N = 8 :=
by
  intros N h
  sorry

end number_of_people_in_group_l67_67575


namespace rational_coordinates_l67_67097

theorem rational_coordinates (x : ℚ) : ∃ y : ℚ, 2 * x^3 + 2 * y^3 - 3 * x^2 - 3 * y^2 + 1 = 0 :=
by
  use (1 - x)
  sorry

end rational_coordinates_l67_67097


namespace tan_eq_123_deg_l67_67347

theorem tan_eq_123_deg (n : ℤ) (h : -180 < n ∧ n < 180) : 
  real.tan (n * real.pi / 180) = real.tan (123 * real.pi / 180) → n = 123 ∨ n = -57 :=
by
  -- to do the proof
  sorry

end tan_eq_123_deg_l67_67347


namespace angle_of_triangle_l67_67069

noncomputable def triangle_angles (A B C D : Point ℝ) (α β γ : Real.Angle) : Prop :=
  Triangle A B C ∧
  Barycentric.CentralAngleBisection A B C D ∧
  (|BD| * |CD| = |AD|^2) ∧
  angle A D B = Real.Angle.pi / 4 ∧
  α = angle B A C ∧
  β = angle A B C ∧
  γ = angle A C B

theorem angle_of_triangle 
  (A B C D : Point ℝ)
  (h1 : Triangle A B C)
  (h2 : Barycentric.CentralAngleBisection A B C D)
  (h3 : |(B - D) * (C - D)| = |A - D|^2)
  (h4 : Real.Angle.pi / 4 = angle A D B) :
  ∃ (α β γ : Real.Angle), triangle_angles A B C D α β γ ∧ 
  α = Real.Angle.pi / 3 ∧ 
  β = 7 * Real.Angle.pi / 12 ∧ 
  γ = Real.Angle.pi / 12 :=
by 
  sorry

end angle_of_triangle_l67_67069


namespace value_of_a_minus_b_l67_67515

theorem value_of_a_minus_b (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : 2 * a - b = 1) : a - b = -1 :=
by
  sorry

end value_of_a_minus_b_l67_67515


namespace find_side_b_l67_67854

theorem find_side_b (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3)
    (hC : C = Real.pi / 3) (hA : A = Real.pi / 6) (hB : B = Real.pi / 2) : b = 4 := by
  sorry

end find_side_b_l67_67854


namespace smallest_class_number_l67_67771

theorem smallest_class_number (x : ℕ)
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = 57) :
  x = 2 :=
by sorry

end smallest_class_number_l67_67771


namespace theta_in_second_quadrant_l67_67827

open Real

-- Definitions for conditions
def cond1 (θ : ℝ) : Prop := sin θ > cos θ
def cond2 (θ : ℝ) : Prop := tan θ < 0

-- Main theorem statement
theorem theta_in_second_quadrant (θ : ℝ) 
  (h1 : cond1 θ) 
  (h2 : cond2 θ) : 
  θ > π/2 ∧ θ < π :=
sorry

end theta_in_second_quadrant_l67_67827


namespace total_price_of_hats_l67_67597

variables (total_hats : ℕ) (blue_hat_cost : ℕ) (green_hat_cost : ℕ) (green_hats : ℕ) (total_price : ℕ)

def total_number_of_hats := 85
def cost_per_blue_hat := 6
def cost_per_green_hat := 7
def number_of_green_hats := 30

theorem total_price_of_hats :
  (number_of_green_hats * cost_per_green_hat) + ((total_number_of_hats - number_of_green_hats) * cost_per_blue_hat) = 540 :=
sorry

end total_price_of_hats_l67_67597


namespace votes_cast_l67_67452

theorem votes_cast (V : ℝ) (h1 : ∃ (x : ℝ), x = 0.35 * V) (h2 : ∃ (y : ℝ), y = x + 2100) : V = 7000 :=
by sorry

end votes_cast_l67_67452


namespace number_of_hexagons_l67_67616

-- Definitions based on conditions
def num_pentagons : ℕ := 12

-- Based on the problem statement, the goal is to prove that the number of hexagons is 20
theorem number_of_hexagons (h : num_pentagons = 12) : ∃ (num_hexagons : ℕ), num_hexagons = 20 :=
by {
  -- proof would be here
  sorry
}

end number_of_hexagons_l67_67616


namespace total_marbles_l67_67676

theorem total_marbles
  (R B Y : ℕ)  -- Red, Blue, and Yellow marbles as natural numbers
  (h_ratio : 2 * (R + B + Y) = 9 * Y)  -- The ratio condition translated
  (h_yellow : Y = 36)  -- The number of yellow marbles condition
  : R + B + Y = 81 :=  -- Statement that the total number of marbles is 81
sorry

end total_marbles_l67_67676


namespace perpendicular_vectors_t_values_l67_67047

variable (t : ℝ)
def a := (t, 0, -1)
def b := (2, 5, t^2)

theorem perpendicular_vectors_t_values (h : (2 * t + 0 * 5 + -1 * t^2) = 0) : t = 0 ∨ t = 2 :=
by sorry

end perpendicular_vectors_t_values_l67_67047


namespace max_sum_of_squares_eq_l67_67446

theorem max_sum_of_squares_eq (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end max_sum_of_squares_eq_l67_67446


namespace DE_plus_FG_equals_19_div_6_l67_67910

theorem DE_plus_FG_equals_19_div_6
    (AB AC : ℝ)
    (BC : ℝ)
    (h_isosceles : AB = 2 ∧ AC = 2 ∧ BC = 1.5)
    (D E G F : ℝ)
    (h_parallel_DE_BC : D = E)
    (h_parallel_FG_BC : F = G)
    (h_same_perimeter : 2 + D = 2 + F ∧ 2 + F = 5.5 - F) :
    D + F = 19 / 6 := by
  sorry

end DE_plus_FG_equals_19_div_6_l67_67910


namespace certain_number_is_17_l67_67830

theorem certain_number_is_17 (x : ℤ) (h : 2994 / x = 177) : x = 17 :=
by
  sorry

end certain_number_is_17_l67_67830


namespace veronica_loss_more_than_seth_l67_67885

noncomputable def seth_loss : ℝ := 17.5
noncomputable def jerome_loss : ℝ := 3 * seth_loss
noncomputable def total_loss : ℝ := 89
noncomputable def veronica_loss : ℝ := total_loss - (seth_loss + jerome_loss)

theorem veronica_loss_more_than_seth :
  veronica_loss - seth_loss = 1.5 :=
by
  have h_seth_loss : seth_loss = 17.5 := rfl
  have h_jerome_loss : jerome_loss = 3 * seth_loss := rfl
  have h_total_loss : total_loss = 89 := rfl
  have h_veronica_loss : veronica_loss = total_loss - (seth_loss + jerome_loss) := rfl
  sorry

end veronica_loss_more_than_seth_l67_67885


namespace sum_of_last_digits_l67_67110

theorem sum_of_last_digits (num : Nat → Nat) (a b : Nat) :
  (∀ i, 1 ≤ i ∧ i < 2000 → (num i * 10 + num (i + 1)) % 17 = 0 ∨ (num i * 10 + num (i + 1)) % 23 = 0) →
  num 1 = 3 →
  (num 2000 = a ∨ num 2000 = b) →
  a = 2 →
  b = 5 →
  a + b = 7 :=
by 
  sorry

end sum_of_last_digits_l67_67110


namespace total_goals_l67_67133

theorem total_goals (B M : ℕ) (hB : B = 4) (hM : M = 3 * B) : B + M = 16 := by
  sorry

end total_goals_l67_67133


namespace sum_of_roots_l67_67431

def quadratic_polynomial (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def condition (a b c : ℝ) (x : ℝ) :=
  quadratic_polynomial a b c (x^3 + x) ≥ quadratic_polynomial a b c (x^2 + 1)

theorem sum_of_roots (a b c : ℝ) (h : ∀ x : ℝ, condition a b c x) :
  b = -4 * a → -(b / a) = 4 :=
by
  sorry

end sum_of_roots_l67_67431


namespace minimum_force_to_submerge_cube_l67_67128

-- Definitions and given conditions
def volume_cube : ℝ := 10e-6 -- 10 cm^3 in m^3
def density_cube : ℝ := 700 -- in kg/m^3
def density_water : ℝ := 1000 -- in kg/m^3
def gravity : ℝ := 10 -- in m/s^2

-- Prove the minimum force required to submerge the cube completely
theorem minimum_force_to_submerge_cube : 
  (density_water * volume_cube * gravity - density_cube * volume_cube * gravity) = 0.03 :=
by
  sorry

end minimum_force_to_submerge_cube_l67_67128


namespace advance_tickets_sold_20_l67_67156

theorem advance_tickets_sold_20 :
  ∃ (A S : ℕ), 20 * A + 30 * S = 1600 ∧ A + S = 60 ∧ A = 20 :=
by
  sorry

end advance_tickets_sold_20_l67_67156


namespace probability_of_less_than_5_is_one_half_l67_67749

noncomputable def probability_of_less_than_5 : ℚ :=
  let total_outcomes := 8
  let successful_outcomes := 4
  successful_outcomes / total_outcomes

theorem probability_of_less_than_5_is_one_half :
  probability_of_less_than_5 = 1 / 2 :=
by
  -- proof omitted
  sorry

end probability_of_less_than_5_is_one_half_l67_67749


namespace rival_awards_l67_67691

theorem rival_awards (scott_awards jessie_awards rival_awards : ℕ)
  (h1 : scott_awards = 4)
  (h2 : jessie_awards = 3 * scott_awards)
  (h3 : rival_awards = 2 * jessie_awards) :
  rival_awards = 24 :=
by sorry

end rival_awards_l67_67691


namespace constant_seq_is_arith_not_always_geom_l67_67004

theorem constant_seq_is_arith_not_always_geom (c : ℝ) (seq : ℕ → ℝ) (h : ∀ n, seq n = c) :
  (∀ n, seq (n + 1) - seq n = 0) ∧ (c = 0 ∨ (∀ n, seq (n + 1) / seq n = 1)) :=
by
  sorry

end constant_seq_is_arith_not_always_geom_l67_67004


namespace problem_one_problem_two_l67_67327

-- Problem 1
theorem problem_one : -9 + 5 * (-6) - 18 / (-3) = -33 :=
by
  sorry

-- Problem 2
theorem problem_two : ((-3/4) - (5/8) + (9/12)) * (-24) + (-8) / (2/3) = -6 :=
by
  sorry

end problem_one_problem_two_l67_67327


namespace area_of_original_square_l67_67187

theorem area_of_original_square 
  (x : ℝ) 
  (h0 : x * (x - 3) = 40) 
  (h1 : 0 < x) : 
  x ^ 2 = 64 := 
sorry

end area_of_original_square_l67_67187


namespace greatest_number_is_2040_l67_67592

theorem greatest_number_is_2040 (certain_number : ℕ) : 
  (∀ d : ℕ, d ∣ certain_number ∧ d ∣ 2037 → d ≤ 1) ∧ 
  (certain_number % 1 = 10) ∧ 
  (2037 % 1 = 7) → 
  certain_number = 2040 :=
by
  sorry

end greatest_number_is_2040_l67_67592


namespace simplify_expression_l67_67963

theorem simplify_expression (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 :=
by 
  sorry

end simplify_expression_l67_67963


namespace find_x_l67_67055

theorem find_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end find_x_l67_67055


namespace quadratic_inequality_solution_l67_67681

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
  sorry

end quadratic_inequality_solution_l67_67681


namespace tips_collected_l67_67878

-- Definitions based on conditions
def total_collected : ℕ := 240
def hourly_wage : ℕ := 10
def hours_worked : ℕ := 19

-- Correct answer translated into a proof problem
theorem tips_collected : total_collected - (hours_worked * hourly_wage) = 50 := by
  sorry

end tips_collected_l67_67878


namespace number_of_possible_m_values_l67_67556

theorem number_of_possible_m_values :
  ∃ m_set : Finset ℤ, (∀ x1 x2 : ℤ, x1 * x2 = 40 → (x1 + x2) ∈ m_set) ∧ m_set.card = 8 :=
sorry

end number_of_possible_m_values_l67_67556


namespace complex_multiplication_imaginary_unit_l67_67201

theorem complex_multiplication_imaginary_unit 
  (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_imaginary_unit_l67_67201


namespace quadratic_root_d_value_l67_67842

theorem quadratic_root_d_value :
  (∃ d : ℝ, ∀ x : ℝ, (2 * x^2 + 8 * x + d = 0) ↔ (x = (-8 + Real.sqrt 12) / 4) ∨ (x = (-8 - Real.sqrt 12) / 4)) → 
  d = 6.5 :=
by
  sorry

end quadratic_root_d_value_l67_67842


namespace bug_on_square_moves_l67_67619

theorem bug_on_square_moves :
  ∃ (m n : ℕ), (∀ k < n, Nat.gcd k n = 1) ∧
  (Q : ℕ → ℚ) (Q_0 : Q 0 = 1) (Q_n : ∀ n, Q (n + 1) = 1 / 3 * (1 - Q n))
  (m = 44287 ∧ n = 177147) ∧ 
  (Q 12 = m / n) ∧
  (m + n = 221434) :=
by
  sorry

end bug_on_square_moves_l67_67619


namespace leak_empty_tank_time_l67_67302

theorem leak_empty_tank_time (A L : ℝ) (hA : A = 1 / 10) (hAL : A - L = 1 / 15) : (1 / L = 30) :=
sorry

end leak_empty_tank_time_l67_67302


namespace friends_count_l67_67018

variables (F : ℕ)
def cindy_initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def marbles_given : ℕ := F * marbles_per_friend
def marbles_remaining := cindy_initial_marbles - marbles_given

theorem friends_count (h : 4 * marbles_remaining = 720) : F = 4 :=
by sorry

end friends_count_l67_67018


namespace area_quadrilateral_EFGH_l67_67161

-- Define the rectangles ABCD and XYZR
def area_rectangle_ABCD : ℝ := 60 
def area_rectangle_XYZR : ℝ := 4

-- Define what needs to be proven: the area of quadrilateral EFGH
theorem area_quadrilateral_EFGH (a b c d : ℝ) :
  (area_rectangle_ABCD = area_rectangle_XYZR + 2 * (a + b + c + d)) →
  (a + b + c + d = 28) →
  (area_rectangle_XYZR = 4) →
  (area_rectangle_ABCD = 60) →
  (a + b + c + d + area_rectangle_XYZR = 32) :=
by
  intros h1 h2 h3 h4
  sorry

end area_quadrilateral_EFGH_l67_67161


namespace unique_solution_exists_l67_67989

theorem unique_solution_exists :
  ∃! (x y : ℝ), 4^(x^2 + 2 * y) + 4^(2 * x + y^2) = Real.cos (Real.pi * x) ∧ (x, y) = (2, -2) :=
by
  sorry

end unique_solution_exists_l67_67989


namespace fraction_meaningful_domain_l67_67836

theorem fraction_meaningful_domain (x : ℝ) : (∃ f : ℝ, f = 1 / (x - 2)) → x ≠ 2 :=
by
  sorry

end fraction_meaningful_domain_l67_67836


namespace parallel_lines_slope_eq_l67_67280

theorem parallel_lines_slope_eq (a : ℝ) : (∀ x y : ℝ, 3 * y - 4 * a = 8 * x) ∧ (∀ x y : ℝ, y - 2 = (a + 4) * x) → a = -4 / 3 :=
by
  sorry

end parallel_lines_slope_eq_l67_67280


namespace units_digit_of_n_cubed_minus_n_squared_l67_67558

-- Define n for the purpose of the problem
def n : ℕ := 9867

-- Prove that the units digit of n^3 - n^2 is 4
theorem units_digit_of_n_cubed_minus_n_squared : ∃ d : ℕ, d = (n^3 - n^2) % 10 ∧ d = 4 := by
  sorry

end units_digit_of_n_cubed_minus_n_squared_l67_67558


namespace find_red_coin_l67_67249

/- Define the function f(n) as the minimum number of scans required to determine the red coin
   - out of n coins with the given conditions.
   - Seyed has 998 white coins, 1 red coin, and 1 red-white coin.
-/

def f (n : Nat) : Nat := sorry

/- The main theorem to be proved: There exists an algorithm that can find the red coin using 
   the scanner at most 17 times for 1000 coins.
-/

theorem find_red_coin (n : Nat) (h : n = 1000) : f n ≤ 17 := sorry

end find_red_coin_l67_67249


namespace find_initial_sum_l67_67321

-- Define the conditions as constants
def A1 : ℝ := 590
def A2 : ℝ := 815
def t1 : ℝ := 2
def t2 : ℝ := 7

-- Define the variables
variable (P r : ℝ)

-- First condition after 2 years
def condition1 : Prop := A1 = P + P * r * t1

-- Second condition after 7 years
def condition2 : Prop := A2 = P + P * r * t2

-- The statement we need to prove: the initial sum of money P is 500
theorem find_initial_sum (h1 : condition1 P r) (h2 : condition2 P r) : P = 500 :=
sorry

end find_initial_sum_l67_67321


namespace halfway_fraction_l67_67914

theorem halfway_fraction : 
  ∃ (x : ℚ), x = 1/2 * ((2/3) + (4/5)) ∧ x = 11/15 :=
by
  sorry

end halfway_fraction_l67_67914


namespace mashas_numbers_l67_67407

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l67_67407


namespace minimum_frosting_time_l67_67776

def ann_time_per_cake := 8 -- Ann's time per cake in minutes
def bob_time_per_cake := 6 -- Bob's time per cake in minutes
def carol_time_per_cake := 10 -- Carol's time per cake in minutes
def passing_time := 1 -- time to pass a cake from one person to another in minutes
def total_cakes := 10 -- total number of cakes to be frosted

theorem minimum_frosting_time : 
  (ann_time_per_cake + passing_time + bob_time_per_cake + passing_time + carol_time_per_cake) + (total_cakes - 1) * carol_time_per_cake = 116 := 
by 
  sorry

end minimum_frosting_time_l67_67776


namespace product_mod_5_l67_67502

-- The sequence 3,13,...,93 can be identified as: 3 + (10 * n) for n = 0,1,2,...,9
def sequence : Nat → Nat := λ n => 3 + 10 * n

-- We state that all elements in the sequence mod 5 is 3
def sequence_mod_5 : ∀ n : Fin 10, sequence n % 5 = 3 :=
  by
  intros n
  fin_cases n
  all_goals simp [sequence]; norm_num

-- We now state the main problem
theorem product_mod_5 : 
    (∏ n in Finset.range 10, sequence n) % 5 = 4 :=
  by
  have h : ∏ n in Finset.range 10, sequence n = 3 ^ 10 :=
    sorry -- The product of the sequence is equal to 3^10
  rw h
  norm_num

#eval (3 ^ 10) % 5 -- this should evaluate to 4

end product_mod_5_l67_67502


namespace part1_part2_l67_67038

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x^2 + 6)

theorem part1 (k : ℝ) :
  (∀ x : ℝ, (f x > k) ↔ (x < -3 ∨ x > -2)) ↔ k = -2/5 :=
by
  sorry

theorem part2 (t : ℝ) :
  (∀ x : ℝ, (x > 0) → (f x ≤ t)) ↔ t ∈ (Set.Ici (Real.sqrt 6 / 6)) :=
by
  sorry

end part1_part2_l67_67038


namespace find_added_number_l67_67591

def S₁₅ := 15 * 17
def S₁₆ := 16 * 20
def added_number := S₁₆ - S₁₅

theorem find_added_number : added_number = 65 :=
by
  sorry

end find_added_number_l67_67591


namespace potato_bag_weight_l67_67922

theorem potato_bag_weight (w : ℕ) (h₁ : w = 36) : w = 36 :=
by
  sorry

end potato_bag_weight_l67_67922


namespace determine_numbers_l67_67408

theorem determine_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11)
  (h4 : S = a + b) (h5 : (∀ (x y : ℕ), x + y = S → x ≠ y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) = false)
  (h6 : a % 2 = 0 ∨ b % 2 = 0) : 
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
  sorry

end determine_numbers_l67_67408


namespace quadratic_equation_with_given_roots_l67_67518

theorem quadratic_equation_with_given_roots :
  (∃ (x : ℝ), (x - 3) * (x + 4) = 0 ↔ x = 3 ∨ x = -4) :=
by
  sorry

end quadratic_equation_with_given_roots_l67_67518


namespace range_of_x_l67_67109

variable (a b x : ℝ)

def conditions : Prop := (a > 0) ∧ (b > 0)

theorem range_of_x (h : conditions a b) : (x^2 + 2*x < 8) -> (-4 < x) ∧ (x < 2) := 
by
  sorry

end range_of_x_l67_67109


namespace probability_of_rolling_number_less_than_5_is_correct_l67_67747

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l67_67747


namespace product_of_binomials_l67_67992

-- Definition of the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x - 3
def binomial2 (x : ℝ) : ℝ := x + 7

-- The theorem to be proved
theorem product_of_binomials (x : ℝ) : 
  binomial1 x * binomial2 x = 4 * x^2 + 25 * x - 21 :=
by
  sorry

end product_of_binomials_l67_67992


namespace tan_sum_identity_l67_67360

theorem tan_sum_identity (α : ℝ) (h : Real.tan α = 1 / 2) : Real.tan (α + π / 4) = 3 := 
by 
  sorry

end tan_sum_identity_l67_67360


namespace product_mod_5_l67_67500

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l67_67500


namespace cheaper_rock_cost_per_ton_l67_67614

theorem cheaper_rock_cost_per_ton (x : ℝ) 
    (h1 : 24 * 1 = 24) 
    (h2 : 800 = 16 * x + 8 * 40) : 
    x = 30 :=
sorry

end cheaper_rock_cost_per_ton_l67_67614


namespace area_to_paint_l67_67442

def wall_height : ℕ := 10
def wall_length : ℕ := 15
def door_height : ℕ := 3
def door_length : ℕ := 5
def window_height : ℕ := 2
def window_length : ℕ := 3

theorem area_to_paint : (wall_height * wall_length) - (door_height * door_length + window_height * window_length) = 129 := by
  sorry

end area_to_paint_l67_67442


namespace divisor_in_first_division_l67_67562

theorem divisor_in_first_division
  (N : ℕ)
  (D : ℕ)
  (Q : ℕ)
  (h1 : N = 8 * D)
  (h2 : N % 5 = 4) :
  D = 3 := 
sorry

end divisor_in_first_division_l67_67562


namespace valid_odd_and_increasing_functions_l67_67448

   def is_odd_function (f : ℝ → ℝ) : Prop :=
     ∀ x, f (-x) = -f (x)

   def is_increasing_function (f : ℝ → ℝ) : Prop :=
     ∀ x y, x < y → f (x) < f (y)

   noncomputable def f1 (x : ℝ) : ℝ := 3 * x^2
   noncomputable def f2 (x : ℝ) : ℝ := 6 * x
   noncomputable def f3 (x : ℝ) : ℝ := x * abs x
   noncomputable def f4 (x : ℝ) : ℝ := x + 1 / x

   theorem valid_odd_and_increasing_functions :
     (is_odd_function f2 ∧ is_increasing_function f2) ∧
     (is_odd_function f3 ∧ is_increasing_function f3) :=
   by
     sorry -- Proof goes here
   
end valid_odd_and_increasing_functions_l67_67448


namespace quadratic_coefficient_c_l67_67223

theorem quadratic_coefficient_c (b c: ℝ) 
  (h_sum: 12 = b) (h_prod: 20 = c) : 
  c = 20 := 
by sorry

end quadratic_coefficient_c_l67_67223


namespace smallest_m_n_l67_67785

noncomputable def g (m n : ℕ) (x : ℝ) : ℝ := Real.arccos (Real.log (↑n * x) / Real.log (↑m))

theorem smallest_m_n (m n : ℕ) (h1 : 1 < m) (h2 : ∀ x : ℝ, -1 ≤ Real.log (↑n * x) / Real.log (↑m) ∧
                      Real.log (↑n * x) / Real.log (↑m) ≤ 1 ∧
                      (forall a b : ℝ,  a ≤ x ∧ x ≤ b -> b - a = 1 / 1007)) :
  m + n = 1026 :=
sorry

end smallest_m_n_l67_67785


namespace pure_imaginary_complex_l67_67516

theorem pure_imaginary_complex (m : ℝ) (i : ℂ) (h : i^2 = -1) :
    (∃ (y : ℂ), (2 - m * i) / (1 + i) = y * i) ↔ m = 2 :=
by
  sorry

end pure_imaginary_complex_l67_67516


namespace total_bottles_in_box_l67_67938

def dozens (n : ℕ) := 12 * n

def water_bottles : ℕ := dozens 2

def apple_bottles : ℕ := water_bottles + 6

def total_bottles : ℕ := water_bottles + apple_bottles

theorem total_bottles_in_box : total_bottles = 54 := 
by
  sorry

end total_bottles_in_box_l67_67938


namespace correct_shelf_probability_l67_67702

theorem correct_shelf_probability:
  ∃ (arrangements : Finset (List ℕ)) (valid : Finset (List ℕ)), 
     arrangements.card = 420 ∧ 
     valid.card = 24 ∧ 
     ((valid.card : ℝ) / (arrangements.card : ℝ) = 2 / 35) := 
by 
  -- Let's define arrangement and valid conditions here 
  sorry

end correct_shelf_probability_l67_67702


namespace value_of_a_l67_67422

theorem value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : (a + b + c) / 3 = 4 * b) (h4 : c / b = 11) : a = 0 :=
by
  sorry

end value_of_a_l67_67422


namespace solution_set_l67_67335

-- Define determinant operation on 2x2 matrices
def determinant (a b c d : ℝ) := a * d - b * c

-- Define the condition inequality
def condition (x : ℝ) : Prop :=
  determinant x 3 (-x) x < determinant 2 0 1 2

-- Prove that the solution to the condition is -4 < x < 1
theorem solution_set : {x : ℝ | condition x} = {x : ℝ | -4 < x ∧ x < 1} :=
by
  sorry

end solution_set_l67_67335


namespace proof_l67_67877

-- Define the propositions
def p : Prop := ∃ x : ℝ, Real.sin x ≥ 1
def q : Prop := ∀ x : ℝ, 0 < x → Real.exp x > Real.log x

-- The theorem statement
theorem proof : p ∧ q := by sorry

end proof_l67_67877


namespace hourly_wage_12_5_l67_67300

theorem hourly_wage_12_5 
  (H : ℝ)
  (work_hours : ℝ := 40)
  (widgets_per_week : ℝ := 1000)
  (widget_earnings_per_widget : ℝ := 0.16)
  (total_earnings : ℝ := 660) :
  (40 * H + 1000 * 0.16 = 660) → (H = 12.5) :=
by
  sorry

end hourly_wage_12_5_l67_67300


namespace multiplication_distributive_example_l67_67305

theorem multiplication_distributive_example : 23 * 4 = 20 * 4 + 3 * 4 := by
  sorry

end multiplication_distributive_example_l67_67305


namespace ganesh_ram_sohan_work_time_l67_67509

theorem ganesh_ram_sohan_work_time (G R S : ℝ)
  (H1 : G + R = 1 / 24)
  (H2 : S = 1 / 48) : (G + R + S = 1 / 16) ∧ (1 / (G + R + S) = 16) :=
by
  sorry

end ganesh_ram_sohan_work_time_l67_67509


namespace joe_first_lift_weight_l67_67454

variable (x y : ℕ)

def joe_lift_conditions (x y : ℕ) : Prop :=
  x + y = 600 ∧ 2 * x = y + 300

theorem joe_first_lift_weight (x y : ℕ) (h : joe_lift_conditions x y) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l67_67454


namespace arithmetic_geom_seq_a5_l67_67203

theorem arithmetic_geom_seq_a5 (a : ℕ → ℝ) (s : ℕ → ℝ) (q : ℝ)
  (a1 : a 1 = 1)
  (S8 : s 8 = 17 * s 4) :
  a 5 = 16 :=
sorry

end arithmetic_geom_seq_a5_l67_67203


namespace three_digit_numbers_with_2_without_4_l67_67532

theorem three_digit_numbers_with_2_without_4 : 
  ∃ n : Nat, n = 200 ∧
  (∀ x : Nat, 100 ≤ x ∧ x ≤ 999 → 
      (∃ d1 d2 d3,
        d1 ≠ 0 ∧ 
        x = d1 * 100 + d2 * 10 + d3 ∧ 
        (d1 ≠ 4 ∧ d2 ≠ 4 ∧ d3 ≠ 4) ∧
        (d1 = 2 ∨ d2 = 2 ∨ d3 = 2))) :=
sorry

end three_digit_numbers_with_2_without_4_l67_67532


namespace total_amount_shared_l67_67136

theorem total_amount_shared (X_share Y_share Z_share total_amount : ℝ) 
                            (h1 : Y_share = 0.45 * X_share) 
                            (h2 : Z_share = 0.50 * X_share) 
                            (h3 : Y_share = 45) : 
                            total_amount = X_share + Y_share + Z_share := 
by 
  -- Sorry to skip the proof
  sorry

end total_amount_shared_l67_67136


namespace leonardo_initial_money_l67_67555

theorem leonardo_initial_money (chocolate_cost : ℝ) (borrowed_amount : ℝ) (needed_amount : ℝ)
  (h_chocolate_cost : chocolate_cost = 5)
  (h_borrowed_amount : borrowed_amount = 0.59)
  (h_needed_amount : needed_amount = 0.41) :
  chocolate_cost + borrowed_amount + needed_amount - (chocolate_cost - borrowed_amount) = 4.41 :=
by
  rw [h_chocolate_cost, h_borrowed_amount, h_needed_amount]
  norm_num
  -- Continue with the proof, eventually obtaining the value 4.41
  sorry

end leonardo_initial_money_l67_67555


namespace bike_covered_distance_l67_67936

theorem bike_covered_distance
  (time : ℕ) 
  (truck_distance : ℕ) 
  (speed_difference : ℕ) 
  (bike_speed truck_speed : ℕ)
  (h_time : time = 8)
  (h_truck_distance : truck_distance = 112)
  (h_speed_difference : speed_difference = 3)
  (h_truck_speed : truck_speed = truck_distance / time)
  (h_speed_relation : truck_speed = bike_speed + speed_difference) :
  bike_speed * time = 88 :=
by
  -- The proof is omitted
  sorry

end bike_covered_distance_l67_67936


namespace sequence_formula_l67_67585

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 5)
  (h3 : ∀ n > 1, a (n + 1) = 2 * a n - a (n - 1)) :
  ∀ n, a n = 4 * n - 3 :=
by
  sorry

end sequence_formula_l67_67585


namespace overtime_hours_l67_67948

theorem overtime_hours (x y : ℕ) 
  (h1 : 60 * x + 90 * y = 3240) 
  (h2 : x + y = 50) : 
  y = 8 :=
by
  sorry

end overtime_hours_l67_67948


namespace number_of_parrots_l67_67247

noncomputable def daily_consumption_parakeet : ℕ := 2
noncomputable def daily_consumption_parrot : ℕ := 14
noncomputable def daily_consumption_finch : ℕ := 1  -- Each finch eats half of what a parakeet eats

noncomputable def num_parakeets : ℕ := 3
noncomputable def num_finches : ℕ := 4
noncomputable def required_birdseed : ℕ := 266
noncomputable def days_in_week : ℕ := 7

theorem number_of_parrots (num_parrots : ℕ) : 
  daily_consumption_parakeet * num_parakeets * days_in_week +
  daily_consumption_finch * num_finches * days_in_week + 
  daily_consumption_parrot * num_parrots * days_in_week = required_birdseed → num_parrots = 2 :=
by 
  -- The proof is omitted as per the instructions
  sorry

end number_of_parrots_l67_67247


namespace sixty_percent_of_total_is_960_l67_67893

variable (number_of_boys number_of_girls total_participants : ℕ)

-- Condition 1: The difference between the number of boys and girls is 400.
def difference_condition : Prop := number_of_girls - number_of_boys = 400

-- Condition 2: There are 600 boys.
def boys_condition : Prop := number_of_boys = 600

-- Condition 3: The number of girls is more than the number of boys.
def girls_condition : Prop := number_of_girls > number_of_boys

-- Given conditions
axiom difference_condition_h : difference_condition number_of_boys number_of_girls
axiom boys_condition_h : boys_condition number_of_boys
axiom girls_condition_h : girls_condition number_of_boys number_of_girls

-- Total number of participants
def total_participants : ℕ := number_of_boys + number_of_girls

theorem sixty_percent_of_total_is_960 :
  0.6 * (number_of_boys + number_of_girls) = 960 :=
by 
  sorry

end sixty_percent_of_total_is_960_l67_67893


namespace fraction_of_cookies_with_nuts_l67_67393

theorem fraction_of_cookies_with_nuts
  (nuts_per_cookie : ℤ)
  (total_cookies : ℤ)
  (total_nuts : ℤ)
  (h1 : nuts_per_cookie = 2)
  (h2 : total_cookies = 60)
  (h3 : total_nuts = 72) :
  (total_nuts / nuts_per_cookie) / total_cookies = 3 / 5 := by
  sorry

end fraction_of_cookies_with_nuts_l67_67393


namespace b_plus_d_over_a_l67_67728

theorem b_plus_d_over_a (a b c d e : ℝ) (h : a ≠ 0) 
  (root1 : a * (5:ℝ)^4 + b * (5:ℝ)^3 + c * (5:ℝ)^2 + d * (5:ℝ) + e = 0)
  (root2 : a * (-3:ℝ)^4 + b * (-3:ℝ)^3 + c * (-3:ℝ)^2 + d * (-3:ℝ) + e = 0)
  (root3 : a * (2:ℝ)^4 + b * (2:ℝ)^3 + c * (2:ℝ)^2 + d * (2:ℝ) + e = 0) :
  (b + d) / a = - (12496 / 3173) :=
sorry

end b_plus_d_over_a_l67_67728


namespace constant_term_in_quadratic_eq_l67_67549

theorem constant_term_in_quadratic_eq : 
  ∀ (x : ℝ), (x^2 - 5 * x = 2) → (∃ a b c : ℝ, a = 1 ∧ a * x^2 + b * x + c = 0 ∧ c = -2) :=
by
  sorry

end constant_term_in_quadratic_eq_l67_67549


namespace set_M_properties_l67_67932

def f (x : ℝ) : ℝ := |x| - |2 * x - 1|

def M : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem set_M_properties :
  M = { x | 0 < x ∧ x < 2 } ∧
  (∀ a, a ∈ M → 
    ((0 < a ∧ a < 1) → (a^2 - a + 1 < 1 / a)) ∧
    (a = 1 → (a^2 - a + 1 = 1 / a)) ∧
    ((1 < a ∧ a < 2) → (a^2 - a + 1 > 1 / a))) := 
by
  sorry

end set_M_properties_l67_67932


namespace distance_between_x_intercepts_l67_67476

theorem distance_between_x_intercepts 
  (s1 s2 : ℝ) (P : ℝ × ℝ)
  (h1 : s1 = 2) 
  (h2 : s2 = -4) 
  (hP : P = (8, 20)) :
  let l1_x_intercept := (0 - (20 - P.2)) / s1 + P.1
  let l2_x_intercept := (0 - (20 - P.2)) / s2 + P.1
  abs (l1_x_intercept - l2_x_intercept) = 15 := 
sorry

end distance_between_x_intercepts_l67_67476


namespace roller_coaster_costs_7_tickets_l67_67301

-- Define the number of tickets for the Ferris wheel, log ride, and the initial and additional tickets Zach needs.
def ferris_wheel_tickets : ℕ := 2
def log_ride_tickets : ℕ := 1
def initial_tickets : ℕ := 1
def additional_tickets : ℕ := 9

-- Define the total number of tickets Zach needs.
def total_tickets : ℕ := initial_tickets + additional_tickets

-- Define the number of tickets needed for the Ferris wheel and log ride together.
def combined_tickets_needed : ℕ := ferris_wheel_tickets + log_ride_tickets

-- Define the number of tickets the roller coaster costs.
def roller_coaster_tickets : ℕ := total_tickets - combined_tickets_needed

-- The theorem stating what we need to prove.
theorem roller_coaster_costs_7_tickets :
  roller_coaster_tickets = 7 :=
by sorry

end roller_coaster_costs_7_tickets_l67_67301


namespace smaller_angle_at_10_15_p_m_l67_67048

-- Definitions of conditions
def clock_hours : ℕ := 12
def degrees_per_hour : ℚ := 360 / clock_hours
def minute_hand_position : ℚ := (15 / 60) * 360
def hour_hand_position : ℚ := 10 * degrees_per_hour + (15 / 60) * degrees_per_hour
def absolute_difference : ℚ := |hour_hand_position - minute_hand_position|
def smaller_angle : ℚ := 360 - absolute_difference

-- Prove that the smaller angle is 142.5°
theorem smaller_angle_at_10_15_p_m : smaller_angle = 142.5 := by
  sorry

end smaller_angle_at_10_15_p_m_l67_67048


namespace expression_evaluation_l67_67489

theorem expression_evaluation : (2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1) := 
by 
  sorry

end expression_evaluation_l67_67489


namespace complex_solution_l67_67835

theorem complex_solution (z : ℂ) (h : z * (0 + 1 * I) = (0 + 1 * I) - 1) : z = 1 + I :=
by
  sorry

end complex_solution_l67_67835


namespace arithmetic_sequence_closed_form_l67_67040

noncomputable def B_n (n : ℕ) : ℝ :=
  2 * (1 - (-2)^n) / 3

theorem arithmetic_sequence_closed_form (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
  (h1 : a_n 1 = 1) (h2 : S_n 3 = 0) :
  B_n n = 2 * (1 - (-2)^n) / 3 := sorry

end arithmetic_sequence_closed_form_l67_67040


namespace value_of_x_l67_67437

theorem value_of_x (x : ℝ) (h : x = -x) : x = 0 := 
by 
  sorry

end value_of_x_l67_67437


namespace cost_of_item_is_200_l67_67946

noncomputable def cost_of_each_item (x : ℕ) : ℕ :=
  let before_discount := 7 * x -- Total cost before discount
  let discount_part := before_discount - 1000 -- Part of the cost over $1000
  let discount := discount_part / 10 -- 10% of the part over $1000
  let after_discount := before_discount - discount -- Total cost after discount
  after_discount

theorem cost_of_item_is_200 :
  (∃ x : ℕ, cost_of_each_item x = 1360) ↔ x = 200 :=
by
  sorry

end cost_of_item_is_200_l67_67946


namespace stickers_at_end_of_week_l67_67701

theorem stickers_at_end_of_week (initial_stickers earned_stickers total_stickers : Nat) :
  initial_stickers = 39 →
  earned_stickers = 22 →
  total_stickers = initial_stickers + earned_stickers →
  total_stickers = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end stickers_at_end_of_week_l67_67701


namespace minimize_triangle_area_minimize_product_PA_PB_l67_67474

-- Define the initial conditions and geometry setup
def point (x y : ℝ) := (x, y)
def line_eq (a b : ℝ) := ∀ x y : ℝ, x / a + y / b = 1

-- Point P
def P := point 2 1

-- Condition: the line passes through point P and intersects the axes
def line_through_P (a b : ℝ) := line_eq a b ∧ (2 / a + 1 / b = 1) ∧ a > 2 ∧ b > 1

-- Prove that the line minimizing the area of triangle AOB is x + 2y - 4 = 0
theorem minimize_triangle_area (a b : ℝ) (h : line_through_P a b) :
  a = 4 ∧ b = 2 → line_eq 4 2 := 
sorry

-- Prove that the line minimizing the product |PA||PB| is x + y - 3 = 0
theorem minimize_product_PA_PB (a b : ℝ) (h : line_through_P a b) :
  a = 3 ∧ b = 3 → line_eq 3 3 := 
sorry

end minimize_triangle_area_minimize_product_PA_PB_l67_67474


namespace remainder_98_pow_50_mod_100_l67_67920

/-- 
Theorem: The remainder when \(98^{50}\) is divided by 100 is 24.
-/
theorem remainder_98_pow_50_mod_100 : (98^50 % 100) = 24 := by
  sorry

end remainder_98_pow_50_mod_100_l67_67920


namespace relay_race_total_time_l67_67184

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 10
  let t3 := t2 - 15
  let t4 := t1 - 25
  t1 + t2 + t3 + t4 = 200 := by
    sorry

end relay_race_total_time_l67_67184


namespace expected_sectors_pizza_l67_67164

/-- Let N be the total number of pizza slices and M be the number of slices taken randomly.
    Given N = 16 and M = 5, the expected number of sectors formed is 11/3. -/
theorem expected_sectors_pizza (N M : ℕ) (hN : N = 16) (hM : M = 5) :
  (N - M) * M / (N - 1) = 11 / 3 :=
  sorry

end expected_sectors_pizza_l67_67164


namespace inequality_with_means_l67_67876

theorem inequality_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_with_means_l67_67876


namespace shaded_square_percentage_l67_67600

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 16) (h2 : shaded_squares = 8) : 
  (shaded_squares : ℚ) / total_squares * 100 = 50 :=
by
  sorry

end shaded_square_percentage_l67_67600


namespace johns_share_l67_67138

theorem johns_share
  (total_amount : ℕ)
  (ratio_john : ℕ)
  (ratio_jose : ℕ)
  (ratio_binoy : ℕ)
  (total_parts : ℕ)
  (value_per_part : ℕ)
  (johns_parts : ℕ)
  (johns_share : ℕ)
  (h1 : total_amount = 4800)
  (h2 : ratio_john = 2)
  (h3 : ratio_jose = 4)
  (h4 : ratio_binoy = 6)
  (h5 : total_parts = ratio_john + ratio_jose + ratio_binoy)
  (h6 : value_per_part = total_amount / total_parts)
  (h7 : johns_parts = ratio_john)
  (h8 : johns_share = value_per_part * johns_parts) :
  johns_share = 800 := by
  sorry

end johns_share_l67_67138


namespace evaluate_polynomial_at_minus_two_l67_67490

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

theorem evaluate_polynomial_at_minus_two :
  P (-2) = -18 :=
by
  sorry

end evaluate_polynomial_at_minus_two_l67_67490


namespace num_solutions_eq_40_l67_67331

theorem num_solutions_eq_40 : 
  ∀ (n : ℕ), 
  (∃ seq : ℕ → ℕ, seq 1 = 4 ∧ (∀ k : ℕ, 1 ≤ k → seq (k + 1) = seq k + 4) ∧ seq 10 = 40) :=
by
  sorry

end num_solutions_eq_40_l67_67331


namespace solve_inequality_l67_67435

theorem solve_inequality (x : ℝ) : -1/3 * x + 1 ≤ -5 → x ≥ 18 := 
  sorry

end solve_inequality_l67_67435


namespace time_upstream_is_correct_l67_67471

-- Define the conditions
def speed_of_stream : ℝ := 3
def speed_in_still_water : ℝ := 15
def downstream_time : ℝ := 1
def downstream_speed : ℝ := speed_in_still_water + speed_of_stream
def distance_downstream : ℝ := downstream_speed * downstream_time
def upstream_speed : ℝ := speed_in_still_water - speed_of_stream

-- Theorem statement
theorem time_upstream_is_correct :
  (distance_downstream / upstream_speed) = 1.5 := by
  sorry

end time_upstream_is_correct_l67_67471


namespace min_liars_in_presidium_l67_67162

-- Define the conditions of the problem
def liars_and_truthlovers (grid : ℕ → ℕ → Prop) : Prop :=
  ∃ n : ℕ, n = 32 ∧ 
  (∀ i j, i < 4 ∧ j < 8 → 
    (∃ ni nj, (ni = i + 1 ∨ ni = i - 1 ∨ ni = i ∨ nj = j + 1 ∨ nj = j - 1 ∨ nj = j) ∧
      (ni < 4 ∧ nj < 8) → (grid i j ↔ ¬ grid ni nj)))

-- Define proof problem
theorem min_liars_in_presidium (grid : ℕ → ℕ → Prop) :
  liars_and_truthlovers grid → (∃ l, l = 8) := by
  sorry

end min_liars_in_presidium_l67_67162


namespace sets_relationship_l67_67205

def set_M : Set ℝ := {x | x^2 - 2 * x > 0}
def set_N : Set ℝ := {x | x > 3}

theorem sets_relationship : set_M ∩ set_N = set_N := by
  sorry

end sets_relationship_l67_67205


namespace solve_a_l67_67809

-- Defining sets A and B
def set_A (a : ℤ) : Set ℤ := {a^2, a + 1, -3}
def set_B (a : ℤ) : Set ℤ := {a - 3, 2 * a - 1, a^2 + 1}

-- Defining the condition of intersection
def intersection_condition (a : ℤ) : Prop :=
  (set_A a) ∩ (set_B a) = {-3}

-- Stating the theorem
theorem solve_a (a : ℤ) (h : intersection_condition a) : a = -1 :=
sorry

end solve_a_l67_67809


namespace expression_nonnegative_l67_67023

theorem expression_nonnegative (x : ℝ) : 
  0 ≤ x → x < 3 → 0 ≤ (x - 12 * x^2 + 36 * x^3) / (9 - x^3) :=
  sorry

end expression_nonnegative_l67_67023


namespace number_of_short_trees_to_plant_l67_67720

-- Definitions of the conditions
def current_short_trees : ℕ := 41
def current_tall_trees : ℕ := 44
def total_short_trees_after_planting : ℕ := 98

-- The statement to be proved
theorem number_of_short_trees_to_plant :
  total_short_trees_after_planting - current_short_trees = 57 :=
by
  -- Proof goes here
  sorry

end number_of_short_trees_to_plant_l67_67720


namespace taqeesha_grade_correct_l67_67849

-- Definitions for conditions
def total_score_of_24_students := 24 * 82
def total_score_of_25_students (T: ℕ) := 25 * 84
def taqeesha_grade := 132

-- Theorem statement forming the proof problem
theorem taqeesha_grade_correct
    (h1: total_score_of_24_students + taqeesha_grade = total_score_of_25_students taqeesha_grade): 
    taqeesha_grade = 132 :=
by
  sorry

end taqeesha_grade_correct_l67_67849


namespace problem1_problem2_l67_67612

-- Using the conditions from a) and the correct answers from b):
-- 1. Given an angle α with a point P(-4,3) on its terminal side

theorem problem1 (α : ℝ) (x y r : ℝ) (h₁ : x = -4) (h₂ : y = 3) (h₃ : r = 5) 
  (hx : r = Real.sqrt (x^2 + y^2)) 
  (hsin : Real.sin α = y / r) 
  (hcos : Real.cos α = x / r) 
  : (Real.cos (π / 2 + α) * Real.sin (-π - α)) / (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3 / 4 :=
by sorry

-- 2. Let k be an integer
theorem problem2 (α : ℝ) (k : ℤ)
  : (Real.sin (k * π - α) * Real.cos ((k + 1) * π - α)) / (Real.sin ((k - 1) * π + α) * Real.cos (k * π + α)) = -1 :=
by sorry

end problem1_problem2_l67_67612


namespace repeating_sixths_denominator_l67_67424

theorem repeating_sixths_denominator :
  let S := (0.succ_div 1 succ : ℚ) in -- Define S to handle repeating decimals
  S.denom = 3 := 
begin
  let S : ℚ := 2 / 3,
  sorry
end

end repeating_sixths_denominator_l67_67424


namespace product_of_integers_l67_67352

theorem product_of_integers (X Y Z W : ℚ) (h_sum : X + Y + Z + W = 100)
  (h_relation : X + 5 = Y - 5 ∧ Y - 5 = 3 * Z ∧ 3 * Z = W / 3) :
  X * Y * Z * W = 29390625 / 256 := by
  sorry

end product_of_integers_l67_67352


namespace max_sum_squares_of_sides_l67_67172

theorem max_sum_squares_of_sides
  (a : ℝ) (α : ℝ) 
  (hα1 : 0 < α) (hα2 : α < Real.pi / 2) : 
  ∃ b c : ℝ, b^2 + c^2 = a^2 / (1 - Real.cos α) := 
sorry

end max_sum_squares_of_sides_l67_67172


namespace polynomial_roots_sum_l67_67726

theorem polynomial_roots_sum (a b c d e : ℝ) (h₀ : a ≠ 0)
  (h1 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
  (h2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h3 : a * 2^4 + b * 2^3 + c * 2^2 + d * 2 + e = 0) :
  (b + d) / a = -2677 := 
sorry

end polynomial_roots_sum_l67_67726


namespace Sergey_full_years_l67_67417

def full_years (years months weeks days hours : ℕ) : ℕ :=
  years + months / 12 + (weeks * 7 + days) / 365

theorem Sergey_full_years 
  (years : ℕ)
  (months : ℕ)
  (weeks : ℕ)
  (days : ℕ)
  (hours : ℕ) :
  years = 36 →
  months = 36 →
  weeks = 36 →
  days = 36 →
  hours = 36 →
  full_years years months weeks days hours = 39 :=
by
  intros
  sorry

end Sergey_full_years_l67_67417


namespace watermelons_with_seeds_l67_67718

def ripe_watermelons : ℕ := 11
def unripe_watermelons : ℕ := 13
def seedless_watermelons : ℕ := 15
def total_watermelons := ripe_watermelons + unripe_watermelons

theorem watermelons_with_seeds :
  total_watermelons - seedless_watermelons = 9 :=
by
  sorry

end watermelons_with_seeds_l67_67718


namespace henry_classical_cds_l67_67845

variable (R C : ℕ)

theorem henry_classical_cds :
  (23 - 3 = R) →
  (R = 2 * C) →
  C = 10 :=
by
  intros h1 h2
  sorry

end henry_classical_cds_l67_67845


namespace problem1_problem2_l67_67675

-- Define sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x : ℝ | x < -2 ∨ x > 6}

-- Define the two proof problems as Lean statements
theorem problem1 (a : ℝ) : setA a ∩ setB = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by
  sorry

theorem problem2 (a : ℝ) : setA a ⊆ setB ↔ (a < -5 ∨ a > 6) := by
  sorry

end problem1_problem2_l67_67675


namespace lower_amount_rent_l67_67411

theorem lower_amount_rent (L : ℚ) (total_rent : ℚ) (reduction : ℚ)
  (h1 : total_rent = 2000)
  (h2 : reduction = 200)
  (h3 : 10 * (60 - L) = reduction) :
  L = 40 := by
  sorry

end lower_amount_rent_l67_67411


namespace probability_of_rolling_number_less_than_5_is_correct_l67_67746

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l67_67746


namespace find_a_b_l67_67537

theorem find_a_b (a b : ℝ) (h : (a - 2) ^ 2 + |b + 4| = 0) : a + b = -2 :=
sorry

end find_a_b_l67_67537


namespace christmas_gift_distribution_l67_67175

theorem christmas_gift_distribution :
  ∃ n : ℕ, n = 30 ∧ 
  ∃ (gifts : Finset α) (students : Finset β) 
    (distribute : α → β) (a b c d : α),
    a ∈ gifts ∧ b ∈ gifts ∧ c ∈ gifts ∧ d ∈ gifts ∧ gifts.card = 4 ∧
    students.card = 3 ∧ 
    (∀ s ∈ students, ∃ g ∈ gifts, distribute g = s) ∧ 
    distribute a ≠ distribute b :=
sorry

end christmas_gift_distribution_l67_67175


namespace coordinate_system_and_parametric_equations_l67_67660

/-- Given the parametric equation of curve C1 is 
  x = 2 * cos φ and y = 3 * sin φ (where φ is the parameter)
  and a coordinate system with the origin as the pole and the positive half-axis of x as the polar axis.
  The polar equation of curve C2 is ρ = 2.
  The vertices of square ABCD are all on C2, arranged counterclockwise,
  with the polar coordinates of point A being (2, π/3).
  Find the Cartesian coordinates of A, B, C, and D, and prove that
  for any point P on C1, |PA|^2 + |PB|^2 + |PC|^2 + |PD|^2 is within the range [32, 52]. -/
theorem coordinate_system_and_parametric_equations
  (φ : ℝ)
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)
  (P : ℝ → ℝ × ℝ)
  (A B C D : ℝ × ℝ)
  (t : ℝ)
  (H1 : ∀ φ, P φ = (2 * Real.cos φ, 3 * Real.sin φ))
  (H2 : A = (1, Real.sqrt 3) ∧ B = (-Real.sqrt 3, 1) ∧ C = (-1, -Real.sqrt 3) ∧ D = (Real.sqrt 3, -1))
  (H3 : ∀ p : ℝ × ℝ, ∃ φ, p = P φ)
  : ∀ x y, ∃ (φ : ℝ), P φ = (x, y) →
    ∃ t, t = |P φ - A|^2 + |P φ - B|^2 + |P φ - C|^2 + |P φ - D|^2 ∧ 32 ≤ t ∧ t ≤ 52 := 
sorry

end coordinate_system_and_parametric_equations_l67_67660


namespace flag_arrangement_remainder_l67_67722

theorem flag_arrangement_remainder :
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  M % div = 441 := 
by
  -- Definitions
  let num_blue := 12
  let num_green := 10
  let div := 1000
  let M := 13 * Nat.choose (num_blue + 2) num_green - 2 * Nat.choose (num_blue + 1) num_green
  -- Proof
  sorry

end flag_arrangement_remainder_l67_67722


namespace total_kids_got_in_equals_148_l67_67792

def total_kids : ℕ := 120 + 90 + 50

def denied_riverside : ℕ := (20 * 120) / 100
def denied_west_side : ℕ := (70 * 90) / 100
def denied_mountaintop : ℕ := 50 / 2

def got_in_riverside : ℕ := 120 - denied_riverside
def got_in_west_side : ℕ := 90 - denied_west_side
def got_in_mountaintop : ℕ := 50 - denied_mountaintop

def total_got_in : ℕ := got_in_riverside + got_in_west_side + got_in_mountaintop

theorem total_kids_got_in_equals_148 :
  total_got_in = 148 := 
by
  unfold total_got_in
  unfold got_in_riverside got_in_west_side got_in_mountaintop
  unfold denied_riverside denied_west_side denied_mountaintop
  sorry

end total_kids_got_in_equals_148_l67_67792


namespace negation_of_proposition_l67_67257

theorem negation_of_proposition :
  (¬ (∃ x : ℝ, x < 0 ∧ x^2 > 0)) ↔ (∀ x : ℝ, x < 0 → x^2 ≤ 0) :=
  sorry

end negation_of_proposition_l67_67257


namespace probability_of_rolling_less_than_5_l67_67744

theorem probability_of_rolling_less_than_5 :
  (let outcomes := 8 in
   let successes := 4 in
   let probability := (successes: ℚ) / outcomes in
   probability = 1 / 2) :=
by
  sorry

end probability_of_rolling_less_than_5_l67_67744


namespace area_of_original_square_l67_67186

theorem area_of_original_square 
  (x : ℝ) 
  (h0 : x * (x - 3) = 40) 
  (h1 : 0 < x) : 
  x ^ 2 = 64 := 
sorry

end area_of_original_square_l67_67186


namespace value_of_x_l67_67057

theorem value_of_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end value_of_x_l67_67057


namespace sum_of_cubes_divisible_by_middle_integer_l67_67096

theorem sum_of_cubes_divisible_by_middle_integer (a : ℤ) : 
  (a - 1)^3 + a^3 + (a + 1)^3 ∣ 3 * a :=
sorry

end sum_of_cubes_divisible_by_middle_integer_l67_67096


namespace min_area_of_triangle_l67_67930

noncomputable def area_of_triangle (p q : ℤ) : ℚ :=
  (1 / 2 : ℚ) * abs (3 * p - 5 * q)

theorem min_area_of_triangle :
  (∀ p q : ℤ, p ≠ 0 ∨ q ≠ 0 → area_of_triangle p q ≥ (1 / 2 : ℚ)) ∧
  (∃ p q : ℤ, p ≠ 0 ∨ q ≠ 0 ∧ area_of_triangle p q = (1 / 2 : ℚ)) := 
by { 
  sorry 
}

end min_area_of_triangle_l67_67930


namespace simplify_f_value_f_second_quadrant_l67_67802

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (3 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (3 * Real.pi / 2 - α)) / 
  (Real.cos (Real.pi - α) * Real.sin (-Real.pi - α))

theorem simplify_f (α : ℝ) : 
  f α = Real.cos α := 
sorry

theorem value_f_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (hcosα : Real.cos (π / 2 + α) = -1 / 3) :
  f α = - (2 * Real.sqrt 2) / 3 := 
sorry

end simplify_f_value_f_second_quadrant_l67_67802


namespace arithmetic_sequence_a_value_l67_67369

theorem arithmetic_sequence_a_value :
  ∀ (a : ℤ), (-7) - a = a - 1 → a = -3 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_a_value_l67_67369


namespace radius_of_circle_l67_67115

theorem radius_of_circle (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ a) : 
  ∃ R, R = (b - a) / 2 ∨ R = (b + a) / 2 :=
by {
  sorry
}

end radius_of_circle_l67_67115


namespace two_pow_ge_two_mul_l67_67611

theorem two_pow_ge_two_mul (n : ℕ) : 2^n ≥ 2 * n :=
sorry

end two_pow_ge_two_mul_l67_67611


namespace saddle_value_l67_67699

theorem saddle_value (S : ℝ) (H : ℝ) (h1 : S + H = 100) (h2 : H = 7 * S) : S = 12.50 :=
by
  sorry

end saddle_value_l67_67699


namespace grasshopper_frog_jump_difference_l67_67579

theorem grasshopper_frog_jump_difference :
  let grasshopper_jump := 19
  let frog_jump := 15
  grasshopper_jump - frog_jump = 4 :=
by
  let grasshopper_jump := 19
  let frog_jump := 15
  sorry

end grasshopper_frog_jump_difference_l67_67579


namespace negation_of_existence_l67_67715

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ ∀ x : ℝ, x^2 + 2 * a * x + a > 0 :=
by
  sorry

end negation_of_existence_l67_67715


namespace expectation_absolute_deviation_l67_67285

noncomputable def absolute_deviation (n : ℕ) (m : ℕ) : ℝ :=
    |(m / n : ℝ) - 0.5|

theorem expectation_absolute_deviation (n m : ℕ) 
    (h₁ : n = 10) (h₂ : n = 100) (pn : ℕ → Real) :
    let E_α := ∫ x, absolute_deviation 10 (pn x): ℝ in
    let E_β := ∫ x, absolute_deviation 100 (pn x): ℝ in
    E_α > E_β :=
begin
  sorry
end

end expectation_absolute_deviation_l67_67285


namespace length_of_platform_l67_67947

theorem length_of_platform (v t_m t_p L_t L_p : ℝ)
    (h1 : v = 33.3333333)
    (h2 : t_m = 22)
    (h3 : t_p = 45)
    (h4 : L_t = v * t_m)
    (h5 : L_t + L_p = v * t_p) :
    L_p = 766.666666 :=
by
  sorry

end length_of_platform_l67_67947


namespace axis_of_symmetry_l67_67346

theorem axis_of_symmetry (a : ℝ) (h : a ≠ 0) : y = - 1 / (4 * a) :=
sorry

end axis_of_symmetry_l67_67346


namespace amount_with_r_l67_67607

theorem amount_with_r (p q r : ℕ) (h1 : p + q + r = 7000) (h2 : r = (2 * (p + q)) / 3) : r = 2800 :=
sorry

end amount_with_r_l67_67607


namespace intersection_of_sets_A_B_l67_67526

def set_A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 > 0 }
def set_B : Set ℝ := { x : ℝ | -2 < x ∧ x ≤ 2 }
def set_intersection : Set ℝ := { x : ℝ | -2 < x ∧ x < -1 }

theorem intersection_of_sets_A_B :
  (set_A ∩ set_B) = set_intersection :=
  sorry

end intersection_of_sets_A_B_l67_67526


namespace joe_selects_all_CHASING_l67_67231

noncomputable def probability_selecting_all_CHASING : ℚ := 
  let p_camp := 1 / (nat.choose 4 2) in
  let p_herbs := 1 / (nat.choose 5 3) in
  let p_glow := 1 / (nat.choose 4 2) in
  p_camp * p_herbs * p_glow

theorem joe_selects_all_CHASING : probability_selecting_all_CHASING = 1 / 360 :=
by
  sorry

end joe_selects_all_CHASING_l67_67231


namespace brendan_remaining_money_l67_67648

-- Definitions based on conditions
def earned_amount : ℕ := 5000
def recharge_rate : ℕ := 1/2
def car_cost : ℕ := 1500

-- Proof Statement
theorem brendan_remaining_money : 
  (earned_amount * recharge_rate) - car_cost = 1000 :=
sorry

end brendan_remaining_money_l67_67648


namespace triangle_area_l67_67268

theorem triangle_area (a b c : ℝ) (h1 : a = 14) (h2 : b = 48) (h3 : c = 50) (h4 : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 336 := 
by 
  rw [h1, h2]
  sorry

end triangle_area_l67_67268


namespace value_of_expression_l67_67678

-- Given conditions
variable (n : ℤ)
def m : ℤ := 4 * n + 3

-- Main theorem statement
theorem value_of_expression (n : ℤ) : 
  (m n)^2 - 8 * (m n) * n + 16 * n^2 = 9 := 
  sorry

end value_of_expression_l67_67678


namespace max_product_l67_67041

theorem max_product (a b : ℕ) (h1: a + b = 100) 
    (h2: a % 3 = 2) (h3: b % 7 = 5) : a * b ≤ 2491 := by
  sorry

end max_product_l67_67041


namespace parabola_standard_equation_l67_67351

theorem parabola_standard_equation :
  ∃ p1 p2 : ℝ, p1 > 0 ∧ p2 > 0 ∧ (y^2 = 2 * p1 * x ∨ x^2 = 2 * p2 * y) ∧ ((6, 4) ∈ {(x, y) | y^2 = 2 * p1 * x} ∨ (6, 4) ∈ {(x, y) | x^2 = 2 * p2 * y}) := 
  sorry

end parabola_standard_equation_l67_67351


namespace typing_time_l67_67958

def original_speed : ℕ := 212
def reduction : ℕ := 40
def new_speed : ℕ := original_speed - reduction
def document_length : ℕ := 3440
def required_time : ℕ := 20

theorem typing_time :
  document_length / new_speed = required_time :=
by
  sorry

end typing_time_l67_67958


namespace find_value_l67_67060

theorem find_value (x y : ℚ) (hx : x = 5 / 7) (hy : y = 7 / 5) :
  (1 / 3 * x^8 * y^9 + 1 / 7) = 64 / 105 := by
  sorry

end find_value_l67_67060


namespace triangle_interior_angle_leq_60_l67_67281

theorem triangle_interior_angle_leq_60 (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (angle_sum : A + B + C = 180)
  (all_gt_60 : A > 60 ∧ B > 60 ∧ C > 60) :
  false :=
by
  sorry

end triangle_interior_angle_leq_60_l67_67281


namespace arthur_additional_muffins_l67_67160

/-- Define the number of muffins Arthur has already baked -/
def muffins_baked : ℕ := 80

/-- Define the multiplier for the total output Arthur wants -/
def desired_multiplier : ℝ := 2.5

/-- Define the equation representing the total desired muffins -/
def total_muffins : ℝ := muffins_baked * desired_multiplier

/-- Define the number of additional muffins Arthur needs to bake -/
def additional_muffins : ℝ := total_muffins - muffins_baked

theorem arthur_additional_muffins : additional_muffins = 120 := by
  sorry

end arthur_additional_muffins_l67_67160


namespace difference_between_median_and_mean_is_five_l67_67224

noncomputable def mean_score : ℝ :=
  0.20 * 60 + 0.20 * 75 + 0.40 * 85 + 0.20 * 95

noncomputable def median_score : ℝ := 85

theorem difference_between_median_and_mean_is_five :
  abs (median_score - mean_score) = 5 :=
by
  unfold mean_score median_score
  -- median_score - mean_score = 85 - 80
  -- thus the absolute value of the difference is 5
  sorry

end difference_between_median_and_mean_is_five_l67_67224


namespace original_price_of_cycle_l67_67764

theorem original_price_of_cycle (SP : ℝ) (gain_percent : ℝ) (original_price : ℝ) 
  (hSP : SP = 1260) (hgain : gain_percent = 0.40) (h_eq : SP = original_price * (1 + gain_percent)) :
  original_price = 900 :=
by
  sorry

end original_price_of_cycle_l67_67764


namespace problem1_problem2_l67_67525

noncomputable def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
noncomputable def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)
noncomputable def condition1 (a : ℝ) : Prop := 
  a = -1 ∧ (∃ x, p x a ∨ q x)

noncomputable def condition2 (a : ℝ) : Prop :=
  ∀ x, ¬ p x a → ¬ q x

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : condition1 a) : -6 ≤ x ∧ x < -3 ∨ 1 < x ∧ x ≤ 12 := 
sorry

theorem problem2 (a : ℝ) (h₂ : condition2 a) : -4 ≤ a ∧ a ≤ -2 :=
sorry

end problem1_problem2_l67_67525


namespace area_ratio_trapezoid_l67_67074

/--
In trapezoid PQRS, the lengths of the bases PQ and RS are 10 and 21 respectively.
The legs of the trapezoid are extended beyond P and Q to meet at point T.
Prove that the ratio of the area of triangle TPQ to the area of trapezoid PQRS is 100/341.
-/
theorem area_ratio_trapezoid (PQ RS TPQ PQRS : ℝ) (hPQ : PQ = 10) (hRS : RS = 21) :
  let area_TPQ := TPQ
  let area_PQRS := PQRS
  area_TPQ / area_PQRS = 100 / 341 :=
by
  sorry

end area_ratio_trapezoid_l67_67074


namespace compute_value_condition_l67_67828

theorem compute_value_condition (x : ℝ) (h : x + (1 / x) = 3) :
  (x - 2) ^ 2 + 25 / (x - 2) ^ 2 = -x + 5 := by
  sorry

end compute_value_condition_l67_67828


namespace length_to_width_ratio_l67_67426

-- Define the conditions
def width := 6
def area := 108

-- The length of the rectangle, derived from the conditions
def length := area / width

-- The main statement: The ratio of the length to the width is 3:1
theorem length_to_width_ratio : length / width = 3 :=
by
  unfold length width area
  -- the following is auto calculated based on the values from the condition
  simp
  sorry


end length_to_width_ratio_l67_67426


namespace v_2015_eq_2_l67_67654

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 4
  | 4 => 1
  | 5 => 2
  | _ => 0  -- assuming g(x) = 0 for other values, though not used here

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n)

theorem v_2015_eq_2 : v 2015 = 2 :=
by
  sorry

end v_2015_eq_2_l67_67654


namespace general_formula_correct_S_k_equals_189_l67_67087

-- Define the arithmetic sequence with initial conditions
def a (n : ℕ) : ℤ :=
  if n = 1 then -11
  else sorry  -- Will be defined by the general formula

-- Given conditions in Lean
def initial_condition (a : ℕ → ℤ) :=
  a 1 = -11 ∧ a 4 + a 6 = -6

-- General formula for the arithmetic sequence to be proven
def general_formula (n : ℕ) : ℤ := 2 * n - 13

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℤ :=
  n^2 - 12 * n

-- Problem 1: Prove the general formula
theorem general_formula_correct : ∀ n : ℕ, initial_condition a → a n = general_formula n :=
by sorry

-- Problem 2: Prove that k = 21 such that S_k = 189
theorem S_k_equals_189 : ∃ k : ℕ, S k = 189 ∧ k = 21 :=
by sorry

end general_formula_correct_S_k_equals_189_l67_67087


namespace right_triangle_cotangent_l67_67381

theorem right_triangle_cotangent
  (A B C : Point)
  (h : ∠ A B C = 90°)
  (AC BC : ℝ)
  (hAC : AC = 3)
  (hBC : BC = 4) :
  Real.cot (∠ A B C) = 3 / 4 := by
  sorry

end right_triangle_cotangent_l67_67381


namespace min_value_of_a_l67_67357

theorem min_value_of_a (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : a + b + c + d = 2004) (h5 : a^2 - b^2 + c^2 - d^2 = 2004) : a = 503 :=
sorry

end min_value_of_a_l67_67357


namespace students_only_biology_students_biology_or_chemistry_but_not_both_l67_67793

def students_enrolled_in_both : ℕ := 15
def total_biology_students : ℕ := 35
def students_only_chemistry : ℕ := 18

theorem students_only_biology (h₀ : students_enrolled_in_both ≤ total_biology_students) :
  total_biology_students - students_enrolled_in_both = 20 := by
  sorry

theorem students_biology_or_chemistry_but_not_both :
  total_biology_students - students_enrolled_in_both + students_only_chemistry = 38 := by
  sorry

end students_only_biology_students_biology_or_chemistry_but_not_both_l67_67793


namespace correct_completion_at_crossroads_l67_67642

theorem correct_completion_at_crossroads :
  (∀ (s : String), 
    s = "An accident happened at a crossroads a few meters away from a bank" → 
    (∃ (general_sense : Bool), general_sense = tt)) :=
by
  sorry

end correct_completion_at_crossroads_l67_67642


namespace extracurricular_books_counts_l67_67901

theorem extracurricular_books_counts 
  (a b c d : ℕ)
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by
  sorry

end extracurricular_books_counts_l67_67901


namespace maximum_f_l67_67620

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def f (p : ℝ) : ℝ :=
  binomial_coefficient 20 2 * p^2 * (1 - p)^18

theorem maximum_f :
  ∃ p_0 : ℝ, 0 < p_0 ∧ p_0 < 1 ∧ f p = f (0.1) := sorry

end maximum_f_l67_67620


namespace solve_system_of_equations_l67_67730

theorem solve_system_of_equations :
  ∀ x y : ℝ,
  (y^2 + 2*x*y + x^2 - 6*y - 6*x + 5 = 0) ∧ (y - x + 1 = x^2 - 3*x) ∧ (x ≠ 0) ∧ (x ≠ 3) →
  (x, y) = (-1, 2) ∨ (x, y) = (2, -1) ∨ (x, y) = (-2, 7) :=
by
  sorry

end solve_system_of_equations_l67_67730


namespace masha_numbers_l67_67395

theorem masha_numbers {a b : ℕ} (h1 : a ≠ b) (h2 : 11 < a) (h3 : 11 < b) 
  (h4 : ∃ S, S = a + b ∧ (∀ x y, x + y = S → x ≠ y → 11 < x ∧ 11 < y → 
       (¬(x = a ∧ y = b) ∧ ¬(x = b ∧ y = a)))) 
  (h5 : even a ∨ even b)
  (h6 : ∀ x y, (even x ∨ even y) → x ≠ y → 11 < x ∧ 11 < y ∧ x + y = a + b → 
       x = a ∧ y = b ∨ x = b ∧ y = a) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by
  sorry

end masha_numbers_l67_67395


namespace find_sum_of_a_and_c_l67_67193

variable (a b c d : ℝ)

theorem find_sum_of_a_and_c (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) :
  a + c = 8 := by sorry

end find_sum_of_a_and_c_l67_67193


namespace distinct_gcd_numbers_l67_67590

theorem distinct_gcd_numbers (nums : Fin 100 → ℕ) (h_distinct : Function.Injective nums) :
  ¬ ∃ a b c : Fin 100, 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
    (nums a + Nat.gcd (nums b) (nums c) = nums b + Nat.gcd (nums a) (nums c)) ∧ 
    (nums b + Nat.gcd (nums a) (nums c) = nums c + Nat.gcd (nums a) (nums b)) := 
sorry

end distinct_gcd_numbers_l67_67590


namespace tetrahedron_volume_l67_67709

theorem tetrahedron_volume (R S1 S2 S3 S4 : ℝ) : 
    V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l67_67709


namespace base_of_1987_with_digit_sum_25_l67_67796

theorem base_of_1987_with_digit_sum_25 (b a c : ℕ) (h₀ : a * b^2 + b * b + c = 1987)
(h₁ : a + b + c = 25) (h₂ : 1 ≤ b ∧ b ≤ 45) : b = 19 :=
sorry

end base_of_1987_with_digit_sum_25_l67_67796


namespace probability_mixed_l67_67005

noncomputable def male_students : ℕ := 220
noncomputable def female_students : ℕ := 380
noncomputable def total_students : ℕ := male_students + female_students

noncomputable def selected_students : ℕ := 10
noncomputable def selected_male_students : ℕ := 4
noncomputable def selected_female_students : ℕ := 6

noncomputable def students_for_discussion : ℕ := 3

noncomputable def total_events := Nat.choose selected_students students_for_discussion
noncomputable def male_only_events := Nat.choose selected_male_students students_for_discussion
noncomputable def female_only_events := Nat.choose selected_female_students students_for_discussion

noncomputable def mixed_events := total_events - male_only_events - female_only_events

theorem probability_mixed :
  (mixed_events : ℚ) / (total_events : ℚ) = 4 / 5 :=
sorry

end probability_mixed_l67_67005


namespace parabola_ellipse_shared_focus_l67_67045

theorem parabola_ellipse_shared_focus (m : ℝ) :
  (∃ (x : ℝ), x^2 = 2 * (1/2)) ∧ (∃ (y : ℝ), y = (1/2)) →
  ∀ (a b : ℝ), (a = 2) ∧ (b = m) →
  m = 9/4 := 
by sorry

end parabola_ellipse_shared_focus_l67_67045


namespace smallest_N_for_circular_table_l67_67940

/--
  Given a circular table with 60 chairs, prove that the smallest number of people, N,
  such that any additional person must sit next to someone already seated is 20.
-/
theorem smallest_N_for_circular_table (N : ℕ) (h : N = 20) : 
  ∀ (next_seated : ℕ), next_seated ≤ N → (∃ i : ℕ, i < N ∧ next_seated = i + 1 ∨ next_seated = i - 1) :=
by
  sorry

end smallest_N_for_circular_table_l67_67940


namespace birds_in_tree_l67_67308

theorem birds_in_tree (initial_birds : ℝ) (birds_flew_away : ℝ) (h : initial_birds = 21.0) (h_flew : birds_flew_away = 14.0) : 
initial_birds - birds_flew_away = 7.0 :=
by
  -- proof goes here
  sorry

end birds_in_tree_l67_67308


namespace log_sum_eval_l67_67344

theorem log_sum_eval :
  (Real.logb 5 625 + Real.logb 5 5 - Real.logb 5 (1 / 25)) = 7 :=
by
  have h1 : Real.logb 5 625 = 4 := by sorry
  have h2 : Real.logb 5 5 = 1 := by sorry
  have h3 : Real.logb 5 (1 / 25) = -2 := by sorry
  rw [h1, h2, h3]
  norm_num

end log_sum_eval_l67_67344


namespace intersection_points_l67_67258

noncomputable def curve1 (x y : ℝ) : Prop := x^2 + 4 * y^2 = 1
noncomputable def curve2 (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

theorem intersection_points : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 2 := 
by 
  sorry

end intersection_points_l67_67258


namespace product_mod_5_l67_67501

-- The sequence 3,13,...,93 can be identified as: 3 + (10 * n) for n = 0,1,2,...,9
def sequence : Nat → Nat := λ n => 3 + 10 * n

-- We state that all elements in the sequence mod 5 is 3
def sequence_mod_5 : ∀ n : Fin 10, sequence n % 5 = 3 :=
  by
  intros n
  fin_cases n
  all_goals simp [sequence]; norm_num

-- We now state the main problem
theorem product_mod_5 : 
    (∏ n in Finset.range 10, sequence n) % 5 = 4 :=
  by
  have h : ∏ n in Finset.range 10, sequence n = 3 ^ 10 :=
    sorry -- The product of the sequence is equal to 3^10
  rw h
  norm_num

#eval (3 ^ 10) % 5 -- this should evaluate to 4

end product_mod_5_l67_67501


namespace three_digit_integers_l67_67533

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 0 ∧ d ≠ 4

def contains_two (n : ℕ) : Prop :=
  ∃ k, n / 10^k % 10 = 2

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def does_not_contain_four (n : ℕ) : Prop :=
  ∀ k, n / 10^k % 10 ≠ 4

theorem three_digit_integers : {n : ℕ | is_three_digit n ∧ contains_two n ∧ does_not_contain_four n}.card = 200 := by
  sorry

end three_digit_integers_l67_67533


namespace parabola_directrix_l67_67494

noncomputable def equation_of_directrix (a h k : ℝ) : ℝ :=
  k - 1 / (4 * a)

theorem parabola_directrix:
  ∀ (a h k : ℝ), a = -3 ∧ h = 1 ∧ k = -2 → equation_of_directrix a h k = - 23 / 12 :=
by
  intro a h k
  intro h_ahk
  sorry

end parabola_directrix_l67_67494


namespace f_7_eq_neg3_l67_67354

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom f_interval  : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4

theorem f_7_eq_neg3 : f 7 = -3 :=
  sorry

end f_7_eq_neg3_l67_67354


namespace fraction_simplifies_l67_67685

def current_age_grant := 25
def current_age_hospital := 40

def age_in_five_years (current_age : Nat) : Nat := current_age + 5

def grant_age_in_5_years := age_in_five_years current_age_grant
def hospital_age_in_5_years := age_in_five_years current_age_hospital

def fraction_of_ages := grant_age_in_5_years / hospital_age_in_5_years

theorem fraction_simplifies : fraction_of_ages = (2 / 3) := by
  sorry

end fraction_simplifies_l67_67685


namespace inequality_with_means_l67_67875

theorem inequality_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_with_means_l67_67875


namespace A_infinite_l67_67363

noncomputable def f : ℝ → ℝ := sorry

def A : Set ℝ := { a : ℝ | f a > a ^ 2 }

theorem A_infinite
  (h_f_def : ∀ x : ℝ, ∃ y : ℝ, y = f x)
  (h_inequality: ∀ x : ℝ, (f x) ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
  (h_A_nonempty : A ≠ ∅) :
  Set.Infinite A := 
sorry

end A_infinite_l67_67363


namespace even_function_condition_iff_l67_67803

theorem even_function_condition_iff (m : ℝ) :
    (∀ x : ℝ, (m * 2^x + 2^(-x)) = (m * 2^(-x) + 2^x)) ↔ (m = 1) :=
by
  sorry

end even_function_condition_iff_l67_67803


namespace moles_of_HC2H3O2_needed_l67_67349

theorem moles_of_HC2H3O2_needed :
  (∀ (HC2H3O2 NaHCO3 H2O : ℕ), 
    (HC2H3O2 + NaHCO3 = NaC2H3O2 + H2O + CO2) → 
    (H2O = 3) → 
    (NaHCO3 = 3) → 
    HC2H3O2 = 3) :=
by
  intros HC2H3O2 NaHCO3 H2O h_eq h_H2O h_NaHCO3
  -- Hint: You can use the balanced chemical equation to derive that HC2H3O2 must be 3
  sorry

end moles_of_HC2H3O2_needed_l67_67349


namespace product_mod5_is_zero_l67_67329

theorem product_mod5_is_zero :
  (2023 * 2024 * 2025 * 2026) % 5 = 0 :=
by
  sorry

end product_mod5_is_zero_l67_67329


namespace area_of_circle_with_given_circumference_l67_67447

-- Defining the given problem's conditions as variables
variables (C : ℝ) (r : ℝ) (A : ℝ)
  
-- The condition that circumference is 12π meters
def circumference_condition : Prop := C = 12 * Real.pi
  
-- The relationship between circumference and radius
def radius_relationship : Prop := C = 2 * Real.pi * r
  
-- The formula to calculate the area of the circle
def area_formula : Prop := A = Real.pi * r^2
  
-- The proof goal that we need to establish
theorem area_of_circle_with_given_circumference :
  circumference_condition C ∧ radius_relationship C r ∧ area_formula A r → A = 36 * Real.pi :=
by
  intros
  sorry -- Skipping the proof, to be done later

end area_of_circle_with_given_circumference_l67_67447


namespace percent_of_carnations_l67_67313

variable (totalFlowers : ℚ)
variable (pinkFlowers : ℚ := 3/5 * totalFlowers)
variable (redFlowers : ℚ := 2/5 * totalFlowers)
variable (pinkRoses : ℚ := 1/3 * pinkFlowers)
variable (pinkCarnations : ℚ := 2/3 * pinkFlowers)
variable (redCarnations : ℚ := 3/4 * redFlowers)

theorem percent_of_carnations
  (h1 : pinkFlowers = 3/5 * totalFlowers)
  (h2 : redFlowers = 2/5 * totalFlowers)
  (h3 : pinkRoses = 1/3 * pinkFlowers)
  (h4 : pinkCarnations = 2/3 * pinkFlowers)
  (h5 : redCarnations = 3/4 * redFlowers) :
  (pinkCarnations + redCarnations) / totalFlowers * 100 = 70 := by
  sorry

end percent_of_carnations_l67_67313


namespace expectation_absolute_deviation_l67_67283

noncomputable def absolute_deviation (n : ℕ) (m : ℕ) : ℝ :=
    |(m / n : ℝ) - 0.5|

theorem expectation_absolute_deviation (n m : ℕ) 
    (h₁ : n = 10) (h₂ : n = 100) (pn : ℕ → Real) :
    let E_α := ∫ x, absolute_deviation 10 (pn x): ℝ in
    let E_β := ∫ x, absolute_deviation 100 (pn x): ℝ in
    E_α > E_β :=
begin
  sorry
end

end expectation_absolute_deviation_l67_67283


namespace comp_figure_perimeter_l67_67708

-- Given conditions
def side_length_square : ℕ := 2
def side_length_triangle : ℕ := 1
def number_of_squares : ℕ := 4
def number_of_triangles : ℕ := 3

-- Define the perimeter calculation
def perimeter_of_figure : ℕ :=
  let perimeter_squares := (2 * (number_of_squares - 2) + 2 * 2 + 2 * 1) * side_length_square
  let perimeter_triangles := number_of_triangles * side_length_triangle
  perimeter_squares + perimeter_triangles

-- Target theorem
theorem comp_figure_perimeter : perimeter_of_figure = 17 := by
  sorry

end comp_figure_perimeter_l67_67708


namespace sin_alpha_plus_7pi_over_12_l67_67192

theorem sin_alpha_plus_7pi_over_12 (α : Real) 
  (h1 : Real.cos (α + π / 12) = 1 / 5) : 
  Real.sin (α + 7 * π / 12) = 1 / 5 :=
by
  sorry

end sin_alpha_plus_7pi_over_12_l67_67192


namespace fraction_inspected_by_Jane_l67_67870

theorem fraction_inspected_by_Jane (P : ℝ) (x y : ℝ) 
    (h1: 0.007 * x * P + 0.008 * y * P = 0.0075 * P) 
    (h2: x + y = 1) : y = 0.5 :=
by sorry

end fraction_inspected_by_Jane_l67_67870


namespace neither_sufficient_nor_necessary_l67_67081

variable {a b : ℝ}

theorem neither_sufficient_nor_necessary (hab_ne_zero : a * b ≠ 0) :
  ¬ (a * b > 1 → a > (1 / b)) ∧ ¬ (a > (1 / b) → a * b > 1) :=
sorry

end neither_sufficient_nor_necessary_l67_67081


namespace original_monthly_bill_l67_67880

-- Define the necessary conditions
def increased_bill (original: ℝ): ℝ := original + 0.3 * original
def total_bill_after_increase : ℝ := 78

-- The proof we need to construct
theorem original_monthly_bill (X : ℝ) (H : increased_bill X = total_bill_after_increase) : X = 60 :=
by {
    sorry -- Proof is not required, only statement
}

end original_monthly_bill_l67_67880


namespace coordinates_of_vertex_B_equation_of_line_BC_l67_67206

noncomputable def vertex_A : (ℝ × ℝ) := (5, 1)
def bisector_expr (x y : ℝ) : Prop := x + y - 5 = 0
def median_CM_expr (x y : ℝ) : Prop := 2 * x - y - 5 = 0

theorem coordinates_of_vertex_B (B : ℝ × ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  B = (2, 3) :=
sorry

theorem equation_of_line_BC (coeff_3x coeff_2y const : ℝ) 
  (h1 : ∃ x y, bisector_expr x y ∧ median_CM_expr x y) :
  coeff_3x = 3 ∧ coeff_2y = 2 ∧ const = -12 :=
sorry

end coordinates_of_vertex_B_equation_of_line_BC_l67_67206


namespace Captain_Zarnin_staffing_scheme_l67_67781

theorem Captain_Zarnin_staffing_scheme :
  let positions := 6
  let candidates := 15
  (Nat.choose candidates positions) * 
  (Nat.factorial positions) = 3276000 :=
by
  let positions := 6
  let candidates := 15
  let ways_to_choose := Nat.choose candidates positions
  let ways_to_permute := Nat.factorial positions
  have h : (ways_to_choose * ways_to_permute) = 3276000 := sorry
  exact h

end Captain_Zarnin_staffing_scheme_l67_67781


namespace real_numbers_int_approximation_l67_67855

theorem real_numbers_int_approximation:
  ∀ (x y : ℝ), ∃ (m n : ℤ),
  (x - m) ^ 2 + (y - n) * (x - m) + (y - n) ^ 2 ≤ (1 / 3) :=
by
  intros x y
  sorry

end real_numbers_int_approximation_l67_67855


namespace evaluate_fraction_l67_67345

theorem evaluate_fraction : (25 * 5 + 5^2) / (5^2 - 15) = 15 := 
by
  sorry

end evaluate_fraction_l67_67345


namespace hawks_score_l67_67342

theorem hawks_score (x y : ℕ) (h1 : x + y = 82) (h2 : x - y = 18) : y = 32 :=
sorry

end hawks_score_l67_67342


namespace answer_l67_67807

def p := ∃ x : ℝ, x - 2 > Real.log x
def q := ∀ x : ℝ, Real.exp x > 1

theorem answer (hp : p) (hq : ¬ q) : p ∧ ¬ q :=
  by
    exact ⟨hp, hq⟩

end answer_l67_67807


namespace range_of_m_values_l67_67669

theorem range_of_m_values {P Q : ℝ × ℝ} (hP : P = (-1, 1)) (hQ : Q = (2, 2)) (m : ℝ) :
  -3 < m ∧ m < -2 / 3 → (∃ (l : ℝ → ℝ), ∀ x y, y = l x → x + m * y + m = 0) :=
sorry

end range_of_m_values_l67_67669


namespace expression_evaluation_l67_67063

variable (x y : ℝ)

theorem expression_evaluation (h1 : x = 2 * y) (h2 : y ≠ 0) : 
  (x + 2 * y) - (2 * x + y) = -y := 
by
  sorry

end expression_evaluation_l67_67063


namespace triangle_interior_angle_l67_67443

-- Define the given values and equations
variables (x : ℝ) 
def arc_DE := x + 80
def arc_EF := 2 * x + 30
def arc_FD := 3 * x - 25

-- The main proof statement
theorem triangle_interior_angle :
  arc_DE x + arc_EF x + arc_FD x = 360 →
  0.5 * (arc_EF x) = 60.83 :=
by sorry

end triangle_interior_angle_l67_67443


namespace men_in_first_group_l67_67102

theorem men_in_first_group (M : ℕ) (h1 : (M * 7 * 18) = (12 * 7 * 12)) : M = 8 :=
by sorry

end men_in_first_group_l67_67102


namespace parabola_focus_l67_67108

theorem parabola_focus (x y : ℝ) (h : y = 4 * x^2) : (0, 1 / 16) = (0, 1 / 16) :=
by
  sorry

end parabola_focus_l67_67108


namespace prob_two_out_of_three_successes_l67_67319

open ProbabilityTheory

noncomputable def prob_two_successes (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) : ℝ :=
  3 * p^2 * (1 - p)

theorem prob_two_out_of_three_successes (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
  (∑ A in ({ [ tt, tt, ff ], [ tt, ff, tt ], [ ff, tt, tt ] } : finset (vector bool 3)),
    probs_of_independent_events A p) = 3 * p^2 * (1 - p) :=
by
  sorry

-- Helper function to calculate the probability of specific outcomes of independent events.
def probs_of_independent_events (outcomes : vector bool 3) (p : ℝ) : ℝ :=
  (outcomes.to_list.map (λ b, if b then p else 1 - p)).prod

end prob_two_out_of_three_successes_l67_67319


namespace arrange_books_l67_67095

open Nat

theorem arrange_books :
    let german_books := 3
    let spanish_books := 4
    let french_books := 3
    let total_books := german_books + spanish_books + french_books
    (total_books == 10) →
    let units := 2
    let items_to_arrange := units + german_books
    factorial items_to_arrange * factorial spanish_books * factorial french_books = 17280 :=
by 
    intros
    sorry

end arrange_books_l67_67095


namespace min_cookies_divisible_by_13_l67_67777

theorem min_cookies_divisible_by_13 (a b : ℕ) : ∃ n : ℕ, n > 0 ∧ n % 13 = 0 ∧ (∃ a b : ℕ, n = 10 * a + 21 * b) ∧ n = 52 :=
by
  sorry

end min_cookies_divisible_by_13_l67_67777


namespace find_certain_number_l67_67373

theorem find_certain_number (x : ℤ) (h : x - 5 = 4) : x = 9 :=
sorry

end find_certain_number_l67_67373


namespace angle_sum_x_y_l67_67225

theorem angle_sum_x_y 
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ) (x : ℝ) (y : ℝ) 
  (hA : angle_A = 34) (hB : angle_B = 80) (hC : angle_C = 30) 
  (hexagon_property : ∀ A B x y : ℝ, A + B + 360 - x + 90 + 120 - y = 720) :
  x + y = 36 :=
by
  sorry

end angle_sum_x_y_l67_67225


namespace equivalent_discount_l67_67942

theorem equivalent_discount (original_price : ℝ) (d1 d2 single_discount : ℝ) :
  original_price = 50 →
  d1 = 0.15 →
  d2 = 0.10 →
  single_discount = 0.235 →
  original_price * (1 - d1) * (1 - d2) = original_price * (1 - single_discount) :=
by
  intros
  sorry

end equivalent_discount_l67_67942


namespace most_reasonable_sampling_method_l67_67122

-- Define the conditions
def significant_difference_by_educational_stage := true
def no_significant_difference_by_gender := true

-- Define the statement
theorem most_reasonable_sampling_method :
  (significant_difference_by_educational_stage ∧ no_significant_difference_by_gender) →
  "Stratified sampling by educational stage" = "most reasonable sampling method" :=
by
  sorry

end most_reasonable_sampling_method_l67_67122


namespace inequality_proof_l67_67804

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                        (hb : 0 ≤ b) (hb1 : b ≤ 1)
                        (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1) :=
by 
  sorry

end inequality_proof_l67_67804


namespace true_discount_correct_l67_67608

noncomputable def true_discount (banker_gain : ℝ) (average_rate : ℝ) (time_years : ℝ) : ℝ :=
  let r := average_rate
  let t := time_years
  let exp_factor := Real.exp (-r * t)
  let face_value := banker_gain / (1 - exp_factor)
  face_value - (face_value * exp_factor)

theorem true_discount_correct : 
  true_discount 15.8 0.145 5 = 15.8 := 
by
  sorry

end true_discount_correct_l67_67608


namespace value_of_a_in_terms_of_b_l67_67453

noncomputable def value_of_a (b : ℝ) : ℝ :=
  b * (38.1966 / 61.8034)

theorem value_of_a_in_terms_of_b (b a : ℝ) :
  (∀ x : ℝ, (b / x = 61.80339887498949 / 100) ∧ (x = (a + b) * (61.80339887498949 / 100)))
  → a = value_of_a b :=
by
  sorry

end value_of_a_in_terms_of_b_l67_67453


namespace angle_terminal_side_l67_67888

theorem angle_terminal_side (k : ℤ) : ∃ α : ℝ, α = k * 360 - 30 ∧ 0 ≤ α ∧ α < 360 →
  α = 330 :=
by
  sorry

end angle_terminal_side_l67_67888


namespace right_triangle_sides_l67_67003

theorem right_triangle_sides (a b c : ℝ)
    (h1 : a + b + c = 30)
    (h2 : a^2 + b^2 = c^2)
    (h3 : ∃ r, a = (5 * r) / 2 ∧ a + b = 5 * r ∧ ∀ x y, x / y = 2 / 3) :
  a = 5 ∧ b = 12 ∧ c = 13 :=
sorry

end right_triangle_sides_l67_67003


namespace even_function_a_value_l67_67368

theorem even_function_a_value {f : ℝ → ℝ} (a : ℝ) :
  (∀ x : ℝ, f x = x^3 * (a * 2^x - 2^(-x)) ∧ f x = f (-x)) → a = 1 :=
by
  intros h,
  sorry

end even_function_a_value_l67_67368


namespace area_of_triangle_aef_l67_67714

noncomputable def length_ab : ℝ := 10
noncomputable def width_ad : ℝ := 6
noncomputable def diagonal_ac : ℝ := Real.sqrt (length_ab^2 + width_ad^2)
noncomputable def segment_length_ac : ℝ := diagonal_ac / 4
noncomputable def area_aef : ℝ := (1/2) * segment_length_ac * ((60 * diagonal_ac) / diagonal_ac^2)

theorem area_of_triangle_aef : area_aef = 7.5 := by
  sorry

end area_of_triangle_aef_l67_67714


namespace tan_240_eq_sqrt3_l67_67171

theorem tan_240_eq_sqrt3 : Real.tan (240 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end tan_240_eq_sqrt3_l67_67171


namespace probability_of_rolling_less_than_five_l67_67741

/-- Rolling an 8-sided die -/
def outcomes := { 1, 2, 3, 4, 5, 6, 7, 8 }

/-- Successful outcomes (numbers less than 5) -/
def successful_outcomes := { 1, 2, 3, 4 }

/-- Number of total outcomes -/
def total_outcomes := 8

/-- Number of successful outcomes -/
def num_successful_outcomes := 4

/-- Probability of rolling a number less than 5 on an 8-sided die -/
def probability : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_rolling_less_than_five : probability = 1 / 2 := by
  sorry

end probability_of_rolling_less_than_five_l67_67741


namespace raindrop_prob_green_slope_l67_67772

-- Define the angles at the apex of the pyramid
variables (α β γ : ℝ)

-- Define that the angles at the apex are right angles
axiom right_angle_sum : α + β + γ = π / 2

-- The main theorem to state the probability result
theorem raindrop_prob_green_slope (h : right_angle_sum α β γ) : 
  (1 - cos α ^ 2 - sin β ^ 2) = cos γ ^ 2 :=
sorry

end raindrop_prob_green_slope_l67_67772


namespace four_digit_numbers_with_one_digit_as_average_l67_67211

noncomputable def count_valid_four_digit_numbers : Nat := 80

theorem four_digit_numbers_with_one_digit_as_average :
  ∃ n : Nat, n = count_valid_four_digit_numbers ∧ n = 80 := by
  use count_valid_four_digit_numbers
  constructor
  · rfl
  · rfl

end four_digit_numbers_with_one_digit_as_average_l67_67211


namespace Jo_has_least_l67_67016

variable (Money : Type) 
variable (Bo Coe Flo Jo Moe Zoe : Money)
variable [LT Money] [LE Money] -- Money type is an ordered type with less than and less than or equal relations.

-- Conditions
axiom h1 : Jo < Flo 
axiom h2 : Flo < Bo
axiom h3 : Jo < Moe
axiom h4 : Moe < Bo
axiom h5 : Bo < Coe
axiom h6 : Coe < Zoe

-- The main statement to prove that Jo has the least money.
theorem Jo_has_least (h1 : Jo < Flo) (h2 : Flo < Bo) (h3 : Jo < Moe) (h4 : Moe < Bo) (h5 : Bo < Coe) (h6 : Coe < Zoe) : 
  ∀ x, x = Jo ∨ x = Bo ∨ x = Flo ∨ x = Moe ∨ x = Coe ∨ x = Zoe → Jo ≤ x :=
by
  -- Proof is skipped using sorry
  sorry

end Jo_has_least_l67_67016


namespace rosie_purchase_price_of_art_piece_l67_67909

-- Define the conditions as hypotheses
variables (P : ℝ)
variables (future_value increase : ℝ)

-- Given conditions
def conditions := future_value = 3 * P ∧ increase = 8000 ∧ increase = future_value - P

-- The statement to be proved
theorem rosie_purchase_price_of_art_piece (h : conditions P future_value increase) : P = 4000 :=
sorry

end rosie_purchase_price_of_art_piece_l67_67909


namespace brendan_remaining_money_l67_67647

-- Definitions given in the conditions
def weekly_pay (total_monthly_earnings : ℕ) (weeks_in_month : ℕ) : ℕ := total_monthly_earnings / weeks_in_month
def weekly_recharge_amount (weekly_pay : ℕ) : ℕ := weekly_pay / 2
def total_recharge_amount (weekly_recharge_amount : ℕ) (weeks_in_month : ℕ) : ℕ := weekly_recharge_amount * weeks_in_month
def remaining_money_after_car_purchase (total_monthly_earnings : ℕ) (car_cost : ℕ) : ℕ := total_monthly_earnings - car_cost
def total_remaining_money (remaining_money_after_car_purchase : ℕ) (total_recharge_amount : ℕ) : ℕ := remaining_money_after_car_purchase - total_recharge_amount

-- The actual statement to prove
theorem brendan_remaining_money
  (total_monthly_earnings : ℕ := 5000)
  (weeks_in_month : ℕ := 4)
  (car_cost : ℕ := 1500)
  (weekly_pay := weekly_pay total_monthly_earnings weeks_in_month)
  (weekly_recharge_amount := weekly_recharge_amount weekly_pay)
  (total_recharge_amount := total_recharge_amount weekly_recharge_amount weeks_in_month)
  (remaining_money_after_car_purchase := remaining_money_after_car_purchase total_monthly_earnings car_cost)
  (total_remaining_money := total_remaining_money remaining_money_after_car_purchase total_recharge_amount) :
  total_remaining_money = 1000 :=
sorry

end brendan_remaining_money_l67_67647


namespace sum_of_geometric_sequence_l67_67021

-- Consider a geometric sequence {a_n} with the first term a_1 = 1 and a common ratio of 1/3.
-- Let S_n denote the sum of the first n terms.
-- We need to prove that S_n = (3 - a_n) / 2, given the above conditions.
noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2

theorem sum_of_geometric_sequence (n : ℕ) : geometric_sequence_sum n = 
  let a_1 := 1
  let r := (1 : ℝ) / 3
  let a_n := r ^ (n - 1)
  (3 - a_n) / 2 := sorry

end sum_of_geometric_sequence_l67_67021


namespace parabola_find_m_l67_67194

theorem parabola_find_m
  (p m : ℝ) (h_p_pos : p > 0) (h_point_on_parabola : (2 * p * m) = 8)
  (h_chord_length : (m + (2 / m))^2 - m^2 = 7) : m = (2 * Real.sqrt 3) / 3 :=
by sorry

end parabola_find_m_l67_67194


namespace Jonah_calories_burn_l67_67098

-- Definitions based on conditions
def burn_calories (hours : ℕ) : ℕ := hours * 30

theorem Jonah_calories_burn (h1 : burn_calories 2 = 60) : burn_calories 5 - burn_calories 2 = 90 :=
by
  have h2 : burn_calories 5 = 150 := rfl
  rw [h1, h2]
  exact rfl

end Jonah_calories_burn_l67_67098


namespace sum_of_altitudes_less_than_sum_of_sides_l67_67414

-- Define a triangle with sides and altitudes properties
structure Triangle :=
(A B C : Point)
(a b c : ℝ)
(m_a m_b m_c : ℝ)
(sides : a + b > c ∧ b + c > a ∧ c + a > b) -- Triangle Inequality

axiom altitude_property (T : Triangle) :
  T.m_a < T.b ∧ T.m_b < T.c ∧ T.m_c < T.a

-- The theorem to prove
theorem sum_of_altitudes_less_than_sum_of_sides (T : Triangle) :
  T.m_a + T.m_b + T.m_c < T.a + T.b + T.c :=
sorry

end sum_of_altitudes_less_than_sum_of_sides_l67_67414


namespace inequality_solution_range_l67_67523

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| + |x| ≤ a) ↔ a ≥ 2 :=
by
  sorry

end inequality_solution_range_l67_67523


namespace inequality_proof_l67_67364

variable (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_ab_bc_ca : a * b + b * c + c * a = 1)

theorem inequality_proof :
  (3 / Real.sqrt (a^2 + 1)) + (4 / Real.sqrt (b^2 + 1)) + (12 / Real.sqrt (c^2 + 1)) < (39 / 2) :=
by
  sorry

end inequality_proof_l67_67364


namespace masha_numbers_l67_67396

theorem masha_numbers (a b : ℕ) (S : ℕ) (h1 : a ≠ b) (h2 : a > 11) (h3 : b > 11) (h4 : S = a + b) 
    (h5 : (∀ x y : ℕ, x + y = S → x = a ∨ y = a → abs x - y = a) ∧ (even a ∨ even b)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := 
by sorry

end masha_numbers_l67_67396


namespace parabola_vertex_coordinates_l67_67576

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, -x^2 + 15 ≥ -x^2 + 15 :=
by
  sorry

end parabola_vertex_coordinates_l67_67576


namespace books_from_second_shop_l67_67568

theorem books_from_second_shop (x : ℕ) (h₁ : 6500 + 2000 = 8500)
    (h₂ : 85 = 8500 / (65 + x)) : x = 35 :=
by
  -- proof goes here
  sorry

end books_from_second_shop_l67_67568


namespace train_length_l67_67006

/-- Given that the jogger runs at 2.5 m/s,
    the train runs at 12.5 m/s, 
    the jogger is initially 260 meters ahead, 
    and the train takes 38 seconds to pass the jogger,
    prove that the length of the train is 120 meters. -/
theorem train_length (speed_jogger speed_train : ℝ) (initial_distance time_passing : ℝ)
  (hjogger : speed_jogger = 2.5) (htrain : speed_train = 12.5)
  (hinitial : initial_distance = 260) (htime : time_passing = 38) :
  ∃ L : ℝ, L = 120 :=
by
  sorry

end train_length_l67_67006


namespace ratio_of_pieces_l67_67615

theorem ratio_of_pieces (total_length : ℝ) (short_piece : ℝ) (total_length_eq : total_length = 70) (short_piece_eq : short_piece = 27.999999999999993) :
  let long_piece := total_length - short_piece
  let ratio := short_piece / long_piece
  ratio = 2 / 3 :=
by
  sorry

end ratio_of_pieces_l67_67615


namespace find_smallest_w_l67_67605

theorem find_smallest_w (w : ℕ) (h : 0 < w) : 
  (∀ k, k = 2^5 ∨ k = 3^3 ∨ k = 12^2 → (k ∣ (936 * w))) ↔ w = 36 := by 
  sorry

end find_smallest_w_l67_67605


namespace quadratic_roots_distinct_l67_67202

theorem quadratic_roots_distinct (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2*x1 + m = 0 ∧ x2^2 + 2*x2 + m = 0) →
  m < 1 := 
by
  sorry

end quadratic_roots_distinct_l67_67202


namespace exp_abs_dev_10_gt_100_l67_67290

open ProbabilityTheory
open MeasureTheory

def deviation (n m : ℕ) : ℝ := (m / n : ℝ) - 0.5
def abs_deviation (n m : ℕ) : ℝ := abs ((m / n : ℝ) - 0.5)
def exp_abs_deviation (n : ℕ) : ℝ := sorry -- The expected value of the absolute deviation

theorem exp_abs_dev_10_gt_100 :
  (exp_abs_deviation 10) > (exp_abs_deviation 100) :=
sorry

end exp_abs_dev_10_gt_100_l67_67290


namespace expression_eval_neg_sqrt_l67_67425

variable (a : ℝ)

theorem expression_eval_neg_sqrt (ha : a < 0) : a * Real.sqrt (-1 / a) = -Real.sqrt (-a) :=
by
  sorry

end expression_eval_neg_sqrt_l67_67425


namespace wrench_force_l67_67577

theorem wrench_force (F L k: ℝ) (h_inv: ∀ F L, F * L = k) (h_given: F * 12 = 240 * 12) : 
  (∀ L, (L = 16) → (F = 180)) ∧ (∀ L, (L = 8) → (F = 360)) := by 
sorry

end wrench_force_l67_67577


namespace problem1_problem2_l67_67043

variable {m x : ℝ}

-- Definition of the function f
def f (x m : ℝ) : ℝ := |x - m| + |x|

-- Statement for Problem (1)
theorem problem1 (h : f 1 m = 1) : 
  ∀ x, f x 1 < 2 ↔ (-1 / 2) < x ∧ x < (3 / 2) := 
sorry

-- Statement for Problem (2)
theorem problem2 (h : ∀ x, f x m ≥ m^2) : 
  -1 ≤ m ∧ m ≤ 1 := 
sorry

end problem1_problem2_l67_67043


namespace number_minus_45_l67_67768

theorem number_minus_45 (x : ℕ) (h1 : (x / 2) / 2 = 85 + 45) : x - 45 = 475 := by
  sorry

end number_minus_45_l67_67768


namespace tan_C_value_l67_67379

theorem tan_C_value (A B C : ℝ)
  (h_cos_A : Real.cos A = 4/5)
  (h_tan_A_minus_B : Real.tan (A - B) = -1/2) :
  Real.tan C = 11/2 :=
sorry

end tan_C_value_l67_67379


namespace celebrity_baby_photo_probability_l67_67763

theorem celebrity_baby_photo_probability : 
  let total_arrangements := Nat.factorial 4
  let correct_arrangements := 1
  let probability := correct_arrangements / total_arrangements
  probability = 1/24 :=
by
  sorry

end celebrity_baby_photo_probability_l67_67763


namespace basketball_team_first_competition_games_l67_67463

-- Definitions given the conditions
def first_competition_games (x : ℕ) := x
def second_competition_games (x : ℕ) := (5 * x) / 8
def third_competition_games (x : ℕ) := x + (5 * x) / 8
def total_games (x : ℕ) := x + (5 * x) / 8 + (x + (5 * x) / 8)

-- Lean 4 statement to prove the correct answer
theorem basketball_team_first_competition_games : 
  ∃ x : ℕ, total_games x = 130 ∧ first_competition_games x = 40 :=
by
  sorry

end basketball_team_first_competition_games_l67_67463


namespace find_x_l67_67052

theorem find_x (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 :=
by
  sorry

end find_x_l67_67052


namespace chips_needed_per_console_l67_67943

-- Definitions based on the conditions
def chips_per_day : ℕ := 467
def consoles_per_day : ℕ := 93

-- The goal is to prove that each video game console needs 5 computer chips
theorem chips_needed_per_console : chips_per_day / consoles_per_day = 5 :=
by sorry

end chips_needed_per_console_l67_67943


namespace vector_dot_product_l67_67543

def vector := ℝ × ℝ

def collinear (a b : vector) : Prop :=
  ∃ (k : ℝ), a = (k * b.1, k * b.2)

noncomputable def dot_product (v w : vector) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_dot_product (k : ℝ) (h_collinear : collinear (3 / 2, 1) (3, k))
  (h_k : k = 2) :
  dot_product ((3 / 2, 1) - (3, k)) (2 * (3 / 2, 1) + (3, k)) = -13 :=
by
  sorry

end vector_dot_product_l67_67543


namespace evaluate_expression_l67_67026

theorem evaluate_expression : 
  3 * (-3)^4 + 3 * (-3)^3 + 3 * (-3)^2 + 3 * 3^2 + 3 * 3^3 + 3 * 3^4 = 540 := 
by 
  sorry

end evaluate_expression_l67_67026


namespace total_cards_given_away_l67_67230

-- Define the conditions in Lean
def Jim_initial_cards : ℕ := 365
def sets_given_to_brother : ℕ := 8
def sets_given_to_sister : ℕ := 5
def sets_given_to_friend : ℕ := 2
def cards_per_set : ℕ := 13

-- Define a theorem to prove the total number of cards given away
theorem total_cards_given_away : 
  sets_given_to_brother + sets_given_to_sister + sets_given_to_friend = 15 ∧
  15 * cards_per_set = 195 := 
by
  sorry

end total_cards_given_away_l67_67230


namespace diameter_of_circle_is_60_l67_67147

noncomputable def diameter_of_circle (M N : ℝ) : ℝ :=
  if h : N ≠ 0 then 2 * (M / N * (1 / (2 * Real.pi))) else 0

theorem diameter_of_circle_is_60 (M N : ℝ) (h : M / N = 15) :
  diameter_of_circle M N = 60 :=
by
  sorry

end diameter_of_circle_is_60_l67_67147


namespace mashas_numbers_l67_67406

def is_even (n : ℕ) : Prop := n % 2 = 0

def problem_statement (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ is_even a ∧ a + b = 28
  
theorem mashas_numbers : ∃ (a b : ℕ), problem_statement a b :=
by
  use 12
  use 16
  unfold problem_statement
  split
  -- a ≠ b
  exact dec_trivial
  split
  -- a > 11
  exact dec_trivial
  split
  -- b > 11
  exact dec_trivial
  split
  -- is_even a
  exact dec_trivial
  -- a + b = 28
  exact dec_trivial

end mashas_numbers_l67_67406


namespace monochromatic_triangle_probability_l67_67025

noncomputable def probability_monochromatic_triangle : ℚ := sorry

theorem monochromatic_triangle_probability :
  -- Condition: Each of the 6 sides and the 9 diagonals of a regular hexagon are randomly and independently colored red, blue, or green with equal probability.
  -- Proof: The probability that at least one triangle whose vertices are among the vertices of the hexagon has all its sides of the same color is equal to 872/1000.
  probability_monochromatic_triangle = 872 / 1000 :=
sorry

end monochromatic_triangle_probability_l67_67025


namespace alice_bob_meet_l67_67323

theorem alice_bob_meet :
  ∃ k : ℕ, (4 * k - 4 * (k / 5) ≡ 8 * k [MOD 15]) ∧ (k = 5) :=
by
  sorry

end alice_bob_meet_l67_67323


namespace expression_value_l67_67651

theorem expression_value (a b c d : ℤ) (h_a : a = 15) (h_b : b = 19) (h_c : c = 3) (h_d : d = 2) :
  (a - (b - c)) - ((a - b) - c + d) = 4 := 
by
  rw [h_a, h_b, h_c, h_d]
  sorry

end expression_value_l67_67651


namespace black_white_area_ratio_l67_67179

theorem black_white_area_ratio :
  let r1 := 2
  let r2 := 6
  let r3 := 10
  let r4 := 14
  let r5 := 18
  let area (r : ℝ) := π * r^2
  let black_area := area r1 + (area r3 - area r2) + (area r5 - area r4)
  let white_area := (area r2 - area r1) + (area r4 - area r3)
  black_area / white_area = (49 : ℝ) / 32 :=
by
  sorry

end black_white_area_ratio_l67_67179


namespace better_offer_saves_800_l67_67759

theorem better_offer_saves_800 :
  let initial_order := 20000
  let discount1 (x : ℝ) := x * 0.70 * 0.90 - 800
  let discount2 (x : ℝ) := x * 0.75 * 0.80 - 1000
  discount1 initial_order - discount2 initial_order = 800 :=
by
  sorry

end better_offer_saves_800_l67_67759


namespace number_of_boys_l67_67221

theorem number_of_boys (b g : ℕ) (h1: (3/5 : ℚ) * b = (5/6 : ℚ) * g) (h2: b + g = 30)
  (h3: g = (b * 18) / 25): b = 17 := by
  sorry

end number_of_boys_l67_67221


namespace avg_score_assigned_day_l67_67545

theorem avg_score_assigned_day
  (total_students : ℕ)
  (exam_assigned_day_students_perc : ℕ)
  (exam_makeup_day_students_perc : ℕ)
  (avg_makeup_day_score : ℕ)
  (total_avg_score : ℕ)
  : exam_assigned_day_students_perc = 70 → 
    exam_makeup_day_students_perc = 30 → 
    avg_makeup_day_score = 95 → 
    total_avg_score = 74 → 
    total_students = 100 → 
    (70 * 65 + 30 * 95 = 7400) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end avg_score_assigned_day_l67_67545


namespace hospital_bed_occupancy_l67_67767

theorem hospital_bed_occupancy 
  (x : ℕ)
  (beds_A := x)
  (beds_B := 2 * x)
  (beds_C := 3 * x)
  (occupied_A := (1 / 3) * x)
  (occupied_B := (1 / 2) * (2 * x))
  (occupied_C := (1 / 4) * (3 * x))
  (max_capacity_B := (3 / 4) * (2 * x))
  (max_capacity_C := (5 / 6) * (3 * x)) :
  (4 / 3 * x) / (2 * x) = 2 / 3 ∧ (3 / 4 * x) / (3 * x) = 1 / 4 := 
  sorry

end hospital_bed_occupancy_l67_67767


namespace product_mod_5_l67_67499

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l67_67499


namespace kevin_hopping_distance_l67_67234

theorem kevin_hopping_distance :
  let hop_distance (n : Nat) : ℚ :=
    let factor : ℚ := (3/4 : ℚ)^n
    1/4 * factor
  let total_distance : ℚ :=
    (hop_distance 0 + hop_distance 1 + hop_distance 2 + hop_distance 3 + hop_distance 4 + hop_distance 5)
  total_distance = 39677 / 40960 :=
by
  sorry

end kevin_hopping_distance_l67_67234


namespace solution_set_of_x_l67_67974

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l67_67974


namespace probability_of_event_E_l67_67739

-- Define the event E as rolling a number less than 5
def event_E (n : ℕ) : Prop := n < 5

-- Define the sample space as the set {1, 2, ..., 8} for an 8-sided die
def sample_space : set ℕ := {n | n ≥ 1 ∧ n ≤ 8}

-- Define the probability function
def probability (E : set ℕ) (S : set ℕ) : ℚ :=
  if S.nonempty then
    E.to_finset.card.to_rat / S.to_finset.card.to_rat
  else 0

-- Prove that the probability of event E is 1/2 given the sample space
theorem probability_of_event_E :
  probability {n | event_E n} sample_space = 1 / 2 := by
  sorry

end probability_of_event_E_l67_67739


namespace agreed_upon_service_period_l67_67007

theorem agreed_upon_service_period (x : ℕ) (hx : 900 + 100 = 1000) 
(assumed_service : x * 1000 = 9 * (650 + 100)) :
  x = 12 :=
by {
  sorry
}

end agreed_upon_service_period_l67_67007


namespace floor_eq_l67_67982

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l67_67982


namespace b_plus_d_over_a_l67_67727

theorem b_plus_d_over_a (a b c d e : ℝ) (h : a ≠ 0) 
  (root1 : a * (5:ℝ)^4 + b * (5:ℝ)^3 + c * (5:ℝ)^2 + d * (5:ℝ) + e = 0)
  (root2 : a * (-3:ℝ)^4 + b * (-3:ℝ)^3 + c * (-3:ℝ)^2 + d * (-3:ℝ) + e = 0)
  (root3 : a * (2:ℝ)^4 + b * (2:ℝ)^3 + c * (2:ℝ)^2 + d * (2:ℝ) + e = 0) :
  (b + d) / a = - (12496 / 3173) :=
sorry

end b_plus_d_over_a_l67_67727


namespace problem_1_problem_2_l67_67176

-- Definitions of the given probabilities
def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/4
def prob_C : ℚ := 2/5

-- Independence implies that the probabilities of combined events are products of individual probabilities.
-- To avoid unnecessary complications, we assume independence holds true without proof.
axiom independence : ∀ A B C : Prop, (A ∧ B ∧ C) ↔ (A ∧ B) ∧ C

-- Problem statement for part (1)
theorem problem_1 : prob_A * prob_B * prob_C = 1/15 := by
  sorry

-- Helper definitions for probabilities of not visiting
def not_prob_A : ℚ := 1 - prob_A
def not_prob_B : ℚ := 1 - prob_B
def not_prob_C : ℚ := 1 - prob_C

-- Problem statement for part (2)
theorem problem_2 : (prob_A * not_prob_B * not_prob_C + not_prob_A * prob_B * not_prob_C + not_prob_A * not_prob_B * prob_C) = 9/20 := by
  sorry

end problem_1_problem_2_l67_67176


namespace second_car_speed_correct_l67_67444

noncomputable def first_car_speed : ℝ := 90

noncomputable def time_elapsed (h : ℕ) (m : ℕ) : ℝ := h + m / 60

noncomputable def distance_travelled (speed : ℝ) (time : ℝ) : ℝ := speed * time

def distance_ratio_at_832 (dist1 dist2 : ℝ) : Prop := dist1 = 1.2 * dist2
def distance_ratio_at_920 (dist1 dist2 : ℝ) : Prop := dist1 = 2 * dist2

noncomputable def time_first_car_832 : ℝ := time_elapsed 0 24
noncomputable def dist_first_car_832 : ℝ := distance_travelled first_car_speed time_first_car_832

noncomputable def dist_second_car_832 : ℝ := dist_first_car_832 / 1.2

noncomputable def time_first_car_920 : ℝ := time_elapsed 1 12
noncomputable def dist_first_car_920 : ℝ := distance_travelled first_car_speed time_first_car_920

noncomputable def dist_second_car_920 : ℝ := dist_first_car_920 / 2

noncomputable def time_second_car_travel : ℝ := time_elapsed 0 42

noncomputable def second_car_speed : ℝ := (dist_second_car_920 - dist_second_car_832) / time_second_car_travel

theorem second_car_speed_correct :
  second_car_speed = 34.2857 := by
  sorry

end second_car_speed_correct_l67_67444


namespace translated_circle_eq_l67_67276

theorem translated_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 16) →
  (x + 5) ^ 2 + (y + 3) ^ 2 = 16 :=
by
  sorry

end translated_circle_eq_l67_67276


namespace find_x_plus_one_over_x_l67_67362

open Real

theorem find_x_plus_one_over_x (x : ℝ) (h : x ^ 3 + 1 / x ^ 3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_plus_one_over_x_l67_67362


namespace wind_power_in_scientific_notation_l67_67153

theorem wind_power_in_scientific_notation :
  (56 * 10^6) = (5.6 * 10^7) :=
by
  sorry

end wind_power_in_scientific_notation_l67_67153


namespace snow_at_least_once_prob_l67_67658

-- Define the conditions for the problem
def prob_snow_day1_to_day4 : ℚ := 1 / 2
def prob_no_snow_day1_to_day4 : ℚ := 1 - prob_snow_day1_to_day4

def prob_snow_day5_to_day7 : ℚ := 1 / 3
def prob_no_snow_day5_to_day7 : ℚ := 1 - prob_snow_day5_to_day7

-- Define the probability of no snow during the first week of February
def prob_no_snow_week : ℚ := (prob_no_snow_day1_to_day4 ^ 4) * (prob_no_snow_day5_to_day7 ^ 3)

-- Define the probability that it snows at least once during the first week of February
def prob_snow_at_least_once : ℚ := 1 - prob_no_snow_week

-- The theorem we want to prove
theorem snow_at_least_once_prob : prob_snow_at_least_once = 53 / 54 :=
by
  sorry

end snow_at_least_once_prob_l67_67658


namespace concave_side_probability_l67_67246

theorem concave_side_probability (tosses : ℕ) (frequency_convex : ℝ) (htosses : tosses = 1000) (hfrequency : frequency_convex = 0.44) :
  ∀ probability_concave : ℝ, probability_concave = 1 - frequency_convex → probability_concave = 0.56 :=
by
  intros probability_concave h
  rw [hfrequency] at h
  rw [h]
  norm_num
  done

end concave_side_probability_l67_67246


namespace F_3_f_5_eq_24_l67_67805

def f (a : ℤ) : ℤ := a - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem F_3_f_5_eq_24 : F 3 (f 5) = 24 := by
  sorry

end F_3_f_5_eq_24_l67_67805


namespace jake_present_weight_l67_67062

theorem jake_present_weight :
  ∃ (J K L : ℕ), J = 194 ∧ J + K = 287 ∧ J - L = 2 * K ∧ J = 194 := by
  sorry

end jake_present_weight_l67_67062


namespace initial_average_marks_is_90_l67_67711

def incorrect_average_marks (A : ℝ) : Prop :=
  let wrong_sum := 10 * A
  let correct_sum := 10 * 95
  wrong_sum + 50 = correct_sum

theorem initial_average_marks_is_90 : ∃ A : ℝ, incorrect_average_marks A ∧ A = 90 :=
by
  use 90
  unfold incorrect_average_marks
  simp
  sorry

end initial_average_marks_is_90_l67_67711


namespace calculate_weight_of_first_batch_jelly_beans_l67_67079

theorem calculate_weight_of_first_batch_jelly_beans (J : ℝ)
    (h1 : 16 = 8 * (J * 4)) : J = 2 := 
  sorry

end calculate_weight_of_first_batch_jelly_beans_l67_67079


namespace most_and_least_l67_67865

variables {Jan Kim Lee Ron Zay : ℝ}

-- Conditions as hypotheses
axiom H1 : Lee < Jan
axiom H2 : Kim < Jan
axiom H3 : Zay < Ron
axiom H4 : Zay < Lee
axiom H5 : Zay < Jan
axiom H6 : Jan < Ron

theorem most_and_least :
  (Ron > Jan) ∧ (Ron > Kim) ∧ (Ron > Lee) ∧ (Ron > Zay) ∧ 
  (Zay < Jan) ∧ (Zay < Kim) ∧ (Zay < Lee) ∧ (Zay < Ron) :=
by {
  -- Proof is omitted
  sorry
}

end most_and_least_l67_67865


namespace range_of_q_eq_eight_inf_l67_67487

noncomputable def q (x : ℝ) : ℝ := (x^2 + 2)^3

theorem range_of_q_eq_eight_inf (x : ℝ) : 0 ≤ x → ∃ y, y = q x ∧ 8 ≤ y := sorry

end range_of_q_eq_eight_inf_l67_67487


namespace train_platform_length_l67_67925

theorem train_platform_length (time_platform : ℝ) (time_man : ℝ) (speed_km_per_hr : ℝ) :
  time_platform = 34 ∧ time_man = 20 ∧ speed_km_per_hr = 54 →
  let speed_m_per_s := speed_km_per_hr * (5/18)
  let length_train := speed_m_per_s * time_man
  let time_to_cover_platform := time_platform - time_man
  let length_platform := speed_m_per_s * time_to_cover_platform
  length_platform = 210 := 
by {
  sorry
}

end train_platform_length_l67_67925


namespace upstream_travel_time_l67_67464

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l67_67464


namespace LindseyMinimumSavings_l67_67241
-- Import the library to bring in the necessary definitions and notations

-- Definitions from the problem conditions
def SeptemberSavings : ℕ := 50
def OctoberSavings : ℕ := 37
def NovemberSavings : ℕ := 11
def MomContribution : ℕ := 25
def VideoGameCost : ℕ := 87
def RemainingMoney : ℕ := 36

-- Problem statement as a Lean theorem
theorem LindseyMinimumSavings : 
  (SeptemberSavings + OctoberSavings + NovemberSavings) > 98 :=
  sorry

end LindseyMinimumSavings_l67_67241


namespace combined_work_rate_l67_67451

theorem combined_work_rate (W : ℝ) 
  (A_rate : ℝ := W / 10) 
  (B_rate : ℝ := W / 5) : 
  A_rate + B_rate = 3 * W / 10 := 
by
  sorry

end combined_work_rate_l67_67451


namespace speed_of_ferry_P_l67_67033

variable (v_P v_Q : ℝ)

noncomputable def condition1 : Prop := v_Q = v_P + 4
noncomputable def condition2 : Prop := (6 * v_P) / v_Q = 4
noncomputable def condition3 : Prop := 2 + 2 = 4

theorem speed_of_ferry_P
    (h1 : condition1 v_P v_Q)
    (h2 : condition2 v_P v_Q)
    (h3 : condition3) :
    v_P = 8 := 
by 
    sorry

end speed_of_ferry_P_l67_67033


namespace probability_of_two_approvals_l67_67632

-- Define the base probabilities
def P_A : ℝ := 0.6
def P_D : ℝ := 1 - P_A

-- Define the binomial coefficient function
def binom (n k : ℕ) := nat.choose n k

-- Define the probability of exactly k successes in n trials
noncomputable def P_exactly_two_approvals :=
  (binom 4 2) * (P_A ^ 2) * (P_D ^ 2)

theorem probability_of_two_approvals : P_exactly_two_approvals = 0.3456 := by
  sorry

end probability_of_two_approvals_l67_67632


namespace percentage_change_difference_l67_67015

-- Define initial and final percentages
def initial_yes : ℝ := 0.4
def initial_no : ℝ := 0.6
def final_yes : ℝ := 0.6
def final_no : ℝ := 0.4

-- Definition for the percentage of students who changed their opinion
def y_min : ℝ := 0.2 -- 20%
def y_max : ℝ := 0.6 -- 60%

-- Calculate the difference
def difference_y : ℝ := y_max - y_min

theorem percentage_change_difference :
  difference_y = 0.4 := by
  sorry

end percentage_change_difference_l67_67015


namespace jonah_total_raisins_l67_67867

-- Define the amounts of yellow and black raisins added
def yellow_raisins : ℝ := 0.3
def black_raisins : ℝ := 0.4

-- The main statement to be proved
theorem jonah_total_raisins : yellow_raisins + black_raisins = 0.7 :=
by 
  sorry

end jonah_total_raisins_l67_67867


namespace find_integer_pairs_l67_67795

def satisfies_conditions (m n : ℤ) : Prop :=
  m^2 = n^5 + n^4 + 1 ∧ ((m - 7 * n) ∣ (m - 4 * n))

theorem find_integer_pairs :
  ∀ (m n : ℤ), satisfies_conditions m n → (m, n) = (-1, 0) ∨ (m, n) = (1, 0) := by
  sorry

end find_integer_pairs_l67_67795


namespace B_work_rate_l67_67309

theorem B_work_rate (A_rate C_rate combined_rate : ℝ) (B_days : ℝ) (hA : A_rate = 1 / 4) (hC : C_rate = 1 / 8) (hCombined : A_rate + 1 / B_days + C_rate = 1 / 2) : B_days = 8 :=
by
  sorry

end B_work_rate_l67_67309


namespace cube_volume_l67_67906

theorem cube_volume (A : ℝ) (h : A = 24) : 
  ∃ V : ℝ, V = 8 :=
by
  sorry

end cube_volume_l67_67906


namespace prob_9_in_decimal_rep_of_3_over_11_l67_67565

def decimal_rep_of_3_over_11 : List ℕ := [2, 7]  -- decimal representation of 3/11 is 0.272727...

theorem prob_9_in_decimal_rep_of_3_over_11 : 
  (1 / (2 : ℚ)) * (decimal_rep_of_3_over_11.count 9) = 0 := by
  have h : 9 ∉ decimal_rep_of_3_over_11 := by simp only [decimal_rep_of_3_over_11, List.mem_cons, List.mem_nil, not_false_iff]; exact dec_trivial
  rw List.count_eq_zero_of_not_mem h
  norm_num
  sorry

end prob_9_in_decimal_rep_of_3_over_11_l67_67565


namespace chef_pies_total_l67_67964

def chefPieSales : ℕ :=
  let small_shepherd_pies := 52 / 4
  let large_shepherd_pies := 76 / 8
  let small_chicken_pies := 80 / 5
  let large_chicken_pies := 130 / 10
  let small_vegetable_pies := 42 / 6
  let large_vegetable_pies := 96 / 12
  let small_beef_pies := 35 / 7
  let large_beef_pies := 105 / 14

  small_shepherd_pies + large_shepherd_pies + small_chicken_pies + large_chicken_pies +
  small_vegetable_pies + large_vegetable_pies +
  small_beef_pies + large_beef_pies

theorem chef_pies_total : chefPieSales = 80 := by
  unfold chefPieSales
  have h1 : 52 / 4 = 13 := by norm_num
  have h2 : 76 / 8 = 9 ∨ 76 / 8 = 10 := by norm_num -- rounding consideration
  have h3 : 80 / 5 = 16 := by norm_num
  have h4 : 130 / 10 = 13 := by norm_num
  have h5 : 42 / 6 = 7 := by norm_num
  have h6 : 96 / 12 = 8 := by norm_num
  have h7 : 35 / 7 = 5 := by norm_num
  have h8 : 105 / 14 = 7 ∨ 105 / 14 = 8 := by norm_num -- rounding consideration
  sorry

end chef_pies_total_l67_67964


namespace smallest_positive_debt_resolved_l67_67732

theorem smallest_positive_debt_resolved : ∃ (D : ℤ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 250 * g) ∧ D = 50 :=
by
  sorry

end smallest_positive_debt_resolved_l67_67732


namespace evaluate_expression_l67_67661

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end evaluate_expression_l67_67661


namespace find_a_l67_67818

-- Define the hyperbola equation and the asymptote conditions
def hyperbola (a : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / 9) = 1

def asymptote1 (x y : ℝ) : Prop := 3 * x + 2 * y = 0
def asymptote2 (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Prove that if asymptote conditions hold, a = 2
theorem find_a (a : ℝ) (ha : a > 0) :
  (∀ x y, asymptote1 x y) ∧ (∀ x y, asymptote2 x y) → a = 2 :=
sorry

end find_a_l67_67818


namespace two_planes_divide_at_most_4_parts_l67_67274

-- Definitions related to the conditions
def Plane := ℝ × ℝ × ℝ → Prop -- Representing a plane in ℝ³ by an equation

-- Axiom: Two given planes
axiom plane1 : Plane
axiom plane2 : Plane

-- Conditions about their relationship
def are_parallel (p1 p2 : Plane) : Prop := 
  ∀ x y z, p1 (x, y, z) → p2 (x, y, z)

def intersect (p1 p2 : Plane) : Prop :=
  ∃ x y z, p1 (x, y, z) ∧ p2 (x, y, z)

-- Main theorem to state
theorem two_planes_divide_at_most_4_parts :
  (∃ p1 p2 : Plane, are_parallel p1 p2 ∨ intersect p1 p2) →
  (exists n : ℕ, n <= 4) :=
sorry

end two_planes_divide_at_most_4_parts_l67_67274


namespace gain_percent_l67_67135

theorem gain_percent (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1440) : 
  ((selling_price - cost_price) / cost_price) * 100 = 60 :=
by
  sorry

end gain_percent_l67_67135


namespace floor_eq_l67_67984

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l67_67984


namespace minimum_handshakes_l67_67068

noncomputable def min_handshakes (n : ℕ) (k : ℕ) : ℕ :=
  (n * k) / 2

theorem minimum_handshakes (n k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  min_handshakes n k = 45 :=
by
  -- We provide the conditions directly
  -- n = 30, k = 3
  rw [h1, h2]
  -- then show that min_handshakes 30 3 = 45
  show min_handshakes 30 3 = 45
  sorry 

end minimum_handshakes_l67_67068


namespace fraction_uncovered_l67_67923

def area_rug (length width : ℕ) : ℕ := length * width
def area_square (side : ℕ) : ℕ := side * side

theorem fraction_uncovered 
  (rug_length rug_width floor_area : ℕ)
  (h_rug_length : rug_length = 2)
  (h_rug_width : rug_width = 7)
  (h_floor_area : floor_area = 64)
  : (floor_area - area_rug rug_length rug_width) / floor_area = 25 / 32 := 
sorry

end fraction_uncovered_l67_67923


namespace intersection_M_N_l67_67674

def M (x : ℝ) : Prop := -2 < x ∧ x < 2
def N (x : ℝ) : Prop := |x - 1| ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x ∧ N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l67_67674


namespace remainder_of_product_mod_5_l67_67497

theorem remainder_of_product_mod_5 : 
  (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := by
  sorry

end remainder_of_product_mod_5_l67_67497


namespace integer_solutions_to_system_l67_67140

theorem integer_solutions_to_system (x y z : ℤ) (h1 : x + y + z = 2) (h2 : x^3 + y^3 + z^3 = -10) :
  (x = 3 ∧ y = 3 ∧ z = -4) ∨
  (x = 3 ∧ y = -4 ∧ z = 3) ∨
  (x = -4 ∧ y = 3 ∧ z = 3) :=
sorry

end integer_solutions_to_system_l67_67140


namespace smaller_angle_at_7_15_l67_67277

theorem smaller_angle_at_7_15 (h_angle : ℝ) (m_angle : ℝ) : 
  h_angle = 210 + 0.5 * 15 →
  m_angle = 90 →
  min (abs (h_angle - m_angle)) (360 - abs (h_angle - m_angle)) = 127.5 :=
  by
    intros h_eq m_eq
    rw [h_eq, m_eq]
    sorry

end smaller_angle_at_7_15_l67_67277


namespace masha_numbers_unique_l67_67400

def natural_numbers : Set ℕ := {n | n > 11}

theorem masha_numbers_unique (a b : ℕ) (ha : a ∈ natural_numbers) (hb : b ∈ natural_numbers) (hne : a ≠ b)
  (hs_equals : ∃ S, S = a + b)
  (sasha_initially_uncertain : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → ¬ (Sasha_can_determine_initially a b S))
  (masha_hint : ∃ (a_even : ℕ), a_even ∈ natural_numbers ∧ (a_even % 2 = 0) ∧ (a_even = a ∨ a_even = b))
  (sasha_then_confident : ∀ (a b : ℕ), a ∈ natural_numbers → b ∈ natural_numbers → a ≠ b → (a + b = S) → (a_even = a ∨ a_even = b) → Sasha_can_determine_confidently a b S) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) := by
  sorry

end masha_numbers_unique_l67_67400


namespace dimes_given_l67_67884

theorem dimes_given (initial_dimes final_dimes dimes_dad_gave : ℕ)
  (h1 : initial_dimes = 9)
  (h2 : final_dimes = 16)
  (h3 : final_dimes = initial_dimes + dimes_dad_gave) :
  dimes_dad_gave = 7 :=
by
  rw [h1, h2] at h3
  linarith

end dimes_given_l67_67884


namespace find_a_for_even_function_l67_67366

-- conditions
def f (a : ℝ) (x : ℝ) : ℝ := x^3 * (a * 2^x - 2^(-x))

-- definition of an even function
def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x = g (-x)

-- the proof problem statement
theorem find_a_for_even_function (a : ℝ) : is_even_function (f a) ↔ a = 1 :=
by
  sorry

end find_a_for_even_function_l67_67366


namespace career_preference_degrees_l67_67609

variable (M F : ℕ)
variable (h1 : M / F = 2 / 3)
variable (preferred_males : ℚ := M / 4)
variable (preferred_females : ℚ := F / 2)
variable (total_students : ℚ := M + F)
variable (preferred_career_students : ℚ := preferred_males + preferred_females)
variable (career_fraction : ℚ := preferred_career_students / total_students)
variable (degrees : ℚ := 360 * career_fraction)

theorem career_preference_degrees :
  degrees = 144 :=
sorry

end career_preference_degrees_l67_67609


namespace hamiltonian_path_exists_l67_67850

open Finset

-- Define a Hamiltonian path
def is_hamiltonian_path {V : Type*} (G : SimpleGraph V) (p : List V) : Prop :=
  p.nodup ∧ ∀ v ∈ p.to_finset, G.degree v > 0 ∧ p.head? = some v ∧ ∀ u ∈ G.neighbor_set v, List.last p = some u ∧ u ∉ p.tail

noncomputable def example_graph : SimpleGraph (Fin n) := sorry -- Define the example graph with 20 vertices and a degree of at least 10 for each vertex.

theorem hamiltonian_path_exists (G : SimpleGraph (Fin 20)) 
  (h1 : ∀ v : Fin 20, G.degree v ≥ 10) : 
  ∃ p : List (Fin 20), is_hamiltonian_path G p :=
begin
  sorry -- The proof itself is omitted as per the instructions.
end

end hamiltonian_path_exists_l67_67850


namespace maximum_distance_proof_l67_67227

noncomputable def maximum_distance_from_point_on_circle_to_line : ℝ :=
  4 * Real.sqrt 2 + 2

theorem maximum_distance_proof :
  ∀ (ρ θ : ℝ), ρ^2 + 2*ρ*Real.cos θ - 3 = 0 → ∃ θ, ρ*Real.cos θ + ρ*Real.sin θ - 7 = 0 → 
  ∃ d, d = 4 * Real.sqrt 2 + 2 := 
by
  intros
  use maximum_distance_from_point_on_circle_to_line
  sorry

end maximum_distance_proof_l67_67227


namespace fraction_meaningful_iff_l67_67839

theorem fraction_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_meaningful_iff_l67_67839


namespace probability_of_rolling_number_less_than_5_is_correct_l67_67748

noncomputable def probability_of_rolling_number_less_than_5 : ℚ :=
  let total_outcomes := 8
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_of_rolling_number_less_than_5_is_correct :
  probability_of_rolling_number_less_than_5 = 1 / 2 := by
  sorry

end probability_of_rolling_number_less_than_5_is_correct_l67_67748


namespace f_f_n_plus_n_eq_n_plus_1_l67_67886

-- Define the function f : ℕ+ → ℕ+ satisfying the given condition
axiom f : ℕ+ → ℕ+

-- Define that for all positive integers n, f satisfies the condition f(f(n)) + f(n+1) = n + 2
axiom f_condition : ∀ n : ℕ+, f (f n) + f (n + 1) = n + 2

-- State that we want to prove that f(f(n) + n) = n + 1 for all positive integers n
theorem f_f_n_plus_n_eq_n_plus_1 : ∀ n : ℕ+, f (f n + n) = n + 1 := 
by sorry

end f_f_n_plus_n_eq_n_plus_1_l67_67886


namespace complement_A_inter_B_range_of_a_l67_67371

open Set

-- Define sets A and B based on the conditions
def A : Set ℝ := {x | -4 ≤ x - 6 ∧ x - 6 ≤ 0}
def B : Set ℝ := {x | 2 * x - 6 ≥ 3 - x}

-- Define set C based on the conditions
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Problem 1: Prove the complement of (A ∩ B) in ℝ is the set of x where (x < 3 or x > 6)
theorem complement_A_inter_B :
  compl (A ∩ B) = {x | x < 3} ∪ {x | x > 6} :=
sorry

-- Problem 2: Prove that A ∩ C = A implies a ∈ [6, ∞)
theorem range_of_a {a : ℝ} (hC : A ∩ C a = A) :
  6 ≤ a :=
sorry

end complement_A_inter_B_range_of_a_l67_67371


namespace root_quadratic_l67_67214

theorem root_quadratic (m : ℝ) (h : m^2 - 2*m - 1 = 0) : m^2 + 1/m^2 = 6 :=
sorry

end root_quadratic_l67_67214


namespace max_rectangle_area_with_prime_dimension_l67_67944

theorem max_rectangle_area_with_prime_dimension :
  ∃ (l w : ℕ), 2 * (l + w) = 120 ∧ (Prime l ∨ Prime w) ∧ l * w = 899 :=
by
  sorry

end max_rectangle_area_with_prime_dimension_l67_67944


namespace circle_area_radius_8_l67_67734

variable (r : ℝ) (π : ℝ)

theorem circle_area_radius_8 : r = 8 → (π * r^2) = 64 * π :=
by
  sorry

end circle_area_radius_8_l67_67734


namespace probability_digit_9_in_3_over_11_is_zero_l67_67564

-- Define the repeating block of the fraction 3/11
def repeating_block_3_over_11 : List ℕ := [2, 7]

-- Define the function to count the occurrences of a digit in a list
def count_occurrences (digit : ℕ) (lst : List ℕ) : ℕ :=
  lst.count digit

-- Define the probability function
def probability_digit_9_in_3_over_11 : ℚ :=
  (count_occurrences 9 repeating_block_3_over_11) / repeating_block_3_over_11.length

-- Theorem statement
theorem probability_digit_9_in_3_over_11_is_zero : 
  probability_digit_9_in_3_over_11 = 0 := 
by 
  sorry

end probability_digit_9_in_3_over_11_is_zero_l67_67564


namespace max_tiles_on_floor_l67_67456

theorem max_tiles_on_floor
  (tile_w tile_h floor_w floor_h : ℕ)
  (h_tile_w : tile_w = 25)
  (h_tile_h : tile_h = 65)
  (h_floor_w : floor_w = 150)
  (h_floor_h : floor_h = 390) :
  max ((floor_h / tile_h) * (floor_w / tile_w))
      ((floor_h / tile_w) * (floor_w / tile_h)) = 36 :=
by
  -- Given conditions and calculations will be proved in the proof.
  sorry

end max_tiles_on_floor_l67_67456


namespace quadratic_equality_l67_67821

theorem quadratic_equality (a_2 : ℝ) (a_1 : ℝ) (a_0 : ℝ) (r : ℝ) (s : ℝ) (x : ℝ)
  (h₁ : a_2 ≠ 0)
  (h₂ : a_0 ≠ 0)
  (h₃ : a_2 * r^2 + a_1 * r + a_0 = 0)
  (h₄ : a_2 * s^2 + a_1 * s + a_0 = 0) :
  a_0 + a_1 * x + a_2 * x^2 = a_0 * (1 - x / r) * (1 - x / s) :=
by
  sorry

end quadratic_equality_l67_67821


namespace speed_of_first_train_l67_67275

-- Define the problem conditions
def distance_between_stations : ℝ := 20
def speed_of_second_train : ℝ := 25
def meet_time : ℝ := 8
def start_time_first_train : ℝ := 7
def start_time_second_train : ℝ := 8
def travel_time_first_train : ℝ := meet_time - start_time_first_train

-- The actual proof statement in Lean
theorem speed_of_first_train : ∀ (v : ℝ),
  v * travel_time_first_train = distance_between_stations → v = 20 :=
by
  intro v
  intro h
  sorry

end speed_of_first_train_l67_67275
