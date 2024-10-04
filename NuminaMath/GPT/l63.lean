import Mathlib

namespace proof_x_minus_y_squared_l63_63974

-- Define the variables x and y
variables (x y : ℝ)

-- Assume the given conditions
def cond1 : (x + y)^2 = 64 := sorry
def cond2 : x * y = 12 := sorry

-- Formulate the main goal to prove
theorem proof_x_minus_y_squared : (x - y)^2 = 16 :=
by
  have hx_add_y_sq : (x + y)^2 = 64 := cond1
  have hxy : x * y = 12 := cond2
  -- Use the identities and the given conditions to prove the statement
  sorry

end proof_x_minus_y_squared_l63_63974


namespace Jeff_Jogging_Extra_Friday_l63_63088

theorem Jeff_Jogging_Extra_Friday :
  let planned_daily_minutes := 60
  let days_in_week := 5
  let planned_weekly_minutes := days_in_week * planned_daily_minutes
  let thursday_cut_short := 20
  let actual_weekly_minutes := 290
  let thursday_run := planned_daily_minutes - thursday_cut_short
  let other_four_days_minutes := actual_weekly_minutes - thursday_run
  let mondays_to_wednesdays_run := 3 * planned_daily_minutes
  let friday_run := other_four_days_minutes - mondays_to_wednesdays_run
  let extra_run_on_friday := friday_run - planned_daily_minutes
  extra_run_on_friday = 10 := by trivial

end Jeff_Jogging_Extra_Friday_l63_63088


namespace ratio_rate_down_to_up_l63_63325

theorem ratio_rate_down_to_up 
  (rate_up : ℝ) (time_up : ℝ) (distance_down : ℝ) (time_down_eq_time_up : time_down = time_up) :
  (time_up = 2) → 
  (rate_up = 3) →
  (distance_down = 9) → 
  (time_down = time_up) →
  (distance_down / time_down / rate_up = 1.5) :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_rate_down_to_up_l63_63325


namespace greatest_integer_solution_l63_63448

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 40 ≤ 0) : n ≤ 8 :=
sorry

end greatest_integer_solution_l63_63448


namespace bankers_discount_is_270_l63_63860

noncomputable def bank_discount (BG r t : ℝ) : ℝ :=
  let BD := 540 / 2
  BD

theorem bankers_discount_is_270 (BG r t : ℝ) (h_BG : BG = 270) (h_r : r = 0.12) (h_t : t = 5) :
  bank_discount BG r t = 270 :=
by
  sorry

end bankers_discount_is_270_l63_63860


namespace find_a_l63_63189

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 2 ↔ |a * x + 2| < 6) → a = -4 :=
by
  intro h
  sorry

end find_a_l63_63189


namespace sequence_solution_l63_63827

theorem sequence_solution (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, 2 / a n = 1 / a (n + 1) + 1 / a (n - 1)) :
  ∀ n, a n = 2 / n :=
by
  sorry

end sequence_solution_l63_63827


namespace quadratic_roots_n_value_l63_63265

theorem quadratic_roots_n_value :
  ∃ m p : ℕ, ∃ (n : ℕ) (h : Nat.gcd m p = 1),
  (∃ x1 x2 : ℝ, (3 * x1^2 - 6 * x1 - 9 = 0 ∧ 3 * x2^2 - 6 * x2 - 9 = 0) ∧
   ∀ x, 3 * x^2 - 6 * x - 9 = 0 → x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :=
by
  use [1, 1, 144, Nat.gcd_one_right 1]
  sorry

end quadratic_roots_n_value_l63_63265


namespace fraction_of_field_planted_l63_63498

theorem fraction_of_field_planted (AB AC : ℕ) (x : ℕ) (shortest_dist : ℕ) (hypotenuse : ℕ)
  (S : ℕ) (total_area : ℕ) (planted_area : ℕ) :
  AB = 5 ∧ AC = 12 ∧ hypotenuse = 13 ∧ shortest_dist = 2 ∧ x * x = S ∧ 
  total_area = 30 ∧ planted_area = total_area - S →
  (planted_area / total_area : ℚ) = 2951 / 3000 :=
by
  sorry

end fraction_of_field_planted_l63_63498


namespace find_x_l63_63007

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 682.50) : x = 210 :=
by
  sorry

end find_x_l63_63007


namespace isosceles_triangle_leg_l63_63673

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop := (a = b ∨ a = c ∨ b = c)

theorem isosceles_triangle_leg
  (a b c : ℝ)
  (h1 : is_isosceles_triangle a b c)
  (h2 : a + b + c = 18)
  (h3 : a = 8 ∨ b = 8 ∨ c = 8) :
  (a = 5 ∨ b = 5 ∨ c = 5 ∨ a = 8 ∨ b = 8 ∨ c = 8) :=
sorry

end isosceles_triangle_leg_l63_63673


namespace total_flour_required_l63_63713

-- Definitions specified based on the given conditions
def flour_already_put_in : ℕ := 10
def flour_needed : ℕ := 2

-- Lean 4 statement to prove the total amount of flour required by the recipe
theorem total_flour_required : (flour_already_put_in + flour_needed) = 12 :=
by
  sorry

end total_flour_required_l63_63713


namespace arithmetic_geometric_mean_inequality_l63_63428

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1 / 3) :=
sorry

end arithmetic_geometric_mean_inequality_l63_63428


namespace simplest_fraction_is_one_l63_63282

theorem simplest_fraction_is_one :
  ∃ m : ℕ, 
  (∃ k : ℕ, 45 * m = k^2) ∧ 
  (∃ n : ℕ, 56 * m = n^3) → 
  45 * m / 56 * m = 1 := by
  sorry

end simplest_fraction_is_one_l63_63282


namespace a_5_eq_16_S_8_eq_255_l63_63658

open Nat

-- Definitions from the conditions
def a : ℕ → ℕ
| 0     => 1
| (n+1) => 2 * a n

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Proof problem statements
theorem a_5_eq_16 : a 4 = 16 := sorry

theorem S_8_eq_255 : S 8 = 255 := sorry

end a_5_eq_16_S_8_eq_255_l63_63658


namespace all_statements_correct_l63_63494

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem all_statements_correct (b : ℝ) (h1 : b > 0) (h2 : b ≠ 1) :
  (f b b = 1) ∧
  (f b 1 = 0) ∧
  (¬(0 ∈ Set.range (f b))) ∧
  (∀ x, 0 < x ∧ x < b → f b x < 1) ∧
  (∀ x, x > b → f b x > 1) := by
  unfold f
  sorry

end all_statements_correct_l63_63494


namespace slices_per_pack_l63_63089

theorem slices_per_pack (sandwiches : ℕ) (slices_per_sandwich : ℕ) (packs_of_bread : ℕ) (total_slices : ℕ) 
  (h1 : sandwiches = 8) (h2 : slices_per_sandwich = 2) (h3 : packs_of_bread = 4) : 
  total_slices = 4 :=
by
  sorry

end slices_per_pack_l63_63089


namespace find_integer_pairs_l63_63049

theorem find_integer_pairs (a b : ℤ) : 
  (∃ d : ℤ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → d ∣ (a^n + b^n + 1)) → 
  (∃ k₁ k₂ : ℤ, ((a = 2 * k₁) ∧ (b = 2 * k₂ + 1)) ∨ ((a = 3 * k₁ + 1) ∧ (b = 3 * k₂ + 1))) :=
by
  sorry

end find_integer_pairs_l63_63049


namespace cost_of_candy_bar_l63_63351

def initial_amount : ℝ := 3.0
def remaining_amount : ℝ := 2.0

theorem cost_of_candy_bar :
  initial_amount - remaining_amount = 1.0 :=
by
  sorry

end cost_of_candy_bar_l63_63351


namespace vectors_parallel_x_value_l63_63397

theorem vectors_parallel_x_value :
  ∀ (x : ℝ), (∀ a b : ℝ × ℝ, a = (2, 1) → b = (4, x+1) → (a.1 / b.1 = a.2 / b.2)) → x = 1 :=
by
  intros x h
  sorry

end vectors_parallel_x_value_l63_63397


namespace proposition_neg_p_and_q_false_l63_63077

theorem proposition_neg_p_and_q_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end proposition_neg_p_and_q_false_l63_63077


namespace rectangle_to_rhombus_l63_63171

def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ D.2 = C.2 ∧ C.1 = B.1 ∧ B.2 = A.2

def is_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) ≠ 0

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

theorem rectangle_to_rhombus (A B C D : ℝ × ℝ) (h1 : is_rectangle A B C D) :
  ∃ X Y Z W : ℝ × ℝ, is_triangle A B C ∧ is_triangle A D C ∧ is_rhombus X Y Z W :=
by
  sorry

end rectangle_to_rhombus_l63_63171


namespace prime_odd_sum_l63_63803

theorem prime_odd_sum (x y : ℕ) (h_prime : Prime x) (h_odd : y % 2 = 1) (h_eq : x^2 + y = 2005) : x + y = 2003 :=
by
  sorry

end prime_odd_sum_l63_63803


namespace max_sin_cos_value_l63_63811

open Real

noncomputable def max_value (α β γ : ℝ) : ℝ :=
  sin (α - γ) + cos (β - γ)

theorem max_sin_cos_value (α β γ : ℝ) (hα : 0 ≤ α ∧ α ≤ 2 * π) (hβ : 0 ≤ β ∧ β ≤ 2 * π)
  (hγ : 0 ≤ γ ∧ γ ≤ 2 * π) (h : sin (α - β) = 1 / 4) :
  max_value α β γ ≤ sqrt 10 / 2 :=
sorry

end max_sin_cos_value_l63_63811


namespace arithmetic_progression_term_l63_63433

variable (n r : ℕ)

-- Given the sum of the first n terms of an arithmetic progression is S_n = 3n + 4n^2
def S (n : ℕ) : ℕ := 3 * n + 4 * n^2

-- Prove that the r-th term of the sequence is 8r - 1
theorem arithmetic_progression_term :
  (S r) - (S (r - 1)) = 8 * r - 1 :=
by
  sorry

end arithmetic_progression_term_l63_63433


namespace conversion_proofs_l63_63145

-- Define the necessary constants for unit conversion
def cm_to_dm2 (cm2: ℚ) : ℚ := cm2 / 100
def m3_to_dm3 (m3: ℚ) : ℚ := m3 * 1000
def dm3_to_liters (dm3: ℚ) : ℚ := dm3
def liters_to_ml (liters: ℚ) : ℚ := liters * 1000

theorem conversion_proofs :
  (cm_to_dm2 628 = 6.28) ∧
  (m3_to_dm3 4.5 = 4500) ∧
  (dm3_to_liters 3.6 = 3.6) ∧
  (liters_to_ml 0.6 = 600) :=
by
  sorry

end conversion_proofs_l63_63145


namespace f_of_3_eq_11_l63_63063

theorem f_of_3_eq_11 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1 / x) = x^2 + 1 / x^2) : f 3 = 11 :=
by
  sorry

end f_of_3_eq_11_l63_63063


namespace max_of_differentiable_function_l63_63149

variable {a b : ℝ}
variable {f : ℝ → ℝ} (hf : ∀ x ∈ set.Icc a b, differentiable ℝ f)

theorem max_of_differentiable_function (hf : ∀ x ∈ set.Icc a b, differentiable ℝ f) :
  ∃ c ∈ set.Icc a b, ∀ x ∈ set.Icc a b, f c ≥ f x :=
begin
  let S := {x ∈ set.Icc a b | ∀ y ∈ set.Icc a b, f x ≥ f y},
  have hS : S.nonempty,
  -- sorry to skip the proof
  sorry,
  have hmax : ∃ c ∈ S, ∀ x ∈ S, f c ≥ f x,
  from exists_maximum_of_compact -- sorry to skip the proof,
  sorry
  use [c, hc, hc'],
  sorry
end

end max_of_differentiable_function_l63_63149


namespace square_difference_l63_63968

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l63_63968


namespace range_of_a_l63_63802

variable {a b c d : ℝ}

theorem range_of_a (h1 : a + b + c + d = 3) (h2 : a^2 + 2 * b^2 + 3 * c^2 + 6 * d^2 = 5) : 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l63_63802


namespace circle_equation_unique_circle_equation_l63_63527

-- Definitions based on conditions
def radius (r : ℝ) : Prop := r = 1
def center_in_first_quadrant (a b : ℝ) : Prop := a > 0 ∧ b > 0
def tangent_to_line (a b : ℝ) : Prop := (|4 * a - 3 * b| / Real.sqrt (4^2 + (-3)^2)) = 1
def tangent_to_x_axis (b : ℝ) : Prop := b = 1

-- Main theorem statement
theorem circle_equation_unique 
  {a b : ℝ} 
  (h_rad : radius 1) 
  (h_center : center_in_first_quadrant a b) 
  (h_tan_line : tangent_to_line a b) 
  (h_tan_x : tangent_to_x_axis b) :
  (a = 2 ∧ b = 1) :=
sorry

-- Final circle equation
theorem circle_equation : 
  (∀ a b : ℝ, ((a = 2) ∧ (b = 1)) → (x - a)^2 + (y - b)^2 = 1) :=
sorry

end circle_equation_unique_circle_equation_l63_63527


namespace combined_market_value_two_years_later_l63_63918

theorem combined_market_value_two_years_later:
  let P_A := 8000
  let P_B := 10000
  let P_C := 12000
  let r_A := 0.20
  let r_B := 0.15
  let r_C := 0.10

  let V_A_year_1 := P_A - r_A * P_A
  let V_A_year_2 := V_A_year_1 - r_A * P_A
  let V_B_year_1 := P_B - r_B * P_B
  let V_B_year_2 := V_B_year_1 - r_B * P_B
  let V_C_year_1 := P_C - r_C * P_C
  let V_C_year_2 := V_C_year_1 - r_C * P_C

  V_A_year_2 + V_B_year_2 + V_C_year_2 = 21400 :=
by
  sorry

end combined_market_value_two_years_later_l63_63918


namespace candy_per_bag_correct_l63_63053

def total_candy : ℕ := 648
def sister_candy : ℕ := 48
def friends : ℕ := 3
def bags : ℕ := 8

def remaining_candy (total candy_kept : ℕ) : ℕ := total - candy_kept
def candy_per_person (remaining people : ℕ) : ℕ := remaining / people
def candy_per_bag (per_person bags : ℕ) : ℕ := per_person / bags

theorem candy_per_bag_correct :
  candy_per_bag (candy_per_person (remaining_candy total_candy sister_candy) (friends + 1)) bags = 18 :=
by
  sorry

end candy_per_bag_correct_l63_63053


namespace complex_problem_l63_63381

open Complex

theorem complex_problem
  (α θ β : ℝ)
  (h : exp (i * (α + θ)) + exp (i * (β + θ)) = 1 / 3 + (4 / 9) * i) :
  exp (-i * (α + θ)) + exp (-i * (β + θ)) = 1 / 3 - (4 / 9) * i :=
by
  sorry

end complex_problem_l63_63381


namespace value_of_x_l63_63137

theorem value_of_x (x : ℝ) (h : (10 - x)^2 = x^2 + 4) : x = 24 / 5 :=
by
  sorry

end value_of_x_l63_63137


namespace andrew_age_l63_63905

theorem andrew_age 
  (g a : ℚ)
  (h1: g = 16 * a)
  (h2: g - 20 - (a - 20) = 45) : 
 a = 17 / 3 := by 
  sorry

end andrew_age_l63_63905


namespace quadratic_conditions_l63_63557

open Polynomial

noncomputable def exampleQuadratic (x : ℝ) : ℝ :=
-2 * x^2 + 12 * x - 10

theorem quadratic_conditions :
  (exampleQuadratic 1 = 0) ∧ (exampleQuadratic 5 = 0) ∧ (exampleQuadratic 3 = 8) :=
by
  sorry

end quadratic_conditions_l63_63557


namespace incorrect_statement_A_l63_63714

theorem incorrect_statement_A (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a > b) :
  ¬ (a - a^2 > b - b^2) := sorry

end incorrect_statement_A_l63_63714


namespace rectangular_solid_edges_sum_l63_63022

theorem rectangular_solid_edges_sum 
  (a b c : ℝ) 
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b^2 = a * c) : 
  4 * (a + b + c) = 32 := 
  sorry

end rectangular_solid_edges_sum_l63_63022


namespace problem_1_problem_2_problem_3_l63_63160

-- First proof statement
theorem problem_1 : 2017^2 - 2016 * 2018 = 1 :=
by
  sorry

-- Definitions for the second problem
variables {a b : ℤ}

-- Second proof statement
theorem problem_2 (h1 : a + b = 7) (h2 : a * b = -1) : (a + b)^2 = 49 :=
by
  sorry

-- Third proof statement (part of the second problem)
theorem problem_3 (h1 : a + b = 7) (h2 : a * b = -1) : a^2 - 3 * a * b + b^2 = 54 :=
by
  sorry

end problem_1_problem_2_problem_3_l63_63160


namespace matching_pair_probability_correct_l63_63252

-- Define the basic assumptions (conditions)
def black_pairs : Nat := 7
def brown_pairs : Nat := 4
def gray_pairs : Nat := 3
def red_pairs : Nat := 2

def total_pairs : Nat := black_pairs + brown_pairs + gray_pairs + red_pairs
def total_shoes : Nat := 2 * total_pairs

-- The probability calculation will be shown as the final proof requirement
def matching_color_probability : Rat :=  (14 * 7 + 8 * 4 + 6 * 3 + 4 * 2 : Int) / (32 * 31 : Int)

-- The target statement to be proven
theorem matching_pair_probability_correct :
  matching_color_probability = (39 / 248 : Rat) :=
by
  sorry

end matching_pair_probability_correct_l63_63252


namespace combined_mpg_l63_63719

-- Definitions based on the conditions
def ray_miles : ℕ := 150
def tom_miles : ℕ := 100
def ray_mpg : ℕ := 30
def tom_mpg : ℕ := 20

-- Theorem statement
theorem combined_mpg : (ray_miles + tom_miles) / ((ray_miles / ray_mpg) + (tom_miles / tom_mpg)) = 25 := by
  sorry

end combined_mpg_l63_63719


namespace opposite_numbers_power_l63_63525

theorem opposite_numbers_power (a b : ℝ) (h : a + b = 0) : (a + b) ^ 2023 = 0 :=
by 
  sorry

end opposite_numbers_power_l63_63525


namespace perfect_square_octal_last_digit_l63_63989

theorem perfect_square_octal_last_digit (a b c : ℕ) (n : ℕ) (h1 : a ≠ 0) (h2 : (abc:ℕ) = n^2) :
  c = 1 :=
sorry

end perfect_square_octal_last_digit_l63_63989


namespace unique_quotient_is_9742_l63_63983

theorem unique_quotient_is_9742 :
  ∃ (d4 d3 d2 d1 : ℕ),
    (d2 = d1 + 2) ∧
    (d4 = d3 + 2) ∧
    (0 ≤ d1 ∧ d1 ≤ 9) ∧
    (0 ≤ d2 ∧ d2 ≤ 9) ∧
    (0 ≤ d3 ∧ d3 ≤ 9) ∧
    (0 ≤ d4 ∧ d4 ≤ 9) ∧
    (d4 * 1000 + d3 * 100 + d2 * 10 + d1 = 9742) :=
by sorry

end unique_quotient_is_9742_l63_63983


namespace students_taking_history_l63_63221

-- Defining the conditions
def num_students (total_students history_students statistics_students both_students : ℕ) : Prop :=
  total_students = 89 ∧
  statistics_students = 32 ∧
  (history_students + statistics_students - both_students) = 59 ∧
  (history_students - both_students) = 27

-- The theorem stating that given the conditions, the number of students taking history is 54
theorem students_taking_history :
  ∃ history_students, ∃ statistics_students, ∃ both_students, 
  num_students 89 history_students statistics_students both_students ∧ history_students = 54 :=
by
  sorry

end students_taking_history_l63_63221


namespace caitlinAgeIsCorrect_l63_63340

-- Define Aunt Anna's age
def auntAnnAge : Nat := 48

-- Define the difference between Aunt Anna's age and 18
def ageDifference : Nat := auntAnnAge - 18

-- Define Brianna's age as twice the difference
def briannaAge : Nat := 2 * ageDifference

-- Define Caitlin's age as 6 years younger than Brianna
def caitlinAge : Nat := briannaAge - 6

-- Theorem to prove Caitlin's age
theorem caitlinAgeIsCorrect : caitlinAge = 54 := by
  sorry -- Proof to be filled in

end caitlinAgeIsCorrect_l63_63340


namespace least_number_to_produce_multiple_of_112_l63_63887

theorem least_number_to_produce_multiple_of_112 : ∃ k : ℕ, 72 * k = 112 * m → k = 14 :=
by
  sorry

end least_number_to_produce_multiple_of_112_l63_63887


namespace find_abc_value_l63_63216

open Real

/- Defining the conditions -/
variables (a b c : ℝ)
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
variables (h4 : a * (b + c) = 156) (h5 : b * (c + a) = 168) (h6 : c * (a + b) = 176)

/- Prove the value of abc -/
theorem find_abc_value :
  a * b * c = 754 :=
sorry

end find_abc_value_l63_63216


namespace john_coffees_per_day_l63_63693

theorem john_coffees_per_day (x : ℕ)
  (h1 : ∀ p : ℕ, p = 2)
  (h2 : ∀ p : ℕ, p = p + p / 2)
  (h3 : ∀ n : ℕ, n = x / 2)
  (h4 : ∀ d : ℕ, 2 * x - 3 * (x / 2) = 2) :
  x = 4 :=
by
  sorry

end john_coffees_per_day_l63_63693


namespace guppies_total_l63_63379

theorem guppies_total :
  let haylee := 3 * 12
  let jose := haylee / 2
  let charliz := jose / 3
  let nicolai := charliz * 4
  haylee + jose + charliz + nicolai = 84 :=
by
  sorry

end guppies_total_l63_63379


namespace binomial_expansion_l63_63285

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end binomial_expansion_l63_63285


namespace possible_values_of_a_l63_63371

def setA := {x : ℝ | x ≥ 3}
def setB (a : ℝ) := {x : ℝ | 2 * a - x > 1}
def complementB (a : ℝ) := {x : ℝ | x ≥ (2 * a - 1)}

theorem possible_values_of_a (a : ℝ) :
  (∀ x, x ∈ setA → x ∈ complementB a) ↔ (a = -2 ∨ a = 0 ∨ a = 2) :=
by
  sorry

end possible_values_of_a_l63_63371


namespace other_tables_have_3_legs_l63_63010

-- Define the given conditions
variables (total_tables four_legged_tables : ℕ)
variables (total_legs legs_four_legged_tables : ℕ)

-- State the conditions as Lean definitions
def dealer_conditions :=
  total_tables = 36 ∧
  four_legged_tables = 16 ∧
  total_legs = 124 ∧
  legs_four_legged_tables = 4 * four_legged_tables

-- Main theorem to prove the number of legs on the other tables
theorem other_tables_have_3_legs (cond : dealer_conditions total_tables four_legged_tables total_legs legs_four_legged_tables) :
  let other_tables := total_tables - four_legged_tables in
  let other_legs := total_legs - legs_four_legged_tables in
  other_tables > 0 →
  other_legs % other_tables = 0 →
  other_legs / other_tables = 3 :=
sorry

end other_tables_have_3_legs_l63_63010


namespace gym_monthly_cost_l63_63226

theorem gym_monthly_cost (down_payment total_cost total_months : ℕ) (h_down_payment : down_payment = 50) (h_total_cost : total_cost = 482) (h_total_months : total_months = 36) : 
  (total_cost - down_payment) / total_months = 12 := by 
  sorry

end gym_monthly_cost_l63_63226


namespace average_price_per_book_l63_63249

theorem average_price_per_book
  (spent1 spent2 spent3 spent4 : ℝ) (books1 books2 books3 books4 : ℕ)
  (h1 : spent1 = 1080) (h2 : spent2 = 840) (h3 : spent3 = 765) (h4 : spent4 = 630)
  (hb1 : books1 = 65) (hb2 : books2 = 55) (hb3 : books3 = 45) (hb4 : books4 = 35) :
  (spent1 + spent2 + spent3 + spent4) / (books1 + books2 + books3 + books4) = 16.575 :=
by {
  sorry
}

end average_price_per_book_l63_63249


namespace cost_of_cd_l63_63549

theorem cost_of_cd 
  (cost_film : ℕ) (cost_book : ℕ) (total_spent : ℕ) (num_cds : ℕ) (total_cost_films : ℕ)
  (total_cost_books : ℕ) (cost_cd : ℕ) : 
  cost_film = 5 → cost_book = 4 → total_spent = 79 →
  total_cost_films = 9 * cost_film → total_cost_books = 4 * cost_book →
  total_spent = total_cost_films + total_cost_books + num_cds * cost_cd →
  num_cds = 6 →
  cost_cd = 3 := 
by {
  -- proof would go here
  sorry
}

end cost_of_cd_l63_63549


namespace remainder_b96_div_50_l63_63998

theorem remainder_b96_div_50 (b : ℕ → ℕ) (h : ∀ n, b n = 7^n + 9^n) : b 96 % 50 = 2 :=
by
  -- The proof is omitted.
  sorry

end remainder_b96_div_50_l63_63998


namespace factorial_fraction_eq_seven_l63_63036

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end factorial_fraction_eq_seven_l63_63036


namespace calculate_expression_l63_63627

theorem calculate_expression : (40 * 1505 - 20 * 1505) / 5 = 6020 := by
  sorry

end calculate_expression_l63_63627


namespace dot_product_value_l63_63664

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (3, 1)

theorem dot_product_value :
  vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 :=
by
  -- Proof goes here
  sorry

end dot_product_value_l63_63664


namespace need_to_sell_more_rolls_l63_63244

variable (goal sold_grandmother sold_uncle_1 sold_uncle_additional sold_neighbor_1 returned_neighbor sold_mothers_friend sold_cousin_1 sold_cousin_additional : ℕ)

theorem need_to_sell_more_rolls
  (h_goal : goal = 100)
  (h_sold_grandmother : sold_grandmother = 5)
  (h_sold_uncle_1 : sold_uncle_1 = 12)
  (h_sold_uncle_additional : sold_uncle_additional = 10)
  (h_sold_neighbor_1 : sold_neighbor_1 = 8)
  (h_returned_neighbor : returned_neighbor = 4)
  (h_sold_mothers_friend : sold_mothers_friend = 25)
  (h_sold_cousin_1 : sold_cousin_1 = 3)
  (h_sold_cousin_additional : sold_cousin_additional = 5) :
  goal - (sold_grandmother + (sold_uncle_1 + sold_uncle_additional) + (sold_neighbor_1 - returned_neighbor) + sold_mothers_friend + (sold_cousin_1 + sold_cousin_additional)) = 36 := by
  sorry

end need_to_sell_more_rolls_l63_63244


namespace non_officers_count_l63_63886

theorem non_officers_count 
    (avg_salary_employees : ℝ) 
    (avg_salary_officers : ℝ) 
    (avg_salary_non_officers : ℝ) 
    (num_officers : ℕ) : 
    avg_salary_employees = 120 ∧ avg_salary_officers = 470 ∧ avg_salary_non_officers = 110 ∧ num_officers = 15 → 
    ∃ N : ℕ, N = 525 ∧ 
    (num_officers * avg_salary_officers + N * avg_salary_non_officers) / (num_officers + N) = avg_salary_employees := 
by 
    sorry

end non_officers_count_l63_63886


namespace equivalent_proof_problem_l63_63293

variables {a b c d e : ℚ}

theorem equivalent_proof_problem
  (h1 : 3 * a + 4 * b + 6 * c + 8 * d + 10 * e = 55)
  (h2 : 4 * (d + c + e) = b)
  (h3 : 4 * b + 2 * c = a)
  (h4 : c - 2 = d)
  (h5 : d + 1 = e) : 
  a * b * c * d * e = -1912397372 / 78364164096 := 
sorry

end equivalent_proof_problem_l63_63293


namespace factor_correct_l63_63048

theorem factor_correct (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end factor_correct_l63_63048


namespace symmedian_length_l63_63938

theorem symmedian_length (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ AS : ℝ, AS = (b * c^2 / (b^2 + c^2)) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2)) :=
sorry

end symmedian_length_l63_63938


namespace problem_1_problem_2_l63_63373

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0
def line2 (x y : ℝ) : Prop := 2 * x + 3 * y - 8 = 0
def line3 (x y : ℝ) : Prop := 3 * x + 2 * y + 5 = 0
def point_M : ℝ × ℝ := (1, 2)
def point_P : ℝ × ℝ := (3, 1)

def line_l1 (x y : ℝ) : Prop := x + 2 * y - 5 = 0
def line_l2 (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

theorem problem_1 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) → 
  (line_l1 point_P.1 point_P.2) ∧ (line_l1 point_M.1 point_M.2) :=
by 
  sorry

theorem problem_2 :
  (line1 point_M.1 point_M.2) ∧ (line2 point_M.1 point_M.2) →
  (∀ (x y : ℝ), line_l2 x y ↔ line3 x y) :=
by
  sorry

end problem_1_problem_2_l63_63373


namespace ernie_can_make_circles_l63_63335

theorem ernie_can_make_circles :
  ∀ (boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle : ℕ),
  Ali_circles = 5 →
  Ali_boxes_per_circle = 8 →
  Ernie_boxes_per_circle = 10 →
  boxes_initial = 80 →
  ((boxes_initial - Ali_circles * Ali_boxes_per_circle) / Ernie_boxes_per_circle) = 4 :=
by
  intros boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle 
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end ernie_can_make_circles_l63_63335


namespace original_cone_volume_l63_63323

theorem original_cone_volume
  (H R h r : ℝ)
  (Vcylinder : ℝ) (Vfrustum : ℝ)
  (cylinder_volume : Vcylinder = π * r^2 * h)
  (frustum_volume : Vfrustum = (1 / 3) * π * (R^2 + R * r + r^2) * (H - h))
  (Vcylinder_value : Vcylinder = 9)
  (Vfrustum_value : Vfrustum = 63) :
  (1 / 3) * π * R^2 * H = 64 :=
by
  sorry

end original_cone_volume_l63_63323


namespace antipov_inequality_l63_63057

theorem antipov_inequality (a b c : ℕ) 
  (h1 : ¬ (a ∣ b ∨ b ∣ a ∨ a ∣ c ∨ c ∣ a ∨ b ∣ c ∨ c ∣ b)) 
  (h2 : (ab + 1) ∣ (abc + 1)) : c ≥ b :=
sorry

end antipov_inequality_l63_63057


namespace sequence_monotonic_and_bounded_l63_63481

theorem sequence_monotonic_and_bounded :
  ∀ (a : ℕ → ℝ), (a 1 = 1 / 2) → (∀ n, a (n + 1) = 1 / 2 + (a n)^2 / 2) →
    (∀ n, a n < 2) ∧ (∀ n, a n < a (n + 1)) :=
by
  sorry

end sequence_monotonic_and_bounded_l63_63481


namespace max_value_of_f_value_of_f_given_tan_half_alpha_l63_63206

noncomputable def f (x : ℝ) := 2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * (Real.sin x)

theorem max_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ 3) ∧ (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 3 ∧ f x = 3) :=
sorry

theorem value_of_f_given_tan_half_alpha (α : ℝ) (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end max_value_of_f_value_of_f_given_tan_half_alpha_l63_63206


namespace total_fruits_picked_l63_63796

theorem total_fruits_picked (g_oranges g_apples a_oranges a_apples o_oranges o_apples : ℕ) :
  g_oranges = 45 →
  g_apples = a_apples + 5 →
  a_oranges = g_oranges - 18 →
  a_apples = 15 →
  o_oranges = 6 * 3 →
  o_apples = 6 * 2 →
  g_oranges + g_apples + a_oranges + a_apples + o_oranges + o_apples = 137 :=
by
  intros
  sorry

end total_fruits_picked_l63_63796


namespace elizabeth_net_profit_is_50_l63_63934

variables (ingredient_cost : ℕ) (bags_made : ℕ) (price_per_bag : ℕ)
variables (bags_sold_initial : ℕ) (discounted_bags : ℕ) (discounted_price : ℕ)
variable total_cost : ℕ := bags_made * ingredient_cost
variable revenue_initial : ℕ := bags_sold_initial * price_per_bag
variable revenue_discounted : ℕ := discounted_bags * discounted_price
variable total_revenue : ℕ := revenue_initial + revenue_discounted
variable net_profit : ℕ := total_revenue - total_cost

theorem elizabeth_net_profit_is_50 :
  ingredient_cost = 3 ∧ bags_made = 20 ∧ price_per_bag = 6 ∧ 
  bags_sold_initial = 15 ∧ discounted_bags = 5 ∧ discounted_price = 4 →
  net_profit = 50 :=
by 
  intros h, 
  cases h with h1 h2, 
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  sorry

end elizabeth_net_profit_is_50_l63_63934


namespace solution_l63_63278

variable (n : ℕ)

def y (x : ℝ) := Real.exp x

-- Coordinates of the quadrilateral vertices
def vertex1 := (n : ℝ , y n)
def vertex2 := (n+2 : ℝ , y (n+2))
def vertex3 := (n+4 : ℝ , y (n+4))
def vertex4 := (n+6 : ℝ , y (n+6))

def shoelace_area : ℝ :=
    0.5 * Real.abs (
      (y n) * (n+2) +
      (y (n+2)) * (n+4) +
      (y (n+4)) * (n+6) +
      (y (n+6)) * n -
      (y (n+2)) * n -
      (y (n+4)) * (n+2) -
      (y (n+6)) * (n+4) -
      (y n) * (n+6)
    )

theorem solution :
  (shoelace_area n = Real.exp 6 - Real.exp 2) → n = 2 :=
by
  sorry

end solution_l63_63278


namespace christmas_tree_problem_l63_63889

theorem christmas_tree_problem (b t : ℕ) (h1 : t = b + 1) (h2 : 2 * b = t - 1) : b = 3 ∧ t = 4 :=
by
  sorry

end christmas_tree_problem_l63_63889


namespace arithmetic_sequence_common_difference_l63_63951

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 2 = 1)
  (h2 : a 3 + a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l63_63951


namespace tuesday_pairs_of_boots_l63_63857

theorem tuesday_pairs_of_boots (S B : ℝ) (x : ℤ) 
  (h1 : 22 * S + 16 * B = 460)
  (h2 : 8 * S + x * B = 560)
  (h3 : B = S + 15) : 
  x = 24 :=
sorry

end tuesday_pairs_of_boots_l63_63857


namespace intersection_A_B_l63_63823

def A : Set ℤ := {-1, 1, 3, 5, 7}
def B : Set ℝ := { x | 2^x > 2 * Real.sqrt 2 }

theorem intersection_A_B :
  A ∩ { x : ℤ | x > 3 / 2 } = {3, 5, 7} :=
by
  sorry

end intersection_A_B_l63_63823


namespace domain_of_ratio_function_l63_63805

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := f (2 ^ x)

theorem domain_of_ratio_function (D : Set ℝ) (hD : D = Set.Icc 1 2):
  ∀ f : ℝ → ℝ, (∀ x, g x = f (2 ^ x)) →
  ∃ D' : Set ℝ, D' = {x | 2 ≤ x ∧ x ≤ 4} →
  ∀ y : ℝ, (2 ≤ y ∧ y ≤ 4) → ∃ x : ℝ, y = x + 1 ∧ x ≠ 1 → (1 < x ∧ x ≤ 3) :=
sorry

end domain_of_ratio_function_l63_63805


namespace evening_minivans_l63_63704

theorem evening_minivans (total_minivans afternoon_minivans : ℕ) (h_total : total_minivans = 5) 
(h_afternoon : afternoon_minivans = 4) : total_minivans - afternoon_minivans = 1 := 
by
  sorry

end evening_minivans_l63_63704


namespace relationship_t_s_l63_63368

theorem relationship_t_s (a b : ℝ) : 
  let t := a + 2 * b
  let s := a + b^2 + 1
  t <= s :=
by
  sorry

end relationship_t_s_l63_63368


namespace quadratic_inequality_solution_set_l63_63205

theorem quadratic_inequality_solution_set (a b : ℝ) :
  (∀ x : ℝ, (2 < x ∧ x < 3) → (ax^2 + 5*x + b > 0)) →
  ∃ x : ℝ, (-1/2 < x ∧ x < -1/3) :=
sorry

end quadratic_inequality_solution_set_l63_63205


namespace inequality_for_a_and_b_l63_63119

theorem inequality_for_a_and_b (a b : ℝ) : 
  (1 / 3 * a - b) ≤ 5 :=
sorry

end inequality_for_a_and_b_l63_63119


namespace Jake_has_more_peaches_than_Jill_l63_63086

variables (Jake Steven Jill : ℕ)
variable (h1 : Jake = Steven - 5)
variable (h2 : Steven = Jill + 18)
variable (h3 : Jill = 87)

theorem Jake_has_more_peaches_than_Jill (Jake Steven Jill : ℕ) (h1 : Jake = Steven - 5) (h2 : Steven = Jill + 18) (h3 : Jill = 87) :
  Jake - Jill = 13 :=
by
  sorry

end Jake_has_more_peaches_than_Jill_l63_63086


namespace intersection_of_A_and_B_l63_63659

open Set

variable {α : Type} [PartialOrder α]

noncomputable def A := { x : ℝ | -1 < x ∧ x < 1 }
noncomputable def B := { x : ℝ | 0 < x }

theorem intersection_of_A_and_B :
  A ∩ B = { x : ℝ | 0 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_l63_63659


namespace solve_for_y_l63_63052

theorem solve_for_y : ∃ y : ℝ, (2010 + y)^2 = y^2 ∧ y = -1005 :=
by
  sorry

end solve_for_y_l63_63052


namespace warehouse_painted_area_l63_63155

theorem warehouse_painted_area :
  let length := 8
  let width := 6
  let height := 3.5
  let door_width := 1
  let door_height := 2
  let front_back_area := 2 * (length * height)
  let left_right_area := 2 * (width * height)
  let total_wall_area := front_back_area + left_right_area
  let door_area := door_width * door_height
  let painted_area := total_wall_area - door_area
  painted_area = 96 :=
by
  -- Sorry to skip the actual proof steps
  sorry

end warehouse_painted_area_l63_63155


namespace negation_proposition_iff_l63_63862

-- Define propositions and their components
def P (x : ℝ) : Prop := x > 1
def Q (x : ℝ) : Prop := x^2 > 1

-- State the proof problem
theorem negation_proposition_iff (x : ℝ) : ¬ (P x → Q x) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by 
  sorry

end negation_proposition_iff_l63_63862


namespace average_tree_height_is_800_l63_63403

def first_tree_height : ℕ := 1000
def other_tree_height : ℕ := first_tree_height / 2
def last_tree_height : ℕ := first_tree_height + 200
def total_height : ℕ := first_tree_height + other_tree_height + other_tree_height + last_tree_height
def average_height : ℕ := total_height / 4

theorem average_tree_height_is_800 :
  average_height = 800 := by
  sorry

end average_tree_height_is_800_l63_63403


namespace factorial_expression_simplification_l63_63034

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l63_63034


namespace max_figures_in_grid_l63_63045

-- Definition of the grid size
def grid_size : ℕ := 9

-- Definition of the figure coverage
def figure_coverage : ℕ := 4

-- The total number of unit squares in the grid is 9 * 9 = 81
def total_unit_squares : ℕ := grid_size * grid_size

-- Each figure covers exactly 4 unit squares
def units_per_figure : ℕ := figure_coverage

-- The number of such 2x2 blocks that can be formed in 9x9 grid.
def maximal_figures_possible : ℕ := (grid_size / 2) * (grid_size / 2)

-- The main theorem to be proved
theorem max_figures_in_grid : 
  maximal_figures_possible = total_unit_squares / units_per_figure := by
  sorry

end max_figures_in_grid_l63_63045


namespace k_values_equation_satisfied_l63_63921

theorem k_values_equation_satisfied : 
  {k : ℕ | k > 0 ∧ ∃ r s : ℕ, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s} = {2, 3, 4, 8} :=
by
  sorry

end k_values_equation_satisfied_l63_63921


namespace product_of_bc_l63_63127

theorem product_of_bc
  (b c : Int)
  (h1 : ∀ r, r^2 - r - 1 = 0 → r^5 - b * r - c = 0) :
  b * c = 15 :=
by
  -- We start the proof assuming the conditions
  sorry

end product_of_bc_l63_63127


namespace minimum_value_expression_l63_63551

theorem minimum_value_expression (x y : ℝ) : 
  ∃ m : ℝ, ∀ x y : ℝ, 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ m ∧ m = 3 :=
sorry

end minimum_value_expression_l63_63551


namespace angela_action_figures_l63_63488

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end angela_action_figures_l63_63488


namespace kayla_total_items_l63_63699

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l63_63699


namespace distance_between_points_l63_63592

def point1 : ℝ × ℝ := (3.5, -2)
def point2 : ℝ × ℝ := (7.5, 5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 65 := by
  sorry

end distance_between_points_l63_63592


namespace area_of_triangle_ADE_l63_63986

noncomputable def triangle_area (A B C: ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_ADE (A B D E F : ℝ × ℝ) (h₁ : A.1 = 0 ∧ A.2 = 0) (h₂ : B.1 = 8 ∧ B.2 = 0)
  (h₃ : D.1 = 8 ∧ D.2= 8) (h₄ : E.1 = 4 * 3 / 5 ∧ E.2 = 0) 
  (h₅ : F.1 = 0 ∧ F.2 = 12) :
  triangle_area A D E = 288 / 25 := 
sorry

end area_of_triangle_ADE_l63_63986


namespace cricket_average_increase_l63_63008

-- Define the conditions as variables
variables (innings_initial : ℕ) (average_initial : ℕ) (runs_next_innings : ℕ)
variables (runs_increase : ℕ)

-- Given conditions
def conditions := (innings_initial = 13) ∧ (average_initial = 22) ∧ (runs_next_innings = 92)

-- Target: Calculate the desired increase in average (runs_increase)
theorem cricket_average_increase (h : conditions innings_initial average_initial runs_next_innings) :
  runs_increase = 5 :=
  sorry

end cricket_average_increase_l63_63008


namespace problem_solution_l63_63201

variable (x y z : ℝ)

theorem problem_solution
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + x * y = 8)
  (h2 : y + z + y * z = 15)
  (h3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 :=
sorry

end problem_solution_l63_63201


namespace table_arrangement_division_l63_63530

theorem table_arrangement_division (total_tables : ℕ) (rows : ℕ) (tables_per_row : ℕ) (tables_left_over : ℕ)
    (h1 : total_tables = 74) (h2 : rows = 8) (h3 : tables_per_row = total_tables / rows) (h4 : tables_left_over = total_tables % rows) :
    tables_per_row = 9 ∧ tables_left_over = 2 := by
  sorry

end table_arrangement_division_l63_63530


namespace ned_washed_shirts_l63_63100

theorem ned_washed_shirts (short_sleeve long_sleeve not_washed: ℕ) (h1: short_sleeve = 9) (h2: long_sleeve = 21) (h3: not_washed = 1) : 
    (short_sleeve + long_sleeve - not_washed = 29) :=
by
  sorry

end ned_washed_shirts_l63_63100


namespace marbles_remaining_l63_63495

def original_marbles : Nat := 64
def given_marbles : Nat := 14
def remaining_marbles : Nat := original_marbles - given_marbles

theorem marbles_remaining : remaining_marbles = 50 :=
  by
    sorry

end marbles_remaining_l63_63495


namespace find_numbers_l63_63297

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l63_63297


namespace no_unboxed_products_l63_63751

-- Definitions based on the conditions
def big_box_capacity : ℕ := 50
def small_box_capacity : ℕ := 40
def total_products : ℕ := 212

-- Theorem statement proving the least number of unboxed products
theorem no_unboxed_products (big_box_capacity small_box_capacity total_products : ℕ) : 
  (total_products - (total_products / big_box_capacity) * big_box_capacity) % small_box_capacity = 0 :=
by
  sorry

end no_unboxed_products_l63_63751


namespace find_k_l63_63079

theorem find_k (x y k : ℤ) (h1 : 2 * x - y = 5 * k + 6) (h2 : 4 * x + 7 * y = k) (h3 : x + y = 2023) : k = 2022 := 
by {
  sorry
}

end find_k_l63_63079


namespace kayla_total_items_l63_63700

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l63_63700


namespace simplify_expression_l63_63382

theorem simplify_expression (x y : ℝ) (h : x - 3 * y = 4) : (x - 3 * y) ^ 2 + 2 * x - 6 * y - 10 = 14 := by
  sorry

end simplify_expression_l63_63382


namespace minimum_value_analysis_l63_63571

theorem minimum_value_analysis
  (a : ℝ) (m n : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : 2 * m + n = 2)
  (h4 : m > 0)
  (h5 : n > 0) :
  (2 / m + 1 / n) ≥ 9 / 2 :=
sorry

end minimum_value_analysis_l63_63571


namespace contribution_of_eight_families_l63_63156

/-- Definition of the given conditions --/
def classroom := 200
def two_families := 2 * 20
def ten_families := 10 * 5
def missing_amount := 30

def total_raised (x : ℝ) : ℝ := two_families + ten_families + 8 * x

/-- The main theorem to prove the contribution of each of the eight families --/
theorem contribution_of_eight_families (x : ℝ) (h : total_raised x = classroom - missing_amount) : x = 10 := by
  sorry

end contribution_of_eight_families_l63_63156


namespace tan_beta_l63_63060

open Real

variable (α β : ℝ)

theorem tan_beta (h₁ : tan α = 1/3) (h₂ : tan (α + β) = 1/2) : tan β = 1/7 :=
by sorry

end tan_beta_l63_63060


namespace sum_of_roots_eq_eight_l63_63564

open Real

theorem sum_of_roots_eq_eight (f : ℝ → ℝ) 
  (h_symm : ∀ x, f(2 + x) = f(2 - x)) 
  (h_roots : set.countable {x | f x = 0}) 
  (h_distinct : (∃ (r1 r2 r3 r4 : ℝ), 
               r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ 
               r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 ∧
               f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0 ∧ f r4 = 0)) : 
  (∀ (r1 r2 r3 r4 : ℝ), 
       f r1 = 0 → f r2 = 0 → f r3 = 0 → f r4 = 0 → 
       r1 ≠ r2 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r3 ∧ r2 ≠ r4 ∧ r3 ≠ r4 → 
       r1 + r2 + r3 + r4 = 8) := 
by
  sorry

end sum_of_roots_eq_eight_l63_63564


namespace joan_seashells_correct_l63_63090

/-- Joan originally found 70 seashells -/
def joan_original_seashells : ℕ := 70

/-- Sam gave Joan 27 seashells -/
def seashells_given_by_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def joan_total_seashells : ℕ := joan_original_seashells + seashells_given_by_sam

theorem joan_seashells_correct : joan_total_seashells = 97 :=
by
  unfold joan_total_seashells
  unfold joan_original_seashells seashells_given_by_sam
  sorry

end joan_seashells_correct_l63_63090


namespace vec_mag_diff_eq_neg_one_l63_63665

variables (a b : ℝ × ℝ)

def vec_add_eq := a + b = (2, 3)

def vec_sub_eq := a - b = (-2, 1)

theorem vec_mag_diff_eq_neg_one (h₁ : vec_add_eq a b) (h₂ : vec_sub_eq a b) :
  (a.1 ^ 2 + a.2 ^ 2) - (b.1 ^ 2 + b.2 ^ 2) = -1 :=
  sorry

end vec_mag_diff_eq_neg_one_l63_63665


namespace cost_of_blue_pill_l63_63766

variable (cost_total : ℝ) (days : ℕ) (daily_cost : ℝ)
variable (blue_pill : ℝ) (red_pill : ℝ)

-- Conditions
def condition1 (days : ℕ) : Prop := days = 21
def condition2 (blue_pill red_pill : ℝ) : Prop := blue_pill = red_pill + 2
def condition3 (cost_total daily_cost : ℝ) (days : ℕ) : Prop := cost_total = daily_cost * days
def condition4 (daily_cost blue_pill red_pill : ℝ) : Prop := daily_cost = blue_pill + red_pill

-- Target to prove
theorem cost_of_blue_pill
  (h1 : condition1 days)
  (h2 : condition2 blue_pill red_pill)
  (h3 : condition3 cost_total daily_cost days)
  (h4 : condition4 daily_cost blue_pill red_pill)
  (h5 : cost_total = 945) :
  blue_pill = 23.5 :=
by sorry

end cost_of_blue_pill_l63_63766


namespace inequality_proof_l63_63960

variable {a b : ℕ → ℝ}

-- Conditions: {a_n} is a geometric sequence with positive terms, {b_n} is an arithmetic sequence, a_6 = b_8
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def is_arithmetic (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

axiom a_pos_terms : ∀ n : ℕ, a n > 0
axiom a_geom_seq : is_geometric a
axiom b_arith_seq : is_arithmetic b
axiom a6_eq_b8 : a 6 = b 8

-- Prove: a_3 + a_9 ≥ b_9 + b_7
theorem inequality_proof : a 3 + a 9 ≥ b 9 + b 7 :=
by sorry

end inequality_proof_l63_63960


namespace div_eq_eight_fifths_l63_63213

theorem div_eq_eight_fifths (a b : ℚ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 5) : a / b = 8 / 5 :=
by
  sorry

end div_eq_eight_fifths_l63_63213


namespace gcd_lcm_product_eq_l63_63050

-- Define the numbers
def a : ℕ := 10
def b : ℕ := 15

-- Define the GCD and LCM
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Proposition that needs to be proved
theorem gcd_lcm_product_eq : gcd_ab * lcm_ab = 150 :=
  by
    -- Proof would go here
    sorry

end gcd_lcm_product_eq_l63_63050


namespace sum_of_numbers_l63_63888

theorem sum_of_numbers (x : ℝ) 
  (h_ratio : ∃ x, (2 * x) / x = 2 ∧ (3 * x) / x = 3)
  (h_squares : x^2 + (2 * x)^2 + (3 * x)^2 = 2744) :
  x + 2 * x + 3 * x = 84 :=
by
  sorry

end sum_of_numbers_l63_63888


namespace function_form_l63_63996

def satisfies_condition (f : ℕ → ℤ) : Prop :=
  ∀ m n : ℕ, m > 0 → n > 0 → ⌊ (f (m * n) : ℚ) / n ⌋ = f m

theorem function_form (f : ℕ → ℤ) (h : satisfies_condition f) :
  ∃ r : ℝ, ∀ n : ℕ, 
    (f n = ⌊ (r * n : ℝ) ⌋) ∨ (f n = ⌈ (r * n : ℝ) ⌉ - 1) := 
  sorry

end function_form_l63_63996


namespace cos_alpha_value_l63_63512

open Real

theorem cos_alpha_value (α : ℝ) (h_cos : cos (α - π/6) = 15/17) (h_range : π/6 < α ∧ α < π/2) : 
  cos α = (15 * Real.sqrt 3 - 8) / 34 :=
by
  sorry

end cos_alpha_value_l63_63512


namespace probability_of_C_l63_63475

theorem probability_of_C (P : ℕ → ℚ) (P_total : P 1 + P 2 + P 3 = 1)
  (P_A : P 1 = 1/3) (P_B : P 2 = 1/2) : P 3 = 1/6 :=
by
  sorry

end probability_of_C_l63_63475


namespace sum_divides_product_iff_l63_63718

theorem sum_divides_product_iff (n : ℕ) : 
  (n*(n+1)/2) ∣ n! ↔ ∃ (a b : ℕ), 1 < a ∧ 1 < b ∧ a * b = n + 1 ∧ a ≤ n ∧ b ≤ n :=
sorry

end sum_divides_product_iff_l63_63718


namespace min_weights_needed_l63_63284

theorem min_weights_needed :
  ∃ (weights : List ℕ), (∀ m : ℕ, 1 ≤ m ∧ m ≤ 100 → ∃ (left right : List ℕ), m = (left.sum - right.sum)) ∧ weights.length = 5 :=
sorry

end min_weights_needed_l63_63284


namespace tammy_investment_change_l63_63398

theorem tammy_investment_change :
  ∀ (initial_investment : ℝ) (loss_percent : ℝ) (gain_percent : ℝ),
    initial_investment = 200 → 
    loss_percent = 0.2 → 
    gain_percent = 0.25 →
    ((initial_investment * (1 - loss_percent)) * (1 + gain_percent)) = initial_investment :=
by
  intros initial_investment loss_percent gain_percent
  sorry

end tammy_investment_change_l63_63398


namespace tetrahedron_volume_l63_63565

-- Definition of the required constants and variables
variables {S1 S2 S3 S4 r : ℝ}

-- The volume formula we need to prove
theorem tetrahedron_volume :
  (V = 1/3 * (S1 + S2 + S3 + S4) * r) :=
sorry

end tetrahedron_volume_l63_63565


namespace part_I_part_II_l63_63683

-- Part I
theorem part_I :
  ∀ (x_0 y_0 : ℝ),
  (x_0 ^ 2 + y_0 ^ 2 = 8) ∧
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) →
  ∃ a b : ℝ, (a = 2 ∧ b = 2) →
  (∀ x y : ℝ, (x - 2) ^ 2 + (y - 2) ^ 2 = 8) :=
by 
sorry

-- Part II
theorem part_II :
  ¬ ∃ (x_0 y_0 k_1 k_2 : ℝ),
  (x_0 ^ 2 / 12 + y_0 ^ 2 / 6 = 1) ∧
  (k_1k_2 = (y_0^2 - 4) / (x_0^2 - 4)) ∧
  (k_1 + k_2 = 2 * x_0 * y_0 / (x_0^2 - 4)) ∧
  (k_1k_2 - (k_1 + k_2) / (x_0 * y_0) + 1 = 0) :=
by 
sorry

end part_I_part_II_l63_63683


namespace expression_identity_l63_63287

theorem expression_identity (a : ℤ) (h : a = 102) : 
  a^4 - 4 * a^3 + 6 * a^2 - 4 * a + 1 = 104060401 :=
by {
  rw h,
  calc 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 101^4 : by sorry
  ... = 104060401 : by sorry
}

end expression_identity_l63_63287


namespace rate_per_kg_grapes_l63_63069

/-- Define the conditions for the problem -/
def rate_per_kg_mangoes : ℕ := 55
def kg_grapes_purchased : ℕ := 3
def kg_mangoes_purchased : ℕ := 9
def total_paid : ℕ := 705

/-- The theorem statement to prove the rate per kg for grapes -/
theorem rate_per_kg_grapes (G : ℕ) :
  kg_grapes_purchased * G + kg_mangoes_purchased * rate_per_kg_mangoes = total_paid →
  G = 70 :=
by
  sorry -- Proof will go here

end rate_per_kg_grapes_l63_63069


namespace impossible_labeling_l63_63124

def labeling (n : ℕ) := Fin n → Fin 4

def satisfies_composite_square (f : labeling n) : Prop :=
∀ i j : Fin n, ∃ a b c d : Fin 4, set.eq (set.of_finset (i, j) { f (i + 1) (j), f (i) (j + 1), f (i + 1) (j + 1), f(i) (j) }) = {a,b,c,d}

def satisfies_row_column (f : labeling n) : Prop :=
∀ k : Fin n, ∃ a b c d : Fin 4,
  (∀ (i : Fin n), set.eq (set.of_finset { f k i }) a b c d) ∧
  (∀ (j : Fin n), set.eq (set.of_finset { f i k }) a b c d)

theorem impossible_labeling : ¬ ∃ f : labeling n, satisfies_composite_square f ∧ satisfies_row_column f :=
sorry

end impossible_labeling_l63_63124


namespace man_older_than_son_l63_63327

variables (M S : ℕ)

theorem man_older_than_son
  (h_son_age : S = 26)
  (h_future_age : M + 2 = 2 * (S + 2)) :
  M - S = 28 :=
by sorry

end man_older_than_son_l63_63327


namespace incorrect_value_in_polynomial_progression_l63_63990

noncomputable def polynomial_values (x : ℕ) : ℕ :=
  match x with
  | 0 => 1
  | 1 => 9
  | 2 => 35
  | 3 => 99
  | 4 => 225
  | 5 => 441
  | 6 => 784
  | 7 => 1296
  | _ => 0  -- This is a dummy value just to complete the function

theorem incorrect_value_in_polynomial_progression :
  ¬ (∃ (a b c d : ℝ), ∀ x : ℕ,
    polynomial_values x = (a * x ^ 3 + b * x ^ 2 + c * x + d + if x ≤ 7 then 0 else 1)) :=
by
  intro h
  sorry

end incorrect_value_in_polynomial_progression_l63_63990


namespace license_plate_difference_l63_63042

theorem license_plate_difference :
  (26^3 * 10^4) - (26^4 * 10^3) = -281216000 :=
by
  sorry

end license_plate_difference_l63_63042


namespace volume_removed_percentage_l63_63482

noncomputable def original_volume : ℕ := 20 * 15 * 10

noncomputable def cube_volume : ℕ := 4 * 4 * 4

noncomputable def total_volume_removed : ℕ := 8 * cube_volume

noncomputable def percentage_volume_removed : ℝ :=
  (total_volume_removed : ℝ) / (original_volume : ℝ) * 100

theorem volume_removed_percentage :
  percentage_volume_removed = 512 / 30 := sorry

end volume_removed_percentage_l63_63482


namespace axis_of_symmetry_range_of_m_l63_63372

/-- The conditions given in the original mathematical problem -/
noncomputable def f (x : ℝ) : ℝ :=
  let OA := (2 * Real.cos x, Real.sqrt 3)
  let OB := (Real.sin x + Real.sqrt 3 * Real.cos x, -1)
  (OA.1 * OB.1 + OA.2 * OB.2) + 2

/-- Question 1: The axis of symmetry for the function f(x) -/
theorem axis_of_symmetry :
  ∃ k : ℤ, ∀ x : ℝ, (2 * x + Real.pi / 3 = Real.pi / 2 + k * Real.pi) ↔ (x = k * Real.pi / 2 + Real.pi / 12) :=
sorry

/-- Question 2: The range of m such that g(x) = f(x) + m has zero points for x in (0, π/2) -/
theorem range_of_m (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (∃ c : ℝ, (f x + c = 0)) ↔ ( -4 ≤ c ∧ c < Real.sqrt 3 - 2) :=
sorry

end axis_of_symmetry_range_of_m_l63_63372


namespace three_digit_combinations_count_l63_63967

theorem three_digit_combinations_count : 
  let digits := [1, 2, 3, 4],
      num_digits := 3,
      valid_numbers := {
        x : Fin 1000 // x represents 3 digits, range 0-999
          | let d1 := x / 100, d2 := (x % 100) / 10, d3 := x % 10 in
            d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧
            (d1 = d2 -> d1 ≠ d3) ∧ (d1 = d3 -> d1 ≠ d2) ∧ (d2 = d3 -> d2 ≠ d1)
      } in
  valid_numbers.card = 60 :=
by
  sorry

end three_digit_combinations_count_l63_63967


namespace jackson_house_visits_l63_63543

theorem jackson_house_visits
  (days_per_week : ℕ)
  (total_goal : ℕ)
  (monday_earnings : ℕ)
  (tuesday_earnings : ℕ)
  (earnings_per_4_houses : ℕ)
  (houses_per_4 : ℝ)
  (remaining_days := days_per_week - 2)
  (remaining_goal := total_goal - monday_earnings - tuesday_earnings)
  (daily_goal := remaining_goal / remaining_days)
  (earnings_per_house := houses_per_4 / 4)
  (houses_per_day := daily_goal / earnings_per_house) :
  days_per_week = 5 ∧
  total_goal = 1000 ∧
  monday_earnings = 300 ∧
  tuesday_earnings = 40 ∧
  earnings_per_4_houses = 10 ∧
  houses_per_4 = earnings_per_4_houses.toReal →
  houses_per_day = 88 := 
by 
  sorry

end jackson_house_visits_l63_63543


namespace cos_double_angle_identity_l63_63507

theorem cos_double_angle_identity (x : ℝ) (h : Real.sin (Real.pi / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 :=
sorry

end cos_double_angle_identity_l63_63507


namespace prove_m_plus_n_eq_one_l63_63813

-- Define coordinates of points A and B
def A (m n : ℝ) : ℝ × ℝ := (1 + m, 1 - n)
def B : ℝ × ℝ := (-3, 2)

-- Define symmetry about the y-axis condition
def symmetric_about_y_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = Q.2

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  symmetric_about_y_axis (A m n) B

-- Statement to prove
theorem prove_m_plus_n_eq_one (m n : ℝ) (h : conditions m n) : m + n = 1 := 
by 
  sorry

end prove_m_plus_n_eq_one_l63_63813


namespace tan_435_eq_2_plus_sqrt3_l63_63342

open Real

theorem tan_435_eq_2_plus_sqrt3 : tan (435 * (π / 180)) = 2 + sqrt 3 :=
  sorry

end tan_435_eq_2_plus_sqrt3_l63_63342


namespace shaded_region_area_eq_l63_63901

noncomputable def areaShadedRegion : ℝ :=
  let side_square := 14
  let side_triangle := 18
  let height := 14
  let H := 9 * Real.sqrt 3
  let BF := (side_square + side_triangle, height - H)
  let base_BF := BF.1 - 0
  let height_BF := BF.2
  let area_triangle_BFH := 0.5 * base_BF * height_BF
  let total_triangle_area := 0.5 * side_triangle * height
  let area_half_BFE := 0.5 * total_triangle_area
  area_half_BFE - area_triangle_BFH

theorem shaded_region_area_eq :
  areaShadedRegion = 9 * Real.sqrt 3 :=
by 
 sorry

end shaded_region_area_eq_l63_63901


namespace tangent_value_of_k_k_range_l63_63098

noncomputable def f (x : Real) : Real := Real.exp (2 * x)
def g (k x : Real) : Real := k * x + 1

theorem tangent_value_of_k (k : Real) :
  (∃ t : Real, f t = g k t ∧ deriv f t = deriv (g k) t) → k = 2 :=
by
  sorry

theorem k_range (k : Real) (h : k > 0) :
  (∃ m : Real, m > 0 ∧ ∀ x : Real, 0 < x → x < m → |f x - g k x| > 2 * x) → 4 < k :=
by
  sorry

end tangent_value_of_k_k_range_l63_63098


namespace find_x_value_l63_63237

variable {x : ℝ}

def opposite_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a = (k * b.1, k * b.2)

theorem find_x_value (h : opposite_directions (x, 1) (4, x)) : x = -2 :=
sorry

end find_x_value_l63_63237


namespace numbers_pairs_sum_prod_l63_63308

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l63_63308


namespace Megan_deleted_pictures_l63_63462

/--
Megan took 15 pictures at the zoo and 18 at the museum. She still has 2 pictures from her vacation.
Prove that Megan deleted 31 pictures.
-/
theorem Megan_deleted_pictures :
  let zoo_pictures := 15
  let museum_pictures := 18
  let remaining_pictures := 2
  let total_pictures := zoo_pictures + museum_pictures
  let deleted_pictures := total_pictures - remaining_pictures
  deleted_pictures = 31 :=
by
  sorry

end Megan_deleted_pictures_l63_63462


namespace find_y_from_condition_l63_63598

variable (y : ℝ) (h : (3 * y) / 7 = 15)

theorem find_y_from_condition : y = 35 :=
by {
  sorry
}

end find_y_from_condition_l63_63598


namespace three_digit_numbers_without_579_l63_63210

def count_valid_digits (exclusions : List Nat) (range : List Nat) : Nat :=
  (range.filter (λ n => n ∉ exclusions)).length

def count_valid_three_digit_numbers : Nat :=
  let hundreds := count_valid_digits [5, 7, 9] [1, 2, 3, 4, 6, 8]
  let tens_units := count_valid_digits [5, 7, 9] [0, 1, 2, 3, 4, 6, 8]
  hundreds * tens_units * tens_units

theorem three_digit_numbers_without_579 : 
  count_valid_three_digit_numbers = 294 :=
by
  unfold count_valid_three_digit_numbers
  /- 
  Here you can add intermediate steps if necessary, 
  but for now we assert the final goal since this is 
  just the problem statement with the proof omitted.
  -/
  sorry

end three_digit_numbers_without_579_l63_63210


namespace value_of_a_l63_63074

theorem value_of_a (a : ℕ) (h1 : a * 9^3 = 3 * 15^5) (h2 : a = 5^5) : a = 3125 := by
  sorry

end value_of_a_l63_63074


namespace inequality_abc_distinct_positive_l63_63851

theorem inequality_abc_distinct_positive
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d) :
  (a^2 / b + b^2 / c + c^2 / d + d^2 / a > a + b + c + d) := 
by
  sorry

end inequality_abc_distinct_positive_l63_63851


namespace tangent_parallel_x_axis_tangent_45_degrees_x_axis_l63_63626

-- Condition: Define the curve
def curve (x : ℝ) : ℝ := x^2 - 1

-- Condition: Calculate derivative
def derivative_curve (x : ℝ) : ℝ := 2 * x

-- Part (a): Point where tangent is parallel to the x-axis
theorem tangent_parallel_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 0 ∧ x = 0 ∧ y = -1) :=
  sorry

-- Part (b): Point where tangent forms a 45 degree angle with the x-axis
theorem tangent_45_degrees_x_axis :
  (∃ x y : ℝ, y = curve x ∧ derivative_curve x = 1 ∧ x = 1/2 ∧ y = -3/4) :=
  sorry

end tangent_parallel_x_axis_tangent_45_degrees_x_axis_l63_63626


namespace solve_for_x_l63_63752

variable (x : ℝ)

-- Define the condition: 20% of x = 300
def twenty_percent_eq_300 := (0.20 * x = 300)

-- Define the goal: 120% of x = 1800
def one_twenty_percent_eq_1800 := (1.20 * x = 1800)

theorem solve_for_x (h : twenty_percent_eq_300 x) : one_twenty_percent_eq_1800 x :=
sorry

end solve_for_x_l63_63752


namespace train_speed_correct_l63_63492

/-- Define the length of the train in meters -/
def length_train : ℝ := 120

/-- Define the length of the bridge in meters -/
def length_bridge : ℝ := 160

/-- Define the time taken to pass the bridge in seconds -/
def time_taken : ℝ := 25.2

/-- Define the expected speed of the train in meters per second -/
def expected_speed : ℝ := 11.1111

/-- Prove that the speed of the train is 11.1111 meters per second given conditions -/
theorem train_speed_correct :
  (length_train + length_bridge) / time_taken = expected_speed :=
by
  sorry

end train_speed_correct_l63_63492


namespace face_opposite_to_turquoise_is_pink_l63_63856

-- Declare the inductive type for the color of the face
inductive Color
| P -- Pink
| V -- Violet
| T -- Turquoise
| O -- Orange

open Color

-- Define the setup conditions of the problem
def cube_faces : List Color :=
  [P, P, P, V, V, T, O]

-- Define the positions of the faces for the particular folded cube configuration
-- Assuming the function cube_configuration gives the face opposite to a given face.
axiom cube_configuration : Color → Color

-- State the main theorem regarding the opposite face
theorem face_opposite_to_turquoise_is_pink : cube_configuration T = P :=
sorry

end face_opposite_to_turquoise_is_pink_l63_63856


namespace smallest_bottom_right_value_l63_63786

theorem smallest_bottom_right_value :
  ∃ (grid : ℕ × ℕ × ℕ → ℕ), -- grid as a function from row/column pairs to natural numbers
    (∀ i j, 1 ≤ i ∧ i ≤ 3 → 1 ≤ j ∧ j ≤ 3 → grid (i, j) ≠ 0) ∧ -- all grid values are non-zero
    (grid (1, 1) ≠ grid (1, 2) ∧ grid (1, 1) ≠ grid (1, 3) ∧ grid (1, 2) ≠ grid (1, 3) ∧
     grid (2, 1) ≠ grid (2, 2) ∧ grid (2, 1) ≠ grid (2, 3) ∧ grid (2, 2) ≠ grid (2, 3) ∧
     grid (3, 1) ≠ grid (3, 2) ∧ grid (3, 1) ≠ grid (3, 3) ∧ grid (3, 2) ≠ grid (3, 3)) ∧ -- all grid values are distinct
    (grid (1, 1) + grid (1, 2) = grid (1, 3)) ∧ 
    (grid (2, 1) + grid (2, 2) = grid (2, 3)) ∧ 
    (grid (3, 1) + grid (3, 2) = grid (3, 3)) ∧ -- row sum conditions
    (grid (1, 1) + grid (2, 1) = grid (3, 1)) ∧ 
    (grid (1, 2) + grid (2, 2) = grid (3, 2)) ∧ 
    (grid (1, 3) + grid (2, 3) = grid (3, 3)) ∧ -- column sum conditions
    (grid (3, 3) = 12) :=
by
  sorry

end smallest_bottom_right_value_l63_63786


namespace find_numbers_l63_63314

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l63_63314


namespace triangle_third_side_length_l63_63535

theorem triangle_third_side_length (a b : ℕ) (h1 : a = 2) (h2 : b = 3) 
(h3 : ∃ x, x^2 - 10 * x + 21 = 0 ∧ (a + b > x) ∧ (a + x > b) ∧ (b + x > a)) :
  ∃ x, x = 3 := 
by 
  sorry

end triangle_third_side_length_l63_63535


namespace initial_men_in_camp_l63_63569

theorem initial_men_in_camp (days_initial men_initial : ℕ) (days_plus_thirty men_plus_thirty : ℕ)
(h1 : days_initial = 20)
(h2 : men_plus_thirty = men_initial + 30)
(h3 : days_plus_thirty = 5)
(h4 : (men_initial * days_initial) = (men_plus_thirty * days_plus_thirty)) :
  men_initial = 10 :=
by sorry

end initial_men_in_camp_l63_63569


namespace area_of_region_B_l63_63041

-- Given conditions
def region_B (z : ℂ) : Prop :=
  (0 ≤ (z.re / 50) ∧ (z.re / 50) ≤ 1 ∧ 0 ≤ (z.im / 50) ∧ (z.im / 50) ≤ 1)
  ∧
  (0 ≤ (50 * z.re / (z.re^2 + z.im^2)) ∧ (50 * z.re / (z.re^2 + z.im^2)) ≤ 1 ∧ 
  0 ≤ (50 * z.im / (z.re^2 + z.im^2)) ∧ (50 * z.im / (z.re^2 + z.im^2)) ≤ 1)

-- Theorem to be proved
theorem area_of_region_B : 
  (∫ z in {z : ℂ | region_B z}, 1) = 1875 - 312.5 * Real.pi :=
by
  sorry

end area_of_region_B_l63_63041


namespace no_such_hexagon_and_point_l63_63927

noncomputable def hexagon_and_point_condition : Prop :=
∀ (hex : List (ℝ × ℝ)), (hex.length = 6) → 
  (∀ i, i < 6 → dist hex[i] hex[(i+1) % 6] > 1) → 
  ¬∃ M : ℝ × ℝ, (convex {p | p ∈ hex}) ∧ (∀ i, i < 6 → dist M hex[i] < 1)

theorem no_such_hexagon_and_point : hexagon_and_point_condition :=
by
  sorry

end no_such_hexagon_and_point_l63_63927


namespace calculate_expression_l63_63162

theorem calculate_expression (y : ℝ) (hy : y ≠ 0) : 
  (18 * y^3) * (4 * y^2) * (1/(2 * y)^3) = 9 * y^2 :=
by
  sorry

end calculate_expression_l63_63162


namespace cost_of_dozen_pens_l63_63568

theorem cost_of_dozen_pens
  (cost_three_pens_five_pencils : ℝ)
  (cost_one_pen : ℝ)
  (pen_to_pencil_ratio : ℝ)
  (h1 : 3 * cost_one_pen + 5 * (cost_three_pens_five_pencils / 8) = 260)
  (h2 : cost_one_pen = 65)
  (h3 : cost_one_pen / (cost_three_pens_five_pencils / 8) = 5/1)
  : 12 * cost_one_pen = 780 := by
    sorry

end cost_of_dozen_pens_l63_63568


namespace cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l63_63513

theorem cos_alpha_plus_5pi_over_12_eq_neg_1_over_3
  (α : ℝ)
  (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 5 * π / 12) = -1 / 3 :=
by
  sorry

end cos_alpha_plus_5pi_over_12_eq_neg_1_over_3_l63_63513


namespace compute_expression_l63_63916

theorem compute_expression :
  (-9 * 5) - (-7 * -2) + (11 * -4) = -103 :=
by
  sorry

end compute_expression_l63_63916


namespace circumcircle_through_midpoint_l63_63402

noncomputable theory -- due to the use of classical geometry, which might involve classical logic

open EuclideanGeometry -- this opens needed geometric constructs and definitions

/-- 
Given an acute-angled triangle ABC with altitudes intersecting opposite sides at points D, E, F.
Consider a line through D that is parallel to EF and intersects AC and AB at points Q and R, respectively.
Let P be the point where EF intersects BC. Prove that the circumcircle of triangle PQR passes through the midpoint M of BC.
-/
theorem circumcircle_through_midpoint 
  (ABC : Triangle)
  (h_acute : acute_triangle ABC)
  (D E F : Point)
  (h1 : foot D A B C)
  (h2 : foot E B C A)
  (h3 : foot F C A B)
  (Q R : Point)
  (h4 : parallel_line D EF Q R)
  (P : Point)
  (h5 : intersect_line EF BC P)
  (M : Point)
  (h6 : midpoint M B C) :
  is_on_circumcircle P Q R M := 
sorry

end circumcircle_through_midpoint_l63_63402


namespace find_unknown_number_l63_63577

theorem find_unknown_number (x : ℝ) : 
  (1000 * 7) / (x * 17) = 10000 → x = 24.285714285714286 := by
  sorry

end find_unknown_number_l63_63577


namespace forgot_days_l63_63331

def July_days : ℕ := 31
def days_took_capsules : ℕ := 27

theorem forgot_days : July_days - days_took_capsules = 4 :=
by
  sorry

end forgot_days_l63_63331


namespace eval_expr_l63_63219

theorem eval_expr (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a > b) :
  (a^(b+1) * b^(a+1)) / (b^(b+1) * a^(a+1)) = (a / b)^(b - a) :=
sorry

end eval_expr_l63_63219


namespace initial_fee_calculation_l63_63406

theorem initial_fee_calculation 
  (charge_per_segment : ℝ)
  (segment_length : ℝ)
  (total_distance : ℝ)
  (total_charge : ℝ)
  (number_of_segments := total_distance / segment_length)
  (cost_for_distance := number_of_segments * charge_per_segment)
  (initial_fee := total_charge - cost_for_distance) :
  charge_per_segment = 0.35 → 
  segment_length = 2 / 5 → 
  total_distance = 3.6 → 
  total_charge = 5.5 → 
  initial_fee = 2.35 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp only [div_eq_mul_inv, mul_comm (3.6:ℝ), mul_assoc, mul_inv_cancel_left₀ (ne_of_gt (by norm_num : (2:ℝ) ≠ 0)), mul_comm 2, ←mul_assoc, mul_comm 0.35, sub_self_add]
  norm_num

end initial_fee_calculation_l63_63406


namespace neg_three_is_square_mod_p_l63_63709

theorem neg_three_is_square_mod_p (q : ℤ) (p : ℕ) (prime_p : Nat.Prime p) (condition : p = 3 * q + 1) :
  ∃ x : ℤ, (x^2 ≡ -3 [ZMOD p]) :=
sorry

end neg_three_is_square_mod_p_l63_63709


namespace f_1986_l63_63072

noncomputable def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 3 * f (a * b)
axiom f_1 : f 1 = 2

theorem f_1986 : f 1986 = 2 :=
by
  sorry

end f_1986_l63_63072


namespace cost_of_child_ticket_is_4_l63_63132

def cost_of_child_ticket (cost_adult cost_total tickets_sold tickets_child receipts_total : ℕ) : ℕ :=
  let tickets_adult := tickets_sold - tickets_child
  let receipts_adult := tickets_adult * cost_adult
  let receipts_child := receipts_total - receipts_adult
  receipts_child / tickets_child

theorem cost_of_child_ticket_is_4 (cost_adult : ℕ) (cost_total : ℕ)
  (tickets_sold : ℕ) (tickets_child : ℕ) (receipts_total : ℕ) :
  cost_of_child_ticket 12 4 130 90 840 = 4 := by
  sorry

end cost_of_child_ticket_is_4_l63_63132


namespace exists_p_q_for_integer_roots_l63_63631

theorem exists_p_q_for_integer_roots : 
  ∃ (p q : ℤ), ∀ k (hk : k ∈ (Finset.range 10)), 
    ∃ (r1 r2 : ℤ), (r1 + r2 = -(p + k)) ∧ (r1 * r2 = (q + k)) :=
sorry

end exists_p_q_for_integer_roots_l63_63631


namespace positive_difference_l63_63451

theorem positive_difference : 496 = abs ((64 + 64) / 8 - (64 * 64) / 8) := by
  have h1 : 8^2 = 64 := rfl
  have h2 : 64 + 64 = 128 := rfl
  have h3 : (128 : ℕ) / 8 = 16 := rfl
  have h4 : 64 * 64 = 4096 := rfl
  have h5 : (4096 : ℕ) / 8 = 512 := rfl
  have h6 : 512 - 16 = 496 := rfl
  sorry

end positive_difference_l63_63451


namespace xiaolin_final_score_l63_63321

-- Define the conditions
def score_situps : ℕ := 80
def score_800m : ℕ := 90
def weight_situps : ℕ := 4
def weight_800m : ℕ := 6

-- Define the final score based on the given conditions
def final_score : ℕ :=
  (score_situps * weight_situps + score_800m * weight_800m) / (weight_situps + weight_800m)

-- Prove that the final score is 86
theorem xiaolin_final_score : final_score = 86 :=
by sorry

end xiaolin_final_score_l63_63321


namespace sum_of_solutions_l63_63051

theorem sum_of_solutions (x : ℝ) (h : ∀ x, (x ≠ 1) ∧ (x ≠ -1) → ( -15 * x / (x^2 - 1) = 3 * x / (x + 1) - 9 / (x - 1) )) : 
  (∀ x, (x ≠ 1) ∧ (x ≠ -1) → -15 * x / (x^2 - 1) = 3 * x / (x+1) - 9 / (x-1)) → (x = ( -1 + Real.sqrt 13 ) / 2 ∨ x = ( -1 - Real.sqrt 13 ) / 2) → (x + ( -x ) = -1) :=
by
  sorry

end sum_of_solutions_l63_63051


namespace fraction_arithmetic_l63_63778

theorem fraction_arithmetic :
  ((5 : ℚ) / 6 - (1 : ℚ) / 3) * (3 / 4) = 3 / 8 :=
by
  sorry

end fraction_arithmetic_l63_63778


namespace remainder_3_pow_89_plus_5_mod_7_l63_63750

theorem remainder_3_pow_89_plus_5_mod_7 :
  (3^1 % 7 = 3) ∧ (3^2 % 7 = 2) ∧ (3^3 % 7 = 6) ∧ (3^4 % 7 = 4) ∧ (3^5 % 7 = 5) ∧ (3^6 % 7 = 1) →
  ((3^89 + 5) % 7 = 3) :=
by
  intros h
  sorry

end remainder_3_pow_89_plus_5_mod_7_l63_63750


namespace minimize_quadratic_expression_l63_63596

theorem minimize_quadratic_expression :
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, (y^2 - 6*y + 8) ≥ (x^2 - 6*x + 8) := by
sorry

end minimize_quadratic_expression_l63_63596


namespace elizabeth_net_profit_l63_63935

theorem elizabeth_net_profit :
  let cost_per_bag := 3.00
  let num_bags := 20
  let price_first_15_bags := 6.00
  let price_last_5_bags := 4.00
  let total_cost := cost_per_bag * num_bags
  let revenue_first_15 := 15 * price_first_15_bags
  let revenue_last_5 := 5 * price_last_5_bags
  let total_revenue := revenue_first_15 + revenue_last_5
  let net_profit := total_revenue - total_cost
  net_profit = 50.00 :=
by
  sorry

end elizabeth_net_profit_l63_63935


namespace Kayla_total_items_l63_63695

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l63_63695


namespace largest_possible_a_l63_63554

theorem largest_possible_a 
  (a b c d : ℕ) 
  (h1 : a < 2 * b)
  (h2 : b < 3 * c)
  (h3 : c < 2 * d)
  (h4 : d < 100) :
  a ≤ 1179 :=
sorry

end largest_possible_a_l63_63554


namespace factorize_m_factorize_x_factorize_xy_l63_63247

theorem factorize_m (m : ℝ) : m^2 + 7 * m - 18 = (m - 2) * (m + 9) := 
sorry

theorem factorize_x (x : ℝ) : x^2 - 2 * x - 8 = (x + 2) * (x - 4) :=
sorry

theorem factorize_xy (x y : ℝ) : (x * y)^2 - 7 * (x * y) + 10 = (x * y - 2) * (x * y - 5) := 
sorry

end factorize_m_factorize_x_factorize_xy_l63_63247


namespace complex_fraction_l63_63161

theorem complex_fraction (h : (1 : ℂ) - I = 1 - (I : ℂ)) :
  ((1 - I) * (1 - (2 * I))) / (1 + I) = -2 - I := 
by
  sorry

end complex_fraction_l63_63161


namespace centrally_symmetric_equidecomposable_l63_63891

-- Assume we have a type for Polyhedra
variable (Polyhedron : Type)

-- Conditions
variable (sameVolume : Polyhedron → Polyhedron → Prop)
variable (centrallySymmetricFaces : Polyhedron → Prop)
variable (equidecomposable : Polyhedron → Polyhedron → Prop)

-- Theorem statement
theorem centrally_symmetric_equidecomposable 
  (P Q : Polyhedron) 
  (h1 : sameVolume P Q) 
  (h2 : centrallySymmetricFaces P) 
  (h3 : centrallySymmetricFaces Q) :
  equidecomposable P Q := 
sorry

end centrally_symmetric_equidecomposable_l63_63891


namespace high_school_heralds_games_lost_percentage_l63_63125

theorem high_school_heralds_games_lost_percentage :
  ∀ (won lost : ℕ) (total_games : ℕ) (ratio_won_lost : ℚ),
    ratio_won_lost = 8 / 5 →
    total_games = won + lost →
    total_games = 52 →
    ∃ percentage_lost : ℚ, percentage_lost = (lost * 100 / total_games) ∧ percentage_lost ≈ 38 :=
by
  sorry

end high_school_heralds_games_lost_percentage_l63_63125


namespace range_of_a_l63_63369

theorem range_of_a (f : ℝ → ℝ) (h1 : ∀ x, f (x - 3) = f (3 - (x - 3))) (h2 : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  {a : ℝ | f (2 - a^2) > f a} = {a | -2 < a ∧ a < 1} :=
by
  sorry

end range_of_a_l63_63369


namespace find_k_l63_63746

-- Define a point and its translation
structure Point where
  x : ℕ
  y : ℕ

-- Original and translated points
def P : Point := { x := 5, y := 3 }
def P' : Point := { x := P.x - 4, y := P.y - 1 }

-- Given function with parameter k
def line (k : ℕ) (p : Point) : ℕ := (k * p.x) - 2

-- Prove the value of k
theorem find_k (k : ℕ) (h : line k P' = P'.y) : k = 4 :=
by
  sorry

end find_k_l63_63746


namespace brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l63_63601

noncomputable def brocard_vertex_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(a * b * c, c^3, b^3)

theorem brocard_vertex_coordinates_correct (a b c : ℝ) :
  brocard_vertex_trilinear_coordinates a b c = (a * b * c, c^3, b^3) :=
sorry

noncomputable def steiner_point_trilinear_coordinates (a b c : ℝ) : ℝ × ℝ × ℝ :=
(1 / (a * (b^2 - c^2)),
  1 / (b * (c^2 - a^2)),
  1 / (c * (a^2 - b^2)))

theorem steiner_point_coordinates_correct (a b c : ℝ) :
  steiner_point_trilinear_coordinates a b c = 
  (1 / (a * (b^2 - c^2)),
   1 / (b * (c^2 - a^2)),
   1 / (c * (a^2 - b^2))) :=
sorry

end brocard_vertex_coordinates_correct_steiner_point_coordinates_correct_l63_63601


namespace chess_tournament_third_place_wins_l63_63985

theorem chess_tournament_third_place_wins :
  ∀ (points : Fin 8 → ℕ)
  (total_games : ℕ)
  (total_points : ℕ),
  (total_games = 28) →
  (∀ i j : Fin 8, i ≠ j → points i ≠ points j) →
  ((points 1) = (points 4 + points 5 + points 6 + points 7)) →
  (points 2 > points 4) →
  ∃ (games_won : Fin 8 → Fin 8 → Prop),
  (games_won 2 4) :=
by
  sorry

end chess_tournament_third_place_wins_l63_63985


namespace find_x_squared_perfect_square_l63_63410

theorem find_x_squared_perfect_square (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n ≠ m)
  (h4 : n > m) (h5 : n % 2 ≠ m % 2) : 
  ∃ x : ℤ, x = 0 ∧ ∀ x, (x = 0) → ∃ k : ℕ, (x ^ (2 ^ n) - 1) / (x ^ (2 ^ m) - 1) = k^2 :=
sorry

end find_x_squared_perfect_square_l63_63410


namespace range_of_x_for_odd_function_l63_63861

theorem range_of_x_for_odd_function (f : ℝ → ℝ) (domain : Set ℝ)
  (h_odd : ∀ x ∈ domain, f (-x) = -f x)
  (h_mono : ∀ x y, 0 < x -> x < y -> f x < f y)
  (h_f3 : f 3 = 0)
  (h_ineq : ∀ x, x ∈ domain -> x * (f x - f (-x)) < 0) : 
  ∀ x, x * f x < 0 ↔ -3 < x ∧ x < 0 ∨ 0 < x ∧ x < 3 :=
by sorry

end range_of_x_for_odd_function_l63_63861


namespace pudding_cost_l63_63404

theorem pudding_cost (P : ℝ) (h1 : 75 = 5 * P + 65) : P = 2 :=
sorry

end pudding_cost_l63_63404


namespace hair_length_correct_l63_63248

-- Define the initial hair length, the cut length, and the growth length as constants
def l_initial : ℕ := 16
def l_cut : ℕ := 11
def l_growth : ℕ := 12

-- Define the final hair length as the result of the operations described
def l_final : ℕ := l_initial - l_cut + l_growth

-- State the theorem we want to prove
theorem hair_length_correct : l_final = 17 :=
by
  sorry

end hair_length_correct_l63_63248


namespace find_cost_price_l63_63296

/-- Define the given conditions -/
def selling_price : ℝ := 100
def profit_percentage : ℝ := 0.15
def cost_price : ℝ := 86.96

/-- Define the relationship between selling price and cost price -/
def relation (CP SP : ℝ) : Prop := SP = CP * (1 + profit_percentage)

/-- State the theorem based on the conditions and required proof -/
theorem find_cost_price 
  (SP : ℝ) (CP : ℝ) 
  (h1 : SP = selling_price) 
  (h2 : relation CP SP) : 
  CP = cost_price := 
by
  sorry

end find_cost_price_l63_63296


namespace chastity_lollipops_l63_63780

theorem chastity_lollipops (initial_money lollipop_cost gummy_cost left_money total_gummies total_spent lollipops : ℝ)
  (h1 : initial_money = 15)
  (h2 : lollipop_cost = 1.50)
  (h3 : gummy_cost = 2)
  (h4 : left_money = 5)
  (h5 : total_gummies = 2)
  (h6 : total_spent = initial_money - left_money)
  (h7 : total_spent = 10)
  (h8 : total_gummies * gummy_cost = 4)
  (h9 : total_spent - (total_gummies * gummy_cost) = 6)
  (h10 : lollipops = (total_spent - (total_gummies * gummy_cost)) / lollipop_cost) :
  lollipops = 4 := 
sorry

end chastity_lollipops_l63_63780


namespace juan_amal_probability_l63_63694

theorem juan_amal_probability :
  let prob := ∑ i in finset.range 1 11, ∑ j in finset.range 1 7, 
             if (i * j) % 4 = 0 then (1 / 10) * (1 / 6) else 0
  in prob = 1 / 3 :=
by {
  let prob := ∑ i in finset.range 1 11, ∑ j in finset.range 1 7, 
             if (i * j) % 4 = 0 then (1 / 10) * (1 / 6) else 0,
  exact eq.trans (prob) 1 / 3 sorry
}

end juan_amal_probability_l63_63694


namespace interest_rate_per_annum_l63_63330

def principal : ℝ := 8945
def simple_interest : ℝ := 4025.25
def time : ℕ := 5

theorem interest_rate_per_annum : (simple_interest * 100) / (principal * time) = 9 := by
  sorry

end interest_rate_per_annum_l63_63330


namespace coefficients_sum_eq_four_l63_63176

noncomputable def simplified_coefficients_sum (y : ℚ → ℚ) : ℚ :=
  let A := 1
  let B := 3
  let C := 2
  let D := -2
  A + B + C + D

theorem coefficients_sum_eq_four : simplified_coefficients_sum (λ x => 
  (x^3 + 5*x^2 + 8*x + 4) / (x + 2)) = 4 := by
  sorry

end coefficients_sum_eq_four_l63_63176


namespace seven_large_power_mod_seventeen_l63_63790

theorem seven_large_power_mod_seventeen :
  (7 : ℤ)^1985 % 17 = 7 :=
by
  have h1 : (7 : ℤ)^2 % 17 = 15 := sorry
  have h2 : (7 : ℤ)^4 % 17 = 16 := sorry
  have h3 : (7 : ℤ)^8 % 17 = 1 := sorry
  have h4 : 1985 = 8 * 248 + 1 := sorry
  sorry

end seven_large_power_mod_seventeen_l63_63790


namespace value_of_e_l63_63603

theorem value_of_e (a : ℕ) (e : ℕ) 
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * 45 * e) : 
  e = 49 := 
by 
  sorry

end value_of_e_l63_63603


namespace a11_a12_a13_eq_105_l63_63997

variable (a : ℕ → ℝ) -- Define the arithmetic sequence
variable (d : ℝ) -- Define the common difference

-- Assume the conditions given in step a)
axiom arith_seq (n : ℕ) : a n = a 0 + n * d
axiom sum_3_eq_15 : a 0 + a 1 + a 2 = 15
axiom prod_3_eq_80 : a 0 * a 1 * a 2 = 80
axiom pos_diff : d > 0

theorem a11_a12_a13_eq_105 : a 10 + a 11 + a 12 = 105 :=
sorry

end a11_a12_a13_eq_105_l63_63997


namespace area_of_circle_l63_63254

/-- Given a circle with circumference 36π, prove that the area is 324π. -/
theorem area_of_circle (C : ℝ) (hC : C = 36 * π) 
  (h1 : ∀ r : ℝ, C = 2 * π * r → 0 ≤ r)
  (h2 : ∀ r : ℝ, 0 ≤ r → ∃ (A : ℝ), A = π * r^2) :
  ∃ k : ℝ, (A = 324 * π → k = 324) := 
sorry


end area_of_circle_l63_63254


namespace find_number_of_pairs_l63_63092

variable (n : ℕ)
variable (prob_same_color : ℚ := 0.09090909090909091)
variable (total_shoes : ℕ := 12)
variable (pairs_of_shoes : ℕ)

-- The condition on the probability of selecting two shoes of the same color
def condition_probability : Prop :=
  (1 : ℚ) / ((2 * n - 1) : ℚ) = prob_same_color

-- The condition on the total number of shoes
def condition_total_shoes : Prop :=
  2 * n = total_shoes

-- The goal to prove that the number of pairs of shoes is 6 given the conditions
theorem find_number_of_pairs (h1 : condition_probability n) (h2 : condition_total_shoes n) : n = 6 :=
by
  sorry

end find_number_of_pairs_l63_63092


namespace perpendicular_vector_l63_63521

theorem perpendicular_vector {a : ℝ × ℝ} (h : a = (1, -2)) : ∃ (b : ℝ × ℝ), b = (2, 1) ∧ (a.1 * b.1 + a.2 * b.2 = 0) :=
by 
  sorry

end perpendicular_vector_l63_63521


namespace ratio_of_apples_l63_63153

/-- The store sold 32 red apples and the combined amount of red and green apples sold was 44. -/
theorem ratio_of_apples (R G : ℕ) (h1 : R = 32) (h2 : R + G = 44) : R / 4 = 8 ∧ G / 4 = 3 :=
by {
  -- Placeholder for the proof
  sorry
}

end ratio_of_apples_l63_63153


namespace S_when_R_is_16_and_T_is_1_div_4_l63_63600

theorem S_when_R_is_16_and_T_is_1_div_4 :
  ∃ (S : ℝ), (∀ (R S T : ℝ) (c : ℝ), (R = c * S / T) →
  (2 = c * 8 / (1/2)) → c = 1 / 8) ∧
  (16 = (1/8) * S / (1/4)) → S = 32 :=
sorry

end S_when_R_is_16_and_T_is_1_div_4_l63_63600


namespace factorial_expression_simplification_l63_63035

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l63_63035


namespace positive_difference_eq_496_l63_63453

theorem positive_difference_eq_496 : 
  let a := 8 ^ 2 in 
  (a + a) / 8 - (a * a) / 8 = 496 :=
by
  let a := 8^2
  have h1 : (a + a) / 8 = 16 := by sorry
  have h2 : (a * a) / 8 = 512 := by sorry
  show (a + a) / 8 - (a * a) / 8 = 496 from by
    calc
      (a + a) / 8 - (a * a) / 8
            = 16 - 512 : by rw [h1, h2]
        ... = -496 : by ring
        ... = 496 : by norm_num

end positive_difference_eq_496_l63_63453


namespace polynomial_divisible_by_x_minus_2_l63_63046

theorem polynomial_divisible_by_x_minus_2 (k : ℝ) :
  (2 * (2 : ℝ)^3 - 8 * (2 : ℝ)^2 + k * (2 : ℝ) - 10 = 0) → 
  k = 13 :=
by 
  intro h
  sorry

end polynomial_divisible_by_x_minus_2_l63_63046


namespace product_of_roots_l63_63782

-- Define the coefficients of the cubic equation
def a : ℝ := 2
def d : ℝ := 12

-- Define the cubic equation
def cubic_eq (x : ℝ) : ℝ := a * x^3 - 3 * x^2 - 8 * x + d

-- Prove the product of the roots is -6 using Vieta's formulas
theorem product_of_roots : -d / a = -6 := by
  sorry

end product_of_roots_l63_63782


namespace find_B_l63_63458

theorem find_B (A B : ℕ) (h : 5 * 100 + 10 * A + 8 - (B * 100 + 14) = 364) : B = 2 :=
sorry

end find_B_l63_63458


namespace original_employee_salary_l63_63180

-- Given conditions
def emily_original_salary : ℝ := 1000000
def emily_new_salary : ℝ := 850000
def number_of_employees : ℕ := 10
def employee_new_salary : ℝ := 35000

-- Prove the original salary of each employee
theorem original_employee_salary :
  (emily_original_salary - emily_new_salary) / number_of_employees = employee_new_salary - 20000 := 
by
  sorry

end original_employee_salary_l63_63180


namespace circumcircle_tangent_ef_l63_63757

variables {A B C D E F P Q M : Point}

-- Define the incenter and incircle properties
noncomputable def incenter (ABC : Triangle) : Point := sorry
noncomputable def incircle (ABC : Triangle) : Circle := sorry

-- Define perpendiculars
def perpendicular (p q : Point) : Prop := sorry

def triangle (A B C : Point) : Prop := sorry

-- Assume given conditions.
variables (h1 : incircle (triangle A B C) touches_side AB at D)
          (h2 : incircle (triangle A B C) touches_side BC at E)
          (h3 : incircle (triangle A B C) touches_side CA at F)
          (h4 : perpendicular D BC P)
          (h5 : perpendicular E BC Q)
          (h6 : second_intersection_point (segment AP) (circle_in (triangle ABC)) M)

-- Prove that the circumcircle of triangle ADQ is tangent to EF.
theorem circumcircle_tangent_ef :
  let circumcircle_ADQ := circumcircle (triangle A D Q) in
  tangent_to circumcircle_ADQ (line_contain E F) :=
sorry

end circumcircle_tangent_ef_l63_63757


namespace find_two_numbers_l63_63300

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l63_63300


namespace households_using_neither_brands_l63_63150

def total_households : Nat := 240
def only_brand_A_households : Nat := 60
def both_brands_households : Nat := 25
def ratio_B_to_both : Nat := 3
def only_brand_B_households : Nat := ratio_B_to_both * both_brands_households
def either_brand_households : Nat := only_brand_A_households + only_brand_B_households + both_brands_households
def neither_brand_households : Nat := total_households - either_brand_households

theorem households_using_neither_brands :
  neither_brand_households = 80 :=
by
  -- Proof can be filled out here
  sorry

end households_using_neither_brands_l63_63150


namespace find_numbers_l63_63299

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l63_63299


namespace general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l63_63196

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Noncomputable sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}
variable (c : ℤ)

axiom h1 : is_arithmetic_sequence a d
axiom h2 : d > 0
axiom h3 : a 1 * a 2 = 45
axiom h4 : a 0 + a 4 = 18

-- General formula for the nth term
theorem general_formula_for_nth_term :
  ∃ a1 d, a 0 = a1 ∧ d > 0 ∧ (∀ n, a n = a1 + n * d) :=
sorry

-- Arithmetic sequence from Sn/(n+c)
theorem exists_c_makes_bn_arithmetic :
  ∃ (c : ℤ), c ≠ 0 ∧ (∀ n, n > 0 → (arithmetic_sum a n) / (n + c) - (arithmetic_sum a (n - 1)) / (n - 1 + c) = d) :=
sorry

end general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l63_63196


namespace that_remaining_money_l63_63739

section
/-- Initial money in Olivia's wallet --/
def initial_money : ℕ := 53

/-- Money collected from ATM --/
def collected_money : ℕ := 91

/-- Money spent at the supermarket --/
def spent_money : ℕ := collected_money + 39

/-- Remaining money after visiting the supermarket --
Theorem that proves Olivia's remaining money is 14 dollars.
-/
theorem remaining_money : initial_money + collected_money - spent_money = 14 := 
by
  unfold initial_money collected_money spent_money
  simp
  sorry
end

end that_remaining_money_l63_63739


namespace part1_part2_l63_63519

-- Define the function, assumptions, and the proof for the first part
theorem part1 (m : ℝ) (x : ℝ) :
  (∀ x > 1, -m * (0 * x + 1) * Real.log x + x - 0 ≥ 0) →
  m ≤ Real.exp 1 := sorry

-- Define the function, assumptions, and the proof for the second part
theorem part2 (x : ℝ) :
  (∀ x > 0, (x - 1) * (-(x + 1) * Real.log x + x - 1) ≤ 0) := sorry

end part1_part2_l63_63519


namespace total_coins_l63_63548
-- Import the necessary library

-- Defining the conditions
def quarters := 22
def dimes := quarters + 3
def nickels := quarters - 6

-- Main theorem statement
theorem total_coins : (quarters + dimes + nickels) = 63 := by
  sorry

end total_coins_l63_63548


namespace units_digit_of_a_l63_63396

theorem units_digit_of_a (a : ℕ) (ha : (∃ b : ℕ, 1 ≤ b ∧ b ≤ 9 ∧ (a*a / 10^1) % 10 = b)) : 
  ((a % 10 = 4) ∨ (a % 10 = 6)) :=
sorry

end units_digit_of_a_l63_63396


namespace Conor_can_chop_116_vegetables_in_a_week_l63_63636

-- Define the conditions
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8
def work_days_per_week : ℕ := 4

-- Define the total vegetables per day
def vegetables_per_day : ℕ := eggplants_per_day + carrots_per_day + potatoes_per_day

-- Define the total vegetables per week
def vegetables_per_week : ℕ := vegetables_per_day * work_days_per_week

-- The proof statement
theorem Conor_can_chop_116_vegetables_in_a_week : vegetables_per_week = 116 :=
by
  sorry  -- The proof step is omitted with sorry

end Conor_can_chop_116_vegetables_in_a_week_l63_63636


namespace saree_blue_stripes_l63_63878

theorem saree_blue_stripes :
  ∀ (brown_stripes gold_stripes blue_stripes : ℕ),
    brown_stripes = 4 →
    gold_stripes = 3 * brown_stripes →
    blue_stripes = 5 * gold_stripes →
    blue_stripes = 60 :=
by
  intros brown_stripes gold_stripes blue_stripes h_brown h_gold h_blue
  sorry

end saree_blue_stripes_l63_63878


namespace polygon_sides_and_diagonals_l63_63806

theorem polygon_sides_and_diagonals (n : ℕ) (h : (n-2) * 180 / 360 = 13 / 2) : 
  n = 15 ∧ (n * (n - 3) / 2 = 90) :=
by {
  sorry
}

end polygon_sides_and_diagonals_l63_63806


namespace exists_zero_in_interval_l63_63733

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) - 1 / x

theorem exists_zero_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- This is just the Lean statement, no proof is provided
  sorry

end exists_zero_in_interval_l63_63733


namespace ernie_can_make_circles_l63_63333

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end ernie_can_make_circles_l63_63333


namespace quadratic_equation_l63_63870

theorem quadratic_equation (p q : ℝ) 
  (h1 : p^2 + 9 * q^2 + 3 * p - p * q = 30)
  (h2 : p - 5 * q - 8 = 0) : 
  p^2 - p - 6 = 0 :=
by sorry

end quadratic_equation_l63_63870


namespace eccentricity_of_hyperbola_l63_63810

-- Definitions and conditions
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a > b) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def regular_hexagon_side_length (a b c : ℝ) : Prop :=
  2 * a = (Real.sqrt 3 + 1) * c

-- Goal: Prove the eccentricity of the hyperbola
theorem eccentricity_of_hyperbola (a b c : ℝ) (x y : ℝ) :
  hyperbola x y a b →
  regular_hexagon_side_length a b c →
  2 * a = (Real.sqrt 3 + 1) * c →
  c ≠ 0 →
  a ≠ 0 →
  b ≠ 0 →
  (c / a = Real.sqrt 3 + 1) :=
by
  intros h_hyp h_hex h_eq h_c_ne_zero h_a_ne_zero h_b_ne_zero
  sorry -- Proof goes here

end eccentricity_of_hyperbola_l63_63810


namespace george_earnings_after_deductions_l63_63192

noncomputable def george_total_earnings : ℕ := 35 + 12 + 20 + 21

noncomputable def tax_deduction (total_earnings : ℕ) : ℚ := total_earnings * 0.10

noncomputable def uniform_fee : ℚ := 15

noncomputable def final_earnings (total_earnings : ℕ) (tax_deduction : ℚ) (uniform_fee : ℚ) : ℚ :=
  total_earnings - tax_deduction - uniform_fee

theorem george_earnings_after_deductions : 
  final_earnings george_total_earnings (tax_deduction george_total_earnings) uniform_fee = 64.2 := 
  by
  sorry

end george_earnings_after_deductions_l63_63192


namespace compare_exponents_l63_63799

theorem compare_exponents (n : ℕ) (hn : n > 8) :
  let a := Real.sqrt n
  let b := Real.sqrt (n + 1)
  a^b > b^a :=
sorry

end compare_exponents_l63_63799


namespace length_of_LN_l63_63420

theorem length_of_LN 
  (sinN : ℝ)
  (LM LN : ℝ)
  (h1 : sinN = 3 / 5)
  (h2 : LM = 20)
  (h3 : sinN = LM / LN) :
  LN = 100 / 3 :=
by
  sorry

end length_of_LN_l63_63420


namespace base_b_square_of_integer_l63_63174

theorem base_b_square_of_integer (b : ℕ) (h : b > 4) : ∃ n : ℕ, (n * n) = b^2 + 4 * b + 4 :=
by 
  sorry

end base_b_square_of_integer_l63_63174


namespace percentage_markup_l63_63123

theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 4800) (h₂ : cost_price = 3840) :
  (selling_price - cost_price) / cost_price * 100 = 25 :=
by
  sorry

end percentage_markup_l63_63123


namespace problem_solution_l63_63097

variables {a b c : ℝ}

theorem problem_solution (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^3 * b^3 / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  a^3 * c^3 / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  b^3 * c^3 / ((b^3 - a^2 * c) * (c^3 - a^2 * b))) = 1 :=
sorry

end problem_solution_l63_63097


namespace length_of_c_l63_63689

theorem length_of_c (A B C : ℝ) (a b c : ℝ) (h1 : (π / 3) - A = B) (h2 : a = 3) (h3 : b = 5) : c = 7 :=
sorry

end length_of_c_l63_63689


namespace find_mode_l63_63266

def scores : List ℕ :=
  [105, 107, 111, 111, 112, 112, 115, 118, 123, 124, 124, 126, 127, 129, 129, 129, 130, 130, 130, 130, 131, 140, 140, 140, 140]

def mode (ls : List ℕ) : ℕ :=
  ls.foldl (λmodeScore score => if ls.count score > ls.count modeScore then score else modeScore) 0

theorem find_mode :
  mode scores = 130 :=
by
  sorry

end find_mode_l63_63266


namespace square_difference_l63_63969

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l63_63969


namespace perp_tangent_line_equation_l63_63185

theorem perp_tangent_line_equation :
  ∃ a b c : ℝ, (∀ x y : ℝ, 3 * x + y + 2 = 0 ↔ y = -3 * x - 2) ∧
               (∀ x y : ℝ, (2 * x - 6 * y + 1 = 0) → (y = -(1/3) * x + 1/6)) ∧
               (∀ x : ℝ, y = x^3 + 3 * x^2 - 1 → derivative y at x = -3) ∧
               (∃ x : ℝ, 3 * x^2 + 6 * x = -3) :=
sorry

end perp_tangent_line_equation_l63_63185


namespace suitable_value_for_x_evaluates_to_neg1_l63_63561

noncomputable def given_expression (x : ℝ) : ℝ :=
  (x^3 + 2 * x^2) / (x^2 - 4 * x + 4) / (4 * x + 8) - 1 / (x - 2)

theorem suitable_value_for_x_evaluates_to_neg1 : 
  given_expression (-6) = -1 :=
by
  sorry

end suitable_value_for_x_evaluates_to_neg1_l63_63561


namespace intersection_correct_l63_63906

open Set

def M : Set ℤ := {0, 1, 2, -1}
def N : Set ℤ := {0, 1, 2, 3}

theorem intersection_correct : M ∩ N = {0, 1, 2} :=
by 
  -- Proof omitted
  sorry

end intersection_correct_l63_63906


namespace archer_total_fish_caught_l63_63769

noncomputable def total_fish_caught (initial : ℕ) (second_extra : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := initial + second_extra
  let total_after_two_rounds := initial + second_round
  let third_round := second_round + (third_round_percentage * second_round) / 100
  total_after_two_rounds + third_round

theorem archer_total_fish_caught :
  total_fish_caught 8 12 60 = 60 :=
by
  -- Theorem statement to prove the total fish caught equals 60 given the conditions.
  sorry

end archer_total_fish_caught_l63_63769


namespace smallest_degree_measure_for_WYZ_l63_63096

def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100
def angle_WYZ : ℝ := angle_XYZ - angle_XYW

theorem smallest_degree_measure_for_WYZ : angle_WYZ = 30 :=
by
  sorry

end smallest_degree_measure_for_WYZ_l63_63096


namespace solve_linear_system_l63_63728

theorem solve_linear_system (x y a : ℝ) (h1 : 4 * x + 3 * y = 1) (h2 : a * x + (a - 1) * y = 3) (hxy : x = y) : a = 11 :=
by
  sorry

end solve_linear_system_l63_63728


namespace paul_packed_total_toys_l63_63245

def small_box_small_toys : ℕ := 8
def medium_box_medium_toys : ℕ := 12
def large_box_large_toys : ℕ := 7
def large_box_small_toys : ℕ := 3
def small_box_medium_toys : ℕ := 5

def small_box : ℕ := small_box_small_toys + small_box_medium_toys
def medium_box : ℕ := medium_box_medium_toys
def large_box : ℕ := large_box_large_toys + large_box_small_toys

def total_toys : ℕ := small_box + medium_box + large_box

theorem paul_packed_total_toys : total_toys = 35 :=
by sorry

end paul_packed_total_toys_l63_63245


namespace no_such_convex_hexagon_exists_l63_63924

theorem no_such_convex_hexagon_exists :
  ¬ ∃ (A B C D E F M : ℝ × ℝ), 
    (∃ (hABCDEF : convex_hull ℝ ({A, B, C, D, E, F})),
     (dist M A < 1) ∧ (dist M B < 1) ∧ (dist M C < 1) ∧ (dist M D < 1) ∧ (dist M E < 1) ∧ (dist M F < 1) ∧
     (dist A B > 1) ∧ (dist B C > 1) ∧ (dist C D > 1) ∧ (dist D E > 1) ∧ (dist E F > 1) ∧ (dist F A > 1)) :=
begin
  sorry
end

end no_such_convex_hexagon_exists_l63_63924


namespace smaller_octagon_area_fraction_l63_63533

noncomputable def ratio_of_areas_of_octagons (A B C D E F G H : ℝ) : ℝ :=
  -- Assume A B C D E F G H represent vertices of a regular octagon
  let larger_octagon_area := sorry in  -- Compute area of the larger octagon based on its geometric properties
  let smaller_octagon_area := sorry in -- Compute area of the smaller octagon based on its geometric properties and the midpoints
  smaller_octagon_area / larger_octagon_area

theorem smaller_octagon_area_fraction (A B C D E F G H : ℝ) 
  (h_regular : regular_octagon A B C D E F G H)
  (h_midpoints : smaller_octagon_created_by_midpoints A B C D E F G H)
  (h_angle_center : angle_at_center A B C D E F G H = 45) :
  ratio_of_areas_of_octagons A B C D E F G H = 1 / 2 :=
by
  sorry

end smaller_octagon_area_fraction_l63_63533


namespace factorial_expression_simplification_l63_63033

theorem factorial_expression_simplification : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 1 := 
by 
  sorry

end factorial_expression_simplification_l63_63033


namespace probability_A_or_B_selected_l63_63506

open Finset

-- Constants representing the students
inductive Student : Type
| A | B | C | D

-- The set of all students
def students : Finset Student := {Student.A, Student.B, Student.C, Student.D}

-- The event of selecting exactly two students
def event (s1 s2 : Student) : Finset Student := {s1, s2}

-- Define a function to calculate combinations
def comb (n k : ℕ) : ℕ := (n.choose k)

-- Main theorem statement
theorem probability_A_or_B_selected : 
  (comb 2 1 * comb 2 1 : ℚ) / comb 4 2 = 2 / 3 := by
  sorry

end probability_A_or_B_selected_l63_63506


namespace ernie_can_make_circles_l63_63336

theorem ernie_can_make_circles :
  ∀ (boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle : ℕ),
  Ali_circles = 5 →
  Ali_boxes_per_circle = 8 →
  Ernie_boxes_per_circle = 10 →
  boxes_initial = 80 →
  ((boxes_initial - Ali_circles * Ali_boxes_per_circle) / Ernie_boxes_per_circle) = 4 :=
by
  intros boxes_initial Ali_circles Ali_boxes_per_circle Ernie_boxes_per_circle 
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end ernie_can_make_circles_l63_63336


namespace area_of_square_II_l63_63534

theorem area_of_square_II {a b : ℝ} (h : a > b) (d : ℝ) (h1 : d = a - b)
    (A1_A : ℝ) (h2 : A1_A = (a - b)^2 / 2) (A2_A : ℝ) (h3 : A2_A = 3 * A1_A) :
  A2_A = 3 * (a - b)^2 / 2 := by
  sorry

end area_of_square_II_l63_63534


namespace percentage_error_is_94_l63_63479

theorem percentage_error_is_94 (x : ℝ) (hx : 0 < x) :
  let correct_result := 4 * x
  let error_result := x / 4
  let error := |correct_result - error_result|
  let percentage_error := (error / correct_result) * 100
  percentage_error = 93.75 := by
    sorry

end percentage_error_is_94_l63_63479


namespace repeating_decimal_as_fraction_l63_63642

-- Define the repeating decimal 0.36666... as a real number
def repeating_decimal : ℝ := 0.366666666666...

-- State the theorem to express the repeating decimal as a fraction
theorem repeating_decimal_as_fraction : repeating_decimal = (11 : ℝ) / 30 := 
sorry

end repeating_decimal_as_fraction_l63_63642


namespace third_candidate_votes_l63_63130

theorem third_candidate_votes
  (total_votes : ℝ)
  (votes_for_two_candidates : ℝ)
  (winning_percentage : ℝ)
  (H1 : votes_for_two_candidates = 4636 + 11628)
  (H2 : winning_percentage = 67.21387283236994 / 100)
  (H3 : total_votes = votes_for_two_candidates / (1 - winning_percentage)) :
  (total_votes - votes_for_two_candidates) = 33336 :=
by
  sorry

end third_candidate_votes_l63_63130


namespace g_29_eq_27_l63_63570

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ x : ℝ, g (x + g x) = 3 * g x
axiom initial_condition : g 2 = 9

theorem g_29_eq_27 : g 29 = 27 := by
  sorry

end g_29_eq_27_l63_63570


namespace circle_condition_l63_63684

theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - 2*m*x - 2*m*y + 2*m^2 + m - 1 = 0) →
  m < 1 :=
sorry

end circle_condition_l63_63684


namespace solve_for_n_l63_63361

-- Define the problem statement
theorem solve_for_n : ∃ n : ℕ, (3 * n^2 + n = 219) ∧ (n = 9) := 
sorry

end solve_for_n_l63_63361


namespace inequality_proof_l63_63553

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (habc : a * b * c = 1)

theorem inequality_proof :
  (a + 1 / b)^2 + (b + 1 / c)^2 + (c + 1 / a)^2 ≥ 3 * (a + b + c + 1) :=
by
  sorry

end inequality_proof_l63_63553


namespace ben_total_distance_walked_l63_63490

-- Definitions based on conditions
def walking_speed : ℝ := 4  -- 4 miles per hour.
def total_time : ℝ := 2  -- 2 hours.
def break_time : ℝ := 0.25  -- 0.25 hours (15 minutes).

-- Proof goal: Prove that the total distance walked is 7.0 miles.
theorem ben_total_distance_walked : (walking_speed * (total_time - break_time) = 7.0) :=
by
  sorry

end ben_total_distance_walked_l63_63490


namespace initial_speed_increase_l63_63473

variables (S : ℝ) (P : ℝ)

/-- Prove that the initial percentage increase in speed P is 0.3 based on the given conditions: 
1. After the first increase by P, the speed becomes S + PS.
2. After the second increase by 10%, the final speed is (S + PS) * 1.10.
3. The total increase results in a speed that is 1.43 times the original speed S. -/
theorem initial_speed_increase (h : (S + P * S) * 1.1 = 1.43 * S) : P = 0.3 :=
sorry

end initial_speed_increase_l63_63473


namespace first_train_travels_more_l63_63443

-- Define the conditions
def velocity_first_train := 50 -- speed of the first train in km/hr
def velocity_second_train := 40 -- speed of the second train in km/hr
def distance_between_P_and_Q := 900 -- distance between P and Q in km

-- Problem statement
theorem first_train_travels_more :
  ∃ t : ℝ, (velocity_first_train * t + velocity_second_train * t = distance_between_P_and_Q)
          → (velocity_first_train * t - velocity_second_train * t = 100) :=
by sorry

end first_train_travels_more_l63_63443


namespace solution_set_f_gt_0_l63_63203

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2*x - 3 else - (x^2 - 2*x - 3)

theorem solution_set_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ (-3 < x ∧ x < 0)} :=
by
  sorry

end solution_set_f_gt_0_l63_63203


namespace geometric_sequence_product_l63_63826

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (a_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r)
  (root_condition : ∃ x y : ℝ, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 :=
sorry

end geometric_sequence_product_l63_63826


namespace fourth_power_mod_7_is_0_l63_63880

def fourth_smallest_prime := 7
def square_of_fourth_smallest_prime := fourth_smallest_prime ^ 2
def fourth_power_of_square := square_of_fourth_smallest_prime ^ 4

theorem fourth_power_mod_7_is_0 : 
  (fourth_power_of_square % 7) = 0 :=
by sorry

end fourth_power_mod_7_is_0_l63_63880


namespace arithmetic_sequence_term_2011_is_671st_l63_63318

theorem arithmetic_sequence_term_2011_is_671st:
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → (3 * n - 2 = 2011) → n = 671 :=
by 
  intros a1 d n ha1 hd h_eq;
  sorry

end arithmetic_sequence_term_2011_is_671st_l63_63318


namespace original_profit_percentage_l63_63326

-- Our definitions based on conditions.
variables (P S : ℝ)
-- Selling at double the price results in 260% profit
axiom h : (2 * S - P) / P * 100 = 260

-- Prove that the original profit percentage is 80%
theorem original_profit_percentage : (S - P) / P * 100 = 80 := 
sorry

end original_profit_percentage_l63_63326


namespace positive_difference_l63_63452

def a := 8^2
def b := a + a
def c := a * a
theorem positive_difference : ((b / 8) - (c / 8)) = 496 := by
  sorry

end positive_difference_l63_63452


namespace find_k_l63_63807

theorem find_k 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (hSn : ∀ n, S n = -2 + 2 * (1 / 3) ^ n) 
  (h_geom : ∀ n, a (n + 1) = a n * a 2 / a 1) :
  k = -2 :=
sorry

end find_k_l63_63807


namespace never_return_to_start_l63_63801

variable {City : Type} [MetricSpace City]

-- Conditions
variable (C : ℕ → City)  -- C is the sequence of cities
variable (dist : City → City → ℝ)  -- distance function
variable (furthest : City → City)  -- function that maps each city to the furthest city from it
variable (start : City)  -- initial city

-- Assuming C satisfies the properties in the problem statement
axiom initial_city : C 1 = start
axiom furthest_city_step : ∀ n, C (n + 1) = furthest (C n)
axiom no_ambiguity : ∀ c1 c2, (dist c1 (furthest c1) > dist c1 c2 ↔ c2 ≠ furthest c1)

-- Define the problem to prove that if C₁ ≠ C₃, then ∀ n ≥ 4, Cₙ ≠ C₁
theorem never_return_to_start (h : C 1 ≠ C 3) : ∀ n ≥ 4, C n ≠ start := sorry

end never_return_to_start_l63_63801


namespace hari_contribution_correct_l63_63102

-- Translate the conditions into definitions
def praveen_investment : ℝ := 3360
def praveen_duration : ℝ := 12
def hari_duration : ℝ := 7
def profit_ratio_praveen : ℝ := 2
def profit_ratio_hari : ℝ := 3

-- The target Hari's contribution that we need to prove
def hari_contribution : ℝ := 2160

-- Problem statement: prove Hari's contribution given the conditions
theorem hari_contribution_correct :
  (praveen_investment * praveen_duration) / (hari_contribution * hari_duration) = profit_ratio_praveen / profit_ratio_hari :=
by {
  -- The statement is set up to prove equality of the ratios as given in the problem
  sorry
}

end hari_contribution_correct_l63_63102


namespace tan_beta_l63_63059

open Real

variable (α β : ℝ)

theorem tan_beta (h₁ : tan α = 1/3) (h₂ : tan (α + β) = 1/2) : tan β = 1/7 :=
by sorry

end tan_beta_l63_63059


namespace terrell_weight_lifting_l63_63423

theorem terrell_weight_lifting (n : ℝ) : 
  (2 * 25 * 10 = 500) → (2 * 20 * n = 500) → n = 12.5 :=
by
  intros h1 h2
  sorry

end terrell_weight_lifting_l63_63423


namespace travel_with_decreasing_ticket_prices_l63_63084

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ℕ × ℕ → ℕ)
  (distinct_prices : ∀ ⦃x y : ℕ × ℕ⦄, x ≠ y → train_prices x ≠ train_prices y)
  (symmetric_prices : ∀ x y : ℕ, train_prices (x, y) = train_prices (y, x)) :
  ∃ start_city : ℕ, ∃ route : List (ℕ × ℕ),
    length route = n-1 ∧ 
    (∀ i, i < route.length - 1 → train_prices (route.nth i).get_or_else (0,0) > train_prices (route.nth (i+1)).get_or_else (0,0)) :=
begin
  -- Proof goes here
  sorry
end

end travel_with_decreasing_ticket_prices_l63_63084


namespace no_three_nat_numbers_with_sum_power_of_three_l63_63773

noncomputable def powers_of_3 (n : ℕ) : ℕ := 3^n

theorem no_three_nat_numbers_with_sum_power_of_three :
  ¬ ∃ (a b c : ℕ) (k m n : ℕ), a + b = powers_of_3 k ∧ b + c = powers_of_3 m ∧ c + a = powers_of_3 n :=
by
  sorry

end no_three_nat_numbers_with_sum_power_of_three_l63_63773


namespace max_value_quadratic_l63_63560

theorem max_value_quadratic (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : x * (1 - x) ≤ 1 / 4 :=
by
  sorry

end max_value_quadratic_l63_63560


namespace number_of_children_l63_63785

theorem number_of_children (total_oranges : ℕ) (oranges_per_child : ℕ) (h1 : oranges_per_child = 3) (h2 : total_oranges = 12) : total_oranges / oranges_per_child = 4 :=
by
  sorry

end number_of_children_l63_63785


namespace simplify_expression_l63_63167

theorem simplify_expression (a : ℝ) : a * (a - 3) = a^2 - 3 * a := 
by 
  sorry

end simplify_expression_l63_63167


namespace unit_digit_of_square_l63_63670

theorem unit_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 := sorry

end unit_digit_of_square_l63_63670


namespace imaginary_part_of_complex_num_l63_63064

def imaginary_unit : ℂ := Complex.I

noncomputable def complex_num : ℂ := 10 * imaginary_unit / (1 - 2 * imaginary_unit)

theorem imaginary_part_of_complex_num : complex_num.im = 2 := by
  sorry

end imaginary_part_of_complex_num_l63_63064


namespace min_value_inequality_l63_63841

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 3) :
  (1 / x) + (4 / y) + (9 / z) ≥ 12 := 
sorry

end min_value_inequality_l63_63841


namespace women_stockbrokers_2005_l63_63824

-- Define the context and conditions
def women_stockbrokers_2000 : ℕ := 10000
def percent_increase_2005 : ℕ := 100

-- Statement to prove the number of women stockbrokers in 2005
theorem women_stockbrokers_2005 : women_stockbrokers_2000 + women_stockbrokers_2000 * percent_increase_2005 / 100 = 20000 := by
  sorry

end women_stockbrokers_2005_l63_63824


namespace solve_quadratic_l63_63432

theorem solve_quadratic (x : ℝ) (h : x^2 - 4 = 0) : x = 2 ∨ x = -2 :=
by sorry

end solve_quadratic_l63_63432


namespace Aryan_owes_1200_l63_63774

variables (A K : ℝ) -- A represents Aryan's debt, K represents Kyro's debt

-- Condition 1: Aryan's debt is twice Kyro's debt
axiom condition1 : A = 2 * K

-- Condition 2: Aryan pays 60% of her debt
axiom condition2 : (0.60 * A) + (0.80 * K) = 1500 - 300

theorem Aryan_owes_1200 : A = 1200 :=
by
  sorry

end Aryan_owes_1200_l63_63774


namespace geom_seq_min_val_l63_63399

-- Definition of geometric sequence with common ratio q
def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Main theorem
theorem geom_seq_min_val (a : ℕ → ℝ) (q : ℝ) 
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_geom : geom_seq a q)
  (h_cond : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) :
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end geom_seq_min_val_l63_63399


namespace inequality_check_l63_63291

theorem inequality_check : (-1 : ℝ) / 3 < -1 / 5 := 
by 
  sorry

end inequality_check_l63_63291


namespace proof_x_minus_y_squared_l63_63976

-- Define the variables x and y
variables (x y : ℝ)

-- Assume the given conditions
def cond1 : (x + y)^2 = 64 := sorry
def cond2 : x * y = 12 := sorry

-- Formulate the main goal to prove
theorem proof_x_minus_y_squared : (x - y)^2 = 16 :=
by
  have hx_add_y_sq : (x + y)^2 = 64 := cond1
  have hxy : x * y = 12 := cond2
  -- Use the identities and the given conditions to prove the statement
  sorry

end proof_x_minus_y_squared_l63_63976


namespace paco_countertop_total_weight_l63_63850

theorem paco_countertop_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 :=
sorry

end paco_countertop_total_weight_l63_63850


namespace eccentricity_of_ellipse_equilateral_triangle_l63_63515

theorem eccentricity_of_ellipse_equilateral_triangle (c b a e : ℝ)
  (h1 : b = Real.sqrt (3 * c))
  (h2 : a = Real.sqrt (b^2 + c^2)) 
  (h3 : e = c / a) :
  e = 1 / 2 :=
by {
  sorry
}

end eccentricity_of_ellipse_equilateral_triangle_l63_63515


namespace log_ab_is_pi_l63_63894

open Real

noncomputable def log_ab (a b : ℝ) : ℝ :=
(log b) / (log a)

theorem log_ab_is_pi (a b : ℝ)  (ha_pos: 0 < a) (ha_ne_one: a ≠ 1) (hb_pos: 0 < b) 
  (cond1 : log (a ^ 3) = log (b ^ 6)) (cond2 : cos (π * log a) = 1) : log_ab a b = π :=
by
  sorry

end log_ab_is_pi_l63_63894


namespace range_of_a_l63_63376

-- Defining the propositions
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

-- Main theorem statement
theorem range_of_a (a : ℝ) : (¬(∃ x, p x) → ¬(∃ x, q x a)) → a < -3 :=
by
  sorry

end range_of_a_l63_63376


namespace calculate_expression_l63_63168

theorem calculate_expression :
  4 * Real.sqrt 24 * (Real.sqrt 6 / 8) / Real.sqrt 3 - 3 * Real.sqrt 3 = -Real.sqrt 3 :=
by
  sorry

end calculate_expression_l63_63168


namespace algebraic_expression_value_l63_63526

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : 2*a^2 - 4*a + 2023 = 2025 :=
sorry

end algebraic_expression_value_l63_63526


namespace no_such_hexagon_exists_l63_63930

-- Define a hexagon (set of six points in 2D space) and a point M
structure Hexagon (α : Type) := 
(vertices : fin 6 → α)

-- Define the property that all sides of the hexagon are greater than 1
def side_lengths_greater_than (h : Hexagon (ℝ × ℝ)) (d : ℝ) :=
∀ i : fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) > d

-- Define the property that the distance from a point M to any vertex is less than 1
def point_within_distance (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ) (d : ℝ) :=
∀ i : fin 6, dist M (h.vertices i) < d

theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon (ℝ × ℝ)) (M : ℝ × ℝ),
    side_lengths_greater_than h 1 ∧ 
    point_within_distance h M 1 :=
by sorry

end no_such_hexagon_exists_l63_63930


namespace ernie_circles_l63_63338

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l63_63338


namespace quadratic_eq_complete_square_l63_63899

theorem quadratic_eq_complete_square (x p q : ℝ) (h : 9 * x^2 - 54 * x + 63 = 0) 
(h_trans : (x + p)^2 = q) : p + q = -1 := sorry

end quadratic_eq_complete_square_l63_63899


namespace factorial_expression_l63_63031

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l63_63031


namespace wrapping_paper_fraction_l63_63547

theorem wrapping_paper_fraction (s l : ℚ) (h1 : 4 * s + 2 * l = 5 / 12) (h2 : l = 2 * s) :
  s = 5 / 96 ∧ l = 5 / 48 :=
by
  sorry

end wrapping_paper_fraction_l63_63547


namespace solve_quadratic_equation_solve_linear_factor_equation_l63_63563

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x^2 - 6 * x + 1 = 0 → (x = 3 - 2 * Real.sqrt 2 ∨ x = 3 + 2 * Real.sqrt 2) :=
by
  intro x
  intro h
  sorry

theorem solve_linear_factor_equation :
  ∀ (x : ℝ), x * (2 * x - 1) = 2 * (2 * x - 1) → (x = 1 / 2 ∨ x = 2) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_equation_solve_linear_factor_equation_l63_63563


namespace cos_pi_minus_2alpha_l63_63058

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin α = 2 / 3) : Real.cos (Real.pi - 2 * α) = -1 / 9 :=
by
  sorry

end cos_pi_minus_2alpha_l63_63058


namespace kamal_marks_physics_l63_63835

-- Define the marks in subjects
def marks_english := 66
def marks_mathematics := 65
def marks_chemistry := 62
def marks_biology := 75
def average_marks := 69
def number_of_subjects := 5

-- Calculate the total marks from the average
def total_marks := average_marks * number_of_subjects

-- Calculate the known total marks
def known_total_marks := marks_english + marks_mathematics + marks_chemistry + marks_biology

-- Define Kamal's marks in Physics
def marks_physics := total_marks - known_total_marks

-- Prove the marks in Physics are 77
theorem kamal_marks_physics : marks_physics = 77 := by
  sorry

end kamal_marks_physics_l63_63835


namespace total_sampled_students_is_80_l63_63224

-- Given conditions
variables (total_students num_freshmen num_sampled_freshmen : ℕ)
variables (total_students := 2400) (num_freshmen := 600) (num_sampled_freshmen := 20)

-- Define the proportion for stratified sampling.
def stratified_sampling (total_students num_freshmen num_sampled_freshmen total_sampled_students : ℕ) : Prop :=
  num_freshmen / total_students = num_sampled_freshmen / total_sampled_students

-- State the theorem: Prove the total number of students to be sampled from the entire school is 80.
theorem total_sampled_students_is_80 : ∃ n, stratified_sampling total_students num_freshmen num_sampled_freshmen n ∧ n = 80 := 
sorry

end total_sampled_students_is_80_l63_63224


namespace area_of_triangle_l63_63017

theorem area_of_triangle (x : ℝ) :
  let t1_area := 16
  let t2_area := 25
  let t3_area := 64
  let total_area_factor := t1_area + t2_area + t3_area
  let side_factor := 17 * 17
  ΔABC_area = side_factor * total_area_factor :=
by {
  -- Placeholder to complete the proof
  sorry
}

end area_of_triangle_l63_63017


namespace smallest_prime_after_seven_consecutive_nonprimes_l63_63682

theorem smallest_prime_after_seven_consecutive_nonprimes :
  ∃ p, p > 96 ∧ Nat.Prime p ∧ ∀ n, 90 ≤ n ∧ n ≤ 96 → ¬Nat.Prime n :=
by
  sorry

end smallest_prime_after_seven_consecutive_nonprimes_l63_63682


namespace total_amount_shared_l63_63675

theorem total_amount_shared (x y z : ℝ) (h1 : x = 1.25 * y) (h2 : y = 1.2 * z) (h3 : z = 400) :
  x + y + z = 1480 :=
by
  sorry

end total_amount_shared_l63_63675


namespace problem_A_value_l63_63523

theorem problem_A_value (x y A : ℝ) (h : (x + 2 * y) ^ 2 = (x - 2 * y) ^ 2 + A) : A = 8 * x * y :=
by {
    sorry
}

end problem_A_value_l63_63523


namespace problem1_problem2_l63_63005

-- Problem 1
theorem problem1 (a b : ℤ) (h1 : a = 4) (h2 : b = 5) : a - b = -1 := 
by {
  sorry
}

-- Problem 2
theorem problem2 (a b m n s : ℤ) (h1 : a + b = 0) (h2 : m * n = 1) (h3 : |s| = 3) :
  a + b + m * n + s = 4 ∨ a + b + m * n + s = -2 := 
by {
  sorry
}

end problem1_problem2_l63_63005


namespace original_average_l63_63114

theorem original_average (A : ℝ) (h : (10 * A = 70)) : A = 7 :=
sorry

end original_average_l63_63114


namespace trig_identity_solution_l63_63652

theorem trig_identity_solution
  (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 1 / 3) :
  Real.sin (4 * x) - 2 * Real.cos (3 * x) * Real.sin x = -7 / 9 :=
by
  sorry

end trig_identity_solution_l63_63652


namespace kite_initial_gain_percentage_l63_63616

noncomputable def initial_gain_percentage (MP CP : ℝ) : ℝ :=
  ((MP - CP) / CP) * 100

theorem kite_initial_gain_percentage :
  ∃ MP CP : ℝ,
    SP = 30 ∧
    SP = MP * 0.9 ∧
    1.035 * CP = SP ∧
    initial_gain_percentage MP CP = 15 :=
sorry

end kite_initial_gain_percentage_l63_63616


namespace dogs_food_consumption_l63_63874

def cups_per_meal_per_dog : ℝ := 1.5
def number_of_dogs : ℝ := 2
def meals_per_day : ℝ := 3
def cups_per_pound : ℝ := 2.25

theorem dogs_food_consumption : 
  ((cups_per_meal_per_dog * number_of_dogs) * meals_per_day) / cups_per_pound = 4 := 
by
  sorry

end dogs_food_consumption_l63_63874


namespace number_of_people_in_group_l63_63115

theorem number_of_people_in_group (n : ℕ) (h1 : 110 - 60 = 5 * n) : n = 10 :=
by 
  sorry

end number_of_people_in_group_l63_63115


namespace ab_fraction_l63_63574

theorem ab_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 9) (h2 : a * b = 20) : 
  (1 / a + 1 / b) = 9 / 20 := 
by 
  sorry

end ab_fraction_l63_63574


namespace intersection_of_M_N_l63_63199

-- Definitions of the sets M and N
def M : Set ℝ := { x | (x + 2) * (x - 1) < 0 }
def N : Set ℝ := { x | x + 1 < 0 }

-- Proposition stating that the intersection of M and N is { x | -2 < x < -1 }
theorem intersection_of_M_N : M ∩ N = { x : ℝ | -2 < x ∧ x < -1 } :=
  by
    sorry

end intersection_of_M_N_l63_63199


namespace quadratic_root_range_l63_63816

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end quadratic_root_range_l63_63816


namespace students_not_taken_test_l63_63844

theorem students_not_taken_test 
  (num_enrolled : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (answered_both : ℕ) 
  (H_num_enrolled : num_enrolled = 40) 
  (H_answered_q1 : answered_q1 = 30) 
  (H_answered_q2 : answered_q2 = 29) 
  (H_answered_both : answered_both = 29) : 
  num_enrolled - (answered_q1 + answered_q2 - answered_both) = 10 :=
by {
  sorry
}

end students_not_taken_test_l63_63844


namespace remainder_is_4_over_3_l63_63597

noncomputable def original_polynomial (z : ℝ) : ℝ := 3 * z ^ 3 - 4 * z ^ 2 - 14 * z + 3
noncomputable def divisor (z : ℝ) : ℝ := 3 * z + 5
noncomputable def quotient (z : ℝ) : ℝ := z ^ 2 - 3 * z + 1 / 3

theorem remainder_is_4_over_3 :
  ∃ r : ℝ, original_polynomial z = divisor z * quotient z + r ∧ r = 4 / 3 :=
sorry

end remainder_is_4_over_3_l63_63597


namespace employee_B_payment_l63_63587

theorem employee_B_payment (total_payment A_payment B_payment : ℝ) 
    (h1 : total_payment = 450) 
    (h2 : A_payment = 1.5 * B_payment) 
    (h3 : total_payment = A_payment + B_payment) : 
    B_payment = 180 := 
by
  sorry

end employee_B_payment_l63_63587


namespace determinant_of_matrix_A_l63_63936

variable (y : ℝ)

def matrix_A : Matrix (Fin 3) (Fin 3) ℝ :=
  !![
    [y + 2, 2y, 2y],
    [2y, y + 2, 2y],
    [2y, 2y, y + 2]
  ]

theorem determinant_of_matrix_A : Matrix.det (matrix_A y) = 5 * y^3 - 10 * y^2 + 12 * y + 8 := by
  sorry

end determinant_of_matrix_A_l63_63936


namespace difference_between_a_b_l63_63957

theorem difference_between_a_b (a b : ℝ) (d : ℝ) : 
  (a - b = d) → (a ^ 2 + b ^ 2 = 150) → (a * b = 25) → d = 10 :=
by
  sorry

end difference_between_a_b_l63_63957


namespace greatest_int_satisfying_inequality_l63_63446

theorem greatest_int_satisfying_inequality : ∃ n : ℤ, (∀ m : ℤ, m^2 - 13 * m + 40 ≤ 0 → m ≤ n) ∧ n = 8 := 
sorry

end greatest_int_satisfying_inequality_l63_63446


namespace find_digit_A_l63_63741

theorem find_digit_A : ∃ A : ℕ, A < 10 ∧ (200 + 10 * A + 4) % 13 = 0 ∧ A = 7 :=
by
  sorry

end find_digit_A_l63_63741


namespace triangular_pyramid_height_l63_63222

noncomputable def pyramid_height (a b c h : ℝ) : Prop :=
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2

theorem triangular_pyramid_height {a b c h : ℝ} (h_gt_0 : h > 0) (a_gt_0 : a > 0) (b_gt_0 : b > 0) (c_gt_0 : c > 0) :
  pyramid_height a b c h := by
  sorry

end triangular_pyramid_height_l63_63222


namespace range_of_k_l63_63982

/-- If the function y = (k + 1) * x is decreasing on the entire real line, then k < -1. -/
theorem range_of_k (k : ℝ) (h : ∀ x y : ℝ, x < y → (k + 1) * x > (k + 1) * y) : k < -1 :=
sorry

end range_of_k_l63_63982


namespace polynomial_perfect_square_trinomial_l63_63076

theorem polynomial_perfect_square_trinomial (k : ℝ) :
  (∀ x : ℝ, 4 * x^2 + 2 * k * x + 25 = (2 * x + 5) * (2 * x + 5)) → (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_trinomial_l63_63076


namespace find_numbers_l63_63305

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l63_63305


namespace product_of_numbers_l63_63434

-- Definitions of the conditions
variables (x y : ℝ)

-- The conditions themselves
def cond1 : Prop := x + y = 20
def cond2 : Prop := x^2 + y^2 = 200

-- Statement of the proof problem
theorem product_of_numbers (h1 : cond1 x y) (h2 : cond2 x y) : x * y = 100 :=
sorry

end product_of_numbers_l63_63434


namespace Kayla_total_items_l63_63702

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l63_63702


namespace find_x_l63_63942

theorem find_x (x : ℕ) (h1 : (31 : ℕ) ≤ 100) (h2 : (58 : ℕ) ≤ 100) (h3 : (98 : ℕ) ≤ 100) (h4 : 0 < x) (h5 : x ≤ 100)
               (h_mean_mode : ((31 + 58 + 98 + x + x) / 5 : ℚ) = 1.5 * x) : x = 34 :=
by
  sorry

end find_x_l63_63942


namespace correct_operation_l63_63461

theorem correct_operation (a b : ℝ) : 
  (a+2)*(a-2) = a^2 - 4 :=
by
  sorry

end correct_operation_l63_63461


namespace male_population_half_total_l63_63575

theorem male_population_half_total (total_population : ℕ) (segments : ℕ) (male_segment : ℕ) :
  total_population = 800 ∧ segments = 4 ∧ male_segment = 1 ∧ male_segment = segments / 2 →
  total_population / 2 = 400 :=
by
  intro h
  sorry

end male_population_half_total_l63_63575


namespace elective_course_schemes_l63_63483

theorem elective_course_schemes : Nat.choose 4 2 = 6 := by
  sorry

end elective_course_schemes_l63_63483


namespace percentage_of_z_equals_39_percent_of_y_l63_63388

theorem percentage_of_z_equals_39_percent_of_y
    (x y z : ℝ)
    (h1 : y = 0.75 * x)
    (h2 : z = 0.65 * x)
    (P : ℝ)
    (h3 : (P / 100) * z = 0.39 * y) :
    P = 45 :=
by sorry

end percentage_of_z_equals_39_percent_of_y_l63_63388


namespace range_of_a_l63_63657

theorem range_of_a {a : ℝ} 
  (hA : ∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))
  (hB : ∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))
  (hC : (a^2 + 1) / (2 * a) > 0)
  (hOnlyOneFalse : (¬(∀ x, (ax - 1) * (x - a) > 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬(∀ x, (ax - 1) * (x - a) < 0 ↔ (x < a ∨ x > 1 / a))) ∨ 
                   (¬((a^2 + 1) / (2 * a) > 0))):
  0 < a ∧ a < 1 := 
sorry

end range_of_a_l63_63657


namespace value_of_a_l63_63516

theorem value_of_a (a : ℝ) :
  ((abs ((1) - (2) + a)) = 1) ↔ (a = 0 ∨ a = 2) :=
by
  sorry

end value_of_a_l63_63516


namespace total_students_at_competition_l63_63131

variable (K H N : ℕ)

theorem total_students_at_competition
  (H_eq : H = (3/5) * K)
  (N_eq : N = 2 * (K + H))
  (total_students : K + H + N = 240) :
  K + H + N = 240 :=
by
  sorry

end total_students_at_competition_l63_63131


namespace largest_possible_s_l63_63231

-- Definitions based on the problem conditions
def polygon_interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

-- Main theorem statement
theorem largest_possible_s (r s : ℕ) (h₁ : r ≥ s) (h₂ : s ≥ 3) 
  (h₃ : polygon_interior_angle r = (5/4) * polygon_interior_angle s) : s ≤ 102 :=
by
  sorry

#eval largest_possible_s -- This line is to test build without actual function call, should be removed in real usage.

end largest_possible_s_l63_63231


namespace geometric_sequence_properties_l63_63948

-- Define the geometric sequence with the given conditions
def geometric_sequence (a₃ : ℕ → ℝ) := (a₃ 3 = 12) ∧ (a₃ 8 = 3 / 8)

-- Define the general formula for the n-th term of a geometric sequence
def general_term (a₁ q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

-- Define the sum of the first n terms of a geometric sequence
def sum_of_geometric_sequence (a₁ q : ℝ) (S_n : ℕ → ℝ) (n : ℕ) : Prop :=
  S_n n = a₁ * (1 - q^n) / (1 - q)

-- The proof problem statement
theorem geometric_sequence_properties : 
  ∃ a₁ q S_n : ℝ,
  geometric_sequence (λ n, general_term a₁ q n)
  →
  ∀ n, (general_term a₁ q n = 48 * (1 / 2)^(n - 1)) 
       ∧ (S_n n = 93 → n = 5) :=
begin
  sorry
end

end geometric_sequence_properties_l63_63948


namespace heather_bicycling_time_l63_63387

theorem heather_bicycling_time (distance speed : ℕ) (h1 : distance = 96) (h2 : speed = 6) : 
(distance / speed) = 16 := by
  sorry

end heather_bicycling_time_l63_63387


namespace proof_part_1_proof_part_2_l63_63649

variable {α : ℝ}

/-- Given tan(α) = 3, prove
  (1) (3 * sin(α) + 2 * cos(α))/(sin(α) - 4 * cos(α)) = -11 -/
theorem proof_part_1
  (h : Real.tan α = 3) :
  (3 * Real.sin α + 2 * Real.cos α) / (Real.sin α - 4 * Real.cos α) = -11 := 
by
  sorry

/-- Given tan(α) = 3, prove
  (2) (5 * cos^2(α) - 3 * sin^2(α))/(1 + sin^2(α)) = -11/5 -/
theorem proof_part_2
  (h : Real.tan α = 3) :
  (5 * (Real.cos α)^2 - 3 * (Real.sin α)^2) / (1 + (Real.sin α)^2) = -11 / 5 :=
by
  sorry

end proof_part_1_proof_part_2_l63_63649


namespace dan_has_13_limes_l63_63783

theorem dan_has_13_limes (picked_limes : ℕ) (given_limes : ℕ) (h1 : picked_limes = 9) (h2 : given_limes = 4) : 
  picked_limes + given_limes = 13 := 
by
  sorry

end dan_has_13_limes_l63_63783


namespace no_integer_pairs_satisfy_equation_l63_63360

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), m^3 + 6 * m^2 + 5 * m ≠ 27 * n^3 + 27 * n^2 + 9 * n + 1 :=
by
  intros m n
  sorry

end no_integer_pairs_satisfy_equation_l63_63360


namespace chairs_carried_per_trip_l63_63408

theorem chairs_carried_per_trip (x : ℕ) (friends : ℕ) (trips : ℕ) (total_chairs : ℕ) 
  (h1 : friends = 4) (h2 : trips = 10) (h3 : total_chairs = 250) 
  (h4 : 5 * (trips * x) = total_chairs) : x = 5 :=
by sorry

end chairs_carried_per_trip_l63_63408


namespace rectangle_area_l63_63260

theorem rectangle_area (b l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 112) : l * b = 588 := by
  sorry

end rectangle_area_l63_63260


namespace gibbs_inequality_l63_63234

noncomputable section

open BigOperators

variable {r : ℕ} (p q : Fin r → ℝ)

/-- (p_i) is a probability distribution -/
def isProbabilityDistribution (p : Fin r → ℝ) : Prop :=
  (∀ i, 0 ≤ p i) ∧ (∑ i, p i = 1)

/-- -\sum_{i=1}^{r} p_i \ln p_i \leqslant -\sum_{i=1}^{r} p_i \ln q_i for probability distributions p and q -/
theorem gibbs_inequality
  (hp : isProbabilityDistribution p)
  (hq : isProbabilityDistribution q) :
  -∑ i, p i * Real.log (p i) ≤ -∑ i, p i * Real.log (q i) := 
by
  sorry

end gibbs_inequality_l63_63234


namespace star_computation_l63_63784

-- Define the operation ☆
def star (m n : Int) := m^2 - m * n + n

-- Define the main proof problem
theorem star_computation :
  star 3 4 = 1 ∧ star (-1) (star 2 (-3)) = 15 := 
by
  sorry

end star_computation_l63_63784


namespace min_max_transformation_a_min_max_transformation_b_l63_63412

theorem min_max_transformation_a {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≥ a) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≤ b) :=
sorry

theorem min_max_transformation_b {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≥ -b) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≤ -a) :=
sorry

end min_max_transformation_a_min_max_transformation_b_l63_63412


namespace no_convex_hexagon_with_point_M_l63_63929

open Real EuclideanGeometry

-- We define a convex hexagon and the condition that all its sides are greater than 1
structure ConvexHexagon where
  vertices : Fin₆ → EuclideanSpace ℝ (Fin 2)
  is_convex : ConvexHull set.range vertices = set.range vertices
  all_sides_greater_than_one : ∀ i : Fin₆, dist (vertices i) (vertices ((i + 1) % 6)) > 1

-- The main theorem to be proved
theorem no_convex_hexagon_with_point_M :
  ¬∃ (H : ConvexHexagon) (M : EuclideanSpace ℝ (Fin 2)),
    (∀ i : Fin₆, dist M (H.vertices i) < 1) := 
  sorry

end no_convex_hexagon_with_point_M_l63_63929


namespace books_more_than_figures_l63_63405

-- Definitions of initial conditions
def initial_action_figures := 2
def initial_books := 10
def added_action_figures := 4

-- Problem statement to prove
theorem books_more_than_figures :
  initial_books - (initial_action_figures + added_action_figures) = 4 :=
by
  -- Proof goes here
  sorry

end books_more_than_figures_l63_63405


namespace find_numbers_l63_63303

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l63_63303


namespace simplify_expression_l63_63383

theorem simplify_expression (x y : ℝ) (h : x - 3 * y = 4) : (x - 3 * y) ^ 2 + 2 * x - 6 * y - 10 = 14 := by
  sorry

end simplify_expression_l63_63383


namespace max_marks_l63_63761

theorem max_marks (M : ℝ) : 0.33 * M = 59 + 40 → M = 300 :=
by
  sorry

end max_marks_l63_63761


namespace problem_part1_problem_part2_l63_63953

open Real

variables {α : ℝ}

theorem problem_part1 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) :
  sin α * cos α = 3 / 10 := sorry

theorem problem_part2 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) (h3 : sin α * cos α = 3 / 10) :
  sin α + cos α = - (2 * sqrt 10 / 5) := sorry

end problem_part1_problem_part2_l63_63953


namespace minimum_value_l63_63999

noncomputable def minValue (x y : ℝ) : ℝ := (2 / x) + (3 / y)

theorem minimum_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 20) : minValue x y = 1 :=
sorry

end minimum_value_l63_63999


namespace second_order_derivative_l63_63362

-- Define the parameterized functions x and y
noncomputable def x (t : ℝ) : ℝ := 1 / t
noncomputable def y (t : ℝ) : ℝ := 1 / (1 + t ^ 2)

-- Define the second-order derivative of y with respect to x
noncomputable def d2y_dx2 (t : ℝ) : ℝ := (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3

-- Prove the relationship based on given conditions
theorem second_order_derivative :
  ∀ t : ℝ, (∃ x y : ℝ, x = 1 / t ∧ y = 1 / (1 + t ^ 2)) → 
    (d2y_dx2 t) = (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3 :=
by
  intros t ht
  -- Proof omitted
  sorry

end second_order_derivative_l63_63362


namespace no_solution_x_l63_63110

theorem no_solution_x : ¬ ∃ x : ℝ, x * (x - 1) * (x - 2) + (100 - x) * (99 - x) * (98 - x) = 0 := 
sorry

end no_solution_x_l63_63110


namespace doug_fires_l63_63932

theorem doug_fires (D : ℝ) (Kai_fires : ℝ) (Eli_fires : ℝ) 
    (hKai : Kai_fires = 3 * D)
    (hEli : Eli_fires = 1.5 * D)
    (hTotal : D + Kai_fires + Eli_fires = 110) : 
  D = 20 := 
by
  sorry

end doug_fires_l63_63932


namespace number_of_tables_l63_63353

-- Define the conditions
def seats_per_table : ℕ := 8
def total_seating_capacity : ℕ := 32

-- Define the main statement using the conditions
theorem number_of_tables : total_seating_capacity / seats_per_table = 4 := by
  sorry

end number_of_tables_l63_63353


namespace count_integers_expression_negative_l63_63505

theorem count_integers_expression_negative :
  ∃ n : ℕ, n = 4 ∧ 
  ∀ x : ℤ, x^4 - 60 * x^2 + 144 < 0 → n = 4 := by
  -- Placeholder for the proof
  sorry

end count_integers_expression_negative_l63_63505


namespace square_side_length_l63_63328

theorem square_side_length (P : ℝ) (s : ℝ) (h1 : P = 36) (h2 : P = 4 * s) : s = 9 := 
by sorry

end square_side_length_l63_63328


namespace garrett_cats_count_l63_63241

def number_of_cats_sheridan : ℕ := 11
def difference_in_cats : ℕ := 13

theorem garrett_cats_count (G : ℕ) (h : G - number_of_cats_sheridan = difference_in_cats) : G = 24 :=
by
  sorry

end garrett_cats_count_l63_63241


namespace rectangle_inscribed_circle_l63_63615

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem rectangle_inscribed_circle (a b : ℝ) (Ha : a = 7) (Hb : b = 24) :
  let d := rectangle_diagonal a b in
  let circumference := real.pi * d in
  let area := a * b in
  d = 25 ∧ circumference = 25 * real.pi ∧ area = 168 :=
by
  sorry

end rectangle_inscribed_circle_l63_63615


namespace hallie_made_100_per_painting_l63_63965

-- Define conditions
def num_paintings : ℕ := 3
def total_money_made : ℕ := 300

-- Define the goal
def money_per_painting : ℕ := total_money_made / num_paintings

theorem hallie_made_100_per_painting :
  money_per_painting = 100 :=
sorry

end hallie_made_100_per_painting_l63_63965


namespace orlie_age_l63_63722

theorem orlie_age (O R : ℕ) (h1 : R = 9) (h2 : R = (3 * O) / 4)
  (h3 : R - 4 = ((O - 4) / 2) + 1) : O = 12 :=
by
  sorry

end orlie_age_l63_63722


namespace non_poli_sci_gpa_below_or_eq_3_is_10_l63_63849

-- Definitions based on conditions
def total_applicants : ℕ := 40
def poli_sci_majors : ℕ := 15
def gpa_above_3 : ℕ := 20
def poli_sci_gpa_above_3 : ℕ := 5

-- Derived conditions from the problem
def poli_sci_gpa_below_or_eq_3 : ℕ := poli_sci_majors - poli_sci_gpa_above_3
def total_gpa_below_or_eq_3 : ℕ := total_applicants - gpa_above_3
def non_poli_sci_gpa_below_or_eq_3 : ℕ := total_gpa_below_or_eq_3 - poli_sci_gpa_below_or_eq_3

-- Statement to be proven
theorem non_poli_sci_gpa_below_or_eq_3_is_10 : non_poli_sci_gpa_below_or_eq_3 = 10 := by
  sorry

end non_poli_sci_gpa_below_or_eq_3_is_10_l63_63849


namespace Archer_catch_total_fish_l63_63771

noncomputable def ArcherFishProblem : ℕ :=
  let firstRound := 8
  let secondRound := firstRound + 12
  let thirdRound := secondRound + (secondRound * 60 / 100)
  firstRound + secondRound + thirdRound

theorem Archer_catch_total_fish : ArcherFishProblem = 60 := by
  sorry

end Archer_catch_total_fish_l63_63771


namespace kevin_watermelons_l63_63228

theorem kevin_watermelons (w1 w2 w_total : ℝ) (h1 : w1 = 9.91) (h2 : w2 = 4.11) (h_total : w_total = 14.02) : 
  w1 + w2 = w_total → 2 = 2 :=
by
  sorry

end kevin_watermelons_l63_63228


namespace equivalent_form_l63_63386

theorem equivalent_form (x y : ℝ) (h : y = x + 1/x) :
  (x^4 + x^3 - 3*x^2 + x + 2 = 0) ↔ (x^2 * (y^2 + y - 5) = 0) :=
sorry

end equivalent_form_l63_63386


namespace boarders_joined_l63_63576

theorem boarders_joined (initial_boarders : ℕ) (initial_day_scholars : ℕ)
  (final_boarders : ℕ) (x : ℕ)
  (ratio_initial : initial_boarders * 16 = initial_day_scholars * 7)
  (ratio_final : final_boarders * 2 = initial_day_scholars)
  (final_boarders_eq : final_boarders = initial_boarders + x)
  (initial_boarders_val : initial_boarders = 560)
  (initial_day_scholars_val : initial_day_scholars = 1280)
  (final_boarders_val : final_boarders = 640) :
  x = 80 :=
by
  sorry

end boarders_joined_l63_63576


namespace find_two_numbers_l63_63302

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l63_63302


namespace beacon_population_l63_63426

variables (Richmond Victoria Beacon : ℕ)

theorem beacon_population :
  (Richmond = Victoria + 1000) →
  (Victoria = 4 * Beacon) →
  (Richmond = 3000) →
  (Beacon = 500) :=
by
  intros h1 h2 h3
  sorry

end beacon_population_l63_63426


namespace find_numbers_l63_63304

theorem find_numbers (S P : ℝ) (x₁ x₂ y₁ y₂ : ℝ) (h₁ : x₁ + y₁ = S) (h₂ : x₁ * y₁ = P) (h₃ : x₂ + y₂ = S) (h₄ : x₂ * y₂ = P) :
  x₁ = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₁ = S - x₁ ∧
  x₂ = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y₂ = S - x₂ := 
by
  sorry

end find_numbers_l63_63304


namespace students_play_both_football_and_cricket_l63_63885

theorem students_play_both_football_and_cricket :
  ∀ (total F C N both : ℕ),
  total = 460 →
  F = 325 →
  C = 175 →
  N = 50 →
  total - N = F + C - both →
  both = 90 :=
by
  intros
  sorry

end students_play_both_football_and_cricket_l63_63885


namespace minimum_rebate_rate_l63_63859

open Real

noncomputable def rebate_rate (s p_M p_N p: ℝ) : ℝ := 100 * (p_M + p_N - p) / s

theorem minimum_rebate_rate 
  (s p_M p_N p : ℝ)
  (h_M : 0.19 * 0.4 * s ≤ p_M ∧ p_M ≤ 0.24 * 0.4 * s)
  (h_N : 0.29 * 0.6 * s ≤ p_N ∧ p_N ≤ 0.34 * 0.6 * s)
  (h_total : 0.10 * s ≤ p ∧ p ≤ 0.15 * s) :
  ∃ r : ℝ, r = rebate_rate s p_M p_N p ∧ 0.1 ≤ r ∧ r ≤ 0.2 :=
sorry

end minimum_rebate_rate_l63_63859


namespace M_intersection_N_l63_63078

noncomputable def M := {x : ℝ | 0 ≤ x ∧ x < 16}
noncomputable def N := {x : ℝ | x ≥ 1 / 3}

theorem M_intersection_N :
  (M ∩ N) = {x : ℝ | 1 / 3 ≤ x ∧ x < 16} := by
sorry

end M_intersection_N_l63_63078


namespace money_made_l63_63705

def initial_amount : ℕ := 26
def final_amount : ℕ := 52

theorem money_made : (final_amount - initial_amount) = 26 :=
by sorry

end money_made_l63_63705


namespace calculate_a_mul_a_sub_3_l63_63164

variable (a : ℝ)

theorem calculate_a_mul_a_sub_3 : a * (a - 3) = a^2 - 3 * a := 
by
  sorry

end calculate_a_mul_a_sub_3_l63_63164


namespace max_difference_in_volume_l63_63141

noncomputable def computed_volume (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def max_possible_volume (length width height : ℕ) (error : ℕ) : ℕ :=
  (length + error) * (width + error) * (height + error)

theorem max_difference_in_volume :
  ∀ (length width height error : ℕ), length = 150 → width = 150 → height = 225 → error = 1 → 
  max_possible_volume length width height error - computed_volume length width height = 90726 :=
by
  intros length width height error h_length h_width h_height h_error
  rw [h_length, h_width, h_height, h_error]
  simp only [computed_volume, max_possible_volume]
  -- Intermediate calculations
  sorry

end max_difference_in_volume_l63_63141


namespace necessary_but_not_sufficient_for_inequality_l63_63651

variables (a b : ℝ)

theorem necessary_but_not_sufficient_for_inequality (h : a ≠ b) (hab_pos : a * b > 0) :
  (b/a + a/b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequality_l63_63651


namespace lemonade_third_intermission_l63_63352

theorem lemonade_third_intermission (a b c T : ℝ) (h1 : a = 0.25) (h2 : b = 0.42) (h3 : T = 0.92) (h4 : T = a + b + c) : c = 0.25 :=
by
  sorry

end lemonade_third_intermission_l63_63352


namespace largest_s_value_l63_63232

theorem largest_s_value (r s : ℕ) (h_r : r ≥ 3) (h_s : s ≥ 3) 
  (h_angle : (r - 2) * 180 / r = (5 * (s - 2) * 180) / (4 * s)) : s ≤ 130 :=
by {
  sorry
}

end largest_s_value_l63_63232


namespace exists_integers_m_n_for_inequalities_l63_63095

theorem exists_integers_m_n_for_inequalities (a b : ℝ) (h : a ≠ b) : ∃ (m n : ℤ), 
  (a * (m : ℝ) + b * (n : ℝ) < 0) ∧ (b * (m : ℝ) + a * (n : ℝ) > 0) :=
sorry

end exists_integers_m_n_for_inequalities_l63_63095


namespace quadratic_minimum_value_interval_l63_63204

theorem quadratic_minimum_value_interval (k : ℝ) :
  (∀ (x : ℝ), 0 ≤ x ∧ x < 2 → (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (2*k^2 + 2*k - 1)) → (0 ≤ k ∧ k < 1) :=
by {
  sorry
}

end quadratic_minimum_value_interval_l63_63204


namespace total_enemies_l63_63988

theorem total_enemies (n : ℕ) : (n - 3) * 9 = 72 → n = 11 :=
by
  sorry

end total_enemies_l63_63988


namespace kayak_rental_cost_l63_63744

theorem kayak_rental_cost (F : ℝ) (C : ℝ) (h1 : ∀ t : ℝ, C = F + 5 * t)
  (h2 : C = 30) : C = 45 :=
sorry

end kayak_rental_cost_l63_63744


namespace ratio_of_bases_l63_63537

-- Definitions for an isosceles trapezoid
def isosceles_trapezoid (s t : ℝ) := ∃ (a b c d : ℝ), s = d ∧ s = a ∧ t = b ∧ (a + c = b + d)

-- Main theorem statement based on conditions and required ratio
theorem ratio_of_bases (s t : ℝ) (h1 : isosceles_trapezoid s t)
  (h2 : s = s) (h3 : t = t) : s / t = 3 / 5 :=
by { sorry }

end ratio_of_bases_l63_63537


namespace total_raisins_l63_63091

theorem total_raisins (yellow raisins black raisins : ℝ) (h_yellow : yellow = 0.3) (h_black : black = 0.4) : yellow + black = 0.7 := 
by
  sorry

end total_raisins_l63_63091


namespace quadratic_two_distinct_real_roots_l63_63818

theorem quadratic_two_distinct_real_roots (k : ℝ) (h : k ≠ 0) : 
  (kx : ℝ) -> (a = k) -> (b = -2) -> (c = -1) -> (b^2 - 4*a*c > 0) -> (-2)^2 - 4*k*(-1) = (4 + 4*k > 0) := sorry

end quadratic_two_distinct_real_roots_l63_63818


namespace solve_problem_l63_63628

noncomputable def problem_statement : Prop :=
  (2015 : ℝ) / (2015^2 - 2016 * 2014) = 2015

theorem solve_problem : problem_statement := by
  -- Proof steps will be filled in here.
  sorry

end solve_problem_l63_63628


namespace decimal_equiv_of_fraction_l63_63607

theorem decimal_equiv_of_fraction : (1 / 5) ^ 2 = 0.04 := by
  sorry

end decimal_equiv_of_fraction_l63_63607


namespace find_missing_number_l63_63148

theorem find_missing_number (x : ℕ) (h : x * 240 = 173 * 240) : x = 173 :=
sorry

end find_missing_number_l63_63148


namespace neg_one_to_zero_l63_63909

theorem neg_one_to_zero : (-1 : ℤ)^0 = 1 := 
by 
  -- Expanding expressions and applying the exponent rule for non-zero numbers
  sorry

end neg_one_to_zero_l63_63909


namespace lesser_fraction_l63_63276

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l63_63276


namespace smallest_perfect_square_greater_than_x_l63_63511

theorem smallest_perfect_square_greater_than_x (x : ℤ)
  (h₁ : ∃ k : ℤ, k^2 ≠ x)
  (h₂ : x ≥ 0) :
  ∃ n : ℤ, n^2 > x ∧ ∀ m : ℤ, m^2 > x → n^2 ≤ m^2 :=
sorry

end smallest_perfect_square_greater_than_x_l63_63511


namespace quadratic_two_distinct_real_roots_l63_63819

theorem quadratic_two_distinct_real_roots (k : ℝ) (h : k ≠ 0) : 
  (kx : ℝ) -> (a = k) -> (b = -2) -> (c = -1) -> (b^2 - 4*a*c > 0) -> (-2)^2 - 4*k*(-1) = (4 + 4*k > 0) := sorry

end quadratic_two_distinct_real_roots_l63_63819


namespace actual_plot_area_l63_63024

noncomputable def area_of_triangle_in_acres : Real :=
  let base_cm : Real := 8
  let height_cm : Real := 5
  let area_cm2 : Real := 0.5 * base_cm * height_cm
  let conversion_factor_cm2_to_km2 : Real := 25
  let area_km2 : Real := area_cm2 * conversion_factor_cm2_to_km2
  let conversion_factor_km2_to_acres : Real := 247.1
  area_km2 * conversion_factor_km2_to_acres

theorem actual_plot_area :
  area_of_triangle_in_acres = 123550 :=
by
  sorry

end actual_plot_area_l63_63024


namespace rhombus_diagonals_perpendicular_l63_63690

section circumscribed_quadrilateral

variables {a b c d : ℝ}

-- Definition of a tangential quadrilateral satisfying Pitot's theorem.
def tangential_quadrilateral (a b c d : ℝ) :=
  a + c = b + d

-- Defining a rhombus in terms of its sides
def rhombus (a b c d : ℝ) :=
  a = b ∧ b = c ∧ c = d

-- The theorem we want to prove
theorem rhombus_diagonals_perpendicular
  (h : tangential_quadrilateral a b c d)
  (hr : rhombus a b c d) : 
  true := sorry

end circumscribed_quadrilateral

end rhombus_diagonals_perpendicular_l63_63690


namespace floor_sub_y_eq_zero_l63_63380

theorem floor_sub_y_eq_zero {y : ℝ} (h : ⌊y⌋ + ⌈y⌉ = 2 * y) : ⌊y⌋ - y = 0 :=
sorry

end floor_sub_y_eq_zero_l63_63380


namespace range_of_x_l63_63950

theorem range_of_x (a : ℕ → ℝ) (x : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_condition : ∀ n, a (n + 1)^2 + a n^2 < (5 / 2) * a (n + 1) * a n)
  (h_a2 : a 2 = 3 / 2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) : 2 < x ∧ x < 3 := by
  sorry

end range_of_x_l63_63950


namespace sin_cos_identity_l63_63502

theorem sin_cos_identity : sin (π / 12) * cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_identity_l63_63502


namespace geometric_seq_min_value_l63_63707

theorem geometric_seq_min_value (b : ℕ → ℝ) (s : ℝ) (h1 : b 1 = 1) (h2 : ∀ n : ℕ, b (n + 1) = s * b n) : 
  ∃ s : ℝ, 3 * b 1 + 4 * b 2 = -9 / 16 :=
by
  sorry

end geometric_seq_min_value_l63_63707


namespace salary_increase_percentage_l63_63845

variable {P : ℝ} (initial_salary : P > 0)

def salary_after_first_year (P: ℝ) : ℝ :=
  P * 1.12

def salary_after_second_year (P: ℝ) : ℝ :=
  (salary_after_first_year P) * 1.12

def salary_after_third_year (P: ℝ) : ℝ :=
  (salary_after_second_year P) * 1.15

theorem salary_increase_percentage (P: ℝ) (h: P > 0) : 
  (salary_after_third_year P - P) / P * 100 = 44 :=
by 
  sorry

end salary_increase_percentage_l63_63845


namespace david_recreation_l63_63229

theorem david_recreation (W : ℝ) (P : ℝ) 
  (h1 : 0.95 * W = this_week_wages) 
  (h2 : 0.5 * this_week_wages = recreation_this_week)
  (h3 : 1.1875 * (P / 100) * W = recreation_this_week) : P = 40 :=
sorry

end david_recreation_l63_63229


namespace expression_value_l63_63384

theorem expression_value
  (x y : ℝ) 
  (h : x - 3 * y = 4) : 
  (x - 3 * y)^2 + 2 * x - 6 * y - 10 = 14 :=
by
  sorry

end expression_value_l63_63384


namespace computer_price_increase_l63_63529

theorem computer_price_increase (c : ℕ) (h : 2 * c = 540) : c + (c * 30 / 100) = 351 :=
by
  sorry

end computer_price_increase_l63_63529


namespace joanne_earnings_l63_63993

theorem joanne_earnings :
  let main_job_hourly_wage := 16.00
  let part_time_job_hourly_wage := 13.50
  let main_job_hours_per_day := 8
  let part_time_job_hours_per_day := 2
  let number_of_days := 5

  let main_job_daily_earnings := main_job_hours_per_day * main_job_hourly_wage
  let main_job_weekly_earnings := main_job_daily_earnings * number_of_days
  let part_time_job_daily_earnings := part_time_job_hours_per_day * part_time_job_hourly_wage
  let part_time_job_weekly_earnings := part_time_job_daily_earnings * number_of_days

  (main_job_weekly_earnings + part_time_job_weekly_earnings = 775)
:= by
  simp only [
    main_job_hourly_wage, part_time_job_hourly_wage,
    main_job_hours_per_day, part_time_job_hours_per_day,
    number_of_days,
    main_job_daily_earnings, main_job_weekly_earnings,
    part_time_job_daily_earnings, part_time_job_weekly_earnings
  ]
  sorry

end joanne_earnings_l63_63993


namespace rectangular_solid_edges_sum_l63_63021

theorem rectangular_solid_edges_sum 
  (a b c : ℝ) 
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b^2 = a * c) : 
  4 * (a + b + c) = 32 := 
  sorry

end rectangular_solid_edges_sum_l63_63021


namespace no_such_convex_hexagon_and_point_exists_l63_63926

theorem no_such_convex_hexagon_and_point_exists :
  ¬(∃ (hexagon : List (ℝ × ℝ)), 
     hexagon.length = 6 ∧ 
     (∀ i j, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → i ≠ j → 
       dist (hexagon.nthLe i (by simp [hexagon.length_le_six])) 
            (hexagon.nthLe j (by simp [hexagon.length_le_six])) > 1) ∧ 
     (∀ i, 0 ≤ i → i < 6 → 
       let M := (x, y) in
       dist M (hexagon.nthLe i (by simp [hexagon.length_le_six])) < 1)) :=
sorry

noncomputable def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

lemma convex (hexagon : List (ℝ × ℝ)) : 
  hexagon.length = 6 → 
  ∀ i j k m, 0 ≤ i → i < 6 → 0 ≤ j → j < 6 → 0 ≤ k → k < 6 → 0 ≤ m → m < 6 → 
  (i ≠ j ∧ j ≠ k ∧ k ≠ m ∧ m ≠ i ∧ i ≠ k ∧ j ≠ m) →
  let A := hexagon.nthLe i (by simp [hexagon.length_le_six]) in
  let B := hexagon.nthLe j (by simp [hexagon.length_le_six]) in
  let C := hexagon.nthLe k (by simp [hexagon.length_le_six]) in
  let D := hexagon.nthLe m (by simp [hexagon.length_le_six]) in
  (C.1 - A.1) * (B.2 - A.2) < (C.2 - A.2) * (B.1 - A.1) ∧ 
  (D.1 - B.1) * (C.2 - B.2) < (D.2 - B.2) * (C.1 - B.1) (by simp) :=
sorry

end no_such_convex_hexagon_and_point_exists_l63_63926


namespace machine_production_percentage_difference_l63_63238

theorem machine_production_percentage_difference 
  (X_production_rate : ℕ := 3)
  (widgets_to_produce : ℕ := 1080)
  (difference_in_hours : ℕ := 60) :
  ((widgets_to_produce / (widgets_to_produce / X_production_rate - difference_in_hours) - 
   X_production_rate) / X_production_rate * 100) = 20 := by
  sorry

end machine_production_percentage_difference_l63_63238


namespace diagonal_perimeter_ratio_l63_63011

theorem diagonal_perimeter_ratio
    (b : ℝ)
    (h : b ≠ 0) -- To ensure the garden has non-zero side lengths
    (a : ℝ) (h1: a = 3 * b) 
    (d : ℝ) (h2: d = (Real.sqrt (b^2 + a^2)))
    (P : ℝ) (h3: P = 2 * a + 2 * b)
    (h4 : d = b * (Real.sqrt 10)) :
  d / P = (Real.sqrt 10) / 8 := by
    sorry

end diagonal_perimeter_ratio_l63_63011


namespace charged_amount_is_35_l63_63836

-- Definitions based on conditions
def annual_interest_rate : ℝ := 0.05
def owed_amount : ℝ := 36.75
def time_in_years : ℝ := 1

-- The amount charged on the account in January
def charged_amount (P : ℝ) : Prop :=
  owed_amount = P + (P * annual_interest_rate * time_in_years)

-- The proof statement
theorem charged_amount_is_35 : charged_amount 35 := by
  sorry

end charged_amount_is_35_l63_63836


namespace find_b_value_l63_63981

theorem find_b_value (x y z : ℝ) (u t : ℕ) (h_pos_xyx : x > 0 ∧ y > 0 ∧ z > 0 ∧ u > 0 ∧ t > 0)
  (h1 : (x + y - z) / z = 1) (h2 : (x - y + z) / y = 1) (h3 : (-x + y + z) / x = 1) 
  (ha : (x + y) * (y + z) * (z + x) / (x * y * z) = 8) (hu_t : u + t + u * t = 34) : (u + t = 10) :=
by
  sorry

end find_b_value_l63_63981


namespace peter_pairs_of_pants_l63_63907

-- Define the conditions
def shirt_cost_condition (S : ℕ) : Prop := 2 * S = 20
def pants_cost (P : ℕ) : Prop := P = 6
def purchase_condition (P S : ℕ) (number_of_pants : ℕ) : Prop :=
  P * number_of_pants + 5 * S = 62

-- State the proof problem:
theorem peter_pairs_of_pants (S P number_of_pants : ℕ) 
  (h1 : shirt_cost_condition S)
  (h2 : pants_cost P) 
  (h3 : purchase_condition P S number_of_pants) :
  number_of_pants = 2 := by
  sorry

end peter_pairs_of_pants_l63_63907


namespace find_value_l63_63578

variable (number : ℝ) (V : ℝ)

theorem find_value
  (h1 : number = 8)
  (h2 : 0.75 * number + V = 8) : V = 2 := by
  sorry

end find_value_l63_63578


namespace total_ticket_cost_is_14_l63_63159

-- Definitions of the ticket costs
def ticket_cost_hat : ℕ := 2
def ticket_cost_stuffed_animal : ℕ := 10
def ticket_cost_yoyo : ℕ := 2

-- Definition of the total ticket cost
def total_ticket_cost : ℕ := ticket_cost_hat + ticket_cost_stuffed_animal + ticket_cost_yoyo

-- Theorem stating the total ticket cost is 14
theorem total_ticket_cost_is_14 : total_ticket_cost = 14 := by
  -- Proof would go here, but sorry is used to skip it
  sorry

end total_ticket_cost_is_14_l63_63159


namespace GP_length_l63_63828

theorem GP_length (X Y Z G P Q : Type) 
  (XY XZ YZ : ℝ) 
  (hXY : XY = 12) 
  (hXZ : XZ = 9) 
  (hYZ : YZ = 15) 
  (hG_centroid : true)  -- Medians intersect at G (Centroid property)
  (hQ_altitude : true)  -- Q is the foot of the altitude from X to YZ
  (hP_below_G : true)  -- P is the point on YZ directly below G
  : GP = 2.4 := 
sorry

end GP_length_l63_63828


namespace adil_older_than_bav_by_732_days_l63_63332

-- Definitions based on the problem conditions
def adilBirthDate : String := "December 31, 2015"
def bavBirthDate : String := "January 1, 2018"

-- Main theorem statement 
theorem adil_older_than_bav_by_732_days :
    let daysIn2016 := 366
    let daysIn2017 := 365
    let transition := 1
    let totalDays := daysIn2016 + daysIn2017 + transition
    totalDays = 732 :=
by
    sorry

end adil_older_than_bav_by_732_days_l63_63332


namespace bin_add_convert_l63_63590

-- Definitions
def bin1 := 11111111
def bin2 := 11111
def dec_value (b : ℕ) : ℕ := (2^(b.digits).length - 1)

-- Main proof statement
theorem bin_add_convert (h1 : dec_value bin1 = 255) (h2 : dec_value bin2 = 31) : 
  let sum := h1 + h2 in 
  let base8 := nat.base_8_val sum in 
  let final_decimal := nat.base_10_of_base_8 base8 in 
  final_decimal = 286 := 
by 
  sorry

end bin_add_convert_l63_63590


namespace find_number_l63_63869

def valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (¬ (n % 10 = 6) ∧ n % 7 = 0)) ∧
  ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (¬ (n > 26) ∧ n % 10 = 8)) ∧
  ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (¬ (n % 13 = 0) ∧ n < 27))

theorem find_number : ∃ n : ℕ, valid_number n ∧ n = 91 := by
  sorry

end find_number_l63_63869


namespace regular_price_of_tire_l63_63126

theorem regular_price_of_tire (x : ℝ) (h : 3 * x + 10 = 250) : x = 80 :=
sorry

end regular_price_of_tire_l63_63126


namespace find_numbers_sum_eq_S_product_eq_P_l63_63309

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l63_63309


namespace ratio_expression_value_l63_63170

theorem ratio_expression_value (x y : ℝ) (h : x ≠ 0) (h' : y ≠ 0) (h_eq : x^2 - y^2 = x + y) : 
  x / y + y / x = 2 + 1 / (y^2 + y) :=
by
  sorry

end ratio_expression_value_l63_63170


namespace binary_multiplication_l63_63777

theorem binary_multiplication :
  0b1101 * 0b110 = 0b1011110 := 
sorry

end binary_multiplication_l63_63777


namespace frustum_has_only_two_parallel_surfaces_l63_63027

-- Definitions for the geometric bodies in terms of their properties
structure Pyramid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 0

structure Prism where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

structure Frustum where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 2

structure Cuboid where
  -- definition indicating the number of parallel surfaces
  parallel_surfaces : Nat := 6

-- The main theorem stating that the Frustum is the one with exactly two parallel surfaces.
theorem frustum_has_only_two_parallel_surfaces (pyramid : Pyramid) (prism : Prism) (frustum : Frustum) (cuboid : Cuboid) :
  frustum.parallel_surfaces = 2 ∧
  pyramid.parallel_surfaces ≠ 2 ∧
  prism.parallel_surfaces ≠ 2 ∧
  cuboid.parallel_surfaces ≠ 2 :=
by
  sorry

end frustum_has_only_two_parallel_surfaces_l63_63027


namespace inequality_holds_for_all_x_iff_a_in_range_l63_63958

theorem inequality_holds_for_all_x_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - 4 * x > 2 * a * x + a) ↔ (-4 < a ∧ a < -1) :=
by
  sorry

end inequality_holds_for_all_x_iff_a_in_range_l63_63958


namespace lesser_fraction_l63_63269

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l63_63269


namespace calculate_a_mul_a_sub_3_l63_63165

variable (a : ℝ)

theorem calculate_a_mul_a_sub_3 : a * (a - 3) = a^2 - 3 * a := 
by
  sorry

end calculate_a_mul_a_sub_3_l63_63165


namespace sum_of_coefficients_l63_63054

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (2 * x + 1)^5 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 →
  a₀ = 1 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 3^5 - 1 :=
by
  intros h_expand h_a₀
  sorry

end sum_of_coefficients_l63_63054


namespace vector_subtraction_magnitude_l63_63800

theorem vector_subtraction_magnitude (a b : EuclideanSpace ℝ (Fin 2))
  (h1 : ‖a‖ = 2) 
  (h2 : ‖b‖ = 3) 
  (h3 : ‖a + b‖ = Real.sqrt 19) : 
  ‖a - b‖ = Real.sqrt 7 :=
sorry

end vector_subtraction_magnitude_l63_63800


namespace sandy_tokens_ratio_l63_63855

theorem sandy_tokens_ratio :
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (difference : ℕ),
  total_tokens = 1000000 →
  num_siblings = 4 →
  difference = 375000 →
  ∃ (sandy_tokens : ℕ),
  sandy_tokens = (total_tokens - (num_siblings * ((total_tokens - difference) / (num_siblings + 1)))) ∧
  sandy_tokens / total_tokens = 1 / 2 :=
by 
  intros total_tokens num_siblings difference h1 h2 h3
  sorry

end sandy_tokens_ratio_l63_63855


namespace no_such_hexagon_exists_l63_63925

open Set Metric

noncomputable def exists_convex_hexagon_and_point_M : Prop :=
  ∃ (hexagon : Fin 6 → Point) (M : Point),
    (∀ i, distance (hexagon i) (hexagon ((i + 1) % 6)) > 1) ∧
    (M ∈ convexHull ℝ (range hexagon)) ∧
    (∀ i, distance M (hexagon i) < 1)

theorem no_such_hexagon_exists :
  ¬ exists_convex_hexagon_and_point_M := by
  sorry

end no_such_hexagon_exists_l63_63925


namespace math_club_members_count_l63_63256

theorem math_club_members_count 
    (n_books : ℕ) 
    (n_borrow_each_member : ℕ) 
    (n_borrow_each_book : ℕ) 
    (total_borrow_count_books : n_books * n_borrow_each_book = 36) 
    (total_borrow_count_members : 2 * x = 36) 
    : x = 18 := 
by
  sorry

end math_club_members_count_l63_63256


namespace plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l63_63026

theorem plan_Y_cheaper_than_X (x : ℕ) : 
  ∃ x, 2500 + 7 * x < 15 * x ∧ ∀ y, y < x → ¬ (2500 + 7 * y < 15 * y) := 
sorry

theorem plan_Z_cheaper_than_X (x : ℕ) : 
  ∃ x, 3000 + 6 * x < 15 * x ∧ ∀ y, y < x → ¬ (3000 + 6 * y < 15 * y) := 
sorry

end plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l63_63026


namespace angela_action_figures_left_l63_63485

theorem angela_action_figures_left :
  ∀ (initial_collection : ℕ), 
  initial_collection = 24 → 
  (let sold := initial_collection / 4 in
   let remaining_after_sold := initial_collection - sold in
   let given_to_daughter := remaining_after_sold / 3 in
   let remaining_after_given := remaining_after_sold - given_to_daughter in
   remaining_after_given = 12) :=
by
  intros
  sorry

end angela_action_figures_left_l63_63485


namespace area_of_triangle_formed_by_line_and_axes_l63_63788

-- Definition of the line equation condition
def line_eq (x y : ℝ) : Prop := 2 * x - 5 * y - 10 = 0

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_line_and_axes :
  (∃ x y : ℝ, line_eq x y ∧ x = 0 ∧ y = -2) ∧
  (∃ x y : ℝ, line_eq x y ∧ x = 5 ∧ y = 0) →
  let base : ℝ := 5
  let height : ℝ := 2
  let area := (1 / 2) * base * height
  area = 5 := 
by
  sorry

end area_of_triangle_formed_by_line_and_axes_l63_63788


namespace find_k_l63_63666

-- Define the vectors and the condition of perpendicularity
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, -1)
def c (k : ℝ) : ℝ × ℝ := (3 + k, 1 - k)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The primary statement we aim to prove
theorem find_k : ∃ k : ℝ, dot_product a (c k) = 0 ∧ k = -5 :=
by
  exists -5
  sorry

end find_k_l63_63666


namespace positive_difference_l63_63457

theorem positive_difference :
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  term2 - term1 = 496 :=
by
  let a := 8^2
  let term1 := (a + a) / 8
  let term2 := (a * a) / 8
  have h1 : a = 64 := rfl
  have h2 : term1 = 16 := by simp [a, term1]
  have h3 : term2 = 512 := by simp [a, term2]
  show 512 - 16 = 496 from sorry

end positive_difference_l63_63457


namespace tom_father_time_saved_correct_l63_63873

def tom_father_jog_time_saved : Prop :=
  let monday_speed := 6
  let tuesday_speed := 5
  let thursday_speed := 4
  let saturday_speed := 5
  let daily_distance := 3
  let hours_to_minutes := 60

  let monday_time := daily_distance / monday_speed
  let tuesday_time := daily_distance / tuesday_speed
  let thursday_time := daily_distance / thursday_speed
  let saturday_time := daily_distance / saturday_speed

  let total_time_original := monday_time + tuesday_time + thursday_time + saturday_time
  let always_5mph_time := 4 * (daily_distance / 5)
  let time_saved := total_time_original - always_5mph_time

  let time_saved_minutes := time_saved * hours_to_minutes

  time_saved_minutes = 3

theorem tom_father_time_saved_correct : tom_father_jog_time_saved := by
  sorry

end tom_father_time_saved_correct_l63_63873


namespace ellie_needs_25ml_of_oil_l63_63179

theorem ellie_needs_25ml_of_oil 
  (oil_per_wheel : ℕ) 
  (number_of_wheels : ℕ) 
  (other_parts_oil : ℕ) 
  (total_oil : ℕ)
  (h1 : oil_per_wheel = 10)
  (h2 : number_of_wheels = 2)
  (h3 : other_parts_oil = 5)
  (h4 : total_oil = oil_per_wheel * number_of_wheels + other_parts_oil) : 
  total_oil = 25 :=
  sorry

end ellie_needs_25ml_of_oil_l63_63179


namespace super_k_teams_l63_63685

theorem super_k_teams (n : ℕ) (h : n * (n - 1) / 2 = 45) : n = 10 :=
sorry

end super_k_teams_l63_63685


namespace percentage_of_failed_candidates_l63_63536

theorem percentage_of_failed_candidates :
  let total_candidates := 2000
  let girls := 900
  let boys := total_candidates - girls
  let boys_passed := 32 / 100 * boys
  let girls_passed := 32 / 100 * girls
  let total_passed := boys_passed + girls_passed
  let total_failed := total_candidates - total_passed
  let percentage_failed := (total_failed / total_candidates) * 100
  percentage_failed = 68 :=
by
  -- Proof goes here
  sorry

end percentage_of_failed_candidates_l63_63536


namespace find_integer_n_l63_63524

theorem find_integer_n (n : ℤ) :
  (⌊ (n^2 : ℤ) / 9 ⌋ - ⌊ n / 3 ⌋^2 = 3) → (n = 8 ∨ n = 10) :=
  sorry

end find_integer_n_l63_63524


namespace parallel_vectors_angle_l63_63963

noncomputable def vec_a (α : ℝ) : ℝ × ℝ := (1 / 2, Real.sin α)
noncomputable def vec_b (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)

theorem parallel_vectors_angle (α : ℝ) (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ (vec_a α).1 = k * (vec_b α).1 ∧ (vec_a α).2 = k * (vec_b α).2) (h_acute : 0 < α ∧ α < π / 2) :
  α = π / 4 :=
sorry

end parallel_vectors_angle_l63_63963


namespace markup_amount_l63_63605

def purchase_price : ℝ := 48
def overhead_percentage : ℝ := 0.35
def net_profit : ℝ := 18

def overhead : ℝ := purchase_price * overhead_percentage
def total_cost : ℝ := purchase_price + overhead
def selling_price : ℝ := total_cost + net_profit
def markup : ℝ := selling_price - purchase_price

theorem markup_amount : markup = 34.80 := by
  sorry

end markup_amount_l63_63605


namespace positive_difference_of_fractions_l63_63454

theorem positive_difference_of_fractions : 
  (let a := 8^2 in (a + a) / 8) = 16 ∧ (let a := 8^2 in (a * a) / 8) = 512 →
  (let a := 8^2 in ((a * a) / 8 - (a + a) / 8)) = 496 := 
by
  sorry

end positive_difference_of_fractions_l63_63454


namespace tommy_needs_4_steaks_l63_63585

noncomputable def tommy_steaks : Nat := 
  let family_members := 5
  let ounces_per_pound := 16
  let ounces_per_steak := 20
  let total_ounces_needed := family_members * ounces_per_pound
  let steaks_needed := total_ounces_needed / ounces_per_steak
  steaks_needed

theorem tommy_needs_4_steaks :
  tommy_steaks = 4 :=
by
  sorry

end tommy_needs_4_steaks_l63_63585


namespace average_increase_l63_63471

-- Definitions
def runs_11 := 90
def avg_11 := 40

-- Conditions
def total_runs_before (A : ℕ) := A * 10
def total_runs_after (runs_11 : ℕ) (total_runs_before : ℕ) := total_runs_before + runs_11
def increased_average (avg_11 : ℕ) (avg_before : ℕ) := avg_11 = avg_before + 5

-- Theorem stating the equivalent proof problem
theorem average_increase
  (A : ℕ)
  (H1 : total_runs_after runs_11 (total_runs_before A) = 40 * 11)
  (H2 : avg_11 = 40) :
  increased_average 40 A := 
sorry

end average_increase_l63_63471


namespace find_multiplier_l63_63146

theorem find_multiplier (x : ℝ) (y : ℝ) (h1 : x = 62.5) (h2 : (y * (x + 5)) / 5 - 5 = 22) : y = 2 :=
sorry

end find_multiplier_l63_63146


namespace rectangle_perimeter_l63_63493

theorem rectangle_perimeter {b : ℕ → ℕ} {W H : ℕ}
  (h1 : ∀ i, b i ≠ b (i+1))
  (h2 : b 9 = W / 2)
  (h3 : gcd W H = 1)

  (h4 : b 1 + b 2 = b 3)
  (h5 : b 1 + b 3 = b 4)
  (h6 : b 3 + b 4 = b 5)
  (h7 : b 4 + b 5 = b 6)
  (h8 : b 2 + b 3 + b 5 = b 7)
  (h9 : b 2 + b 7 = b 8)
  (h10 : b 1 + b 4 + b 6 = b 9)
  (h11 : b 6 + b 9 = b 7 + b 8) : 
  2 * (W + H) = 266 :=
  sorry

end rectangle_perimeter_l63_63493


namespace distance_between_nails_l63_63582

theorem distance_between_nails (banner_length : ℕ) (num_nails : ℕ) (end_distance : ℕ) :
  banner_length = 20 → num_nails = 7 → end_distance = 1 → 
  (banner_length - 2 * end_distance) / (num_nails - 1) = 3 :=
by
  intros
  sorry

end distance_between_nails_l63_63582


namespace hats_in_box_total_l63_63129

theorem hats_in_box_total : 
  (∃ (n : ℕ), (∀ (r b y : ℕ), r + y = n - 2 ∧ r + b = n - 2 ∧ b + y = n - 2)) → (∃ n, n = 3) :=
by
  sorry

end hats_in_box_total_l63_63129


namespace simple_interest_for_2_years_l63_63847

noncomputable def calculate_simple_interest (P r t : ℝ) : ℝ :=
  (P * r * t) / 100

theorem simple_interest_for_2_years (CI P r t : ℝ) (hCI : CI = P * (1 + r / 100)^t - P)
  (hCI_value : CI = 615) (r_value : r = 5) (t_value : t = 2) : 
  calculate_simple_interest P r t = 600 :=
by
  sorry

end simple_interest_for_2_years_l63_63847


namespace repeated_number_divisibility_l63_63015

theorem repeated_number_divisibility (x : ℕ) (h : 1000 ≤ x ∧ x < 10000) :
  73 ∣ (10001 * x) ∧ 137 ∣ (10001 * x) :=
sorry

end repeated_number_divisibility_l63_63015


namespace replacement_fraction_l63_63465

variable (Q : ℝ) (x : ℝ)

def initial_concentration : ℝ := 0.70
def new_concentration : ℝ := 0.35
def replacement_concentration : ℝ := 0.25

theorem replacement_fraction (h1 : 0.70 * Q - 0.70 * x * Q + 0.25 * x * Q = 0.35 * Q) :
  x = 7 / 9 :=
by
  sorry

end replacement_fraction_l63_63465


namespace binomial_distribution_parameters_l63_63236

noncomputable def E (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def D (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_distribution_parameters (n : ℕ) (p : ℝ) 
  (h1 : E n p = 2.4) (h2 : D n p = 1.44) : 
  n = 6 ∧ p = 0.4 :=
by
  sorry

end binomial_distribution_parameters_l63_63236


namespace problem_1_problem_2_l63_63108

-- Problem 1 proof statement
theorem problem_1 (x : ℝ) (h : x = -1) : 
  (1 * (-x^2 + 5 * x) - (x - 3) - 4 * x) = 2 := by
  -- Placeholder for the proof
  sorry

-- Problem 2 proof statement
theorem problem_2 (m n : ℝ) (h_m : m = -1/2) (h_n : n = 1/3) : 
  (5 * (3 * m^2 * n - m * n^2) - (m * n^2 + 3 * m^2 * n)) = 4/3 := by
  -- Placeholder for the proof
  sorry

end problem_1_problem_2_l63_63108


namespace pentagon_sum_of_sides_and_vertices_eq_10_l63_63187

-- Define the number of sides of a pentagon
def number_of_sides : ℕ := 5

-- Define the number of vertices of a pentagon
def number_of_vertices : ℕ := 5

-- Define the sum of sides and vertices
def sum_of_sides_and_vertices : ℕ :=
  number_of_sides + number_of_vertices

-- The theorem to prove that the sum is 10
theorem pentagon_sum_of_sides_and_vertices_eq_10 : sum_of_sides_and_vertices = 10 :=
by
  sorry

end pentagon_sum_of_sides_and_vertices_eq_10_l63_63187


namespace find_pairs_l63_63939

theorem find_pairs (x y : ℝ) (h1 : |x| + |y| = 1340) (h2 : x^3 + y^3 + 2010 * x * y = 670^3) :
  x + y = 670 ∧ x * y = -673350 :=
sorry

end find_pairs_l63_63939


namespace proof_sum_of_drawn_kinds_l63_63478

def kindsGrains : Nat := 40
def kindsVegetableOils : Nat := 10
def kindsAnimalFoods : Nat := 30
def kindsFruitsAndVegetables : Nat := 20
def totalKindsFood : Nat := kindsGrains + kindsVegetableOils + kindsAnimalFoods + kindsFruitsAndVegetables
def sampleSize : Nat := 20
def samplingRatio : Nat := sampleSize / totalKindsFood

def numKindsVegetableOilsDrawn : Nat := kindsVegetableOils / 5
def numKindsFruitsAndVegetablesDrawn : Nat := kindsFruitsAndVegetables / 5
def sumVegetableOilsAndFruitsAndVegetablesDrawn : Nat := numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn

theorem proof_sum_of_drawn_kinds : sumVegetableOilsAndFruitsAndVegetablesDrawn = 6 := by
  have h1 : totalKindsFood = 100 := by rfl
  have h2 : samplingRatio = 1 / 5 := by
    calc
      sampleSize / totalKindsFood
      _ = 20 / 100 := rfl
      _ = 1 / 5 := by norm_num
  have h3 : numKindsVegetableOilsDrawn = 2 := by
    calc
      kindsVegetableOils / 5
      _ = 10 / 5 := rfl
      _ = 2 := by norm_num
  have h4 : numKindsFruitsAndVegetablesDrawn = 4 := by
    calc
      kindsFruitsAndVegetables / 5
      _ = 20 / 5 := rfl
      _ = 4 := by norm_num
  calc
    sumVegetableOilsAndFruitsAndVegetablesDrawn
    _ = numKindsVegetableOilsDrawn + numKindsFruitsAndVegetablesDrawn := rfl
    _ = 2 + 4 := by rw [h3, h4]
    _ = 6 := by norm_num

end proof_sum_of_drawn_kinds_l63_63478


namespace ernie_circles_l63_63337

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes circles_ali : ℕ) 
  (h1 : boxes_per_circle_ali = 8)
  (h2 : boxes_per_circle_ernie = 10)
  (h3 : total_boxes = 80)
  (h4 : circles_ali = 5) : 
  (total_boxes - circles_ali * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l63_63337


namespace average_runs_in_second_set_l63_63566

theorem average_runs_in_second_set
  (avg_first_set : ℕ → ℕ → ℕ)
  (avg_all_matches : ℕ → ℕ → ℕ)
  (avg1 : ℕ := avg_first_set 20 30)
  (avg2 : ℕ := avg_all_matches 30 25) :
  ∃ (A : ℕ), A = 15 := by
  sorry

end average_runs_in_second_set_l63_63566


namespace definite_integral_example_l63_63638

theorem definite_integral_example : ∫ x in (0 : ℝ)..(π/2), 2 * x = π^2 / 4 := 
by 
  sorry

end definite_integral_example_l63_63638


namespace find_numbers_l63_63313

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l63_63313


namespace sum_M_N_K_l63_63075

theorem sum_M_N_K (d K M N : ℤ) 
(h : ∀ x : ℤ, (x^2 + 3*x + 1) ∣ (x^4 - d*x^3 + M*x^2 + N*x + K)) :
  M + N + K = 5*K - 4*d - 11 := 
sorry

end sum_M_N_K_l63_63075


namespace tiffany_final_lives_l63_63437

def initial_lives : ℕ := 43
def lost_lives : ℕ := 14
def gained_lives : ℕ := 27

theorem tiffany_final_lives : (initial_lives - lost_lives + gained_lives) = 56 := by
    sorry

end tiffany_final_lives_l63_63437


namespace min_value_expression_l63_63650

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 10 + 6 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (a + 2 * y = 1) → ( (y^2 + a + 1) / (a * y)  ≥  c )) :=
sorry

end min_value_expression_l63_63650


namespace isosceles_trapezoid_side_length_is_five_l63_63113

noncomputable def isosceles_trapezoid_side_length (b1 b2 area : ℝ) : ℝ :=
  let h := 2 * area / (b1 + b2)
  let base_diff_half := (b2 - b1) / 2
  Real.sqrt (h^2 + base_diff_half^2)
  
theorem isosceles_trapezoid_side_length_is_five :
  isosceles_trapezoid_side_length 6 12 36 = 5 := by
  sorry

end isosceles_trapezoid_side_length_is_five_l63_63113


namespace monthly_manufacturing_expenses_l63_63617

theorem monthly_manufacturing_expenses 
  (num_looms : ℕ) (total_sales_value : ℚ) 
  (monthly_establishment_charges : ℚ) 
  (decrease_in_profit : ℚ) 
  (sales_per_loom : ℚ) 
  (manufacturing_expenses_per_loom : ℚ) 
  (total_manufacturing_expenses : ℚ) : 
  num_looms = 80 → 
  total_sales_value = 500000 → 
  monthly_establishment_charges = 75000 → 
  decrease_in_profit = 4375 → 
  sales_per_loom = total_sales_value / num_looms → 
  manufacturing_expenses_per_loom = sales_per_loom - decrease_in_profit → 
  total_manufacturing_expenses = manufacturing_expenses_per_loom * num_looms →
  total_manufacturing_expenses = 150000 :=
by
  intros h_num_looms h_total_sales h_monthly_est_charges h_decrease_in_profit h_sales_per_loom h_manufacturing_expenses_per_loom h_total_manufacturing_expenses
  sorry

end monthly_manufacturing_expenses_l63_63617


namespace crayons_per_pack_l63_63243

theorem crayons_per_pack (total_crayons : ℕ) (packs : ℕ) (crayons_per_pack : ℕ) : 
  total_crayons = 615 ∧ packs = 41 → crayons_per_pack = 15 :=
by
  intro h
  cases h with hc hp
  sorry

end crayons_per_pack_l63_63243


namespace distinct_quadrilateral_areas_l63_63933

theorem distinct_quadrilateral_areas (A B C D E F : ℝ) 
  (h : A + B + C + D + E + F = 156) :
  ∃ (Q1 Q2 Q3 : ℝ), Q1 = 78 ∧ Q2 = 104 ∧ Q3 = 104 :=
sorry

end distinct_quadrilateral_areas_l63_63933


namespace domain_and_parity_range_of_a_l63_63055

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a

theorem domain_and_parity (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  (∀ x, f a x * g a x = f a (-x) * g a (-x)) ∧ (∀ x, -1 < x ∧ x < 1) :=
sorry

theorem range_of_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : f a 1 + g a (1/4) < 1) :
  (a ∈ (Set.Ioo 0 1 ∪ Set.Ioi (3/2))) :=
sorry

end domain_and_parity_range_of_a_l63_63055


namespace thousands_digit_is_0_or_5_l63_63614

theorem thousands_digit_is_0_or_5 (n t : ℕ) (h₁ : n > 1000000) (h₂ : n % 40 = t) (h₃ : n % 625 = t) : 
  ((n / 1000) % 10 = 0) ∨ ((n / 1000) % 10 = 5) :=
sorry

end thousands_digit_is_0_or_5_l63_63614


namespace john_profit_is_1500_l63_63831

noncomputable def john_profit (total_puppies : ℕ) (half_given_away : ℕ) 
  (puppies_kept : ℕ) (sell_price : ℕ) (stud_fee : ℕ) : ℕ :=
  (total_puppies - half_given_away - puppies_kept) * sell_price - stud_fee

theorem john_profit_is_1500 : john_profit 8 4 1 600 300 = 1500 := 
by simp [john_profit]; sorry

end john_profit_is_1500_l63_63831


namespace ernie_can_make_circles_l63_63334

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end ernie_can_make_circles_l63_63334


namespace find_values_l63_63843

theorem find_values (x y : ℤ) 
  (h1 : x / 5 + 7 = y / 4 - 7)
  (h2 : x / 3 - 4 = y / 2 + 4) : 
  x = -660 ∧ y = -472 :=
by 
  sorry

end find_values_l63_63843


namespace Conor_can_chop_116_vegetables_in_a_week_l63_63635

-- Define the conditions
def eggplants_per_day : ℕ := 12
def carrots_per_day : ℕ := 9
def potatoes_per_day : ℕ := 8
def work_days_per_week : ℕ := 4

-- Define the total vegetables per day
def vegetables_per_day : ℕ := eggplants_per_day + carrots_per_day + potatoes_per_day

-- Define the total vegetables per week
def vegetables_per_week : ℕ := vegetables_per_day * work_days_per_week

-- The proof statement
theorem Conor_can_chop_116_vegetables_in_a_week : vegetables_per_week = 116 :=
by
  sorry  -- The proof step is omitted with sorry

end Conor_can_chop_116_vegetables_in_a_week_l63_63635


namespace ceil_eq_intervals_l63_63643

theorem ceil_eq_intervals (x : ℝ) :
  (⌈⌈ 3 * x ⌉ + 1 / 2⌉ = ⌈ x - 2 ⌉) ↔ (-1 : ℝ) ≤ x ∧ x < -2 / 3 := 
by
  sorry

end ceil_eq_intervals_l63_63643


namespace tracy_two_dogs_food_consumption_l63_63876

theorem tracy_two_dogs_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℝ)
  (pounds_per_cup : ℝ)
  (num_dogs : ℝ) :
  cups_per_meal = 1.5 →
  meals_per_day = 3 →
  pounds_per_cup = 1 / 2.25 →
  num_dogs = 2 →
  num_dogs * (cups_per_meal * meals_per_day) * pounds_per_cup = 4 := by
  sorry

end tracy_two_dogs_food_consumption_l63_63876


namespace seven_balls_expected_positions_l63_63559

theorem seven_balls_expected_positions :
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  expected_positions = 3.61 :=
by
  let n := 7
  let swaps := 4
  let p_stay := (1 - 2/7)^4 + 6 * (2/7)^2 * (5/7)^2 + (2/7)^4
  let expected_positions := n * p_stay
  exact sorry

end seven_balls_expected_positions_l63_63559


namespace trig_identity_example_l63_63315

theorem trig_identity_example : 4 * Real.sin (Real.pi / 12) * Real.cos (Real.pi / 12) = 1 := 
by
  -- The statement "π/12" is mathematically equivalent to 15 degrees.
  sorry

end trig_identity_example_l63_63315


namespace count_multiples_of_70_in_range_200_to_500_l63_63070

theorem count_multiples_of_70_in_range_200_to_500 : 
  ∃! count, count = 5 ∧ (∀ n, 200 ≤ n ∧ n ≤ 500 ∧ (n % 70 = 0) ↔ n = 210 ∨ n = 280 ∨ n = 350 ∨ n = 420 ∨ n = 490) :=
by
  sorry

end count_multiples_of_70_in_range_200_to_500_l63_63070


namespace no_such_hexagon_exists_l63_63931

theorem no_such_hexagon_exists (H : Π (hexagon : list (ℝ × ℝ)), hexagon.length = 6 → 
    convex hull) : ¬ ∃ M : ℝ × ℝ, 
  (∀ (hexagon : list (ℝ × ℝ)), hexagon.length = 6 →
    (∀ (i j : ℕ), i < 6 → j < 6 → i ≠ j → dist (hexagon.nth_le i sorry) (hexagon.nth_le j sorry) > 1) →
    (∀ (i : ℕ), i < 6 → dist M (hexagon.nth_le i sorry) < 1)) :=
by
  sorry

end no_such_hexagon_exists_l63_63931


namespace smallest_n_Sn_gt_2023_l63_63808

open Nat

theorem smallest_n_Sn_gt_2023 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (a 1 = 4) →
  (∀ n : ℕ, n > 0 → a n + a (n + 1) = 4 * n + 2) →
  (∀ m : ℕ, S m = if m % 2 = 0 then m ^ 2 + m else m ^ 2 + m + 2) →
  ∃ n : ℕ, S n > 2023 ∧ ∀ k : ℕ, k < n → S k ≤ 2023 :=
sorry

end smallest_n_Sn_gt_2023_l63_63808


namespace students_per_bus_l63_63715

theorem students_per_bus
  (total_students : ℕ)
  (buses : ℕ)
  (students_in_cars : ℕ)
  (h1 : total_students = 375)
  (h2 : buses = 7)
  (h3 : students_in_cars = 4) :
  (total_students - students_in_cars) / buses = 53 :=
by
  sorry

end students_per_bus_l63_63715


namespace propositions_correctness_l63_63040

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def P : Prop := ∃ x : ℝ, x^2 - x - 1 > 0
def negP : Prop := ∀ x : ℝ, x^2 - x - 1 ≤ 0

theorem propositions_correctness :
    (∀ a, a ∈ M → a ∈ N) = false ∧
    (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
    (∀ p q, ¬(p ∧ q) → ¬p ∧ ¬q) = false ∧ 
    (¬P ↔ negP) :=
by
  sorry

end propositions_correctness_l63_63040


namespace solve_for_x_l63_63109

theorem solve_for_x (x : ℝ) (h : (3 / 4) - (1 / 2) = 1 / x) : x = 4 :=
sorry

end solve_for_x_l63_63109


namespace yellow_highlighters_count_l63_63531

theorem yellow_highlighters_count 
  (Y : ℕ) 
  (pink_highlighters : ℕ := Y + 7) 
  (blue_highlighters : ℕ := Y + 12) 
  (total_highlighters : ℕ := Y + pink_highlighters + blue_highlighters) : 
  total_highlighters = 40 → Y = 7 :=
by
  sorry

end yellow_highlighters_count_l63_63531


namespace pages_written_in_a_year_l63_63544

def pages_per_friend_per_letter : ℕ := 3
def friends : ℕ := 2
def letters_per_week : ℕ := 2
def weeks_per_year : ℕ := 52

theorem pages_written_in_a_year : 
  (pages_per_friend_per_letter * friends * letters_per_week * weeks_per_year) = 624 :=
by
  sorry

end pages_written_in_a_year_l63_63544


namespace find_remainder_l63_63002

theorem find_remainder : 
    ∃ (d q r : ℕ), 472 = d * q + r ∧ 427 = d * (q - 5) + r ∧ r = 4 :=
by
  sorry

end find_remainder_l63_63002


namespace will_3_point_shots_l63_63081

theorem will_3_point_shots :
  ∃ x y : ℕ, 3 * x + 2 * y = 26 ∧ x + y = 11 ∧ x = 4 :=
by
  sorry

end will_3_point_shots_l63_63081


namespace number_of_solutions_proof_l63_63346

noncomputable def number_of_real_solutions (x y z w : ℝ) : ℝ :=
  if (x = z + w + 2 * z * w * x) ∧ (y = w + x + 2 * w * x * y) ∧ (z = x + y + 2 * x * y * z) ∧ (w = y + z + 2 * y * z * w) then
    5
  else
    0

theorem number_of_solutions_proof :
  ∃ x y z w : ℝ, x = z + w + 2 * z * w * x ∧ y = w + x + 2 * w * x * y ∧ z = x + y + 2 * x * y * z ∧ w = y + z + 2 * y * z * w → number_of_real_solutions x y z w = 5 :=
by
  sorry

end number_of_solutions_proof_l63_63346


namespace women_ratio_l63_63676

theorem women_ratio (pop : ℕ) (w_retail : ℕ) (w_fraction : ℚ) (h_pop : pop = 6000000) (h_w_retail : w_retail = 1000000) (h_w_fraction : w_fraction = 1 / 3) : 
  (3000000 : ℚ) / (6000000 : ℚ) = 1 / 2 :=
by sorry

end women_ratio_l63_63676


namespace gcd_840_1764_l63_63259

-- Define the numbers according to the conditions
def a : ℕ := 1764
def b : ℕ := 840

-- The goal is to prove that the GCD of a and b is 84
theorem gcd_840_1764 : Nat.gcd a b = 84 := 
by
  -- The proof steps would normally go here
  sorry

end gcd_840_1764_l63_63259


namespace base_number_unique_l63_63394

theorem base_number_unique (y : ℕ) : (3 : ℝ) ^ 16 = (9 : ℝ) ^ y → y = 8 → (9 : ℝ) = 3 ^ (16 / y) :=
by
  sorry

end base_number_unique_l63_63394


namespace equivalent_representations_l63_63892

theorem equivalent_representations :
  (16 / 20 = 24 / 30) ∧
  (80 / 100 = 4 / 5) ∧
  (4 / 5 = 0.8) :=
by 
  sorry

end equivalent_representations_l63_63892


namespace choir_members_minimum_l63_63322

theorem choir_members_minimum (n : ℕ) : (∃ n, n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m, (m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0) → n ≤ m) → n = 360 :=
by
  sorry

end choir_members_minimum_l63_63322


namespace petya_wins_if_and_only_if_m_ne_n_l63_63056

theorem petya_wins_if_and_only_if_m_ne_n 
  (m n : ℕ) 
  (game : ∀ m n : ℕ, Prop)
  (win_condition : (game m n ↔ m ≠ n)) : 
  Prop := 
by 
  sorry

end petya_wins_if_and_only_if_m_ne_n_l63_63056


namespace Kayla_total_items_l63_63697

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l63_63697


namespace div_by_7_iff_sum_div_by_7_l63_63508

theorem div_by_7_iff_sum_div_by_7 (a b : ℕ) : 
  (101 * a + 10 * b) % 7 = 0 ↔ (a + b) % 7 = 0 := 
by
  sorry

end div_by_7_iff_sum_div_by_7_l63_63508


namespace value_of_c_l63_63727

-- Define a structure representing conditions of the problem
structure ProblemConditions where
  c : Real

-- Define the problem in terms of given conditions and required proof
theorem value_of_c (conditions : ProblemConditions) : conditions.c = 5 / 2 := by
  sorry

end value_of_c_l63_63727


namespace shaded_area_eq_2_25_l63_63501

/-- Problem statement:
Prove that the area of the shaded region defined by the lines passing through points (0, 3) and (6, 3)
for the first line, and points (0, 6) and (3, 0) for the second line, from \(x = 0\) to the 
intersection point is equal to \(2.25\) square units.
-/

noncomputable def line1 (x : ℝ) : ℝ := 3

noncomputable def line2 (x : ℝ) : ℝ := -2 * x + 6

theorem shaded_area_eq_2_25 :
  let f := λ x : ℝ, line2 x - line1 x in
  ∫ x in (0 : ℝ)..1.5, f x = 2.25 :=
by
  -- Definitions and integration setup
  let f := λ x : ℝ, line2 x - line1 x
  sorry

end shaded_area_eq_2_25_l63_63501


namespace salary_reduction_l63_63898

theorem salary_reduction (S : ℝ) (R : ℝ) 
  (h : (S - (R / 100) * S) * (4 / 3) = S) :
  R = 25 := 
  sorry

end salary_reduction_l63_63898


namespace endpoint_of_parallel_segment_l63_63612

theorem endpoint_of_parallel_segment (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hA : A = (2, 1)) (h_parallel : B.snd = A.snd) (h_length : abs (B.fst - A.fst) = 5) :
  B = (7, 1) ∨ B = (-3, 1) :=
by
  -- Proof goes here
  sorry

end endpoint_of_parallel_segment_l63_63612


namespace cone_height_correct_l63_63959

noncomputable def cone_height (radius : ℝ) (central_angle : ℝ) : ℝ := 
  let base_radius := (central_angle * radius) / (2 * Real.pi)
  let height := Real.sqrt (radius ^ 2 - base_radius ^ 2)
  height

theorem cone_height_correct:
  cone_height 3 (2 * Real.pi / 3) = 2 * Real.sqrt 2 := 
by
  sorry

end cone_height_correct_l63_63959


namespace points_lie_on_hyperbola_l63_63367

theorem points_lie_on_hyperbola (s : ℝ) :
  let x := 2 * (Real.exp s + Real.exp (-s))
  let y := 4 * (Real.exp s - Real.exp (-s))
  (x^2) / 16 - (y^2) / 64 = 1 :=
by
  sorry

end points_lie_on_hyperbola_l63_63367


namespace group_size_l63_63255

theorem group_size (n : ℕ) (T : ℕ) (h1 : T = 14 * n) (h2 : T + 32 = 16 * (n + 1)) : n = 8 :=
by
  -- We skip the proof steps
  sorry

end group_size_l63_63255


namespace trapezoid_perimeter_l63_63541

noncomputable def perimeter_trapezoid 
  (AB CD AD BC : ℝ) 
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : ℝ :=
AB + BC + CD + AD

theorem trapezoid_perimeter
  (AB CD AD BC : ℝ)
  (h_AB_CD_parallel : AB = CD) 
  (h_AD_perpendicular : AD = 4 * Real.sqrt 2)
  (h_BC_perpendicular : BC = 4 * Real.sqrt 2)
  (h_AB_eq : AB = 10)
  (h_CD_eq : CD = 18)
  (h_height : Real.sqrt (AD ^ 2 - 1) = 4) 
  : perimeter_trapezoid AB CD AD BC h_AB_CD_parallel h_AD_perpendicular h_BC_perpendicular h_AB_eq h_CD_eq h_height = 28 + 8 * Real.sqrt 2 :=
by
  sorry

end trapezoid_perimeter_l63_63541


namespace lesser_fraction_exists_l63_63274

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l63_63274


namespace maximize_GDP_growth_l63_63743

def projectA_investment : ℕ := 20  -- million yuan
def projectB_investment : ℕ := 10  -- million yuan

def total_investment (a b : ℕ) : ℕ := a + b
def total_electricity (a b : ℕ) : ℕ := 20000 * a + 40000 * b
def total_jobs (a b : ℕ) : ℕ := 24 * a + 36 * b
def total_GDP_increase (a b : ℕ) : ℕ := 26 * a + 20 * b  -- scaled by 10 to avoid decimals

theorem maximize_GDP_growth : 
  total_investment projectA_investment projectB_investment ≤ 30 ∧
  total_electricity projectA_investment projectB_investment ≤ 1000000 ∧
  total_jobs projectA_investment projectB_investment ≥ 840 → 
  total_GDP_increase projectA_investment projectB_investment = 860 := 
by
  -- Proof would be provided here
  sorry

end maximize_GDP_growth_l63_63743


namespace inequality_holds_l63_63977

theorem inequality_holds (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := 
by
  sorry

end inequality_holds_l63_63977


namespace construct_rhombus_l63_63846

-- Define data structure representing a point in a 2-dimensional Euclidean space.
structure Point where
  x : ℝ
  y : ℝ

-- Define what it means for four points to form a rhombus.
def isRhombus (A B C D : Point) : Prop :=
  (A.x - B.x) ^ 2 + (A.y - B.y) ^ 2 = (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 ∧
  (B.x - C.x) ^ 2 + (B.y - C.y) ^ 2 = (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 ∧
  (C.x - D.x) ^ 2 + (C.y - D.y) ^ 2 = (D.x - A.x) ^ 2 + (D.y - A.y) ^ 2

-- Define circumradius condition for triangle ABC
def circumradius (A B C : Point) (R : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- Define inradius condition for triangle BCD
def inradius (B C D : Point) (r : ℝ) : Prop := sorry -- Detailed definition would be added here.

-- The proposition to be proved: We can construct the rhombus ABCD given R and r.
theorem construct_rhombus (A B C D : Point) (R r : ℝ) :
  (circumradius A B C R) →
  (inradius B C D r) →
  isRhombus A B C D :=
by
  sorry

end construct_rhombus_l63_63846


namespace no_such_hexagon_exists_l63_63928

theorem no_such_hexagon_exists :
  ¬ ∃ (hexagon : Fin 6 → ℝ × ℝ) (M : ℝ × ℝ),
    convex (set.range hexagon) ∧
    (∀ i j, 0 ≤ dist (hexagon i) (hexagon j) ∧ dist (hexagon i) (hexagon j) > 1) ∧
    (∀ i, dist M (hexagon i) < 1) :=
sorry

end no_such_hexagon_exists_l63_63928


namespace solve_for_x_l63_63686

theorem solve_for_x (x : ℚ) (h : 1/4 + 7/x = 13/x + 1/9) : x = 216/5 :=
by
  sorry

end solve_for_x_l63_63686


namespace final_investment_amount_l63_63879

noncomputable def final_amount (P1 P2 : ℝ) (r1 r2 t1 t2 n1 n2 : ℝ) : ℝ :=
  let A1 := P1 * (1 + r1 / n1) ^ (n1 * t1)
  let A2 := (A1 + P2) * (1 + r2 / n2) ^ (n2 * t2)
  A2

theorem final_investment_amount :
  final_amount 6000 2000 0.10 0.08 2 1.5 2 4 = 10467.05 :=
by
  sorry

end final_investment_amount_l63_63879


namespace measure_equality_l63_63413

open MeasureTheory

variables {n : ℕ} (μ ν : MeasureTheory.Measure (fin n → ℝ))

theorem measure_equality (h : ∀ t : (fin n → ℝ), (∫ x, Complex.exp (Complex.I * t • x) ∂μ) =
  (∫ x, Complex.exp (Complex.I * t • x) ∂ν)) :
  μ = ν :=
sorry

end measure_equality_l63_63413


namespace value_of_a_l63_63001

theorem value_of_a (a : ℝ) (h : ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 → (a * x + 6 ≤ 10)) :
  a = 2 ∨ a = -4 ∨ a = 0 :=
sorry

end value_of_a_l63_63001


namespace distance_B_to_center_l63_63347

/-- Definitions for the geometrical scenario -/
structure NotchedCircleGeom where
  radius : ℝ
  A_pos : ℝ × ℝ
  B_pos : ℝ × ℝ
  C_pos : ℝ × ℝ
  AB_len : ℝ
  BC_len : ℝ
  angle_ABC_right : Prop
  
  -- Conditions derived from problem statement
  radius_eq_sqrt72 : radius = Real.sqrt 72
  AB_len_eq_8 : AB_len = 8
  BC_len_eq_3 : BC_len = 3
  angle_ABC_right_angle : angle_ABC_right
  
/-- Problem statement -/
theorem distance_B_to_center (geom : NotchedCircleGeom) :
  let x := geom.B_pos.1
  let y := geom.B_pos.2
  x^2 + y^2 = 50 :=
sorry

end distance_B_to_center_l63_63347


namespace investment_amount_correct_l63_63608

noncomputable def investment_problem : Prop :=
  let initial_investment_rubles : ℝ := 10000
  let initial_exchange_rate : ℝ := 50
  let annual_return_rate : ℝ := 0.12
  let end_year_exchange_rate : ℝ := 80
  let currency_conversion_commission : ℝ := 0.05
  let broker_profit_commission_rate : ℝ := 0.3

  -- Computations
  let initial_investment_dollars := initial_investment_rubles / initial_exchange_rate
  let profit_dollars := initial_investment_dollars * annual_return_rate
  let total_dollars := initial_investment_dollars + profit_dollars
  let broker_commission_dollars := profit_dollars * broker_profit_commission_rate
  let post_commission_dollars := total_dollars - broker_commission_dollars
  let amount_in_rubles_before_conversion_commission := post_commission_dollars * end_year_exchange_rate
  let conversion_commission := amount_in_rubles_before_conversion_commission * currency_conversion_commission
  let final_amount_rubles := amount_in_rubles_before_conversion_commission - conversion_commission

  -- Proof goal
  final_amount_rubles = 16476.8

theorem investment_amount_correct : investment_problem := by {
  sorry
}

end investment_amount_correct_l63_63608


namespace greatest_integer_solution_l63_63135

theorem greatest_integer_solution :
  ∃ n : ℤ, (n^2 - 17 * n + 72 ≤ 0) ∧ (∀ m : ℤ, (m^2 - 17 * m + 72 ≤ 0) → m ≤ n) ∧ n = 9 :=
sorry

end greatest_integer_solution_l63_63135


namespace sum_of_edges_l63_63020

theorem sum_of_edges (a b c : ℝ)
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b ^ 2 = a * c) :
  4 * (a + b + c) = 32 := 
sorry

end sum_of_edges_l63_63020


namespace mean_of_data_is_5_l63_63065

theorem mean_of_data_is_5 (h : s^2 = (1 / 4) * ((3.2 - x)^2 + (5.7 - x)^2 + (4.3 - x)^2 + (6.8 - x)^2))
  : x = 5 := 
sorry

end mean_of_data_is_5_l63_63065


namespace tan_435_eq_2_add_sqrt_3_l63_63345

theorem tan_435_eq_2_add_sqrt_3 :
  Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  sorry

end tan_435_eq_2_add_sqrt_3_l63_63345


namespace factory_production_schedule_l63_63441

noncomputable def production_equation (x : ℝ) : Prop :=
  (1000 / x) - (1000 / (1.2 * x)) = 2

theorem factory_production_schedule (x : ℝ) (hx : x ≠ 0) : production_equation x := 
by 
  -- Assumptions based on conditions:
  -- Factory plans to produce total of 1000 sets of protective clothing.
  -- Actual production is 20% more than planned.
  -- Task completed 2 days ahead of original schedule.
  -- We need to show: (1000 / x) - (1000 / (1.2 * x)) = 2
  sorry

end factory_production_schedule_l63_63441


namespace find_a_b_l63_63365

theorem find_a_b (a b : ℝ)
  (h1 : a < 0)
  (h2 : (-b / a) = -((1 / 2) - (1 / 3)))
  (h3 : (2 / a) = -((1 / 2) * (1 / 3))) : 
  a + b = -14 :=
sorry

end find_a_b_l63_63365


namespace circle_radius_5_l63_63000

-- The circle equation given
def circle_eq (x y : ℝ) (c : ℝ) : Prop :=
  x^2 + 4 * x + y^2 + 8 * y + c = 0

-- The radius condition given
def radius_condition : Prop :=
  5 = (25 : ℝ).sqrt

-- The final proof statement
theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, circle_eq x y c) → radius_condition → c = -5 := 
by
  sorry

end circle_radius_5_l63_63000


namespace range_of_m_l63_63066

noncomputable def f (x : ℝ) : ℝ := 1 + Real.sin (2 * x)

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + m

theorem range_of_m (m : ℝ) :
  (∃ x₀ ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x₀ ≥ g x₀ m) → m ≤ Real.sqrt 2 :=
by
  intro h
  sorry

end range_of_m_l63_63066


namespace factor_exp_l63_63358

variable (x : ℤ)

theorem factor_exp : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) :=
by
  sorry

end factor_exp_l63_63358


namespace solve_for_y_l63_63211

theorem solve_for_y (x y : ℝ) (h : (x + y)^5 - x^5 + y = 0) : y = 0 :=
sorry

end solve_for_y_l63_63211


namespace sum_first_7_terms_is_105_l63_63654

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables {a d : ℕ}
axiom a4_is_15 : arithmetic_seq a d 4 = 15

-- Goal/theorem to be proven
theorem sum_first_7_terms_is_105 : sum_arithmetic_seq a d 7 = 105 :=
sorry

end sum_first_7_terms_is_105_l63_63654


namespace min_value_trig_expr_l63_63639

theorem min_value_trig_expr : 
  ∃ A : ℝ, A ∈ set.Icc 0 (2 * Real.pi) ∧
  2 * Real.sin (A / 2) + Real.sin A = -4 :=
sorry

end min_value_trig_expr_l63_63639


namespace equal_functions_A_l63_63028

-- Define the functions
def f₁ (x : ℝ) : ℝ := x^2 - 2*x - 1
def f₂ (t : ℝ) : ℝ := t^2 - 2*t - 1

-- Theorem stating that f₁ is equal to f₂
theorem equal_functions_A : ∀ x : ℝ, f₁ x = f₂ x :=
by
  intros x
  sorry

end equal_functions_A_l63_63028


namespace range_of_m_l63_63198

variable (m : ℝ) -- variable m in the real numbers

-- Definition of proposition p
def p : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0

-- Definition of proposition q
def q : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- The theorem statement with the given conditions
theorem range_of_m (h : p m ∧ q m) : -2 < m ∧ m < 0 :=
by
  sorry

end range_of_m_l63_63198


namespace derivative_of_y_correct_l63_63143

noncomputable def derivative_of_y (x : ℝ) : ℝ :=
  let y := (4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))) / (16 + (Real.log 4) ^ 2)
  let u := 4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))
  let v := 16 + (Real.log 4) ^ 2
  let du_dx := (4^x * Real.log 4) * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x)) +
               (4^x) * (4 * Real.log 4 * Real.cos (4 * x) + 16 * Real.sin (4 * x))
  let dv_dx := 0
  (du_dx * v - u * dv_dx) / (v ^ 2)

theorem derivative_of_y_correct (x : ℝ) : derivative_of_y x = 4^x * Real.sin (4 * x) :=
  sorry

end derivative_of_y_correct_l63_63143


namespace proof_x_minus_y_squared_l63_63975

-- Define the variables x and y
variables (x y : ℝ)

-- Assume the given conditions
def cond1 : (x + y)^2 = 64 := sorry
def cond2 : x * y = 12 := sorry

-- Formulate the main goal to prove
theorem proof_x_minus_y_squared : (x - y)^2 = 16 :=
by
  have hx_add_y_sq : (x + y)^2 = 64 := cond1
  have hxy : x * y = 12 := cond2
  -- Use the identities and the given conditions to prove the statement
  sorry

end proof_x_minus_y_squared_l63_63975


namespace bug_visits_tiles_l63_63154

theorem bug_visits_tiles (width length : ℕ) (gcd_width_length : ℕ) (broken_tile : ℕ × ℕ)
  (h_width : width = 12) (h_length : length = 25) (h_gcd : gcd_width_length = Nat.gcd width length)
  (h_broken_tile : broken_tile = (12, 18)) :
  width + length - gcd_width_length = 36 := by
  sorry

end bug_visits_tiles_l63_63154


namespace miles_left_l63_63240

theorem miles_left (d_total d_covered d_left : ℕ) 
  (h₁ : d_total = 78) 
  (h₂ : d_covered = 32) 
  (h₃ : d_left = d_total - d_covered):
  d_left = 46 := 
by {
  sorry 
}

end miles_left_l63_63240


namespace income_expenditure_ratio_l63_63118

noncomputable def I : ℝ := 19000
noncomputable def S : ℝ := 3800
noncomputable def E : ℝ := I - S

theorem income_expenditure_ratio : (I / E) = 5 / 4 := by
  sorry

end income_expenditure_ratio_l63_63118


namespace valerie_initial_money_l63_63281

theorem valerie_initial_money (n m C_s C_l L I : ℕ) 
  (h1 : n = 3) (h2 : m = 1) (h3 : C_s = 8) (h4 : C_l = 12) (h5 : L = 24) :
  I = (n * C_s) + (m * C_l) + L :=
  sorry

end valerie_initial_money_l63_63281


namespace problem_false_proposition_l63_63962

def p : Prop := ∀ x : ℝ, |x| = x ↔ x > 0

def q : Prop := (¬ ∃ x : ℝ, x^2 - x > 0) ↔ ∀ x : ℝ, x^2 - x ≤ 0

theorem problem_false_proposition : ¬ (p ∧ q) :=
by
  sorry

end problem_false_proposition_l63_63962


namespace tangent_line_equation_l63_63246

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P.2 = P.1^2)
  (h_perpendicular : ∃ k : ℝ, k * -1/2 = -1) : 
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end tangent_line_equation_l63_63246


namespace roots_of_cubic_eq_l63_63073

theorem roots_of_cubic_eq (r s t p q : ℝ) (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) 
(h3 : r * s * t = r) : r^2 + s^2 + t^2 = p^2 - 2 * q := 
by 
  sorry

end roots_of_cubic_eq_l63_63073


namespace lesser_fraction_exists_l63_63272

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l63_63272


namespace benny_start_cards_l63_63491

--- Benny bought 4 new cards before the dog ate half of his collection.
def new_cards : Int := 4

--- The remaining cards after the dog ate half of the collection is 34.
def remaining_cards : Int := 34

--- The total number of cards Benny had before adding the new cards and the dog ate half.
def total_before_eating := remaining_cards * 2

theorem benny_start_cards : total_before_eating - new_cards = 64 :=
sorry

end benny_start_cards_l63_63491


namespace interest_rate_difference_l63_63762

theorem interest_rate_difference:
  ∀ (R H: ℝ),
    (300 * (H / 100) * 5 = 300 * (R / 100) * 5 + 90) →
    (H - R = 6) :=
by
  intros R H h
  sorry

end interest_rate_difference_l63_63762


namespace piravena_flight_cost_l63_63775

noncomputable def cost_of_flight (distance_km : ℕ) (booking_fee : ℕ) (rate_per_km : ℕ) : ℕ :=
  booking_fee + (distance_km * rate_per_km / 100)

def check_cost_of_flight : Prop :=
  let distance_bc := 1000
  let booking_fee := 100
  let rate_per_km := 10
  cost_of_flight distance_bc booking_fee rate_per_km = 200

theorem piravena_flight_cost : check_cost_of_flight := 
by {
  sorry
}

end piravena_flight_cost_l63_63775


namespace correct_value_two_decimal_places_l63_63292

theorem correct_value_two_decimal_places (x : ℝ) 
  (h1 : 8 * x + 8 = 56) : 
  (x / 8) + 7 = 7.75 :=
sorry

end correct_value_two_decimal_places_l63_63292


namespace jim_age_in_2_years_l63_63745

theorem jim_age_in_2_years (c1 : ∀ t : ℕ, t = 37) (c2 : ∀ j : ℕ, j = 27) : ∀ j2 : ℕ, j2 = 29 :=
by
  sorry

end jim_age_in_2_years_l63_63745


namespace function_domain_l63_63117

noncomputable def sqrt_domain : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ 2 - x > 0 ∧ 2 - x ≠ 1}

theorem function_domain :
  sqrt_domain = {x | -1 ≤ x ∧ x < 1} ∪ {x | 1 < x ∧ x < 2} :=
by
  sorry

end function_domain_l63_63117


namespace find_fx_for_negative_x_l63_63943

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  ∀ x, (0 < x) → f x = x * (1 - x)

theorem find_fx_for_negative_x (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_given : given_function f) :
  ∀ x, (x < 0) → f x = x + x^2 :=
by
  sorry

end find_fx_for_negative_x_l63_63943


namespace tangent_line_at_1_l63_63375

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

noncomputable def f' (x : ℝ) : ℝ := 1 / x - 3

theorem tangent_line_at_1 :
  let y := f 1
  let k := f' 1
  y = -3 ∧ k = -2 →
  ∀ (x y : ℝ), y = k * (x - 1) + f 1 ↔ 2 * x + y + 1 = 0 :=
by
  sorry

end tangent_line_at_1_l63_63375


namespace initial_marbles_count_l63_63350

-- Define the conditions
def marbles_given_to_mary : ℕ := 14
def marbles_remaining : ℕ := 50

-- Prove that Dan's initial number of marbles is 64
theorem initial_marbles_count : marbles_given_to_mary + marbles_remaining = 64 := 
by {
  sorry
}

end initial_marbles_count_l63_63350


namespace allocation_of_branches_and_toys_l63_63890

-- Define the number of branches and toys as natural numbers
variables (b t : ℕ)

-- Define the conditions
def condition_1 := t = b + 1
def condition_2 := 2 * b = t - 1

-- The main theorem to prove
theorem allocation_of_branches_and_toys (hb : b = 3) (ht : t = 4) :
  condition_1 b t ∧ condition_2 b t :=
begin
  -- This is the correct number of branches and toys
  -- Proof is omitted
  sorry
end

end allocation_of_branches_and_toys_l63_63890


namespace radius_first_circle_l63_63737

theorem radius_first_circle
  (A B C D E : Type)
  [MetricSpace B]
  (d : ℝ)
  (h1 : Segment A B = Diameter Circle1)
  (h2 : Center Circle2 = B)
  (h3 : radius Circle2 = 2)
  (h4 : intersects Circle1 Circle2 C)
  (h5 : Chord.Cons Circle2 CE: Chord CE = 3)
  (h6 : tangent_to Circle1 CE) :
  radius Circle1 = 4 / Real.sqrt 7 :=
by
  sorry

end radius_first_circle_l63_63737


namespace perpendicular_tangent_line_l63_63186

theorem perpendicular_tangent_line :
  ∃ m : ℝ, ∃ x₀ : ℝ, y₀ = x₀ ^ 3 + 3 * x₀ ^ 2 - 1 ∧ y₀ = -3 * x₀ + m ∧ 
  (∀ x, x ≠ x₀ → x ^ 3 + 3 * x ^ 2 - 1 ≠ -3 * x + m) ∧ m = -2 := 
sorry

end perpendicular_tangent_line_l63_63186


namespace polynomial_102_l63_63290

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end polynomial_102_l63_63290


namespace fraction_simplest_form_l63_63882

def fracA (a b : ℤ) : ℤ × ℤ := (|2 * a|, 5 * a^2 * b)
def fracB (a : ℤ) : ℤ × ℤ := (a, a^2 - 2 * a)
def fracC (a b : ℤ) : ℤ × ℤ := (3 * a + b, a + b)
def fracD (a b : ℤ) : ℤ × ℤ := (a^2 - a * b, a^2 - b^2)

theorem fraction_simplest_form (a b : ℤ) : (fracC a b).1 / (fracC a b).2 = (3 * a + b) / (a + b) :=
by sorry

end fraction_simplest_form_l63_63882


namespace simplify_expression_l63_63725

theorem simplify_expression (x : ℝ) :
  (3 * x)^5 + (4 * x^2) * (3 * x^2) = 243 * x^5 + 12 * x^4 :=
by
  sorry

end simplify_expression_l63_63725


namespace range_of_a_l63_63674

noncomputable def range_of_a_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, |x + 1| + |x - a| ≤ 2

theorem range_of_a : ∀ a : ℝ, range_of_a_condition a → (-3 : ℝ) ≤ a ∧ a ≤ 1 :=
by
  intros a h
  sorry

end range_of_a_l63_63674


namespace class_average_correct_l63_63532

-- Define the constants as per the problem data
def total_students : ℕ := 30
def students_group_1 : ℕ := 24
def students_group_2 : ℕ := 6
def avg_score_group_1 : ℚ := 85 / 100  -- 85%
def avg_score_group_2 : ℚ := 92 / 100  -- 92%

-- Calculate total scores and averages based on the defined constants
def total_score_group_1 : ℚ := students_group_1 * avg_score_group_1
def total_score_group_2 : ℚ := students_group_2 * avg_score_group_2
def total_class_score : ℚ := total_score_group_1 + total_score_group_2
def class_average : ℚ := total_class_score / total_students

-- Goal: Prove that class_average is 86.4%
theorem class_average_correct : class_average = 86.4 / 100 := sorry

end class_average_correct_l63_63532


namespace value_of_f_3x_minus_7_l63_63235

def f (x : ℝ) : ℝ := 3 * x + 5

theorem value_of_f_3x_minus_7 (x : ℝ) : f (3 * x - 7) = 9 * x - 16 :=
by
  -- Proof goes here
  sorry

end value_of_f_3x_minus_7_l63_63235


namespace total_value_of_gold_l63_63094

theorem total_value_of_gold (legacy_bars : ℕ) (aleena_bars : ℕ) (bar_value : ℕ) (total_gold_value : ℕ) 
  (h1 : legacy_bars = 12) 
  (h2 : aleena_bars = legacy_bars - 4)
  (h3 : bar_value = 3500) : 
  total_gold_value = (legacy_bars + aleena_bars) * bar_value := 
by 
  sorry

end total_value_of_gold_l63_63094


namespace find_interest_rate_l63_63264

noncomputable def compound_interest_rate (A P : ℝ) (t n : ℕ) : ℝ := sorry

theorem find_interest_rate :
  compound_interest_rate 676 625 2 1 = 0.04 := 
sorry

end find_interest_rate_l63_63264


namespace boat_crossing_l63_63866

theorem boat_crossing (students teacher trips people_in_boat : ℕ) (h_students : students = 13) (h_teacher : teacher = 1) (h_boat_capacity : people_in_boat = 5) :
  trips = (students + teacher + people_in_boat - 1) / (people_in_boat - 1) :=
by
  sorry

end boat_crossing_l63_63866


namespace Kayla_total_items_l63_63696

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l63_63696


namespace gabriel_forgot_days_l63_63945

def days_in_july : ℕ := 31
def days_taken : ℕ := 28

theorem gabriel_forgot_days : days_in_july - days_taken = 3 := by
  sorry

end gabriel_forgot_days_l63_63945


namespace problem_solution_l63_63793

-- Given non-zero numbers x and y such that x = 1 / y,
-- prove that (2x - 1/x) * (y - 1/y) = -2x^2 + y^2.
theorem problem_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x = 1 / y) :
  (2 * x - 1 / x) * (y - 1 / y) = -2 * x^2 + y^2 :=
by
  sorry

end problem_solution_l63_63793


namespace neg_one_power_zero_l63_63912

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end neg_one_power_zero_l63_63912


namespace archer_total_fish_caught_l63_63770

noncomputable def total_fish_caught (initial : ℕ) (second_extra : ℕ) (third_round_percentage : ℕ) : ℕ :=
  let second_round := initial + second_extra
  let total_after_two_rounds := initial + second_round
  let third_round := second_round + (third_round_percentage * second_round) / 100
  total_after_two_rounds + third_round

theorem archer_total_fish_caught :
  total_fish_caught 8 12 60 = 60 :=
by
  -- Theorem statement to prove the total fish caught equals 60 given the conditions.
  sorry

end archer_total_fish_caught_l63_63770


namespace least_number_to_subtract_l63_63594

theorem least_number_to_subtract (n : ℕ) (h : n = 9876543210) : 
  ∃ m, m = 6 ∧ (n - m) % 29 = 0 := 
sorry

end least_number_to_subtract_l63_63594


namespace total_fish_count_l63_63629

theorem total_fish_count (kyle_caught_same_as_tasha : ∀ kyle tasha : ℕ, kyle = tasha) 
  (carla_caught : ℕ) (kyle_caught : ℕ) (tasha_caught : ℕ)
  (h0 : carla_caught = 8) (h1 : kyle_caught = 14) (h2 : tasha_caught = kyle_caught) : 
  8 + 14 + 14 = 36 :=
by sorry

end total_fish_count_l63_63629


namespace student_average_vs_true_average_l63_63329

theorem student_average_vs_true_average (w x y z : ℝ) (h : w < x ∧ x < y ∧ y < z) : 
  (2 * w + 2 * x + y + z) / 6 < (w + x + y + z) / 4 :=
by
  sorry

end student_average_vs_true_average_l63_63329


namespace sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l63_63364

theorem sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5 :
  let smallest := 125
  let largest := 521
  smallest + largest = 646 := by
  sorry

end sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l63_63364


namespace boat_travel_difference_l63_63319

-- Define the speeds
variables (a b : ℝ) (ha : a > b)

-- Define the travel times
def downstream_time := 3
def upstream_time := 2

-- Define the distances
def downstream_distance := downstream_time * (a + b)
def upstream_distance := upstream_time * (a - b)

-- Prove the mathematical statement
theorem boat_travel_difference : downstream_distance a b - upstream_distance a b = a + 5 * b := by
  -- sorry can be used to skip the proof
  sorry

end boat_travel_difference_l63_63319


namespace arithmetic_sequence_general_formula_and_extremum_l63_63952

noncomputable def a (n : ℕ) : ℤ := sorry
def S (n : ℕ) : ℤ := sorry

theorem arithmetic_sequence_general_formula_and_extremum :
  (a 1 + a 4 = 8) ∧ (a 2 * a 3 = 15) →
  (∃ c d : ℤ, (∀ n : ℕ, a n = c * n + d) ∨ (∀ n : ℕ, a n = -c * n + d)) ∧
  ((∃ n_min : ℕ, n_min > 0 ∧ S n_min = 1) ∧ (∃ n_max : ℕ, n_max > 0 ∧ S n_max = 16)) :=
by
  sorry

end arithmetic_sequence_general_formula_and_extremum_l63_63952


namespace range_f_l63_63517

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x - Real.cos x

theorem range_f : Set.range f = Set.Icc (-2 : ℝ) 2 := 
by
  sorry

end range_f_l63_63517


namespace find_percentage_l63_63151

theorem find_percentage (P N : ℕ) (h₁ : N = 125) (h₂ : N = (P * N / 100) + 105) : P = 16 :=
by
  sorry

end find_percentage_l63_63151


namespace sachin_younger_than_rahul_l63_63107

theorem sachin_younger_than_rahul :
  ∀ (sachin_age rahul_age : ℕ),
  (sachin_age / rahul_age = 6 / 9) →
  (sachin_age = 14) →
  (rahul_age - sachin_age = 7) :=
by
  sorry

end sachin_younger_than_rahul_l63_63107


namespace good_oranges_per_month_l63_63854

/-- Salaria has 50% of tree A and 50% of tree B, totaling to 10 trees.
    Tree A gives 10 oranges a month and 60% are good.
    Tree B gives 15 oranges a month and 1/3 are good.
    Prove that the total number of good oranges Salaria gets per month is 55. -/
theorem good_oranges_per_month 
  (total_trees : ℕ) 
  (percent_tree_A : ℝ) 
  (percent_tree_B : ℝ) 
  (oranges_tree_A : ℕ)
  (good_percent_A : ℝ)
  (oranges_tree_B : ℕ)
  (good_ratio_B : ℝ)
  (H1 : total_trees = 10)
  (H2 : percent_tree_A = 0.5)
  (H3 : percent_tree_B = 0.5)
  (H4 : oranges_tree_A = 10)
  (H5 : good_percent_A = 0.6)
  (H6 : oranges_tree_B = 15)
  (H7 : good_ratio_B = 1/3)
  : (total_trees * percent_tree_A * oranges_tree_A * good_percent_A) + 
    (total_trees * percent_tree_B * oranges_tree_B * good_ratio_B) = 55 := 
  by 
    sorry

end good_oranges_per_month_l63_63854


namespace log_expression_value_l63_63955

theorem log_expression_value (x : ℝ) (hx : x < 1) (h : (Real.log x / Real.log 10)^3 - 2 * (Real.log (x^3) / Real.log 10) = 150) :
  (Real.log x / Real.log 10)^4 - (Real.log (x^4) / Real.log 10) = 645 := 
sorry

end log_expression_value_l63_63955


namespace cheryl_material_used_l63_63604

theorem cheryl_material_used :
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  used = (52 / 247 : ℚ) :=
by
  let material1 := (4 / 19 : ℚ)
  let material2 := (2 / 13 : ℚ)
  let bought := material1 + material2
  let leftover := (4 / 26 : ℚ)
  let used := bought - leftover
  have : used = (52 / 247 : ℚ) := sorry
  exact this

end cheryl_material_used_l63_63604


namespace probability_of_at_least_one_three_l63_63895

noncomputable def is_valid_roll (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 8

noncomputable def valid_rolls : List ℕ := [1,2,3,4,5,6,7,8]

noncomputable def at_least_one_three (rolls : List ℕ) : Prop := 3 ∈ rolls

theorem probability_of_at_least_one_three :
  ∀ (X1 X2 X3 X4 : ℕ),
    is_valid_roll X1 →
    is_valid_roll X2 →
    is_valid_roll X3 →
    is_valid_roll X4 →
    (X1 + X2 + X3 = X4) →
    (∃ n : ℚ, n = (5:ℚ) / 12) :=
  sorry

end probability_of_at_least_one_three_l63_63895


namespace avg_salary_l63_63425

-- Conditions as definitions
def number_of_technicians : Nat := 7
def salary_per_technician : Nat := 10000
def number_of_workers : Nat := 14
def salary_per_non_technician : Nat := 6000

-- Total salary of technicians
def total_salary_technicians : Nat := number_of_technicians * salary_per_technician

-- Number of non-technicians
def number_of_non_technicians : Nat := number_of_workers - number_of_technicians

-- Total salary of non-technicians
def total_salary_non_technicians : Nat := number_of_non_technicians * salary_per_non_technician

-- Total salary
def total_salary_all_workers : Nat := total_salary_technicians + total_salary_non_technicians

-- Average salary of all workers
def avg_salary_all_workers : Nat := total_salary_all_workers / number_of_workers

-- Theorem to prove
theorem avg_salary (A : Nat) (h : A = avg_salary_all_workers) : A = 8000 := by
  sorry

end avg_salary_l63_63425


namespace conor_vegetables_per_week_l63_63634

theorem conor_vegetables_per_week : 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := 
by 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  show (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 
  sorry

end conor_vegetables_per_week_l63_63634


namespace factorial_expression_l63_63030

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l63_63030


namespace triangle_side_ratio_l63_63220

theorem triangle_side_ratio (a b c: ℝ) (A B C: ℝ) (h1: b * Real.cos C + c * Real.cos B = 2 * b) :
  a / b = 2 :=
sorry

end triangle_side_ratio_l63_63220


namespace kayla_total_items_l63_63698

theorem kayla_total_items (Tc : ℕ) (Ts : ℕ) (Kc : ℕ) (Ks : ℕ) 
  (h1 : Tc = 2 * Kc) (h2 : Ts = 2 * Ks) (h3 : Tc = 12) (h4 : Ts = 18) : Kc + Ks = 15 :=
by
  sorry

end kayla_total_items_l63_63698


namespace exponential_monotonicity_example_l63_63193

theorem exponential_monotonicity_example (m n : ℕ) (a b : ℝ) (h1 : a = 0.2 ^ m) (h2 : b = 0.2 ^ n) (h3 : m > n) : a < b :=
by
  sorry

end exponential_monotonicity_example_l63_63193


namespace factory_production_schedule_l63_63442

noncomputable def production_equation (x : ℝ) : Prop :=
  (1000 / x) - (1000 / (1.2 * x)) = 2

theorem factory_production_schedule (x : ℝ) (hx : x ≠ 0) : production_equation x := 
by 
  -- Assumptions based on conditions:
  -- Factory plans to produce total of 1000 sets of protective clothing.
  -- Actual production is 20% more than planned.
  -- Task completed 2 days ahead of original schedule.
  -- We need to show: (1000 / x) - (1000 / (1.2 * x)) = 2
  sorry

end factory_production_schedule_l63_63442


namespace correct_log_values_l63_63609

theorem correct_log_values (a b c : ℝ)
                          (log_027 : ℝ) (log_21 : ℝ) (log_1_5 : ℝ) (log_2_8 : ℝ)
                          (log_3 : ℝ) (log_5 : ℝ) (log_6 : ℝ) (log_7 : ℝ)
                          (log_8 : ℝ) (log_9 : ℝ) (log_14 : ℝ) :
  (log_3 = 2 * a - b) →
  (log_5 = a + c) →
  (log_6 = 1 + a - b - c) →
  (log_7 = 2 * (b + c)) →
  (log_9 = 4 * a - 2 * b) →
  (log_1_5 = 3 * a - b + c) →
  (log_14 = 1 - c + 2 * b) →
  (log_1_5 = 3 * a - b + c - 1) ∧ (log_7 = 2 * b + c) := sorry

end correct_log_values_l63_63609


namespace joanne_total_weekly_earnings_l63_63994

-- Define the earnings per hour and hours worked per day for the main job
def mainJobHourlyWage : ℝ := 16
def mainJobDailyHours : ℝ := 8

-- Compute daily earnings from the main job
def mainJobDailyEarnings : ℝ := mainJobHourlyWage * mainJobDailyHours

-- Define the earnings per hour and hours worked per day for the part-time job
def partTimeJobHourlyWage : ℝ := 13.5
def partTimeJobDailyHours : ℝ := 2

-- Compute daily earnings from the part-time job
def partTimeJobDailyEarnings : ℝ := partTimeJobHourlyWage * partTimeJobDailyHours

-- Compute total daily earnings from both jobs
def totalDailyEarnings : ℝ := mainJobDailyEarnings + partTimeJobDailyEarnings

-- Define the number of workdays per week
def workDaysPerWeek : ℝ := 5

-- Compute total weekly earnings
def totalWeeklyEarnings : ℝ := totalDailyEarnings * workDaysPerWeek

-- The problem statement to prove: Joanne's total weekly earnings = 775
theorem joanne_total_weekly_earnings :
  totalWeeklyEarnings = 775 :=
by
  sorry

end joanne_total_weekly_earnings_l63_63994


namespace range_of_a_l63_63946

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → (x < a ∨ x > a + 4)) ∧ ¬(∀ x : ℝ, (x < a ∨ x > a + 4) → -2 ≤ x ∧ x ≤ 1) ↔
  a > 1 ∨ a < -6 :=
by {
  sorry
}

end range_of_a_l63_63946


namespace determine_location_with_coords_l63_63484

-- Define the conditions as a Lean structure
structure Location where
  longitude : ℝ
  latitude : ℝ

-- Define the specific location given in option ①
def location_118_40 : Location :=
  {longitude := 118, latitude := 40}

-- Define the theorem and its statement
theorem determine_location_with_coords :
  ∃ loc : Location, loc = location_118_40 := 
  by
  sorry -- Placeholder for the proof

end determine_location_with_coords_l63_63484


namespace product_of_first_nine_terms_l63_63539

-- Declare the geometric sequence and given condition
variable {α : Type*} [Field α]
variable {a : ℕ → α}
variable (r : α) (a1 : α)

-- Define that the sequence is geometric
def is_geometric_sequence (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  ∀ n : ℕ, a n = a1 * r ^ n

-- Given a_5 = -2 in the sequence
def geometric_sequence_with_a5 (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  is_geometric_sequence a r a1 ∧ a 5 = -2

-- Prove that the product of the first 9 terms is -512
theorem product_of_first_nine_terms 
  (a : ℕ → α) 
  (r : α) 
  (a₁ : α) 
  (h : geometric_sequence_with_a5 a r a₁) : 
  (a 0) * (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = -512 := 
by
  sorry

end product_of_first_nine_terms_l63_63539


namespace square_difference_l63_63970

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 64) (h₂ : x * y = 12) : (x - y)^2 = 16 := by
  -- proof would go here
  sorry

end square_difference_l63_63970


namespace problem_statement_l63_63712

-- Define the conditions:
def f (x : ℚ) : ℚ := sorry

axiom f_mul (a b : ℚ) : f (a * b) = f a + f b
axiom f_int (n : ℤ) : f (n : ℚ) = (n : ℚ)

-- The problem statement:
theorem problem_statement : f (8/13) < 0 :=
sorry

end problem_statement_l63_63712


namespace paint_required_for_small_statues_l63_63669

-- Constants definition
def pint_per_8ft_statue : ℕ := 1
def height_original_statue : ℕ := 8
def height_small_statue : ℕ := 2
def number_of_small_statues : ℕ := 400

-- Theorem statement
theorem paint_required_for_small_statues :
  pint_per_8ft_statue = 1 →
  height_original_statue = 8 →
  height_small_statue = 2 →
  number_of_small_statues = 400 →
  (number_of_small_statues * (pint_per_8ft_statue * (height_small_statue / height_original_statue)^2)) = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end paint_required_for_small_statues_l63_63669


namespace unit_digit_product_7858_1086_4582_9783_l63_63606

theorem unit_digit_product_7858_1086_4582_9783 : 
  (7858 * 1086 * 4582 * 9783) % 10 = 8 :=
by
  -- Given that the unit digits of the numbers are 8, 6, 2, and 3.
  let d1 := 7858 % 10 -- This unit digit is 8
  let d2 := 1086 % 10 -- This unit digit is 6
  let d3 := 4582 % 10 -- This unit digit is 2
  let d4 := 9783 % 10 -- This unit digit is 3
  -- We need to prove that the unit digit of the product is 8
  sorry -- The actual proof steps are skipped

end unit_digit_product_7858_1086_4582_9783_l63_63606


namespace circle_radius_l63_63653

theorem circle_radius (m : ℝ) (h : 2 * 1 + (-m / 2) = 0) :
  let radius := 1 / 2 * Real.sqrt (4 + m ^ 2 + 16)
  radius = 3 :=
by
  sorry

end circle_radius_l63_63653


namespace permutation_six_two_l63_63144

-- Definition for permutation
def permutation (n k : ℕ) : ℕ := n * (n - 1)

-- Theorem stating that the permutation of 6 taken 2 at a time is 30
theorem permutation_six_two : permutation 6 2 = 30 :=
by
  -- proof will be filled here
  sorry

end permutation_six_two_l63_63144


namespace proof_f_f_f_3_l63_63995

def f (n : ℤ) : ℤ :=
  if n < 5
  then n^2 + 1
  else 2 * n - 3

theorem proof_f_f_f_3 :
  f (f (f 3)) = 31 :=
by 
  -- Here, we skip the proof as instructed
  sorry

end proof_f_f_f_3_l63_63995


namespace d_value_l63_63392

theorem d_value (d : ℝ) : (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ (d = 5) := by 
sorry

end d_value_l63_63392


namespace prove_x_minus_y_squared_l63_63973

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l63_63973


namespace correct_operation_l63_63883

theorem correct_operation (a b : ℝ) : 
  ¬(a^2 + a^3 = a^5) ∧ ¬((a^2)^3 = a^8) ∧ (a^3 / a^2 = a) ∧ ¬((a - b)^2 = a^2 - b^2) := 
by {
  sorry
}

end correct_operation_l63_63883


namespace peter_age_problem_l63_63317

theorem peter_age_problem
  (P J : ℕ) 
  (h1 : J = P + 12)
  (h2 : P - 10 = 1/3 * (J - 10)) : P = 16 :=
sorry

end peter_age_problem_l63_63317


namespace geometric_sequence_general_term_formula_no_arithmetic_sequence_l63_63195

-- Assume we have a sequence {a_n} and its sum of the first n terms S_n where S_n = 2a_n - n (for n ∈ ℕ*)
variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}

-- Condition 1: S_n = 2a_n - n
axiom Sn_condition (n : ℕ) (h : n > 0) : S_n n = 2 * a_n n - n

-- 1. Prove that the sequence {a_n + 1} is a geometric sequence with first term and common ratio equal to 2
theorem geometric_sequence (n : ℕ) (h : n > 0) : ∃ r : ℕ, r = 2 ∧ ∀ m : ℕ, a_n (m + 1) + 1 = r * (a_n m + 1) :=
by
  sorry

-- 2. Prove the general term formula an = 2^n - 1
theorem general_term_formula (n : ℕ) (h : n > 0) : a_n n = 2^n - 1 :=
by
  sorry

-- 3. Prove that there do not exist three consecutive terms in {a_n} that form an arithmetic sequence
theorem no_arithmetic_sequence (n k : ℕ) (h : n > 0 ∧ k > 0 ∧ k + 2 < n) : ¬(a_n k + a_n (k + 2) = 2 * a_n (k + 1)) :=
by
  sorry

end geometric_sequence_general_term_formula_no_arithmetic_sequence_l63_63195


namespace factorial_expression_l63_63032

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l63_63032


namespace sequoia_taller_than_maple_l63_63992

def height_maple_tree : ℚ := 13 + 3/4
def height_sequoia : ℚ := 20 + 1/2

theorem sequoia_taller_than_maple : (height_sequoia - height_maple_tree) = 6 + 3/4 :=
by
  sorry

end sequoia_taller_than_maple_l63_63992


namespace fraction_simplification_l63_63941

open Real -- Open the Real namespace for real number operations

theorem fraction_simplification (a x : ℝ) : 
  (sqrt (a^2 + x^2) - (x^2 + a^2) / sqrt (a^2 + x^2)) / (a^2 + x^2) = 0 := 
sorry

end fraction_simplification_l63_63941


namespace number_of_ways_is_64_l63_63071

-- Definition of the problem conditions
def ways_to_sign_up (students groups : ℕ) : ℕ :=
  groups ^ students

-- Theorem statement asserting that for 3 students and 4 groups, the number of ways is 64
theorem number_of_ways_is_64 : ways_to_sign_up 3 4 = 64 :=
by sorry

end number_of_ways_is_64_l63_63071


namespace equivalence_of_statements_l63_63599

theorem equivalence_of_statements 
  (Q P : Prop) :
  (Q → ¬ P) ↔ (P → ¬ Q) := sorry

end equivalence_of_statements_l63_63599


namespace sum_of_infinite_perimeters_l63_63767

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_infinite_perimeters (a : ℝ) :
  let first_perimeter := 3 * a
  let common_ratio := (1/3 : ℝ)
  let S := geometric_series_sum first_perimeter common_ratio 0
  S = (9 * a / 2) :=
by
  sorry

end sum_of_infinite_perimeters_l63_63767


namespace angle_cosine_third_quadrant_l63_63389

theorem angle_cosine_third_quadrant (B : ℝ) (h1 : π < B ∧ B < 3 * π / 2) (h2 : Real.sin B = 4 / 5) :
  Real.cos B = -3 / 5 :=
sorry

end angle_cosine_third_quadrant_l63_63389


namespace lesser_fraction_exists_l63_63273

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end lesser_fraction_exists_l63_63273


namespace find_reduced_price_l63_63023

noncomputable def reduced_price_per_kg 
  (total_spent : ℝ) (original_quantity : ℝ) (additional_quantity : ℝ) (price_reduction_rate : ℝ) : ℝ :=
  let original_price := total_spent / original_quantity
  let reduced_price := original_price * (1 - price_reduction_rate)
  reduced_price

theorem find_reduced_price 
  (total_spent : ℝ := 800)
  (original_quantity : ℝ := 20)
  (additional_quantity : ℝ := 5)
  (price_reduction_rate : ℝ := 0.15) :
  reduced_price_per_kg total_spent original_quantity additional_quantity price_reduction_rate = 34 :=
by
  sorry

end find_reduced_price_l63_63023


namespace number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l63_63562

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of rooms
def total_rooms : ℕ := 2

-- For the first question, define: each room must receive exact three people
def room_receives_three_people (n m : ℕ) : Prop :=
  n = 3 ∧ m = 3

-- For the second question, define: each room must receive at least one person
def room_receives_at_least_one_person (n m : ℕ) : Prop :=
  n ≥ 1 ∧ m ≥ 1

theorem number_of_distributions_room_receives_three_people :
  ∃ (ways : ℕ), ways = 20 :=
by
  sorry

theorem number_of_distributions_room_receives_at_least_one_person :
  ∃ (ways : ℕ), ways = 62 :=
by
  sorry

end number_of_distributions_room_receives_three_people_number_of_distributions_room_receives_at_least_one_person_l63_63562


namespace g_symmetric_about_pi_div_12_l63_63395

noncomputable def g (x : ℝ) : ℝ := 1/2 - Real.sin (2 * x + Real.pi / 3)

theorem g_symmetric_about_pi_div_12 :
  ∀ x : ℝ, g (2 * (Real.pi / 12) - x) = g x :=
sorry

end g_symmetric_about_pi_div_12_l63_63395


namespace smallest_positive_period_max_value_in_interval_l63_63518

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sin x) ^ 2 - Real.cos (2 * x + Real.pi / 3)

theorem smallest_positive_period :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ T = Real.pi :=
sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 5 / 2 :=
sorry

end smallest_positive_period_max_value_in_interval_l63_63518


namespace part1_part2_part3_l63_63662

universe u

def A : Set ℝ := {x | -3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12 * x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}
def CR_A : Set ℝ := {x | x < -3 ∨ x ≥ 7}

theorem part1 : A ∪ B = {x | -3 ≤ x ∧ x < 10} := by
  sorry

theorem part2 : CR_A ∩ B = {x | 7 ≤ x ∧ x < 10} := by
  sorry

theorem part3 (a : ℝ) (h : (A ∩ C a).Nonempty) : a > -3 := by
  sorry

end part1_part2_part3_l63_63662


namespace length_AM_is_correct_l63_63464

-- Definitions of the problem conditions
def length_of_square : ℝ := 9

def ratio_AP_PB : ℝ × ℝ := (7, 2)

def radius_of_quarter_circle : ℝ := 9

-- The theorem to prove
theorem length_AM_is_correct
  (AP PB PE : ℝ)
  (x : ℝ)
  (AM : ℝ) 
  (H_AP_PB  : AP = 7 ∧ PB = 2 ∧ PE = 2)
  (H_QD_QE : x = 63 / 11)
  (H_PQ : PQ = 2 + x) :
  AM = 85 / 22 :=
by
  sorry

end length_AM_is_correct_l63_63464


namespace a_1_value_l63_63949

variable (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ)

axiom a_n_def : ∀ n ≥ 2, a n + 2 * (S n) * (S (n - 1)) = 0
axiom S_5_value : S 5 = 1/11
axiom summation_def : ∀ k ≥ 1, S k = S (k - 1) + a k

theorem a_1_value : a 1 = 1/3 := by
  sorry

end a_1_value_l63_63949


namespace expression_equals_sqrt2_l63_63914

theorem expression_equals_sqrt2 :
  (1 + Real.pi)^0 + 2 - abs (-3) + 2 * Real.sin (Real.pi / 4) = Real.sqrt 2 := by
  sorry

end expression_equals_sqrt2_l63_63914


namespace jerome_bought_last_month_l63_63225

-- Definitions representing the conditions in the problem
def total_toy_cars_now := 40
def original_toy_cars := 25
def bought_this_month (bought_last_month : ℕ) := 2 * bought_last_month

-- The main statement to prove
theorem jerome_bought_last_month : ∃ x : ℕ, original_toy_cars + x + bought_this_month x = total_toy_cars_now ∧ x = 5 :=
by
  sorry

end jerome_bought_last_month_l63_63225


namespace cos2_minus_sin2_pi_over_12_l63_63913

theorem cos2_minus_sin2_pi_over_12 : 
  (Real.cos (Real.pi / 12))^2 - (Real.sin (Real.pi / 12))^2 = Real.cos (Real.pi / 6) := 
by
  sorry

end cos2_minus_sin2_pi_over_12_l63_63913


namespace task_completion_l63_63316

theorem task_completion (x y z : ℝ) 
  (h1 : 1 / x + 1 / y = 1 / 2)
  (h2 : 1 / y + 1 / z = 1 / 4)
  (h3 : 1 / z + 1 / x = 5 / 12) :
  x = 3 := 
sorry

end task_completion_l63_63316


namespace trader_sold_40_meters_of_cloth_l63_63618

theorem trader_sold_40_meters_of_cloth 
  (total_profit_per_meter : ℕ) 
  (total_profit : ℕ) 
  (meters_sold : ℕ) 
  (h1 : total_profit_per_meter = 30) 
  (h2 : total_profit = 1200) 
  (h3 : total_profit = total_profit_per_meter * meters_sold) : 
  meters_sold = 40 := by
  sorry

end trader_sold_40_meters_of_cloth_l63_63618


namespace largest_integer_inequality_l63_63283

theorem largest_integer_inequality (x : ℤ) (h : 10 - 3 * x > 25) : x = -6 :=
sorry

end largest_integer_inequality_l63_63283


namespace min_rho_squared_l63_63753

noncomputable def rho_squared (x t : ℝ) : ℝ :=
  (x - t)^2 + (x^2 - 4 * x + 7 + t)^2

theorem min_rho_squared : 
  ∃ (x t : ℝ), x = 3/2 ∧ t = -7/8 ∧ 
  ∀ (x' t' : ℝ), rho_squared x' t' ≥ rho_squared (3/2) (-7/8) :=
by
  sorry

end min_rho_squared_l63_63753


namespace lcm_is_only_function_l63_63414

noncomputable def f (x y : ℕ) : ℕ := Nat.lcm x y

theorem lcm_is_only_function 
    (f : ℕ → ℕ → ℕ)
    (h1 : ∀ x : ℕ, f x x = x) 
    (h2 : ∀ x y : ℕ, f x y = f y x) 
    (h3 : ∀ x y : ℕ, (x + y) * f x y = y * f x (x + y)) : 
  ∀ x y : ℕ, f x y = Nat.lcm x y := 
sorry

end lcm_is_only_function_l63_63414


namespace license_plate_count_l63_63044

def num_license_plates : Nat :=
  26 * 10 * 36

theorem license_plate_count : num_license_plates = 9360 :=
by
  sorry

end license_plate_count_l63_63044


namespace triangle_sine_cosine_l63_63080

theorem triangle_sine_cosine (a b A : ℝ) (B C : ℝ) (c : ℝ) 
  (ha : a = Real.sqrt 7) 
  (hb : b = 2) 
  (hA : A = 60 * Real.pi / 180) 
  (hsinB : Real.sin B = Real.sin B := by sorry)
  (hc : c = 3 := by sorry) :
  (Real.sin B = Real.sqrt 21 / 7) ∧ (c = 3) := 
sorry

end triangle_sine_cosine_l63_63080


namespace simplify_expression_l63_63223

theorem simplify_expression (x y : ℝ) :
  3 * (x + y) ^ 2 - 7 * (x + y) + 8 * (x + y) ^ 2 + 6 * (x + y) = 
  11 * (x + y) ^ 2 - (x + y) :=
by
  sorry

end simplify_expression_l63_63223


namespace binomial_expansion_l63_63286

theorem binomial_expansion : 
  (102: ℕ)^4 - 4 * (102: ℕ)^3 + 6 * (102: ℕ)^2 - 4 * (102: ℕ) + 1 = (101: ℕ)^4 :=
by sorry

end binomial_expansion_l63_63286


namespace percentage_support_of_surveyed_population_l63_63018

-- Definitions based on the conditions
def men_percentage_support : ℝ := 0.70
def women_percentage_support : ℝ := 0.75
def men_surveyed : ℕ := 200
def women_surveyed : ℕ := 800

-- Proof statement
theorem percentage_support_of_surveyed_population : 
  ((men_percentage_support * men_surveyed + women_percentage_support * women_surveyed) / 
   (men_surveyed + women_surveyed) * 100) = 74 := 
by
  sorry

end percentage_support_of_surveyed_population_l63_63018


namespace bus_stop_minutes_per_hour_l63_63047

/-- Given the average speed of a bus excluding stoppages is 60 km/hr
and including stoppages is 15 km/hr, prove that the bus stops for 45 minutes per hour. -/
theorem bus_stop_minutes_per_hour
  (speed_no_stops : ℝ := 60)
  (speed_with_stops : ℝ := 15) :
  ∃ t : ℝ, t = 45 :=
by
  sorry

end bus_stop_minutes_per_hour_l63_63047


namespace total_height_of_tower_l63_63025

theorem total_height_of_tower :
  let S₃₅ : ℕ := (35 * (35 + 1)) / 2
  let S₆₅ : ℕ := (65 * (65 + 1)) / 2
  S₃₅ + S₆₅ = 2775 :=
by
  let S₃₅ := (35 * (35 + 1)) / 2
  let S₆₅ := (65 * (65 + 1)) / 2
  sorry

end total_height_of_tower_l63_63025


namespace complement_union_l63_63209

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 4}

theorem complement_union : U \ (A ∪ B) = {3, 5} :=
by
  sorry

end complement_union_l63_63209


namespace angle_in_gradians_l63_63681

noncomputable def gradians_in_full_circle : ℝ := 600
noncomputable def degrees_in_full_circle : ℝ := 360
noncomputable def angle_in_degrees : ℝ := 45

theorem angle_in_gradians :
  angle_in_degrees / degrees_in_full_circle * gradians_in_full_circle = 75 := 
by
  sorry

end angle_in_gradians_l63_63681


namespace eugene_pencils_left_l63_63497

-- Define the total number of pencils Eugene initially has
def initial_pencils : ℝ := 234.0

-- Define the number of pencils Eugene gives away
def pencils_given_away : ℝ := 35.0

-- Define the expected number of pencils left
def expected_pencils_left : ℝ := 199.0

-- Prove the number of pencils left after giving away 35.0 equals 199.0
theorem eugene_pencils_left : initial_pencils - pencils_given_away = expected_pencils_left := by
  -- This is where the proof would go, if needed
  sorry

end eugene_pencils_left_l63_63497


namespace postcards_initial_count_l63_63908

theorem postcards_initial_count (P : ℕ) 
  (h1 : ∀ n, n = P / 2)
  (h2 : ∀ n, n = (P / 2) * 15 / 5) 
  (h3 : P / 2 + 3 * P / 2 = 36) : 
  P = 18 := 
sorry

end postcards_initial_count_l63_63908


namespace distance_p_runs_l63_63003

-- Given conditions
def runs_faster (speed_q : ℝ) : ℝ := 1.20 * speed_q
def head_start : ℝ := 50

-- Proof statement
theorem distance_p_runs (speed_q distance_q : ℝ) (h1 : runs_faster speed_q = 1.20 * speed_q)
                         (h2 : head_start = 50)
                         (h3 : (distance_q / speed_q) = ((distance_q + head_start) / (runs_faster speed_q))) :
                         (distance_q + head_start = 300) :=
by
  sorry

end distance_p_runs_l63_63003


namespace ratio_shoes_sandals_simplified_l63_63431

-- Define the given conditions
def shoes_sold : ℕ := 72
def sandals_sold : ℕ := 40

-- Define the GCD of the two given conditions
def gcd_shoes_sandals : ℕ := Nat.gcd shoes_sold sandals_sold

-- Define the simplified ratio of shoes to sandals
def simplified_shoes : ℕ := shoes_sold / gcd_shoes_sandals
def simplified_sandals : ℕ := sandals_sold / gcd_shoes_sandals

-- Prove that the ratio of shoes sold to sandals sold is 9:5
theorem ratio_shoes_sandals_simplified : simplified_shoes = 9 ∧ simplified_sandals = 5 :=
by
  -- We only state the theorem without proof
  sorry

end ratio_shoes_sandals_simplified_l63_63431


namespace evaluate_expression_l63_63837

theorem evaluate_expression (x y : ℕ) (hx : 2^x ∣ 360 ∧ ¬ 2^(x+1) ∣ 360) (hy : 3^y ∣ 360 ∧ ¬ 3^(y+1) ∣ 360) :
  (3 / 7)^(y - x) = 7 / 3 := by
  sorry

end evaluate_expression_l63_63837


namespace polynomial_102_l63_63289

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end polynomial_102_l63_63289


namespace no_such_hexagon_exists_l63_63923

def convex (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ (i j k : ℕ), i ≠ j → j ≠ k → k ≠ i → 
    let (x_i, y_i) := hexagon i;
        (x_j, y_j) := hexagon j;
        (x_k, y_k) := hexagon k in
    ((x_j - x_i) * (y_k - y_i)) - ((y_j - y_i) * (x_k - x_i)) > 0

def side_length_gt_one (hexagon : ℕ → (ℝ × ℝ)) : Prop :=
  ∀ i, let (x1, y1) := hexagon i;
           (x2, y2) := hexagon ((i + 1) % 6) in
           (x2 - x1)^2 + (y2 - y1)^2 > 1

def distance_lt_one (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)) : Prop :=
  ∀ i, let (x, y) := M;
           (x_i, y_i) := hexagon i in
           (x - x_i)^2 + (y - y_i)^2 < 1

theorem no_such_hexagon_exists : 
  ¬ (∃ (hexagon : ℕ → (ℝ × ℝ)) (M : (ℝ × ℝ)),
        convex hexagon ∧
        side_length_gt_one hexagon ∧ 
        distance_lt_one hexagon M) :=
sorry

end no_such_hexagon_exists_l63_63923


namespace probability_region_C_l63_63474

theorem probability_region_C (P_A P_B P_C P_D : ℚ) 
  (h₁ : P_A = 1/4) 
  (h₂ : P_B = 1/3) 
  (h₃ : P_A + P_B + P_C + P_D = 1) : 
  P_C = 5/12 := 
by 
  sorry

end probability_region_C_l63_63474


namespace arithmetic_sequence_general_term_l63_63510

theorem arithmetic_sequence_general_term (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 7)
  (h_a7 : a 7 = 3) :
  ∀ n, a n = -↑n + 10 :=
by
  sorry

end arithmetic_sequence_general_term_l63_63510


namespace revenue_function_correct_strategy_not_profitable_l63_63830

-- Given conditions 
def purchase_price : ℝ := 1
def last_year_price : ℝ := 2
def last_year_sales_volume : ℕ := 10000
def last_year_revenue : ℝ := 20000
def proportionality_constant : ℝ := 4
def increased_sales_volume (x : ℝ) : ℝ := proportionality_constant * (2 - x) ^ 2

-- Questions translated to Lean statements
def revenue_this_year (x : ℝ) : ℝ := 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17

theorem revenue_function_correct (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x = 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17 :=
by
  sorry

theorem strategy_not_profitable (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x ≤ last_year_revenue :=
by
  sorry

end revenue_function_correct_strategy_not_profitable_l63_63830


namespace power_equivalence_l63_63798

theorem power_equivalence (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (x y : ℕ) 
  (hx : 2^m = x) (hy : 2^(2 * n) = y) : 4^(m + 2 * n) = x^2 * y^2 := 
by 
  sorry

end power_equivalence_l63_63798


namespace lesser_fraction_solution_l63_63579

noncomputable def lesser_fraction (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) : ℚ :=
  if x ≤ y then x else y

theorem lesser_fraction_solution (x y : ℚ) (h₁ : x + y = 7/8) (h₂ : x * y = 1/12) :
  lesser_fraction x y h₁ h₂ = (7 - Real.sqrt 17) / 16 := by
  sorry

end lesser_fraction_solution_l63_63579


namespace number_of_steaks_needed_l63_63583

-- Definitions based on the conditions
def family_members : ℕ := 5
def pounds_per_member : ℕ := 1
def ounces_per_pound : ℕ := 16
def ounces_per_steak : ℕ := 20

-- Prove the number of steaks needed equals 4
theorem number_of_steaks_needed : (family_members * pounds_per_member * ounces_per_pound) / ounces_per_steak = 4 := by
  sorry

end number_of_steaks_needed_l63_63583


namespace divides_2pow18_minus_1_l63_63177

theorem divides_2pow18_minus_1 (n : ℕ) : 20 ≤ n ∧ n < 30 ∧ (n ∣ 2^18 - 1) ↔ (n = 19 ∨ n = 27) := by
  sorry

end divides_2pow18_minus_1_l63_63177


namespace reciprocal_fraction_addition_l63_63595

theorem reciprocal_fraction_addition (a b c : ℝ) (h : a ≠ b) :
  (a + c) / (b + c) = b / a ↔ c = - (a + b) := 
by
  sorry

end reciprocal_fraction_addition_l63_63595


namespace max_profit_correctness_l63_63902

noncomputable def daily_purchase_max_profit := 
  let purchase_price := 4.2
  let selling_price := 6
  let return_price := 1.2
  let days_sold_10kg := 10
  let days_sold_6kg := 20
  let days_in_month := 30
  let profit_function (x : ℝ) := 
    10 * x * (selling_price - purchase_price) + 
    days_sold_6kg * 6 * (selling_price - purchase_price) + 
    days_sold_6kg * (x - 6) * (return_price - purchase_price)
  (6, profit_function 6)

theorem max_profit_correctness : daily_purchase_max_profit = (6, 324) :=
  sorry

end max_profit_correctness_l63_63902


namespace three_pow_gt_n_add_two_mul_two_pow_sub_one_l63_63104

theorem three_pow_gt_n_add_two_mul_two_pow_sub_one (n : ℕ) (hn1 : 2 < n) :
  3^n > (n+2) * 2^(n-1) := sorry

end three_pow_gt_n_add_two_mul_two_pow_sub_one_l63_63104


namespace max_product_h_k_l63_63421

theorem max_product_h_k {h k : ℝ → ℝ} (h_bound : ∀ x, -3 ≤ h x ∧ h x ≤ 5) (k_bound : ∀ x, -1 ≤ k x ∧ k x ≤ 4) :
  ∃ x y, h x * k y = 20 :=
by
  sorry

end max_product_h_k_l63_63421


namespace max_three_cell_corners_l63_63900

-- Define the grid size
def grid_height : ℕ := 7
def grid_width : ℕ := 14

-- Define the concept of a three-cell corner removal
def three_cell_corner (region : ℕ) : ℕ := region / 3

-- Define the problem statement in Lean
theorem max_three_cell_corners : three_cell_corner (grid_height * grid_width) = 32 := by
  sorry

end max_three_cell_corners_l63_63900


namespace not_kth_power_l63_63105

theorem not_kth_power (m k : ℕ) (hk : k > 1) : ¬ ∃ a : ℤ, m * (m + 1) = a^k :=
by
  sorry

end not_kth_power_l63_63105


namespace inequality_transformation_l63_63978

variable {a b c d : ℝ}

theorem inequality_transformation
  (h1 : a < b)
  (h2 : b < 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (d / a) < (c / a) :=
by
  sorry

end inequality_transformation_l63_63978


namespace remaining_volume_of_cube_l63_63477

theorem remaining_volume_of_cube (s : ℝ) (r : ℝ) (h : ℝ) (π : ℝ) 
    (cube_volume : s = 5) 
    (cylinder_radius : r = 1.5) 
    (cylinder_height : h = 5) :
    s^3 - π * r^2 * h = 125 - 11.25 * π := by
  sorry

end remaining_volume_of_cube_l63_63477


namespace math_problem_l63_63655

noncomputable def ellipse_standard_equation (a b : ℝ) : Prop :=
  a = 2 ∧ b = Real.sqrt 3 ∧ (∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1)

noncomputable def constant_slope_sum (T R S : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  T = (4, 0) ∧ l (1, 0) ∧ 
  (∀ TR TS : ℝ, (TR = (R.2 / (R.1 - 4)) ∧ TS = (S.2 / (S.1 - 4)) ∧ 
  (TR + TS = 0)))

theorem math_problem 
  {a b : ℝ} {T R S : ℝ × ℝ} {l : ℝ × ℝ → Prop} : 
  ellipse_standard_equation a b ∧ constant_slope_sum T R S l :=
by
  sorry

end math_problem_l63_63655


namespace find_numbers_sum_eq_S_product_eq_P_l63_63311

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l63_63311


namespace problem1_eval_problem2_eval_l63_63917

-- Problem 1
theorem problem1_eval :
  (1 : ℚ) * (-4.5) - (-5.6667) - (2.5) - 7.6667 = -9 := 
by
  sorry

-- Problem 2
theorem problem2_eval :
  (-(4^2) / (-2)^3) - ((4 / 9) * ((-3 / 2)^2)) = 1 := 
by
  sorry

end problem1_eval_problem2_eval_l63_63917


namespace accounting_majors_l63_63679

theorem accounting_majors (p q r s t u : ℕ) 
  (hpqt : (p * q * r * s * t * u = 51030)) 
  (hineq : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) :
  p = 2 :=
sorry

end accounting_majors_l63_63679


namespace preimage_of_8_is_5_image_of_8_is_64_l63_63815

noncomputable def f (x : ℝ) : ℝ := 2^(x - 2)

theorem preimage_of_8_is_5 : ∃ x, f x = 8 := by
  use 5
  sorry

theorem image_of_8_is_64 : f 8 = 64 := by
  sorry

end preimage_of_8_is_5_image_of_8_is_64_l63_63815


namespace egyptian_fraction_l63_63720

theorem egyptian_fraction (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) : 
  (2 : ℚ) / 7 = (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c :=
by
  sorry

end egyptian_fraction_l63_63720


namespace initial_loss_percentage_l63_63904

theorem initial_loss_percentage 
  (CP : ℝ := 250) 
  (SP : ℝ) 
  (h1 : SP + 50 = 1.10 * CP) : 
  (CP - SP) / CP * 100 = 10 := 
sorry

end initial_loss_percentage_l63_63904


namespace jackson_collection_goal_l63_63542

theorem jackson_collection_goal 
  (days_in_week : ℕ)
  (goal : ℕ)
  (earned_mon : ℕ)
  (earned_tue : ℕ)
  (avg_collect_per_4house : ℕ)
  (remaining_days : ℕ)
  (remaining_goal : ℕ)
  (daily_target : ℕ)
  (collect_per_house : ℚ)
  :
  days_in_week = 5 →
  goal = 1000 →
  earned_mon = 300 →
  earned_tue = 40 →
  avg_collect_per_4house = 10 →
  remaining_goal = goal - earned_mon - earned_tue →
  remaining_days = days_in_week - 2 →
  daily_target = remaining_goal / remaining_days →
  collect_per_house = avg_collect_per_4house / 4 →
  (daily_target : ℚ) / collect_per_house = 88 := 
by sorry

end jackson_collection_goal_l63_63542


namespace probability_five_collinear_in_5x5_grid_l63_63987

/-- In a square array of 25 dots (5x5 grid), what is the probability that five randomly chosen dots are collinear? -/
theorem probability_five_collinear_in_5x5_grid :
  (12 : ℚ) / Nat.choose 25 5 = 2 / 8855 := by
  sorry

end probability_five_collinear_in_5x5_grid_l63_63987


namespace g_inv_equals_g_l63_63409

variable {x l : ℝ}

def g (x : ℝ) (l : ℝ) : ℝ := (3 * x + 4) / (l * x - 3)

theorem g_inv_equals_g (l : ℝ) : 
  (∀ x : ℝ, 4 * l + 9 ≠ 0 → (∃ y : ℝ, g y l = x ∧ g x l = y)) ↔ 
  l ≠ -9/4 :=
by sorry

end g_inv_equals_g_l63_63409


namespace area_of_trapezoid_l63_63687

-- Definitions of geometric properties and conditions
def is_perpendicular (a b c : ℝ) : Prop := a + b = 90 -- representing ∠ABC = 90°
def tangent_length (bc ad : ℝ) (O : ℝ) : Prop := bc * ad = O -- representing BC tangent to O with diameter AD
def is_diameter (ad r : ℝ) : Prop := ad = 2 * r -- AD being the diameter of the circle with radius r

-- Given conditions in the problem
variables (AB BC CD AD r O : ℝ) (h1 : is_perpendicular AB BC 90) (h2 : is_perpendicular BC CD 90)
          (h3 : tangent_length BC AD O) (h4 : is_diameter AD r) (h5 : BC = 2 * CD)
          (h6 : AB = 9) (h7 : CD = 3)

-- Statement to prove the area is 36
theorem area_of_trapezoid : (AB + CD) * CD = 36 := by
  sorry

end area_of_trapezoid_l63_63687


namespace inequality_one_solution_l63_63173

theorem inequality_one_solution (a : ℝ) :
  (∀ x : ℝ, |x^2 + 2 * a * x + 4 * a| ≤ 4 → x = -a) ↔ a = 2 :=
by sorry

end inequality_one_solution_l63_63173


namespace smallest_c_in_progressions_l63_63550

def is_arithmetic_progression (a b c : ℤ) : Prop := b - a = c - b

def is_geometric_progression (b c a : ℤ) : Prop := c^2 = a*b

theorem smallest_c_in_progressions :
  ∃ (a b c : ℤ), is_arithmetic_progression a b c ∧ is_geometric_progression b c a ∧ 
  (∀ (a' b' c' : ℤ), is_arithmetic_progression a' b' c' ∧ is_geometric_progression b' c' a' → c ≤ c') ∧ c = 2 :=
by
  sorry

end smallest_c_in_progressions_l63_63550


namespace base9_minus_base6_l63_63937

-- Definitions from conditions
def base9_to_base10 (n : Nat) : Nat :=
  match n with
  | 325 => 3 * 9^2 + 2 * 9^1 + 5 * 9^0
  | _ => 0

def base6_to_base10 (n : Nat) : Nat :=
  match n with
  | 231 => 2 * 6^2 + 3 * 6^1 + 1 * 6^0
  | _ => 0

-- Main theorem statement
theorem base9_minus_base6 : base9_to_base10 325 - base6_to_base10 231 = 175 :=
by
  sorry

end base9_minus_base6_l63_63937


namespace batsman_average_increases_l63_63147

theorem batsman_average_increases
  (score_17th: ℕ)
  (avg_increase: ℕ)
  (initial_avg: ℕ)
  (final_avg: ℕ)
  (initial_innings: ℕ):
  score_17th = 74 →
  avg_increase = 3 →
  initial_innings = 16 →
  initial_avg = 23 →
  final_avg = initial_avg + avg_increase →
  (final_avg * (initial_innings + 1) = score_17th + (initial_avg * initial_innings)) →
  final_avg = 26 :=
by
  sorry

end batsman_average_increases_l63_63147


namespace greatest_integer_solution_l63_63449

theorem greatest_integer_solution (n : ℤ) (h : n^2 - 13 * n + 40 ≤ 0) : n ≤ 8 :=
sorry

end greatest_integer_solution_l63_63449


namespace exists_function_f_l63_63106

theorem exists_function_f :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n * n :=
by
  sorry

end exists_function_f_l63_63106


namespace simplify_expression_l63_63166

theorem simplify_expression (a : ℝ) : a * (a - 3) = a^2 - 3 * a := 
by 
  sorry

end simplify_expression_l63_63166


namespace platform_length_l63_63294

theorem platform_length (train_length : ℝ) (time_pole : ℝ) (time_platform : ℝ) (speed : ℝ) (platform_length : ℝ) :
  train_length = 300 → time_pole = 18 → time_platform = 38 → speed = train_length / time_pole →
  platform_length = (speed * time_platform) - train_length → platform_length = 333.46 :=
by
  introv h1 h2 h3 h4 h5
  sorry

end platform_length_l63_63294


namespace quadratic_has_distinct_real_roots_l63_63820

open Classical

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_distinct_real_roots (k : ℝ) (h_nonzero : k ≠ 0) : 
  (k > -1) ↔ (discriminant k (-2) (-1) > 0) :=
by
  unfold discriminant
  simp
  linarith

end quadratic_has_distinct_real_roots_l63_63820


namespace total_passengers_l63_63467

theorem total_passengers (P : ℕ)
  (h1 : P / 12 + P / 8 + P / 3 + P / 6 + 35 = P) : 
  P = 120 :=
by
  sorry

end total_passengers_l63_63467


namespace find_xyz_l63_63710

theorem find_xyz (x y z : ℝ) (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 8/3) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  xyz = (11 + Real.sqrt 117) / 2 :=
begin
  sorry
end

end find_xyz_l63_63710


namespace find_m_l63_63233

noncomputable def m : ℕ :=
  let S := {d : ℕ | d ∣ 15^8 ∧ d > 0}
  let total_ways := 9^6
  let strictly_increasing_ways := (Nat.choose 9 3) * (Nat.choose 10 3)
  let probability := strictly_increasing_ways / total_ways
  let gcd := Nat.gcd strictly_increasing_ways total_ways
  strictly_increasing_ways / gcd

theorem find_m : m = 112 :=
by
  sorry

end find_m_l63_63233


namespace product_of_five_consecutive_numbers_not_square_l63_63724

theorem product_of_five_consecutive_numbers_not_square (n : ℤ) : 
  ¬ ∃ k : ℤ, k * k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) :=
by
  sorry

end product_of_five_consecutive_numbers_not_square_l63_63724


namespace no_common_root_l63_63646

theorem no_common_root (a b c d : ℝ) (ha : 0 < a) (hb : a < b) (hc : b < c) (hd : c < d) :
  ¬ ∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) :=
by
  sorry

end no_common_root_l63_63646


namespace harrison_croissant_expenditure_l63_63522

-- Define the conditions
def cost_regular_croissant : ℝ := 3.50
def cost_almond_croissant : ℝ := 5.50
def weeks_in_year : ℕ := 52

-- Define the total cost of croissants in a year
def total_cost (cost_regular cost_almond : ℝ) (weeks : ℕ) : ℝ :=
  (weeks * cost_regular) + (weeks * cost_almond)

-- State the proof problem
theorem harrison_croissant_expenditure :
  total_cost cost_regular_croissant cost_almond_croissant weeks_in_year = 468.00 :=
by
  sorry

end harrison_croissant_expenditure_l63_63522


namespace negation_example_l63_63262

open Real

theorem negation_example : 
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n ≥ x^2) ↔ ∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2 := 
  sorry

end negation_example_l63_63262


namespace average_length_is_21_08_l63_63128

def lengths : List ℕ := [20, 21, 22]
def quantities : List ℕ := [23, 64, 32]

def total_length := List.sum (List.zipWith (· * ·) lengths quantities)
def total_quantity := List.sum quantities

def average_length := total_length / total_quantity

theorem average_length_is_21_08 :
  average_length = 2508 / 119 := by
  sorry

end average_length_is_21_08_l63_63128


namespace range_of_m_l63_63528

theorem range_of_m (m : ℝ) : 
    (∀ x y : ℝ, (x^2 / (4 - m) + y^2 / (m - 3) = 1) → 
    4 - m > 0 ∧ m - 3 > 0 ∧ m - 3 > 4 - m) → 
    (7/2 < m ∧ m < 4) :=
sorry

end range_of_m_l63_63528


namespace max_cars_and_quotient_l63_63716

-- Definition of the problem parameters
def car_length : ℕ := 5
def speed_per_car_length : ℕ := 10
def hour_in_seconds : ℕ := 3600
def one_kilometer_in_meters : ℕ := 1000
def distance_in_meters_per_hour (n : ℕ) : ℕ := (10 * n) * one_kilometer_in_meters
def unit_distance (n : ℕ) : ℕ := car_length * (n + 1)

-- Hypotheses
axiom car_spacing : ∀ n : ℕ, unit_distance n = car_length * (n + 1)
axiom car_speed : ∀ n : ℕ, distance_in_meters_per_hour n = (10 * n) * one_kilometer_in_meters

-- Maximum whole number of cars M that can pass in one hour and the quotient when M is divided by 10
theorem max_cars_and_quotient : ∃ (M : ℕ), M = 3000 ∧ M / 10 = 300 := by
  sorry

end max_cars_and_quotient_l63_63716


namespace lower_water_level_by_inches_l63_63279

theorem lower_water_level_by_inches
  (length width : ℝ) (gallons_removed : ℝ) (gallons_to_cubic_feet : ℝ) (feet_to_inches : ℝ) : 
  length = 20 → 
  width = 25 → 
  gallons_removed = 1875 → 
  gallons_to_cubic_feet = 7.48052 → 
  feet_to_inches = 12 → 
  (gallons_removed / gallons_to_cubic_feet) / (length * width) * feet_to_inches = 6.012 := 
by 
  sorry

end lower_water_level_by_inches_l63_63279


namespace problem_equiv_l63_63867

theorem problem_equiv :
  ∃ n : ℕ, 
    10 ≤ n ∧ n < 100 ∧ 
    ((n % 10 = 6 ∧ ¬ (n % 7 = 0)) ∨ (n % 10 ≠ 6 ∧ n % 7 = 0)) ∧
    ((n > 26 ∧ ¬ (n % 10 = 8)) ∨ (n ≤ 26 ∧ n % 10 = 8)) ∧
    ((n % 13 = 0 ∧ ¬ (n < 27)) ∨ (n % 13 ≠ 0 ∧ n < 27)) ∧
    n = 91 :=
begin
  sorry
end

end problem_equiv_l63_63867


namespace find_principal_l63_63903

theorem find_principal (R : ℝ) : ∃ P : ℝ, (P * (R + 5) * 10) / 100 - (P * R * 10) / 100 = 100 :=
by {
  use 200,
  sorry
}

end find_principal_l63_63903


namespace functional_equation_solution_l63_63499

open Nat

theorem functional_equation_solution :
  (∀ (f : ℕ → ℕ), 
    (∀ (x y : ℕ), 0 ≤ y + f x - (Nat.iterate f (f y) x) ∧ (y + f x - (Nat.iterate f (f y) x) ≤ 1)) →
    (∀ n, f n = n + 1)) :=
by
  intro f h
  sorry

end functional_equation_solution_l63_63499


namespace total_cookies_prepared_l63_63776

-- State the conditions as definitions
def num_guests : ℕ := 10
def cookies_per_guest : ℕ := 18

-- The theorem stating the problem
theorem total_cookies_prepared (num_guests cookies_per_guest : ℕ) : 
  num_guests * cookies_per_guest = 180 := 
by 
  -- Here, we would have the proof, but we're using sorry to skip it
  sorry

end total_cookies_prepared_l63_63776


namespace problem_l63_63068

theorem problem (a b : ℝ) (h1 : ∀ x : ℝ, 1 < x ∧ x < 2 → ax^2 - bx + 2 < 0) : a + b = 4 :=
sorry

end problem_l63_63068


namespace numDogsInPetStore_l63_63758

-- Definitions from conditions
variables {D P : Nat}

-- Theorem statement - no proof provided
theorem numDogsInPetStore (h1 : D + P = 15) (h2 : 4 * D + 2 * P = 42) : D = 6 :=
by
  sorry

end numDogsInPetStore_l63_63758


namespace product_of_base8_digits_of_5432_l63_63881

open Nat

def base8_digits (n : ℕ) : List ℕ :=
  let rec digits_helper (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc
    else digits_helper (n / 8) ((n % 8) :: acc)
  digits_helper n []

def product_of_digits (digits : List ℕ) : ℕ :=
  digits.foldl (· * ·) 1

theorem product_of_base8_digits_of_5432 : 
    product_of_digits (base8_digits 5432) = 0 :=
by
  sorry

end product_of_base8_digits_of_5432_l63_63881


namespace expression_value_l63_63385

theorem expression_value
  (x y : ℝ) 
  (h : x - 3 * y = 4) : 
  (x - 3 * y)^2 + 2 * x - 6 * y - 10 = 14 :=
by
  sorry

end expression_value_l63_63385


namespace quotient_when_divided_by_44_is_3_l63_63016

/-
A number, when divided by 44, gives a certain quotient and 0 as remainder.
When dividing the same number by 30, the remainder is 18.
Prove that the quotient in the first division is 3.
-/

theorem quotient_when_divided_by_44_is_3 (N : ℕ) (Q : ℕ) (P : ℕ) 
  (h1 : N % 44 = 0)
  (h2 : N % 30 = 18) :
  N = 44 * Q →
  Q = 3 := 
by
  -- since no proof is required, we use sorry
  sorry

end quotient_when_divided_by_44_is_3_l63_63016


namespace pq_sum_l63_63540

def single_digit (n : ℕ) : Prop := n < 10

theorem pq_sum (P Q : ℕ) (hP : single_digit P) (hQ : single_digit Q)
  (hSum : P * 100 + Q * 10 + Q + P * 110 + Q + Q * 111 = 876) : P + Q = 5 :=
by 
  -- Here we assume the expected outcome based on the problem solution
  sorry

end pq_sum_l63_63540


namespace total_routes_A_to_B_l63_63871

-- Define the conditions
def routes_A_to_C : ℕ := 4
def routes_C_to_B : ℕ := 2

-- Statement to prove
theorem total_routes_A_to_B : (routes_A_to_C * routes_C_to_B = 8) :=
by
  -- Omitting the proof, but stating that there is a total of 8 routes from A to B
  sorry

end total_routes_A_to_B_l63_63871


namespace necessary_but_not_sufficient_condition_l63_63116

open Real

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 ≤ a ∧ a ≤ 4) → (a^2 - 4 * a < 0) := 
by
  sorry

end necessary_but_not_sufficient_condition_l63_63116


namespace factorial_fraction_eq_seven_l63_63037

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end factorial_fraction_eq_seven_l63_63037


namespace elsa_data_remaining_l63_63354

variable (data_total : ℕ) (data_youtube : ℕ)

def data_remaining_after_youtube (data_total data_youtube : ℕ) : ℕ := data_total - data_youtube

def data_fraction_spent_on_facebook (data_left : ℕ) : ℕ := (2 * data_left) / 5

theorem elsa_data_remaining
  (h_data_total : data_total = 500)
  (h_data_youtube : data_youtube = 300) :
  data_remaining_after_youtube data_total data_youtube
  - data_fraction_spent_on_facebook (data_remaining_after_youtube data_total data_youtube) 
  = 120 :=
by
  sorry

end elsa_data_remaining_l63_63354


namespace no_infinite_monochromatic_arithmetic_progression_l63_63779

theorem no_infinite_monochromatic_arithmetic_progression : 
  ∃ (coloring : ℕ → ℕ), (∀ (q r : ℕ), ∃ (n1 n2 : ℕ), coloring (q * n1 + r) ≠ coloring (q * n2 + r)) := sorry

end no_infinite_monochromatic_arithmetic_progression_l63_63779


namespace calc_expression_l63_63915

variable {x : ℝ}

theorem calc_expression :
    (2 + 3 * x) * (-2 + 3 * x) = 9 * x ^ 2 - 4 := sorry

end calc_expression_l63_63915


namespace unrealistic_data_l63_63621

theorem unrealistic_data :
  let A := 1000
  let A1 := 265
  let A2 := 51
  let A3 := 803
  let A1U2 := 287
  let A2U3 := 843
  let A1U3 := 919
  let A1I2 := A1 + A2 - A1U2
  let A2I3 := A2 + A3 - A2U3
  let A3I1 := A3 + A1 - A1U3
  let U := A1 + A2 + A3 - A1I2 - A2I3 - A3I1
  let A1I2I3 := A - U
  A1I2I3 > A2 :=
by
   sorry

end unrealistic_data_l63_63621


namespace calculate_wholesale_price_l63_63480

noncomputable def retail_price : ℝ := 108

noncomputable def selling_price (retail_price : ℝ) : ℝ := retail_price * 0.90

noncomputable def selling_price_alt (wholesale_price : ℝ) : ℝ := wholesale_price * 1.20

theorem calculate_wholesale_price (W : ℝ) (R : ℝ) (SP : ℝ)
  (hR : R = 108)
  (hSP1 : SP = selling_price R)
  (hSP2 : SP = selling_price_alt W) : W = 81 :=
by
  -- Proof omitted
  sorry

end calculate_wholesale_price_l63_63480


namespace sin_neg_pi_l63_63738

theorem sin_neg_pi : Real.sin (-Real.pi) = 0 := by
  sorry

end sin_neg_pi_l63_63738


namespace find_t_l63_63552

noncomputable def g (x : ℝ) (p q s t : ℝ) : ℝ :=
  x^4 + p*x^3 + q*x^2 + s*x + t

theorem find_t {p q s t : ℝ}
  (h1 : ∀ r : ℝ, g r p q s t = 0 → r < 0 ∧ Int.mod (round r) 2 = 1)
  (h2 : p + q + s + t = 2047) :
  t = 5715 :=
sorry

end find_t_l63_63552


namespace consecutive_product_solution_l63_63500

theorem consecutive_product_solution :
  ∀ (n : ℤ), (∃ a : ℤ, n^4 + 8 * n + 11 = a * (a + 1)) ↔ n = 1 :=
by
  sorry

end consecutive_product_solution_l63_63500


namespace find_z_l63_63961

-- Definitions of the conditions
def equation_1 (x y : ℝ) : Prop := x^2 - 3 * x + 6 = y - 10
def equation_2 (y z : ℝ) : Prop := y = 2 * z
def x_value (x : ℝ) : Prop := x = -5

-- Lean theorem statement
theorem find_z (x y z : ℝ) (h1 : equation_1 x y) (h2 : equation_2 y z) (h3 : x_value x) : z = 28 :=
sorry

end find_z_l63_63961


namespace polynomial_expansion_sum_l63_63706

theorem polynomial_expansion_sum (a_6 a_5 a_4 a_3 a_2 a_1 a : ℝ) :
  (∀ x : ℝ, (3 * x - 1)^6 = a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a) →
  a_6 + a_5 + a_4 + a_3 + a_2 + a_1 + a = 64 :=
by
  -- Proof is not needed, placeholder here.
  sorry

end polynomial_expansion_sum_l63_63706


namespace tracy_two_dogs_food_consumption_l63_63877

theorem tracy_two_dogs_food_consumption
  (cups_per_meal : ℝ)
  (meals_per_day : ℝ)
  (pounds_per_cup : ℝ)
  (num_dogs : ℝ) :
  cups_per_meal = 1.5 →
  meals_per_day = 3 →
  pounds_per_cup = 1 / 2.25 →
  num_dogs = 2 →
  num_dogs * (cups_per_meal * meals_per_day) * pounds_per_cup = 4 := by
  sorry

end tracy_two_dogs_food_consumption_l63_63877


namespace no_solution_exists_l63_63602

theorem no_solution_exists (x y : ℝ) : 9^(y + 1) / (1 + 4 / x^2) ≠ 1 :=
by
  sorry

end no_solution_exists_l63_63602


namespace remainder_t100_mod_7_l63_63120

theorem remainder_t100_mod_7 :
  ∀ T : ℕ → ℕ, (T 1 = 3) →
  (∀ n : ℕ, n > 1 → T n = 3 ^ (T (n - 1))) →
  (T 100 % 7 = 6) :=
by
  intro T h1 h2
  -- sorry to skip the actual proof
  sorry

end remainder_t100_mod_7_l63_63120


namespace fifth_friend_contribution_l63_63190

variables (a b c d e : ℕ)

theorem fifth_friend_contribution:
  a + b + c + d + e = 120 ∧
  a = 2 * b ∧
  b = (c + d) / 3 ∧
  c = 2 * e →
  e = 12 :=
sorry

end fifth_friend_contribution_l63_63190


namespace pentagon_area_l63_63349

theorem pentagon_area 
  (edge_length : ℝ) 
  (triangle_height : ℝ) 
  (n_pentagons : ℕ) 
  (equal_convex_pentagons : ℕ) 
  (pentagon_area : ℝ) : 
  edge_length = 5 ∧ triangle_height = 2 ∧ n_pentagons = 5 ∧ equal_convex_pentagons = 5 → pentagon_area = 30 := 
by
  sorry

end pentagon_area_l63_63349


namespace sum_of_digits_least_N_l63_63839

-- Define the function P(N)
def P (N : ℕ) : ℚ := (Nat.ceil (3 * N / 5 + 1) : ℕ) / (N + 1)

-- Define the predicate that checks if P(N) is less than 321/400
def P_lt_321_over_400 (N : ℕ) : Prop := P N < (321 / 400 : ℚ)

-- Define a function that sums the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- The main statement: we claim the least multiple of 5 satisfying the condition
-- That the sum of its digits is 12
theorem sum_of_digits_least_N :
  ∃ N : ℕ, 
    (N % 5 = 0) ∧ 
    P_lt_321_over_400 N ∧ 
    (∀ N' : ℕ, (N' % 5 = 0) → P_lt_321_over_400 N' → N' ≥ N) ∧ 
    sum_of_digits N = 12 := 
sorry

end sum_of_digits_least_N_l63_63839


namespace nine_skiers_four_overtakes_impossible_l63_63418

theorem nine_skiers_four_overtakes_impossible :
  ∀ (skiers : Fin 9 → ℝ),  -- skiers are represented by their speeds
  (∀ i j, i < j → skiers i ≤ skiers j) →  -- skiers start sequentially and maintain constant speeds
  ¬(∀ i, (∃ a b : Fin 9, (a ≠ i ∧ b ≠ i ∧ (skiers a < skiers i ∧ skiers i < skiers b ∨ skiers b < skiers i ∧ skiers i < skiers a)))) →
    false := 
by
  sorry

end nine_skiers_four_overtakes_impossible_l63_63418


namespace instantaneous_velocity_at_t5_l63_63152

noncomputable def s (t : ℝ) : ℝ := 4 * t ^ 2 - 3

theorem instantaneous_velocity_at_t5 : (deriv s 5) = 40 := by
  sorry

end instantaneous_velocity_at_t5_l63_63152


namespace balloons_total_l63_63729

theorem balloons_total (number_of_groups balloons_per_group : ℕ)
  (h1 : number_of_groups = 7) (h2 : balloons_per_group = 5) : 
  number_of_groups * balloons_per_group = 35 := by
  sorry

end balloons_total_l63_63729


namespace find_nonnegative_solutions_l63_63183

theorem find_nonnegative_solutions :
  ∀ (x y z : ℕ), 5^x + 7^y = 2^z ↔ (x = 0 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = 1 ∧ z = 5) :=
by
  sorry

end find_nonnegative_solutions_l63_63183


namespace comb_7_2_equals_21_l63_63795

theorem comb_7_2_equals_21 : (Nat.choose 7 2) = 21 := by
  sorry

end comb_7_2_equals_21_l63_63795


namespace production_company_keeps_60_percent_l63_63897

noncomputable def openingWeekendRevenue : ℝ := 120
noncomputable def productionCost : ℝ := 60
noncomputable def profit : ℝ := 192
noncomputable def totalRevenue : ℝ := 3.5 * openingWeekendRevenue
noncomputable def amountKept : ℝ := profit + productionCost
noncomputable def percentageKept : ℝ := (amountKept / totalRevenue) * 100

theorem production_company_keeps_60_percent :
  percentageKept = 60 :=
by
  sorry

end production_company_keeps_60_percent_l63_63897


namespace shop_length_l63_63573

def monthly_rent : ℝ := 2244
def width : ℝ := 18
def annual_rent_per_sqft : ℝ := 68

theorem shop_length : 
  (monthly_rent * 12 / annual_rent_per_sqft / width) = 22 := 
by
  -- Proof omitted
  sorry

end shop_length_l63_63573


namespace concert_parking_fee_l63_63723

theorem concert_parking_fee :
  let ticket_cost := 50 
  let processing_fee_percentage := 0.15 
  let entrance_fee_per_person := 5 
  let total_cost_concert := 135
  let num_people := 2 

  let total_ticket_cost := ticket_cost * num_people
  let processing_fee := total_ticket_cost * processing_fee_percentage
  let total_ticktet_cost_with_fee := total_ticket_cost + processing_fee
  let total_entrance_fee := entrance_fee_per_person * num_people
  let total_cost_without_parking := total_ticktet_cost_with_fee + total_entrance_fee
  total_cost_concert - total_cost_without_parking = 10 := by 
  sorry

end concert_parking_fee_l63_63723


namespace find_numbers_l63_63312

theorem find_numbers (S P : ℝ) 
  (h_nond : S^2 ≥ 4 * P) :
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2,
      x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2,
      y1 := S - x1,
      y2 := S - x2
  in (x1 + y1 = S ∧ x1 * y1 = P) ∧ (x2 + y2 = S ∧ x2 * y2 = P) :=
by 
  sorry

end find_numbers_l63_63312


namespace molecular_weight_AlPO4_correct_l63_63039

-- Noncomputable because we are working with specific numerical values.
noncomputable def atomic_weight_Al : ℝ := 26.98
noncomputable def atomic_weight_P : ℝ := 30.97
noncomputable def atomic_weight_O : ℝ := 16.00

noncomputable def molecular_weight_AlPO4 : ℝ := 
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

theorem molecular_weight_AlPO4_correct : molecular_weight_AlPO4 = 121.95 := by
  sorry

end molecular_weight_AlPO4_correct_l63_63039


namespace value_of_2022_plus_a_minus_b_l63_63804

theorem value_of_2022_plus_a_minus_b (x a b : ℚ) (h_distinct : x ≠ a ∧ x ≠ b ∧ a ≠ b) 
  (h_gt : a > b) (h_min : ∀ y : ℚ, |y - a| + |y - b| ≥ 2 ∧ |x - a| + |x - b| = 2) :
  2022 + a - b = 2024 := 
by 
  sorry

end value_of_2022_plus_a_minus_b_l63_63804


namespace d_value_l63_63393

theorem d_value (d : ℝ) : (∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15) ↔ (d = 5) := by 
sorry

end d_value_l63_63393


namespace xy_product_l63_63956

noncomputable def f (t : ℝ) : ℝ := Real.sqrt (t^2 + 1) - t + 1

theorem xy_product (x y : ℝ)
  (h : (Real.sqrt (x^2 + 1) - x + 1) * (Real.sqrt (y^2 + 1) - y + 1) = 2) :
  x * y = 1 := by
  sorry

end xy_product_l63_63956


namespace factorial_fraction_eq_seven_l63_63038

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end factorial_fraction_eq_seven_l63_63038


namespace coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l63_63212

theorem coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5 :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (x - 2) ^ 5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 → a_2 = -80 := by
  sorry

end coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5_l63_63212


namespace arcsin_of_neg_one_l63_63169

theorem arcsin_of_neg_one : Real.arcsin (-1) = -Real.pi / 2 :=
by
  sorry

end arcsin_of_neg_one_l63_63169


namespace correct_equation_l63_63439

variables (x : ℝ) (production_planned total_clothings : ℝ)
variables (increase_rate days_ahead : ℝ)

noncomputable def daily_production (x : ℝ) := x
noncomputable def total_production := 1000
noncomputable def production_per_day_due_to_overtime (x : ℝ) := x * (1 + 0.2 : ℝ)
noncomputable def original_completion_days (x : ℝ) := total_production / daily_production x
noncomputable def increased_production_completion_days (x : ℝ) := total_production / production_per_day_due_to_overtime x
noncomputable def days_difference := original_completion_days x - increased_production_completion_days x

theorem correct_equation : days_difference x = 2 := by
  sorry

end correct_equation_l63_63439


namespace measure_angle_PQR_given_conditions_l63_63538

-- Definitions based on conditions
variables {R P Q S : Type} [LinearOrder R] [AddGroup Q] [LinearOrder P] [LinearOrder S]

-- Assume given conditions
def is_straight_line (r s p : ℝ) : Prop := r + p = 2 * s

def is_isosceles_triangle (p s q : ℝ) : Prop := p = q

def angle (q s p : ℝ) := (q - s) - (s - p)

variables (r p q s : ℝ)

-- Define the given angles and equality conditions
def given_conditions : Prop := 
  is_straight_line r s p ∧
  angle q s p = 60 ∧
  is_isosceles_triangle p s q ∧
  r ≠ q 

-- The theorem we want to prove
theorem measure_angle_PQR_given_conditions : given_conditions r p q s → angle p q r = 120 := by
  sorry

end measure_angle_PQR_given_conditions_l63_63538


namespace number_on_board_is_91_l63_63868

-- Definitions based on conditions
def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def ends_in_digit (d n : ℕ) : Prop := n % 10 = d

def divisible_by (m n : ℕ) : Prop := n % m = 0

def andrey_statements (n : ℕ) : Prop :=
  (ends_in_digit 6 n ∨ divisible_by 7 n) ∧ ¬(ends_in_digit 6 n ∧ divisible_by 7 n)

def borya_statements (n : ℕ) : Prop :=
  (n > 26 ∨ ends_in_digit 8 n) ∧ ¬(n > 26 ∧ ends_in_digit 8 n)

def sasha_statements (n : ℕ) : Prop :=
  (divisible_by 13 n ∨ n < 27) ∧ ¬(divisible_by 13 n ∧ n < 27)

-- Mathematical equivalent proof problem
theorem number_on_board_is_91 (n : ℕ) :
  is_two_digit n →
  andrey_statements n →
  borya_statements n →
  sasha_statements n →
  n = 91 :=
by {
  intro _ _ _ _,
  -- Proof goes here, skipped with sorry
  sorry
}

end number_on_board_is_91_l63_63868


namespace sum_smallest_largest_even_integers_l63_63424

theorem sum_smallest_largest_even_integers (m b z : ℕ) (hm_even : m % 2 = 0)
  (h_mean : z = (b + (b + 2 * (m - 1))) / 2) :
  (b + (b + 2 * (m - 1))) = 2 * z :=
by
  sorry

end sum_smallest_largest_even_integers_l63_63424


namespace table_legs_l63_63009

theorem table_legs (total_tables : ℕ) (total_legs : ℕ) (four_legged_tables : ℕ) (four_legged_count : ℕ) 
  (other_legged_tables : ℕ) (other_legged_count : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_legged_tables = 16 →
  four_legged_count = 4 →
  other_legged_tables = total_tables - four_legged_tables →
  total_legs = (four_legged_tables * four_legged_count) + (other_legged_tables * other_legged_count) →
  other_legged_count = 3 := 
by
  sorry

end table_legs_l63_63009


namespace compute_expression_l63_63755

theorem compute_expression : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end compute_expression_l63_63755


namespace positive_difference_is_496_l63_63456

def square (n: ℕ) : ℕ := n * n
def term1 := (square 8 + square 8) / 8
def term2 := (square 8 * square 8) / 8
def positive_difference := abs (term2 - term1)

theorem positive_difference_is_496 : positive_difference = 496 :=
by
  -- This is where the proof would go
  sorry

end positive_difference_is_496_l63_63456


namespace probability_top_card_heart_l63_63760

def specially_designed_deck (n_cards n_ranks n_suits cards_per_suit : ℕ) : Prop :=
  n_cards = 60 ∧ n_ranks = 15 ∧ n_suits = 4 ∧ cards_per_suit = n_ranks

theorem probability_top_card_heart (n_cards n_ranks n_suits cards_per_suit : ℕ)
  (h_deck : specially_designed_deck n_cards n_ranks n_suits cards_per_suit) :
  (15 / 60 : ℝ) = 1 / 4 :=
by
  sorry

end probability_top_card_heart_l63_63760


namespace solve_eq1_solve_eq2_l63_63419

-- Proof for the first equation
theorem solve_eq1 (y : ℝ) : 8 * y - 4 * (3 * y + 2) = 6 ↔ y = -7 / 2 := 
by 
  sorry

-- Proof for the second equation
theorem solve_eq2 (x : ℝ) : 2 - (x + 2) / 3 = x - (x - 1) / 6 ↔ x = 1 := 
by 
  sorry

end solve_eq1_solve_eq2_l63_63419


namespace length_of_first_train_is_270_04_l63_63764

noncomputable def length_of_first_train (speed_first_train_kmph : ℕ) (speed_second_train_kmph : ℕ) 
  (time_seconds : ℕ) (length_second_train_m : ℕ) : ℕ :=
  let combined_speed_mps := ((speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600) 
  let combined_length := combined_speed_mps * time_seconds
  combined_length - length_second_train_m

theorem length_of_first_train_is_270_04 :
  length_of_first_train 120 80 9 230 = 270 :=
by
  sorry

end length_of_first_train_is_270_04_l63_63764


namespace initial_mat_weavers_eq_4_l63_63111

theorem initial_mat_weavers_eq_4 :
  ∃ x : ℕ, (x * 4 = 4) ∧ (14 * 14 = 49) ∧ (x = 4) :=
sorry

end initial_mat_weavers_eq_4_l63_63111


namespace probability_of_b_in_rabbit_l63_63401

theorem probability_of_b_in_rabbit : 
  let word := "rabbit"
  let total_letters := 6
  let num_b_letters := 2
  (num_b_letters : ℚ) / total_letters = 1 / 3 :=
by
  sorry

end probability_of_b_in_rabbit_l63_63401


namespace difference_of_numbers_l63_63142

-- Definitions for the digits and the numbers formed
def digits : List ℕ := [5, 3, 1, 4]

def largestNumber : ℕ := 5431
def leastNumber : ℕ := 1345

-- The problem statement
theorem difference_of_numbers (digits : List ℕ) (n_largest n_least : ℕ) :
  n_largest = 5431 ∧ n_least = 1345 → (n_largest - n_least) = 4086 :=
by
  sorry

end difference_of_numbers_l63_63142


namespace population_of_Beacon_l63_63427

-- Defining the populations of Richmond, Victoria, and Beacon
variables (Richmond Victoria Beacon : ℕ)

-- Given conditions
def condition1 : Prop := Richmond = Victoria + 1000
def condition2 : Prop := Victoria = 4 * Beacon
def condition3  : Prop := Richmond = 3000

-- The theorem to prove the population of Beacon
theorem population_of_Beacon 
  (h1 : condition1) 
  (h2 : condition2) 
  (h3 : condition3) : 
  Beacon = 500 :=
sorry

end population_of_Beacon_l63_63427


namespace largest_by_changing_first_digit_l63_63460

-- Define the original number
def original_number : ℝ := 0.7162534

-- Define the transformation that changes a specific digit to 8
def transform_to_8 (n : ℕ) (d : ℝ) : ℝ :=
  match n with
  | 1 => 0.8162534
  | 2 => 0.7862534
  | 3 => 0.7182534
  | 4 => 0.7168534
  | 5 => 0.7162834
  | 6 => 0.7162584
  | 7 => 0.7162538
  | _ => d

-- State the theorem
theorem largest_by_changing_first_digit :
  ∀ (n : ℕ), transform_to_8 1 original_number ≥ transform_to_8 n original_number :=
by
  sorry

end largest_by_changing_first_digit_l63_63460


namespace factory_output_increase_l63_63430

theorem factory_output_increase (x : ℝ) (h : (1 + x / 100) ^ 4 = 4) : x = 41.4 :=
by
  -- Given (1 + x / 100) ^ 4 = 4
  sorry

end factory_output_increase_l63_63430


namespace xyz_ineq_l63_63253

theorem xyz_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := 
sorry

end xyz_ineq_l63_63253


namespace calculate_probability_two_cards_sum_to_15_l63_63747

-- Define the probability calculation as per the problem statement
noncomputable def probability_two_cards_sum_to_15 : ℚ :=
  let total_cards := 52
  let number_cards := 36 -- 9 values (2 through 10) each with 4 cards
  let card_combinations := (number_cards * (number_cards - 1)) / 2 -- Total pairs to choose from
  let favourable_combinations := 144 -- Manually calculated from cases in the solution
  favourable_combinations / card_combinations

theorem calculate_probability_two_cards_sum_to_15 :
  probability_two_cards_sum_to_15 = 8 / 221 :=
by
  -- Here we ignore the proof steps and directly state it assuming the provided assumption
  admit

end calculate_probability_two_cards_sum_to_15_l63_63747


namespace dogs_food_consumption_l63_63875

def cups_per_meal_per_dog : ℝ := 1.5
def number_of_dogs : ℝ := 2
def meals_per_day : ℝ := 3
def cups_per_pound : ℝ := 2.25

theorem dogs_food_consumption : 
  ((cups_per_meal_per_dog * number_of_dogs) * meals_per_day) / cups_per_pound = 4 := 
by
  sorry

end dogs_food_consumption_l63_63875


namespace discriminant_greater_than_four_l63_63263

theorem discriminant_greater_than_four {p q : ℝ} 
  (h₁ : (999:ℝ)^2 + p * 999 + q < 0) 
  (h₂ : (1001:ℝ)^2 + p * 1001 + q < 0) :
  (p^2 - 4 * q) > 4 :=
sorry

end discriminant_greater_than_four_l63_63263


namespace real_roots_range_l63_63944

theorem real_roots_range (a : ℝ) : (∃ x : ℝ, a*x^2 + 2*x - 1 = 0) ↔ (a >= -1 ∧ a ≠ 0) :=
by 
  sorry

end real_roots_range_l63_63944


namespace quotient_when_m_divided_by_11_is_2_l63_63730

theorem quotient_when_m_divided_by_11_is_2 :
  let n_values := [1, 2, 3, 4, 5]
  let squares := n_values.map (λ n => n^2)
  let remainders := List.eraseDup (squares.map (λ x => x % 11))
  let m := remainders.sum
  m / 11 = 2 :=
by
  sorry

end quotient_when_m_divided_by_11_is_2_l63_63730


namespace find_x_for_slope_l63_63197

theorem find_x_for_slope (x : ℝ) (h : (2 - 5) / (x - (-3)) = -1 / 4) : x = 9 :=
by 
  -- Proof skipped
  sorry

end find_x_for_slope_l63_63197


namespace expected_value_of_biased_coin_l63_63472

noncomputable def expected_value : ℚ :=
  (2 / 3) * 5 + (1 / 3) * -6

theorem expected_value_of_biased_coin :
  expected_value = 4 / 3 := by
  sorry

end expected_value_of_biased_coin_l63_63472


namespace exists_travel_route_l63_63083

theorem exists_travel_route (n : ℕ) (cities : Finset ℕ) 
  (ticket_price : ℕ → ℕ → ℕ)
  (h1 : cities.card = n)
  (h2 : ∀ c1 c2, c1 ≠ c2 → ∃ p, (ticket_price c1 c2 = p ∧ ticket_price c1 c2 = ticket_price c2 c1))
  (h3 : ∀ p1 p2 c1 c2 c3 c4,
    p1 ≠ p2 ∧ (ticket_price c1 c2 = p1) ∧ (ticket_price c3 c4 = p2) →
    p1 ≠ p2) :
  ∃ city : ℕ, ∀ m : ℕ, m = n - 1 →
  ∃ route : Finset (ℕ × ℕ),
  route.card = m ∧
  ∀ (t₁ t₂ : ℕ × ℕ), t₁ ∈ route → t₂ ∈ route → (t₁ ≠ t₂ → ticket_price t₁.1 t₁.2 < ticket_price t₂.1 t₂.2) :=
by
  sorry

end exists_travel_route_l63_63083


namespace students_chose_greek_food_l63_63567
  
theorem students_chose_greek_food (total_students : ℕ) (percentage_greek : ℝ) (h1 : total_students = 200) (h2 : percentage_greek = 0.5) :
  (percentage_greek * total_students : ℝ) = 100 :=
by
  rw [h1, h2]
  norm_num
  sorry

end students_chose_greek_food_l63_63567


namespace rabbit_total_apples_90_l63_63191

-- Define the number of apples each animal places in a basket
def rabbit_apple_per_basket : ℕ := 5
def deer_apple_per_basket : ℕ := 6

-- Define the number of baskets each animal uses
variable (h_r h_d : ℕ)

-- Define the total number of apples collected by both animals
def total_apples : ℕ := rabbit_apple_per_basket * h_r

-- Conditions
axiom deer_basket_count_eq_rabbit : h_d = h_r - 3
axiom same_total_apples : total_apples = deer_apple_per_basket * h_d

-- Goal: Prove that the total number of apples the rabbit collected is 90
theorem rabbit_total_apples_90 : total_apples = 90 := sorry

end rabbit_total_apples_90_l63_63191


namespace dog_speed_correct_l63_63029

-- Definitions of the conditions
def football_field_length_yards : ℕ := 200
def total_football_fields : ℕ := 6
def yards_to_feet_conversion : ℕ := 3
def time_to_fetch_minutes : ℕ := 9

-- The goal is to find the dog's speed in feet per minute
def dog_speed_feet_per_minute : ℕ :=
  (total_football_fields * football_field_length_yards * yards_to_feet_conversion) / time_to_fetch_minutes

-- Statement for the proof
theorem dog_speed_correct : dog_speed_feet_per_minute = 400 := by
  sorry

end dog_speed_correct_l63_63029


namespace locus_of_point_M_l63_63632

theorem locus_of_point_M 
  (O A : EuclideanSpace ℝ (Fin 2)) 
  (r m n : ℝ) 
  (h_r : r > 0) 
  (h_mn : m > 0 ∧ n > 0)
  (P : EuclideanSpace ℝ (Fin 2))
  (hP : dist O P = r) 
  (M : EuclideanSpace ℝ (Fin 2))
  (hM : ∃ k : ℝ, k = m / (m + n) ∧ M = k • P + (1 - k) • A)
  (hA_inside : dist O A < r): 
  ∃ O' : EuclideanSpace ℝ (Fin 2), 
    (∃ l : ℝ, l = n / (m + n) ∧ O' = l • A + (1 - l) • O) ∧
    (∃ R : ℝ, R = (n / (m + n)) * r ∧ ∀ P : EuclideanSpace ℝ (Fin 2), dist P O = r → dist M O' = R) := 
sorry

end locus_of_point_M_l63_63632


namespace geometric_arithmetic_sequence_l63_63194

theorem geometric_arithmetic_sequence (a_n : ℕ → ℕ) (q : ℕ) (a1_eq : a_n 1 = 3)
  (an_geometric : ∀ n, a_n (n + 1) = a_n n * q)
  (arithmetic_condition : 4 * a_n 1 + a_n 3 = 8 * a_n 2) :
  a_n 3 + a_n 4 + a_n 5 = 84 := by
  sorry

end geometric_arithmetic_sequence_l63_63194


namespace Jaron_prize_points_l63_63087

def points_bunnies (bunnies: Nat) (points_per_bunny: Nat) : Nat :=
  bunnies * points_per_bunny

def points_snickers (snickers: Nat) (points_per_snicker: Nat) : Nat :=
  snickers * points_per_snicker

def total_points (bunny_points: Nat) (snicker_points: Nat) : Nat :=
  bunny_points + snicker_points

theorem Jaron_prize_points :
  let bunnies := 8
  let points_per_bunny := 100
  let snickers := 48
  let points_per_snicker := 25
  let bunny_points := points_bunnies bunnies points_per_bunny
  let snicker_points := points_snickers snickers points_per_snicker
  total_points bunny_points snicker_points = 2000 := 
by
  sorry

end Jaron_prize_points_l63_63087


namespace third_motorcyclist_speed_l63_63748

theorem third_motorcyclist_speed 
  (t₁ t₂ : ℝ)
  (x : ℝ)
  (h1 : t₁ - t₂ = 1.25)
  (h2 : 80 * t₁ = x * (t₁ - 0.5))
  (h3 : 60 * t₂ = x * (t₂ - 0.5))
  (h4 : x ≠ 60)
  (h5 : x ≠ 80):
  x = 100 :=
by
  sorry

end third_motorcyclist_speed_l63_63748


namespace expression_value_l63_63188

theorem expression_value : (2^2003 + 5^2004)^2 - (2^2003 - 5^2004)^2 = 40 * 10^2003 := 
by
  sorry

end expression_value_l63_63188


namespace sequence_an_l63_63822

theorem sequence_an (a : ℕ → ℝ) (h0 : a 1 = 1)
  (h1 : ∀ n, 4 * a n * a (n + 1) = (a n + a (n + 1) - 1)^2)
  (h2 : ∀ n > 1, a n > a (n - 1)) :
  ∀ n, a n = n^2 := 
sorry

end sequence_an_l63_63822


namespace atomic_number_R_l63_63858

noncomputable def atomic_number_Pb := 82
def electron_shell_difference := 32

def same_group_atomic_number 
  (atomic_number_Pb : ℕ) 
  (electron_shell_difference : ℕ) : 
  ℕ := 
  atomic_number_Pb + electron_shell_difference

theorem atomic_number_R (R : ℕ) : 
  same_group_atomic_number atomic_number_Pb electron_shell_difference = 114 := 
by
  sorry

end atomic_number_R_l63_63858


namespace find_power_l63_63217

theorem find_power (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^11) (h2 : x = 13) : y = 11 :=
sorry

end find_power_l63_63217


namespace perimeter_of_square_l63_63363

theorem perimeter_of_square (s : ℕ) (h : s = 13) : 4 * s = 52 :=
by {
  sorry
}

end perimeter_of_square_l63_63363


namespace numbers_pairs_sum_prod_l63_63307

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l63_63307


namespace find_a_l63_63208

def setA (a : ℤ) : Set ℤ := {a, 0}

def setB : Set ℤ := {x : ℤ | 3 * x^2 - 10 * x < 0}

theorem find_a (a : ℤ) (h : (setA a ∩ setB).Nonempty) : a = 1 ∨ a = 2 ∨ a = 3 :=
sorry

end find_a_l63_63208


namespace books_bought_l63_63445

theorem books_bought (math_price : ℕ) (hist_price : ℕ) (total_cost : ℕ) (math_books : ℕ) (hist_books : ℕ) 
  (H : math_price = 4) (H1 : hist_price = 5) (H2 : total_cost = 396) (H3 : math_books = 54) 
  (H4 : math_books * math_price + hist_books * hist_price = total_cost) :
  math_books + hist_books = 90 :=
by sorry

end books_bought_l63_63445


namespace h_two_n_mul_h_2024_l63_63749

variable {h : ℕ → ℝ}
variable {k : ℝ}
variable (n : ℕ) (k_ne_zero : k ≠ 0)

-- Condition 1: h(m + n) = h(m) * h(n)
axiom h_add_mul (m n : ℕ) : h (m + n) = h m * h n

-- Condition 2: h(2) = k
axiom h_two : h 2 = k

theorem h_two_n_mul_h_2024 : h (2 * n) * h 2024 = k^(n + 1012) := 
  sorry

end h_two_n_mul_h_2024_l63_63749


namespace smallest_discount_l63_63504

theorem smallest_discount (n : ℕ) (h1 : (1 - 0.12) * (1 - 0.18) = 0.88 * 0.82)
  (h2 : (1 - 0.08) * (1 - 0.08) * (1 - 0.08) = 0.92 * 0.92 * 0.92)
  (h3 : (1 - 0.20) * (1 - 0.10) = 0.80 * 0.90) :
  (29 > 27.84 ∧ 29 > 22.1312 ∧ 29 > 28) :=
by {
  sorry
}

end smallest_discount_l63_63504


namespace ratio_of_B_to_C_l63_63896

theorem ratio_of_B_to_C
  (A B C : ℕ) 
  (h1 : A = B + 2) 
  (h2 : A + B + C = 47) 
  (h3 : B = 18) : B / C = 2 := 
by 
  sorry

end ratio_of_B_to_C_l63_63896


namespace prime_solution_l63_63359

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem prime_solution : ∀ (p q : ℕ), 
  is_prime p → is_prime q → 7 * p * q^2 + p = q^3 + 43 * p^3 + 1 → (p = 2 ∧ q = 7) :=
by
  intros p q hp hq h
  sorry

end prime_solution_l63_63359


namespace students_no_A_l63_63082

def total_students : Nat := 40
def students_A_chemistry : Nat := 10
def students_A_physics : Nat := 18
def students_A_both : Nat := 6

theorem students_no_A : (total_students - (students_A_chemistry + students_A_physics - students_A_both)) = 18 :=
by
  sorry

end students_no_A_l63_63082


namespace determine_b_l63_63637

theorem determine_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Iio 2 ∪ Set.Ioi 6 → -x^2 + b * x - 7 < 0) ∧ 
  (∀ x : ℝ, ¬(x ∈ Set.Iio 2 ∪ Set.Ioi 6) → ¬(-x^2 + b * x - 7 < 0)) → 
  b = 8 :=
sorry

end determine_b_l63_63637


namespace find_b6_l63_63572

def fib (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

theorem find_b6 (b : ℕ → ℕ) (b1 b2 : ℕ)
  (h1 : b 1 = b1) (h2 : b 2 = b2) (h3 : b 5 = 55)
  (hfib : fib b) : b 6 = 84 :=
  sorry

end find_b6_l63_63572


namespace eval_abs_a_plus_b_l63_63797

theorem eval_abs_a_plus_b (a b : ℤ) (x : ℤ) 
(h : (7 * x - a) ^ 2 = 49 * x ^ 2 - b * x + 9) : |a + b| = 45 :=
sorry

end eval_abs_a_plus_b_l63_63797


namespace price_25_bag_l63_63610

noncomputable def price_per_bag_25 : ℝ := 28.97

def price_per_bag_5 : ℝ := 13.85
def price_per_bag_10 : ℝ := 20.42

def total_cost (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ) : ℝ :=
  n5 * p5 + n10 * p10 + n25 * p25

theorem price_25_bag :
  ∃ (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ),
    p5 = price_per_bag_5 ∧
    p10 = price_per_bag_10 ∧
    p25 = price_per_bag_25 ∧
    65 ≤ 5 * n5 + 10 * n10 + 25 * n25 ∧
    5 * n5 + 10 * n10 + 25 * n25 ≤ 80 ∧
    total_cost p5 p10 p25 n5 n10 n25 = 98.77 :=
by
  sorry

end price_25_bag_l63_63610


namespace find_k_intersect_lines_l63_63261

theorem find_k_intersect_lines :
  ∃ (k : ℚ), ∀ (x y : ℚ), 
  (2 * x + 3 * y + 8 = 0) → (x - y - 1 = 0) → (x + k * y = 0) → k = -1/2 :=
by sorry

end find_k_intersect_lines_l63_63261


namespace integer_solution_l63_63812

theorem integer_solution (x : ℤ) (h : (Int.natAbs x - 1) * x ^ 2 - 9 = 1) : x = 2 ∨ x = -2 ∨ x = 3 ∨ x = -3 :=
by
  sorry

end integer_solution_l63_63812


namespace no_arrangement_of_1_to_1978_coprime_l63_63829

theorem no_arrangement_of_1_to_1978_coprime :
  ¬ ∃ (a : Fin 1978 → ℕ), 
    (∀ i : Fin 1977, Nat.gcd (a i) (a (i + 1)) = 1) ∧ 
    (∀ i : Fin 1976, Nat.gcd (a i) (a (i + 2)) = 1) ∧ 
    (∀ i : Fin 1978, 1 ≤ a i ∧ a i ≤ 1978 ∧ ∀ j : Fin 1978, (i ≠ j → a i ≠ a j)) :=
sorry

end no_arrangement_of_1_to_1978_coprime_l63_63829


namespace max_value_x2_plus_2xy_l63_63112

open Real

theorem max_value_x2_plus_2xy (x y : ℝ) (h : x + y = 5) : 
  ∃ (M : ℝ), (M = x^2 + 2 * x * y) ∧ (∀ z w : ℝ, z + w = 5 → z^2 + 2 * z * w ≤ M) :=
by
  sorry

end max_value_x2_plus_2xy_l63_63112


namespace simplify_expression_l63_63356

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3 / 4 := by
  sorry

end simplify_expression_l63_63356


namespace lemons_needed_for_3_dozen_is_9_l63_63833

-- Define the conditions
def lemon_tbs : ℕ := 4
def juice_needed_per_dozen : ℕ := 12
def dozens_needed : ℕ := 3
def total_juice_needed : ℕ := juice_needed_per_dozen * dozens_needed

-- The number of lemons needed to make 3 dozen cupcakes
def lemons_needed (total_juice : ℕ) (lemon_juice : ℕ) : ℕ :=
  total_juice / lemon_juice

-- Prove the number of lemons needed == 9
theorem lemons_needed_for_3_dozen_is_9 : lemons_needed total_juice_needed lemon_tbs = 9 :=
  by sorry

end lemons_needed_for_3_dozen_is_9_l63_63833


namespace intersecting_chords_ratio_l63_63133

theorem intersecting_chords_ratio {XO YO WO ZO : ℝ} 
    (hXO : XO = 5) 
    (hWO : WO = 7) 
    (h_power_of_point : XO * YO = WO * ZO) : 
    ZO / YO = 5 / 7 :=
by
    rw [hXO, hWO] at h_power_of_point
    sorry

end intersecting_chords_ratio_l63_63133


namespace smallest_number_of_marbles_l63_63624

theorem smallest_number_of_marbles 
  (r w b bl n : ℕ) 
  (h : r + w + b + bl = n)
  (h1 : r * (r - 1) * (r - 2) * (r - 3) = 24 * w * b * (r * (r - 1) / 2))
  (h2 : r * (r - 1) * (r - 2) * (r - 3) = 24 * bl * b * (r * (r - 1) / 2))
  (h_no_neg : 4 ≤ r):
  n = 18 :=
sorry

end smallest_number_of_marbles_l63_63624


namespace initial_amount_of_A_l63_63466

theorem initial_amount_of_A (A B : ℕ) (h1 : A / B = 4 / 1)
  (h2 : (A - 24) / (B - 6 + 30) = 2 / 3) : A = 48 := by
  sorry

end initial_amount_of_A_l63_63466


namespace product_of_complex_conjugates_l63_63200

theorem product_of_complex_conjugates (i : ℂ) (h : i^2 = -1) : (1 + i) * (1 - i) = 2 :=
by
  sorry

end product_of_complex_conjugates_l63_63200


namespace single_point_graph_l63_63732

theorem single_point_graph (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 8 * y + d = 0 → x = -1 ∧ y = 4) → d = 19 :=
by
  sorry

end single_point_graph_l63_63732


namespace number_of_valid_sequences_l63_63122

/--
The measures of the interior angles of a convex pentagon form an increasing arithmetic sequence.
Determine the number of such sequences possible if the pentagon is not equiangular, all of the angle
degree measures are positive integers less than 150 degrees, and the smallest angle is at least 60 degrees.
-/

theorem number_of_valid_sequences : ∃ n : ℕ, n = 5 ∧
  ∀ (x d : ℕ),
  x ≥ 60 ∧ x + 4 * d < 150 ∧ 5 * x + 10 * d = 540 ∧ (x + d ≠ x + 2 * d) := 
sorry

end number_of_valid_sequences_l63_63122


namespace minimize_sum_first_n_terms_l63_63509

noncomputable def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

noncomputable def sum_first_n_terms (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n-1) / 2) * d

theorem minimize_sum_first_n_terms (a₁ : ℤ) (a₃_plus_a₅ : ℤ) (n_min : ℕ) :
  a₁ = -9 → a₃_plus_a₅ = -6 → n_min = 5 := by
  sorry

end minimize_sum_first_n_terms_l63_63509


namespace fourth_power_nested_sqrt_l63_63341

noncomputable def nested_sqrt : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt : nested_sqrt ^ 4 = 16 := by
  sorry

end fourth_power_nested_sqrt_l63_63341


namespace DeepakAgeProof_l63_63140

def RahulAgeAfter10Years (RahulAge : ℕ) : Prop := RahulAge + 10 = 26

def DeepakPresentAge (ratioRahul ratioDeepak : ℕ) (RahulAge : ℕ) : ℕ :=
  (2 * RahulAge) / ratioRahul

theorem DeepakAgeProof {DeepakCurrentAge : ℕ}
  (ratioRahul ratioDeepak RahulAge : ℕ)
  (hRatio : ratioRahul = 4)
  (hDeepakRatio : ratioDeepak = 2) :
  RahulAgeAfter10Years RahulAge →
  DeepakCurrentAge = DeepakPresentAge ratioRahul ratioDeepak RahulAge :=
  sorry

end DeepakAgeProof_l63_63140


namespace num_of_factors_l63_63872

theorem num_of_factors (a b c : ℕ) (ha : ∃ p₁ : ℕ, prime p₁ ∧ (a = p₁ ^ 2 ∨ a = p₁ ^ 3))
                               (hb : ∃ p₂ : ℕ, prime p₂ ∧ (b = p₂ ^ 2 ∨ b = p₂ ^ 3))
                               (hc : ∃ p₃ : ℕ, prime p³ ∧ (c = p₃ ^ 2 ∨ c = p₃ ^ 3))
                               (habc_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c) :
  nat.num_divisors (a^2 * b^3 * c^4) = 850 :=
sorry

end num_of_factors_l63_63872


namespace elsa_data_remaining_l63_63355

variable (initial_data : ℕ) (youtube_data : ℕ) (facebook_fraction_num : ℕ) (facebook_fraction_den : ℕ)

def remaining_data (initial_data youtube_data facebook_fraction_num facebook_fraction_den : ℕ) : ℕ :=
  let remaining_after_youtube := initial_data - youtube_data
  let facebook_data := facebook_fraction_num * remaining_after_youtube / facebook_fraction_den
  remaining_after_youtube - facebook_data

theorem elsa_data_remaining : 
  remaining_data 500 300 2 5 = 120 := 
by 
  simp [remaining_data]
  sorry

end elsa_data_remaining_l63_63355


namespace modular_home_total_cost_l63_63496

theorem modular_home_total_cost :
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  total_cost = 249000 := 
by
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  have h : total_cost = 249000 := sorry
  exact h

end modular_home_total_cost_l63_63496


namespace Kayla_total_items_l63_63703

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l63_63703


namespace find_a_b_find_range_of_x_l63_63708

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (Real.log x / Real.log 2)^2 - 2 * a * (Real.log x / Real.log 2) + b

theorem find_a_b (a b : ℝ) :
  (f (1/4) a b = -1) → (a = -2 ∧ b = 3) :=
by
  sorry

theorem find_range_of_x (a b : ℝ) :
  a = -2 → b = 3 →
  ∀ x : ℝ, (f x a b < 0) → (1/8 < x ∧ x < 1/2) :=
by
  sorry

end find_a_b_find_range_of_x_l63_63708


namespace solution_x_y_l63_63920

theorem solution_x_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : 
    x^4 - 6 * x^2 + 1 = 7 * 2^y ↔ (x = 3 ∧ y = 2) :=
by {
    sorry
}

end solution_x_y_l63_63920


namespace same_school_probability_l63_63280

theorem same_school_probability :
  let total_teachers : ℕ := 6
  let teachers_from_school_A : ℕ := 3
  let teachers_from_school_B : ℕ := 3
  let ways_to_choose_2_from_6 : ℕ := Nat.choose total_teachers 2
  let ways_to_choose_2_from_A := Nat.choose teachers_from_school_A 2
  let ways_to_choose_2_from_B := Nat.choose teachers_from_school_B 2
  let same_school_ways : ℕ := ways_to_choose_2_from_A + ways_to_choose_2_from_B
  let probability := (same_school_ways : ℚ) / ways_to_choose_2_from_6 
  probability = (2 : ℚ) / (5 : ℚ) := by sorry

end same_school_probability_l63_63280


namespace four_digit_number_exists_l63_63645

theorem four_digit_number_exists :
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 4 * n = (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000) :=
sorry

end four_digit_number_exists_l63_63645


namespace sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l63_63250

theorem sqrt_12_eq_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := sorry

theorem sqrt_1_div_2_eq_sqrt_2_div_2 : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 := sorry

end sqrt_12_eq_2_sqrt_3_sqrt_1_div_2_eq_sqrt_2_div_2_l63_63250


namespace find_five_digit_number_l63_63182

theorem find_five_digit_number : 
  ∃ (A B C D E : ℕ), 
    (0 < A ∧ A ≤ 9) ∧ 
    (0 < B ∧ B ≤ 9) ∧ 
    (0 < C ∧ C ≤ 9) ∧ 
    (0 < D ∧ D ≤ 9) ∧ 
    (0 < E ∧ E ≤ 9) ∧ 
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E) ∧ 
    (B ≠ C ∧ B ≠ D ∧ B ≠ E) ∧ 
    (C ≠ D ∧ C ≠ E) ∧ 
    (D ≠ E) ∧ 
    (2016 = (10 * D + E) * A * B) ∧ 
    (¬ (10 * D + E) % 3 = 0) ∧ 
    (10^4 * A + 10^3 * B + 10^2 * C + 10 * D + E = 85132) :=
sorry

end find_five_digit_number_l63_63182


namespace triangle_area_l63_63688

theorem triangle_area (A B C : ℝ) (AB AC : ℝ) (A_angle : ℝ) (h1 : A_angle = π / 6)
  (h2 : AB * AC * Real.cos A_angle = Real.tan A_angle) :
  1 / 2 * AB * AC * Real.sin A_angle = 1 / 6 :=
by
  sorry

end triangle_area_l63_63688


namespace lesser_fraction_l63_63271

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l63_63271


namespace find_triplets_l63_63184

theorem find_triplets (x y z : ℝ) :
  (2 * x^3 + 1 = 3 * z * x) ∧ (2 * y^3 + 1 = 3 * x * y) ∧ (2 * z^3 + 1 = 3 * y * z) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1 / 2 ∧ y = -1 / 2 ∧ z = -1 / 2) :=
by 
  intro h
  sorry

end find_triplets_l63_63184


namespace book_exchange_ways_l63_63726

-- Conditions: 6 friends should exchange books without direct trading
-- Problem Statement: Prove that there are 160 ways for this exchange to happen
theorem book_exchange_ways :
  fintype.card (equiv.perm (fin 6)) = 160 :=
by
  sorry

end book_exchange_ways_l63_63726


namespace numbers_in_circle_are_zero_l63_63581

theorem numbers_in_circle_are_zero (a : Fin 55 → ℤ) 
  (h : ∀ i, a i = a ((i + 54) % 55) + a ((i + 1) % 55)) : 
  ∀ i, a i = 0 := 
by
  sorry

end numbers_in_circle_are_zero_l63_63581


namespace total_guppies_correct_l63_63378

noncomputable def total_guppies : ℕ :=
  let haylee := 3 * 12 in
  let jose := haylee / 2 in
  let charliz := jose / 3 in
  let nicolai := charliz * 4 in
  haylee + jose + charliz + nicolai

theorem total_guppies_correct : total_guppies = 84 :=
by 
  -- skip proof
  sorry

end total_guppies_correct_l63_63378


namespace value_of_c_l63_63593

theorem value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 10 < 0) ↔ (x < 2 ∨ x > 8)) → c = 10 :=
by
  sorry

end value_of_c_l63_63593


namespace max_int_solution_of_inequality_system_l63_63121

theorem max_int_solution_of_inequality_system :
  ∃ (x : ℤ), (∀ (y : ℤ), (3 * y - 1 < y + 1) ∧ (2 * (2 * y - 1) ≤ 5 * y + 1) → y ≤ x) ∧
             (3 * x - 1 < x + 1) ∧ (2 * (2 * x - 1) ≤ 5 * x + 1) ∧
             x = 0 :=
by
  sorry

end max_int_solution_of_inequality_system_l63_63121


namespace travel_with_decreasing_ticket_prices_l63_63085

theorem travel_with_decreasing_ticket_prices (n : ℕ) (cities : Finset ℕ) (train_prices : ∀ (i j : ℕ), i ≠ j → ℕ) : 
  cities.card = n ∧
  (∀ i j, i ≠ j → train_prices i j = train_prices j i) ∧
  (∀ i j k l, (i ≠ j ∧ k ≠ l ∧ (i ≠ k ∨ j ≠ l)) → train_prices i j ≠ train_prices k l) →
  ∃ (start : ℕ), ∃ (route : list (ℕ × ℕ)), 
  route.length = n - 1 ∧ 
  (∀ (m : ℕ), m < route.length - 1 → train_prices route.nth m route.nth (m+1) > train_prices route.nth (m+1) route.nth (m+2)) :=
by 
  sorry

end travel_with_decreasing_ticket_prices_l63_63085


namespace probability_of_pink_l63_63691

variable (B P : ℕ) -- number of blue and pink gumballs
variable (h_total : B + P > 0) -- there is at least one gumball in the jar
variable (h_prob_two_blue : (B / (B + P)) * (B / (B + P)) = 16 / 49) -- the probability of drawing two blue gumballs in a row

theorem probability_of_pink : (P / (B + P)) = 3 / 7 :=
sorry

end probability_of_pink_l63_63691


namespace min_value_of_fraction_l63_63672

noncomputable def problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  problem_statement a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_fraction_l63_63672


namespace thyme_pots_count_l63_63489

theorem thyme_pots_count
  (basil_pots : ℕ := 3)
  (rosemary_pots : ℕ := 9)
  (leaves_per_basil_pot : ℕ := 4)
  (leaves_per_rosemary_pot : ℕ := 18)
  (leaves_per_thyme_pot : ℕ := 30)
  (total_leaves : ℕ := 354)
  : (total_leaves - (basil_pots * leaves_per_basil_pot + rosemary_pots * leaves_per_rosemary_pot)) / leaves_per_thyme_pot = 6 :=
by
  sorry

end thyme_pots_count_l63_63489


namespace bill_before_tax_l63_63139

theorem bill_before_tax (T E : ℝ) (h1 : E = 2) (h2 : 3 * T + 5 * E = 12.70) : 2 * T + 3 * E = 7.80 :=
by
  sorry

end bill_before_tax_l63_63139


namespace selling_price_to_achieve_profit_l63_63759

theorem selling_price_to_achieve_profit (num_pencils : ℝ) (cost_per_pencil : ℝ) (desired_profit : ℝ) (selling_price : ℝ) :
  num_pencils = 1800 →
  cost_per_pencil = 0.15 →
  desired_profit = 100 →
  selling_price = 0.21 :=
by
  sorry

end selling_price_to_achieve_profit_l63_63759


namespace cistern_problem_l63_63476

noncomputable def cistern_problem_statement : Prop :=
∀ (x : ℝ),
  (1 / 5 - 1 / x = 1 / 11.25) → x = 9

theorem cistern_problem : cistern_problem_statement :=
sorry

end cistern_problem_l63_63476


namespace sam_seashell_count_l63_63463

/-!
# Problem statement:
-/
def initialSeashells := 35
def seashellsGivenToJoan := 18
def seashellsFoundToday := 20
def seashellsGivenToTom := 5

/-!
# Proof goal: Prove that the current number of seashells Sam has is 32.
-/
theorem sam_seashell_count :
  initialSeashells - seashellsGivenToJoan + seashellsFoundToday - seashellsGivenToTom = 32 :=
  sorry

end sam_seashell_count_l63_63463


namespace probability_circle_l63_63848

theorem probability_circle (total_figures triangles circles squares : ℕ)
  (h_total : total_figures = 10)
  (h_triangles : triangles = 4)
  (h_circles : circles = 3)
  (h_squares : squares = 3) :
  circles / total_figures = 3 / 10 :=
by
  sorry

end probability_circle_l63_63848


namespace John_profit_is_1500_l63_63832

-- Defining the conditions
def P_initial : ℕ := 8
def Puppies_given_away : ℕ := P_initial / 2
def Puppies_kept : ℕ := 1
def Price_per_puppy : ℕ := 600
def Payment_stud_owner : ℕ := 300

-- Define the number of puppies John's selling
def Puppies_selling := Puppies_given_away - Puppies_kept

-- Define the total revenue from selling the puppies
def Total_revenue := Puppies_selling * Price_per_puppy

-- Define John’s profit 
def John_profit := Total_revenue - Payment_stud_owner

-- The statement to prove
theorem John_profit_is_1500 : John_profit = 1500 := by
  sorry

end John_profit_is_1500_l63_63832


namespace clock_angle_solution_l63_63339

theorem clock_angle_solution (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 360) :
    (θ = 15) ∨ (θ = 165) :=
by
  sorry

end clock_angle_solution_l63_63339


namespace angela_action_figures_left_l63_63486

theorem angela_action_figures_left :
  ∀ (initial_collection : ℕ), 
  initial_collection = 24 → 
  (let sold := initial_collection / 4 in
   let remaining_after_sold := initial_collection - sold in
   let given_to_daughter := remaining_after_sold / 3 in
   let remaining_after_given := remaining_after_sold - given_to_daughter in
   remaining_after_given = 12) :=
by
  intros
  sorry

end angela_action_figures_left_l63_63486


namespace hcf_of_numbers_is_five_l63_63134

theorem hcf_of_numbers_is_five (a b x : ℕ) (ratio : a = 3 * x) (ratio_b : b = 4 * x)
  (lcm_ab : Nat.lcm a b = 60) (hcf_ab : Nat.gcd a b = 5) : Nat.gcd a b = 5 :=
by
  sorry

end hcf_of_numbers_is_five_l63_63134


namespace equation_of_perpendicular_line_l63_63012

theorem equation_of_perpendicular_line (c : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0 ∧ 2 * x + y - 5 = 0) → (x - 2 * y - 3 = 0) := 
by
  sorry

end equation_of_perpendicular_line_l63_63012


namespace two_digit_square_difference_l63_63175

-- Define the problem in Lean
theorem two_digit_square_difference :
  ∃ (X Y : ℕ), (10 ≤ X ∧ X ≤ 99) ∧ (10 ≤ Y ∧ Y ≤ 99) ∧ (X > Y) ∧
  (∃ (t : ℕ), (1 ≤ t ∧ t ≤ 9) ∧ (X^2 - Y^2 = 100 * t)) :=
sorry

end two_digit_square_difference_l63_63175


namespace sara_height_l63_63558

def Julie := 33
def Mark := Julie + 1
def Roy := Mark + 2
def Joe := Roy + 3
def Sara := Joe + 6

theorem sara_height : Sara = 45 := by
  sorry

end sara_height_l63_63558


namespace number_of_mappings_l63_63660

open Finset

-- Definitions of the sets M and N
def M : Finset (Fin 3) := {0, 1, 2}
def N : Finset (Fin 2) := {0, 1}

-- The number of different mappings from set M to set N
theorem number_of_mappings : (M.card * N.card) = 8 := by
  sorry

end number_of_mappings_l63_63660


namespace stacy_days_to_complete_paper_l63_63731

variable (total_pages pages_per_day : ℕ)
variable (d : ℕ)

theorem stacy_days_to_complete_paper 
  (h1 : total_pages = 63) 
  (h2 : pages_per_day = 21) 
  (h3 : total_pages = pages_per_day * d) : 
  d = 3 := 
sorry

end stacy_days_to_complete_paper_l63_63731


namespace correct_total_count_l63_63984

variable (x : ℕ)

-- Define the miscalculation values
def value_of_quarter := 25
def value_of_dime := 10
def value_of_half_dollar := 50
def value_of_nickel := 5

-- Calculate the individual overestimations and underestimations
def overestimation_from_quarters := (value_of_quarter - value_of_dime) * (2 * x)
def underestimation_from_half_dollars := (value_of_half_dollar - value_of_nickel) * x

-- Calculate the net correction needed
def net_correction := overestimation_from_quarters - underestimation_from_half_dollars

theorem correct_total_count :
  net_correction x = 15 * x :=
by
  sorry

end correct_total_count_l63_63984


namespace product_of_two_numbers_l63_63865

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 40) (h₂ : x - y = 16) : x * y = 336 :=
sorry

end product_of_two_numbers_l63_63865


namespace winning_candidate_votes_l63_63740

-- Define the conditions as hypotheses in Lean.
def two_candidates (candidates : ℕ) : Prop := candidates = 2
def winner_received_62_percent (V : ℝ) (votes_winner : ℝ) : Prop := votes_winner = 0.62 * V
def winning_margin (V : ℝ) : Prop := 0.24 * V = 384

-- The main theorem to prove: the winner candidate received 992 votes.
theorem winning_candidate_votes (V votes_winner : ℝ) (candidates : ℕ) 
  (h1 : two_candidates candidates) 
  (h2 : winner_received_62_percent V votes_winner)
  (h3 : winning_margin V) : 
  votes_winner = 992 :=
by
  sorry

end winning_candidate_votes_l63_63740


namespace three_sleep_simultaneously_l63_63004

noncomputable def professors := Finset.range 5

def sleeping_times (p: professors) : Finset ℕ 
-- definition to be filled in, stating that p falls asleep twice.
:= sorry 

def moment_two_asleep (p q: professors) : ℕ 
-- definition to be filled in, stating that p and q are asleep together once.
:= sorry

theorem three_sleep_simultaneously :
  ∃ t : ℕ, ∃ p1 p2 p3 : professors, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
  (t ∈ sleeping_times p1) ∧
  (t ∈ sleeping_times p2) ∧
  (t ∈ sleeping_times p3) := by
  sorry

end three_sleep_simultaneously_l63_63004


namespace unique_cd_exists_l63_63556

open Real

theorem unique_cd_exists (h₀ : 0 < π / 2):
  ∃! (c d : ℝ), (0 < c) ∧ (c < π / 2) ∧ (0 < d) ∧ (d < π / 2) ∧ (c < d) ∧ 
  (sin (cos c) = c) ∧ (cos (sin d) = d) := sorry

end unique_cd_exists_l63_63556


namespace problem_1_problem_2_l63_63809

-- Condition for Question 1
def f (x : ℝ) (a : ℝ) := |x - a|

-- Proof Problem for Question 1
theorem problem_1 (a : ℝ) (h : a = 1) : {x : ℝ | f x a > 1/2 * (x + 1)} = {x | x > 3 ∨ x < 1/3} :=
sorry

-- Condition for Question 2
def g (x : ℝ) (a : ℝ) := |x - a| + |x - 2|

-- Proof Problem for Question 2
theorem problem_2 (a : ℝ) : (∃ x : ℝ, g x a ≤ 3) → (-1 ≤ a ∧ a ≤ 5) :=
sorry

end problem_1_problem_2_l63_63809


namespace fraction_of_males_on_time_l63_63625

theorem fraction_of_males_on_time (A : ℕ) :
  (2 / 9 : ℚ) * A = (2 / 9 : ℚ) * A → 
  (2 / 3 : ℚ) * A = (2 / 3 : ℚ) * A → 
  (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) = (5 / 6 : ℚ) * ((1 / 3 : ℚ) * A) → 
  ((7 / 9 : ℚ) * A - (5 / 18 : ℚ) * A) / ((2 / 3 : ℚ) * A) = (1 / 2 : ℚ) :=
by
  intros h1 h2 h3
  sorry

end fraction_of_males_on_time_l63_63625


namespace product_of_roots_cubic_eq_l63_63781

-- Define a cubic equation 2x^3 - 3x^2 - 8x + 12 = 0
def cubic_eq : Polynomial ℝ := Polynomial.mk [2, -3, -8, 12]

-- Define a function to compute the product of the roots of a cubic equation using Vieta's formulas
def product_of_roots (p : Polynomial ℝ) : ℝ :=
  let a := p.coeff 3
  let d := p.coeff 0
  -d / a

-- The proof statement: Prove that the product of the roots of this specific cubic equation is -6
theorem product_of_roots_cubic_eq : product_of_roots cubic_eq = -6 := 
  sorry

end product_of_roots_cubic_eq_l63_63781


namespace mom_twice_alex_l63_63101

-- Definitions based on the conditions
def alex_age_in_2010 : ℕ := 10
def mom_age_in_2010 : ℕ := 5 * alex_age_in_2010
def future_years_after_2010 (x : ℕ) : ℕ := 2010 + x

-- Defining the ages in the future year
def alex_age_future (x : ℕ) : ℕ := alex_age_in_2010 + x
def mom_age_future (x : ℕ) : ℕ := mom_age_in_2010 + x

-- The theorem to prove
theorem mom_twice_alex (x : ℕ) (h : mom_age_future x = 2 * alex_age_future x) : future_years_after_2010 x = 2040 :=
  by
  sorry

end mom_twice_alex_l63_63101


namespace alice_next_birthday_age_l63_63622

theorem alice_next_birthday_age (a b c : ℝ) 
  (h1 : a = 1.25 * b)
  (h2 : b = 0.7 * c)
  (h3 : a + b + c = 30) : a + 1 = 11 :=
by {
  sorry
}

end alice_next_birthday_age_l63_63622


namespace range_of_a_l63_63520

def condition1 (a : ℝ) : Prop := (2 - a) ^ 2 < 1
def condition2 (a : ℝ) : Prop := (3 - a) ^ 2 ≥ 1

theorem range_of_a (a : ℝ) (h1 : condition1 a) (h2 : condition2 a) :
  1 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l63_63520


namespace prove_x_minus_y_squared_l63_63972

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l63_63972


namespace problem_part1_problem_part2_l63_63648

theorem problem_part1
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (x y : ℤ)
  (hA : A x y = 2 * x ^ 2 + 4 * x * y - 2 * x - 3)
  (hB : B x y = -x^2 + x*y + 2) :
  3 * A x y - 2 * (A x y + 2 * B x y) = 6 * x ^ 2 - 2 * x - 11 := by
  sorry

theorem problem_part2
  (A : ℤ → ℤ → ℤ)
  (B : ℤ → ℤ → ℤ)
  (y : ℤ)
  (H : ∀ x, B x y + (1 / 2) * A x y = C) :
  y = 1 / 3 := by
  sorry

end problem_part1_problem_part2_l63_63648


namespace square_area_l63_63623

theorem square_area (x : ℚ) (side_length : ℚ) 
  (h1 : side_length = 3 * x - 12) 
  (h2 : side_length = 24 - 2 * x) : 
  side_length ^ 2 = 92.16 := 
by 
  sorry

end square_area_l63_63623


namespace triangular_pyramid_volume_l63_63619

theorem triangular_pyramid_volume (a b c : ℝ)
  (h1 : 1/2 * a * b = 1.5)
  (h2 : 1/2 * b * c = 2)
  (h3 : 1/2 * a * c = 6) :
  (1/6 * a * b * c = 2) :=
by {
  -- Here, we would provide the proof steps, but for now we leave it as sorry
  sorry
}

end triangular_pyramid_volume_l63_63619


namespace arithmetic_sequence_common_difference_l63_63514

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith_seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h_a30 : a 30 = 100)
  (h_a100 : a 100 = 30) :
  d = -1 := sorry

end arithmetic_sequence_common_difference_l63_63514


namespace tan_435_eq_2_plus_sqrt3_l63_63343

open Real

theorem tan_435_eq_2_plus_sqrt3 : tan (435 * (π / 180)) = 2 + sqrt 3 :=
  sorry

end tan_435_eq_2_plus_sqrt3_l63_63343


namespace spending_spring_months_l63_63258

theorem spending_spring_months (spend_end_March spend_end_June : ℝ)
  (h1 : spend_end_March = 1) (h2 : spend_end_June = 4) :
  (spend_end_June - spend_end_March) = 3 :=
by
  rw [h1, h2]
  norm_num

end spending_spring_months_l63_63258


namespace inequality_proof_l63_63842

open Real

-- Given conditions
variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1)

-- Goal to prove
theorem inequality_proof : 
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
sorry

end inequality_proof_l63_63842


namespace greatest_number_of_kits_l63_63444

-- Given conditions
def bottles_of_water := 20
def cans_of_food := 12
def flashlights := 30
def blankets := 18

def no_more_than_10_items_per_kit (kits : ℕ) := 
  (bottles_of_water / kits ≤ 10) ∧ 
  (cans_of_food / kits ≤ 10) ∧ 
  (flashlights / kits ≤ 10) ∧ 
  (blankets / kits ≤ 10)

def greater_than_or_equal_to_5_kits (kits : ℕ) := kits ≥ 5

def all_items_distributed_equally (kits : ℕ) := 
  (bottles_of_water % kits = 0) ∧ 
  (cans_of_food % kits = 0) ∧ 
  (flashlights % kits = 0) ∧ 
  (blankets % kits = 0)

-- Proof goal
theorem greatest_number_of_kits : 
  ∃ kits : ℕ, 
    no_more_than_10_items_per_kit kits ∧ 
    greater_than_or_equal_to_5_kits kits ∧ 
    all_items_distributed_equally kits ∧ 
    kits = 6 := 
sorry

end greatest_number_of_kits_l63_63444


namespace geometric_sequence_general_term_geometric_sequence_sum_n_l63_63947

theorem geometric_sequence_general_term (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) : 
  a n = 48 * (1 / 2) ^ (n - 1) := 
by {
  sorry
}

theorem geometric_sequence_sum_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 3 = 12) (h2 : a 8 = 3 / 8) 
  (h3 : ∀ (n : ℕ), a n = 48 * (1 / 2) ^ (n - 1)) 
  (h4 : ∀ (n : ℕ), S n = 48 * (1 - (1 / 2) ^ n) / (1 - 1 / 2))
  (h5 : S 5 = 93) : 
  5 = 5 := 
by {
  sorry
}

end geometric_sequence_general_term_geometric_sequence_sum_n_l63_63947


namespace largest_of_four_consecutive_primes_l63_63754

noncomputable def sum_of_primes_is_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ Prime (p1 + p2 + p3 + p4)

theorem largest_of_four_consecutive_primes :
  ∃ p1 p2 p3 p4, 
  sum_of_primes_is_prime p1 p2 p3 p4 ∧ 
  p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ 
  (p1, p2, p3, p4) = (2, 3, 5, 7) ∧ 
  max p1 (max p2 (max p3 p4)) = 7 :=
by {
  sorry                                 -- solve this in Lean
}

end largest_of_four_consecutive_primes_l63_63754


namespace quadratic_root_range_l63_63817

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end quadratic_root_range_l63_63817


namespace area_of_given_triangle_l63_63644

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
1 / 2 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

def vertex_A : ℝ × ℝ := (-1, 4)
def vertex_B : ℝ × ℝ := (7, 0)
def vertex_C : ℝ × ℝ := (11, 5)

theorem area_of_given_triangle : area_of_triangle vertex_A vertex_B vertex_C = 28 :=
by
  show 1 / 2 * |(-1) * (0 - 5) + 7 * (5 - 4) + 11 * (4 - 0)| = 28
  sorry

end area_of_given_triangle_l63_63644


namespace find_unknown_number_l63_63940

theorem find_unknown_number
  (n : ℕ)
  (h_lcm : Nat.lcm n 1491 = 5964) :
  n = 4 :=
sorry

end find_unknown_number_l63_63940


namespace find_prob_l63_63954

-- Define the normal distribution with mean μ and variance σ²
variables {μ σ : ℝ}

def normal_prob_cond (ξ : ℝ → ℝ) : Prop :=
  (ξ ∼ Normal μ σ) ∧
  ((ξ > 4) = (ξ < 2)) ∧
  (P(ξ ≤ 0) = 0.2)

theorem find_prob (ξ : ℝ → ℝ) (h : normal_prob_cond ξ) :
  P(0 < ξ < 6) = 0.6 :=
by 
  sorry

end find_prob_l63_63954


namespace selection_of_village_assistants_l63_63647

theorem selection_of_village_assistants (A B C : ℕ) (n : ℕ) (total_candidates : ℕ) (r : ℕ) :
  (total_candidates = 10) → (r = 3) → (C ≠ total_candidates) → 
  (A ≠ total_candidates) → (B ≠ total_candidates) → 
  ∑ i in (Finset.Ico 1 (total_candidates - 2)), 
    choose (total_candidates - 1) r - choose (total_candidates - 3) r = 49 := by
  intro h1 h2 h3 h4 h5
  sorry

end selection_of_village_assistants_l63_63647


namespace abc_sum_71_l63_63415

theorem abc_sum_71 (a b c : ℝ) (h₁ : ∀ x, (x ≤ -3 ∨ 23 ≤ x ∧ x < 27) ↔ ( (x - a) * (x - b) / (x - c) ≥ 0)) (h₂ : a < b) : 
  a + 2 * b + 3 * c = 71 :=
sorry

end abc_sum_71_l63_63415


namespace expense_recording_l63_63671

-- Define the recording of income and expenses
def record_income (amount : Int) : Int := amount
def record_expense (amount : Int) : Int := -amount

-- Given conditions
def income_example := record_income 500
def expense_example := record_expense 400

-- Prove that an expense of 400 yuan is recorded as -400 yuan
theorem expense_recording : record_expense 400 = -400 :=
  by sorry

end expense_recording_l63_63671


namespace prove_x_minus_y_squared_l63_63971

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l63_63971


namespace least_multiple_of_15_greater_than_520_l63_63450

theorem least_multiple_of_15_greater_than_520 : ∃ n : ℕ, n > 520 ∧ n % 15 = 0 ∧ (∀ m : ℕ, m > 520 ∧ m % 15 = 0 → n ≤ m) ∧ n = 525 := 
by
  sorry

end least_multiple_of_15_greater_than_520_l63_63450


namespace prob_john_meets_train_l63_63546

noncomputable def john_arrival_time : MeasureTheory.Measure ℝ := MeasureTheory.Measure.unif 90 -- John arrives uniformly between 1:30 (0 minutes) and 3:00 (90 minutes)
noncomputable def train_arrival_time : MeasureTheory.Measure ℝ := MeasureTheory.Measure.unif 60 -- Train arrives uniformly between 2:00 (0 minutes) and 3:00 (60 minutes)
noncomputable def prob_train_at_station_when_john_arrives : ℝ := 4/27

theorem prob_john_meets_train :
  MeasureTheory.Probability (set.prod {j | j > -0.5} {t | t < j + 20}) = 4/27 := sorry

end prob_john_meets_train_l63_63546


namespace cannot_determine_students_answered_both_correctly_l63_63239

-- Definitions based on the given conditions
def students_enrolled : ℕ := 25
def students_answered_q1_correctly : ℕ := 22
def students_not_taken_test : ℕ := 3
def some_students_answered_q2_correctly : Prop := -- definition stating that there's an undefined number of students that answered question 2 correctly
  ∃ n : ℕ, (n ≤ students_enrolled) ∧ n > 0

-- Statement for the proof problem
theorem cannot_determine_students_answered_both_correctly :
  ∃ n, (n ≤ students_answered_q1_correctly) ∧ n > 0 → false :=
by sorry

end cannot_determine_students_answered_both_correctly_l63_63239


namespace last_digit_322_pow_369_l63_63789

theorem last_digit_322_pow_369 : (322^369) % 10 = 2 := by
  sorry

end last_digit_322_pow_369_l63_63789


namespace avg_visitors_sundays_l63_63611

-- Definitions
def days_in_month := 30
def avg_visitors_per_day_month := 750
def avg_visitors_other_days := 700
def sundays_in_month := 5
def other_days := days_in_month - sundays_in_month

-- Main statement to prove
theorem avg_visitors_sundays (S : ℕ) 
  (H1 : days_in_month = 30) 
  (H2 : avg_visitors_per_day_month = 750) 
  (H3 : avg_visitors_other_days = 700) 
  (H4 : sundays_in_month = 5) 
  (H5 : other_days = days_in_month - sundays_in_month) 
  :
  (sundays_in_month * S + other_days * avg_visitors_other_days) = avg_visitors_per_day_month * days_in_month 
  → S = 1000 :=
by 
  sorry

end avg_visitors_sundays_l63_63611


namespace sum_of_edges_l63_63019

theorem sum_of_edges (a b c : ℝ)
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b ^ 2 = a * c) :
  4 * (a + b + c) = 32 := 
sorry

end sum_of_edges_l63_63019


namespace total_hamburger_varieties_l63_63966

def num_condiments : ℕ := 9
def num_condiment_combinations : ℕ := 2 ^ num_condiments
def num_patties_choices : ℕ := 4
def num_bread_choices : ℕ := 2

theorem total_hamburger_varieties : num_condiment_combinations * num_patties_choices * num_bread_choices = 4096 :=
by
  -- conditions
  have h1 : num_condiments = 9 := rfl
  have h2 : num_condiment_combinations = 2 ^ num_condiments := rfl
  have h3 : num_patties_choices = 4 := rfl
  have h4 : num_bread_choices = 2 := rfl

  -- correct answer
  sorry

end total_hamburger_varieties_l63_63966


namespace train_length_150_m_l63_63763

def speed_in_m_s (speed_kmh : ℕ) : ℕ :=
  speed_kmh * 1000 / 3600

def length_of_train (speed_in_m_s : ℕ) (time_s : ℕ) : ℕ :=
  speed_in_m_s * time_s

theorem train_length_150_m (speed_kmh : ℕ) (time_s : ℕ) (speed_m_s : speed_in_m_s speed_kmh = 15) (time_pass_pole : time_s = 10) : length_of_train (speed_in_m_s speed_kmh) time_s = 150 := by
  sorry

end train_length_150_m_l63_63763


namespace complement_intersection_eq_l63_63661

open Set

def P : Set ℝ := { x | x^2 - 2 * x ≥ 0 }
def Q : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

theorem complement_intersection_eq :
  (compl P) ∩ Q = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end complement_intersection_eq_l63_63661


namespace lesser_fraction_l63_63277

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l63_63277


namespace dots_not_visible_on_3_dice_l63_63436

theorem dots_not_visible_on_3_dice :
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  hidden_dots = 35 := 
by 
  let total_dots := 18 * 21 / 6
  let visible_dots := 1 + 2 + 2 + 3 + 5 + 4 + 5 + 6
  let hidden_dots := total_dots - visible_dots
  show total_dots - visible_dots = 35
  sorry

end dots_not_visible_on_3_dice_l63_63436


namespace TileD_in_AreaZ_l63_63438

namespace Tiles

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def TileA : Tile := {top := 5, right := 3, bottom := 2, left := 4}
def TileB : Tile := {top := 2, right := 4, bottom := 5, left := 3}
def TileC : Tile := {top := 3, right := 6, bottom := 1, left := 5}
def TileD : Tile := {top := 5, right := 2, bottom := 3, left := 6}

variables (X Y Z W : Tile)
variable (tiles : List Tile := [TileA, TileB, TileC, TileD])

noncomputable def areaZContains : Tile := sorry

theorem TileD_in_AreaZ  : areaZContains = TileD := sorry

end Tiles

end TileD_in_AreaZ_l63_63438


namespace frustum_volume_correct_l63_63014

noncomputable def volume_of_frustum 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) : ℝ :=
  let base_area_original := base_edge_original_pyramid ^ 2
  let volume_original := 1 / 3 * base_area_original * height_original_pyramid
  let similarity_ratio := base_edge_smaller_pyramid / base_edge_original_pyramid
  let volume_smaller := volume_original * (similarity_ratio ^ 3)
  volume_original - volume_smaller

theorem frustum_volume_correct 
(base_edge_original_pyramid : ℝ) (height_original_pyramid : ℝ) 
(base_edge_smaller_pyramid : ℝ) (height_smaller_pyramid : ℝ) 
(h_orig_base_edge : base_edge_original_pyramid = 16) 
(h_orig_height : height_original_pyramid = 10) 
(h_smaller_base_edge : base_edge_smaller_pyramid = 8) 
(h_smaller_height : height_smaller_pyramid = 5) : 
  volume_of_frustum base_edge_original_pyramid height_original_pyramid base_edge_smaller_pyramid height_smaller_pyramid = 746.66 :=
by 
  sorry

end frustum_volume_correct_l63_63014


namespace system_no_solution_iff_n_eq_neg_one_l63_63218

def no_solution_system (n : ℝ) : Prop :=
  ¬∃ x y z : ℝ, (n * x + y = 1) ∧ (n * y + z = 1) ∧ (x + n * z = 1)

theorem system_no_solution_iff_n_eq_neg_one (n : ℝ) : no_solution_system n ↔ n = -1 :=
sorry

end system_no_solution_iff_n_eq_neg_one_l63_63218


namespace prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l63_63158

-- Definition of the inequalities to be proven using the rearrangement inequality
def inequality1 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def inequality2 (a b c : ℝ) : Prop := a^2 + b^2 + c^2 ≥ a * b + b * c + c * a
def inequality3 (a b : ℝ) : Prop := a^2 + b^2 + 1 ≥ a * b + b + a
def inequality5 (x y : ℝ) : Prop := x^3 + y^3 ≥ x^2 * y + x * y^2

-- Proofs required for each inequality
theorem prove_inequality1 (a b : ℝ) : inequality1 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality2 (a b c : ℝ) : inequality2 a b c := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality3 (a b : ℝ) : inequality3 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality5 (x y : ℝ) (hx : x ≥ y) (hy : 0 < y) : inequality5 x y := 
by sorry  -- This can be proved using the rearrangement inequality

end prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5_l63_63158


namespace original_fund_was_830_l63_63613

/- Define the number of employees as a variable -/
variables (n : ℕ)

/- Define the conditions given in the problem -/
def initial_fund := 60 * n - 10
def new_fund_after_distributing_50 := initial_fund - 50 * n
def remaining_fund := 130

/- State the proof goal -/
theorem original_fund_was_830 :
  initial_fund = 830 :=
by sorry

end original_fund_was_830_l63_63613


namespace conditional_probability_l63_63893

/- Conditions for the problem. -/
variables {Ω : Type} [ProbabilitySpace Ω]

-- There are 4 products, 3 first-class and 1 second-class
def num_products : ℕ := 4
def num_first_class : ℕ := 3
def num_second_class : ℕ := 1

-- Events A and B
def event_A (ω : Ω) : Prop := first_draw_first_class ω
def event_B (ω : Ω) : Prop := second_draw_first_class ω

-- Probabilities
def P (p : Prop) [decidable p] : ℝ := probability_measure p

-- The goal is to find P(B|A), which is the conditional probability of event B given event A.
theorem conditional_probability (hA : P event_A > 0) :
  P (event_B ∧ event_A) / P event_A = 2 / 3 :=
sorry

end conditional_probability_l63_63893


namespace prob_constraint_sum_digits_l63_63840

noncomputable def P (N : ℕ) :=
  let favorable_positions := (Nat.floor (2 * N / 5)) + 1 + (N - (Nat.ceil (3 * N / 5)) + 1)
  favorable_positions / (N + 1 : ℝ)

-- The objective is to establish the sum of the digits of the smallest multiple of 5 where P(N) < 321/400

theorem prob_constraint_sum_digits :
  let min_N := (List.range 1000).find (λ n, n % 5 = 0 ∧ P n < 321 / 400) ∨ 480 -- Use 480 as per problem's solution boundary
  let digit_sum := List.sum (List.map (λ c, c.toNat - '0'.toNat) (min_N.digits 10))
  digit_sum = 12 := 
sorry

end prob_constraint_sum_digits_l63_63840


namespace find_f_729_l63_63172

variable (f : ℕ+ → ℕ+) -- Define the function f on the positive integers.

-- Conditions of the problem.
axiom h1 : ∀ n : ℕ+, f (f n) = 3 * n
axiom h2 : ∀ n : ℕ+, f (3 * n + 1) = 3 * n + 2 

-- Proof statement.
theorem find_f_729 : f 729 = 729 :=
by
  sorry -- Placeholder for the proof.

end find_f_729_l63_63172


namespace cartesian_to_polar_circle_l63_63656

open Real

theorem cartesian_to_polar_circle (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = ρ * cos θ) 
  (h2 : y = ρ * sin θ) 
  (h3 : x^2 + y^2 - 2 * x = 0) : 
  ρ = 2 * cos θ :=
sorry

end cartesian_to_polar_circle_l63_63656


namespace vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l63_63435

def f (x : ℝ) : ℝ := -x^2 + 2 * x + 15
def g (x a : ℝ) : ℝ := (2 - 2 * a) * x - f x

theorem vertex_and_segment_condition : 
  (f 1 = 16) ∧ ∃ x1 x2 : ℝ, (f x1 = 0) ∧ (f x2 = 0) ∧ (x2 - x1 = 8) := 
sorry

theorem g_monotonically_increasing (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2 → g x1 a ≤ g x2 a) ↔ a ≤ 0 :=
sorry

theorem g_minimum_value (a : ℝ) :
  (0 < a ∧ g 2 a = -4 * a - 11) ∨ (a < 0 ∧ g 0 a = -15) ∨ (0 ≤ a ∧ a ≤ 2 ∧ g a a = -a^2 - 15) :=
sorry

end vertex_and_segment_condition_g_monotonically_increasing_g_minimum_value_l63_63435


namespace kaleb_games_per_box_l63_63407

theorem kaleb_games_per_box (initial_games sold_games boxes remaining_games games_per_box : ℕ)
  (h1 : initial_games = 76)
  (h2 : sold_games = 46)
  (h3 : boxes = 6)
  (h4 : remaining_games = initial_games - sold_games)
  (h5 : games_per_box = remaining_games / boxes) :
  games_per_box = 5 :=
sorry

end kaleb_games_per_box_l63_63407


namespace accounting_majors_l63_63680

theorem accounting_majors (p q r s t u : ℕ) 
  (hpqt : (p * q * r * s * t * u = 51030)) 
  (hineq : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) :
  p = 2 :=
sorry

end accounting_majors_l63_63680


namespace positive_difference_l63_63455

theorem positive_difference (a k : ℕ) (h1 : a = 8^2) (h2 : k = 8) :
  abs ((a + a) / k - (a * a) / k) = 496 :=
by
  sorry

end positive_difference_l63_63455


namespace smallest_n_rel_prime_to_300_l63_63136

theorem smallest_n_rel_prime_to_300 : ∃ n : ℕ, n > 1 ∧ Nat.gcd n 300 = 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → Nat.gcd m 300 ≠ 1 :=
by
  sorry

end smallest_n_rel_prime_to_300_l63_63136


namespace rectangle_length_l63_63736

variable (L W : ℕ)

theorem rectangle_length (h1 : 6 * W = 5 * L) (h2 : W = 20) : L = 24 := by
  sorry

end rectangle_length_l63_63736


namespace sandwich_cost_90_cents_l63_63545

theorem sandwich_cost_90_cents :
  let cost_bread := 0.15
  let cost_ham := 0.25
  let cost_cheese := 0.35
  (2 * cost_bread + cost_ham + cost_cheese) * 100 = 90 := 
by
  sorry

end sandwich_cost_90_cents_l63_63545


namespace divisible_iff_exists_t_l63_63852

theorem divisible_iff_exists_t (a b m α : ℤ) (h_coprime : Int.gcd a m = 1) (h_divisible : a * α + b ≡ 0 [ZMOD m]):
  ∀ x : ℤ, (a * x + b ≡ 0 [ZMOD m]) ↔ ∃ t : ℤ, x = α + m * t :=
sorry

end divisible_iff_exists_t_l63_63852


namespace neg_one_power_zero_l63_63911

theorem neg_one_power_zero : (-1: ℤ)^0 = 1 := 
sorry

end neg_one_power_zero_l63_63911


namespace problem1_problem2_l63_63067

-- Problem 1: Prove the range of k for any real number x
theorem problem1 (k : ℝ) (x : ℝ) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  1 ≤ k ∧ k < 13 :=
sorry

-- Problem 2: Prove the range of k for any x in the interval (0, 1]
theorem problem2 (k : ℝ) (x : ℝ) (hx : 0 < x) (hx1 : x ≤ 1) (h : (k*x^2 + k*x + 4) / (x^2 + x + 1) > 1) :
  k > -1/2 :=
sorry

end problem1_problem2_l63_63067


namespace natural_numbers_equal_l63_63099

theorem natural_numbers_equal (a b : ℕ) (h : ∀ n : ℕ, ¬ Nat.coprime (a + n) (b + n)) : a = b := by
  sorry

end natural_numbers_equal_l63_63099


namespace quadratic_has_distinct_real_roots_l63_63821

open Classical

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_distinct_real_roots (k : ℝ) (h_nonzero : k ≠ 0) : 
  (k > -1) ↔ (discriminant k (-2) (-1) > 0) :=
by
  unfold discriminant
  simp
  linarith

end quadratic_has_distinct_real_roots_l63_63821


namespace accounting_majors_count_l63_63678

theorem accounting_majors_count (p q r s t u : ℕ) 
  (h_eq : p * q * r * s * t * u = 51030)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) : 
  p = 2 :=
sorry

end accounting_majors_count_l63_63678


namespace find_Q_l63_63470

variable (Q U P k : ℝ)

noncomputable def varies_directly_and_inversely : Prop :=
  P = k * (Q / U)

theorem find_Q (h : varies_directly_and_inversely Q U P k)
  (h1 : P = 6) (h2 : Q = 8) (h3 : U = 4)
  (h4 : P = 18) (h5 : U = 9) :
  Q = 54 :=
sorry

end find_Q_l63_63470


namespace intersection_M_N_l63_63814

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 - 3 * x = 0}

theorem intersection_M_N : M ∩ N = {0} :=
by sorry

end intersection_M_N_l63_63814


namespace triangle_problem_l63_63370

/-- 
Given a triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively, 
if b = 2 and 2*b*cos B = a*cos C + c*cos A,
prove that B = π/3 and find the maximum area of ΔABC.
-/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : b = 2) (h2 : 2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) :
  B = Real.pi / 3 ∧
  (∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ max_area = (1/2) * a * c * Real.sin B) :=
by
  sorry

end triangle_problem_l63_63370


namespace ms_emily_inheritance_l63_63093

theorem ms_emily_inheritance :
  ∃ (y : ℝ), 
    (0.25 * y + 0.15 * (y - 0.25 * y) = 19500) ∧
    (y = 53800) :=
by
  sorry

end ms_emily_inheritance_l63_63093


namespace find_second_number_l63_63864

theorem find_second_number
  (x y z : ℚ)
  (h1 : x + y + z = 120)
  (h2 : x = (3 : ℚ) / 4 * y)
  (h3 : z = (7 : ℚ) / 5 * y) :
  y = 800 / 21 :=
by
  sorry

end find_second_number_l63_63864


namespace correct_equation_l63_63440

variables (x : ℝ) (production_planned total_clothings : ℝ)
variables (increase_rate days_ahead : ℝ)

noncomputable def daily_production (x : ℝ) := x
noncomputable def total_production := 1000
noncomputable def production_per_day_due_to_overtime (x : ℝ) := x * (1 + 0.2 : ℝ)
noncomputable def original_completion_days (x : ℝ) := total_production / daily_production x
noncomputable def increased_production_completion_days (x : ℝ) := total_production / production_per_day_due_to_overtime x
noncomputable def days_difference := original_completion_days x - increased_production_completion_days x

theorem correct_equation : days_difference x = 2 := by
  sorry

end correct_equation_l63_63440


namespace remainder_of_n_when_divided_by_7_l63_63503

theorem remainder_of_n_when_divided_by_7 (n : ℕ) :
  (n^2 ≡ 2 [MOD 7]) ∧ (n^3 ≡ 6 [MOD 7]) → (n ≡ 3 [MOD 7]) :=
by sorry

end remainder_of_n_when_divided_by_7_l63_63503


namespace mark_min_correct_problems_l63_63400

noncomputable def mark_score (x : ℕ) : ℤ :=
  8 * x - 21

theorem mark_min_correct_problems (x : ℕ) :
  (4 * 2) + mark_score x ≥ 120 ↔ x ≥ 17 :=
by
  sorry

end mark_min_correct_problems_l63_63400


namespace find_m_for_all_n_l63_63459

def sum_of_digits (k: ℕ) : ℕ :=
  k.digits 10 |>.sum

def A (k: ℕ) : ℕ :=
  -- Constructing the number A_k as described
  -- This is a placeholder for the actual implementation
  sorry

theorem find_m_for_all_n (n: ℕ) (hn: 0 < n) :
  ∃ m: ℕ, 0 < m ∧ n ∣ A m ∧ n ∣ m ∧ n ∣ sum_of_digits (A m) :=
sorry

end find_m_for_all_n_l63_63459


namespace distance_from_center_to_tangent_chord_l63_63207

theorem distance_from_center_to_tangent_chord
  (R a m x : ℝ)
  (h1 : m^2 = 4 * R^2)
  (h2 : 16 * R^2 * x^4 - 16 * R^2 * x^2 * (a^2 + R^2) + 16 * a^4 * R^4 - a^2 * (4 * R^2 - m^2)^2 = 0) :
  x = R :=
sorry

end distance_from_center_to_tangent_chord_l63_63207


namespace claire_balloon_count_l63_63630

variable (start_balloons lost_balloons initial_give_away more_give_away final_balloons grabbed_from_coworker : ℕ)

theorem claire_balloon_count (h1 : start_balloons = 50)
                           (h2 : lost_balloons = 12)
                           (h3 : initial_give_away = 1)
                           (h4 : more_give_away = 9)
                           (h5 : final_balloons = 39)
                           (h6 : start_balloons - initial_give_away - lost_balloons - more_give_away + grabbed_from_coworker = final_balloons) :
                           grabbed_from_coworker = 11 :=
by
  sorry

end claire_balloon_count_l63_63630


namespace range_of_m_l63_63062

theorem range_of_m (a b m : ℝ) (h1 : 2 * b = 2 * a + b) (h2 : b * b = a * a * b) (h3 : 0 < Real.log b / Real.log m) (h4 : Real.log b / Real.log m < 1) : m > 8 :=
sorry

end range_of_m_l63_63062


namespace find_numbers_sum_eq_S_product_eq_P_l63_63310

theorem find_numbers (S P : ℝ) (h : S^2 ≥ 4 * P) :
  ∃ x y : ℝ, (x + y = S) ∧ (x * y = P) :=
by
  have x1 : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
  have x2 : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2
  use x1, x2
  split
  sorry

-- additional definitions if needed for simplicity
def x1 (S P : ℝ) : ℝ := (S + real.sqrt (S^2 - 4 * P)) / 2
def x2 (S P : ℝ) : ℝ := (S - real.sqrt (S^2 - 4 * P)) / 2

theorem sum_eq_S (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P + x2 S P = S :=
by
  sorry

theorem product_eq_P (S P : ℝ) (h : S^2 ≥ 4 * P) : 
  x1 S P * x2 S P = P :=
by
  sorry

end find_numbers_sum_eq_S_product_eq_P_l63_63310


namespace number_of_steaks_needed_l63_63584

-- Definitions based on the conditions
def family_members : ℕ := 5
def pounds_per_member : ℕ := 1
def ounces_per_pound : ℕ := 16
def ounces_per_steak : ℕ := 20

-- Prove the number of steaks needed equals 4
theorem number_of_steaks_needed : (family_members * pounds_per_member * ounces_per_pound) / ounces_per_steak = 4 := by
  sorry

end number_of_steaks_needed_l63_63584


namespace repeating_decimal_fraction_l63_63641

theorem repeating_decimal_fraction : (0.366666... : ℝ) = 33 / 90 := 
sorry

end repeating_decimal_fraction_l63_63641


namespace initial_value_amount_l63_63181

theorem initial_value_amount (P : ℝ) 
  (h1 : ∀ t, t ≥ 0 → t = P * (1 + (1/8)) ^ t) 
  (h2 : P * (1 + (1/8)) ^ 2 = 105300) : 
  P = 83200 := 
sorry

end initial_value_amount_l63_63181


namespace michael_height_l63_63417

theorem michael_height (flagpole_height flagpole_shadow michael_shadow : ℝ) 
                        (h1 : flagpole_height = 50) 
                        (h2 : flagpole_shadow = 25) 
                        (h3 : michael_shadow = 5) : 
                        (michael_shadow * (flagpole_height / flagpole_shadow) = 10) :=
by
  sorry

end michael_height_l63_63417


namespace find_y_value_l63_63792

theorem find_y_value (x y : ℝ) (h1 : x^2 + y^2 - 4 = 0) (h2 : x^2 - y + 2 = 0) : y = 2 :=
by sorry

end find_y_value_l63_63792


namespace local_food_drive_correct_l63_63013

def local_food_drive_condition1 (R J x : ℕ) : Prop :=
  J = 2 * R + x

def local_food_drive_condition2 (J : ℕ) : Prop :=
  4 * J = 100

def local_food_drive_condition3 (R J : ℕ) : Prop :=
  R + J = 35

theorem local_food_drive_correct (R J x : ℕ)
  (h1 : local_food_drive_condition1 R J x)
  (h2 : local_food_drive_condition2 J)
  (h3 : local_food_drive_condition3 R J) :
  x = 5 :=
by
  sorry

end local_food_drive_correct_l63_63013


namespace pictures_per_day_calc_l63_63227

def years : ℕ := 3
def images_per_card : ℕ := 50
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

def number_of_cards : ℕ := total_spent / cost_per_card
def total_images : ℕ := number_of_cards * images_per_card
def days_in_year : ℕ := 365
def total_days : ℕ := years * days_in_year

theorem pictures_per_day_calc : 
  (total_images / total_days) = 10 := 
by
  sorry

end pictures_per_day_calc_l63_63227


namespace exists_sequences_satisfying_conditions_l63_63469

noncomputable def satisfies_conditions (n : ℕ) (hn : Odd n) 
  (a : Fin n → ℕ) (b : Fin n → ℕ) : Prop :=
  ∀ (k : Fin n), 0 < k.val → k.val < n →
    ∀ (i : Fin n),
      let in3n := 3 * n;
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n ≠
      (a i + b i) % in3n ∧
      (a i + b i) % in3n ≠
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ∧
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ≠
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n

theorem exists_sequences_satisfying_conditions :
  ∀ n : ℕ, Odd n → ∃ (a : Fin n → ℕ) (b : Fin n → ℕ),
    satisfies_conditions n sorry a b :=
sorry

end exists_sequences_satisfying_conditions_l63_63469


namespace priyas_speed_is_30_l63_63721

noncomputable def find_priyas_speed (v : ℝ) : Prop :=
  let riya_speed := 20
  let time := 0.5  -- in hours
  let distance_apart := 25
  (riya_speed + v) * time = distance_apart

theorem priyas_speed_is_30 : ∃ v : ℝ, find_priyas_speed v ∧ v = 30 :=
by
  sorry

end priyas_speed_is_30_l63_63721


namespace ellipse_equation_with_m_l63_63991

theorem ellipse_equation_with_m (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m ∈ Set.Ioi 5 := 
sorry

end ellipse_equation_with_m_l63_63991


namespace crayons_per_pack_l63_63242

theorem crayons_per_pack (total_crayons : ℕ) (num_packs : ℕ) (crayons_per_pack : ℕ) 
  (h1 : total_crayons = 615) (h2 : num_packs = 41) : crayons_per_pack = 15 := by
sorry

end crayons_per_pack_l63_63242


namespace total_songs_listened_l63_63589

theorem total_songs_listened (vivian_daily : ℕ) (fewer_songs : ℕ) (days_in_june : ℕ) (weekend_days : ℕ) :
  vivian_daily = 10 →
  fewer_songs = 2 →
  days_in_june = 30 →
  weekend_days = 8 →
  (vivian_daily * (days_in_june - weekend_days)) + ((vivian_daily - fewer_songs) * (days_in_june - weekend_days)) = 396 := 
by
  intros h1 h2 h3 h4
  sorry

end total_songs_listened_l63_63589


namespace initial_number_of_numbers_is_five_l63_63735

-- Define the conditions and the given problem
theorem initial_number_of_numbers_is_five
  (n : ℕ) (S : ℕ)
  (h1 : S / n = 27)
  (h2 : (S - 35) / (n - 1) = 25) : n = 5 :=
by
  sorry

end initial_number_of_numbers_is_five_l63_63735


namespace find_shirt_cost_l63_63768

def cost_each_shirt (x : ℝ) : Prop :=
  let total_purchase_price := x + 5 + 30 + 14
  let shipping_cost := if total_purchase_price > 50 then 0.2 * total_purchase_price else 5
  let total_bill := total_purchase_price + shipping_cost
  total_bill = 102

theorem find_shirt_cost (x : ℝ) (h : cost_each_shirt x) : x = 36 :=
sorry

end find_shirt_cost_l63_63768


namespace neg_one_to_zero_l63_63910

theorem neg_one_to_zero : (-1 : ℤ)^0 = 1 := 
by 
  -- Expanding expressions and applying the exponent rule for non-zero numbers
  sorry

end neg_one_to_zero_l63_63910


namespace sum_of_angles_subtended_by_arcs_l63_63717

theorem sum_of_angles_subtended_by_arcs
  (A B X Y C : Type)
  (arc_AX arc_XC : ℝ)
  (h1 : arc_AX = 58)
  (h2 : arc_XC = 62)
  (R S : ℝ)
  (hR : R = arc_AX / 2)
  (hS : S = arc_XC / 2) :
  R + S = 60 :=
by
  rw [hR, hS, h1, h2]
  norm_num

end sum_of_angles_subtended_by_arcs_l63_63717


namespace quadratic_solution_1_l63_63251

theorem quadratic_solution_1 :
  (∃ x, x^2 - 4 * x + 3 = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

end quadratic_solution_1_l63_63251


namespace accounting_majors_count_l63_63677

theorem accounting_majors_count (p q r s t u : ℕ) 
  (h_eq : p * q * r * s * t * u = 51030)
  (h_order : 1 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < t ∧ t < u) : 
  p = 2 :=
sorry

end accounting_majors_count_l63_63677


namespace lesser_fraction_l63_63275

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l63_63275


namespace jordan_more_novels_than_maxime_l63_63178

theorem jordan_more_novels_than_maxime :
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  jordan_novels - maxime_novels = 51 :=
by
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  sorry

end jordan_more_novels_than_maxime_l63_63178


namespace inequality_problem_l63_63202

theorem inequality_problem (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_sum : a + b + c + d = 4) : 
    a^2 * b * c + b^2 * d * a + c^2 * d * a + d^2 * b * c ≤ 4 := 
sorry

end inequality_problem_l63_63202


namespace even_num_Z_tetrominoes_l63_63043

-- Definitions based on the conditions of the problem
def is_tiled_with_S_tetrominoes (P : Type) : Prop := sorry
def tiling_uses_S_Z_tetrominoes (P : Type) : Prop := sorry
def num_Z_tetrominoes (P : Type) : ℕ := sorry

-- The theorem statement
theorem even_num_Z_tetrominoes (P : Type) 
  (hTiledWithS : is_tiled_with_S_tetrominoes P) 
  (hTilingWithSZ : tiling_uses_S_Z_tetrominoes P) : num_Z_tetrominoes P % 2 = 0 :=
sorry

end even_num_Z_tetrominoes_l63_63043


namespace coloring_scheme_formula_l63_63588

noncomputable def number_of_coloring_schemes (m n : ℕ) : ℕ :=
  if h : (m ≥ 2) ∧ (n ≥ 2) then
    m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n
  else 0

-- Formal statement verifying the formula for coloring schemes
theorem coloring_scheme_formula (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  number_of_coloring_schemes m n = m * ((-1 : ℤ)^n * (m - 2 : ℤ)).natAbs + (m - 2)^n :=
by sorry

end coloring_scheme_formula_l63_63588


namespace zero_of_fn_exists_between_2_and_3_l63_63580

open Real

noncomputable def f (x : ℝ) : ℝ := log x + 3 * x - 9

theorem zero_of_fn_exists_between_2_and_3 :
  ∃ x : ℝ, 2 < x ∧ x < 3 ∧ f x = 0 :=
sorry

end zero_of_fn_exists_between_2_and_3_l63_63580


namespace Archer_catch_total_fish_l63_63772

noncomputable def ArcherFishProblem : ℕ :=
  let firstRound := 8
  let secondRound := firstRound + 12
  let thirdRound := secondRound + (secondRound * 60 / 100)
  firstRound + secondRound + thirdRound

theorem Archer_catch_total_fish : ArcherFishProblem = 60 := by
  sorry

end Archer_catch_total_fish_l63_63772


namespace star_computation_l63_63979

def star (x y : ℝ) := x * y - 3 * x + y

theorem star_computation :
  (star 5 8) - (star 8 5) = 12 := by
  sorry

end star_computation_l63_63979


namespace max_sum_of_three_integers_with_product_24_l63_63863

theorem max_sum_of_three_integers_with_product_24 : ∃ (a b c : ℤ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧ a + b + c = 15) :=
by
  sorry

end max_sum_of_three_integers_with_product_24_l63_63863


namespace value_of_a_range_of_m_l63_63374

def f (x a : ℝ) : ℝ := abs (x - a)

-- Given the following conditions
axiom cond1 (x : ℝ) (a : ℝ) : f x a = abs (x - a)
axiom cond2 (x : ℝ) (a : ℝ) : (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)

-- Prove that a = 2
theorem value_of_a (a : ℝ) : (∀ x : ℝ, (f x a >= 3) ↔ (x <= 1 ∨ x >= 5)) → a = 2 := by
  sorry

-- Additional condition for m
axiom cond3 (x : ℝ) (a : ℝ) (m : ℝ) : ∀ x : ℝ, f x a + f (x + 4) a >= m

-- Prove that m ≤ 4
theorem range_of_m (a : ℝ) (m : ℝ) : (∀ x : ℝ, f x a + f (x + 4) a >= m) → a = 2 → m ≤ 4 := by
  sorry

end value_of_a_range_of_m_l63_63374


namespace remainder_sum_l63_63377

theorem remainder_sum (x y z : ℕ) (h1 : x % 15 = 11) (h2 : y % 15 = 13) (h3 : z % 15 = 9) :
  ((2 * (x % 15) + (y % 15) + (z % 15)) % 15) = 14 :=
by
  sorry

end remainder_sum_l63_63377


namespace find_numbers_l63_63298

theorem find_numbers (x y S P : ℝ) (h_sum : x + y = S) (h_prod : x * y = P) : 
  {x, y} = { (S + Real.sqrt (S^2 - 4*P)) / 2, (S - Real.sqrt (S^2 - 4*P)) / 2 } :=
by
  sorry

end find_numbers_l63_63298


namespace conor_vegetables_per_week_l63_63633

theorem conor_vegetables_per_week : 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := 
by 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  show (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 
  sorry

end conor_vegetables_per_week_l63_63633


namespace distinct_cubed_mod_7_units_digits_l63_63667

theorem distinct_cubed_mod_7_units_digits : 
  (∃ S : Finset ℕ, S.card = 3 ∧ ∀ n ∈ (Finset.range 7), (n^3 % 7) ∈ S) :=
  sorry

end distinct_cubed_mod_7_units_digits_l63_63667


namespace ab_plus_cd_l63_63215

variables (a b c d : ℝ)

axiom h1 : a + b + c = 1
axiom h2 : a + b + d = 6
axiom h3 : a + c + d = 15
axiom h4 : b + c + d = 10

theorem ab_plus_cd : a * b + c * d = 45.33333333333333 := 
by 
  sorry

end ab_plus_cd_l63_63215


namespace decompose_trig_expression_l63_63919

theorem decompose_trig_expression (x : ℝ) :
  1 - (sin x)^5 - (cos x)^5 = (1 - sin x) * (1 - cos x) * (3 + 2 * (sin x + cos x) + 2 * (sin x * cos x) + (sin x * cos x) * (sin x + cos x)) :=
by
  sorry

end decompose_trig_expression_l63_63919


namespace eighth_day_of_april_2000_is_saturday_l63_63429

noncomputable def april_2000_eight_day_is_saturday : Prop :=
  (∃ n : ℕ, (1 ≤ n ∧ n ≤ 7) ∧
            ((n + 0 * 7) = 2 ∨ (n + 1 * 7) = 2 ∨ (n + 2 * 7) = 2 ∨
             (n + 3 * 7) = 2 ∨ (n + 4 * 7) = 2) ∧
            ((n + 0 * 7) % 2 = 0 ∨ (n + 1 * 7) % 2 = 0 ∨
             (n + 2 * 7) % 2 = 0 ∨ (n + 3 * 7) % 2 = 0 ∨
             (n + 4 * 7) % 2 = 0) ∧
            (∃ k : ℕ, k ≤ 4 ∧ (n + k * 7 = 8))) ∧
            (8 % 7) = 1 ∧ (1 ≠ 0)

theorem eighth_day_of_april_2000_is_saturday :
  april_2000_eight_day_is_saturday := 
sorry

end eighth_day_of_april_2000_is_saturday_l63_63429


namespace xyz_inequality_l63_63103

theorem xyz_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z ≥ 1) :
    (x^4 + y) * (y^4 + z) * (z^4 + x) ≥ (x + y^2) * (y + z^2) * (z + x^2) :=
by
  sorry

end xyz_inequality_l63_63103


namespace sum_of_circle_areas_l63_63348

theorem sum_of_circle_areas 
    (r s t : ℝ)
    (h1 : r + s = 6)
    (h2 : r + t = 8)
    (h3 : s + t = 10) : 
    (π * r^2 + π * s^2 + π * t^2) = 36 * π := 
by
    sorry

end sum_of_circle_areas_l63_63348


namespace original_price_per_kg_l63_63295

theorem original_price_per_kg (P : ℝ) (S : ℝ) (reduced_price : ℝ := 0.8 * P) (total_cost : ℝ := 400) (extra_salt : ℝ := 10) :
  S * P = total_cost ∧ (S + extra_salt) * reduced_price = total_cost → P = 10 :=
by
  intros
  sorry

end original_price_per_kg_l63_63295


namespace evaluate_expression_l63_63357

theorem evaluate_expression : - (16 / 4 * 7 + 25 - 2 * 7) = -39 :=
by sorry

end evaluate_expression_l63_63357


namespace union_of_A_and_B_l63_63838

def setA : Set ℝ := {x : ℝ | x > 1 / 2}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

theorem union_of_A_and_B : setA ∪ setB = {x : ℝ | -1 < x} :=
by
  sorry

end union_of_A_and_B_l63_63838


namespace greatest_int_satisfying_inequality_l63_63447

theorem greatest_int_satisfying_inequality : ∃ n : ℤ, (∀ m : ℤ, m^2 - 13 * m + 40 ≤ 0 → m ≤ n) ∧ n = 8 := 
sorry

end greatest_int_satisfying_inequality_l63_63447


namespace angela_action_figures_l63_63487

theorem angela_action_figures (n s r g : ℕ) (hn : n = 24) (hs : s = n * 1 / 4) (hr : r = n - s) (hg : g = r * 1 / 3) :
  r - g = 12 :=
sorry

end angela_action_figures_l63_63487


namespace value_of_d_l63_63390

theorem value_of_d (d : ℝ) (h : ∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15 → True) : d = 5 :=
sorry

end value_of_d_l63_63390


namespace sum_of_coordinates_of_A_l63_63230

noncomputable def point := (ℝ × ℝ)
def B : point := (2, 6)
def C : point := (4, 12)
def AC (A C : point) : ℝ := (A.1 - C.1)^2 + (A.2 - C.2)^2
def AB (A B : point) : ℝ := (A.1 - B.1)^2 + (A.2 - B.2)^2
def BC (B C : point) : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2

theorem sum_of_coordinates_of_A :
  ∃ A : point, AC A C / AB A B = (1/3) ∧ BC B C / AB A B = (1/3) ∧ A.1 + A.2 = 24 :=
by
  sorry

end sum_of_coordinates_of_A_l63_63230


namespace election_valid_votes_l63_63468

variable (V : ℕ)
variable (invalid_pct : ℝ)
variable (exceed_pct : ℝ)
variable (total_votes : ℕ)
variable (invalid_votes : ℝ)
variable (valid_votes : ℕ)
variable (A_votes : ℕ)
variable (B_votes : ℕ)

theorem election_valid_votes :
  V = 9720 →
  invalid_pct = 0.20 →
  exceed_pct = 0.15 →
  total_votes = V →
  invalid_votes = invalid_pct * V →
  valid_votes = total_votes - invalid_votes →
  A_votes = B_votes + exceed_pct * total_votes →
  A_votes + B_votes = valid_votes →
  B_votes = 3159 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end election_valid_votes_l63_63468


namespace verify_sum_l63_63787

-- Definitions and conditions
def C : ℕ := 1
def D : ℕ := 2
def E : ℕ := 5

-- Base-6 addition representation
def is_valid_base_6_addition (a b c d : ℕ) : Prop :=
  (a + b) % 6 = c ∧ (a + b) / 6 = d

-- Given the addition problem:
def addition_problem : Prop :=
  is_valid_base_6_addition 2 5 C 0 ∧
  is_valid_base_6_addition 4 C E 0 ∧
  is_valid_base_6_addition D 2 4 0

-- Goal to prove
theorem verify_sum : addition_problem → C + D + E = 6 :=
by
  sorry

end verify_sum_l63_63787


namespace expression_identity_l63_63288

theorem expression_identity (a : ℤ) (h : a = 102) : 
  a^4 - 4 * a^3 + 6 * a^2 - 4 * a + 1 = 104060401 :=
by {
  rw h,
  calc 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 101^4 : by sorry
  ... = 104060401 : by sorry
}

end expression_identity_l63_63288


namespace richard_twice_as_old_as_scott_in_8_years_l63_63320

theorem richard_twice_as_old_as_scott_in_8_years :
  (richard_age - david_age = 6) ∧ (david_age - scott_age = 8) ∧ (david_age = 14) →
  (richard_age + 8 = 2 * (scott_age + 8)) :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end richard_twice_as_old_as_scott_in_8_years_l63_63320


namespace ellipse_properties_l63_63257

namespace MathProof

-- Definitions and conditions
variables {a b c : ℝ} {x y : ℝ}

def is_ellipse (a b : ℝ) (x y : ℝ) : Prop := (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1
def sum_of_distances_eq (F1 F2 G : ℝ × ℝ) (D : ℝ × ℝ) : ℝ :=
  (F1.1 - G.1) + (F2.1 - G.1) + (D.1 - G.1)
def DF1_eq_3GF1 (D F1 G : ℝ × ℝ) : Prop :=
  abs (D.1 - F1.1) = 3 * abs (G.1 - F1.1)
def line_through_fixed_point (x y : ℝ) : Prop := 3 * x = -2 * y - 2

-- Theorem statement
theorem ellipse_properties
  (h₁ : 0 < b) (h₂ : b < a)
  (hE : is_ellipse a b x y)
  (hDGF2 : sum_of_distances_eq (0,0) (0,0) (x,y) (0,0) = 8)
  (h_relation : DF1_eq_3GF1 (0,0) (c,0) (x,y))
  (hyp : x = -4/3 * c ∧ y = 1/3 * b) :
  (is_ellipse 2 (sqrt 2) x y) ∧ (line_through_fixed_point (-2/3) 0) :=
sorry

end MathProof

end ellipse_properties_l63_63257


namespace min_value_fraction_l63_63061

theorem min_value_fraction (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b > 0) (h₃ : 2 * a + b = 1) : 
  ∃ x, x = 8 ∧ ∀ y, (y = (1 / a) + (2 / b)) → y ≥ x :=
sorry

end min_value_fraction_l63_63061


namespace geometric_condition_l63_63663

def Sn (p : ℤ) (n : ℕ) : ℤ := p * 2^n + 2

def an (p : ℤ) (n : ℕ) : ℤ :=
  if n = 1 then Sn p n
  else Sn p n - Sn p (n - 1)

def is_geometric_progression (p : ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → ∃ r : ℤ, an p n = an p (n - 1) * r

theorem geometric_condition (p : ℤ) :
  is_geometric_progression p ↔ p = -2 :=
sorry

end geometric_condition_l63_63663


namespace tommy_needs_4_steaks_l63_63586

noncomputable def tommy_steaks : Nat := 
  let family_members := 5
  let ounces_per_pound := 16
  let ounces_per_steak := 20
  let total_ounces_needed := family_members * ounces_per_pound
  let steaks_needed := total_ounces_needed / ounces_per_steak
  steaks_needed

theorem tommy_needs_4_steaks :
  tommy_steaks = 4 :=
by
  sorry

end tommy_needs_4_steaks_l63_63586


namespace arrangements_count_correct_l63_63157

def arrangements_total : Nat :=
  let total_with_A_first := (Nat.factorial 5) -- A^5_5 = 120
  let total_with_B_first := (Nat.factorial 4) * 1 -- A^1_4 * A^4_4 = 96
  total_with_A_first + total_with_B_first

theorem arrangements_count_correct : arrangements_total = 216 := 
by
  -- Proof is required here
  sorry

end arrangements_count_correct_l63_63157


namespace single_cow_single_bag_l63_63825

-- Definitions given in the problem conditions
def cows : ℕ := 26
def bags : ℕ := 26
def days : ℕ := 26

-- Statement to be proved
theorem single_cow_single_bag : (1 : ℕ) = 26 := sorry

end single_cow_single_bag_l63_63825


namespace johns_original_earnings_l63_63692

-- Define the conditions
def raises (original : ℝ) (percentage : ℝ) := original + original * percentage

-- The theorem stating the equivalent problem proof
theorem johns_original_earnings :
  ∃ (x : ℝ), raises x 0.375 = 55 ↔ x = 40 :=
sorry

end johns_original_earnings_l63_63692


namespace value_of_d_l63_63391

theorem value_of_d (d : ℝ) (h : ∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15 → True) : d = 5 :=
sorry

end value_of_d_l63_63391


namespace woman_first_half_speed_l63_63620

noncomputable def first_half_speed (total_time : ℕ) (second_half_speed : ℕ) (total_distance : ℕ) : ℕ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem woman_first_half_speed : first_half_speed 20 24 448 = 21 := by
  sorry

end woman_first_half_speed_l63_63620


namespace angle_SQR_l63_63411

-- Define angles
def PQR : ℝ := 40
def PQS : ℝ := 28

-- State the theorem
theorem angle_SQR : PQR - PQS = 12 := by
  sorry

end angle_SQR_l63_63411


namespace find_varphi_intervals_of_increase_l63_63555

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem find_varphi (φ : ℝ) (h1 : -Real.pi < φ) (h2 : φ < 0)
  (h3 : ∃ k : ℤ, 2 * (Real.pi / 8) + φ = (Real.pi / 2) + k * Real.pi) :
  φ = -3 * Real.pi / 4 :=
sorry

theorem intervals_of_increase (m : ℤ) :
  ∀ x : ℝ, (π / 8 + m * π ≤ x ∧ x ≤ 5 * π / 8 + m * π) ↔
  Real.sin (2 * x - 3 * π / 4) > 0 :=
sorry

end find_varphi_intervals_of_increase_l63_63555


namespace prove_inequality_l63_63711

open Real

noncomputable def inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ 
  3 * (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1)

theorem prove_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  inequality a b c h1 h2 h3 := 
  sorry

end prove_inequality_l63_63711


namespace growth_rate_yield_per_acre_l63_63756

theorem growth_rate_yield_per_acre (x : ℝ) (a_i y_i y_f : ℝ) (h1 : a_i = 5) (h2 : y_i = 10000) (h3 : y_f = 30000) 
  (h4 : y_f = 5 * (1 + 2 * x) * (y_i / a_i) * (1 + x)) : x = 0.5 := 
by
  -- Insert the proof here
  sorry

end growth_rate_yield_per_acre_l63_63756


namespace arithmetic_sequence_problem_l63_63267

variable {a_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}
variable (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * d)
variable (h_S_n : ∀ n, S_n n = (n * (a_n 1 + a_n n)) / 2)

theorem arithmetic_sequence_problem
  (h1 : S_n 5 = 2 * a_n 5)
  (h2 : a_n 3 = -4) :
  a_n 9 = -22 := sorry

end arithmetic_sequence_problem_l63_63267


namespace no_such_convex_hexagon_and_point_l63_63922

noncomputable def convex_hexagon_and_point_statement : Prop :=
  ∀ (hexagon : List (ℝ × ℝ)) (M : ℝ × ℝ),
    hexagon.length = 6 ∧ 
    (∀ (p q : ℝ × ℝ), p ∈ hexagon → q ∈ hexagon → p ≠ q → convex (p, q)) ∧
    (∀ (i : ℕ) (hi : i < hexagon.length), dist (hexagon.nth_le i hi) (hexagon.nth_le ((i + 1) % hexagon.length) ((i + 1) % hexagon.length).lt_of_lt nat.succ_pos') > 1) ∧
    (∀ (v : ℝ × ℝ), v ∈ hexagon → dist M v < 1) →
    ∃ (v : ℝ × ℝ), v ∈ hexagon → dist (hexagon.head) v < dist M v

theorem no_such_convex_hexagon_and_point : ¬ convex_hexagon_and_point_statement :=
by
  sorry

end no_such_convex_hexagon_and_point_l63_63922


namespace angle_A_is_30_degrees_l63_63964

theorem angle_A_is_30_degrees {A : ℝ} (hA_acute : 0 < A ∧ A < π / 2) (hA_sin : Real.sin A = 1 / 2) : A = π / 6 :=
sorry

end angle_A_is_30_degrees_l63_63964


namespace remainder_when_squared_expression_divided_by_40_l63_63668

theorem remainder_when_squared_expression_divided_by_40
  (k : ℤ) : ((40 * k - 1)^2 - 3 * (40 * k - 1) + 5) % 40 = 9 := 
by {
  -- calculation steps are omitted
  sorry
}

end remainder_when_squared_expression_divided_by_40_l63_63668


namespace percentage_of_girls_who_like_basketball_l63_63834

theorem percentage_of_girls_who_like_basketball 
  (total_students : ℕ)
  (percentage_girls : ℝ)
  (percentage_boys_basketball : ℝ)
  (factor_girls_to_boys_not_basketball : ℝ)
  (total_students_eq : total_students = 25)
  (percentage_girls_eq : percentage_girls = 0.60)
  (percentage_boys_basketball_eq : percentage_boys_basketball = 0.40)
  (factor_girls_to_boys_not_basketball_eq : factor_girls_to_boys_not_basketball = 2) 
  : 
  ((factor_girls_to_boys_not_basketball * (total_students * (1 - percentage_girls) * (1 - percentage_boys_basketball))) / 
  (total_students * percentage_girls)) * 100 = 80 :=
by
  sorry

end percentage_of_girls_who_like_basketball_l63_63834


namespace find_two_numbers_l63_63301

theorem find_two_numbers (S P : ℝ) : 
  let x₁ := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let x₂ := (S - Real.sqrt (S^2 - 4 * P)) / 2
  ∃ x y : ℝ, (x + y = S ∧ x * y = P) ∧ (x = x₁ ∨ x = x₂) ∧ (y = S - x) :=
by
  sorry

end find_two_numbers_l63_63301


namespace problem_ab_cd_eq_l63_63214

theorem problem_ab_cd_eq (a b c d : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a + b + d = 5)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 14) :
  ab + cd = 45 := 
by
  sorry

end problem_ab_cd_eq_l63_63214


namespace volume_inhaled_per_breath_is_correct_l63_63853

def breaths_per_minute : ℤ := 17
def volume_inhaled_24_hours : ℤ := 13600
def minutes_per_hour : ℤ := 60
def hours_per_day : ℤ := 24

def total_minutes_24_hours : ℤ := hours_per_day * minutes_per_hour
def total_breaths_24_hours : ℤ := total_minutes_24_hours * breaths_per_minute
def volume_per_breath := (volume_inhaled_24_hours : ℚ) / (total_breaths_24_hours : ℚ)

theorem volume_inhaled_per_breath_is_correct :
  volume_per_breath = 0.5556 := by
  sorry

end volume_inhaled_per_breath_is_correct_l63_63853


namespace difference_of_sums_l63_63591

noncomputable def sum_of_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

noncomputable def sum_of_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem difference_of_sums : 
  sum_of_first_n_even 2004 - sum_of_first_n_odd 2003 = 6017 := 
by sorry

end difference_of_sums_l63_63591


namespace distance_between_homes_l63_63416

theorem distance_between_homes (Maxwell_distance : ℝ) (Maxwell_speed : ℝ) (Brad_speed : ℝ) (midpoint : ℝ) 
    (h1 : Maxwell_speed = 2) 
    (h2 : Brad_speed = 4) 
    (h3 : Maxwell_distance = 12) 
    (h4 : midpoint = Maxwell_distance * 2 * (Brad_speed / Maxwell_speed) + Maxwell_distance) :
midpoint = 36 :=
by
  sorry

end distance_between_homes_l63_63416


namespace largest_value_of_n_l63_63422

theorem largest_value_of_n :
  ∃ (n : ℕ) (X Y Z : ℕ),
    n = 25 * X + 5 * Y + Z ∧
    n = 81 * Z + 9 * Y + X ∧
    X < 5 ∧ Y < 5 ∧ Z < 5 ∧
    n = 121 := by
  sorry

end largest_value_of_n_l63_63422


namespace lesser_fraction_l63_63270

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3 / 4) (h₂ : x * y = 1 / 8) : min x y = 1 / 4 :=
sorry

end lesser_fraction_l63_63270


namespace numbers_pairs_sum_prod_l63_63306

noncomputable def find_numbers_pairs (S P : ℝ) 
  (h_real_sol : S^2 ≥ 4 * P) :
  (ℝ × ℝ) × (ℝ × ℝ) :=
  let x1 := (S + Real.sqrt (S^2 - 4 * P)) / 2
  let y1 := S - x1
  let x2 := (S - Real.sqrt (S^2 - 4 * P)) / 2
  let y2 := S - x2
  ((x1, y1), (x2, y2))

theorem numbers_pairs_sum_prod (S P : ℝ) (h_real_sol : S^2 ≥ 4 * P) :
  let ((x1, y1), (x2, y2)) := find_numbers_pairs S P h_real_sol in
  (x1 + y1 = S ∧ x2 + y2 = S) ∧ (x1 * y1 = P ∧ x2 * y2 = P) :=
by
  sorry

end numbers_pairs_sum_prod_l63_63306


namespace number_of_rabbits_l63_63324

theorem number_of_rabbits
  (dogs : ℕ) (cats : ℕ) (total_animals : ℕ)
  (joins_each_cat : ℕ → ℕ)
  (hares_per_rabbit : ℕ)
  (h_dogs : dogs = 1)
  (h_cats : cats = 4)
  (h_total : total_animals = 37)
  (h_hares_per_rabbit : hares_per_rabbit = 3)
  (H : total_animals = dogs + cats + 4 * joins_each_cat cats + 3 * 4 * joins_each_cat cats) :
  joins_each_cat cats = 2 :=
by
  sorry

end number_of_rabbits_l63_63324


namespace max_equilateral_triangle_area_l63_63268

theorem max_equilateral_triangle_area (length width : ℝ) (h_len : length = 15) (h_width : width = 12) 
: ∃ (area : ℝ), area = 200.25 * Real.sqrt 3 - 450 := by
  sorry

end max_equilateral_triangle_area_l63_63268


namespace binom_10_4_l63_63163

theorem binom_10_4 : Nat.choose 10 4 = 210 := 
by sorry

end binom_10_4_l63_63163


namespace tim_surprises_combinations_l63_63742

theorem tim_surprises_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 120 :=
by
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  sorry

end tim_surprises_combinations_l63_63742


namespace Kayla_total_items_l63_63701

theorem Kayla_total_items (T_bars : ℕ) (T_cans : ℕ) (K_bars : ℕ) (K_cans : ℕ)
  (h1 : T_bars = 2 * K_bars) (h2 : T_cans = 2 * K_cans)
  (h3 : T_bars = 12) (h4 : T_cans = 18) : 
  K_bars + K_cans = 15 :=
by {
  -- In order to focus only on statement definition, we use sorry here
  sorry
}

end Kayla_total_items_l63_63701


namespace find_pictures_museum_l63_63138

-- Define the given conditions
def pictures_zoo : Nat := 24
def pictures_deleted : Nat := 14
def pictures_remaining : Nat := 22

-- Define the target: the number of pictures taken at the museum
def pictures_museum : Nat := 12

-- State the goal to be proved
theorem find_pictures_museum :
  pictures_zoo + pictures_museum - pictures_deleted = pictures_remaining :=
sorry

end find_pictures_museum_l63_63138


namespace repeat_decimal_to_fraction_l63_63640

theorem repeat_decimal_to_fraction : 0.36666 = 11 / 30 :=
by {
    sorry
}

end repeat_decimal_to_fraction_l63_63640


namespace julie_savings_fraction_l63_63884

variables (S : ℝ) (x : ℝ)
theorem julie_savings_fraction (h : 12 * S * x = 4 * S * (1 - x)) : 1 - x = 3 / 4 :=
sorry

end julie_savings_fraction_l63_63884


namespace problem1_problem2_l63_63366

-- Problem 1: Prove that x = ±7/2 given 4x^2 - 49 = 0
theorem problem1 (x : ℝ) : 4 * x^2 - 49 = 0 → x = 7 / 2 ∨ x = -7 / 2 := 
by
  sorry

-- Problem 2: Prove that x = 2 given (x + 1)^3 - 27 = 0
theorem problem2 (x : ℝ) : (x + 1)^3 - 27 = 0 → x = 2 := 
by
  sorry

end problem1_problem2_l63_63366


namespace total_players_l63_63006

-- Definitions for conditions
def K : Nat := 10
def KK : Nat := 30
def B : Nat := 5

-- Statement of the proof problem
theorem total_players : K + KK - B = 35 :=
by
  -- Proof not required, just providing the statement
  sorry

end total_players_l63_63006


namespace mod_arith_example_l63_63791

theorem mod_arith_example :
  (3 * 7⁻¹ + 5 * 13⁻¹) % 63 = 13 % 63 :=
by
  -- Given inverses
  have h7_inv : 7⁻¹ % 63 = 19 % 63 := sorry,
  have h13_inv : 13⁻¹ % 63 = 29 % 63 := sorry,
  -- The main goal is
  calc
    (3 * 7⁻¹ + 5 * 13⁻¹) % 63
        = (3 * 19 + 5 * 29) % 63 : by
          rw [h7_inv, h13_inv]
        ... = 202 % 63 : by
          norm_num
        ... = 13 % 63 : by
          norm_num

end mod_arith_example_l63_63791


namespace find_base_l63_63980

theorem find_base (x y : ℕ) (b : ℕ) (h1 : 3 ^ x * b ^ y = 19683) (h2 : x - y = 9) (h3 : x = 9) : b = 1 := 
by
  sorry

end find_base_l63_63980


namespace tan_435_eq_2_add_sqrt_3_l63_63344

theorem tan_435_eq_2_add_sqrt_3 :
  Real.tan (435 * Real.pi / 180) = 2 + Real.sqrt 3 :=
by
  sorry

end tan_435_eq_2_add_sqrt_3_l63_63344


namespace train_passes_man_in_correct_time_l63_63765

-- Definitions for the given conditions
def platform_length : ℝ := 270
def train_length : ℝ := 180
def crossing_time : ℝ := 20

-- Theorem to prove the time taken to pass the man is 8 seconds
theorem train_passes_man_in_correct_time
  (p: ℝ) (l: ℝ) (t_cross: ℝ)
  (h1: p = platform_length)
  (h2: l = train_length)
  (h3: t_cross = crossing_time) :
  l / ((l + p) / t_cross) = 8 := by
  -- Proof goes here
  sorry

end train_passes_man_in_correct_time_l63_63765


namespace average_of_11_numbers_l63_63734

theorem average_of_11_numbers (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ)
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = 58)
  (h2 : (a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 6 = 65)
  (h3 : a₆ = 78) : 
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁) / 11 = 60 := 
by 
  sorry 

end average_of_11_numbers_l63_63734


namespace find_unique_pair_l63_63794

theorem find_unique_pair (x y : ℝ) :
  (∀ (u v : ℝ), (u * x + v * y = u) ∧ (u * y + v * x = v)) ↔ (x = 1 ∧ y = 0) :=
by
  -- This is to ignore the proof part
  sorry

end find_unique_pair_l63_63794
