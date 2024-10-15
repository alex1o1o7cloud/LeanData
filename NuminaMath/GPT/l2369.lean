import Mathlib

namespace NUMINAMATH_GPT_arithmetic_seq_a7_a8_l2369_236934

theorem arithmetic_seq_a7_a8 (a : ℕ → ℤ) (d : ℤ) (h₁ : a 1 + a 2 = 4) (h₂ : d = 2) :
  a 7 + a 8 = 28 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_a8_l2369_236934


namespace NUMINAMATH_GPT_seokgi_share_is_67_l2369_236914

-- The total length of the wire
def length_of_wire := 150

-- Seokgi's share is 16 cm shorter than Yeseul's share
def is_shorter_by (Y S : ℕ) := S = Y - 16

-- The sum of Yeseul's and Seokgi's shares equals the total length
def total_share (Y S : ℕ) := Y + S = length_of_wire

-- Prove that Seokgi's share is 67 cm
theorem seokgi_share_is_67 (Y S : ℕ) (h1 : is_shorter_by Y S) (h2 : total_share Y S) : 
  S = 67 :=
sorry

end NUMINAMATH_GPT_seokgi_share_is_67_l2369_236914


namespace NUMINAMATH_GPT_job_completion_time_l2369_236981

theorem job_completion_time (x : ℤ) (hx : (4 : ℝ) / x + (2 : ℝ) / 3 = 1) : x = 12 := by
  sorry

end NUMINAMATH_GPT_job_completion_time_l2369_236981


namespace NUMINAMATH_GPT_cos_2alpha_plus_pi_over_3_l2369_236982

open Real

theorem cos_2alpha_plus_pi_over_3 
  (alpha : ℝ) 
  (h1 : cos (alpha - π / 12) = 3 / 5) 
  (h2 : 0 < alpha ∧ alpha < π / 2) : 
  cos (2 * alpha + π / 3) = -24 / 25 := 
sorry

end NUMINAMATH_GPT_cos_2alpha_plus_pi_over_3_l2369_236982


namespace NUMINAMATH_GPT_evaluate_expression_l2369_236913

theorem evaluate_expression : 
  (-2 : ℤ)^2004 + 3 * (-2)^2003 = (-2)^2003 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2369_236913


namespace NUMINAMATH_GPT_laptop_sticker_price_l2369_236936

theorem laptop_sticker_price (x : ℝ) (h₁ : 0.70 * x = 0.80 * x - 50 - 30) : x = 800 := 
  sorry

end NUMINAMATH_GPT_laptop_sticker_price_l2369_236936


namespace NUMINAMATH_GPT_nikka_us_stamp_percentage_l2369_236995

/-- 
Prove that 20% of Nikka's stamp collection are US stamps given the following conditions:
1. Nikka has a total of 100 stamps.
2. 35 of those stamps are Chinese.
3. 45 of those stamps are Japanese.
-/
theorem nikka_us_stamp_percentage
  (total_stamps : ℕ)
  (chinese_stamps : ℕ)
  (japanese_stamps : ℕ)
  (h1 : total_stamps = 100)
  (h2 : chinese_stamps = 35)
  (h3 : japanese_stamps = 45) :
  ((total_stamps - (chinese_stamps + japanese_stamps)) / total_stamps) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_nikka_us_stamp_percentage_l2369_236995


namespace NUMINAMATH_GPT_exist_a_b_for_every_n_l2369_236992

theorem exist_a_b_for_every_n (n : ℕ) (hn : 0 < n) : 
  ∃ (a b : ℤ), 1 < a ∧ 1 < b ∧ a^2 + 1 = 2 * b^2 ∧ (a - b) % n = 0 := 
sorry

end NUMINAMATH_GPT_exist_a_b_for_every_n_l2369_236992


namespace NUMINAMATH_GPT_c_investment_l2369_236985

theorem c_investment 
  (A_investment B_investment : ℝ)
  (C_share total_profit : ℝ)
  (hA : A_investment = 8000)
  (hB : B_investment = 4000)
  (hC_share : C_share = 36000)
  (h_profit : total_profit = 252000) :
  ∃ (x : ℝ), (x / 4000) / (2 + 1 + x / 4000) = (36000 / 252000) ∧ x = 2000 :=
by
  sorry

end NUMINAMATH_GPT_c_investment_l2369_236985


namespace NUMINAMATH_GPT_valid_b_values_count_l2369_236965

theorem valid_b_values_count : 
  (∃! b : ℤ, ∃ x1 x2 x3 : ℤ, 
    (∀ x : ℤ, x^2 + b * x + 5 ≤ 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧ 
    (20 ≤ b^2 ∧ b^2 < 29)) :=
sorry

end NUMINAMATH_GPT_valid_b_values_count_l2369_236965


namespace NUMINAMATH_GPT_christian_sue_need_more_money_l2369_236996

-- Definitions based on the given conditions
def bottle_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def christian_mowing_rate : ℕ := 5
def christian_mowing_count : ℕ := 4
def sue_walking_rate : ℕ := 2
def sue_walking_count : ℕ := 6

-- Prove that Christian and Sue will need 6 more dollars to buy the bottle of perfume
theorem christian_sue_need_more_money :
  let christian_earning := christian_mowing_rate * christian_mowing_count
  let christian_total := christian_initial + christian_earning
  let sue_earning := sue_walking_rate * sue_walking_count
  let sue_total := sue_initial + sue_earning
  let total_money := christian_total + sue_total
  50 - total_money = 6 :=
by
  sorry

end NUMINAMATH_GPT_christian_sue_need_more_money_l2369_236996


namespace NUMINAMATH_GPT_percent_runs_by_running_eq_18_75_l2369_236903

/-
Define required conditions.
-/
def total_runs : ℕ := 224
def boundaries_runs : ℕ := 9 * 4
def sixes_runs : ℕ := 8 * 6
def twos_runs : ℕ := 12 * 2
def threes_runs : ℕ := 4 * 3
def byes_runs : ℕ := 6 * 1
def running_runs : ℕ := twos_runs + threes_runs + byes_runs

/-
Define the proof problem to show that the percentage of the total score made by running between the wickets is 18.75%.
-/
theorem percent_runs_by_running_eq_18_75 : (running_runs : ℚ) / total_runs * 100 = 18.75 := by
  sorry

end NUMINAMATH_GPT_percent_runs_by_running_eq_18_75_l2369_236903


namespace NUMINAMATH_GPT_non_rain_hours_correct_l2369_236970

def total_hours : ℕ := 9
def rain_hours : ℕ := 4

theorem non_rain_hours_correct : (total_hours - rain_hours) = 5 := 
by
  sorry

end NUMINAMATH_GPT_non_rain_hours_correct_l2369_236970


namespace NUMINAMATH_GPT_x_100_equals_2_power_397_l2369_236993

-- Define the sequences
noncomputable def a_n (n : ℕ) : ℕ := 2^(n-1)
noncomputable def b_n (n : ℕ) : ℕ := 5*n - 3

-- Define the merged sequence x_n
noncomputable def x_n (k : ℕ) : ℕ := 2^(4*k - 3)

-- Prove x_100 is 2^397
theorem x_100_equals_2_power_397 : x_n 100 = 2^397 := by
  unfold x_n
  show 2^(4*100 - 3) = 2^397
  rfl

end NUMINAMATH_GPT_x_100_equals_2_power_397_l2369_236993


namespace NUMINAMATH_GPT_fixed_monthly_fee_l2369_236912

theorem fixed_monthly_fee (x y z : ℝ) 
  (h1 : x + y = 18.50) 
  (h2 : x + y + 3 * z = 23.45) : 
  x = 7.42 := 
by 
  sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l2369_236912


namespace NUMINAMATH_GPT_jasmine_cookies_l2369_236950

theorem jasmine_cookies (J : ℕ) (h1 : 20 + J + (J + 10) = 60) : J = 15 :=
sorry

end NUMINAMATH_GPT_jasmine_cookies_l2369_236950


namespace NUMINAMATH_GPT_isosceles_triangle_angle_condition_l2369_236964

theorem isosceles_triangle_angle_condition (A B C : ℝ) (h_iso : A = B) (h_angle_eq : A = 2 * C ∨ C = 2 * A) :
    (A = 45 ∨ A = 72) ∧ (B = 45 ∨ B = 72) :=
by
  -- Given isosceles triangle properties.
  sorry

end NUMINAMATH_GPT_isosceles_triangle_angle_condition_l2369_236964


namespace NUMINAMATH_GPT_probability_of_defective_product_is_0_032_l2369_236944

-- Defining the events and their probabilities
def P_H1 : ℝ := 0.30
def P_H2 : ℝ := 0.25
def P_H3 : ℝ := 0.45

-- Defining the probabilities of defects given each production line
def P_A_given_H1 : ℝ := 0.03
def P_A_given_H2 : ℝ := 0.02
def P_A_given_H3 : ℝ := 0.04

-- Summing up the total probabilities
def P_A : ℝ :=
  P_H1 * P_A_given_H1 +
  P_H2 * P_A_given_H2 +
  P_H3 * P_A_given_H3

-- The statement to be proven
theorem probability_of_defective_product_is_0_032 :
  P_A = 0.032 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_probability_of_defective_product_is_0_032_l2369_236944


namespace NUMINAMATH_GPT_sqrt_mul_sqrt_eq_six_l2369_236978

theorem sqrt_mul_sqrt_eq_six : (Real.sqrt 3) * (Real.sqrt 12) = 6 := 
sorry

end NUMINAMATH_GPT_sqrt_mul_sqrt_eq_six_l2369_236978


namespace NUMINAMATH_GPT_storks_more_than_birds_l2369_236909

def birds := 4
def initial_storks := 3
def additional_storks := 6

theorem storks_more_than_birds :
  (initial_storks + additional_storks) - birds = 5 := 
by
  sorry

end NUMINAMATH_GPT_storks_more_than_birds_l2369_236909


namespace NUMINAMATH_GPT_sum_of_octal_numbers_l2369_236955

theorem sum_of_octal_numbers :
  (176 : ℕ) + 725 + 63 = 1066 := by
sorry

end NUMINAMATH_GPT_sum_of_octal_numbers_l2369_236955


namespace NUMINAMATH_GPT_find_inheritance_amount_l2369_236972

noncomputable def totalInheritance (tax_amount : ℕ) : ℕ :=
  let federal_rate := 0.20
  let state_rate := 0.10
  let combined_rate := federal_rate + (state_rate * (1 - federal_rate))
  sorry

theorem find_inheritance_amount : totalInheritance 10500 = 37500 := 
  sorry

end NUMINAMATH_GPT_find_inheritance_amount_l2369_236972


namespace NUMINAMATH_GPT_find_r_in_parallelogram_l2369_236916

theorem find_r_in_parallelogram 
  (θ : ℝ) 
  (r : ℝ)
  (CAB DBA DBC ACB AOB : ℝ)
  (h1 : CAB = 3 * DBA)
  (h2 : DBC = 2 * DBA)
  (h3 : ACB = r * (t * AOB))
  (h4 : t = 4 / 3)
  (h5 : AOB = 180 - 2 * DBA)
  (h6 : ACB = 180 - 4 * DBA) :
  r = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_r_in_parallelogram_l2369_236916


namespace NUMINAMATH_GPT_adam_bought_26_books_l2369_236935

-- Conditions
def initial_books : ℕ := 56
def shelves : ℕ := 4
def avg_books_per_shelf : ℕ := 20
def leftover_books : ℕ := 2

-- Definitions based on conditions
def capacity_books : ℕ := shelves * avg_books_per_shelf
def total_books_after_trip : ℕ := capacity_books + leftover_books

-- Question: How many books did Adam buy on his shopping trip?
def books_bought : ℕ := total_books_after_trip - initial_books

theorem adam_bought_26_books :
  books_bought = 26 :=
by
  sorry

end NUMINAMATH_GPT_adam_bought_26_books_l2369_236935


namespace NUMINAMATH_GPT_part1_part2_l2369_236974

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

theorem part1 (x : ℝ) : (f x 1) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > -a) ↔ (a > -3/2) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2369_236974


namespace NUMINAMATH_GPT_number_of_sets_l2369_236929

theorem number_of_sets (M : Set ℕ) : 
  {1, 2} ⊆ M → M ⊆ {1, 2, 3, 4} → ∃ n : ℕ, n = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sets_l2369_236929


namespace NUMINAMATH_GPT_tangent_line_at_1_l2369_236947

def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_1 (f' : ℝ → ℝ) (h1 : ∀ x, deriv f x = f' x) (h2 : ∀ y, 2 * 1 + y - 3 = 0) :
  f' 1 + f 1 = -1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_l2369_236947


namespace NUMINAMATH_GPT_triangle_inequality_l2369_236923

theorem triangle_inequality (a b c m_A : ℝ)
  (h1 : 2*m_A ≤ b + c)
  (h2 : a^2 + (2*m_A)^2 = (b^2) + (c^2)) :
  a^2 + 4*m_A^2 ≤ (b + c)^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_l2369_236923


namespace NUMINAMATH_GPT_petya_digits_l2369_236922

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end NUMINAMATH_GPT_petya_digits_l2369_236922


namespace NUMINAMATH_GPT_probability_in_dark_l2369_236904

theorem probability_in_dark (rev_per_min : ℕ) (given_prob : ℝ) (h1 : rev_per_min = 3) (h2 : given_prob = 0.25) :
  given_prob = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_probability_in_dark_l2369_236904


namespace NUMINAMATH_GPT_inequality_proof_l2369_236946

open Real

-- Given conditions
variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1)

-- Goal to prove
theorem inequality_proof : 
  (1 + x) * (1 + y) * (1 + z) ≥ 2 * (1 + (y / x)^(1/3) + (z / y)^(1/3) + (x / z)^(1/3)) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2369_236946


namespace NUMINAMATH_GPT_customerPaidPercentGreater_l2369_236959

-- Definitions for the conditions
def costOfManufacture (C : ℝ) : ℝ := C
def designerPrice (C : ℝ) : ℝ := C * 1.40
def retailerTaxedPrice (C : ℝ) : ℝ := (C * 1.40) * 1.05
def customerInitialPrice (C : ℝ) : ℝ := ((C * 1.40) * 1.05) * 1.10
def customerFinalPrice (C : ℝ) : ℝ := (((C * 1.40) * 1.05) * 1.10) * 0.90

-- The theorem statement
theorem customerPaidPercentGreater (C : ℝ) (hC : 0 < C) : 
    (customerFinalPrice C - costOfManufacture C) / costOfManufacture C * 100 = 45.53 := by 
  sorry

end NUMINAMATH_GPT_customerPaidPercentGreater_l2369_236959


namespace NUMINAMATH_GPT_exists_centrally_symmetric_inscribed_convex_hexagon_l2369_236921

-- Definition of a convex polygon with vertices
def convex_polygon (W : Type) : Prop := sorry

-- Definition of the unit area condition
def has_unit_area (W : Type) : Prop := sorry

-- Definition of being centrally symmetric
def is_centrally_symmetric (V : Type) : Prop := sorry

-- Definition of being inscribed
def is_inscribed_polygon (V W : Type) : Prop := sorry

-- Definition of a convex hexagon
def convex_hexagon (V : Type) : Prop := sorry

-- Main theorem statement
theorem exists_centrally_symmetric_inscribed_convex_hexagon (W : Type) 
  (hW_convex : convex_polygon W) (hW_area : has_unit_area W) : 
  ∃ V : Type, convex_hexagon V ∧ is_centrally_symmetric V ∧ is_inscribed_polygon V W ∧ sorry :=
  sorry

end NUMINAMATH_GPT_exists_centrally_symmetric_inscribed_convex_hexagon_l2369_236921


namespace NUMINAMATH_GPT_f_2007_2007_l2369_236994

def f (n : ℕ) : ℕ :=
  n.digits 10 |>.map (fun d => d * d) |>.sum

def f_k : ℕ → ℕ → ℕ
| 0, n => n
| (k+1), n => f (f_k k n)

theorem f_2007_2007 : f_k 2007 2007 = 145 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_f_2007_2007_l2369_236994


namespace NUMINAMATH_GPT_eval_log32_4_l2369_236908

noncomputable def log_base_change (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem eval_log32_4 : log_base_change 32 4 = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_eval_log32_4_l2369_236908


namespace NUMINAMATH_GPT_senior_tickets_count_l2369_236957

-- Define variables and problem conditions
variables (A S : ℕ)

-- Total number of tickets equation
def total_tickets (A S : ℕ) : Prop := A + S = 510

-- Total receipts equation
def total_receipts (A S : ℕ) : Prop := 21 * A + 15 * S = 8748

-- Prove that the number of senior citizen tickets S is 327
theorem senior_tickets_count (A S : ℕ) (h1 : total_tickets A S) (h2 : total_receipts A S) : S = 327 :=
sorry

end NUMINAMATH_GPT_senior_tickets_count_l2369_236957


namespace NUMINAMATH_GPT_series_sum_is_correct_l2369_236948

noncomputable def series_sum : ℝ := ∑' k, 5^((2 : ℕ)^k) / (25^((2 : ℕ)^k) - 1)

theorem series_sum_is_correct : series_sum = 1 / (Real.sqrt 5 - 1) := 
by
  sorry

end NUMINAMATH_GPT_series_sum_is_correct_l2369_236948


namespace NUMINAMATH_GPT_systematic_sampling_sequence_l2369_236968

theorem systematic_sampling_sequence :
  ∃ (s : Set ℕ), s = {3, 13, 23, 33, 43} ∧
  (∀ n, n ∈ s → n ≤ 50 ∧ ∃ k, k < 5 ∧ n = 3 + k * 10) :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_sequence_l2369_236968


namespace NUMINAMATH_GPT_smallest_repeating_block_length_l2369_236980

-- Define the decimal expansion of 3/11
noncomputable def decimalExpansion : Rational → List Nat :=
  sorry

-- Define the repeating block determination of a given decimal expansion
noncomputable def repeatingBlockLength : List Nat → Nat :=
  sorry

-- Define the fraction 3/11
def frac := (3 : Rat) / 11

-- State the theorem
theorem smallest_repeating_block_length :
  repeatingBlockLength (decimalExpansion frac) = 2 :=
  sorry

end NUMINAMATH_GPT_smallest_repeating_block_length_l2369_236980


namespace NUMINAMATH_GPT_server_processes_21600000_requests_l2369_236902

theorem server_processes_21600000_requests :
  (15000 * 1440 = 21600000) :=
by
  -- Calculations and step-by-step proof
  sorry

end NUMINAMATH_GPT_server_processes_21600000_requests_l2369_236902


namespace NUMINAMATH_GPT_moles_of_water_from_reaction_l2369_236998

def moles_of_water_formed (nh4cl_moles : ℕ) (naoh_moles : ℕ) : ℕ :=
  nh4cl_moles -- Because 1:1 ratio of reactants producing water

theorem moles_of_water_from_reaction :
  moles_of_water_formed 3 3 = 3 := by
  -- Use the condition of the 1:1 reaction ratio derivable from the problem's setup.
  sorry

end NUMINAMATH_GPT_moles_of_water_from_reaction_l2369_236998


namespace NUMINAMATH_GPT_total_cards_in_stack_l2369_236966

theorem total_cards_in_stack (n : ℕ) (H1: 252 ≤ 2 * n) (H2 : (2 * n) % 2 = 0)
                             (H3 : ∀ k : ℕ, k ≤ 2 * n → (if k % 2 = 0 then k / 2 else (k + 1) / 2) * 2 = k) :
  2 * n = 504 := sorry

end NUMINAMATH_GPT_total_cards_in_stack_l2369_236966


namespace NUMINAMATH_GPT_gain_per_year_is_200_l2369_236933

noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem gain_per_year_is_200 :
  let borrowed_principal := 5000
  let borrowing_rate := 4
  let borrowing_time := 2
  let lent_principal := 5000
  let lending_rate := 8
  let lending_time := 2

  let interest_paid := simple_interest borrowed_principal borrowing_rate borrowing_time
  let interest_earned := simple_interest lent_principal lending_rate lending_time

  let total_gain := interest_earned - interest_paid
  let gain_per_year := total_gain / 2

  gain_per_year = 200 := by
  sorry

end NUMINAMATH_GPT_gain_per_year_is_200_l2369_236933


namespace NUMINAMATH_GPT_equation_pattern_l2369_236958
open Nat

theorem equation_pattern (n : ℕ) (h_pos : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 := by
  sorry

end NUMINAMATH_GPT_equation_pattern_l2369_236958


namespace NUMINAMATH_GPT_gain_percent_is_correct_l2369_236967

theorem gain_percent_is_correct :
  let CP : ℝ := 450
  let SP : ℝ := 520
  let gain : ℝ := SP - CP
  let gain_percent : ℝ := (gain / CP) * 100
  gain_percent = 15.56 :=
by
  sorry

end NUMINAMATH_GPT_gain_percent_is_correct_l2369_236967


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2369_236956

theorem solution_set_of_inequality (x : ℝ) (h : x ≠ 2) : 
  ((2 * x) / (x - 2) ≤ 1) ↔ (-2 ≤ x ∧ x < 2) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2369_236956


namespace NUMINAMATH_GPT_fourth_square_area_l2369_236915

theorem fourth_square_area (PQ QR RS QS : ℝ)
  (h1 : PQ^2 = 25)
  (h2 : QR^2 = 49)
  (h3 : RS^2 = 64) :
  QS^2 = 138 :=
by
  sorry

end NUMINAMATH_GPT_fourth_square_area_l2369_236915


namespace NUMINAMATH_GPT_grinder_price_l2369_236962

variable (G : ℝ) (PurchasedMobile : ℝ) (SoldMobile : ℝ) (overallProfit : ℝ)

theorem grinder_price (h1 : PurchasedMobile = 10000)
                      (h2 : SoldMobile = 11000)
                      (h3 : overallProfit = 400)
                      (h4 : 0.96 * G + SoldMobile = G + PurchasedMobile + overallProfit) :
                      G = 15000 := by
  sorry

end NUMINAMATH_GPT_grinder_price_l2369_236962


namespace NUMINAMATH_GPT_factor_polynomial_l2369_236911

theorem factor_polynomial (x y : ℝ) : 
  x^4 + 4 * y^4 = (x^2 - 2 * x * y + 2 * y^2) * (x^2 + 2 * x * y + 2 * y^2) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l2369_236911


namespace NUMINAMATH_GPT_no_stromino_covering_of_5x5_board_l2369_236953

-- Define the conditions
def isStromino (r : ℕ) (c : ℕ) : Prop := 
  (r = 3 ∧ c = 1) ∨ (r = 1 ∧ c = 3)

def is5x5Board (r c : ℕ) : Prop := 
  r = 5 ∧ c = 5

-- The main goal is to show this proposition
theorem no_stromino_covering_of_5x5_board : 
  ∀ (board_size : ℕ × ℕ),
    is5x5Board board_size.1 board_size.2 →
    ∀ (stromino_count : ℕ),
      stromino_count = 16 →
      (∀ (stromino : ℕ × ℕ), 
        isStromino stromino.1 stromino.2 →
        ∀ (cover : ℕ), 
          3 = cover) →
      ¬(∃ (cover_fn : ℕ × ℕ → ℕ), 
          (∀ (pos : ℕ × ℕ), pos.fst < 5 ∧ pos.snd < 5 →
            cover_fn pos = 1 ∨ cover_fn pos = 2) ∧
          (∀ (i : ℕ), i < 25 → 
            ∃ (stromino_pos : ℕ × ℕ), 
              stromino_pos.fst < 5 ∧ stromino_pos.snd < 5 ∧ 
              -- Each stromino must cover exactly 3 squares, 
              -- which implies that the covering function must work appropriately.
              (cover_fn (stromino_pos.fst, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 1, stromino_pos.snd) +
               cover_fn (stromino_pos.fst + 2, stromino_pos.snd) = 3 ∨
               cover_fn (stromino_pos.fst, stromino_pos.snd + 1) +
               cover_fn (stromino_pos.fst, stromino_pos.snd + 2) = 3))) :=
by sorry

end NUMINAMATH_GPT_no_stromino_covering_of_5x5_board_l2369_236953


namespace NUMINAMATH_GPT_brianna_marbles_lost_l2369_236940

theorem brianna_marbles_lost
  (total_marbles : ℕ)
  (remaining_marbles : ℕ)
  (L : ℕ)
  (gave_away : ℕ)
  (dog_ate : ℚ)
  (h1 : total_marbles = 24)
  (h2 : remaining_marbles = 10)
  (h3 : gave_away = 2 * L)
  (h4 : dog_ate = L / 2)
  (h5 : total_marbles - remaining_marbles = L + gave_away + dog_ate) : L = 4 := 
by
  sorry

end NUMINAMATH_GPT_brianna_marbles_lost_l2369_236940


namespace NUMINAMATH_GPT_mark_owes_joanna_l2369_236930

def dollars_per_room : ℚ := 12 / 3
def rooms_cleaned : ℚ := 9 / 4
def total_amount_owed : ℚ := 9

theorem mark_owes_joanna :
  dollars_per_room * rooms_cleaned = total_amount_owed :=
by
  sorry

end NUMINAMATH_GPT_mark_owes_joanna_l2369_236930


namespace NUMINAMATH_GPT_find_R_when_S_7_l2369_236949

-- Define the variables and equations in Lean
variables (R S g : ℕ)

-- The theorem statement based on the given conditions and desired conclusion
theorem find_R_when_S_7 (h1 : R = 2 * g * S + 3) (h2: R = 23) (h3 : S = 5) : (∃ g : ℕ, R = 2 * g * 7 + 3) :=
by {
  -- This part enforces the proof will be handled later
  sorry
}

end NUMINAMATH_GPT_find_R_when_S_7_l2369_236949


namespace NUMINAMATH_GPT_abc_sum_zero_l2369_236961

variable (a b c : ℝ)

-- Conditions given in the original problem
axiom h1 : a + b / c = 1
axiom h2 : b + c / a = 1
axiom h3 : c + a / b = 1

theorem abc_sum_zero : a * b + b * c + c * a = 0 :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_zero_l2369_236961


namespace NUMINAMATH_GPT_smallest_even_number_l2369_236937

theorem smallest_even_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6)) = 140) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_smallest_even_number_l2369_236937


namespace NUMINAMATH_GPT_find_k_and_prove_geometric_sequence_l2369_236945

/-
Given conditions:
1. Sequence sa : ℕ → ℝ with sum sequence S : ℕ → ℝ satisfying the recurrence relation S (n + 1) = (k + 1) * S n + 2
2. Initial terms a_1 = 2 and a_2 = 1
-/

def sequence_sum_relation (S : ℕ → ℝ) (k : ℝ) : Prop :=
∀ n : ℕ, S (n + 1) = (k + 1) * S n + 2

def init_sequence_terms (a : ℕ → ℝ) : Prop :=
a 1 = 2 ∧ a 2 = 1

/-
Proof goal:
1. Prove k = -1/2 given the conditions.
2. Prove sequence a is a geometric sequence with common ratio 1/2 given the conditions.
-/

theorem find_k_and_prove_geometric_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) (k : ℝ) :
  sequence_sum_relation S k →
  init_sequence_terms a →
  (k = (-1:ℝ)/2) ∧ (∀ n: ℕ, n ≥ 1 → a (n+1) = (1/2) * a n) :=
by
  sorry

end NUMINAMATH_GPT_find_k_and_prove_geometric_sequence_l2369_236945


namespace NUMINAMATH_GPT_five_segments_acute_angle_l2369_236954

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_obtuse (a b c : ℝ) : Prop :=
  c^2 > a^2 + b^2

def is_acute (a b c : ℝ) : Prop :=
  c^2 < a^2 + b^2

theorem five_segments_acute_angle (a b c d e : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (T1 : is_triangle a b c) (T2 : is_triangle a b d) (T3 : is_triangle a b e)
  (T4 : is_triangle a c d) (T5 : is_triangle a c e) (T6 : is_triangle a d e)
  (T7 : is_triangle b c d) (T8 : is_triangle b c e) (T9 : is_triangle b d e)
  (T10 : is_triangle c d e) : 
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
           is_triangle x y z ∧ is_acute x y z :=
by
  sorry

end NUMINAMATH_GPT_five_segments_acute_angle_l2369_236954


namespace NUMINAMATH_GPT_fraction_of_7000_l2369_236928

theorem fraction_of_7000 (x : ℝ) 
  (h1 : (1 / 10 / 100) * 7000 = 7) 
  (h2 : x * 7000 - 7 = 700) : 
  x = 0.101 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_7000_l2369_236928


namespace NUMINAMATH_GPT_cards_remaining_l2369_236941

theorem cards_remaining (initial_cards : ℕ) (cards_given : ℕ) (remaining_cards : ℕ) :
  initial_cards = 242 → cards_given = 136 → remaining_cards = initial_cards - cards_given → remaining_cards = 106 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_cards_remaining_l2369_236941


namespace NUMINAMATH_GPT_exponentiation_problem_l2369_236920

theorem exponentiation_problem (a b : ℤ) (h : 3 ^ a * 9 ^ b = (1 / 3 : ℚ)) : a + 2 * b = -1 :=
sorry

end NUMINAMATH_GPT_exponentiation_problem_l2369_236920


namespace NUMINAMATH_GPT_range_of_a_l2369_236943

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 4 * x + a ≥ 0) → a ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2369_236943


namespace NUMINAMATH_GPT_count_3_digit_multiples_of_13_l2369_236917

noncomputable def smallest_3_digit_number : ℕ := 100
noncomputable def largest_3_digit_number : ℕ := 999
noncomputable def divisor : ℕ := 13

theorem count_3_digit_multiples_of_13 : 
  ∃ n : ℕ, n = 69 ∧ (∀ k : ℕ, (k > 0 ∧ 13 * k ≥ smallest_3_digit_number ∧ 13 * k ≤ largest_3_digit_number ↔ k ≥ 9 ∧ k ≤ 76)) :=
sorry

end NUMINAMATH_GPT_count_3_digit_multiples_of_13_l2369_236917


namespace NUMINAMATH_GPT_joan_spent_on_jacket_l2369_236918

def total_spent : ℝ := 42.33
def shorts_spent : ℝ := 15.00
def shirt_spent : ℝ := 12.51
def jacket_spent : ℝ := 14.82

theorem joan_spent_on_jacket :
  total_spent - shorts_spent - shirt_spent = jacket_spent :=
by
  sorry

end NUMINAMATH_GPT_joan_spent_on_jacket_l2369_236918


namespace NUMINAMATH_GPT_base8_9257_digits_product_sum_l2369_236975

theorem base8_9257_digits_product_sum :
  let base10 := 9257
  let base8_digits := [2, 2, 0, 5, 1] -- base 8 representation of 9257
  let product_of_digits := 2 * 2 * 0 * 5 * 1
  let sum_of_digits := 2 + 2 + 0 + 5 + 1
  product_of_digits = 0 ∧ sum_of_digits = 10 := 
by
  sorry

end NUMINAMATH_GPT_base8_9257_digits_product_sum_l2369_236975


namespace NUMINAMATH_GPT_maximum_value_squared_l2369_236901

theorem maximum_value_squared (a b : ℝ) (h₁ : 0 < b) (h₂ : b ≤ a) :
  (∃ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧ a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
  (a / b)^2 ≤ 4 / 3 := 
sorry

end NUMINAMATH_GPT_maximum_value_squared_l2369_236901


namespace NUMINAMATH_GPT_square_area_problem_l2369_236986

theorem square_area_problem 
  (BM : ℝ) 
  (ABCD_is_divided : Prop)
  (hBM : BM = 4)
  (hABCD_is_divided : ABCD_is_divided) : 
  ∃ (side_length : ℝ), side_length * side_length = 144 := 
by
-- We skip the proof part for this task
sorry

end NUMINAMATH_GPT_square_area_problem_l2369_236986


namespace NUMINAMATH_GPT_tony_squat_weight_l2369_236927

-- Definitions from conditions
def curl_weight := 90
def military_press_weight := 2 * curl_weight
def squat_weight := 5 * military_press_weight

-- Theorem statement
theorem tony_squat_weight : squat_weight = 900 := by
  sorry

end NUMINAMATH_GPT_tony_squat_weight_l2369_236927


namespace NUMINAMATH_GPT_darla_total_payment_l2369_236960

-- Define the cost per watt, total watts used, and late fee
def cost_per_watt : ℝ := 4
def total_watts : ℝ := 300
def late_fee : ℝ := 150

-- Define the total cost of electricity
def electricity_cost : ℝ := cost_per_watt * total_watts

-- Define the total amount Darla needs to pay
def total_amount : ℝ := electricity_cost + late_fee

-- The theorem to prove the total amount equals $1350
theorem darla_total_payment : total_amount = 1350 := by
  sorry

end NUMINAMATH_GPT_darla_total_payment_l2369_236960


namespace NUMINAMATH_GPT_inequality_x2_8_over_xy_y2_l2369_236990

open Real

theorem inequality_x2_8_over_xy_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x^2 + 8 / (x * y) + y^2 ≥ 8 := 
sorry

end NUMINAMATH_GPT_inequality_x2_8_over_xy_y2_l2369_236990


namespace NUMINAMATH_GPT_graph_symmetric_about_x_2_l2369_236963

variables {D : Set ℝ} {f : ℝ → ℝ}

theorem graph_symmetric_about_x_2 (h : ∀ x ∈ D, f (x + 1) = f (-x + 3)) : 
  ∀ x ∈ D, f (x) = f (4 - x) :=
by
  sorry

end NUMINAMATH_GPT_graph_symmetric_about_x_2_l2369_236963


namespace NUMINAMATH_GPT_cost_of_one_shirt_l2369_236910

theorem cost_of_one_shirt (J S K : ℕ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 71) (h3 : 3 * J + 2 * S + K = 90) : S = 15 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_one_shirt_l2369_236910


namespace NUMINAMATH_GPT_investment_principal_l2369_236931

theorem investment_principal (A r : ℝ) (n t : ℕ) (P : ℝ) : 
  r = 0.07 → n = 4 → t = 5 → A = 60000 → 
  A = P * (1 + r / n)^(n * t) →
  P = 42409 :=
by
  sorry

end NUMINAMATH_GPT_investment_principal_l2369_236931


namespace NUMINAMATH_GPT_base_of_exponential_function_l2369_236907

theorem base_of_exponential_function (a : ℝ) (h : ∀ x : ℝ, y = a^x) :
  (a > 1 ∧ (a - 1 / a = 1)) ∨ (0 < a ∧ a < 1 ∧ (1 / a - a = 1)) → 
  a = (1 + Real.sqrt 5) / 2 ∨ a = (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_GPT_base_of_exponential_function_l2369_236907


namespace NUMINAMATH_GPT_modulus_of_complex_raised_to_eight_l2369_236925

-- Define the complex number 2 + i in Lean
def z : Complex := Complex.mk 2 1

-- State the proof problem with conditions
theorem modulus_of_complex_raised_to_eight : Complex.abs (z ^ 8) = 625 := by
  sorry

end NUMINAMATH_GPT_modulus_of_complex_raised_to_eight_l2369_236925


namespace NUMINAMATH_GPT_tent_cost_solution_l2369_236989

-- We define the prices of the tents and other relevant conditions.
def tent_costs (m n : ℕ) : Prop :=
  2 * m + 4 * n = 5200 ∧ 3 * m + n = 2800

-- Define the condition for the number of tents and constraints.
def optimal_tent_count (x : ℕ) (w : ℕ) : Prop :=
  x + (20 - x) = 20 ∧ x ≤ (20 - x) / 3 ∧ w = 600 * x + 1000 * (20 - x)

-- The main theorem to be proven in Lean.
theorem tent_cost_solution :
  ∃ m n, tent_costs m n ∧ m = 600 ∧ n = 1000 ∧
  ∃ x, optimal_tent_count x 18000 ∧ x = 5 ∧ (20 - x) = 15 :=
by
  sorry

end NUMINAMATH_GPT_tent_cost_solution_l2369_236989


namespace NUMINAMATH_GPT_max_ways_to_ascend_and_descend_l2369_236951

theorem max_ways_to_ascend_and_descend :
  let east := 2
  let west := 3
  let south := 4
  let north := 1
  let ascend_descend_ways (ascend: ℕ) (n_1 n_2 n_3: ℕ) := ascend * (n_1 + n_2 + n_3)
  (ascend_descend_ways south east west north > ascend_descend_ways east west south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways west east south north) ∧ 
  (ascend_descend_ways south east west north > ascend_descend_ways north east west south) := sorry

end NUMINAMATH_GPT_max_ways_to_ascend_and_descend_l2369_236951


namespace NUMINAMATH_GPT_cos_A_value_l2369_236997

theorem cos_A_value (A B C : ℝ) 
  (A_internal : A + B + C = Real.pi) 
  (cos_B : Real.cos B = 1 / 2)
  (sin_C : Real.sin C = 3 / 5) : 
  Real.cos A = (3 * Real.sqrt 3 - 4) / 10 := 
by
  sorry

end NUMINAMATH_GPT_cos_A_value_l2369_236997


namespace NUMINAMATH_GPT_hexagon_coloring_count_l2369_236977

-- Defining the conditions
def has7Colors : Type := Fin 7

-- Hexagon vertices
inductive Vertex
| A | B | C | D | E | F

-- Adjacent vertices
def adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.B => true
| Vertex.B, Vertex.C => true
| Vertex.C, Vertex.D => true
| Vertex.D, Vertex.E => true
| Vertex.E, Vertex.F => true
| Vertex.F, Vertex.A => true
| _, _ => false

-- Non-adjacent vertices (diagonals)
def non_adjacent : Vertex → Vertex → Prop
| Vertex.A, Vertex.C => true
| Vertex.A, Vertex.D => true
| Vertex.B, Vertex.D => true
| Vertex.B, Vertex.E => true
| Vertex.C, Vertex.E => true
| Vertex.C, Vertex.F => true
| Vertex.D, Vertex.F => true
| Vertex.E, Vertex.A => true
| Vertex.F, Vertex.A => true
| Vertex.F, Vertex.B => true
| _, _ => false

-- Coloring function
def valid_coloring (coloring : Vertex → has7Colors) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2, non_adjacent v1 v2 → coloring v1 ≠ coloring v2)
  ∧ (∀ v1 v2 v3, adjacent v1 v2 → adjacent v2 v3 → adjacent v1 v3 → coloring v1 ≠ coloring v3)

noncomputable def count_valid_colorings : Nat :=
  -- This is a placeholder for the count function
  sorry

theorem hexagon_coloring_count : count_valid_colorings = 21000 := 
  sorry

end NUMINAMATH_GPT_hexagon_coloring_count_l2369_236977


namespace NUMINAMATH_GPT_total_amount_l2369_236905

-- Definitions based on the problem conditions
def jack_amount : ℕ := 26
def ben_amount : ℕ := jack_amount - 9
def eric_amount : ℕ := ben_amount - 10

-- Proof statement
theorem total_amount : jack_amount + ben_amount + eric_amount = 50 :=
by
  -- Sorry serves as a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_total_amount_l2369_236905


namespace NUMINAMATH_GPT_maria_remaining_towels_l2369_236939

-- Define the number of green towels Maria bought
def greenTowels : ℕ := 58

-- Define the number of white towels Maria bought
def whiteTowels : ℕ := 43

-- Define the total number of towels Maria bought
def totalTowels : ℕ := greenTowels + whiteTowels

-- Define the number of towels Maria gave to her mother
def towelsGiven : ℕ := 87

-- Define the resulting number of towels Maria has
def remainingTowels : ℕ := totalTowels - towelsGiven

-- Prove that the remaining number of towels is 14
theorem maria_remaining_towels : remainingTowels = 14 :=
by
  sorry

end NUMINAMATH_GPT_maria_remaining_towels_l2369_236939


namespace NUMINAMATH_GPT_log_product_l2369_236926

theorem log_product : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 2 := by
  sorry

end NUMINAMATH_GPT_log_product_l2369_236926


namespace NUMINAMATH_GPT_boy_scouts_percentage_l2369_236952

variable (S B G : ℝ)

-- Conditions
-- Given B + G = S
axiom condition1 : B + G = S

-- Given 0.75B + 0.625G = 0.7S
axiom condition2 : 0.75 * B + 0.625 * G = 0.7 * S

-- Goal
theorem boy_scouts_percentage : B / S = 0.6 :=
by sorry

end NUMINAMATH_GPT_boy_scouts_percentage_l2369_236952


namespace NUMINAMATH_GPT_cds_per_rack_l2369_236924

theorem cds_per_rack (total_cds : ℕ) (racks_per_shelf : ℕ) (cds_per_rack : ℕ) 
  (h1 : total_cds = 32) 
  (h2 : racks_per_shelf = 4) : 
  cds_per_rack = total_cds / racks_per_shelf :=
by 
  sorry

end NUMINAMATH_GPT_cds_per_rack_l2369_236924


namespace NUMINAMATH_GPT_percentage_students_qualified_school_A_l2369_236991

theorem percentage_students_qualified_school_A 
  (A Q : ℝ)
  (h1 : 1.20 * A = A + 0.20 * A)
  (h2 : 1.50 * Q = Q + 0.50 * Q)
  (h3 : (1.50 * Q / 1.20 * A) * 100 = 87.5) :
  (Q / A) * 100 = 58.33 := sorry

end NUMINAMATH_GPT_percentage_students_qualified_school_A_l2369_236991


namespace NUMINAMATH_GPT_find_c_l2369_236987

theorem find_c (a b c : ℝ) (h1 : a * 2 = 3 * b / 2) (h2 : a * 2 + 9 = c) (h3 : 4 - 3 * b = -c) : 
  c = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2369_236987


namespace NUMINAMATH_GPT_second_offset_l2369_236988

theorem second_offset (d : ℝ) (h1 : ℝ) (A : ℝ) (h2 : ℝ) : 
  d = 28 → h1 = 9 → A = 210 → h2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_second_offset_l2369_236988


namespace NUMINAMATH_GPT_general_term_of_geometric_sequence_l2369_236971

theorem general_term_of_geometric_sequence 
  (positive_terms : ∀ n : ℕ, 0 < a_n) 
  (h1 : a_1 = 1) 
  (h2 : ∃ a : ℕ, a_2 = a + 1 ∧ a_3 = 2 * a + 5) : 
  ∃ q : ℕ, ∀ n : ℕ, a_n = q^(n-1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_of_geometric_sequence_l2369_236971


namespace NUMINAMATH_GPT_luke_payments_difference_l2369_236900

theorem luke_payments_difference :
  let principal := 12000
  let rate := 0.08
  let years := 10
  let n_quarterly := 4
  let n_annually := 1
  let quarterly_rate := rate / n_quarterly
  let annually_rate := rate / n_annually
  let balance_plan1_5years := principal * (1 + quarterly_rate)^(n_quarterly * 5)
  let payment_plan1_5years := balance_plan1_5years / 3
  let remaining_balance_plan1_5years := balance_plan1_5years - payment_plan1_5years
  let final_balance_plan1_10years := remaining_balance_plan1_5years * (1 + quarterly_rate)^(n_quarterly * 5)
  let total_payment_plan1 := payment_plan1_5years + final_balance_plan1_10years
  let final_balance_plan2_10years := principal * (1 + annually_rate)^years
  (total_payment_plan1 - final_balance_plan2_10years).abs = 1022 :=
by
  sorry

end NUMINAMATH_GPT_luke_payments_difference_l2369_236900


namespace NUMINAMATH_GPT_students_taking_neither_580_l2369_236983

noncomputable def numberOfStudentsTakingNeither (total students_m students_a students_d students_ma students_md students_ad students_mad : ℕ) : ℕ :=
  let total_taking_at_least_one := (students_m + students_a + students_d) 
                                - (students_ma + students_md + students_ad) 
                                + students_mad
  total - total_taking_at_least_one

theorem students_taking_neither_580 :
  let total := 800
  let students_m := 140
  let students_a := 90
  let students_d := 75
  let students_ma := 50
  let students_md := 30
  let students_ad := 25
  let students_mad := 20
  numberOfStudentsTakingNeither total students_m students_a students_d students_ma students_md students_ad students_mad = 580 :=
by
  sorry

end NUMINAMATH_GPT_students_taking_neither_580_l2369_236983


namespace NUMINAMATH_GPT_axis_of_symmetry_of_shifted_function_l2369_236942

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry_of_shifted_function :
  (∃ x : ℝ, g x = 1 ∧ x = Real.pi / 12) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_of_shifted_function_l2369_236942


namespace NUMINAMATH_GPT_basic_computer_price_l2369_236984

theorem basic_computer_price (C P : ℝ)
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3)
  : C = 1500 :=
sorry

end NUMINAMATH_GPT_basic_computer_price_l2369_236984


namespace NUMINAMATH_GPT_number_of_parents_who_volunteered_to_bring_refreshments_l2369_236979

theorem number_of_parents_who_volunteered_to_bring_refreshments 
  (total : ℕ) (supervise : ℕ) (supervise_and_refreshments : ℕ) (N : ℕ) (R : ℕ)
  (h_total : total = 84)
  (h_supervise : supervise = 25)
  (h_supervise_and_refreshments : supervise_and_refreshments = 11)
  (h_R_eq_1_5N : R = 3 * N / 2)
  (h_eq : total = (supervise - supervise_and_refreshments) + (R - supervise_and_refreshments) + supervise_and_refreshments + N) :
  R = 42 :=
by
  sorry

end NUMINAMATH_GPT_number_of_parents_who_volunteered_to_bring_refreshments_l2369_236979


namespace NUMINAMATH_GPT_action_figures_per_shelf_l2369_236906

/-- Mike has 64 action figures he wants to display. If each shelf 
    in his room can hold a certain number of figures and he needs 8 
    shelves, prove that each shelf can hold 8 figures. -/
theorem action_figures_per_shelf :
  (64 / 8) = 8 :=
by
  sorry

end NUMINAMATH_GPT_action_figures_per_shelf_l2369_236906


namespace NUMINAMATH_GPT_average_of_first_20_even_numbers_not_divisible_by_3_or_5_l2369_236999

def first_20_valid_even_numbers : List ℕ :=
  [2, 4, 8, 14, 16, 22, 26, 28, 32, 34, 38, 44, 46, 52, 56, 58, 62, 64, 68, 74]

-- Check the sum of these numbers
def sum_first_20_valid_even_numbers : ℕ :=
  first_20_valid_even_numbers.sum

-- Define average calculation
def average_first_20_valid_even_numbers : ℕ :=
  sum_first_20_valid_even_numbers / 20

theorem average_of_first_20_even_numbers_not_divisible_by_3_or_5 :
  average_first_20_valid_even_numbers = 35 :=
by
  sorry

end NUMINAMATH_GPT_average_of_first_20_even_numbers_not_divisible_by_3_or_5_l2369_236999


namespace NUMINAMATH_GPT_curve_representation_l2369_236976

   theorem curve_representation :
     ∀ (x y : ℝ), x^4 - y^4 - 4*x^2 + 4*y^2 = 0 ↔ (x + y = 0 ∨ x - y = 0 ∨ x^2 + y^2 = 4) :=
   by
     sorry
   
end NUMINAMATH_GPT_curve_representation_l2369_236976


namespace NUMINAMATH_GPT_length_of_de_equals_eight_l2369_236938

theorem length_of_de_equals_eight
  (a b c d e : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (bc : c - b = 3 * (d - c))
  (ab : b - a = 5)
  (ac : c - a = 11)
  (ae : e - a = 21) :
  e - d = 8 := by
  sorry

end NUMINAMATH_GPT_length_of_de_equals_eight_l2369_236938


namespace NUMINAMATH_GPT_cost_of_insulation_l2369_236919

def rectangular_tank_dimension_l : ℕ := 6
def rectangular_tank_dimension_w : ℕ := 3
def rectangular_tank_dimension_h : ℕ := 2
def total_cost : ℕ := 1440

def surface_area (l w h : ℕ) : ℕ := 2 * (l * w + l * h + w * h)

def cost_per_square_foot (total_cost surface_area : ℕ) : ℕ := total_cost / surface_area

theorem cost_of_insulation : 
  cost_per_square_foot total_cost (surface_area rectangular_tank_dimension_l rectangular_tank_dimension_w rectangular_tank_dimension_h) = 20 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_insulation_l2369_236919


namespace NUMINAMATH_GPT_Delaney_missed_bus_by_l2369_236932

def time_in_minutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

def Delaney_start_time : ℕ := time_in_minutes 7 50
def bus_departure_time : ℕ := time_in_minutes 8 0
def travel_duration : ℕ := 30

theorem Delaney_missed_bus_by :
  Delaney_start_time + travel_duration - bus_departure_time = 20 :=
by
  sorry

end NUMINAMATH_GPT_Delaney_missed_bus_by_l2369_236932


namespace NUMINAMATH_GPT_triangle_non_existent_l2369_236973

theorem triangle_non_existent (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
    (tangent_condition : (c^2) = 2 * (a^2) + 2 * (b^2)) : False := by
  sorry

end NUMINAMATH_GPT_triangle_non_existent_l2369_236973


namespace NUMINAMATH_GPT_div_gcd_iff_div_ab_gcd_mul_l2369_236969

variable (a b n c : ℕ)
variables (h₀ : a ≠ 0) (d : ℕ)
variable (hd : d = Nat.gcd a b)

theorem div_gcd_iff_div_ab : (n ∣ a ∧ n ∣ b) ↔ n ∣ d :=
by
  sorry

theorem gcd_mul (h₁ : c > 0) : Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by
  sorry

end NUMINAMATH_GPT_div_gcd_iff_div_ab_gcd_mul_l2369_236969
