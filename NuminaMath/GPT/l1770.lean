import Mathlib

namespace transform_equation_l1770_177023

theorem transform_equation (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x - 1)^2 = 3 :=
sorry

end transform_equation_l1770_177023


namespace division_of_fractions_l1770_177029

theorem division_of_fractions : (4 : ℚ) / (5 / 7) = 28 / 5 := sorry

end division_of_fractions_l1770_177029


namespace tan_A_minus_B_l1770_177040

theorem tan_A_minus_B (A B : ℝ) (h1: Real.cos A = -Real.sqrt 2 / 2) (h2 : Real.tan B = 1 / 3) : 
  Real.tan (A - B) = -2 := by
  sorry

end tan_A_minus_B_l1770_177040


namespace min_value_of_function_l1770_177085

theorem min_value_of_function : ∃ x : ℝ, ∀ x : ℝ, x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
by
  sorry

end min_value_of_function_l1770_177085


namespace bruno_pens_l1770_177092

-- Define Bruno's purchase of pens
def one_dozen : Nat := 12
def half_dozen : Nat := one_dozen / 2
def two_and_half_dozens : Nat := 2 * one_dozen + half_dozen

-- State the theorem to be proved
theorem bruno_pens : two_and_half_dozens = 30 :=
by sorry

end bruno_pens_l1770_177092


namespace domain_of_f_l1770_177055

noncomputable def domain_of_function (x : ℝ) : Set ℝ :=
  {x | 4 - x ^ 2 ≥ 0 ∧ x ≠ 1}

theorem domain_of_f (x : ℝ) : domain_of_function x = {x | -2 ≤ x ∧ x < 1 ∨ 1 < x ∧ x ≤ 2} :=
by
  sorry

end domain_of_f_l1770_177055


namespace monotonic_intervals_minimum_m_value_l1770_177011

noncomputable def f (x : ℝ) (a : ℝ) := (2 * Real.exp 1 + 1) * Real.log x - (3 * a / 2) * x + 1

theorem monotonic_intervals (a : ℝ) : 
  if a ≤ 0 then ∀ x ∈ Set.Ioi 0, 0 < (2 * Real.exp 1 + 1) / x - (3 * a / 2) 
  else ∀ x ∈ Set.Ioc 0 ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) > 0 ∧
       ∀ x ∈ Set.Ioi ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) < 0 := sorry

noncomputable def g (x : ℝ) (m : ℝ) := x * Real.exp x + m - ((2 * Real.exp 1 + 1) * Real.log x + x - 1)

theorem minimum_m_value :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 < x → g x m ≥ 0) ↔ m ≥ - Real.exp 1 := sorry

end monotonic_intervals_minimum_m_value_l1770_177011


namespace log2_75_in_terms_of_a_b_l1770_177035

noncomputable def log_base2 (x : ℝ) : ℝ := Real.log x / Real.log 2

variables (a b : ℝ)
variables (log2_9_eq_a : log_base2 9 = a)
variables (log2_5_eq_b : log_base2 5 = b)

theorem log2_75_in_terms_of_a_b : log_base2 75 = (1 / 2) * a + 2 * b :=
by sorry

end log2_75_in_terms_of_a_b_l1770_177035


namespace trapezoid_median_l1770_177049

noncomputable def median_trapezoid (base₁ base₂ height : ℝ) : ℝ :=
(base₁ + base₂) / 2

theorem trapezoid_median (b_t : ℝ) (a_t : ℝ) (h_t : ℝ) (a_tp : ℝ) 
  (h_eq : h_t = 16) (a_eq : a_t = 192) (area_tp_eq : a_tp = a_t) : median_trapezoid h_t h_t h_t = 12 :=
by
  have h_t_eq : h_t = 16 := by sorry
  have a_t_eq : a_t = 192 := by sorry
  have area_tp : a_tp = 192 := by sorry
  sorry

end trapezoid_median_l1770_177049


namespace algebraic_expression_value_l1770_177081

variables (m n x y : ℤ)

def condition1 := m - n = 100
def condition2 := x + y = -1

theorem algebraic_expression_value :
  condition1 m n → condition2 x y → (n + x) - (m - y) = -101 :=
by
  intro h1 h2
  sorry

end algebraic_expression_value_l1770_177081


namespace math_problem_l1770_177080

theorem math_problem (x y : ℝ) 
  (h1 : 1/5 + x + y = 1) 
  (h2 : 1/5 * 1 + 2 * x + 3 * y = 11/5) : 
  (x = 2/5) ∧ 
  (y = 2/5) ∧ 
  (1/5 + x = 3/5) ∧ 
  ((1 - 11/5)^2 * (1/5) + (2 - 11/5)^2 * (2/5) + (3 - 11/5)^2 * (2/5) = 14/25) :=
by {
  sorry
}

end math_problem_l1770_177080


namespace triangle_with_sticks_l1770_177098

theorem triangle_with_sticks (c : ℕ) (h₁ : 4 + 9 > c) (h₂ : 9 - 4 < c) :
  c = 9 :=
by
  sorry

end triangle_with_sticks_l1770_177098


namespace savings_calculation_l1770_177041

noncomputable def calculate_savings (spent_price : ℝ) (saving_pct : ℝ) : ℝ :=
  let original_price := spent_price / (1 - (saving_pct / 100))
  original_price - spent_price

-- Define the spent price and saving percentage
def spent_price : ℝ := 20
def saving_pct : ℝ := 12.087912087912088

-- Statement to be proved
theorem savings_calculation : calculate_savings spent_price saving_pct = 2.75 :=
  sorry

end savings_calculation_l1770_177041


namespace cosine_value_l1770_177026

theorem cosine_value (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos (α + Real.pi / 6) = -1 / 3 :=
by
  sorry

end cosine_value_l1770_177026


namespace binom_15_4_eq_1365_l1770_177056

theorem binom_15_4_eq_1365 : (Nat.choose 15 4) = 1365 := 
by 
  sorry

end binom_15_4_eq_1365_l1770_177056


namespace min_intersection_l1770_177072

open Finset

-- Definition of subset count function
def n (S : Finset ℕ) : ℕ :=
  2 ^ S.card

theorem min_intersection {A B C : Finset ℕ} (hA : A.card = 100) (hB : B.card = 100) 
  (h_subsets : n A + n B + n C = n (A ∪ B ∪ C)) :
  (A ∩ B ∩ C).card ≥ 97 := by
  sorry

end min_intersection_l1770_177072


namespace remainder_when_2n_divided_by_4_l1770_177031

theorem remainder_when_2n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 :=
by
  sorry

end remainder_when_2n_divided_by_4_l1770_177031


namespace money_left_after_purchases_l1770_177091

variable (initial_money : ℝ) (fraction_for_cupcakes : ℝ) (money_spent_on_milkshake : ℝ)

theorem money_left_after_purchases (h_initial : initial_money = 10)
  (h_fraction : fraction_for_cupcakes = 1/5)
  (h_milkshake : money_spent_on_milkshake = 5) :
  initial_money - (initial_money * fraction_for_cupcakes) - money_spent_on_milkshake = 3 := 
by
  sorry

end money_left_after_purchases_l1770_177091


namespace contrapositive_statement_l1770_177066

theorem contrapositive_statement (m : ℝ) : 
  (¬ ∃ (x : ℝ), x^2 + x - m = 0) → m > 0 :=
by
  sorry

end contrapositive_statement_l1770_177066


namespace range_of_m_l1770_177004

theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ y = 0) ∧ 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ x = 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l1770_177004


namespace line_through_intersection_and_origin_l1770_177086

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := 2023 * x - 2022 * y - 1 = 0
def line2 (x y : ℝ) : Prop := 2022 * x + 2023 * y + 1 = 0

-- Define the line passing through the origin
def line_pass_origin (x y : ℝ) : Prop := 4045 * x + y = 0

-- Define the intersection point of the two lines
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define the theorem stating the desired property
theorem line_through_intersection_and_origin (x y : ℝ)
    (h1 : intersection x y)
    (h2 : x = 0 ∧ y = 0) :
    line_pass_origin x y :=
by
    sorry

end line_through_intersection_and_origin_l1770_177086


namespace real_roots_range_real_roots_specific_value_l1770_177067

-- Part 1
theorem real_roots_range (a b m : ℝ) (h_eq : a ≠ 0) (h_discriminant : b^2 - 4 * a * m ≥ 0) :
  m ≤ (b^2) / (4 * a) :=
sorry

-- Part 2
theorem real_roots_specific_value (x1 x2 m : ℝ) (h_sum : x1 + x2 = 4) (h_product : x1 * x2 = m)
  (h_condition : x1^2 + x2^2 + (x1 * x2)^2 = 40) (h_range : m ≤ 4) :
  m = -4 :=
sorry

end real_roots_range_real_roots_specific_value_l1770_177067


namespace cos_sum_identity_l1770_177032

theorem cos_sum_identity (θ : ℝ) (h1 : Real.tan θ = -5 / 12) (h2 : θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  Real.cos (θ + Real.pi / 4) = 17 * Real.sqrt 2 / 26 :=
sorry

end cos_sum_identity_l1770_177032


namespace welders_that_left_first_day_l1770_177008

-- Definitions of conditions
def welders := 12
def days_to_complete_order := 3
def days_remaining_work_after_first_day := 8
def work_done_first_day (r : ℝ) := welders * r * 1
def total_work (r : ℝ) := welders * r * days_to_complete_order

-- Theorem statement
theorem welders_that_left_first_day (r : ℝ) : 
  ∃ x : ℝ, 
    (welders - x) * r * days_remaining_work_after_first_day = total_work r - work_done_first_day r 
    ∧ x = 9 :=
by
  sorry

end welders_that_left_first_day_l1770_177008


namespace relay_race_total_time_l1770_177094

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 0.25 * t1
  let t3 := t2 - 0.20 * t2
  let t4 := t1 + 0.30 * t1
  let t5 := 80
  let t6 := t5 - 0.20 * t5
  let t7 := t5 + 0.15 * t5
  let t8 := t7 - 0.05 * t7
  t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 = 573.65 :=
by
  sorry

end relay_race_total_time_l1770_177094


namespace remainder_when_divided_by_5_l1770_177009

theorem remainder_when_divided_by_5 (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^3 % 5 = 4) : n % 5 = 4 :=
sorry

end remainder_when_divided_by_5_l1770_177009


namespace common_difference_l1770_177063

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

theorem common_difference (d : ℕ) (a1 : ℕ) (h1 : a1 = 18) (h2 : d ≠ 0) 
  (h3 : (a1 + 3 * d)^2 = a1 * (a1 + 7 * d)) : d = 2 :=
by
  sorry

end common_difference_l1770_177063


namespace jane_crayons_l1770_177016

theorem jane_crayons :
  let start := 87
  let eaten := 7
  start - eaten = 80 :=
by
  sorry

end jane_crayons_l1770_177016


namespace greatest_possible_remainder_l1770_177045

theorem greatest_possible_remainder {x : ℤ} (h : ∃ (k : ℤ), x = 11 * k + 10) : 
  ∃ y, y = 10 := sorry

end greatest_possible_remainder_l1770_177045


namespace solve_for_m_l1770_177027

noncomputable def operation (a b c x y : ℝ) := a * x + b * y + c * x * y

theorem solve_for_m (a b c : ℝ) (h1 : operation a b c 1 2 = 3)
                              (h2 : operation a b c 2 3 = 4) 
                              (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), operation a b c x m = x) :
  ∃ (m : ℝ), m = 4 :=
sorry

end solve_for_m_l1770_177027


namespace arithmetic_sequence_common_difference_l1770_177076

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a6 : a 6 = 5) (h_a10 : a 10 = 6) : 
  (a 10 - a 6) / 4 = 1 / 4 := 
by
  sorry

end arithmetic_sequence_common_difference_l1770_177076


namespace graph_passes_through_point_l1770_177074

theorem graph_passes_through_point :
  ∀ (a : ℝ), 0 < a ∧ a < 1 → (∃ (x y : ℝ), (x = 2) ∧ (y = -1) ∧ (y = 2 * a * x - 1)) :=
by
  sorry

end graph_passes_through_point_l1770_177074


namespace roots_distinct_and_real_l1770_177042

variables (b d : ℝ)
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_distinct_and_real (h₁ : discriminant b (-3 * Real.sqrt 5) d = 25) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 :=
by 
  sorry

end roots_distinct_and_real_l1770_177042


namespace smallest_d_l1770_177039

noncomputable def abc_identity_conditions (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3 * d * x^2 + 3 * x + e^3

theorem smallest_d (a b c d e : ℝ) (h : abc_identity_conditions a b c d e) : d = 1 := 
sorry

end smallest_d_l1770_177039


namespace sin2alpha_plus_cosalpha_l1770_177051

theorem sin2alpha_plus_cosalpha (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) + Real.cos α = (4 + Real.sqrt 5) / 5 :=
by
  sorry

end sin2alpha_plus_cosalpha_l1770_177051


namespace triangle_equilateral_if_arithmetic_sequences_l1770_177007

theorem triangle_equilateral_if_arithmetic_sequences
  (A B C : ℝ) (a b c : ℝ)
  (h_angles : A + B + C = 180)
  (h_angle_seq : ∃ (N : ℝ), A = B - N ∧ C = B + N)
  (h_sides : ∃ (n : ℝ), a = b - n ∧ c = b + n) :
  A = B ∧ B = C ∧ a = b ∧ b = c :=
sorry

end triangle_equilateral_if_arithmetic_sequences_l1770_177007


namespace Haley_sweaters_l1770_177073

theorem Haley_sweaters (machine_capacity loads shirts sweaters : ℕ) 
    (h_capacity : machine_capacity = 7)
    (h_loads : loads = 5)
    (h_shirts : shirts = 2)
    (h_sweaters_total : sweaters = loads * machine_capacity - shirts) :
  sweaters = 33 :=
by 
  rw [h_capacity, h_loads, h_shirts] at h_sweaters_total
  exact h_sweaters_total

end Haley_sweaters_l1770_177073


namespace hydrogen_moles_formed_l1770_177020

open Function

-- Define types for the substances involved in the reaction
structure Substance :=
  (name : String)
  (moles : ℕ)

-- Define the reaction
def reaction (NaH H2O NaOH H2 : Substance) : Prop :=
  NaH.moles = H2O.moles ∧ NaOH.moles = H2.moles

-- Given conditions
def NaH_initial : Substance := ⟨"NaH", 2⟩
def H2O_initial : Substance := ⟨"H2O", 2⟩
def NaOH_final : Substance := ⟨"NaOH", 2⟩
def H2_final : Substance := ⟨"H2", 2⟩

-- Problem statement in Lean
theorem hydrogen_moles_formed :
  reaction NaH_initial H2O_initial NaOH_final H2_final → H2_final.moles = 2 :=
by
  -- Skip proof
  sorry

end hydrogen_moles_formed_l1770_177020


namespace min_value_expression_l1770_177058

theorem min_value_expression (x : ℝ) (h : x > 1) : x + 9 / x - 2 ≥ 4 :=
sorry

end min_value_expression_l1770_177058


namespace smallest_integer_proof_l1770_177050

noncomputable def smallestInteger (s : ℝ) (h : s < 1 / 2000) : ℤ :=
  Nat.ceil (Real.sqrt (1999 / 3))

theorem smallest_integer_proof (s : ℝ) (h : s < 1 / 2000) (m : ℤ) (hm : m = (smallestInteger s h + s)^3) : smallestInteger s h = 26 :=
by 
  sorry

end smallest_integer_proof_l1770_177050


namespace probability_Xavier_Yvonne_not_Zelda_l1770_177070

theorem probability_Xavier_Yvonne_not_Zelda
    (P_Xavier : ℚ)
    (P_Yvonne : ℚ)
    (P_Zelda : ℚ)
    (hXavier : P_Xavier = 1/3)
    (hYvonne : P_Yvonne = 1/2)
    (hZelda : P_Zelda = 5/8) :
    (P_Xavier * P_Yvonne * (1 - P_Zelda) = 1/16) :=
  by
  rw [hXavier, hYvonne, hZelda]
  sorry

end probability_Xavier_Yvonne_not_Zelda_l1770_177070


namespace school_seat_payment_l1770_177006

def seat_cost (num_rows : ℕ) (seats_per_row : ℕ) (cost_per_seat : ℕ) (discount : ℕ → ℕ → ℕ) : ℕ :=
  let total_seats := num_rows * seats_per_row
  let total_cost := total_seats * cost_per_seat
  let groups_of_ten := total_seats / 10
  let total_discount := groups_of_ten * discount 10 cost_per_seat
  total_cost - total_discount

-- Define the discount function as 10% of the cost of a group of 10 seats
def discount (group_size : ℕ) (cost_per_seat : ℕ) : ℕ := (group_size * cost_per_seat) / 10

theorem school_seat_payment :
  seat_cost 5 8 30 discount = 1080 :=
sorry

end school_seat_payment_l1770_177006


namespace convert_rect_to_polar_l1770_177024

theorem convert_rect_to_polar (y x : ℝ) (h : y = x) : ∃ θ : ℝ, θ = π / 4 :=
by
  sorry

end convert_rect_to_polar_l1770_177024


namespace max_value_function_l1770_177012

theorem max_value_function (x : ℝ) (h : x < 0) : 
  ∃ y_max, (∀ x', x' < 0 → (x' + 4 / x') ≤ y_max) ∧ y_max = -4 := 
sorry

end max_value_function_l1770_177012


namespace bridge_length_is_235_l1770_177095

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * time_sec
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_is_235 :
  length_of_bridge 140 45 30 = 235 :=
by 
  sorry

end bridge_length_is_235_l1770_177095


namespace sum_of_three_numbers_l1770_177043

theorem sum_of_three_numbers :
  1.35 + 0.123 + 0.321 = 1.794 :=
sorry

end sum_of_three_numbers_l1770_177043


namespace model_x_completion_time_l1770_177068

theorem model_x_completion_time (T_x : ℝ) : 
  (24 : ℕ) * (1 / T_x + 1 / 36) = 1 → T_x = 72 := 
by 
  sorry

end model_x_completion_time_l1770_177068


namespace total_cows_l1770_177088

theorem total_cows (cows_per_herd : Nat) (herds : Nat) (total_cows : Nat) : 
  cows_per_herd = 40 → herds = 8 → total_cows = cows_per_herd * herds → total_cows = 320 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_cows_l1770_177088


namespace base_video_card_cost_l1770_177048

theorem base_video_card_cost
    (cost_computer : ℕ)
    (fraction_monitor_peripherals : ℕ → ℕ → ℕ)
    (twice : ℕ → ℕ)
    (total_spent : ℕ)
    (cost_monitor_peripherals_eq : fraction_monitor_peripherals cost_computer 5 = 300)
    (twice_eq : ∀ x, twice x = 2 * x)
    (eq_total : ∀ (base_video_card : ℕ), cost_computer + fraction_monitor_peripherals cost_computer 5 + twice base_video_card = total_spent)
    : ∃ x, total_spent = 2100 ∧ cost_computer = 1500 ∧ x = 150 :=
by
  sorry

end base_video_card_cost_l1770_177048


namespace find_g1_l1770_177021

open Function

-- Definitions based on the conditions
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem find_g1 (g : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0) 
  (h2 : g (-1) = 1) 
  : g 1 = -3 :=
sorry

end find_g1_l1770_177021


namespace question1_question2_l1770_177054

noncomputable def f (x : ℝ) : ℝ :=
  if x < -4 then -x - 9
  else if x < 1 then 3 * x + 7
  else x + 9

theorem question1 (x : ℝ) (h : -10 ≤ x ∧ x ≤ -2) : f x ≤ 1 := sorry

theorem question2 (x a : ℝ) (hx : x > 1) (h : f x > -x^2 + a * x) : a < 7 := sorry

end question1_question2_l1770_177054


namespace p_at_0_l1770_177064

noncomputable def p : Polynomial ℚ := sorry

theorem p_at_0 :
  (∀ n : ℕ, n ≤ 6 → p.eval (2^n) = 1 / (2^n))
  ∧ p.degree = 6 → 
  p.eval 0 = 127 / 64 :=
sorry

end p_at_0_l1770_177064


namespace terminating_decimal_of_7_div_200_l1770_177038

theorem terminating_decimal_of_7_div_200 : (7 / 200 : ℝ) = 0.028 := sorry

end terminating_decimal_of_7_div_200_l1770_177038


namespace continuous_stripe_probability_l1770_177047

noncomputable def probability_continuous_stripe : ℚ :=
  let total_configurations := 4^6
  let favorable_configurations := 48
  favorable_configurations / total_configurations

theorem continuous_stripe_probability : probability_continuous_stripe = 3 / 256 :=
  by
  sorry

end continuous_stripe_probability_l1770_177047


namespace sequence_general_term_l1770_177059

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, n > 0 → a (n + 1) = a n + 2^n) :
  ∀ n : ℕ, a n = 2^n :=
sorry

end sequence_general_term_l1770_177059


namespace geometric_sequence_sum_l1770_177033

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_ratio_ne_one : q ≠ 1)
  (S : ℕ → ℝ) (h_a1 : S 1 = 1) (h_S4_eq_5S2 : S 4 - 5 * S 2 = 0) :
  S 5 = 31 :=
sorry

end geometric_sequence_sum_l1770_177033


namespace calcium_iodide_weight_l1770_177036

theorem calcium_iodide_weight
  (atomic_weight_Ca : ℝ)
  (atomic_weight_I : ℝ)
  (moles : ℝ) :
  atomic_weight_Ca = 40.08 →
  atomic_weight_I = 126.90 →
  moles = 5 →
  (atomic_weight_Ca + 2 * atomic_weight_I) * moles = 1469.4 :=
by
  intros
  sorry

end calcium_iodide_weight_l1770_177036


namespace simplify_expression_l1770_177096

noncomputable def expr1 := (Real.sqrt 462) / (Real.sqrt 330)
noncomputable def expr2 := (Real.sqrt 245) / (Real.sqrt 175)
noncomputable def expr_simplified := (12 * Real.sqrt 35) / 25

theorem simplify_expression :
  expr1 + expr2 = expr_simplified :=
sorry

end simplify_expression_l1770_177096


namespace total_rowing_campers_l1770_177089

theorem total_rowing_campers (morning_rowing afternoon_rowing : ℕ) : 
  morning_rowing = 13 -> 
  afternoon_rowing = 21 -> 
  morning_rowing + afternoon_rowing = 34 :=
by
  sorry

end total_rowing_campers_l1770_177089


namespace algebraic_expression_value_l1770_177084

theorem algebraic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x + 7 = 8) : 2 * x^2 + 3 * x - 7 = -6 :=
by sorry

end algebraic_expression_value_l1770_177084


namespace solve_quadratic_l1770_177075

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = -2 + Real.sqrt 2) ∧ (x2 = -2 - Real.sqrt 2) ∧ (∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end solve_quadratic_l1770_177075


namespace probability_f_ge1_l1770_177001

noncomputable def f (x: ℝ) : ℝ := 3*x^2 - x - 1

def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def valid_intervals : Set ℝ := { x | -1 ≤ x ∧ x ≤ -2/3 } ∪ { x | 1 ≤ x ∧ x ≤ 2 }

def interval_length (a b : ℝ) : ℝ := b - a

theorem probability_f_ge1 : 
  (interval_length (-2/3) (-1) + interval_length 1 2) / interval_length (-1) 2 = 4 / 9 := 
by
  sorry

end probability_f_ge1_l1770_177001


namespace strictly_positive_integers_equal_l1770_177010

theorem strictly_positive_integers_equal 
  (a b : ℤ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : 
  a = b :=
sorry

end strictly_positive_integers_equal_l1770_177010


namespace constant_term_expanded_eq_neg12_l1770_177025

theorem constant_term_expanded_eq_neg12
  (a w c d : ℤ)
  (h_eq : (a * x + w) * (c * x + d) = 6 * x ^ 2 + x - 12)
  (h_abs_sum : abs a + abs w + abs c + abs d = 12) :
  w * d = -12 := by
  sorry

end constant_term_expanded_eq_neg12_l1770_177025


namespace cost_of_marker_l1770_177083

theorem cost_of_marker (s c m : ℕ) (h1 : s > 12) (h2 : m > 1) (h3 : c > m) (h4 : s * c * m = 924) : c = 11 :=
sorry

end cost_of_marker_l1770_177083


namespace last_two_digits_condition_l1770_177013

-- Define the function to get last two digits of a number
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

-- Given numbers
def n1 := 122
def n2 := 123
def n3 := 125
def n4 := 129

-- The missing number
variable (x : ℕ)

theorem last_two_digits_condition : 
  last_two_digits (last_two_digits n1 * last_two_digits n2 * last_two_digits n3 * last_two_digits n4 * last_two_digits x) = 50 ↔ last_two_digits x = 1 :=
by 
  sorry

end last_two_digits_condition_l1770_177013


namespace maximum_value_x2y_y2z_z2x_l1770_177093

theorem maximum_value_x2y_y2z_z2x (x y z : ℝ) (h_sum : x + y + z = 0) (h_squares : x^2 + y^2 + z^2 = 6) :
  x^2 * y + y^2 * z + z^2 * x ≤ 6 :=
sorry

end maximum_value_x2y_y2z_z2x_l1770_177093


namespace min_female_students_l1770_177097

theorem min_female_students (males females : ℕ) (total : ℕ) (percent_participated : ℕ) (participated : ℕ) (min_females : ℕ)
  (h1 : males = 22) 
  (h2 : females = 18) 
  (h3 : total = males + females)
  (h4 : percent_participated = 60) 
  (h5 : participated = (percent_participated * total) / 100)
  (h6 : min_females = participated - males) :
  min_females = 2 := 
sorry

end min_female_students_l1770_177097


namespace solve_rational_equation_l1770_177030

theorem solve_rational_equation (x : ℝ) (h : x ≠ (2/3)) : 
  (6*x + 4) / (3*x^2 + 6*x - 8) = 3*x / (3*x - 2) ↔ x = -4/3 ∨ x = 3 :=
sorry

end solve_rational_equation_l1770_177030


namespace units_digit_product_l1770_177060

theorem units_digit_product (k l : ℕ) (h1 : ∀ n : ℕ, (5^n % 10) = 5) (h2 : ∀ m < 4, (6^m % 10) = 6) :
  ((5^k * 6^l) % 10) = 0 :=
by
  have h5 : (5^k % 10) = 5 := h1 k
  have h6 : (6^4 % 10) = 6 := h2 4 (by sorry)
  have h_product : (5^k * 6^l % 10) = ((5 % 10) * (6 % 10) % 10) := sorry
  norm_num at h_product
  exact h_product

end units_digit_product_l1770_177060


namespace identify_wise_l1770_177079

def total_people : ℕ := 30

def is_wise (p : ℕ) : Prop := True   -- This can be further detailed to specify wise characteristics
def is_fool (p : ℕ) : Prop := True    -- This can be further detailed to specify fool characteristics

def wise_count (w : ℕ) : Prop := True -- This indicates the count of wise people
def fool_count (f : ℕ) : Prop := True -- This indicates the count of fool people

def sum_of_groups (wise_groups fool_groups : ℕ) : Prop :=
  wise_groups + fool_groups = total_people

def sum_of_fools (fool_groups : ℕ) (F : ℕ) : Prop :=
  fool_groups = F

theorem identify_wise (F : ℕ) (h1 : F ≤ 8) :
  ∃ (wise_person : ℕ), (wise_person < 30 ∧ is_wise wise_person) :=
by
  sorry

end identify_wise_l1770_177079


namespace probability_not_red_light_l1770_177018

theorem probability_not_red_light :
  ∀ (red_light yellow_light green_light : ℕ),
    red_light = 30 →
    yellow_light = 5 →
    green_light = 40 →
    (yellow_light + green_light) / (red_light + yellow_light + green_light) = (3 : ℚ) / 5 :=
by intros red_light yellow_light green_light h_red h_yellow h_green
   sorry

end probability_not_red_light_l1770_177018


namespace sin_double_angle_value_l1770_177000

theorem sin_double_angle_value 
  (α : ℝ) 
  (hα1 : π / 2 < α) 
  (hα2 : α < π)
  (h : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = - 17 / 18 := 
by
  sorry

end sin_double_angle_value_l1770_177000


namespace quadratic_has_two_distinct_real_roots_l1770_177082

theorem quadratic_has_two_distinct_real_roots (k : ℝ) (h1 : 4 + 4 * k > 0) (h2 : k ≠ 0) :
  k > -1 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l1770_177082


namespace beads_per_necklace_l1770_177069

-- Definitions based on conditions
def total_beads_used (N : ℕ) : ℕ :=
  10 * N + 2 * N + 50 + 35

-- Main theorem to prove the number of beads needed for one beaded necklace
theorem beads_per_necklace (N : ℕ) (h : total_beads_used N = 325) : N = 20 :=
by
  sorry

end beads_per_necklace_l1770_177069


namespace max_watches_two_hours_l1770_177090

noncomputable def show_watched_each_day : ℕ := 30 -- Time in minutes
def days_watched : ℕ := 4 -- Monday to Thursday

theorem max_watches_two_hours :
  (days_watched * show_watched_each_day) / 60 = 2 := by
  sorry

end max_watches_two_hours_l1770_177090


namespace loss_percentage_l1770_177078

theorem loss_percentage (C : ℝ) (h : 40 * C = 100 * C) : 
  ∃ L : ℝ, L = 60 := 
sorry

end loss_percentage_l1770_177078


namespace Stuart_reward_points_l1770_177046

theorem Stuart_reward_points (reward_points_per_unit : ℝ) (spending : ℝ) (unit_amount : ℝ) : 
  reward_points_per_unit = 5 → 
  spending = 200 → 
  unit_amount = 25 → 
  (spending / unit_amount) * reward_points_per_unit = 40 :=
by 
  intros h_points h_spending h_unit
  sorry

end Stuart_reward_points_l1770_177046


namespace shaded_solid_volume_l1770_177061

noncomputable def volume_rectangular_prism (length width height : ℕ) : ℕ :=
  length * width * height

theorem shaded_solid_volume :
  volume_rectangular_prism 4 5 6 - volume_rectangular_prism 1 2 4 = 112 :=
by
  sorry

end shaded_solid_volume_l1770_177061


namespace find_m_n_l1770_177087

theorem find_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 := 
by {
  sorry
}

end find_m_n_l1770_177087


namespace find_g_inv_84_l1770_177005

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem find_g_inv_84 : g 3 = 84 → ∃ x, g x = 84 ∧ x = 3 :=
by
  sorry

end find_g_inv_84_l1770_177005


namespace solve_for_w_l1770_177037

theorem solve_for_w (w : ℂ) (i : ℂ) (i_squared : i^2 = -1) 
  (h : 3 - i * w = 1 + 2 * i * w) : 
  w = -2 * i / 3 := 
sorry

end solve_for_w_l1770_177037


namespace abs_inequality_proof_by_contradiction_l1770_177034

theorem abs_inequality_proof_by_contradiction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  |a| > |b| :=
by
  let h := |a| ≤ |b|
  sorry

end abs_inequality_proof_by_contradiction_l1770_177034


namespace matrix_cube_l1770_177003

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l1770_177003


namespace estimate_3_sqrt_2_range_l1770_177028

theorem estimate_3_sqrt_2_range :
  4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 :=
by
  sorry

end estimate_3_sqrt_2_range_l1770_177028


namespace slope_of_chord_in_ellipse_l1770_177077

noncomputable def slope_of_chord (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y1 - y2) / (x1 - x2)

theorem slope_of_chord_in_ellipse :
  ∀ (x1 y1 x2 y2 : ℝ),
    (x1^2 / 16 + y1^2 / 9 = 1) →
    (x2^2 / 16 + y2^2 / 9 = 1) →
    ((x1 + x2) = -2) →
    ((y1 + y2) = 4) →
    slope_of_chord x1 y1 x2 y2 = 9 / 32 :=
by
  intro x1 y1 x2 y2 h1 h2 h3 h4
  sorry

end slope_of_chord_in_ellipse_l1770_177077


namespace correct_transformation_l1770_177044

theorem correct_transformation (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end correct_transformation_l1770_177044


namespace samantha_birth_year_l1770_177019

theorem samantha_birth_year :
  ∀ (first_amc : ℕ) (amc9_year : ℕ) (samantha_age_in_amc9 : ℕ),
  (first_amc = 1983) →
  (amc9_year = first_amc + 8) →
  (samantha_age_in_amc9 = 13) →
  (amc9_year - samantha_age_in_amc9 = 1978) :=
by
  intros first_amc amc9_year samantha_age_in_amc9 h1 h2 h3
  sorry

end samantha_birth_year_l1770_177019


namespace find_12th_term_l1770_177002

noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ
| 0 => a
| (n+1) => r * geometric_sequence a r n

theorem find_12th_term : ∃ a r, geometric_sequence a r 4 = 5 ∧ geometric_sequence a r 7 = 40 ∧ geometric_sequence a r 11 = 640 :=
by
  -- statement only, no proof provided
  sorry

end find_12th_term_l1770_177002


namespace part_a_l1770_177053

theorem part_a (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) : 
  |a - b| + |b - c| + |c - a| ≤ 2 * Real.sqrt 2 :=
sorry

end part_a_l1770_177053


namespace relationship_between_a_and_b_l1770_177022

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : Real.exp a + 2 * a = Real.exp b + 3 * b) : 
  a > b :=
sorry

end relationship_between_a_and_b_l1770_177022


namespace opposite_of_2023_l1770_177052

-- Statements for the condition and question
theorem opposite_of_2023 (x : ℤ) (h : 2023 + x = 0) : x = -2023 :=
sorry

end opposite_of_2023_l1770_177052


namespace min_percentage_of_people_owning_95_percent_money_l1770_177062

theorem min_percentage_of_people_owning_95_percent_money 
  (total_people: ℕ) (total_money: ℕ) 
  (P: ℕ) (M: ℕ) 
  (H1: P = total_people * 10 / 100) 
  (H2: M = total_money * 90 / 100)
  (H3: ∀ (people_owning_90_percent: ℕ), people_owning_90_percent = P → people_owning_90_percent * some_money = M) :
      P = total_people * 55 / 100 := 
sorry

end min_percentage_of_people_owning_95_percent_money_l1770_177062


namespace simplify_and_evaluate_expression_l1770_177015

variables (a b : ℚ)

theorem simplify_and_evaluate_expression : 
  (4 * (a^2 - 2 * a * b) - (3 * a^2 - 5 * a * b + 1)) = 5 :=
by
  let a := -2
  let b := (1 : ℚ) / 3
  sorry

end simplify_and_evaluate_expression_l1770_177015


namespace min_fraction_value_l1770_177017

-- Define the conditions: geometric sequence, specific term relationship, product of terms

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

def specific_term_relationship (a : ℕ → ℝ) : Prop :=
  a 3 = a 2 + 2 * a 1

def product_of_terms (a : ℕ → ℝ) (m n : ℕ) : Prop :=
  a m * a n = 64 * (a 1)^2

def min_value_fraction (m n : ℕ) : Prop :=
  1 / m + 9 / n = 2

theorem min_fraction_value (a : ℕ → ℝ) (m n : ℕ)
  (h1 : geometric_sequence a)
  (h2 : specific_term_relationship a)
  (h3 : product_of_terms a m n)
  : min_value_fraction m n := by
  sorry

end min_fraction_value_l1770_177017


namespace gardener_b_time_l1770_177099

theorem gardener_b_time :
  ∃ x : ℝ, (1 / 3 + 1 / x = 1 / 1.875) → (x = 5) := by
  sorry

end gardener_b_time_l1770_177099


namespace combined_degrees_l1770_177057

variable (Summer_deg Jolly_deg : ℕ)

def Summer_has_150_degrees := Summer_deg = 150

def Summer_has_5_more_degrees_than_Jolly := Summer_deg = Jolly_deg + 5

theorem combined_degrees (h1 : Summer_has_150_degrees Summer_deg) (h2 : Summer_has_5_more_degrees_than_Jolly Summer_deg Jolly_deg) :
  Summer_deg + Jolly_deg = 295 :=
by
  sorry

end combined_degrees_l1770_177057


namespace sandwiches_count_l1770_177065

theorem sandwiches_count (M : ℕ) (C : ℕ) (S : ℕ) (hM : M = 12) (hC : C = 12) (hS : S = 5) :
  M * (C * (C - 1) / 2) * S = 3960 := 
  by sorry

end sandwiches_count_l1770_177065


namespace lines_intersect_l1770_177071

theorem lines_intersect (m b : ℝ) (h1 : 17 = 2 * m * 4 + 5) (h2 : 17 = 4 * 4 + b) : b + m = 2.5 :=
by {
    sorry
}

end lines_intersect_l1770_177071


namespace problem_solution_l1770_177014

theorem problem_solution (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by 
  sorry

end problem_solution_l1770_177014
