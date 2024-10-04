import Mathlib
import Mathlib.Algebra.Arith
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parity
import Mathlib.Analysis.SpecialFunctions.Complex.Logarithm
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Combinatorics.Perm
import Mathlib.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Noncomputable
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Init.Data.Int.Default
import Mathlib.Order.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import Mathlib.Topology.Bounds

namespace probability_of_sum_multiple_of_4_l737_737660

noncomputable def prob_sum_multiple_of_4 : ℚ := 
  let n_success := (3 : ℚ) + 5 + 1
  let n_total := (6 : ℚ) * 6
  n_success / n_total

theorem probability_of_sum_multiple_of_4 (success_count total_count : ℕ) 
  (H_success: success_count = 9) (H_total: total_count = 36) 
  (p : ℚ := (success_count : ℚ) / (total_count : ℚ)) :
  p = 1/4 :=
by 
  simp [H_success, H_total, p, prob_sum_multiple_of_4]
  sorry

end probability_of_sum_multiple_of_4_l737_737660


namespace masha_clay_pieces_l737_737178

theorem masha_clay_pieces :
  let red_init := 4
      blue_init := 3
      yellow_init := 5
      blue_after_first_division := blue_init * 2
      yellow_after_first_division := yellow_init * 2
      red_after_second_division := red_init * 2
      blue_after_second_division := blue_after_first_division * 2
      yellow_after_second_division := yellow_after_first_division
  in
  red_after_second_division + blue_after_second_division + yellow_after_second_division = 30 :=
by
  let red_init := 4
  let blue_init := 3
  let yellow_init := 5
  let blue_after_first_division := blue_init * 2
  let yellow_after_first_division := yellow_init * 2
  let red_after_second_division := red_init * 2
  let blue_after_second_division := blue_after_first_division * 2
  let yellow_after_second_division := yellow_after_first_division
  show red_after_second_division + blue_after_second_division + yellow_after_second_division = 30
  sorry

end masha_clay_pieces_l737_737178


namespace correct_statement_D_l737_737681

theorem correct_statement_D :
  let number_of_red_balls := 3
  let number_of_black_balls := 4
  let total_balls := 7
  let probability_red_ball := number_of_red_balls / total_balls
  let number_of_students := 3200
  let sampled_students := 200
  let sampled_jumprope_enthusiasts := 85
  let percentage_jumprope := (sampled_jumprope_enthusiasts / sampled_students) * 100
  let estimated_jumprope := number_of_students * (percentage_jumprope / 100)
  in estimated_jumprope = 1360 :=
by
  sorry

end correct_statement_D_l737_737681


namespace sum_of_squares_of_solutions_l737_737430

theorem sum_of_squares_of_solutions :
  ∑ x in ({x : ℝ | |x^2 - x - 1/402| = 1/201}).toFinset, x^2 = 26500/13467 :=
by
  sorry

end sum_of_squares_of_solutions_l737_737430


namespace floor_sqrt_80_l737_737877

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737877


namespace effective_percent_increase_l737_737010

variable (P : ℝ) -- Let's use a real number for prices

def final_price_after_discounts (P : ℝ) : ℝ := 
  (P * 0.8) * 0.9

def price_with_tax (P : ℝ) : ℝ := 
  P * 1.08

def percent_increase (P : ℝ) : ℝ  := 
  let final_price := final_price_after_discounts P
  let price_with_tax := price_with_tax P
  ((price_with_tax - final_price) / final_price) * 100

theorem effective_percent_increase (P : ℝ) : 
  percent_increase P = 50 := 
by
  sorry

end effective_percent_increase_l737_737010


namespace largest_multiple_of_7_negation_gt_neg150_l737_737273

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l737_737273


namespace find_A_l737_737435

variable {a b : ℝ}

theorem find_A (h : (5 * a + 3 * b)^2 = (5 * a - 3 * b)^2 + A) : A = 60 * a * b :=
sorry

end find_A_l737_737435


namespace at_least_one_heart_or_king_l737_737341

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l737_737341


namespace max_value_of_sums_l737_737568

noncomputable def max_of_sums (a b c d : ℝ) : ℝ :=
  a^4 + b^4 + c^4 + d^4

theorem max_value_of_sums (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  max_of_sums a b c d ≤ 16 :=
sorry

end max_value_of_sums_l737_737568


namespace floor_of_sqrt_80_l737_737923

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737923


namespace cost_per_mile_l737_737324

theorem cost_per_mile (x : ℝ) (daily_fee : ℝ) (daily_budget : ℝ) (max_miles : ℝ)
  (h1 : daily_fee = 50)
  (h2 : daily_budget = 88)
  (h3 : max_miles = 190)
  (h4 : daily_budget = daily_fee + x * max_miles) :
  x = 0.20 :=
by
  sorry

end cost_per_mile_l737_737324


namespace shepherd_flock_l737_737734

theorem shepherd_flock (x y : ℕ) (h1 : (x - 1) * 5 = 7 * y) (h2 : x * 3 = 5 * (y - 1)) :
  x + y = 25 :=
sorry

end shepherd_flock_l737_737734


namespace positive_integer_pairs_l737_737422

theorem positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  a^b = b^(a^2) ↔ (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27) :=
by sorry

end positive_integer_pairs_l737_737422


namespace square_of_negative_is_positive_l737_737121

-- Define P as a negative integer
variable (P : ℤ) (hP : P < 0)

-- Theorem statement that P² is always positive.
theorem square_of_negative_is_positive : P^2 > 0 :=
sorry

end square_of_negative_is_positive_l737_737121


namespace triangles_area_ratio_l737_737656

noncomputable def ratio_area_triangles 
  (A B : ℕ → ℝ × ℝ) 
  (angle_30 : ∀ i, ∠(0, 0) (A i) (B i) = 30)
  (angle_120 : ∀ i, ∠(A i) (0, 0) (B i) = 120)
  (right_angle : ∀ i, ∠(0, 0) (B i) (A (i+1)) = 90) 
  : ℚ :=
1 / (2 ^ 94)

theorem triangles_area_ratio 
  (A B : ℕ → ℝ × ℝ)
  (angle_30 : ∀ i, ∠(0, 0) (A i) (B i) = 30)
  (angle_120 : ∀ i, ∠(A i) (0, 0) (B i) = 120)
  (right_angle : ∀ i, ∠(0, 0) (B i) (A (i+1)) = 90) 
  : ratio_area_triangles A B angle_30 angle_120 right_angle = 1 / (2 ^ 94) :=
sorry

end triangles_area_ratio_l737_737656


namespace sphere_radius_same_volume_as_cone_l737_737708

theorem sphere_radius_same_volume_as_cone :
  let r_c : ℝ := 2
  let h_c : ℝ := 3
  let V_cone := (1/3) * π * (r_c ^ 2) * h_c
  let r_s := (3.sqrt) // cube root of 3
  let V_sphere := (4/3) * π * (r_s ^ 3)
  V_cone = V_sphere :=
by
  let r_c : ℝ := 2
  let h_c : ℝ := 3
  have V_cone : ℝ := (1/3) * π * (r_c ^ 2) * h_c
  have r_s : ℝ := real.cbrt 3
  have V_sphere : ℝ := (4/3) * π * (r_s ^ 3)
  show V_cone = V_sphere
  sorry

end sphere_radius_same_volume_as_cone_l737_737708


namespace ratio_of_boys_to_girls_simplify_ratio_ratio_of_boys_to_girls_simplified_l737_737522

def number_of_girls := 210
def total_students := 546

def number_of_boys (G : ℕ) (T : ℕ) := T - G

theorem ratio_of_boys_to_girls (G := number_of_girls) (T := total_students) :
  (number_of_boys G T) / G = 336 / 210 := 
  sorry

theorem simplify_ratio :
  336 / 210 = 8 / 5 := 
  sorry

theorem ratio_of_boys_to_girls_simplified (G := number_of_girls) (T := total_students) :
  (number_of_boys G T) / G = 8 / 5 :=
by {
  apply Eq.trans,
  apply ratio_of_boys_to_girls,
  apply simplify_ratio,
}

end ratio_of_boys_to_girls_simplify_ratio_ratio_of_boys_to_girls_simplified_l737_737522


namespace sphere_radius_same_volume_as_cone_l737_737707

theorem sphere_radius_same_volume_as_cone :
  let r_c : ℝ := 2
  let h_c : ℝ := 3
  let V_cone := (1/3) * π * (r_c ^ 2) * h_c
  let r_s := (3.sqrt) // cube root of 3
  let V_sphere := (4/3) * π * (r_s ^ 3)
  V_cone = V_sphere :=
by
  let r_c : ℝ := 2
  let h_c : ℝ := 3
  have V_cone : ℝ := (1/3) * π * (r_c ^ 2) * h_c
  have r_s : ℝ := real.cbrt 3
  have V_sphere : ℝ := (4/3) * π * (r_s ^ 3)
  show V_cone = V_sphere
  sorry

end sphere_radius_same_volume_as_cone_l737_737707


namespace largest_neg_multiple_of_7_greater_than_neg_150_l737_737260

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l737_737260


namespace probability_heart_or_king_l737_737330

theorem probability_heart_or_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let overlap := 1
  let unique_hearts_or_kings := hearts + kings - overlap
  let non_hearts_or_kings := total_cards - unique_hearts_or_kings
  let p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  let p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  let p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  p_at_least_one_heart_or_king = 88 / 169 :=
by
  have total_cards := 52
  have hearts := 13
  have kings := 4
  have overlap := 1
  have unique_hearts_or_kings := hearts + kings - overlap
  have non_hearts_or_kings := total_cards - unique_hearts_or_kings
  have p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  have p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  have p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  show p_at_least_one_heart_or_king = 88 / 169
  sorry

end probability_heart_or_king_l737_737330


namespace new_volume_of_balloon_l737_737698

def initial_volume : ℝ := 2.00  -- Initial volume in liters
def initial_pressure : ℝ := 745  -- Initial pressure in mmHg
def initial_temperature : ℝ := 293.15  -- Initial temperature in Kelvin
def final_pressure : ℝ := 700  -- Final pressure in mmHg
def final_temperature : ℝ := 283.15  -- Final temperature in Kelvin
def final_volume : ℝ := 2.06  -- Expected final volume in liters

theorem new_volume_of_balloon :
  (initial_pressure * initial_volume / initial_temperature) = (final_pressure * final_volume / final_temperature) :=
  sorry  -- Proof to be filled in later

end new_volume_of_balloon_l737_737698


namespace odd_function_D_l737_737678

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def f_A (x : ℝ) : ℝ := x + 1
def f_B (x : ℝ) : ℝ := 3 * x^2 - 1
def f_C (x : ℝ) : ℝ := 2 * (x + 1)^3 - 1
def f_D (x : ℝ) : ℝ := -4 / x

theorem odd_function_D : is_odd_function f_D :=
by
  intro x
  sorry

end odd_function_D_l737_737678


namespace max_value_of_expression_l737_737122

theorem max_value_of_expression (x y : ℝ) 
  (h : (x - 4)^2 / 4 + y^2 / 9 = 1) : 
  (x^2 / 4 + y^2 / 9 ≤ 9) ∧ ∃ x y, (x - 4)^2 / 4 + y^2 / 9 = 1 ∧ x^2 / 4 + y^2 / 9 = 9 :=
by
  sorry

end max_value_of_expression_l737_737122


namespace birthday_money_l737_737192

theorem birthday_money (x : ℤ) (h₀ : 16 + x - 25 = 19) : x = 28 :=
by
  sorry

end birthday_money_l737_737192


namespace largest_multiple_of_7_negation_greater_than_neg_150_l737_737280

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l737_737280


namespace complement_union_of_M_N_l737_737090

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def M := {1, 3, 5, 7}
def N := {5, 6, 7}

theorem complement_union_of_M_N : (U \ (M ∪ N)) = {2, 4, 8} := by
  sorry

end complement_union_of_M_N_l737_737090


namespace floor_sqrt_80_l737_737933

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737933


namespace num_real_a_has_integer_roots_l737_737041

theorem num_real_a_has_integer_roots : 
  (∃ a : ℝ, ∀ x ∈ ℝ, (x^2 + a * x + 12 * a = 0) → (∃ p q : ℤ, x = p ∧ x = q ∧ (p + q = -a) ∧ (p * q = 12 * a))) → 8 := sorry

end num_real_a_has_integer_roots_l737_737041


namespace value_of_y_l737_737110

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l737_737110


namespace Erica_Ice_Cream_Spend_l737_737033

theorem Erica_Ice_Cream_Spend :
  (6 * ((3 * 2.00) + (2 * 1.50) + (2 * 3.00))) = 90 := sorry

end Erica_Ice_Cream_Spend_l737_737033


namespace common_ratio_of_geometric_sequence_l737_737532

-- Define positive geometric sequence a_n with common ratio q
def geometric_sequence (a q : ℝ) (n : ℕ) : ℝ := a * q^n

-- Define the relevant conditions
variable {a q : ℝ}
variable (h1 : a * q^4 + 2 * a * q^2 * q^6 + a * q^4 * q^8 = 16)
variable (h2 : (a * q^4 + a * q^8) / 2 = 4)
variable (pos_q : q > 0)

-- Define the goal: proving the common ratio q is sqrt(2)
theorem common_ratio_of_geometric_sequence : q = Real.sqrt 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l737_737532


namespace initial_number_of_girls_l737_737616

theorem initial_number_of_girls (n : ℕ) (A : ℝ) 
  (h1 : (n + 1) * (A + 3) - 70 = n * A + 94) :
  n = 8 :=
by {
  sorry
}

end initial_number_of_girls_l737_737616


namespace magnitude_projection_eq_2_l737_737552

variables (u z : EuclideanSpace ℝ (Fin 3))
variables (h1 : inner u z = 6) (h2 : ∥z∥ = 3)

theorem magnitude_projection_eq_2 : ∥projection z u∥ = 2 :=
by sorry

end magnitude_projection_eq_2_l737_737552


namespace prove_angle_APC_l737_737524

-- Assuming the angles and midpoints
variable {A B C D M N P : Type}
variables {angleDAB angleABC angleBCD angleAPC : ℝ}
variables (isMidpoint_M : M = (A + B) / 2) (isMidpoint_N : N = (C + D) / 2)
variables (AP_eq_CP : AP = CP)
variables (ratio_MP_PN : AM / CN = MP / PN)

-- The given conditions
def given_conditions : Prop :=
  angleDAB = 110 ∧ angleABC = 50 ∧ angleBCD = 70 ∧
  isMidpoint_M ∧ isMidpoint_N ∧ AP_eq_CP ∧ ratio_MP_PN

-- The angle we need to prove
def target_angle : ℝ := 120

-- The theorem statement
theorem prove_angle_APC :
  given_conditions → angleAPC = target_angle :=
begin
  sorry
end

end prove_angle_APC_l737_737524


namespace sqrt_floor_eight_l737_737858

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737858


namespace largest_multiple_of_7_neg_greater_than_neg_150_l737_737282

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l737_737282


namespace largest_neg_multiple_of_7_greater_than_neg_150_l737_737259

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l737_737259


namespace floor_sqrt_80_l737_737944

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737944


namespace B_correct_C_correct_D_correct_l737_737082

-- Definitions from conditions in part a.
def parabola_equation (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x
def focus_position {p : ℝ} (p_pos : p > 0) : (ℝ × ℝ) := (p / 2, 0)
def distance_from_directrix_to_focus {p : ℝ} (p_pos : p > 0) : ℝ := 2

-- Define points and distances mentioned in part c.
def on_parabola {p : ℝ} (p_pos : p > 0) (x y : ℝ) : Prop := parabola_equation p x y
def point_distance (P Q : (ℝ × ℝ)) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2 |> real.sqrt
def midpoint (A B : (ℝ × ℝ)) : (ℝ × ℝ) := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Theorems from correct options identified in part b.
theorem B_correct (p : ℝ) (p_pos : p > 0) (A B : ℝ × ℝ) (F : ℝ × ℝ) (FM_x_dist : ℝ)
  (AB_len : point_distance A B = 8) : 
  let M := midpoint A B in
  FM_x_dist = 3 :=
sorry

theorem C_correct (P Q F : ℝ × ℝ) (four_times_Q : (4 * Q.1, 4 * Q.2) = P)
  : point_distance F P = 6 :=
sorry

theorem D_correct (A F B : (ℝ × ℝ)) (prop_focal_chord : (1 / point_distance A F) + (1 / point_distance B F) = 1)
  : 9 * point_distance A F + point_distance B F >= 16 :=
sorry

end B_correct_C_correct_D_correct_l737_737082


namespace min_moves_to_checkerboard_l737_737651

noncomputable def minimum_moves_checkerboard (n : ℕ) : ℕ :=
if n = 6 then 18
else 0

theorem min_moves_to_checkerboard :
  minimum_moves_checkerboard 6 = 18 :=
by sorry

end min_moves_to_checkerboard_l737_737651


namespace maximize_profit_at_57_point_5_yuan_l737_737299

theorem maximize_profit_at_57_point_5_yuan :
  ∃ x : ℝ, (∀ y : ℝ, profit y ≤ profit 57.5) ∧ x = 57.5 :=
by
  let profit (x : ℝ) := (x - 40) * (500 - 20 * (x - 50))
  have h1 : profit = λ x, -20 * (x - 40) * (x - 75)
  have h2 : ∀ x, profit x = -20 * ((x - 57.5)^2 + 6125)
  use 57.5
  split
  · intros y
    rw h2
    rw h2 y
    linarith
  · refl

end maximize_profit_at_57_point_5_yuan_l737_737299


namespace roots_in_interval_l737_737642

noncomputable def region_A := {pq | p^2 - 4 * q < 0}
noncomputable def region_B := {pq | p^2 - 4 * q > 0 ∧ -2 * p + q + 4 > 0 ∧ p > 4}
noncomputable def region_C := {pq | p^2 - 4 * q > 0 ∧ -2 * p + q + 4 < 0 ∧ p + q + 1 > 0}
noncomputable def region_D := {pq | p^2 - 4 * q > 0 ∧ -2 * p + q + 4 < 0 ∧ p + q + 1 < 0}
noncomputable def region_E := {pq | p^2 - 4 * q > 0 ∧ -2 * p + q + 4 > 0 ∧ p + q + 1 < 0}
noncomputable def region_F := {pq | p^2 - 4 * q > 0 ∧ -2 * p + q + 4 > 0 ∧ p < -2}
noncomputable def region_G := {pq | p^2 - 4 * q > 0 ∧ -2 * p + q + 4 > 0 ∧ p + q + 1 > 0 ∧ p < 4 ∧ -2 > -p/2}

theorem roots_in_interval (p q : ℝ) : 
  (p, q) ∈ region_A → (∀ x ∈ Set.Ioo (-2:ℝ) 1, (x^2 + p * x + q) ≠ 0) ∧
  (p, q) ∈ region_B → (∀ x ∈ Set.Ioo (-2:ℝ) 1, (x^2 + p * x + q) ≠ 0) ∧
  (p, q) ∈ region_C → (∃ x ∈ Set.Ioo (-2:ℝ) 1, (x^2 + p * x + q) = 0 ∧ ∀ y ∈ Set.Ioo (-2:ℝ) 1, y ≠ x → (y^2 + p * y + q) ≠ 0) ∧
  (p, q) ∈ region_D → (∃ x₁ x₂ ∈ Set.Ioo (-2:ℝ) 1, x₁ < x₂ ∧ (x₁^2 + p * x₁ + q) = 0 ∧ (x₂^2 + p * x₂ + q) = 0) ∧
  (p, q) ∈ region_E → (∃ x ∈ Set.Ioo (-2:ℝ) 1, (x^2 + p * x + q) = 0 ∧ ∀ y ∈ Set.Ioo (-2:ℝ) 1, y ≠ x → (y^2 + p * y + q) ≠ 0) ∧
  (p, q) ∈ region_F → (∀ x ∈ Set.Ioo (-2:ℝ) 1, (x^2 + p * x + q) ≠ 0) ∧
  (p, q) ∈ region_G → (∃ x₁ x₂ ∈ Set.Ioo (-2:ℝ) 1, x₁ < x₂ ∧ (x₁^2 + p * x₁ + q) = 0 ∧ (x₂^2 + p * x₂ + q) = 0) :=
sorry

end roots_in_interval_l737_737642


namespace calculate_payment_difference_l737_737177

theorem calculate_payment_difference 
  (P : ℝ) 
  (r1 r2 : ℝ) 
  (n : ℕ)
  (semi_annual_factor : ℝ := 2)
  (annual_factor : ℝ := 1)
  (years : ℝ := 10) 
  (P_eq : P = 10000)
  (r1_eq : r1 = 0.10)
  (r2_eq : r2 = 0.08)
  (payment_1: ℝ := let A := P * (1 + r1 / semi_annual_factor)^(semi_annual_factor * (years / 2))
                   in let one_third := A / 3
                   in let remaining := A - one_third
                   in one_third + remaining * (1 + r1 / semi_annual_factor)^(semi_annual_factor * (years / 2)))
  (payment_2: ℝ := P * (1 + r2 / annual_factor)^years)
  (diff := abs (payment_1 - payment_2))
  :
  round diff = 1579 :=
by
  sorry

end calculate_payment_difference_l737_737177


namespace floor_sqrt_80_l737_737950

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737950


namespace distinct_ab_plus_a_plus_b_values_l737_737001

theorem distinct_ab_plus_a_plus_b_values :
  let L := {x | x < 15 ∧ x % 2 = 1}
  let P := { (a, b) | a ∈ L ∧ b ∈ L }
  let S := {abp | ∃ a b, a ∈ L ∧ b ∈ L ∧ abp = a * b + a + b}
  S.card = 24 := sorry

end distinct_ab_plus_a_plus_b_values_l737_737001


namespace determine_c_for_inverse_l737_737209

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def f_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem determine_c_for_inverse :
  (∀ x : ℝ, x ≠ 0 → f (f_inv x) c = x) ↔ c = 1 :=
sorry

end determine_c_for_inverse_l737_737209


namespace power_of_54_l737_737159

theorem power_of_54 (a b : ℕ) (h_a_pos : a > 0) (h_b_pos : b > 0) 
(h_eq : 54^a = a^b) : ∃ k : ℕ, a = 54^k := by
  sorry

end power_of_54_l737_737159


namespace triangle_subdivision_l737_737748

theorem triangle_subdivision (n : ℕ) (polygons : Finset (Finset ℕ)) 
  (h_subdiv : ∀ p ∈ polygons, 4 ≤ p.card ∨ p.card = 3) :
  (∃ p ∈ polygons, p.card = 3) ∨ (∃ p₁ p₂ ∈ polygons, p₁ ≠ p₂ ∧ p₁.card = p₂.card) :=
sorry

end triangle_subdivision_l737_737748


namespace percent_non_gymnastics_basketball_players_l737_737754

-- Definitions for the problem conditions
variables (N : ℕ) -- Total number of students
def basketball_students : ℕ := 0.5 * N
def gymnastics_students : ℕ := 0.4 * N
def basketball_and_gymnastics_students : ℕ := 0.3 * (0.5 * N)

-- Statement to prove
theorem percent_non_gymnastics_basketball_players
  (h1 : basketball_students N = 0.5 * N)
  (h2 : gymnastics_students N = 0.4 * N)
  (h3 : basketball_and_gymnastics_students N = 0.3 * (0.5 * N)) :
  (0.35 * N) / (0.6 * N) * 100 = 58 :=
by
  sorry

end percent_non_gymnastics_basketball_players_l737_737754


namespace probability_card_10_and_spade_l737_737657

theorem probability_card_10_and_spade : 
  let deck_size := 52 
  let num_10s := 4 
  let num_spades := 13
  let first_card_10_prob := num_10s / deck_size
  let second_card_spade_prob_after_10 := (num_spades / (deck_size - 1))
  let total_prob := first_card_10_prob * second_card_spade_prob_after_10
  total_prob = 12 / 663 :=
by
  sorry

end probability_card_10_and_spade_l737_737657


namespace zinc_copper_mixture_weight_l737_737692

theorem zinc_copper_mixture_weight (Z C : ℝ) (h1 : Z / C = 9 / 11) (h2 : Z = 31.5) : Z + C = 70 := by
  sorry

end zinc_copper_mixture_weight_l737_737692


namespace floor_sqrt_80_l737_737807

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737807


namespace proposition_b_proposition_d_l737_737304

-- Proposition B: For a > 0 and b > 0, if ab = 2, then the minimum value of a + 2b is 4
theorem proposition_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2) : a + 2 * b ≥ 4 :=
  sorry

-- Proposition D: For a > 0 and b > 0, if a² + b² = 1, then the maximum value of a + b is sqrt(2).
theorem proposition_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 :=
  sorry

end proposition_b_proposition_d_l737_737304


namespace floor_sqrt_80_l737_737869

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737869


namespace probability_heart_or_king_l737_737332

theorem probability_heart_or_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let overlap := 1
  let unique_hearts_or_kings := hearts + kings - overlap
  let non_hearts_or_kings := total_cards - unique_hearts_or_kings
  let p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  let p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  let p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  p_at_least_one_heart_or_king = 88 / 169 :=
by
  have total_cards := 52
  have hearts := 13
  have kings := 4
  have overlap := 1
  have unique_hearts_or_kings := hearts + kings - overlap
  have non_hearts_or_kings := total_cards - unique_hearts_or_kings
  have p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  have p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  have p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  show p_at_least_one_heart_or_king = 88 / 169
  sorry

end probability_heart_or_king_l737_737332


namespace poly_factorization_l737_737408

theorem poly_factorization (p q : ℚ) :
  ∃ a b c : ℚ, 
    (px^4 + qx^3 + 40x^2 - 24x + 9 = (4x^2 - 3x + 2) * (ax^2 + bx + c)) ∧ 
    p = 4 * a ∧ 
    q = 4 * b - 3 * a ∧ 
    40 = 4 * c - 3 * b + 2 * a ∧ 
    -24 = -3 * c + 2 * b ∧ 
    9 = 2 * c ∧ 
    p = 12.5 ∧ 
    q = -30.375 :=
begin
  sorry
end

end poly_factorization_l737_737408


namespace valid_number_count_is_300_l737_737433

-- Define the set of digits
def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6]

-- Define the set of odd digits
def odd_digits : List ℕ := [1, 3, 5]

-- Define a function to count valid four-digit numbers
noncomputable def count_valid_numbers : ℕ :=
  (odd_digits.length * (digits.length - 2) * (digits.length - 2) * (digits.length - 3))

-- State the theorem
theorem valid_number_count_is_300 : count_valid_numbers = 300 :=
  sorry

end valid_number_count_is_300_l737_737433


namespace floor_sqrt_80_eq_8_l737_737913

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737913


namespace parabola_intersections_l737_737776

def parabolas_count (slope_count : Nat) (intercept_count : Nat) : Nat := 
  slope_count * intercept_count
  
def total_pairs (n : Nat) : Nat := 
  n * (n - 1) / 2

def parallel_pairs_count (slope_count : Nat) (intercept_count : Nat) : Nat := 
  let per_slope_comb : Nat := (intercept_count * (intercept_count - 1) / 2) + (intercept_count - 1) * intercept_count / 2 + 1
  slope_count * per_slope_comb 

theorem parabola_intersections : 
  let slopes := 7
  let intercepts := 9
  let total_parabolas := parabolas_count slopes intercepts in
  total_pairs total_parabolas - parallel_pairs_count slopes intercepts * 2 = 2212 :=
by
  let total_parabolas := parabolas_count 7 9
  let intersection_pairs := total_pairs total_parabolas - parallel_pairs_count 7 9
  have h : 2 * intersection_pairs = 2212 := sorry
  exact h

end parabola_intersections_l737_737776


namespace order_of_values_l737_737445

-- Define the properties and function
variables (f : ℝ → ℝ) (a b c : ℝ)

-- Given conditions
def condition1 : Prop := ∀ x1 x2 ∈ set.Icc (4 : ℝ) 8, x1 < x2 → (f x1 - f x2) / (x1 - x2) > 0
def condition2 : Prop := ∀ x : ℝ, f (x + 4) = - f x
def condition3 : Prop := ∀ x : ℝ, f (x + 4) = f (-(x + 4))

-- Values assignment
def a_def : Prop := a = f 6
def b_def : Prop := b = f 11
def c_def : Prop := c = f 2017

-- The theorem (proof statement)
theorem order_of_values (h1 : condition1 f) (h2 : condition2 f) (h3 : condition3 f) (ha : a_def f a) (hb : b_def f b) (hc : c_def f c) : b < a ∧ a < c := by
  sorry

end order_of_values_l737_737445


namespace probability_at_least_one_heart_or_king_l737_737337
   
   noncomputable def probability_non_favorable : ℚ := 81 / 169

   theorem probability_at_least_one_heart_or_king :
     1 - probability_non_favorable = 88 / 169 := 
   sorry
   
end probability_at_least_one_heart_or_king_l737_737337


namespace floor_sqrt_80_l737_737937

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737937


namespace largest_neg_multiple_of_7_greater_than_neg_150_l737_737263

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l737_737263


namespace fractions_equal_l737_737152

theorem fractions_equal (a b c d : ℚ) (h1 : a = 2/7) (h2 : b = 3) (h3 : c = 3/7) (h4 : d = 2) :
  a * b = c * d := 
sorry

end fractions_equal_l737_737152


namespace maximum_area_of_sector_l737_737222

theorem maximum_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 10) : 
  (1 / 2 * l * r) ≤ 25 / 4 := 
sorry

end maximum_area_of_sector_l737_737222


namespace smallest_square_area_l737_737699

noncomputable def three_four_diagonal := real.sqrt (3^2 + 4^2)  -- diagonal length of 3x4 rectangle
noncomputable def four_five_diagonal := real.sqrt (4^2 + 5^2)  -- diagonal length of 4x5 rectangle

theorem smallest_square_area :
  ∀ (d1 d2 : ℝ), d1 = real.sqrt (3^2 + 4^2) → d2 = real.sqrt (4^2 + 5^2) →
  (d1 ≤ d2 ∨ d2 ≤ d1) →
  (∃ (s : ℝ), s = max d1 d2 ∧ s^2 = 41) :=
by
  intros d1 d2 h1 h2 h_or
  use max d1 d2
  split
  · apply le_refl (max d1 d2)
  · sorry

end smallest_square_area_l737_737699


namespace periodic_special_words_min_number_remainder_1000_l737_737548

-- Define the conditions
def W : Set Char := {'a', 'b'}
def periodic (W : List Char) : Prop := 
  ∃ p : ℕ, p = 2^(2016) ∧ ∀ i : ℕ, W.nth (i + p) = W.nth i

def appears (U : List Char) (W : List Char) : Prop := 
  ∃ k l : ℕ, k ≤ l ∧ U = W.slice k l

def special (U : List Char) (W : List Char) : Prop :=
  appears (U ++ ['a']) W ∧ appears (U ++ ['b']) W ∧ appears (['a'] ++ U) W ∧ appears (['b'] ++ U) W

def no_special_words_longer_than (n : ℕ) (W : List Char) : Prop :=
  ∀ U : List Char, U.length > n → ¬ special U W

def N := 2^(2016) - 1

-- Theorem statement
theorem periodic_special_words_min_number_remainder_1000 :
  ∀ W : List Char, periodic W →
  no_special_words_longer_than 2015 W →
  (N % 1000 = 535) := by
  sorry

end periodic_special_words_min_number_remainder_1000_l737_737548


namespace geometric_sequence_problem_l737_737523

noncomputable def a (n : ℕ) : ℝ := sorry -- arbitrary definition for the geometric sequence

def q : ℝ := - (1 / 2)  -- from solution step, q is calculated as -1/2

-- Define the sum of the first n terms of the sequence \{1/a_n\}
def S (n : ℕ) : ℝ := sorry -- needs actual implementation of the sum function for proof

theorem geometric_sequence_problem (h : a 1 ≠ 0) (condition : a 2 + 8 * a 5 = 0) : 
  (S 5 / S 2) = -11 :=
by
  sorry

end geometric_sequence_problem_l737_737523


namespace largest_multiple_of_7_negation_gt_neg150_l737_737275

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l737_737275


namespace floor_sqrt_80_eq_8_l737_737911

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737911


namespace perimeter_of_8_sides_each_12_cm_l737_737290

theorem perimeter_of_8_sides_each_12_cm :
  let num_sides := 8
  let side_length := 12
  let perimeter := num_sides * side_length
  perimeter = 96 :=
by
  let num_sides := 8
  let side_length := 12
  let perimeter := num_sides * side_length
  have h : perimeter = 8 * 12 := rfl
  have h1 : 8 * 12 = 96 := by norm_num
  rwa [h, h1]

end perimeter_of_8_sides_each_12_cm_l737_737290


namespace sum_of_series_l737_737693

theorem sum_of_series :
  (3 + 13 + 23 + 33 + 43) + (11 + 21 + 31 + 41 + 51) = 270 := 
by
  sorry

end sum_of_series_l737_737693


namespace find_a_l737_737506

theorem find_a (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : 
  a = real.cbrt 4 := 
sorry

end find_a_l737_737506


namespace floor_sqrt_80_l737_737828

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737828


namespace floor_sqrt_80_l737_737971

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737971


namespace tetrahedron_area_inequality_l737_737183

theorem tetrahedron_area_inequality (A B C D : Point)
  (A_ABC A_ABD A_ACD A_BCD : ℝ) :
  A_ABC < A_ABD + A_ACD + A_BCD :=
begin
  sorry
end

end tetrahedron_area_inequality_l737_737183


namespace largest_multiple_of_7_negation_gt_neg150_l737_737270

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l737_737270


namespace eq_a_no_real_roots_l737_737371

theorem eq_a_no_real_roots (x : ℝ) : 
  (∀ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = 1 → (b^2 - 4*a*c < 0)) ∧
  (∀ a b c : ℝ, a = 1 ∧ b = 2 ∧ c = 1 → (b^2 - 4*a*c ≥ 0)) ∧
  (∀ a b c : ℝ, a = 1 ∧ b = -2 ∧ c = -1 → (b^2 - 4*a*c ≥ 0)) ∧
  (∀ a b c : ℝ, a = 1 ∧ b = -1 ∧ c = -2 → (b^2 - 4*a*c ≥ 0))
: 
(∀ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = 1 → (b^2 - 4*a*c < 0)) 
:= by
  sorry

end eq_a_no_real_roots_l737_737371


namespace correct_card_assignment_l737_737236

theorem correct_card_assignment :
  ∃ (cards : Fin 4 → Fin 4), 
    (¬ (cards 1 = 3 ∨ cards 2 = 3) ∧
     ¬ (cards 0 = 2 ∨ cards 2 = 2) ∧
     ¬ (cards 0 = 1) ∧
     ¬ (cards 0 = 3)) →
    (cards 0 = 4 ∧ cards 1 = 2 ∧ cards 2 = 1 ∧ cards 3 = 3) := 
by {
  sorry
}

end correct_card_assignment_l737_737236


namespace AP_equals_BP_plus_CP_l737_737590

-- Define the equilateral triangle ABC and the circumcircle
structure EquilateralTriangle (A B C : Type) :=
(eq_sides : ∀ a b c : Type, a = b ∧ b = c ∧ c = a)

-- Condition P lies on the arc BC of the circumcircle of triangle ABC
def OnArcBCOfCircumcircle {A B C P : Type} (circumcircle : Type) (H : EquilateralTriangle A B C) : Prop :=
true -- Placeholder for actual geometric condition of P on the arc BC

-- Prove the equality AP = BP + CP
theorem AP_equals_BP_plus_CP (A B C P : Type) (H : EquilateralTriangle A B C) (H' : OnArcBCOfCircumcircle A B C P) :
  AP = BP + CP :=
sorry

end AP_equals_BP_plus_CP_l737_737590


namespace irreducible_denominator_l737_737585

noncomputable def denominator_of_irreducible_fraction : ℕ := 5 ^ 26

theorem irreducible_denominator :
  ∀ (n : ℕ), (n = (100! / 10 ^ 50).den) → n = denominator_of_irreducible_fraction :=
by
  sorry

end irreducible_denominator_l737_737585


namespace calculate_fraction_value_l737_737770

theorem calculate_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 11 / 7 := 
  sorry

end calculate_fraction_value_l737_737770


namespace jeep_time_fraction_l737_737720

def original_speed := 440 / 3
def new_speed := 293.3333333333333
def time_fraction := original_speed / new_speed

theorem jeep_time_fraction : time_fraction = 1 / 2 :=
by
  -- sorry is used to skip the proof
  sorry

end jeep_time_fraction_l737_737720


namespace floor_sqrt_80_l737_737884

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737884


namespace max_enclosed_area_perimeter_160_length_twice_width_l737_737600

theorem max_enclosed_area_perimeter_160_length_twice_width 
  (W L : ℕ) 
  (h1 : 2 * (L + W) = 160) 
  (h2 : L = 2 * W) : 
  L * W = 1352 := 
sorry

end max_enclosed_area_perimeter_160_length_twice_width_l737_737600


namespace passes_through_point_l737_737626

variable (a : ℝ) (x : ℝ) 

def function_graph := a^(x-2) + 1

theorem passes_through_point (h₀ : a > 0) (h₁ : a ≠ 1) : function_graph a 2 = 2 := by
  unfold function_graph
  calc
    a^(2-2) + 1 = a^0 + 1   : by rw [show 2-2 = 0 by norm_num]
                _ = 1 + 1   : by rw [pow_zero a]
                _ = 2       : by norm_num

end passes_through_point_l737_737626


namespace certain_event_white_balls_l737_737677

open Finset

theorem certain_event_white_balls :
  ∀ (box : Finset ℕ) (n : ℕ) (h_box : box.card = 5) 
  (h_all_white : ∀ x ∈ box, x < 5 ∧ x ≥ 0),
  ∃ (draw : Finset ℕ), draw.card = 2 ∧ draw ⊆ box :=
by
  sorry

end certain_event_white_balls_l737_737677


namespace floor_sqrt_80_eq_8_l737_737846

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737846


namespace quadratic_no_real_roots_probability_l737_737243

noncomputable def probability_no_real_roots : ℝ :=
let a := Uniform.random_variable 0 1 in
Prob { a | a > 1/4 }

theorem quadratic_no_real_roots_probability :
  probability_no_real_roots = 3 / 4 := 
sorry

end quadratic_no_real_roots_probability_l737_737243


namespace floor_sqrt_80_l737_737940

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737940


namespace community_B_selection_l737_737401

theorem community_B_selection :
  let A := 360
  let B := 270
  let C := 180
  let total_households := 90
  let total := A + B + C
  (B / total) * total_households = 30 :=
by
  let A := 360
  let B := 270
  let C := 180
  let total_households := 90
  let total := A + B + C
  have h1 : total = 360 + 270 + 180 := rfl
  have h2 : B / total = 270 / total := rfl
  have h3 : (270 / total) * total_households = (270 / 810) * 90 := by simp [h1, B]
  have h4 : (270 / 810) * 90 = 1 / 3 * 90 := by norm_num
  have h5 : 1 / 3 * 90 = 30 := by norm_num
  exact (h5 : 1 / 3 * 90 = 30)

end community_B_selection_l737_737401


namespace floor_sqrt_80_l737_737935

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737935


namespace sqrt_floor_eight_l737_737849

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737849


namespace compare_f_values_l737_737069

noncomputable theory
open Real

-- Define the function f and declare it as an odd function, periodic with period 8, and increasing on [0, 2]
variables {f : ℝ → ℝ}

-- Conditions
axiom odd_function : ∀ x : ℝ, f(-x) = -f(x)
axiom periodicity : ∀ x : ℝ, f(x - 4) = -f(x)
axiom increasing_on_0_2 : ∀ {x y : ℝ}, (0 ≤ x ∧ x ≤ 2) → (0 ≤ y ∧ y ≤ 2) → (x < y → f(x) < f(y))

-- Define the property to be proven between f(-25), f(80), and f(11)
theorem compare_f_values : f(-25) < f(80) ∧ f(80) < f(11) :=
sorry

end compare_f_values_l737_737069


namespace largest_multiple_of_7_neg_greater_than_neg_150_l737_737284

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l737_737284


namespace ratio_eq_thirteen_fifths_l737_737115

theorem ratio_eq_thirteen_fifths
  (a b c : ℝ)
  (h₁ : b / a = 4)
  (h₂ : c / b = 2) :
  (a + b + c) / (a + b) = 13 / 5 :=
sorry

end ratio_eq_thirteen_fifths_l737_737115


namespace solve_for_x_l737_737606

theorem solve_for_x (x : ℝ) : 7^(x+3) = 343^x ↔ x = 3 / 2 := by
  sorry

end solve_for_x_l737_737606


namespace find_side_BC_l737_737536

theorem find_side_BC (A B AC : ℝ) (h1 : A = 45) (h2 : B = 60) (h3 : AC = 6) : 
  let C := 180 - A - B in
  BC = 2 * Real.sqrt 6 := by
  sorry

end find_side_BC_l737_737536


namespace remainder_polynomial_division_l737_737030

theorem remainder_polynomial_division :
  let f : ℝ → ℝ := λ x, x^4 - 4 * x^2 + 7 * x - 8 in
  f 3 = 58 :=
by
  intro f
  sorry

end remainder_polynomial_division_l737_737030


namespace sum_of_remainders_l737_737671

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l737_737671


namespace geometric_sequence_first_term_l737_737624

open Real Nat

theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^4 = (7! : ℝ))
  (h2 : a * r^7 = (8! : ℝ)) : a = 315 := by
  sorry

end geometric_sequence_first_term_l737_737624


namespace solve_for_y_l737_737109

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l737_737109


namespace total_time_taken_l737_737321

variables (distTotal dist40 : ℕ) (speed40 speed60 : ℕ) (timeTotal : ℤ)

theorem total_time_taken
  (h1 : distTotal = 250)
  (h2 : dist40 = 124)
  (h3 : speed40 = 40)
  (h4 : speed60 = 60)
  (h5 : timeTotal = (dist40 / speed40) + ((distTotal - dist40) / speed60)) :
  timeTotal = 52 / 10 := 
begin
  sorry
end

end total_time_taken_l737_737321


namespace positive_difference_l737_737404

noncomputable def f (n : ℝ) : ℝ :=
if n < -1 then n^2 + 2 * n else real.exp n - 10

theorem positive_difference :
  let a1 := -3 in
  let a2 := real.log 16 in
  f(-3) + f(0) + f(a1) = 0 → f(-3) + f(0) + f(a2) = 0 → |a2 - a1| = real.log 16 + 3 :=
by
  sorry

end positive_difference_l737_737404


namespace number_of_x_intercepts_l737_737994

def is_x_intercept (x : ℝ) : Prop :=
  ∃ k : ℤ, k ≠ 0 ∧ x = 1 / (k * Real.pi)

def within_interval (x : ℝ) : Prop :=
  0.0001 < x ∧ x < 0.0005

theorem number_of_x_intercepts :
  (Finset.filter within_interval (Finset.filter is_x_intercept (Finset.Icc 0.0001 0.0005))).card = 2547 := by
  sorry

end number_of_x_intercepts_l737_737994


namespace cube_surface_area_increase_cube_volume_increase_l737_737302

theorem cube_surface_area_increase (s : ℝ) :
  let original_surface_area := 6 * s ^ 2,
      new_edge_length := 1.6 * s,
      new_surface_area := 6 * (new_edge_length) ^ 2
  in (new_surface_area - original_surface_area) / original_surface_area * 100 = 156 := 
sorry

theorem cube_volume_increase (s : ℝ) :
  let original_volume := s ^ 3,
      new_edge_length := 1.6 * s,
      new_volume := (new_edge_length) ^ 3
  in (new_volume - original_volume) / original_volume * 100 = 309.6 := 
sorry

end cube_surface_area_increase_cube_volume_increase_l737_737302


namespace sum_of_proper_divisors_256_l737_737297

theorem sum_of_proper_divisors_256 : 
  let n := 256
  let proper_divisors := (∑ i in finset.range 8, 2^i)
  proper_divisors = 255 := 
by {
  let n := 256,
  let proper_divisors := finset.range 8,
  have h : n = 2^8 := by norm_num,
  have h_proper_div : ∑ i in proper_divisors, 2^i = 255 := sorry,
  exact h_proper_div,
}

end sum_of_proper_divisors_256_l737_737297


namespace select_51_boxes_half_fruits_l737_737518

-- Assume there are 100 boxes
variable (n : ℕ)
variable (A : Fin n → ℕ) -- Number of apples in each box
variable (O : Fin n → ℕ) -- Number of oranges in each box
variable (B : Fin n → ℕ) -- Number of bananas in each box
variable (h_n : n = 100)

-- Define total count of each fruit
def total_apples := (Finset.univ : Finset (Fin n)).sum A
def total_oranges := (Finset.univ : Finset (Fin n)).sum O
def total_bananas := (Finset.univ : Finset (Fin n)).sum B

-- Define the subset with 51 boxes
variable (S : Finset (Fin n))
variable (h_S_card : S.card = 51)

-- Define the sum of fruits in subset S
def sum_apples := S.sum A
def sum_oranges := S.sum O
def sum_bananas := S.sum B

-- Prove that the subset S contains at least half of each fruit
theorem select_51_boxes_half_fruits 
  (h_n : n = 100) 
  (h_half_apples : sum_apples A S ≥ total_apples A / 2) 
  (h_half_oranges : sum_apples O S ≥ total_oranges O / 2) 
  (h_half_bananas : sum_apples B S ≥ total_bananas B / 2) : 
  ∃ (S : Finset (Fin n)), S.card = 51 ∧ 
    sum_apples A S ≥ total_apples A / 2 ∧ 
    sum_oranges O S ≥ total_oranges O / 2 ∧ 
    sum_bananas B S ≥ total_bananas B / 2 := 
by 
  have Sn := 2 * 51
  linarith 
  sorry

end select_51_boxes_half_fruits_l737_737518


namespace zeta_sum_equivalence_l737_737035

open BigOperators

def Riemann_zeta (y : ℝ) : ℝ := ∑' m : ℕ, 1 / m ^ y

theorem zeta_sum_equivalence :
  (∑' j : ℕ in finset.Ico 3 ⬝, fractional_part (Riemann_zeta (3 * j - 2)) = 
  ∑' m : ℕ in finset.Ico 2 ⬝, 1 / (m ^ 7 - m ^ 4) :=
by {
  sorry
}

end zeta_sum_equivalence_l737_737035


namespace angle_C_is_60_degrees_l737_737068

theorem angle_C_is_60_degrees (a b c : ℝ) (h : a ^ 2 + b ^ 2 - c ^ 2 = a * b) : 
  ∠C = π / 3 :=
begin
  -- Proof steps go here
  sorry
end

end angle_C_is_60_degrees_l737_737068


namespace quadratic_function_inequality_l737_737475

variable (a x x₁ x₂ : ℝ)

def f (x : ℝ) := a * x^2 + 2 * a * x + 4

theorem quadratic_function_inequality
  (h₀ : 0 < a) (h₁ : a < 3)
  (h₂ : x₁ + x₂ = 0)
  (h₃ : x₁ < x₂) :
  f a x₁ < f a x₂ := 
sorry

end quadratic_function_inequality_l737_737475


namespace trigonometric_polynomial_equiv_polynomial_l737_737594

noncomputable def trigonometric_polynomial (n : ℕ) : Type :=
  { f : ℝ → ℝ // ∃ (a : fin (n + 1) → ℝ), f = λ φ, (a 0) + ∑ i in finset.range (n + 1), (a i) * (real.cos φ)^i }

noncomputable def polynomial (n : ℕ) : Type :=
  { P : ℝ → ℝ // ∃ (b : fin (n + 1) → ℝ), P = λ x, (b 0) + ∑ i in finset.range (n + 1), (b i) * x^i }

theorem trigonometric_polynomial_equiv_polynomial (n : ℕ)
  (f : trigonometric_polynomial n) :
  ∃ P : polynomial n, (λ φ, (P.val (real.cos φ))) = f.val ∧
  ∃ f' : trigonometric_polynomial n, (λ x, (f'.val (real.acos x))) = P.val := sorry

end trigonometric_polynomial_equiv_polynomial_l737_737594


namespace jenny_change_l737_737541

/-!
## Problem statement

Jenny is printing 7 copies of her 25-page essay. It costs $0.10 to print one page.
She also buys 7 pens, each costing $1.50. If she pays with $40, calculate the change she should get.
-/

def cost_per_page : ℝ := 0.10
def pages_per_copy : ℕ := 25
def num_copies : ℕ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℕ := 7
def amount_paid : ℝ := 40.0

def total_pages : ℕ := num_copies * pages_per_copy

def cost_printing : ℝ := total_pages * cost_per_page
def cost_pens : ℝ := num_pens * cost_per_pen

def total_cost : ℝ := cost_printing + cost_pens

theorem jenny_change : amount_paid - total_cost = 12 := by
  -- proof here
  sorry

end jenny_change_l737_737541


namespace partition_vertices_into_3_sets_l737_737712

-- Define the graph structure and conditions
structure DirectedGraph (V : Type) :=
  (edges : V → V × V)

variable {V : Type}

-- Define the partitioning function and the target partition property
def valid_partition (g : DirectedGraph V) (partition : V → Fin 3) : Prop :=
  ∀ v : V, let (v1, v2) := g.edges v in partition v ≠ partition v1 ∨ partition v ≠ partition v2

-- Main theorem statement
theorem partition_vertices_into_3_sets (g : DirectedGraph V) (h : ∀ v : V, (g.edges v).fst ≠ v ∧ (g.edges v).snd ≠ v) :
  ∃ partition : V → Fin 3, valid_partition g partition :=
sorry

end partition_vertices_into_3_sets_l737_737712


namespace find_area_ratio_l737_737537

def triangle_area_ratio (AB BC AC AD AE : ℝ) (hAB : AB = 20) (hBC : BC = 30) (hAC : AC = 34) 
                        (hAD : AD = 12) (hAE : AE = 18) : Prop :=
  ∃ (D E : Type) (hD : D ∈ segment AB) (hE : E ∈ segment AC),
  ratio_of_areas ADE BCED = 9/16

theorem find_area_ratio :
  triangle_area_ratio 20 30 34 12 18
  (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) := sorry

end find_area_ratio_l737_737537


namespace inequality_proof_l737_737046

noncomputable def a : ℝ := Real.pi ^ (1/2)
noncomputable def b : ℝ := Real.logBase Real.pi (1/2)
noncomputable def c : ℝ := Real.logBase (1/Real.pi) (1/2)

theorem inequality_proof : a > c ∧ c > b := by
  sorry

end inequality_proof_l737_737046


namespace floor_of_sqrt_80_l737_737930

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737930


namespace units_digit_expression_l737_737008

lemma sqrt_256 : real.sqrt 256 = 16 := sorry

def A : ℝ := 17 + real.sqrt 256
def B : ℝ := 17 - real.sqrt 256

theorem units_digit_expression :
  (A ^ 21 - B ^ 21) % 10 = 2 := sorry

end units_digit_expression_l737_737008


namespace largest_multiple_of_7_negated_gt_neg_150_l737_737252

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l737_737252


namespace three_digit_with_repeated_digits_l737_737662

theorem three_digit_with_repeated_digits :
  let total_three_digit_numbers := 900 in
  let total_three_digit_numbers_without_repeated_digits := 9 * 9 * 8 in
  total_three_digit_numbers - total_three_digit_numbers_without_repeated_digits = 252 :=
by
  sorry

end three_digit_with_repeated_digits_l737_737662


namespace lengths_equal_l737_737380

variables {α : ℝ} {a b c : ℝ} -- Scalars representing lengths and angles of the triangle

noncomputable def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def length_O1O2 (a b : ℝ) : ℝ :=
  (a + b) / Real.sqrt 2

noncomputable def length_CO3 (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / Real.sqrt 2

theorem lengths_equal (a b c : ℝ) (h : right_triangle a b c) : 
  length_O1O2 a b = length_CO3 a b :=
by
  unfold length_O1O2
  unfold length_CO3
  rw h
  slice_rhs 3 2
  rw [Real.sqrt_div, Real.sqrt_mul_self, Real.sqrt_add_eq]
  sorry

end lengths_equal_l737_737380


namespace product_value_l737_737765

def telescoping_product : ℕ → ℚ
| 0 := 1
| (n + 1) := (4 * (n + 1)) / (4 * (n + 1) + 4) * telescoping_product n

theorem product_value : telescoping_product 501 = 1 / 502 := by
  sorry

end product_value_l737_737765


namespace floor_sqrt_80_l737_737974

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737974


namespace floor_sqrt_80_l737_737965

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737965


namespace deepit_worked_hours_l737_737627

theorem deepit_worked_hours (hours_saturday hours_sunday : ℕ)
  (h_saturday : hours_saturday = 6)
  (h_sunday : hours_sunday = 4) :
  hours_saturday + hours_sunday = 10 :=
by
  rw [h_saturday, h_sunday]
  norm_num -- norm_num replaces 6 + 4 with 10

end deepit_worked_hours_l737_737627


namespace exists_divisor_in_sequence_l737_737595

theorem exists_divisor_in_sequence (n : ℕ) (hn1 : n > 1) (hn2 : odd n) :
  ∃ m, 1 ≤ m ∧ m < n ∧ n ∣ (2 ^ m - 1) :=
sorry

end exists_divisor_in_sequence_l737_737595


namespace floor_sqrt_80_l737_737969

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737969


namespace sum_series_fraction_l737_737988

theorem sum_series_fraction :
  (∑ n in Finset.range 9, (1 / ((n + 1 : ℝ) * (n + 2 : ℝ)))) = 9 / 10 :=
by
  sorry

end sum_series_fraction_l737_737988


namespace combined_height_of_trees_l737_737581

noncomputable def growth_rate_A (weeks : ℝ) : ℝ := (weeks / 2) * 50
noncomputable def growth_rate_B (weeks : ℝ) : ℝ := (weeks / 3) * 70
noncomputable def growth_rate_C (weeks : ℝ) : ℝ := (weeks / 4) * 90
noncomputable def initial_height_A : ℝ := 200
noncomputable def initial_height_B : ℝ := 150
noncomputable def initial_height_C : ℝ := 250
noncomputable def total_weeks : ℝ := 16
noncomputable def total_growth_A := growth_rate_A total_weeks
noncomputable def total_growth_B := growth_rate_B total_weeks
noncomputable def total_growth_C := growth_rate_C total_weeks
noncomputable def final_height_A := initial_height_A + total_growth_A
noncomputable def final_height_B := initial_height_B + total_growth_B
noncomputable def final_height_C := initial_height_C + total_growth_C
noncomputable def final_combined_height := final_height_A + final_height_B + final_height_C

theorem combined_height_of_trees :
  final_combined_height = 1733.33 := by
  sorry

end combined_height_of_trees_l737_737581


namespace floor_of_sqrt_80_l737_737917

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737917


namespace part1_part2_l737_737080

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + (1 + a) * Real.exp (-x)

theorem part1 (a : ℝ) : (∀ x : ℝ, f x a = f (-x) a) ↔ a = 0 := by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, 0 < x → f x a ≥ a + 1) → a ≤ 3 := by
  sorry

end part1_part2_l737_737080


namespace evaluate_expression_l737_737417

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 5)) = 15 / 16 :=
by 
  sorry

end evaluate_expression_l737_737417


namespace floor_of_sqrt_80_l737_737928

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737928


namespace avg_weight_B_is_approx_70_01_l737_737646

-- Definitions for conditions
def num_students_A := 50
def avg_weight_A := 50.0
def num_students_B := 70
def avg_weight_whole := 61.67

-- Helper definition for total weights based on the conditions
def total_weight_A := num_students_A * avg_weight_A
def total_weight_whole := (num_students_A + num_students_B) * avg_weight_whole
def total_weight_B := total_weight_whole - total_weight_A

-- Main theorem
theorem avg_weight_B_is_approx_70_01 :
  (total_weight_B / num_students_B).to_string(2) = "70.01" :=
by sorry

end avg_weight_B_is_approx_70_01_l737_737646


namespace limit_of_difference_l737_737163

noncomputable def f : ℝ → ℝ := sorry  -- assume f(x) is a differentiable function
variable {h : ℝ}

theorem limit_of_difference (hf : differentiable ℝ f) (hf' : deriv f 2 = 1/2) :
  filter.tendsto (λ h, (f (2 - h) - f (2 + h)) / h) (nhds 0) (nhds (-1/2)) := 
by 
  -- proof is omitted as per problem specification
  sorry

end limit_of_difference_l737_737163


namespace exponential_inequality_solution_l737_737456

theorem exponential_inequality_solution (x : ℝ) (h : 2^(2*x - 7) < 2^(x - 3)) : x < 4 :=
sorry

end exponential_inequality_solution_l737_737456


namespace angle_CPD_l737_737140

noncomputable def arc_AS := 70
noncomputable def arc_BT := 45
def PC_tangent_SAR : Prop := sorry
def PD_tangent_RBT : Prop := sorry
def SRT_straight_line : Prop := sorry

theorem angle_CPD :
  PC_tangent_SAR →
  PD_tangent_RBT →
  SRT_straight_line →
  arc_AS = 70 →
  arc_BT = 45 →
  ∠ CPD = 115 := by
  sorry

end angle_CPD_l737_737140


namespace probability_heart_or_king_l737_737326

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l737_737326


namespace min_max_in_first_50_terms_l737_737087

noncomputable def a_n (n : ℕ) : ℝ :=
  (n - Real.sqrt 2015) / (n - Real.sqrt 2016)

theorem min_max_in_first_50_terms :
  ∃ a b, a = a_n 44 ∧ b = a_n 45 ∧ ∀ n ∈ {1, ..., 50}, a_n n ≤ a ∧ b ≤ a_n n :=
sorry

end min_max_in_first_50_terms_l737_737087


namespace gg1_eq_13_l737_737405

def g (n : ℕ) : ℕ :=
if n < 3 then n^2 + 1
else if n < 6 then 2 * n + 3
else 4 * n - 2

theorem gg1_eq_13 : g (g (g 1)) = 13 :=
by
  sorry

end gg1_eq_13_l737_737405


namespace floor_sqrt_80_l737_737949

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737949


namespace base_rate_first_company_proof_l737_737742

noncomputable def base_rate_first_company : ℝ := 8.00
def charge_per_minute_first_company : ℝ := 0.25
def base_rate_second_company : ℝ := 12.00
def charge_per_minute_second_company : ℝ := 0.20
def minutes : ℕ := 80

theorem base_rate_first_company_proof :
  base_rate_first_company = 8.00 :=
sorry

end base_rate_first_company_proof_l737_737742


namespace range_f_subset_interval_l737_737446

-- Define the function f on real numbers
def f : ℝ → ℝ := sorry

-- The given condition for all real numbers x and y such that x > y
axiom condition (x y : ℝ) (h : x > y) : (f x)^2 ≤ f y

-- The main theorem that needs to be proven
theorem range_f_subset_interval : ∀ x, 0 ≤ f x ∧ f x ≤ 1 := 
by
  intro x
  apply And.intro
  -- Proof for 0 ≤ f x
  sorry
  -- Proof for f x ≤ 1
  sorry

end range_f_subset_interval_l737_737446


namespace greatest_perimeter_among_six_pieces_l737_737207

-- Define the isosceles triangle with the given conditions
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  base_pos : 0 < base
  height_pos : 0 < height

-- Define a Lean 4 theorem that corresponds to the proof problem
theorem greatest_perimeter_among_six_pieces (T : IsoscelesTriangle) (hT : T.base = 10) (hH : T.height = 12) :
  ∃ p, p = 33.27 ∧ ∀ k : ℕ, k < 6 → p = 1.67 + (sqrt (12^2 + (10 * k / 6 - 5)^2) + sqrt (12^2 + (10 * (k + 1) / 6 - 5)^2)) 
  :=
begin
  sorry
end

end greatest_perimeter_among_six_pieces_l737_737207


namespace prob_not_on_x_axis_prob_in_second_quadrant_l737_737488
-- Import the required libraries

-- Define the set A
def A : Set ℤ := {-9, -7, -5, -3, -1, 0, 2, 4, 6, 8}

-- Define the conditions (x, y) ∈ A and x ≠ y
def valid_pairs (x y : ℤ) : Prop := x ∈ A ∧ y ∈ A ∧ x ≠ y

-- Define the event A' where (x, y) is not on the x-axis
def event_not_on_x_axis (x y : ℤ) : Prop := valid_pairs x y ∧ y ≠ 0

-- Define the event B where (x, y) is in the second quadrant
def event_in_second_quadrant (x y : ℤ) : Prop := valid_pairs x y ∧ x < 0 ∧ y > 0

-- The proof statements in Lean

-- The probability that (x, y) is not on the x-axis
theorem prob_not_on_x_axis : 
  (∑ (x y : ℤ) in A ×ˢ A, if event_not_on_x_axis x y then 1 else 0) / 
  (∑ (x y : ℤ) in A ×ˢ A, if valid_pairs x y then 1 else 0) = 0.9 := 
  sorry

-- The probability that (x, y) is in the second quadrant
theorem prob_in_second_quadrant : 
  (∑ (x y : ℤ) in A ×ˢ A, if event_in_second_quadrant x y then 1 else 0) / 
  (∑ (x y : ℤ) in A ×ˢ A, if valid_pairs x y then 1 else 0) = 2 / 9 := 
  sorry

end prob_not_on_x_axis_prob_in_second_quadrant_l737_737488


namespace complete_square_identity_l737_737775

theorem complete_square_identity (x : ℝ) : ∃ (d e : ℤ), (x^2 - 10 * x + 13 = 0 → (x + d)^2 = e ∧ d + e = 7) :=
sorry

end complete_square_identity_l737_737775


namespace lottery_game_l737_737135

noncomputable def eligible_numbers : set ℕ := {1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100}

def include_twenty : ∀ (s : set ℕ), 20 ∈ s → s ⊆ eligible_numbers → set ℕ := {n | n = 20 ∨ n = 2 ∨ n = 5 ∨ n = 10 ∨ n = 25 ∨ n = 8 ∨ n = 4}

theorem lottery_game :
  let k := 10 in
  let total_combinations := nat.choose 14 5 in
  let prob := k / (total_combinations : ℝ) in
  prob = 10 / 3003 := by 
  sorry

end lottery_game_l737_737135


namespace probability_heart_or_king_l737_737331

theorem probability_heart_or_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let overlap := 1
  let unique_hearts_or_kings := hearts + kings - overlap
  let non_hearts_or_kings := total_cards - unique_hearts_or_kings
  let p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  let p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  let p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  p_at_least_one_heart_or_king = 88 / 169 :=
by
  have total_cards := 52
  have hearts := 13
  have kings := 4
  have overlap := 1
  have unique_hearts_or_kings := hearts + kings - overlap
  have non_hearts_or_kings := total_cards - unique_hearts_or_kings
  have p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  have p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  have p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  show p_at_least_one_heart_or_king = 88 / 169
  sorry

end probability_heart_or_king_l737_737331


namespace neg_p_l737_737084

def p : Prop := ∀ x : ℝ, sin x ≤ 1

theorem neg_p : ¬p ↔ ∃ x : ℝ, sin x > 1 :=
by
  sorry

end neg_p_l737_737084


namespace smallest_b_greater_than_three_l737_737292

theorem smallest_b_greater_than_three (b : ℕ) (h : b > 3) : 
  (∃ b, b = 5 ∧ (∃ n : ℕ, 4 * b + 5 = n^2)) :=
by
  use 5
  constructor
  · rfl
  · use 5
  sorry

end smallest_b_greater_than_three_l737_737292


namespace repeating_decimal_to_fraction_l737_737120

theorem repeating_decimal_to_fraction :
  let x := 0.565656...
  let (a, b) := (8, 33)
  gcd a b = 1 → 
  (∃ a b : ℕ, x = (a:ℝ) / (b:ℝ) ∧ a + b = 41) :=
by
  sorry

end repeating_decimal_to_fraction_l737_737120


namespace exists_x_y_mod_p_l737_737168

theorem exists_x_y_mod_p (p : ℕ) (hp : Nat.Prime p) (a : ℤ) : ∃ x y : ℤ, (x^2 + y^3) % p = a % p :=
by
  sorry

end exists_x_y_mod_p_l737_737168


namespace math_problem_l737_737462

variable {x : ℂ}

theorem math_problem (h : x - 1/x = Complex.i * Real.sqrt 2) : 
  x^2187 - 1/x^2187 = Complex.i * Real.sqrt 2 := 
sorry

end math_problem_l737_737462


namespace opposite_of_neg_six_l737_737635

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_l737_737635


namespace area_of_isosceles_right_triangle_l737_737045

theorem area_of_isosceles_right_triangle 
  (a b : ℝ × ℝ)
  (h_a : a = (-1, 1))
  (OA OB : ℝ × ℝ)
  (h_OA : OA = a - b)
  (h_OB : OB = a + b)
  (h_iso_right : OA.1 * OB.1 + OA.2 * OB.2 = 0)
  (hab : abs a = abs b) :
  let area := 1 / 2 * abs (OA.1) * abs (OB.1)
  in area = 2 :=
by
  let a := (-1, 1)
  let b : ℝ × ℝ := sorry
  let OA := a - b
  let OB := a + b
  have h_a : a = (-1, 1) := rfl
  have h_OA : OA = a - b := rfl
  have h_OB : OB = a + b := rfl
  have h_iso_right : OA.1 * OB.1 + OA.2 * OB.2 = 0 := sorry
  have hab : abs a = abs b := sorry
  let area := 1 / 2 * abs (OA.1) * abs (OB.1)
  show area = 2
  sorry

end area_of_isosceles_right_triangle_l737_737045


namespace floor_neg_seven_fourths_l737_737797

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l737_737797


namespace jenny_change_l737_737540

def cost_per_page : ℝ := 0.10
def pages_per_essay : ℝ := 25
def num_essays : ℝ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℝ := 7
def amount_paid : ℝ := 40.00

theorem jenny_change : 
  let cost_of_printing := num_essays * pages_per_essay * cost_per_page in
  let cost_of_pens := num_pens * cost_per_pen in
  let total_cost := cost_of_printing + cost_of_pens in
  amount_paid - total_cost = 12.00 :=
by
  -- Definitions
  let cost_of_printing := num_essays * pages_per_essay * cost_per_page
  let cost_of_pens := num_pens * cost_per_pen
  let total_cost := cost_of_printing + cost_of_pens

  -- Proof
  sorry

end jenny_change_l737_737540


namespace range_of_a_l737_737612

theorem range_of_a (m n a : ℝ) (hz : ∃ z : ℂ, z = m + n * complex.I ∧ m < 0 ∧ n > 0 ∧ z * complex.conj z + 2 * complex.I * z = 8 + a * complex.I) :
    a ∈ set.Ico (-6 : ℝ) 0 :=
sorry

end range_of_a_l737_737612


namespace sum_primes_less_than_20_l737_737665

theorem sum_primes_less_than_20 : (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19) = 77 := sorry

end sum_primes_less_than_20_l737_737665


namespace hyperbola_properties_l737_737466

theorem hyperbola_properties (c a b : ℝ) (hecc : 2 * a = c) (hb : b^2 = a^2 - c^2) :
  (a = 2) ∧ (c = 4) →
  (∃ eqn : String, eqn = "x^2 / 4 - y^2 / 12 = 1") ∧
  (∃ asymptotes : String, asymptotes = "y = ±√3x") ∧
  (∃ d : ℝ, d = 2 * sqrt 3) :=
by
  intro ha hc
  split
  { exact ⟨"x^2 / 4 - y^2 / 12 = 1", rfl⟩ }
  split
  { exact ⟨"y = ±√3x", rfl⟩ }
  { exact ⟨2 * sqrt 3, rfl⟩ }

-- sorry at the end to skip proof since it is not considered

end hyperbola_properties_l737_737466


namespace measure_of_angle_BAC_l737_737577

theorem measure_of_angle_BAC 
  (A B C X Y Z : Type)
  (AX XYZ ZY YB BC : Real) 
  (angle_ABC : Real) 
  (h1 : AX = XYZ)
  (h2 : XYZ = ZY)
  (h3 : ZY = YB)
  (h4 : YB = BC)
  (h5 : angle_ABC = 150) :
  ∃ (t : Real), t = 15 ∧ ∠BAC = t :=
sorry

end measure_of_angle_BAC_l737_737577


namespace floor_sqrt_80_eq_8_l737_737898

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737898


namespace exists_zero_in_interval_l737_737484

noncomputable def f : ℝ → ℝ := λ x, Real.exp x - x - 2

theorem exists_zero_in_interval :
  (∃ c ∈ set.Ioo 1 2, f c = 0) :=
by {
  -- Given conditions
  have h_cont : continuous f := continuous_exp.sub continuous_id.sub continuous_const,
  have h1 : f 1 < 0,
    calc f 1 = Real.exp 1 - 1 - 2 : by rfl
         ... = Real.exp 1 - 3 : by ring
         ... < 0 : by norm_num,
  have h2 : f 2 > 0,
    calc f 2 = Real.exp 2 - 2 - 2 : by rfl
         ... = Real.exp 2 - 4 : by ring
         ... > 0 : by norm_num,

  have h_zero_in_Ioc : ∃ c ∈ set.Icc 1 2, f c = 0,
    from intermediate_value_Icc h_cont (le_of_lt h1) h2 rfl,
  
  cases h_zero_in_Ioc with c hc,
  cases hc with hc1 hc2,

  -- Refining the c found to be within the open interval (1, 2)
  refine ⟨c, ⟨hc1.1, hc1.2⟩, hc2⟩,
  sorry
}

end exists_zero_in_interval_l737_737484


namespace cube_surface_area_increase_cube_volume_increase_l737_737301

theorem cube_surface_area_increase (s : ℝ) :
  let original_surface_area := 6 * s ^ 2,
      new_edge_length := 1.6 * s,
      new_surface_area := 6 * (new_edge_length) ^ 2
  in (new_surface_area - original_surface_area) / original_surface_area * 100 = 156 := 
sorry

theorem cube_volume_increase (s : ℝ) :
  let original_volume := s ^ 3,
      new_edge_length := 1.6 * s,
      new_volume := (new_edge_length) ^ 3
  in (new_volume - original_volume) / original_volume * 100 = 309.6 := 
sorry

end cube_surface_area_increase_cube_volume_increase_l737_737301


namespace floor_sqrt_80_l737_737805

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737805


namespace floor_sqrt_80_l737_737806

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737806


namespace largest_multiple_of_7_negation_greater_than_neg_150_l737_737277

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l737_737277


namespace sum_faces_edges_vertices_triangular_prism_l737_737296

-- Given conditions for triangular prism:
def triangular_prism_faces : Nat := 2 + 3  -- 2 triangular faces and 3 rectangular faces
def triangular_prism_edges : Nat := 3 + 3 + 3  -- 3 top edges, 3 bottom edges, 3 connecting edges
def triangular_prism_vertices : Nat := 3 + 3  -- 3 vertices on the top base, 3 on the bottom base

-- Proof statement for the sum of the faces, edges, and vertices of a triangular prism
theorem sum_faces_edges_vertices_triangular_prism : 
  triangular_prism_faces + triangular_prism_edges + triangular_prism_vertices = 20 := by
  sorry

end sum_faces_edges_vertices_triangular_prism_l737_737296


namespace tetrahedron_area_inequality_l737_737185

-- Defining the tetrahedron and areas of its faces
structure Tetrahedron :=
  (A1 A2 A3 A4 : Prop)
  (S1 S2 S3 S4 : ℝ) -- the areas of the faces

-- Axiom stating the theorem we need to prove
axiom areas_lt_sum_of_others (T : Tetrahedron) (i : Fin 4) :
  let areas := [T.S1, T.S2, T.S3, T.S4]
  in areas[i] < areas[(i + 1) % 4] + areas[(i + 2) % 4] + areas[(i + 3) % 4]

-- The theorem statement
theorem tetrahedron_area_inequality (T : Tetrahedron) :
  ∀ i : Fin 4, let areas := [T.S1, T.S2, T.S3, T.S4]
  in areas[i] < areas[(i + 1) % 4] + areas[(i + 2) % 4] + areas[(i + 3) % 4] :=
by
  intro i
  exact areas_lt_sum_of_others T i

end tetrahedron_area_inequality_l737_737185


namespace range_of_m_l737_737130

variable (a b : ℝ)

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^3 - m ≤ a * x + b ∧ a * x + b ≤ x^3 + m) ↔ m ∈ Set.Ici (Real.sqrt 3 / 9) :=
by
  sorry

end range_of_m_l737_737130


namespace unique_fixed_point_l737_737058

noncomputable def midpoint (P Q : Point) : Point :=
  ⟨(P.x + Q.x)/2, (P.y + Q.y)/2⟩

noncomputable def transformation (A B C P : Point) : Point :=
  let Q := midpoint P A
  let R := midpoint Q B
  let P' := midpoint R C
  P'

theorem unique_fixed_point {A B C : Point} :
  ∃! P : Point, transformation A B C P = P :=
  sorry

end unique_fixed_point_l737_737058


namespace equal_charges_at_4_hours_l737_737181

-- Define the charges for both companies
def PaulsPlumbingCharge (h : ℝ) : ℝ := 55 + 35 * h
def ReliablePlumbingCharge (h : ℝ) : ℝ := 75 + 30 * h

-- Prove that for 4 hours of labor, the charges are equal
theorem equal_charges_at_4_hours : PaulsPlumbingCharge 4 = ReliablePlumbingCharge 4 :=
by
  sorry

end equal_charges_at_4_hours_l737_737181


namespace prob_hits_exactly_two_expectation_total_score_l737_737650

namespace ShooterProblem

-- Events and conditions
def hits_target_A : Event := sorry
def hits_target_B : Event := sorry

-- Probabilities
def prob_hits_A : ℝ := 3/4
def prob_hits_B : ℝ := 2/3

-- The first query: Probability of hitting exactly two shots
/-- The probability that the shooter hits exactly two shots is 5/16 -/
theorem prob_hits_exactly_two 
  (h_ind_A1_A2 : Independent (hits_target_A 1) (hits_target_A 2)) 
  (h_ind_A_B : Independent (hits_target_A) (hits_target_B)) :
  P (hits_target_A 1 ∧ hits_target_A 2 ∧ ¬hits_target_B ∨ 
     hits_target_A 1 ∧ ¬hits_target_A 2 ∧ hits_target_B ∨ 
     ¬hits_target_A 1 ∧ hits_target_A 2 ∧ hits_target_B) = 5/16 :=
sorry

-- The second query: Mathematical expectation of shooter's total score
def score (a1 a2 b : Bool) : ℝ :=
  (if a1 then 1 else 0) + (if a2 then 1 else 0) + (if b then 2 else 0)

/-- The expectation of the shooter's total score is 11/4 -/
theorem expectation_total_score :
  E[X] = 11/4 :=
sorry

end ShooterProblem

end prob_hits_exactly_two_expectation_total_score_l737_737650


namespace small_boxes_folded_l737_737103

theorem small_boxes_folded (hugo_folds_small_box_in : ℕ) (tom_folds_small_box_in : ℕ) (total_folding_time : ℕ) : 
  hugo_folds_small_box_in = 3 ∧ tom_folds_small_box_in = 4 ∧ total_folding_time = 7200 → 
  let hugo_small_boxes := total_folding_time / hugo_folds_small_box_in in
  let tom_small_boxes := total_folding_time / tom_folds_small_box_in in
  hugo_small_boxes + tom_small_boxes = 4200 :=
begin
  intros h,
  simp [h.1, h.2.1, h.2.2],
  let hugo_small_boxes := 7200 / 3,
  let tom_small_boxes := 7200 / 4,
  have h1 : hugo_small_boxes = 2400,
  { rw nat.div_eq_of_lt, refl, exact dec_trivial },
  have h2 : tom_small_boxes = 1800,
  { rw nat.div_eq_of_lt, refl, exact dec_trivial },
  rw [h1, h2],
  exact rfl,
end

end small_boxes_folded_l737_737103


namespace floor_sqrt_80_l737_737972

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737972


namespace stickers_unclaimed_l737_737368

theorem stickers_unclaimed (x : ℕ) :
  let total_pile := x in
  let al_takes := (4/9) * total_pile in
  let remaining_after_al := total_pile - al_takes in
  let bert_takes := (1/3) * remaining_after_al in
  let remaining_after_bert := remaining_after_al - bert_takes in
  let carl_takes := (2/9) * remaining_after_bert in
  let remaining_after_carl := remaining_after_bert - carl_takes in
  remaining_after_carl / total_pile = (230 / 243) :=
by
  sorry

end stickers_unclaimed_l737_737368


namespace solve_quadratic_l737_737206

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = 2 + Real.sqrt 11) ∧ (x2 = 2 - (Real.sqrt 11)) ∧ 
  (∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ x = x1 ∨ x = x2) := 
sorry

end solve_quadratic_l737_737206


namespace problem_statement_l737_737373

theorem problem_statement :
  let p q : Prop := sorry in
  let emo_proof_equiv := (∃ x : ℝ, x^2 + 2 * x ≤ 0) in
  let emo_proof_neg := (∀ x : ℝ, x^2 + 2 * x > 0) in
  let prop1 := "If p and q are two propositions, then 'both p and q are true' is a necessary but not sufficient condition for 'either p or q is true'" in
  let prop2 := "If p is: ∃ x ∈ 𝕉, x^2 + 2x ≤ 0, then ¬ p is: ∀ x ∈ 𝕉, x^2 + 2x > 0" in
  let prop3 := "If p is true and q is false, then p ∧ (¬ q) and (¬ p) ∨ q are both true" in
  let prop4 := "The contrapositive of the proposition 'If ¬ p, then q' is 'If p, then ¬ q'" in
  (prop1 = "incorrect") ∧ (prop2 = "correct") ∧ (prop3 = "incorrect") ∧ (prop4 = "incorrect") → 1 = 1 :=
begin
  sorry
end

end problem_statement_l737_737373


namespace floor_sqrt_80_l737_737936

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737936


namespace largest_multiple_of_7_gt_neg_150_l737_737265

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l737_737265


namespace additional_discount_percentage_l737_737354

theorem additional_discount_percentage
  (MSRP : ℝ)
  (p : ℝ)
  (d : ℝ)
  (sale_price : ℝ)
  (H1 : MSRP = 45.0)
  (H2 : p = 0.30)
  (H3 : d = MSRP - (p * MSRP))
  (H4 : d = 31.50)
  (H5 : sale_price = 25.20) :
  sale_price = d - (0.20 * d) :=
by
  sorry

end additional_discount_percentage_l737_737354


namespace inequality_product_ge_cyclic_product_l737_737162

theorem inequality_product_ge_cyclic_product 
  (n : ℕ) (h : n ≥ 2) (x : ℕ → ℝ) (h_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x i) :
  (∏ i in Finset.range n, (x (i + 1) + 1 / x (i + 1))) ≥ 
  (∏ i in Finset.range n, (x (i+1) + (1 / x ((i+1) % n + 1)))) :=
by
  sorry

end inequality_product_ge_cyclic_product_l737_737162


namespace count_valid_five_digit_numbers_l737_737713

def odd_digits := {1, 3, 5, 7, 9}
def is_valid_five_digit_number (n : Nat) : Prop :=
  let digits := Int.digits 10 n
  n >= 10000 ∧ n < 100000 ∧
  ∀ d ∈ digits, d ∈ odd_digits ∧
  (List.nodup digits) ∧
  (digits[3] > digits[2]) ∧
  (digits[3] > digits[4]) ∧
  (digits[1] > digits[2]) ∧
  (digits[1] > digits[0])

theorem count_valid_five_digit_numbers :
  Finset.card (Finset.filter is_valid_five_digit_number (Finset.range 100000)) = 16 :=
by 
  sorry

end count_valid_five_digit_numbers_l737_737713


namespace boat_travel_l737_737320

theorem boat_travel (T_against T_with : ℝ) (V_b D V_c : ℝ) 
  (hT_against : T_against = 10) 
  (hT_with : T_with = 6) 
  (hV_b : V_b = 12)
  (hD1 : D = (V_b - V_c) * T_against)
  (hD2 : D = (V_b + V_c) * T_with) :
  V_c = 3 ∧ D = 90 :=
by
  sorry

end boat_travel_l737_737320


namespace square_perimeter_l737_737359

theorem square_perimeter (side_length : ℕ) (h : side_length = 11) : 
  let P := 4 * side_length in P = 44 := 
by
  sorry

end square_perimeter_l737_737359


namespace difference_in_areas_l737_737547

def S1 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 2 + Real.log (x + y) / Real.log 2

def S2 (x y : ℝ) : Prop :=
  Real.log (3 + x ^ 2 + y ^ 2) / Real.log 2 ≤ 3 + Real.log (x + y) / Real.log 2

theorem difference_in_areas : 
  let area_S1 := π * 1 ^ 2
  let area_S2 := π * (Real.sqrt 13) ^ 2
  area_S2 - area_S1 = 12 * π :=
by
  sorry

end difference_in_areas_l737_737547


namespace floor_sqrt_80_l737_737832

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737832


namespace cubic_meter_to_cubic_centimeters_l737_737095

theorem cubic_meter_to_cubic_centimeters : (1 : ℝ)^3 = (100 : ℝ)^3 → (1 : ℝ)^3 = (1000000 : ℝ) :=
by
  intro h
  calc
    (1 : ℝ)^3 = (100 : ℝ)^3 : h
    ... = 1000000 : by norm_num

end cubic_meter_to_cubic_centimeters_l737_737095


namespace find_first_number_l737_737213

theorem find_first_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (x + 70 + 13) / 3 + 9 → x = 10 :=
by
  sorry

end find_first_number_l737_737213


namespace floor_sqrt_80_l737_737883

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737883


namespace sqrt_floor_eight_l737_737859

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737859


namespace tan_half_alpha_l737_737070

-- Define the conditions
variables {α : ℝ}
axiom sin_alpha : real.sin α = -24 / 25
axiom alpha_in_third_quadrant : ∃ k : ℤ, (2 * k * real.pi + real.pi < α ∧ α < 2 * k * real.pi + 3 * real.pi / 2)

-- Define the proof
theorem tan_half_alpha : real.tan (α / 2) = -4 / 3 :=
by sorry

end tan_half_alpha_l737_737070


namespace probability_of_one_correct_letter_l737_737235

noncomputable def derangements (n : ℕ) : ℕ :=
  if h : n = 0 then 1
  else if h : n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

theorem probability_of_one_correct_letter : 
  let total_ways := 6.factorial,
      choose_one := 6,
      derangements := derangements 5
  in (choose_one * derangements / total_ways : ℚ) = 11 / 30 := 
by
  -- Definitions to make the problem easier to follow
  have h1 : 6.factorial = 720 := by norm_num,
  have h2 : derangements 5 = 44 := by sorry,
  have h3 : (choose_one * derangements / total_ways : ℚ) = (6 * 44) / 720 := by norm_num,
  rw [h1, h2] at h3,
  norm_num at h3,
  exact h3

end probability_of_one_correct_letter_l737_737235


namespace floor_sqrt_80_l737_737822

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737822


namespace value_of_expression_l737_737048

theorem value_of_expression (x1 x2 : ℝ) 
  (h1 : x1 ^ 2 - 3 * x1 - 4 = 0) 
  (h2 : x2 ^ 2 - 3 * x2 - 4 = 0)
  (h3 : x1 + x2 = 3) 
  (h4 : x1 * x2 = -4) : 
  x1 ^ 2 - 4 * x1 - x2 + 2 * x1 * x2 = -7 := by
  sorry

end value_of_expression_l737_737048


namespace floor_sqrt_80_l737_737984

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737984


namespace floor_sqrt_80_l737_737980

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737980


namespace floor_neg_seven_fourths_l737_737798

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l737_737798


namespace k_greater_than_inv_e_l737_737439

theorem k_greater_than_inv_e (k : ℝ) (x : ℝ) (hx_pos : 0 < x) (hcond : k * (Real.exp (k * x) + 1) - (1 + (1 / x)) * Real.log x > 0) : 
  k > 1 / Real.exp 1 :=
sorry

end k_greater_than_inv_e_l737_737439


namespace tangent_line_at_one_eq_e_range_of_a_for_two_zeros_l737_737480

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp x) / (x ^ a)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - (Real.exp 1 * (x - a * Real.log x))

theorem tangent_line_at_one_eq_e :
  ∃ (x : ℝ), f 1 x = Real.exp 1 ∧ (f 1)'(x) = 0 := by
  sorry

theorem range_of_a_for_two_zeros :
  ∀ (a : ℝ), (a > 0) → 
    (∃ (c1 c2 : ℝ), 0 < c1 ∧ c1 < c2 ∧ g a c1 = 0 ∧ g a c2 = 0) →
      ((0 < a ∧ a < 1) ∨ a > 1) := by
  sorry

end tangent_line_at_one_eq_e_range_of_a_for_two_zeros_l737_737480


namespace floor_sqrt_80_eq_8_l737_737897

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737897


namespace product_value_l737_737764

def telescoping_product : ℕ → ℚ
| 0 := 1
| (n + 1) := (4 * (n + 1)) / (4 * (n + 1) + 4) * telescoping_product n

theorem product_value : telescoping_product 501 = 1 / 502 := by
  sorry

end product_value_l737_737764


namespace floor_sqrt_80_l737_737960

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737960


namespace floor_sqrt_80_l737_737956

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737956


namespace find_x_value_l737_737500

theorem find_x_value (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : xy / (x + y) = a) (h5 : xz / (x + z) = b) (h6 : yz / (y + z) = c)
  (h7 : x + y + z = abc) : 
  x = (2 * a * b * c) / (a * b + b * c + a * c) :=
sorry

end find_x_value_l737_737500


namespace floor_sqrt_80_l737_737945

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737945


namespace floor_sqrt_80_eq_8_l737_737895

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737895


namespace system_of_equations_solution_l737_737465

noncomputable def x0 : ℚ := 15 / 8
noncomputable def y0 : ℚ := 15 / 8
def B := 1 / x0 + 1 / y0

theorem system_of_equations_solution :
  (x0 / 3 + y0 / 5 = 1 ∧ x0 / 5 + y0 / 3 = 1) → B = 16 / 15 := by
  sorry

end system_of_equations_solution_l737_737465


namespace probability_divisible_by_3_of_prime_digit_two_digit_numbers_l737_737509

open Nat

def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def valid_two_digit_numbers : List ℕ := 
  [22, 23, 25, 27, 32, 33, 35, 37, 52, 53, 55, 57, 72, 73, 75, 77]

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem probability_divisible_by_3_of_prime_digit_two_digit_numbers :
  let favorable := valid_two_digit_numbers.filter is_divisible_by_3
  (favorable.length : ℚ) / (valid_two_digit_numbers.length : ℚ) = 5 / 16
:= sorry

end probability_divisible_by_3_of_prime_digit_two_digit_numbers_l737_737509


namespace part_I_part_II_l737_737482

noncomputable def f (a b x : ℝ) := Real.exp x + a * x^2 - b * x - 1
noncomputable def g (a b x : ℝ) := deriv (λ x : ℝ, f a b x) x

theorem part_I (a b : ℝ)
  (h1 : ∀ x ∈ set.Icc (0 : ℝ) 1, DerivableAt (λ x, f a b x) x) :
  (if a ≥ -1/2 then (∀ x ∈ set.Icc (0 : ℝ) 1, g a b x ≥ g a b 0) else true) ∧
  (if a ≤ - Real.exp (1/2) then (∀ x ∈ set.Icc (0 : ℝ) 1, g a b x ≤ g a b 1) else true) ∧
  (if -Real.exp (1/2) < a ∧ a < -1/2 then
  (let x := Real.log (-2 * a) in 
  x ∈ set.Ioo 0 1 ∧
  g a b (Real.log (-2 * a)) = 2 * a * Real.log (-2 * a) - 2 * a - b) 
  else true) :=
by sorry

theorem part_II (a b : ℝ) (h2 : f a b 1 = 0)
  (h3 : ∃ x ∈ set.Ioo (0 : ℝ) 1, f a b x = 0) :
  -1 < a ∧ a < 2 - Real.exp 1 :=
by sorry

end part_I_part_II_l737_737482


namespace average_sum_problem_l737_737514

theorem average_sum_problem (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 10) : ∃ sum : ℝ, sum = avg * n ∧ sum = 53 :=
by
  sorry

end average_sum_problem_l737_737514


namespace find_length_of_PQ_l737_737658

structure Circle (center : Type*) (radius : ℝ)

def external_tangent_point {C₁ C₂ : Circle} (C₁.radius = 2) (C₁.center = O₁)
  (C₂.radius = 3) (C₂.center = O₂) : Type* :=
P : Type*

def common_external_tangent_point (P: external_tangent_point O₁ O₂) : Type* :=
Q : Type* 

theorem find_length_of_PQ (O₁ O₂ : Type*) (h1 : Circle O₁ 2) (h2 : Circle O₂ 3) 
  (P : external_tangent_point O₁ O₂) (Q : common_external_tangent_point P) :
  PQ = 12 :=
sorry

end find_length_of_PQ_l737_737658


namespace distance_covered_l737_737717

-- Define the conditions
def speed_still_water : ℕ := 30   -- 30 kmph
def current_speed : ℕ := 6        -- 6 kmph
def time_downstream : ℕ := 24     -- 24 seconds

-- Proving the distance covered downstream
theorem distance_covered (s_still s_current t : ℕ) (h_s_still : s_still = speed_still_water) (h_s_current : s_current = current_speed) (h_t : t = time_downstream):
  (s_still + s_current) * 1000 / 3600 * t = 240 :=
by sorry

end distance_covered_l737_737717


namespace angle_NPC_eq_angle_MPC_angle_EPC_eq_angle_DPC_l737_737377

noncomputable theory
open_locale classical
open_locale big_operators

variables {A B C H M N O P D E : Type}
variables [Euclidean_space.{0} A] [Euclidean_space.{0} B] [Euclidean_space.{0} C] 
          [oH : Euclidean_space.{0} H] [Euclidean_space.{0} M] [Euclidean_space.{0} N]
          [Euclidean_space.{0} O] [Euclidean_space.{0} P] [Euclidean_space.{0} D] [Euclidean_space.{0} E]

-- Conditions
def altitude (triangle : triangle A B C) (alt : line_segment C P) : Prop := sorry
def intersect (point : H) (line1 : line_segment A H) (line2 : line_segment B H) (intersect_points : M × N) : Prop := sorry
def intersection (line1 : line_segment M N) (line2 : line_segment C P) (point : O) : Prop := sorry
def arbitrary_line_through_O (line : line_segment O D) (quad : quadrilateral C N H M) (intersect_points : D × E) : Prop := sorry

-- Questions
theorem angle_NPC_eq_angle_MPC (triangle : triangle A B C) (alt : line_segment C P) (point : H) (line1 : line_segment A H) (line2 : line_segment B H) 
  (intersect_points : M × N) (h_altitude : altitude triangle alt) 
  (h_intersect : intersect point line1 line2 intersect_points) :
  ∠(N P C) = ∠(M P C) :=
sorry

theorem angle_EPC_eq_angle_DPC (triangle : triangle A B C) (alt : line_segment C P) (point : H) (line1 : line_segment A H) (line2 : line_segment B H)
  (intersect_points : M × N) (intersection_point : O) (arbitrary_line : line_segment O D) (quad : quadrilateral C N H M)
  (intersect_points_2 : D × E) 
  (h_altitude : altitude triangle alt) 
  (h_intersect : intersect point line1 line2 intersect_points)
  (h_intersection : intersection (line_segment.mk M N) (line_segment.mk C P) intersection_point)
  (h_arbitrary : arbitrary_line_through_O arbitrary_line quad intersect_points_2) : 
  ∠(E P C) = ∠(D P C) :=
sorry

end angle_NPC_eq_angle_MPC_angle_EPC_eq_angle_DPC_l737_737377


namespace ways_to_color_grid_l737_737102

theorem ways_to_color_grid :
  let grid_width := 2008
  in ∃ (f : Fin 2 → Fin grid_width → Fin 3),
  (∀ i j, (i < 1 ∧ f i j ≠ f (i + 1) j) ∧ (j < grid_width - 1 ∧ f i j ≠ f i (j + 1))) →
  card (set_of f) = 6 * 3^(2007) :=
sorry

end ways_to_color_grid_l737_737102


namespace angle_inequalities_l737_737491

variables (α β : Plane) (m a b : Line)
variables (θ₁ θ₂ θ₃ : ℝ)

-- Hypotheses
def plane_intersection (h : α ∩ β = m) : Prop := True
def line_in_plane_a (h : a ⊆ α) : Prop := True
def line_perpendicular_to_m (h : a ⊥ m) : Prop := True
def line_in_plane_b (h : b ⊆ β) : Prop := True

-- Angles
def dihedral_angle (h : θ₁ = dihedral α β) : Prop := True
def angle_line_plane (h : θ₂ = angle_between_line_and_plane a β) : Prop := True
def angle_line_line (h : θ₃ = angle_between_lines a b) : Prop := True

theorem angle_inequalities
  (h1 : α ∩ β = m)
  (h2 : a ⊆ α)
  (h3 : a ⊥ m)
  (h4 : b ⊆ β)
  (h5 : θ₁ = dihedral α β)
  (h6 : θ₂ = angle_between_line_and_plane a β)
  (h7 : θ₃ = angle_between_lines a b) :
  θ₁ ≥ θ₂ ∧ θ₃ ≥ θ₂ :=
  sorry

end angle_inequalities_l737_737491


namespace smallest_d_value_l737_737356

theorem smallest_d_value : 
  ∃ d : ℝ, (d ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d + 5) = 4 * d) ∧ ∀ d' : ℝ, (d' ≥ 0) ∧ (dist (0, 0) (4 * Real.sqrt 5, d' + 5) = 4 * d') → (3 ≤ d') → d = 3 := 
by
  sorry

end smallest_d_value_l737_737356


namespace dihedral_angle_cosine_l737_737450

def point (α : Type) := (x y z : α)

def tetrahedron (α : Type) := (A B C S : point α)

def is_equilateral_triangle {α : Type} [field α] (A B C : point α) (side_length : α) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

def regular_tetrahedron {α : Type} [field α] (t : tetrahedron α) (side_length : α) (side_edge : α) : Prop :=
  is_equilateral_triangle t.A t.B t.C side_length ∧ 
  dist t.S t.A = side_edge ∧ dist t.S t.B = side_edge ∧ dist t.S t.C = side_edge

def divides_volume_into_two_equal_parts {α : Type} [field α] (plane : point α → α) (t : tetrahedron α) : Prop := sorry -- Requires detailed geometric definition

def cos_dihedral_angle {α : Type} [field α] (plane1 plane2 : point α → α) : α := sorry -- Law of Cosines

theorem dihedral_angle_cosine (A B C S : point ℝ)
  (t : tetrahedron ℝ := (A, B, C, S))
  (plane1 := λ p : point ℝ, p.x)
  (plane2 := λ p : point ℝ, p.x + p.y + p.z)
  (h_regular : regular_tetrahedron t 1 2)
  (h_division : divides_volume_into_two_equal_parts plane1 t) :
  cos_dihedral_angle plane1 plane2 = 2 * Real.sqrt 15 / 15 :=
sorry

end dihedral_angle_cosine_l737_737450


namespace find_perpendicular_tangent_line_l737_737026

noncomputable def tangent_line (c : ℝ) : affine.affineSubspaces ℝ := 
affine.affineSubspaces.affine Span ℝ ℝ (4, -1, c)

theorem find_perpendicular_tangent_line :
  ∃ (c : ℝ), (affine.affineSubspaces.affine Span ℝ ℝ.1 4) = (affine.affineSubspaces.affine Span ℝ ℝ (0,2))
  ∧ linear.linearMap ℝ (4, -1) = (2c - 4) * (linear.linearMap ℝ (2, -1))
  ∧ 4x - y - 2 = 0 :=
begin
  sorry
end

end find_perpendicular_tangent_line_l737_737026


namespace arithmetic_progression_sum_l737_737171

theorem arithmetic_progression_sum (n : ℕ) (a : ℕ → ℝ) (d : ℝ)
  (h_ap : ∀ i : ℕ, 1 ≤ i ∧ i < n → a (i + 1) = a i + d) :
  (∑ i in Finset.range (n - 1), 1 / (a i * a (i + 1))) = (n - 1) / (a 0 * a (n - 1)) :=
sorry

end arithmetic_progression_sum_l737_737171


namespace double_tangent_sine_l737_737529

theorem double_tangent_sine (t1 t2 : ℝ) (k : ℤ) :
  (cos t1 = cos t2 ∧ cos t1 ≠ 0) →
  (sin t1 - t1 * cos t1 = sin t2 - t2 * cos t2) →
  t2 = 2 * k * Real.pi - t1 →
  t1 = t1 - k * Real.pi → 
  ∃ p q : ℝ, (sin p = sin q) ∧ (p ≠ q) :=
by
  sorry

end double_tangent_sine_l737_737529


namespace sum_of_remainders_l737_737672

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l737_737672


namespace minimum_period_l737_737073

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem minimum_period (ω : ℝ) (hω : ω > 0) 
  (h : ∀ x1 x2 : ℝ, |f ω x1 - f ω x2| = 2 → |x1 - x2| = Real.pi / 2) :
  ∃ T > 0, ∀ x : ℝ, f ω (x + T) = f ω x ∧ T = Real.pi := sorry

end minimum_period_l737_737073


namespace problem_statement_l737_737473

variable (a b : ℝ)

def P1 := a > b → a + b > 0
def P1_converse := a + b > 0 → a > b

def P2 := a ≠ b → a^2 ≠ b^2
def P2_converse := a^2 ≠ b^2 → a ≠ b

def P3 := ∀ (x : point), on_angle_bisector x → equidistant_from_sides x
def P3_converse := ∀ (x : point), equidistant_from_sides x → on_angle_bisector x

def P4 := ∀ (p : parallelogram), diagonals_bisect_each_other p
def P4_converse := ∀ (p : parallelogram), diagonals_bisect_each_other_converse p

theorem problem_statement : 
  let propositions : list (Prop × Prop) := [(P1, P1_converse), (P2, P2_converse), (P3, P3_converse), (P4, P4_converse)] in
  list.count (λ p, p.1 ∧ p.2) propositions = 2 := 
begin
  sorry
end

end problem_statement_l737_737473


namespace correct_conclusion_l737_737644

-- Regression equation definition
def regression_equation (x : ℝ) : ℝ :=
  -10 * x + 200

-- The linear relationship between sales volume and selling price
-- Here we are proving the condition "When the selling price is 10 yuan/piece, the sales volume is around 100 pieces"

theorem correct_conclusion (x : ℝ) (y : ℝ) :
  (regression_equation x = y) → (x = 10) → (y ≈ 100) := sorry

end correct_conclusion_l737_737644


namespace liars_count_l737_737755

inductive PersonType
| knight : PersonType
| liar : PersonType
| weird : PersonType

def behavior (left right : PersonType) : Prop :=
  match left, right with
  | PersonType.knight, PersonType.liar => True
  | PersonType.knight, _ => False
  | PersonType.liar, PersonType.knight => True
  | PersonType.liar, PersonType.weird => True
  | PersonType.liar, _ => False
  | PersonType.weird, PersonType.liar => True
  | PersonType.weird, PersonType.knight => False
  | PersonType.weird, _ => _ 

theorem liars_count (table : list PersonType) (H : ∀ i, behavior (list.get table i) (list.get table ((i + 1) % 100))) :
  (table.count PersonType.liar = 0 ∨ table.count PersonType.liar = 50) :=
sorry

end liars_count_l737_737755


namespace value_of_y_l737_737105

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l737_737105


namespace area_of_triangle_l737_737093

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def heron_formula (a b c : ℝ) : ℝ :=
  let p := semi_perimeter a b c
  in Real.sqrt (p * (p - a) * (p - b) * (p - c))

theorem area_of_triangle
  (a b c : ℝ)
  (h_perimeter : a + b + c = 18)
  (h_ratio : a / b = 2 / 3 ∧ b / c = 3 / 4) :
  heron_formula a b c = 3 * Real.sqrt 15 := 
sorry

end area_of_triangle_l737_737093


namespace Vishal_investment_percentage_more_than_Trishul_l737_737245

-- Definitions from the conditions
def R : ℚ := 2400
def T : ℚ := 0.90 * R
def total_investments : ℚ := 6936

-- Mathematically equivalent statement to prove
theorem Vishal_investment_percentage_more_than_Trishul :
  ∃ V : ℚ, V + T + R = total_investments ∧ (V - T) / T * 100 = 10 := 
by
  sorry

end Vishal_investment_percentage_more_than_Trishul_l737_737245


namespace sum_A_C_l737_737367

theorem sum_A_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : B + C = 340) (h3 : C = 40) : A + C = 200 :=
by
  sorry

end sum_A_C_l737_737367


namespace expand_product_l737_737014

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := 
  sorry

end expand_product_l737_737014


namespace floor_of_sqrt_80_l737_737922

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737922


namespace find_line_equation_l737_737083

theorem find_line_equation (k x y x₁ y₁ x₂ y₂ : ℝ) (h_parabola : y ^ 2 = 2 * x) 
  (h_line_ny_eq : y = k * x + 2) (h_intersect_1 : (y₁ - (k * x₁ + 2)) = 0)
  (h_intersect_2 : (y₂ - (k * x₂ + 2)) = 0) 
  (h_y_intercept : (0,2) = (x,y))-- the line has y-intercept 2 
  (h_origin : (0,0) = (x, y)) -- origin 
  (h_orthogonal : x₁ * x₂ + y₁ * y₂ = 0): 
  y = -x + 2 :=
by {
  sorry
}

end find_line_equation_l737_737083


namespace sides_of_triangle_l737_737317

variable (a b c : ℝ)

theorem sides_of_triangle (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
  sorry

end sides_of_triangle_l737_737317


namespace S_subset_T_l737_737167

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

def S : Set (ℝ × ℝ) :=
  {p | let x := p.1; let y := p.2 in
       ∃ k : ℤ, x^2 - y^2 = k ∧ is_odd k}

def T : Set (ℝ × ℝ) :=
  {p | let x := p.1; let y := p.2 in
       sin (2 * Real.pi * x^2) - sin (2 * Real.pi * y^2) =
       cos (2 * Real.pi * x^2) - cos (2 * Real.pi * y^2)}

theorem S_subset_T : S ⊆ T :=
  by 
  sorry

end S_subset_T_l737_737167


namespace express_set_A_l737_737124

def A := {x : ℤ | -1 < abs (x - 1) ∧ abs (x - 1) < 2}

theorem express_set_A : A = {0, 1, 2} := 
by
  sorry

end express_set_A_l737_737124


namespace christmas_trees_in_each_box_l737_737413

theorem christmas_trees_in_each_box
  (T : ℕ)
  (pieces_of_tinsel_in_each_box : ℕ := 4)
  (snow_globes_in_each_box : ℕ := 5)
  (total_boxes : ℕ := 12)
  (total_decorations : ℕ := 120)
  (decorations_per_box : ℕ := pieces_of_tinsel_in_each_box + T + snow_globes_in_each_box)
  (total_decorations_distributed : ℕ := total_boxes * decorations_per_box) :
  total_decorations_distributed = total_decorations → T = 1 := by
  sorry

end christmas_trees_in_each_box_l737_737413


namespace success_vowel_last_l737_737096

open Finset

noncomputable def count_vowel_last_arrangements : ℕ :=
  (factorial 2) * (factorial 5 / ((factorial 3) * (factorial 2)))

theorem success_vowel_last : count_vowel_last_arrangements = 20 :=
by
  sorry

end success_vowel_last_l737_737096


namespace regular_polygon_radius_not_unique_triangle_l737_737682

theorem regular_polygon_radius_not_unique_triangle (P : Type) [regular_polygon P] (r : ℝ) : 
  ∀ (circumradius : P → ℝ), ¬ (circumradius P = r → is_triangle P) := by
  sorry

end regular_polygon_radius_not_unique_triangle_l737_737682


namespace smallest_insightful_marking_l737_737034

-- Define the set based on given conditions
def R (n : ℕ) : Set ℤ :=
  if even n then {k : ℤ | abs k < n / 2 + 1} \ {0} else {k : ℤ | abs k ≤ n / 2}

-- Given good and insightful markings
def is_good_marking (n : ℕ) (colors : ℕ → ℤ) : Prop :=
  -- every pair of connected balls have different colors
  ∀ i j, (connected i j) → (colors i ≠ colors j)

def is_insightful_marking (n m : ℕ) (colors_n : ℕ → ℤ) (colors_m : ℕ → ℤ) : Prop :=
  -- conditions for insightful marking
  ∀ i j, (white_string i j → colors_m i ≠ colors_m j) ∧ 
         (red_string i j → colors_n i + colors_n j ≠ 0)

-- Formalize the smallest m for insightful markings
noncomputable def min_insightful_mark (n : ℕ) : ℕ := 2 * n - 1

-- Theorem statement asserting the smallest m for insightful marking with required properties
theorem smallest_insightful_marking (n : ℕ) (hn : n ≥ 3) :
  ∀ m, is_good_marking n (colors n) → is_insightful_marking n (min_insightful_mark n) (colors (min_insightful_mark n)) :=
begin
  sorry, -- proof omitted
end

end smallest_insightful_marking_l737_737034


namespace floor_sqrt_80_l737_737938

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737938


namespace value_of_y_l737_737111

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l737_737111


namespace product_of_sequence_l737_737762

def sequence_term (k : Nat) : Rat :=
  (4 * k) / (4 * k + 4)

theorem product_of_sequence :
  ∏ k in Finset.range (501 + 1), sequence_term k = 1 / 502 := 
by
  sorry

end product_of_sequence_l737_737762


namespace floor_sqrt_80_l737_737941

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737941


namespace floor_sqrt_80_l737_737812

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737812


namespace solve_for_n_l737_737201

theorem solve_for_n (n : ℕ) (h : 3^n * 9^n = 81^(n - 12)) : n = 48 :=
sorry

end solve_for_n_l737_737201


namespace floor_sqrt_80_l737_737879

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737879


namespace find_number_l737_737727

theorem find_number (m : ℤ) (h1 : ∃ k1 : ℤ, k1 * k1 = m + 100) (h2 : ∃ k2 : ℤ, k2 * k2 = m + 168) : m = 156 :=
sorry

end find_number_l737_737727


namespace min_xy_solution_l737_737047

theorem min_xy_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 2 * x + 8 * y) :
  (x = 16 ∧ y = 4) :=
by
  sorry

end min_xy_solution_l737_737047


namespace floor_sqrt_80_l737_737867

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737867


namespace increasing_function_l737_737476

noncomputable def is_increasing_f (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (f a x ≤ f a y)

def f (a : ℝ) : (ℝ → ℝ) :=
  λ x, if x < 1 then (5 - a) * x - 3 else Real.log x / Real.log a

theorem increasing_function (a : ℝ) :
  (5 - a > 0) ∧ (a > 1) ∧ (5 - a - 3 ≤ 0) → a ∈ Icc 2 5 :=
sorry

end increasing_function_l737_737476


namespace Caden_total_money_l737_737382

-- Define the conditions as assumptions
variables (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (quarters : ℕ)

-- Given conditions
axiom pennies_eq_120 : pennies = 120
axiom nickels_eq_pennies_div_3 : nickels = pennies / 3
axiom dimes_eq_nickels_div_5 : dimes = nickels / 5
axiom quarters_eq_dimes_mul_2 : quarters = dimes * 2

-- Theorem to prove the total amount of money
theorem Caden_total_money : 
  0.01 * pennies + 0.05 * nickels + 0.10 * dimes + 0.25 * quarters = 8 := 
by
  sorry

end Caden_total_money_l737_737382


namespace largest_integer_log_sum_l737_737426

theorem largest_integer_log_sum: 
  (⌊(∑ k in finset.range 1007, real.log (k + 3) / (real.log (k + 2))) / (real.log 3)⌋) = 6 :=
by
  sorry

end largest_integer_log_sum_l737_737426


namespace product_of_sequence_l737_737760

def sequence_term (k : Nat) : Rat :=
  (4 * k) / (4 * k + 4)

theorem product_of_sequence :
  ∏ k in Finset.range (501 + 1), sequence_term k = 1 / 502 := 
by
  sorry

end product_of_sequence_l737_737760


namespace sqrt_floor_eight_l737_737857

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737857


namespace sqrt_floor_eight_l737_737852

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737852


namespace sidewalk_and_border_concrete_volume_l737_737739

/-
Question: How many cubic yards of concrete are needed, given that concrete must be ordered in whole cubic yards?
Conditions:
1. A straight concrete sidewalk with dimensions 3 feet wide, 90 feet long, and 4 inches thick.
2. A border of 6 inches wide and 2 inches thick is to be constructed around the sidewalk on both sides along the length only.
Answer: 4 cubic yards of concrete.
-/

noncomputable def cubic_yards_of_concrete_needed : ℕ := 4

theorem sidewalk_and_border_concrete_volume 
    (width : ℝ) (length : ℝ) (thickness : ℝ) (border_width : ℝ) (border_thickness : ℝ)
    (width = 3 / 3) -- 1 yard
    (length = 90 / 3) -- 30 yards
    (thickness = 4 / 36) -- 1/9 yard
    (border_width = 6 / 36) -- 1/6 yard
    (border_thickness = 2 / 36) -- 1/18 yard :
    cubic_yards_of_concrete_needed = 4 :=
by
  -- calculations (actual proofs would be here)
  sorry

end sidewalk_and_border_concrete_volume_l737_737739


namespace ellipse_equation_line_passes_fixed_point_l737_737571

noncomputable def ellipse_eq : String := 
  "The equation of ellipse with b = sqrt(3) and eccentricity e = 1/2 is (x^2)/4 + (y^2)/3 = 1."

theorem ellipse_equation (a b : ℝ) (h1 : b = sqrt 3) (h2 : (1/2) = sqrt (1 - (b^2) / (a^2))) : 
  (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1 :=
  sorry

theorem line_passes_fixed_point (P Q A : ℝ × ℝ) (h1 : A = (0, - sqrt 3))
  (l : ℝ → ℝ × ℝ) (hp : ∃ x, l x = P) (hq : ∃ y, l y = Q) (h3 : (sum_of_slopes (fst A) (snd A) (fst P) (snd P) (fst Q) (snd Q)) = 2) :
  ∃ k, ∀ x, l x = (sqrt 3, sqrt 3) :=
  sorry

end ellipse_equation_line_passes_fixed_point_l737_737571


namespace don_raise_l737_737409

variable (D R : ℝ)

theorem don_raise 
  (h1 : R = 0.08 * D)
  (h2 : 840 = 0.08 * 10500)
  (h3 : (D + R) - (10500 + 840) = 540) : 
  R = 880 :=
by sorry

end don_raise_l737_737409


namespace incorrect_statement_D_l737_737732

def temperatures : List ℤ := [-20, -10, 0, 10, 20, 30]
def speeds : List ℤ := [318, 324, 330, 336, 342, 348]

theorem incorrect_statement_D :
  let f := fun temp =>
    match temp with
    | -20 => 318
    | -10 => 324
    | 0   => 330
    | 10  => 336
    | 20  => 342
    | 30  => 348
    | _   => 0
  let distance := fun speed time => speed * time
  ∃ (T : ℤ) (H : T = 10) (speed := f T), ¬ distance speed 4 = 1304 :=
by
  sorry

end incorrect_statement_D_l737_737732


namespace largest_multiple_of_7_negated_gt_neg_150_l737_737255

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l737_737255


namespace floor_sqrt_80_l737_737888

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737888


namespace necessary_but_not_sufficient_period_l737_737318

def y (a x : ℝ) : ℝ := cos (a * x) ^ 2 - sin (a * x) ^ 2

theorem necessary_but_not_sufficient_period (a : ℝ) : 
  (∀ x, y 1 x = y 1 (x + π))
  → (∀ b, b ≠ 1 → ∃ x, (y b x ≠ y b (x + π)))
  → (a = 1) is necessary but not sufficient condition for function (λ x, y a x) to have period π :=
  sorry

end necessary_but_not_sufficient_period_l737_737318


namespace overall_gain_percentage_10point51_l737_737349

-- Defining the costs and revenues for each item based on given conditions
def costA := 10 * 8
def revenueA := 10 * 10
def costB := 7 * 15
def revenueB := 7 * 18
def costC := 5 * 22
def revenueC := 5 * 20

-- Total cost and revenue
def totalCost := costA + costB + costC
def totalRevenue := revenueA + revenueB + revenueC

-- Calculating profit
def totalProfit := totalRevenue - totalCost

-- Calculating the overall gain percentage
def overallGainPercentage := (totalProfit : ℝ) / totalCost * 100

-- The goal is to prove that the overall gain percentage is 10.51%
theorem overall_gain_percentage_10point51 :
  overallGainPercentage = 10.51 := by
  sorry

end overall_gain_percentage_10point51_l737_737349


namespace pyramid_volume_l737_737238

noncomputable def volume_of_pyramid (l : ℝ) : ℝ :=
  (l^3 / 24) * (Real.sqrt (Real.sqrt 2 + 1))

theorem pyramid_volume (l : ℝ) (α β : ℝ)
  (hα : α = π / 8)
  (hβ : β = π / 4)
  (hl : l = 6) :
  volume_of_pyramid l = 9 * Real.sqrt (Real.sqrt 2 + 1) := by
  sorry

end pyramid_volume_l737_737238


namespace line_does_not_pass_third_quadrant_l737_737503

theorem line_does_not_pass_third_quadrant (a b c x y : ℝ) (h_ac : a * c < 0) (h_bc : b * c < 0) :
  ¬(x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0) :=
sorry

end line_does_not_pass_third_quadrant_l737_737503


namespace magnitude_squared_of_complex_number_l737_737618

theorem magnitude_squared_of_complex_number 
  (z : ℂ)
  (h1 : z + complex.abs z = 3 + 12 * complex.I) :
  complex.abs z ^ 2 = 650.25 :=
sorry

end magnitude_squared_of_complex_number_l737_737618


namespace floor_of_sqrt_80_l737_737924

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737924


namespace physics_value_sum_l737_737630

def letter_value (n : ℕ) : ℤ :=
  match n % 9 with
  | 0 => 0
  | 1 => 2
  | 2 => 3
  | 3 => 2
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 8 => -2
  | _ => 0 -- unreachable

def letter_position (c : Char) : ℕ :=
  c.toNat - 'A'.toNat + 1

def word_value (word : String) : ℤ :=
  word.toList.map (letter_position ∘ letter_value).sum

theorem physics_value_sum : word_value "PHYSICS" = 1 := sorry

end physics_value_sum_l737_737630


namespace original_price_of_shoes_l737_737497

theorem original_price_of_shoes (P : ℝ) (h1 : 0.25 * P = 51) : P = 204 := 
by 
  sorry

end original_price_of_shoes_l737_737497


namespace floor_sqrt_80_l737_737863

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737863


namespace number_of_articles_l737_737126

-- Conditions
variables (C S : ℚ)
-- Given that the cost price of 50 articles is equal to the selling price of some number of articles N.
variables (N : ℚ) (h1 : 50 * C = N * S)
-- Given that the gain is 11.11111111111111 percent.
variables (gain : ℚ := 1/9) (h2 : S = C * (1 + gain))

-- Prove that N = 45
theorem number_of_articles (C S : ℚ) (N : ℚ) (h1 : 50 * C = N * S)
    (gain : ℚ := 1/9) (h2 : S = C * (1 + gain)) : N = 45 :=
by
  sorry

end number_of_articles_l737_737126


namespace remainder_when_divided_l737_737018
-- First, import the entire Mathlib library

-- Define the polynomial p(x) and the conditions given in the problem
noncomputable def p (x : ℝ) := (x+1) * (x-2)^2 * (a * x + b)

-- State the conditions on p(x)
def condition1 := p 2 = 6
def condition2 := p (-1) = 0

-- Define the remainder when p(x) is divided by (x+1)(x-2)^2
def remainder (x : ℝ) := 2 * x + 2

-- The formal theorem statement
theorem remainder_when_divided (a b : ℝ) (r : ℝ -> ℝ) :
  (∀ x, p x = (x+1) * (x-2)^2 * r x + a * x + b) →
  (p 2 = 6) →
  (p (-1) = 0) →
  (remainder x = 2 * x + 2) :=
  by
    sorry

end remainder_when_divided_l737_737018


namespace winning_candidate_votes_l737_737237

theorem winning_candidate_votes (V : ℕ) (h1 : 3136 + 7636 + V ≠ 0)
  (h2 : 0.51910714285714285 = (7636 : ℝ) / (3136 + 7636 + V)) : 7636 = 0.51910714285714285 * (3136 + 7636 + V) := by
  sorry

end winning_candidate_votes_l737_737237


namespace nail_pierces_one_cardboard_only_l737_737602

/--
Seryozha cut out two identical figures from cardboard. He placed them overlapping
at the bottom of a rectangular box. The bottom turned out to be completely covered. 
A nail was driven into the center of the bottom. Prove that it is possible for the 
nail to pierce one cardboard piece without piercing the other.
-/
theorem nail_pierces_one_cardboard_only 
  (identical_cardboards : Prop)
  (overlapping : Prop)
  (fully_covered_bottom : Prop)
  (nail_center : Prop) 
  : ∃ (layout : Prop), layout ∧ nail_center → nail_pierces_one :=
sorry

end nail_pierces_one_cardboard_only_l737_737602


namespace z_z_6_eq_0_l737_737319

def factorial_trailing_zeroes (n : ℕ) : ℕ :=
  (List.range (n // 5 + 1)).sum_by (fun k => n / 5 ^ (k + 1))

def z (n : ℕ) : ℕ := factorial_trailing_zeroes n

theorem z_z_6_eq_0 : z (z (6! )) = 0 := by
  have h1 : z 6! = 1 := by
    unfold z factorial_trailing_zeroes
    rw [Nat.factorial_succ, Nat.factorial, /, //]
    sorry

  rw [h1]
  have h2 : z 1 = 0 := by
    unfold z factorial_trailing_zeroes
    rw [/]
    sorry
  
  rw [h2]
  exact rfl

end z_z_6_eq_0_l737_737319


namespace sqrt_floor_eight_l737_737855

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737855


namespace floor_sqrt_80_l737_737942

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737942


namespace rate_percent_simple_interest_l737_737381

-- Given conditions
def principal : ℝ := 750
def amount : ℝ := 900
def time : ℝ := 4

-- Simple Interest Formula
def simple_interest (P R T : ℝ) := P * R * T / 100

-- Goal: Prove that the rate percent R is 5%
theorem rate_percent_simple_interest : 
  ∃ R : ℝ, amount = principal + simple_interest principal R time ∧ R = 5 := 
by 
  use 5
  split 
  sorry

end rate_percent_simple_interest_l737_737381


namespace min_triangle_perimeter_l737_737144

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    such that ∠C = max {∠A, ∠B, ∠C}, sin C = 1 + cos C * cos (A - B), and 2/a + 1/b = 1,
    prove that the minimum perimeter of the triangle is 10. -/
theorem min_triangle_perimeter 
  (A B C : ℝ) (a b c : ℝ) 
  (hC_max : C = max A (max B C))
  (h_sinc : sin C = 1 + cos C * cos (A - B))
  (h_sides : 2/a + 1/b = 1) :
  a + b + c = 10 :=
sorry

end min_triangle_perimeter_l737_737144


namespace quadratic_function_expression_quadratic_function_extrema_l737_737086

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem quadratic_function_expression (a b : ℝ) :
  (f 0 a b = 6) →
  (f 1 a b = 5) →
  f (x : ℝ) (-2) 6 = x^2 - 2 * x + 6 :=
by
  sorry

theorem quadratic_function_extrema :
  ∀ x ∈ Icc (-2 : ℝ) 2, 
      f x (-2) 6 ≥ 5 ∧ f x (-2) 6 ≤ 14 :=
by
  sorry

end quadratic_function_expression_quadratic_function_extrema_l737_737086


namespace largest_multiple_of_7_neg_greater_than_neg_150_l737_737286

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l737_737286


namespace at_least_one_neg_l737_737661

theorem at_least_one_neg (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
sorry

end at_least_one_neg_l737_737661


namespace monotonic_intervals_range_of_k_l737_737486

-- Part 1: Monotonic Intervals for k = e
theorem monotonic_intervals : 
  ∀ x : ℝ, (f(x) = exp x - Real.exp 1 * x) → 
    ((∀ x > 1, deriv f x > 0) ∧ (∀ x < 1, deriv f x < 0)) :=
begin
  sorry
end

-- Part 2: Range of k for f(|x|) to have exactly 4 zeros
theorem range_of_k (k : ℝ) (hk : k > 1) : 
  (∀ x : ℝ, f(|x|) = exp(abs x) - k * abs x) → 
    (4 = (card {x : ℝ | f(|x|) = 0}) → k ∈ (Real.exp 1, ∞)) :=
begin
  sorry
end

end monotonic_intervals_range_of_k_l737_737486


namespace count_pairs_l737_737224

theorem count_pairs (log2_5 : ℝ) (h_log2_5 : log2_5 = Real.logb 2 5) :
  (∃ (m n : ℕ), 1 ≤ m ∧ m ≤ 2140 ∧ 5^n < 2^m ∧ 2^m < 2^(m + 1) ∧ 2^(m + 1) < 5^(n + 1)) ∧
  (∃ k, log2_5 ≈ k / 900 ∧ 2080 < k < 2081) → ∑ (m : ℕ) in (finset.range 2140).filter (λ m, ∃ n : ℕ, 5^n < 2^m ∧ 2^m < 2^(m + 1) ∧ 2^(m + 1) < 5^(n + 1)), 1 = 1240
  :=
sorry

end count_pairs_l737_737224


namespace part_one_part_two_l737_737072

def f (a x : ℝ) := -x^2 + a * x + 4
def g (x : ℝ) := |x + 1| + |x - 1|

theorem part_one (a : ℝ) (h_a: a = 1) :
  { x : ℝ | f a x ≥ g x } = { x : ℝ | x ∈ set.Icc (-1) ((real.sqrt 17 - 1) / 2) } :=
sorry

theorem part_two :
  (-1 ≤ a ∧ a ≤ 1) ↔ ∀ x ∈ set.Icc (-1) 1, f a x ≥ g x :=
sorry

end part_one_part_two_l737_737072


namespace minimum_value_2_cos_x_minus_1_l737_737223

theorem minimum_value_2_cos_x_minus_1 : ∃ x ∈ ℝ, ∀ y = 2 * cos x - 1, y = -3 := by
sorry

end minimum_value_2_cos_x_minus_1_l737_737223


namespace problem_1_part1_problem_1_part2_problem_2_l737_737437

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + 2 + 2 * cos (x) ^ 2

theorem problem_1_part1 : (∃ T > 0, ∀ x, f (x + T) = f x) := sorry

theorem problem_1_part2 : (∀ k : ℤ, ∀ x ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), ∀ y ∈ Set.Icc (k * π - π / 3) (k * π + π / 6), x < y → f x > f y) := sorry

noncomputable def S_triangle (A B C : ℝ) (a b c : ℝ) : ℝ := 1 / 2 * b * c * sin A

theorem problem_2 :
  ∀ (A B C a b c : ℝ), f A = 4 → b = 1 → S_triangle A B C a b c = sqrt 3 / 2 →
    a^2 = b^2 + c^2 - 2 * b * c * cos A → a = sqrt 3 := sorry

end problem_1_part1_problem_1_part2_problem_2_l737_737437


namespace intersection_of_M_and_N_l737_737089

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-2, -1, 0} := sorry

end intersection_of_M_and_N_l737_737089


namespace probability_of_prime_pairs_l737_737659

def set_of_integers : finset ℕ := finset.Icc 1 25

def is_prime (n : ℕ) : Prop := nat.prime n

def prime_set : finset ℕ := set_of_integers.filter is_prime

def num_pairs (s : finset ℕ) : ℕ := s.card.choose 2

def probability_both_prime : ℚ := (num_pairs prime_set : ℚ) / (num_pairs set_of_integers : ℚ)

theorem probability_of_prime_pairs :
  probability_both_prime = 3 / 25 :=
sorry

end probability_of_prime_pairs_l737_737659


namespace floor_sqrt_80_l737_737970

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737970


namespace daniel_earnings_l737_737779

theorem daniel_earnings :
  let monday_fabric := 20
  let monday_yarn := 15
  let tuesday_fabric := 2 * monday_fabric
  let tuesday_yarn := monday_yarn + 10
  let wednesday_fabric := (1 / 4) * tuesday_fabric
  let wednesday_yarn := (1 / 2) * tuesday_yarn
  let total_fabric := monday_fabric + tuesday_fabric + wednesday_fabric
  let total_yarn := monday_yarn + tuesday_yarn + wednesday_yarn
  let fabric_cost := 2
  let yarn_cost := 3
  let fabric_earnings_before_discount := total_fabric * fabric_cost
  let yarn_earnings_before_discount := total_yarn * yarn_cost
  let fabric_discount := if total_fabric > 30 then 0.10 * fabric_earnings_before_discount else 0
  let yarn_discount := if total_yarn > 20 then 0.05 * yarn_earnings_before_discount else 0
  let fabric_earnings_after_discount := fabric_earnings_before_discount - fabric_discount
  let yarn_earnings_after_discount := yarn_earnings_before_discount - yarn_discount
  let total_earnings := fabric_earnings_after_discount + yarn_earnings_after_discount
  total_earnings = 275.625 := by
  {
    sorry
  }

end daniel_earnings_l737_737779


namespace min_value_of_some_expression_l737_737289

-- Define the absolute value function for reference
def abs (x : ℝ) : ℝ := if x < 0 then -x else x

-- Define the given expression composed of absolute values
def given_expression (x some_expression : ℝ) : ℝ := abs (x - 4) + abs (x + 7) + abs some_expression

-- Statement of the problem in Lean 4
theorem min_value_of_some_expression :
  (∀ x some_expression, abs (x - 4) + abs (x + 7) + abs some_expression ≥ 12) →
  (∃ some_expression, abs (given_expression (-3/2) some_expression) = 1) :=
  by
  sorry -- Proof to be filled in

end min_value_of_some_expression_l737_737289


namespace floor_sqrt_80_eq_8_l737_737910

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737910


namespace floor_sqrt_80_l737_737947

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737947


namespace not_equivalent_to_l737_737306

theorem not_equivalent_to (h1 : (5.25 * 10^-6) = 0.00000525)
  (h2 : (52.5 * 10^-7) = 0.00000525)
  (h3 : (525 * 10^-8) = 0.00000525)
  (h4 : (5 / 2 * 10^-6) ≠ 0.00000525)
  (h5 : (1 / 200000) = 5 * 10^-6) :
  (5 / 2 * 10^-6) ≠ 0.00000525 := 
sorry

end not_equivalent_to_l737_737306


namespace g_at_3_value_l737_737507

theorem g_at_3_value (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : g 1 = 7)
  (h2 : g 2 = 11)
  (h3 : ∀ x : ℝ, g x = c * x + d * x + 3) : 
  g 3 = 15 :=
by
  sorry

end g_at_3_value_l737_737507


namespace slices_remaining_is_correct_l737_737129

def slices_per_pizza : ℕ := 8
def pizzas_ordered : ℕ := 2
def slices_eaten : ℕ := 7
def total_slices : ℕ := slices_per_pizza * pizzas_ordered
def slices_remaining : ℕ := total_slices - slices_eaten

theorem slices_remaining_is_correct : slices_remaining = 9 := by
  sorry

end slices_remaining_is_correct_l737_737129


namespace solve_for_y_l737_737107

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l737_737107


namespace sum_of_angles_WYX_and_YZW_l737_737704

theorem sum_of_angles_WYX_and_YZW 
  (W X Y Z : Type) [circle_around_quadrilateral : circumscribed_quadrilateral W X Y Z]
  (angle_WZY angle_XWY : ℝ) 
  (h1 : angle_WZY = 50)
  (h2 : angle_XWY = 20) : 
  angle_WYX + angle_YZW = 110 := 
by
  sorry

end sum_of_angles_WYX_and_YZW_l737_737704


namespace minimal_total_time_single_tap_minimal_total_time_two_taps_l737_737648

-- Definitions for the problem
variable {T : Fin 10 → ℝ}

-- Problem (1): Single Water Tap
theorem minimal_total_time_single_tap (h_distinct: ∀ {a b : Fin 10}, a ≠ b → T a ≠ T b) :
  let sorted_times := Finset.sort (≤) (Finset.univ.image T) in 
  let S_min := ∑ i in Finset.range 10, (10 - i) * sorted_times[i] in
  S_min = ∑ i in Finset.range 10, (10 - (Fin.val i + 1)) * T i :=
sorry

-- Problem (2): Two Water Taps
theorem minimal_total_time_two_taps (h_distinct: ∀ {a b : Fin 10}, a ≠ b → T a ≠ T b) :
  let sorted_times := Finset.sort (≤) (Finset.univ.image T) in
  let group_A := sorted_times.take 5 in
  let group_B := sorted_times.drop 5 in
  let S_min_A := ∑ i in Finset.range 5, (5 - i) * group_A[i] in
  let S_min_B := ∑ i in Finset.range 5, (5 - i) * group_B[i] in
  let S_min := S_min_A + S_min_B in
  S_min = sorry :=
sorry

end minimal_total_time_single_tap_minimal_total_time_two_taps_l737_737648


namespace perfect_square_trinomial_l737_737113

theorem perfect_square_trinomial (a b : ℝ) :
  (∃ c : ℝ, 4 * (c^2) = 9 ∧ 4 * c = a - b) → 2 * a - 2 * b = 24 ∨ 2 * a - 2 * b = -24 :=
by
  sorry

end perfect_square_trinomial_l737_737113


namespace function_eq_const_half_l737_737021

variable (f : ℝ → ℝ)

theorem function_eq_const_half :
  (∀ x : ℝ, f(f(x) * f(1 - x)) = f(x)) →
  (∀ x : ℝ, f(f(x)) = 1 - f(x)) →
  ∀ x : ℝ, f(x) = 1 / 2 :=
by
  intros h1 h2 x
  sorry

end function_eq_const_half_l737_737021


namespace floor_sqrt_80_l737_737808

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737808


namespace greatest_perimeter_approx_l737_737609

-- Define the base and height of the isosceles triangle
def base : ℝ := 12
def height : ℝ := 10

-- Define the perimeter function for sub-triangles
def perimeter (k : ℕ) : ℝ :=
  1 + Real.sqrt (height^2 + k^2) + Real.sqrt (height^2 + (k + 1)^2)

-- Prove that the greatest perimeter among the triangles is 30.01 inches
theorem greatest_perimeter_approx :
  (∃ k, k < 12 ∧ perimeter k = 30.01) :=
sorry

end greatest_perimeter_approx_l737_737609


namespace floor_sqrt_80_l737_737967

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737967


namespace floor_sqrt_80_eq_8_l737_737838

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737838


namespace count_multiples_of_2310_l737_737098

/--
Given that \(2310 = 2 \times 3 \times 5 \times 7 \times 11\), 
and \(i\) and \(j\) are integers within \(0 \leq i < j \leq 99\), 
the number of positive integer multiples of \(2310\) 
that can be expressed in the form \(10^j - 10^i\) is 110.
-/
theorem count_multiples_of_2310 :
  let n := 2310 in
  let m := (2 * 3 * 5 * 7 * 11 : ℕ) in
  m = n ∧ (∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ 99 → 2310 ∣ (10^j - 10^i)) ↔ ∃ k : ℕ, k = 110 :=
by
  sorry

end count_multiples_of_2310_l737_737098


namespace greatest_integer_b_l737_737425

theorem greatest_integer_b (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 7 ≠ 0) → b ≤ 5 :=
by sorry

end greatest_integer_b_l737_737425


namespace largest_five_digit_product_l737_737250

theorem largest_five_digit_product
  (digs : List ℕ)
  (h_digit_count : digs.length = 5)
  (h_product : (digs.foldr (· * ·) 1) = 9 * 8 * 7 * 6 * 5) :
  (digs.foldr (λ a b => if a > b then 10 * a + b else 10 * b + a) 0) = 98765 :=
sorry

end largest_five_digit_product_l737_737250


namespace rearrangement_hours_l737_737583

theorem rearrangement_hours (name_length : ℕ) (rate : ℕ) (factorial : ℕ → ℕ)
  (factorial_6 : factorial name_length = 720)
  (rate_15 : rate = 15) :
  (factorial name_length) / (rate * 60) = 0.8 :=
by
  have rearrangements := factorial_6
  have rate_per_minute := rate_15
  sorry

end rearrangement_hours_l737_737583


namespace floor_sqrt_80_eq_8_l737_737914

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737914


namespace terminating_fraction_count_l737_737431

theorem terminating_fraction_count : 
  (Finset.filter (λ n : ℕ, n % 21 = 0) (Finset.Icc 1 1500)).card = 71 :=
by
  sorry

end terminating_fraction_count_l737_737431


namespace total_amount_is_2500_l737_737342

noncomputable def total_amount_divided (P1 : ℝ) (annual_income : ℝ) : ℝ :=
  let P2 := 2500 - P1
  let income_from_P1 := (5 / 100) * P1
  let income_from_P2 := (6 / 100) * P2
  income_from_P1 + income_from_P2

theorem total_amount_is_2500 : 
  (total_amount_divided 2000 130) = 130 :=
by
  sorry

end total_amount_is_2500_l737_737342


namespace find_expression_value_l737_737574

-- Define the points D and E as pairs of real numbers
def D : ℝ × ℝ := (30, 10)
def E : ℝ × ℝ := (6, 1)

-- Calculate the midpoint F
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def F : ℝ × ℝ := midpoint D E

-- The theorem to prove
theorem find_expression_value : (2 * F.1 - 4 * F.2) = 14 := by
  sorry

end find_expression_value_l737_737574


namespace find_max_a_l737_737458

def f (a x : ℝ) := a * x^3 - x

theorem find_max_a (a : ℝ) (h : ∃ t : ℝ, |f a (t + 2) - f a t| ≤ 2 / 3) :
  a ≤ 4 / 3 :=
sorry

end find_max_a_l737_737458


namespace percentage_return_on_investment_l737_737706

variable (r : ℝ) (F : ℝ) (P : ℝ)

theorem percentage_return_on_investment (h_r : r = 0.125) (h_F : F = 60) (h_P : P = 30) :
  ((r * F / P) * 100) = 25 :=
by
  rw [h_r, h_F, h_P]
  norm_num

#eval percentage_return_on_investment 0.125 60 30 rfl rfl rfl

end percentage_return_on_investment_l737_737706


namespace angle_acb_60_degrees_l737_737146

theorem angle_acb_60_degrees 
  (A B C D E F : Type) 
  [IsTriangle A B C]
  (h1 : AC = 2 * AB)
  (h2 : D ∈ LineSegment A B)
  (h3 : E ∈ LineSegment B C)
  (h4 : ∠BAE = ∠ACD)
  (h5 : Intersection F LineSegment AE LineSegment CD)
  (h6 : IsIsoscelesTriangle A F E)
  (h7 : AF = FE)
  (h8 : IsEquilateralTriangle C F E)
  (h9 : ∠CFE = 60) :
  ∠ACB = 60 := by
  sorry

end angle_acb_60_degrees_l737_737146


namespace ordered_pair_unique_solution_l737_737230

open Matrix

noncomputable def cross_product {α : Type*} [Field α] (u v : Matrix (Fin 3) (Fin 1) α) : Matrix (Fin 3) (Fin 1) α :=
  ![(u 1 0 * v 2 0 - u 2 0 * v 1 0),
    (u 2 0 * v 0 0 - u 0 0 * v 2 0),
    (u 0 0 * v 1 0 - u 1 0 * v 0 0)]

theorem ordered_pair_unique_solution (x y : ℝ)
  (h : cross_product ![(3 : ℝ), x, (-9 : ℝ)] ![(7 : ℝ), 5, y] = ![(0 : ℝ), 0, 0]) :
  (x, y) = (15/7, -21) :=
by
  sorry

end ordered_pair_unique_solution_l737_737230


namespace max_singers_l737_737703

theorem max_singers (y x m : ℕ) 
  (h1 : m = y * x + 4)
  (h2 : m = (y - 3) * (x + 2))
  (h3 : m < 150) :
  m ≤ 144 :=
begin
  sorry
end

end max_singers_l737_737703


namespace floor_sqrt_80_l737_737943

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737943


namespace simplify_expr1_simplify_expr2_l737_737603

-- (1) Simplify the expression: 3a(a+1) - (3+a)(3-a) - (2a-1)^2 == 7a - 10
theorem simplify_expr1 (a : ℝ) : 
  3 * a * (a + 1) - (3 + a) * (3 - a) - (2 * a - 1) ^ 2 = 7 * a - 10 :=
sorry

-- (2) Simplify the expression: ((x^2 - 2x + 4) / (x - 1) + 2 - x) / (x^2 + 4x + 4) / (1 - x) == -2 / (x + 2)^2
theorem simplify_expr2 (x : ℝ) (h : x ≠ 1) (h1 : x ≠ 0) : 
  (((x^2 - 2 * x + 4) / (x - 1) + 2 - x) / ((x^2 + 4 * x + 4) / (1 - x))) = -2 / (x + 2)^2 :=
sorry

end simplify_expr1_simplify_expr2_l737_737603


namespace subset_M_N_l737_737088

def is_element_of_M (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)

def is_element_of_N (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)

theorem subset_M_N : ∀ x, is_element_of_M x → is_element_of_N x :=
by
  sorry

end subset_M_N_l737_737088


namespace calc_expression_l737_737386

theorem calc_expression : 
  (Real.sqrt 16 - 4 * (Real.sqrt 2) / 2 + abs (- (Real.sqrt 3 * Real.sqrt 6)) + (-1) ^ 2023) = 
  (3 + Real.sqrt 2) :=
by
  sorry

end calc_expression_l737_737386


namespace area_percentage_change_l737_737220

variable (a b : ℝ)

def initial_area : ℝ := a * b

def new_length (a : ℝ) : ℝ := a * 1.35

def new_width (b : ℝ) : ℝ := b * 0.86

def new_area (a b : ℝ) : ℝ := (new_length a) * (new_width b)

theorem area_percentage_change :
    ((new_area a b) / (initial_area a b)) = 1.161 :=
by
  sorry

end area_percentage_change_l737_737220


namespace sqrt_floor_eight_l737_737854

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737854


namespace at_least_one_heart_or_king_l737_737340

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l737_737340


namespace sum_of_roots_abs_eqn_zero_l737_737202

theorem sum_of_roots_abs_eqn_zero (x : ℝ) (hx : |x|^2 - 4*|x| - 5 = 0) : (5 + (-5) = 0) :=
  sorry

end sum_of_roots_abs_eqn_zero_l737_737202


namespace chessboard_completion_l737_737161

theorem chessboard_completion (n : ℕ) (h : n ≥ 3) : 
  ∃ (f : ℕ × ℕ → ℕ), (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → f (i, j) = 1 ∨ f (i, j) = 2) ∧
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n-1 ∧ 1 ≤ j ∧ j ≤ n-2 → (f (i, j) + f (i, j+1) + f (i, j+2) + 
                                                     f (i+1, j) + f (i+1, j+1) + f (i+1, j+2)) % 2 = 0) ∧ 
  (∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n-2 ∧ 1 ≤ j ∧ j ≤ n-1 → (f (i, j) + f (i+1, j) + f (i+2, j) + 
                                                     f (i, j+1) + f (i+1, j+1) + f (i+2, j+1)) % 2 = 0) ∧ 
  (∃ s : finset (ℕ × ℕ → ℕ), s.card = 3^2 ∧ s ⊆ finset.univ ∧ 
    ∀ f ∈ s, ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ n-2 ∧ 1 ≤ j ∧ j ≤ n-2 → 
      (f (i, j) + f (i, j+1) + f (i, j+2) + 
       f (i+1, j) + f (i+1, j+1) + f (i+1, j+2)) % 2 = 0 ∧ 
      (f (i, j) + f (i+1, j) + f (i+2, j) + 
       f (i, j+1) + f (i+1, j+1) + f (i+2, j+1)) % 2 = 0) :=
sorry

end chessboard_completion_l737_737161


namespace tan_of_alpha_l737_737499

noncomputable theory

def trigonometric_identity (α : ℝ) : Prop :=
  2 * Real.sin α + Real.cos α = -Real.sqrt 5

theorem tan_of_alpha (α : ℝ) (h : trigonometric_identity α) : Real.tan α = 2 :=
sorry

end tan_of_alpha_l737_737499


namespace max_non_overlapping_areas_l737_737705

theorem max_non_overlapping_areas (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, k = 4 * n + 4 := 
sorry

end max_non_overlapping_areas_l737_737705


namespace inequality_solution_min_value_expression_l737_737697

-- Statement for the first problem (inequality)
theorem inequality_solution (x : ℝ) (h1 : x ≠ 3) :
(\frac{2}{3} ≤ x ∧ x < 2) ∨ (x > 2) → \frac{2x+1}{3-x} ≥ 1 := sorry

-- Statement for the second problem (minimum value)
theorem min_value_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  \frac{4}{x} + \frac{9}{y} ≥ 25 := sorry

end inequality_solution_min_value_expression_l737_737697


namespace find_symmetric_point_l737_737533

-- Define the function to calculate the midpoint between two points
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the symmetric point calculation
def symmetric_point (p q : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (2 * q.1 - p.1, 2 * q.2 - p.2, 2 * q.3 - p.3)

-- Given point P and the midpoint Q, find the symmetric point R
theorem find_symmetric_point :
  let P := (1, 2, -2)
  let Q := (-1, 0, 1)
  symmetric_point P Q = (-3, -2, 4) :=
by
  simp [P, Q, symmetric_point]
  sorry

end find_symmetric_point_l737_737533


namespace floor_sqrt_80_l737_737870

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737870


namespace largest_multiple_of_7_negated_gt_neg_150_l737_737256

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l737_737256


namespace jerry_added_action_figures_l737_737154

-- Given conditions
def initial_action_figures : ℕ := 8
def total_action_figures : ℕ := 10

-- Question to be proved
theorem jerry_added_action_figures :
  initial_action_figures + ?m_1 = total_action_figures → ?m_1 = 2 :=
sorry

end jerry_added_action_figures_l737_737154


namespace center_of_gravity_divides_segment_l737_737572

variables {a b : ℝ}

def trapezoid_ratio (a b : ℝ) : ℝ :=
  (2 * b + a) / (2 * a + b)

theorem center_of_gravity_divides_segment :
  ∀ (a b : ℝ), trapezoid_ratio a b = ((2 * b + a) / (2 * a + b)) :=
by 
  intros a b
  rw trapezoid_ratio
  sorry

end center_of_gravity_divides_segment_l737_737572


namespace polygon_number_of_sides_and_interior_sum_l737_737726

-- Given conditions
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)
def exterior_angle_sum : ℝ := 360

-- Proof problem statement
theorem polygon_number_of_sides_and_interior_sum (n : ℕ)
  (h : interior_angle_sum n = 3 * exterior_angle_sum) :
  n = 8 ∧ interior_angle_sum n = 1080 :=
by
  sorry

end polygon_number_of_sides_and_interior_sum_l737_737726


namespace problem_statement_l737_737440

noncomputable theory
open_locale classical

def f (x : ℝ) : ℝ := real.sqrt (3 * x ^ 2 + 4)

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
nat.iterate (λ g, λ x, real.sqrt(3 * (g x) ^ 2 + 4)) n f x

def sum_f_squared (n : ℕ) (x : ℝ) : ℝ :=
∑ k in finset.range n, (f_iter (k + 1) x) ^ 2

theorem problem_statement (n : ℕ) (hn : 0 < n) (x : ℝ) :
  sum_f_squared n x ≥ 3^(n+1) - 2 * n - 3 :=
sorry

end problem_statement_l737_737440


namespace largest_multiple_of_7_gt_neg_150_l737_737269

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l737_737269


namespace floor_sqrt_80_l737_737817

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737817


namespace shortest_minor_arc_line_eqn_l737_737347

theorem shortest_minor_arc_line_eqn :
  let M := (1, 2)
  let C := (2, 0)
  let l : ℝ → ℝ := λ x, (1 / 2) * (x - 1) + 2
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 9 → (2 * (y - 2) = x - 1 → x - 2 * y + 3 = 0) :=
by sorry

end shortest_minor_arc_line_eqn_l737_737347


namespace original_deal_games_l737_737364

noncomputable def price_per_game (total_price : ℝ) (number_of_games : ℕ) : ℝ :=
  total_price / number_of_games

noncomputable def number_of_games (total_price : ℝ) (price_per_game : ℝ) : ℕ :=
  Real.floor (total_price / price_per_game)

theorem original_deal_games :
  let price_2_games := 22.84
  let price_original_deal := 34.26
  let games_2 := 2
  let p := price_per_game price_2_games games_2
  number_of_games price_original_deal p = 3 :=
by
  let price_2_games := 22.84
  let price_original_deal := 34.26
  let games_2 := 2
  let p := price_per_game price_2_games games_2
  have h1 : p = 11.42 := by
    rw [price_per_game, Real.div_eq_mul_inv]
    norm_num
  have h2 : number_of_games price_original_deal p = 3 := by
    rw [number_of_games, Real.floor_eq]
    norm_num
  exact h2

end original_deal_games_l737_737364


namespace log_sum_equality_l737_737419

theorem log_sum_equality : ∀ (x y : ℕ), x = 50 → y = 30 → log 5 50 + log 5 30 = 3 + log 5 12 :=
by
  intros x y hx hy
  rw [hx, hy]
  -- Here we acknowledge the logarithmic identity
  have h_log_identity : log 5 (x * y) = log 5 x + log 5 y := sorry -- The proof of the identity is omitted for simplicity
  rw ← h_log_identity
  -- Simplify 50 * 30 = 1500
  have h1500 : 50 * 30 = 1500 := by norm_num
  rw h1500
  -- Now, the problem reduces to proving log 5 1500 = 3 + log 5 12, which might be a composite proof
  sorry

end log_sum_equality_l737_737419


namespace find_length_EC_l737_737091

-- Definitions related to the triangle and given conditions
variables (A B C D E : Type) [linear_ordered_field A] (angle measure seg : A)
variables (m_angle : A → A) (seg_len : A → A)

-- Conditions described in the problem
hypothesis angle_A : m_angle A = 45
hypothesis angle_B : m_angle B = 75
hypothesis seg_AC : seg_len AC = 16
hypothesis BD_perp_AC : ∃ B D, B ≠ D ∧ BD ⊥ AC
hypothesis CE_perp_AB : ∃ C E, C ≠ E ∧ CE ⊥ AB
hypothesis angle_relation : m_angle DBC = 2 * m_angle ECB

-- Question translated into a theorem statement,
-- that needs to be proven as per given conditions
theorem find_length_EC : seg_len EC = 7.75 := by
    sorry

end find_length_EC_l737_737091


namespace floor_sqrt_80_l737_737823

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737823


namespace floor_sqrt_80_eq_8_l737_737839

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737839


namespace jellybean_addition_l737_737652

-- Definitions related to the problem
def initial_jellybeans : ℕ := 37
def removed_jellybeans_initial : ℕ := 15
def added_jellybeans (x : ℕ) : ℕ := x
def removed_jellybeans_again : ℕ := 4
def final_jellybeans : ℕ := 23

-- Prove that the number of jellybeans added back (x) is 5
theorem jellybean_addition (x : ℕ) 
  (h1 : initial_jellybeans - removed_jellybeans_initial + added_jellybeans x - removed_jellybeans_again = final_jellybeans) : 
  x = 5 :=
sorry

end jellybean_addition_l737_737652


namespace total_fish_approximation_l737_737134

variable (TotalA TotalB TotalC : ℕ)

-- Given conditions
def conditions :=
  let prop1 := 180 = 90 + 60 + 30
  let prop2 := 100 = 45 + 35 + 20
  -- tagged fish proportions
  let proportionA := 4 / 45
  let proportionB := 3 / 35
  let proportionC := 1 / 20
  -- equations representing the total number of fish in the pond
  let eqA := (90 / TotalA : ℚ) = proportionA
  let eqB := (60 / TotalB : ℚ) = proportionB
  let eqC := (30 / TotalC : ℚ) = proportionC
  prop1 ∧ prop2 ∧ eqA ∧ eqB ∧ eqC

theorem total_fish_approximation (h : conditions TotalA TotalB TotalC) :
  TotalA + TotalB + TotalC = 2313 :=
sorry

end total_fish_approximation_l737_737134


namespace correct_option_l737_737455

variables {ξ : Type} [random_variable ξ]
variables {α β : Plane} {u v : vector}

/-- Definition for Variance -/
def D (x : ξ) : ℝ := sorry -- Placeholder for variance definition

/-- Given conditions -/
axiom D_xi_eq_1 : D(ξ) = 1
axiom dot_product_perpendicular : (u • v) = 0 → perpendicular α β

/-- Proposition definitions -/
def p := D(2 * ξ + 1) = 2
def q := (u • v) = 0 → perpendicular α β

/-- Correct option -/
theorem correct_option : (¬ p) ∧ q := 
by
  sorry

end correct_option_l737_737455


namespace largest_multiple_of_7_negated_gt_neg_150_l737_737253

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l737_737253


namespace largest_multiple_of_7_negation_greater_than_neg_150_l737_737276

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l737_737276


namespace parabola_vertex_sum_l737_737420

theorem parabola_vertex_sum (p q r : ℝ)
  (h1 : ∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 4 → y = p * x^2 + q * x + r)
  (h2 : ∀ y1 : ℝ, y1 = p * (1 : ℝ)^2 + q * (1 : ℝ) + r → y1 = 10)
  (h3 : ∀ y2 : ℝ, y2 = p * (-1 : ℝ)^2 + q * (-1 : ℝ) + r → y2 = 14) :
  p + q + r = 10 :=
sorry

end parabola_vertex_sum_l737_737420


namespace sum_a_b_l737_737118

theorem sum_a_b (a b : ℝ) (h : ∀ x, f(x) = (x + 5) / (x^2 + a*x + b)) 
  (h_asym : x = 2 ∨ x = -3) : a + b = -5 :=
sorry

end sum_a_b_l737_737118


namespace guards_sufficient_l737_737312

theorem guards_sufficient (n : ℕ) (h : n ≥ 3) : 
  ∃ guards : ℕ, guards = ⌊n / 3⌋ ∧ 
  (∀ (polygon : Type) [non_convex_polygon polygon n], 
  sufficient_guards polygon guards) :=
sorry

end guards_sufficient_l737_737312


namespace problem_statement_l737_737189

noncomputable def probability_real_root : ℝ :=
  let a := Icc 0 real.pi
  let b := Icc 0 real.pi
  ((∫ x in a, ∫ y in b, indicator (λ p, p.1^2 + p.2^2 ≥ real.pi) (x, y)) / (real.pi * real.pi))

theorem problem_statement : probability_real_root = 3 / 4 :=
sorry

end problem_statement_l737_737189


namespace intersection_square_area_l737_737346

noncomputable def intersection_area (side_large side_small rotation_angle segment_length : ℝ) : ℝ :=
2 * 2 -- area of the large square
- (4 * (1 / 2 * segment_length * sin (rotation_angle : ℝ) * 2)) -- subtract the area contribution from the rotations

theorem intersection_square_area :
∀ (side_large side_small rotation_angle segment_length : ℝ), side_large = 2 → side_small = 1 → rotation_angle = 50 → segment_length = 26 / 50 →
(p q : ℤ), 
p = 3203 → q = 1000 → (coprime p q) →
intersection_area side_large side_small rotation_angle segment_length = (p : ℝ) / q →
p + q = 4203 := 
by
  intros side_large side_small rotation_angle segment_length h_side_large h_side_small h_rotation_angle h_segment_length p q h_p h_q h_coprime h_area_eq
  sorry

end intersection_square_area_l737_737346


namespace geometric_solution_l737_737248

theorem geometric_solution (x y : ℝ) (h : x^2 + 2 * y^2 - 10 * x + 12 * y + 43 = 0) : x = 5 ∧ y = -3 := 
  by sorry

end geometric_solution_l737_737248


namespace quadrilateral_is_kite_l737_737510

-- Define a Quadrilateral structure
structure Quadrilateral (P Q R S : Type) :=
(diag_bisect : ∀ A B (segments : Set P), (segments A B) → A.dist * B.dist)
(diag_perpendicular : ∀ A B (segments : Set P), ∠AOB = 90°)
(adj_sides_equal : ∀ A B : Type, A ≡ B)

-- Our theorem statement, given the conditions, we conclude the quadrilateral is a kite
theorem quadrilateral_is_kite {P Q R S : Type} [Quadrilateral P Q R S]
  (h1 : diag_bisect P Q R S ∧ diag_perpendicular P Q R S ∧ adj_sides_equal P Q R S) :
  quadrilateral_type = kite :=
sorry

end quadrilateral_is_kite_l737_737510


namespace range_of_f_l737_737078

def f (x : ℝ) : ℝ :=
  x^2 + 2 * x - 3

theorem range_of_f : set.range (λ x, f x) = set.Icc (-3) 5 :=
by
  sorry

end range_of_f_l737_737078


namespace unique_solution_l737_737023

theorem unique_solution:
  ∃! (x y z : ℕ), 2^x + 9 * 7^y = z^3 ∧ x = 0 ∧ y = 1 ∧ z = 4 :=
by
  sorry

end unique_solution_l737_737023


namespace floor_sqrt_80_eq_8_l737_737903

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737903


namespace solution_f_2008_l737_737780

noncomputable def f : ℝ → ℝ := sorry

lemma smallest_period_of_f : ∀ x : ℝ, f (x + 4) = f x := sorry
lemma even_function_f : ∀ x : ℝ, f (-x) = f x := sorry
lemma f_on_interval : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 2) → f x = 2 - x := sorry 

theorem solution_f_2008 : f 2008 = 2 :=
by 
  have h_period : f 2008 = f 0 := smallest_period_of_f 2008
  have h_even : f 0 = 2 := even_function_f 0
  rw [h_period, h_even]
  exact h_even

end solution_f_2008_l737_737780


namespace triangle_circumscribed_angle_l737_737521

theorem triangle_circumscribed_angle {O A B C : Type*}
  [MetricSpace O] [MeasureSpace O] [measurable_space.angle_space O] [metric.has_circles O A B C]
  (h1 : IsCircumscribed O A B C)
  (h2 : MeasuredAngle B O C = 150)
  (h3 : MeasuredAngle A O B = 130) :
  MeasuredAngle A B C = 40 :=
sorry

end triangle_circumscribed_angle_l737_737521


namespace tan_x_neg7_l737_737436

theorem tan_x_neg7 (x : ℝ) (h1 : Real.sin (x + π / 4) = 3 / 5) (h2 : Real.sin (x - π / 4) = 4 / 5) : 
  Real.tan x = -7 :=
sorry

end tan_x_neg7_l737_737436


namespace floor_neg_seven_fourths_l737_737793

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l737_737793


namespace Prob_inequal_genders_l737_737180

noncomputable def binomial_probability (n k: ℕ) (p: ℚ): ℚ :=
  (nat.choose n k : ℚ) * p^k * (1-p)^(n-k)

theorem Prob_inequal_genders:
  let n := 8
  let p := 0.6
  let q := 0.4
  let prob_more_sons := (binomial_probability n 5 p) + (binomial_probability n 6 p) + (binomial_probability n 7 p) + (binomial_probability n 8 p)
  let prob_more_daughters := (binomial_probability n 0 p) + (binomial_probability n 1 p) + (binomial_probability n 2 p) + (binomial_probability n 3 p)
  in prob_more_sons + prob_more_daughters = 0.484 := 
by
  sorry

end Prob_inequal_genders_l737_737180


namespace hundreds_digit_of_factorial_subtraction_is_zero_l737_737249

theorem hundreds_digit_of_factorial_subtraction_is_zero :
  (∃ k : ℕ, 20! = 1000 * k) → (∃ m : ℕ, 25! = 1000 * m) → (∃ n : ℕ, (25! - 20!) = 1000 * n) :=
by
  intros h1 h2
  cases h1 with k hk
  cases h2 with m hm
  use m - k
  simp [hk, hm, factorial]
  ring -- This simplifies the expression and verifies the hundreds digit is zero.
  sorry -- skipping actual proof as instructed

end hundreds_digit_of_factorial_subtraction_is_zero_l737_737249


namespace postal_clerk_sold_for_4_80_l737_737728

-- Define the conditions and the problem
def total_stamps : Nat := 75
def stamp_5_cents : Nat := 5
def stamp_8_cents : Nat := 8
def count_5_cent_stamps : Nat := 40
def count_8_cent_stamps := total_stamps - count_5_cent_stamps

-- The theorem to prove
theorem postal_clerk_sold_for_4_80 (h1 : count_5_cent_stamps = 40)
                                    (h2 : count_8_cent_stamps = 35)
                                    (h3 : total_stamps = 75) :
  (40 * 5 + 35 * 8) / 100 = 4.80 :=
begin
  sorry
end

end postal_clerk_sold_for_4_80_l737_737728


namespace remainder_polynomial_division_l737_737031

theorem remainder_polynomial_division :
  let f : ℝ → ℝ := λ x, x^4 - 4 * x^2 + 7 * x - 8 in
  f 3 = 58 :=
by
  intro f
  sorry

end remainder_polynomial_division_l737_737031


namespace floor_sqrt_80_l737_737957

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737957


namespace sqrt_expression_equals_neg_two_l737_737633

theorem sqrt_expression_equals_neg_two : 
  (sqrt 6 + sqrt 2) * (sqrt 3 - 2) * sqrt (sqrt 3 + 2) = -2 :=
sorry

end sqrt_expression_equals_neg_two_l737_737633


namespace floor_sqrt_80_l737_737955

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737955


namespace incorrect_statement_C_l737_737483

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.exp x - a * x
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := real.exp x - a

theorem incorrect_statement_C (a x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) (h3 : x1 < x2) : ¬ (x1 * x2 > 1) :=
sorry

end incorrect_statement_C_l737_737483


namespace sqrt_floor_eight_l737_737850

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737850


namespace sqrt_floor_eight_l737_737851

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737851


namespace total_garbage_collected_correct_l737_737579

def Lizzie_group_collected : ℕ := 387
def other_group_collected : ℕ := Lizzie_group_collected - 39
def total_garbage_collected : ℕ := Lizzie_group_collected + other_group_collected

theorem total_garbage_collected_correct :
  total_garbage_collected = 735 :=
sorry

end total_garbage_collected_correct_l737_737579


namespace floor_sqrt_80_eq_8_l737_737842

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737842


namespace alice_zoe_difference_l737_737194

-- Definitions of the conditions
def AliceApples := 8
def ZoeApples := 2

-- Theorem statement to prove the difference in apples eaten
theorem alice_zoe_difference : AliceApples - ZoeApples = 6 := by
  -- Proof
  sorry

end alice_zoe_difference_l737_737194


namespace find_y_l737_737557

def diamond (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y : ∃ y : ℝ, diamond 4 y = 44 ∧ y = 48 / 7 :=
by
  sorry

end find_y_l737_737557


namespace floor_sqrt_80_eq_8_l737_737844

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737844


namespace solve_quadratic_eq_l737_737204

theorem solve_quadratic_eq {x : ℝ} : (x^2 - 4 * x - 7 = 0) ↔ (x = 2 + sqrt 11 ∨ x = 2 - sqrt 11) :=
sorry

end solve_quadratic_eq_l737_737204


namespace smallest_b_greater_than_three_l737_737293

theorem smallest_b_greater_than_three (b : ℕ) (h : b > 3) : 
  (∃ b, b = 5 ∧ (∃ n : ℕ, 4 * b + 5 = n^2)) :=
by
  use 5
  constructor
  · rfl
  · use 5
  sorry

end smallest_b_greater_than_three_l737_737293


namespace floor_of_sqrt_80_l737_737925

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737925


namespace right_triangle_area_l737_737137

theorem right_triangle_area (h : ℝ = 13) (a : angle = 30) : area = 21.125 * real.sqrt 3 := sorry

end right_triangle_area_l737_737137


namespace algebraic_expression_opposite_l737_737298

theorem algebraic_expression_opposite (a b x : ℝ) (h : b^2 * x^2 + |a| = -(b^2 * x^2 + |a|)) : a * b = 0 :=
by 
  sorry

end algebraic_expression_opposite_l737_737298


namespace max_possible_salary_l737_737723

theorem max_possible_salary (n : ℕ) (min_salary : ℕ) (total_salary : ℕ) (h1 : n = 18)
  (h2 : min_salary = 20000) (h3 : total_salary = 600000) :
  ∃ max_salary : ℕ, max_salary = 260000 ∧ 
  (∀ p : ℕ, p < n → min_salary <= s p) ∧
  (sum (λ p, s p) (range n) <= total_salary) :=
sorry

end max_possible_salary_l737_737723


namespace table_impossible_l737_737151

theorem table_impossible (a : ℕ → ℕ → ℤ)
    (row_sum_positive : ∀ i : ℕ, 0 ≤ i ∧ i < 5 → ∑ j in Finset.range 5, a i j > 0)
    (col_sum_negative : ∀ j : ℕ, 0 ≤ j ∧ j < 5 → ∑ i in Finset.range 5, a i j < 0) :
  false := by
  sorry

end table_impossible_l737_737151


namespace opposite_of_neg_six_is_six_l737_737637

theorem opposite_of_neg_six_is_six : ∃ x, -6 + x = 0 ∧ x = 6 := by
  use 6
  split
  · rfl
  · rfl

end opposite_of_neg_six_is_six_l737_737637


namespace rounding_45_26489_l737_737599

def number : ℝ := 45.26489

def round_to_tenth (x : ℝ) : ℝ := (Real.floor (10 * x) / 10) + if 10 * x % 1 >= 0.5 then 0.1 else 0

theorem rounding_45_26489 :
  round_to_tenth number = 45.3 :=
by
  sorry

end rounding_45_26489_l737_737599


namespace min_birthday_employees_wednesday_l737_737687

theorem min_birthday_employees_wednesday :
  ∀ (employees : ℕ) (n : ℕ), 
  employees = 50 → 
  n ≥ 1 →
  ∃ (x : ℕ), 6 * x + (x + n) = employees ∧ x + n ≥ 8 :=
by
  sorry

end min_birthday_employees_wednesday_l737_737687


namespace floor_sqrt_80_l737_737977

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737977


namespace floor_sqrt_80_l737_737830

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737830


namespace ellipse_properties_max_AB_distance_l737_737452

noncomputable theory
open Real

def ellipse_eq (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_properties :
  ∃ a b : ℝ, 0 < b ∧ b < a ∧ ellipse_eq 1 (sqrt 3 / 2) a b ∧ a^2 = b^2 + 3 ∧ ellipse_eq x y 2 1 :=
begin
  sorry
end

theorem max_AB_distance (m : ℝ) :
  |m| ≥ 1 →
  ∃ (f : ℝ → ℝ), (∀ (a b : ℝ), ellipse_eq a b 2 1 → 
  |f m| = (4 * sqrt 3 * |m|) / (m^2 + 1)) ∧
  ∀ (m : ℝ), (f m ≤ 2 ∧ (f 1 = sqrt 3 ∧ f (-1) = sqrt 3)) :=
begin
  sorry
end

end ellipse_properties_max_AB_distance_l737_737452


namespace nutmeg_amount_l737_737580

def amount_of_cinnamon : ℝ := 0.6666666666666666
def difference_cinnamon_nutmeg : ℝ := 0.16666666666666666

theorem nutmeg_amount (x : ℝ) 
  (h1 : amount_of_cinnamon = x + difference_cinnamon_nutmeg) : 
  x = 0.5 :=
by 
  sorry

end nutmeg_amount_l737_737580


namespace train_speed_kmh_l737_737746

theorem train_speed_kmh
  (length_train : ℕ)
  (crossing_time : ℕ)
  (length_bridge : ℕ)
  (total_distance : length_train + length_bridge = 375)
  (speed_mps : total_distance / crossing_time = 12.5) :
  (speed_kmh : 12.5 * 3.6 = 45) :
  (speed_kmh_proof : speed_kmh = 45) :=
sorry

end train_speed_kmh_l737_737746


namespace floor_sqrt_80_l737_737886

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737886


namespace b_arithmetic_b_formula_T_formula_l737_737143

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {c : ℕ → ℝ}

-- Define the sequence {a_n}
axiom a1 : a 1 = 2
axiom a_recurrence : ∀ n : ℕ, a n * a (n + 1) - 2 * a n + 1 = 0

-- Define the sequence {b_n}
def b (n : ℕ) := 2 / (a n - 1)

-- Question 1: Prove that {b_n} is an arithmetic sequence
theorem b_arithmetic : ∀ n : ℕ, b (n + 1) - b n = 2 := sorry

-- Proof that b_n = 2n
theorem b_formula : ∀ n : ℕ, b n = 2 * n := sorry

-- Define the sequence {c_n}
def c (n : ℕ) := if n = 1 then b 1 else 2 * 3^(n - 1)

-- Define the sum of nc_n
def T (n : ℕ) := ∑ i in Finset.range n, i * c (i + 1)

-- Question 2: Prove the formula for T_n
theorem T_formula : ∀ n : ℕ, T n = (n - 1/2) * 3^n + 1/2 := sorry

end b_arithmetic_b_formula_T_formula_l737_737143


namespace floor_of_sqrt_80_l737_737921

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737921


namespace problem_solution_l737_737674

def original_number : ℕ := 123456789

def expected_value_swapped (N : ℕ) : ℚ :=
  -- Placeholder for the correct expected value calculation logic
  555555555

theorem problem_solution :
  (expected_value_swapped original_number).numerator + (expected_value_swapped original_number).denominator % 10^6 = 555556 :=
by
  sorry

end problem_solution_l737_737674


namespace max_value_l737_737535

def a_n (n : ℕ) : ℤ := -2 * (n : ℤ)^2 + 29 * (n : ℤ) + 3

theorem max_value : ∃ n : ℕ, a_n n = 108 ∧ ∀ m : ℕ, a_n m ≤ 108 := by
  sorry

end max_value_l737_737535


namespace floor_neg_seven_fourths_l737_737795

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l737_737795


namespace floor_of_sqrt_80_l737_737929

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737929


namespace floor_sqrt_80_eq_8_l737_737893

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737893


namespace six_points_circle_l737_737653

-- Define the conditions and question
theorem six_points_circle
  (A B P Q R P' Q' R' : Type)
  -- (1) Triangles PAB, AQB and ABR are similar
  (similar_PAB_AQB_ABR : ∀ (a b c : Type), similar (triangle a b c))
  -- (2) Triangles P'AB, AQ'B and ABR' are similar and symmetric about the perpendicular bisector of AB
  (symm_similarity_P'_A_Q'_B_R' : symmetric_about (perpendicular_bisector A B) ∧
    ∀ (a b c' : Type), similar (triangle a b c'))
  : cyclical_on_same_circle P Q R P' Q' R :=
sorry

end six_points_circle_l737_737653


namespace series_proof_l737_737559

noncomputable def series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / (b ^ (n + 1))

noncomputable def transformed_series_sum (a b : ℝ) : ℝ :=
  ∑' (n : ℕ), a / ((a + 2 * b) ^ (n + 1))

theorem series_proof (a b : ℝ)
  (h1 : series_sum a b = 7)
  (h2 : a = 7 * (b - 1)) :
  transformed_series_sum a b = 7 * (b - 1) / (9 * b - 8) :=
by sorry

end series_proof_l737_737559


namespace prob_distribution_ξ_prob_event_C_l737_737741

noncomputable def P_A : Real := 2 / 3
noncomputable def P_B : Real := 3 / 4

-- Definition of probabilities for each value of ξ
noncomputable def P_ξ_0 : Real := (1 - P_A) * (1 - P_B)
noncomputable def P_ξ_1 : Real := (1 - P_A) * P_B + P_A * (1 - P_B)
noncomputable def P_ξ_2 : Real := P_A * P_B

-- Expected value of ξ
noncomputable def E_ξ : Real := 0 * P_ξ_0 + 1 * P_ξ_1 + 2 * P_ξ_2

-- Definition of η
noncomputable def η (ξ : Nat) : Nat := if ξ = 0 then 4 else if ξ = 1 then 0 else 4

-- Definition of event C
def event_C : Real := η 2 * P_ξ_2 + η 0 * P_ξ_0

-- Theorems to prove
theorem prob_distribution_ξ :
  P_ξ_0 = 1 / 12 ∧ P_ξ_1 = 5 / 12 ∧ P_ξ_2 = 1 / 2 ∧ E_ξ = 17 / 12 := by
  sorry

theorem prob_event_C :
  event_C = 7 / 12 := by
  sorry

end prob_distribution_ξ_prob_event_C_l737_737741


namespace floor_sqrt_80_l737_737825

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737825


namespace find_a_proof_l737_737625

open Real

def point1 : ℝ × ℝ := (-3, 4)
def point2 : ℝ × ℝ := (2, -1)
def direction_vector : ℝ × ℝ := (point2.1 - point1.1, point2.2 - point1.2)
def adjusted_direction_vector : ℝ × ℝ := let scalar := -2 / direction_vector.2
                                         in (scalar * direction_vector.1, scalar * direction_vector.2)

-- The condition states the form of the direction vector
def condition (v : ℝ × ℝ) : Prop := v = ⟨a, 2⟩   -- vector must have y-component 2
def find_a : Prop := adjusted_direction_vector = ⟨-2, 2⟩   -- Therefore, a = -2

-- Proof obligation: show that our adjusted vector matches the required form with a = -2
theorem find_a_proof (a : ℝ) : find_a ↔ condition ⟨a, 2⟩ := sorry

end find_a_proof_l737_737625


namespace simplify_expression_l737_737694

theorem simplify_expression :
  (∛125) - ((-Real.sqrt 3)^2) + (1 + (1 / Real.sqrt 2) - Real.sqrt 2) * Real.sqrt 2 - (-1)^2023 = 2 + Real.sqrt 2 :=
by sorry

end simplify_expression_l737_737694


namespace lends_bicycle_time_l737_737505

def lends_bicycle (choc_bars bonbons : ℕ) : ℝ :=
  (choc_bars * 1.5) + (bonbons * (1 / 6))

theorem lends_bicycle_time :
  lends_bicycle 1 3 = 2 :=
by
  sorry

end lends_bicycle_time_l737_737505


namespace floor_sqrt_80_eq_8_l737_737837

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737837


namespace range_a_l737_737006

theorem range_a (a : ℝ) :
  (∀ x y : ℝ, x ≠ 0 → |x + x⁻¹| ≥ |a - 2| + sin y) ↔ (1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_a_l737_737006


namespace triangle_similarity_property_l737_737376

-- Let P, B, A, O, C, D, F be points in a Euclidean plane, with O being the center of a circle, and PBA a secant that passes through the center O.
-- Additionally, let CD be a chord intersecting PA at point F, with given conditions:
-- PB = 2, OA = 2, and triangles COF and PDF are similar.
-- We need to prove that PF = 3 under these conditions.

variable {P B A O C D F : Type}
variable [Field P] [MetricSpace P O]
variable {PB OA : P}
variable (CO DF PF : P)
variable (tri_CO_F tri_PD_F : Triangle P)

-- given that missing trianlge definitions (COF and PDF) a part of metric space is analogous to Euclidean plane embedded in metric space
def Triangle.similar (Δ1 Δ2 : Triangle P) : Prop := sorry  -- similarity definition placeholder

theorem triangle_similarity_property (h_sim: Triangle.similar tri_CO_F tri_PD_F)
    (h_PB: PB = 2)
    (h_OA: OA = 2):
  PF = 3 :=
by
  sorry

end triangle_similarity_property_l737_737376


namespace area_of_rectangular_plot_l737_737219

theorem area_of_rectangular_plot (B L : ℕ) (h1 : L = 3 * B) (h2 : B = 18) : L * B = 972 := by
  sorry

end area_of_rectangular_plot_l737_737219


namespace no_real_roots_range_l737_737085

theorem no_real_roots_range (a : ℝ) : (¬ ∃ x : ℝ, x^2 + a * x - 4 * a = 0) ↔ (-16 < a ∧ a < 0) := by
  sorry

end no_real_roots_range_l737_737085


namespace probability_heart_or_king_l737_737333

theorem probability_heart_or_king :
  let total_cards := 52
  let hearts := 13
  let kings := 4
  let overlap := 1
  let unique_hearts_or_kings := hearts + kings - overlap
  let non_hearts_or_kings := total_cards - unique_hearts_or_kings
  let p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  let p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  let p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  p_at_least_one_heart_or_king = 88 / 169 :=
by
  have total_cards := 52
  have hearts := 13
  have kings := 4
  have overlap := 1
  have unique_hearts_or_kings := hearts + kings - overlap
  have non_hearts_or_kings := total_cards - unique_hearts_or_kings
  have p_non_heart_or_king := (non_hearts_or_kings : ℚ) / (total_cards : ℚ)
  have p_non_heart_or_king_twice := p_non_heart_or_king * p_non_heart_or_king
  have p_at_least_one_heart_or_king := 1 - p_non_heart_or_king_twice
  show p_at_least_one_heart_or_king = 88 / 169
  sorry

end probability_heart_or_king_l737_737333


namespace floor_sqrt_80_l737_737865

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737865


namespace only_C_is_quadratic_l737_737676

def is_quadratic_equation (eq : String) : Prop :=
  match eq with
  | "x^2 - 13 = 0" => True    -- Recognized as the only quadratic equation in this context
  | _ => False                 -- All other options are not considered quadratic

theorem only_C_is_quadratic :
  ∀ eq, eq = "A: x^2 + y = 11" ∨ eq = "B: x^2 - 1/x = 1" ∨ eq = "C: x^2 - 13 = 0" ∨ eq = "D: 2x + 1 = 0" → 
  eq = "C: x^2 - 13 = 0" ↔ is_quadratic_equation "x^2 - 13 = 0" :=
by
  intro eq H
  cases H with
  | inl H1 =>
    rw [H1]
    exact IffFalseIntro _
  | inr H1 =>
    cases H1 with
    | inl H2 =>
      rw [H2]
      exact IffFalseIntro _
    | inr H2 =>
      cases H2 with
      | inl H3 =>
        rw [H3]
        exact IffTrueIntro _
      | inr H3 =>
        rw [H3]
        exact IffFalseIntro _
where
  IffFalseIntro {P : Prop} (z : P → False) : P ↔ False := ⟨z, False.elim⟩
  IffTrueIntro {P : Prop} (h : P) : P ↔ True := ⟨fun _ => trivial, fun _ => h⟩

#eval only_C_is_quadratic "A: x^2 + y = 11" (Or.inl rfl) -- Should be false
#eval only_C_is_quadratic "B: x^2 - 1/x = 1" (Or.inr (Or.inl rfl)) -- Should be false
#eval only_C_is_quadratic "C: x^2 - 13 = 0" (Or.inr (Or.inr (Or.inl rfl))) -- Should be true
#eval only_C_is_quadratic "D: 2x + 1 = 0" (Or.inr (Or.inr (Or.inr rfl))) -- Should be false

end only_C_is_quadratic_l737_737676


namespace area_of_triangle_l737_737468

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem area_of_triangle :
  let F1 := (-2 : ℝ, 0 : ℝ) in
  let F2 := (2 : ℝ, 0 : ℝ) in
  let h : conic_hyperbola := conic_hyperbola.mk 1 3 in
  ∃ P : ℝ × ℝ, h.in_hyperbola P ∧ 3 * distance P F1 = 4 * distance P F2 → 
  1 / 2 * distance P F1 * distance P F2 * real.sin (real.arccos ((distance P F1) ^ 2 + (distance P F2) ^ 2 - 16) / (2 * distance P F1 * distance P F2)) = 3 * real.sqrt 15 := 
by 
  -- Proof omitted
  sorry

end area_of_triangle_l737_737468


namespace floor_sqrt_80_l737_737881

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737881


namespace seq_inequality_l737_737645

noncomputable def seq (n : ℕ) : ℕ
| 1 => 1
| (2 * k) => seq (2 * k - 1) + seq k
| (2 * k + 1) => seq (2 * k)

theorem seq_inequality (n : ℕ) (h : 0 < n) :
  seq (2^n) > 2^(n^2 / 4) :=
sorry

end seq_inequality_l737_737645


namespace sqrt_floor_eight_l737_737847

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737847


namespace sum_local_values_2345_l737_737664

theorem sum_local_values_2345 : 
  let n := 2345
  let digit_2_value := 2000
  let digit_3_value := 300
  let digit_4_value := 40
  let digit_5_value := 5
  digit_2_value + digit_3_value + digit_4_value + digit_5_value = n := 
by
  sorry

end sum_local_values_2345_l737_737664


namespace infinite_nested_sqrt_six_l737_737212

theorem infinite_nested_sqrt_six : ∀ (m : ℝ), (m > 0) ∧ (m^2 = 6 + m) → m = 3 :=
by
  intro m
  intros h
  cases h with hm_pos hm_eq
  sorry

end infinite_nested_sqrt_six_l737_737212


namespace count_real_numbers_a_with_integer_roots_l737_737039

def integer_roots_count : Nat := 15

theorem count_real_numbers_a_with_integer_roots :
  ∀ a : ℝ, (∃ r s : ℤ, r * s = 12 * a ∧ r + s = -a) ↔ a ∈ (Finset.range 15).map (λ k, (k.succ : ℝ)) := sorry

end count_real_numbers_a_with_integer_roots_l737_737039


namespace largest_multiple_of_7_negation_greater_than_neg_150_l737_737281

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l737_737281


namespace domain_of_f_f_is_odd_l737_737076

noncomputable def f (a : ℝ) (x : ℝ) := log a (x + 1) - log a (1 - x)

theorem domain_of_f {a : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  ∀ x : ℝ, (f a x ≠ f a x) ↔ -1 < x ∧ x < 1 :=
sorry

theorem f_is_odd {a : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  ∀ x : ℝ, f a (-x) = -f a (x) :=
sorry

end domain_of_f_f_is_odd_l737_737076


namespace john_age_l737_737546

/-
Problem statement:
John is 24 years younger than his dad. The sum of their ages is 68 years.
We need to prove that John is 22 years old.
-/

theorem john_age:
  ∃ (j d : ℕ), (j = d - 24 ∧ j + d = 68) → j = 22 :=
by
  sorry

end john_age_l737_737546


namespace pages_per_book_l737_737601

theorem pages_per_book (words_per_minute : ℕ) (words_per_page : ℕ) (hours_reading : ℕ) (num_books : ℕ) :
  words_per_minute = 40 →
  words_per_page = 100 →
  hours_reading = 20 →
  num_books = 6 →
  (words_per_minute * hours_reading * 60) / (words_per_page * num_books) = 80 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end pages_per_book_l737_737601


namespace floor_sqrt_80_l737_737958

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737958


namespace sum_remainder_l737_737669

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l737_737669


namespace floor_neg_seven_fourths_l737_737796

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l737_737796


namespace floor_sqrt_80_l737_737864

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737864


namespace right_angled_isosceles_triangle_can_be_cut_l737_737004

structure RightAngledIsoscelesTriangle :=
(base : ℝ)
(height : ℝ)
(right_angle : ∃ A B C, ∠ A B C = 90 ∧ A B = B C)

def can_cut_into_four_identical_smaller_corners (triangle : RightAngledIsoscelesTriangle) : Prop :=
  ∃ (smaller_triangles : Finset RightAngledIsoscelesTriangle),
    smaller_triangles.card = 4 ∧
    ∀ t ∈ smaller_triangles, 
      t.base = triangle.base / 2 ∧
      t.height = triangle.height / 2 ∧
      ∀ A' B' C', ∠ A' B' C' = 90 ∧ A' B' = B' C'

theorem right_angled_isosceles_triangle_can_be_cut :
  ∀ (triangle : RightAngledIsoscelesTriangle),
    can_cut_into_four_identical_smaller_corners triangle :=
by
  sorry

end right_angled_isosceles_triangle_can_be_cut_l737_737004


namespace floor_sqrt_80_l737_737986

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737986


namespace floor_sqrt_80_l737_737952

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737952


namespace number_of_puppies_l737_737094

variable (P : ℕ)
variable (cats : ℕ)
variable (cat_weight puppy_weight cats_total_weight puppies_total_weight : ℝ)

def total_weight_of_cats (cats : ℕ) (cat_weight : ℝ) : ℝ :=
  cats * cat_weight

noncomputable def total_weight_of_puppies (P : ℕ) (puppy_weight : ℝ) : ℝ :=
  P * puppy_weight

axiom cats_total_weight_given : cats_total_weight = total_weight_of_cats 14 2.5
axiom weight_difference : cats_total_weight = puppies_total_weight + 5
axiom puppies_total_weight_30 : puppies_total_weight = 30
axiom puppy_weight_given : puppy_weight = 7.5

theorem number_of_puppies (P : ℕ) : P = 4 :=
by
  have h1 : total_weight_of_cats 14 2.5 = 35 := sorry
  have h2 : 35 = total_weight_of_puppies P 7.5 + 5 := sorry
  have h3 : total_weight_of_puppies P puppy_weight = 30 := sorry
  have h4 : P * 7.5 = 30 := sorry
  have h5 : P = 4 := sorry
  exact h5

end number_of_puppies_l737_737094


namespace smallest_n_value_l737_737610

theorem smallest_n_value (a b c m n : ℕ) (a_even : Even a) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (sum_3000 : a + b + c = 3000) (factorial_eq : factorial a * factorial b * factorial c = m * 10^n) (m_not_div10 : ∀ k, m = 10 * k → False) : n = 496 :=
sorry

end smallest_n_value_l737_737610


namespace trapezoid_two_equal_angles_l737_737527

structure Trapezoid (T : Type) :=
(a b c d : T)
(parallel : a ∥ c ∨ b ∥ d)
(not_parallel : ¬ (a ∥ b ∧ c ∥ d))

def is_isosceles (T : Type) [Trapezoid T](t: Trapezoid T) : Prop :=
t.a = t.b

def is_right_angled (T: Type) [Trapezoid T](t: Trapezoid T) : Prop :=
∃ (angle: Prop), angle = π/2

theorem trapezoid_two_equal_angles (T : Type) [Trapezoid T] (t : Trapezoid T) 
  (h : ∃ (angles : T → T → Prop), ∀ x y, angles x y -> x = y): 
  is_isosceles T t ∨ is_right_angled T t := 
sorry

end trapezoid_two_equal_angles_l737_737527


namespace rectangle_area_in_ellipse_l737_737729

theorem rectangle_area_in_ellipse :
  (∃ (a b: ℝ), (a = 2 * b) ∧ (a^2 / 4 + b^2 / 8 = 1) ∧
  let A := 4 * a * b in A = 32 / 3) :=
sorry

end rectangle_area_in_ellipse_l737_737729


namespace arithmetic_sequence_vertex_sum_l737_737063

theorem arithmetic_sequence_vertex_sum {a b c d k : ℕ} 
  (h1 : d = a + 3 * k)
  (h2 : b = a + k)
  (h3 : c = a + 2 * k)
  (h_vertex : a = 1 ∧ d = 4) : b + c = 5 :=
by
  have h_a : a = 1 := h_vertex.1
  have h_d : d = 4 := h_vertex.2
  have h_k : 3 * k = 3 := by rw [h_d, h_a, Nat.add_comm, Nat.sub_eq_iff_eq_add]
  have k_val : k = 1 := Nat.eq_of_mul_eq_mul_left (by simp) h_k
  rw [h2, h3, h_a, k_val]
  exact by simp

end arithmetic_sequence_vertex_sum_l737_737063


namespace range_of_a_l737_737075

variable (a : ℝ) (x : ℝ)

def f (x : ℝ) : ℝ := -a * x^2 + Real.log x

theorem range_of_a (h : ∃ x : ℝ, 1 < x ∧ f a x > -a) : 0 < a ∧ a < 1 / 2 :=
by
  sorry

end range_of_a_l737_737075


namespace expand_product_l737_737017

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 :=
by 
  sorry

end expand_product_l737_737017


namespace interval_of_decrease_l737_737406

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 3 * Real.log x

theorem interval_of_decrease :
  ∀ x : ℝ, 0 < x ∧ x < Real.sqrt 2 / 2 → f'(x) < 0 := 
sorry

end interval_of_decrease_l737_737406


namespace floor_sqrt_80_l737_737985

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737985


namespace floor_sqrt_80_eq_8_l737_737835

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737835


namespace kite_shape_cannot_be_determined_l737_737683

-- Definitions of the conditions
structure Quadrilateral :=
(a b c d : ℝ)

structure KiteShaped (Q : Quadrilateral) :=
(diagonals_eq : Q.a + Q.c = Q.b + Q.d)  -- This assumes the sum of some properties like diagonals in a kite shape are equal

def cannot_determine_shape (Q : Quadrilateral) [KiteShaped Q] : Prop :=
∃ (shapes : Set (Quadrilateral → Prop)), (shapes = {Rectangle Q, Square Q, IsoscelesTrapezoid Q}) ∧ 
¬ ∀ S ∈ shapes, S Q

-- The theorem stating the problem and answer
theorem kite_shape_cannot_be_determined
  (Q : Quadrilateral) (K : KiteShaped Q) :
  cannot_determine_shape Q :=
sorry

end kite_shape_cannot_be_determined_l737_737683


namespace layla_earnings_l737_737614

def rate_donaldsons : ℕ := 15
def bonus_donaldsons : ℕ := 5
def hours_donaldsons : ℕ := 7
def rate_merck : ℕ := 18
def discount_merck : ℝ := 0.10
def hours_merck : ℕ := 6
def rate_hille : ℕ := 20
def bonus_hille : ℕ := 10
def hours_hille : ℕ := 3
def rate_johnson : ℕ := 22
def flat_rate_johnson : ℕ := 80
def hours_johnson : ℕ := 4
def rate_ramos : ℕ := 25
def bonus_ramos : ℕ := 20
def hours_ramos : ℕ := 2

def donaldsons_earnings := rate_donaldsons * hours_donaldsons + bonus_donaldsons
def merck_earnings := rate_merck * hours_merck - (rate_merck * hours_merck * discount_merck : ℝ)
def hille_earnings := rate_hille * hours_hille + bonus_hille
def johnson_earnings := rate_johnson * hours_johnson
def ramos_earnings := rate_ramos * hours_ramos + bonus_ramos

noncomputable def total_earnings : ℝ :=
  donaldsons_earnings + merck_earnings + hille_earnings + johnson_earnings + ramos_earnings

theorem layla_earnings : total_earnings = 435.2 :=
by
  sorry

end layla_earnings_l737_737614


namespace range_of_m_for_real_roots_l737_737489

-- Define the three quadratic equations
def eq1 (m x : ℝ) : Prop := x^2 - x + m = 0
def eq2 (m x : ℝ) : Prop := (m-1) * x^2 + 2 * x + 1 = 0
def eq3 (m x : ℝ) : Prop := (m-2) * x^2 + 2 * x - 1 = 0

-- The statement that at least two of these equations have real roots if and only if
def has_at_least_two_real_roots (equations : (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop) × (ℝ → ℝ → Prop)) : ℝ → Prop := 
  λ m, 
  let e1 := equations.1 in
  let e2 := equations.2 in
  let e3 := equations.2.1 in
  (∃ x : ℝ, e1 m x) ∧ (∃ x : ℝ, e2 m x) ∨
  (∃ x : ℝ, e1 m x) ∧ (∃ x : ℝ, e3 m x) ∨
  (∃ x : ℝ, e2 m x) ∧ (∃ x : ℝ, e3 m x)

-- Prove the range for m such that at least two equations have real roots
theorem range_of_m_for_real_roots : 
  ∀ m : ℝ, has_at_least_two_real_roots (eq1, eq2, eq3) m ↔ m ≤ 1/4 ∨ (1 ≤ m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_for_real_roots_l737_737489


namespace cos_sum_of_squares_l737_737423

theorem cos_sum_of_squares :
    ∀ (b1 b2 b3 b4 b5 b6 b7 : ℝ),
        (∀ θ : ℝ, cos θ ^ 7 = b1 * cos θ + b2 * cos (2 * θ) + b3 * cos (3 * θ) + b4 * cos (4 * θ) + b5 * cos (5 * θ) + b6 * cos (6 * θ) + b7 * cos (7 * θ)) →
        b1 = 35 / 64 ∧ b2 = 0 ∧ b3 = 21 / 64 ∧ b4 = 0 ∧ b5 = 7 / 64 ∧ b6 = 0 ∧ b7 = 1 / 64 →
        b1^2 + b2^2 + b3^2 + b4^2 + b5^2 + b6^2 + b7^2 = 1716 / 4096 := by
  intros b1 b2 b3 b4 b5 b6 b7 h1 h2
  sorry

end cos_sum_of_squares_l737_737423


namespace hyperbola_equation_l737_737066

/-- Given the foci and eccentricity of a hyperbola, 
    prove that the equation of the hyperbola is as specified. -/
theorem hyperbola_equation (C : Type) [MetricSpace C]
  (foci1 foci2 : C) (ecc : ℝ) 
  (h_foci1 : foci1 = (-2, 0)) 
  (h_foci2 : foci2 = (2, 0)) 
  (h_ecc : ecc = √2) : 
  C → Prop :=
  (λ x y, (x^2 / 2) - (y^2 / 2) = 1)

-- Proof of the theorem is left as an exercise
sorry

end hyperbola_equation_l737_737066


namespace reciprocal_of_proper_fraction_l737_737643

-- Define the proper fraction
def is_proper_fraction (x : ℚ) : Prop := 0 < x ∧ x < 1

-- Define the reciprocal of a number
def reciprocal (x : ℚ) : ℚ := if x ≠ 0 then 1 / x else 0

-- Main theorem to be proven
theorem reciprocal_of_proper_fraction (x : ℚ) (hx : is_proper_fraction x) : reciprocal x > 1 :=
begin
  -- Ensuring x is a proper fraction
  have h1 : 0 < x := hx.1,
  have h2 : x < 1 := hx.2,
  -- Definition of reciprocal when x is non-zero
  have h3 : reciprocal x = 1 / x,
  { simp [reciprocal, ne_of_gt h1], },
  -- Proving that 1/x > 1 when 0 < x < 1
  linarith,
end

end reciprocal_of_proper_fraction_l737_737643


namespace domain_f_l737_737216

noncomputable def f (x : ℝ) := Real.sqrt (3 - x) + Real.log (x - 1)

theorem domain_f : { x : ℝ | 1 < x ∧ x ≤ 3 } = { x : ℝ | True } ∩ { x : ℝ | x ≤ 3 } ∩ { x : ℝ | x > 1 } :=
by
  sorry

end domain_f_l737_737216


namespace solve_for_y_l737_737108

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l737_737108


namespace floor_sqrt_80_l737_737814

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737814


namespace ratio_of_areas_l737_737528

-- Let ABCD be a parallelogram
variables (A B C D E F : Type) [parallelogram A B C D]

-- E is the midpoint of diagonal BD
variable (E : midpoint B D)

-- Line EF bisects angle BEC
variables (F : Type) [line EF_bisects_angle_BEC E F]

-- The area of triangle BEC is twice the area of triangle DEF
variables [two_times_area_triangle_DEF := area_triangle B E C = 2 * area_triangle D E F]

-- The goal is to prove the ratio of the area of triangle DEF to the area of the quadrilateral ABEF
theorem ratio_of_areas (h₁ : parallelogram A B C D)
  (h₂ : midpoint E B D)
  (h₃ : line EF_bisects_angle_BEC E F)
  (h₄ : area_triangle B E C = 2 * area_triangle D E F) :
  ∃ k, k = area_triangle D E F / area_quadrilateral A B E F :=
by sorry

end ratio_of_areas_l737_737528


namespace floor_sqrt_80_l737_737862

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737862


namespace floor_sqrt_80_eq_8_l737_737892

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737892


namespace remaining_cupcakes_l737_737402

def total_cupcakes : ℕ := 2.5 * 12
def total_people_initial : ℕ := 27 + 1 + 1
def absent_students : ℕ := 3
def people_present : ℕ := total_people_initial - absent_students
def cupcakes_left : ℕ := total_cupcakes - people_present

theorem remaining_cupcakes : cupcakes_left = 4 := by
  sorry

end remaining_cupcakes_l737_737402


namespace rate_is_five_l737_737731

noncomputable def rate_per_sq_meter (total_cost : ℕ) (total_area : ℕ) : ℕ :=
  total_cost / total_area

theorem rate_is_five :
  let length := 80
  let breadth := 60
  let road_width := 10
  let total_cost := 6500
  let area_road1 := road_width * breadth
  let area_road2 := road_width * length
  let area_intersection := road_width * road_width
  let total_area := area_road1 + area_road2 - area_intersection
  rate_per_sq_meter total_cost total_area = 5 :=
by
  sorry

end rate_is_five_l737_737731


namespace sqrt_floor_eight_l737_737853

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737853


namespace convex_quadrilateral_count_l737_737050

theorem convex_quadrilateral_count (n : ℕ) (h₀ : n > 4) (h₁ : ∀ (a b c : ℝ × ℝ), a ≠ b → b ≠ c → c ≠ a → ¬ collinear a b c) :
  ∃ S : set (ℝ × ℝ), |S| = n ∧ (∃ Q : set (set (ℝ × ℝ)), (∀ q ∈ Q, 4 ≤ |q| ∧ convex q) ∧ Q.card ≥ binomial (n - 3) 2) :=
sorry

end convex_quadrilateral_count_l737_737050


namespace fraction_sum_l737_737383

theorem fraction_sum :
  (3 / 30 : ℝ) + (5 / 300) + (7 / 3000) = 0.119 := by
  sorry

end fraction_sum_l737_737383


namespace sumSeq_infinite_l737_737231

-- Define the sequence y_n
def seq (n : ℕ) : ℕ :=
  if n = 1 then 3 else seq (n - 1) ^ 2 + seq (n - 1) + 2

-- Define the sum of the sequence
def sumSeq : ℕ → ℝ
  | 0           := 0
  | (n + 1)     := sumSeq n + 1 / (seq (n + 1) + 1)

-- Theorem stating the desired result
theorem sumSeq_infinite :
  (∑' k, 1 / (seq k + 1 : ℝ)) = (1 / 4) :=
by
  sorry

end sumSeq_infinite_l737_737231


namespace floor_sqrt_80_l737_737810

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737810


namespace find_multiple_l737_737179
-- Importing Mathlib to access any necessary math definitions.

-- Define the constants based on the given conditions.
def Darwin_money : ℝ := 45
def Mia_money : ℝ := 110
def additional_amount : ℝ := 20

-- The Lean theorem which encapsulates the proof problem.
theorem find_multiple (x : ℝ) : 
  Mia_money = x * Darwin_money + additional_amount → x = 2 :=
by
  sorry

end find_multiple_l737_737179


namespace solve_linear_system_l737_737508

theorem solve_linear_system (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = -10 - 4 * y)
  (h3 : x + y = 14 - 4 * z)
  : 2 * x + 2 * y + 2 * z = 8 :=
by
  sorry

end solve_linear_system_l737_737508


namespace option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l737_737679

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Definitions of the given options as natural numbers
def A := 3^3 * 4^4 * 5^5
def B := 3^4 * 4^5 * 5^6
def C := 3^6 * 4^4 * 5^6
def D := 3^5 * 4^6 * 5^5
def E := 3^6 * 4^6 * 5^4

-- Lean statements for each option being a perfect square
theorem option_B_is_perfect_square : is_perfect_square B := sorry
theorem option_C_is_perfect_square : is_perfect_square C := sorry
theorem option_E_is_perfect_square : is_perfect_square E := sorry

end option_B_is_perfect_square_option_C_is_perfect_square_option_E_is_perfect_square_l737_737679


namespace floor_sqrt_80_l737_737875

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737875


namespace value_of_y_l737_737112

theorem value_of_y (x y : ℝ) (h₁ : 1.5 * x = 0.75 * y) (h₂ : x = 20) : y = 40 :=
sorry

end value_of_y_l737_737112


namespace courtyard_area_increase_l737_737730

def rectangular_area (length width : ℝ) : ℝ :=
  length * width

def circumference (length width : ℝ) : ℝ :=
  2 * (length + width)

def circular_radius (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

def circular_area (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem courtyard_area_increase :
  let length := 60;
      width := 20;
      rect_area := rectangular_area length width;
      C := circumference length width;
      radius := circular_radius C;
      circle_area := circular_area radius
  in (circle_area - rect_area) = (6400 - 1200 * Real.pi) / Real.pi :=
by
  sorry

end courtyard_area_increase_l737_737730


namespace find_area_of_trapezoid_l737_737145

-- Defining the conditions of the trapezoid
variables {A B C D : Type}
variable [inhabited A]
-- Let b be a real number representing the total sum of the parallel sides
variables {b : ℝ} 
-- Define diagonals AC and BD, where AC = x and BD = y
variables {x y : ℝ}
-- Define the angles as per the problem
variables {alpha : ℝ}

-- Define the conditions
def trapezoid_ABCD_parallel_and_sum (AB CD : ℝ) :=
  AB + CD = b

def diagonals_relation (AC BD : ℝ) :=
  5 * AC = 3 * BD

def angle_relation (alpha : ℝ) :=
  ∠BAC = 2 * α ∧ ∠DBA = α

-- Define the main theorem to be proved in the Lean statement
theorem find_area_of_trapezoid (AB CD : ℝ) (AC BD : ℝ) (α : ℝ)
  (h1 : trapezoid_ABCD_parallel_and_sum AB CD)
  (h2 : diagonals_relation AC BD)
  (h3 : angle_relation α) :
  area_trapezoid AB CD = (5 * real.sqrt 11 / 64) * b^2 :=
sorry

end find_area_of_trapezoid_l737_737145


namespace floor_sqrt_80_eq_8_l737_737900

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737900


namespace sum_of_distances_x_plus_y_l737_737410

def Point := (ℝ × ℝ)

def dist (P Q : Point) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def A : Point := (0, 0)
def B : Point := (12, 0)
def C : Point := (4, 6)
def P : Point := (5, 3)

theorem sum_of_distances :
  dist P A + dist P B + dist P C = Real.sqrt 34 + Real.sqrt 58 + Real.sqrt 10 :=
by
  sorry

theorem x_plus_y :
  1 + 1 = 2 :=
by
  rfl

end sum_of_distances_x_plus_y_l737_737410


namespace coolant_replacement_l737_737690

theorem coolant_replacement 
    (total_volume : ℝ) 
    (initial_concentration : ℝ) 
    (final_concentration : ℝ) 
    (replacement_concentration : ℝ)
    (initial_antifreeze : ℝ)
    (desired_antifreeze : ℝ)
    (x : ℝ) 
    (original_coolant_left : ℝ) 
    (drained_replaced : ℝ) 
  : total_volume = 19 
  → initial_concentration = 0.30 
  → final_concentration = 0.50 
  → replacement_concentration = 0.80 
  → initial_antifreeze = 0.30 * 19 
  → desired_antifreeze = 0.50 * 19 
  → x = (desired_antifreeze - initial_antifreeze) / (replacement_concentration - initial_concentration)
  → drained_replaced = x 
  → original_coolant_left = total_volume - drained_replaced 
  → total_volume = 19 
  ∧ drained_replaced = 7.6 
  ∧ original_coolant_left = 11.4 :=
begin
  sorry
end

end coolant_replacement_l737_737690


namespace area_gray_region_correct_l737_737390

def center_C : ℝ × ℝ := (3, 5)
def radius_C : ℝ := 3
def center_D : ℝ × ℝ := (9, 5)
def radius_D : ℝ := 3

noncomputable def area_gray_region : ℝ :=
  let rectangle_area := (center_D.1 - center_C.1) * (center_C.2 - (center_C.2 - radius_C))
  let sector_area := (1 / 4) * radius_C ^ 2 * Real.pi
  rectangle_area - 2 * sector_area

theorem area_gray_region_correct :
  area_gray_region = 18 - 9 / 2 * Real.pi :=
by
  sorry

end area_gray_region_correct_l737_737390


namespace minimum_S_value_l737_737160

noncomputable def S (a : ℝ) (h : a > 1) : ℝ :=
  ∫ x in 0..1, a^4 / real.sqrt ((a^2 - x^2)^3)

theorem minimum_S_value : ∀ a : ℝ, a > 1 → S a = 2 :=
begin
  sorry
end

end minimum_S_value_l737_737160


namespace f_order_l737_737444

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Condition: Reflection symmetry f(x) = f(2-x)
axiom f_symm : ∀ x : ℝ, f(x) = f(2 - x)

-- Condition: (x - 1)f'(x) > 0 for all x ≠ 1
axiom f_deriv_pos : ∀ x : ℝ, x ≠ 1 → (x - 1) * f'(x) > 0

-- Additional given condition: 1 < a < 2
variable {a : ℝ}
axiom a_bound : 1 < a ∧ a < 2

-- Goal: Prove f(log2(a)) < f(2) < f(2^a)
theorem f_order : f (Real.log 2 a) < f 2 ∧ f 2 < f (2 ^ a) :=
by
  sorry

end f_order_l737_737444


namespace floor_sqrt_80_l737_737979

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737979


namespace closest_miles_walker_l737_737593

-- Define the conditions as constants
def pedometer_max_steps : Nat := 99999
def initial_reading : Nat := 0
def flips : Nat := 44
def final_reading : Nat := 50000
def steps_per_mile : Nat := 1800

-- Define a function to calculate steps from flips
def steps_from_flips (flips : Nat) : Nat := ((pedometer_max_steps + 1) * flips)

-- Define a function to calculate total steps
def total_steps (flips : Nat) (final_reading : Nat) : Nat := (steps_from_flips flips) + final_reading

-- Define a function to calculate miles from total steps
def miles_walked (total_steps : Nat) (steps_per_mile : Nat) : Float := (total_steps : Float) / (steps_per_mile : Float)

-- Calculate the total number of steps Pete walked during the year
def total_steps_walked := total_steps flips final_reading

-- Calculate the number of miles Pete walked during the year
def miles_walked_during_year := miles_walked total_steps_walked steps_per_mile

-- The proof statement
theorem closest_miles_walker : abs (miles_walked_during_year - 2500) <= abs (miles_walked_during_year - 3000) ∧
                                abs (miles_walked_during_year - 2500) <= abs (miles_walked_during_year - 3500) ∧
                                abs (miles_walked_during_year - 2500) <= abs (miles_walked_during_year - 4000) ∧
                                abs (miles_walked_during_year - 2500) <= abs (miles_walked_during_year - 4500) :=
by
  sorry

end closest_miles_walker_l737_737593


namespace students_end_year_10_l737_737138

def students_at_end_of_year (initial_students : ℕ) (left_students : ℕ) (increase_percent : ℕ) : ℕ :=
  let remaining_students := initial_students - left_students
  let increased_students := (remaining_students * increase_percent) / 100
  remaining_students + increased_students

theorem students_end_year_10 : 
  students_at_end_of_year 10 4 70 = 10 := by 
  sorry

end students_end_year_10_l737_737138


namespace lilliputian_matchboxes_fit_l737_737666

theorem lilliputian_matchboxes_fit (L W H : ℝ) (k : ℝ) (V_G V_L : ℝ) 
    (h_k : k = 12) 
    (h_VG : V_G = L * W * H) 
    (h_VL : V_L = (L / k) * (W / k) * (H / k)) : 
    V_G / V_L = 1728 := 
by
  have h1 : V_L = V_G / (k ^ 3) := 
  by 
    rw [h_VL, h_VG]
    field_simp
    ring
  
  rw [h1]
  have h2 : k ^ 3 = 1728 := 
  by
    rw [h_k]
    norm_num

  rw [h2]
  field_simp
  norm_num
  sorry

end lilliputian_matchboxes_fit_l737_737666


namespace white_copy_cost_l737_737774

-- Define the given conditions
def cost_per_colored_copy : ℝ := 0.10
def total_copies : ℝ := 400
def colored_copies : ℝ := 50
def total_bill : ℝ := 22.50

-- Calculate the total cost for colored copies
def total_cost_colored : ℝ := colored_copies * cost_per_colored_copy

-- Calculate the cost for white copies
def total_cost_white : ℝ := total_bill - total_cost_colored

-- Calculate the number of white copies
def white_copies : ℝ := total_copies - colored_copies

-- Define the cost per white copy (what we want to prove)
def cost_per_white_copy : ℝ := total_cost_white / white_copies

-- Proof statement (goal)
theorem white_copy_cost :
  cost_per_white_copy = 0.05 := 
  sorry

end white_copy_cost_l737_737774


namespace largest_neg_multiple_of_7_greater_than_neg_150_l737_737261

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l737_737261


namespace sequence_sum_geometric_inverse_sequence_geometric_l737_737052

-- Define the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (a1 q : ℝ) : Prop :=
  a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

-- Problem statement 1
theorem sequence_sum_geometric (a : ℕ → ℝ) (a1 q : ℝ) (h : is_geometric_sequence a a1 q) :
  is_geometric_sequence (λ n, a n + a (n + 1) + a (n + 2)) (a1 * (1 + q + q^2)) q :=
sorry

-- Problem statement 2
theorem inverse_sequence_geometric (a : ℕ → ℝ) (a1 q : ℝ) (h : is_geometric_sequence a a1 q) :
  is_geometric_sequence (λ n, 1 / (a n)) (1 / a1) (1 / q) :=
sorry

end sequence_sum_geometric_inverse_sequence_geometric_l737_737052


namespace floor_sqrt_80_l737_737968

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737968


namespace problem_1_problem_2_l737_737591

noncomputable def f (x a : ℝ) : ℝ := abs x + 2 * abs (x - a)

theorem problem_1 (x : ℝ) : (f x 1 ≤ 4) ↔ (- 2 / 3 ≤ x ∧ x ≤ 2) := 
sorry

theorem problem_2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ (4 ≤ a) := 
sorry

end problem_1_problem_2_l737_737591


namespace angles_equal_condition_l737_737700

variables {A B C D S E F : Type} [EuclideanGeometry]

/-- A (convex) trapezoid ABCD is called "good" if it has a circumcircle, the sides AB and CD are the parallel sides, and CD is shorter than AB. 
Given such a good trapezoid ABCD with point S defined such that the parallel to AD passing through B intersects the extension of CD at point S,
and points E and F are where the tangents from S to the circumcircle of the trapezoid touch it, where E lies on the same side of line CD as A,
prove that the angles ∠BSE and ∠FSC are equal if and only if ∠BAD = 60° or AB = AD. -/
theorem angles_equal_condition (h_good : good_trapezoid A B C D)
  (h_parallel: A B ∥ C D) (h_cd_less_ab: length C D < length A B)
  (h_circumcircle : has_circumcircle A B C D)
  (h_s : S = intersection (parallel B A D) (extension C D))
  (h_tangents : tangents_from S (circumcircle A B C D) E F)
  (h_E_side : on_same_side E C D A) :
  (angle B S E = angle F S C) ↔ (angle B A D = 60⁰ ∨ length A B = length A D) :=
sorry

end angles_equal_condition_l737_737700


namespace solution_system_l737_737990

theorem solution_system :
  ∀ (x y : ℝ), (y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y) ↔
  (x = 0 ∧ y = 0) ∨ (x = 2 + sqrt(2) ∧ y = 2 + sqrt(2)) ∨ (x = 2 - sqrt(2) ∧ y = 2 - sqrt(2)) :=
by
  intros x y
  split
  {
    intro h
    sorry -- Include detailed proof steps here
  }
  {
    intro h
    sorry -- Include detailed proof steps here
  }

end solution_system_l737_737990


namespace find_possible_k_l737_737470

-- Define the conditions: the roots of the polynomial must be positive integers
def roots_are_positive_integers (r1 r2 : ℕ) : Prop :=
  r1 > 0 ∧ r2 > 0 ∧ r1 * r2 = 36

-- Define Vieta's formulas which relate the roots to the coefficients via sums and products
def sum_of_roots_equals_k (r1 r2 k : ℕ) : Prop :=
  r1 + r2 = k

theorem find_possible_k (k : ℕ) :
  (∃ r1 r2 : ℕ, roots_are_positive_integers r1 r2 ∧ sum_of_roots_equals_k r1 r2 k) →
  k ∈ {12, 13, 15, 20, 37} :=
sorry

end find_possible_k_l737_737470


namespace S_15_value_l737_737396

-- S_n denotes the sum of the elements in the nth set
def S : ℕ → ℕ
| 1 := 1
| n + 1 := let first_element := 1 + n * (n + 1) / 2 in
           let last_element := first_element + n in
           (n + 1) * (first_element + last_element) / 2

theorem S_15_value : S 15 = 1695 :=
by sorry

end S_15_value_l737_737396


namespace solve_quadratic_eq_l737_737203

theorem solve_quadratic_eq {x : ℝ} : (x^2 - 4 * x - 7 = 0) ↔ (x = 2 + sqrt 11 ∨ x = 2 - sqrt 11) :=
sorry

end solve_quadratic_eq_l737_737203


namespace arith_seq_ratio_l737_737208

-- Definitions related to arithmetic sequence and sum
def arithmetic_seq (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arith_seq (S a : ℕ → ℝ) := ∀ n : ℕ, S n = (n : ℝ) / 2 * (a 1 + a n)

-- Given condition
def condition (a : ℕ → ℝ) := a 8 / a 7 = 13 / 5

-- Prove statement
theorem arith_seq_ratio (a S : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_sum : sum_of_arith_seq S a)
  (h_cond : condition a) :
  S 15 / S 13 = 3 := 
sorry

end arith_seq_ratio_l737_737208


namespace find_sine_cosine_find_period_and_interval_l737_737472

variables {θ : ℝ} (hθ : θ ∈ set.Ioo 0 (real.pi / 2))

def parallel_vectors : Prop := (sqrt 3, real.sin θ) = λ k : ℝ, (k * 1, k * real.cos θ)

theorem find_sine_cosine (h_parallel : parallel_vectors hθ) :
  real.sin θ = sqrt 3 / 2 ∧ real.cos θ = 1 / 2 :=
sorry

def f (x : ℝ) : ℝ := real.sin (2 * x + θ)

theorem find_period_and_interval (h_parallel : parallel_vectors hθ)
  (hθ_val : θ = real.pi / 3) :
  (real.periodic (λ x, f hθ_val x) real.pi) ∧
  set.eq (set.Icc (-(5 * real.pi / 12) + k * real.pi) (real.pi / 12 + k * real.pi)) 
         (λ x, ∃ k : ℤ, x ∈ set.Icc (-(5 * real.pi / 12) + k * real.pi) (real.pi / 12 + k * real.pi)) :=
sorry

end find_sine_cosine_find_period_and_interval_l737_737472


namespace simplify_expression_l737_737198

theorem simplify_expression (a : ℝ) (h : -2 ≤ a ∧ a ≤ 2) :
  let expr := (a^2 + 1)/a - 2 / ((a + 2) * (a - 1) / (a^2 + 2 * a)) in
  expr = a - 1 :=
by
  sorry

end simplify_expression_l737_737198


namespace velocity_at_fifth_second_l737_737225

theorem velocity_at_fifth_second :
  let s := λ t : ℝ, 3 * t ^ 2 - 2 * t + 4 in
  let v := λ t : ℝ, Deriv s t in
  v 5 = 28 := by
  sorry

end velocity_at_fifth_second_l737_737225


namespace comprehensive_survey_correct_l737_737305

/-- Which of the following questions is suitable for a comprehensive survey?
    A: The viewership of a certain episode of "The Brain" on CCTV.
    B: The average number of online shopping times per person in a community in Jiaokou County in May.
    C: The quality of components of Tianzhou-6.
    D: The maximum cruising range of the BYD new energy car escort 07.
    The suitable option for a comprehensive survey is C. -/
theorem comprehensive_survey_correct :
  ∃ (suitable_option : string),
    (suitable_option = "The quality of components of Tianzhou-6.") :=
by
  use "The quality of components of Tianzhou-6."
  sorry

end comprehensive_survey_correct_l737_737305


namespace valid_mappings_equal_54_l737_737175

def S : set (fin 9 → fin 3) := { f | ∀ i, f i ∈ {0, 1, 2} }

noncomputable def valid_mapping_count : ℕ :=
  { f : S → {0, 1, 2} // 
    ∀ (x y : S), (∀ i, x i ≠ y i) → f x ≠ f y }.card

theorem valid_mappings_equal_54 : valid_mapping_count = 54 := 
  sorry

end valid_mappings_equal_54_l737_737175


namespace floor_sqrt_80_l737_737831

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737831


namespace floor_sqrt_80_l737_737975

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737975


namespace find_continuous_functions_l737_737020

open Real

noncomputable def continuous_solutions (f : ℝ⁺ → ℝ⁺) : Prop :=
  ∀ x : ℝ⁺, x + 1/x = f x + 1/(f x)

theorem find_continuous_functions (f : ℝ⁺ → ℝ⁺) :
  continuous_solutions f ↔
    f = (fun x => x) ∨
    f = (fun x => 1/x) ∨
    f = (fun x => max x (1/x)) ∨
    f = (fun x => min x (1/x)) :=
sorry

end find_continuous_functions_l737_737020


namespace maximum_value_is_one_div_sqrt_two_l737_737170

noncomputable def maximum_value_2ab_root2_plus_2ac_plus_2bc (a b c : ℝ) : ℝ :=
  2 * a * b * Real.sqrt 2 + 2 * a * c + 2 * b * c

theorem maximum_value_is_one_div_sqrt_two (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h : a^2 + b^2 + c^2 = 1) :
  maximum_value_2ab_root2_plus_2ac_plus_2bc a b c ≤ 1 / Real.sqrt 2 :=
by
  sorry

end maximum_value_is_one_div_sqrt_two_l737_737170


namespace probability_at_least_one_heart_or_king_l737_737335
   
   noncomputable def probability_non_favorable : ℚ := 81 / 169

   theorem probability_at_least_one_heart_or_king :
     1 - probability_non_favorable = 88 / 169 := 
   sorry
   
end probability_at_least_one_heart_or_king_l737_737335


namespace decreasing_interval_l737_737628

noncomputable def f : ℝ → ℝ := fun x => x^3 - 3 * x^2 + 2

theorem decreasing_interval :
  {x : ℝ | x > 0 ∧ x < 2} = {x : ℝ | (f.derivative.eval x) < 0} := 
sorry

end decreasing_interval_l737_737628


namespace problem_a_b_n_l737_737453

theorem problem_a_b_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ 0 → (b - k) ∣ (a - k^n)) : a = b^n := 
sorry

end problem_a_b_n_l737_737453


namespace largest_4_digit_palindrome_divisible_by_3_exists_largest_4_digit_palindrome_divisible_by_3_l737_737251

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem largest_4_digit_palindrome_divisible_by_3 :
  ∀ n : ℕ, is_4_digit n ∧ is_palindrome n ∧ is_divisible_by_3 n → n ≤ 9999 :=
by
  intros n hn
  sorry

theorem exists_largest_4_digit_palindrome_divisible_by_3 :
  ∃ n : ℕ, is_4_digit n ∧ is_palindrome n ∧ is_divisible_by_3 n ∧ ∀ m : ℕ, is_4_digit m ∧ is_palindrome m ∧ is_divisible_by_3 m → m ≤ n :=
by
  existsi 9999
  split
  { show 1000 ≤ 9999 ∧ 9999 < 10000, from ⟨le_refl 9999, dec_trivial⟩ }
  split
  { show is_palindrome 9999, from dec_trivial }
  split
  { show is_divisible_by_3 9999, from dec_trivial }
  { intros m hm
    sorry }

end largest_4_digit_palindrome_divisible_by_3_exists_largest_4_digit_palindrome_divisible_by_3_l737_737251


namespace problem1_problem2_l737_737415

noncomputable def f (x a : ℝ) : ℝ := |2 * x + a| + |x - (1 / a)|

theorem problem1 (a : ℝ) (h_a : a < 0) (h_f : |a| + |1 / a| > 5 / 2) : a < -2 ∨ -1 / 2 < a ∧ a < 0 :=
sorry

theorem problem2 (a x : ℝ) (h_a : a < 0) (hx : x ∈ ℝ) : 
  f x a ≥ sqrt 2 :=
sorry

end problem1_problem2_l737_737415


namespace coeff_x_squared_power_six_l737_737586

theorem coeff_x_squared_power_six : 
  let f (n : ℕ) := (1 + x + x^2)^n,
      a₂ := (1 + 2 + 3 + ... + 6) in
  a₂ = 21 := by
  sorry

end coeff_x_squared_power_six_l737_737586


namespace part_I_part_II_l737_737056

variable {a : ℕ → ℝ}
variable (n : ℕ)

-- Hypotheses about the sequence
def sequence_conditions : Prop :=
  (∀ n, a n > 0) ∧ (a 1 = 2) ∧ (∀ n, (↑n + 1) * (a (n + 1))^2 = n * (a n)^2 + a n)

-- Proving each part under the given conditions
theorem part_I (h : sequence_conditions) : ∀ n, a n > 1 :=
sorry

theorem part_II (h : sequence_conditions) (n : ℕ) : ∑ i in (finset.range n).filter (λ i, 2 ≤ i + 2) (a (i + 2))^2 / (i + 2)^2 < 7 / 4 :=
sorry

end part_I_part_II_l737_737056


namespace sam_average_speed_l737_737218

theorem sam_average_speed
  (total_distance : ℝ) 
  (start_time end_time : ℝ) 
  (break_time : ℝ) 
  (break_duration : ℝ) 
  (driving_distance : total_distance = 180)
  (start_time_is : start_time = 8) 
  (end_time_is : end_time = 14) 
  (break_time_is : break_time = 11) 
  (break_duration_is : break_duration = 0.5) :
  (total_distance / ((end_time - start_time) - break_duration) = 32.73) :=
by
  rw [driving_distance, start_time_is, end_time_is, break_duration_is]
  dsimp
  norm_num
  sorry

end sam_average_speed_l737_737218


namespace axis_of_symmetry_l737_737474

-- Define the function f
def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.sin (2 * x + ϕ)

-- Statement of the proof
theorem axis_of_symmetry {ϕ : ℝ} (h : ∀ x : ℝ, f x ϕ ≤ |f (π / 6) ϕ|) : ∃ k ∈ ℤ, ϕ = k * π + π / 6 → ∃ m ∈ ℤ, x = (m - k) * π / 2 + π / 6 → x = 2 * π / 3 :=
sorry

end axis_of_symmetry_l737_737474


namespace smallest_n_binary_representation_l737_737995

theorem smallest_n_binary_representation:
  ∃ n : ℕ, n = 2053 ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ 1990 → (nat.binary_repr k ∈ (real.to_nat (1 / n) - floor (1 / n)))
  sorry

end smallest_n_binary_representation_l737_737995


namespace green_more_than_blue_l737_737149

theorem green_more_than_blue (B₀ G₀ initial_diff first_border_second_border_diff : ℕ)
  (h₀ : B₀ = 12)
  (h₁ : G₀ = 8)
  (h₂ : initial_diff = G₀ - B₀)
  (h₃ : first_border_second_border_diff = 18 + 24)
  (h₄ : B₀ + first_border_second_border_diff = initial_diff + 38) :
  G₀ + 18 + 24 - B₀ = 38 := by
  rw [h₁, h₀, h₂, h₃, h₄]
  sorry

end green_more_than_blue_l737_737149


namespace sum_remainder_l737_737670

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l737_737670


namespace find_fake_pearl_weighings_l737_737647

/-- Given 9 pearls with one being fake and lighter, and access to a balance scale,
prove that the minimum number of weighings needed to find the fake pearl is 2. -/
theorem find_fake_pearl_weighings :
  ∃ (n : ℕ), n = 2 ∧
  ∀ (pearls : Fin 9 → ℤ) (is_fake : Fin 9 → Prop),
  (∃! i, is_fake i ∧ pearls i < ∀ j, j ≠ i → pearls j) →
  ∃ (weighings : list (Fin 9 × Fin 9 → Prop)), weighings.length = n :=
begin
  sorry,
end

end find_fake_pearl_weighings_l737_737647


namespace path_count_l737_737716

-- Define the game conditions
def checkered_board := 
    ∃ (rows columns : ℕ) (color : ℕ → ℕ → Prop), rows = 6 ∧ columns = 8 ∧
        (∀ r c, color r c ↔ (r + c) % 2 = 0)

def valid_move (from to : ℕ × ℕ) (color : ℕ → ℕ → Prop) : Prop :=
    let (x₁, y₁) := from
    let (x₂, y₂) := to
    (x₂ = x₁) ∧ (y₂ = y₁ + 1) ∧ (color x₁ y₁ = ff) ∧ (color x₂ y₂ = tt)

def path_exists (moves : ℕ) (start end : ℕ × ℕ) : Prop :=
    moves = 8 ∧ start = (0, 0) ∧ end = (7, 5) ∧
    ∀ path, (list.length path = 8 ∧
    list.head path = start ∧ list.last path = end ∧
    list.pairwise valid_move path)

-- The final proof goal
theorem path_count : checkered_board → path_exists 8 (0, 0) (7, 5) → 56 
:= by sorry

end path_count_l737_737716


namespace floor_sqrt_80_eq_8_l737_737909

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737909


namespace external_angle_bisector_lengths_l737_737363

noncomputable def f_a (a b c : ℝ) : ℝ := 4 * Real.sqrt 3
noncomputable def f_b (b : ℝ) : ℝ := 6 / Real.sqrt 7
noncomputable def f_c (a b c : ℝ) : ℝ := 4 * Real.sqrt 3

theorem external_angle_bisector_lengths (a b c : ℝ) 
  (ha : a = 5 - Real.sqrt 7)
  (hb : b = 6)
  (hc : c = 5 + Real.sqrt 7) :
  f_a a b c = 4 * Real.sqrt 3 ∧
  f_b b = 6 / Real.sqrt 7 ∧
  f_c a b c = 4 * Real.sqrt 3 := by
  sorry

end external_angle_bisector_lengths_l737_737363


namespace clea_ride_time_l737_737772

noncomputable def walk_down_stopped (x y : ℝ) : Prop := 90 * x = y
noncomputable def walk_down_moving (x y k : ℝ) : Prop := 30 * (x + k) = y
noncomputable def ride_time (y k t : ℝ) : Prop := t = y / k

theorem clea_ride_time (x y k t : ℝ) (h1 : walk_down_stopped x y) (h2 : walk_down_moving x y k) :
  ride_time y k t → t = 45 :=
sorry

end clea_ride_time_l737_737772


namespace floor_neg_7_over_4_l737_737804

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l737_737804


namespace find_prob_A_complement_l737_737013

noncomputable def P (A : Type _) [MeasureSpace A] (s : Set A) : ℝ :=
MeasureTheory.Measure.measure_instance A s

variable {A B : Set A} {pA pB : ℝ}

-- Our conditions:
axiom mutually_exclusive : Disjoint A B
axiom neither_occurs : P(Aᶜ ∩ Bᶜ) = 2 / 5
axiom twice_prob_B : P(A) = 2 * P(B)

-- The statement we need to prove:
theorem find_prob_A_complement : P(Aᶜ) = 3 / 5 :=
sorry

end find_prob_A_complement_l737_737013


namespace matrix_scaling_transformation_l737_737427

theorem matrix_scaling_transformation (a b c d : ℝ) :
  let M := ![
    [5, 0],
    [0, 3]
  ]
  in M * ![
    [a, b],
    [c, d]
  ] = ![
    [5 * a, 5 * b],
    [3 * c, 3 * d]
  ] :=
by sorry

end matrix_scaling_transformation_l737_737427


namespace complex_eq_l737_737461

open Complex

theorem complex_eq (z : ℂ) (i : ℂ) (hz : z = 1 + I) :
  z * conj(z) + abs (conj (z)) - 1 = sqrt 2 + 1 :=
by
  sorry

end complex_eq_l737_737461


namespace problem1_problem2_l737_737492

open Set

variables {α : Type*} [Preorder α]

-- Define the sets A and B
def A (a : α) : Set α := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set α := {x | x < -1 ∨ x > 5 }

-- First problem: A ∩ B = ∅ implies a ≤ -4 ∨ a ≥ 5.
theorem problem1 {a : α} : A a ∩ B = ∅ → a ≤ -4 ∨ a ≥ 5 := by
  sorry

-- Second problem: A ∪ B = B implies a > 2.
theorem problem2 {a : α} : A a ∪ B = B → a > 2 := by
  sorry

end problem1_problem2_l737_737492


namespace satisfies_natural_solution_l737_737127

theorem satisfies_natural_solution (m : ℤ) :
  (∃ x : ℕ, x = 6 / (m - 1)) → (m = 2 ∨ m = 3 ∨ m = 4 ∨ m = 7) :=
by
  sorry

end satisfies_natural_solution_l737_737127


namespace permutation_inequality_l737_737560

theorem permutation_inequality (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) 
  (ha : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i) 
  (hb : ∃ σ : Perm (Fin n), ∀ i : Fin n, b i = a (σ i)) : 
  (∑ i in Finset.range n, a i / b i) ≥ n := 
by 
  sorry

end permutation_inequality_l737_737560


namespace advance_agency_fees_eq_8280_l737_737667

-- Conditions
variables (Commission GivenFees Incentive AdvanceAgencyFees : ℝ)
-- Given values
variables (h_comm : Commission = 25000) 
          (h_given : GivenFees = 18500) 
          (h_incent : Incentive = 1780)

-- The problem statement to prove
theorem advance_agency_fees_eq_8280 
    (h_comm : Commission = 25000) 
    (h_given : GivenFees = 18500) 
    (h_incent : Incentive = 1780)
    : AdvanceAgencyFees = 26780 - GivenFees :=
by
  sorry

end advance_agency_fees_eq_8280_l737_737667


namespace solve_for_a_l737_737403

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem solve_for_a (a : ℝ) : E a 3 2 = E a 5 3 ↔ a = -1/16 :=
by
  sorry

end solve_for_a_l737_737403


namespace floor_sqrt_80_l737_737946

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737946


namespace problem_statement_l737_737384

theorem problem_statement : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end problem_statement_l737_737384


namespace original_price_l737_737156

theorem original_price (sale_price : ℝ) (discount : ℝ) : 
  sale_price = 55 → discount = 0.45 → 
  ∃ (P : ℝ), 0.55 * P = sale_price ∧ P = 100 :=
by
  sorry

end original_price_l737_737156


namespace largest_multiple_of_7_neg_greater_than_neg_150_l737_737283

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l737_737283


namespace properties_of_F_l737_737165

variable {R : Type*} [LinearOrderedField R] {f : R → R} (x : R)

def F (x : R) : R := f x - x^3

noncomputable def even_F (f : R → R) : Prop :=
  ∀ x, F f x = F f (-x)

noncomputable def decreasing_F_neg (f : R → R) : Prop :=
  ∀ x < 0, deriv (F f) x < 0

noncomputable def inequality_solution_set (f : R → R) : Set R :=
  {x : R | f x - f (x - 1) > 3 * x ^ 2 - 3 * x + 1}

theorem properties_of_F :
  (∀ x, f x - f (-x) = 2 * x^3) ∧
  (∀ x > 0, deriv f x > 3 * x^2) →
  even_F f ∧
  decreasing_F_neg f ∧
  inequality_solution_set f = {x : R | 1/2 < x} :=
by
  intros h
  sorry

end properties_of_F_l737_737165


namespace ava_distance_l737_737757

theorem ava_distance (d_total d_remaining d_covered : ℕ) (h_total : d_total = 1000) (h_remaining : d_remaining = 167) (h_covered : d_covered = d_total - d_remaining) : d_covered = 833 :=
by
  rw [h_total, h_remaining, h_covered]
  norm_num

end ava_distance_l737_737757


namespace find_m_plus_n_l737_737399

-- Definition for the first set
def first_set (n : ℕ) := {n, n + 6, n + 8, n + 12, n + 18}

-- Definition for the second set
def second_set (m : ℕ) := {m, m + 2, m + 4, m + 6, m + 8}

-- Condition: The median of the first set is 12
def median_first_set (n : ℕ) : Prop := n + 8 = 12

-- Condition: The mean of the second set is m + 5
def mean_second_set (m : ℕ) : Prop := (m + (m+2) + (m+4) + (m+6) + (m+8)) / 5 = m + 5

-- The final proof statement
theorem find_m_plus_n (n m : ℕ) (h1 : median_first_set n) (h2 : mean_second_set m) : m + n = 7 :=
sorry

end find_m_plus_n_l737_737399


namespace floor_sqrt_80_l737_737959

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737959


namespace floor_sqrt_80_l737_737820

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737820


namespace minimum_major_axis_length_l737_737054

def line (x y : ℝ) : Prop := x + y - 4 = 0
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1
def focus1 := (-2, 0) : ℝ × ℝ
def focus2 := (2, 0) : ℝ × ℝ

theorem minimum_major_axis_length 
  (M : ℝ × ℝ)
  (H_M_on_line : line M.1 M.2)
  (H_ellipse_focus : focus1 ∨ focus2 ∈ {f : ℝ × ℝ | ∃ x y, ellipse x y ∧ f = (x, y)}):
  ∃ a b : ℝ, a > b ∧ a > 0 ∧ 2 * a = 2 * real.sqrt 10 :=
sorry

end minimum_major_axis_length_l737_737054


namespace number_of_subsets_sum_of_min_and_max_is_11_l737_737495

/-- Prove the number of nonempty subsets of {1, 2, ..., 10} that have the property
    that the sum of their largest and smallest element is 11 is 341. --/
theorem number_of_subsets_sum_of_min_and_max_is_11 :
  let s := {1, 2, ..., 10} in
  ∃ n, (∀ (t : Finset ℕ), t ⊆ s → t.Nonempty → (t.min' t.nonempty_witness + t.max' t.nonempty_witness = 11) → ∑ (k = 0) ^ 4, 4 ^ k = 341) := 
by
  sorry

end number_of_subsets_sum_of_min_and_max_is_11_l737_737495


namespace find_lengths_OM_ON_l737_737147

-- Definitions of the conditions
variables {A B C M N O : Type}
variables (triangle_ABC : triangle A B C)
variables (bisector_AM : bisector A M)
variables (bisector_BN : bisector B N)
variables (incenter_O : is_incenter O A B C)
variables (on_circle_CM : on_circle C O M N)
variables (length_MN : MN = real.sqrt 3)

-- Lean 4 statement to prove lengths
theorem find_lengths_OM_ON :
  OM = 1 ∧ ON = 1 :=
sorry

end find_lengths_OM_ON_l737_737147


namespace largest_multiple_of_7_negation_gt_neg150_l737_737272

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l737_737272


namespace ratio_of_distances_l737_737061

noncomputable def focus_parabola := (1, 0 : ℝ)  -- Focus F

def parabola := {p : ℝ × ℝ | p.snd ^ 2 = 4 * p.fst}

def line_through_focus (m : ℝ) := {p : ℝ × ℝ | p.snd = m * (p.fst - 1)}

def intersect_parabola_line (m : ℝ) := 
  {p : ℝ × ℝ | p ∈ parabola ∧ p ∈ line_through_focus m}

theorem ratio_of_distances (m : ℝ) (h_m_sqrt3 : m = real.sqrt 3)
  (A B : ℝ × ℝ) 
  (hA : A ∈ intersect_parabola_line m)
  (hB : B ∈ intersect_parabola_line m)
  (h_FA_gt_FB : dist focus_parabola A > dist focus_parabola B) : 
  dist focus_parabola A / dist focus_parabola B = 3 := 
sorry

end ratio_of_distances_l737_737061


namespace a1_value_S5_value_l737_737570

-- Define the sequence a_n and the sum S_n
def S : ℕ → ℕ
| 0       := 0
| (n + 1) := ∑ i in finset.range (n + 1), (λ a : ℕ, if a = 0 then 1 else 2 * S n + 1) i

theorem a1_value (a : ℕ → ℕ) (S : ℕ → ℕ) (h₁ : S 2 = 4) (h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * S n + 1) :
  a 1 = 1 :=
sorry

theorem S5_value (S : ℕ → ℕ) (h₁ : S 2 = 4) (h₂ : ∀ n : ℕ, n > 0 → S (n + 1) = 3 * S n + 1) :
  S 5 = 121 :=
sorry

end a1_value_S5_value_l737_737570


namespace floor_of_sqrt_80_l737_737919

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737919


namespace max_marked_points_on_circle_l737_737525

theorem max_marked_points_on_circle (n : ℕ) (hn : 3 ≤ n) : 
  let marked_points : Set (ℝ × ℝ) := {p | midpoint_of_side_or_diagonal n p} in
  ∃ single_circle : Set (ℝ × ℝ), ∀ p ∈ marked_points, p ∈ single_circle → card single_circle = n :=
sorry

end max_marked_points_on_circle_l737_737525


namespace inequality_proof_l737_737187

theorem inequality_proof (a1 a2 a3 b1 b2 b3 : ℝ) 
  (h1 : a1 ≥ a2) (h2 : a2 ≥ a3) (h3 : b1 ≥ b2) (h4 : b2 ≥ b3) :
  3 * (a1 * b1 + a2 * b2 + a3 * b3) ≥ (a1 + a2 + a3) * (b1 + b2 + b3) :=
begin
  sorry
end

end inequality_proof_l737_737187


namespace tetrahedron_area_inequality_l737_737184

theorem tetrahedron_area_inequality (A B C D : Point)
  (A_ABC A_ABD A_ACD A_BCD : ℝ) :
  A_ABC < A_ABD + A_ACD + A_BCD :=
begin
  sorry
end

end tetrahedron_area_inequality_l737_737184


namespace mandatory_state_tax_rate_l737_737157

theorem mandatory_state_tax_rate 
  (MSRP : ℝ) (total_paid : ℝ) (insurance_rate : ℝ) (tax_rate : ℝ) 
  (insurance_cost : ℝ := insurance_rate * MSRP)
  (cost_before_tax : ℝ := MSRP + insurance_cost)
  (tax_amount : ℝ := total_paid - cost_before_tax) :
  MSRP = 30 → total_paid = 54 → insurance_rate = 0.2 → 
  tax_amount / cost_before_tax * 100 = tax_rate →
  tax_rate = 50 :=
by
  intros MSRP_val paid_val ins_rate_val comp_tax_rate
  sorry

end mandatory_state_tax_rate_l737_737157


namespace prime_product_le_four_pow_l737_737169

theorem prime_product_le_four_pow {k n : ℕ} (p : ℕ → ℕ) (H_prime : ∀ i, i ≤ k → Prime (p i))
  (H_consec : ∀ i, 2 ≤ i → p i = Nat.succ (p (i - 1))) (H_le_n : p k ≤ n) (H_n_ge_two : n ≥ 2) :
  (∏ i in Finset.range (k + 1), p i) ≤ 4 ^ n := 
sorry

end prime_product_le_four_pow_l737_737169


namespace telescoping_product_l737_737766

theorem telescoping_product :
  (∏ k in Finset.range 501, (4 * (k + 1)) / (4 * (k + 1) + 4)) = (1 / 502) := 
by
  sorry

end telescoping_product_l737_737766


namespace floor_sqrt_80_l737_737976

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737976


namespace y_sum_equals_three_l737_737575

noncomputable def sum_of_y_values (solutions : List (ℝ × ℝ × ℝ)) : ℝ :=
  solutions.foldl (fun acc (_, y, _) => acc + y) 0

theorem y_sum_equals_three (solutions : List (ℝ × ℝ × ℝ))
  (h1 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → x + y * z = 5)
  (h2 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → y + x * z = 8)
  (h3 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → z + x * y = 12) :
  sum_of_y_values solutions = 3 := sorry

end y_sum_equals_three_l737_737575


namespace sphere_radius_eq_cbrt_three_l737_737710

-- Definitions based on the given problem
def cone_radius : ℝ := 2
def cone_height : ℝ := 3

-- Volume of the cone
def volume_cone : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height

-- Volume of the sphere
def volume_sphere (R : ℝ) : ℝ := (4/3) * Real.pi * R^3

theorem sphere_radius_eq_cbrt_three :
  ∃ R : ℝ, volume_cone = volume_sphere R ∧ R = Real.cbrt 3 :=
sorry

end sphere_radius_eq_cbrt_three_l737_737710


namespace floor_sqrt_80_l737_737951

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737951


namespace floor_sqrt_80_l737_737818

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737818


namespace plane_equation_l737_737355

theorem plane_equation 
  (s t : ℝ)
  (x y z : ℝ)
  (parametric_plane : ℝ → ℝ → ℝ × ℝ × ℝ)
  (plane_eq : ℝ × ℝ × ℝ → Prop) :
  parametric_plane s t = (2 + 2 * s - t, 1 + 2 * s, 4 - 3 * s + t) →
  plane_eq (x, y, z) ↔ 2 * x - 5 * y + 2 * z - 7 = 0 :=
by
  sorry

end plane_equation_l737_737355


namespace distribution_properties_l737_737345

theorem distribution_properties (m d j s k : ℝ) (h1 : True)
  (h2 : True)
  (h3 : True)
  (h4 : 68 ≤ 100 ∧ 68 ≥ 0) -- 68% being a valid percentage
  : j = 84 ∧ s = s ∧ k = k :=
by
  -- sorry is used to highlight the proof is not included
  sorry

end distribution_properties_l737_737345


namespace no_such_function_exists_l737_737596

open Set Filter

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, (∀ x y : ℝ, 0 < x → 0 < y → f (x + y) ≥ f x + y * f (f x)) ∧
                  (∀ x : ℝ, 0 < x → f x > 0) :=
begin
  sorry
end

end no_such_function_exists_l737_737596


namespace monotonic_increasing_range_l737_737479

noncomputable def f (x a : ℝ) : ℝ := (Real.exp x) * (x + a) / x

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, x > 0 → (∀ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 < x2 → f x1 a ≤ f x2 a)) ↔ -4 ≤ a ∧ a ≤ 0 :=
sorry

end monotonic_increasing_range_l737_737479


namespace floor_sqrt_80_l737_737973

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737973


namespace largest_neg_multiple_of_7_greater_than_neg_150_l737_737262

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l737_737262


namespace interval_for_x_l737_737119

theorem interval_for_x (x : ℝ) 
  (hx1 : 1/x < 2) 
  (hx2 : 1/x > -3) : 
  x > 1/2 ∨ x < -1/3 :=
  sorry

end interval_for_x_l737_737119


namespace floor_sqrt_80_eq_8_l737_737904

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737904


namespace only_composite_positive_integer_with_divisors_form_l737_737019

theorem only_composite_positive_integer_with_divisors_form (n : ℕ) (composite : ¬Nat.Prime n ∧ 1 < n)
  (H : ∀ d ∈ Nat.divisors n, ∃ (a r : ℕ), a ≥ 0 ∧ r ≥ 2 ∧ d = a^r + 1) : n = 10 :=
by
  sorry

end only_composite_positive_integer_with_divisors_form_l737_737019


namespace find_phi_l737_737464

theorem find_phi 
  (ω : ℝ) (φ : ℝ) 
  (hω : ω > 0) 
  (hφ : 0 < φ ∧ φ < π) 
  (h_symm : ∀ x, f(x) = sin(ω * x + φ) → x = π / 4 ∨ x = 5 * π / 4) :
  φ = π / 4 :=
sorry

end find_phi_l737_737464


namespace floor_sqrt_80_l737_737962

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737962


namespace product_value_l737_737763

def telescoping_product : ℕ → ℚ
| 0 := 1
| (n + 1) := (4 * (n + 1)) / (4 * (n + 1) + 4) * telescoping_product n

theorem product_value : telescoping_product 501 = 1 / 502 := by
  sorry

end product_value_l737_737763


namespace pillar_height_at_E_l737_737375

-- Define the vertices A, B, C, E and their heights
def height (p : ℝ × ℝ × ℝ) := p.2.2

def A := (0, 0, 0)
def B := (10, 0, 0)
def C := (10 * Real.sqrt 2 / 2, 10 * Real.sqrt 2 / 2, 0)
def E := (-10, 0, 0)

-- Given heights
def height_A := 15
def height_B := 12
def height_C := 13

-- Points above the vertices
def P := (0, 0, height_A)
def Q := (10, 0, height_B)
def R := (10 * Real.sqrt 2 / 2, 10 * Real.sqrt 2 / 2, height_C)

-- Plane equation coefficients
def n1 := 20
def n2 := -10 * Real.sqrt 2
def n3 := -10
def d := -150

-- Calculate the height of the pillar at E
def height_at_E := 5

-- Statement
theorem pillar_height_at_E : 
  let z_E := height_at_E in
  n1 * (-10) + n2 * 0 + n3 * z_E = d :=
by
  -- Skip proof steps with sorry
  sorry

end pillar_height_at_E_l737_737375


namespace sum_of_coefficients_of_parabolas_kite_formed_l737_737398

theorem sum_of_coefficients_of_parabolas_kite_formed (a b : ℝ) 
  (h1 : ∃ (x : ℝ), y = ax^2 - 4)
  (h2 : ∃ (y : ℝ), y = 6 - bx^2)
  (h3 : (a > 0) ∧ (b > 0) ∧ (ax^2 - 4 = 0) ∧ (6 - bx^2 = 0))
  (h4 : kite_area = 18) :
  a + b = 125/36 := 
by sorry

end sum_of_coefficients_of_parabolas_kite_formed_l737_737398


namespace parabola_opening_downwards_l737_737448

theorem parabola_opening_downwards (a : ℝ) :
  (∀ x, 0 < x ∧ x < 3 → ax^2 - 2 * a * x + 3 > 0) → -1 < a ∧ a < 0 :=
by 
  intro h
  sorry

end parabola_opening_downwards_l737_737448


namespace largest_multiple_of_7_neg_greater_than_neg_150_l737_737285

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l737_737285


namespace silk_dyed_total_l737_737240

theorem silk_dyed_total (green_silk : ℕ) (pink_silk : ℕ) (h1 : green_silk = 61921) (h2 : pink_silk = 49500) :
  green_silk + pink_silk = 111421 :=
by {
  rw [h1, h2],
  exact rfl,
}

end silk_dyed_total_l737_737240


namespace smallest_k_rightmost_digit_is_1_l737_737036

def a_n (n : ℕ) : ℕ := (n + 7) ! / (n - 1) !

theorem smallest_k_rightmost_digit_is_1 :
  ∃ (k : ℕ), (∀ (m : ℕ), m < k → (a_n m % 10 ≠ 1)) ∧ (a_n k % 10 = 1) := by
  sorry

end smallest_k_rightmost_digit_is_1_l737_737036


namespace find_length_DM_l737_737166

-- Defining the problem statement and the necessary conditions
variables {A B C D M : Point}
variables {AB BC : ℕ}
variables {midpoint : Point → Point → Point}
variables {dist : Point → Point → ℝ}

-- Assume given conditions
variables (h1 : ∠BAD = ∠BCD)
variables (h2 : ∠BDC = 90)
variables (h3 : dist A B = 5)
variables (h4 : dist B C = 6)
variables (h5 : M = midpoint A C)

-- Define the goal
theorem find_length_DM :
  dist D M = (sqrt 11) / 2 :=
sorry

end find_length_DM_l737_737166


namespace analytical_expression_of_f_range_of_k_l737_737485

noncomputable def f : ℝ → ℝ := λ x, Real.log x / Real.log 2

theorem analytical_expression_of_f :
  (∀ a, (∀ x, f(x + 1) = a * Real.log (x + 1) / Real.log 3) → f(2) = 1 → f = λ x, Real.log x / Real.log 2) :=
by
  sorry

theorem range_of_k (k : ℝ) :
  (∀ x, (f(4^x + 1) - f(k * 2^x + k) = x) → (k = 1)) :=
by
  sorry

end analytical_expression_of_f_range_of_k_l737_737485


namespace chocolate_distribution_number_of_ways_to_distribute_chocolates_l737_737750

theorem chocolate_distribution : 
  ∃ (a b c : ℕ), a + b + c = 30 ∧ a ≥ 3 ∧ b ≥ 3 ∧ c ≥ 3 := sorry

theorem number_of_ways_to_distribute_chocolates (a b c : ℕ) 
  (ha : a + b + c = 30) (h1 : a ≥ 3) (h2 : b ≥ 3) (h3 : c ≥ 3) :
  nat.choose (21 + 3 - 1) (3 - 1) = 253 := sorry

end chocolate_distribution_number_of_ways_to_distribute_chocolates_l737_737750


namespace three_digit_numbers_l737_737100

theorem three_digit_numbers (n : ℕ) :
  n = 4 ↔ ∃ (x y : ℕ), 
  (100 ≤ 101 * x + 10 * y ∧ 101 * x + 10 * y < 1000) ∧ 
  (x ≠ 0 ∧ x ≠ 5) ∧ 
  (2 * x + y = 15) ∧ 
  (y < 10) :=
by { sorry }

end three_digit_numbers_l737_737100


namespace grazing_oxen_l737_737686

variable (x : ℕ)

theorem grazing_oxen (h1 : b_oxen = 12) (h2 : b_months = 5)
                   (h3 : c_oxen = 15) (h4 : c_months = 3)
                   (total_rent : ℕ) (c_share : ℕ)
                   (h5 : total_rent = 140) (h6 : c_share = 36) :
  let total_oxen_months := 7 * x + b_oxen * b_months + c_oxen * c_months in
  (c_oxen * c_months / total_oxen_months) * total_rent = c_share →
  x = 10 :=
by
  intros
  unfold total_oxen_months
  let total_oxen_months := 7 * x + 12 * 5 + 15 * 3
  dsimp only at *
  let total_oxen_months := 7 * x + 60 + 45
  let total_oxen_months := 7 * x + 105
  let total_c_months := 15 * 3
  let total_c_months := 45
  have h : 45 / total_oxen_months = c_share / total_rent,
  { sorry }
  have H2 : 45 * total_rent = total_oxen_months * c_share,
  { sorry }
  have h3 := 36 * total_oxen_months = 45 * 140,
  { sorry }
  have h4 : total_oxen_months := 140 * 45, { sorry }
  have h5 : total_oxen_months := 6300 / 36, { sorry }

  have h6 :=  7 * x +105 == 175,
  { sorry }
  have x := (175 - 105)/7,
  { sorry }

  exact h5
  sorry


end grazing_oxen_l737_737686


namespace floor_sqrt_80_eq_8_l737_737891

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737891


namespace largest_multiple_of_7_negation_greater_than_neg_150_l737_737278

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l737_737278


namespace floor_sqrt_80_l737_737966

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737966


namespace lengths_of_triangle_l737_737131

-- We define the lengths of the sides, their relationships, and the sine of the largest angle.
variables {a b c : ℝ}
noncomputable def determine_lengths : Prop :=
  a = b + 2 ∧
  b = c + 2 ∧
  a = c + 4 ∧
  sin (real.arcsin (sqrt 3 / 2)) = sqrt 3 / 2 ∧
  a = 7 ∧
  b = 5 ∧
  c = 3

-- We now state the theorem to prove that the lengths of the sides are as specified.
theorem lengths_of_triangle : determine_lengths :=
by sorry

end lengths_of_triangle_l737_737131


namespace floor_sqrt_80_eq_8_l737_737834

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737834


namespace train_speed_correct_l737_737745

-- Define the conditions
def train_length : ℝ := 55 -- meters
def man_speed_kmph : ℝ := 6 -- kmph
def passing_time : ℝ := 3 -- seconds

-- Man's speed in m/s
def man_speed_ms : ℝ := man_speed_kmph * (1000 / 3600)

-- Define the relative speed
def relative_speed : ℝ := train_length / passing_time

-- Define the train's speed in m/s
def train_speed_ms : ℝ := relative_speed - man_speed_ms

-- Convert train's speed to kmph
def train_speed_kmph : ℝ := train_speed_ms * (3600 / 1000)

-- The theorem to prove
theorem train_speed_correct : train_speed_kmph = 60 :=
by
  -- The proof would go here, but we'll skip it using 'sorry'
  sorry

end train_speed_correct_l737_737745


namespace exists_f_l737_737538

noncomputable def f : ℕ → ℝ := λ x, 2^x

theorem exists_f (f : ℕ → ℝ) (h1 : ∀ x : ℕ, f(x) > 0) (h2 : ∀ a b : ℕ, f(a + b) = f(a) * f(b)) (h3 : f 2 = 4) :
  ∃ c : ℕ → ℝ, ∀ x : ℕ, c x = 2^x :=
by
  have f_def : ∀ x : ℕ, f x = 2^x, from sorry
  use f
  intro x
  exact f_def x

end exists_f_l737_737538


namespace floor_sqrt_80_eq_8_l737_737836

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737836


namespace incorrect_statement_about_empty_set_l737_737680

theorem incorrect_statement_about_empty_set :
  ¬ (0 ∈ ∅) := 
by {
  sorry 
}

end incorrect_statement_about_empty_set_l737_737680


namespace opposite_of_neg_six_is_six_l737_737638

theorem opposite_of_neg_six_is_six : ∃ x, -6 + x = 0 ∧ x = 6 := by
  use 6
  split
  · rfl
  · rfl

end opposite_of_neg_six_is_six_l737_737638


namespace floor_sqrt_80_eq_8_l737_737906

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737906


namespace problem1_problem2_l737_737081

-- Definitions used directly from conditions
def inequality (m x : ℝ) : Prop := m * x ^ 2 - 2 * m * x - 1 < 0

-- Proof problem (1)
theorem problem1 (m : ℝ) (h : ∀ x : ℝ, inequality m x) : -1 < m ∧ m ≤ 0 :=
sorry

-- Proof problem (2)
theorem problem2 (x : ℝ) (h : ∀ m : ℝ, |m| ≤ 1 → inequality m x) :
  (1 - Real.sqrt 2 < x ∧ x < 1) ∨ (1 < x ∧ x < 1 + Real.sqrt 2) :=
sorry

end problem1_problem2_l737_737081


namespace total_clothes_washed_l737_737388

theorem total_clothes_washed (cally_white_shirts : ℕ) (cally_colored_shirts : ℕ) (cally_shorts : ℕ) (cally_pants : ℕ) 
                             (danny_white_shirts : ℕ) (danny_colored_shirts : ℕ) (danny_shorts : ℕ) (danny_pants : ℕ) 
                             (total_clothes : ℕ)
                             (hcally : cally_white_shirts = 10 ∧ cally_colored_shirts = 5 ∧ cally_shorts = 7 ∧ cally_pants = 6)
                             (hdanny : danny_white_shirts = 6 ∧ danny_colored_shirts = 8 ∧ danny_shorts = 10 ∧ danny_pants = 6)
                             (htotal : total_clothes = 58) : 
  cally_white_shirts + cally_colored_shirts + cally_shorts + cally_pants + 
  danny_white_shirts + danny_colored_shirts + danny_shorts + danny_pants = total_clothes := 
by {
  sorry
}

end total_clothes_washed_l737_737388


namespace geometric_seq_cond_l737_737057

-- Define the sequence and its sum
def seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, a 0 = 3 → (sqrt ((a.sum n) + 1)) = (sqrt ((a.sum 0) + 1) * (2 ^ (n - 1)))

-- Define a_n
def a_n (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  if n = 0 then a 0
  else (a.sum n) - (a.sum (n-1))

-- The main statement of equivalence
theorem geometric_seq_cond (a : ℕ → ℝ) : 
  (∀ n : ℕ, (a 0 = 3 ∧ sqrt ((a.sum n) + 1) = (sqrt ((a.sum 0) + 1) * (2 ^ (n - 1)))) ↔ (∀ n : ℕ, n ≥ 1 → a (n + 1) = 4 * a n)) :=
sorry

end geometric_seq_cond_l737_737057


namespace floor_of_sqrt_80_l737_737926

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737926


namespace probability_of_negative_product_l737_737689

def m : Set ℤ := {-6, -5, -4, -3, -2, -1}
def t : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2}

-- Define the count of negative elements in set m
def count_neg_m : ℤ := 6

-- Define the count of positive elements in set t
def count_pos_t : ℤ := 3

-- Define the total combinations count
def total_combinations : ℤ := (Set.cardinal m).toInt * (Set.cardinal t).toInt -- cardinal function returns cardinality, convert cardinal to int

-- Define the favorable combinations count
def favorable_combinations : ℤ := count_neg_m * count_pos_t

-- Statement to prove
theorem probability_of_negative_product : 
    (favorable_combinations : ℚ) / (total_combinations : ℚ) = 3 / 8 := 
sorry

end probability_of_negative_product_l737_737689


namespace color_diff_l737_737588

def is_correct_coloring (grid : List (List Bool)) : Prop :=
  ∀ i j, (i < grid.length ∧ j < grid.head.length) →
  grid.geti i j →
  ¬ (i > 0 ∧ grid.geti (i - 1) j) ∧
  ¬ (i < grid.length - 1 ∧ grid.geti (i + 1) j) ∧
  ¬ (j > 0 ∧ grid.geti i (j - 1)) ∧
  ¬ (j < grid.head.length - 1 ∧ grid.geti i (j + 1))

def num_colorings (n : ℕ) (is_even : Bool) : ℕ :=
  (List.list_product (list.replicate 2 [Bool.ff, Bool.tt])).count
  (λ grid, is_correct_coloring grid ∧
              grid.foldr (λ row acc, acc + row.count id) 0 % 2 == if is_even then 0 else 1)

def A_n (n : ℕ) : ℕ := num_colorings n true
def B_n (n : ℕ) : ℕ := num_colorings n false

theorem color_diff (n : ℕ) : A_n n - B_n n = 1 ∨ A_n n - B_n n = -1 := 
sorry

end color_diff_l737_737588


namespace opposite_of_2_is_neg_2_l737_737640

theorem opposite_of_2_is_neg_2 : ∃ x : ℤ, 2 + x = 0 ∧ x = -2 :=
by
  use -2
  split
  sorry

end opposite_of_2_is_neg_2_l737_737640


namespace area_of_OBEC_is_72_l737_737722

variable {Point : Type}
variable Line : Type

structure point (x y : ℝ := 0)

def A : point := point.mk 5 0
def B : point := point.mk 0 15
def C : point := point.mk 6 0
def E : point := point.mk 3 6
def origin : point := point.mk 0 0

def firstLine (p : point) : Prop := p.y = -3 * p.x + 15
def secondLine (p : point) : Prop := p.y = if p.x = 6 then 0 else 2 * p.x - 12

theorem area_of_OBEC_is_72 (points_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ E ∧ E ≠ A ∧ origin ≠ A ∧ origin ≠ B ∧ origin ≠ C ∧ origin ≠ E) 
  (A_on_firstLine : firstLine A) (B_on_firstLine : firstLine B) (E_on_firstLine : firstLine E) 
  (C_on_secondLine : secondLine C) (E_on_secondLine : secondLine E) : 
  let area_OBEC := 1 / 2 * (abs ((0 - 15) * (6 - 0) + (6 - 0) * (6 - 15) + (6 - 3) * (0 - 6) + (3 - 0) * (15 - 0))) in
  area_OBEC = 72 := 
by sorry

end area_of_OBEC_is_72_l737_737722


namespace find_x_value_l737_737210

theorem find_x_value (x : ℂ) (h1 : x ^ 2018 - 3 * x + 2 = 0) (h2 : x ≠ 2) : 
  x ^ 2017 + x ^ 2016 + ... + x + 1 = 3 := sorry

end find_x_value_l737_737210


namespace floor_sqrt_80_l737_737829

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737829


namespace stamp_problem_l737_737999

/-- Define the context where we have stamps of 7, n, and (n + 2) cents, and 120 cents being the largest
    value that cannot be formed using these stamps -/
theorem stamp_problem (n : ℕ) (h : ∀ k, k > 120 → ∃ a b c, k = 7 * a + n * b + (n + 2) * c) (hn : ¬ ∃ a b c, 120 = 7 * a + n * b + (n + 2) * c) : n = 22 :=
sorry

end stamp_problem_l737_737999


namespace minimal_triangle_area_y_coordinate_l737_737352

theorem minimal_triangle_area_y_coordinate :
  ∀ (A B : ℝ × ℝ), A.2 = A.1^2 ∧ B.2 = B.1^2 ∧ ∃ (x₀ x₁ : ℝ), A = (x₀, x₀^2) ∧ B = (x₁, x₁^2) ∧ 
  2 * x₀ * (x₀ + x₁) = -1 ∧ x₁ = - (1 / (2 * x₀)) - x₀ ∧ 
  ∀ (S : ℝ), S = 1/2 * abs (x₀ - x₁) * (x₀^2 + 1/2) ∧ 
  (∀ y : ℝ, y = A.2 → y = (1 / 24) * (-3 + real.sqrt 33)) → 
  A.2 = (1 / 24) * (-3 + real.sqrt 33) :=
by 
  -- Proof goes here
  sorry

end minimal_triangle_area_y_coordinate_l737_737352


namespace multiples_of_5_not_10_or_6_count_l737_737099

theorem multiples_of_5_not_10_or_6_count : 
    (finset.filter 
        (λ n, n % 5 = 0 ∧ n % 10 ≠ 0 ∧ n % 6 ≠ 0) 
        (finset.range 200)).card 
    = 20 := 
by sorry

end multiples_of_5_not_10_or_6_count_l737_737099


namespace probability_of_2_pow_n_ends_with_digit_2_probability_of_2_pow_n_ends_with_digits_12_l737_737291

noncomputable def prob_ends_with_digit_2 : ℚ :=
  if n : ℕ, n > 0 ∧ ∃ k : ℕ, n = 4 * k + 1 then 1 / 4 else 0
  
noncomputable def prob_ends_with_digits_12 : ℚ :=
  if n : ℕ, n > 0 ∧ ∃ k : ℕ, n = 20 * k + 9 then 1 / 20 else 0

theorem probability_of_2_pow_n_ends_with_digit_2 :
  prob_ends_with_digit_2 = 0.25 :=
by
  sorry

theorem probability_of_2_pow_n_ends_with_digits_12 :
  prob_ends_with_digits_12 = 0.05 :=
by
  sorry

end probability_of_2_pow_n_ends_with_digit_2_probability_of_2_pow_n_ends_with_digits_12_l737_737291


namespace probability_heart_or_king_l737_737329

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l737_737329


namespace floor_sqrt_80_l737_737824

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737824


namespace correct_equation_by_moving_digit_l737_737002

theorem correct_equation_by_moving_digit :
  (10^2 - 1 = 99) → (101 = 102 - 1) :=
by
  intro h
  sorry

end correct_equation_by_moving_digit_l737_737002


namespace emails_received_l737_737543

variable (x y : ℕ)

theorem emails_received (h1 : 3 + 6 = 9) (h2 : x + y + 9 = 10) : x + y = 1 := by
  sorry

end emails_received_l737_737543


namespace largest_neg_multiple_of_7_greater_than_neg_150_l737_737258

theorem largest_neg_multiple_of_7_greater_than_neg_150 : 
  ∃ (n : ℤ), (n % 7 = 0) ∧ (-n > -150) ∧ (∀ m : ℤ, (m % 7 = 0) ∧ (-m > -150) → m ≤ n) :=
begin
  use 147,
  split,
  { norm_num }, -- Verifies that 147 is a multiple of 7
  split,
  { norm_num }, -- Verifies that -147 > -150
  { intros m h,
    obtain ⟨k, rfl⟩ := (zmod.int_coe_zmod_eq_zero_iff_dvd m 7).mp h.1,
    suffices : k ≤ 21, { rwa [int.nat_abs_of_nonneg (by norm_num : (7 : ℤ) ≥ 0), ←abs_eq_nat_abs, int.abs_eq_nat_abs, nat.abs_of_nonneg (zero_le 21), ← int.le_nat_abs_iff_coe_nat_le] at this },
    have : -m > -150 := h.2,
    rwa [int.lt_neg, neg_le_neg_iff] at this,
    norm_cast at this,
    exact this
  }
end

end largest_neg_multiple_of_7_greater_than_neg_150_l737_737258


namespace at_least_one_heart_or_king_l737_737338

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l737_737338


namespace parallelogram_count_l737_737592

-- Define a triangle with vertices A, B, and C
structure Triangle (α : Type) :=
  (A B C : α)

-- Define equilateral triangle constructions on each side of the initial triangle
structure EquilateralTriangle (α : Type) :=
  (inner_outer : α × α) -- Indicates if the triangle is inward or outward

-- Define a function that constructs both inward and outward equilateral triangles
def constructEquilateralTriangles {α : Type} (T : Triangle α) : List (EquilateralTriangle α) :=
  sorry

-- Function to determine if four points form a parallelogram
def isParallelogram {α : Type} (p1 p2 p3 p4 : α) : Prop :=
  sorry

-- Formalize the conditions and the proof problem
theorem parallelogram_count (α : Type) [Inhabited α] :
  ∀ (T : Triangle α),
    9 = (constructEquilateralTriangles T).choose 4 |
        (λ vertices, isParallelogram vertices.head vertices.tail.head vertices.tail.tail.head vertices.tail.tail.tail.head) :=
begin
  sorry
end

end parallelogram_count_l737_737592


namespace ratio_of_percent_changes_l737_737684

variables (P U U_new : ℝ)
noncomputable def new_price := 0.8 * P
noncomputable def unchanged_revenue_condition : Prop := (P * U = new_price * U_new)

theorem ratio_of_percent_changes (h : unchanged_revenue_condition P U U_new) :
  let percent_increase_units_sold := 25
  let percent_decrease_price := 20
  (percent_increase_units_sold / percent_decrease_price).to_real = 1.25 :=
by
  have h1 : U_new = U / 0.8,
  { sorry }
  have h2 : U_new = 1.25 * U,
  { sorry }
  have percent_increase_units_sold := ((U_new - U) / U) * 100,
  have percent_decrease_price := 20,
  have h3 : (percent_increase_units_sold / percent_decrease_price).to_real = 1.25,
  { sorry }
  exact h3

end ratio_of_percent_changes_l737_737684


namespace determine_d_l737_737534

theorem determine_d (m n d : ℝ) (p : ℝ) (hp : p = 0.6666666666666666) 
  (h1 : m = 3 * n + 5) (h2 : m + d = 3 * (n + p) + 5) : d = 2 :=
by {
  sorry
}

end determine_d_l737_737534


namespace juice_cans_count_l737_737738

theorem juice_cans_count :
  let original_price := 12 
  let discount := 2 
  let tub_sale_price := original_price - discount 
  let tub_quantity := 2 
  let ice_cream_total := tub_quantity * tub_sale_price 
  let total_payment := 24 
  let juice_cost_per_5cans := 2 
  let remaining_amount := total_payment - ice_cream_total 
  let sets_of_juice_cans := remaining_amount / juice_cost_per_5cans 
  let cans_per_set := 5 
  2 * cans_per_set = 10 :=
by
  sorry

end juice_cans_count_l737_737738


namespace men_took_dip_l737_737608

theorem men_took_dip 
  (tank_length : ℝ) (tank_breadth : ℝ) (water_rise_cm : ℝ) (man_displacement : ℝ)
  (H1 : tank_length = 40) (H2 : tank_breadth = 20) (H3 : water_rise_cm = 25) (H4 : man_displacement = 4) :
  let water_rise_m := water_rise_cm / 100
  let total_volume_displaced := tank_length * tank_breadth * water_rise_m
  let number_of_men := total_volume_displaced / man_displacement
  number_of_men = 50 :=
by
  sorry

end men_took_dip_l737_737608


namespace derivative_at_pi_over_three_l737_737315

def f (x : ℝ) : ℝ := x + Real.sin x

theorem derivative_at_pi_over_three :
  (derivative f (π / 3)) = 3 / 2 :=
by
  sorry

end derivative_at_pi_over_three_l737_737315


namespace total_amount_lent_l737_737244

theorem total_amount_lent (A T : ℝ) (hA : A = 15008) (hInterest : 0.08 * A + 0.10 * (T - A) = 850) : 
  T = 11501.6 :=
by
  sorry

end total_amount_lent_l737_737244


namespace train_cross_tunnel_time_l737_737493

theorem train_cross_tunnel_time 
  (train_length : ℕ)
  (train_speed_kmph : ℕ)
  (tunnel_length : ℕ)
  (train_length = 100) 
  (train_speed_kmph = 72) 
  (tunnel_length = 1400) : 
  (train_length + tunnel_length) / (train_speed_kmph * 1000 / 3600) = 75 := 
by 
  sorry

end train_cross_tunnel_time_l737_737493


namespace count_real_numbers_a_with_integer_roots_l737_737038

def integer_roots_count : Nat := 15

theorem count_real_numbers_a_with_integer_roots :
  ∀ a : ℝ, (∃ r s : ℤ, r * s = 12 * a ∧ r + s = -a) ↔ a ∈ (Finset.range 15).map (λ k, (k.succ : ℝ)) := sorry

end count_real_numbers_a_with_integer_roots_l737_737038


namespace area_increase_factor_l737_737991

theorem area_increase_factor (s : ℝ) :
  let A_original := s^2
  let A_new := (3 * s)^2
  A_new / A_original = 9 := by
  sorry

end area_increase_factor_l737_737991


namespace part1_part2_l737_737597

-- Part 1
theorem part1 (x y : ℝ) 
  (h1 : x + 2 * y = 9) 
  (h2 : 2 * x + y = 6) :
  (x - y = -3) ∧ (x + y = 5) :=
sorry

-- Part 2
theorem part2 (x y : ℝ) 
  (h1 : x + 2 = 5) 
  (h2 : y - 1 = 4) :
  x = 3 ∧ y = 5 :=
sorry

end part1_part2_l737_737597


namespace average_speed_to_SF_l737_737325

theorem average_speed_to_SF (v d : ℝ) (h1 : d ≠ 0) (h2 : v ≠ 0) :
  (2 * d / ((d / v) + (2 * d / v)) = 34) → v = 51 :=
by
  -- proof goes here
  sorry

end average_speed_to_SF_l737_737325


namespace find_f_4_l737_737459

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_4 : (∀ x : ℝ, f (x / 2 - 1) = 2 * x + 3) → f 4 = 23 :=
by
  sorry

end find_f_4_l737_737459


namespace floor_sqrt_80_l737_737821

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737821


namespace sum_remainder_l737_737668

theorem sum_remainder (p q r : ℕ) (hp : p % 15 = 11) (hq : q % 15 = 13) (hr : r % 15 = 14) : 
  (p + q + r) % 15 = 8 :=
by
  sorry

end sum_remainder_l737_737668


namespace point_in_circle_probability_l737_737759

noncomputable def probability_point_in_circle : ℚ := sorry

theorem point_in_circle_probability :
  (∃ (P : ℚ), P = probability_point_in_circle ∧ P = 1 / 9) :=
by
  sorry
  
-- Definitions to use within Lean
def valid_pairs : finset (ℕ × ℕ) := 
({1, 2, 3, 4, 5, 6}).product {1, 2, 3, 4, 5, 6}

def within_circle (p : ℕ × ℕ) : Prop := p.1^2 + p.2^2 < 9

def favorable_pairs : finset (ℕ × ℕ) := valid_pairs.filter within_circle

def probability_point_in_circle : ℚ :=
(favorable_pairs.card : ℚ) / (valid_pairs.card : ℚ)

end point_in_circle_probability_l737_737759


namespace feasible_cube_net_l737_737003

theorem feasible_cube_net : 
  ∃ shape : set (ℕ × ℕ), shape ⊆ (finset.product (finset.range 3) (finset.range 3)) ∧ 
                          set.card shape = 6 ∧ 
                          is_cube_net shape :=
sorry

end feasible_cube_net_l737_737003


namespace floor_sqrt_80_l737_737811

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737811


namespace largest_multiple_of_7_negation_greater_than_neg_150_l737_737279

theorem largest_multiple_of_7_negation_greater_than_neg_150 : 
  ∃ k : ℤ, k * 7 = 147 ∧ ∀ n : ℤ, (k < n → n * 7 ≤ 150) :=
by
  use 21
  sorry

end largest_multiple_of_7_negation_greater_than_neg_150_l737_737279


namespace tan_alpha_plus_beta_l737_737457

open Real

theorem tan_alpha_plus_beta (A alpha beta : ℝ) (h1 : sin alpha = A * sin (alpha + beta)) (h2 : abs A > 1) :
  tan (alpha + beta) = sin beta / (cos beta - A) :=
by
  sorry

end tan_alpha_plus_beta_l737_737457


namespace consultation_session_probability_l737_737350

noncomputable def consultation_probability : ℝ :=
  let volume_cube := 3 * 3 * 3
  let volume_valid := 9 - 2 * (1/3 * 2.25 * 1.5)
  volume_valid / volume_cube

theorem consultation_session_probability : consultation_probability = 1 / 4 :=
by
  sorry

end consultation_session_probability_l737_737350


namespace solution_exists_l737_737158

-- Define the primary condition that must hold for any solution.
def condition (p : Nat) : Prop :=
  ∃ n : Nat, n = 3 * p + 1

-- Define the goal based on the correct answer identified in the solution.
theorem solution_exists (p : Nat) :
  condition p → ∃ q : Nat, q = 25 :=
by
  intro h
  use 3 * p + 1
  rw ← h
  sorry

end solution_exists_l737_737158


namespace opposite_of_neg_six_is_six_l737_737639

theorem opposite_of_neg_six_is_six : ∃ x, -6 + x = 0 ∧ x = 6 := by
  use 6
  split
  · rfl
  · rfl

end opposite_of_neg_six_is_six_l737_737639


namespace a_eq_formula_a_5_eq_2034_l737_737447

def num_adj_not_adjacent (n : ℕ) (digits : Fin 5 → ℕ) : ℕ := 
  -- To be implemented: counts the number of n-digit numbers with 1, 2, 3, 4, 5 where 1 and 2 are not adjacent.
  sorry

noncomputable def a (n : ℕ) : ℕ := 
  let x := (3 + Real.sqrt 7) / (2 * Real.sqrt 7) * (2 + Real.sqrt 7) ^ n 
          - (3 - Real.sqrt 7) / (2 * Real.sqrt 7) * (2 - Real.sqrt 7) ^ n in
  let y := 9 * 2 ^ n - 13 in
  x - y

theorem a_eq_formula (n : ℕ) (h : n ≥ 3) : 
  a n = (3 + Real.sqrt 7) / (2 * Real.sqrt 7) * (2 + Real.sqrt 7) ^ n 
        - (3 - Real.sqrt 7) / (2 * Real.sqrt 7) * (2 - Real.sqrt 7) ^ n 
        - (9 * 2 ^ n - 13) := sorry

theorem a_5_eq_2034 : a 5 = 2034 := by
  have h : 5 ≥ 3 := by linarith
  rw [a_eq_formula 5 h]
  -- Computation to verify the result matches the expected value
  sorry

end a_eq_formula_a_5_eq_2034_l737_737447


namespace number_of_expressible_integers_l737_737496

-- Define the floor functions as they appear in the problem.
noncomputable def floor10 (x : ℝ) : ℤ := int.floor (10 * x)
noncomputable def floor12 (x : ℝ) : ℤ := int.floor (12 * x)
noncomputable def floor14 (x : ℝ) : ℤ := int.floor (14 * x)
noncomputable def floor16 (x : ℝ) : ℤ := int.floor (16 * x)

-- Define the main function g(x)
noncomputable def g (x : ℝ) : ℤ := floor10 x + floor12 x + floor14 x + floor16 x

-- Prove the main statement
theorem number_of_expressible_integers :
  (∃ (f : ℤ → bool), (∀ n, 1 ≤ n ∧ n ≤ 5000 → f n ↔ ∃ x : ℝ, g x = n) ∧ (∑ i in finset.range 5000, if f i then 1 else 0) = 3365) :=
sorry

end number_of_expressible_integers_l737_737496


namespace jenny_change_l737_737539

def cost_per_page : ℝ := 0.10
def pages_per_essay : ℝ := 25
def num_essays : ℝ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℝ := 7
def amount_paid : ℝ := 40.00

theorem jenny_change : 
  let cost_of_printing := num_essays * pages_per_essay * cost_per_page in
  let cost_of_pens := num_pens * cost_per_pen in
  let total_cost := cost_of_printing + cost_of_pens in
  amount_paid - total_cost = 12.00 :=
by
  -- Definitions
  let cost_of_printing := num_essays * pages_per_essay * cost_per_page
  let cost_of_pens := num_pens * cost_per_pen
  let total_cost := cost_of_printing + cost_of_pens

  -- Proof
  sorry

end jenny_change_l737_737539


namespace count_correct_propositions_l737_737490

variables {Line : Type*} 
variables (a b c : Line)

-- Correct proposition (from the solution)
def parallel_transitive (h1 : a ∥ b) (h2 : b ∥ c) : a ∥ c := sorry

-- Incorrect proposition 1 (perpendicular relationship not transitive)
example (h1 : a ⟂ b) (h2 : b ⟂ c) : ¬ (a ⟂ c) := sorry

-- Incorrect proposition 2 (coplanarity not transitive)
example (ha : ∃ p : set Line, a ∈ p ∧ b ∈ p) 
        (hb : ∃ q : set Line, b ∈ q ∧ c ∈ q) : 
        ¬ (∃ r : set Line, a ∈ r ∧ c ∈ r) := sorry

-- Theorem: The number of correct propositions is 1.
theorem count_correct_propositions : 1 = 1 := rfl

end count_correct_propositions_l737_737490


namespace remove_zero_maximizes_pairs_l737_737663

def integers : List ℤ := [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

def pairs_generate (l : List ℤ) : List (ℤ × ℤ) :=
  List.filter (λ (p : ℤ × ℤ), p.1 + p.2 = 9 ∧ p.1 ≠ p.2) (l.product l)

def number_of_pairs (l : List ℤ) : ℕ :=
  (pairs_generate l).length

theorem remove_zero_maximizes_pairs :
  ∀ r ∈ integers, number_of_pairs (List.erase integers r) ≤ number_of_pairs (List.erase integers 0) :=
  by
  sorry

end remove_zero_maximizes_pairs_l737_737663


namespace num_of_fractions_l737_737374

def is_fraction (x : Real) : Prop :=
  ∃ a b : Int, b ≠ 0 ∧ x = a / b

def given_numbers : List Real := [2 / 5, -6, 25, 0, 3.14, 20 / 100]

def count_fractions (lst : List Real) : Nat :=
  lst.countp is_fraction

theorem num_of_fractions : count_fractions given_numbers = 3 := 
  by sorry

end num_of_fractions_l737_737374


namespace floor_sqrt_80_eq_8_l737_737890

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737890


namespace x_intercept_of_line_l737_737992

theorem x_intercept_of_line :
  ∃ x : ℝ, 4 * x - 3 * 0 = 24 ∧ (x, 0) = (6, 0) :=
by
  use 6
  split
  · simp
  · rfl

end x_intercept_of_line_l737_737992


namespace minimum_value_of_k_l737_737077

noncomputable def f(x : ℝ) : ℝ := x * (1 + x) ^ 2

def F(a : ℝ) : ℝ := if a ≥ -1 then -4 / 27 else f a

theorem minimum_value_of_k (a : ℝ) (ha : a < 0) :
  let k := F(a) / a in
  k ≥ 1 / 9 ∧ (a = -4 / 3 → k = 1 / 9) := by
  sorry

end minimum_value_of_k_l737_737077


namespace find_p_l737_737092

def vector (α : Type*) := (α × α × α)

def a := (2, -2, 4) : vector ℚ
def b := (0, 6, 0) : vector ℚ
def p := (8/7, 10/7, 16/7) : vector ℚ

def collinear (u v w : vector ℚ) : Prop :=
  let (ux, uy, uz) := u
  let (vx, vy, vz) := v
  let (wx, wy, wz) := w
  ∃ k : ℚ, v = (k * ux, k * uy, k * uz) ∧ ∃ l : ℚ, w = (l * ux, l * uy, l * uz)

theorem find_p :
  (∃ v : vector ℚ, p = a ∧ p = b) ∧ collinear a b p ∧ (let (px, py, pz) := p in -2 * px + 8 * py - 4 * pz = 0) :=
  sorry

end find_p_l737_737092


namespace range_of_a_l737_737443

noncomputable def f : ℝ → ℝ := sorry

def is_even_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f(x + 1) = f(-x + 1)

def monotonic_decreasing (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, 1 ≤ x1 → 1 ≤ x2 → x1 ≠ x2 → (x2 - x1) * (f x2 - f x1) < 0

variables (f : ℝ → ℝ) [is_even : is_even_function f] [mono_decr : monotonic_decreasing f]

theorem range_of_a (a : ℝ) (h : f (Real.log a) ≥ f (-1)) : 
  a ∈ Set.Icc (1 / Real.exp 1) (Real.exp 3) :=
sorry

end range_of_a_l737_737443


namespace smallest_four_digit_number_with_distinct_digits_l737_737307

theorem smallest_four_digit_number_with_distinct_digits : 
  ∃ n : ℕ, n = 1023 ∧ (1000 ≤ n ∧ n < 10000) ∧ (∀ i j : ℕ, i ≠ j → ∀ d1 d2 : ℕ, digit n i d1 → digit n j d2 → d1 ≠ d2) :=
by
  sorry

end smallest_four_digit_number_with_distinct_digits_l737_737307


namespace problem_1_problem_2_l737_737051

-- Define the function f(x) = |x + a| + |x|
def f (x : ℝ) (a : ℝ) : ℝ := abs (x + a) + abs x

-- (Ⅰ) Prove that for a = 1, the solution set for f(x) ≥ 2 is (-∞, -1/2] ∪ [3/2, +∞)
theorem problem_1 : 
  ∀ (x : ℝ), f x 1 ≥ 2 ↔ (x ≤ -1/2 ∨ x ≥ 3/2) :=
by
  intro x
  sorry

-- (Ⅱ) Prove that if there exists x ∈ ℝ such that f(x) < 2, then -2 < a < 2
theorem problem_2 :
  (∃ (x : ℝ), f x a < 2) → -2 < a ∧ a < 2 :=
by
  intro h
  sorry

end problem_1_problem_2_l737_737051


namespace value_of_y_l737_737104

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l737_737104


namespace product_of_large_and_small_l737_737308

theorem product_of_large_and_small : 
  ∃ (a b : ℕ), 
  (a = 642 ∧ b = 204) ∧ 
  (∀ (x y : ℕ), 
    (x = 642 → y = 204 → a * b = 130968)) :=
begin
  use [642, 204],
  split,
  { split; refl },
  { intros x y hx hy,
    rw [hx, hy],
    norm_num }
end

end product_of_large_and_small_l737_737308


namespace sqrt_floor_eight_l737_737860

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737860


namespace max_parts_by_rectangles_l737_737150

theorem max_parts_by_rectangles (n : ℕ) : 
  ∃ S : ℕ, S = 2 * n^2 - 2 * n + 2 :=
by
  sorry

end max_parts_by_rectangles_l737_737150


namespace proof_equiv_expression_l737_737385

variable (x y : ℝ)

def P : ℝ := x^2 + y^2
def Q : ℝ := x^2 - y^2

theorem proof_equiv_expression :
  ( (P x y)^2 + (Q x y)^2 ) / ( (P x y)^2 - (Q x y)^2 ) - 
  ( (P x y)^2 - (Q x y)^2 ) / ( (P x y)^2 + (Q x y)^2 ) = 
  (x^4 - y^4) / (x^2 * y^2) :=
by
  sorry

end proof_equiv_expression_l737_737385


namespace find_angle_GG2G1_l737_737564

-- Defining the ten points on a circle
variables {G A1 A2 A3 A4 B1 B2 B3 B4 B5 : Point}

-- Defining the conditions
def is_regular_pentagon (G A1 A2 A3 A4 : Point) : Prop := 
-- Definition of a regular pentagon here (angles, sides equal)

def is_regular_hexagon (G B1 B2 B3 B4 B5 : Point) : Prop := 
-- Definition of a regular hexagon here (angles, sides equal)

def on_minor_arc (B1 G A1 : Point) : Prop :=
-- Definition to indicate that B1 lies on the minor arc between G and A1

def intersects (X Y Z : Point) (M : Point) : Prop :=
-- Definition to indicate that segments XY and YZ intersect at M

theorem find_angle_GG2G1 :
    is_on_circle G ∧ is_on_circle A1 ∧ is_on_circle A2 ∧ is_on_circle A3 ∧ is_on_circle A4 ∧ 
    is_on_circle B1 ∧ is_on_circle B2 ∧ is_on_circle B3 ∧ is_on_circle B4 ∧ is_on_circle B5 ∧
    is_regular_pentagon G A1 A2 A3 A4 ∧ 
    is_regular_hexagon G B1 B2 B3 B4 B5 ∧ 
    on_minor_arc B1 G A1 ∧ 
    intersects B5 B3 B1 A2 G1 ∧ 
    intersects B5 A3 G B3 G2 
    → angle G G2 G1 = 12 :=
by
    sorry

end find_angle_GG2G1_l737_737564


namespace steve_speed_on_way_back_l737_737691

theorem steve_speed_on_way_back (d t v : ℝ) (h1: d = 28) (h2: t = 6) (h3: v = 7) :
  let to_work_time := d / v,
      back_work_time := d / (2 * v),
      total_time := to_work_time + back_work_time
  in total_time = t → 2 * v = 14 :=
by
  sorry

end steve_speed_on_way_back_l737_737691


namespace floor_of_sqrt_80_l737_737920

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737920


namespace total_glass_panels_in_house_l737_737365

/-- Prove the total number of glass panels in the whole house is 80. -/
theorem total_glass_panels_in_house : 
  (∀ (window_panels : ℕ)(double_windows : ℕ)(single_windows : ℕ)(panels_per_double_window : ℕ)(panels_per_single_window : ℕ),
     (window_panels = 4) → 
     (double_windows = 6) → 
     (single_windows = 8) → 
     (panels_per_double_window = 2 * window_panels) → 
     (panels_per_single_window = window_panels) → 
     (double_windows * panels_per_double_window + single_windows * panels_per_single_window = 80)) :=
by 
  intros window_panels double_windows single_windows panels_per_double_window panels_per_single_window h_window_panels 
         h_double_windows h_single_windows h_panels_per_double_window h_panels_per_single_window 
  rw [h_window_panels, h_double_windows, h_single_windows, h_panels_per_double_window, h_panels_per_single_window]
  exact eq.refl 80

end total_glass_panels_in_house_l737_737365


namespace jenny_change_l737_737542

/-!
## Problem statement

Jenny is printing 7 copies of her 25-page essay. It costs $0.10 to print one page.
She also buys 7 pens, each costing $1.50. If she pays with $40, calculate the change she should get.
-/

def cost_per_page : ℝ := 0.10
def pages_per_copy : ℕ := 25
def num_copies : ℕ := 7
def cost_per_pen : ℝ := 1.50
def num_pens : ℕ := 7
def amount_paid : ℝ := 40.0

def total_pages : ℕ := num_copies * pages_per_copy

def cost_printing : ℝ := total_pages * cost_per_page
def cost_pens : ℝ := num_pens * cost_per_pen

def total_cost : ℝ := cost_printing + cost_pens

theorem jenny_change : amount_paid - total_cost = 12 := by
  -- proof here
  sorry

end jenny_change_l737_737542


namespace f_1001_l737_737562

noncomputable def f : ℝ → ℝ := sorry -- since we're dealing with a non-specified function

axiom f_pos (x : ℝ) (h : x > 0) : f(x) > 0

axiom f_eq_sqrt (x y : ℝ) (h : x > y) : f(x - y) = sqrt(f(x * y) + 3)

theorem f_1001 : f(1001) = 3 := 
by {
  -- proof goes here
  sorry
}

end f_1001_l737_737562


namespace jason_nickels_is_52_l737_737613

theorem jason_nickels_is_52 (n q : ℕ) (h1 : 5 * n + 10 * q = 680) (h2 : q = n - 10) : n = 52 :=
sorry

end jason_nickels_is_52_l737_737613


namespace meeting_distance_from_top_l737_737545

section

def total_distance : ℝ := 12
def uphill_distance : ℝ := 6
def downhill_distance : ℝ := 6
def john_start_time : ℝ := 0.25
def john_uphill_speed : ℝ := 12
def john_downhill_speed : ℝ := 18
def jenny_uphill_speed : ℝ := 14
def jenny_downhill_speed : ℝ := 21

theorem meeting_distance_from_top : 
  ∃ (d : ℝ), d = 6 - 14 * ((0.25) + 6 / 14 - (1 / 2) - (6 - 18 * ((1 / 2) + d / 18))) / 14 ∧ d = 45 / 32 :=
sorry

end

end meeting_distance_from_top_l737_737545


namespace f_n_expression_can_form_triangle_l737_737438

def f1 (x : ℝ) : ℝ := x

def f (n : ℕ) (x : ℝ) : ℕ → ℝ
| 0 := f1 x
| (n+1) := f n x + x * (f n x).derivative

axiom f_base : ∀ (n : ℕ), n > 0 → (f n 1 = 1)

def g (n : ℕ) (x m : ℝ) : ℝ :=
f n x + f n (m - x)

axiom x_in_interval (x1 x2 x3 m : ℝ) : x1 ∈ set.Icc (m / 2) (2 * m / 3) ∧ 
                                       x2 ∈ set.Icc (m / 2) (2 * m / 3) ∧ 
                                       x3 ∈ set.Icc (m / 2) (2 * m / 3)

theorem f_n_expression (n : ℕ) (x : ℝ) (h : n > 0) : f n x = x^n := sorry

theorem can_form_triangle (x1 x2 x3 m : ℝ) :
  x_in_interval x1 x2 x3 m →
  let lengths := [g 3 x1 m, g 3 x2 m, g 3 x3 m] in
  ∃ (a b c : ℝ), a = lengths.nth 0 ∧ b = lengths.nth 1 ∧ c = lengths.nth 2 ∧ 
  a + b > c ∧ a + c > b ∧ b + c > a := sorry

end f_n_expression_can_form_triangle_l737_737438


namespace largest_multiple_of_7_negated_gt_neg_150_l737_737257

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l737_737257


namespace angle_B_of_isosceles_triangle_l737_737695

theorem angle_B_of_isosceles_triangle 
  (A B C P Q : Type) 
  (dist_AC_BC : dist A C = dist B C) 
  (point_P_on_AB : point P ∈ line_segment A B) 
  (point_Q_on_BC : point Q ∈ line_segment B C) 
  (dist_AP_PQ_QB_BC : dist A P = dist P Q ∧ dist P Q = dist Q B ∧ dist Q B = dist B C) 
  : ∠ B = 60 :=
sorry

end angle_B_of_isosceles_triangle_l737_737695


namespace range_of_m_l737_737174

noncomputable def f : ℝ → ℝ := sorry

theorem range_of_m (m : ℝ) (h1 : ∀ x : ℝ, HasDerivAt f (f' x) x)
  (h2 : ∀ x : ℝ, f (-x) + f x = x^2) (h3 : ∀ x : ℝ, 0 < x → f' x < x)
  (h4 : ∀ m : ℝ, f (2 - m) - f m > 2 - 2 * m) : 1 < m :=
sorry

end range_of_m_l737_737174


namespace fraction_arithmetic_l737_737758

theorem fraction_arithmetic : 
  (2 / 5 + 3 / 7) / (4 / 9 * 1 / 8) = 522 / 35 := by
  sorry

end fraction_arithmetic_l737_737758


namespace floor_sqrt_80_l737_737981

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737981


namespace measure_angle_ABC_l737_737434

theorem measure_angle_ABC (r : ℝ) (V : ℝ) (slant_height : ℝ) (angle : ℝ)
  (radius_given : r = 10)
  (volume_given : V = 250 * Real.pi)
  (slant_height_given : slant_height = 12.5)
  (height : ℝ)
  (height_given : height = 7.5)
  (central_angle_maj_arc : angle = 288) :
  let ABC := 360 - angle in
  ABC = 72 :=
by sorry

end measure_angle_ABC_l737_737434


namespace time_for_A_to_complete_work_alone_l737_737702

noncomputable def rateOfWork (days: Real) : Real := 1 / days

theorem time_for_A_to_complete_work_alone :
  ∃ x : Real, (
    let workRateA := rateOfWork x;
    let workRateB := rateOfWork 5;
    let totalWorkTogether := workRateA + workRateB;
    let remainingWorkB := 2.928571428571429 * workRateB;
    totalWorkTogether + remainingWorkB = 1
  )
    ∧ abs (x - 4.67) < 0.01  -- Approximation with a small tolerance
 :=
begin
    existsi 4.67, -- We'll assume the solution found by manual computation here
    unfold rateOfWork,
    have workRateA := 1 / 4.67,
    have workRateB := 1 / 5,
    have totalWorkTogether := workRateA + workRateB,
    have remainingWorkB := 2.928571428571429 * workRateB,
    have totalWork := totalWorkTogether + remainingWorkB,
    have h1 : abs (totalWork - 1) < 0.01, sorry,
    have h2 : abs (4.67 - 4.67) = 0, sorry, -- Trivial as same number subtracts to zero
    exact ⟨rfl, h1⟩
end

end time_for_A_to_complete_work_alone_l737_737702


namespace equation_of_curve_C_range_of_AB_l737_737065

noncomputable def circle_c1 : set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 9}

theorem equation_of_curve_C :
  ∀ (N A : ℝ × ℝ) (M : ℝ × ℝ),
  (∃ (x₀ y₀ : ℝ), A = (x₀, y₀) ∧ (x₀^2 + y₀^2 = 9)) →
  (M = (A.1, 0)) →
  N = ( \frac{2\sqrt{2}}{3} * A.1, \frac{2}{3} * A.2) →
  (N.1 / 2) ^ 2 + N.2 ^ 2 = 4 :=
by sorry

theorem range_of_AB :
  ∀ (A B : ℝ × ℝ),
  (A.1^2 / 8 + A.2^2 / 4 = 1) →
  (B.1^2 / 8 + B.2^2 / 4 = 1) →
  A ≠ B →
  A.1 * B.1 + A.2 * B.2 = 0 →
  ∃ (d : ℝ), 
    (d = (dist A B)) ∧
    (d ≥ \frac{4 * \sqrt{6}}{3} ∧ d ≤ 2 * \sqrt{3}) :=
by sorry

end equation_of_curve_C_range_of_AB_l737_737065


namespace tetrahedron_properties_l737_737042

theorem tetrahedron_properties (A B C D : Type) : 
  -- Assuming A, B, C, and D are vertices of a tetrahedron
  tetrahedron ABCD →
  -- Proposition 1
  (are_skew_lines (line_through A B) (line_through C D)) ∧ 
  -- Proposition 2
  (perpendicular (segment A B) (segment C D) ∧ perpendicular (segment B C) (segment A D) → 
  perpendicular (segment A C) (segment B D)) ∧ 
  -- Proposition 3
  intersect_at_one_point (midpoint_of (segment A B)) (midpoint_of (segment C D)) (midpoint_of (segment A C)) :=
sorry

end tetrahedron_properties_l737_737042


namespace floor_sqrt_80_eq_8_l737_737905

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737905


namespace floor_sqrt_80_l737_737982

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737982


namespace extreme_value_at_3_tangent_line_at_A_l737_737573

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

-- Define the first theorem related to (1)
theorem extreme_value_at_3 (a : ℝ) (h : f 3 a = 0) : f = (λ x, 2 * x^3 - 12 * x^2 + 18 * x + 8) :=
by sorry

-- Define the second theorem related to (2)
theorem tangent_line_at_A (a : ℝ) (ha : f 1 3 = 16) : 
  let fp := (6 : ℝ) * (1 : ℝ)^2 - (6 * (3 + 1)) * 1 + 6 * 3 in fp = 0 → f 1 3 = 16 :=
by sorry

end extreme_value_at_3_tangent_line_at_A_l737_737573


namespace floor_sqrt_80_l737_737816

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737816


namespace degree_of_sum_l737_737611

-- Define polynomial degrees
def deg_f : ℕ := 3
def deg_g : ℕ := 1

-- Define the polynomials f(z) and g(z) with given degrees
noncomputable def f (z : ℤ) : ℤ := a_3 * z^3 + a_2 * z^2 + a_1 * z + a_0
noncomputable def g (z : ℤ) : ℤ := b_1 * z + b_0

-- The mathematical proof problem
theorem degree_of_sum (a_3 a_2 a_1 a_0 b_1 b_0 : ℤ) (h : a_3 ≠ 0) :
  degree (f + g) = 3 :=
sorry

end degree_of_sum_l737_737611


namespace total_faculty_students_l737_737141

-- Define the conditions 
axiom N : ℕ 
axiom A : ℕ 
axiom B : ℕ 
axiom percent : ℝ

-- Assign the values given in the problem
def numeric_students := 226
def automatic_control_students := 450
def both_subject_students := 134
def percentage := 0.80

-- Define total second-year students studying at least one subject
def total_second_year_students := numeric_students + automatic_control_students - both_subject_students

-- Prove the total number of students in the faculty
theorem total_faculty_students : 
percent * total_second_year_students = 542 → 
(542 / percent).ceil.to_nat = 678 := 
by sorry

end total_faculty_students_l737_737141


namespace average_sum_problem_l737_737513

theorem average_sum_problem (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 10) : ∃ sum : ℝ, sum = avg * n ∧ sum = 53 :=
by
  sorry

end average_sum_problem_l737_737513


namespace opposite_of_neg_six_l737_737636

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_l737_737636


namespace radius_of_cyclic_quadrilateral_is_3sqrt7_l737_737395

noncomputable def radius_of_largest_circle (AB BC CD DA : ℝ) (hAB : AB = 10) (hBC : BC = 11) (hCD : CD = 13) (hDA : DA = 12)
  (cyclic : IsCyclicQuadrilateral AB BC CD DA) : ℝ :=
3 * Real.sqrt 7

theorem radius_of_cyclic_quadrilateral_is_3sqrt7 :
  ∀ (AB BC CD DA : ℝ), (AB = 10) → (BC = 11) → (CD = 13) → (DA = 12) → (IsCyclicQuadrilateral AB BC CD DA) →
    radius_of_largest_circle AB BC CD DA = 3 * Real.sqrt 7 :=
by
  intros AB BC CD DA hAB hBC hCD hDA cyclic
  unfold radius_of_largest_circle
  sorry

end radius_of_cyclic_quadrilateral_is_3sqrt7_l737_737395


namespace instantaneous_velocity_at_1_l737_737357

noncomputable def S (t : ℝ) : ℝ := t^2 + 2 * t

theorem instantaneous_velocity_at_1 : (deriv S 1) = 4 :=
by 
  -- The proof is left as an exercise
  sorry

end instantaneous_velocity_at_1_l737_737357


namespace exists_degree_at_most_2n_div_5_l737_737379

theorem exists_degree_at_most_2n_div_5
  (V : Type) [Fintype V] [DecidableEq V] (E : set (V × V))
  (n : ℕ) (hV : Fintype.card V = n)
  (h1 : ∀ (a b c : V), (a, b) ∈ E → (b, c) ∈ E → (a, c) ∈ E → false)
  (h2 : ∀ (A B : set V), A ∪ B = set.univ → A ∩ B = ∅ → 
    (∃ a b ∈ A, (a, b) ∈ E) ∨ (∃ a b ∈ B, (a, b) ∈ E)) :
  ∃ v : V, ∑ w in (univ.filter (λ w, (v, w) ∈ E)).toFinset, 1 ≤ 2 * n / 5 :=
begin
  sorry
end

end exists_degree_at_most_2n_div_5_l737_737379


namespace max_balanced_cells_l737_737414

/-- Define the concept of a board, cells, colors and balanced cells -/
def Board := Fin 100 × Fin 100
inductive Color
| blue
| white

def neighbors (cell : Board) : List Board :=
  let (x, y) := cell
  [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)].filter (λ (a, b), a.val < 100 ∧ b.val < 100)

def is_balanced (board : Board → Color) (cell : Board) : Prop :=
  let colors := neighbors cell |>.map (board)
  colors.count Color.blue = colors.count Color.white

/- The main theorem statement -/
theorem max_balanced_cells (board : Board → Color) :
  ∃ (balanced_set : Set Board), (∀ cell ∈ balanced_set, is_balanced board cell) ∧ balanced_set.size = 9608 :=
sorry

end max_balanced_cells_l737_737414


namespace dave_bought_packs_l737_737005

def packs_of_white_shirts (bought_total : ℕ) (white_per_pack : ℕ) (blue_packs : ℕ) (blue_per_pack : ℕ) : ℕ :=
  (bought_total - blue_packs * blue_per_pack) / white_per_pack

theorem dave_bought_packs : packs_of_white_shirts 26 6 2 4 = 3 :=
by
  sorry

end dave_bought_packs_l737_737005


namespace proof_problem_l737_737242

noncomputable def scores_A : list ℕ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
noncomputable def scores_B : list ℕ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

-- Define the calculation of the percentile
def percentile (p : ℝ) (l : list ℕ) : ℝ :=
  let k := p * (l.length).to_real in
  (l.nth_le (k.floor.to_nat - 1) (by linarith)) + (l.nth_le (k.ceil.to_nat - 1) (by linarith)) / 2

-- Define average calculations
def average (l : list ℕ) : ℝ :=
  (l.sum.to_real) / (l.length).to_real

-- Defining the combined scores to calculate the overall average
def combined_scores : list ℕ := scores_A ++ scores_B

-- Define variance calculations
def variance (l : list ℕ) : ℝ :=
  let μ := average l in
  (l.map (λ x, (x.to_real - μ) ^ 2)).sum / (l.length).to_real

theorem proof_problem :
  percentile 0.70 scores_A = 7.5 ∧
  average scores_A < average scores_B ∧
  average combined_scores = 6.6 ∧
  variance scores_A < variance scores_B :=
by
  sorry

end proof_problem_l737_737242


namespace simplify_expression_l737_737197

theorem simplify_expression (p : ℝ) : 
  (2 * (3 * p + 4) - 5 * p * 2)^2 + (6 - 2 / 2) * (9 * p - 12) = 16 * p^2 - 19 * p + 4 := 
by 
  sorry

end simplify_expression_l737_737197


namespace smallest_even_five_digit_tens_place_l737_737622

theorem smallest_even_five_digit_tens_place (a b c d e : ℕ) (h_distinct : list.nodup [a, b, c, d, e]) (h_digits : list.perm [a, b, c, d, e] [1, 2, 3, 4, 9]) (h_even : d = 4 ∨ d = 2) : 
  let n := 10000 * a + 1000 * b + 100 * c + 10 * e + d in
  n % 2 = 0 → a = 1 → b = 2 → c = 3 → d = 4 →
  e = 9 :=
sorry

end smallest_even_five_digit_tens_place_l737_737622


namespace expand_binomials_l737_737685

theorem expand_binomials :
  (a + 2 * b) ^ 3 = a ^ 3 + 6 * a ^ 2 * b + 12 * a * b ^ 2 + 8 * b ^ 3 ∧
  (5 * a - b) ^ 3 = 125 * a ^ 3 - 75 * a ^ 2 * b + 15 * a * b ^ 2 - b ^ 3 ∧
  (2 * a + 3 * b) ^ 3 = 8 * a ^ 3 + 36 * a ^ 2 * b + 54 * a * b ^ 2 + 27 * b ^ 3 ∧
  (m ^ 3 - n ^ 2) ^ 3 = m ^ 9 - 3 * m ^ 6 * n ^ 2 + 3 * m ^ 3 * n ^ 4 - n ^ 6 :=
by
  sorry

end expand_binomials_l737_737685


namespace smallest_value_among_options_l737_737504

theorem smallest_value_among_options (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < 3*y ∧ y^3 < y^0.5 ∧ y^3 < 1/y ∧ y^3 < Real.exp y := by
  sorry

end smallest_value_among_options_l737_737504


namespace polygon_has_8_sides_l737_737566

variable (a : ℝ) (x y : ℝ)
variables (T : Set (ℝ × ℝ))

def convexSetDefinition (a : ℝ) : Set (ℝ × ℝ) :=
  { p | let ⟨x, y⟩ := p; a ≤ x ∧ x ≤ 3 * a / 2 ∧ a ≤ y ∧ y ≤ 3 * a / 2 ∧
                    x + y ≥ a ∧ x + a ≥ y ∧ y + a ≥ x ∧ x^2 + y^2 ≤ 5 * a^2 }

theorem polygon_has_8_sides (ha : 0 < a) :
  (∂ (convexSetDefinition a)). polygonSides = 8 :=
sorry

end polygon_has_8_sides_l737_737566


namespace floor_sqrt_80_eq_8_l737_737833

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737833


namespace angle_EFG_is_60_degrees_l737_737116

theorem angle_EFG_is_60_degrees
  (A D F G C E : Point)
  (x : ℝ)
  (h_parallel : AD ∥ FG)
  (h_CFG_eq_1_5x : ∠CFG = 1.5 * x)
  (h_CEA_eq_sum : ∠CEA = x + 2 * x)
  (h_sum_angles : ∠CFG + ∠CEA = 180) :
  ∠EFG = 60 :=
by
  sorry

end angle_EFG_is_60_degrees_l737_737116


namespace speed_conversion_l737_737987

theorem speed_conversion (v_kmph : ℝ) (km_to_m : ℝ) (hr_to_s : ℝ) :
  v_kmph * (km_to_m / hr_to_s) = 18.33 :=
by
  let v_kmph := 66
  let km_to_m := 1000
  let hr_to_s := 3600
  show v_kmph * (km_to_m / hr_to_s) = 18.33
  sorry

end speed_conversion_l737_737987


namespace floor_sqrt_80_eq_8_l737_737889

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737889


namespace derivative_equals_l737_737993

noncomputable def func (x : ℝ) : ℝ :=
  (3 / (8 * Real.sqrt 2) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)))
  - (Real.tanh x / (4 * (2 - (Real.tanh x)^2)))

theorem derivative_equals :
  ∀ x : ℝ, deriv func x = 1 / (2 + (Real.cosh x)^2)^2 :=
by {
  sorry
}

end derivative_equals_l737_737993


namespace sandwich_non_condiment_percentage_l737_737733

/-
  Given that a sandwich at Deli Gourmet weighs 200 grams and 50 grams are condiments,
  prove that the percentage of the sandwich that is not condiments is 75%.
-/

theorem sandwich_non_condiment_percentage 
  (total_weight : ℕ) (condiment_weight : ℕ)
  (h1 : total_weight = 200) (h2 : condiment_weight = 50) :
  100 * (total_weight - condiment_weight) / total_weight = 75 := 
by
  rw [h1, h2]
  -- Now we would continue the proof if required, but we use sorry to skip it
  sorry

end sandwich_non_condiment_percentage_l737_737733


namespace final_mark_is_correct_l737_737752

def term_mark : ℝ := 80
def term_weight : ℝ := 0.70
def exam_mark : ℝ := 90
def exam_weight : ℝ := 0.30

theorem final_mark_is_correct :
  (term_mark * term_weight + exam_mark * exam_weight) = 83 :=
by
  sorry

end final_mark_is_correct_l737_737752


namespace petya_vitya_hat_l737_737182

-- Set up the conditions
variables (v_e v_p v_v : ℝ)
-- v_e: Speed of the escalator
-- v_p: Speed of Petya relative to the escalator
-- v_v: Speed of Vitya relative to the escalator

-- Define the main theorem
theorem petya_vitya_hat (h1 : v_p ≥ 2 * v_e) (h2 : v_v ≥ 2 * v_e) : 
  let time_to_hat_p := 2 / (v_p - v_e) + 2 / (v_p + v_e)
  let time_to_hat_v := 2 / (v_v + v_e) + 2 / (v_v - v_e)
  in time_to_hat_p = time_to_hat_v := 
sorry

end petya_vitya_hat_l737_737182


namespace magnitude_projection_eq_2_l737_737553

variables (u z : EuclideanSpace ℝ (Fin 3))
variables (h1 : inner u z = 6) (h2 : ∥z∥ = 3)

theorem magnitude_projection_eq_2 : ∥projection z u∥ = 2 :=
by sorry

end magnitude_projection_eq_2_l737_737553


namespace remainder_division_l737_737028

def polynomial (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

theorem remainder_division : polynomial 3 = 58 :=
by
  sorry

end remainder_division_l737_737028


namespace hexagon_area_l737_737378

-- Define necessary given conditions and angles
variables {AF CD AB EF BC ED : ℝ}
variables {α β : ℝ}

-- Given conditions
axiom parallel_AF_CD : AF ∥ CD
axiom parallel_AB_EF : AB ∥ EF
axiom parallel_BC_ED : BC ∥ ED

axiom length_AF : AF = 1
axiom length_AB : AB = 1
axiom length_BC : BC = 1

axiom angle_FAB : α = 60
axiom angle_BCD : β = 60

-- The final proof goal
theorem hexagon_area : AF ∥ CD → AB ∥ EF → BC ∥ ED → 
  AF = 1 → AB = 1 → BC = 1 → α = 60 → β = 60 → 
  (area_of_hexagon AF AB BC CD EF ED α β = sqrt(3)) :=
begin
  sorry
end

end hexagon_area_l737_737378


namespace distance_covered_downstream_l737_737366

theorem distance_covered_downstream :
  let speed_boat := 120 -- speed of the boat in still water in kmph
  let speed_current := 60 -- speed of the current in kmph
  let time_seconds := 9.99920006399488 -- time in seconds
  let speed_downstream := speed_boat + speed_current -- effective speed downstream in kmph
  let speed_downstream_mps := (speed_downstream * 1000) / 3600 -- convert speed from kmph to m/s
  let distance := speed_downstream_mps * time_seconds -- distance covered downstream in meters
  distance ≈ 500 := -- the approximate distance covered should be 500 meters
by
  sorry

end distance_covered_downstream_l737_737366


namespace minimal_polynomial_of_a_plus_sqrt2_l737_737064

theorem minimal_polynomial_of_a_plus_sqrt2 (a x : ℝ) 
  (h_a : a^3 - a - 1 = 0)
  (h_x : x = a + Real.sqrt 2) :
  ∃ p : Polynomial ℤ, p.leadingCoeff = 1 ∧ p.degree = 6 ∧ Polynomial.aeval (x) p = 0 ∧ 
  p = Polynomial.ofIntPoly [(-1 : ℤ), (-10 : ℤ), (13 : ℤ), (-2 : ℤ), (-8 : ℤ), 0, 1] :=
by sorry

end minimal_polynomial_of_a_plus_sqrt2_l737_737064


namespace cyclic_quadrilateral_circumference_l737_737123

-- Define the cyclic quadrilateral properties and prove the circumference of its circumscribed circle.
theorem cyclic_quadrilateral_circumference (AB BC CD DA : ℝ) (h₁ : AB = 25) (h₂ : BC = 39) (h₃ : CD = 52) (h₄ : DA = 60) 
  (inscribed : ∃ O: Point, QuadrilateralInscribedInCircle AB BC CD DA O) : 
  circumference (circumscribed_circle AB BC CD DA) = 65 * real.pi := 
begin
  sorry
end

end cyclic_quadrilateral_circumference_l737_737123


namespace floor_sqrt_80_l737_737939

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737939


namespace geometric_sequence_sum_first_five_terms_l737_737142

theorem geometric_sequence_sum_first_five_terms
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 + a 3 = 10)
  (h2 : a 2 + a 4 = 30)
  (h_geom : ∀ n, a (n + 1) = a n * q) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 121 :=
sorry

end geometric_sequence_sum_first_five_terms_l737_737142


namespace area_of_polygon_l737_737411

-- Define the coordinates of the vertices
def vertices : list (ℕ × ℕ) := [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0), (3, 3), (2, 2), (0, 0)]

-- Define the formula to calculate the area of the polygon using the shoelace formula
def polygon_area (vertices : list (ℕ × ℕ)) : ℕ :=
  let x := vertices.map Prod.fst
  let y := vertices.map Prod.snd
  abs ((x.zipWith (*) (y.tail ++ [y.head])).sum - (y.zipWith (*) (x.tail ++ [x.head])).sum) / 2

-- The Lean statement to prove
theorem area_of_polygon : polygon_area vertices = 17 / 2 := by sorry

end area_of_polygon_l737_737411


namespace convert_to_general_form_l737_737400

theorem convert_to_general_form (x : ℝ) :
  5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 :=
by
  sorry

end convert_to_general_form_l737_737400


namespace floor_sqrt_80_l737_737873

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737873


namespace no_right_triangle_solution_l737_737442

noncomputable def circle_center : Type := ℝ × ℝ
noncomputable def radius := ℝ
noncomputable def point : Type := ℝ × ℝ

variable (O : circle_center) (r : radius)
variable (P Q : point)

noncomputable def midpoint (P Q : point) : point :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def distance (A B : point) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem no_right_triangle_solution (O : circle_center) (r : radius) (P Q : point) :
  let M := midpoint P Q,
      OM := distance O M,
      PQ := distance P Q in
  OM + PQ / 2 < r →
  ¬∃ A B : point, (distance O A = r) ∧ (distance O B = r) ∧
    (distance P A + distance P B + distance A B = distance P Q) ∧ 
    (distance Q A + distance Q B + distance A B = distance P Q) ∧
    ((distance P A = distance Q B) ∧ (distance P B = distance Q A)) :=
sorry

end no_right_triangle_solution_l737_737442


namespace telescoping_product_l737_737768

theorem telescoping_product :
  (∏ k in Finset.range 501, (4 * (k + 1)) / (4 * (k + 1) + 4)) = (1 / 502) := 
by
  sorry

end telescoping_product_l737_737768


namespace area_of_triangle_is_18_l737_737247

-- Define the vertices of the triangle
def point1 : ℝ × ℝ := (1, 4)
def point2 : ℝ × ℝ := (7, 4)
def point3 : ℝ × ℝ := (1, 10)

-- Define a function to calculate the area of a triangle given three vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

-- Statement of the problem
theorem area_of_triangle_is_18 :
  triangle_area point1 point2 point3 = 18 :=
by
  -- skipping the proof
  sorry

end area_of_triangle_is_18_l737_737247


namespace vertices_of_reflected_triangle_on_incircle_l737_737550

variables {A B C H1 H2 H3 T1 T2 T3 : Point}
variables {AH1 BH2 CH3 : Line}
variables {l1 l2 l3 : Line}
variables {incircle : Circle}

-- Conditions
axiom is_altitude_A_H1 (h1: is_altitude A H1)
axiom is_altitude_B_H2 (h2: is_altitude B H2)
axiom is_altitude_C_H3 (h3: is_altitude C H3)

axiom triangle_acute (hacute: is_acute_triangle A B C)

axiom incircle_touch_points (htouch: touches_incircle incircle A B C T1 T2 T3)

-- Definitions of specific lines and reflections
def l1 := reflect_line H2 H3 (line_through T2 T3)
def l2 := reflect_line H3 H1 (line_through T3 T1)
def l3 := reflect_line H1 H2 (line_through T1 T2)

-- Question to prove
theorem vertices_of_reflected_triangle_on_incircle :
  vertices_on_incircle (triangle_formed_by l1 l2 l3) incircle :=
sorry

end vertices_of_reflected_triangle_on_incircle_l737_737550


namespace floor_sqrt_80_eq_8_l737_737901

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737901


namespace value_of_f_at_7_l737_737460

theorem value_of_f_at_7
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_periodic : ∀ x, f (x + 4) = f x)
  (h_definition : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) :
  f 7 = 2 :=
by
  -- Proof will be filled here
  sorry

end value_of_f_at_7_l737_737460


namespace floor_sqrt_80_l737_737861

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737861


namespace floor_neg_seven_fourths_l737_737794

theorem floor_neg_seven_fourths : Int.floor (-7 / 4) = -2 := 
by
  sorry

end floor_neg_seven_fourths_l737_737794


namespace sum_of_non_solutions_l737_737551

-- Let D, E, and F be constants
variables (D E F : ℝ)

-- Assume the given initial conditions
axiom cond1 : ∀ x : ℝ, (x + E) * (D * x + 36) = 3 * (x + F) * (x + 9)

-- Derive D, E, and F values as stated in the solution
axiom cond2 : D = 3
axiom cond3 : E = 9
axiom cond4 : F = 12

-- The proof problem: Sum of the values x which makes the denominator zero
theorem sum_of_non_solutions : 
  let non_solution_x := [-9, -12] in
  ∑ x in non_solution_x, x = -21 :=
by
  sorry

end sum_of_non_solutions_l737_737551


namespace new_recipe_water_cups_l737_737229

theorem new_recipe_water_cups (sugar_cups : ℕ) (h_sugar : sugar_cups = 2) :
  let original_ratio_flour_water_sugar := (11, 8, 1)
  let new_ratio_flour_water := (22, 8)
  let new_ratio_flour_sugar := (11, 2)
  let combined_ratio := (22, 8, 4)
  4 * (sugar_cups : ℕ) = 8 * 2 →
  (x : ℕ) (h_ratio : 4 * x = 16) (h_x : x = sugar_cups * 4 / 2),
  x = 4 := 
  by sorry

end new_recipe_water_cups_l737_737229


namespace smallest_b_base_45b_perfect_square_l737_737295

theorem smallest_b_base_45b_perfect_square : ∃ b : ℕ, b > 3 ∧ (∃ n : ℕ, n^2 = 4 * b + 5) ∧ ∀ b' : ℕ, b' > 3 ∧ (∃ n' : ℕ, n'^2 = 4 * b' + 5) → b ≤ b' := 
sorry

end smallest_b_base_45b_perfect_square_l737_737295


namespace tetrahedron_area_inequality_l737_737186

-- Defining the tetrahedron and areas of its faces
structure Tetrahedron :=
  (A1 A2 A3 A4 : Prop)
  (S1 S2 S3 S4 : ℝ) -- the areas of the faces

-- Axiom stating the theorem we need to prove
axiom areas_lt_sum_of_others (T : Tetrahedron) (i : Fin 4) :
  let areas := [T.S1, T.S2, T.S3, T.S4]
  in areas[i] < areas[(i + 1) % 4] + areas[(i + 2) % 4] + areas[(i + 3) % 4]

-- The theorem statement
theorem tetrahedron_area_inequality (T : Tetrahedron) :
  ∀ i : Fin 4, let areas := [T.S1, T.S2, T.S3, T.S4]
  in areas[i] < areas[(i + 1) % 4] + areas[(i + 2) % 4] + areas[(i + 3) % 4] :=
by
  intro i
  exact areas_lt_sum_of_others T i

end tetrahedron_area_inequality_l737_737186


namespace gas_pipe_usability_probability_l737_737715
open Set

def gas_pipe_probability : ℝ :=
  let total_area := (400 * 400) / 2 -- Original triangle area
  let feasible_area := (200 * 100) / 2 -- Feasible triangle area
  feasible_area / total_area

theorem gas_pipe_usability_probability :
  gas_pipe_probability = 1 / 8 := sorry

end gas_pipe_usability_probability_l737_737715


namespace number_of_divisors_of_720_l737_737176

theorem number_of_divisors_of_720 : 
  (finset.filter (λ n, 720 % n = 0) (finset.range 721)).card = 30 :=
sorry

end number_of_divisors_of_720_l737_737176


namespace find_a_monotonicity_zero_count_l737_737071
-- Importing the entirety of Mathlib to bring in necessary functionalities

open Real

-- Define the function f(x)
def f (a x : ℝ) : ℝ := log x - (1 / 2) * a * x^2

-- Statement for the first part: find the value of 'a'
theorem find_a (a : ℝ) (h_tangent_perpendicular : is_perpendicular (2, f a 2) (2x + y + 2)) : a = 0 :=
sorry

-- Statement for the second part: determine the intervals of monotonicity for the function f(x)
theorem monotonicity (a : ℝ) :
  (∀ x > 0, f' a x > 0) → (a ≤ 0 → ∀ x > 0, f' a x > 0) ∧
  (a > 0 → ∀ x ∈ Ioo 0 (sqrt (1/a)), f' a x > 0 ∧ ∀ x > sqrt (1/a), f' a x < 0) :=
sorry

-- Statement for the third part: discuss the number of zeros in the interval [1, e^2]
theorem zero_count (a : ℝ) :
  (0 ≤ a ∧ a < (4 / exp 4) ∨ a = 1 / exp 1 → mz a (1, exp 2)) →
  (a < 0 ∨ a > 1 / exp 1 → nz a (1, exp 2)) →
  ((4 / exp 4 ≤ a ∧ a < 1 / exp 1) → tz a (1, exp 2)) :=
sorry

/-- Helper Definitions --/
-- Derivative of f with respect to x
noncomputable def f' (a x : ℝ) : ℝ := 1 / x - a * x

-- Number of zeros in the interval
def mz (a : ℝ) (I : set ℝ) := ∃! x ∈ I, f a x = 0 -- Exactly one zero
def nz (a : ℝ) (I : set ℝ) := ∀ x ∈ I, f a x ≠ 0 -- No zeros
def tz (a : ℝ) (I : set ℝ) := ∃ x1 x2 ∈ I, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0 -- Exactly two zeros

-- Perpendicular lines verification
noncomputable def is_perpendicular (p : ℝ × ℝ) (l : ℝ → ℝ) := ∀ x, l x = -1 * x / (p.2 - p.1)

end find_a_monotonicity_zero_count_l737_737071


namespace smallest_b_base_45b_perfect_square_l737_737294

theorem smallest_b_base_45b_perfect_square : ∃ b : ℕ, b > 3 ∧ (∃ n : ℕ, n^2 = 4 * b + 5) ∧ ∀ b' : ℕ, b' > 3 ∧ (∃ n' : ℕ, n'^2 = 4 * b' + 5) → b ≤ b' := 
sorry

end smallest_b_base_45b_perfect_square_l737_737294


namespace James_baked_muffins_l737_737753

theorem James_baked_muffins (arthur_muffins : Nat) (multiplier : Nat) (james_muffins : Nat) : 
  arthur_muffins = 115 → 
  multiplier = 12 → 
  james_muffins = arthur_muffins * multiplier → 
  james_muffins = 1380 :=
by
  intros haf ham hmul
  rw [haf, ham] at hmul
  simp at hmul
  exact hmul

end James_baked_muffins_l737_737753


namespace find_abc_l737_737501

variable (a b c : ℝ)

def conditions : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  ab = 45 * Real.cbrt 3 ∧
  ac = 75 * Real.cbrt 3 ∧
  bc = 27 * Real.cbrt 3

theorem find_abc (h : conditions a b c) : abc = 135 * Real.sqrt 15 :=
sorry

end find_abc_l737_737501


namespace last_a_decodes_to_s_in_message_l737_737351

-- Define the conditions given in the problem
def shift_function (n : ℕ) : ℕ := n + 1 + n

-- Proof statement that last 'a' in the given message translates to 's'
theorem last_a_decodes_to_s_in_message : 
  let occurrences := 8
  let shift := (List.sum (List.range occurrences).map shift_function + 2)
  let modulo_shift := shift % 26
  let decoded_letter := char.of_nat ((nat.mod (char.to_nat 'a' + modulo_shift) 26) + nat.to_nat 'a')
  in decoded_letter = 's' :=
by
  sorry

end last_a_decodes_to_s_in_message_l737_737351


namespace combined_marbles_l737_737370

def Rhonda_marbles : ℕ := 80
def Amon_marbles : ℕ := Rhonda_marbles + 55

theorem combined_marbles : Amon_marbles + Rhonda_marbles = 215 :=
by
  sorry

end combined_marbles_l737_737370


namespace find_n_eq_l737_737785

variable (x p k d n : ℝ)

def condition1 : Prop := ∀ n, (∀ x, x ^ 2 - p * x = k * x - d)

def condition2 : Prop := ∃ r, r ≠ 0 ∧ (x ^ 2 - p * x).roots = {r, -r}

def condition3 : Prop := ∃ r, r ≠ 0 ∧ r * (-r) = 1

theorem find_n_eq : condition1 x p k d n ∧ condition2 x p ∧ condition3 x → n = 2 * (k - p) / (k + p) :=
sorry

end find_n_eq_l737_737785


namespace max_f_value_l737_737631

noncomputable def f (x : ℝ) : ℝ :=
  sin (π / 2 + 2 * x) - 5 * sin x

theorem max_f_value : 
  ∃ x : ℝ, ∀ y : ℝ, f(x) ≥ f(y) ∧ f(x) = 4 :=
sorry

end max_f_value_l737_737631


namespace apples_remaining_after_one_week_l737_737578

def x : ℕ := 4
def y : ℕ := 2
def z : ℕ := 3
def d : ℝ := 0.25

def remaining_apples : ℕ := x - y
def new_total_apples : ℕ := remaining_apples + z
def apples_that_fall : ℕ := Int.floor (d * new_total_apples)
def final_apples : ℕ := new_total_apples - apples_that_fall

theorem apples_remaining_after_one_week : final_apples = 4 :=
by
  sorry

end apples_remaining_after_one_week_l737_737578


namespace sphere_radius_eq_cbrt_three_l737_737709

-- Definitions based on the given problem
def cone_radius : ℝ := 2
def cone_height : ℝ := 3

-- Volume of the cone
def volume_cone : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height

-- Volume of the sphere
def volume_sphere (R : ℝ) : ℝ := (4/3) * Real.pi * R^3

theorem sphere_radius_eq_cbrt_three :
  ∃ R : ℝ, volume_cone = volume_sphere R ∧ R = Real.cbrt 3 :=
sorry

end sphere_radius_eq_cbrt_three_l737_737709


namespace price_difference_is_7_42_l737_737233

def total_cost : ℝ := 80.34
def shirt_price : ℝ := 36.46
def sweater_price : ℝ := total_cost - shirt_price
def price_difference : ℝ := sweater_price - shirt_price

theorem price_difference_is_7_42 : price_difference = 7.42 :=
  by
    sorry

end price_difference_is_7_42_l737_737233


namespace floor_sqrt_80_l737_737882

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737882


namespace pyramid_cross_section_area_l737_737526

noncomputable theory

-- Definition of the problem conditions and question
variables (a : ℝ)

-- The theorem statement
theorem pyramid_cross_section_area (a_pos : 0 < a) :
  ∃ (S : ℝ), S = 3 * a^2 :=
sorry

end pyramid_cross_section_area_l737_737526


namespace resultant_polynomials_l737_737432

variables {a b p q x1 x2 y1 y2 : ℝ}

def f (x : ℝ) : ℝ := x^2 + a * x + b
def g (y : ℝ) : ℝ := y^2 + p * y + q

-- Assume x1 and x2 are roots of f
def areRootsF : Prop := (f x1 = 0) ∧ (f x2 = 0)

-- Assume y1 and y2 are roots of g
def areRootsG : Prop := (g y1 = 0) ∧ (g y2 = 0)

theorem resultant_polynomials :
  areRootsF → areRootsG → (p - a) * (p * b - a * q) + (q - b)^2 = 
  (x1 - y1) * (x1 - y2) * (x2 - y1) * (x2 - y2) := 
by { sorry }

end resultant_polynomials_l737_737432


namespace floor_sqrt_80_l737_737931

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737931


namespace units_digit_2749_987_l737_737498

def mod_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  (base % 10)^(exp % 2) % 10

theorem units_digit_2749_987 : mod_units_digit 2749 987 = 9 := 
by 
  sorry

end units_digit_2749_987_l737_737498


namespace paving_rate_l737_737629

theorem paving_rate
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
  sorry

end paving_rate_l737_737629


namespace smallest_positive_period_max_min_values_l737_737074

noncomputable def f (x : ℝ) : ℝ := sin (x - π / 6) * cos x + 1

theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x := 
sorry

theorem max_min_values :
  (∀ x ∈ Set.Icc (π / 12) (π / 2), f x ≤ 5 / 4) ∧
  (f (π / 3) = 5 / 4) ∧
  (∀ x ∈ Set.Icc (π / 12) (π / 2), 3 / 4 ≤ f x) ∧
  (f (π / 12) = 3 / 4) :=
sorry

end smallest_positive_period_max_min_values_l737_737074


namespace probability_at_least_one_heart_or_king_l737_737336
   
   noncomputable def probability_non_favorable : ℚ := 81 / 169

   theorem probability_at_least_one_heart_or_king :
     1 - probability_non_favorable = 88 / 169 := 
   sorry
   
end probability_at_least_one_heart_or_king_l737_737336


namespace angle_relation_l737_737517

noncomputable theory
open_locale classical

variables (A B C D M : Type) [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D] [decidable_eq M]
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
variables (a b c d m: A)

-- Conditions
def right_angle (a b c : A) : Prop := angle b c a = π / 2

def midpoint (m a b : A) : Prop := dist a m = dist m b

def same_angle (a b c d m : A) : Prop := angle a m d = angle b m d

-- Given conditions
variables 
(h1 : right_angle a c b)
(h2 : on_line d a b)
(h3 : midpoint m c d)
(h4 : same_angle a b c d m)

-- The theorem to prove
theorem angle_relation : angle a d c = 2 * angle a c d :=
sorry

end angle_relation_l737_737517


namespace clock_strikes_in_12_hours_l737_737101

theorem clock_strikes_in_12_hours :
  ∑ k in Finset.range (12 + 1), k + 12 = 90 := 
by
  sorry

end clock_strikes_in_12_hours_l737_737101


namespace trajectory_of_Q_l737_737467

open Real

theorem trajectory_of_Q :
  ∀ (P Q M : ℝ × ℝ),
  (P.1 ^ 2 + P.2 ^ 2 = 1) →
  (M = (2, 0)) →
  (Q = ((P.1 + M.1) / 2, (P.2 + M.2) / 2)) →
  ((Q.1 - 1) ^ 2 + Q.2 ^ 2 = 1 / 4) :=
by
  intros P Q M
  assume hP hM hQ
  sorry

end trajectory_of_Q_l737_737467


namespace cards_value_1_count_l737_737362

/-- There are 4 different suits in a deck of cards containing a total of 52 cards.
  Each suit has 13 cards numbered from 1 to 13.
  Feifei draws 2 hearts, 3 spades, 4 diamonds, and 5 clubs.
  The sum of the face values of these 14 cards is exactly 35.
  Prove that 4 of these cards have a face value of 1. -/
theorem cards_value_1_count :
  ∃ (hearts spades diamonds clubs : List ℕ),
  hearts.length = 2 ∧ spades.length = 3 ∧ diamonds.length = 4 ∧ clubs.length = 5 ∧
  (∀ v, v ∈ hearts → v ∈ List.range 13) ∧ 
  (∀ v, v ∈ spades → v ∈ List.range 13) ∧
  (∀ v, v ∈ diamonds → v ∈ List.range 13) ∧
  (∀ v, v ∈ clubs → v ∈ List.range 13) ∧
  (hearts.sum + spades.sum + diamonds.sum + clubs.sum = 35) ∧
  ((hearts ++ spades ++ diamonds ++ clubs).count 1 = 4) := sorry

end cards_value_1_count_l737_737362


namespace joe_fish_times_sam_l737_737587

-- Define the number of fish Sam has
def sam_fish : ℕ := 7

-- Define the number of fish Harry has
def harry_fish : ℕ := 224

-- Define the number of times Joe has as many fish as Sam
def joe_times_sam (x : ℕ) : Prop :=
  4 * (sam_fish * x) = harry_fish

-- The theorem to prove Joe has 8 times as many fish as Sam
theorem joe_fish_times_sam : ∃ x, joe_times_sam x ∧ x = 8 :=
by
  sorry

end joe_fish_times_sam_l737_737587


namespace value_of_a_l737_737576

theorem value_of_a (X : ℝ → ℝ) (hX_norm : X ∼ Normal 1 (3 ^ 2))
  (h_prob : ∀ (P : ℝ → Prop), P(X ≤ 0) = P(X > a - 6)) :
  a = 8 :=
sorry

end value_of_a_l737_737576


namespace opposite_of_neg_six_l737_737634

theorem opposite_of_neg_six : -(-6) = 6 := 
by
  sorry

end opposite_of_neg_six_l737_737634


namespace largest_multiple_of_7_negation_gt_neg150_l737_737271

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l737_737271


namespace length_of_crease_l737_737358

theorem length_of_crease (θ : ℝ) (h : ℝ) (L : ℝ) : 
  θ ≠ 0 ∧ θ ≠ π/2 ∧ tan θ = 3 → AB = 8 ∧ DE = 2 ∧ EC = 6 → L = 2/3 :=
by
  sorry

end length_of_crease_l737_737358


namespace floor_neg_7_over_4_l737_737801

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l737_737801


namespace problem_statement_l737_737477

noncomputable def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (2 * x + φ) + 1

theorem problem_statement 
  (A : ℝ)
  (φ : ℝ)
  (hA : A > 0)
  (hf : ∀ x : ℝ, f A φ x ≤ f A φ (Real.pi / 3))
  (hφ : abs φ < Real.pi / 2) :
  (B : (f A φ (-(2 * Real.pi / 3)) = f A φ (2 * Real.pi / 3)) ∧
  D : (∀ x ∈ Icc Real.pi (5 * Real.pi / 4), ∀ y ∈ Icc Real.pi (5 * Real.pi / 4), x < y → f A φ x < f A φ y)) :=
by
  sorry

end problem_statement_l737_737477


namespace fenced_yard_area_l737_737343

theorem fenced_yard_area :
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  yard - cutout1 - cutout2 = 343 := by
  let yard := 20 * 18
  let cutout1 := 3 * 3
  let cutout2 := 4 * 2
  have h : yard - cutout1 - cutout2 = 343 := sorry
  exact h

end fenced_yard_area_l737_737343


namespace magnitude_proj_is_two_l737_737555

noncomputable def magnitude_proj (u z : ℝ^n) (h1 : dot_product u z = 6) (h2 : norm z = 3) : ℝ :=
norm ((dot_product u z / (norm z)^2) • z)

theorem magnitude_proj_is_two (u z : ℝ^n) (h1 : dot_product u z = 6) (h2 : norm z = 3) :
  magnitude_proj u z h1 h2 = 2 :=
by
  sorry

end magnitude_proj_is_two_l737_737555


namespace total_embroidery_time_l737_737389

-- Defining the constants as given in the problem
def stitches_per_minute : ℕ := 4
def stitches_per_flower : ℕ := 60
def stitches_per_unicorn : ℕ := 180
def stitches_per_godzilla : ℕ := 800
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzillas : ℕ := 1 -- Implicitly from the problem statement

-- Total time calculation as a Lean theorem
theorem total_embroidery_time : 
  (stitches_per_godzilla * num_godzillas + 
   stitches_per_unicorn * num_unicorns + 
   stitches_per_flower * num_flowers) / stitches_per_minute = 1085 := 
by
  sorry

end total_embroidery_time_l737_737389


namespace quad_inequality_necessary_but_not_sufficient_l737_737314

def quad_inequality (x : ℝ) : Prop := x^2 - x - 6 > 0
def less_than_negative_five (x : ℝ) : Prop := x < -5

theorem quad_inequality_necessary_but_not_sufficient :
  (∀ x : ℝ, less_than_negative_five x → quad_inequality x) ∧ 
  (∃ x : ℝ, quad_inequality x ∧ ¬ less_than_negative_five x) :=
by
  sorry

end quad_inequality_necessary_but_not_sufficient_l737_737314


namespace hypotenuse_of_cones_l737_737300

noncomputable def hypotenuse_length (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

theorem hypotenuse_of_cones (x y : ℝ) 
  (h₁ : (1/3) * π * y^2 * x = 1250 * π)
  (h₂ : (1/3) * π * x^2 * y = 2700 * π) :
  hypotenuse_length x y ≈ 21.3 :=
sorry

end hypotenuse_of_cones_l737_737300


namespace calculate_postage_cost_l737_737632

def base_rate : ℕ := 45 -- base rate in cents
def additional_rate : ℕ := 25 -- additional rate in cents per ounce or fraction
def weight : ℝ := 3.7 -- weight of the package in ounces

def total_postage_cost : ℝ :=
  let additional_weight := weight - 1
  let additional_units := additional_weight.ceil
  let additional_cost := additional_units * additional_rate
  let total_cost_in_cents := base_rate + additional_cost
  total_cost_in_cents / 100 -- convert cents to dollars

theorem calculate_postage_cost :
  total_postage_cost = 1.20 :=
by
  sorry

end calculate_postage_cost_l737_737632


namespace rectangle_area_l737_737701

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w^2 + (2 * w)^2 = x^2) : 
  2 * (w^2) = (2 / 5) * x^2 :=
by
  sorry

end rectangle_area_l737_737701


namespace milk_per_school_day_l737_737416

/--
Emma buys some containers of milk every school day for lunch. 
She does not go to school on the weekends. 
In 3 weeks, she buys 30 containers of milk. 
Prove that she buys 2 containers of milk each school day.
-/
theorem milk_per_school_day (weeks : ℕ) (total_containers : ℕ) (school_days : ℕ) :
  weeks = 3 → total_containers = 30 → school_days = 5 →
  (total_containers / (weeks * school_days) = 2) :=
by
  intros h_weeks h_containers h_days
  rw [h_weeks, h_containers, h_days]
  norm_num

end milk_per_school_day_l737_737416


namespace acceleration_vector_line_touches_unit_circle_f_decreases_in_interval_area_S_l737_737619

noncomputable def P (t : ℝ) : ℝ × ℝ :=
  (cos (2 * t) + t * sin (2 * t), sin (2 * t) - t * cos (2 * t))

def alpha (t : ℝ) : ℝ × ℝ :=
  (-4 * t * sin (2 * t), 4 * t * cos (2 * t))

theorem acceleration_vector (t : ℝ) :
  ∀ t : ℝ, alpha t = (-4 * t * sin (2 * t), 4 * t * cos (2 * t)) :=
sorry

theorem line_touches_unit_circle (t : ℝ) :
  ∀ t : ℝ, ∃ Q : ℝ × ℝ, (Q.1 = cos (2 * t)) ∧ (Q.2 = sin (2 * t)) :=
sorry

theorem f_decreases_in_interval :
  ∀ t : ℝ, 0 ≤ t → t ≤ π / 2 → (deriv (λ t => cos (2 * t) + t * sin (2 * t)) t < 0) :=
sorry

theorem area_S :
  ∫ t in (π / 4) .. (π / 2), (P t).2 * real.abs (deriv (λ t => (P t).1) t) = (7 * π^3 / 192) :=
sorry

end acceleration_vector_line_touches_unit_circle_f_decreases_in_interval_area_S_l737_737619


namespace floor_neg_7_over_4_l737_737803

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l737_737803


namespace calculator_display_exceeds_1000_after_three_presses_l737_737322

-- Define the operation of pressing the squaring key
def square_key (n : ℕ) : ℕ := n * n

-- Define the initial display number
def initial_display : ℕ := 3

-- Prove that after pressing the squaring key 3 times, the display is greater than 1000.
theorem calculator_display_exceeds_1000_after_three_presses : 
  square_key (square_key (square_key initial_display)) > 1000 :=
by
  sorry

end calculator_display_exceeds_1000_after_three_presses_l737_737322


namespace remainder_division_l737_737029

def polynomial (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7 * x - 8

theorem remainder_division : polynomial 3 = 58 :=
by
  sorry

end remainder_division_l737_737029


namespace floor_sqrt_80_eq_8_l737_737907

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737907


namespace compute_nested_f_l737_737561

def f(x : ℤ) : ℤ := x^2 - 4 * x + 3

theorem compute_nested_f : f (f (f (f (f (f 2))))) = f 1179395 := 
  sorry

end compute_nested_f_l737_737561


namespace product_is_zero_l737_737012

theorem product_is_zero (b : ℕ) (h : b = 5) : (b-12) * (b-11) * (b-10) * (b-9) * (b-8) * (b-7) * (b-6) * (b-5) * (b-4) * (b-3) * (b-2) * (b-1) * b = 0 := 
by
  rw [h]
  sorry

end product_is_zero_l737_737012


namespace probability_of_inequality_l737_737397

def f (x : ℝ) : ℝ := x^2 - x - 2

def interval : set ℝ := {x | -5 ≤ x ∧ x ≤ 5}

def satisfies_inequality (x : ℝ) : Prop := f(x) ≤ 0

theorem probability_of_inequality : 
  let interval_satisfaction := {x | x ∈ interval ∧ satisfies_inequality x}
  (measure_of interval_satisfaction) / (measure_of interval) = 3 / 10 :=
sorry

end probability_of_inequality_l737_737397


namespace round_down_7453_497_l737_737598

theorem round_down_7453_497 : Int.round 7453.497 = 7453 :=
by
  sorry

end round_down_7453_497_l737_737598


namespace Alyssa_total_spent_l737_737369

-- define the amounts spent on grapes and cherries
def costGrapes: ℝ := 12.08
def costCherries: ℝ := 9.85

-- define the total cost based on the given conditions
def totalCost: ℝ := costGrapes + costCherries

-- prove that the total cost equals 21.93
theorem Alyssa_total_spent:
  totalCost = 21.93 := 
  by
  -- proof to be completed
  sorry

end Alyssa_total_spent_l737_737369


namespace determine_valid_n_l737_737789

def satisfies_property (n : ℕ) : Prop :=
  ∀ m : ℤ, ∃ π : Π (i : fin n), fin n, ∀ k : fin n, 
  (nat.cast (π (π k)) : ℤ) % n = (m * nat.cast k : ℤ) % n

theorem determine_valid_n (n : ℕ) : n = 1 ∨ nat.factorization n 2 = 1 ↔ satisfies_property n :=
sorry

end determine_valid_n_l737_737789


namespace angle_YOZ_29_l737_737148

-- Let O be the incenter of triangle XYZ
variables {X Y Z O : Type} {angle_XYZ angle_YXZ : ℝ}

-- Assign given angle measures
def angle_XYZ_deg := 65
def angle_YXZ_deg := 57
def angle_XZY := 180 - angle_XYZ_deg - angle_YXZ_deg

-- Define the problem in Lean 4
theorem angle_YOZ_29 (h1 : angle_XYZ = 65) 
                     (h2 : angle_YXZ = 57) 
                     (h3 : O_incenter : incircle O X Y Z) :
  angle _YOZ = 29 :=
sorry

end angle_YOZ_29_l737_737148


namespace cliff_shiny_igneous_l737_737132

variables (I S : ℕ)

theorem cliff_shiny_igneous :
  I = S / 2 ∧ I + S = 270 → I / 3 = 30 := 
by
  intro h
  sorry

end cliff_shiny_igneous_l737_737132


namespace truthful_responses_l737_737520

-- Conditions and problem definition
def number_of_students (n : ℕ) (truth_responses liar_responses : ℕ) : Prop :=
  n * (n - 1) = truth_responses + liar_responses

def liar_responses_eq (t n liar_responses : ℕ) : Prop :=
  2 * t * (n - t) = liar_responses

def truth_responses_eq (t n truth_responses : ℕ) : Prop :=
  truth_responses = t * (n - 1)

-- Theorem statement
theorem truthful_responses (n t truth_responses liar_responses : ℕ) :
  number_of_students n truth_responses liar_responses ∧ liar_responses_eq t n liar_responses →
  (truth_responses = 16 ∨ truth_responses = 56) :=
by
  intros h
  cases h with h1 h2
  sorry

end truthful_responses_l737_737520


namespace construct_triangle_l737_737777

/-- Given a side c, an angle β opposite side b, and the difference d = a - b, 
    construct a triangle such that AB = c, ∠ABC = β, and d = a - b. -/
theorem construct_triangle 
  (c : ℝ) (β : ℝ) (d : ℝ) 
  (hc : 0 < c)
  (hβ : 0 < β ∧ β < π)
  (hd : 0 < d) :
  ∃ (a b : ℝ) (A B C : Type) (triangle_ABC : A ∧ B ∧ C),
  distance A B = c ∧
  angle A B C = β ∧
  (a - b = d) :=
sorry

end construct_triangle_l737_737777


namespace floor_sqrt_80_l737_737876

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737876


namespace angle_measure_of_ENG_l737_737190

theorem angle_measure_of_ENG 
  (EFGH : Type) [rectangle EFGH]
  (E F G H N : EFGH)
  (EF : dist(E, F) = 10)
  (FG : dist(F, G) = 4)
  (EN : dist(E, N) = 4)
  (angle_eq : ∠ ENG = ∠ FNG) 
  (triangle_ENG : right_triangle E N G)
  (triangle_FNG : right_triangle F N G) :
  ∠ ENG = 45 :=
sorry

end angle_measure_of_ENG_l737_737190


namespace sqrt_floor_eight_l737_737848

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737848


namespace mouse_seed_hiding_l737_737136

theorem mouse_seed_hiding : 
  ∀ (h_m h_r x : ℕ), 
  4 * h_m = x →
  7 * h_r = x →
  h_m = h_r + 3 →
  x = 28 :=
by
  intros h_m h_r x H1 H2 H3
  sorry

end mouse_seed_hiding_l737_737136


namespace shem_earnings_l737_737196

theorem shem_earnings (kem_hourly: ℝ) (ratio: ℝ) (workday_hours: ℝ) (shem_hourly: ℝ) (shem_daily: ℝ) :
  kem_hourly = 4 →
  ratio = 2.5 →
  shem_hourly = kem_hourly * ratio →
  workday_hours = 8 →
  shem_daily = shem_hourly * workday_hours →
  shem_daily = 80 :=
by
  -- Proof omitted
  sorry

end shem_earnings_l737_737196


namespace value_of_k_single_solution_l737_737125

-- Define the problem in Lean 4
theorem value_of_k_single_solution 
  (k : ℝ) 
  (A : set ℝ := { x | k * x^2 + 4 * x + 4 = 0 }) : 
  (∃ a ∈ A, ∀ b ∈ A, a = b) → (k = 0 ∨ k = 1) :=
by 
  sorry

end value_of_k_single_solution_l737_737125


namespace janet_extra_flowers_l737_737153

-- Define the number of flowers Janet picked for each type
def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4

-- Define the number of flowers Janet used
def used : ℕ := 19

-- Calculate the total number of flowers Janet picked
def total_picked : ℕ := tulips + roses + daisies + lilies

-- Calculate the number of extra flowers
def extra_flowers : ℕ := total_picked - used

-- The theorem to be proven
theorem janet_extra_flowers : extra_flowers = 8 :=
by
  -- You would provide the proof here, but it's not required as per instructions
  sorry

end janet_extra_flowers_l737_737153


namespace floor_sqrt_80_eq_8_l737_737841

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737841


namespace largest_multiple_of_7_negated_gt_neg_150_l737_737254

theorem largest_multiple_of_7_negated_gt_neg_150 :
  ∃ (n : ℕ), (negate (n * 7) > -150) ∧ (∀ m : ℕ, (negate (m * 7) > -150) → m ≤ n) ∧ (n * 7 = 147) :=
sorry

end largest_multiple_of_7_negated_gt_neg_150_l737_737254


namespace floor_sqrt_80_eq_8_l737_737915

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737915


namespace initial_weight_of_solution_y_is_six_l737_737200

variables (W : ℝ) (liquid_x : ℝ) (water : ℝ)

-- The initial amounts of liquid_x and water in solution y
def initial_liquid_x := 0.3 * W
def initial_water := 0.7 * W

-- After evaporation of 2 kg of water
def evaporated_water := initial_water - 2
def remaining_solution := W - 2

-- Adding 2 kg of solution y, with 30% liquid x and 70% water
def added_liquid_x := 0.6
def added_water := 1.4

-- Total amounts of liquid_x and water in the new solution
def new_liquid_x := initial_liquid_x + added_liquid_x
def new_water := evaporated_water + added_water

-- Total weight of the new solution
def new_total_weight := W -- Since (W - 2) + 2 = W

-- Condition for the new solution being 40% liquid x
def new_liquid_x_condition := 0.4 * new_total_weight = new_liquid_x

theorem initial_weight_of_solution_y_is_six (h : new_liquid_x_condition) : W = 6 :=
by {
  sorry
}

end initial_weight_of_solution_y_is_six_l737_737200


namespace distribute_chapters_l737_737791

theorem distribute_chapters :
  let num_authors := 8
  let num_chapters := 16
  let distribution := [3, 3, 2, 2, 2, 2, 1, 1]
  (num_authors = 8 ∧ num_chapters = 16 ∧
  list_sum distribution = num_chapters ∧
  list_occurrences distribution 3 = 2 ∧
  list_occurrences distribution 2 = 4 ∧
  list_occurrences distribution 1 = 2)
  → (num_ways_distribution num_chapters distribution = (16.factorial / (2^6 * 3^2)))
 :=
begin
  sorry
end

end distribute_chapters_l737_737791


namespace remainder_when_13_add_x_div_31_eq_22_l737_737164

open BigOperators

theorem remainder_when_13_add_x_div_31_eq_22
  (x : ℕ) (hx : x > 0) (hmod : 7 * x ≡ 1 [MOD 31]) :
  (13 + x) % 31 = 22 := 
  sorry

end remainder_when_13_add_x_div_31_eq_22_l737_737164


namespace probability_of_intersecting_diagonals_l737_737394

-- Define a regular dodecagon
def regular_dodecagon := Finset (Fin 12)

-- Define number of diagonals
def num_diagonals := (regular_dodecagon.card.choose 2) - 12

-- Define number of pairs of diagonals
def num_pairs_diagonals := (num_diagonals.choose 2)

-- Define number of intersecting diagonals
def num_intersecting_diagonals := (regular_dodecagon.card.choose 4)

-- Define the probability that two randomly chosen diagonals intersect inside the dodecagon
def intersection_probability := (num_intersecting_diagonals : ℝ) / (num_pairs_diagonals : ℝ)

theorem probability_of_intersecting_diagonals :
  intersection_probability = (495 / 1431 : ℝ) :=
sorry

end probability_of_intersecting_diagonals_l737_737394


namespace postage_problem_l737_737996

noncomputable def sum_all_positive_integers (n1 n2 : ℕ) : ℕ :=
  n1 + n2

theorem postage_problem : sum_all_positive_integers 21 22 = 43 :=
by
  have h1 : ∀ x y z : ℕ, 7 * x + 21 * y + 23 * z ≠ 120 := sorry
  have h2 : ∀ x y z : ℕ, 7 * x + 22 * y + 24 * z ≠ 120 := sorry
  exact rfl

end postage_problem_l737_737996


namespace lim_Sn_div_Tn_l737_737565

def M (n : ℕ) : Set ℝ :=
  { x : ℝ | ∃ (a : Fin n → Fin 2), x = ∑ i in Finset.range (n - 1), (a i : ℝ) / 10^(i+1) + 1 / 10^n }

def T (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def S (n : ℕ) : ℝ :=
  ∑ x in M n, x

theorem lim_Sn_div_Tn : 
  ∀ (Sn : ℕ → ℝ) (Tn : ℕ → ℕ), 
  (∀ n, Sn n = S n) → (∀ n, Tn n = T n) → 
  (Sn = S ∧ Tn = T) →
  ∃ l, (filter.at_top (λ n, Sn n / Tn n) ⟶ l) ∧ l = 1/18 :=
by
  sorry

end lim_Sn_div_Tn_l737_737565


namespace longest_side_of_triangle_l737_737747

-- Definition of the vertices of the triangle
def vertex1 := (3 : ℝ, 3 : ℝ)
def vertex2 := (5 : ℝ, 7 : ℝ)
def vertex3 := (7 : ℝ, 3 : ℝ)

-- Function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- The distances between the vertices
def d1 := distance vertex1 vertex2
def d2 := distance vertex2 vertex3
def d3 := distance vertex3 vertex1

-- The length of the longest side calculation proof
theorem longest_side_of_triangle :
  max d1 (max d2 d3) = 2 * real.sqrt 5 :=
by sorry

end longest_side_of_triangle_l737_737747


namespace sum_third_largest_and_smallest_l737_737309

-- Condition 1: Digits available are 7, 4, 0, 3, and 5
def digits : List ℕ := [7, 4, 0, 3, 5]

-- Condition 2 and 3: Form two-digit numbers with distinct digits, and 0 cannot be the first digit.
def valid_two_digit (a b : ℕ) : Bool :=
  a ≠ b ∧ a ≠ 0 

-- A function to generate all valid two-digit numbers from the given digits
def two_digit_numbers : List ℕ :=
  (digits.product digits).filter (λ pair => valid_two_digit pair.fst pair.snd = true).map (λ pair => 10 * pair.fst + pair.snd)

-- All valid two-digit numbers (as per the constraints)
def sorted_desc := two_digit_numbers.qsort (λ x y => x > y)
def sorted_asc := two_digit_numbers.qsort (λ x y => x < y)

-- Define the third largest and third smallest values
def third_largest := sorted_desc.nth 2
def third_smallest := sorted_asc.nth 2

-- The proof statement
theorem sum_third_largest_and_smallest : third_largest + third_smallest = 108 :=
by
  -- Placeholder for proof
  sorry

end sum_third_largest_and_smallest_l737_737309


namespace translated_axis_of_symmetry_left_pi_over_6_l737_737515

theorem translated_axis_of_symmetry_left_pi_over_6 
    (k : ℤ) :
    let f := λ x, sin (2 * x) + sqrt 3 * cos (2 * x)
    let g := λ x, sin (2 * (x + π/6)) + sqrt 3 * cos (2 * (x + π/6))
    ∃ x₀, g x₀ = f (x₀ + π/6) ∧ (∃ t : ℝ, 2 * t + 2 * π / 3 = k * π + π / 2) 
    → (x₀ = k * π / 2 - π / 12)
:= sorry

end translated_axis_of_symmetry_left_pi_over_6_l737_737515


namespace ratio_of_areas_l737_737604

theorem ratio_of_areas (s : ℝ) (h_s_pos : 0 < s) :
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  total_small_triangles_area / large_triangle_area = 1 / 6 :=
by
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  sorry
 
end ratio_of_areas_l737_737604


namespace exist_tangents_with_slope_3_l737_737481

noncomputable def f (x a : ℝ) := exp (2 * x) - 2 * exp x + a * x - 1
noncomputable def f' (x a : ℝ) := 2 * exp (2 * x) - 2 * exp x + a

theorem exist_tangents_with_slope_3 (a : ℝ) :
  (3 < a ∧ a < 7 / 2) ↔
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' x₁ a = 3 ∧ f' x₂ a = 3 :=
begin
  sorry
end

end exist_tangents_with_slope_3_l737_737481


namespace rational_points_coloring_l737_737589

def is_rational_point (p : ℚ × ℚ) : Prop :=
∃ (a b c d : ℤ), b > 0 ∧ d > 0 ∧ Int.gcd a b = 1 ∧ Int.gcd c d = 1 ∧ p = (a / b, c / d)

theorem rational_points_coloring (n : ℕ) (hn : 0 < n) : 
  ∃ (coloring : ℚ × ℚ → ℕ), 
  (∀ (p : ℚ × ℚ), is_rational_point p → coloring p < n) ∧
  (∀ (p1 p2 : ℚ × ℚ), is_rational_point p1 → is_rational_point p2 → p1 ≠ p2 → 
    ∃ q : ℚ × ℚ, is_rational_point q ∧ q ∈ segment ℚ p1 p2 ∧ 
    (∀ k : ℕ, k < n → q ≠ p1 ∧ q ≠ p2 → coloring q = k)) :=
sorry

end rational_points_coloring_l737_737589


namespace largest_multiple_of_7_neg_greater_than_neg_150_l737_737287

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l737_737287


namespace triangle_angle_contradiction_l737_737675

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l737_737675


namespace divisible_by_13_l737_737558

theorem divisible_by_13 (a : ℤ) (h₀ : 0 ≤ a) (h₁ : a ≤ 13) : (51^2015 + a) % 13 = 0 → a = 1 :=
by
  sorry

end divisible_by_13_l737_737558


namespace biology_score_range_l737_737740

-- Define the constraints
variable {M P C B : ℝ}

-- Conditions
def sum_M_P := M + P = 110
def diff_C_P := C - P = 20
def weighted_average_88 := (M + P + C + B) / 4 ≥ 88
def passing_marks : 40 ≤ M ∧ M ≤ 70 ∧ 40 ≤ P
def eligibility_for_scholarship := ∃ B : ℝ, 152 ≤ B ∧ B ≤ 182

-- The Lean 4 statement
theorem biology_score_range
  (h1 : sum_M_P)
  (h2 : diff_C_P)
  (h3 : weighted_average_88)
  (h4 : passing_marks) :
  eligibility_for_scholarship :=
begin
  -- We assert there exists a B such that it satisfies the eligibility condition.
  sorry -- Proof omitted
end

end biology_score_range_l737_737740


namespace gift_card_value_l737_737393

def latte_cost : ℝ := 3.75
def croissant_cost : ℝ := 3.50
def daily_treat_cost : ℝ := latte_cost + croissant_cost
def weekly_treat_cost : ℝ := daily_treat_cost * 7

def cookie_cost : ℝ := 1.25
def total_cookie_cost : ℝ := cookie_cost * 5

def total_spent : ℝ := weekly_treat_cost + total_cookie_cost
def remaining_balance : ℝ := 43.00

theorem gift_card_value : (total_spent + remaining_balance) = 100 := 
by sorry

end gift_card_value_l737_737393


namespace sqrt_floor_eight_l737_737856

theorem sqrt_floor_eight : (⌊real.sqrt 80⌋ = 8) :=
begin
  -- conditions
  have h1 : 8^2 = 64 := by norm_num,
  have h2 : 9^2 = 81 := by norm_num,
  have h3 : 8 < real.sqrt 80 := by { apply real.sqrt_lt, norm_num, },
  have h4 : real.sqrt 80 < 9 := by { apply real.sqrt_lt, norm_num, },

  -- combine conditions to prove the statement
  rw real.floor_eq_iff,
  split,
  { exact h3, },
  { exact h4, }
end

end sqrt_floor_eight_l737_737856


namespace probability_of_A_and_B_l737_737226

-- Definitions and conditions
variable (A B : Type)
variable (P : A → ℝ)
variable (h1 : P A = 5 / 6)
variable (h2 : P B = 1 / 2)

-- The statement to prove
theorem probability_of_A_and_B (A B : Type) (P : A → ℝ) 
  (h1 : P A = 5 / 6) (h2 : P B = 1 / 2) : 
  P (A ∩ B) = (5 / 6) * (1 / 2) :=
by 
  sorry

end probability_of_A_and_B_l737_737226


namespace floor_sqrt_80_l737_737948

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737948


namespace find_principal_on_SI_l737_737232

-- Definitions of conditions
def CI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * ((1 + r / 100) ^ t) - P

def SI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  (P * r * t) / 100

-- Define constants given in the problem
def P_CI : ℝ := 6000
def r_CI : ℝ := 15
def t_CI : ℕ := 2
def r_SI : ℝ := 12
def t_SI : ℕ := 4

-- Calculation for compound interest and principal on simple interest.
theorem find_principal_on_SI :
  let C.I := CI P_CI r_CI t_CI
  let S.I := C.I / 2
  let P := 2015.625
  SI P r_SI t_SI = S.I :=
by
  let CI_val := CI P_CI r_CI t_CI
  let SI_val := CI_val / 2
  have : SI 2015.625 r_SI t_SI = SI_val := sorry
  exact this

end find_principal_on_SI_l737_737232


namespace multiplicative_inverse_exists_and_is_correct_l737_737569

theorem multiplicative_inverse_exists_and_is_correct :
  ∃ N : ℤ, N > 0 ∧ (123456 * 171717) * N % 1000003 = 1 :=
sorry

end multiplicative_inverse_exists_and_is_correct_l737_737569


namespace f_f_one_fourth_l737_737478

noncomputable def f : ℝ → ℝ := 
  λ x, if x > 0 then real.log x / real.log 2 else 3 ^ x

theorem f_f_one_fourth : f (f (1 / 4)) = 1 / 9 :=
  by
    sorry

end f_f_one_fourth_l737_737478


namespace box_marble_problem_l737_737737

theorem box_marble_problem :
  ∃ n, ∀ k > n, ∃ a b c : ℕ, k = 13 * a + 11 * b + 7 * c :=
begin
  use 30,
  intros k hk,
  sorry
end

end box_marble_problem_l737_737737


namespace determine_m_minus_n_l737_737784

-- Definitions of the conditions
variables {m n : ℝ}

-- The proof statement
theorem determine_m_minus_n (h_eq : ∀ x y : ℝ, x^(4 - 3 * |m|) + y^(3 * |n|) = 2009 → x + y = 2009)
  (h_prod_lt_zero : m * n < 0)
  (h_sum : 0 < m + n ∧ m + n ≤ 3) : m - n = 4/3 := 
sorry

end determine_m_minus_n_l737_737784


namespace floor_neg_seven_fourths_l737_737799

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l737_737799


namespace quadratic_transformation_l737_737228

theorem quadratic_transformation (c b : ℤ)
  (h₁ : (∃ b c : ℤ, ∀ x : ℝ, x^2 + 2200 * x + 4200 = (x + b)^2 + c))
  (h₂ : b = 1100)
  (h₃ : c = -1205800) :
  c / b = -1096 :=
  by
  rcases h₁ with ⟨b, c, h⟩,
  rw [h₂, h₃],
  norm_num,
  sorry

end quadratic_transformation_l737_737228


namespace order_of_values_l737_737502

noncomputable def a : ℝ := Real.sin 5
noncomputable def b : ℝ := Real.logBase 3 2
noncomputable def c : ℝ := Real.log 2
noncomputable def d : ℝ := Real.exp 0.001

-- Note that these Lean definitions directly appear in the conditions from step a).

theorem order_of_values : d > c ∧ c > b ∧ b > a := by
  sorry

end order_of_values_l737_737502


namespace intersection_P_Q_l737_737172

def P : Set ℕ := {0, 1, 2, 3}
def Q : Set ℝ := {x : ℝ | |x| < 2}

theorem intersection_P_Q : (P ∩ (Q ∩ Set.univ : Set ℕ)) = {0, 1} :=
by
  -- Lean understands that P is a subset of natural numbers,
  -- hence we intersect Q with the universal set of naturals
  sorry

end intersection_P_Q_l737_737172


namespace num_real_a_has_integer_roots_l737_737040

theorem num_real_a_has_integer_roots : 
  (∃ a : ℝ, ∀ x ∈ ℝ, (x^2 + a * x + 12 * a = 0) → (∃ p q : ℤ, x = p ∧ x = q ∧ (p + q = -a) ∧ (p * q = 12 * a))) → 8 := sorry

end num_real_a_has_integer_roots_l737_737040


namespace find_f_prime_one_l737_737469

noncomputable def f (x : ℝ) : ℝ := 2 * x * f' 1 + Real.log x
noncomputable def f' (x : ℝ) : ℝ := deriv f x

theorem find_f_prime_one : f' 1 = -1 := by
  -- Given conditions translated into Lean.
  sorry

end find_f_prime_one_l737_737469


namespace derivative_y_x_l737_737025

-- Define the given functions
def x (t : ℝ) : ℝ := Real.log (Real.tan t)
def y (t : ℝ) : ℝ := 1 / (Real.sin t) ^ 2

-- Define the derivative of x with respect to t
def dx_dt (t : ℝ) : ℝ := 1 / (Real.sin t * Real.cos t)

-- Define the derivative of y with respect to t
def dy_dt (t : ℝ) : ℝ := -2 * (Real.cos t) / (Real.sin t) ^ 3

-- State the theorem to be proved
theorem derivative_y_x (t : ℝ) : (dy_dt t) / (dx_dt t) = -2 * (Real.cot t) ^ 2 := by
  sorry

end derivative_y_x_l737_737025


namespace no_integer_root_quadratic_trinomials_l737_737009

theorem no_integer_root_quadratic_trinomials :
  ¬ ∃ (a b c : ℤ),
    (∃ r1 r2 : ℤ, a * r1^2 + b * r1 + c = 0 ∧ a * r2^2 + b * r2 + c = 0 ∧ r1 ≠ r2) ∧
    (∃ s1 s2 : ℤ, (a + 1) * s1^2 + (b + 1) * s1 + (c + 1) = 0 ∧ (a + 1) * s2^2 + (b + 1) * s2 + (c + 1) = 0 ∧ s1 ≠ s2) :=
by
  sorry

end no_integer_root_quadratic_trinomials_l737_737009


namespace bicycle_speed_l737_737344

theorem bicycle_speed
  (dist : ℝ := 15) -- Distance between the school and the museum
  (bus_factor : ℝ := 1.5) -- Bus speed is 1.5 times the bicycle speed
  (time_diff : ℝ := 1 / 4) -- Bicycle students leave 1/4 hour earlier
  (x : ℝ) -- Speed of bicycles
  (h : (dist / x) - (dist / (bus_factor * x)) = time_diff) :
  x = 20 :=
sorry

end bicycle_speed_l737_737344


namespace roots_of_polynomial_l737_737989

noncomputable def polynomial : Polynomial ℚ := Polynomial.Coeff [(-36), 6, 11, -6, 1]

theorem roots_of_polynomial :
  (Polynomial.roots polynomial).to_list = [-2, -2, 3, 3] :=
sorry

end roots_of_polynomial_l737_737989


namespace sequence_property_l737_737173

variable (a : ℕ → ℕ)

theorem sequence_property
  (h_bij : Function.Bijective a) (n : ℕ) :
  ∃ k, k < n ∧ a (n - k) < a n ∧ a n < a (n + k) :=
sorry

end sequence_property_l737_737173


namespace floor_sqrt_80_l737_737815

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737815


namespace factor_expression_l737_737418

theorem factor_expression (x : ℝ) : 5 * x^2 * (x - 2) - 9 * (x - 2) = (x - 2) * (5 * x^2 - 9) :=
sorry

end factor_expression_l737_737418


namespace two_pairs_same_distance_l737_737199

-- Lean 4 statement
theorem two_pairs_same_distance (points : Fin 6 → (ℝ × ℝ)) (Hbounded : ∀ i, points i.1 < 10 ∧ points i.2 < 10) 
  (Hint_dist: ∀ i j, i ≠ j → ∃ (k : ℕ), k ∈ Finset.range 15 ∧ (dist (points i) (points j) = k)) : 
  ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ dist (points i) (points j) = dist (points k) (points l) := 
sorry

end two_pairs_same_distance_l737_737199


namespace length_of_BD_l737_737556

-- Given definitions and conditions
variables {A B C D : Type} [MetricSpace A B C D]

noncomputable def right_triangle (A B C : Point) (angleB : is_right_angle ∠ABC) : Prop := sorry
noncomputable def circle_diameter (A C : Point) (circle : Circle) : Prop := sorry
noncomputable def intersects (line BC : Line) (circle : Circle) (point D : Point) : Prop := sorry

-- AB and AC lengths
noncomputable def AB_length : ℝ := 18
noncomputable def AC_length : ℝ := 30

-- Proof goal: BD length
noncomputable def BD_length (BD : Line) : ℝ := 14.4

theorem length_of_BD
  (A B C D : Point)
  (ABC_right : right_triangle A B C ∠ABC)
  (circle_with_diameter_AC : circle_diameter A C circle)
  (D_on_BC_intersect_circle : intersects (line B C) circle D)
  (AB_eq_18 : dist A B = AB_length)
  (AC_eq_30 : dist A C = AC_length) : 
  dist B D = BD_length :=
begin
  sorry
end

end length_of_BD_l737_737556


namespace floor_sqrt_80_eq_8_l737_737894

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737894


namespace telescoping_product_l737_737767

theorem telescoping_product :
  (∏ k in Finset.range 501, (4 * (k + 1)) / (4 * (k + 1) + 4)) = (1 / 502) := 
by
  sorry

end telescoping_product_l737_737767


namespace floor_sqrt_80_l737_737964

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737964


namespace floor_sqrt_80_l737_737871

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737871


namespace square_perimeter_not_necessarily_integer_l737_737361

theorem square_perimeter_not_necessarily_integer
  (a : ℚ) (x : ℚ) (hx : 0 < x ∧ x < a)
  (P1_whole : 2 * (a + x) ∈ ℤ)
  (P2_whole : 2 * (2 * a - x) ∈ ℤ) :
  ¬ (4 * a ∈ ℤ) :=
by
  sorry

end square_perimeter_not_necessarily_integer_l737_737361


namespace population_weight_of_500_students_l737_737654

-- Definitions
def number_of_students : ℕ := 500
def number_of_selected_students : ℕ := 60

-- Conditions
def condition1 := number_of_students = 500
def condition2 := number_of_selected_students = 60

-- Statement
theorem population_weight_of_500_students : 
  condition1 → condition2 → 
  (∃ p, p = "the weight of the 500 students") := by
  intros _ _
  existsi "the weight of the 500 students"
  rfl

end population_weight_of_500_students_l737_737654


namespace correct_statements_l737_737316

variables {l m : Type} {alpha beta gamma : Type}

-- Define line, plane, parallelism
-- Assume necessary properties of lines, planes, and their relationships for simplicity

-- Conditions for Statement A
def Line.l_is_parallel_to_m := sorry
def Line.m_is_in_alpha := sorry
def Line.l_is_not_in_alpha := sorry
def Line.l_is_parallel_to_alpha := sorry

-- Conditions for Statement C
def Plane.alpha_is_parallel_to_beta := sorry
def Plane.beta_is_parallel_to_gamma := sorry
def Plane.alpha_is_parallel_to_gamma := sorry

-- Conditions for Statement D
def Line.l_is_skew_with_m := sorry
def Line.l_is_parallel_to_alpha_and_beta := sorry
def Line.m_is_parallel_to_alpha_and_beta := sorry
def Plane.alpha_is_parallel_to_beta_via_skew_lines := sorry

theorem correct_statements (A : Type) (C : Type) (D : Type) :
  (Line.l_is_parallel_to_m → Line.m_is_in_alpha → Line.l_is_not_in_alpha → Line.l_is_parallel_to_alpha) ∧
  (Plane.alpha_is_parallel_to_beta → Plane.beta_is_parallel_to_gamma → Plane.alpha_is_parallel_to_gamma) ∧
  (Line.l_is_skew_with_m → Line.l_is_parallel_to_alpha_and_beta → Line.m_is_parallel_to_alpha_and_beta → Plane.alpha_is_parallel_to_beta_via_skew_lines) :=
by { sorry }

end correct_statements_l737_737316


namespace num_five_digit_even_except_thousands_div_by_5_and_3_l737_737494

/--
How many 5-digit positive integers having even digits (except the thousands digit, which can be odd) are divisible by both 5 and 3?
-/
theorem num_five_digit_even_except_thousands_div_by_5_and_3 :
  { n : ℕ // 10000 ≤ n ∧ n ≤ 99999 ∧ 
  ((n % 10 = 0 ∧ (even (n / 10000)) ∧ 
  (∃ (a : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ (odd a ∨ even a) ∧ 
  (∀ (d : ℕ), d ∈ [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10] → even d) ∧
  (n % 3 = 0))) }) .to_finset.card = 200 :=
sorry

end num_five_digit_even_except_thousands_div_by_5_and_3_l737_737494


namespace canoes_more_than_kayaks_l737_737246

noncomputable def canoes_difference (C K : ℕ) : Prop :=
  15 * C + 18 * K = 405 ∧ 2 * C = 3 * K → C - K = 5

theorem canoes_more_than_kayaks (C K : ℕ) : canoes_difference C K :=
by
  sorry

end canoes_more_than_kayaks_l737_737246


namespace tax_percentage_l737_737191

-- Definitions
def salary_before_taxes := 5000
def rent_expense_per_month := 1350
def total_late_rent_payments := 2 * rent_expense_per_month
def fraction_of_next_salary_after_taxes := (3 / 5 : ℚ)

-- Main statement to prove
theorem tax_percentage (T : ℚ) : 
  fraction_of_next_salary_after_taxes * (salary_before_taxes - (T / 100) * salary_before_taxes) = total_late_rent_payments → 
  T = 10 :=
by
  sorry

end tax_percentage_l737_737191


namespace prove_identity_l737_737062

namespace Trigonometry

variables (α : ℝ)

theorem prove_identity 
  (h_cos : cos (π/6 - α) = sqrt 3 / 3) :
  sin^2 (α - π/6) - cos (5 * π/6 + α) = (2 + sqrt 3) / 3 :=
by
  sorry

end Trigonometry

end prove_identity_l737_737062


namespace problem_solution_l737_737516

variables (A B C D E F : Type)
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F]
variables [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D] [Module ℝ E] [Module ℝ F]

open_locale classical

noncomputable def triangle_midpoints (A B C D E F : Type) [AddCommGroup A] [Module ℝ A] :=
  ∃ (D E : A), 
  2•D = B + C ∧
  2•E = A + C ∧
  ∃ (F : A),
  4•F = A - B ∧
  ∃ (x y : ℝ),
  D = x•F + y•E

theorem problem_solution {A B C D E F : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C] [AddCommGroup D] [AddCommGroup E] [AddCommGroup F]
  [Module ℝ A] [Module ℝ B] [Module ℝ C] [Module ℝ D] [Module ℝ E] [Module ℝ F]
  (h1 : triangle_midpoints A B C D E F)
  (h2 : ∃ (x y : ℝ), D = x • F + y • E) : 
  ∀ (x y : ℝ),
  x = 2 → y = 1 → x + y = 3 :=
by
  intros x y hx hy
  rw [hx, hy]
  exact rfl

end problem_solution_l737_737516


namespace z_in_fourth_quadrant_l737_737060

-- Given complex numbers z1 and z2
def z1 : ℂ := 3 - 2 * Complex.I
def z2 : ℂ := 1 + Complex.I

-- Define the multiplication of z1 and z2
def z : ℂ := z1 * z2

-- Prove that z is located in the fourth quadrant
theorem z_in_fourth_quadrant : z.re > 0 ∧ z.im < 0 :=
by
  -- Construction and calculations skipped for the math proof,
  -- the result should satisfy the conditions for being in the fourth quadrant
  sorry

end z_in_fourth_quadrant_l737_737060


namespace floor_sqrt_80_eq_8_l737_737908

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737908


namespace largest_positive_integer_n_l737_737027

noncomputable def largest_n (x : ℝ) : ℕ :=
  let n := 8 in
  if ∀ x : ℝ, Real.sin x ^ n + Real.cos x ^ n ≥ 1 / (2 * n) then n else 0

theorem largest_positive_integer_n (x : ℝ) : largest_n x = 8 :=
begin
  sorry
end

end largest_positive_integer_n_l737_737027


namespace increase_in_area_l737_737641

variable {p x y a : ℝ}
variable (h₁ : x + y = p) (h₂ : a > 0)

theorem increase_in_area (hx : x + y = p) (ha : a > 0) : 
  let initial_area := x * (p - x) in
  let new_area := (x + a) * ((p - x) + a) in
  new_area - initial_area = a * (p + a) :=
by
  sorry

end increase_in_area_l737_737641


namespace floor_sqrt_80_eq_8_l737_737916

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737916


namespace perpendicular_vectors_implies_y_eq_5_l737_737454

-- Definitions
variable {y : ℝ}
def A : ℝ × ℝ := (10, 1)
def B (y : ℝ) : ℝ × ℝ := (2, y)
def vector_a : ℝ × ℝ := (1, 2)
def vector_AB (y : ℝ) : ℝ × ℝ := (2 - 10, y - 1)

-- Theorem to be proved
theorem perpendicular_vectors_implies_y_eq_5 (h : vector_AB y ⋅ vector_a = 0) : y = 5 :=
by
  sorry

end perpendicular_vectors_implies_y_eq_5_l737_737454


namespace possible_slopes_of_line_intersects_ellipse_l737_737721

/-- 
A line whose y-intercept is (0, 3) intersects the ellipse 4x^2 + 9y^2 = 36. 
Find all possible slopes of this line. 
-/
theorem possible_slopes_of_line_intersects_ellipse :
  (∀ m : ℝ, ∃ x : ℝ, 4 * x^2 + 9 * (m * x + 3)^2 = 36) ↔ 
  (m <= - (Real.sqrt 5) / 3 ∨ m >= (Real.sqrt 5) / 3) :=
sorry

end possible_slopes_of_line_intersects_ellipse_l737_737721


namespace at_least_one_heart_or_king_l737_737339

-- Define the conditions
def total_cards := 52
def hearts := 13
def kings := 4
def king_of_hearts := 1
def cards_hearts_or_kings := hearts + kings - king_of_hearts

-- Calculate probabilities based on the above conditions
def probability_not_heart_or_king := 
  1 - (cards_hearts_or_kings / total_cards)

def probability_neither_heart_nor_king :=
  (probability_not_heart_or_king) ^ 2

def probability_at_least_one_heart_or_king :=
  1 - probability_neither_heart_nor_king

-- State the theorem to be proved
theorem at_least_one_heart_or_king : 
  probability_at_least_one_heart_or_king = (88 / 169) :=
by
  sorry

end at_least_one_heart_or_king_l737_737339


namespace Marcus_walking_speed_l737_737582

def bath_time : ℕ := 20  -- in minutes
def blow_dry_time : ℕ := bath_time / 2  -- in minutes
def trail_distance : ℝ := 3  -- in miles
def total_dog_time : ℕ := 60  -- in minutes

theorem Marcus_walking_speed :
  let walking_time := total_dog_time - (bath_time + blow_dry_time)
  let walking_time_hours := (walking_time:ℝ) / 60
  (trail_distance / walking_time_hours) = 6 := by
  sorry

end Marcus_walking_speed_l737_737582


namespace floor_sqrt_80_eq_8_l737_737896

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737896


namespace floor_sqrt_80_l737_737932

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737932


namespace floor_sqrt_80_l737_737827

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737827


namespace floor_sqrt_80_eq_8_l737_737843

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737843


namespace floor_sqrt_80_l737_737978

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737978


namespace JackOfHeartsIsSane_l737_737788

inductive Card
  | Ace
  | Two
  | Three
  | Four
  | Five
  | Six
  | Seven
  | JackOfHearts

open Card

def Sane (c : Card) : Prop := sorry

axiom Condition1 : Sane Three → ¬ Sane Ace
axiom Condition2 : Sane Four → (¬ Sane Three ∨ ¬ Sane Two)
axiom Condition3 : Sane Five → (Sane Ace ↔ Sane Four)
axiom Condition4 : Sane Six → (Sane Ace ∧ Sane Two)
axiom Condition5 : Sane Seven → ¬ Sane Five
axiom Condition6 : Sane JackOfHearts → (¬ Sane Six ∨ ¬ Sane Seven)

theorem JackOfHeartsIsSane : Sane JackOfHearts := by
  sorry

end JackOfHeartsIsSane_l737_737788


namespace abc_is_ratio_of_cubes_and_squares_l737_737567

theorem abc_is_ratio_of_cubes_and_squares (a b c : ℚ) (t : ℤ) 
  (h1 : a + b + c = t) 
  (h2 : a^2 + b^2 + c^2 = t) 
  : ∃ x y : ℤ, nat.gcd x y = 1 ∧ abc = x^3 / y^2 :=
sorry

end abc_is_ratio_of_cubes_and_squares_l737_737567


namespace birds_count_214_l737_737412

def two_legged_birds_count (b m i : Nat) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 3 * i = 686 → b = 214

theorem birds_count_214 (b m i : Nat) : two_legged_birds_count b m i :=
by
  sorry

end birds_count_214_l737_737412


namespace speed_of_man_l737_737744

theorem speed_of_man (length_train : ℝ) (speed_train_kmph : ℝ) (passing_time : ℝ)
  (speed_of_man : ℝ) : 
  length_train = 110 ∧ speed_train_kmph = 84 ∧ passing_time = 4.399648028157747 →
  speed_of_man = 6 :=
begin
  sorry
end

end speed_of_man_l737_737744


namespace ratio_of_areas_l737_737735

-- Variables and conditions
variables (h : ℝ)
def s : ℝ := 3 * h -- Side length of the square is three times the height of the triangle
def a : ℝ := (2 * h) / (Real.sqrt 3) -- Side length of the equilateral triangle derived from height

-- Areas
def A_square : ℝ := s h * s h -- Area of the square
def A_triangle : ℝ := (Real.sqrt 3 / 4) * (a h * a h) -- Area of the equilateral triangle

-- Theorem: The ratio of the area of the square to the area of the equilateral triangle is 6
theorem ratio_of_areas (h : ℝ) (h_pos : h > 0) : 
  A_square h / A_triangle h = 6 := 
by 
  sorry

end ratio_of_areas_l737_737735


namespace emily_selects_green_apples_l737_737792

theorem emily_selects_green_apples :
  let total_apples := 10
  let red_apples := 6
  let green_apples := 4
  let selected_apples := 3
  let total_combinations := Nat.choose total_apples selected_apples
  let green_combinations := Nat.choose green_apples selected_apples
  (green_combinations / total_combinations : ℚ) = 1 / 30 :=
by
  sorry

end emily_selects_green_apples_l737_737792


namespace all_propositions_correct_l737_737372

noncomputable def proposition_one (a b : ℝ) : Prop :=
  |a + b| - 2 * |a| ≤ |a - b|

noncomputable def proposition_two (a b : ℝ) : Prop :=
  |a - b| < 1 → |a| < |b| + 1

noncomputable def proposition_three (x y : ℝ) : Prop :=
  |x| < 2 ∧ |y| > 3 → |x / y| < (2 / 3)

noncomputable def proposition_four (A B : ℝ) : Prop :=
  A * B ≠ 0 → log (((|A| + |B|) / 2) : ℝ) ≥ (1 / 2) * (log |A| + log |B|)

theorem all_propositions_correct :
  (∀ (a b : ℝ), proposition_one a b) ∧
  (∀ (a b : ℝ), proposition_two a b) ∧
  (∀ (x y : ℝ), proposition_three x y) ∧
  (∀ (A B : ℝ), proposition_four A B) :=
by
  sorry

end all_propositions_correct_l737_737372


namespace product_of_sequence_l737_737761

def sequence_term (k : Nat) : Rat :=
  (4 * k) / (4 * k + 4)

theorem product_of_sequence :
  ∏ k in Finset.range (501 + 1), sequence_term k = 1 / 502 := 
by
  sorry

end product_of_sequence_l737_737761


namespace expand_product_l737_737015

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := 
  sorry

end expand_product_l737_737015


namespace min_value_eq_six_l737_737114

theorem min_value_eq_six
    (α β : ℝ)
    (k : ℝ)
    (h1 : α^2 + 2 * (k + 3) * α + (k^2 + 3) = 0)
    (h2 : β^2 + 2 * (k + 3) * β + (k^2 + 3) = 0)
    (h3 : (2 * (k + 3))^2 - 4 * (k^2 + 3) ≥ 0) :
    ( (α - 1)^2 + (β - 1)^2 = 6 ) := 
sorry

end min_value_eq_six_l737_737114


namespace equation_of_line_l737_737053

noncomputable def check_line_eq (A C : ℝ × ℝ) (r : ℝ) (k : ℝ) : Bool :=
  let d := (k - 1).abs / (Real.sqrt (k^2 + 1))
  let chord_length := 2 * (Real.sqrt (r^2 - d^2))
  chord_length = 2 * Real.sqrt 3

theorem equation_of_line (A C : ℝ × ℝ) (r : ℝ) (k : ℝ) (l : ℝ) :
  A = (-1, 1) ∧ C = (-2, 0) ∧ r = 2 ∧ check_line_eq A C r k = true ∧ check_line_eq A C r 0 = true →
  (∀ x y : ℝ, (x = -1 ∨ y = 1)) :=
by
  sorry

end equation_of_line_l737_737053


namespace max_m_value_real_roots_interval_l737_737623

theorem max_m_value_real_roots_interval :
  (∃ x ∈ (Set.Icc 0 1), x^3 - 3 * x - m = 0) → m ≤ 0 :=
by
  sorry 

end max_m_value_real_roots_interval_l737_737623


namespace find_smallest_n_l737_737429

open Matrix Complex

noncomputable def rotation_matrix := ![
  ![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
  ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]
]

def I_2 := (1 : Matrix (Fin 2) (Fin 2) ℝ)

theorem find_smallest_n (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (hA : A = rotation_matrix) : 
  ∃ (n : ℕ), 0 < n ∧ A ^ n = I_2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → A ^ m ≠ I_2 :=
by {
  sorry
}

end find_smallest_n_l737_737429


namespace probability_of_yellow_jelly_bean_l737_737719

theorem probability_of_yellow_jelly_bean (P_red P_orange P_yellow : ℝ) 
  (h1 : P_red = 0.2) 
  (h2 : P_orange = 0.5) 
  (h3 : P_red + P_orange + P_yellow = 1) : 
  P_yellow = 0.3 :=
sorry

end probability_of_yellow_jelly_bean_l737_737719


namespace minimum_cuts_to_divide_cube_l737_737313

theorem minimum_cuts_to_divide_cube (large_cube_edge small_cube_edge : ℕ) (num_small_cubes : ℕ) :
  large_cube_edge = 40 ∧ small_cube_edge = 10 ∧ num_small_cubes = 64 → 
  ∃ (cuts : ℕ), cuts = 6 :=
begin
  sorry
end

end minimum_cuts_to_divide_cube_l737_737313


namespace vertex_of_parabola_l737_737786

theorem vertex_of_parabola (a b c : ℝ) (h_eq : ∀ x : ℝ, (3 : ℝ) * x^2 - 6 * x + 2 = a * x^2 + b * x + c) :
  (1 : ℝ, -1 : ℝ) = (-b / (2 * a), (4 * a * c - b^2) / (4 * a)) := 
  by 
  have h_eq_params : a = 3 ∧ b = -6 ∧ c = 2 := sorry
  have h_vertex_formula : (-b / (2 * a), (4 * a * c - b^2) / (4 * a)) = (1, -1) := sorry
  exact h_vertex_formula

end vertex_of_parabola_l737_737786


namespace solve_quadratic_l737_737205

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = 2 + Real.sqrt 11) ∧ (x2 = 2 - (Real.sqrt 11)) ∧ 
  (∀ x : ℝ, x^2 - 4*x - 7 = 0 ↔ x = x1 ∨ x = x2) := 
sorry

end solve_quadratic_l737_737205


namespace floor_sqrt_80_l737_737866

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737866


namespace broken_line_path_length_l737_737790

/-
  Problem Statement:
  Diameter AB of a circle with center O is 12 units.
  Point C is located 3 units from A along AB, and point D is located 5 units from B along AB.
  P is any point on the circle and the highest point.
  Prove that the length of the broken-line path from C to P to D is 3√5 + √37 units.
-/

-- Given conditions
def diameter (A B O : Point) (radius : ℝ) : Prop :=
  dist A B = 2 * radius ∧ midpoint A B = O

def loc_C (A C : Point) : Prop := 
  dist A C = 3

def loc_D (B D : Point) : Prop := 
  dist B D = 5

def highest_point (O P : Point) (radius : ℝ) : Prop :=
  dist O P = radius ∧ ∀ Q, Q ≠ P → dist O Q < dist O P

noncomputable def broken_line_length (C P D : Point) : ℝ := 
  dist C P + dist P D

-- Main theorem
theorem broken_line_path_length (A B C D O P : Point)
  (h_diameter: diameter A B O 6)
  (hC: loc_C A C) 
  (hD: loc_D B D)
  (hP: highest_point O P 6):
  broken_line_length C P D = 3 * Real.sqrt 5 + Real.sqrt 37 :=
sorry -- Proof to be filled in later

end broken_line_path_length_l737_737790


namespace negation_of_exists_leq_l737_737620

theorem negation_of_exists_leq (
  P : ∃ x : ℝ, x^2 - 2 * x + 4 ≤ 0
) : ∀ x : ℝ, x^2 - 2 * x + 4 > 0 :=
sorry

end negation_of_exists_leq_l737_737620


namespace floor_sqrt_80_eq_8_l737_737912

theorem floor_sqrt_80_eq_8 : ∀ (x : ℝ), 8 < x ∧ x < 9 → ∃ y : ℕ, y = 8 ∧ (⌊x⌋ : ℝ) = y :=
by {
  intros x h,
  use 8,
  split,
  { refl },
  {
    sorry
  }
}

end floor_sqrt_80_eq_8_l737_737912


namespace fly_max_path_in_cube_l737_737714

-- Define the main problem constants and assumptions
def side_length : ℝ := 2
def distance_adjacent : ℝ := side_length
def distance_face_diagonal : ℝ := real.sqrt (side_length * side_length * 2)
def distance_space_diagonal : ℝ := real.sqrt (side_length * side_length * 3)

-- Define the conditions 
def fly_visits_six_corners : Prop := true -- placeholder, meaning it visits six distinct corners

def fly_flies_between_non_adjacent (d: ℝ) : Prop := d > distance_adjacent -- placeholder for non-adjacency

def max_path_length : ℝ := 3 * distance_space_diagonal

-- The theorem statement we need to prove
theorem fly_max_path_in_cube :
  fly_visits_six_corners →
  (∀ d, fly_flies_between_non_adjacent d) →
  max_path_length = 6 * real.sqrt 3 := 
sorry

end fly_max_path_in_cube_l737_737714


namespace average_typed_words_per_minute_l737_737617

def rudy_wpm := 64
def joyce_wpm := 76
def gladys_wpm := 91
def lisa_wpm := 80
def mike_wpm := 89
def num_team_members := 5

theorem average_typed_words_per_minute : 
  (rudy_wpm + joyce_wpm + gladys_wpm + lisa_wpm + mike_wpm) / num_team_members = 80 := 
by
  sorry

end average_typed_words_per_minute_l737_737617


namespace angle_QUR_is_71_l737_737530

theorem angle_QUR_is_71 
  (P S Q R T U V W : Type) 
  (PS : P ≄ S)
  (hQ : Q ∈ P ∨ Q ∈ S)
  (hR: R ∈ P ∨ R ∈ S)
  (hQWR : ∠ QWR = 38)
  (x y : ℝ)
  (hTQP : ∠ TQP = x)
  (hTQW : ∠ TQW = x)
  (hVRS : ∠ VRS = y)
  (hVRW : ∠ VRW = y)
  (hU_def : U = line_through T Q ∩ line_through V R)
  : ∠ QUR = 71 := 
sorry

end angle_QUR_is_71_l737_737530


namespace integral_calculation_l737_737769

noncomputable def integral_example : ℝ :=
  \int_{0}^{1} (1 + real.sqrt(1 - x^2)) dx

theorem integral_calculation :
  ∫ x in (0:ℝ)..(1:ℝ), (1 + real.sqrt(1 - x^2)) = 1 + (real.pi / 4) :=
by
  sorry

end integral_calculation_l737_737769


namespace hiring_manager_criteria_l737_737214

-- Define the parameter values
def average_age : ℝ := 31
def std_dev : ℝ := 8
def max_range : ℝ := 17
def num_stddev_away := 1.0625

-- Statement to prove
theorem hiring_manager_criteria :
  max_range / 2 / std_dev = num_stddev_away := 
sorry

end hiring_manager_criteria_l737_737214


namespace find_A_B_l737_737787

theorem find_A_B : ∀ (x B A : ℝ),
  (B * x - 13) / (x^2 - 7 * x + 10) = A / (x - 2) + 5 / (x - 5) →
  A = 3 / 5 ∧ B = 28 / 5 →
  A + B = 6.2 :=
by
  intros x B A H_eq H_vals
  cases H_vals with H_A H_B
  rw [H_A, H_B]
  norm_num
  done

end find_A_B_l737_737787


namespace max_value_l737_737067

def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def is_increasing (f : ℝ → ℝ) := ∀ {a b}, a < b → f a < f b

theorem max_value (f : ℝ → ℝ) (x y : ℝ)
  (h_odd : is_odd f)
  (h_increasing : is_increasing f)
  (h_eq : f (x^2 - 2 * x) + f y = 0) :
  2 * x + y ≤ 4 :=
sorry

end max_value_l737_737067


namespace floor_of_sqrt_80_l737_737927

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737927


namespace binary_representation_of_51_l737_737778

theorem binary_representation_of_51 : nat.binary_repr 51 = "110011" := 
sorry

end binary_representation_of_51_l737_737778


namespace expand_product_l737_737016

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 :=
by 
  sorry

end expand_product_l737_737016


namespace math_problem_l737_737387

noncomputable def problem : Real :=
  (2 * Real.sqrt 2 - 1) ^ 2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5)

theorem math_problem :
  problem = 5 - 4 * Real.sqrt 2 :=
by
  sorry

end math_problem_l737_737387


namespace part1_part2_l737_737188

/-- Part 1 -/
theorem part1 (a b x1 x2 : ℝ) (f : ℝ → ℝ) : f = λ x, a * x + b → 
  f ((x1 + x2) / 2) = (f x1 + f x2) / 2 :=
by
  intros h
  rw h
  sorry

/-- Part 2 -/
theorem part2 (a b x1 x2 : ℝ) (g : ℝ → ℝ) : g = λ x, x^2 + a * x + b → 
  g ((x1 + x2) / 2) ≤ (g x1 + g x2) / 2 :=
by
  intros h
  rw h
  sorry

end part1_part2_l737_737188


namespace determine_b20_l737_737055

theorem determine_b20 (b : Fin 21 → ℕ)
  (h : (1 - (z : ℂ)) ^ b 1 * (∏ i in range 2 21, (1 - z ^ i) ^ b i) = 1 - 3 * z + ∑ i in range 2 21, c i * z ^ i) :
  b 20 = 3 := 
by {
  sorry
}

end determine_b20_l737_737055


namespace floor_sqrt_80_l737_737887

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737887


namespace find_n_l737_737743

theorem find_n (x y m n : ℕ) (hx : 0 ≤ x ∧ x < 10) (hy : 0 ≤ y ∧ y < 10) 
  (h1 : 100 * y + x = (x + y) * m) (h2 : 100 * x + y = (x + y) * n) : n = 101 - m :=
by
  sorry

end find_n_l737_737743


namespace floor_sqrt_80_l737_737872

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737872


namespace largest_prime_divisor_l737_737407

theorem largest_prime_divisor : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (17^2 + 60^2) → q ≤ p :=
  sorry

end largest_prime_divisor_l737_737407


namespace unique_prime_sum_diff_l737_737022

theorem unique_prime_sum_diff :
  ∀ p : ℕ, Prime p ∧ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ (p = p1 + 2) ∧ (p = p3 - 2)) → p = 5 :=
by
  sorry

end unique_prime_sum_diff_l737_737022


namespace floor_sqrt_80_eq_8_l737_737899

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737899


namespace g_eq_g_g_l737_737563

noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 1

theorem g_eq_g_g (x : ℝ) : 
  g (g x) = g x ↔ x = 2 + Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 + 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 + Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) 
             ∨ x = 2 - Real.sqrt ((11 - 2 * Real.sqrt 21) / 2) := 
by
  sorry

end g_eq_g_g_l737_737563


namespace ordered_pairs_eq_count_l737_737097

open Nat

theorem ordered_pairs_eq_count:
    (∃! (a b : ℕ), a > 0 ∧ b > 0 ∧ a * b + 83 = 24 * lcm a b + 17 * gcd a b ∧
    (a, b) ∈ ({(30, 49), (49, 30)} : Finset (ℕ × ℕ))) :=
sorry

end ordered_pairs_eq_count_l737_737097


namespace average_sales_is_104_l737_737211

-- Define the sales data for the months January to May
def january_sales : ℕ := 150
def february_sales : ℕ := 90
def march_sales : ℕ := 60
def april_sales : ℕ := 140
def may_sales : ℕ := 100
def may_discount : ℕ := 20

-- Define the adjusted sales for May after applying the discount
def adjusted_may_sales : ℕ := may_sales - (may_sales * may_discount / 100)

-- Define the total sales from January to May
def total_sales : ℕ := january_sales + february_sales + march_sales + april_sales + adjusted_may_sales

-- Define the number of months
def number_of_months : ℕ := 5

-- Define the average sales per month
def average_sales_per_month : ℕ := total_sales / number_of_months

-- Prove that the average sales per month is equal to 104
theorem average_sales_is_104 : average_sales_per_month = 104 := by
  -- Here, we'd write the proof, but we'll leave it as 'sorry' for now
  sorry

end average_sales_is_104_l737_737211


namespace calculate_principal_l737_737428

theorem calculate_principal
  (I : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hI : I = 8625)
  (hR : R = 50 / 3)
  (hT : T = 3 / 4)
  (hInterest : I = (P * (R / 100) * T)) :
  P = 6900000 := by
  sorry

end calculate_principal_l737_737428


namespace find_number_l737_737311

theorem find_number (x : ℝ) (h : (x / 3) * 12 = 9) : x = 9 / 4 := by
  sorry

end find_number_l737_737311


namespace sum_a_b_zero_l737_737117

theorem sum_a_b_zero {a b : ℝ} (h1 : (a : ℂ) + complex.I * complex.I = (b : ℂ) + complex.I) :
  a + b = 0 :=
sorry

end sum_a_b_zero_l737_737117


namespace solve_for_a_b_find_range_of_k_l737_737079

variables (a b k : ℝ) (x : ℝ)

def f (x : ℝ) := Real.exp (3 * a * x)
def g (x : ℝ) := k * x + b

-- Prove that a = 1/3 and b = 0
theorem solve_for_a_b (h_tangent_slope : 3 * a * Real.exp (3 * a) = Real.exp 1)
    (h_odd : ∀ x, g (-x) = -g x) : a = 1 / 3 ∧ b = 0 :=
by
  sorry

-- Prove the range of k such that f(x) > g(x) for all x in (-2, 2)
theorem find_range_of_k 
    (h_a : a = 1 / 3)
    (h_b : b = 0)
    (h_range : ∀ x : ℝ, x ∈ Set.Ioo (-2) 2 → f x > g x) 
    : k ∈ Set.Ico (-(1 / (2 * Real.exp 2))) Real.exp 1 :=
by
  sorry

end solve_for_a_b_find_range_of_k_l737_737079


namespace num_bases_with_final_digit_one_l737_737037

def hasFinalDigitOne (n b : ℕ) : Prop := n % b = 1

def validBases (n lower upper : ℕ) : List ℕ :=
  List.filter (λ b, hasFinalDigitOne n b) (List.range' lower (upper - lower + 1))

theorem num_bases_with_final_digit_one (n lower upper result : ℕ) :
  validBases n lower upper = [2, 3, 4, 6, 8] → result = 5 :=
by
  intros h
  unfold validBases at h
  simp at h
  simp [validBases, hasFinalDigitOne] ------ defnitions
  sorry

end num_bases_with_final_digit_one_l737_737037


namespace postage_problem_l737_737997

noncomputable def sum_all_positive_integers (n1 n2 : ℕ) : ℕ :=
  n1 + n2

theorem postage_problem : sum_all_positive_integers 21 22 = 43 :=
by
  have h1 : ∀ x y z : ℕ, 7 * x + 21 * y + 23 * z ≠ 120 := sorry
  have h2 : ∀ x y z : ℕ, 7 * x + 22 * y + 24 * z ≠ 120 := sorry
  exact rfl

end postage_problem_l737_737997


namespace largest_multiple_of_7_gt_neg_150_l737_737268

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l737_737268


namespace floor_sqrt_80_l737_737874

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737874


namespace maximize_container_volume_l737_737360

theorem maximize_container_volume :
  ∃ x : ℝ, 0 < x ∧ x < 24 ∧ ∀ y : ℝ, 0 < y ∧ y < 24 → 
  ( (48 - 2 * x)^2 * x ≥ (48 - 2 * y)^2 * y ) ∧ x = 8 :=
sorry

end maximize_container_volume_l737_737360


namespace equal_candy_distribution_l737_737155

theorem equal_candy_distribution :
  ∀ (candies friends : ℕ), candies = 30 → friends = 4 → candies % friends = 2 :=
by
  sorry

end equal_candy_distribution_l737_737155


namespace floor_sqrt_80_l737_737885

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737885


namespace largest_multiple_of_7_gt_neg_150_l737_737267

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l737_737267


namespace third_divisor_l737_737032

theorem third_divisor (x : ℕ) (h12 : 12 ∣ (x + 3)) (h15 : 15 ∣ (x + 3)) (h40 : 40 ∣ (x + 3)) :
  ∃ d : ℕ, d ≠ 12 ∧ d ≠ 15 ∧ d ≠ 40 ∧ d ∣ (x + 3) ∧ d = 2 :=
by
  sorry

end third_divisor_l737_737032


namespace probability_heart_or_king_l737_737327

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l737_737327


namespace maximum_product_of_sums_l737_737549

open BigOperators

noncomputable theory

-- Statement of the problem
theorem maximum_product_of_sums (n : ℕ) (h : n ≥ 2) :
  let m := n / 3 in
  let r := n % 3 in
  ∃ (a : List ℕ), (a.sum = n) ∧
    match r with
    | 0    => list.product (@List.map ℕ ℕ (\(a_i : ℕ), (a_i * (a_i + 1)) / 2) a) = 6^m
    | 1    => list.product (@List.map ℕ ℕ (\(a_i : ℕ), (a_i * (a_i + 1)) / 2) a) = 10 * 6^(m - 1)
    | 2    => list.product (@List.map ℕ ℕ (\(a_i : ℕ), (a_i * (a_i + 1)) / 2) a) = 3 * 6^m
    | _    => false
  := 
sorry

end maximum_product_of_sums_l737_737549


namespace floor_sqrt_80_l737_737963

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737963


namespace magnitude_proj_is_two_l737_737554

noncomputable def magnitude_proj (u z : ℝ^n) (h1 : dot_product u z = 6) (h2 : norm z = 3) : ℝ :=
norm ((dot_product u z / (norm z)^2) • z)

theorem magnitude_proj_is_two (u z : ℝ^n) (h1 : dot_product u z = 6) (h2 : norm z = 3) :
  magnitude_proj u z h1 h2 = 2 :=
by
  sorry

end magnitude_proj_is_two_l737_737554


namespace smallest_a_exists_l737_737783

theorem smallest_a_exists (a p b : ℕ) (hp : prime p) (ha : a ≥ 2) (hb : b ≥ 2) :
  (∃ (p b : ℕ), prime p ∧ b ≥ 2 ∧ (a^p - a) / p = b^2) → a = 9 :=
by
  sorry

end smallest_a_exists_l737_737783


namespace floor_neg_seven_fourths_l737_737800

theorem floor_neg_seven_fourths : Int.floor (-7 / 4 : ℚ) = -2 := 
by 
  sorry

end floor_neg_seven_fourths_l737_737800


namespace floor_sqrt_80_l737_737809

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737809


namespace floor_sqrt_80_l737_737934

noncomputable def floor_sqrt (n : ℕ) : ℕ :=
  int.to_nat (Int.floor (Real.sqrt n))

theorem floor_sqrt_80 : floor_sqrt 80 = 8 := by
  -- Conditions
  have h1 : 64 < 80 := by norm_num
  have h2 : 80 < 81 := by norm_num
  have h3 : 8 < Real.sqrt 80 := by norm_num; exact Real.sqrt_pos.mpr (by norm_num)
  have h4 : Real.sqrt 80 < 9 := by 
    apply Real.sqrt_lt; norm_num
  -- Thus, we conclude
  sorry

end floor_sqrt_80_l737_737934


namespace tetrahedron_coloring_count_l737_737773

def tetrahedron_coloring (V E : Type) [Fintype V] [Fintype E] (incidence : E → V × V) : Prop :=
  ∀ (colors : V → Fin 4), ∃ (valid : Prop), valid = 
    ∀ e, (colors (incidence e).fst ≠ colors (incidence e).snd)

theorem tetrahedron_coloring_count : ∀ 
  (V E : Type) [Fintype V] [Fintype E] (incidence : E → V × V), 
  tetrahedron_coloring V E incidence → Fintype.card {f : V → Fin 4 // 
  ∀ e, (f (incidence e).fst ≠ f (incidence e).snd)} = 24 :=
by admit

end tetrahedron_coloring_count_l737_737773


namespace floor_sqrt_80_l737_737878

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737878


namespace min_height_of_cuboid_l737_737711

theorem min_height_of_cuboid (h : ℝ) (side_len : ℝ) (small_spheres_r : ℝ) (large_sphere_r : ℝ) :
  side_len = 4 → 
  small_spheres_r = 1 → 
  large_sphere_r = 2 → 
  ∃ h_min : ℝ, h_min = 2 + 2 * Real.sqrt 7 ∧ h ≥ h_min := 
by
  sorry

end min_height_of_cuboid_l737_737711


namespace smallest_enclosing_sphere_radius_l737_737011

-- Define the radius of each small sphere and the center set
def radius (r : ℝ) : Prop := r = 2

def center_set (C : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ c ∈ C, ∃ x y z : ℝ, 
    (x = 2 ∨ x = -2) ∧ 
    (y = 2 ∨ y = -2) ∧ 
    (z = 2 ∨ z = -2) ∧
    (c = (x, y, z))

-- Prove the radius of the smallest enclosing sphere is 2√3 + 2
theorem smallest_enclosing_sphere_radius (r : ℝ) (C : Set (ℝ × ℝ × ℝ)) 
  (h_radius : radius r) (h_center_set : center_set C) :
  ∃ R : ℝ, R = 2 * Real.sqrt 3 + 2 :=
sorry

end smallest_enclosing_sphere_radius_l737_737011


namespace sample_of_third_grade_students_l737_737718

/-- Problem statement: calculate number of third grade students in a sample -/
theorem sample_of_third_grade_students (total_students sample_size first_grade_students : ℕ)
  (proportion_second_grade : ℚ)
  (h_total : total_students = 2800)
  (h_sample : sample_size = 40)
  (h_first_grade : first_grade_students = 910)
  (h_proportion_second_grade : proportion_second_grade = 3 / 10) :
  ∃ third_grade_students_sample : ℕ,
    third_grade_students_sample = sample_size - ((proportion_second_grade * sample_size).to_nat + (first_grade_students * sample_size / total_students).to_nat) := 
sorry

end sample_of_third_grade_students_l737_737718


namespace Claire_will_earn_about_92_13_l737_737391

noncomputable def ClaireEarnings : ℝ :=
  let total_flowers := 400
  let tulips := 120
  let white_roses := 80
  let total_roses := total_flowers - tulips
  let red_roses := total_roses - white_roses
  let small_red_roses := 40
  let medium_red_roses := 60
  let large_red_roses := red_roses - (small_red_roses + medium_red_roses)
  let sell_half (n : ℝ) := n / 2
  let earnings (n : ℝ) (price : ℝ) := n * price
  let discount (n : ℝ) (amount : ℝ) := if n ≥ 30 then 0.15 else if n ≥ 20 then 0.10 else if n ≥ 10 then 0.05 else 0
  let discounted_earnings (n : ℝ) (price : ℝ) := let e := earnings n price in e - e * discount n e
  let small_red_roses_sold := sell_half small_red_roses
  let medium_red_roses_sold := sell_half medium_red_roses
  let large_red_roses_sold := sell_half large_red_roses
  discounted_earnings small_red_roses_sold 0.75 +
  discounted_earnings medium_red_roses_sold 1 +
  discounted_earnings large_red_roses_sold 1.25

theorem Claire_will_earn_about_92_13 : abs (ClaireEarnings - 92.13) < 0.01 := by
  sorry

end Claire_will_earn_about_92_13_l737_737391


namespace probability_of_drawing_specific_cards_l737_737043

theorem probability_of_drawing_specific_cards :
  (probability_of_drawing_cards [⟨card_type.heart, false, false⟩, ⟨card_type.king, true, true⟩, ⟨card_type.king, true, true⟩, ⟨card_type.ace, false, false⟩] from standard_deck) = (1 / 12317) :=
  sorry

end probability_of_drawing_specific_cards_l737_737043


namespace floor_of_sqrt_80_l737_737918

theorem floor_of_sqrt_80 : 
  ∀ (n: ℕ), n^2 = 64 → (n+1)^2 = 81 → 64 < 80 → 80 < 81 → ⌊real.sqrt 80⌋ = 8 :=
begin
  intros,
  sorry
end

end floor_of_sqrt_80_l737_737918


namespace stamp_problem_l737_737998

/-- Define the context where we have stamps of 7, n, and (n + 2) cents, and 120 cents being the largest
    value that cannot be formed using these stamps -/
theorem stamp_problem (n : ℕ) (h : ∀ k, k > 120 → ∃ a b c, k = 7 * a + n * b + (n + 2) * c) (hn : ¬ ∃ a b c, 120 = 7 * a + n * b + (n + 2) * c) : n = 22 :=
sorry

end stamp_problem_l737_737998


namespace sum_of_ten_numbers_l737_737512

theorem sum_of_ten_numbers (average count : ℝ) (h_avg : average = 5.3) (h_count : count = 10) : 
  average * count = 53 :=
by
  sorry

end sum_of_ten_numbers_l737_737512


namespace chad_savings_l737_737771

variable (savings_rate : ℝ) (mowing : ℤ) (birthday : ℤ) (odd_jobs : ℤ) (total_savings : ℤ)

theorem chad_savings :
  savings_rate = 0.40 →
  mowing = 600 →
  birthday = 250 →
  odd_jobs = 150 →
  total_savings = 460 →
  ∃ (video_games_earnings : ℤ),
    total_savings = (savings_rate * (mowing + birthday + odd_jobs + video_games_earnings : ℝ)).toInt ∧
    video_games_earnings = 150 :=
by
  sorry

end chad_savings_l737_737771


namespace floor_sqrt_80_l737_737983

theorem floor_sqrt_80 : (Nat.floor (Real.sqrt 80)) = 8 := by
  have h₁ : 8^2 = 64 := by norm_num
  have h₂ : 9^2 = 81 := by norm_num
  have h₃ : 8 < Real.sqrt 80 := by
    norm_num
    rw [Real.sqrt_lt_iff]
    linarith
  have h₄ : Real.sqrt 80 < 9 := by
    norm_num
    rw [←Real.sqrt_inj]
    linarith
  apply Nat.floor_eq
  apply lt.trans
  exact h₃
  exact h₄

end floor_sqrt_80_l737_737983


namespace partner_q_investment_time_l737_737239

theorem partner_q_investment_time 
  (P Q R : ℝ)
  (Profit_p Profit_q Profit_r : ℝ)
  (Tp Tq Tr : ℝ)
  (h1 : P / Q = 7 / 5)
  (h2 : Q / R = 5 / 3)
  (h3 : Profit_p / Profit_q = 7 / 14)
  (h4 : Profit_q / Profit_r = 14 / 9)
  (h5 : Tp = 5)
  (h6 : Tr = 9) :
  Tq = 14 :=
by
  sorry

end partner_q_investment_time_l737_737239


namespace product_of_roots_of_polynomial_l737_737000

theorem product_of_roots_of_polynomial : 
  ∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x^2 - x - 34 = 0) ∧ (a * b = -34) :=
by
  sorry

end product_of_roots_of_polynomial_l737_737000


namespace simplify_f_value_f_cos_value_f_specific_l737_737463

variable (α : ℝ)
variable (hα : (π < α ∧ α < 3 * π))
noncomputable def f (α : ℝ) : ℝ :=
  (sin (π - α) * cos (2 * π - α) * tan (-α - π)) / (tan (-α) * sin (-π - α))

theorem simplify_f : f α = cos α :=
  sorry

theorem value_f_cos (h : cos (α - 3/2 * π) = 1/5) : f α = -2 * sqrt 6 / 5 :=
  sorry

theorem value_f_specific : f (-1860 * π / 180) = 1/2 :=
  sorry

end simplify_f_value_f_cos_value_f_specific_l737_737463


namespace parabola_equation_l737_737724

noncomputable def parabola_focus : (ℝ × ℝ) := (5, -2)

noncomputable def parabola_directrix (x y : ℝ) : Prop := 4 * x - 5 * y = 20

theorem parabola_equation (x y : ℝ) :
  (parabola_focus = (5, -2)) →
  (parabola_directrix x y) →
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009 = 0 :=
by
  sorry

end parabola_equation_l737_737724


namespace floor_sqrt_80_l737_737826

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737826


namespace geometric_sequence_a1_l737_737471

theorem geometric_sequence_a1 (a1 a2 a3 S3 : ℝ) (q : ℝ)
  (h1 : S3 = a1 + (1 / 2) * a2)
  (h2 : a3 = (1 / 4))
  (h3 : S3 = a1 * (1 + q + q^2))
  (h4 : a2 = a1 * q)
  (h5 : a3 = a1 * q^2) :
  a1 = 1 :=
sorry

end geometric_sequence_a1_l737_737471


namespace floor_sqrt_80_l737_737953

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737953


namespace find_positive_integer_pairs_l737_737421

theorem find_positive_integer_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a^2 = 3 * b^3) ↔ ∃ d : ℕ, 0 < d ∧ a = 18 * d^3 ∧ b = 6 * d^2 :=
by
  sorry

end find_positive_integer_pairs_l737_737421


namespace two_pipes_fill_time_l737_737310

theorem two_pipes_fill_time (R : ℝ) (h1 : (3 : ℝ) * R * (8 : ℝ) = 1) : (2 : ℝ) * R * (12 : ℝ) = 1 :=
by 
  have hR : R = 1 / 24 := by linarith
  rw [hR]
  sorry

end two_pipes_fill_time_l737_737310


namespace transform_quadratic_l737_737241

theorem transform_quadratic (x m n : ℝ) 
  (h : x^2 - 6 * x - 1 = 0) : 
  (x + m)^2 = n ↔ (m = 3 ∧ n = 10) :=
by sorry

end transform_quadratic_l737_737241


namespace probability_heart_or_king_l737_737328

theorem probability_heart_or_king (cards hearts kings : ℕ) (prob_non_heart_king : ℚ) 
    (prob_two_non_heart_king : ℚ) : 
    cards = 52 → hearts = 13 → kings = 4 → 
    prob_non_heart_king = 36 / 52 → prob_two_non_heart_king = (36 / 52) ^ 2 → 
    1 - prob_two_non_heart_king = 88 / 169 :=
by
  intros h_cards h_hearts h_kings h_prob_non_heart_king h_prob_two_non_heart_king
  sorry

end probability_heart_or_king_l737_737328


namespace distance_BN_fixed_l737_737059

theorem distance_BN_fixed : 
  ∀ (m : ℝ), 
  (m > -Real.sqrt 2 ∧ m < Real.sqrt 2) → 
  let l := λ x, (1 / 2) * x + m,
      N := (x : ℝ) -> (l x = 0) -> (x, 0 : ℝ × ℝ),
      x1 := (-2 * m), x2 := (2 * m^2 - 2),
      A := (-m, (1 / 2) * m : ℝ × ℝ),
      C := ((-2) * m, 0 : ℝ × ℝ),
      M := λ m, (-m, (1 / 2) * m : ℝ × ℝ),
      distance := λ p1 p2 : ℝ × ℝ, Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  in distance B N = Real.sqrt 10 / 2 := 
sorry

end distance_BN_fixed_l737_737059


namespace constant_term_binomial_expansion_l737_737024

theorem constant_term_binomial_expansion : 
  let x_term (r : ℕ) := (-2)^r * Nat.choose 8 r * x^(8 / 3 - 4 / 3 * r)
  let constant_term := (-2)^2 * Nat.choose 8 2
  x_term 2 = constant_term := by
  sorry

end constant_term_binomial_expansion_l737_737024


namespace profit_percent_correct_l737_737348

noncomputable def profit_percent (P : ℝ) : ℝ :=
  let cp := 46 * P in
  let sp_per_pen := 0.99 * P in
  let total_sp := 50 * sp_per_pen in
  let profit := total_sp - cp in
  (profit / cp) * 100

theorem profit_percent_correct (P : ℝ) (hP : P > 0) :
  profit_percent P = 350 / 46 :=
by
  let cp := 46 * P
  let sp_per_pen := 0.99 * P
  let total_sp := 50 * sp_per_pen
  let profit := total_sp - cp
  have h : (profit / cp) * 100 = (3.5 * P / (46 * P)) * 100,
  { sorry },
  have h_eq : 3.5 / 46 = 350 / 4600 := by sorry,
  rw [h_eq] at h,
  have h_eq2 : 350 / 4600 = 350 / 46 := by sorry,
  rw [h_eq2] at h,
  exact h

end profit_percent_correct_l737_737348


namespace misread_weight_l737_737615

theorem misread_weight (n : ℕ) (average_incorrect : ℚ) (average_correct : ℚ) (corrected_weight : ℚ) (incorrect_total correct_total diff : ℚ)
  (h1 : n = 20)
  (h2 : average_incorrect = 58.4)
  (h3 : average_correct = 59)
  (h4 : corrected_weight = 68)
  (h5 : incorrect_total = n * average_incorrect)
  (h6 : correct_total = n * average_correct)
  (h7 : diff = correct_total - incorrect_total)
  (h8 : diff = corrected_weight - x) : x = 56 := 
sorry

end misread_weight_l737_737615


namespace infinite_geometric_series_limit_l737_737128

theorem infinite_geometric_series_limit (x : ℝ) (h : (9.choose 2) * (-2)^x ^ 2 = 288) :
  (∃ x, x = 3 / 2 ∧ ∀n, (n → ∞) ((∑ i in range n, 1 / x ^ i) = 2 )) := by
  sorry

end infinite_geometric_series_limit_l737_737128


namespace largest_multiple_of_7_gt_neg_150_l737_737264

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l737_737264


namespace circle_equation_and_line_slope_l737_737441

open Real

-- Definitions of points A and B
def A : Real × Real := (0, 2)
def B : Real × Real := (2, -2)

-- Condition: the center of circle C lies on the line x - y + 1 = 0
def center_on_line (center : Real × Real) : Prop :=
  (center.1 - center.2 + 1 = 0)

-- Given a point through which the line with slope m passes
def point_through_line : Real × Real := (1, 4)

-- The distance formula to validate the chord length condition
def distance_from_center_to_line
  (center : Real × Real) (k : Real) (b : Real) : Real :=
  abs (-center.1 * k + center.2 + b) / sqrt (k^2 + 1)

theorem circle_equation_and_line_slope :
  ∃ (center : Real × Real) (r : Real), center ∈ {center | center_on_line center} ∧ 
  ((center.1 + 3)^2 + (center.2 + 2)^2 = r^2) ∧ r = 5 ∧ 
  ∃ (k b : Real), 
    k = 5 / 12 ∧ 
    b = 43 / 12 ∧ 
    distance_from_center_to_line center k b = 4 := 
sorry

end circle_equation_and_line_slope_l737_737441


namespace find_a_l737_737449

theorem find_a (a : ℝ) (h : ∃ x, x = 3 ∧ x^2 + a * x + a - 1 = 0) : a = -2 :=
sorry

end find_a_l737_737449


namespace floor_neg_7_over_4_l737_737802

theorem floor_neg_7_over_4 : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_7_over_4_l737_737802


namespace parabola_translation_l737_737655

theorem parabola_translation :
  ∀ f g : ℝ → ℝ,
    (∀ x, f x = - (x - 1) ^ 2) →
    (∀ x, g x = f (x - 1) + 2) →
    ∀ x, g x = - (x - 2) ^ 2 + 2 :=
by
  -- Add the proof steps here if needed
  sorry

end parabola_translation_l737_737655


namespace floor_sqrt_80_l737_737813

theorem floor_sqrt_80 : (Int.floor (Real.sqrt 80) = 8) :=
by
  have h1 : (64 = 8^2) := by norm_num
  have h2 : (81 = 9^2) := by norm_num
  have h3 : (64 < 80 ∧ 80 < 81) := by norm_num
  have h4 : (8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) :=
    by
      rw [←h1, ←h2]
      exact Real.sqrt_lt_sq ((lt_add_one 8).mpr rfl) (by linarith)
  have h5 : (Int.floor (Real.sqrt 80) = 8) := sorry
  exact h5

end floor_sqrt_80_l737_737813


namespace problem_l737_737303

noncomputable def A := 3.75 * 10^(-7)
noncomputable def B := (3 / 4) * 10^(-7)
noncomputable def C := (3 / 8) * 10^(-6)
noncomputable def D := (3 / 8) * 10^(-7)
noncomputable def target := 0.000000375

theorem problem : D ≠ target :=
by {
  sorry
}

end problem_l737_737303


namespace sum_of_remainders_l737_737673

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l737_737673


namespace largest_multiple_of_7_negation_gt_neg150_l737_737274

theorem largest_multiple_of_7_negation_gt_neg150 : 
  ∃ (k : ℤ), (k % 7 = 0 ∧ -k > -150 ∧ ∀ (m : ℤ), (m % 7 = 0 ∧ -m > -150 → m ≤ k)) :=
sorry

end largest_multiple_of_7_negation_gt_neg150_l737_737274


namespace shem_earnings_l737_737195

theorem shem_earnings (kem_hourly: ℝ) (ratio: ℝ) (workday_hours: ℝ) (shem_hourly: ℝ) (shem_daily: ℝ) :
  kem_hourly = 4 →
  ratio = 2.5 →
  shem_hourly = kem_hourly * ratio →
  workday_hours = 8 →
  shem_daily = shem_hourly * workday_hours →
  shem_daily = 80 :=
by
  -- Proof omitted
  sorry

end shem_earnings_l737_737195


namespace floor_sqrt_80_l737_737961

theorem floor_sqrt_80 : ⌊real.sqrt 80⌋ = 8 := 
by {
  let sqrt80 := real.sqrt 80,
  have sqrt80_between : 8 < sqrt80 ∧ sqrt80 < 9,
  { split;
    linarith [real.sqrt_lt.2 (by norm_num : 64 < (80 : ℝ)),
              real.lt_sqrt.2 (by norm_num : (80 : ℝ) < 81)] },
  rw real.floor_eq_iff,
  use (and.intro (by linarith [sqrt80_between.1]) (by linarith [sqrt80_between.2])),
  linarith
}

end floor_sqrt_80_l737_737961


namespace feed_sequences_2880_l737_737749

-- Define the pairs of animals and the initial condition
structure AnimalPair :=
  (male : String)
  (female : String)

-- Define the function to calculate feeding sequences
noncomputable def feed_sequences (pairs : List AnimalPair) : Nat :=
  let males := pairs.length
  let females := pairs.length
  let sequence_count : ℕ := (List.range females.length).foldl (λ acc, n => acc * (females - n) * 2.max(1 - n // 2)) 1
  sequence_count

-- Define the main theorem to check the sequence count
theorem feed_sequences_2880 (pairs : List AnimalPair) (h_pairs : pairs.length = 5) : 
  feed_sequences pairs = 2880 := 
  by
    -- We cannot prove this within a code segment
    sorry

end feed_sequences_2880_l737_737749


namespace max_m_min_value_inequality_l737_737487

theorem max_m (m : ℝ) :
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m) → m ≤ 1 :=
by
  assume h : ∀ x : ℝ, |x - 3| + |x - m| ≥ 2 * m
  sorry

theorem min_value_inequality (a b c : ℝ) (m : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = m) 
  (h_max_m : m = 1):
  4 * a^2 + 9 * b^2 + c^2 ≥ 36 / 49 ∧ 
  (4 * a^2 + 9 * b^2 + c^2 = 36 / 49 ↔ a = 9 / 49 ∧ b = 4 / 49 ∧ c = 36 / 49) :=
by
  assume h : a > 0 ∧ b > 0 ∧ c > 0
  assume h_sum : a + b + c = m 
  assume h_max_m : m = 1
  sorry

end max_m_min_value_inequality_l737_737487


namespace floor_sqrt_80_l737_737819

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l737_737819


namespace floor_sqrt_80_l737_737868

theorem floor_sqrt_80 : (⌊Real.sqrt 80⌋ = 8) :=
by
  -- Use the conditions
  have h64 : 8^2 = 64 := by norm_num
  have h81 : 9^2 = 81 := by norm_num
  have h_sqrt64 : Real.sqrt 64 = 8 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_eight]
  have h_sqrt81 : Real.sqrt 81 = 9 := by rw [Real.sqrt_sq_eq_abs, abs_of_nonneg zero_le_nine]
  -- Establish inequality
  have h_ineq : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9 := 
    by 
      split
      -- 8 < Real.sqrt 80 
      · apply lt_of_lt_of_le _ (Real.sqrt_le_sqrt (le_refl 80) h81.le)
        exact lt_add_one 8
      -- Real.sqrt 80 < 9
      · apply le_of_lt
        apply lt_trans (Real.sqrt_lt_sqrt _ _) h_sqrt81
        exact zero_le 64
        exact le_of_lt h
  -- Conclude using the floor definition
  exact sorry

end floor_sqrt_80_l737_737868


namespace length_AP_l737_737139
open Real

-- Definitions and conditions
def side_length : ℝ := 2
def radius : ℝ := side_length / 2
def center : (ℝ × ℝ) := (0, 0)
def A : (ℝ × ℝ) := (-radius, radius)
def M : (ℝ × ℝ) := (0, -radius)

noncomputable def AM_slope : ℝ := (A.snd - M.snd) / (A.fst - M.fst)
noncomputable def AM_line (x : ℝ) : ℝ := AM_slope * x + A.snd

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = radius^2

-- Main theorem
theorem length_AP : dist A (AM_line (-4 / 5), (-3 / 5)) = sqrt 65 / 5 := by
    sorry

end length_AP_l737_737139


namespace probability_of_pink_flower_is_five_over_nine_l737_737649

-- Definitions as per the conditions
def flowersInBagA := 9
def pinkFlowersInBagA := 3
def flowersInBagB := 9
def pinkFlowersInBagB := 7
def probChoosingBag := (1:ℚ) / 2

-- Definition of the probabilities
def probPinkFromA := pinkFlowersInBagA / flowersInBagA
def probPinkFromB := pinkFlowersInBagB / flowersInBagB

-- Total probability calculation using the law of total probability
def probPink := probPinkFromA * probChoosingBag + probPinkFromB * probChoosingBag

-- Statement to be proved
theorem probability_of_pink_flower_is_five_over_nine : probPink = (5:ℚ) / 9 := 
by
  sorry

end probability_of_pink_flower_is_five_over_nine_l737_737649


namespace solve_first_system_solve_second_system_l737_737607

theorem solve_first_system :
  (exists x y : ℝ, 3 * x + 2 * y = 6 ∧ y = x - 2) ->
  (∃ (x y : ℝ), x = 2 ∧ y = 0) := by
  sorry

theorem solve_second_system :
  (exists m n : ℝ, m + 2 * n = 7 ∧ -3 * m + 5 * n = 1) ->
  (∃ (m n : ℝ), m = 3 ∧ n = 2) := by
  sorry

end solve_first_system_solve_second_system_l737_737607


namespace number_of_solutions_l737_737049

def f (x : ℝ) : ℝ := |1 - 2 * x|

theorem number_of_solutions :
  (∃ n : ℕ, n = 8 ∧ ∀ x ∈ [0,1], f (f (f x)) = (1 / 2) * x) :=
sorry

end number_of_solutions_l737_737049


namespace value_of_x_l737_737044

theorem value_of_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 :=
by
  sorry

end value_of_x_l737_737044


namespace find_d_plus_f_l737_737234

noncomputable def complex_nums_equation (a b c d e f : ℂ) : Prop :=
  b = 4 ∧
  e = -2 * a - c ∧
  a + c + e = 0 ∧
  b + d + f = 3

theorem find_d_plus_f (a b c d e f : ℂ) (h : complex_nums_equation a b c d e f) : d + f = -1 :=
by
  cases h with hb h_rest
  cases h_rest with he h_rest
  cases h_rest with ha hbd
  rw [hb] at hbd
  have h : d + f = 3 - 4 := by linarith
  exact h

end find_d_plus_f_l737_737234


namespace polar_center_of_circle_l737_737531

theorem polar_center_of_circle :
  ∀ (ρ θ : ℝ),
    (∃ (θ : ℝ), ρ = cos (θ + π / 3)) →
    (∃ (ρ θ : ℝ), 
      (ρ = 1/2 ∧ θ = -π / 3) ∧ 
      (x = ρ * cos θ) ∧ 
      (y = ρ * sin θ)) :=
begin
  -- proof to be filled in
  sorry
end

end polar_center_of_circle_l737_737531


namespace value_of_y_l737_737106

theorem value_of_y (x y : ℝ) (cond1 : 1.5 * x = 0.75 * y) (cond2 : x = 20) : y = 40 :=
by
  sorry

end value_of_y_l737_737106


namespace floor_sqrt_80_eq_8_l737_737840

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737840


namespace sum_of_ten_numbers_l737_737511

theorem sum_of_ten_numbers (average count : ℝ) (h_avg : average = 5.3) (h_count : count = 10) : 
  average * count = 53 :=
by
  sorry

end sum_of_ten_numbers_l737_737511


namespace ada_initial_position_l737_737605

-- Define the conditions and the question as constants and variables
constant Seats : Type
constant Ada Bea Ceci Dee Edie Flo : Seats

-- Define initial positions of friends as variables
variable (Pos : Seats → ℕ) -- Pos assigns a seat number to each friend

-- Define the conditions based on movements
axiom A1 : Pos Bea = Pos Bea + 1
axiom A2 : Pos Ceci = Pos Ceci + 2
axiom A3 : Pos Dee = Pos Dee - 1
axiom A4 : Pos Edie = Pos Flo ∧ Pos Flo = Pos Edie

-- Ada returns to an end seat (either 1 or 6)
axiom A5 : Pos Ada = 1 ∨ Pos Ada = 6

-- Deduction about the initial position of Ada
theorem ada_initial_position : Pos Ada = 3 :=
sorry

end ada_initial_position_l737_737605


namespace shortest_path_length_l737_737621

-- The conditions and transformations used in the problem
structure Dimensions :=
  (duct_length : ℕ)
  (duct_width : ℕ)
  (duct_height : ℕ)
  (cube_size : ℕ)

-- Coordinates of the starting and ending points
structure Coordinates :=
  (start : ℕ × ℕ × ℕ)
  (end : ℕ × ℕ × ℕ)

-- The proof problem statement
theorem shortest_path_length (dims : Dimensions)
                               (coords : Coordinates)
                               (m n : ℕ)
                               (path_length : ℕ → ℕ → ℝ) :
  dims.duct_length = 10 ∧
  dims.duct_width = 2 ∧
  dims.duct_height = 1 ∧
  dims.cube_size = 2 ∧
  coords.start = (2, 0, 2) ∧
  coords.end = (0, 10, -3) ∧
  (path_length 32 45 = √32 + √45) →
  m = 32 ∧ n = 45 ∧
  m + n = 77 :=
sorry

end shortest_path_length_l737_737621


namespace proof_seq_l737_737696

open Nat

-- Definition of sequence {a_n}
def seq_a : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * seq_a n

-- Definition of sum S_n of sequence {b_n}
def sum_S : ℕ → ℕ
| 0 => 0
| n + 1 => sum_S n + (2^n)

-- Definition of sequence {b_n}
def seq_b : ℕ → ℕ
| 0 => 1
| n + 1 => 2 * seq_b n

-- Definition of sequence {c_n}
def seq_c (n : ℕ) : ℕ := seq_b n * log 3 (seq_a n) -- Note: log base 3

-- Sum of first n terms of {c_n}
def sum_T : ℕ → ℕ
| 0 => 0
| n + 1 => sum_T n + seq_c n

-- Proof statement
theorem proof_seq (n : ℕ) :
  (seq_a n = 3 ^ n) ∧
  (2 * seq_b n - 1 = sum_S 0 * sum_S n) ∧
  (sum_T n = (n - 2) * 2 ^ (n + 2)) :=
sorry

end proof_seq_l737_737696


namespace number_of_red_dresses_l737_737544

-- Define context for Jane's dress shop problem
def dresses_problem (R B : Nat) : Prop :=
  R + B = 200 ∧ B = R + 34

-- Prove that the number of red dresses (R) should be 83
theorem number_of_red_dresses : ∃ R B : Nat, dresses_problem R B ∧ R = 83 :=
by
  sorry

end number_of_red_dresses_l737_737544


namespace domain_log_condition_l737_737781

theorem domain_log_condition : ∀ x : ℝ, (1 - 2^x > 0) ↔ (x < 0) :=
by
  intros x
  sorry

end domain_log_condition_l737_737781


namespace largest_multiple_of_7_gt_neg_150_l737_737266

theorem largest_multiple_of_7_gt_neg_150 : ∃ (x : ℕ), (x % 7 = 0) ∧ ((- (x : ℤ)) > -150) ∧ ∀ y : ℕ, (y % 7 = 0 ∧ (- (y : ℤ)) > -150) → y ≤ x :=
by
  sorry

end largest_multiple_of_7_gt_neg_150_l737_737266


namespace right_triangle_hypotenuse_and_area_l737_737288

theorem right_triangle_hypotenuse_and_area (a b : ℕ) (h₁ : a = 60) (h₂ : b = 80) : 
  let hypotenuse := (Int.sqrt (a*a + b*b)) in
  let area := (a * b) / 2 in
  hypotenuse = 100 ∧ area = 2400 := 
  by
  rw [h₁, h₂]
  let hypotenuse := (Int.sqrt (60 * 60 + 80 * 80))
  let area := (60 * 80) / 2
  have : hypotenuse = 100 := by sorry
  have : area = 2400 := by sorry
  exact ⟨this, this⟩

end right_triangle_hypotenuse_and_area_l737_737288


namespace floor_sqrt_80_eq_8_l737_737902

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l737_737902


namespace coefficient_monomial_is_l737_737215

def monomial (x y : ℝ) : ℝ := (x * y^2) / 5

theorem coefficient_monomial_is (x y : ℝ) : (monomial x y) = (1 / 5) * x * y^2 :=
by
  sorry

end coefficient_monomial_is_l737_737215


namespace creases_match_pattern_C_l737_737392

-- Definitions for the conditions
structure FoldedPaper where
  folds : ℕ -- Number of folds
  shape : Type -- Shape of the paper, e.g., a square

def isosceles_right_triangle (angle : ℝ) : Prop :=
  angle = 90 -- Defines an isosceles right triangle with right angle

def fold (paper : FoldedPaper) (times : ℕ) : FoldedPaper :=
  { paper with folds := times }

def unfold (paper : FoldedPaper) : FoldedPaper :=
  { paper with shape := paper.shape }

-- Main theorem stating the problem
theorem creases_match_pattern_C : 
  ∀ (paper : FoldedPaper),
  paper.shape = Square →
  fold paper 4.fold =
  unfold paper →
  creases paper = pattern_C :=
by sorry

-- Representation of initial paper
def Square : Type := sorry

-- Representation of the crease pattern C
def pattern_C : Type := sorry

-- Initial paper instance
def initial_paper : FoldedPaper :=
  { folds := 0, shape := Square }

-- Sorrys used to bypass actual definitions and proof for demonstration

end creases_match_pattern_C_l737_737392


namespace sum_of_common_roots_l737_737007

theorem sum_of_common_roots :
  (∑ k in {k : Type* | ∃ x : ℝ, (x^3 - 3 * x^2 + 2 * x = 0) ∧ (x^2 + 3 * x + k = 0)}, id k) = -14 :=
sorry

end sum_of_common_roots_l737_737007


namespace mean_of_set_with_given_median_l737_737782

def mean_set (s : Finset ℝ) : ℝ := (s.sum id) / (s.card)

noncomputable def set_m (m : ℝ) : Finset ℝ := {m, m + 6, m + 8, m + 14, m + 21}

theorem mean_of_set_with_given_median (m : ℝ) (h : m + 8 = 16) : mean_set (set_m m) = 17.8 := 
by 
  sorry

end mean_of_set_with_given_median_l737_737782


namespace most_likely_sum_exceeds_12_l737_737736

/-- A six-sided die with faces 1, 2, 3, 4, 5, and 6. -/
def six_side_die := {1, 2, 3, 4, 5, 6}

/-- Calculate the most likely sum of points after the sum of die rolls exceeds 12. -/
theorem most_likely_sum_exceeds_12 : ∃ m, m = 13 :=
by 
  -- Assuming rolling a six-sided die until the sum exceeds 12
  sorry

end most_likely_sum_exceeds_12_l737_737736


namespace hexagon_side_length_l737_737221

theorem hexagon_side_length (p : ℕ) (s : ℕ) (h₁ : p = 24) (h₂ : s = 6) : p / s = 4 := by
  sorry

end hexagon_side_length_l737_737221


namespace rounding_example_l737_737193

def round_to_nearest_whole (x : ℝ) : ℤ :=
  if x - x.floor < 0.5 then x.floor else x.ceil

theorem rounding_example : round_to_nearest_whole 24567.4999997 = 24567 := by
  sorry

end rounding_example_l737_737193


namespace cross_product_example_l737_737424

def vector_cross (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := 
  (u.2.1 * v.2.2 - u.2.2 * v.2.1, 
   u.2.2 * v.1 - u.1 * v.2.2, 
   u.1 * v.1 - u.2.1 * v.1)
   
theorem cross_product_example : 
  vector_cross (4, 3, -7) (2, 0, 5) = (15, -34, -6) :=
by
  -- The proof will go here
  sorry

end cross_product_example_l737_737424


namespace surface_area_of_cuboid_volume_of_cuboid_l737_737688

-- Define the basic parameters for the cuboid
def length : ℝ := 10
def breadth : ℝ := 8
def height : ℝ := 6

-- The surface area of the cuboid
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

-- The volume of the cuboid
def volume (l w h : ℝ) : ℝ := l * w * h

-- The theorem about the surface area
theorem surface_area_of_cuboid :
  surface_area length breadth height = 376 :=
sorry

-- The theorem about the volume
theorem volume_of_cuboid :
  volume length breadth height = 480 :=
sorry

end surface_area_of_cuboid_volume_of_cuboid_l737_737688


namespace bus_capacity_l737_737519

theorem bus_capacity : 
  ∀ (left_side_seats right_side_seats people_per_seat backseat_people : ℕ),
  left_side_seats = 15 →
  right_side_seats = left_side_seats - 3 →
  people_per_seat = 3 →
  backseat_people = 11 →
  ((left_side_seats + right_side_seats) * people_per_seat + backseat_people) = 92 :=
by
  intros left_side_seats right_side_seats people_per_seat backseat_people h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end bus_capacity_l737_737519


namespace floor_sqrt_80_eq_8_l737_737845

theorem floor_sqrt_80_eq_8 :
  ∀ x : ℝ, (8:ℝ)^2 < 80 ∧ 80 < (9:ℝ)^2 → ⌊real.sqrt 80⌋ = 8 :=
by
  intro x
  assume h
  sorry

end floor_sqrt_80_eq_8_l737_737845


namespace probability_at_least_one_heart_or_king_l737_737334
   
   noncomputable def probability_non_favorable : ℚ := 81 / 169

   theorem probability_at_least_one_heart_or_king :
     1 - probability_non_favorable = 88 / 169 := 
   sorry
   
end probability_at_least_one_heart_or_king_l737_737334


namespace A_days_to_complete_job_l737_737323

noncomputable def time_for_A (x : ℝ) (work_left : ℝ) : ℝ :=
  let work_rate_A := 1 / x
  let work_rate_B := 1 / 30
  let combined_work_rate := work_rate_A + work_rate_B
  let completed_work := 4 * combined_work_rate
  let fraction_work_left := 1 - completed_work
  fraction_work_left

theorem A_days_to_complete_job : ∃ x : ℝ, time_for_A x 0.6 = 0.6 ∧ x = 15 :=
by {
  use 15,
  sorry
}

end A_days_to_complete_job_l737_737323


namespace sum_first_70_terms_l737_737451

-- Define the sequence a_n
def seq (n : ℕ) : ℚ := if n = 0 then 0 else 1 / 2 ^ n

-- Define the sum condition
def sum_condition (n : ℕ) : Prop :=
  ∑ i in Finset.range n, 2 ^ i * seq (i + 1) = n / 2

-- Define the new sequence
def new_seq : List ℚ :=
  List.bind (List.range 1 71) (λ n, List.replicate (2 * n - 1) (seq n))

-- Define the sum of the first 70 terms of the new sequence
def first_70_terms_sum : ℚ :=
  ∑ i in Finset.range 70, new_seq.get_or_else i 0

theorem sum_first_70_terms : first_70_terms_sum = 47 / 16 := 
by sorry

end sum_first_70_terms_l737_737451


namespace sin_monotonically_decreasing_l737_737217

open Real

theorem sin_monotonically_decreasing (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = sin (2 * x + π / 3)) →
  (0 ≤ x ∧ x ≤ π) →
  (∀ x, (π / 12) ≤ x ∧ x ≤ (7 * π / 12)) →
  ∀ x y, (x < y → f y ≤ f x) := by
  sorry

end sin_monotonically_decreasing_l737_737217


namespace find_rate_of_interest_l737_737725

-- Definitions based on conditions
def Principal : ℝ := 7200
def SimpleInterest : ℝ := 3150
def Time : ℝ := 2.5
def RatePerAnnum (R : ℝ) : Prop := SimpleInterest = (Principal * R * Time) / 100

-- Theorem statement
theorem find_rate_of_interest (R : ℝ) (h : RatePerAnnum R) : R = 17.5 :=
by { sorry }

end find_rate_of_interest_l737_737725


namespace prob_interval_0_2_l737_737133

variables (σ: ℝ) (hσ: σ > 0)

noncomputable def ξ := measure_theory.probability_measure.normal 1 σ

theorem prob_interval_0_2 :
  measure_theory.measure_space.measure (ξ : measure_theory.measure ℝ) (set.Ioc 0 2) = 0.7 :=
sorry

end prob_interval_0_2_l737_737133


namespace valid_paintings_count_l737_737751

-- There are 8 faces on the die, numbered 1 through 8
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Opposite face pairs whose numbers sum to 9
def opposite_pairs : Finset (ℕ × ℕ) := {(1, 8), (2, 7), (3, 6), (4, 5)}

-- Calculate the number of ways to paint two faces red ensuring their numbers don't sum to 9
def valid_paintings : ℕ := (faces.card choose 2) - opposite_pairs.card

theorem valid_paintings_count : valid_paintings = 24 := by
  unfold valid_paintings
  unfold faces
  unfold opposite_pairs
  norm_num
  -- Here, you'd compute and verify if needed, since we place a sorry to conclude for now
  sorry

end valid_paintings_count_l737_737751


namespace projection_of_3_neg2_onto_v_l737_737227

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product (a b : ℝ × ℝ) : ℝ := (a.1 * b.1 + a.2 * b.2)
  let scalar := (dot_product u v) / (dot_product v v)
  (scalar * v.1, scalar * v.2)

def v : ℝ × ℝ := (2, -8)

theorem projection_of_3_neg2_onto_v :
  projection (3, -2) v = (11/17, -44/17) :=
by sorry

end projection_of_3_neg2_onto_v_l737_737227


namespace distance_between_intersections_parabola_ellipse_l737_737353

theorem distance_between_intersections_parabola_ellipse :
  (exists (f : ℝ) (V_posLine : ℝ → Prop) (intersections : set (ℝ × ℝ)),
       let ellipse := λ (P : ℝ × ℝ), P.1 ^ 2 / 16 + P.2 ^ 2 / 36 = 1 in
       let parabola := λ (P : ℝ × ℝ), ∃ (A : ℝ), P.1 = A * P.2 ^ 2 in
       let shared_focus := abs (f * sqrt 5) in
       let directrix := V_posLine in
       ellipse (f, 0) ∧
       parabola (f, 0) ∧
       V_posLine (-f) ∧
       intersections ⊆ ellipse ∧
       intersections ⊆ parabola ∧
       ∃ P1 P2 ∈ intersections, P1 ≠ P2 ∧
       dist P1 P2 = 24 * sqrt 5 / sqrt (9 + 5 * sqrt 5)) :=
sorry

end distance_between_intersections_parabola_ellipse_l737_737353


namespace floor_sqrt_80_l737_737954

theorem floor_sqrt_80 : int.floor (real.sqrt 80) = 8 := by
  -- Definitions of the conditions in Lean
  have h1 : 64 < 80 := by
    norm_num
  have h2 : 80 < 81 := by
    norm_num
  have h3 : 8 < real.sqrt 80 := sorry
  have h4 : real.sqrt 80 < 9 := sorry
  -- Using the conditions to complete the proof
  sorry

end floor_sqrt_80_l737_737954


namespace comic_books_stacking_order_l737_737584

-- Definitions of the conditions
def num_spiderman_books : ℕ := 6
def num_archie_books : ℕ := 5
def num_garfield_books : ℕ := 4

-- Calculations of factorials
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Grouping and order calculation
def ways_to_arrange_group_books : ℕ :=
  factorial num_spiderman_books *
  factorial num_archie_books *
  factorial num_garfield_books

def num_groups : ℕ := 3

def ways_to_arrange_groups : ℕ :=
  factorial num_groups

def total_ways_to_stack_books : ℕ :=
  ways_to_arrange_group_books * ways_to_arrange_groups

-- Theorem stating the total number of different orders
theorem comic_books_stacking_order :
  total_ways_to_stack_books = 12441600 :=
by
  sorry

end comic_books_stacking_order_l737_737584


namespace floor_sqrt_80_l737_737880

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := 
by 
  have h : 64 ≤ 80 := by norm_num
  have h1 : 80 < 81 := by norm_num
  have h2 : 8 ≤ Real.sqrt 80 := Real.sqrt_le.mpr h
  have h3 : Real.sqrt 80 < 9 := Real.sqrt_lt.mpr h1
  exact Int.floor_of_nonneg_of_lt (Real.sqrt_nonneg 80) (Real.sqrt_pos.mpr h.to_lt) h3

end floor_sqrt_80_l737_737880


namespace dealership_truck_sales_l737_737756

theorem dealership_truck_sales (SUVs Trucks : ℕ) (h1 : SUVs = 45) (h2 : 3 * Trucks = 5 * SUVs) : Trucks = 75 :=
by
  sorry

end dealership_truck_sales_l737_737756
